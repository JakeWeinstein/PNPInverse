# Handoff — k0 Inference for PNP-BV Solver

**Date**: 2026-04-13
**Status**: Working recipe established. Multiple follow-up experiments still needed.
**Author of this handoff**: Previous Claude instance; context is otherwise intact in git history.

## What to read first (in order)

1. `docs/k0_inference_status.md` — narrative of the investigation + full recipe + integration plan
2. `docs/physics_validation_log.md` — validation framework, failure mode catalog
3. `StudyResults/v18_convergence_extension_log.md` — prior investigation that concluded k0 was "non-identifiable"; this work overturns that conclusion
4. This file — things still to try and how

## Environment

- **Python**: `../venv-firedrake/bin/python` (from the PNPInverse directory). Do NOT use conda.
- **Platform**: macOS, Firedrake installed in the venv
- **Working directory**: `/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse`
- **Typical script runtime**: a single 6-voltage curve takes ~80-90 seconds. Nelder-Mead inference takes 30-40 evaluations = ~30-60 minutes. Plan for background execution.

Run example:
```bash
../venv-firedrake/bin/python scripts/studies/v18_logc_inference_test.py 2>&1 | tee /tmp/run.log
```

## The working recipe (summary)

The only combination that gives physically-correct onset physics AND identifiable k0:

1. **Forms**: use `Forward.bv_solver.forms_logc` (log-concentration transform, u = ln c)
2. **Species**: 3-species (O2, H2O2, H+); drop ClO4- as dynamic species
3. **Boltzmann background**: add `c_ClO4 = c_bulk * exp(phi)` to Poisson RHS (see `add_boltzmann_background` in `scripts/studies/v18_test_3species_boltzmann.py` for the monkey-patch pattern)
4. **H2O2 seed**: `C_H2O2_HAT = 1e-4` (NOT 0 — log(0) is singular)
5. **Voltage range**: V_RHE ∈ [-0.10, +0.30] for onset physics; V=-0.30 and V≥+0.40 don't converge in log-c
6. **Regularization**: Tikhonov on `log(k0)` with weight λ=0.01 and a prior from literature/EIS

Reference constants (from `scripts/_bv_common.py`):
- `K0_HAT_R1 = 1.2632e-3` (nondim k0 for O2/H2O2 reaction)
- `K0_HAT_R2 = 5.263e-5` (nondim k0 for H2O2/H2O reaction)
- `ALPHA_R1 = 0.627`, `ALPHA_R2 = 0.500`
- `E_EQ_R1 = 0.68V`, `E_EQ_R2 = 1.78V` (RHE)
- `V_T = 0.0257V` (thermal voltage)

## Things still to try (prioritized)

### HIGH PRIORITY

#### 1. Noise-seed robustness — untested
Only tested with `np.random.default_rng(42)`. Real noise is not seed 42.

**Do**: Run `scripts/studies/v18_logc_regularized.py` with 5+ different seeds (e.g., 1, 13, 42, 100, 999). For each seed, record (k0 error, α error). Report median and 95th percentile.

**Why it matters**: v13 previously showed 5-28% error variation across 5 seeds. Our result (0.3% error) is from one seed; needs confirmation.

**Expected outcome**: With regularization, errors should be noise-limited and therefore more stable than unregularized. But unconfirmed.

#### 2. Prior robustness — untested
Tested with `k0_prior = K0_HAT_R1` (i.e., prior == true, cheating). Real priors will be off.

**Do**: Re-run `v18_logc_regularized.py` with `log_k0_prior = np.log(K0_HAT_R1 * X)` for X ∈ {0.3, 0.5, 1.0, 2.0, 3.0}. For each, record recovered k0 and α.

**Why it matters**: a 2-3x off prior is realistic for literature k0 values. We need to know if the method still works, or biases toward the wrong prior.

**Modify**: the line `log_k0_prior = np.log(K0_HAT_R1)` in `v18_logc_regularized.py` main().

#### 3. k0_2 inference — untested
Only k0_r1 was inferred. k0_r2 is also a parameter.

**Do**: Extend `v18_logc_regularized.py` to also infer k0_r2 and α_r2. The inference variable becomes 4D: `(log k0_1, α_1, log k0_2, α_2)`.

**Why it matters**: Real inference cares about both reactions. v13 found k0_2 is harder to identify than k0_1. Need to see if log-c + prior fixes that too.

**Modify**: The `objective()` function to unpack 4 parameters and call `solve_curve(k0_1, k0_2, α_1, α_2)` with all four varying.

#### 4. Adjoint gradients through log-c — untested but high-value
Nelder-Mead takes 30+ evals (~45 min). Adjoint L-BFGS-B would take ~5 iterations × 2 evals (forward+adjoint) = ~15 min. 3x speedup.

**Do**: Write `scripts/studies/v18_logc_adjoint.py` modeled after `scripts/Inference/v18_adjoint_simple.py` but using `forms_logc` instead of standard forms. Verify gradient accuracy against finite differences at 3-5 test points. Then run L-BFGS-B.

**Why it matters**: For production use, NM is too slow. Adjoint is essential.

**Gotcha**: firedrake.adjoint should tape through `fd.exp(u)` automatically, but test it — there have been issues with some transform combinations.

### MEDIUM PRIORITY

#### 5. Extended voltage grid via warm-start chain
Log-c fails at V=-0.30 because no warm-start. But if we solve V=-0.10 first, then V=-0.15 using that as IC, then V=-0.20, etc., the continuation should work.

**Do**: Write a warm-started voltage sweep script that uses log-c and extends to V=-0.50V cathodic + V=+0.30V onset. This would give ~15-20 voltage points.

**Why it matters**: More points → tighter posterior → less reliance on prior strength.

**Reference**: `scripts/studies/v18_logc_k0_sensitivity.py` already uses the basic log-c pattern; extend it with warm-starting between voltage points.

#### 6. λ selection via L-curve or discrepancy principle
Currently λ=0.01 is arbitrary. There's standard theory for choosing λ optimally.

**Do**: Run inference for λ ∈ [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]. Plot `log(J_data)` vs `log(J_prior)` (the L-curve). The "corner" is the optimal λ.

**Why it matters**: With a wrong prior, too-strong λ biases k0 toward the wrong value. Too-weak λ doesn't break the ridge. L-curve balances these.

#### 7. Hybrid cathodic+onset data
v13 used full-cathodic (V_RHE ∈ [-1.2, -0.03]V) with standard 4sp and got 4-5% k0 error. This work uses onset (V_RHE ∈ [-0.10, +0.30]V) with log-c. Combine them.

**Do**: Forward-solve at V_RHE ∈ [-0.5, -0.15] using standard forms (which work there) AND V_RHE ∈ [-0.10, +0.30] using log-c forms. Concatenate the cd arrays. Run inference on the union.

**Why it matters**: More data → smaller prior weight needed → more robust to prior errors.

**Challenge**: have to harmonize the two solver setups; standard 4sp vs 3sp+Boltzmann have slightly different physics (ClO4- dynamics vs Boltzmann background).

#### 8. Peroxide current (PC) as a second observable
Only current density (CD) was used. Peroxide current (PC) is a separate observable with different k0/α sensitivity.

**Do**: Extend the inference objective to include PC residuals. The multi-observable form:
```python
J = Σ_V (cd_sim(V) - cd_target(V))² + w_pc * Σ_V (pc_sim(V) - pc_target(V))²
```

**Why it matters**: PC has different k0_2 sensitivity (k0_2 is the H2O2-reducing rate). Multi-observable was key to v13's success.

**Reference**: `FluxCurve/bv_curve_eval.py` has a `evaluate_bv_multi_observable_objective_and_gradient` function that handles this pattern. Adapt to log-c.

### LOW PRIORITY / SPECULATIVE

#### 9. Test on real experimental data
All testing has been synthetic targets. Real experimental data (from Mangan et al. 2025 per README) is the ultimate test.

**Do**: Load experimental I-V data, run the log-c + regularized inference. Compare recovered k0/α to literature values for the same system.

**Location**: Experimental data files are in `data/` directory.

#### 10. Adaptive λ (hierarchical Bayes)
Treat λ as a parameter to infer rather than fixed.

**Do**: Empirical Bayes — marginalize over λ. Requires more sophisticated MCMC or variational methods.

**Why**: Removes the arbitrary choice of λ.

#### 11. Multi-experiment fitting
Vary L_ref or c_O2_bulk across experiments. k0 identifies from shift in I-V curves between experiments.

**Do**: Simulate 2-3 experiments with different L_ref (e.g., 65μm, 32.5μm, 130μm). Joint inference fits all with shared k0/α.

**Why it matters**: Breaks k0-α degeneracy without needing a prior. Standard technique in electrochemistry.

**Challenge**: Significant solver changes; need to re-run forward for each experiment.

#### 12. Measure the ridge directly
Map the (log k0, α) landscape around the true minimum via grid scan to visualize the ridge and its curvature.

**Do**: Grid of (log k0, α) at ±50% k0 and ±10% α (say 11×11 grid). Compute J at each. Plot contours.

**Why**: Visualization aids understanding and confirms the ridge nature.

## Failure modes and diagnosis

### Symptom: solver diverges (c → ∞ or cd → 1e30)
**Likely cause**: IC inconsistency at first voltage. Log-c's IC for H2O2 is log(H2O2_SEED); if this is combined with a Dirichlet BC of c=0, the first Newton step has a huge gradient at the bulk boundary.

**Fix**: Keep H2O2_SEED at 1e-4 (not smaller). Ensure BC and IC consistency (both use c0_model[1]). Use warm-start from a neighboring voltage when possible.

### Symptom: inference converges to wrong k0 with near-zero J
**Likely cause**: Ridge degeneracy. Unregularized optimization will walk the ridge indefinitely.

**Fix**: Add Tikhonov regularization (see recipe). The prior doesn't need to be accurate — even a rough estimate (within 10x of true) is enough to close the ridge.

### Symptom: F1 (negative c) validation fires in log-c
**Shouldn't happen**. In log-c, c = exp(u) > 0 mathematically. If F1 fires, there's a bug in the validation (it's reading u_min as c_min).

**Fix**: Make sure validation calls in log-c scripts use `c = np.exp(U.dat[i].data_ro)`, not raw `U.dat[i].data_ro`. See `v18_test_3sp_logc.py` for the pattern.

### Symptom: Nelder-Mead takes 40+ evals
**Cause**: NM is derivative-free and 2D problems need many function evals, especially with ridge landscapes.

**Fix**: Use adjoint gradients + L-BFGS-B (see item #4 in "things to try").

## Important context and "gotchas"

### The venv is in the parent directory
Per user memory: `Use venv-firedrake in parent dir, not conda, for Python/pytest`. Scripts call `setup_firedrake_env()` which sets paths, but you must invoke with `../venv-firedrake/bin/python`.

### Adjoint annotation
When running forward-only solves (not inference), wrap in `with adj.stop_annotating():` to prevent pyadjoint from taping everything (memory bloat).

### The Boltzmann background is currently monkey-patched
The 3sp+Boltzmann model patches `build_forms` at runtime by modifying `ctx["F_res"]` after the standard build. See `add_boltzmann_background` in `v18_test_3species_boltzmann.py` for the pattern. This is fragile; a proper integration would make it a first-class solver_params option.

### forms_logc has its own per-reaction E_eq logic (line 297)
There's a known bug (flagged in earlier verification): `if E_eq_j_val is not None and E_eq_j_val != 0.0` should be `if E_eq_j_val is not None`. A reaction with E_eq=0 falls through to the global E_eq. In our 3sp+Boltzmann setup all reactions have non-zero E_eq so it doesn't matter, but a fresh integration should fix this.

### The physics validation framework
`Forward/bv_solver/validation.py` was added during this investigation. Checks F1/F2/F3/F4/F6/F7 (failures) and W1/W2/W3/W5/W8 (warnings). Already wired into all solver return points in the main pipeline (solvers.py, grid_charge_continuation.py, robust_forward.py, hybrid_forward.py, FluxCurve/bv_point_solve/*, Surrogate/training.py, overnight_train_v16.py). NOT wired into log-c paths yet since log-c guarantees F1 by construction.

### I_lim thresholds across files
The F2 (current exceeds diffusion limit) check uses `I_lim = 2.0 * max(c0)` via `compute_i_lim_from_params()`. This was threaded through training.py, bv_curve_eval.py, overnight_train_v16.py. If you add log-c to these, use the same helper.

## Key file locations

### Created this session
| File | Purpose |
|------|---------|
| `Forward/bv_solver/validation.py` | Physics validation framework (new) |
| `scripts/studies/v18_test_3sp_h2o2_seed.py` | Failed seeding experiment |
| `scripts/studies/v18_test_3sp_logc.py` | Log-c convergence test |
| `scripts/studies/v18_logc_k0_sensitivity.py` | k0 sensitivity scan |
| `scripts/studies/v18_logc_inference_test.py` | Unregularized inference (ridge walk) |
| `scripts/studies/v18_logc_noise_sensitivity.py` | Noise sweep (not run) |
| `scripts/studies/v18_logc_regularized.py` | Regularized inference (WORKS) |
| `docs/k0_inference_status.md` | Main status doc |
| `docs/physics_validation_log.md` | Validation framework log |
| `docs/physics_validation_plan.md` | Per-file implementation plan for validation |
| `.verification/REPORT.md` | Level-1 verification after fixes |

### Existing files (not modified here)
| File | What it does |
|------|--------------|
| `Forward/bv_solver/forms_logc.py` | Log-c forms (pre-existing, used as-is) |
| `Forward/bv_solver/forms.py` | Standard forms (modified to add diagnostic metadata) |
| `Forward/bv_solver/observables.py` | Observable form builders (modified with validated assemble) |
| `Forward/bv_solver/solvers.py` | Main solver functions (modified with validation) |
| `Forward/bv_solver/grid_charge_continuation.py` | Unified continuation module (modified with validation) |
| `FluxCurve/bv_curve_eval.py` | Inverse problem evaluator (modified with I_lim) |
| `Surrogate/training.py` | Surrogate training pipeline (modified with validation) |
| `scripts/_bv_common.py` | Shared constants (used as-is) |

## Contact / recovery

If lost:
1. `git log --oneline | head -20` shows recent commits for context
2. `docs/k0_inference_status.md` is the narrative
3. `StudyResults/v18_convergence_extension_log.md` is the earlier context (2026-04-07)
4. User's memory (`~/.claude/projects/*/memory/`) has: venv location, voltage window, adjoint preference
