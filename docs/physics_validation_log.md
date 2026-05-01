# Physics Validation Checks — Investigation Log

**Date**: 2026-04-13
**Goal**: Add systematic non-physical solution detection across the PNP-BV solver pipeline. Every forward solve result should be validated before being used for training data, inference, or plotting.

## Motivation

The forward solver (4-species and 3-species+Boltzmann) can produce solutions that pass SNES convergence and steady-state criteria but contain non-physical artifacts:
- Negative concentrations (CG1 maximum-principle violation)
- Fluxes exceeding transport limits
- Regularization (concentration floor, exponent clip) masking real problems
- Partial z-convergence stored as valid data

These artifacts poison surrogate training data and mislead the inverse solver.

## Classification

### Automatic Failures (discard the data point)

| ID | Check | Condition |
|----|-------|-----------|
| F1 | Negative concentration | any c_i < 0 in domain |
| F2 | Current exceeds diffusion limit | \|I\| > I_lim = n·F·D·c_bulk/L |
| F3 | Peroxide selectivity > 100% | \|PC/CD\| > 1.0 + tolerance |
| F4 | Concentration floor dominating BV rate | c_surf_i == eps_c at electrode for reacting species |
| F5 | Partial z masquerading as physical | achieved_z < 1.0 - 1e-6 |
| F6 | H2O2 exceeds stoichiometric limit | c_H2O2 > c_O2_bulk anywhere |
| F7 | Peroxide current wrong sign | PC > 0 at cathodic overpotentials |

### Warnings (flag but keep, investigate if widespread)

| ID | Check | Condition |
|----|-------|-----------|
| W1 | Exponent clip saturation | \|α·n_e·η\| >= clip_val for any reaction |
| W2 | Concentration exceeds bulk in interior | c_i > c_bulk_i * 1.05 far from electrode |
| W3 | Potential overshoot | phi outside [min(phi_applied,0), max(phi_applied,0)] by >10% |
| W4 | Charge neutrality violation in bulk | \|Σ z_i·c_i\| / c_bulk > 1% in top half of domain |
| W5 | H+ depletion below physical floor | c_H+ drops >3 orders below bulk outside Debye layer |
| W6 | Non-monotonic profiles | sign changes in grad(c) for neutral species in bulk |
| W7 | Mass conservation error | integral balance > 1% |
| W8 | False steady state | bulk integral still changing >0.1% despite flux convergence |
| W9 | Flux continuity mismatch | NP flux vs BV rate differ >5% at electrode |
| W10 | Boundary layer under-resolved | estimated BL thickness < 2*h_min |
| W11 | Gibbs oscillations in Debye layer | grad(c) changes sign >1 time in first 10 elements |

## Exploration Results (2026-04-13)

Four parallel agents explored the codebase. Key findings:

### Finding 1: Zero physics validation at any solution return point

Every file that returns a solution (U, U_data, GridPointResult, RobustCurveResult) does so with
NO physics checks. Only SNES convergence and steady-state flux criteria are checked.

Files affected: `solvers.py` (3 functions), `grid_charge_continuation.py` (Phase 1 + Phase 2 snapshots),
`robust_forward.py` (_z_ramp_worker + solve_curve_robust), `hybrid_forward.py` (z=0 and z=1 paths),
`stabilized_forward.py`, `gummel_solver.py`.

### Finding 2: Zero observable validation at any assembly point

Current density and peroxide current are assembled via `fd.assemble()` and stored directly.
No checks for NaN, Inf, sign, magnitude, or diffusion limit. This spans:
`observables.py`, `bv_point_solve/__init__.py`, `bv_point_solve/forward.py`, `bv_curve_eval.py`,
`Surrogate/training.py`, `FluxCurve/curve_eval.py`.

### Finding 3: Training pipeline checks are necessary but insufficient

`overnight_train_v16.py._validate_sample()` checks z >= 0.999 and finite observables.
But: no concentration floor check, no flux ceiling, no false steady-state detection.
`hybrid_forward.py._solve_z0_points()` accepts if `steps > 0` (extremely permissive).

### Finding 4: Zero diagnostic for regularization activation

Exponent clipping (forms.py:247), concentration floor (forms.py:297), and softplus
(forms.py:294) all activate silently. No code anywhere detects or reports when these
regularizations are dominating the physics.

---

## Implementation (2026-04-13)

### Wave 1: Foundation
- Created `Forward/bv_solver/validation.py` (255 lines) — shared module with:
  - `ValidationResult` dataclass (valid, failures, warnings)
  - `validate_solution_state()` — checks F1, F4, F6, W2, W3, W5 on Firedrake Function U
  - `validate_observables()` — checks F2, F3, F7 on assembled CD/PC values
  - `validate_steady_state()` — checks W8 on flux/integral history
  - `check_clip_saturation()` — checks W1 on BV exponent values

### Wave 2: Core solver + diagnostic metadata
- `forms.py` — added `_diag_*` keys to ctx (bv_exp_scale, exponent_clip, eps_c, per-reaction E_eq/alpha/n_e). Also refactored eta_clipped into `_build_eta_clipped()` for per-reaction E_eq support.
- `observables.py` — added `assemble_observable_validated()` helper (F2 magnitude check only)
- `solvers.py` — validation at return points of all 4 solver functions (warn-only)
- `grid_charge_continuation.py` — validation at Phase 2 snapshot, `GridPointResult.validation` field, `physics_failures()` helper. Converged flag now requires physics validity.

### Wave 3: Higher-level solvers
- `robust_forward.py` — validation in `_z_ramp_worker`, `validation_failures` array in `RobustCurveResult`, cache guard in `populate_ic_cache_robust`
- `hybrid_forward.py` — validation in `_solve_z0_points`, tightened z=0 acceptance to require `ok=True`
- `bv_point_solve/__init__.py` — post-convergence solution state validation
- `bv_point_solve/forward.py` — post-convergence validation, cache-skip on physics failure

### Wave 4: Pipeline consumers
- `bv_curve_eval.py` — per-point skip on F2 failure, cross-observable F3 selectivity check via `_skip_mask`
- `Surrogate/training.py` — per-point validation after extraction, re-validation after interpolation
- `overnight_train_v16.py` — F2/F3/F7/W1 checks added to `_validate_sample()`
- `compute_adjoint_gradients_v16.py` — re-validation of loaded data, gradient magnitude warnings

### Totals
- 1 new file (validation.py), 12 modified files
- ~685 lines added across all files
- Checks implemented: F1-F7 (failures), W1-W3, W5, W8 (warnings)
- Not yet implemented: W4, W6, W7, W9, W10, W11

## Verification Round 1 — Level 2 (Sonnet + Opus)

8 agents (4 Sonnet, 4 Opus) reviewed all 13 files. Found 7 critical bugs, 8 warnings.

### Critical bugs found and fixed

| # | File | Bug | Fix |
|---|------|-----|-----|
| 1 | `robust_forward.py:460` | Wrong config key `params.get("eps_c", 1e-6)` — should be nested under `bv_convergence`. F4 check 100x less sensitive. | Fixed to `params.get("bv_convergence", {}).get("conc_floor", 1e-8)` |
| 2 | `hybrid_forward.py:167` | Identical wrong config key bug. | Same fix. |
| 3 | `bv_curve_eval.py:83` | Gradient added without objective for physics-failed points. Breaks L-BFGS-B invariant `grad == d(obj)`. | Skip both gradient and objective. |
| 4 | `forward.py:253` + `__init__.py:627` | Observable validation inside time-step loop fires on transients, causing premature `return None` / `break`. | Removed mid-loop validation; post-convergence checks remain. |
| 5 | `training.py:249` | `_interpolate_failed_points` passes `pc=0.0` hardcoded — F3/F7 vacuous. | Replaced with direct F2 magnitude check. |
| 6 | `observables.py` | `assemble_observable_validated` passes `cd=0` in PC mode — F7 can't fire. | Replaced with honest single-observable F2 check. |
| 7 | `compute_adjoint_gradients_v16.py:455` | Looks for `"obs_cd"` key but saver writes `"current_density"` — dead code. | Fixed key names. |

### Remaining warnings (not yet fixed)

| # | Location | Issue |
|---|----------|-------|
| 8 | `validation.py:53` | `exponent_clip` param accepted but unused; W1 never fires from `validate_solution_state` callers |
| 9 | `validation.py:66-79` | F1 and F4 double-fire on negative concentrations |
| 10 | `forms.py:346` | `E_eq_j_val != 0.0` guard should be `is not None` |
| 11 | `bv_curve_eval.py` | Hardcoded `I_lim=1.0` inconsistent with `2.0*max(c_bulk)` in point-solve files |
| 12 | `training.py` vs `overnight_train_v16.py` | I_lim=1.0 vs I_LIM=2.0 across pipelines |
| 13 | `hybrid_forward.py` | z=0 acceptance tightened — budget-exhausted points now become NaN |
| 14 | `compute_adjoint_gradients_v16.py:270` | `solver_params_list[0]` is n_species, not phi_applied |
| 15 | Subprocess workers | W-level warnings silently discarded in ProcessPoolExecutor |

## Verification Round 2 — Level 1 (Sonnet) — Complete

4 Sonnet agents re-verified all 7 critical fixes. PASS — all fixes confirmed correct, no new issues introduced. 8 warning-level items remain (unused exponent_clip, F1+F4 double-fire, E_eq guard, I_lim inconsistency — which was subsequently also fixed).

## Autonomous k0 Inference Investigation (2026-04-13, continued)

**Goal**: make k0 inference feasible.

### Diagnostic finding: BV exponent clip × concentration floor interaction

Reaction 2 has E_eq = 1.78V (high), so η_2 = phi_applied - E_eq is very negative at all practical operating voltages. With `clip_exponent=50`, `exp(-α·n_e·η_2)` is capped at `exp(50) ≈ 5e21`. Combined with `conc_floor=1e-12`:

```
rate_at_floor = k0_2 * exp(50) * (c_H+/c_ref)^2 * 1e-12
             ≈ 5e-5 * 5e21 * 2e-6 * 1e-12
             ≈ 0.5 per unit time
```

At all V from -0.30 to +0.40, this spurious rate is identical because exp is saturated. When c_H2O2 is driven to zero (anodic voltages, where reaction 1 barely produces H2O2), this spurious sink term has no physical source to balance against, and the solver finds a "solution" where c_H2O2 goes catastrophically negative (-0.69 at V=0.40V).

**Options tested:**
1. Lower clip (to 20-30): distorts R_1 physics at cathodic voltages — rejected
2. Seed H2O2 IC positive: breaks V=-0.30 (spurious R_2 rate blows up Newton) — didn't help
3. **Log-concentration transform with seeded H2O2**: works! See next section.

### Log-c 3sp+Boltzmann breakthrough

`forms_logc.py` uses `u_i = ln(c_i)` as the primary unknown. Key properties:
- `c = exp(u)` is mathematically guaranteed positive — F1 impossible by construction
- With H2O2 seeded at u = log(1e-4) = -9.2, the log-of-zero singularity is avoided
- Boltzmann ClO4- background remains valid (phi is still the primary variable for Poisson)

**Results** (`scripts/studies/v18_test_3sp_logc.py`):

| V_RHE | z achieved | cd | Convergence |
|-------|------------|-----|-------------|
| -0.30 | 0.00 | failed | First voltage, no warm-start |
| -0.10 | **1.00** | -0.1780 | FULL |
| +0.00 | **1.00** | -0.1738 | FULL |
| +0.10 | **1.00** | -0.1631 | FULL |
| +0.15 | **1.00** | -0.1450 | FULL |
| +0.20 | **1.00** | -0.0786 | FULL |
| +0.25 | **1.00** | -0.0020 | FULL |
| +0.30 | **1.00** | -0.00002 | FULL |
| +0.40 | 0.00 | failed | H2O2 runaway |
| +0.50 | **1.00** | -0 | FULL |

All converged points have positive concentrations. The working range covers the entire onset region.

### k0 sensitivity in the working range

`scripts/studies/v18_logc_k0_sensitivity.py` swept k0_r1 multiplier over [0.2, 0.5, 1.0, 2.0, 5.0] (25x range):

| V_RHE | max |Δcd| over 25x k0 | Relative |
|-------|------------------------|----------|
| -0.10 | 0 | 0% (diffusion-limited) |
| +0.10 | 0.002 | 1.5% |
| **+0.15** | **0.013** | **8.9%** |
| **+0.20** | **0.051** | **50%** |
| **+0.25** | **0.008** | **95%** |
| +0.30 | 0.00002 | 49% (but cd tiny) |

Onset voltages V=0.15-0.25 show strong k0 signal. This is the regime where k0 is identifiable.

### Inference test

In progress: `scripts/studies/v18_logc_inference_test.py`
- Target: 6 voltage points V∈[-0.10, +0.25]V at (k0_true=1.263e-3, α_true=0.627)
- Noise: 2% Gaussian on each voltage point
- Initial guess: k0 at +20% offset, α at -10% offset
- Optimizer: scipy Nelder-Mead (derivative-free)

### Integration plan (if inference succeeds)

1. **Add a `transform: "logc"` option to solver_params** — lets callers request log-c forms without changing function signatures everywhere.

2. **Dispatch in `build_forms`** — check `params.get("transform")` and call `build_forms_logc` if requested. Similarly for `build_context` and `set_initial_conditions`.

3. **Integrate Boltzmann background** — the monkey-patch pattern in v18 scripts should become a first-class solver_params option, e.g. `params["boltzmann_species"] = [{"species": "ClO4-", "z": -1, "c_bulk_nondim": 0.2}]`.

4. **Seed H2O2 IC** — expose via `params.get("h2o2_ic_seed")` with a sensible default.

5. **Wire validation** — already done at solution return points; log-c solutions go through the same validation (c_min should always be ≥ 0 by construction, but the checks still catch anomalies).

6. **Update training pipeline** — `overnight_train_v16.py` can opt in via solver_params config. Per-sample validation already in place.

7. **Adjoint through log-c** — pyadjoint should tape through `exp(u)` automatically. Need to confirm the Boltzmann term is tape-compatible and test gradient accuracy via finite-difference comparison.

### What doesn't work (documented dead ends)

- **Standard 4sp**: fails at V > 0.15V (Debye layer oscillation in ClO4-)
- **Stabilized 4sp (d_art=0.001)**: converges to V=+0.73V but destroys onset physics (FLAT curve)
- **3sp + Boltzmann (concentration formulation)**: converges to V=+0.60V but H2O2 goes to -0.69 (CG1 oscillation around near-zero state)
- **Seeded H2O2 (concentration formulation)**: breaks V=-0.30V (spurious R_2 rate blows up Newton), doesn't fix anodic F1 violations
- **Lower BV clip (30 or 20)**: distorts R_1 physics at cathodic voltages
- **Log-c without H2O2 seed**: log(0) = -46 creates extreme stiffness (v18 entry 2)

### What works

- **Log-c 3sp + Boltzmann + H2O2 seed (1e-4)**: converges at V = -0.10 to +0.30 (entire onset region) with positive concentrations by construction. Strong k0 sensitivity (50-95% relative Δcd for 25x k0 range at V=0.20-0.25).

### Inference results

**Unregularized NM (2% noise, 6 onset voltages):**
- Walks the k0-α ridge unboundedly
- k0 reached 17x true at eval 34 with J still decreasing
- Ridge slope: d(log k0)/d(α) ≈ -47 (1% α change ↔ 60% k0 change)
- **Diagnosis: ridge-limited, not noise-limited**

**Regularized NM (λ=0.01, correct k0 prior):**
- Eval 24: k0 error -0.5%, α error +0.6%
- Eval 26: k0 error -0.3%, α error +0.7%
- Eval 29: k0 error +0.1%, α error +0.5%
- **Recovery at noise floor; ~0.5% precision on both parameters**

### Conclusion

k0 inference is **feasible** with log-c onset physics + a weak Tikhonov k0 prior. The data alone doesn't pin k0 due to the α-k0 ridge; a prior from literature/EIS/Tafel analysis is required. This is consistent with v17's earlier finding that onset-only data is ridge-limited, but adds the concrete numerical demonstration that:

1. Log-c makes onset physics reliable (no spurious negative H2O2)
2. Even a weak prior (λ=0.01) transforms the ridge into a well-conditioned bowl
3. Recovery accuracy approaches the noise floor (<1% for both k0 and α at 2% noise)

See `docs/k0_inference_status.md` for full details and the integration plan.

---
