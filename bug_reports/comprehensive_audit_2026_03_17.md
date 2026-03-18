# Comprehensive Bug Audit — PNPInverse Codebase
**Date:** 2026-03-17
**Method:** 13 parallel specialized agents scanning ~56k lines across 130 Python files
**Scope:** Mathematical correctness, nondimensionalization, surrogate models, FluxCurve pipeline, adjoint/inverse, convergence strategies, tests, I/O, numerical stability, cross-module interfaces, scripts, steady-state solvers, ISMO

---

## Executive Summary

| Severity | Count | Key Themes |
|----------|-------|------------|
| CRITICAL | 10 | BV math error, solver state corruption, ISMO data loss, script crashes, test inverse crime |
| HIGH | 18 | PTC bugs, ensemble inconsistency, gradient errors, coverage gaps, IO races |
| MEDIUM | 19 | Interface contracts, plotting, nondim gaps, frozen-dataclass violations |
| LOW | 15 | Labeling, code quality, latent issues |

---

## CRITICAL Findings

### C1. `n_electrons` parsed but never used in BV exponent
**File:** `Forward/bv_solver/forms.py:282,295`
**Agent:** math-forms-auditor
**Impact:** For the default 2-electron O2 reduction, the Tafel slope is wrong by a factor of 2. The exponent should be `exp(-alpha * n_e * eta)` but computes `exp(-alpha * eta)`. The `n_electrons` field in config is dead code.
**Note:** May be intentional if `alpha` is pre-multiplied by `n_electrons`, but then the config field is misleading.

### C2. Charge continuation restores from wrong state on failure
**File:** `Forward/bv_solver/solvers.py:494-501`
**Agent:** convergence-auditor
**Impact:** When Phase 2 fails mid-timestep, `U_prev` is from a partial solve at the **failing** z_factor, not the last converged z_factor. The returned solution is silently incorrect. The final status print also reports wrong z_factor.

### C3. Cathodic concentration factor `c_ref_nondim` can become zero after scaling
**File:** `Forward/bv_solver/forms.py:289`
**Agent:** numerical-auditor
**Impact:** `(c_surf / c_ref_f) ** power` produces inf/NaN when `c_ref_nondim` rounds to zero after nondimensionalization. No floor is applied post-scaling.

### C4. ISMO candidate-to-PDE-result matching uses fragile float comparison
**File:** `Surrogate/ismo.py:701-714`
**Agent:** ismo-auditor
**Impact:** `np.allclose(..., atol=1e-15)` can mismatch after float arithmetic, silently discarding valid PDE training data and wasting compute budget.

### C5. ISMO temporary directory deleted before results are fully processed
**File:** `Surrogate/ismo.py:660-678`
**Agent:** ismo-auditor
**Impact:** Forces fallback to the fragile float-matching logic (C4) instead of reading the proper `converged` mask from disk.

### C6. `run_ismo.py` references undefined `evaluate_pde_batch`
**File:** `scripts/surrogate/run_ismo.py:498,500`
**Agent:** script-auditor
**Impact:** `NameError` at runtime. Should be `evaluate_candidates_with_pde`.

### C7. `validate_surrogate.py` uses wrong NPZ key `splits["test"]`
**File:** `scripts/surrogate/validate_surrogate.py:62`
**Agent:** script-auditor
**Impact:** `KeyError` at runtime. Key is `"test_idx"`, not `"test"`.

### C8. Test inverse crime: PDE target NaN fallback uses surrogate predictions
**File:** `tests/test_inverse_verification.py:196-204`
**Agent:** test-auditor
**Impact:** Failed PDE solves are silently replaced with surrogate predictions in tests designed to validate surrogate-free parameter recovery. Defeats the purpose of the V&V test.

### C9. V&V tests silently skip when data/models are missing
**File:** `tests/test_surrogate_fidelity.py:69-76`, `tests/test_inverse_verification.py:320-359`
**Agent:** test-auditor
**Impact:** Core verification tests (`SUR-01` through `INV-03`) produce green CI with zero actual coverage if `data/surrogate_models/` doesn't exist.

### C10. Dirichlet solver Poisson equation uses `eps=1.0` with `F` on RHS
**File:** `Forward/dirichlet_solver.py:156-157`
**Agent:** nondim-auditor
**Impact:** Assembles `Laplacian(phi) = 96485 * sum(z_i * c_i)` which is physically meaningless. Masked for symmetric 1:1 electrolytes but wrong for any non-symmetric case. This fix was applied to Robin/BV solvers but never ported here.

---

## HIGH Findings

### H1. PTC doesn't reset `dt_const` at start of each voltage step
**File:** `Forward/bv_solver/solvers.py:333`
**Agent:** convergence-auditor
**Impact:** Carries over huge dt (~1e6) from previous voltage, defeating PTC's conservative start. Can cause Newton divergence.

### H2. PTC Newton failure leaves inconsistent U/U_prev for next voltage
**File:** `Forward/bv_solver/solvers.py:342`
**Agent:** convergence-auditor
**Impact:** Corrupted state propagates through subsequent voltage steps.

### H3. PTC permanently mutates `dt_const` to ~1e6
**File:** `Forward/bv_solver/solvers.py:363`
**Agent:** numerical-auditor
**Impact:** Corrupts context for any downstream reuse — changes PDE semantics from time-stepping to direct steady-state.

### H4. `io.py` `load_surrogate()` crashes on NN/GP/PCE models
**File:** `Surrogate/io.py:78-81`
**Agents:** surrogate-auditor, interface-auditor
**Impact:** `AttributeError` — unconditional access to `.config.smoothing_cd` which only exists on `BVSurrogateModel`.

### H5. Ensemble std `ddof=1` vs `ddof=0` inconsistency
**Files:** `Surrogate/ensemble.py:100` vs `Surrogate/nn_training.py:677`
**Agent:** surrogate-auditor
**Impact:** 12% disagreement in reported uncertainty between production wrapper and training code. With 1-member ensemble, `ddof=1` produces NaN.

### H6. NN forward pass in float32 limits gradient accuracy
**File:** `Surrogate/nn_model.py:564-577`
**Agent:** numerical-auditor
**Impact:** Gradient accuracy limited to ~1e-4 relative error. L-BFGS-B gets degraded gradients, potentially converging to wrong minima.

### H7. BV pipeline skips failed-point gradients; Robin includes them
**Files:** `FluxCurve/bv_curve_eval.py:72-78` vs `FluxCurve/curve_eval.py:61-66`
**Agent:** fluxcurve-auditor
**Impact:** BV optimizer gets misleading zero gradient with high objective, causing stalling or wild steps.

### H8. `evaluate_curve_loss_forward` missing `observable_species_index`
**File:** `FluxCurve/curve_eval.py:116-154`
**Agent:** fluxcurve-auditor
**Impact:** Crashes or wrong observable when `observable_mode="species"` in Robin pipeline verification.

### H9. Fixed LHS seed (777) across ISMO iterations
**File:** `Surrogate/acquisition.py:89`
**Agent:** ismo-auditor
**Impact:** Space-filling budget generates identical points every iteration; wasted after iteration 1.

### H10. `k0_2_sensitivity_weight` is a no-op
**File:** `Surrogate/acquisition.py:514-525`
**Agent:** ismo-auditor
**Impact:** Config parameter misleads users; k0_2 (hardest parameter) gets no extra exploration despite non-trivial default weight of 2.0.

### H11. ISMO convergence checked at acquisition point, not optimizer's best
**File:** `Surrogate/ismo.py:1430-1432`
**Agent:** ismo-auditor
**Impact:** Surrogate-PDE gap measured at wrong point; can cause premature/delayed convergence.

### H12. ISMO stagnation fires after only 2 iterations
**File:** `Surrogate/ismo.py:1587`
**Agent:** ismo-auditor
**Impact:** Premature termination before surrogate can improve.

### H13. Non-atomic checkpoint writes risk corruption
**File:** `Surrogate/training.py:492-513`
**Agent:** data-io-auditor
**Impact:** Kill during `np.savez_compressed` destroys the only checkpoint, losing hours of training data.

### H14. Race condition on shared CSV in multi-seed runner
**File:** `scripts/studies/run_multi_seed_v13.py:202-218`
**Agent:** data-io-auditor
**Impact:** All seeds write to same hardcoded path; timestamp check fragile on some filesystems.

### H15. `training_data_audit.py` column name mismatch
**File:** `scripts/studies/training_data_audit.py:381-382`
**Agent:** script-auditor
**Impact:** `KeyError` — expects `nn_ensemble_cd_nrmse` but CSV has `cd_nrmse`.

### H16. 15% tolerance for noiseless parameter recovery test
**File:** `tests/test_inverse_verification.py:551-558`
**Agent:** test-auditor
**Impact:** Extremely generous; masks systematic surrogate bias. 5%-noise gate is "informational only".

### H17. `ForwardSolverAdapter` can't discover `forsolve_bv`
**File:** `Inverse/solver_interface.py:119`
**Agent:** interface-auditor
**Impact:** BV solver uses different function name (`forsolve_bv`) and split module architecture. Adapter raises ValueError.

### H18. Monotonicity penalty assumes ascending voltage sort order
**File:** `Surrogate/nn_training.py:96-127`
**Agent:** surrogate-auditor
**Impact:** Penalty direction inverted if voltage grid is descending.

---

## MEDIUM Findings

### M1. Dirichlet solver truncates z_vals to int
**File:** `Forward/dirichlet_solver.py:118`
**Impact:** `int()` cast drops fractional charges. Robin/BV solvers use `float()`.

### M2. Diffusion bounds not validated as log-space
**File:** `Inverse/parameter_targets.py:137`
**Impact:** Users can silently provide physical-space bounds; optimizer expects log-space.

### M3. Post-optimization `rf()` re-evaluation risks exception after success
**File:** `Inverse/inference_runner/__init__.py:96`

### M4. `robin_kappa` apply_value_inplace mutates solver_options dict through aliasing
**File:** `Inverse/parameter_targets.py:186-199`

### M5. Sequential vs parallel checkpoint resume path asymmetry
**File:** `Surrogate/training.py`
**Impact:** Cross-path resume not tested; could cause silent data loss.

### M6. No model version field in surrogate serialization
**File:** `Surrogate/io.py`
**Impact:** Fragile `hasattr` backcompat accumulates tech debt.

### M7. Unshuffled K-fold CV in POD-RBF
**File:** `Surrogate/pod_rbf_model.py:262-308`
**Impact:** Correlated folds give biased smoothing parameter selection.

### M8. Multi-pH mutates `base_solver_params[8]` — fails for SolverParams dataclass
**File:** `FluxCurve/bv_curve_eval.py:448-463`

### M9. `U_prev.assign(U)` annotated on tape every step
**File:** `FluxCurve/point_solve.py:274`
**Impact:** Tape bloat; potential OOM for large problems.

### M10. Sign convention double-negation in BV target generation vs inference
**File:** `FluxCurve/bv_run/io.py:76`, `FluxCurve/bv_observables.py:35-40`
**Impact:** Currently self-consistent but fragile.

### M11. `_clone_params_with_phi` converts SolverParams to plain list
**File:** `Forward/bv_solver/solvers.py:13-17`

### M12. `n_electrons` not propagated into nondim current density scale
**File:** `Nondim/transform.py:406`
**Impact:** Callers must manually multiply; consistency trap.

### M13. Jacobian FD step is absolute, not relative
**File:** `scripts/studies/sensitivity_visualization.py:410-434`
**Impact:** k0_2 (~5e-5) perturbed by ~19%; sensitivity analysis quantitatively wrong.

### M14. Softplus regularization fails for negative concentrations
**File:** `Forward/bv_solver/forms.py:237-238`
**Impact:** `exp(-1e10)` underflows; floor fails during Newton iterations.

### M15. Copy-paste: alpha_2 initial guess uses alpha_1_range
**File:** `scripts/studies/inverse_benchmark_all_models.py:370-371`
**Impact:** Latent — ranges currently identical but will break if they diverge.

### M16. All seeds overwrite same CSV with no per-seed isolation
**File:** `scripts/studies/run_multi_seed_v13.py`

### M17. ISMO augmented data missing `phi_applied` key
**File:** `Surrogate/ismo.py:886-893`

### M18. Two competing ISMO implementations (legacy vs acquisition.py)
**File:** `Surrogate/ismo.py` vs `Surrogate/acquisition.py`
**Impact:** Production `run_ismo()` uses legacy versions, missing improvements.

### M19. No version/shape validation on surrogate model load
**File:** `Surrogate/io.py`, `Surrogate/nn_model.py`

---

## LOW Findings (15 total)

| ID | File | Issue |
|----|------|-------|
| L1 | `Forward/bv_solver/forms.py:208` | Steric check uses exact float `!= 0.0` |
| L2 | `Nondim/transform.py:399` | Debye length name misleading if `potential_scale != V_T` |
| L3 | `scripts/_bv_common.py:159` | `compute_i_scale` relies on implicit operator precedence |
| L4 | `scripts/studies/sensitivity_visualization.py:482` | X-axis labeled "Overpotential" but plots `phi_applied` |
| L5 | `FluxCurve/plot.py:187` | `_LiveFitPlot` hardcodes 2-element kappa display |
| L6 | `FluxCurve/results.py:113` | Gradient default hardcoded to `[0.0, 0.0]` |
| L7 | `scripts/benchmark_autograd_vs_fd.py` | Missing `sys.path` setup |
| L8 | `scripts/verification/mms_bv_4species.py:190` | Dead function has wrong stoichiometry format |
| L9 | `scripts/studies/inverse_benchmark_all_models.py:332` | Hardcoded multistart seed=42 across noise realizations |
| L10 | `Surrogate/nn_model.py:80` | Z-score std floor of 1e-15 is near subnormal |
| L11 | `Surrogate/pce_model.py:519-580` | `predict_gradient` returns physical-space gradient; optimizer expects log-space |
| L12 | `Forward/params.py:90-100` | `__setitem__` breaks frozen dataclass contract |
| L13 | `Nondim/compat.py:21-22` | Hardcoded 2-species defaults |
| L14 | `Surrogate/training.py:199-231` | `np.interp` uses constant extrapolation at boundaries |
| L15 | Multiple test files | Redundant `sys.path` manipulation |

---

## Coverage Gaps (No Tests)

| Code Path | Status |
|-----------|--------|
| `Forward/robin_solver.py` | **No dedicated tests** |
| `Forward/dirichlet_solver.py` | **No dedicated tests** |
| `Forward/plotter.py` | **No tests** |
| `FluxCurve/bv_parallel.py` | **No tests** |
| `FluxCurve/curve_eval.py` | **No tests** |
| `FluxCurve/observables.py` | **No tests** |
| `Inverse/objectives.py` | Indirect only |
| `Surrogate/cascade.py` | **No tests** |
| `Surrogate/training.py` | **No tests** |
| `Forward/bv_solver/solvers.py` | Integration only |
| `add_percent_noise` mode="signal" | **Not tested** |

---

## Recommended Priority Actions

1. **Fix C1** (n_electrons in BV exponent) — verify convention and either use it or remove the config field
2. **Fix C2** (charge continuation checkpoint) — save `fd.Function(ctx["U"])` before each z_factor step
3. **Fix C3** (c_ref_nondim floor) — add `max(c_ref_nondim, 1e-12)` after scaling
4. **Fix C6+C7** (script NameError/KeyError) — trivial fixes that currently crash at runtime
5. **Fix H1-H3** (PTC dt_const) — reset at each voltage step, restore after solve
6. **Fix H4** (load_surrogate) — guard `.config` access with `hasattr`
7. **Fix H5** (ddof consistency) — standardize on `ddof=0` or guard single-member case
8. **Fix C8** (test inverse crime) — assert `n_nan == 0` or cap allowed NaN fraction
9. **Fix H13** (atomic checkpoints) — write to `.tmp` then `os.rename()`
10. **Fix C4+C5** (ISMO matching) — return `converged` mask from PDE eval instead of float matching
