# Bug Report: ISMO Module Correctness

**Focus:** All bug types in ISMO (Iterative Surrogate Model Optimization) components
**Agent:** ISMO Module Correctness

---

## BUG 1: Bogus voltage grid validation (always True)
**File:** `Surrogate/ismo_pde_eval.py:532`
**Severity:** CRITICAL
**Description:** `np.allclose(old_phi, pde_result.current_density.shape[1] and old_phi, atol=0)` always evaluates to True due to Python `and` operator precedence. Voltage grid mismatches go undetected, enabling silent data corruption during ISMO data integration.
**Suggested fix:** Remove broken check or replace with proper comparison.

## BUG 2: PDE failure parameter mismatch in `run_ismo.py`
**File:** `scripts/surrogate/run_ismo.py:508-537`
**Severity:** HIGH
**Description:** When PDE solves fail mid-loop, `n_pde_failures` increments and the result is skipped via `continue`, but `new_params = new_params[:n_successful]` trims from the FRONT. Failures at intermediate indices break parameter-to-curve correspondence.
**Suggested fix:** Track successful indices explicitly and filter `new_params` by those indices.

## BUG 3: `retrain_surrogate` called with wrong signature in runner
**File:** `scripts/surrogate/run_ismo.py:551-553`
**Severity:** HIGH
**Description:** Calls `retrain_surrogate(surrogate, params_train, cd_train, pc_train, phi_applied)` (5 positional args) but the actual function expects `(surrogate, new_data, existing_data, config, iteration, train_idx, test_idx, device)` (8 args with dicts). Will produce TypeError at runtime.
**Suggested fix:** Match the correct function signature.

## BUG 4: Wrong module names in ISMO runner imports
**File:** `scripts/surrogate/run_ismo.py:73-76, 84`
**Severity:** HIGH (silent fallback to inferior code paths)
**Description:** Tries `from Surrogate.ismo_acquisition import ...` but module is `Surrogate.acquisition`. Tries `from Surrogate.ismo_pde import ...` but module is `Surrogate.ismo_pde_eval`. Import always fails silently, causing fallback to LHS sampling instead of hybrid acquisition.
**Suggested fix:** Fix import paths to `Surrogate.acquisition` and `Surrogate.ismo_pde_eval`.

## BUG 5: NN ensemble warm-start without normalizer correction
**File:** `Surrogate/ismo.py:1020-1037`
**Severity:** HIGH
**Description:** Old state dict passed directly as warm_start without the analytical weight correction for normalizer shift (implemented in `ismo_retrain._correct_weights_for_normalizer_shift`). When training data changes, normalizers are recomputed, making old weights inconsistent. Can cause catastrophic quality degradation.
**Suggested fix:** Port normalizer correction from `ismo_retrain` or use `ismo_retrain.retrain_nn_ensemble`.

## BUG 6: In-place mutation of non-NN surrogates during retraining
**File:** `Surrogate/ismo.py:948-955`
**Severity:** HIGH
**Description:** For non-NN models, `surrogate.fit(...)` is called directly on original object, mutating it in place. Pre-retrain state is lost; no rollback if retraining degrades model. Contrasts with NN ensemble path which creates new objects.
**Suggested fix:** Deep copy before calling `.fit()`.

## BUG 7: Stagnation detection fires too early
**File:** `Surrogate/ismo_convergence.py:251-265`
**Severity:** HIGH
**Description:** No `min_iterations` guard on stagnation check. With default `stagnation_window=3` and `min_iterations=2`, stagnation fires at iteration 3 before meaningful improvement. Flat initial disagreement triggers premature termination.
**Suggested fix:** Add `if n >= max(w, c.min_iterations):`.

## BUG 8: `_compute_pde_loss` uses wrong target when `subset_idx` provided
**File:** `Surrogate/ismo.py:582-602`
**Severity:** HIGH
**Description:** Simulated curves are subsetted (`cd_s = cd_sim[subset_idx]`) but target curves are NOT. Shape mismatch or silent broadcasting error if caller passes full-length targets.
**Suggested fix:** Apply `subset_idx` to both simulated and target arrays.

## BUG 9: `k0_2_sensitivity_weight` is a no-op
**File:** `Surrogate/acquisition.py:518-519`
**Severity:** MEDIUM
**Description:** Creates `k0_2_sensitivity = np.ones(n_candidates)`, then multiplies scores uniformly. Does not change ranking. Config parameter exposed to users has zero effect.
**Suggested fix:** Implement actual sensitivity weighting or remove parameter.

## BUG 10: Misleading log message after GP redistribution
**File:** `Surrogate/acquisition.py:635-644`
**Severity:** MEDIUM
**Description:** `n_unc` is zeroed before the log message, so it logs "0 -> optimizer" instead of the actual count redistributed.
**Suggested fix:** Capture original `n_unc` before zeroing.

## BUG 11: `_compute_nrmse_with_reference_range` doesn't use reference_range as denominator
**File:** `Surrogate/ismo_pde_eval.py:346-374`
**Severity:** MEDIUM
**Description:** Despite name and docstring, `reference_range` is only used as a 1% floor. Per-sample ptp still dominates. Inconsistent with `_nrmse` in `ismo_convergence.py`.
**Suggested fix:** If fixed-range normalization intended, use `reference_range` directly as denominator.

## BUG 12: Uncertainty acquisition loops 5000 times individually
**File:** `Surrogate/ismo.py:479-493`
**Severity:** MEDIUM (performance)
**Description:** Calls `predict_with_uncertainty` per candidate point instead of batched. Extremely slow for GP models.
**Suggested fix:** Use `predict_batch_with_uncertainty(lhs_pts)`.

## BUG 13: Runner uses first test sample as target for all iterations
**File:** `scripts/surrogate/run_ismo.py:351-353`
**Severity:** LOW
**Description:** `target_cd = test_cd[0]` may be trivially solvable if the point is in or near the training set.

---

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 1     |
| HIGH     | 7     |
| MEDIUM   | 4     |
| LOW      | 1     |

**Immediate action items:**
1. Bug 1 -- broken validation enables silent data corruption
2. Bugs 3, 4 -- wrong imports/signatures make ISMO runner non-functional for real acquisition/retrain
3. Bug 2 -- parameter-curve misalignment on PDE failures
4. Bug 5 -- warm-start without normalizer correction degrades quality
