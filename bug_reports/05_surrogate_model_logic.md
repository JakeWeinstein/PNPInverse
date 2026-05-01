# Bug Report: Surrogate Model Logic Bugs

**Focus:** Array shapes, NaN propagation, ensemble, training in Surrogate/
**Agent:** Surrogate Model Code Quality (logic focus)

---

## BUG 1: Smoothness penalty crosses CD/PC boundary
**File:** `Surrogate/nn_training.py:155-159`
**Severity:** MEDIUM
**Description:** The smoothness penalty computes second-order finite differences across the entire output vector `pred[:, 2:] - 2*pred[:, 1:-1] + pred[:, :-2]`. Since the output is `[CD(n_eta) | PC(n_eta)]` concatenated, the differences at the CD/PC boundary mix current density values with peroxide current values, producing physically meaningless penalty terms.
**Suggested fix:** Split `pred` into CD and PC portions before computing differences.

## BUG 2: Double forward pass in physics regularization
**File:** `Surrogate/nn_training.py:322-333`
**Severity:** MEDIUM
**Description:** Both `_monotonicity_penalty()` and `_smoothness_penalty()` call `model(X_batch)` again internally (lines 124, 156), even though `pred = net(xb)` was already computed. The penalty is computed on a fresh forward pass rather than the same predictions used for MSE loss.
**Suggested fix:** Pass the pre-computed `pred` tensor to penalty functions.

## BUG 3: NaN predictions silently zeroed in grid evaluation
**File:** `Surrogate/multistart.py:253-254`
**Severity:** HIGH
**Description:** NaN residuals are replaced with 0.0 before summing. NaN-producing parameter combinations contribute zero to the objective instead of a penalty. The later NaN check (lines 262-264) operates on the original (non-subsetted) predictions, creating an inconsistency.
**Suggested fix:** Remove the `np.where(np.isnan(...), 0.0, ...)` lines. Use `np.where(np.isnan(objectives), np.inf, objectives)` as the final step.

## BUG 4: Bogus voltage grid validation in `integrate_new_data`
**File:** `Surrogate/ismo_pde_eval.py:532`
**Severity:** HIGH
**Description:** `np.allclose(old_phi, pde_result.current_density.shape[1] and old_phi, atol=0)` is nonsensical. Due to Python operator precedence, `shape[1] and old_phi` evaluates to `old_phi`, making this `np.allclose(old_phi, old_phi)` -- always True. The validation is a no-op.
**Suggested fix:** Replace with a proper grid comparison or remove entirely (length check on lines 536-540 partially covers this).

## BUG 5: GP `predict_torch` API mismatch with NN interface
**File:** `Surrogate/gp_model.py:615-649`
**Severity:** HIGH
**Description:** `GPSurrogateModel.predict_torch` expects Z-score normalized input and returns Z-score normalized output. `NNSurrogateModel.predict_torch` expects log-space input and returns physical-space output. The `SurrogateObjective` autograd path detects `predict_torch` exists and uses it, but passes raw log-space inputs. A GP model used with these objectives would silently produce wrong gradients.
**Suggested fix:** Make GP's `predict_torch` match the NN API, or add a check that excludes GP models from the autograd path.

## BUG 6: Test set used as validation during ISMO retraining (data leakage)
**File:** `Surrogate/ismo_retrain.py:596-606`
**Severity:** HIGH
**Description:** Test indices are used to create `val_params/val_cd/val_pc`, then passed to `NNSurrogateModel.fit()` for early stopping. The test set is used for model selection during retraining. After retraining, the quality check also uses the same test set, making it unreliable.
**Suggested fix:** Split training indices into train/val subsets for fine-tuning, keeping test indices truly held out.

## BUG 7: `EnsembleMeanWrapper` uses `ddof=1` while `nn_training.predict_ensemble` uses `ddof=0`
**File:** `Surrogate/ensemble.py:100` vs `Surrogate/nn_training.py:685`
**Severity:** LOW
**Description:** Inconsistent standard deviation computation. With 5 ensemble members, the ~12% difference in uncertainty estimates could affect acquisition decisions.
**Suggested fix:** Standardize on `ddof=1` (sample std).

## BUG 8: `load_surrogate` only accepts `BVSurrogateModel` type
**File:** `Surrogate/io.py:48`
**Severity:** MEDIUM
**Description:** `isinstance(model, BVSurrogateModel)` check rejects valid surrogate types like `PODRBFSurrogateModel`, `GPSurrogateModel`, `PCESurrogateModel`, and `EnsembleMeanWrapper`.
**Suggested fix:** Broaden the isinstance check or use duck typing.

## BUG 9: Bare `except` clauses in PCE Sobol computation
**File:** `Surrogate/pce_model.py:630, 639`
**Severity:** MEDIUM
**Description:** Two bare `except Exception` clauses silently swallow errors in Sobol index computation with `pass` body. Failed Sobol indices are left as zeros with no indication.
**Suggested fix:** Log the exception at warning level.

## BUG 10: CSV file resource leak in training
**File:** `Surrogate/nn_training.py:283`
**Severity:** MEDIUM
**Description:** CSV file opened without context manager. If training throws an exception, the file handle leaks and data may be unflushed.
**Suggested fix:** Use `with open(...) as csv_file:` or try/finally.

## BUG 11: `objective_and_gradient` double-counts evals on FD path
**File:** `Surrogate/objectives.py:247-251`
**Severity:** LOW
**Description:** On the FD path, `n_evals` over-reports by 1 per `objective_and_gradient` call. Bookkeeping issue only.

## BUG 12: Pickle serialization fragile for RBF models across scipy versions
**File:** `Surrogate/io.py:28-29`
**Severity:** MEDIUM
**Description:** `RBFInterpolator` internal representation can change across scipy versions, causing pickle.load to fail or produce incorrect predictions.
**Suggested fix:** Consider saving training data and re-fitting on load, or pin scipy version.

---

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 0     |
| HIGH     | 4     |
| MEDIUM   | 6     |
| LOW      | 2     |

**Top priority:** Bug 5 (GP autograd mismatch), Bug 6 (data leakage), Bug 3 (NaN zeroing), Bug 4 (broken validation).
