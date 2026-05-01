# Bug Report: Surrogate Code Quality

**Focus:** Python code quality bugs in Surrogate/ package
**Agent:** Surrogate Code Quality

---

## BUG 1: Bare `except` clauses in PCE Sobol computation
**File:** `Surrogate/pce_model.py:630, 639`
**Severity:** MEDIUM
**Description:** Two bare `except Exception` clauses silently swallow errors with only `pass` body. Failed Sobol indices left as zeros with no logging.
**Suggested fix:** Log at `logger.warning` level.

## BUG 2: Bare `except` in ismo_retrain.py
**File:** `Surrogate/ismo_retrain.py:664`
**Severity:** LOW
**Description:** `except Exception:` catches all exceptions when retraining. Does log traceback, so practical impact is low.

## BUG 3: CSV file resource leak
**File:** `Surrogate/nn_training.py:283`
**Severity:** HIGH
**Description:** CSV file opened at line 283 but only closed at line 423. Not protected by try/finally. If training is interrupted, file handle leaked with potentially unflushed data.
**Suggested fix:** Use context manager or try/finally.

## BUG 4: Double-counting of `_n_evals` in SurrogateObjective
**File:** `Surrogate/objectives.py:198, 214-216, 234-251`
**Severity:** MEDIUM
**Description:** When autograd is NOT used, `objective_and_gradient()` inflates eval counter. Same issue in `AlphaOnlySurrogateObjective` and `ReactionBlockSurrogateObjective`.

## BUG 5: `_has_autograd` cached at construction time may become stale
**File:** `Surrogate/objectives.py:97, 298, 474, 702`
**Severity:** LOW
**Description:** Flag set once at init. If surrogate replaced during ISMO, flag would be stale. In practice objectives are constructed fresh.

## BUG 6: Bogus validation expression in `integrate_new_data`
**File:** `Surrogate/ismo_pde_eval.py:532`
**Severity:** HIGH
**Description:** `np.allclose(old_phi, pde_result.current_density.shape[1] and old_phi, atol=0)` is always True due to Python operator precedence. Validation is a complete no-op.
**Suggested fix:** Replace with proper grid comparison.

## BUG 7: Global state mutation in `_TRAIN_WORKER_STATE`
**File:** `Surrogate/training.py:572, 595`
**Severity:** LOW
**Description:** Module-level mutable dict used by ProcessPoolExecutor initializers. Correct for multiprocessing spawn context, but unsafe if ever used in threads.

## BUG 8: Double forward pass in physics regularization
**File:** `Surrogate/nn_training.py:124, 155, 326-333`
**Severity:** MEDIUM
**Description:** Model forward pass computed 2-3 times per batch when physics penalties enabled. Performance bug.
**Suggested fix:** Pass pre-computed predictions to penalty functions.

## BUG 9: `__init__.py` missing exports for diagnostics module
**File:** `Surrogate/__init__.py`
**Severity:** LOW
**Description:** `ISMODiagnostics` and plotting functions from `ismo_diagnostics` not exported. Likely intentional.

## BUG 10: `load_surrogate()` only checks `BVSurrogateModel` type
**File:** `Surrogate/io.py:48`
**Severity:** MEDIUM
**Description:** Rejects valid surrogate types like `PODRBFSurrogateModel`, `GPSurrogateModel`, etc.
**Suggested fix:** Broaden isinstance check or use duck typing.

## BUG 11: `predict_ensemble` ddof inconsistency
**File:** `Surrogate/nn_training.py:683` vs `Surrogate/ensemble.py:100`
**Severity:** LOW
**Description:** `ddof=0` (population) vs `ddof=1` (sample). ~12% difference in uncertainty with 5 members.
**Suggested fix:** Standardize on `ddof=1`.

## BUG 12: Dead import guard in ismo_convergence.py
**File:** `Surrogate/ismo_convergence.py:37-41`
**Severity:** LOW
**Description:** try/except ImportError for `Surrogate.ismo` is dead code -- module always available.

---

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 0     |
| HIGH     | 2     |
| MEDIUM   | 4     |
| LOW      | 6     |

**Top priority:** Bug 3 (CSV resource leak), Bug 6 (bogus validation no-op).
