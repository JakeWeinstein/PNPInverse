# Round 2 Bug Audit: Surrogate/ Package

Audited: 2026-03-17
Auditor: Claude Opus 4.6 (automated re-audit)
Scope: All 22 files in `Surrogate/`

---

## Verified Fixes (Round 1 bugs confirmed resolved)

### VF-1: Smoothness penalty split (nn_training.py, line 130-158)
**Status: VERIFIED FIXED.** The `_smoothness_penalty` function now correctly splits predictions into CD and PC halves before computing second-order finite differences. The boundary between output types no longer introduces a spurious penalty.

### VF-2: CSV try/finally (nn_training.py, line 307-421)
**Status: VERIFIED FIXED.** The training loop is wrapped in `try: ... finally: csv_file.close()` ensuring the CSV file handle is always closed even on exceptions.

### VF-3: Double forward pass eliminated (nn_training.py, line 320-331)
**Status: VERIFIED FIXED.** Physics regularization (`_monotonicity_penalty`, `_smoothness_penalty`) now reuses the `pred` tensor from the MSE loss computation rather than running a separate forward pass.

### VF-4: Sobol except -> logger.warning (pce_model.py, line 629-652)
**Status: VERIFIED FIXED.** Bare `except:` blocks in `compute_sobol_indices` replaced with `except Exception:` + `logger.warning(... exc_info=True)`. Exceptions no longer silently swallowed.

### VF-5: load_surrogate broadened type check (io.py, line 67)
**Status: VERIFIED FIXED.** `isinstance` check uses `_SURROGATE_TYPES` tuple containing all 5 model types (BVSurrogateModel, PODRBFSurrogateModel, NNSurrogateModel, GPSurrogateModel, PCESurrogateModel).

### VF-6: GP supports_autograd = False (gp_model.py, line 231)
**Status: VERIFIED FIXED.** `GPSurrogateModel.__init__` sets `self.supports_autograd = False`, preventing the autograd objective path from using the GP's Z-score-space `predict_torch`.

### VF-7: supports_autograd check in _has_autograd (objectives.py, line 36-47)
**Status: VERIFIED FIXED.** `_has_autograd()` now checks both `callable(surrogate.predict_torch)` and `getattr(surrogate, "supports_autograd", True)`, allowing models to explicitly opt out.

### VF-8: Polish cache (multistart.py, line 340-374)
**Status: VERIFIED FIXED.** `_polish_candidate` autograd path implements a `_cache` dict keyed by `x_bytes` to avoid recomputation when L-BFGS-B calls objective and gradient at the same point.

### VF-9: NaN reorder protection (multistart.py, line 253-267)
**Status: VERIFIED FIXED.** `_evaluate_grid_objectives` detects NaN predictions in valid columns before zeroing residuals and marks those rows as `np.inf`, preventing NaN grid points from rising to the top of the sort.

### VF-10: ISMO warm_start default (ismo.py, line 127)
**Status: VERIFIED FIXED.** `ISMOConfig.warm_start_retrain` defaults to `False`, avoiding accidental warm-start when no state dict is available.

### VF-11: Stagnation min_iterations guard (ismo_convergence.py, line 252)
**Status: VERIFIED FIXED.** Stagnation check uses `n >= max(w, c.min_iterations)`, preventing premature stagnation declarations before the minimum iteration threshold.

### VF-12: allow_pickle in ismo_pde_eval.py (line 178, 312, 528, 539)
**Status: VERIFIED FIXED.** `np.load` calls include `allow_pickle=True` where needed for loading arrays with object dtypes.

### VF-13: Relative path fix in ismo_pde_eval.py (line 144-148)
**Status: VERIFIED FIXED.** `make_standard_pde_bundle` resolves `training_data_path` relative to the package root using `os.path.dirname(os.path.abspath(__file__))` rather than relying on CWD.

### VF-14: k0_2 warning in acquisition.py (line 518-523)
**Status: VERIFIED FIXED.** The k0_2_sensitivity_weight issues a `warnings.warn` explaining it is currently a no-op placeholder, rather than silently applying a meaningless uniform scaling.

### VF-15: ISMO deepcopy protection (ismo.py)
**Status: VERIFIED per docstring.** Functions return new model objects; originals are not mutated (confirmed in ismo_retrain.py which creates new model instances).

### VF-16: Train/val/test split in ismo_retrain.py (line 544-548)
**Status: VERIFIED FIXED.** `retrain_nn_ensemble` splits `train_idx` into 85% training / 15% validation subsets, keeping `test_idx` truly held out for quality checks.

---

## NEW Bugs Found

### BUG-N1: `load_surrogate` backward-compat patching crashes on non-RBF models
- **Severity:** MEDIUM
- **File:** `Surrogate/io.py`
- **Lines:** 78-81
- **Description:** The backward-compatibility code unconditionally accesses `model.config.smoothing_cd` and `model.config.smoothing_pc`. However, `NNSurrogateModel`, `GPSurrogateModel`, and `PCESurrogateModel` do not have a `.config` attribute (NN uses `_hidden`/`_n_blocks`, GP has no config object, PCE has `self.config` but a `PCEConfig` which lacks `smoothing_cd`). Loading an NN, GP, or PCE model via `load_surrogate` will crash with `AttributeError`.
- **Fix suggestion:** Guard with `if hasattr(model, 'config') and model.config is not None:` before accessing `smoothing_cd`/`smoothing_pc`, or restrict the check to `BVSurrogateModel`/`PODRBFSurrogateModel` types.

### BUG-N2: `nn_training.py` references `epoch` variable after loop may not execute
- **Severity:** LOW
- **File:** `Surrogate/nn_training.py`
- **Lines:** 418, 445
- **Description:** After the `finally` block closes the CSV file, lines 418 and 445 reference the `epoch` variable (for plotting and verbose output). If `config.epochs` is 0, the loop body never executes and `epoch` is undefined, causing `UnboundLocalError`. This is an edge case since epochs=0 is not a practical configuration, but it is a latent bug.
- **Fix suggestion:** Initialize `epoch = 0` before the training loop at line ~307.

### BUG-N3: Cascade autograd path recomputes on every call (no caching)
- **Severity:** LOW
- **File:** `Surrogate/cascade.py`
- **Lines:** 194-226
- **Description:** Unlike `multistart.py` which now caches the `(J, grad)` result to avoid recomputation when L-BFGS-B calls objective and gradient at the same x, the cascade's `_make_subset_objective_fn` autograd path does not cache. Each L-BFGS-B step calls `_autograd_obj_and_grad` twice (once via `_objective`, once via `_gradient`), doubling the forward+backward passes.
- **Fix suggestion:** Add an `_cache` dict (same pattern as multistart.py line 340) to `_make_subset_objective_fn` and `_make_subset_block_objective_fn`.

### BUG-N4: `EnsembleMeanWrapper` std with ddof=1 fails for single-member ensemble
- **Severity:** LOW
- **File:** `Surrogate/ensemble.py`
- **Line:** 100-102
- **Description:** `_predict_ensemble_raw` computes `np.std(..., ddof=1)`. With a single-member ensemble (E=1), this produces `NaN` for all std values. While `EnsembleMeanWrapper` typically has 5 members, ISMO's `_retrain_single_nn` wraps a single model in a 1-member ensemble, and `predict_with_uncertainty` on that wrapper would return NaN std.
- **Fix suggestion:** Use `ddof=0` when `len(self.models) == 1`, or guard `predict_with_uncertainty` to raise a clear error for single-member ensembles.

### BUG-N5: `_evaluate_grid_objectives` mutates input `target_cd`/`target_pc` when subset_idx is None
- **Severity:** NEGLIGIBLE
- **File:** `Surrogate/multistart.py`
- **Lines:** 238-242
- **Description:** When `subset_idx is not None`, the function rebinds `target_cd = target_cd[subset_idx]` and `target_pc = target_pc[subset_idx]`, creating new arrays. When `subset_idx is None`, the original references are used. This is not a mutation bug per se (no in-place modification), but the asymmetric rebinding pattern is fragile. Currently safe.
- **Fix suggestion:** No action needed; documenting for awareness.

### BUG-N6: `io.py` `load_surrogate` prints to stdout unconditionally
- **Severity:** LOW
- **File:** `Surrogate/io.py`
- **Lines:** 83-85
- **Description:** `load_surrogate` uses `print()` for status messages rather than `logging`. This is inconsistent with `ismo_retrain.py`, `ismo_pde_eval.py`, `acquisition.py`, and `pce_model.py` which all use `logger`. In library code, `print` to stdout can interfere with callers that capture stdout.
- **Fix suggestion:** Replace `print()` calls with `logger.info()` (add a module-level logger).

### BUG-N7: `acquisition.py` redundant `import warnings` inside function
- **Severity:** NEGLIGIBLE
- **File:** `Surrogate/acquisition.py`
- **Lines:** 29, 519
- **Description:** `warnings` is imported at module level (line 29) and again inside `_acquire_uncertainty` (line 519). The inner import is harmless but unnecessary.
- **Fix suggestion:** Remove the inner `import warnings` at line 519.

---

## REMAINING Bugs (Missed in Round 1)

### BUG-R1: `ismo_pde_eval.py` `_compute_nrmse_with_reference_range` uses per-sample ptp despite docstring claiming reference_range
- **Severity:** LOW
- **File:** `Surrogate/ismo_pde_eval.py`
- **Lines:** 354-389
- **Description:** The docstring was updated to clarify that `reference_range` is only used for the floor, but the function name `_compute_nrmse_with_reference_range` is misleading. It actually normalizes per-sample by `ptp(truth[i])`, not by the reference range. The `compare_surrogate_vs_pde` function passes `cd_reference_range` / `pc_reference_range` as arguments, but they only control the floor. This makes the "convergence_threshold" in `compare_surrogate_vs_pde` batch-dependent (different sample compositions yield different NRMSE values), which could cause flaky convergence decisions.
- **Fix suggestion:** Consider offering a mode that actually normalizes by the fixed reference range for stable convergence checking, or rename the function to `_compute_nrmse_per_sample_range`.

### BUG-R2: `ismo_retrain.py` `retrain_surrogate_full` dispatch misses PCE when GP import succeeds
- **Severity:** MEDIUM
- **File:** `Surrogate/ismo_retrain.py`
- **Lines:** 1207-1234
- **Description:** In `retrain_surrogate_full`, the `else` branch (line 1207) tries GP first, and only falls through to PCE if the GP import fails. But if `GPSurrogateModel` imports successfully and the surrogate is a `PCESurrogateModel`, the code reaches `raise TypeError` (line 1217) because `isinstance(surrogate, GPSurrogateModel)` is False. The PCE path is only reached if the GP import itself raises `ImportError`. The simpler `retrain_surrogate` function (line 1016) handles this correctly by trying both imports independently.
- **Fix suggestion:** Restructure the `else` branch to check PCE independently of GP import success, matching the pattern in `retrain_surrogate`.

### BUG-R3: `ismo.py` `_acquire_uncertainty` loops over 5000 points with single-point predict
- **Severity:** LOW (performance)
- **File:** `Surrogate/ismo.py`
- **Lines:** 479-493
- **Description:** When using uncertainty-based acquisition in the old ISMO path, the function evaluates 5000 LHS points one at a time via `predict_with_uncertainty(pt[0], pt[1], pt[2], pt[3])` rather than using `predict_batch_with_uncertainty`. For GP models this is extremely slow (5000 separate Cholesky decompositions).
- **Fix suggestion:** Use `predict_batch_with_uncertainty` for models that support it, falling back to per-point evaluation only when needed.

### BUG-R4: `pce_model.py` `predict_batch` returns 1D array for single-sample input
- **Severity:** LOW
- **File:** `Surrogate/pce_model.py`
- **Lines:** 502-507
- **Description:** When `predict_batch` is called with a single row (M=1), `chaospy` polynomial evaluation may return a scalar instead of a (1,) array. `np.column_stack` of scalars produces a 1D array of shape `(n_eta,)` rather than `(1, n_eta)`. This would break callers that expect a 2D array. All other model types (`NNSurrogateModel`, `GPSurrogateModel`, `BVSurrogateModel`) correctly return (1, n_eta).
- **Fix suggestion:** Add `cd = np.atleast_2d(cd)` and `pc = np.atleast_2d(pc)` before the return.

### BUG-R5: `gp_model.py` `training_bounds` is a property on GPSurrogateModel but plain attribute on others
- **Severity:** NEGLIGIBLE
- **File:** `Surrogate/gp_model.py`
- **Line:** 742
- **Description:** `GPSurrogateModel` exposes `training_bounds` as a `@property` that returns `self._training_bounds`, while all other model types use a plain attribute `self.training_bounds`. This makes `hasattr(model, 'training_bounds')` return True for both, but assignment `model.training_bounds = X` would fail on GPSurrogateModel (property has no setter). The backward-compat code in `io.py` line 75 (`model.training_bounds = None`) would crash for GP models loaded via the generic `load_surrogate`.
- **Fix suggestion:** Add a setter to the `training_bounds` property on `GPSurrogateModel`, or change to a plain attribute.

---

## Summary

| Category | Count |
|----------|-------|
| Verified fixes from round 1 | 16 |
| New bugs introduced by fixes | 7 (1 MEDIUM, 3 LOW, 3 NEGLIGIBLE) |
| Remaining bugs missed in round 1 | 5 (1 MEDIUM, 3 LOW, 1 NEGLIGIBLE) |
| **Total new findings** | **12** |

### Priority items requiring action:
1. **BUG-N1** (MEDIUM): `load_surrogate` crashes on NN/GP/PCE models due to unconditional `model.config.smoothing_cd` access
2. **BUG-R2** (MEDIUM): `retrain_surrogate_full` PCE dispatch unreachable when GP is installed
3. **BUG-R5** (interacts with BUG-N1): GP `training_bounds` property incompatible with `io.py` backward-compat assignment
