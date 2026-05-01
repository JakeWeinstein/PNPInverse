# Round 2 Integration Audit: Cross-Cutting Issues

**Date**: 2026-03-17
**Auditor**: Claude Opus 4.6 (1M context)
**Scope**: All modified files across ~40 files with ~53 bug fixes. Checking import chains, API contract breaks, semantic conflicts, and missing propagation.

---

## 1. Import Chain Breaks

### All Clear

All import chains across modified files resolve correctly:

- **`Forward/bv_solver/forms.py`** imports `build_model_scaling`, `_get_nondim_cfg`, `_as_list`, `_bool`, `_pos` from `Nondim.transform` -- all five symbols verified present at `Nondim/transform.py` lines 90, 102, 109, 121, 166.
- **`Forward/bv_solver/forms.py`** imports `_get_bv_cfg`, `_get_bv_convergence_cfg`, `_get_bv_reactions_cfg` from `.config` -- all three verified at `Forward/bv_solver/config.py` lines 14, 43, 81.
- **`Surrogate/__init__.py`** imports from 12 submodules (`sampling`, `surrogate_model`, `training`, `objectives`, `validation`, `io`, `bcd`, `multistart`, `cascade`, `ensemble`, `gp_model`, `pce_model`, `ismo`, `acquisition`, `ismo_retrain`, `ismo_pde_eval`, `ismo_convergence`). All modules exist on disk; all imported symbols verified present in their respective files.
- **`Surrogate/io.py`** imports `BVSurrogateModel`, `PODRBFSurrogateModel`, `NNSurrogateModel`, `GPSurrogateModel`, `PCESurrogateModel` -- all five model classes verified present in their respective modules.
- **`Surrogate/multistart.py`** imports `_has_autograd` from `Surrogate.objectives` -- verified present at line 36.
- **`Surrogate/cascade.py`** imports `_has_autograd` from `Surrogate.objectives` -- verified present at line 36.
- **`Surrogate/bcd.py`** imports `ReactionBlockSurrogateObjective`, `SurrogateObjective` from `Surrogate.objectives` -- both verified present.
- **`Surrogate/nn_training.py`** imports `NNSurrogateModel`, `ResNetMLP`, `ZScoreNormalizer`, `_check_torch` from `Surrogate.nn_model` -- all four verified present at lines 37, 130, 52, 173.
- **`Inverse/__init__.py`** re-exports from `solver_interface`, `parameter_targets`, `inference_runner`, `objectives` -- all verified.
- **`Inverse/inference_runner/__init__.py`** re-exports `build_reduced_functional` from `.objective` -- verified at `objective.py` line 19.
- **`FluxCurve/bv_curve_eval.py`** imports `solve_bv_curve_points_with_warmstart` from `FluxCurve.bv_point_solve` and `_solve_cached_fast_path_parallel_multi_obs`, `_clear_caches`, `_parallel_pool`, `_cache_populated`, `_all_points_cache` -- all verified exported from `FluxCurve/bv_point_solve/__init__.py`.

No broken import chains found.

---

## 2. API Contract Breaks

### 2a. `_monotonicity_penalty` and `_smoothness_penalty` signature change
**Status**: ALL CLEAR

Both functions in `Surrogate/nn_training.py` now take `pred` (pre-computed model output tensor) instead of `(model, X_batch)`. The only callers are within the same file (`nn_training.py` lines 325-327 and 329-331), and both call sites pass the correct new signature:
```python
_monotonicity_penalty(pred, n_eta, phi_applied)
_smoothness_penalty(pred, n_eta)
```
No external callers exist -- verified by searching the entire codebase.

### 2b. `evaluate_curve_loss_forward` -- new `observable_mode` parameter
**Status**: ALL CLEAR

The function at `FluxCurve/curve_eval.py` line 116 now accepts `observable_mode: str = "total_species"`. Both callers in `FluxCurve/run.py` (lines 452 and 659) do NOT pass `observable_mode`, which means they receive the default `"total_species"`. This is correct behavior for the Robin flux curve pipeline -- the robin solver's observable is always total species flux.

The BV pipeline uses a separate function `evaluate_bv_curve_objective_and_gradient` in `FluxCurve/bv_curve_eval.py` which has its own `observable_mode` handling via the `request` object.

### 2c. `build_reduced_functional` -- new `phi_target=None` parameter
**Status**: ALL CLEAR

The function at `Inverse/inference_runner/objective.py` line 19 accepts `phi_target: Optional[Sequence[float]]`. All callers handle this correctly:

- `Inverse/objectives.py` line 48: `make_diffusion_objective_and_grad` passes `phi_target=None` when phi is not in objective_fields (which is always the case for diffusion). Lines 90, 129: other factories pass `phi_target=phi_vec` explicitly.
- `Inverse/inference_runner/recovery.py` line 73: `resilient_minimize` passes `phi_target=phi_target` which comes from the caller's context (can be None for diffusion targets).

The function correctly handles `phi_target=None` at lines 50-54 (sets `phi_target_f = None`) and lines 69-73 (raises `ValueError` if phi is requested but target is None).

### 2d. `_get_bv_convergence_cfg` -- new defaults
**Status**: ALL CLEAR

The function at `Forward/bv_solver/config.py` line 43 now returns complete default dicts with `packing_floor` and `softplus_regularization` in all code paths (non-dict params, non-dict raw bv_convergence, and parsed path). The sole consumer is `Forward/bv_solver/forms.py` line 100 (`conv_cfg = _get_bv_convergence_cfg(params)`), which accesses:
- `conv_cfg["use_eta_in_bv"]` (line 190) -- present in all paths
- `conv_cfg["clip_exponent"]` (line 198) -- present in all paths
- `conv_cfg["exponent_clip"]` (line 199) -- present in all paths
- `conv_cfg["regularize_concentration"]` (line 231) -- present in all paths
- `conv_cfg["conc_floor"]` (line 232) -- present in all paths
- `conv_cfg.get("softplus_regularization", False)` (line 230) -- present, but also uses `.get()` with default as defense
- `conv_cfg.get("packing_floor", 1e-8)` (line 213) -- present, but also uses `.get()` with default as defense

All keys are consistent across the three return paths in `_get_bv_convergence_cfg`. No KeyError risk.

### 2e. `load_surrogate` -- broadened type check
**Status**: ALL CLEAR (with one stale test comment noted below)

The `_SURROGATE_TYPES` tuple at `Surrogate/io.py` line 21 now includes all five model types: `BVSurrogateModel`, `PODRBFSurrogateModel`, `NNSurrogateModel`, `GPSurrogateModel`, `PCESurrogateModel`. All five are imported at lines 14-18. All imports resolve to existing classes.

The backward-compatibility patches at lines 74-81 (adding `training_bounds`, `smoothing_cd`, `smoothing_pc` to older pickles) are safe -- they use `hasattr` checks before patching.

---

## 3. Semantic Conflicts

### All Clear

No two fixes touch the same code region in conflicting ways. Specifically:

- The `_get_bv_convergence_cfg` default-dict fix and the `forms.py` `.get()` defense-in-depth are complementary, not conflicting.
- The `_monotonicity_penalty`/`_smoothness_penalty` signature change and the training loop caller update were done atomically in `nn_training.py`.
- The `build_reduced_functional` phi_target fix and the `Inverse/objectives.py` factory phi_target=None change work together correctly.

---

## 4. Missing Propagation

### ISSUE-1: Stale test bypass of `load_surrogate` for POD-RBF models

- **Severity**: LOW
- **File**: `tests/test_surrogate_fidelity.py`, lines 205-207
- **Description**: The test comment says "POD-RBF models are PODRBFSurrogateModel (not a BVSurrogateModel subclass) -- Use direct pickle loading to avoid isinstance check" and uses a custom `_load_pickle_model()` function instead of `load_surrogate()`. However, `Surrogate/io.py` was updated to include `PODRBFSurrogateModel` in `_SURROGATE_TYPES`, so `load_surrogate()` now handles POD-RBF models correctly. The test's bypass is no longer necessary and misses the backward-compatibility patches that `load_surrogate` applies (training_bounds, smoothing_cd, smoothing_pc).
- **Fix suggestion**: Replace `_load_pickle_model(_POD_RBF_LOG_PATH)` and `_load_pickle_model(_POD_RBF_NOLOG_PATH)` calls with `load_surrogate(path)` calls. Remove the `_load_pickle_model` helper function.

### ISSUE-2: `evaluate_curve_loss_forward` does not propagate `observable_species_index` or `observable_scale` to the sweep

- **Severity**: LOW
- **File**: `FluxCurve/curve_eval.py`, lines 116-154
- **Description**: The function accepts `observable_mode` and `observable_scale` but does NOT accept `observable_species_index` (which exists on `RobinFluxCurveInferenceRequest`). The full curve evaluation function `evaluate_curve_objective_and_gradient` (lines 16-113 in the same file) passes `observable_species_index` from the request to the per-point solver, but the forward-only loss function does not. This means the forward loss fallback in `FluxCurve/run.py` always uses the default species index. For the Robin pipeline this is fine (default works), but this is a latent asymmetry between the adjoint path and the forward-only path.
- **Fix suggestion**: Add `observable_species_index: Optional[int] = None` parameter to `evaluate_curve_loss_forward` and forward it to `steady.species_index` before the sweep.

### ISSUE-3: `FluxCurve/run.py` callers of `evaluate_curve_loss_forward` don't propagate `observable_mode`

- **Severity**: LOW
- **File**: `FluxCurve/run.py`, lines 452-461 and 659-668
- **Description**: Both call sites in `FluxCurve/run.py` do NOT pass the `observable_mode` parameter to `evaluate_curve_loss_forward`, relying on the default `"total_species"`. However, the runtime request may have a different `observable_mode` (e.g., `"peroxide_current"`). The forward-only loss evaluation will always use `"total_species"`, potentially producing incorrect loss values when the optimizer is targeting a different observable.
  - At line 452, this is in a fallback path (`if not np.isfinite(best_loss)`), so it may rarely fire.
  - At line 659, this is the final loss evaluation, which is always executed. If `request_runtime.observable_mode != "total_species"`, the reported `final_loss` would be computed with the wrong observable.
- **Fix suggestion**: Pass `observable_mode=str(request.observable_mode)` (or `request_runtime.observable_mode`) to both `evaluate_curve_loss_forward` calls in `FluxCurve/run.py`.

---

## 5. Summary

| Check Category | Result |
|---|---|
| Import chain breaks | ALL CLEAR -- 0 issues |
| API contract breaks (`_monotonicity_penalty`/`_smoothness_penalty`) | ALL CLEAR -- callers updated |
| API contract breaks (`evaluate_curve_loss_forward`) | 1 LOW (observable_mode not propagated by callers) |
| API contract breaks (`build_reduced_functional`) | ALL CLEAR -- phi_target handled correctly |
| API contract breaks (`_get_bv_convergence_cfg`) | ALL CLEAR -- forms.py works with new defaults |
| API contract breaks (`load_surrogate`) | ALL CLEAR -- type check broadened, imports available |
| Semantic conflicts | ALL CLEAR -- 0 conflicts |
| Missing propagation | 3 issues (all LOW severity) |

### Actionable Items (by priority)

1. **LOW -- ISSUE-3**: Pass `observable_mode` from the runtime request to `evaluate_curve_loss_forward` in `FluxCurve/run.py` lines 452 and 659. This ensures the final forward loss evaluation uses the same observable as the optimizer.

2. **LOW -- ISSUE-2**: Add `observable_species_index` parameter to `evaluate_curve_loss_forward` to maintain parity with the adjoint evaluation path.

3. **LOW -- ISSUE-1**: Update `tests/test_surrogate_fidelity.py` to use `load_surrogate()` for POD-RBF models instead of the custom `_load_pickle_model()` bypass, now that `load_surrogate` accepts all five surrogate types.
