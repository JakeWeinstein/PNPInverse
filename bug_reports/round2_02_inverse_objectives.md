# Round 2 Bug Audit: Inverse/ and Surrogate/objectives.py

Auditor: Claude Opus 4.6 (1M context)
Date: 2026-03-17
Scope: All files in `Inverse/`, `Surrogate/objectives.py`, `Surrogate/multistart.py`

---

## Verified Fixes

### VF-1: Inverse/objectives.py -- phi placeholder replaced with None
- **File:** `Inverse/objectives.py`, line 46
- **Status:** VERIFIED CORRECT
- The diffusion target now correctly sets `phi_target = None` when `"phi"` is not in the target's `objective_fields`. Previously this was a placeholder value. The conditional logic `None if "phi" not in target.objective_fields else c_targets[0]` is correct -- the diffusion target has `objective_fields=("concentration",)` so phi is excluded and `None` is passed.

### VF-2: Inverse/inference_runner/objective.py -- warnings.warn in eval_cb_post
- **File:** `Inverse/inference_runner/objective.py`, lines 131-135
- **Status:** VERIFIED CORRECT
- `_AttemptMonitor.eval_cb_post` now wraps `estimate_from_controls` in a try/except and emits `warnings.warn` instead of crashing the optimization loop. The early `return` on failure prevents corrupting `best_estimate`/`best_objective`.

### VF-3: Inverse/inference_runner/objective.py -- phi_target guard in build_reduced_functional
- **File:** `Inverse/inference_runner/objective.py`, lines 50-54, 69-73
- **Status:** VERIFIED CORRECT
- `phi_target_f` is set to `None` when `phi_target is None`, and the `"phi"` objective branch raises a clear `ValueError` if `phi_target_f is None` but the target requests phi. This prevents a crash from trying to convert `None` to a Firedrake Function.

### VF-4: Inverse/parameter_targets.py -- bounds format fix
- **File:** `Inverse/parameter_targets.py`, line 154 and line 209
- **Status:** VERIFIED CORRECT
- `_build_phi0_target.default_bounds_factory` returns `(1e-8, None)` (single tuple for a scalar control).
- `_build_robin_kappa_target.default_bounds_factory` returns `[(1e-8, None) for _ in range(n_species)]` (list of tuples for per-species controls). Both formats are correct for their respective use with `scipy.optimize`.

### VF-5: Surrogate/objectives.py -- _has_autograd check
- **File:** `Surrogate/objectives.py`, lines 36-47
- **Status:** VERIFIED CORRECT
- `_has_autograd` checks for callable `predict_torch` AND respects `supports_autograd` attribute (defaulting to `True`). GP models can opt out by setting `supports_autograd = False`. All four objective classes (`SurrogateObjective`, `AlphaOnlySurrogateObjective`, `ReactionBlockSurrogateObjective`, `SubsetSurrogateObjective`) use this helper.

### VF-6: Surrogate/multistart.py -- NaN check reorder in grid evaluation
- **File:** `Surrogate/multistart.py`, lines 254-267
- **Status:** VERIFIED CORRECT
- NaN detection (`has_nan_cd`, `has_nan_pc`) is computed BEFORE zeroing NaN residuals, then NaN-containing points are marked as `np.inf`. This ensures NaN predictions are penalized rather than silently treated as zero residual.

### VF-7: Surrogate/multistart.py -- polish cache
- **File:** `Surrogate/multistart.py`, lines 340-374
- **Status:** VERIFIED CORRECT
- The autograd polish path caches `(x_bytes, J, grad)` to avoid recomputation when L-BFGS-B calls `_objective` and `_gradient` at the same point sequentially. Uses `tobytes()` for exact comparison.

---

## New Bugs Found

### BUG-1: Double-counting eval count in SurrogateObjective when using FD gradient path
- **Severity:** LOW
- **File:** `Surrogate/objectives.py`, lines 206-208 and 236-239
- **Line:** 207, 237-238
- **Description:** When `_use_autograd` is False, `objective()` increments `_n_evals` (line 207). Then `gradient()` calls `self.objective(x_plus)` and `self.objective(x_minus)` in a loop, each incrementing `_n_evals` again. When `objective_and_gradient()` calls both `objective()` and `gradient()` (lines 258-260), the center-point evaluation in `objective()` is counted once, plus 2*N evaluations in `gradient()`, all correctly counted. However, the autograd path in `_autograd_objective_and_gradient` also increments `_n_evals` (line 186), so calling `gradient()` alone (line 224) triggers `_autograd_objective_and_gradient` which increments the counter, but a separate call to `objective()` would also increment it -- no double-counting issue there. **Actually upon closer inspection, this is consistent behavior, not a bug.** RETRACTED.

### BUG-2: Stale cached torch tensors when target masks change between calls
- **Severity:** MEDIUM
- **File:** `Surrogate/objectives.py`, lines 159-171 (and analogous at 339-351, 534-546, 747-759)
- **Description:** The autograd path caches `_target_cd_t`, `_target_pc_t`, `_valid_cd_idx`, `_valid_pc_idx` on first call using `if not hasattr(self, "_target_cd_t")`. If the same objective instance were reused with different targets (e.g., `self.target_cd` reassigned), the cached tensors would be stale. This is a latent fragility rather than an active bug because no current code path mutates targets after construction. However, it violates defensive programming principles.
- **Fix suggestion:** Cache in `__init__` rather than lazily on first autograd call, or invalidate the cache if targets change.

### BUG-3: Inverse/objectives.py -- make_diffusion_objective_and_grad phi_target logic incorrect for future targets
- **Severity:** LOW
- **File:** `Inverse/objectives.py`, line 46
- **Description:** The line `phi_target = None if "phi" not in target.objective_fields else c_targets[0]` uses `c_targets[0]` as the phi_target when phi IS in objective_fields. But `c_targets[0]` is a concentration vector, not a potential vector. Currently this code path is never hit because the diffusion target has `objective_fields=("concentration",)`, so phi is always excluded. If someone added phi to the diffusion target's objective_fields in the future, it would silently use concentration data as the phi target.
- **Fix suggestion:** This function doesn't accept a `phi_vec` parameter (unlike the other two factories). If phi support is needed, add a `phi_vec` parameter. As-is, the fallback to `c_targets[0]` is misleading dead code.

### BUG-4: Inconsistent n_evals accounting in SubsetSurrogateObjective autograd path
- **Severity:** LOW
- **File:** `Surrogate/objectives.py`, line 774
- **Description:** `SubsetSurrogateObjective._autograd_objective_and_gradient` increments `_n_evals` (line 774), but `SubsetSurrogateObjective.objective()` also increments `_n_evals` (line 804). When the user calls `objective_and_gradient()` via the autograd path, only the autograd counter fires (1 eval). When the user calls it via the FD path, `objective()` fires once + `gradient()` fires `objective()` 2*N more times. The semantics of `n_evals` differ between the two paths: autograd counts forward+backward as 1, FD counts each surrogate.predict as 1. This is consistent but undocumented, and could confuse users comparing eval counts across surrogate types.
- **Fix suggestion:** Document that `n_evals` counts "forward model evaluations" which is 1 per autograd call but 1 per `predict()` call in FD mode.

---

## Remaining Bugs (Missed in Round 1)

### REM-1: recovery.py -- solver_options access pattern not robust to non-dict solver_params[10]
- **Severity:** MEDIUM
- **File:** `Inverse/inference_runner/recovery.py`, lines 56, 87, 111, 136, 229
- **Description:** Multiple lines use `hasattr(inverse_solver_params, 'solver_options')` to switch between SolverParams attribute access and list index access. However, `_relax_solver_options_for_attempt` (line 229) accesses `solver_params.solver_options if hasattr(solver_params, 'solver_options') else solver_params[10]` and then checks `if not isinstance(params, dict): return`. If `solver_params[10]` is `None` or missing, this silently skips all relaxation. The function signature says `solver_params: List[Any]` but receives SolverParams instances. The type annotation is stale.
- **Fix suggestion:** Update the type annotation to `Union[List[Any], SolverParams]` and add a guard that raises if solver_options is not a dict when relaxation is required.

### REM-2: recovery.py -- _relax_solver_options_for_attempt mutates shared dict reference
- **Severity:** MEDIUM
- **File:** `Inverse/inference_runner/recovery.py`, lines 229-280
- **Description:** The function mutates `params` (which is `solver_params[10]` or `.solver_options`) in-place. The caller passes `baseline_options` as a deep copy, but the mutation affects the live solver_params dict that is later passed to `build_reduced_functional`. If `build_reduced_functional` stores or reads these options, the mutations persist across attempts. This is intentional for the current attempt but could leak into the next iteration. Looking at line 55, `inverse_solver_params` is rebuilt each iteration via `apply_value`, which deep-copies, so the mutation is contained. **Not a bug in practice** -- the deep copy at line 55 via `apply_value` prevents leakage. RETRACTED.

### REM-3: config.py -- RecoveryConfig.line_search_schedule type annotation uses Tuple but default uses tuple literal
- **Severity:** NEGLIGIBLE
- **File:** `Inverse/inference_runner/config.py`, line 117
- **Description:** `line_search_schedule: Tuple[str, ...] = ("bt", "l2", "cp", "basic")` -- this is fine in Python 3.9+ but the `from __future__ import annotations` makes all annotations strings. No runtime issue but the type is `tuple` at runtime, not `Tuple`. This is purely cosmetic.

### REM-4: Surrogate/multistart.py -- _generate_lhs_grid returns physical-space params but column names in docstring could confuse
- **Severity:** NEGLIGIBLE
- **File:** `Surrogate/multistart.py`, line 176
- **Description:** The docstring says columns are `[k0_1, k0_2, alpha_1, alpha_2]` in physical space, which is correct. No bug.

### REM-5: Surrogate/multistart.py -- _polish_candidate NaN in initial x0 not guarded
- **Severity:** LOW
- **File:** `Surrogate/multistart.py`, lines 418-423
- **Description:** If `x0_physical[0]` or `x0_physical[1]` is exactly 0 or negative (which shouldn't happen from LHS but could from a bad caller), `np.log10(max(..., 1e-30))` clamps to `log10(1e-30) = -30`, which is safe. The `max(..., 1e-30)` guard is correct. No bug.

### REM-6: Inverse/inference_runner/objective.py -- build_reduced_functional does not validate objective_fields values
- **Severity:** LOW
- **File:** `Inverse/inference_runner/objective.py`, lines 64-80
- **Description:** The function checks for `"concentration"` and `"phi"` in `target.objective_fields` but silently ignores any other values. If a future target includes a typo like `"concentraton"`, no objective terms would be generated and the "must request at least one objective field" error (line 78) would fire, but the error message would not indicate the typo.
- **Fix suggestion:** Add a validation step that checks all entries in `objective_fields` are in `{"concentration", "phi"}` and raises a descriptive error for unrecognized fields.

### REM-7: Inverse/inference_runner/objective.py -- chained_pre/chained_post closures capture mutable references
- **Severity:** LOW
- **File:** `Inverse/inference_runner/objective.py`, lines 98-113
- **Description:** The `chained_pre` and `chained_post` closures capture `target_pre`, `target_post`, `extra_eval_cb_pre`, `extra_eval_cb_post` from the enclosing scope. These are bound at definition time in Python, so late-binding issues don't apply here. **Not a bug.** RETRACTED.

---

## Summary

| Category | Count |
|----------|-------|
| Verified Fixes | 7 |
| New Bugs (from fixes) | 1 (MEDIUM) |
| Remaining Bugs (missed in round 1) | 3 (1 MEDIUM, 2 LOW) |
| Retracted (false positives) | 3 |

### Actionable Items (by priority)

1. **MEDIUM -- BUG-2:** Move torch tensor caching from lazy `hasattr` check to `__init__` in all four `Surrogate/objectives.py` classes to prevent stale-cache fragility.
2. **MEDIUM -- REM-1:** Fix type annotations in `recovery.py` and add guard for non-dict solver_options when relaxation is needed.
3. **LOW -- BUG-3:** Fix misleading `c_targets[0]` fallback in `make_diffusion_objective_and_grad` -- either remove the dead branch or add a proper `phi_vec` parameter.
4. **LOW -- REM-6:** Add validation of `objective_fields` entries against known field names in `build_reduced_functional`.
5. **LOW -- BUG-4:** Document `n_evals` semantics difference between autograd and FD paths.
