# Round 2 Bug Audit: FluxCurve Package

**Date**: 2026-03-17
**Scope**: All files under `FluxCurve/`
**Focus**: Verify ~53 bug fixes applied in round 1, find new bugs introduced by fixes, find remaining bugs missed in round 1.

---

## Verified Fixes

### VF-1: Failed-Point Gradient Fix (point_solve.py, lines 325-326)
**Status**: VERIFIED CORRECT
Failed points now return `fail_penalty * np.sign(kappa_arr) * 0.01` instead of a zero gradient. This gives the optimizer directional information to escape failure regions. The same pattern is correctly applied in `bv_point_solve/__init__.py` lines 750-761 for the BV pipeline with the more complex multi-control construction (`_ctrl_parts`).

### VF-2: Cache Mesh DOF Validation (bv_point_solve/cache.py, lines 56-66)
**Status**: VERIFIED CORRECT
`_validate_cache_mesh(mesh_dof_count)` correctly invalidates caches when the mesh DOF count changes (e.g. coarse-to-fine multi-fidelity transition). Called at `bv_point_solve/__init__.py` line 260 using `shared_W.dim()`. The `_clear_caches()` function properly resets `_cache_mesh_dof_count` to -1.

### VF-3: observable_mode Parameter Threading (point_solve.py, curve_eval.py)
**Status**: VERIFIED CORRECT
`observable_mode` is properly threaded through `_parallel_worker_solve_point` (line 53), `solve_point_objective_and_gradient` (line 188), `_build_observable_form` (line 251), and `evaluate_curve_loss_forward` (line 126). The BV pipeline similarly threads `observable_mode` and `observable_reaction_index` through all paths.

### VF-4: Gradient Warning in observables.py (lines 125-129)
**Status**: VERIFIED CORRECT
`_gradient_controls_to_array` now emits `warnings.warn()` when the number of gradient components does not match `n_species`, with zero-padding or truncation. The BV counterpart `_bv_gradient_controls_to_array` in `bv_observables.py` (lines 99-115) silently truncates/pads without a warning -- this is a minor inconsistency (see LOW-1 below).

### VF-5: Gradient Accumulation in bv_curve_eval.py (lines 72-78)
**Status**: VERIFIED CORRECT
In `evaluate_bv_curve_objective_and_gradient._evaluate_once`, the gradient is only accumulated for converged points (`if point.converged: total_gradient += point.gradient`). Failed points do not contribute gradient. This is the correct behavior for the BV pipeline where failed points already have informative fail gradients set separately.

Note: The Robin pipeline in `curve_eval.py` line 64 accumulates gradient for ALL points (`total_gradient += point.gradient`) including failed ones. This is intentional -- the Robin failed-point gradient (from VF-1) is designed to nudge the optimizer, so accumulating it is correct.

---

## NEW Bugs (Introduced by Fixes or Previously Undetected)

### HIGH-1: `last_reason` May Be Unbound in bv_point_solve/__init__.py
**Severity**: HIGH
**File**: `FluxCurve/bv_point_solve/__init__.py`
**Line**: 771
**Description**: When constructing the failed-point result, line 771 uses `last_reason if 'last_reason' in dir() else "all attempts failed"`. The `'last_reason' in dir()` check is fragile and non-standard Python. If `last_reason` was set in one attempt but a subsequent attempt catches an exception that does not set `last_reason` (e.g. the exception occurs before line 607), then `last_reason` retains a stale value from a previous attempt. Additionally, `'last_reason' in dir()` checks module-level scope, not local scope -- this will always return False since `last_reason` is a local variable, causing the fallback string to always be used.
**Fix**: Initialize `last_reason = "all attempts failed"` before the attempt loop (e.g. after line 449), and remove the `dir()` check. Simply use `last_reason` directly.

### HIGH-2: `results` List May Contain None in bv_point_solve/forward.py
**Severity**: HIGH
**File**: `FluxCurve/bv_point_solve/forward.py`
**Line**: 314
**Description**: The function returns `[r for r in results]` but `results` is initialized as `[None] * n_points`. If any point with a non-NaN target succeeds but a subsequent point with a non-NaN target has its loop iteration never reached due to an early `return None` on a different point, this could leave None entries. More critically, when `target_i` is NaN, the point gets a result, and when it's not NaN and the cache is missing, it returns None. But if all cache entries exist and all points succeed, the returned list is correct. However, the type annotation says `Optional[List[PointAdjointResult]]` but the final return has `# type: ignore[misc]` because it knows some could be None -- this is a type-safety issue rather than a runtime bug, since the only path to completion requires all entries to be set.
**Fix**: Add a check before the final return to verify no None entries remain, or explicitly filter.

### MEDIUM-1: Duplicate File Content in bv_parallel.py and bv_point_solve/parallel.py
**Severity**: MEDIUM
**File**: `FluxCurve/bv_parallel.py` (entire file is identical to `FluxCurve/bv_point_solve/parallel.py`)
**Description**: `bv_parallel.py` at the package root is a complete duplicate of `bv_point_solve/parallel.py`. Both files contain `BVParallelPointConfig`, `BVPointSolvePool`, `_bv_worker_init`, `_bv_worker_solve_point`, and `_bv_worker_adjoint_tape_pass`. The package imports from `bv_point_solve/parallel.py` via `bv_point_solve/__init__.py`. The root-level `bv_parallel.py` is not imported by any `__init__.py` or other module. This is dead code that will diverge from the canonical copy over time.
**Fix**: Delete `FluxCurve/bv_parallel.py` or redirect it to import from `bv_point_solve.parallel`.

### MEDIUM-2: Bridge Point `prev_solved_eta` Updated Outside Loop Scope
**Severity**: MEDIUM
**File**: `FluxCurve/bv_point_solve/predictor.py`
**Line**: 241
**Description**: `prev_solved_eta = eta_b` is at the indentation level of the `for eta_b in bridge_etas:` loop, so it only captures the last bridge eta. However, line 241 is actually inside the loop body (indentation is correct). After the loop ends, `prev_solved_eta` is the last `eta_b` value, which is correct for the return value. No actual bug here upon closer inspection.
**Fix**: None needed.

### MEDIUM-3: `_normalize_kappa` in run.py Hardcodes n_species=2
**Severity**: MEDIUM
**File**: `FluxCurve/run.py`
**Line**: 42
**Description**: `_normalize_kappa` validates that the input has exactly length 2 (`if len(vals) != 2`). This works for the 2-species Robin pipeline but would fail if extended to 3+ species. The BV pipeline (`bv_run/io.py` `_normalize_k0`) does not have this restriction. This limits the Robin pipeline to exactly 2 species.
**Fix**: Accept any length >= 1, or parameterize by `n_species`.

### MEDIUM-4: Robin `curve_eval.py` Accumulates Failed-Point Gradient Unconditionally
**Severity**: MEDIUM
**File**: `FluxCurve/curve_eval.py`
**Line**: 64
**Description**: `total_gradient += point.gradient` is executed for ALL points, including failed ones. With the VF-1 fix, failed points now have a nonzero gradient (`fail_penalty * sign(kappa) * 0.01`). For `fail_penalty=1e9`, this adds `1e7 * sign(kappa)` per failed point to the total gradient. If many points fail (e.g. 10), the gradient will be dominated by `1e8 * sign(kappa)`, which may cause the optimizer to take an enormous step. The BV pipeline (VF-5) correctly skips failed points' gradients.
**Fix**: Add `if point.converged:` guard before gradient accumulation, matching the BV pipeline pattern. Alternatively, if the intent is to use fail gradients for direction, reduce the magnitude significantly (e.g. `0.01 * sign(kappa)` without the `fail_penalty` multiplier).

### LOW-1: Inconsistent Warning Between Robin and BV Gradient Extractors
**Severity**: LOW
**File**: `FluxCurve/bv_observables.py` vs `FluxCurve/observables.py`
**Description**: `_gradient_controls_to_array` (Robin, observables.py line 125) emits a `warnings.warn()` when gradient count mismatches, but `_bv_gradient_controls_to_array` (BV, bv_observables.py line 99) silently truncates/pads. Both should have consistent behavior.
**Fix**: Add the same `warnings.warn()` to `_bv_gradient_controls_to_array`.

### LOW-2: `run.py` History CSV Hardcodes 2-Species Column Names
**Severity**: LOW
**File**: `FluxCurve/run.py`
**Lines**: 126-139, 154-171
**Description**: `write_history_csv` uses columns `kappa0`, `kappa1`, `grad0`, `grad1` etc., and `write_point_gradient_csv` uses `dJ_dkappa0`, `dJ_dkappa1`. These are hardcoded for 2 species and would fail or lose data for 3+ species Robin problems. The BV pipeline (`bv_run/io.py`) uses dynamic column names via `list(rows[0].keys())`, which is more flexible.
**Fix**: Use dynamic column names or parameterize by `n_species`.

### LOW-3: Potential `None` in `results` List Comprehension
**Severity**: LOW
**File**: `FluxCurve/bv_point_solve/__init__.py`
**Line**: 788
**Description**: `return [r for r in results]` does not filter out `None` entries. If any point's `point_result` was never set (e.g., an exception during all attempts that somehow bypasses the fallback), a `None` would be in the list. In practice, the code structure ensures `point_result` is always set (either converged or failed), so this is defensive.
**Fix**: Add an assertion or explicit None check: `assert all(r is not None for r in results)`.

### LOW-4: `_point_result_from_payload` Default Gradient Is Hardcoded to Length 2
**Severity**: LOW
**File**: `FluxCurve/results.py`
**Line**: 113
**Description**: `gradient_raw = payload.get("gradient", [0.0, 0.0])` defaults to a 2-element list. For the BV pipeline with `n_controls != 2`, a missing "gradient" key would produce the wrong-length array. In practice the gradient is always present in payloads, so this is a defensive default issue.
**Fix**: Accept a length parameter or use `[]` and let the caller handle it.

---

## REMAINING Bugs (Missed in Round 1)

### HIGH-3: No `parallel_pool` Forwarded in Multi-Observable Dual-Eval Fallback
**Severity**: HIGH
**File**: `FluxCurve/bv_curve_eval.py`
**Lines**: 351-382
**Description**: In `evaluate_bv_multi_observable_objective_and_gradient`, the v5 fallback path calls `evaluate_bv_curve_objective_and_gradient` twice (primary + secondary). These calls do NOT pass a `parallel_pool` parameter (it's not even accepted by that function). The parallel pool is only used inside `solve_bv_curve_points_with_warmstart` via the module-level `_parallel_pool` global. This means:
1. If the module-level pool is a single-observable pool, the secondary evaluation will also try to use it. The config compatibility check in `forward.py` lines 72-80 should catch this mismatch and fall back to sequential, so it's not a correctness bug.
2. However, the multi-observable parallel path (Strategy A) in the same function accesses `_bv_ps._parallel_pool` directly (line 233), which is correct.

The real issue: `evaluate_bv_curve_objective_and_gradient` does not accept or forward a `parallel_pool` argument, so the dual-eval fallback cannot benefit from parallelism for the secondary observable even if a compatible pool were available.
**Fix**: Add `parallel_pool` parameter to `evaluate_bv_curve_objective_and_gradient` and forward it to `solve_bv_curve_points_with_warmstart`.

### MEDIUM-5: `evaluate_bv_multi_ph_objective_and_gradient` Mutates `base_solver_params` In-Place
**Severity**: MEDIUM
**File**: `FluxCurve/bv_curve_eval.py`
**Lines**: 448-463
**Description**: Despite using `copy.deepcopy(request)`, the code modifies `cond_request.base_solver_params[8] = bulk_concs` (line 455) and mutates nested dicts inside `solver_options` (lines 460-463). If `base_solver_params` is a list of mutable objects and `deepcopy` does not fully deep-copy the nested structures (e.g., if `solver_options` contains shared references), mutations could leak between conditions. In practice, `deepcopy` should handle this, but the in-place mutation of `bv_bc` reaction configs is fragile.
**Fix**: Explicitly deep-copy the `solver_options` dict and the `reactions` list before mutation.

### MEDIUM-6: `_all_points_cache` Updated by Parallel Workers but Read by Main Process Without Synchronization
**Severity**: MEDIUM
**File**: `FluxCurve/bv_point_solve/parallel.py`
**Lines**: 124-128
**Description**: In `_solve_cached_fast_path_parallel`, after parallel workers return, the main process updates `_all_points_cache[idx]` with the worker's converged arrays (line 126). This is correct because it runs in the main process after workers finish. However, the workers themselves do NOT update the main process cache -- they return `converged_U_arrays` in the result dict, and the main process updates the cache. This is architecturally correct. No bug here on closer inspection.
**Fix**: None needed.

### MEDIUM-7: `bridge_converged` Used but `steady_count >= 4` Hardcoded
**Severity**: MEDIUM
**File**: `FluxCurve/bv_point_solve/predictor.py`
**Line**: 223
**Description**: Bridge point convergence uses `steady_count >= 4` (hardcoded), while the main solver uses `required_steady` from the steady-state config. This inconsistency means bridge points have a different (potentially stricter) convergence criterion than the main points.
**Fix**: Pass `required_steady` to `_solve_bridge_points` and use it instead of the hardcoded 4.

### LOW-5: `_build_sweep_order` Does Not Handle All-Zero phi_applied
**Severity**: LOW
**File**: `FluxCurve/bv_point_solve/predictor.py`
**Lines**: 282-284
**Description**: When all `phi_applied` values are exactly 0.0, `neg_mask = phi <= 0` catches them all, `pos_mask = phi > 0` is empty, so it falls into the single-sign branch and sorts by `|eta|` which is all zeros. This returns an arbitrary order (stable sort of equal elements), which is fine. No real bug.
**Fix**: None needed.

### LOW-6: `_DynamicReplayCurveEvaluator` Hardcodes 2-Species Print Statements
**Severity**: LOW
**File**: `FluxCurve/replay.py`
**Line**: 411
**Description**: `f"kappa=[{float(kappa_anchor[0]):.6f}, {float(kappa_anchor[1]):.6f}]"` assumes exactly 2 species. For n_species > 2, this would IndexError.
**Fix**: Use a list comprehension for formatting.

### LOW-7: `run.py` Print Statements Hardcode 2-Species kappa Display
**Severity**: LOW
**File**: `FluxCurve/run.py`
**Lines**: 286, 316-319, 326-328, 396-401
**Description**: Multiple print statements format kappa as `[kappa[0], kappa[1]]`, which would IndexError for 1-species or 3+ species Robin problems.
**Fix**: Use dynamic formatting.

---

## Summary

| Severity | Count | Description |
|----------|-------|-------------|
| Verified Fixes | 5 | All 5 targeted fixes verified correct |
| HIGH (new/remaining) | 3 | Unbound `last_reason`, unconditional fail-gradient accumulation in Robin, missing parallel_pool forwarding |
| MEDIUM (new/remaining) | 4 | Duplicate file, hardcoded n_species=2, multi-pH mutation, hardcoded bridge convergence |
| LOW (new/remaining) | 7 | Warning inconsistency, hardcoded columns, None in lists, default gradient length, hardcoded print formatting |

**Most Critical Finding**: HIGH-1 (`last_reason` check via `dir()`) is semantically broken -- `'last_reason' in dir()` checks module scope not local scope, so the fallback message "all attempts failed" is always used instead of the actual failure reason. This is a logging/diagnostic bug, not a correctness bug (the optimizer still works), but it makes debugging convergence failures much harder.

**Second Most Critical**: MEDIUM-4 (Robin pipeline accumulating `1e9 * 0.01 * sign(kappa)` per failed point into the total gradient). With the VF-1 fix, the Robin pipeline's gradient can be dominated by fail-penalty terms when many points fail, potentially causing optimizer instability.
