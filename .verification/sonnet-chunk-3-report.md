# Re-verification Report: Chunk 3 (bv_point_solve/__init__.py, forward.py, bv_curve_eval.py)

Verifier: claude-sonnet-4-6
Scope: Fix #3 and Fix #4

---

## Fix #3 Verification: Physics-failed points skip both objective AND gradient

### `evaluate_bv_curve_objective_and_gradient()` — bv_curve_eval.py lines 75–94

**PARTIALLY CORRECT but has a logic bug.**

The fix comment says "skip both objective AND gradient" for physics-failed converged points, and the `continue` on line 88 does skip `total_objective +=` correctly. However the control flow after `continue` skips the entire `total_objective` and `total_gradient` block — but the structure is wrong:

```python
for i, point in enumerate(points):
    simulated_flux[i] = point.simulated_flux
    # Physics validation: skip non-physical points from objective
    if point.converged:
        vr = validate_observables(...)
        if not vr.valid:
            n_failed += 1
            continue           # <-- skips objective AND gradient: CORRECT
    total_objective += float(point.objective)
    if point.converged:
        total_gradient += point.gradient
    else:
        n_failed += 1
        total_gradient += point.gradient  # Include fail-penalty gradient
```

**Problem**: A converged-but-physics-invalid point falls through the `if point.converged:` block after `continue`, which is fine. BUT a *non-converged* point (converged=False) skips the `validate_observables` block entirely (the outer `if point.converged:` guard), falls to `total_objective += float(point.objective)` (adds the fail_penalty), and then increments `n_failed += 1` AND adds `point.gradient`. This means `n_failed` is incremented for both the "physics invalid converged" path AND the "not converged" path — which is correct. The objective/gradient accounting is also correct for both paths:

- Converged + physics-invalid: `n_failed += 1`, `continue` → no obj/grad added. CORRECT.
- Converged + physics-valid: obj and grad added. CORRECT.
- Not-converged: obj (fail_penalty) added, n_failed incremented, fail-penalty gradient added. CORRECT.

**Conclusion for primary loop: Fix #3 is correctly applied.**

### `evaluate_bv_multi_observable_objective_and_gradient()` — bv_curve_eval.py lines 296–358 (parallel path)

The `_skip_mask` is applied in both the primary and secondary aggregation loops. For skipped points:
```python
if _skip_mask[i]:
    n_failed_primary += 1
    continue   # skips total_obj_primary += and total_grad_primary +=
```
Both objective and gradient are skipped. CORRECT.

The v5 fallback path (lines 391–446) calls `evaluate_bv_curve_objective_and_gradient` twice, which inherits the single-function fix. CORRECT by delegation.

**Conclusion for multi-observable loops: Fix #3 is correctly applied.**

---

## Fix #4 Verification: Mid-loop validation removed; post-convergence validation retained

### `forward.py` (sequential cached fast path)

**No `validate_observables` import present.** Searched: no matches.

Post-convergence `validate_solution_state` is present at lines 303–334 (after `steady_count >= required_steady` is confirmed and after adjoint gradient is computed). It guards cache updates:
- If valid: updates `_all_points_cache[orig_idx]`. CORRECT.
- If invalid: emits a warning and skips cache update. CORRECT.

No mid-loop `validate_observables` or `validate_solution_state` call exists inside the `for step in range(...)` loop (lines 235–269). CORRECT.

**Fix #4 in forward.py: VERIFIED.**

### `__init__.py` (main sequential sweep orchestrator)

**No `validate_observables` import present.** Only `validate_solution_state` from `Forward.bv_solver.validation` is imported (line 37).

`validate_solution_state` is called post-convergence at lines 708–735:
- Guarded by `if not failed_by_exception and steady_count >= required_steady:` (line 666), i.e., only on converged points.
- Called after adjoint gradient computation.
- On failure: sets `failed_by_exception = True` and records the reason. This downgrades the point to failed, preventing the `point_result` from being set as converged.

No mid-loop validation inside `for step in range(1, effective_max_steps + 1):` (lines 611–654). CORRECT.

**Fix #4 in __init__.py: VERIFIED.**

---

## Fix #4 Sub-check: Unused `validate_observables` import cleanup

- `bv_point_solve/__init__.py`: No `validate_observables` import. CLEAN.
- `bv_point_solve/forward.py`: No `validate_observables` import. CLEAN.
- `bv_curve_eval.py` line 14: `from Forward.bv_solver.validation import validate_observables` — this import IS still present and IS still used in `evaluate_bv_curve_objective_and_gradient()` (line 79) and in the multi-obs parallel path (line 303). CORRECT — import is needed, not unused.

---

## Objective/Gradient Accounting When Points Are Skipped

### Primary loop (`evaluate_bv_curve_objective_and_gradient`):

Three exclusive paths for each point:
1. `converged=True`, physics-valid: `total_objective += obj`, `total_gradient += grad`. `n_failed` unchanged.
2. `converged=True`, physics-invalid: `n_failed += 1`, `continue`. Neither obj nor grad added.
3. `converged=False`: `total_objective += fail_penalty`, `n_failed += 1`, `total_gradient += fail_grad`.

Invariant `total_gradient == d(total_objective)/d(params)` holds for paths 1 and 3. Path 2 correctly excludes both. CORRECT.

### Multi-obs parallel path:

Same three-path logic with `_skip_mask[i]` replacing the `validate_observables` check. Both primary and secondary loops handle the skip mask identically. CORRECT.

---

## New Issues Introduced by the Fixes

### ISSUE 1 (Minor — pre-existing, not introduced by fixes): Double `n_failed` in multi-obs combined count

In `evaluate_bv_multi_observable_objective_and_gradient()` (both parallel and v5 fallback), `combined.n_failed` is set to `int(primary_result.n_failed) + int(secondary_result.n_failed)` (lines 379, 440). For a physics-failed point that appears in both primary and secondary, it is counted twice. This inflates the failure count used by anisotropy recovery triggers in `evaluate_bv_curve_objective_and_gradient`. However, this issue predates the current fixes and is not introduced by them.

### ISSUE 2 (Minor — potential, not introduced by fixes): `validate_observables` called with hardcoded sentinel values

In the primary loop (line 79–83):
```python
vr = validate_observables(
    float(point.simulated_flux), 0.0,
    I_lim=1.0, phi_applied=float(phi_applied_values[i]), V_T=1.0,
)
```
The secondary observable flux is passed as `0.0` and `I_lim=1.0`, `V_T=1.0` are hardcoded. This is the pre-existing design, not introduced by the fix.

### No new control-flow issues introduced by Fix #3 or Fix #4.

---

## Summary

| Check | Result |
|---|---|
| Fix #3: Primary loop skips both obj and grad for physics-failed converged points | PASS |
| Fix #3: Multi-obs parallel path skips both obj and grad via `_skip_mask` | PASS |
| Fix #3: Multi-obs v5 fallback inherits correct behavior by delegation | PASS |
| Fix #4: Mid-loop validation removed from `forward.py` | PASS |
| Fix #4: Mid-loop validation removed from `__init__.py` | PASS |
| Fix #4: Post-convergence `validate_solution_state` still present in `forward.py` | PASS |
| Fix #4: Post-convergence `validate_solution_state` still present in `__init__.py` | PASS |
| Unused `validate_observables` cleaned from `forward.py` | PASS (was never there) |
| Unused `validate_observables` cleaned from `__init__.py` | PASS (was never there) |
| `validate_observables` in `bv_curve_eval.py` is still used (not orphaned) | PASS |
| Objective/gradient sum correctly accounts for skipped points | PASS |
| New bugs introduced by fixes | NONE FOUND |
