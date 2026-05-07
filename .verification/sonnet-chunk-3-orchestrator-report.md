# Verification Report: C+D Orchestrator and Formulation Dispatcher
## Files Reviewed
- `Forward/bv_solver/grid_per_voltage.py` (611 lines)
- `Forward/bv_solver/sweep_order.py` (163 lines)
- `Forward/bv_solver/dispatch.py` (137 lines)

Cross-referenced against:
- `Forward/bv_solver/forms_logc.py` (function names and adjoint wrapping)
- `Forward/bv_solver/forms_logc_muh.py` (function names and adjoint wrapping)
- `Forward/bv_solver/diagnostics.py` (collect_diagnostics signature)
- `Forward/bv_solver/observables.py` (_build_bv_observable_form signature)
- `Forward/bv_solver/solvers.py` (_clone_params_with_phi)
- `Forward/params.py` (SolverParams.__getitem__, with_phi_applied)
- `Forward/bv_solver/__init__.py` (public re-exports)
- `scripts/studies/peroxide_window_3sp_bikerman_muh.py` (production driver)

---

## Issues Found

### WARNING-1: Cold-failed interior points between non-contiguous cold anchors are never warm-walked

**SEVERITY**: warning

**LOCATION**: `grid_per_voltage.py` lines 503–600 (Phase 2 orchestration)

**DESCRIPTION**:
Phase 2 defines two sweep ranges: cathodic `range(anchor_lo - 1, -1, -1)` and anodic `range(anchor_hi + 1, n_points)`. Only `anchor_lo = cold_idxs[0]` and `anchor_hi = cold_idxs[-1]` are used. Any voltage index `k` satisfying `anchor_lo < k < anchor_hi` that failed Phase 1 cold-start is never visited in Phase 2 and remains `converged=False` in the output.

**EVIDENCE**:
```python
anchor_lo = cold_idxs[0]
anchor_hi = cold_idxs[-1]
# Cathodic walk: from anchor_lo down toward index 0
for orig_idx in range(anchor_lo - 1, -1, -1):   # only indices < anchor_lo
    ...
# Anodic walk: from anchor_hi up toward index NV-1
for orig_idx in range(anchor_hi + 1, n_points):  # only indices > anchor_hi
    ...
```

Cold successes at interior non-contiguous indices (e.g., indices 2 and 7 with a gap at 3–6) leave the interior failures silently unremedied. The cathodic walk covers nothing below the overall lowest anchor, and the anodic walk covers nothing above the overall highest anchor.

**PRACTICAL RISK**: Low for the production voltage grid `[-0.5, +1.0] V`. Cold failures at high anodic voltages (above `+0.60 V`) land above `anchor_hi` and are correctly handled by the anodic walk. Interior gaps would require a non-monotone cold failure pattern (e.g., a point near 0 V failing cold while points on both sides succeed), which is unusual. However, if it occurs, the gap is silent — no warning is printed and the point is returned as `method="cold-failed"`.

---

### NOTE-1: Adjoint tape annotation is caller-controlled, not self-suppressed inside the orchestrator

**SEVERITY**: note

**LOCATION**: `grid_per_voltage.py` — the entire function body

**DESCRIPTION**:
The z-ramp solves (`run_ss` inside `_solve_cold`), warm-walk solves (`run_ss` inside `_solve_warm`), and the linear-phi IC (`set_initial_conditions_logc`) are not wrapped in `firedrake.adjoint.stop_annotating()` at the orchestrator level. Only the debye_boltzmann IC routines self-suppress annotation (inside `forms_logc.py:set_initial_conditions_debye_boltzmann_logc` and `forms_logc_muh.py:set_initial_conditions_debye_boltzmann_logc_muh`).

**EVIDENCE**: The production reference driver (`scripts/studies/peroxide_window_3sp_bikerman_muh.py` line 142–151) wraps the entire call in `with adj.stop_annotating():`:
```python
with adj.stop_annotating():
    result = solve_grid_per_voltage_cold_with_warm_fallback(...)
```
No `stop_annotating` appears anywhere in `grid_per_voltage.py`. The debye_boltzmann ICs are internally wrapped (verified in `forms_logc.py:609` and `forms_logc_muh.py:690`).

**ASSESSMENT**: This is the intended design. CLAUDE.md guidance ("wrap unannotated cold-ramp / continuation work in `with adj.stop_annotating():`") refers to the call site, not the orchestrator internals. The production driver correctly follows this. If the orchestrator is ever called without the `stop_annotating` wrapper, the z-ramp solves and linear-phi IC interpolations would pollute the pyadjoint tape. A defensive internal suppression would be safer, but this is consistent with the current inverse-paused status and the documented call pattern.

---

### NOTE-2: PerVoltagePointResult stores solution data but not computed observables

**SEVERITY**: note

**LOCATION**: `grid_per_voltage.py` lines 66–76 (dataclass definition)

**DESCRIPTION**:
`PerVoltagePointResult` fields are: `orig_idx`, `phi_applied`, `U_data`, `achieved_z_factor`, `converged`, `method`, `diagnostics`. There is no built-in field for current density, peroxide current, or mass balance. The task specification says the result should record "observables (currents, mass balance)."

**EVIDENCE**:
The `diagnostics` dict (from `collect_diagnostics`) contains surface concentrations and phi extrema but not the assembled current density or peroxide current. Those are extracted only if the caller provides `per_point_callback`. The production driver uses this pattern:
```python
def _grab(orig_idx, eta, ctx):
    cd[orig_idx] = float(fd.assemble(f_cd))
    pc[orig_idx] = float(fd.assemble(f_pc))
```

**ASSESSMENT**: Functional for current production use (the driver extracts observables via callback). However, callers who forget to supply `per_point_callback` get no observable data at all from the result object. A future-facing improvement would store assembled observables directly in the dataclass, but this is not a correctness bug in the current production flow.

---

### NOTE-3: IC waste in warm-walk (_solve_warm ignores the IC it builds)

**SEVERITY**: note

**LOCATION**: `grid_per_voltage.py` lines 379–388 (`_solve_warm`)

**DESCRIPTION**:
`_build_for_voltage(V_target_eta)` calls `set_initial_conditions(ctx, sp)` which computes and assigns an IC at `V_target`. Immediately after, `_restore_U(anchor_snap, U, U_prev)` overwrites `U` and `U_prev` with the anchor snapshot. The IC computation is wasted.

**EVIDENCE**:
```python
ctx, solver, of_cd = _build_for_voltage(V_target_eta)  # sets IC at V_target
...
_restore_U(anchor_snap, U, U_prev)  # immediately overwrites IC with anchor state
```

**ASSESSMENT**: No correctness impact. The anchor snapshot overwrites the IC before any SS call. The cost is one extra IC computation per warm-walk point. For the debye_boltzmann IC (which runs a Picard solve), this is a non-trivial wasted cost per warm-walked voltage. No bug, but a performance inefficiency.

---

### NOTE-4: sweep_order._build_sweep_order and _apply_predictor are not used by the C+D orchestrator

**SEVERITY**: note

**LOCATION**: `Forward/bv_solver/sweep_order.py`

**DESCRIPTION**:
Neither `_build_sweep_order` nor `_apply_predictor` is imported or called by `grid_per_voltage.py` or `dispatch.py`. Both are re-exported via `FluxCurve/bv_point_solve/predictor.py` for use in the older `FluxCurve` warm-walk-only path. The module is effectively utility code for FluxCurve, not for C+D.

**EVIDENCE**:
```python
# FluxCurve/bv_point_solve/predictor.py line 22:
from Forward.bv_solver.sweep_order import _apply_predictor, _build_sweep_order  # noqa: F401 — re-export
```
No import of `sweep_order` anywhere in `Forward/bv_solver/grid_per_voltage.py` or `dispatch.py`.

**ASSESSMENT**: The two functions are correctly implemented (both handle edge cases, have safety validation, and are off-by-one clean). Their placement in `Forward/bv_solver/` is appropriate since they belong to the Forward layer's utility surface even if only FluxCurve currently uses them.

---

## Correctness Arguments (things that are right)

### grid_per_voltage.py

**1. Sweep order for Phase 1 (cold)**
Each voltage is solved independently in grid order. Order doesn't matter for Phase 1 because each call is a fresh context with no state shared between voltages. Cold failures cannot cascade.

**2. Deterministic seeding / NaN isolation**
`_build_for_voltage` creates a new `fd.Function(W)` for `U` and `U_prev` at each call. There is no shared mutable state between different voltage contexts. A diverged or NaN-contaminated `U` from one voltage cannot carry into another.

**3. Path B (debye_boltzmann) fallback to z-ramp**
When the direct z=1 SS fails for the debye_boltzmann IC, the code calls `set_initial_conditions_logc` (imported directly from `forms_logc`, bypassing the dispatcher to avoid re-dispatching to the analytical IC) and then proceeds to the z-ramp path. The `set_initial_conditions_logc` call resets `U` and `U_prev` via `.assign()`, clearing any diverged state before the z=0 SS.

**4. Warm-walk paf assignment**
`paf.assign(float(v_sub))` is called inside `_march` before each `run_ss` call. `paf` is a `fd.Constant` used in both the BV eta expression and the Dirichlet electrode BC, so all reassignments take effect immediately. The `_march` function does NOT use `paf` as a source of `v_prev_substep` (it tracks `v_prev_substep` explicitly), which correctly avoids the degeneracy noted in the comment on line 392.

**5. Checkpoint-rollback in z-ramp**
At each z-step, `ckpt = _snapshot_U(U)` is taken before `_set_z_factor`. On failure, `_restore_U(ckpt, U, U_prev)` reinstates the last good state and `_set_z_factor(ctx, achieved_z)` restores the charge factor. The z-ramp breaks cleanly and the achieved_z is recorded accurately.

**6. Bisection in warm-walk**
The bisection recursion correctly: (a) restores `ckpt_inner` before recursing; (b) restores `ckpt_outer` if either half fails; (c) sets `v_prev_substep = float(v_sub)` after both halves succeed, so the outer loop's next substep correctly continues from `v_sub`. The recursion depth is capped at `bisect_depth_warm` with a clean `return False` and `ckpt_outer` restore.

**7. Edge cases**
- Empty grid (`n_points=0`): Phase 1 loop doesn't run. `last_dof_count` stays `None`. Early return at line 487 fires (cold_idxs is empty). Returns `PerVoltageContinuationResult(points={}, mesh_dof_count=0)`.
- Single voltage: Phase 1 runs. anchor_lo=anchor_hi=0. Cathodic range is `range(-1,-1,-1)` (empty). Anodic range is `range(1,1)` (empty). Returns cleanly.
- All-fail grid: cold_idxs is empty. Early return with all `converged=False` results preserved.

**8. Cathodic/anodic chain propagation**
After a successful warm-walk at `orig_idx`, `snapshots[orig_idx]` and `points[orig_idx].converged` are both updated. The next iteration finds the just-converged neighbor immediately (j = orig_idx, now converged). The chain propagates correctly outward.

**9. No Strategy B callsite**
`solve_grid_with_charge_continuation` is referenced only in comments within `boltzmann.py`, `observables.py`, and `solvers.py`. It is not called or imported in any of the three reviewed files.

**10. No double-counted Boltzmann**
Neither `grid_per_voltage.py` nor `dispatch.py` calls `add_boltzmann_counterion_residual` or `add_boltzmann`. Boltzmann counterion handling is delegated entirely to `build_forms_logc` / `build_forms_logc_muh` via the dispatcher. The anti-pattern from CLAUDE.md is absent.

### dispatch.py

**11. Formulation dispatch**
`_read_bv_convergence_field` reads `solver_params[10]` via `__getitem__` (works for `SolverParams` via `to_list()` and for legacy 11-tuples). It guards with `isinstance(params_dict, dict)` and `isinstance(bv_conv, dict)` before any `.get()`. Falls back to "logc" for any unknown formulation. Returns "logc_muh" only on exact `.strip().lower()` match.

**12. Initializer dispatch**
Same defensive path as formulation dispatch. Falls back to `"linear_phi"` IC. Picks `debye_boltzmann` variant only on exact match. The `blob` kwarg is silently accepted but not used (documented in docstring).

**13. Import name correctness**
All imported names in `dispatch.py` match the actual function names in `forms_logc.py` and `forms_logc_muh.py`:
- `build_context_logc`, `build_forms_logc`, `set_initial_conditions_logc`, `set_initial_conditions_debye_boltzmann_logc` — all present in `forms_logc.py`
- `build_context_logc_muh`, `build_forms_logc_muh`, `set_initial_conditions_logc_muh`, `set_initial_conditions_debye_boltzmann_logc_muh` — all present in `forms_logc_muh.py`

**14. Adjoint annotation suppression in IC routines**
`set_initial_conditions_debye_boltzmann_logc` (`forms_logc.py:609`) and `set_initial_conditions_debye_boltzmann_logc_muh` (`forms_logc_muh.py:690`) both wrap their bodies in `with adj.stop_annotating():`. The linear-phi ICs do not suppress annotation (they contain only Firedrake `.assign()` and `.interpolate()` calls), but these are harmless when the production driver wraps the entire forward solve in `stop_annotating`.

### sweep_order.py

**15. _build_sweep_order correctness**
Two-branch case correctly places the branch with smallest `|eta|` first. Within each branch, sort order is outward from equilibrium (ascending `|eta|` for negative, ascending `eta` for positive). Index arithmetic (`np.where`, `np.argsort`) uses safe numpy operations with no off-by-one exposure. All-zero case falls to stable argsort (benign).

**16. _apply_predictor safety**
Validates predicted state against `_MAX_PREDICTOR_RATIO=10` and reverts to simple warm-start if too aggressive. Concentration clamping at `1e-10` skips the last component (phi, which can be negative). Quadratic predictor only activates when extrapolation distance is `<= 2x` the max prior gap, preventing runaway extrapolation.

---

## Summary

**Critical issues**: None.

**Warnings (1)**:
- WARNING-1: Cold-failed interior points between non-contiguous cold anchors in Phase 2 are silently left as failed. Low practical risk given the production voltage grid's typical cold-failure topology (failures clustered at high anodic end, above `anchor_hi`), but the behavior is undocumented and silent.

**Notes (4)**:
- NOTE-1: Adjoint tape suppression is correctly handled by the production caller, not internally. No bug, but callers that forget `with adj.stop_annotating()` would pollute the tape.
- NOTE-2: Assembled observables (currents) are not stored in `PerVoltagePointResult`; they require `per_point_callback`. Not a bug in current use.
- NOTE-3: `_solve_warm` wastes one IC computation per voltage (the IC is immediately overwritten by the anchor snapshot). Performance inefficiency, not a correctness issue.
- NOTE-4: `sweep_order.py` functions are unused by the C+D path; they serve `FluxCurve` via re-export. Both are correctly implemented.
