# Plan Review (v2)

**Verdict:** APPROVED WITH MINOR CAVEATS
**Reviewer:** Sonnet 4.6

---

## Fix Verification

### Issue 1: Frozen SolverParams mutation — FIXED

The plan correctly checks `isinstance(solver_params, SolverParams)` and dispatches to
`.with_phi_applied(0.0).with_z_vals([0.0] * n_s)` for the frozen dataclass path (Step 1,
lines 97-98). The fallback list-mutation path (`params_neutral[4] = [0.0] * n_s`) is
retained for legacy callers.

Verified against `Forward/params.py`: `.with_z_vals()` exists at line 198 and calls
`dataclasses.replace(self, z_vals=list(z_vals))`. Correct.

One minor note: the plan also unpacks `solver_params` with:
```python
n_s, order, dt, t_end, z_v, D_v, a_v, phi_applied, c0, phi0, params = solver_params
```
For a frozen `SolverParams`, this works because `__iter__` delegates to `to_list()` (params.py:82-84). No issue.

---

### Issue 2: `_try_timestep` returns True on max_steps exhaustion — FIXED (with design intent preserved)

`_run_to_steady_state` now correctly returns `(False, steps_taken)` when the step budget
is exhausted without STEADY_CONSEC consecutive steady steps (plan line 252). The return
is `False` — not `True`.

`_try_timestep` (Step 5b) then receives this `(converged=False, steps=max_steps)` pair
and deliberately promotes it to `True` with an explicit warning log (plan lines 277-282).
The comment explains the design intent: budget exhaustion without SNES failure means the
solution is "usable but not fully converged."

This is a defensible choice for Phase 1 (voltage sweep), where partial convergence still
provides a better warm-start than failing entirely. The same treatment appears in
`_try_z` (plan lines 478-480) for Phase 2 — also intentional and consistent.

The separation between `_run_to_steady_state` (strict: False on exhaustion) and
`_try_timestep`/`_try_z` (lenient: True on exhaustion) is now clean and explicit.

---

### Issue 3: Phase 2 U_prev checkpoint restore delta=0 bug — FIXED

The plan pre-allocates a separate `U_prev_z_ckpt = fd.Function(ctx["U"])` (plan line 587)
and `_adaptive_z_ramp` accepts both `U_z_ckpt` and `U_prev_z_ckpt` as separate parameters.

The `_checkpoint()` helper saves both `ctx["U"]` to `U_z_ckpt` AND `ctx["U_prev"]` to
`U_prev_z_ckpt` (plan lines 483-485). The `_restore()` helper restores both (plan lines
488-490). This prevents the delta=0 false convergence that occurred when U_prev was
restored to an already-solved state identical to U.

The fix is applied consistently in all three `_restore()` call sites within
`_adaptive_z_ramp`.

Note: the Phase 2 outer loop at plan line 596 also sets `ctx["U_prev"].assign(ctx["U"])`
immediately after loading the neutral solution. This is correct — it ensures U_prev
matches U at the start of z-ramping (z=0 state), so the first PTC step computes a
meaningful delta.

---

### Issue 4: Undefined `_bisect_to_target` — FIXED

The extracted `_bisect_eta(eta_lo, eta_target, U_ckpt, U_prev_ckpt, max_sub=6)` helper
is defined in Step 6a (plan lines 291-318). It is used correctly in:
- Bridge point failure fallback (plan line 371)
- Main sweep point failure fallback (plan line 417)

Both call sites pass the correct `U_ckpt`/`U_prev_ckpt` from Phase 1's pre-allocated
checkpoints. The signature is consistent across all callers.

---

### Issue 5: SER convergence metric mismatch — FIXED

`_run_to_steady_state` uses observable-flux-based convergence (plan lines 222-228),
matching `bv_point_solve/__init__.py:622-648` exactly:
- `flux_val = float(fd.assemble(observable_form))` — matches line 622
- `delta = abs(flux_val - prev_flux_val)` — matches line 625
- `scale_val = max(abs(flux_val), abs(prev_flux_val), STEADY_ABS_TOL)` — matches line 626
- `rel_metric = delta / scale_val` — matches line 627

`observable_form` is built once after solver construction using
`_build_bv_observable_form(ctx, mode="current_density", reaction_index=None, scale=1.0)`.
Verified: `_build_bv_observable_form` exists in `FluxCurve/bv_observables.py` lines 10-60
with the expected signature and "current_density" mode support.

---

## New Issues Introduced by the Fixes

### MINOR-1: `_bisect_eta` mutates U_ckpt/U_prev_ckpt in place, then caller re-uses them

In `_bisect_eta` (plan lines 307-308), on successful midpoint:
```python
U_ckpt.assign(ctx["U"])
U_prev_ckpt.assign(ctx["U_prev"])
eta_lo = eta_mid
```
The Phase 1 checkpoints (`U_ckpt`, `U_prev_ckpt`) are passed by reference and mutated
inside `_bisect_eta`. After `_bisect_eta` returns (whether True or False), `U_ckpt` no
longer holds the pre-attempt state — it holds the best intermediate state reached during
bisection.

In the sweep loop, after each main point (plan lines 407-418), the caller saves a fresh
checkpoint with `U_ckpt.assign(ctx["U"])` at the START of each iteration before calling
`_try_timestep`. So the mutation inside `_bisect_eta` is safe for the MAIN SWEEP (it
gets overwritten at the next iteration's checkpoint).

However, for BRIDGE POINTS (plan lines 365-378), the pattern is:
```python
U_ckpt.assign(ctx["U"])
U_prev_ckpt.assign(ctx["U_prev"])
if not _try_timestep(eta_b):
    ctx["U"].assign(U_ckpt)
    ctx["U_prev"].assign(U_prev_ckpt)
    if not _bisect_eta(prev_solved_eta, eta_b, U_ckpt, U_prev_ckpt):
        print(f"  WARNING: bridge ...")
# After bridge: U_ckpt may now hold a mid-bisection state, not the pre-bridge state
```
After a failed bridge with bisection, `U_ckpt` holds the last successful midpoint
reached (not the pre-bridge state). The subsequent main sweep checkpoint
(`U_ckpt.assign(ctx["U"])` at plan line 407) will overwrite this correctly — so the
practical impact is nil. But this is worth noting as a subtle invariant violation.

**Verdict: MINOR** — no actual bug given the sweep loop's checkpoint-at-start pattern.
Recommend adding a comment in `_bisect_eta` noting that U_ckpt/U_prev_ckpt are
modified in place.

---

### MINOR-2: `_run_to_steady_state` resets dt to `dt_initial` on each call but does NOT reset at exit

The helper sets `dt_const.assign(dt_initial)` at the top (plan line 208), then grows
`dt_current` during the loop. It does NOT reset `dt_const` back to `dt_initial` before
returning. The result is that when `_run_to_steady_state` returns (whether converged or
not), `ctx["dt_const"]` holds whatever `dt_current` was at the last step.

The reference implementation in `bv_point_solve/__init__.py:653-656` explicitly resets:
```python
if dt_const is not None:
    dt_const.assign(dt_initial)
```
after each point's forward solve. The plan's `_run_to_steady_state` does NOT do this.

Each call to `_run_to_steady_state` DOES start by resetting `dt_const.assign(dt_initial)`
(plan line 208), so this is not a correctness bug per se — the next call always resets.
However, in Phase 2, between `_try_z` calls, the dt_const is left in an intermediate
state. If any code between z-ramp steps reads `ctx["dt_const"]` for any purpose (e.g.,
logging, diagnostics), it would see a stale value.

**Verdict: MINOR** — no correctness impact given the reset-on-entry pattern. Add a
`dt_const.assign(dt_initial)` at the end of `_run_to_steady_state` to match the
reference implementation.

---

### MINOR-3: Phase 2 outer loop iterates `range(n_points)` but `n_points` is never defined in the plan

The Phase 2 loop (plan line 590) uses `n_points`:
```python
for orig_idx in range(n_points):
```
But `n_points` is not defined anywhere in the plan's code excerpts. It should be:
```python
n_points = len(phi_applied_values)
```
This must be added before the Phase 2 loop, or the loop line should read
`range(len(phi_applied_values))`. This is a straightforward omission — the implementer
will likely catch it, but it should be explicit.

**Verdict: MINOR** — implementation omission. Add `n_points = len(phi_applied_values)`
before the Phase 2 outer loop.

---

### MINOR-4: `_try_z` in `_adaptive_z_ramp` does not reset dt before calling `_run_to_steady_state`

`_run_to_steady_state` resets `dt_const.assign(dt_initial)` on entry (plan line 208).
But `_adaptive_z_ramp`'s fast-path shortcut at Step 3 calls `_try_z(1.0)` after a
successful midpoint, then immediately calls `_try_z(1.0)` again if the first succeeds
and moves to the next acceleration iteration. Since `_run_to_steady_state` resets on
entry, this is fine. No issue — just confirming the reset-on-entry design is consistent.

**Verdict: NOT AN ISSUE** — included for completeness.

---

### MINOR-5: `predictor.py` re-import introduces a circular dependency risk

Step 4 of the plan says:
```python
# predictor.py
from Forward.bv_solver.sweep_order import _build_sweep_order, _apply_predictor
```
And `sweep_order.py` is a new file in `Forward/bv_solver/`. The plan also says
`grid_charge_continuation.py` imports:
```python
from FluxCurve.bv_observables import _build_bv_observable_form
from Forward.bv_solver.sweep_order import _build_sweep_order
```

Current `predictor.py` already imports from `FluxCurve.bv_observables` (line 15 of
predictor.py). The new `sweep_order.py` is a pure-numpy file with no Firedrake imports.
The chain is: `grid_charge_continuation` → `sweep_order` → (nothing from FluxCurve).

Separately: `grid_charge_continuation` → `FluxCurve.bv_observables` (direct import).
This creates a `Forward` → `FluxCurve` dependency, which the plan explicitly acknowledges
(Step 1: "This avoids a Forward → FluxCurve dependency" for sweep_order itself, but
`grid_charge_continuation.py` still imports directly from FluxCurve at plan line 177).

The plan notes `sweep_order.py` avoids the `Forward` → `FluxCurve` dependency for the
pure numpy helpers. But `grid_charge_continuation.py` directly imports
`_build_bv_observable_form` from `FluxCurve` (plan line 177-183). This is a new
cross-package dependency from `Forward` to `FluxCurve`. Whether this is acceptable
depends on the project's package architecture, which is not stated in the plan.

**Verdict: MINOR** — if `Forward` is meant to be a dependency-free lower layer (as
implied by "This avoids a Forward → FluxCurve dependency"), then the observable import
in `grid_charge_continuation.py` violates that principle. The plan should either
acknowledge this trade-off explicitly or propose moving `_build_bv_observable_form` to
`Forward/bv_solver/observables.py`.

---

## Remaining Issues

All 5 previously identified MAJOR issues are addressed. No MAJOR issues remain.

The following MINOR items from the above analysis are new (not from the v1 review):
- MINOR-1: `_bisect_eta` mutates checkpoint args (safe but undocumented)
- MINOR-2: `dt_const` not reset after `_run_to_steady_state` returns (reset-on-entry is sufficient but non-standard)
- MINOR-3: `n_points` undefined in Phase 2 outer loop
- MINOR-5: `Forward` → `FluxCurve` cross-package dependency via `_build_bv_observable_form`

---

## Summary

The v2 plan correctly fixes all 5 major issues from the first review. The core logic is
sound:
- Frozen dataclass handled immutably via `.with_z_vals()`
- `_run_to_steady_state` returns `(bool, int)` cleanly, with step-budget exhaustion
  propagated to `_try_timestep`/`_try_z` where it is intentionally treated as success
- Phase 2 checkpointing uses separate `U_z_ckpt` / `U_prev_z_ckpt` to prevent delta=0
- `_bisect_eta` is properly extracted and used at all call sites
- Observable-flux-based convergence metric matches the reference implementation exactly

The four remaining issues are all MINOR and non-blocking. MINOR-3 (`n_points` undefined)
is the most likely to cause an immediate `NameError` at runtime if not caught during
implementation — it should be addressed explicitly. The cross-package dependency
(MINOR-5) is an architectural note that may be acceptable given the existing structure.

**The plan is ready for implementation** provided the implementer resolves MINOR-3
before running and keeps the other MINOR items in mind.
