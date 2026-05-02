# Code Correctness Verification Report — Chunk 4
**Verifier:** claude-sonnet-4-6  
**Date:** 2026-05-02  
**Scope:** `grid_per_voltage.py`, `observables.py`, `mesh.py`, `solvers._clone_params_with_phi`

---

## Summary

Two confirmed bugs found (one critical, one major). Several design choices that are correct but
worth documenting as intentional. No issues found with the snapshot/restore logic, z-factor
ramping, Phase 1 structure, Phase 2 structure, observables, mesh, or `_clone_params_with_phi`.

---

## A. `_snapshot_U` / `_restore_U` (grid_per_voltage.py:101-108)

**VERDICT: CORRECT**

`_snapshot_U` returns `tuple(d.data_ro.copy() for d in U.dat)` — the `.copy()` detaches each
NumPy array from the Firedrake DAT; the tuple is immutable. Safe.

`_restore_U(snap, U, U_prev)` writes `dst.data[:] = src` into `U.dat[i].data` for each slot,
then calls `U_prev.assign(U)`. After restore, both `U` and `U_prev` hold the snapshotted state.

Snapshot timing is correct in both call sites:
- `_solve_cold` (line 315): `ckpt = _snapshot_U(U)` is taken immediately before
  `_set_z_factor(ctx, float(z_val))`, i.e. while `U` still holds the previous z's converged
  solution. If the new z fails, `_restore_U(ckpt, U, U_prev)` restores both `U` and `U_prev`
  to the previous z's converged state, and `_set_z_factor(ctx, achieved_z)` reverts the charge
  constants to match. Atomic.
- `_march` (line 347): `ckpt_inner = _snapshot_U(U)` is taken before `paf.assign(v_sub)` and
  `run_ss(...)`. If `run_ss` fails, `_restore_U(ckpt_inner, U, U_prev)` reverts `U`/`U_prev`
  correctly.
- `_march` (line 345): `ckpt_outer` is the state at march entry, used to roll back the entire
  march on deep bisection failure. Correct.

`U_prev` is set to the snapshotted `U` value (not to a separate `U_prev` snapshot). This is
correct because: each `run_ss` step does `U_prev.assign(U)` after a successful `solver.solve()`,
so by the time `_snapshot_U` is called, `U` is the converged state and `U_prev` was just set to
that same state. Restoring `U_prev := U_restored` is therefore equivalent to restoring the actual
`U_prev` at snapshot time.

---

## B. `_set_z_factor` (grid_per_voltage.py:288-293)

**VERDICT: CORRECT**

`n_s` is bound on line 193 from the same `solver_params` unpack:
```python
n_s, order, dt, t_end, z_v, D_v, a_v, _, c0, phi0, params = solver_params
```
`ctx["z_consts"]` is set by `build_forms_logc` (forms_logc.py:454) as `z` which is a list of
`fd.Constant` objects of length `n_species` = `n_s`. The loop `for i in range(n_s)` indexes
the same length list. Correct.

`z_nominal` (lines 200-202) is constructed as:
```python
[z_v] * n_s if np.isscalar(z_v) else list(z_v)
```
truncated to `[:n_s]`, so it always has length `n_s`. Matches `ctx["z_consts"]`.

`ctx.get("boltzmann_z_scale")` is set by `add_boltzmann_counterion_residual`
(boltzmann.py:128) when counterions are configured; otherwise absent. The conditional
assignment is correct.

---

## C. `_params_with_phi` (grid_per_voltage.py:204-210)

**VERDICT: CORRECT (for the script's path)**

The `isinstance(solver_params, SolverParams)` check on line 206 is `True` for the script's path
because `make_bv_solver_params` returns `SolverParams.from_list(...)` (a `SolverParams` instance).
`SolverParams.with_phi_applied(float(phi_applied_target))` calls `dataclasses.replace(self,
phi_applied=float(phi))` — returns a new frozen instance. Correct.

The list fallback (line 208-210) calls `_clone_params_with_phi` and wraps in `list(...)`, returning
a plain Python list. This is only taken for non-`SolverParams` input.

---

## D. `_build_for_voltage` — solver_parameters pollution (grid_per_voltage.py:225-237)

**SEVERITY: MAJOR BUG**
**LOCATION:** `grid_per_voltage.py:228-234`

```python
solve_opts = dict(params) if isinstance(params, dict) else {}
solver = fd.NonlinearVariationalSolver(problem, solver_parameters=solve_opts)
```

`params` here is `solver_params[10]` (the full `solver_options` dict). In the production stack,
this dict contains:

```python
{
    "snes_max_it": 400,
    "snes_atol": 1e-7,
    ...                    # PETSc/SNES keys — OK
    "bv_convergence": {...},   # NOT a PETSc option
    "nondim": {...},           # NOT a PETSc option
    "bv_bc": {...},            # NOT a PETSc option
}
```

`NonlinearVariationalSolver(solver_parameters=solve_opts)` passes this entire dict to PETSc as
solver options. PETSc does not silently ignore unknown options by default; unknown string keys
typically trigger a `PETSc.Error` or at minimum a loud warning on stdout.

**Contrast with `forsolve_bv`** (solvers.py:68-70):
```python
solve_opts = dict(params) if isinstance(params, dict) else {}
solve_opts.setdefault("snes_error_if_not_converged", True)
solver = fd.NonlinearVariationalSolver(problem, solver_parameters=solve_opts)
```
`forsolve_bv` also does not filter out `bv_bc`/`bv_convergence`/`nondim`. This is the same
pattern — so if PETSc truly ignores nested dict values it would not be a new problem introduced
here. However:

1. PETSc option parsing depends on whether Firedrake flattens the dict. Firedrake's
   `NonlinearVariationalSolver` uses `parameters.update()` which converts nested dicts to
   PETSc option strings (e.g. `"bv_convergence_formulation"` → a PETSc option key). Nested
   dicts are thus unintentionally injected into the PETSc options database.
2. The SNES options (flat string keys like `"snes_max_it"`) are valid. The nested dict keys
   (`"bv_convergence"`, `"nondim"`, `"bv_bc"`) produce garbage PETSc option strings that may or
   may not cause a hard error depending on PETSc build configuration
   (`-on_error_abort`/`-options_left`).

**Observed behavior in this codebase:** The script succeeds empirically (v24 8/8 PASS), so either
Firedrake's option flattening skips non-string leaf values, or PETSc silently ignores the
unparseable entries. But this is fragile and version-dependent.

**Smallest fix:** Filter the dict before passing to `NonlinearVariationalSolver`:
```python
_PETSC_SCALAR_TYPES = (int, float, str, bool)
solve_opts = {
    k: v for k, v in params.items()
    if isinstance(v, _PETSC_SCALAR_TYPES)
} if isinstance(params, dict) else {}
```
This keeps all flat PETSc/SNES keys and drops the nested sub-dicts. The same fix should be
applied in `forsolve_bv`, `solve_bv_with_continuation`, and `solve_bv_with_charge_continuation`
for consistency.

---

## E. `_make_run_ss` — SNES non-convergence detection (grid_per_voltage.py:243-283)

**SEVERITY: CRITICAL BUG**
**LOCATION:** `grid_per_voltage.py:229-234` (the solver construction), `grid_per_voltage.py:257-259` (the try/except)

The orchestrator's run_ss loop relies on SNES raising an exception on non-convergence:
```python
try:
    solver.solve()
except Exception:
    return False
```

However, `_build_for_voltage` constructs the solver **without** `snes_error_if_not_converged`:
```python
solver = fd.NonlinearVariationalSolver(
    problem, solver_parameters=solve_opts,
)
```

In Firedrake, the default behavior when SNES reaches `snes_max_it` without satisfying tolerances
is to **complete without raising** unless `snes_error_if_not_converged` is explicitly set to
`True` in the solver parameters. The comment on line 229 says "Don't force
snes_error_if_not_converged — the orchestrator handles non-convergence via
checkpoint+rollback at the adaptive-step level." But this reasoning is circular: the
checkpoint+rollback is triggered only by an exception from `solver.solve()`, which will not be
raised if `snes_error_if_not_converged` is not set.

**Consequence:** When SNES fails to converge (e.g. reaches `snes_max_it=400` without meeting
`snes_atol`/`snes_rtol`/`snes_stol`), `solver.solve()` returns silently. `run_ss` then proceeds
to measure the flux observable on a non-converged iterate, which will often pass the steady-state
plateau test (because a non-converged SNES iterate can look numerically "stuck"). `run_ss`
returns `True` (falsely claiming convergence), and the orchestrator accepts a garbage solution.

**Contrast with `forsolve_bv`** (solvers.py:69):
```python
solve_opts.setdefault("snes_error_if_not_converged", True)
```
That function explicitly sets the flag. `solve_bv_with_continuation` and
`solve_bv_with_charge_continuation` both do the same.

**Smallest fix:** Add to `_build_for_voltage` after building `solve_opts`:
```python
solve_opts["snes_error_if_not_converged"] = True
```
With this, a failed SNES solve raises `firedrake.exceptions.ConvergenceError` (a subclass of
`Exception`), which `run_ss`'s `except Exception` clause catches correctly and returns `False`.

**Why v24 still reported 8/8 PASS:** The production SNES tolerances (`snes_atol=1e-7`,
`snes_stol=1e-12`, `snes_max_it=400`) are tight enough that for well-conditioned voltages, true
non-convergence is rare. For difficult voltages (e.g. `-0.50 V`) the z-ramp may actually keep
each step easy enough for SNES to converge. But the safety net (returning `False` on failure) is
absent, so at edge cases the orchestrator will silently use corrupted iterates.

---

## F. `_solve_cold` (grid_per_voltage.py:298-326)

**VERDICT: CORRECT** (modulo bug E)

Structure:
1. Build fresh context + run_ss closure (fresh `dt_val` per call since `run_ss` is re-created
   for each `_solve_cold` invocation — note `_make_run_ss` returns a closure that initializes
   `dt_val = float(dt_init)` on each call to `run_ss(max_steps)`, so dt state is correctly
   reset between the z=0 step and each z-ramp step). Correct.
2. `_set_z_factor(ctx, 0.0)` before `run_ss(max_ss_steps_cold=200)` at z=0. Correct.
3. Loop `np.linspace(0.0, 1.0, max_z_steps + 1)[1:]` = 20 steps `[0.05, ..., 1.0]`. Correct.
4. Snapshot before `_set_z_factor`, restore and revert z on failure, break. Atomic. Correct.
5. Return `_snapshot_U(U)` (a snapshot of the full-z converged state) alongside `ctx`. The
   returned `ctx` holds the live converged U — correct for the per_point_callback that needs it.

The `achieved_z < 1.0 - 1e-3` threshold (line 324) correctly handles floating-point
representation of the last linspace step (1.0 exactly is representable, so `1.0 - 1e-3 = 0.999`
is always less than 1.0). Safe.

---

## G. `_solve_warm` — redundant final run_ss (grid_per_voltage.py:365-371)

**VERDICT: MINOR / QUESTION (not a bug)**

After `_march(V_anchor, V_target, 0)` succeeds:
```python
paf.assign(float(V_target_eta))
if not run_ss(max_ss_steps_warm_final):
    return None, ctx
```

The last substep of `_march` is `v1 = V_target` exactly (because
`np.linspace(v0, v1, n_substeps_warm+1)[1:][-1] = v1`). So `_march` already ends with
`paf.assign(V_target)` and a converged `run_ss`. The final `paf.assign(V_target); run_ss(200)`
is a redundant re-solve at the same voltage.

This is not a bug — it's a defensive "re-anchor at target" that consumes at most 200 extra
time steps. It could return `True` quickly once the plateau detector fires (ss_consec=4
consecutive near-flat steps). The cost is at most modest. No correctness concern.

---

## H. `_march` — v_prev reads wrong value after failed substep (grid_per_voltage.py:343-363)

**SEVERITY: CRITICAL BUG**
**LOCATION:** `grid_per_voltage.py:354-356`

After a substep `v_sub` fails:
```python
_restore_U(ckpt_inner, U, U_prev)   # restores U/U_prev to pre-substep state
if depth >= bisect_depth_warm:
    _restore_U(ckpt_outer, U, U_prev)
    return False
v_prev = float(paf.dat.data_ro[0])   # BUG: reads current paf value
v_mid = 0.5 * (v_prev + float(v_sub))
```

`_restore_U` restores only `U` and `U_prev`. It does NOT restore `paf` (`ctx["phi_applied_func"]`),
which is a separate R-space `fd.Function`. `paf` was assigned `v_sub` at line 348 before the
failed `run_ss`. After `_restore_U`, `paf.dat.data[0]` still holds `v_sub` (the failed substep's
voltage).

Therefore:
```python
v_prev = float(paf.dat.data_ro[0])  # = v_sub  (NOT the previous substep's voltage)
v_mid  = 0.5 * (v_sub + v_sub)      # = v_sub  (degenerate midpoint)
```

The recursive calls become:
```python
_march(v_sub, v_sub, depth + 1)   # zero-width interval
_march(v_sub, v_sub, depth + 1)   # same
```

These zero-width marches produce `np.linspace(v_sub, v_sub, 5)[1:] = [v_sub, v_sub, v_sub, v_sub]`
as substeps, so each substep does `paf.assign(v_sub)` and re-runs `run_ss` at the same voltage
that just failed, guaranteed to fail again. Both recursive calls return `False`, so `ckpt_outer`
is restored and `_march` returns `False`. This is the **correct outcome** but via degenerate
bisection logic — no progress is made, and the bisection is wasted.

In normal operation this means the warm-walk will fail on any substep that doesn't converge
on the first try, even when there is a valid intermediate voltage that bisection would reach.
The depth budget (bisect_depth_warm=3) is consumed without making voltage progress, and the
walk reports failure. **Voltages that would have been reachable via bisection are reported
as warm-walk failures.**

**Smallest fix:** Track `v_prev` explicitly across loop iterations:
```python
def _march(v0: float, v1: float, depth: int) -> bool:
    substeps = np.linspace(v0, v1, n_substeps_warm + 1)[1:]
    ckpt_outer = _snapshot_U(U)
    v_current = v0                          # <-- track explicitly
    for v_sub in substeps:
        ckpt_inner = _snapshot_U(U)
        paf.assign(float(v_sub))
        if run_ss(max_ss_steps_warm):
            v_current = float(v_sub)        # <-- update on success
            continue
        _restore_U(ckpt_inner, U, U_prev)
        # paf still holds v_sub after restore; reset it to v_current
        paf.assign(float(v_current))        # <-- restore paf too
        if depth >= bisect_depth_warm:
            _restore_U(ckpt_outer, U, U_prev)
            paf.assign(float(v0))           # restore paf to march entry
            return False
        v_mid = 0.5 * (v_current + float(v_sub))
        if not _march(v_current, v_mid, depth + 1):
            _restore_U(ckpt_outer, U, U_prev)
            paf.assign(float(v0))
            return False
        if not _march(v_mid, float(v_sub), depth + 1):
            _restore_U(ckpt_outer, U, U_prev)
            paf.assign(float(v0))
            return False
        v_current = float(v_sub)
    return True
```

Note the fix also resets `paf` to the march entry voltage when restoring `ckpt_outer`, so the
caller sees a consistent (U, U_prev, paf) state on failure.

**Why v24 reported 8/8 PASS despite this bug:** The production cold-start (Phase 1) succeeded
for most voltages in the `V_RHE ∈ [−0.50, +0.10] V` window. Only the most cathodic voltages
(−0.30 to −0.50) required Phase 2 warm-walk, and those walks likely succeeded on the first
substep attempt (adjacent voltages are 0.10 V apart, normalized by V_T ≈ 26 mV gives a step
of ~3.8 in nondim units; with n_substeps=4 each substep is ~1.0 nondim). The bisection path
(depth > 0) may not have been exercised in those tests, so the bug was never triggered.

---

## I. Phase 1 outer loop — callback with live ctx (grid_per_voltage.py:385-412)

**VERDICT: CORRECT**

`_solve_cold` returns `(snap, achieved_z, ctx)` where `ctx` is the context that was built and
converged at `V_target`. The orchestrator passes `ctx` directly to `per_point_callback`:
```python
snap, achieved_z, ctx = _solve_cold(orig_idx, eta_target)
...
if per_point_callback is not None:
    per_point_callback(orig_idx, eta_target, ctx)
```

The `ctx` at this point has:
- `U` holding the converged z=1 solution (or the last successful z if partial).
- `bv_rate_exprs` defined (the UFL expressions for observable assembly).
- `bv_settings` with `electrode_marker`.

The script's `_grab_observables` builds new observable forms from `ctx` and assembles them using
the live `U`. Since the observable forms are UFL expressions that evaluate `U` at assemble time,
and `U` is the converged solution, this is correct.

The callback is only called when `snap is not None` (i.e. `converged=True`), so failed points
do not trigger a callback. This is the correct behavior — calling with a non-converged U would
give garbage observables.

---

## J. Phase 2 cathodic walk (grid_per_voltage.py:438-477)

**VERDICT: CORRECT**

The inner search for `j` (nearest converged neighbor to the right) starts at `j = orig_idx + 1`
and scans upward until `points[j].converged`. Since we start from `anchor_lo - 1` and walk
downward, and the walk breaks on first failure, any `j` found is guaranteed to be in `cold_idxs`
or a previously-warm-converged index. `snapshots[j]` is guaranteed non-None because:
- `snapshots[j]` is set to the snapshot from `_solve_cold` when `converged=True` (line 391).
- `snapshots[j]` is set to the snapshot from `_solve_warm` when warm-converged (line 453).
- The `while j < n_points and not points[j].converged` loop skips any unconverged index, so the
  first `j` found is always converged and has a valid snapshot.

The `if j >= n_points: break` guard handles the case where no converged neighbor exists (would
only happen if `cold_idxs` is empty, but that's already handled before Phase 2 starts).

On failure, `break` is used (not `continue`), so the cathodic walk stops at the first failure —
correct, because there is no converged neighbor closer than the failed one.

---

## K. Phase 2 anodic walk (grid_per_voltage.py:480-515)

**VERDICT: CORRECT**

Symmetric to the cathodic walk. Scans downward for `j = orig_idx - 1` to find nearest converged
neighbor, passes `snapshots[j]` (non-None by same argument as J). Breaks on failure. The inner
while loop `while j >= 0 and not points[j].converged` is the mirror of the cathodic loop.
`if j < 0: break` is the mirror of `if j >= n_points: break`.

---

## L. `observables._build_bv_observable_form` (observables.py:13-68)

**VERDICT: CORRECT**

**current_density mode:** Sums all `R_j` from `ctx["bv_rate_exprs"]` and multiplies by
`scale · ds(electrode_marker)`. With `scale = -I_SCALE` (I_SCALE > 0), cathodic ORR gives
negative total current. Consistent with the script's convention.

**peroxide_current mode:** `R_0 - R_1`. R_0 is the O2 → H2O2 reaction (H2O2 production); R_1 is
the H2O2 → H2O reaction (H2O2 consumption). Net H2O2 production rate = R_0 - R_1. With
`scale = -I_SCALE`, positive production (R_0 > R_1) gives a negative peroxide current (cathodic
convention). This matches the sign used in the inverse pipeline. Correct.

**Sign chain check:** In `forms_logc.py` with `bv_log_rate=True`, the cathodic branch is:
```python
cathodic = exp(ln(k0) + u_cat - alpha * n_e * eta_j)
```
At cathodic overpotential (eta_j < 0), `exp(... - alpha*n_e*eta_j)` has the negative-exponent
term dominating, making cathodic large and positive. `R_j = cathodic - anodic > 0` for cathodic
conditions. The BV residual subtracts `stoichiometry * R_j * v * ds`: consumption of reactants
(negative stoichiometry) reduces flux, producing the correct mass balance.

The observable form uses `R_j` directly (same UFL expression as used in the residual), so it
correctly measures the net reaction rate. With `scale = -I_SCALE`:
- `I_SCALE = n_e * F * (kinetic rate scale)` > 0
- Total current observable = `-I_SCALE * Σ R_j * ds` < 0 for cathodic reaction. Correct.

**electrode_marker:** `int(bv_cfg.get("electrode_marker", 1))`. In the production stack,
`bv_cfg["electrode_marker"] = 3` (bottom boundary of the rectangle mesh). The observable
integrates over `ds(3)` which is the electrode. Correct.

**Fallback default of 1:** If `bv_settings` is missing `electrode_marker`, the function defaults
to marker 1 (left boundary). This would be wrong for the rectangle mesh where the electrode is
marker 3. However, in the production path `bv_settings` is always populated by `build_forms_logc`
with the parsed `bv_cfg` which always contains `electrode_marker`. This fallback is a minor
concern but not triggered in this codebase.

---

## M. `mesh.make_graded_rectangle_mesh` (mesh.py:37-65)

**VERDICT: CORRECT**

```python
mesh = fd.RectangleMesh(Nx, Ny, 1.0, 1.0)
coords = mesh.coordinates.dat.data
coords[:, 1] = coords[:, 1] ** beta
```

In Firedrake, `mesh.coordinates.dat.data` is a writable NumPy array view into the mesh's
coordinate DG1 function. In-place modification of this array directly updates the mesh geometry
without any additional call needed. This is the standard Firedrake pattern for custom mesh
grading.

`RectangleMesh(Nx, Ny, 1.0, 1.0)` boundary markers: Firedrake's convention for
`RectangleMesh(nx, ny, Lx, Ly)` is:
- 1 = left (x=0)
- 2 = right (x=Lx)
- 3 = bottom (y=0, electrode)
- 4 = top (y=Ly, bulk)

This matches the docstring. `beta=3` gives `y -> y^3`, clustering nodes more strongly near y=0
than `beta=2` (`y -> y^2`). For beta > 1, the mapping is convex and increasing on [0,1], so the
unit square geometry is preserved. Correct.

**Potential concern:** After `coords[:, 1] = coords[:, 1] ** beta`, Firedrake must have the
mesh geometry updated for all subsequent assembly operations. This works because the coordinates
are live views; however, if the mesh is used in a context where Firedrake caches mesh-level data
structures (e.g. facet normals, cell volumes) before the coordinate modification, those caches
may be stale. In practice, the mesh is built and then immediately passed to
`solve_grid_per_voltage_cold_with_warm_fallback`, and `build_context_logc` creates function
spaces from it after the modification. The assembly operations happen after the modification is
complete. This is the standard safe usage. No concern.

---

## N. `solvers._clone_params_with_phi` (solvers.py:19-25)

**VERDICT: CORRECT**

```python
def _clone_params_with_phi(solver_params, *, phi_applied: float):
    if hasattr(solver_params, 'with_phi_applied'):
        return solver_params.with_phi_applied(float(phi_applied))
    lst = list(solver_params)
    lst[7] = float(phi_applied)
    return lst
```

For `SolverParams` input: `hasattr(..., 'with_phi_applied')` is True; calls
`dataclasses.replace(self, phi_applied=float(phi))` which returns a new frozen `SolverParams`
instance. Correct.

For list input: creates a plain list copy and sets index 7. Returns a list (not a `SolverParams`).
The caller must treat it as a list. `_params_with_phi` in `grid_per_voltage.py` wraps this in
`list(...)` for clarity (redundant but harmless).

**Reachability from this script:** Unreachable. `grid_per_voltage.py:206` takes the
`isinstance(solver_params, SolverParams)` branch first, calling `solver_params.with_phi_applied(...)`.
`_clone_params_with_phi` is only reached for non-`SolverParams` input, which does not occur in
the script's path.

---

## O. Context isolation across voltages

**VERDICT: CORRECT**

Each call to `_solve_cold` or `_solve_warm` calls `_build_for_voltage` which calls
`build_context(sp, mesh=mesh)`. The mesh is shared, but `build_context_logc` creates new
`fd.Function` and `fd.MixedFunctionSpace` objects per call. The returned `ctx` is independent
per voltage: different `U`, `U_prev`, `F_res`, `J_form`, `bcs`, `z_consts`, `phi_applied_func`,
`dt_const`, `bv_rate_exprs`.

The `NonlinearVariationalProblem` and `NonlinearVariationalSolver` are local to
`_build_for_voltage`. They are not stored in any shared dict.

The only shared mutable state across voltages is `snapshots` (a list of numpy tuple snapshots)
and `points` (a dict of `PerVoltagePointResult` frozen dataclasses). Both are written once per
voltage and never mutated after. Correct.

---

## P. `adj.stop_annotating()` leakage

**VERDICT: NO ISSUE**

The script wraps the orchestrator call with `with adj.stop_annotating()`. The orchestrator does
not import or call `firedrake.adjoint.continue_annotating()` or start a tape. The only adjoint-
relevant call inside the orchestrator is `solver.solve()`, which is called inside the
`adj.stop_annotating()` context and therefore not annotated. The `_build_for_voltage` function
creates `NonlinearVariationalProblem` and `NonlinearVariationalSolver` inside the same context,
which would also be unannotated.

There is no mechanism by which the orchestrator could re-enable annotation that would defeat the
script's `with adj.stop_annotating()` block.

---

## Severity Summary

| ID | Finding | Severity | File:Line |
|----|---------|----------|-----------|
| E  | SNES non-convergence not detected: missing `snes_error_if_not_converged=True` in `_build_for_voltage`; run_ss silently accepts non-converged iterates | CRITICAL | grid_per_voltage.py:232-234 |
| H  | `_march` reads `paf.dat.data_ro[0]` after `_restore_U`; paf is not restored, so `v_prev = v_sub` and bisection is degenerate | CRITICAL | grid_per_voltage.py:355-356 |
| D  | `solve_opts = dict(params)` passes `bv_bc`/`bv_convergence`/`nondim` sub-dicts as PETSc option keys; fragile/version-dependent | MAJOR | grid_per_voltage.py:228 |
| G  | Final `paf.assign(V_target); run_ss(...)` in `_solve_warm` is redundant (last march substep already ends at V_target) | MINOR (not a bug) | grid_per_voltage.py:368-370 |

All other items (A, B, C, F, I, J, K, L, M, N, O, P) are correct.

---

## Fixes Required Before Production Use

**Fix 1 (Critical — E):** In `_build_for_voltage`, set `snes_error_if_not_converged = True`
so the `except Exception` in `run_ss` actually catches SNES failures:
```python
solve_opts["snes_error_if_not_converged"] = True   # add after building solve_opts
solver = fd.NonlinearVariationalSolver(problem, solver_parameters=solve_opts)
```

**Fix 2 (Critical — H):** In `_march`, track `v_current` explicitly across substeps and reset
`paf` after `_restore_U`. See the full corrected implementation in section H above.

**Fix 3 (Major — D):** Filter `solve_opts` to exclude nested dict values before passing to
`NonlinearVariationalSolver`. Filter pattern: keep only entries whose values are `int`, `float`,
`str`, or `bool`.
