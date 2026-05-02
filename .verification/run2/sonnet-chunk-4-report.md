# RE-VERIFICATION: grid_per_voltage.py (post-patch) — Chunk 4 Report

**Verifier**: claude-sonnet-4-6  
**Date**: 2026-05-02  
**Files read**:
- `Forward/bv_solver/grid_per_voltage.py` (full, 547 lines)
- `Forward/bv_solver/observables.py` (full, 115 lines)
- `Forward/bv_solver/mesh.py` (full, 65 lines)
- `Forward/bv_solver/solvers.py` (lines 1–25, `_clone_params_with_phi` only)

---

## Summary of Findings

**Previously-identified critical bugs E and H: CONFIRMED CLOSED.**  
**Previously-identified major bug D: CONFIRMED CLOSED.**  
**No new critical bugs introduced by the patch.**  
**Two minor issues found; one question flagged.**

---

## A. Fix 1 — `_build_for_voltage` solve_opts filter (lines 231–243)

### A1. Semantics of the dict-comprehension conditional

```python
_NON_PETSC_KEYS = {"bv_bc", "bv_convergence", "nondim", "robin_bc"}
solve_opts = {
    k: v for k, v in (params or {}).items()
    if isinstance(params, dict) and k not in _NON_PETSC_KEYS
}
```

The generator expression evaluates `(params or {}).items()` **before** the first iteration of the comprehension body. If `params` is a non-dict, non-None, non-falsy object that does not have `.items()`, this raises `AttributeError` regardless of what the `if isinstance(params, dict)` guard would return. The guard only controls whether each yielded `(k, v)` pair is included; it does NOT short-circuit the `.items()` call.

**In practice**: the unpack at line 193 always binds `params` to `solver_params[10]`, which is `SolverParams.solver_options` (a `Dict[str, Any]`), or the 11th element of a plain list (also always a dict in production). The AttributeError path cannot fire with well-formed inputs. However, the code is slightly misleading: the guard `isinstance(params, dict)` inside the comprehension filter _looks_ like it should make the `(params or {})` fallback defensive when `params` is a non-dict, but it does not because `.items()` is called unconditionally. This is a latent correctness mismatch.

**SEVERITY**: minor  
**LOCATION**: `grid_per_voltage.py`, lines 232–235  
**DESCRIPTION**: `isinstance(params, dict)` in the filter body does not protect `.items()` from being called on a non-dict. Safe in production (params is always a dict), but the defensive intent is unfulfilled.  
**SMALLEST FIX**:
```python
solve_opts = {
    k: v for k, v in (params.items() if isinstance(params, dict) else [])
    if k not in _NON_PETSC_KEYS
}
```
This calls `.items()` only when `params` is a dict, and the filter body is simpler.

### A2. _NON_PETSC_KEYS correctness

`{"bv_bc", "bv_convergence", "nondim", "robin_bc"}` — none of these are PETSc option strings (PETSc options use underscore_prefixed names like `snes_type`, `ksp_type`). No legitimate PETSc option is accidentally dropped. **CLOSED / correct.**

### A3. `setdefault` preserves user overrides

`solve_opts.setdefault("snes_error_if_not_converged", True)` only inserts the key if absent. A user-provided `"snes_error_if_not_converged": False` in `solver_options` would survive. In practice the production script does not set this key, so `True` is injected. **Correct.**

### A4. SNES convergence error chain

With `snes_error_if_not_converged=True`, Firedrake raises `firedrake.exceptions.ConvergenceError` (a subclass of `Exception`) when SNES diverges. `run_ss` catches `except Exception: return False` at line 267. The chain `SNES diverges → ConvergenceError → caught by run_ss → returns False → orchestration logic handles failure` is intact. **Bug E: CONFIRMED CLOSED.**

### A5. Bug D (solve_opts pollution with sub-dicts) — CONFIRMED CLOSED

The filter `k not in _NON_PETSC_KEYS` strips `bv_bc`, `bv_convergence`, `nondim`, `robin_bc` — all the nested-dict keys that previously polluted the PETSc option namespace. **Closed.**

---

## B. Fix 2 — `_march` bisection in `_solve_warm` (lines 352–384)

### B1. `v_prev_substep` tracking — success path

At the top of `_march`, `v_prev_substep = float(v0)`. On each successful substep, `v_prev_substep = float(v_sub)` is set before `continue`. After all substeps succeed, `v_prev_substep` holds `float(v1)` (the last substep, which equals `v1` by `linspace` construction). `paf` is also at `v1` because the last `paf.assign(float(v_sub))` ran just before the successful `run_ss`. Return `True`. **Correct.**

### B2. `v_prev_substep` tracking — first-substep failure

First substep `v_sub = substeps[0]`. `v_prev_substep` is still `float(v0)`.  
- `paf.assign(float(v_sub))` runs, SS fails.  
- `_restore_U(ckpt_inner, U, U_prev)`: restores U to the `v0` state (ckpt_inner was snapped immediately before `paf.assign(v_sub)`, which is the start of the first iteration, i.e., the same as `ckpt_outer` for the first iteration).  
- `paf.assign(v_prev_substep)` = `paf.assign(float(v0))`: correct, paf is back at v0.  
- `v_mid = 0.5 * (v_prev_substep + float(v_sub))` = `0.5 * (v0 + substeps[0])`: non-degenerate (v0 < substeps[0] when n_substeps_warm >= 1). **Correct.**

### B3. `v_prev_substep` tracking — mid-loop failure

Say substeps[0] and substeps[1] succeed, substeps[2] fails.  
- `v_prev_substep = float(substeps[1])` at that point.  
- `ckpt_inner` = snapshot taken at the start of the substeps[2] iteration = U at substeps[1] state.  
- `_restore_U(ckpt_inner, U, U_prev)`: U back to substeps[1] state.  
- `paf.assign(v_prev_substep)` = `paf.assign(substeps[1])`: correct, paf matches U's voltage.  
- `v_mid = 0.5 * (substeps[1] + substeps[2])`: non-degenerate.

**Bug H: CONFIRMED CLOSED.**

### B4. `ckpt_inner` capture timing

`ckpt_inner = _snapshot_U(U)` is at line 363, **before** `paf.assign(float(v_sub))` and `run_ss`. On the first iteration, U is at `v0` state (or the previous good substep's state for subsequent iterations). This is the correct "previous good state" to roll back to on failure. **Correct.**

### B5. `ckpt_outer` semantics — mid-march bisection failure propagation

`ckpt_outer` is captured once (line 360) before the for-loop. If a successful bisection in iteration i is followed by a depth-cap failure in iteration i+1:
- `_restore_U(ckpt_outer, U, U_prev)`: restores U to the march-entry state (v0).
- `paf.assign(float(v0))`: restores paf to v0.
- Returns `False`.

This discards work done in iterations 0..i. That is **conservative but semantically correct**: from the parent's perspective, `_march(v0, v1, depth)` returning False means the interval [v0, v1] could not be fully traversed. The parent _march or _solve_warm handles this correctly (by also rolling back and returning False, or breaking the walk). **Correct.**

### B6. paf state after successful child `_march` calls

After `_march(v_prev_substep, v_mid, depth+1)` succeeds, paf is at `v_mid` (the child's last substep = v_mid). After `_march(v_mid, float(v_sub), depth+1)` succeeds, paf is at `float(v_sub)` (the grandchild's last substep). Then `v_prev_substep = float(v_sub)`. Outer loop continues with the next `v_sub` from `substeps`. `ckpt_inner` is taken at the start of that iteration (U at `float(v_sub)` state). **Correct.**

### B7. paf not restored by `_restore_U` — design contract is now explicit

The comment at lines 353–358 documents this explicitly. The `_restore_U` function (lines 105–108) only copies `U.dat` component arrays and calls `U_prev.assign(U)`. It has no knowledge of `paf` (a separate `fd.Constant` or `Function`). Every code path that calls `_restore_U` in `_march` is immediately followed by `paf.assign(...)`. **Design contract is explicit, fully satisfied.**

### B8. Redundant final `run_ss` after `_march` in `_solve_warm` (lines 388–391)

```python
paf.assign(float(V_target_eta))
if not run_ss(max_ss_steps_warm_final):
    return None, ctx
```

At this point, `_march(V_anchor_eta, V_target_eta, 0)` has returned True, meaning paf = V_target_eta already (by the success-path analysis in B1). The `paf.assign` is a no-op assignment (same value). The additional `run_ss` uses `max_ss_steps_warm_final=200` vs the march's `max_ss_steps_warm=150` — this is a "tightening pass" to ensure a firm landing. **Harmless, as noted in the prior pass. No regression introduced.**

---

## C. `_solve_cold` — paf-not-restored bug does NOT apply (verification)

In `_solve_cold` (lines 307–335):
1. Line 314: `ctx["phi_applied_func"].assign(float(V_target_eta))` — paf is set once.
2. The z-ramp loop (lines 323–331) calls `_set_z_factor` and `run_ss` only; it never changes `paf`.
3. On z-step failure: `_restore_U(ckpt, U, U_prev)` then `_set_z_factor(ctx, achieved_z)`. paf is unchanged.
4. The paf-not-restored bug only bites when paf is modified within a loop that also calls `_restore_U`. In `_solve_cold`, paf is modified exactly once before any loop and never inside a loop. **Safe. No paf-restore issue.**

---

## D. Per-point callback contract

### D1. Phase 1 (cold success, line 415–416)

```python
if per_point_callback is not None:
    per_point_callback(orig_idx, eta_target, ctx)
```

Called after `_solve_cold` returns `snap is not None`, with `ctx` from `_solve_cold`. At this point, `ctx["U"]` is at the z=1 converged state for `eta_target`. `_build_bv_observable_form(ctx, mode="current_density", ...)` in the script's callback would assemble from these live forms. **Correct.**

### D2. Phase 2 cathodic walk (lines 477–478) and anodic walk (lines 517–518)

```python
if per_point_callback is not None:
    per_point_callback(orig_idx, eta_target, ctx)
```

Called after `_solve_warm` returns `snap is not None`, with `ctx` from `_solve_warm`. At this point, `_march` returned True and the final `run_ss(max_ss_steps_warm_final)` succeeded, leaving `ctx["U"]` at `V_target_eta` converged state. The `ctx` structure is identical (same keys: `U`, `U_prev`, `phi_applied_func`, `bv_rate_exprs`, `bv_settings`, `mesh`, etc.) because `_build_for_voltage(V_target_eta)` was called at the top of `_solve_warm`. **Correct.**

### D3. ctx structure unchanged by patch

The patch only modified the `solve_opts` computation and the `_march` internal loop variables. `build_context`, `build_forms`, `set_initial_conditions` are not changed. `ctx` keys used by the callback (`bv_rate_exprs`, `bv_settings`, `mesh`) come from these unchanged dispatch functions. **No regression to callback contract.**

---

## E. `_build_for_voltage` — additional edge case

**QUESTION (minor)**:

`_build_for_voltage` calls `_build_bv_observable_form(ctx, mode="current_density", reaction_index=None, scale=1.0)` (line 244). This builds `of_cd` solely for use in `_make_run_ss` as a convergence-detecting flux probe. The `reaction_index=None` is correct because `mode="current_density"` does not use `reaction_index`. **No issue.**

---

## F. `_clone_params_with_phi` in `solvers.py` (lines 19–25)

```python
def _clone_params_with_phi(solver_params, *, phi_applied: float):
    """Return a new SolverParams-like object with phi_applied replaced."""
    if hasattr(solver_params, 'with_phi_applied'):
        return solver_params.with_phi_applied(float(phi_applied))
    lst = list(solver_params)
    lst[7] = float(phi_applied)
    return lst
```

- `hasattr` check: dispatches to `SolverParams.with_phi_applied` (an immutable copy method) if available.  
- Fallback: `list(solver_params)` copies the tuple/list, then replaces index 7 (`phi_applied`). Returns a new list, not mutating the original. **Correct; unchanged by patch.**

`_params_with_phi` in `grid_per_voltage.py` (line 204–210) calls this correctly:
```python
if isinstance(solver_params, SolverParams):
    return solver_params.with_phi_applied(float(phi_applied_target))
return list(_clone_params_with_phi(
    solver_params, phi_applied=float(phi_applied_target)
))
```
The `list(...)` wrapping is redundant (`_clone_params_with_phi` already returns a list for non-SolverParams), but harmless. **No issue.**

---

## G. `observables.py` — `_build_bv_observable_form`

Used in two places in `grid_per_voltage.py`:
1. `_build_for_voltage` (line 244): `mode="current_density"`, `reaction_index=None`, `scale=1.0`. Used as SS convergence probe.
2. Externally by the script's callback: `mode="current_density"`, `scale=-I_SCALE`. Used to extract IV curve data.

The function is purely form-building (no mutation, no state). The `electrode_marker` is read from `ctx["bv_settings"]["electrode_marker"]` with a default of 1. The `bv_rate_exprs` list is read from `ctx`. Both are populated by `build_forms`. **Correct and unchanged.**

---

## H. `mesh.py` — `make_graded_rectangle_mesh`

Mutates `mesh.coordinates.dat.data` in-place after mesh creation. This is the standard Firedrake pattern for post-hoc coordinate transformation (Firedrake's `IntervalMesh`/`RectangleMesh` do not support custom coordinate functions at construction). The mutation is on a freshly-created mesh and is not repeated, so there are no double-application issues. **Correct and unchanged.**

---

## I. Phase 2 cathodic/anodic walk logic — edge case analysis

### I1. Cathodic walk `break` on failure (line 498)

When warm-walk to `orig_idx` fails, the code breaks out of the cathodic loop rather than attempting further-cathodic voltages. This is correct: if V[orig_idx] is unreachable from V[orig_idx+1] (the nearest converged neighbor), then V[orig_idx-1], ..., V[0] are even further from any converged point and will also fail. Breaking avoids wasteful solves. **Correct.**

### I2. Cathodic walk nearest-neighbor search (lines 464–468)

```python
j = orig_idx + 1
while j < n_points and not points[j].converged:
    j += 1
if j >= n_points:
    break
```

This searches rightward for the nearest converged neighbor. Since `anchor_lo` is the lowest cold-success index and the cathodic walk starts from `anchor_lo - 1`, and since the walk progresses downward (`range(anchor_lo - 1, -1, -1)`), all indices `orig_idx + 1, ..., anchor_lo` have been processed (either cold-success or just warm-converged) by the time we get to `orig_idx`. The first `j` that satisfies `points[j].converged` is the most recently warm-converged (or cold-converged) neighbor. **Correct.**

### I3. Anodic walk logic — same analysis, symmetric

The anodic walk is the mirror image of cathodic (upward search for nearest converged neighbor, break on failure). **Correct.**

### I4. MINOR: `points[orig_idx]` not updated on warm-walk failure

In the cathodic and anodic loops, on warm-walk failure (snap is None), the code calls `break` but does NOT update `points[orig_idx]`. The entry created during Phase 1 already exists (method="cold-failed", converged=False). This is functionally correct — the point remains marked as failed. However, the diagnostic info (method string) doesn't reflect that a warm-walk attempt was made. This is cosmetic only.

**SEVERITY**: minor  
**LOCATION**: `grid_per_voltage.py`, lines 492–498 and 532–536  
**DESCRIPTION**: On warm-walk failure, `points[orig_idx]` retains method="cold-failed" rather than "warm-failed". No correctness impact.  
**SMALLEST FIX**: Add `points[orig_idx] = dataclasses.replace(points[orig_idx], method="warm-failed")` before `break` in each walk's failure branch.

---

## Closure Status of Previously-Identified Bugs

| Bug | Severity | Status |
|-----|----------|--------|
| Bug D — solve_opts sub-dict pollution | MAJOR | **CLOSED** by Fix 1 `_NON_PETSC_KEYS` filter |
| Bug E — SNES non-convergence silently accepted | CRITICAL | **CLOSED** by Fix 1 `setdefault("snes_error_if_not_converged", True)` |
| Bug H — _march bisection degenerate midpoint | CRITICAL | **CLOSED** by Fix 2 `v_prev_substep` explicit tracking + `paf.assign(v_prev_substep)` after `_restore_U` |

---

## New Issues Found

| ID | Severity | Location | Description |
|----|----------|----------|-------------|
| N1 | minor | `grid_per_voltage.py:232–235` | `isinstance(params, dict)` guard in comprehension body does not protect `.items()` call from non-dict `params`; safe in production but intent is unfulfilled |
| N2 | minor | `grid_per_voltage.py:492–498, 532–536` | On warm-walk failure, `points[orig_idx].method` stays "cold-failed" instead of being updated to "warm-failed"; cosmetic only |

**No new critical or major bugs introduced by the patch.**

---

## Overall Verdict

The two critical bugs (E, H) and one major bug (D) are **confirmed closed**. The patch is logically sound. The two new findings are both minor: one is a latent defensive-code mismatch that is safe with well-formed inputs, and one is cosmetic diagnostic labeling. Neither affects solver correctness or convergence.
