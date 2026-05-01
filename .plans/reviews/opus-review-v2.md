# Plan Review (v2) -- Unified Grid Charge Continuation Module

**Verdict:** NEEDS REVISION (1 blocking issue, 2 moderate issues)
**Reviewer:** Opus
**Date:** 2026-04-02

---

## Fix Verification

### Fix #1: `params_neutral[4]` mutation on frozen SolverParams -- FIXED

The plan correctly uses `solver_params.with_phi_applied(0.0).with_z_vals([0.0] * n_s)` for `SolverParams` instances, with a fallback path for plain lists. The `with_z_vals()` method (params.py:198-200) returns a new frozen instance via `dataclasses.replace`. No mutation of the original. Correct.

### Fix #2: `_try_timestep` returns True on max_steps exhaustion -- FIXED (with caveat)

`_run_to_steady_state` now correctly returns `(converged: bool, steps_taken: int)` where `converged=False` when the step budget is exhausted. The `_try_timestep` wrapper then re-interprets exhaustion as "usable" (`return True`) with a warning. This is a deliberate design choice: in Phase 1, if the solver ran all N steps without SNES failure, the solution has physically evolved and is likely close enough for warm-starting the next voltage point. Bisection would be wasteful here.

**Caveat:** The predictor history IS updated with a non-converged-but-usable solution (plan line 424-427). This is acceptable because the Lagrange predictor already has safety validation (rejects predictions that deviate >10x from carry state, clamps concentrations). A slightly imprecise warm-start point will not poison the predictor.

Verdict: **FIXED** -- the deliberate True-on-exhaustion is a reasonable engineering tradeoff, well-documented in the code.

### Fix #3: Phase 2 U_prev checkpoint restore -- FIXED

The plan introduces `U_prev_z_ckpt` (line 587) alongside `U_z_ckpt`, and the `_checkpoint()` / `_restore()` helpers (lines 483-489) correctly save and restore both `U` and `U_prev` independently. After restore, `_try_z` calls `_run_to_steady_state`, which resets `dt_const.assign(dt_initial)` at line 208, so the SER state is clean. The observable state (`prev_flux_val`, `prev_delta`, `steady_count`) are all local variables inside `_run_to_steady_state`, so they reset on each call. Correct.

One subtlety: after restoring from checkpoint, `U_prev` holds the *previous* state and `U` holds the *current* state at the checkpoint, so the first `solver.solve()` in `_run_to_steady_state` will advance from the correct starting point, and the first `U_prev.assign(U)` at line 219 will set `U_prev` to the newly solved state. The first convergence check (step 2) will compare the new flux against the step-1 flux. This is correct behavior.

Verdict: **FIXED**

### Fix #4: `_bisect_eta` shared helper -- FIXED

The plan extracts `_bisect_eta` (lines 291-318) as a shared closure used by both bridge fallback and main sweep fallback. After calling `_bisect_eta`:
- On success: `ctx["U"]` holds the converged state at `eta_target`, and `U_ckpt`/`U_prev_ckpt` hold the last successful midpoint state. For the next sweep point, the main loop saves a new neutral solution from `ctx["U"]` (line 422), so the stale checkpoint values are harmless.
- On failure: `ctx["U"]`/`ctx["U_prev"]` hold the best-reached state (at `eta_lo` after last successful midpoint), and the main loop still snapshots this partial state. Correct for continuation.

The `U_ckpt.assign()` calls inside `_bisect_eta` (lines 307-308) do mutate the checkpoint in place. After `_bisect_eta` returns, the caller in the bridge loop (line 374) takes a fresh snapshot `bridge_U_data = tuple(...)` from `ctx["U"]`, not from `U_ckpt`. The main sweep loop (lines 407-408) re-saves `U_ckpt` before each new point. So there is no stale-checkpoint bug.

Verdict: **FIXED**

### Fix #5: Observable-based convergence metric -- PARTIALLY FIXED (blocking issue at z=0)

The plan imports `_build_bv_observable_form` from `FluxCurve.bv_observables` and builds the observable form once after solver construction (lines 177-183). The SER loop then uses `float(fd.assemble(observable_form))` as the convergence metric, matching `bv_point_solve/__init__.py:622`.

**BLOCKING ISSUE: Observable form at z=0 during Phase 1**

The BV rate expressions (`bv_rate_exprs`) depend on:
- `eta_clipped` = `bv_exp_scale * (phi_applied_func - E_eq_model)` (with `use_eta_in_bv=True`, which is the default)
- Surface concentrations `c_surf[i]` (regularized from solution field `U`)
- Rate constants `k0_j` and transfer coefficients `alpha_j`

At z=0 during Phase 1, the Nernst-Planck equations DECOUPLE the electromigration term. However, the BV rate expressions are still active in the weak form (they are UFL expressions referencing `phi_applied_func` and `c_surf`, which are solution-dependent). When `eta_clipped` is nonzero (i.e., when phi_applied != E_eq), the BV rates produce a nonzero flux. So the observable IS nonzero at z=0 for most eta values. It varies with concentration, which evolves during time-stepping.

**However**, the concern about `delta=0` false convergence is valid for a specific scenario: at eta=0 (the first sweep point), with all z=0, if the initial condition already satisfies the BV equilibrium (c_surf = c_ref, eta=0 => exp(0)=1, so R_j = k0*(c_surf - c_ref) which is zero or near-zero when concentrations are at bulk), the observable flux starts at ~0 and stays at ~0. Then:
- `delta = |0 - 0| = 0`
- `is_steady = (0 <= abs_tol)` = True immediately
- After `STEADY_CONSEC=4` steps, convergence is declared

This is actually **correct behavior** -- at eta=0 with initial conditions at equilibrium, the system IS already at steady state and should converge immediately. This is not a false positive; it is a true positive. The first point is always the nearest-to-equilibrium point in the sweep order, so this is expected.

For subsequent Phase 1 points at larger |eta|, the BV rate expressions produce meaningful nonzero flux that changes with concentration, so the observable metric works correctly.

**Revised verdict on the z=0 observable concern:** The observable form IS valid at z=0. The BV rate expressions are UFL symbolic expressions that reference the live solution field -- they do not depend on z_consts directly (z_consts affect the Nernst-Planck flux via the electromigration term `z_i * c_i * grad(phi)`, not the BV rate expression itself). The BV rate depends on surface concentration and overpotential, both of which are nontrivial at z=0 for nonzero eta.

Verdict: **FIXED** -- the z=0 concern is not a real issue. The observable metric works correctly in Phase 1.

---

## New Issues

### ISSUE A (BLOCKING): Forward -> FluxCurve dependency inversion

**Severity: Architectural / Blocking**

The plan introduces `from FluxCurve.bv_observables import _build_bv_observable_form` in `Forward/bv_solver/grid_charge_continuation.py` (line 177). Currently, the dependency graph is strictly:

```
FluxCurve -> Forward (many imports)
Forward -> FluxCurve (zero imports)
```

This is a clean layered architecture where `Forward` is the low-level solver layer and `FluxCurve` is the higher-level inference/analysis layer. The plan would create a **circular dependency** path: `Forward.bv_solver.grid_charge_continuation` -> `FluxCurve.bv_observables` while `FluxCurve.bv_point_solve` -> `Forward.bv_solver`.

Python won't raise an `ImportError` for this (it is a cross-module import, not a direct circular import within `__init__.py`), but it is an architectural regression that will cause problems:
1. `Forward` can no longer be tested or used independently of `FluxCurve`
2. Future refactoring becomes harder (cannot split packages)
3. It violates the existing clean layering

**Recommended fix:** Move `_build_bv_observable_form` to `Forward/bv_solver/observables.py` (or `Forward/steady_state/bv.py` which already has `compute_bv_rates`). Then have `FluxCurve/bv_observables.py` re-export from the new location. The function only depends on `ctx["bv_rate_exprs"]`, `ctx["mesh"]`, and `ctx["bv_settings"]` -- all Forward-layer constructs. It belongs in `Forward`.

### ISSUE B (MODERATE): `_try_z` treats step-budget exhaustion as usable, masking Phase 2 convergence failures

In `_adaptive_z_ramp`, `_try_z` (lines 469-479) returns `True` when the step budget is exhausted:

```python
if not converged and steps == MAX_COLD_STEPS:
    print(f"    z={z_val:.4f}: step budget exhausted ({steps} steps)")
    return True
```

In Phase 1, this is reasonable (warm-started sequential points, solution is close). In Phase 2, this is riskier: jumping from z=0 to z=1.0 is a large perturbation, and if the SER loop runs 100 steps without converging, the solution may be far from steady state. Declaring it "usable" and returning `achieved_z = 1.0` means the point appears fully converged when it may not be.

The existing `solve_bv_with_charge_continuation` (solvers.py:517-531) does NOT treat exhaustion as success -- it catches only SNES exceptions. The fixed-step loop `for _ in range(num_steps): solver.solve()` either completes (all solves succeeded) or throws. There is no concept of "step budget exhaustion" because every step must succeed.

**Recommended fix:** In `_try_z`, return `False` on budget exhaustion (or at minimum, only return `True` if the last few deltas were small, indicating the solution is close to steady state even if not formally converged). Alternatively, reduce `MAX_COLD_STEPS` for z-ramp calls and require actual convergence.

### ISSUE C (MODERATE): Missing `abs_metric` variable in convergence check

In `_run_to_steady_state` (line 228):

```python
is_steady = (rel_metric <= STEADY_REL_TOL) or (delta <= STEADY_ABS_TOL)
```

The plan uses `delta` for the absolute check (correct), but the reference implementation in `bv_point_solve/__init__.py:629` uses `abs_metric`:

```python
abs_metric = delta
is_steady = (rel_metric <= rel_tol) or (abs_metric <= abs_tol)
```

This is functionally identical (`abs_metric = delta`), so it is not a bug. But for consistency and debuggability, the plan should use the same variable name. Minor nit -- not blocking.

---

## Remaining Issues

### R1: No dt reset between Phase 1 and Phase 2

In `bv_point_solve/__init__.py:653-656`, after each forward solve, dt_const is reset to `dt_initial`. The plan's `_run_to_steady_state` does this at the start (line 208: `dt_const.assign(dt_initial)`), which is correct. But verify that between Phase 1 completion and Phase 2 start, the dt_const is in a clean state. Since Phase 2 calls `_run_to_steady_state` which resets dt_const at the top, this is handled. No issue.

### R2: Bridge point bisection uses `prev_solved_eta` as eta_lo

In the bridge loop (line 371):
```python
if not _bisect_eta(prev_solved_eta, eta_b, U_ckpt, U_prev_ckpt):
```

After a successful bridge point, `prev_solved_eta = eta_b` (line 378). If the NEXT bridge point also fails, bisection starts from the last successful bridge eta. This is correct.

But if the first bridge point fails and bisection also fails (line 372), `prev_solved_eta` is NOT updated. The next bridge point will try from the original `prev_solved_eta`, which is fine. The warning is printed. No issue.

### R3: Predictor history updated with bridge solutions

After each bridge point (lines 374-378), the predictor history is updated. This means bridge solutions participate in Lagrange extrapolation for the actual target point. This is correct -- bridge solutions are valid intermediate points that improve predictor accuracy.

---

## Summary

The five previously-identified bugs have all been correctly addressed. The fixes are sound and well-designed: the `_run_to_steady_state` shared helper eliminates duplication, the `(converged, steps_taken)` return tuple enables proper caller decision-making, the separate `U_prev_z_ckpt` prevents false convergence, and `_bisect_eta` is cleanly shared.

**One blocking issue remains:** The `Forward -> FluxCurve` import of `_build_bv_observable_form` creates a dependency inversion that breaks the existing clean layered architecture. The fix is straightforward: move the function to `Forward/bv_solver/observables.py` (it only depends on Forward-layer ctx fields).

**One moderate issue:** `_try_z` treating step-budget exhaustion as success in Phase 2 is riskier than in Phase 1 and differs from the existing `solve_bv_with_charge_continuation` behavior. Consider requiring actual convergence or a near-convergence check.

After addressing Issue A (blocking), the plan is ready for implementation.
