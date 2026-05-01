# Round 2 Bug Audit: Forward/ and Nondim/ Packages

**Date**: 2026-03-17
**Auditor**: Claude Opus 4.6
**Scope**: All `.py` files in `Forward/` and `Nondim/`

---

## Verified Fixes (Round 1 fixes confirmed correct)

### 1. PTC dt_const fix -- Forward/bv_solver/forms.py line 177
**Status**: VERIFIED CORRECT
The `dt_const` is now created as `fd.Constant(float(scaling["dt_model"]))` (mutable),
and `solve_bv_with_ptc` in `solvers.py` line 363 properly calls
`ctx["dt_const"].assign(dt_ptc)` to push the adapted pseudo-timestep into the weak form.
The `dt_const` is stored in `ctx` at forms.py line 360. No issues.

### 2. conv_cfg defaults fix -- Forward/bv_solver/config.py lines 46-54, 56-64
**Status**: VERIFIED CORRECT
`_get_bv_convergence_cfg` now returns complete default dicts (including `packing_floor`
and `softplus_regularization`) when `params` is not a dict or when `bv_convergence` is
not a dict. Both early-return paths have identical key sets matching the parsed path
(lines 66-74). No missing keys.

### 3. setdefault fix -- Forward/steady_state/bv.py lines 49, 108
**Status**: VERIFIED CORRECT
`configure_bv_solver_params` uses `opts.setdefault("bv_bc", {})` (line 49) and then
mutates the returned reference `bv_cfg` in-place. The same `opts` dict is later passed
to `dataclasses.replace(..., solver_options=opts)`, so mutations to `bv_cfg` are
correctly reflected. Same pattern used in `steady_state/robin.py` lines 50, 68. Correct.

### 4. float z_vals fix -- Forward/robin_solver.py line 130
**Status**: VERIFIED CORRECT
`z = [fd.Constant(float(z_vals[i])) for i in range(n)]` -- uses `float()`, which
correctly handles both int and float charge valences for Firedrake `Constant`.

### 5. num_steps rounding -- Forward/robin_solver.py line 279, Forward/dirichlet_solver.py line 232
**Status**: VERIFIED CORRECT
Robin: `num_steps = max(1, int(round(t_end_model / dt_model)))` (line 279).
Dirichlet: `num_steps = max(1, int(round(t_end / dt)))` (line 232).
Both use `round()` before `int()` to avoid truncation of floating-point division.
BV solver (solvers.py lines 54, 128, 456) also uses the same pattern. Consistent.

### 6. except ImportError -- Forward/steady_state/common.py lines 27-29
**Status**: VERIFIED CORRECT
`try: import firedrake.adjoint as adj / except ImportError: adj = None` correctly
handles environments where pyadjoint is not installed. The `_maybe_stop_annotating`
context manager (lines 304-311) properly checks `adj is None` before using it.

### 7. except ImportError -- Forward/plotter.py lines 21-23
**Status**: VERIFIED CORRECT
`try: import imageio / except ImportError: imageio = None` and all downstream usage
guards on `imageio is not None` (lines 241, 250, 281). Falls back to PIL or
matplotlib writers.

### 8. warnings import, removed import copy -- Nondim/transform.py
**Status**: VERIFIED CORRECT
`import warnings` at line 72. Used for 5 warning calls (lines 336-359) about
auto-computed scales when inputs are marked dimensionless. No `import copy` present.
All dict construction uses `dict(scaling)` (shallow copy sufficient since values are
scalars/lists).

---

## NEW Bugs Found (introduced or exposed by round 1 fixes)

### Bug N1: Dirichlet solver truncates z_vals to int, losing fractional charges
- **Severity**: MEDIUM
- **File**: `Forward/dirichlet_solver.py`
- **Line**: 118
- **Description**: `z = [fd.Constant(int(z_vals[i])) for i in range(n)]` casts charge
  valences to `int`. While the robin solver (line 130) and BV solver (forms.py line 165)
  both correctly use `float(z_vals[i])`, the Dirichlet solver truncates. This would
  silently produce wrong results for any species with a non-integer effective charge
  (e.g., z=0.5 in mean-field models). Even for integer charges, `int()` is unnecessary
  since `fd.Constant` accepts float.
- **Fix suggestion**: Change `int(z_vals[i])` to `float(z_vals[i])` to match the other
  two solvers.

---

## REMAINING Bugs (missed in round 1)

### Bug R1: Legacy list path in configure_bv_solver_params loses bv_cfg mutations
- **Severity**: MEDIUM
- **File**: `Forward/steady_state/bv.py`
- **Lines**: 108-138
- **Description**: In the legacy (non-SolverParams) branch, both the `k0_values` block
  (line 108) and `alpha_values` block (line 123) independently call
  `bv_cfg = p.get("bv_bc", {})`. If `"bv_bc"` is not already in `p`, `.get()` returns
  a new empty dict each time that is never stored back into `p`. Mutations to
  `bv_cfg["k0"]` on line 120 would be lost. The SolverParams branch correctly uses
  `setdefault` (line 49), but the legacy branch does not.
  Note: This only matters if `bv_bc` is missing from `params[10]` AND both k0 and alpha
  are being set, which is unlikely in practice since BV configs always include `bv_bc`.
- **Fix suggestion**: Change `p.get("bv_bc", {})` to `p.setdefault("bv_bc", {})` on
  lines 108 and 123 (or factor out to a single call before both blocks).

### Bug R2: Robin sweep does not warm-start (rebuilds context per voltage point)
- **Severity**: LOW (performance, not correctness)
- **File**: `Forward/steady_state/robin.py`
- **Lines**: 240-253
- **Description**: `sweep_phi_applied_steady_flux` calls
  `solve_to_steady_state_for_phi_applied` in a loop, which rebuilds the mesh, function
  spaces, and solver for every voltage point. The BV equivalent
  (`sweep_phi_applied_steady_bv_flux` in `bv.py` lines 311-423) correctly uses
  warm-start: builds once, sweeps `phi_applied_func.assign()`. The Robin sweep wastes
  significant computation rebuilding for each point and does not carry the converged
  solution forward as an initial guess for the next voltage.
- **Fix suggestion**: Refactor `sweep_phi_applied_steady_flux` to match the BV pattern:
  build context once, loop over voltages with `ctx["phi_applied_func"].assign()`.

### Bug R3: BV steady-state solver passes solver_options directly to NonlinearVariationalSolver
- **Severity**: LOW
- **File**: `Forward/steady_state/bv.py`
- **Line**: 231
- **Description**: `fd.NonlinearVariationalSolver(problem, solver_parameters=params.solver_options)`
  passes the entire `solver_options` dict (which contains sub-dicts like `"bv_bc"`,
  `"bv_convergence"`, `"nondim"`, `"robin_bc"`) as PETSc solver parameters. PETSc will
  silently ignore unknown keys, so this works but is messy. The sub-dicts are not PETSc
  options and could cause warnings in strict Firedrake builds.
- **Fix suggestion**: Filter `solver_options` to only PETSc-recognized keys before
  passing, or extract the PETSc options into a separate sub-dict.

### Bug R4: Inconsistent z_vals type in SolverParams vs solver consumption
- **Severity**: LOW
- **File**: `Forward/params.py`
- **Line**: 50
- **Description**: `SolverParams` declares `z_vals: List[float]` and `__post_init__`
  stores `list(self.z_vals)` without converting elements to float. If constructed with
  integer charge valences (e.g., `z_vals=[1, -1]`), the elements remain `int`. Combined
  with Bug N1 above, downstream consumers may get different behavior depending on whether
  they cast to `int` or `float`. The SolverParams type annotation says `List[float]`
  but does not enforce it.
- **Fix suggestion**: In `__post_init__`, normalize z_vals:
  `object.__setattr__(self, "z_vals", [float(v) for v in self.z_vals])`

### Bug R5: Potential division by zero in steady-state BV if all reaction rates are zero
- **Severity**: LOW
- **File**: `Forward/steady_state/bv.py`
- **Lines**: 258-263
- **Description**: The steady-state metric computation uses
  `scale = max(abs(current_density), abs(prev_current), float(max(steady.absolute_tolerance, 1e-16)))`.
  The `max` with `absolute_tolerance` prevents division by zero, so this is technically
  safe. However, when `current_density` and `prev_current` are both exactly 0.0 (which
  happens for neutral species or at equilibrium), `delta=0` and `rel_metric=0`, which
  immediately satisfies the steady-state criterion. This may cause premature convergence
  declaration at the very first comparison step (step 2) before the solution has actually
  evolved.
- **Fix suggestion**: Consider requiring a minimum number of steps before checking
  steady-state, or start `prev_current` at a sentinel value.

### Bug R6: noise.py _add_percent_noise does not handle NaN values
- **Severity**: LOW
- **File**: `Forward/noise.py`
- **Line**: 31-40
- **Description**: The local `_add_percent_noise` helper in `noise.py` does not filter
  NaN values before computing RMS, unlike the more robust version in
  `steady_state/common.py` (lines 191-205) which has `finite_mask = np.isfinite(v)`.
  If a solver produces NaN in any field component, the RMS and noise will propagate NaN.
- **Fix suggestion**: Either replace with the robust version from `common.py`, or add a
  finite-value filter before the RMS computation.

---

## Summary

| Category | Count |
|----------|-------|
| Verified round-1 fixes | 8 |
| New bugs from fixes | 1 (N1) |
| Remaining bugs missed | 6 (R1-R6) |

**Critical/High**: None
**Medium**: 2 (N1, R1)
**Low**: 4 (R2, R3, R4, R5, R6)

The round 1 fixes were all applied correctly with no syntax errors or broken logic.
The most actionable finding is Bug N1 (`int(z_vals[i])` in dirichlet_solver.py) which
is an inconsistency with the other two solvers that could produce silently wrong results
for non-integer charge species.
