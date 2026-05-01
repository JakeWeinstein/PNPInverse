# Bug Report: Forward Solver Numerics

**Focus:** Numerical and scientific correctness in Forward/ package
**Agent:** Forward Solver Numerics

---

## BUG 4.1: PTC solver never actually modifies `dt_const` in the weak form
**File:** `Forward/bv_solver/solvers.py:297-369`
**Severity:** HIGH
**Description:** The `solve_bv_with_ptc` function is documented as pseudo-transient continuation that adapts `dt_ptc`. However, the code **never calls `ctx["dt_const"].assign(dt_ptc)`**. The variable `dt_ptc` is computed and adapted (lines 333, 356-359) but only used for the stopping criterion (`dt_ptc >= max_dt`). The actual `dt_const` in the weak form remains at its original value throughout the solve. This means PTC is effectively just running multiple fixed-dt time steps with a heuristic stopping criterion -- it is NOT performing pseudo-transient continuation.
**Suggested fix:** Add `ctx["dt_const"].assign(float(dt_ptc))` after the adaptation logic on line 359, so the weak-form time step is actually modified.

## BUG 3.2: Robin solver casts z_vals to int, losing fractional charges
**File:** `Forward/robin_solver.py:130`
**Severity:** MEDIUM
**Description:** The Robin solver creates charge constants as `z = [fd.Constant(int(z_vals[i])) for i in range(n)]`. The `int()` cast truncates any non-integer charge number. The BV solver at `forms.py:165` correctly uses `fd.Constant(float(z_vals[i]))` without truncation.
**Suggested fix:** Change to `z = [fd.Constant(float(z_vals[i])) for i in range(n)]`

## BUG 5.1: Concentration Dirichlet BC uses `c0` instead of `c_inf`
**File:** `Forward/bv_solver/forms.py:341-348`
**Severity:** MEDIUM
**Description:** The BV solver applies concentration Dirichlet BCs using `scaling["c0_model_vals"]` (bulk initial concentrations). In most standard PNP setups `c0 == c_inf`, but if a user intends initial conditions to differ from far-field boundary concentrations, this BC would be wrong. The Robin solver separately tracks `c_inf_model_vals` for the Robin BC itself.
**Suggested fix:** Consider using `c_inf_model_vals` for the concentration Dirichlet BC if the problem distinguishes between initial and boundary concentrations.

## BUG 3.1: Debye length formula uses `potential_scale` instead of `thermal_voltage`
**File:** `Nondim/transform.py:370-374`
**Severity:** LOW
**Description:** The Debye length is computed using `potential_scale` rather than `thermal_voltage_v`. When `potential_scale = V_T` (the default), this is correct. But if a user sets a custom `potential_scale_v` different from `V_T`, the computed Debye length will be wrong. The `scales.py` module (line 186) correctly uses `v_thermal`.
**Suggested fix:** Replace `potential_scale` with `thermal_voltage_v` on line 367.

## BUG 8.1: Default stoichiometry is all -1 for every species
**File:** `Forward/bv_solver/config.py:24`
**Severity:** LOW
**Description:** `stoichiometry = raw.get("stoichiometry", [-1] * n_species)`. The default assumes all species are consumed at the electrode, which is physically unreasonable for most multi-species systems (e.g., should be `[-1, +1]` for a typical two-species system).
**Suggested fix:** Consider raising an error if stoichiometry is not explicitly provided.

## BUG 9.1: `robin_solver.py` `forsolve` can produce zero time steps
**File:** `Forward/robin_solver.py:279`
**Severity:** LOW
**Description:** `num_steps = int(t_end_model / dt_model)` uses `int()` which truncates. If `t_end_model < dt_model`, `num_steps = 0` and no time steps are taken. The BV solver at `solvers.py:54` uses `max(1, int(round(...)))` which is safer.
**Suggested fix:** Change to `num_steps = max(1, int(round(t_end_model / dt_model)))`.

## BUG 4.2: `_get_bv_convergence_cfg` returns empty dict (unreachable)
**File:** `Forward/bv_solver/config.py:43-56`
**Severity:** LOW
**Description:** When `params` is not a dict, returns `{}`. Downstream code accesses keys with bracket notation. Effectively unreachable in normal operation due to upstream validation.

## BUG 6.2: Full options dict passed as PETSc solver_parameters
**File:** `Forward/bv_solver/solvers.py:58, 124, 307, 458`
**Severity:** LOW
**Description:** The full options dictionary (including non-PETSc keys like `bv_bc`, `bv_convergence`, `nondim`) is passed as `solver_parameters` to Firedrake's `NonlinearVariationalSolver`. Modern Firedrake typically ignores unrecognized keys, but this is messy.

---

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 0     |
| HIGH     | 1     |
| MEDIUM   | 2     |
| LOW      | 4     |

**Most impactful:** Bug 4.1 -- the PTC solver is non-functional as pseudo-transient continuation. It never assigns the adapted `dt_ptc` to the weak-form `dt_const`.
