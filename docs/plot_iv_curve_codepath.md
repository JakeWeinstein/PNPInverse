# `scripts/plot_iv_curve_unified.py` — Full Codepath Map

**Last updated:** 2026-05-02

This document traces every function and subfunction reached when
`scripts/plot_iv_curve_unified.py` runs. The script is the canonical
"forward pipeline at TRUE parameters" driver and exercises the
production 3sp + analytic Boltzmann counterion + log-c + log-rate BV
stack documented in
`writeups/WeekOfApr27/PNP Inverse Solver Revised.tex` and
`docs/bv_solver_unified_api.md`.

The map is organized as:

1. Module-load / import-time work.
2. `main()` top-level flow with file:line references.
3. Per-file inventory of what is live on this path.
4. Compressed call graph.

---

## 0. Module-load / import-time

| What | Where | Notes |
|---|---|---|
| `os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")` | the script | Avoids OpenMP duplicate-init crash. |
| `sys.stdout.reconfigure(line_buffering=True)` | the script | Streamed log output. |
| Path bootstrap (PNPInverse root added to `sys.path`) | the script | Standard PNPInverse setup. |
| `import numpy as np` | the script | |

When `_bv_common` is imported (top of `main()`), its module body runs
and reads physical constants from `Nondim.constants`
(FARADAY_CONSTANT, GAS_CONSTANT, DEFAULT_TEMPERATURE_K). Those become
`F_CONST`, `R_GAS`, `T_REF`, `V_T = R_GAS*T_REF/F_CONST` and finally
`I_SCALE = compute_i_scale()` (`scripts/_bv_common.py:152-165`). The
species presets (`THREE_SPECIES_LOGC_BOLTZMANN` etc.) and
`DEFAULT_CLO4_BOLTZMANN_COUNTERION` are also instantiated at import.

---

## 1. `main()` top-level flow

### 1a. `setup_firedrake_env()` — `scripts/_bv_common.py:66`

Sets `FIREDRAKE_TSFC_KERNEL_CACHE_DIR`, `PYOP2_CACHE_DIR`,
`XDG_CACHE_HOME`, `MPLCONFIGDIR`, `OMP_NUM_THREADS=1`. No further
calls.

### 1b. Imports (deferred until inside `main()`)

- `import firedrake as fd`
- `import firedrake.adjoint as adj`
- `from Forward.bv_solver import make_graded_rectangle_mesh, solve_grid_per_voltage_cold_with_warm_fallback`
  - Triggers `Forward/bv_solver/__init__.py:91-126`, which imports:
    - `Forward.bv_solver.mesh` → `make_graded_rectangle_mesh`.
    - `Forward.bv_solver.config` → BV/convergence/Boltzmann config parsers.
    - `Forward.bv_solver.nondim` → BV scaling helpers.
    - `Forward.bv_solver.boltzmann` → `add_boltzmann_counterion_residual`.
    - `Forward.bv_solver.dispatch` → `build_context`/`build_forms`/
      `set_initial_conditions` (which in turn import `forms.py` AND
      `forms_logc.py` at module load).
    - `Forward.bv_solver.solvers` → `forsolve_bv`,
      `solve_bv_with_continuation`, `solve_bv_with_ptc`,
      `solve_bv_with_charge_continuation` (none invoked from this script).
    - `Forward.bv_solver.grid_charge_continuation` → strategy-B
      orchestrator (NOT used here).
    - `Forward.bv_solver.grid_per_voltage` → C+D orchestrator
      (the one this script uses).
- `from Forward.bv_solver.observables import _build_bv_observable_form`
  — `Forward/bv_solver/observables.py:13`.

### 1c. `make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)` — `Forward/bv_solver/mesh.py:37`

- `fd.RectangleMesh(8, 200, 1.0, 1.0)` then in-place
  `coords[:, 1] **= beta`.
- Boundary markers it produces (Firedrake `RectangleMesh`
  convention): 1=left, 2=right, 3=bottom (electrode), 4=top (bulk).

### 1d. `make_bv_solver_params(...)` — `scripts/_bv_common.py:441`

Builds a `Forward.params.SolverParams` (the canonical 11-tuple),
assembling its `solver_options` dict from three sub-configs:

- `_make_bv_convergence_cfg(softplus=False, log_rate=True,
  u_clamp=30.0, formulation="logc")` — `_bv_common.py:285`. Produces
  `{clip_exponent, exponent_clip=50, regularize_concentration,
  conc_floor=1e-12, use_eta_in_bv=True, bv_log_rate=True, u_clamp=30,
  formulation="logc"}`.
- `_make_nondim_cfg()` — `_bv_common.py:324`. Sets `enabled=True`,
  the model-space scales (`D_REF`, `C_SCALE`, `L_REF`, `V_T`,
  `K_SCALE`), and `*_inputs_are_dimensionless=True` for diffusivity/
  concentration/potential/time.
- `_make_bv_bc_cfg(species=THREE_SPECIES_LOGC_BOLTZMANN,
  k0_hat_r1=K0_HAT_R1, k0_hat_r2=K0_HAT_R2,
  alpha_r1=0.627, alpha_r2=0.5,
  E_eq_r1=0.68, E_eq_r2=1.78,
  boltzmann_counterions=[ClO4-])` — `_bv_common.py:342`.
  - Builds two reaction dicts (R1 reversible O₂↔H₂O₂, R2
    irreversible H₂O₂→H₂O), each with `cathodic_conc_factors=[
    {species:2 (H+), power:2, c_ref_nondim=C_HP_HAT}]` (auto-attached
    because `n_species ≥ 3`).
  - Markers: `electrode_marker=3, concentration_marker=4,
    ground_marker=4` (matches `make_graded_rectangle_mesh`).
  - Returns dict including `boltzmann_counterions=[{z:-1,
    c_bulk_nondim=C_CLO4_HAT, phi_clamp=50}]`.
- `SolverParams.from_list([...])` — `Forward/params.py:126`. Final
  11-tuple: `(n_species=3, order=1, dt=0.25, t_end=80.0,
  z=[0,0,1], D=[D_O2_HAT, D_H2O2_HAT, D_HP_HAT],
  a=[A_DEFAULT]*3, phi_applied=0.0,
  c0=[1.0, 1e-4, C_HP_HAT], phi0=0.0, params={...})`.

### 1e. `solve_grid_per_voltage_cold_with_warm_fallback(...)` — `Forward/bv_solver/grid_per_voltage.py:111`

Call site (script lines 174-183):

```python
with adj.stop_annotating():
    result = solve_grid_per_voltage_cold_with_warm_fallback(
        sp,
        phi_applied_values=phi_hat_grid,
        mesh=mesh,
        max_z_steps=20,
        n_substeps_warm=4,
        bisect_depth_warm=3,
        per_point_callback=_grab_observables,
    )
```

`adj.stop_annotating()` keeps the entire SS loop off the pyadjoint
tape — the orchestrator never builds adjoints in this script.

#### 1e-i. Inner setup

Imports lazily inside the function body (`grid_per_voltage.py:181-186`):

- `firedrake as fd`
- `from .dispatch import build_context, build_forms, set_initial_conditions`
- `from .observables import _build_bv_observable_form`
- `from .solvers import _clone_params_with_phi` (only used as the
  list-fallback in `_params_with_phi`; `SolverParams` takes the
  `with_phi_applied` branch).
- `from Forward.params import SolverParams`

Unpacks the 11-tuple, derives `z_nominal`, defines
`_params_with_phi(phi_target)`.

#### 1e-ii. Closures defined

##### `_build_for_voltage(phi_target)` — `grid_per_voltage.py:215`

For each voltage:

1. `sp = _params_with_phi(phi_target)` →
   `SolverParams.with_phi_applied()` →
   `dataclasses.replace(...)` (`Forward/params.py:170`).
2. `ctx = build_context(sp, mesh=mesh)` —
   `Forward/bv_solver/dispatch.py:78` reads
   `params['bv_convergence']['formulation']` (= `"logc"` here) via
   `_get_formulation` and dispatches to
   `build_context_logc(sp, mesh=mesh)` — `forms_logc.py:37`. That
   builds:
   - `V_scalar = FunctionSpace(mesh, "CG", 1)`
   - `W = MixedFunctionSpace([V_scalar]*3 + [V_scalar])` (three
     log-c species + φ).
   - `U`, `U_prev`. Returns ctx with `logc_transform=True`.
3. `ctx = build_forms(ctx, sp)` — `dispatch.py:89` again routes on
   formulation. For `logc`, calls `build_forms_logc(ctx, sp)` —
   `forms_logc.py:72`. This is the heavy lifting:
   - `_get_bv_cfg(params, n)` — `config.py:14`: parses markers,
     k0/alpha legacy lists, Stern.
   - `_get_bv_convergence_cfg(params)` — `config.py:96`: parses
     `clip_exponent, exponent_clip=50, conc_floor, use_eta_in_bv,
     bv_log_rate=True, u_clamp=30, formulation="logc"`.
   - `_get_bv_reactions_cfg(params, n)` — `config.py:200`: parses
     the multi-reaction list (R1, R2) into structured dicts.
   - `build_model_scaling(params, n_species, dt, t_end, D_vals,
     c0_vals, phi_applied, phi0, robin=dummy)` —
     `Nondim/transform.py:172`. Reads the `nondim` config block
     (also `_get_nondim_cfg` — `transform.py:127`); produces
     `dt_model, t_end_model, D_model_vals, c0_model_vals,
     kappa_scale_m_s, charge_rhs_prefactor, poisson_coefficient,
     electromigration_prefactor, phi_applied_model, ...`. Since
     `enabled=True` and inputs are flagged dimensionless, values
     pass through largely unchanged.
   - `_add_bv_reactions_scaling_to_transform(...)` — `nondim.py:104`:
     per-reaction `k0_model`, `c_ref_model`, scales
     `cathodic_conc_factors.c_ref_nondim`, sets `bv_exponent_scale=1.0`
     and `bv_E_eq_model = E_eq/V_T`.
   - Builds Firedrake objects: `D = exp(logD)`, `z`,
     `phi_applied_func` (R-space `Function`), `phi0_func`,
     `E_eq_model_global`, `bv_exp_scale`. Defines a closure
     `_build_eta_clipped(E_eq)` that builds
     `eta_scaled = bv_exp_scale*(phi_applied - E_eq)` (with
     `use_eta_in_bv=True`) and applies the symmetric clip at
     `±exponent_clip=50`.
   - Reconstructs `ci = exp(min/max-clamped ui)` with `_U_CLAMP=30`.
   - Loops `for i in range(n)` building Nernst-Planck residual
     chunks: `(c_i - c_old)/dt · v · dx` +
     `D[i]·c_i·(∇u + z[i]·em·∇φ)·∇v · dx`. Steric is active because
     `A_DEFAULT=0.01` ≠ 0, so `mu_steric` is added.
   - Loops `for j, rxn in enumerate(rxns_scaled)` building each
     `R_j`. Because `bv_log_rate=True` is set, takes the **log-rate**
     branch (`forms_logc.py:325-357`):
     - `log_cathodic = ln(k0) + u_cat - α·n_e·η + Σ power·(u_sp - ln c_ref)`
     - `log_anodic = ln(k0) + u_anod + (1-α)·n_e·η`
     - `R_j = exp(log_cathodic) - exp(log_anodic)`
     - Anodic for R2 collapses to `Constant(0)` because R2 is
       irreversible.
   - Adds `−stoi[i]·R_j · v_list[i] · ds(electrode_marker=3)` per
     species per reaction.
   - Poisson term: `eps·∇φ·∇w·dx − charge_rhs·Σ z_i·c_i·w·dx`.
   - Dirichlet BCs: `u_i = ln(c0_model[i])` at
     `concentration_marker=4`; `φ=0` at `ground_marker=4`;
     `φ = phi_applied_func` at `electrode_marker=3` (no Stern in
     this run).
   - `J_form = derivative(F_res, U)`.
   - `add_boltzmann_counterion_residual(ctx, params)` —
     `boltzmann.py:37`. This:
     - `_get_bv_boltzmann_counterions_cfg(params)` —
       `config.py:134` returns `[{z:-1, c_bulk_nondim=C_CLO4_HAT,
       phi_clamp=50}]`.
     - Creates `R_space` Function `boltzmann_z_scale = 1.0`.
     - Appends `−z_scale·charge_rhs·z·c_bulk·exp(−z·clamp(φ))·w·dx`
       to `F_res`.
     - Recomputes `J_form = derivative(F_res, U)`.
     - Stores `ctx['boltzmann_z_scale']` and
       `ctx['boltzmann_counterions']`.
4. `set_initial_conditions(ctx, sp)` — `dispatch.py:108` →
   `set_initial_conditions_logc(ctx, sp)` —
   `forms_logc.py:479`. Sets `U_prev.sub(i) = ln(c0_model[i])` for
   species, and `U_prev.sub(n) = phi_applied_model · (1 − y)`
   (linear ramp from electrode to bulk). Then `U.assign(U_prev)`.
5. `problem = fd.NonlinearVariationalProblem(F_res, U, bcs, J_form)`;
   `solver = fd.NonlinearVariationalSolver(problem,
   solver_parameters=SNES_OPTS_CHARGED + tightened atol/rtol/stol/
   linesearch overrides)`. Note: `snes_error_if_not_converged` is
   intentionally NOT forced here — non-convergence is handled by the
   orchestrator via checkpoint/rollback.
6. `of_cd = _build_bv_observable_form(ctx, mode="current_density",
   reaction_index=None, scale=1.0)` — `observables.py:13`. Returns
   `1.0 · (R_0 + R_1) · ds(electrode_marker=3)`, used internally
   for SS-detection only.

##### `_make_run_ss(ctx, solver, of_cd)` — `grid_per_voltage.py:243`

Returns a closure `run_ss(max_steps)` that:

- Sets `dt_const = dt_init = 0.25`, then loops up to `max_steps`:
  - `solver.solve()` (PETSc SNES Newton with MUMPS LU).
  - `U_prev.assign(U)`.
  - `fv = float(fd.assemble(of_cd))` — checks Δflux against
    `(ss_rel_tol=1e-4, ss_abs_tol=1e-8)`. After `ss_consec=4`
    consecutive steady steps, returns True.
  - SER step-control: dt grows by `min(prev_delta/delta, 4×)` if
    converging, halves down to `dt_init` otherwise; capped at
    `dt_init·dt_max_ratio = 20·`.

##### `_set_z_factor(ctx, z_val)` — `grid_per_voltage.py:288`

Assigns `z_consts[i] = z_nominal[i]·z_val` and
`boltzmann_z_scale = z_val` (so the analytic ClO4⁻ also ramps).

##### `_solve_cold(orig_idx, V_target_eta)` — `grid_per_voltage.py:298`

Phase-1 strategy C:

1. `_build_for_voltage(V_target)` → `_make_run_ss`.
2. Set `phi_applied_func = V_target` explicitly.
3. `_set_z_factor(ctx, 0.0)` → `run_ss(200)` at z=0. If failed,
   return `(None, 0.0, ctx)`.
4. Linear z-ramp `np.linspace(0, 1, 21)[1:]`, with checkpoint
   (`_snapshot_U`) and rollback (`_restore_U`) on failure. Records
   `achieved_z`.
5. Returns `(snapshot, achieved_z, ctx)` if z reached 1, else
   `(None, achieved_z, ctx)`.

##### `_solve_warm(orig_idx, V_target, V_anchor, anchor_snap)` — `grid_per_voltage.py:331`

Phase-2 strategy D:

1. `_build_for_voltage(V_target)` → `_make_run_ss`.
2. `_restore_U(anchor_snap, U, U_prev)`,
   `_set_z_factor(ctx, 1.0)` (full charge from the start).
3. `_march(v0, v1, depth)` recursive bisection: divides `[v0, v1]`
   into `n_substeps_warm=4` pieces; on substep failure, bisects up
   to `bisect_depth_warm=3`.
4. Final `run_ss(max_ss_steps_warm_final=200)` at exact V_target.

#### 1e-iii. Outer loops

- **Phase 1** (`grid_per_voltage.py:385-412`): for
  `orig_idx in range(n_points)` call `_solve_cold`; on success,
  fill `snapshots[orig_idx]` and call
  `per_point_callback(orig_idx, eta_target, ctx)` — that is
  `_grab_observables` from the script.
- Identify `cold_idxs` (sorted converged indices), `anchor_lo`,
  `anchor_hi`. If no cold success, return early.
- **Phase 2 cathodic walk** (`grid_per_voltage.py:438-477`): march
  from `anchor_lo - 1` down to 0; for each gap, find nearest
  converged neighbor at higher index, call `_solve_warm`; chain
  breaks on failure.
- **Phase 2 anodic walk** (`grid_per_voltage.py:480-515`):
  symmetric upward walk from `anchor_hi + 1`.
- Returns `PerVoltageContinuationResult(points, mesh_dof_count)`.

### 1f. Per-point callback `_grab_observables(orig_idx, _phi_eta, ctx)` — script lines 161-167

For each successful voltage:

- `_build_bv_observable_form(ctx, mode="current_density",
  reaction_index=None, scale=-I_SCALE)` — sums all `R_j` over
  `ds(electrode)` and multiplies by `-I_SCALE` (mA/cm²; sign flips
  so cathodic ORR is negative).
- `_build_bv_observable_form(ctx, mode="peroxide_current",
  reaction_index=None, scale=-I_SCALE)` — `R_0 - R_1` over
  `ds(electrode)` × `-I_SCALE`.
- `fd.assemble(...)` on each → fills `cd_nondim[orig_idx]` and
  `pc_nondim[orig_idx]`.

### 1g. Result post-processing (script lines 186-280)

- For each voltage, copies `point.achieved_z_factor` and
  `point.method` from `result.points`.
- Pure-Python writes: `iv_curve.csv`, `iv_curve.json` (config +
  rows). Optional `matplotlib.use("Agg")` plot to `iv_curve.png`.
  Final per-V summary print.

---

## 2. Per-file inventory of what is live on this path

| File | Functions touched on this path |
|---|---|
| `scripts/plot_iv_curve_unified.py` | `main`, `_grab_observables` |
| `scripts/_bv_common.py` | `setup_firedrake_env`, module-level constants, `compute_i_scale`, `make_bv_solver_params`, `_make_bv_convergence_cfg`, `_make_nondim_cfg`, `_make_bv_bc_cfg`; `THREE_SPECIES_LOGC_BOLTZMANN`, `DEFAULT_CLO4_BOLTZMANN_COUNTERION` |
| `Forward/params.py` | `SolverParams.from_list`, `SolverParams.with_phi_applied` |
| `Forward/bv_solver/__init__.py` | re-exports only |
| `Forward/bv_solver/mesh.py` | `make_graded_rectangle_mesh` |
| `Forward/bv_solver/dispatch.py` | `build_context`, `build_forms`, `set_initial_conditions`, `_get_formulation`, `_params_dict` |
| `Forward/bv_solver/forms_logc.py` | `build_context_logc`, `build_forms_logc` (incl. nested `_build_eta_clipped`), `set_initial_conditions_logc` |
| `Forward/bv_solver/forms.py` | imported only (concentration backend); not invoked because `formulation="logc"` |
| `Forward/bv_solver/config.py` | `_get_bv_cfg`, `_get_bv_convergence_cfg`, `_validate_formulation`, `_default_bv_convergence_cfg`, `_get_bv_reactions_cfg`, `_get_bv_boltzmann_counterions_cfg` |
| `Forward/bv_solver/nondim.py` | `_add_bv_reactions_scaling_to_transform` |
| `Forward/bv_solver/boltzmann.py` | `add_boltzmann_counterion_residual` |
| `Forward/bv_solver/observables.py` | `_build_bv_observable_form` |
| `Forward/bv_solver/grid_per_voltage.py` | `solve_grid_per_voltage_cold_with_warm_fallback`, `_snapshot_U`, `_restore_U`; closures `_params_with_phi`, `_build_for_voltage`, `_make_run_ss`, `_set_z_factor`, `_solve_cold`, `_solve_warm`, `_march` |
| `Forward/bv_solver/solvers.py` | `_clone_params_with_phi` (only as a list-fallback path that is **not** taken when `SolverParams` is passed; `SolverParams.with_phi_applied` is used instead) |
| `Nondim/transform.py` | `build_model_scaling`, `_get_nondim_cfg`, `_get_robin_cfg`, `_as_list`, `_pos`, `_bool` |
| `Nondim/constants.py` | constants only |
| Firedrake | `RectangleMesh`, `FunctionSpace`, `MixedFunctionSpace`, `Function`, `Constant`, `TestFunctions`, `split`, `grad`, `dot`, `exp`, `ln`, `min_value`, `max_value`, `derivative`, `Measure`, `DirichletBC`, `NonlinearVariationalProblem`, `NonlinearVariationalSolver` (PETSc SNES + MUMPS LU), `assemble`; `firedrake.adjoint.stop_annotating` |

### What is NOT in this codepath (despite living next to it)

- `Forward/bv_solver/solvers.py` heavyweight helpers (`forsolve_bv`,
  `solve_bv_with_continuation`, `solve_bv_with_ptc`,
  `solve_bv_with_charge_continuation`) — the orchestrator does its
  own SS loop.
- `Forward/bv_solver/validation.py` (`validate_solution_state`) —
  only `forsolve_bv` calls it; the orchestrator does not.
- `Forward/bv_solver/grid_charge_continuation.py` (Strategy B) —
  only `grid_per_voltage.py` (C+D) is used here.
- `Forward/bv_solver/forms.py` (concentration backend) — not
  invoked because `formulation="logc"`.
- `FluxCurve/bv_point_solve/*` — Strategy A; only used by
  inverse-pipeline scripts.
- pyadjoint tape / k0/α controls (`bv_k0_funcs`, `bv_alpha_funcs`)
  — present in ctx but never read because the whole run is wrapped
  in `adj.stop_annotating()`.

---

## 3. Compressed call graph

```
plot_iv_curve_unified.main()
├─ setup_firedrake_env()                       [_bv_common.py:66]
├─ make_graded_rectangle_mesh(8, 200, 3.0)     [mesh.py:37]
├─ make_bv_solver_params(...)                  [_bv_common.py:441]
│  ├─ _make_bv_convergence_cfg                 [_bv_common.py:285]
│  ├─ _make_nondim_cfg                         [_bv_common.py:324]
│  ├─ _make_bv_bc_cfg(species, ..., boltzmann_counterions=[ClO4-])
│  └─ SolverParams.from_list                   [params.py:126]
└─ solve_grid_per_voltage_cold_with_warm_fallback(...)   [grid_per_voltage.py:111]
   ├─ Phase 1 (C): for each V — _solve_cold
   │  └─ _build_for_voltage(V)
   │     ├─ SolverParams.with_phi_applied         [params.py:170]
   │     ├─ build_context (dispatch.py:78)
   │     │  └─ build_context_logc                 [forms_logc.py:37]
   │     ├─ build_forms (dispatch.py:89)
   │     │  └─ build_forms_logc                   [forms_logc.py:72]
   │     │     ├─ _get_bv_cfg / _get_bv_convergence_cfg / _get_bv_reactions_cfg [config.py]
   │     │     ├─ build_model_scaling             [Nondim/transform.py:172]
   │     │     ├─ _add_bv_reactions_scaling_to_transform [nondim.py:104]
   │     │     ├─ assemble F_res (NP, BV log-rate, Poisson) + bcs + J_form
   │     │     └─ add_boltzmann_counterion_residual    [boltzmann.py:37]
   │     │        └─ _get_bv_boltzmann_counterions_cfg [config.py:134]
   │     ├─ set_initial_conditions (dispatch.py:108)
   │     │  └─ set_initial_conditions_logc        [forms_logc.py:479]
   │     ├─ fd.NonlinearVariationalProblem / Solver (PETSc SNES + MUMPS)
   │     └─ _build_bv_observable_form(mode="current_density", scale=1.0)
   │  ├─ _make_run_ss → run_ss (SS loop with SER dt control + assemble of_cd)
   │  ├─ _set_z_factor(ctx, 0.0) → run_ss(200)         [z=0 anchor]
   │  └─ z-ramp 0→1 in 20 steps with _snapshot_U / _restore_U checkpoints
   ├─ per_point_callback = _grab_observables           [script]
   │  ├─ _build_bv_observable_form(mode="current_density",  scale=-I_SCALE)
   │  ├─ _build_bv_observable_form(mode="peroxide_current", scale=-I_SCALE)
   │  └─ fd.assemble(...) on each
   ├─ Phase 2 (D): warm-walk from anchors
   │  └─ _solve_warm
   │     ├─ _build_for_voltage(V_target)
   │     ├─ _restore_U(anchor_snap), _set_z_factor(ctx, 1.0)
   │     ├─ _march(v_anchor, v_target, depth=0) (4 substeps × bisect_depth=3)
   │     └─ final run_ss(200) at V_target
   └─ returns PerVoltageContinuationResult(points, mesh_dof_count)

→ post-processing: CSV/JSON write, optional matplotlib plot, summary print.
```

---

## Pointers

- The unified API surface this script consumes:
  `docs/bv_solver_unified_api.md`.
- Continuation-strategy rationale (why C+D is used here):
  `docs/CONTINUATION_STRATEGY_HANDOFF.md`.
- Apr 27 production rebuild writeup (formulation/log-rate/Boltzmann
  rationale and validation):
  `writeups/WeekOfApr27/PNP Inverse Solver Revised.tex`.
- Standalone-vs-main parity test for the same stack:
  `scripts/studies/v25_main_pipeline_vs_standalone_logc.py`.
- 4sp ↔ 3sp+Boltzmann reduction validation:
  `scripts/studies/v24_3sp_logc_vs_4sp_validation.py`.
