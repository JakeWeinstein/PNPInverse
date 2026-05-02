# BV-PNP Forward Solver: Unified API

**Last updated:** 2026-05-01

This is the canonical reference for calling the BV-PNP forward solver
after the Apr 27 production rebuild
(`writeups/WeekOfApr27/PNP Inverse Solver Revised.tex`).  The three
stacked changes from that rebuild — 3-species + analytic Boltzmann
counterion, log-concentration primary variable, log-rate Butler-Volmer
— are all selectable via configuration flags through a single
formulation dispatcher.  No inline helpers required.

The legacy v13 4-species concentration path remains the default for
backward compatibility, so existing inverse scripts (`v13` family)
keep working unchanged.

## TL;DR

```python
from scripts._bv_common import (
    make_bv_solver_params,
    THREE_SPECIES_LOGC_BOLTZMANN,
    DEFAULT_CLO4_BOLTZMANN_COUNTERION,
    SNES_OPTS_CHARGED, V_T, I_SCALE,
    K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2,
)
from Forward.bv_solver import (
    make_graded_rectangle_mesh,
    solve_grid_per_voltage_cold_with_warm_fallback,
)
from Forward.bv_solver.observables import _build_bv_observable_form
import firedrake as fd
import firedrake.adjoint as adj
import numpy as np

mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)

# Build solver params for the production stack:
#   3sp + Boltzmann counterion + log-c + log-rate BV
sp = make_bv_solver_params(
    eta_hat=0.0, dt=0.25, t_end=80.0,
    species=THREE_SPECIES_LOGC_BOLTZMANN,
    snes_opts=SNES_OPTS_CHARGED,
    formulation="logc",                                # Change 2
    log_rate=True,                                     # Change 3
    boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION],  # Change 1
    k0_hat_r1=K0_HAT_R1, k0_hat_r2=K0_HAT_R2,
    alpha_r1=ALPHA_R1,   alpha_r2=ALPHA_R2,
    E_eq_r1=0.68,        E_eq_r2=1.78,
)

V_GRID = np.array([-0.50, -0.40, -0.30, -0.20, -0.10,
                   0.00, 0.05, 0.10])
phi_hat_grid = V_GRID / V_T

cd = np.full(len(V_GRID), np.nan)
pc = np.full(len(V_GRID), np.nan)

def _grab(orig_idx, _phi, ctx):
    f_cd = _build_bv_observable_form(
        ctx, mode="current_density", reaction_index=None, scale=-I_SCALE)
    f_pc = _build_bv_observable_form(
        ctx, mode="peroxide_current", reaction_index=None, scale=-I_SCALE)
    cd[orig_idx] = float(fd.assemble(f_cd))
    pc[orig_idx] = float(fd.assemble(f_pc))

with adj.stop_annotating():
    res = solve_grid_per_voltage_cold_with_warm_fallback(
        sp,
        phi_applied_values=phi_hat_grid,
        mesh=mesh,
        per_point_callback=_grab,
    )

# res.points[i].converged / .achieved_z_factor / .method per voltage
# cd, pc populated in place by the callback for every converged V.
```

## Architecture

```
Forward/bv_solver/
    __init__.py                # public re-exports + dispatcher
    dispatch.py                # build_context / build_forms /
                               #   set_initial_conditions dispatch on
                               #   params['bv_convergence']['formulation']
    forms.py                   # concentration backend (legacy, default)
    forms_logc.py              # log-c backend (production)
    boltzmann.py               # add_boltzmann_counterion_residual()
                               #   — called automatically by both backends
                               #     when bv_bc.boltzmann_counterions is set
    config.py                  # parses the three new toggles
    grid_charge_continuation.py    # B-shape orchestrator (legacy 4sp path)
    grid_per_voltage.py            # C+D orchestrator (production stack)
    solvers.py                 # per-point helpers (forsolve_bv,
                               #   solve_bv_with_continuation, etc.)
    validation.py              # is_logc-aware physics validation
    observables.py             # current_density / peroxide_current /
                               #   reaction-N observables (formulation-agnostic)

FluxCurve/bv_point_solve/      # inverse-problem pipeline
    __init__.py                # solve_bv_curve_points_with_warmstart
                               #   automatically routes via the dispatcher
```

## The three config flags (`params['bv_convergence']` / `params['bv_bc']`)

| Flag | Location | Default | Effect |
|---|---|---|---|
| `formulation` | `params['bv_convergence']['formulation']` | `"concentration"` | `"concentration"` → `forms.py` (primary unknown `c_i`).  `"logc"` → `forms_logc.py` (primary unknown `u_i = ln c_i`). |
| `bv_log_rate` | `params['bv_convergence']['bv_log_rate']` | `False` | Compute the BV rate as `r = exp(log r)` with surface concentrations entering additively inside the exponent.  Eliminates the surface-clip phantom R₂ sink at high anodic η. |
| `boltzmann_counterions` | `params['bv_bc']['boltzmann_counterions']` | `[]` (empty) | List of analytic Boltzmann counterions in Poisson (PBNP reduction).  Each entry is `{"z": int, "c_bulk_nondim": float, "phi_clamp": float}`.  When non-empty, the forms modules append `−charge_rhs · z · c_bulk · exp(−z·φ) · w · dx · z_scale` per ion to the residual.  `z_scale` is a Function exposed as `ctx['boltzmann_z_scale']` so orchestrators can ramp it alongside dynamic z's during continuation. |

The three flags are independent of each other, but the production
stack uses all three together.  See
`scripts/_bv_common.py:make_bv_solver_params(...)` for the canonical
factory.

## Pipeline entry points

| Function | Strategy | When to use |
|---|---|---|
| `Forward.bv_solver.solve_grid_per_voltage_cold_with_warm_fallback` | C+D — per-V cold + z-ramp, then warm-walk from cold anchors with paf substepping + bisection | Production logc+Boltzmann grid sweep; covers V_RHE ∈ [−0.50, +0.60] V on the writeup's grid. |
| `Forward.bv_solver.solve_grid_with_charge_continuation` | B — neutral V-sweep at z=0 then per-V z-ramp | Legacy 4sp concentration path.  Works for the 4sp case across V_RHE ∈ [−0.50, +0.10] V.  Now also routes via the dispatcher and respects `boltzmann_z_scale`, but B's V-sweep at z=0 has BV-flux fragility for the logc stack — prefer C+D for the new formulation. |
| `FluxCurve.bv_point_solve.solve_bv_curve_points_with_warmstart` | A — sequential two-branch warm-start at full z, with bridge points and recovery | Inverse-problem pipeline used by v13 (`Infer_BVMaster_charged_v13_ultimate.py`).  Does the per-V adjoint solves needed by `BVFluxCurveInferenceRequest`.  Routes via the dispatcher. |
| `Forward.bv_solver.solvers.forsolve_bv` / `solve_bv_with_continuation` / `solve_bv_with_ptc` / `solve_bv_with_charge_continuation` | Single-V helpers | Lower-level — pre-built ctx, time-stepping, etc. |

## Choosing a species preset

Defined in `scripts/_bv_common.py`:

| Preset | Species | Use with |
|---|---|---|
| `FOUR_SPECIES_CHARGED` | O₂, H₂O₂, H⁺, ClO₄⁻ (all dynamic) | `formulation="concentration"`, no Boltzmann counterion.  v13 path. |
| `THREE_SPECIES_LOGC_BOLTZMANN` | O₂, H₂O₂, H⁺ (dynamic) + ClO₄⁻ analytic | `formulation="logc"`, `boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION]`.  Production stack. |
| `TWO_SPECIES_NEUTRAL` | O₂, H₂O₂ only, no charges | Neutral-system tests; not used in production inverse work. |

H₂O₂ in `THREE_SPECIES_LOGC_BOLTZMANN` is initialized at a small
positive seed `H2O2_SEED_NONDIM = 1e-4` so `ln(c0_H2O2)` is finite
in the log-c primary variable.

## Default Boltzmann counterion

```python
from scripts._bv_common import DEFAULT_CLO4_BOLTZMANN_COUNTERION

# {"z": -1, "c_bulk_nondim": C_CLO4_HAT, "phi_clamp": 50.0}
# matches the inline add_boltzmann() helper used in scripts/studies/v18*
```

## Common patterns

### Forward target curve at TRUE parameters

```python
sp = make_bv_solver_params(..., formulation="logc", log_rate=True,
                            boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION])
res = solve_grid_per_voltage_cold_with_warm_fallback(
    sp, phi_applied_values=V_GRID/V_T, mesh=mesh,
    per_point_callback=extract_observables_callback,
)
```

### Inverse problem (FluxCurve / v13-style)

```python
sp = make_bv_solver_params(...,
    formulation="logc", log_rate=True,
    boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION])
# Pass sp into BVFluxCurveInferenceRequest.base_solver_params; the
# FluxCurve pipeline picks up the formulation via the dispatcher.
```

### Single-V cold solve with adjoint annotation

```python
import firedrake.adjoint as adj
from Forward.bv_solver import build_context, build_forms, set_initial_conditions

sp = make_bv_solver_params(..., formulation="logc", log_rate=True,
                            boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION])
ctx = build_context(sp, mesh=mesh)
ctx = build_forms(ctx, sp)
set_initial_conditions(ctx, sp)

# ... cold-ramp z (and ctx["boltzmann_z_scale"]) from 0 to 1 unannotated,
# then re-solve a few SS steps with annotation on for the adjoint tape.
```

The pyadjoint controls are exposed as `ctx['bv_k0_funcs']` (one per
reaction) and `ctx['bv_alpha_funcs']` (one per reaction), in both
formulations.

## Backward compatibility

Existing v13/v15/v16 scripts that call `make_bv_solver_params(...)`
without the new keyword arguments get:

- `formulation="concentration"` (legacy `forms.py` backend)
- `log_rate=False`
- `boltzmann_counterions=None` (no Boltzmann residual)
- `species=FOUR_SPECIES_CHARGED` (4sp PNP)

That is exactly what they were doing pre-patch, so behavior is
unchanged.

## Validation reference

The canonical numerical-equivalence test is
`scripts/studies/v25_main_pipeline_vs_standalone_logc.py`.  At
production resolution (Ny=200) on V_RHE ∈ [−0.50, +0.10] V it shows:

```
overlap voltages: 5/8           (v18 cold-only doesn't reach below -0.20)
max |Δcd|/cd_max:  0.0002 %
max |Δpc|/pc_max:  0.0003 %
5/5 PASS @ 1% tolerance
```

The patched main pipeline additionally extends to V_RHE = −0.30, −0.40,
−0.50 via the warm-walk fallback, exactly reproducing v24's full
window.

For the broader 4sp-vs-3sp+Boltzmann reduction equivalence (i.e. that
the new physical reduction matches the original 4sp PNP within F2
tolerance), see `StudyResults/v24_3sp_logc_vs_4sp_validation/` and
`scripts/studies/v24_3sp_logc_vs_4sp_validation.py`.

## When to call which orchestrator

- **Building a target curve at known parameters across a grid** →
  `solve_grid_per_voltage_cold_with_warm_fallback`.
- **Running an inverse with adjoint Jacobians (e.g. v13's L-BFGS-B
  with `BVFluxCurveInferenceRequest`)** →
  `FluxCurve.solve_bv_curve_points_with_warmstart` (called for you
  via `BVFluxCurveInferenceRequest`); it handles the per-V adjoint
  annotation.
- **Single voltage with manual control (e.g. a hand-rolled inverse
  loop, like v18's `solve_warm_annotated`)** → use `build_context`
  + `build_forms` + `set_initial_conditions` directly through the
  dispatcher; manage your own SS time-stepping and adjoint tape.

## Common gotchas

- **Don't add the inline `add_boltzmann(ctx)` helper to user scripts
  *and* set `boltzmann_counterions` in the config.**  That double-counts
  the counterion contribution.  Pick one.  The unified API uses the
  config; legacy v18/v19/v23/v24 scripts use the inline helper.
- **`set_initial_conditions(ctx, sp, blob=True)` is silently ignored
  in log-c mode** because there's no blob IC for `u_i = ln(c_i)`.
- **`validate_solution_state` needs `is_logc=` when called on a log-c
  context.**  The orchestrators in `Forward.bv_solver` and
  `FluxCurve.bv_point_solve` pass it for you; if you call it
  directly, set `is_logc=ctx.get("logc_transform", False)`.
- **`H2O2_SEED_NONDIM = 1e-4` is not a physics tweak**, it's the
  finite seed for `ln(c_H2O2)` at the bulk Dirichlet BC.  The actual
  H₂O₂ behavior is governed by the BV anodic term and species
  transport, not by this seed.

## Pointers

- Apr 27 writeup (formulation rationale and validation):
  `writeups/WeekOfApr27/PNP Inverse Solver Revised.tex`
- Continuation strategy handoff (background on why C+D):
  `docs/CONTINUATION_STRATEGY_HANDOFF.md`
- Production inverse script using the new stack via the standalone
  build path:  `scripts/studies/v18_logc_lsq_inverse.py`
- Standalone-vs-main parity test:
  `scripts/studies/v25_main_pipeline_vs_standalone_logc.py`
- 3sp+Boltzmann vs 4sp reduction validation:
  `scripts/studies/v24_3sp_logc_vs_4sp_validation.py`
