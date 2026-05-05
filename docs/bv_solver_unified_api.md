# BV-PNP Forward Solver: Unified API

**Last updated:** 2026-05-05

This is the canonical reference for calling the BV-PNP forward solver
on the May 2026 production stack. The current production target is:

- **3 dynamic species** (O₂, H₂O₂, H⁺)
- **analytic Bikerman counterion** for ClO₄⁻
  (`steric_mode='bikerman'`)
- **proton electrochemical-potential primary variable**
  (`formulation='logc_muh'`, `mu_H = u_H + em·z_H·phi`)
- **log-rate Butler–Volmer**
- **finite Stern compact layer** (`stern_capacitance_f_m2 ≈ 0.10`)
- **Bikerman-consistent IC** (`initializer='debye_boltzmann'`,
  composite-ψ + multispecies-γ)

Selected via configuration flags through a single dispatcher; no
inline residual helpers required. The legacy 4-species concentration
backend (`forms.py`) was removed in the May 2026 cleanup; both live
backends are log-concentration based. See
`docs/4sp_bikerman_ic_option_2b_results.md` for the production-target
reference sweep (15/15 V_RHE [−0.5, +1.0]).

## TL;DR

```python
from scripts._bv_common import (
    make_bv_solver_params,
    THREE_SPECIES_LOGC_BOLTZMANN,
    DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC,
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

# Build solver params for the production stack.
sp = make_bv_solver_params(
    eta_hat=0.0, dt=0.25, t_end=80.0,
    species=THREE_SPECIES_LOGC_BOLTZMANN,
    snes_opts=SNES_OPTS_CHARGED,
    formulation="logc_muh",
    log_rate=True,
    boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC],
    stern_capacitance_f_m2=0.10,
    initializer="debye_boltzmann",
    k0_hat_r1=K0_HAT_R1, k0_hat_r2=K0_HAT_R2,
    alpha_r1=ALPHA_R1,   alpha_r2=ALPHA_R2,
    E_eq_r1=0.68,        E_eq_r2=1.78,
)

V_GRID = np.array([-0.50, -0.30, -0.10, 0.00, 0.10, 0.30, 0.50,
                   0.55, 0.60, 0.65, 0.70, 0.75, 1.00])
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

# res.points[i].converged / .achieved_z_factor / .method per voltage.
# cd, pc populated in place by the callback for every converged V.
```

## Architecture

```
Forward/bv_solver/
    __init__.py                # public re-exports + dispatcher
    dispatch.py                # build_context / build_forms /
                               #   set_initial_conditions dispatch on
                               #   formulation + initializer
    forms_logc.py              # log-c backend (production)
    forms_logc_muh.py          # log-c muh-variant backend (production
                               #   when formulation="logc_muh")
    boltzmann.py               # ideal + Bikerman analytic counterions:
                               #   add_boltzmann_counterion_residual,
                               #   build_steric_boltzmann_expressions,
                               #   StericBoltzmannBundle
    config.py                  # parses formulation / initializer /
                               #   steric_mode / a_nondim /
                               #   stern_capacitance_f_m2
    grid_per_voltage.py        # C+D orchestrator (production)
    diagnostics.py             # per-voltage diagnostics (surface
                               #   field means, SNES, steric saturation)
    solvers.py                 # per-point helpers
    validation.py              # is_logc + muh-aware physics validation
    observables.py             # current_density / peroxide_current /
                               #   reaction-N observables
```

The legacy 4sp concentration backend (`forms.py`) and Strategy-B
orchestrator (`grid_charge_continuation.py`) were removed in the May
2026 cleanup.

## Configuration flags

All under `params['bv_convergence']` and `params['bv_bc']` in
`solver_params[10]`. Defaults are produced by
`scripts/_bv_common.py:make_bv_solver_params`.

| Flag | Location | Default | Effect |
|---|---|---|---|
| `formulation` | `bv_convergence.formulation` | `"concentration"` (factory legacy default — set to `"logc_muh"` for production) | `"logc"` → `forms_logc.py`. `"logc_muh"` → `forms_logc_muh.py` (proton stored as `mu_H = u_H + em·z_H·phi`; Newton-smooth in deep-ψ regions). |
| `initializer` | `bv_convergence.initializer` | `"linear_phi"` | `"linear_phi"` → tanh Gouy-Chapman φ + bulk-flat species. `"debye_boltzmann"` → composite-ψ (BKSA) + multispecies-γ; required for the Bikerman-saturated regime to converge cold. |
| `bv_log_rate` | `bv_convergence.bv_log_rate` | `False` (set to `True` for production) | Compute the BV rate as `r = exp(log r)` with surface concentrations entering additively inside the exponent. Eliminates the surface-clip phantom R₂ sink at high anodic η. |
| `exponent_clip` | `bv_convergence.exponent_clip` | `100.0` | Symmetric clamp on `eta_scaled = (V_RHE − E_eq)/V_T` *before* the α·n_e multiplication. **Use 100 for any PC observable.** clip=50 produces fictitious PC at V_RHE < −0.1 V (sign-flipped, 3–4 OOM off; CD is unaffected). Raised 50 → 100 on 2026-05-04. See `docs/clipping_conventions.md` for the operational rule. |
| `u_clamp` | `bv_convergence.u_clamp` | `100.0` | Symmetric clamp on `u_i = ln c_i` in the bulk forms (raised 30 → 100 alongside `exponent_clip`). |
| `boltzmann_counterions` | `bv_bc.boltzmann_counterions` | `[]` | List of analytic Boltzmann counterions in Poisson. Each entry: `{"z": int, "c_bulk_nondim": float, "phi_clamp": float, "steric_mode": "ideal"\|"bikerman", "a_nondim": float}` (`a_nondim` required iff `steric_mode == "bikerman"`). When `bikerman`, the closure is `c = c_b·exp(−z·φ)·(1−A_dyn) / (θ_b + a_b·c_b·exp(−z·φ))`, applied to *both* the Poisson source *and* the dynamic-species packing fraction (`build_steric_boltzmann_expressions`). |
| `stern_capacitance_f_m2` | `bv_bc.stern_capacitance_f_m2` | `None` | `None` → no-Stern Dirichlet BC `phi_s = phi_m`. Positive → Robin BC for the compact-layer drop; the BV overpotential becomes `eta = phi_applied − phi − E_eq` with `phi` solved at the diffuse-layer edge. Production target uses ≈ 0.10 F/m² (10 µF/cm²). |

The flags are independent. The production stack uses
`formulation='logc_muh'` + `initializer='debye_boltzmann'` +
`steric_mode='bikerman'` + `stern_capacitance_f_m2 ≈ 0.10` + log-rate
together; see `docs/4sp_bikerman_ic_option_2b_results.md` for why all
four are needed to reach V_RHE = +1.0 V.

## Pipeline entry points

| Function | Strategy | When to use |
|---|---|---|
| `Forward.bv_solver.solve_grid_per_voltage_cold_with_warm_fallback` | C+D — per-V cold + z-ramp, then warm-walk from cold anchors with substepping + bisection | Production grid sweep. Reaches V_RHE ∈ [−0.5, +1.0] V on the production stack. |
| `Forward.bv_solver.solvers.{forsolve_bv, solve_bv_with_continuation, solve_bv_with_ptc, solve_bv_with_charge_continuation}` | Single-V helpers | Lower-level — pre-built ctx, manual time-stepping, etc. |

The legacy Strategy-B orchestrator
(`solve_grid_with_charge_continuation`) was removed in May 2026
along with the concentration backend.

The inverse-problem entry points
(`FluxCurve.bv_point_solve.solve_bv_curve_points_with_warmstart`,
`BVFluxCurveInferenceRequest`, the `v13`/`v18`/`v23` study scripts)
are **legacy / non-operational** while the inverse pipeline is
paused; they remain in the tree as historical reference but are not
maintained against the current production stack.

## Species presets

Defined in `scripts/_bv_common.py`:

| Preset | Species | Use with |
|---|---|---|
| `THREE_SPECIES_LOGC_BOLTZMANN` | O₂, H₂O₂, H⁺ (dynamic) + ClO₄⁻ analytic | `formulation="logc"` or `"logc_muh"`, `boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC]`. **Production stack.** |
| `FOUR_SPECIES_LOGC_DYNAMIC` | O₂, H₂O₂, H⁺, ClO₄⁻ (all dynamic) | `formulation="logc"` or `"logc_muh"`, `boltzmann_counterions=None`. Equivalence reference for the analytic-vs-dynamic ClO₄⁻ check; ceiling V_RHE ≤ +0.5 V even with Stern. |
| `TWO_SPECIES_NEUTRAL` | O₂, H₂O₂ only, no charges | Neutral-system tests; not used in production. |

H₂O₂ in the 3sp preset is initialized at a small positive seed
`H2O2_SEED_NONDIM = 1e-4` so `ln(c0_H2O2)` is finite in the log-c
primary variable.

## Counterion presets

```python
from scripts._bv_common import (
    DEFAULT_CLO4_BOLTZMANN_COUNTERION,           # legacy: steric_mode='ideal'
    DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC,    # production: 'bikerman'
)
# DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC =
#   {"z": -1, "c_bulk_nondim": C_CLO4_HAT, "phi_clamp": 50.0,
#    "steric_mode": "bikerman", "a_nondim": A_DEFAULT}
```

The ideal preset is retained for cross-validation against legacy
runs; new work should use the steric variant.

## Common patterns

### Forward target curve at TRUE parameters

Use the TL;DR snippet at the top of this doc.

### Single-V cold solve with adjoint annotation (for future inverse)

```python
import firedrake.adjoint as adj
from Forward.bv_solver import build_context, build_forms, set_initial_conditions

sp = make_bv_solver_params(
    ...,
    formulation="logc_muh",
    log_rate=True,
    boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC],
    stern_capacitance_f_m2=0.10,
    initializer="debye_boltzmann",
)

with adj.stop_annotating():
    ctx = build_context(sp, mesh=mesh)
    ctx = build_forms(ctx, sp)
    set_initial_conditions(ctx, sp)
    # cold-ramp z (and ctx["boltzmann_z_scale"]) from 0 to 1 unannotated...

# ... then re-solve a few SS steps with annotation on for the adjoint tape.
```

The pyadjoint controls are exposed as `ctx['bv_k0_funcs']` (one per
reaction) and `ctx['bv_alpha_funcs']` (one per reaction). Inverse
work is paused; this pattern is documented for future use.

## Common gotchas

- **Don't add an inline `add_boltzmann(ctx)` helper to user scripts
  *and* set `boltzmann_counterions` in the config.** That double-counts
  the counterion contribution. The unified API uses the config; only
  legacy v18/v19/v23/v24 scripts use the inline helper.
- **`set_initial_conditions(ctx, sp, blob=True)` is silently ignored
  in log-c mode** because there's no blob IC for `u_i = ln(c_i)`.
- **The `debye_boltzmann` IC requires either a `synthesised_4sp` ClO₄⁻
  counterion *or* a `boltzmann_counterions=[…]` entry with
  `steric_mode='bikerman'`.** With `steric_mode='ideal'` the legacy
  tanh-Gouy-Chapman seed is used (byte-identical to pre-2b on that
  path).
- **`validate_solution_state` needs `is_logc=` when called on a
  log-c context.** Orchestrators pass it for you. On the muh
  formulation, also pass `mu_species=ctx.get('mu_species')` and
  `em=ctx['nondim'].get('electromigration_prefactor', 1.0)` so
  per-dof `log(c_i)` is recovered from `mu_i − em·z_i·phi` before
  the physics checks run.
- **The IC and the residual must agree about steric saturation.** A
  bikerman IC paired with `steric_mode='ideal'` on the residual (or
  vice-versa) cold-fails on the saturated manifold. The dispatcher
  + factory enforce consistency, but ad-hoc config edits can break
  it; if Newton fails on V ≥ +0.30 V check both sides.
- **`H2O2_SEED_NONDIM = 1e-4` is not a physics tweak**, it's the
  finite seed for `ln(c_H2O2)` at the bulk Dirichlet BC.

## Validation references

- **Production-target sweep:**
  `StudyResults/peroxide_window_3sp_bikerman_muh_2b/` (15/15 over
  V_RHE [−0.5, +1.0] V). Driver:
  `scripts/studies/peroxide_window_3sp_bikerman_muh.py`.
- **MMS recovery on the production graded mesh:**
  `tests/test_mms_convergence.py:TestMMSProductionGradedMesh`,
  driven by
  `scripts/verification/mms_bv_3sp_logc_boltzmann.py:verify_on_graded_production_mesh`.
- **Steric closure unit tests:**
  `tests/test_steric_boltzmann_closure.py`,
  `tests/test_steric_boltzmann_closure_algebra.py`,
  `tests/test_steric_sign.py`.
- **IC consistency tests:**
  `tests/test_initializer_debye_boltzmann_3sp_bikerman.py`,
  `tests/test_initializer_debye_boltzmann_4sp.py`,
  `tests/test_initializer_debye_boltzmann_4sp_muh.py`.
- **4sp dynamic vs 3sp+bikerman analytic equivalence:**
  `tests/test_solver_equivalence.py` (cathodic overlap to ~10⁻⁹;
  +0.5 V edge to ~5·10⁻³).

## Pointers

- May 2026 production-target writeup:
  `writeups/ForwardSolverChangesMay26/forward_solver_changes_may2026.pdf`
- Apr 27 forward-solver rebuild narrative:
  `writeups/WeekOfApr27/PNP Inverse Solver Revised.pdf`
- Bikerman residual closure derivation:
  `docs/steric_analytic_clo4_reduction_handoff.md`
- Continuation strategy background:
  `docs/CONTINUATION_STRATEGY_HANDOFF.md`
- Clipping conventions (the three distinct clips, including the
  `exponent_clip` 50 → 100 raise):
  `docs/clipping_conventions.md`
- Investigation log behind the production target (sec 11–13):
  `docs/4sp_drop_boltzmann_investigation.md`
- Experimental-alignment gap audit (Mangan 2025 deck):
  `docs/Mangan2025_experimental_alignment.md`
