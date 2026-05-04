"""Positive tests for the Bikerman steric chemical potential post-sign-fix.

The pre-fix `mu_steric = +ln(1-Phi)` at `Forward/bv_solver/forms_logc.py:266`
inverted the variational derivative of the lattice-gas entropy.
Behavioural consequence: at psi_D > psi_crit ~ ln(1/(4*a*c_bulk)) the dynamic
counterion (ClO4-) had no equilibrium SS and the warm-walk diverged.

Post-fix `mu_steric = -ln(1-Phi)` (Borukhov-Andelman-Orland 1997 eq (3);
Bazant-Kilic-Storey-Ajdari 2009 eq (20)) restores conventional Bikerman
saturation: c_ClO4 at the electrode surface is bounded by `1/a` and grows
toward the cap as psi_D increases.

These tests would FAIL under the old sign (Newton would not find a SS).
They confirm the fix delivers physical saturation behaviour.

Both tests use the production 4sp + debye_boltzmann + Stern stack, with
identical parameter wiring to scripts/studies/peroxide_window_4sp_extended.py
(`4sp_stern_0p10_clip50` pass).  No deviations from production: same
K0/ALPHA, same clip=50, same Stern C_S=0.10 F/m^2.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from conftest import skip_without_firedrake


# Mesh resolution: production sweep uses Ny=200; we use Ny=100 to keep
# the test budget reasonable (per test_solver_equivalence.py:172-180).
# Saturation behaviour is set at the surface BC and is robust to
# discretization in y above ~50 graded cells.
_MESH_NY = 100

# Production sweep parameters mirrored from
# scripts/studies/peroxide_window_4sp_extended.py:51-56.
_EXPONENT_CLIP = 50.0
_U_CLAMP = 100.0
_N_SUBSTEPS_WARM = 8
_BISECT_DEPTH_WARM = 5
_STERN_CS_F_M2 = 0.10  # matches the `4sp_stern_0p10_clip50` production pass


def _run_v_grid(v_rhe_grid, *, formulation: str = "logc"):
    """Run the production 4sp + debye_boltzmann + Stern stack on a
    V_RHE grid via the cold-with-warm-fallback orchestrator.  Mirrors
    scripts/studies/peroxide_window_4sp_extended.py:_run_one_pass for
    the `4sp_stern_0p10_clip50` pass.

    Parameters
    ----------
    v_rhe_grid:
        Voltage grid in V vs RHE.
    formulation:
        ``"logc"`` (default, production) or ``"logc_muh"`` (experimental
        Phase 2 hybrid: H+ stored as electrochemical potential mu_H).

    Returns (per_voltage_fields, per_voltage_converged) where:
      - per_voltage_fields[i] is the surface_field_means(ctx) dict at
        each voltage (or {} if the per_point_callback didn't fire there)
      - per_voltage_converged[i] is True iff that V reached z=1
    """
    from scripts._bv_common import (
        setup_firedrake_env,
        V_T,
        K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2,
        FOUR_SPECIES_LOGC_DYNAMIC,
        SNES_OPTS_CHARGED,
        make_bv_solver_params,
    )
    setup_firedrake_env()

    import firedrake.adjoint as adj
    from Forward.bv_solver import (
        make_graded_rectangle_mesh,
        solve_grid_per_voltage_cold_with_warm_fallback,
    )
    from Forward.bv_solver.diagnostics import surface_field_means

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=int(_MESH_NY), beta=3.0)

    snes_opts = {**SNES_OPTS_CHARGED}
    snes_opts.update({
        "snes_max_it":               400,
        "snes_atol":                 1e-7,
        "snes_rtol":                 1e-10,
        "snes_stol":                 1e-12,
        "snes_linesearch_type":      "l2",
        "snes_linesearch_maxlambda": 0.3,
        "snes_divergence_tolerance": 1e10,
    })

    sp = make_bv_solver_params(
        eta_hat=0.0, dt=0.25, t_end=80.0,
        species=FOUR_SPECIES_LOGC_DYNAMIC,
        snes_opts=snes_opts,
        formulation=formulation, log_rate=True,
        u_clamp=_U_CLAMP,
        boltzmann_counterions=None,
        stern_capacitance_f_m2=_STERN_CS_F_M2,
        k0_hat_r1=K0_HAT_R1, k0_hat_r2=K0_HAT_R2,
        alpha_r1=ALPHA_R1,   alpha_r2=ALPHA_R2,
        E_eq_r1=0.68,        E_eq_r2=1.78,
        initializer="debye_boltzmann",
    )
    new_opts = dict(sp.solver_options)
    new_bv = dict(new_opts["bv_convergence"])
    new_bv["exponent_clip"] = float(_EXPONENT_CLIP)
    new_opts["bv_convergence"] = new_bv
    sp = sp.with_solver_options(new_opts)

    NV = len(v_rhe_grid)
    per_voltage_fields: list[dict] = [{} for _ in range(NV)]

    def _grab(orig_idx, _phi_eta, ctx):
        per_voltage_fields[orig_idx] = dict(surface_field_means(ctx))

    phi_hat_grid = np.array(v_rhe_grid, dtype=float) / V_T
    with adj.stop_annotating():
        result = solve_grid_per_voltage_cold_with_warm_fallback(
            sp,
            phi_applied_values=phi_hat_grid,
            mesh=mesh,
            max_z_steps=20,
            n_substeps_warm=_N_SUBSTEPS_WARM,
            bisect_depth_warm=_BISECT_DEPTH_WARM,
            per_point_callback=_grab,
        )

    converged = [bool(result.points[i].converged) for i in range(NV)]
    return per_voltage_fields, converged


# Steric saturation cap: 1 / a_3 with a_3 = A_DEFAULT = 0.01
# (scripts/_bv_common.py:119, FOUR_SPECIES_LOGC_DYNAMIC).
_A_CLO4 = 0.01
_STERIC_CAP = 1.0 / _A_CLO4  # = 100 nondim


@skip_without_firedrake
@pytest.mark.slow
def test_4sp_clo4_saturates_at_v0p3():
    """At V_RHE=+0.3 V the production 4sp+debye_boltzmann+Stern stack
    must converge with surface c_ClO4 bounded by 1/a and visibly
    approaching saturation.

    Pre-fix this voltage was above psi_crit ~ 4.83 nondim (~+0.124 V vs
    RHE) and Newton diverged because no equilibrium SS exists under the
    inverted sign.  Post-fix the conventional Bikerman saturation holds.

    Uses a small V grid [-0.10, 0.00, +0.10, +0.30] so the orchestrator
    has cold-success anchors for warm-walk fallback if cold at +0.3
    doesn't make it on its own.
    """
    v_grid = [-0.10, 0.00, 0.10, 0.30]
    fields_per_v, converged_per_v = _run_v_grid(v_grid)

    target_idx = v_grid.index(0.30)
    assert converged_per_v[target_idx], (
        f"V_RHE=+0.3 V must converge post-fix.  "
        f"Convergence map (V_RHE -> converged): "
        f"{list(zip(v_grid, converged_per_v))}.  "
        f"Pre-fix this voltage diverged at z=1 because no equilibrium SS "
        f"exists under the inverted steric sign."
    )

    fields = fields_per_v[target_idx]
    assert "c3_surface_mean" in fields, (
        f"surface_field_means missing c3_surface_mean at V=+0.3; "
        f"keys={list(fields)}"
    )

    c3_surf = fields["c3_surface_mean"]
    cap_tol = 1.0  # 1% slack for FE quadrature + packing_floor
    assert c3_surf <= _STERIC_CAP + cap_tol, (
        f"c_ClO4 surface mean must respect Bikerman saturation cap "
        f"1/a={_STERIC_CAP}; got {c3_surf}"
    )
    # Saturation must be visibly approached (else the test is vacuously
    # true — this asserts the steric term is actually engaging).
    assert c3_surf >= 0.5 * _STERIC_CAP, (
        f"c_ClO4 surface mean should approach saturation "
        f"(>= 50% of cap = {0.5 * _STERIC_CAP}); got {c3_surf}.  "
        f"If small, steric is not engaging or psi_D is below psi_crit."
    )


# Note: a separate V=+0.5 V test was prototyped but deemed redundant.
# Sweep S1 (scripts/studies/peroxide_window_4sp_extended.py debye_boltzmann)
# tests V=+0.5 V on the 15-voltage production grid where warm-walk
# anchors are dense.  A unit-test version with a 5-V grid forced a
# single 7.78-nondim eta warm-walk leg from V=+0.3 -> V=+0.5, which is
# slow (>15 min) and brittle to test budget without giving information
# beyond what the sweep produces.  Saturation at the higher voltage is
# validated by the sweep artifacts in
# StudyResults/peroxide_window_4sp_extended_debye_boltzmann/.


@skip_without_firedrake
@pytest.mark.slow
def test_4sp_clo4_saturates_at_v0p3_muh():
    """Same physics as test_4sp_clo4_saturates_at_v0p3 but exercising
    the experimental ``formulation="logc_muh"`` (Phase 2 hybrid: H+
    stored as electrochemical potential mu_H = u_H + em*z_H*phi).

    The same one-character sign fix has been applied to
    ``forms_logc_muh.py:321``; this test confirms the muh path also
    saturates correctly at psi_D > psi_crit.

    The muh path has historically only been tested on 3sp+Boltzmann
    (no steric).  This test is the first to exercise muh + 4sp dynamic
    + Bikerman steric on the residual.

    Same V grid as the logc P1 so the two tests are directly
    comparable: muh and logc are mathematically equivalent residuals
    (differ only in primary-variable representation), so the surface
    c_ClO4 mean should match between formulations to within FE
    discretization error.
    """
    v_grid = [-0.10, 0.00, 0.10, 0.30]
    fields_per_v, converged_per_v = _run_v_grid(v_grid, formulation="logc_muh")

    target_idx = v_grid.index(0.30)
    assert converged_per_v[target_idx], (
        f"V_RHE=+0.3 V must converge post-fix on the muh path.  "
        f"Convergence map (V_RHE -> converged): "
        f"{list(zip(v_grid, converged_per_v))}.  "
        f"Pre-fix the muh path had the same inverted steric sign at "
        f"forms_logc_muh.py:321."
    )

    fields = fields_per_v[target_idx]
    assert "c3_surface_mean" in fields, (
        f"surface_field_means missing c3_surface_mean at V=+0.3 (muh); "
        f"keys={list(fields)}"
    )

    c3_surf = fields["c3_surface_mean"]
    cap_tol = 1.0
    assert c3_surf <= _STERIC_CAP + cap_tol, (
        f"c_ClO4 surface mean (muh) must respect Bikerman saturation cap "
        f"1/a={_STERIC_CAP}; got {c3_surf}"
    )
    assert c3_surf >= 0.5 * _STERIC_CAP, (
        f"c_ClO4 surface mean (muh) should approach saturation "
        f"(>= 50% of cap = {0.5 * _STERIC_CAP}); got {c3_surf}."
    )


def _run_v_grid_3sp_bikerman(v_rhe_grid):
    """Run the production 3sp + bikerman-counterion + muh stack on a
    V_RHE grid via the cold-with-warm-fallback orchestrator.

    The user's target final config: 3 dynamic species (O2, H2O2, H+),
    analytic ClO4- counterion with steric_mode='bikerman' (saturating
    closure), Stern layer, formulation='logc_muh'.  Both the residual
    side (build_steric_boltzmann_expressions) and the IC side
    (gamma + composite-psi) must be Bikerman-consistent for V > +0.5
    to converge cold.
    """
    from scripts._bv_common import (
        setup_firedrake_env,
        V_T,
        K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2,
        THREE_SPECIES_LOGC_BOLTZMANN,
        DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC,
        SNES_OPTS_CHARGED,
        make_bv_solver_params,
    )
    setup_firedrake_env()

    import firedrake.adjoint as adj
    from Forward.bv_solver import (
        make_graded_rectangle_mesh,
        solve_grid_per_voltage_cold_with_warm_fallback,
    )
    from Forward.bv_solver.diagnostics import collect_diagnostics

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=int(_MESH_NY), beta=3.0)

    snes_opts = {**SNES_OPTS_CHARGED}
    snes_opts.update({
        "snes_max_it":               400,
        "snes_atol":                 1e-7,
        "snes_rtol":                 1e-10,
        "snes_stol":                 1e-12,
        "snes_linesearch_type":      "l2",
        "snes_linesearch_maxlambda": 0.3,
        "snes_divergence_tolerance": 1e10,
    })

    sp = make_bv_solver_params(
        eta_hat=0.0, dt=0.25, t_end=80.0,
        species=THREE_SPECIES_LOGC_BOLTZMANN,
        snes_opts=snes_opts,
        formulation="logc_muh", log_rate=True,
        u_clamp=_U_CLAMP,
        boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC],
        stern_capacitance_f_m2=_STERN_CS_F_M2,
        k0_hat_r1=K0_HAT_R1, k0_hat_r2=K0_HAT_R2,
        alpha_r1=ALPHA_R1,   alpha_r2=ALPHA_R2,
        E_eq_r1=0.68,        E_eq_r2=1.78,
        initializer="debye_boltzmann",
    )
    new_opts = dict(sp.solver_options)
    new_bv = dict(new_opts["bv_convergence"])
    new_bv["exponent_clip"] = float(_EXPONENT_CLIP)
    new_opts["bv_convergence"] = new_bv
    sp = sp.with_solver_options(new_opts)

    NV = len(v_rhe_grid)
    per_voltage_diags: list[dict] = [{} for _ in range(NV)]
    params = sp.solver_options if hasattr(sp, "solver_options") else sp[10]

    def _grab(orig_idx, _phi_eta, ctx):
        per_voltage_diags[orig_idx] = dict(
            collect_diagnostics(
                ctx, phase="callback", params=params, picard_iters=0,
            )
        )

    phi_hat_grid = np.array(v_rhe_grid, dtype=float) / V_T
    with adj.stop_annotating():
        result = solve_grid_per_voltage_cold_with_warm_fallback(
            sp,
            phi_applied_values=phi_hat_grid,
            mesh=mesh,
            max_z_steps=20,
            n_substeps_warm=_N_SUBSTEPS_WARM,
            bisect_depth_warm=_BISECT_DEPTH_WARM,
            per_point_callback=_grab,
        )

    converged = [bool(result.points[i].converged) for i in range(NV)]
    return per_voltage_diags, converged


@skip_without_firedrake
@pytest.mark.slow
def test_3sp_bikerman_v0p55_muh():
    """User's target stack: 3sp + analytic ClO4- counterion with
    ``steric_mode='bikerman'`` + muh formulation.

    Pass criteria: V_RHE=+0.55 V converges cold (or via warm-walk) with
    the analytic counterion's surface concentration saturating in
    [0.5/a_b, 1/a_b + cap_tol].  The residual side already produces
    saturation via ``build_steric_boltzmann_expressions``; this test
    confirms that the IC side (composite psi + multispecies gamma)
    seeds Newton with a state that doesn't violate the Bikerman cap
    on the analytic counterion or the dynamic species' total packing.
    """
    v_grid = [-0.10, 0.00, 0.10, 0.30, 0.55]
    diags_per_v, converged_per_v = _run_v_grid_3sp_bikerman(v_grid)

    target_idx = v_grid.index(0.55)
    assert converged_per_v[target_idx], (
        f"V_RHE=+0.55 V must converge for the user's target stack "
        f"(3sp+bikerman counterion+muh).  Convergence map: "
        f"{list(zip(v_grid, converged_per_v))}."
    )

    diag = diags_per_v[target_idx]
    # The analytic counterion's surface c_steric is reported by
    # collect_diagnostics as ``c_counterion0_surface_mean``.
    key = "c_counterion0_surface_mean"
    assert key in diag, (
        f"diagnostics missing {key!r} at V=+0.55; keys={sorted(diag)}"
    )
    c_steric_surf = float(diag[key])

    cap_tol = 1.0
    assert c_steric_surf <= _STERIC_CAP + cap_tol, (
        f"c_steric surface must respect Bikerman cap 1/a_b={_STERIC_CAP}; "
        f"got {c_steric_surf}"
    )
    assert c_steric_surf >= 0.5 * _STERIC_CAP, (
        f"c_steric surface should approach saturation "
        f"(>= 50% of cap = {0.5 * _STERIC_CAP}); got {c_steric_surf}.  "
        f"At V=+0.55 the closure should saturate; if low, the IC seed "
        f"is collapsing the closure prematurely."
    )
