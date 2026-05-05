"""Snapshot regression: Stern wiring must not perturb no-Stern CD/PC.

The strongest gate for the Stern factory wiring landing
(see ``docs/stern_layer_physics_and_next_steps.md``).  Pins the
production no-Stern stack at ``V_RHE = +0.66 V`` to the existing
baseline in
``StudyResults/peroxide_window_pb_init_test/iv_curve.json``
(the ``debye_boltzmann`` initializer row).  Two scenarios:

1. ``stern_capacitance_f_m2 = None``  — the default, omits the cfg key
   entirely.  Must reproduce the baseline.
2. ``stern_capacitance_f_m2 = 0.0``   — writes ``0.0`` into ``bv_bc``
   but ``forms_logc.py:229`` requires ``> 0`` to activate Stern, so
   runtime behaviour is identical to ``None``.  Must reproduce the
   baseline.

Tolerance ``rel_tol = 1e-6`` is the Newton convergence floor; any
deviation larger than that means the wiring landing has perturbed the
solver path.

Marked ``slow`` — requires Firedrake and runs one cold solve at
production resolution (Ny=200), which takes ~60-120 seconds.
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


# Baseline at V_RHE = +0.66 V, no Stern, Ny=200, exponent_clip=100,
# log-rate, 3sp+Boltzmann.  Originally pinned from
# StudyResults/peroxide_window_pb_init_test/iv_curve.json (pre-steric-
# sign-fix); refreshed 2026-05-04 after the steric mu sign was corrected
# at forms_logc.py:266 (mu_steric = -ln(1-Phi), per Borukhov-Andelman-
# Orland 1997 eq (3) / Bazant-Kilic-Storey-Ajdari 2009 eq (20)).
# The 3sp+Boltzmann path uses ``a_vals_hat = [A_DEFAULT]*3 = [0.01]*3``
# (production preset; see scripts/_bv_common.py:THREE_SPECIES_LOGC_BOLTZMANN),
# so its residual *does* include the steric activity term and was
# affected by the sign fix.  Drift from the prior baseline: ~0.36% for
# both CD and PC, well above ``rel_tol = 1e-6`` but small in absolute
# terms.  See docs/4sp_drop_boltzmann_investigation.md (Resolution
# section) for context.
BASELINE_CD_MA_CM2 = 1.2968453558282709e-08
BASELINE_PC_MA_CM2 = 1.2969358369725412e-08
BASELINE_V_RHE = 0.66
REL_TOL = 1e-6


def _solve_no_stern_at_066(*, stern_capacitance_f_m2):
    """Production-stack cold solve at V_RHE = +0.66 V.  Returns (cd, pc)."""
    from scripts._bv_common import (
        setup_firedrake_env,
        V_T, I_SCALE,
        K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2,
        THREE_SPECIES_LOGC_BOLTZMANN,
        DEFAULT_CLO4_BOLTZMANN_COUNTERION,
        SNES_OPTS_CHARGED,
        make_bv_solver_params,
    )
    setup_firedrake_env()

    import firedrake as fd
    import firedrake.adjoint as adj
    from Forward.bv_solver import (
        make_graded_rectangle_mesh,
        solve_grid_per_voltage_cold_with_warm_fallback,
    )
    from Forward.bv_solver.observables import _build_bv_observable_form

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)

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
        formulation="logc", log_rate=True,
        boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION],
        stern_capacitance_f_m2=stern_capacitance_f_m2,
        k0_hat_r1=K0_HAT_R1, k0_hat_r2=K0_HAT_R2,
        alpha_r1=ALPHA_R1, alpha_r2=ALPHA_R2,
        E_eq_r1=0.68, E_eq_r2=1.78,
        initializer="debye_boltzmann",
    )
    new_opts = dict(sp.solver_options)
    new_bv = dict(new_opts["bv_convergence"])
    new_bv["exponent_clip"] = 100.0
    new_opts["bv_convergence"] = new_bv
    sp = sp.with_solver_options(new_opts)

    cd = np.full(1, np.nan)
    pc = np.full(1, np.nan)

    def _grab(orig_idx, _phi_eta, ctx):
        f_cd = _build_bv_observable_form(
            ctx, mode="current_density", reaction_index=None, scale=-I_SCALE)
        f_pc = _build_bv_observable_form(
            ctx, mode="peroxide_current", reaction_index=None, scale=-I_SCALE)
        cd[orig_idx] = float(fd.assemble(f_cd))
        pc[orig_idx] = float(fd.assemble(f_pc))

    phi_hat_grid = np.array([BASELINE_V_RHE]) / V_T
    with adj.stop_annotating():
        result = solve_grid_per_voltage_cold_with_warm_fallback(
            sp,
            phi_applied_values=phi_hat_grid,
            mesh=mesh,
            max_z_steps=20,
            n_substeps_warm=8,
            bisect_depth_warm=5,
            per_point_callback=_grab,
        )

    assert result.points[0].converged, (
        f"No-Stern V=+0.66 V did not converge with "
        f"stern_capacitance_f_m2={stern_capacitance_f_m2!r}; "
        f"method={result.points[0].method!r}"
    )
    return float(cd[0]), float(pc[0])


@skip_without_firedrake
@pytest.mark.slow
@pytest.mark.parametrize("stern_capacitance_f_m2", [None, 0.0],
                         ids=["stern_None", "stern_0p0"])
def test_no_stern_cd_pc_matches_baseline(stern_capacitance_f_m2):
    """The production no-Stern stack at V=+0.66 V must reproduce the
    pre-Stern baseline within Newton tolerance, regardless of whether
    ``stern_capacitance_f_m2`` is omitted (``None``) or written as
    ``0.0`` (debuggable but runtime-inactive).
    """
    cd, pc = _solve_no_stern_at_066(
        stern_capacitance_f_m2=stern_capacitance_f_m2,
    )
    assert cd == pytest.approx(BASELINE_CD_MA_CM2, rel=REL_TOL), (
        f"CD drifted from baseline: got {cd!r}, "
        f"expected {BASELINE_CD_MA_CM2!r} (rel_tol={REL_TOL})"
    )
    assert pc == pytest.approx(BASELINE_PC_MA_CM2, rel=REL_TOL), (
        f"PC drifted from baseline: got {pc!r}, "
        f"expected {BASELINE_PC_MA_CM2!r} (rel_tol={REL_TOL})"
    )
