"""Unit tests for the debye_boltzmann analytical initializer.

Two tests, both ``@pytest.mark.slow`` (Firedrake required):

* ``test_ic_implements_analytical_formula`` -- at V_RHE = 0.5 V (psi_D ~ 19)
  and V_RHE = +1.0 V (psi_D ~ 39), the IC produces a Boltzmann-depleted
  surface H+ profile and a finite-everywhere phi field.  This catches
  the atanh-saturation edge case noted in the plan: at psi_D > ~30 a
  naive ``4*atanh(tanh(psi_D/4)*exp(-y/lambda_D))`` evaluates ``atanh(1.0)``
  and produces ``+inf``.  The IC uses a clamped log-form expression
  instead and must produce finite values everywhere.

* ``test_ic_steric_warning`` -- at V_RHE = +0.5 V, surface c_ClO4 from
  the analytical Gouy-Chapman profile is ~C_CLO4_HAT * exp(19) ~ 1e7,
  far above the Bikerman steric cap (~100 nondim).  The
  ``check_steric_saturation`` helper must emit a ``UserWarning`` flagging
  the converged state as non-physical despite Newton convergence.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from conftest import skip_without_firedrake


def _build_ctx_for_v(v_rhe: float, *, mesh_n: int = 8):
    """Build a fresh ctx with the production logc+Boltzmann config and run
    the debye_boltzmann initializer on a UnitSquareMesh(mesh_n).
    """
    from scripts._bv_common import (
        setup_firedrake_env,
        V_T, K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2,
        THREE_SPECIES_LOGC_BOLTZMANN, DEFAULT_CLO4_BOLTZMANN_COUNTERION,
        SNES_OPTS_CHARGED,
        make_bv_solver_params,
    )
    setup_firedrake_env()

    import firedrake as fd

    from Forward.bv_solver import build_context, build_forms, set_initial_conditions

    mesh = fd.UnitSquareMesh(mesh_n, mesh_n)

    sp = make_bv_solver_params(
        eta_hat=v_rhe / V_T, dt=0.25, t_end=80.0,
        species=THREE_SPECIES_LOGC_BOLTZMANN,
        snes_opts=SNES_OPTS_CHARGED,
        formulation="logc", log_rate=True,
        boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION],
        k0_hat_r1=K0_HAT_R1, k0_hat_r2=K0_HAT_R2,
        alpha_r1=ALPHA_R1, alpha_r2=ALPHA_R2,
        E_eq_r1=0.68, E_eq_r2=1.78,
        initializer="debye_boltzmann",
    )
    ctx = build_context(sp, mesh=mesh)
    ctx = build_forms(ctx, sp)
    set_initial_conditions(ctx, sp)
    return ctx, sp


@skip_without_firedrake
@pytest.mark.slow
class TestDebyeBoltzmannInitializer:
    """Analytical-IC behavior on a small UnitSquareMesh."""

    def test_ic_implements_analytical_formula_v05(self):
        """At V_RHE=+0.5 V the IC produces depleted H+ and a Gouy-Chapman phi."""
        ctx, _ = _build_ctx_for_v(0.5)
        U = ctx["U"]
        n = ctx["n_species"]

        assert ctx.get("initializer_fallback") is False, (
            f"Picard should converge at V=+0.5 V; got fallback "
            f"reason={ctx.get('initializer_fallback_reason')!r}"
        )
        assert ctx.get("initializer_picard_iters", 0) <= 50

        u_h = U.dat[2].data_ro
        phi = U.dat[n].data_ro

        assert np.all(np.isfinite(u_h)), "u_H has non-finite entries"
        assert np.all(np.isfinite(phi)), "phi has non-finite entries"

        # u_H should be depleted at the electrode (y=0) by ~psi_D ~ 19 nondim.
        from scripts._bv_common import C_HP_HAT
        c_h_bulk = C_HP_HAT
        u_h_bulk = float(np.log(c_h_bulk))
        assert u_h.min() < u_h_bulk - 15.0, (
            f"H+ should be depleted by ~19 nondim at electrode; "
            f"got min(u_H)={u_h.min():.4f}, bulk={u_h_bulk:.4f}"
        )
        # u_H at bulk (y=1) should match the Dirichlet bulk value.
        assert abs(u_h.max() - u_h_bulk) < 1e-6, (
            f"u_H bulk should be ln(C_HP_HAT)={u_h_bulk:.4f}; got max={u_h.max():.4f}"
        )

        # phi should reach phi_applied at the electrode and ~0 at bulk.
        from scripts._bv_common import V_T
        phi_applied_nondim = 0.5 / V_T
        assert abs(phi.max() - phi_applied_nondim) < 1.0, (
            f"phi at electrode should be ~{phi_applied_nondim:.2f}; "
            f"got max={phi.max():.4f}"
        )
        assert abs(phi.min()) < 1.0, (
            f"phi at bulk should be ~0; got min={phi.min():.4f}"
        )

    def test_ic_completes_at_v10_no_inf_nan(self):
        """V_RHE=+1.0 V (psi_D ~ 39): atanh-saturation edge case.

        At this voltage tanh(psi_D/4) is bit-equal to 1.0 in IEEE doubles.
        A naive 4*atanh(tanh(psi_D/4)*exp(-y/lambda_D)) at y=0 evaluates
        atanh(1.0) = +inf and produces NaN-poisoned fields.  The IC's
        clamped log-form expression must produce finite values everywhere.
        """
        ctx, _ = _build_ctx_for_v(1.0)
        U = ctx["U"]
        n = ctx["n_species"]

        u_h = U.dat[2].data_ro
        phi = U.dat[n].data_ro
        for i in range(n + 1):
            data = U.dat[i].data_ro
            assert np.all(np.isfinite(data)), (
                f"sub({i}) has non-finite entries at V=+1.0 V "
                f"(min={data.min()}, max={data.max()})"
            )

        # u_H at electrode should be deeply depleted (~ -40 nondim).
        from scripts._bv_common import C_HP_HAT
        c_h_bulk = C_HP_HAT
        u_h_bulk = float(np.log(c_h_bulk))
        assert u_h.min() < u_h_bulk - 30.0, (
            f"H+ should be deeply depleted at V=+1.0 V; got "
            f"min(u_H)={u_h.min():.4f}"
        )

    def test_ic_steric_warning_at_v05(self):
        """Surface c_ClO4 from analytical IC exceeds steric cap at V=+0.5 V.

        c_ClO4_surf ~ C_CLO4_HAT * exp(19) ~ 1e7 >> 100 (Bikerman cap).  The
        ``check_steric_saturation`` helper must emit a UserWarning.
        """
        from Forward.bv_solver.diagnostics import check_steric_saturation

        ctx, sp = _build_ctx_for_v(0.5)
        params = sp.solver_options if hasattr(sp, "solver_options") else sp[10]

        with pytest.warns(UserWarning, match="steric saturation exceeded"):
            result = check_steric_saturation(ctx, params=params, cap=100.0)

        assert result["within_steric"] is False
        assert len(result["counterions"]) == 1
        assert result["counterions"][0]["c_surf"] > 1e5
