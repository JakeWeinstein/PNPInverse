"""Initializer tests for 4sp dynamic + ``formulation='logc_muh'``.

Parallel of ``test_initializer_debye_boltzmann_4sp.py`` but for the muh
formulation.  Pre-2b the muh IC body did NOT apply the multispecies
gamma correction that the logc body had (the muh IC at
``forms_logc_muh.py:909`` only seeded pure Boltzmann profiles).  After
2b lands, the muh IC also applies gamma + composite-psi for the
synthesised 4sp ClO4- counterion.

These tests assert the same gamma + saturation properties as the logc
4sp test suite, plus the muh-specific ``mu_H = u_H + em*z_H*phi``
composition (psi cancels; log_gamma propagates).

Slow tests; require Firedrake.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from conftest import skip_without_firedrake


def _build_ctx_4sp_muh(v_rhe: float, *, mesh_n: int = 8):
    """Build a fresh ctx for 4sp dynamic + muh + debye_boltzmann."""
    from scripts._bv_common import (
        setup_firedrake_env,
        V_T, K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2,
        FOUR_SPECIES_LOGC_DYNAMIC,
        SNES_OPTS_CHARGED,
        make_bv_solver_params,
    )
    setup_firedrake_env()

    import firedrake as fd
    from Forward.bv_solver import build_context, build_forms, set_initial_conditions

    mesh = fd.UnitSquareMesh(mesh_n, mesh_n)
    sp = make_bv_solver_params(
        eta_hat=v_rhe / V_T, dt=0.25, t_end=80.0,
        species=FOUR_SPECIES_LOGC_DYNAMIC,
        snes_opts=SNES_OPTS_CHARGED,
        formulation="logc_muh", log_rate=True,
        boltzmann_counterions=None,
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
class TestDebyeBoltzmann4spMuhExtension:
    """4sp dynamic + muh + bikerman-consistent IC."""

    def test_ic_fires_at_v05(self):
        ctx, _ = _build_ctx_4sp_muh(0.5)
        assert ctx.get("initializer_fallback") is False, (
            f"4sp+muh debye_boltzmann should not fall back at V=+0.5 V; "
            f"reason={ctx.get('initializer_fallback_reason')!r}"
        )
        assert ctx.get("initializer_picard_iters", 0) > 0

    def test_ic_seeds_clo4_with_gamma_correction(self):
        """At V=+0.5 V, c_ClO4 from u_3 must respect the Bikerman cap.

        Pre-2b muh: pure Boltzmann seed -- c_ClO4 ~ 4e7 at the electrode.
        Post-2b muh: gamma-corrected, c_ClO4 saturates at ~1/a = 100.
        """
        ctx, _ = _build_ctx_4sp_muh(0.5)
        U = ctx["U"]

        u3 = U.dat[3].data_ro  # ClO4- stored as u_3 = ln(c_3) (Phase 2 hybrid muh)
        assert np.all(np.isfinite(u3))
        c_clo4 = np.exp(u3)

        a_clo4 = 0.01
        steric_cap = 1.0 / a_clo4

        cap_tol = 0.5
        assert c_clo4.max() <= steric_cap + cap_tol, (
            f"c_ClO4 must respect Bikerman cap 1/a={steric_cap}; "
            f"got max={c_clo4.max():.4f}.  Without gamma correction in "
            f"muh path, would be ~4e7 at V=+0.5 V."
        )
        assert c_clo4.max() >= 0.5 * steric_cap, (
            f"c_ClO4 should approach saturation; got {c_clo4.max():.4f}"
        )

        from scripts._bv_common import C_CLO4_HAT
        c_clo4_bulk = C_CLO4_HAT
        assert abs(c_clo4.min() - c_clo4_bulk) / c_clo4_bulk < 0.05

    def test_mu_h_propagates_log_gamma(self):
        """Check log_gamma propagates into mu_H.

            mu_H_init = (ln H_outer - psi + log_gamma)
                       + em*z_H * (ln(H_outer/c_clo4_bulk) + psi)
                      = 2*ln(H_outer) - ln(c_clo4_bulk) + log_gamma
              (em*z_H = 1, psi cancels)

        At bulk (y=1) log_gamma ~ 0 so mu_H_bulk ~ ln(C_HP_HAT).
        At electrode log_gamma ~ -12.78 so mu_H_surf ~ ln(C_HP_HAT) - 12.78.
        """
        ctx, _ = _build_ctx_4sp_muh(0.5)
        U = ctx["U"]

        mu_h = U.dat[2].data_ro
        from scripts._bv_common import C_HP_HAT, C_CLO4_HAT
        c_h_bulk = C_HP_HAT
        c_clo4_bulk = C_CLO4_HAT
        baseline = 2.0 * math.log(c_h_bulk) - math.log(c_clo4_bulk)
        # Pre-2b muh (no gamma): mu_H_surf ~ baseline (no offset).
        # Post-2b muh: mu_H_surf < baseline - 8 (clear offset).
        assert mu_h.min() < baseline - 8.0, (
            f"mu_H must include log_gamma at electrode; got "
            f"min(mu_H)={mu_h.min():.4f}, baseline (no gamma)={baseline:.4f}"
        )
        # Bulk mu_H should match baseline (log_gamma -> 0 at psi -> 0).
        assert abs(mu_h.max() - baseline) < 0.1

    def test_ic_total_packing_positive_at_v0p3(self):
        """4sp + muh at V=+0.3 V: total dynamic packing must be < 1
        with comfortable margin.  At V=+0.3 V psi_D ~ 11.7, gamma at
        electrode ~ exp(-2.2) ~ 0.11, so c_ClO4 ~ 0.2*0.11*exp(11.7)
        ~ 22 (well below cap 100), c_H ~ 0.2*0.11*exp(-11.7) ~ 1.8e-7.
        Total packing ~ 0.01*(c_O2 + c_H2O2 + c_H + c_ClO4) ~ 0.01*23
        ~ 0.23 << 1.
        """
        ctx, _ = _build_ctx_4sp_muh(0.3)
        U = ctx["U"]
        n = ctx["n_species"]

        u_O2 = U.dat[0].data_ro
        u_H2O2 = U.dat[1].data_ro
        mu_h = U.dat[2].data_ro
        u_clo4 = U.dat[3].data_ro
        phi_field = U.dat[n].data_ro

        # Recover c_H from mu_H - phi (em*z_H = 1).
        u_h = mu_h - phi_field
        c_O2 = np.exp(u_O2)
        c_H2O2 = np.exp(u_H2O2)
        c_H = np.exp(u_h)
        c_ClO4 = np.exp(u_clo4)

        a = 0.01
        occupancy = a * (c_O2 + c_H2O2 + c_H + c_ClO4)
        assert np.all(np.isfinite(occupancy))
        assert occupancy.max() < 1.0 - 1e-3, (
            f"4sp+muh total packing must satisfy a*sum(c) < 1 - 1e-3; "
            f"got max={occupancy.max():.6f} at V=+0.3 V"
        )
        assert occupancy.min() >= 0.0

    def test_ic_reaches_phi_applied_at_electrode(self):
        ctx, _ = _build_ctx_4sp_muh(0.5)
        U = ctx["U"]
        n = ctx["n_species"]
        phi = U.dat[n].data_ro
        from scripts._bv_common import V_T
        phi_applied_nondim = 0.5 / V_T
        assert abs(phi.max() - phi_applied_nondim) < 1.0
