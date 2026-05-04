"""Unit tests for the debye_boltzmann initializer extended to 4sp dynamic.

Companion to ``tests/test_initializer_debye_boltzmann.py`` (3sp+Boltzmann).
The IC was extended in ``Forward/bv_solver/forms_logc.py`` to detect
``FOUR_SPECIES_LOGC_DYNAMIC`` configs (no Boltzmann counterion config,
species 3 = ClO4- with z=-1) and synthesise a virtual counterion entry
from ``c0_model[3]`` so the same Picard + Gouy-Chapman analytical IC
fires.  Without the extension the IC bails out and falls back to
linear-phi, producing a far-from-SS guess that the warm-walk cannot
extend past V≈+0.1 V (see
``StudyResults/peroxide_window_4sp_extended/``).

Slow tests; require Firedrake.
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


def _build_ctx_4sp(v_rhe: float, *, mesh_n: int = 8):
    """Build a fresh ctx with 4sp dynamic + linear_phi-fallback-disabled
    initializer, on a UnitSquareMesh(mesh_n)."""
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
        formulation="logc", log_rate=True,
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
class TestDebyeBoltzmann4spExtension:
    """The IC fires for 4sp dynamic instead of falling back."""

    def test_ic_fires_at_v05(self):
        """At V_RHE=+0.5 V (kinetic regime), the synthesised-counterion
        path runs the Picard + Gouy-Chapman IC successfully."""
        ctx, _ = _build_ctx_4sp(0.5)
        assert ctx.get("initializer_fallback") is False, (
            f"4sp debye_boltzmann should not fall back at V=+0.5 V; "
            f"reason={ctx.get('initializer_fallback_reason')!r}"
        )
        assert ctx.get("initializer_picard_iters", 0) > 0, (
            "Picard should have iterated at least once"
        )

    def test_ic_seeds_clo4_with_gamma_correction(self):
        """At V_RHE=+0.5 V the IC writes c_ClO4(y) on the Bikerman
        γ-corrected manifold:

            c_ClO4(y) = H_outer(y) · γ(ψ(y)) · exp(+ψ(y))

        where γ saturates the counterion concentration at 1/a (the
        Bikerman cap).  Pre-2a′ the IC seeded pure-Boltzmann
        c_ClO4 = c_bulk · exp(+ψ), which placed c_ClO4 ~ 4e7 at the
        electrode at V=+0.5 V — far above 1/a = 100.  The γ correction
        is what makes the IC compatible with the sign-corrected
        Bikerman residual."""
        ctx, _ = _build_ctx_4sp(0.5)
        U = ctx["U"]
        n = ctx["n_species"]  # = 4
        u3 = U.dat[3].data_ro
        phi = U.dat[n].data_ro

        assert np.all(np.isfinite(u3)), "u_3 has non-finite entries"
        assert np.all(np.isfinite(phi)), "phi has non-finite entries"

        c_clo4 = np.exp(u3)
        a_clo4 = 0.01  # A_DEFAULT
        steric_cap = 1.0 / a_clo4  # = 100 nondim

        assert np.all(c_clo4 > 0.0), "c_ClO4 must be positive everywhere"

        # Saturation cap: γ correction must keep c_ClO4 below the
        # Bikerman steric limit.  Allow tiny FE-interpolation slack.
        cap_tol = 0.5  # 0.5% above 1/a is FE noise; >>1% would be a bug
        assert c_clo4.max() <= steric_cap + cap_tol, (
            f"c_ClO4 surface max must respect Bikerman cap 1/a={steric_cap}; "
            f"got max={c_clo4.max():.4f}.  Without γ correction this would "
            f"be ~4e7 at V=+0.5 V — see docs/4sp_drop_boltzmann_investigation.md."
        )
        # Saturation must visibly approach the cap (else γ is too aggressive
        # and the IC has shrunk c_ClO4 below where Newton needs to start).
        assert c_clo4.max() >= 0.5 * steric_cap, (
            f"c_ClO4 surface max should approach saturation "
            f"(>= 50% of cap = {0.5 * steric_cap}); got {c_clo4.max():.4f}.  "
            f"If small, γ is over-correcting."
        )
        # Bulk recovery: at y=1 (top of unit square) γ → 1, ψ → 0,
        # H_outer → c_H_bulk = c_ClO4_bulk, so c_ClO4 → c_ClO4_bulk.
        c_clo4_bulk = 0.2  # C_CLO4_HAT
        assert abs(c_clo4.min() - c_clo4_bulk) / c_clo4_bulk < 0.05, (
            f"c_ClO4 bulk min should recover c_ClO4_bulk={c_clo4_bulk}; "
            f"got min={c_clo4.min():.4f}"
        )

    def test_ic_total_packing_positive_at_v0p3(self):
        """At V_RHE=+0.3 V (the binding case from the investigation log)
        the γ-corrected IC must keep total Bikerman packing comfortably
        positive at every node:

            1 - Σ_j a_j · c_j(y) > margin

        Without γ correction, max(Σ a c) ≈ 5.7e+5 at V=+0.5 (verified by
        running this test on pre-2a′ code) and ~200 at V=+0.3 — both
        catastrophically above 1, packing_floor clamps to 1e-8, Newton
        Jacobian sees a/packing ~ 1e+6.  With γ correction at V=+0.3
        the margin is ~5e-3 per the multispecies Bikerman equilibrium
        (see docs/4sp_bikerman_ic_option_2a_plan.md §2.3).

        At V=+0.5 the natural Bikerman packing margin shrinks to ~1e-6
        (still > packing_floor=1e-8), but that's deep-saturation
        physics, not an IC bug.  V=+0.3 is the right test voltage for
        the IC's "Newton-healthy starting state" property."""
        ctx, _ = _build_ctx_4sp(0.3)
        U = ctx["U"]
        n = ctx["n_species"]  # = 4

        c_O2   = np.exp(U.dat[0].data_ro)
        c_H2O2 = np.exp(U.dat[1].data_ro)
        c_H    = np.exp(U.dat[2].data_ro)
        c_ClO4 = np.exp(U.dat[3].data_ro)

        # All a_j = A_DEFAULT = 0.01 in FOUR_SPECIES_LOGC_DYNAMIC.
        a = 0.01
        occupancy = a * (c_O2 + c_H2O2 + c_H + c_ClO4)

        margin = 1e-3  # binds at ~5e-3 per Bikerman analytics; 1e-3 safety
        assert occupancy.max() < 1.0 - margin, (
            f"Total Bikerman packing fraction must satisfy "
            f"max(Σ a_j c_j) < 1 - {margin}; got max={occupancy.max():.6f}.  "
            f"If at or above 1.0, γ correction is missing on neutral species "
            f"or anchor is wrong (see 4sp_bikerman_ic_gpt_assessment.md)."
        )
        assert occupancy.min() >= 0.0, (
            "Occupancy must be non-negative (positive concentrations + a_j > 0)"
        )

    def test_ic_reaches_phi_applied_at_electrode(self):
        """Sanity: phi at the electrode (y=0) is approximately
        phi_applied_nondim, matching the Dirichlet BC."""
        ctx, _ = _build_ctx_4sp(0.5)
        U = ctx["U"]
        n = ctx["n_species"]
        phi = U.dat[n].data_ro
        from scripts._bv_common import V_T
        phi_applied_nondim = 0.5 / V_T
        assert abs(phi.max() - phi_applied_nondim) < 1.0, (
            f"phi.max should be ~{phi_applied_nondim:.2f}; got {phi.max():.2f}"
        )

    def test_ic_depletes_h_at_electrode(self):
        """Sanity: H+ at the electrode (y=0) is depleted by ~psi_D
        (the Boltzmann depletion for z=+1)."""
        ctx, _ = _build_ctx_4sp(0.5)
        U = ctx["U"]
        u_h = U.dat[2].data_ro
        c_h_bulk = 0.2  # C_HP_HAT
        u_h_bulk = float(np.log(c_h_bulk))
        assert u_h.min() < u_h_bulk - 15.0, (
            f"H+ should be depleted by ~19 nondim at electrode at "
            f"V=+0.5 V; got min(u_H)={u_h.min():.4f}, "
            f"bulk={u_h_bulk:.4f}"
        )

    def test_ic_uses_composite_psi_at_v0p66(self):
        """Regression target for Option 2b composite psi.

        At V_RHE=+0.66 V, psi_D ~ 25.7 >> psi_sat ~ 6.21.  The composite
        profile's saturated zone extends out to y_match ~ 0.25 * lam_D
        (zone 1, near-linear at slope alpha(psi_D)) before transitioning
        to the outer exponential decay at psi_sat.  The IC's phi field
        therefore reaches phi_applied at the electrode and reaches the
        outer baseline (~0) within the unit cell, with an intermediate
        plateau-like behaviour that pure Gouy-Chapman (psi=4*atanh tanh)
        does not exhibit (pre-2b GC drops from psi_D to ~0 in ~lam_D).

        We assert the gross signature: phi must reach BOTH ends (peak
        near phi_applied, trough near 0) within the unit square mesh.
        """
        ctx, _ = _build_ctx_4sp(0.66)
        U = ctx["U"]
        n = ctx["n_species"]
        phi = U.dat[n].data_ro

        from scripts._bv_common import V_T
        phi_applied_nondim = 0.66 / V_T  # ~25.7
        assert phi.max() > 0.5 * phi_applied_nondim, (
            f"phi.max should be near phi_applied={phi_applied_nondim:.2f} "
            f"at the electrode; got max={phi.max():.4f}"
        )
        assert phi.min() < 5.0, (
            f"phi.min must reach the outer baseline ~0 within the unit "
            f"square mesh; got min={phi.min():.4f}.  If too large, "
            f"composite psi is not transitioning to outer decay properly."
        )


@skip_without_firedrake
@pytest.mark.slow
class TestRegression3spStillWorks:
    """The extension must not perturb the existing 3sp+Boltzmann path."""

    def test_3sp_still_fires(self):
        """The original 3sp+Boltzmann debye_boltzmann path is unaffected
        by the synthesis logic (it only fires when counterions is empty)."""
        from scripts._bv_common import (
            setup_firedrake_env,
            V_T, K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2,
            THREE_SPECIES_LOGC_BOLTZMANN, DEFAULT_CLO4_BOLTZMANN_COUNTERION,
            SNES_OPTS_CHARGED, make_bv_solver_params,
        )
        setup_firedrake_env()
        import firedrake as fd
        from Forward.bv_solver import (
            build_context, build_forms, set_initial_conditions,
        )

        mesh = fd.UnitSquareMesh(8, 8)
        sp = make_bv_solver_params(
            eta_hat=0.5 / V_T, dt=0.25, t_end=80.0,
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

        assert ctx.get("initializer_fallback") is False
        assert ctx.get("initializer_picard_iters", 0) > 0
