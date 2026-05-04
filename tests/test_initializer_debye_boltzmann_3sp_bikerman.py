"""Initializer tests for 3sp + bikerman-mode analytic counterion.

The production target stack uses

    species     = THREE_SPECIES_LOGC_BOLTZMANN  (O2, H2O2, H+)
    counterion  = boltzmann_counterions=[{
                       z=-1, c_bulk_nondim=0.2, a_nondim=0.01,
                       steric_mode='bikerman', phi_clamp=50.0
                   }]

The residual side already uses ``build_steric_boltzmann_expressions`` to
saturate c_steric at ~1/a_b (see ``Forward/bv_solver/boltzmann.py:90``).
The IC side must seed the same Bikerman-consistent state:

    psi(y)        = composite (saturated zone + outer exponential decay)
    phi_init(y)   = ln(H_outer/c_clo4_bulk) + psi
    u_i_init(y)   = ln(c_outer_i) + log_gamma(psi(y))   for z=0 species
    u_H_init(y)   = ln(H_outer)   - psi(y) + log_gamma(psi(y))
    mu_H_init(y)  = u_H_init + em*z_H*phi_init    (muh path; psi cancels)

with

    gamma(psi) = 1 / [1 + a_h * H_outer * (exp(-psi) - 1)
                       + a_cl * c_clo4_bulk * (exp(+psi) - 1)]

(neutral species drop out -- exp(0) - 1 = 0).  The anchor for the
analytic counterion is its bulk c_clo4_bulk (constant), since c_outer
of the analytic species is the bulk in the matched-asymptotic outer
region.  This is the only structural difference from the 4sp dynamic
case (where the outer anchor for ClO4- is H_outer(y) by electroneutrality).

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


# Canonical bikerman counterion entry for tests below.
_BIKERMAN_CLO4 = {
    "z": -1,
    "c_bulk_nondim": 0.2,    # = C_CLO4_HAT
    "a_nondim": 0.01,        # = A_DEFAULT
    "steric_mode": "bikerman",
    "phi_clamp": 50.0,
}


def _build_ctx_3sp(
    v_rhe: float, *, mesh_n: int = 8,
    formulation: str = "logc",
    bikerman: bool = True,
):
    """Build a fresh ctx with 3sp + Boltzmann counterion at the requested
    voltage.  When ``bikerman=True``, the counterion entry uses
    ``steric_mode='bikerman'`` and the IC must apply the composite-psi +
    multispecies-gamma seeding.  When ``bikerman=False`` (legacy ideal),
    the IC must fall through to the byte-identical pre-2a'/pre-2b path.
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

    if bikerman:
        counterions = [dict(_BIKERMAN_CLO4)]
    else:
        counterions = [dict(DEFAULT_CLO4_BOLTZMANN_COUNTERION)]

    sp = make_bv_solver_params(
        eta_hat=v_rhe / V_T, dt=0.25, t_end=80.0,
        species=THREE_SPECIES_LOGC_BOLTZMANN,
        snes_opts=SNES_OPTS_CHARGED,
        formulation=formulation, log_rate=True,
        boltzmann_counterions=counterions,
        k0_hat_r1=K0_HAT_R1, k0_hat_r2=K0_HAT_R2,
        alpha_r1=ALPHA_R1, alpha_r2=ALPHA_R2,
        E_eq_r1=0.68, E_eq_r2=1.78,
        initializer="debye_boltzmann",
    )
    ctx = build_context(sp, mesh=mesh)
    ctx = build_forms(ctx, sp)
    set_initial_conditions(ctx, sp)
    return ctx, sp


# ============================================================================
# 3sp + bikerman, formulation="logc"
# ============================================================================

@skip_without_firedrake
@pytest.mark.slow
class Test3spBikermanLogc:
    """The Picard + composite-psi + gamma IC fires for 3sp + bikerman in logc."""

    def test_ic_fires_at_v05(self):
        ctx, _ = _build_ctx_3sp(0.5, formulation="logc", bikerman=True)
        assert ctx.get("initializer_fallback") is False, (
            f"3sp+bikerman debye_boltzmann should not fall back at "
            f"V=+0.5 V; reason={ctx.get('initializer_fallback_reason')!r}"
        )
        assert ctx.get("initializer_picard_iters", 0) > 0

    def test_ic_seeds_h_with_gamma_correction(self):
        """log_gamma must visibly enter u_H at the electrode.

        Without gamma:  u_H_init(y=0) = ln(H_outer) - psi(0) ~ ln(0.2) - 19 = -20.6
        With gamma:     u_H_init(y=0) = ln(H_outer) - psi(0) + log_gamma
                                       ~ -20.6 + (-12.8)             = -33.4

        The ~13 nondim shift is the multispecies Bikerman correction;
        it is the same one already landed in the 4sp dynamic IC at
        ``forms_logc.py:849`` (``synthesised_4sp_counterion`` branch).
        """
        ctx, _ = _build_ctx_3sp(0.5, formulation="logc", bikerman=True)
        U = ctx["U"]
        u_h = U.dat[2].data_ro

        c_h_bulk = 0.2  # C_HP_HAT
        u_h_bulk = float(np.log(c_h_bulk))

        # log_gamma at electrode is ~ -12.78 for V=+0.5 V (psi_D ~ 19).
        # Pre-correction surface u_H ~ -20.6; with correction ~ -33.4.
        # Use 25 as a conservative discriminator (between the two values).
        assert u_h.min() < u_h_bulk - 25.0, (
            f"u_H must include log_gamma at electrode (V=+0.5 V); got "
            f"min(u_H)={u_h.min():.4f}, bulk={u_h_bulk:.4f}.  Without "
            f"gamma the depletion would be ~19 nondim, not >25."
        )

    def test_ic_total_packing_positive_at_v0p3(self):
        """At V=+0.3 V, total Bikerman packing including the analytic
        counterion stays comfortably below 1.

        The analytic counterion's saturated concentration is bounded by
        ~1/a_b (= 100 nondim), so its packing contribution a_b * c_b_max
        is bounded by 1.  Combined with the dynamic species' tiny
        packing (< 0.012 in bulk) and the gamma correction on the
        dynamic species, total occupancy at any node should stay
        clearly below 1.
        """
        ctx, _ = _build_ctx_3sp(0.3, formulation="logc", bikerman=True)
        U = ctx["U"]

        c_O2 = np.exp(U.dat[0].data_ro)
        c_H2O2 = np.exp(U.dat[1].data_ro)
        c_H = np.exp(U.dat[2].data_ro)

        # All a_j = A_DEFAULT = 0.01 for the dynamic species.
        a_dyn = 0.01
        # Analytic counterion's saturated cap is 1/a_b = 100 (a_b = 0.01),
        # contributing at most a_b * c_b_steric = 1.0 to packing.
        # We don't have c_steric on the IC fields directly but the dynamic
        # contribution alone bounds the answer for this V.
        a_dyn_occupancy = a_dyn * (c_O2 + c_H2O2 + c_H)

        # Dynamic occupancy comfortably below 1 (the analytic counterion
        # adds at most ~1 but is bounded by gamma; the multispecies
        # gamma is what makes the total IC packing-safe).  Without gamma
        # on H+, the dynamic occupancy at the electrode would explode
        # (Boltzmann amplification of c_H by exp(-psi) for z=+1 actually
        # depletes H+, but neutral species would still see a positive
        # gamma shift).  Here we just check the dynamic side is sane.
        assert np.all(np.isfinite(a_dyn_occupancy))
        assert a_dyn_occupancy.max() < 1.0, (
            f"Dynamic-species occupancy must satisfy a*sum(c) < 1; got "
            f"max={a_dyn_occupancy.max():.6f} at V=+0.3 V"
        )
        assert np.all(a_dyn_occupancy >= 0.0)

    def test_ic_reaches_phi_applied_at_electrode(self):
        ctx, _ = _build_ctx_3sp(0.5, formulation="logc", bikerman=True)
        U = ctx["U"]
        n = ctx["n_species"]
        phi = U.dat[n].data_ro
        from scripts._bv_common import V_T
        phi_applied_nondim = 0.5 / V_T
        assert abs(phi.max() - phi_applied_nondim) < 1.0

    def test_ic_depletes_h_at_electrode(self):
        ctx, _ = _build_ctx_3sp(0.5, formulation="logc", bikerman=True)
        U = ctx["U"]
        u_h = U.dat[2].data_ro
        c_h_bulk = 0.2
        u_h_bulk = float(np.log(c_h_bulk))
        # H+ is z=+1, so deeply depleted at positive psi_D.
        assert u_h.min() < u_h_bulk - 15.0

    def test_ic_uses_composite_psi_at_v0p66(self):
        """At V=+0.66 V (psi_D ~ 25.7) the composite psi has a saturated
        zone ~y_match wide (zone 1 linear at slope alpha(psi_D)) before
        it transitions to the outer exponential decay at psi_sat ~ 6.21.

        For a UnitSquareMesh(8,8) on y in [0, 1], lam_D in nondim units
        depends on poisson_coefficient; the y_match should still be
        within the first cell.  Test indirectly via the surface phi
        gradient: pure-tanh GC at psi_D=25.7 would sit at psi(0)=25.7,
        decay rapidly, but the composite has a near-linear approach at
        the surface that's distinct.

        We test a softer property: phi values at the electrode and one
        cell in (y=1/8) differ by less than psi_D - psi_sat (i.e. zone 1
        does not collapse the entire potential to ~0 in one cell).  Pure
        tanh-GC at psi_D=25.7 would have psi(1/8 * lam_D-ish) drop to ~0,
        triggering FE node values consistent with the unsaturated case.
        """
        ctx, _ = _build_ctx_3sp(0.66, formulation="logc", bikerman=True)
        U = ctx["U"]
        n = ctx["n_species"]
        phi = U.dat[n].data_ro

        from scripts._bv_common import V_T
        phi_applied_nondim = 0.66 / V_T  # ~25.7
        # Composite zone 1 is linear at slope alpha(psi_D); at the
        # electrode (y=0) phi reaches phi_applied as enforced by the
        # IC.  Just before y_match it sits at phi_applied - alpha*y_match
        # = phi_applied - (psi_D_abs - psi_sat) ~ 6.21 above the bulk
        # background (which is ln(H_o/c_clo4_bulk) ~ 0).
        #
        # Thus min(phi) must be O(0): outer decay reaches the bulk
        # baseline ln(H_o/c_clo4_bulk) ~ 0 within the unit cell.
        assert phi.max() > 0.5 * phi_applied_nondim, (
            f"phi.max should be near phi_applied={phi_applied_nondim:.2f} "
            f"at electrode; got max={phi.max():.4f}"
        )
        assert phi.min() < 5.0, (
            f"phi.min must reach the outer baseline (~0); got "
            f"min={phi.min():.4f}.  If too large, the composite is not "
            f"transitioning to the outer exponential decay correctly."
        )


# ============================================================================
# 3sp + bikerman, formulation="logc_muh"
# ============================================================================

@skip_without_firedrake
@pytest.mark.slow
class Test3spBikermanMuh:
    """Same physics, exercised on the muh formulation."""

    def test_ic_fires_at_v05(self):
        ctx, _ = _build_ctx_3sp(0.5, formulation="logc_muh", bikerman=True)
        assert ctx.get("initializer_fallback") is False, (
            f"3sp+bikerman debye_boltzmann (muh) should not fall back at "
            f"V=+0.5 V; reason={ctx.get('initializer_fallback_reason')!r}"
        )
        assert ctx.get("initializer_picard_iters", 0) > 0

    def test_mu_h_propagates_log_gamma(self):
        """mu_H = u_H + em*z_H*phi.  With em*z_H = 1, psi cancels:

            mu_H_init = (ln H_outer - psi + log_gamma)
                       + (ln(H_outer/c_clo4_bulk) + psi)
                      = 2*ln(H_outer) - ln(c_clo4_bulk) + log_gamma

        Without log_gamma propagation, mu_H_init at the electrode is
        ~ 2*ln(0.2) - ln(0.2) = ln(0.2) ~ -1.61.  WITH log_gamma it
        shifts by ~ -12.78 to ~ -14.39.  Test that mu_H surface mean
        is below the log_gamma-free baseline by a clear margin.
        """
        ctx, _ = _build_ctx_3sp(0.5, formulation="logc_muh", bikerman=True)
        U = ctx["U"]
        mu_h = U.dat[2].data_ro

        c_h_bulk = 0.2
        c_clo4_bulk = 0.2
        # Pre-gamma baseline at the electrode: 2*ln(c_h_bulk) - ln(c_clo4_bulk).
        baseline = 2.0 * math.log(c_h_bulk) - math.log(c_clo4_bulk)
        # log_gamma at psi_D=19 ~ -12.8; mu_H surface should be at least
        # 8 below the baseline.
        assert mu_h.min() < baseline - 8.0, (
            f"mu_H must include log_gamma at electrode; got "
            f"min(mu_H)={mu_h.min():.4f}, baseline (no gamma)={baseline:.4f}.  "
            f"If close to baseline, log_gamma propagation is missing."
        )

    def test_ic_reaches_phi_applied_at_electrode(self):
        ctx, _ = _build_ctx_3sp(0.5, formulation="logc_muh", bikerman=True)
        U = ctx["U"]
        n = ctx["n_species"]
        phi = U.dat[n].data_ro
        from scripts._bv_common import V_T
        phi_applied_nondim = 0.5 / V_T
        assert abs(phi.max() - phi_applied_nondim) < 1.0

    def test_ic_total_packing_positive_at_v0p3(self):
        """Same packing-positive check, on the muh path.  The dynamic
        species' c_i are reconstructed via ``ctx['ci_exprs']`` because
        the proton storage is mu_H, but for the IC fields directly we
        decode c_H from mu_H via mu_H - phi (em*z_H = 1)."""
        ctx, _ = _build_ctx_3sp(0.3, formulation="logc_muh", bikerman=True)
        U = ctx["U"]
        n = ctx["n_species"]

        u_O2 = U.dat[0].data_ro
        u_H2O2 = U.dat[1].data_ro
        mu_h = U.dat[2].data_ro
        phi_field = U.dat[n].data_ro

        # u_H (logc-equivalent) = mu_H - em*z_H*phi (em*z_H = 1 here).
        u_h = mu_h - phi_field
        c_O2 = np.exp(u_O2)
        c_H2O2 = np.exp(u_H2O2)
        c_H = np.exp(u_h)

        a_dyn = 0.01
        a_dyn_occupancy = a_dyn * (c_O2 + c_H2O2 + c_H)
        assert np.all(np.isfinite(a_dyn_occupancy))
        assert a_dyn_occupancy.max() < 1.0


# ============================================================================
# Regression: 3sp + ideal counterion (no bikerman) -- byte-identical IC
# ============================================================================

@skip_without_firedrake
@pytest.mark.slow
class TestRegression3spIdealStillWorks:
    """3sp + ideal Boltzmann counterion: the bikerman/composite-psi
    gating must not perturb the legacy IC.

    The branch ``apply_bikerman_ic`` should evaluate False for the
    DEFAULT_CLO4_BOLTZMANN_COUNTERION (no ``steric_mode='bikerman'`` key,
    no ``synthesised_4sp_counterion``), so the IC walks the legacy
    Gouy-Chapman + linear-outer branch unchanged.

    Cross-checks against ``test_initializer_debye_boltzmann.py`` (same
    physics, slightly different assertions to catch byte-level drift).
    """

    def test_3sp_ideal_logc_unchanged(self):
        ctx, _ = _build_ctx_3sp(0.5, formulation="logc", bikerman=False)
        U = ctx["U"]
        n = ctx["n_species"]

        assert ctx.get("initializer_fallback") is False
        assert ctx.get("initializer_picard_iters", 0) > 0

        # Pre-2b 3sp+ideal seeds:
        #   u_O2(y)  = ln(c_O2_outer)             (no log_gamma)
        #   u_H2O2(y) = ln(c_H2O2_outer)          (no log_gamma)
        #   u_H(y)   = ln(H_outer) - psi          (no log_gamma)
        #   phi(y)   = ln(H_outer/c_clo4_bulk) + psi
        # In particular u_H bulk = ln(c_H_bulk) exactly (no offset).
        u_h = U.dat[2].data_ro
        c_h_bulk = 0.2
        u_h_bulk = float(np.log(c_h_bulk))
        assert abs(u_h.max() - u_h_bulk) < 1e-6, (
            f"3sp+ideal u_H bulk must equal ln(c_h_bulk)={u_h_bulk:.4f} "
            f"to bit-precision (no log_gamma offset); got max={u_h.max():.4f}"
        )

        # Surface depletion: ~psi_D = 19 nondim (no log_gamma offset).
        # Use a corridor that EXCLUDES the gamma-corrected value (~33).
        assert -22.0 < u_h.min() < -15.0, (
            f"3sp+ideal u_H surface should be ~ln(c_h_bulk) - psi_D ~ -20.6 "
            f"(no log_gamma offset); got min={u_h.min():.4f}.  If outside "
            f"this corridor, the bikerman gating leaked into the ideal path."
        )

    def test_3sp_ideal_logc_muh_unchanged(self):
        ctx, _ = _build_ctx_3sp(0.5, formulation="logc_muh", bikerman=False)
        U = ctx["U"]

        assert ctx.get("initializer_fallback") is False
        assert ctx.get("initializer_picard_iters", 0) > 0

        # Pre-2b 3sp+ideal in muh:
        #   mu_H_init = (ln H_outer - psi) + (ln(H_outer/c_clo4_bulk) + psi)
        #             = 2*ln(H_outer) - ln(c_clo4_bulk)        (no gamma)
        # At bulk (y=1) H_outer = c_h_bulk, so mu_H_bulk = 2*ln(0.2) - ln(0.2)
        # = ln(0.2) ~ -1.61.
        mu_h = U.dat[2].data_ro
        c_h_bulk = 0.2
        c_clo4_bulk = 0.2
        baseline_bulk = 2.0 * math.log(c_h_bulk) - math.log(c_clo4_bulk)
        # Bulk mu_H must hit the baseline within FE interpolation slack.
        assert abs(mu_h.max() - baseline_bulk) < 0.1, (
            f"3sp+ideal mu_H bulk must equal {baseline_bulk:.4f} "
            f"(no log_gamma); got max={mu_h.max():.4f}"
        )
        # Surface mu_H: still 2*ln(H_outer) - ln(c_clo4_bulk) but H_outer at
        # the electrode is depleted by ~psi_D in the surface picard solve.
        # No log_gamma offset means mu_H surface stays close to baseline:
        # corridor [baseline - 8, baseline + 1].
        assert mu_h.min() > baseline_bulk - 8.0, (
            f"3sp+ideal mu_H surface should NOT show log_gamma offset; "
            f"got min={mu_h.min():.4f}, baseline={baseline_bulk:.4f}.  "
            f"If much lower (e.g. < -10), bikerman gating leaked into "
            f"the ideal path."
        )
