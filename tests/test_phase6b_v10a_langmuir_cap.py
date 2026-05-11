"""Phase 6β v10a — Langmuir capacity cap regression tests.

Phase 6β v9's cation-hydrolysis residual had no upper bound on the
surface coverage ``Γ``: at converged ``k_hyd ≥ 1e-3`` every λ=1
solution sat at Γ ≈ 6+ monolayers (physically invalid; ~64 monolayers
at ``k_hyd = 1e-2``).  v10a adds a Langmuir vacancy factor
``(1 − Γ/Γ_max)`` to the forward branch so coverage saturates at one
monolayer instead.

This file tests the closed-form Γ_ss formula, the residual-side
vacancy factor, the Singh anode-clamp invariant, the
``sigma_C_m2_to_counts_pm2`` helper round-trip, and the
``v9 byte-equivalence as Γ_max → ∞`` invariant.

Fast unit tests live up top; slow Firedrake-backed tests are marked
``@pytest.mark.slow`` and require the venv-firedrake interpreter.

References
----------
* ``.claude/plans/cryptic-honking-cosmos.md`` — v10a plan.
* ``docs/phase6/PHASE_6B_V9_PHASES_A_B_RESULTS_2026-05-10.md`` — the
  v9 outcome that motivates the cap.
"""
from __future__ import annotations

import math
import os
import sys
import warnings

import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ===================================================================
# Smoke baseline constants — sanity check against
# Forward.bv_solver.cation_hydrolysis
# ===================================================================


class TestSmokeConstantConsistency:
    """The ``GAMMA_MAX_HAT_V10A_SMOKE`` constant must agree between the
    solver module and the script-level helper, so the gate4 driver
    and the bundle never disagree on the default.

    v10b refactor (2026-05-10): references the explicit V10A frozen
    historical constant rather than the deprecated SMOKE alias to keep
    this test stable across future calibration cycles.
    """

    def test_constants_agree(self):
        from Forward.bv_solver.cation_hydrolysis import (
            GAMMA_MAX_HAT_V10A_SMOKE as SOLVER_V10A,
        )
        from scripts._bv_common import (
            GAMMA_MAX_HAT_V10A_SMOKE as SCRIPT_V10A,
        )
        assert SOLVER_V10A == pytest.approx(SCRIPT_V10A, rel=1e-15)
        # Sanity: ~0.047 nondim ≈ 1 monolayer at C_SCALE=1.2, L_REF=1e-4.
        # (See module docstring derivation.)
        assert SOLVER_V10A == pytest.approx(0.047, rel=1e-2)

    def test_smoke_alias_still_points_at_v10a_smoke(self):
        """The deprecated SMOKE alias keeps backward compat by pointing
        at V10A_SMOKE -- never at V10B."""
        from Forward.bv_solver.cation_hydrolysis import (
            GAMMA_MAX_HAT_SMOKE,
            GAMMA_MAX_HAT_V10A_SMOKE,
            GAMMA_MAX_HAT_V10B,
        )
        assert GAMMA_MAX_HAT_SMOKE == pytest.approx(
            GAMMA_MAX_HAT_V10A_SMOKE, rel=1e-15
        )
        # V10B is currently numerically equal to V10A (the v10b 4-test
        # compatibility check tightened the V10A chain rather than
        # replacing the value), but the alias semantic is still
        # SMOKE -> V10A_SMOKE, NEVER SMOKE -> V10B.
        assert GAMMA_MAX_HAT_V10B == pytest.approx(0.047, rel=1e-15)


# ===================================================================
# Pure-Python Langmuir closed-form (gamma_ss_langmuir)
# ===================================================================


def _v9_gamma_ss(*, lam, k_hyd, k_prot, k_des, delta, F_avg, c_H):
    """v9 closed-form Γ_ss (no Langmuir cap) — used as the byte-equivalence
    reference for the Γ_max → ∞ limit."""
    F0 = k_hyd * F_avg
    numerator = lam * F0
    denominator = lam * k_des + (1.0 - lam) + lam * k_prot * c_H / delta
    return numerator / denominator


class TestLangmuirClosedForm:
    """Pure-Python tests on the closed-form ``gamma_ss_langmuir``
    helper.  No Firedrake required."""

    def test_lambda_zero_returns_zero(self):
        """Dirichlet pin invariant: λ=0 ⇒ Γ_ss = 0 (Gate 3D
        invariant survives the v10a cap)."""
        from Forward.bv_solver.cation_hydrolysis import (
            GAMMA_MAX_HAT_V10A_SMOKE,
            gamma_ss_langmuir,
        )
        g, g_un, _ = gamma_ss_langmuir(
            lambda_val=0.0,
            k_hyd=1e-3, k_prot=1e-3, k_des=1.0,
            delta_ohp=4e-6, forward_avg=2.5,
            c_H_avg=0.0833, gamma_max=GAMMA_MAX_HAT_V10A_SMOKE,
        )
        assert g == pytest.approx(0.0, abs=1e-30)
        assert g_un == pytest.approx(0.0, abs=1e-30)

    def test_reduces_to_v9_when_gamma_max_large(self):
        """Γ_max → ∞ ⇒ closed-form reduces to v9 formula."""
        from Forward.bv_solver.cation_hydrolysis import gamma_ss_langmuir
        # Use a finite but huge Γ_max to avoid 0/∞ in the formula —
        # the plan's recommended ~1e10 sentinel (see Risk #5).
        kwargs = dict(
            lambda_val=0.6,
            k_hyd=2e-3, k_prot=5e-3, k_des=1.5,
            delta_ohp=4e-6, forward_avg=2.5,
            c_H_avg=0.0833,
        )
        v9_ref = _v9_gamma_ss(
            lam=kwargs["lambda_val"], k_hyd=kwargs["k_hyd"],
            k_prot=kwargs["k_prot"], k_des=kwargs["k_des"],
            delta=kwargs["delta_ohp"], F_avg=kwargs["forward_avg"],
            c_H=kwargs["c_H_avg"],
        )
        g, g_un, _ = gamma_ss_langmuir(gamma_max=1.0e10, **kwargs)
        # Cap term λ·F₀/Γ_max = 0.6·(2e-3·2.5)/1e10 ≈ 3e-13 — far below
        # the v9 denominator scale (O(1)), so byte-equivalence holds to
        # better than ~1e-12 relative.
        assert g_un == pytest.approx(v9_ref, rel=1e-12)
        # Γ_max=1e10 is well above any physical Γ; clamp is a no-op.
        assert g == pytest.approx(g_un, rel=1e-15)

    def test_saturates_to_gamma_max_as_k_hyd_diverges(self):
        """F₀ → ∞ at fixed λ > 0 ⇒ Γ_ss → Γ_max (the cap actually saturates)."""
        from Forward.bv_solver.cation_hydrolysis import (
            GAMMA_MAX_HAT_V10A_SMOKE,
            gamma_ss_langmuir,
        )
        gamma_max = GAMMA_MAX_HAT_V10A_SMOKE
        # Sweep k_hyd over six orders of magnitude.
        ks = [1e0, 1e2, 1e4, 1e6, 1e8, 1e10]
        prev_g = 0.0
        for k_hyd in ks:
            g, _, _ = gamma_ss_langmuir(
                lambda_val=1.0,
                k_hyd=k_hyd, k_prot=1e-3, k_des=1.0,
                delta_ohp=4e-6, forward_avg=2.5,
                c_H_avg=0.0833, gamma_max=gamma_max,
            )
            assert g <= gamma_max + 1e-15, (
                f"Γ exceeded Γ_max at k_hyd={k_hyd}: Γ={g}, Γ_max={gamma_max}"
            )
            # Monotonic in k_hyd (more forward forcing ⇒ Γ closer to cap).
            assert g >= prev_g - 1e-15
            prev_g = g
        # Final Γ at the largest k_hyd should be essentially Γ_max.
        assert prev_g == pytest.approx(gamma_max, rel=1e-6)

    def test_vacancy_factor_nonneg_implied_by_clamp(self):
        """The closed form respects (1 − Γ/Γ_max) ≥ 0 for any inputs
        in the supported parameter ranges."""
        from Forward.bv_solver.cation_hydrolysis import (
            GAMMA_MAX_HAT_V10A_SMOKE,
            gamma_ss_langmuir,
        )
        gamma_max = GAMMA_MAX_HAT_V10A_SMOKE
        for lam in (0.1, 0.25, 0.5, 0.75, 1.0):
            for k_hyd in (1e-6, 1e-3, 1e0, 1e3, 1e6):
                g, _, _ = gamma_ss_langmuir(
                    lambda_val=lam,
                    k_hyd=k_hyd, k_prot=1e-3, k_des=1.0,
                    delta_ohp=4e-6, forward_avg=2.5,
                    c_H_avg=0.0833, gamma_max=gamma_max,
                )
                vacancy = 1.0 - g / gamma_max
                assert vacancy >= -1e-12, (
                    f"vacancy factor went negative: λ={lam} k_hyd={k_hyd} "
                    f"Γ={g} Γ_max={gamma_max} ⇒ 1−θ={vacancy}"
                )
                assert vacancy <= 1.0 + 1e-15

    def test_denominator_decomposition_sums_correctly(self):
        """Denominator dict sums to the total denominator and matches
        the formula term-by-term."""
        from Forward.bv_solver.cation_hydrolysis import gamma_ss_langmuir
        _, _, terms = gamma_ss_langmuir(
            lambda_val=0.7,
            k_hyd=1e-3, k_prot=5e-3, k_des=1.5,
            delta_ohp=4e-6, forward_avg=2.0,
            c_H_avg=0.0833, gamma_max=0.05,
        )
        total = terms["constant"] + terms["kdes"] + terms["kprot"] + terms["cap"]
        assert total == pytest.approx(terms["total"], rel=1e-15)
        # Spot-check individual terms.
        assert terms["constant"] == pytest.approx(0.3, rel=1e-15)
        assert terms["kdes"] == pytest.approx(0.7 * 1.5, rel=1e-15)
        # cap = λ · k_hyd · forward_avg / Γ_max
        expected_cap = 0.7 * 1e-3 * 2.0 / 0.05
        assert terms["cap"] == pytest.approx(expected_cap, rel=1e-15)

    def test_invalid_gamma_max_raises(self):
        from Forward.bv_solver.cation_hydrolysis import gamma_ss_langmuir
        for bad in (0.0, -1e-6):
            with pytest.raises(ValueError, match="gamma_max must be positive"):
                gamma_ss_langmuir(
                    lambda_val=0.5,
                    k_hyd=1e-3, k_prot=1e-3, k_des=1.0,
                    delta_ohp=4e-6, forward_avg=1.0,
                    c_H_avg=0.0833, gamma_max=bad,
                )

    def test_invalid_lambda_raises(self):
        from Forward.bv_solver.cation_hydrolysis import gamma_ss_langmuir
        for bad in (-0.1, 1.5):
            with pytest.raises(ValueError, match="lambda_val"):
                gamma_ss_langmuir(
                    lambda_val=bad,
                    k_hyd=1e-3, k_prot=1e-3, k_des=1.0,
                    delta_ohp=4e-6, forward_avg=1.0,
                    c_H_avg=0.0833, gamma_max=0.047,
                )

    def test_invalid_kinetic_rates_raise(self):
        from Forward.bv_solver.cation_hydrolysis import gamma_ss_langmuir
        with pytest.raises(ValueError, match="k_hyd"):
            gamma_ss_langmuir(
                lambda_val=0.5, k_hyd=-1.0,
                k_prot=1e-3, k_des=1.0,
                delta_ohp=4e-6, forward_avg=1.0,
                c_H_avg=0.0833, gamma_max=0.047,
            )
        with pytest.raises(ValueError, match="delta_ohp must be positive"):
            gamma_ss_langmuir(
                lambda_val=0.5, k_hyd=1e-3,
                k_prot=1e-3, k_des=1.0,
                delta_ohp=0.0, forward_avg=1.0,
                c_H_avg=0.0833, gamma_max=0.047,
            )


class TestNumericalSelfConsistency:
    """Risk #2 mitigation: derive Γ_ss numerically by Newton-solving
    the residual ``F₀·(1−θ) − B·Γ − k_des·Γ = 0`` (the λ=1 limit) and
    compare to the closed-form helper.  Catches algebra errors that
    R→0 / R→∞ limit tests would miss.
    """

    def _residual_lambda_one(self, *, gamma, F0, B, k_des, gamma_max):
        """Residual: F₀·(1 − Γ/Γ_max) − (B + k_des)·Γ."""
        return F0 * (1.0 - gamma / gamma_max) - (B + k_des) * gamma

    def test_closed_form_matches_newton_solve_at_lambda_one(self):
        from Forward.bv_solver.cation_hydrolysis import gamma_ss_langmuir
        # Realistic v10a parameters.
        k_hyd = 1e-3
        k_prot = 5e-3
        k_des = 1.0
        delta = 4e-6
        forward_avg = 2.5
        c_H_avg = 0.0833
        gamma_max = 0.047

        F0 = k_hyd * forward_avg
        B = k_prot * c_H_avg / delta
        g_closed, _, terms = gamma_ss_langmuir(
            lambda_val=1.0,
            k_hyd=k_hyd, k_prot=k_prot, k_des=k_des,
            delta_ohp=delta, forward_avg=forward_avg,
            c_H_avg=c_H_avg, gamma_max=gamma_max,
        )
        # Residual must be ~0 at the closed-form root.
        r = self._residual_lambda_one(
            gamma=g_closed, F0=F0, B=B, k_des=k_des, gamma_max=gamma_max,
        )
        assert abs(r) < 1e-15 * max(abs(F0), 1.0)

        # Bisection-solve independently.  Residual is linear in Γ so
        # Newton converges in one step; bisection is overkill but
        # offers a fully independent path.
        lo, hi = 0.0, gamma_max
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if self._residual_lambda_one(
                gamma=mid, F0=F0, B=B, k_des=k_des, gamma_max=gamma_max,
            ) > 0.0:
                lo = mid
            else:
                hi = mid
        g_bisect = 0.5 * (lo + hi)
        assert g_closed == pytest.approx(g_bisect, rel=1e-10)


# ===================================================================
# σ unit-conversion helper round-trip (Singh Cu K⁺ values)
# ===================================================================


class TestSigmaUnitHelper:
    """``sigma_C_m2_to_counts_pm2`` must round-trip values used in
    the Singh 2016 pKa formula.  Plan #5 + Risk #2: keep the in-form
    UFL converter and the Python diagnostic helper in lockstep.
    """

    def test_zero_in_zero_out(self):
        from Forward.bv_solver.units import sigma_C_m2_to_counts_pm2
        assert sigma_C_m2_to_counts_pm2(0.0) == 0.0

    def test_sign_preserved(self):
        from Forward.bv_solver.units import sigma_C_m2_to_counts_pm2
        assert sigma_C_m2_to_counts_pm2(+1.0) > 0.0
        assert sigma_C_m2_to_counts_pm2(-1.0) < 0.0
        # Magnitude matches up to sign.
        assert sigma_C_m2_to_counts_pm2(-1.0) == -sigma_C_m2_to_counts_pm2(+1.0)

    def test_round_trip_at_generic_cross_check_value(self):
        """Generic round-trip cross-check at σ = 0.226 C/m² →
        ~1.41e-6 counts/pm² (= 0.226 · 6.2415e-6).

        NOTE: 0.226 C/m² is NOT Singh Table S3's K⁺/Cu σ value.
        Singh's Table S3 σ is 0.141 counts/pm² in Singh's
        cell-level convention (cell-level σ from C_dl·Δφ_cell;
        see `docs/phase6/singh_2016_pka_formula.md` §5.2 +
        acceptance bundle § "Σ_S mapping convention").  The
        equivalent Stern-σ-in-C/m² for Singh's 0.141 counts/pm²
        is `0.141 / 6.2415e-6 ≈ 22,591 C/m²` — unphysically large
        for a local Stern σ_S.  Our PNP-Stern model uses the LOCAL
        Stern σ_S convention (~0.017 C/m² magnitude at V_kin),
        which produces `pka_shift_avg ≈ −4.88e-6` per A.2 record
        — a different physical quantity than Singh's table value.

        The conversion-factor algebra round-trips regardless of
        which σ convention is in use; that's what this test pins.
        """
        from Forward.bv_solver.units import sigma_C_m2_to_counts_pm2
        val = sigma_C_m2_to_counts_pm2(0.226)
        # 0.226 C/m² · 6.2415e-6 ≈ 1.4106e-6 counts/pm².
        expected = 0.226 * 6.241509e-6
        assert val == pytest.approx(expected, rel=1e-6)

    def test_conversion_factor_matches_inverse_e(self):
        """Helper agrees with N_A/F · 1e-24 to floating-point precision."""
        from Forward.bv_solver.units import sigma_C_m2_to_counts_pm2
        # 1 C/m² → (1/e) · 1e-24 counts/pm²
        e_charge = 1.602176634e-19
        expected_per_unit = (1.0 / e_charge) * 1.0e-24
        assert sigma_C_m2_to_counts_pm2(1.0) == pytest.approx(
            expected_per_unit, rel=1e-15
        )

    def test_in_form_ufl_constant_agrees_with_helper(self):
        """The form-build code uses N_A/F * 1e-24 = (1/e) · 1e-24.
        Verify the helper matches the same numerical constant the
        residual UFL bakes in.
        """
        # The form-build code (cation_hydrolysis.py) computes
        # ``counts_per_m2_per_C_per_m2 * pm2_per_m2`` =
        # ``(N_A / F) * 1e-24``.  Replicate that here.
        from Forward.bv_solver.cation_hydrolysis import _avogadro_per_faraday
        from Forward.bv_solver.units import sigma_C_m2_to_counts_pm2

        form_factor = _avogadro_per_faraday() * 1.0e-24
        helper_factor = sigma_C_m2_to_counts_pm2(1.0)
        assert form_factor == pytest.approx(helper_factor, rel=1e-15)


# ===================================================================
# Singh pKa shift — anode-clamp + β-sign invariants (UFL backend)
# ===================================================================


@pytest.mark.slow
class TestSinghAnodeClampInvariant:
    """At σ_S > 0 (anodic), Singh's anode_clamp drives ΔpKa = 0 →
    10^(−ΔpKa) = 1 → no extra forward forcing.  Combined with v10a,
    this means Γ_ss at λ=1 and σ_S > 0 reduces to the K⁺ bulk c_M
    × 10^0 = c_M case.  Verifies that v10a does not subtly break the
    anode clamp.
    """

    def test_anodic_sigma_clamps_to_zero(self):
        import firedrake as fd
        from Forward.bv_solver.cation_hydrolysis import build_pka_shift
        from scripts._bv_common import make_singh_pka_shift_params

        mesh = fd.UnitSquareMesh(4, 4)
        params = make_singh_pka_shift_params("K+")
        # σ_S = +1.0 C/m² → anode → clamp → ΔpKa = 0 exactly.
        pka_shift = build_pka_shift(
            cation_params={**params, "pka_shift_form": "singh_2016_eq_4"},
            sigma_S=fd.Constant(1.0),
        )
        val = float(fd.assemble(pka_shift * fd.dx(domain=mesh))) / float(
            fd.assemble(fd.Constant(1.0) * fd.dx(domain=mesh))
        )
        assert val == pytest.approx(0.0, abs=1e-30)


@pytest.mark.slow
class TestSinghCathodicSignGuard:
    """β sign-guard: cathodic σ_S < 0 produces ΔpKa < 0 for K⁺.

    Singh's Eq. (4) ``ΔpKa = +2·A·z·σ_singh·r_H_El·(1 − r_M-O² / r_H_El²)``
    with ``σ_singh = max(0, −σ_S) · (N_A/F) · 1e-24`` (anode-clamped
    magnitude) gives:
       cathodic σ_S < 0  ⇒  σ_singh > 0
       K⁺ geometry: r_H_El = 200.98 pm, r_M-O = 138 + 63 = 201 pm
       (r_M-O > r_H_El by 0.02 pm)
       ⇒ (1 − r_M-O²/r_H_El²) < 0
       ⇒ ΔpKa < 0  ⇒  10^(−ΔpKa) > 1  ⇒  forward forcing amplified.

    This matches Singh's intent (cathodic field lowers the hydrolysis
    pKa, producing more H⁺).
    """

    def test_cathodic_drives_pka_shift_negative(self):
        import firedrake as fd
        from Forward.bv_solver.cation_hydrolysis import build_pka_shift
        from scripts._bv_common import make_singh_pka_shift_params

        mesh = fd.UnitSquareMesh(4, 4)
        params = make_singh_pka_shift_params("K+")
        pka_shift = build_pka_shift(
            cation_params={**params, "pka_shift_form": "singh_2016_eq_4"},
            sigma_S=fd.Constant(-0.5),  # cathodic C/m²
        )
        val = float(fd.assemble(pka_shift * fd.dx(domain=mesh))) / float(
            fd.assemble(fd.Constant(1.0) * fd.dx(domain=mesh))
        )
        assert val < 0.0, (
            f"K⁺ cathodic ΔpKa expected negative; got {val}"
        )

    def test_zero_sigma_gives_zero_shift(self):
        """Sign-guard floor: σ_S = 0 → ΔpKa = 0 exactly."""
        import firedrake as fd
        from Forward.bv_solver.cation_hydrolysis import build_pka_shift
        from scripts._bv_common import make_singh_pka_shift_params

        mesh = fd.UnitSquareMesh(4, 4)
        params = make_singh_pka_shift_params("K+")
        pka_shift = build_pka_shift(
            cation_params={**params, "pka_shift_form": "singh_2016_eq_4"},
            sigma_S=fd.Constant(0.0),
        )
        val = float(fd.assemble(pka_shift * fd.dx(domain=mesh))) / float(
            fd.assemble(fd.Constant(1.0) * fd.dx(domain=mesh))
        )
        assert val == pytest.approx(0.0, abs=1e-30)


# ===================================================================
# Bundle build — gamma_max_func appears on the bundle with the right default
# ===================================================================


class TestBundleSchema:
    """Module-level sanity on the v10a additions (no Firedrake needed —
    only checks the bundle's class shape + the public surface)."""

    def test_bundle_dataclass_has_gamma_max_field(self):
        import dataclasses
        from Forward.bv_solver.cation_hydrolysis import CationHydrolysisBundle
        names = {f.name for f in dataclasses.fields(CationHydrolysisBundle)}
        assert "gamma_max_func" in names

    def test_module_exports_v10a_public_surface(self):
        import Forward.bv_solver.cation_hydrolysis as ch
        assert "gamma_ss_langmuir" in ch.__all__
        assert "GAMMA_MAX_HAT_SMOKE" in ch.__all__
        assert "clamp_gamma_to_max" in ch.__all__
        assert "collect_v10a_rung_diagnostics" in ch.__all__
        assert "build_forward_branch_uncapped" in ch.__all__

    def test_anchor_continuation_exports_setter(self):
        import Forward.bv_solver.anchor_continuation as ac
        assert "set_reaction_gamma_max_model" in ac.__all__


class TestGammaMaxSetterErrors:
    """Fast unit tests on the accessor error paths."""

    def test_set_gamma_max_no_bundle_raises(self):
        from Forward.bv_solver.anchor_continuation import (
            set_reaction_gamma_max_model,
        )
        with pytest.raises(ValueError, match="no 'cation_hydrolysis' bundle"):
            set_reaction_gamma_max_model({}, 0.047)

    def test_set_gamma_max_zero_raises(self):
        from Forward.bv_solver.anchor_continuation import (
            set_reaction_gamma_max_model,
        )
        with pytest.raises(ValueError, match="must be positive"):
            set_reaction_gamma_max_model({}, 0.0)

    def test_set_gamma_max_negative_raises(self):
        from Forward.bv_solver.anchor_continuation import (
            set_reaction_gamma_max_model,
        )
        with pytest.raises(ValueError, match="must be non-negative"):
            set_reaction_gamma_max_model({}, -0.01)

    def test_warm_start_override_supports_gamma_max(self):
        """``solve_lambda_ramp_from_warm_start`` allows the
        ``gamma_max_nondim`` key in ``parameter_overrides``.

        We cannot easily call the full orchestrator without Firedrake,
        so we inspect the source for the literal key.  If the
        ``_OVERRIDE_DISPATCH`` dict ever loses the key, this string
        check trips.
        """
        import inspect
        from Forward.bv_solver import anchor_continuation as ac
        src = inspect.getsource(ac.solve_lambda_ramp_from_warm_start)
        assert '"gamma_max_nondim"' in src


# ===================================================================
# Bundle build (slow) — verify the bundle picks up gamma_max_nondim
# from cation_hydrolysis_config and the residual sees the cap.
# ===================================================================


def _build_v10a_sp(*, gamma_max_nondim=None):
    """Build solver-params for slow v10a integration tests.  Mirrors
    the gate3 gamma-machinery test fixture but adds an explicit
    ``gamma_max_nondim``."""
    from scripts._bv_common import (
        DEFAULT_SULFATE_ANALYTIC_BIKERMAN_FOR_K2SO4,
        FOUR_SPECIES_LOGC_DYNAMIC_K2SO4,
        GAMMA_MAX_HAT_SMOKE,
        make_bv_solver_params,
        make_cation_hydrolysis_config,
    )

    cation_cfg = make_cation_hydrolysis_config(
        k_hyd=1e-4, k_prot=1e-4, k_des=1.0,
        delta_ohp_hat=1e-2,
        cation="K+",
        pka_shift_form="placeholder",
        gamma_max_nondim=(
            GAMMA_MAX_HAT_SMOKE if gamma_max_nondim is None
            else float(gamma_max_nondim)
        ),
    )

    return make_bv_solver_params(
        eta_hat=0.0, dt=0.25, t_end=10.0,
        species=FOUR_SPECIES_LOGC_DYNAMIC_K2SO4,
        formulation="logc_muh",
        log_rate=True,
        boltzmann_counterions=[DEFAULT_SULFATE_ANALYTIC_BIKERMAN_FOR_K2SO4],
        stern_capacitance_f_m2=0.10,
        initializer="linear_phi",
        enable_cation_hydrolysis=True,
        cation_hydrolysis_config=cation_cfg,
    )


@pytest.mark.slow
class TestBundlePicksUpGammaMaxFromConfig:
    """Slow Firedrake-backed: the bundle's ``gamma_max_func`` reflects
    whatever the caller passed in ``cation_hydrolysis_config``."""

    def test_gamma_max_default_is_smoke_baseline(self):
        import firedrake as fd
        from Forward.bv_solver.cation_hydrolysis import GAMMA_MAX_HAT_SMOKE
        from Forward.bv_solver.dispatch import build_context, build_forms

        sp = _build_v10a_sp(gamma_max_nondim=None)
        mesh = fd.UnitSquareMesh(8, 8)
        ctx = build_context(sp, mesh=mesh)
        ctx = build_forms(ctx, sp)
        bundle = ctx["cation_hydrolysis"]
        assert float(bundle.gamma_max_func) == pytest.approx(
            GAMMA_MAX_HAT_SMOKE, rel=1e-12
        )

    def test_gamma_max_custom_value_propagates(self):
        import firedrake as fd
        from Forward.bv_solver.dispatch import build_context, build_forms

        sp = _build_v10a_sp(gamma_max_nondim=0.10)
        mesh = fd.UnitSquareMesh(8, 8)
        ctx = build_context(sp, mesh=mesh)
        ctx = build_forms(ctx, sp)
        bundle = ctx["cation_hydrolysis"]
        assert float(bundle.gamma_max_func) == pytest.approx(0.10, rel=1e-12)

    def test_setter_round_trips_through_metadata_and_function(self):
        """``set_reaction_gamma_max_model`` writes both the metadata
        layer and the live FE Function."""
        import firedrake as fd
        from Forward.bv_solver.anchor_continuation import (
            set_reaction_gamma_max_model,
        )
        from Forward.bv_solver.dispatch import build_context, build_forms

        sp = _build_v10a_sp(gamma_max_nondim=0.05)
        mesh = fd.UnitSquareMesh(8, 8)
        ctx = build_context(sp, mesh=mesh)
        ctx = build_forms(ctx, sp)
        bundle = ctx["cation_hydrolysis"]

        # Initial value pulled from the config.
        assert float(bundle.gamma_max_func) == pytest.approx(0.05, rel=1e-12)

        # Setter writes the FE Function...
        set_reaction_gamma_max_model(ctx, 0.20)
        assert float(bundle.gamma_max_func) == pytest.approx(0.20, rel=1e-12)
        # ...AND the metadata layer.
        assert (
            ctx["bv_convergence"]["cation_hydrolysis_config"]
            ["gamma_max_nondim"] == pytest.approx(0.20, rel=1e-12)
        )


@pytest.mark.slow
class TestClampGammaToMaxHelper:
    """``clamp_gamma_to_max`` enforces ``Γ ∈ [0, Γ_max]`` on warm
    restarts silently (no warning)."""

    def test_clamp_to_zero(self):
        import firedrake as fd
        from Forward.bv_solver.cation_hydrolysis import clamp_gamma_to_max
        from Forward.bv_solver.dispatch import build_context, build_forms

        sp = _build_v10a_sp(gamma_max_nondim=0.10)
        mesh = fd.UnitSquareMesh(8, 8)
        ctx = build_context(sp, mesh=mesh)
        ctx = build_forms(ctx, sp)
        bundle = ctx["cation_hydrolysis"]

        bundle.gamma_func.assign(-0.5)        # below zero
        # No warning — warm-restart clamp is intentionally silent.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            new_gamma = clamp_gamma_to_max(ctx)
        assert new_gamma == pytest.approx(0.0, abs=1e-15)
        assert float(bundle.gamma_func) == pytest.approx(0.0, abs=1e-15)

    def test_clamp_to_gamma_max(self):
        import firedrake as fd
        from Forward.bv_solver.cation_hydrolysis import clamp_gamma_to_max
        from Forward.bv_solver.dispatch import build_context, build_forms

        sp = _build_v10a_sp(gamma_max_nondim=0.10)
        mesh = fd.UnitSquareMesh(8, 8)
        ctx = build_context(sp, mesh=mesh)
        ctx = build_forms(ctx, sp)
        bundle = ctx["cation_hydrolysis"]

        bundle.gamma_func.assign(0.5)         # above gamma_max=0.10
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            new_gamma = clamp_gamma_to_max(ctx)
        assert new_gamma == pytest.approx(0.10, rel=1e-12)

    def test_clamp_in_range_noop(self):
        import firedrake as fd
        from Forward.bv_solver.cation_hydrolysis import clamp_gamma_to_max
        from Forward.bv_solver.dispatch import build_context, build_forms

        sp = _build_v10a_sp(gamma_max_nondim=0.10)
        mesh = fd.UnitSquareMesh(8, 8)
        ctx = build_context(sp, mesh=mesh)
        ctx = build_forms(ctx, sp)
        bundle = ctx["cation_hydrolysis"]

        bundle.gamma_func.assign(0.025)        # inside [0, 0.10]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            new_gamma = clamp_gamma_to_max(ctx)
        assert new_gamma == pytest.approx(0.025, rel=1e-12)


@pytest.mark.slow
class TestCollectV10aRungDiagnostics:
    """``collect_v10a_rung_diagnostics`` returns the documented keys
    and finite values when called on a freshly-built ctx."""

    def test_diagnostics_keys_present(self):
        import firedrake as fd
        from Forward.bv_solver.cation_hydrolysis import (
            collect_v10a_rung_diagnostics,
        )
        from Forward.bv_solver.dispatch import build_context, build_forms

        sp = _build_v10a_sp(gamma_max_nondim=0.10)
        # Use the production graded rectangle mesh so the marker layout
        # matches what the form expects.
        from Forward.bv_solver import make_graded_rectangle_mesh
        mesh = make_graded_rectangle_mesh(
            Nx=4, Ny=8, beta=3.0, domain_height_hat=1.0,
        )
        ctx = build_context(sp, mesh=mesh)
        ctx = build_forms(ctx, sp)

        diag = collect_v10a_rung_diagnostics(ctx)
        # Schema check — these keys are the v10a rung_diag contract.
        required = {
            "gamma", "gamma_max", "theta",
            "lambda_hydrolysis", "k_hyd", "k_prot", "k_des",
            "delta_ohp_hat",
            "F0_avg", "forward_avg_no_k_hyd",
            "c_H_avg", "pka_shift_avg",
            "R_forward_capped",
            "denominator_constant", "denominator_kdes",
            "denominator_kprot", "denominator_cap",
            "denominator_total", "numerator",
            "sigma_S_C_per_m2", "sigma_S_counts_per_pm2",
        }
        missing = required - set(diag)
        assert not missing, f"diagnostics missing keys: {missing}"

        # Finite scalars on a freshly-built context.
        for k in (
            "gamma", "gamma_max", "theta",
            "F0_avg", "c_H_avg", "denominator_total",
        ):
            assert math.isfinite(float(diag[k])), (
                f"{k} not finite: {diag[k]}"
            )

        # Default state: Γ=0 ⇒ θ=0 ⇒ R_forward_capped = F0_avg.
        assert diag["gamma"] == pytest.approx(0.0, abs=1e-15)
        assert diag["theta"] == pytest.approx(0.0, abs=1e-15)
        assert diag["R_forward_capped"] == pytest.approx(
            diag["F0_avg"], rel=1e-12,
        )

    def test_diagnostics_empty_dict_when_no_bundle(self):
        """Disabled-feature path: helper returns empty dict, never
        raises."""
        from Forward.bv_solver.cation_hydrolysis import (
            collect_v10a_rung_diagnostics,
        )
        out = collect_v10a_rung_diagnostics({})
        assert out == {}
