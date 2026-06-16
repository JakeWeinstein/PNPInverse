"""Phase 6β v10a — V_kin selection rule unit tests.

Pure-Python tests on the ``select_v_kin`` helper in
``scripts/studies/phase6b_v10a_v_sweep_diagnostic.py``.  No Firedrake
required.

The selection rule was hardened via GPT critique session 32 (3
rounds, APPROVED).  These tests pin the GPT-approved invariants so
future drifts surface here rather than in a downstream Phase A.2
run.

Invariants under test
---------------------

* ``select_v_kin`` performs Stage 1 (estimator validity) → Stage 2
  (locked physics rule) and returns the *most informative* failure
  status per critique session 32 R3 nit #3 (``abort_to_v10c`` beats
  ``no_valid_stern_capacitance_sensitivity`` when both would apply).
* Locked filter set is the *exact three* from
  ``docs/phase6/PHASE_0_ACCEPTANCE_BUNDLE_LOCK_2026-05-10.md`` —
  σ_S<0, |cd|/I_lim_4e<0.9, R_2e/(R_2e+R_4e) ∈ [0.05, 0.95].
* Numerical-quality gating (one-sided slope agreement, σ-gap
  floors) is upstream of the locked rule, not a hidden 4th filter.
* ``σ_abs_min`` absolute floor cannot be silently bypassed by an
  adaptive median rescue.
* O₂-flux Levich ratio is dimensionally normalised by
  ``electrode_area_nondim``.
"""
from __future__ import annotations

import math
import os
import sys

import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _make_record(
    *,
    v_rhe: float,
    sigma_S_C_per_m2: float = -0.20,
    cd_mA_cm2: float = -2.0,
    R_2e_current_nondim: float = 0.5,
    R_4e_current_nondim: float = 0.5,
    perturbation_converged: bool = True,
    gamma: float = 0.01,
    k_des: float = 1.0,
    sigma_low_offset: float = -0.04,    # σ at C_S·(1−ε) shifts by this
    sigma_high_offset: float = +0.04,   # σ at C_S·(1+ε) shifts by this
    R_low_offset: float = -0.001,
    R_high_offset: float = +0.001,
    cs_unperturbed: float = 0.10,
    epsilon: float = 0.05,
    o2_flux_levich_ratio: float = 0.30,
) -> dict:
    """Build a synthetic per-V record with sensible defaults.

    Defaults give a record that passes all locked filters + primary
    estimator validity.  Override individual fields per test case.
    """
    R_net_base = k_des * gamma   # mass balance at λ=1
    return {
        "v_rhe": float(v_rhe),
        "sigma_S_C_per_m2": float(sigma_S_C_per_m2),
        "cd_mA_cm2": float(cd_mA_cm2),
        "R_2e_current_nondim": float(R_2e_current_nondim),
        "R_4e_current_nondim": float(R_4e_current_nondim),
        "gamma": float(gamma),
        "k_des": float(k_des),
        # Perturbation triple — driver-collected.
        "perturbation_converged": bool(perturbation_converged),
        "C_s_unperturbed": float(cs_unperturbed),
        "C_s_low": float(cs_unperturbed * (1.0 - epsilon)),
        "C_s_high": float(cs_unperturbed * (1.0 + epsilon)),
        "sigma_S_unperturbed": float(sigma_S_C_per_m2),
        "sigma_S_low": float(sigma_S_C_per_m2 + sigma_low_offset),
        "sigma_S_high": float(sigma_S_C_per_m2 + sigma_high_offset),
        "R_net_unperturbed": float(R_net_base),
        "R_net_low": float(R_net_base + R_low_offset),
        "R_net_high": float(R_net_base + R_high_offset),
        "epsilon": float(epsilon),
        "o2_flux_levich_ratio": float(o2_flux_levich_ratio),
    }


@pytest.fixture
def i_lim_4e():
    """Levich limit at l_eff = 16 µm with codebase constants.

    Independent of the driver's helper (we want to test the rule, not
    the rule's coupling to _bv_common).
    """
    return 5.50    # mA/cm² — matches _i_lim_4e_mA_cm2(16e-6) within rounding


# ===========================================================================
# Stage 0 — abort_to_v10c precedence
# ===========================================================================


class TestAbortToV10c:
    """No V with σ_S < 0 ⇒ abort_to_v10c=True.  Always fires first."""

    def test_all_anodic_aborts(self, i_lim_4e):
        from scripts.studies.drivers.phase6b_v10a_v_sweep_diagnostic import (
            select_v_kin,
        )
        records = [
            _make_record(v_rhe=0.55, sigma_S_C_per_m2=+0.30),
            _make_record(v_rhe=0.30, sigma_S_C_per_m2=+0.20),
            _make_record(v_rhe=0.10, sigma_S_C_per_m2=+0.10),
        ]
        decision = select_v_kin(records, i_lim_4e_mA_cm2=i_lim_4e)
        assert decision.abort_to_v10c is True
        assert decision.v_kin is None
        assert decision.score is None
        # Lower-precedence flags should NOT also be set.
        assert decision.no_valid_stern_capacitance_sensitivity is False
        assert decision.no_candidate_passed_locked_rule is False

    def test_abort_wins_over_estimator_failure(self, i_lim_4e):
        """R3 nit #3: abort_to_v10c is more informative than the
        estimator-failure status when both could apply."""
        from scripts.studies.drivers.phase6b_v10a_v_sweep_diagnostic import (
            select_v_kin,
        )
        # All σ_S > 0 AND all estimators bad.
        records = [
            _make_record(
                v_rhe=0.55, sigma_S_C_per_m2=+0.30,
                perturbation_converged=False,
            ),
            _make_record(
                v_rhe=0.30, sigma_S_C_per_m2=+0.20,
                perturbation_converged=False,
            ),
        ]
        decision = select_v_kin(records, i_lim_4e_mA_cm2=i_lim_4e)
        assert decision.abort_to_v10c is True
        # Critical: do NOT also flag estimator failure when abort wins.
        assert decision.no_valid_stern_capacitance_sensitivity is False


# ===========================================================================
# Stage 1 — estimator-validity gating
# ===========================================================================


class TestEstimatorValidityGate:
    """Stage 1 must keep numerical-quality issues upstream of the
    locked rule."""

    def test_no_valid_estimator_fires_when_some_sigma_neg(self, i_lim_4e):
        from scripts.studies.drivers.phase6b_v10a_v_sweep_diagnostic import (
            select_v_kin,
        )
        # σ_S < 0 at all V but perturbation never converged.
        records = [
            _make_record(
                v_rhe=-0.10, sigma_S_C_per_m2=-0.10,
                perturbation_converged=False,
            ),
            _make_record(
                v_rhe=-0.20, sigma_S_C_per_m2=-0.20,
                perturbation_converged=False,
            ),
        ]
        decision = select_v_kin(records, i_lim_4e_mA_cm2=i_lim_4e)
        assert decision.no_valid_stern_capacitance_sensitivity is True
        assert decision.abort_to_v10c is False
        assert decision.v_kin is None

    def test_sigma_abs_min_floor_cannot_be_bypassed(self, i_lim_4e):
        """R2 issue #1: even if every V's |σ_+ − σ_−| is tiny, the
        absolute floor (1e-4 C/m²) keeps unidentifiable estimators
        out of the valid set."""
        from scripts.studies.drivers.phase6b_v10a_v_sweep_diagnostic import (
            select_v_kin, SIGMA_ABS_MIN_C_PER_M2,
        )
        # σ-gap of 1e-7 << σ_abs_min = 1e-4.
        records = [
            _make_record(
                v_rhe=-0.10, sigma_S_C_per_m2=-0.20,
                sigma_low_offset=-5e-8, sigma_high_offset=+5e-8,
            ),
            _make_record(
                v_rhe=-0.20, sigma_S_C_per_m2=-0.30,
                sigma_low_offset=-5e-8, sigma_high_offset=+5e-8,
            ),
        ]
        decision = select_v_kin(records, i_lim_4e_mA_cm2=i_lim_4e)
        assert decision.no_valid_stern_capacitance_sensitivity is True

    def test_per_side_floor_catches_lopsided_perturbation(self, i_lim_4e):
        """R2 issue #2: if one side has σ_+ ≈ σ_0 even though
        |σ_+ − σ_−| is healthy, the per-side floor must reject."""
        from scripts.studies.drivers.phase6b_v10a_v_sweep_diagnostic import (
            select_v_kin,
        )
        # σ_low - σ_0 = -0.20 (huge), σ_high - σ_0 = +1e-7 (tiny).
        records = [
            _make_record(
                v_rhe=-0.10, sigma_S_C_per_m2=-0.20,
                sigma_low_offset=-0.20, sigma_high_offset=+1e-7,
            ),
            _make_record(
                v_rhe=-0.20, sigma_S_C_per_m2=-0.30,
                sigma_low_offset=-0.20, sigma_high_offset=+1e-7,
            ),
        ]
        decision = select_v_kin(records, i_lim_4e_mA_cm2=i_lim_4e)
        assert decision.no_valid_stern_capacitance_sensitivity is True

    def test_one_sided_disagreement_excludes_from_primary(self, i_lim_4e):
        """A V with wildly different S_+ vs S_- gets pushed out of
        the primary set (one_sided_disagreement > 0.25)."""
        from scripts.studies.drivers.phase6b_v10a_v_sweep_diagnostic import (
            select_v_kin,
        )
        # Two V — one with disagreement ≈ 0.10 (clean), one ≈ 0.80
        # (noisy).  Use sigma_low_offset = sigma_high_offset = ε so
        # S_± are well-defined (not zero).
        records = [
            _make_record(
                v_rhe=-0.10, sigma_S_C_per_m2=-0.10,
                sigma_low_offset=-0.04, sigma_high_offset=+0.04,
                R_low_offset=-0.002, R_high_offset=+0.0024,  # ratio diff ~10 %
            ),
            _make_record(
                v_rhe=-0.20, sigma_S_C_per_m2=-0.20,
                sigma_low_offset=-0.04, sigma_high_offset=+0.04,
                R_low_offset=-0.001, R_high_offset=+0.005,   # very different
            ),
        ]
        decision = select_v_kin(records, i_lim_4e_mA_cm2=i_lim_4e)
        # Clean V should be picked.
        assert decision.v_kin == pytest.approx(-0.10)
        # Noisy V's primary_valid should be False.
        noisy = [
            r for r in decision.per_v_decisions
            if r["v_rhe"] == pytest.approx(-0.20)
        ][0]
        assert noisy["primary_valid"] is False


# ===========================================================================
# Stage 2 — locked physics rule
# ===========================================================================


class TestLockedRuleHappyPath:
    """Three V with all filters passing; picker selects argmax score."""

    def test_argmax_score_picked(self, i_lim_4e):
        from scripts.studies.drivers.phase6b_v10a_v_sweep_diagnostic import (
            select_v_kin,
        )
        # Each V has different R_low/R_high asymmetry → different
        # slope magnitudes.  Middle V has the largest slope.
        records = [
            _make_record(
                v_rhe=-0.10, sigma_S_C_per_m2=-0.10,
                R_low_offset=-0.0005, R_high_offset=+0.0005,   # |slope| small
            ),
            _make_record(
                v_rhe=-0.30, sigma_S_C_per_m2=-0.30,
                R_low_offset=-0.002, R_high_offset=+0.002,     # |slope| large
            ),
            _make_record(
                v_rhe=-0.50, sigma_S_C_per_m2=-0.40,
                R_low_offset=-0.0005, R_high_offset=+0.0005,   # |slope| small
            ),
        ]
        decision = select_v_kin(records, i_lim_4e_mA_cm2=i_lim_4e)
        assert decision.abort_to_v10c is False
        assert decision.no_valid_stern_capacitance_sensitivity is False
        assert decision.no_candidate_passed_locked_rule is False
        assert decision.fallback_used is False
        assert decision.v_kin == pytest.approx(-0.30)
        assert decision.score is not None and decision.score > 0.0

    def test_locked_filter_booleans_per_v(self, i_lim_4e):
        """Each per-V record gets explicit locked_*_filter_passed
        booleans (R2 issue #6)."""
        from scripts.studies.drivers.phase6b_v10a_v_sweep_diagnostic import (
            select_v_kin,
        )
        records = [_make_record(v_rhe=-0.20)]
        decision = select_v_kin(records, i_lim_4e_mA_cm2=i_lim_4e)
        rec = decision.per_v_decisions[0]
        # Explicit names — NOT an ambiguous "locked_filter_passed".
        for key in (
            "locked_sigma_neg_filter_passed",
            "locked_current_filter_passed",
            "locked_branch_filter_passed",
            "locked_three_filters_passed",
        ):
            assert key in rec, f"missing per-V boolean: {key}"
            assert isinstance(rec[key], bool)
        assert rec["locked_three_filters_passed"] is True


class TestLockedRuleFilters:
    """Each individual locked filter eliminates a candidate as designed."""

    def test_branch_filter_eliminates_outside_window(self, i_lim_4e):
        """V with R_2e/(R_2e+R_4e) outside [0.05, 0.95] should drop
        from the primary set; argmax falls to a different V."""
        from scripts.studies.drivers.phase6b_v10a_v_sweep_diagnostic import (
            select_v_kin,
        )
        # V at -0.30 has the largest |slope| but pure-4e selectivity
        # (branch ratio = 0).  V at -0.10 is mixed and picks up.
        records = [
            _make_record(
                v_rhe=-0.10, sigma_S_C_per_m2=-0.10,
                R_low_offset=-0.0005, R_high_offset=+0.0005,
            ),
            _make_record(
                v_rhe=-0.30, sigma_S_C_per_m2=-0.30,
                R_2e_current_nondim=0.0, R_4e_current_nondim=1.0,
                R_low_offset=-0.002, R_high_offset=+0.002,
            ),
        ]
        decision = select_v_kin(records, i_lim_4e_mA_cm2=i_lim_4e)
        # Pure-4e V failed branch filter; mixed V at -0.10 wins.
        assert decision.v_kin == pytest.approx(-0.10)
        assert decision.fallback_used is False
        # Verify the eliminated V's branch boolean is False.
        eliminated = [
            r for r in decision.per_v_decisions
            if r["v_rhe"] == pytest.approx(-0.30)
        ][0]
        assert eliminated["locked_branch_filter_passed"] is False

    def test_current_filter_eliminates_levich_plateau(self, i_lim_4e):
        """V with |cd|/I_lim_4e ≥ 0.9 fails the current filter."""
        from scripts.studies.drivers.phase6b_v10a_v_sweep_diagnostic import (
            select_v_kin,
        )
        # V at -0.30 hits the cathodic plateau (|cd| = 5.0 mA/cm² vs
        # I_lim_4e = 5.5 → 91% > 90%).
        records = [
            _make_record(
                v_rhe=-0.10, sigma_S_C_per_m2=-0.10, cd_mA_cm2=-2.0,
                R_low_offset=-0.0005, R_high_offset=+0.0005,
            ),
            _make_record(
                v_rhe=-0.30, sigma_S_C_per_m2=-0.30, cd_mA_cm2=-5.0,
                R_low_offset=-0.002, R_high_offset=+0.002,
            ),
        ]
        decision = select_v_kin(records, i_lim_4e_mA_cm2=i_lim_4e)
        # Plateau V dropped; non-plateau V at -0.10 wins.
        assert decision.v_kin == pytest.approx(-0.10)


class TestFallbackPath:
    """Drop branch filter when no V passes the primary set."""

    def test_fallback_used_when_all_branch_filtered(self, i_lim_4e):
        """All V have pure-4e selectivity → primary set empty →
        fallback drops branch filter, picks argmax in fallback set."""
        from scripts.studies.drivers.phase6b_v10a_v_sweep_diagnostic import (
            select_v_kin,
        )
        records = [
            _make_record(
                v_rhe=-0.10, sigma_S_C_per_m2=-0.10,
                R_2e_current_nondim=0.0, R_4e_current_nondim=1.0,
                R_low_offset=-0.0005, R_high_offset=+0.0005,
            ),
            _make_record(
                v_rhe=-0.30, sigma_S_C_per_m2=-0.30,
                R_2e_current_nondim=0.0, R_4e_current_nondim=1.0,
                R_low_offset=-0.002, R_high_offset=+0.002,
            ),
        ]
        decision = select_v_kin(records, i_lim_4e_mA_cm2=i_lim_4e)
        assert decision.v_kin == pytest.approx(-0.30)
        assert decision.fallback_used is True

    def test_no_candidate_passes_locked_rule(self, i_lim_4e):
        """Some V have σ_S < 0 (so abort doesn't fire), perturbation
        valid, but the current filter eliminates them all even after
        dropping branch."""
        from scripts.studies.drivers.phase6b_v10a_v_sweep_diagnostic import (
            select_v_kin,
        )
        records = [
            _make_record(
                v_rhe=-0.20, sigma_S_C_per_m2=-0.20,
                cd_mA_cm2=-5.5,    # AT Levich; |cd|/I_lim_4e = 1.0 > 0.9
            ),
            _make_record(
                v_rhe=-0.30, sigma_S_C_per_m2=-0.30,
                cd_mA_cm2=-5.5,
            ),
        ]
        decision = select_v_kin(records, i_lim_4e_mA_cm2=i_lim_4e)
        assert decision.no_candidate_passed_locked_rule is True
        assert decision.abort_to_v10c is False
        assert decision.no_valid_stern_capacitance_sensitivity is False
        assert decision.v_kin is None


# ===========================================================================
# Levich helper + O₂ flux ratio
# ===========================================================================


class TestLevichHelper:

    def test_i_lim_4e_at_16um(self):
        from scripts.studies.drivers.phase6b_v10a_v_sweep_diagnostic import (
            _i_lim_4e_mA_cm2,
        )
        # 4 · 96485 · 1.9e-9 · 1.2 / 16e-6 · 0.1 ≈ 5.50 mA/cm²
        assert _i_lim_4e_mA_cm2(16e-6) == pytest.approx(5.50, rel=2e-2)

    def test_i_lim_4e_inverse_in_l_eff(self):
        from scripts.studies.drivers.phase6b_v10a_v_sweep_diagnostic import (
            _i_lim_4e_mA_cm2,
        )
        # Doubling l_eff halves I_lim.
        assert _i_lim_4e_mA_cm2(32e-6) == pytest.approx(
            _i_lim_4e_mA_cm2(16e-6) / 2.0, rel=1e-12,
        )

    def test_does_not_use_water_lit_d_o2(self):
        """R1 issue #4: the helper must NOT silently use 2.18e-9
        (water literature D_O2).  Verify by checking the value at
        a known l_eff against the codebase D_O2 = 1.9e-9."""
        from scripts.studies.drivers.phase6b_v10a_v_sweep_diagnostic import (
            _i_lim_4e_mA_cm2,
        )
        # If D_O2 = 2.18e-9 the result would be ~6.30 mA/cm²;
        # with 1.9e-9 it's ~5.50.  A 0.10 mA/cm² tolerance separates
        # them cleanly.
        val = _i_lim_4e_mA_cm2(16e-6)
        assert val < 5.80, (
            f"I_lim_4e = {val:.3f}; D_O2 may have drifted to water-lit value"
        )


class TestO2FluxLevichRatio:
    """Dimensional sanity per R2 issue #5."""

    def test_basic_dimensional_invariance(self):
        """Doubling electrode_area halves the ratio."""
        from scripts.studies.drivers.phase6b_v10a_v_sweep_diagnostic import (
            _compute_o2_flux_levich_ratio,
        )
        kwargs = dict(
            R_2e_current_nondim=0.5,
            R_4e_current_nondim=0.5,
            domain_height_hat=0.16,
        )
        r1 = _compute_o2_flux_levich_ratio(electrode_area_nondim=1.0, **kwargs)
        r2 = _compute_o2_flux_levich_ratio(electrode_area_nondim=2.0, **kwargs)
        assert r1 == pytest.approx(2.0 * r2, rel=1e-12)

    def test_ratio_in_zero_one_at_levich_limit(self):
        """At full O₂-flux Levich limit, ratio = 1 regardless of
        branch selectivity (the whole point of this indicator)."""
        from scripts.studies.drivers.phase6b_v10a_v_sweep_diagnostic import (
            _compute_o2_flux_levich_ratio,
        )
        # At Levich: total O₂ consumption rate per area = D·c/l = 1/0.16 = 6.25.
        # Pure 2e: R_2e_current_nondim = 6.25, R_4e = 0
        # Pure 4e: R_2e = 0, R_4e = 6.25
        # 50/50: R_2e = R_4e = 3.125
        ratio_pure_2e = _compute_o2_flux_levich_ratio(
            R_2e_current_nondim=6.25, R_4e_current_nondim=0.0,
            electrode_area_nondim=1.0, domain_height_hat=0.16,
        )
        ratio_pure_4e = _compute_o2_flux_levich_ratio(
            R_2e_current_nondim=0.0, R_4e_current_nondim=6.25,
            electrode_area_nondim=1.0, domain_height_hat=0.16,
        )
        ratio_mixed = _compute_o2_flux_levich_ratio(
            R_2e_current_nondim=3.125, R_4e_current_nondim=3.125,
            electrode_area_nondim=1.0, domain_height_hat=0.16,
        )
        for r in (ratio_pure_2e, ratio_pure_4e, ratio_mixed):
            assert r == pytest.approx(1.0, rel=1e-12)

    def test_handles_zero_area(self):
        """Defensive: bad electrode_area returns 0, doesn't crash."""
        from scripts.studies.drivers.phase6b_v10a_v_sweep_diagnostic import (
            _compute_o2_flux_levich_ratio,
        )
        assert _compute_o2_flux_levich_ratio(
            R_2e_current_nondim=1.0, R_4e_current_nondim=1.0,
            electrode_area_nondim=0.0, domain_height_hat=0.16,
        ) == 0.0


class TestLevichAsymmetryFlag:
    """The locked-rule asymmetry warning (R1 issue #5) fires when the
    locked current filter passes BUT the branch-independent O₂-flux
    ratio shows transport saturation."""

    def test_warning_fires_in_2e_plateau(self, i_lim_4e):
        """Pure-2e plateau: |cd|/I_lim_4e = 0.5 (passes 0.9 filter)
        but o2_flux_levich_ratio = 1.0 (transport-limited)."""
        from scripts.studies.drivers.phase6b_v10a_v_sweep_diagnostic import (
            select_v_kin,
        )
        # Pure-2e at Levich: cd contribution = R_2e (1.0 weight) + 0
        # = R_2e_nondim.  I_SCALE depends on units; just set cd to
        # 0.5·I_lim_4e and o2_flux_levich_ratio = 1.0.
        rec = _make_record(
            v_rhe=-0.20, sigma_S_C_per_m2=-0.20,
            cd_mA_cm2=-i_lim_4e * 0.5,            # pure-2e plateau magnitude
            R_2e_current_nondim=6.25, R_4e_current_nondim=0.0,
            o2_flux_levich_ratio=1.0,
        )
        # Branch filter rejects pure-2e (R_2e / total = 1.0 > 0.95),
        # so this falls to fallback; we just need the per-V flag.
        decision = select_v_kin([rec], i_lim_4e_mA_cm2=i_lim_4e)
        per_v = decision.per_v_decisions[0]
        # locked_current_filter passes (50% < 90%)
        assert per_v["locked_current_filter_passed"] is True
        # warning fires
        assert (
            per_v["locked_current_filter_passes_but_o2_transport_limited"]
            is True
        )


# ===========================================================================
# Auxiliary: VKinDecision shape + JSON round-trip
# ===========================================================================


class TestVKinDecisionShape:

    def test_dataclass_frozen(self, i_lim_4e):
        from scripts.studies.drivers.phase6b_v10a_v_sweep_diagnostic import (
            VKinDecision,
        )
        d = VKinDecision(v_kin=-0.30, score=1.23)
        with pytest.raises(Exception):
            d.v_kin = 0.0    # type: ignore[misc]

    def test_to_json_round_trips(self, i_lim_4e):
        import json
        from scripts.studies.drivers.phase6b_v10a_v_sweep_diagnostic import (
            select_v_kin,
        )
        records = [_make_record(v_rhe=-0.30, sigma_S_C_per_m2=-0.30)]
        decision = select_v_kin(records, i_lim_4e_mA_cm2=i_lim_4e)
        # Should serialise (per_v_decisions contains plain floats / bools).
        payload = decision.to_json()
        s = json.dumps(payload, default=str)
        loaded = json.loads(s)
        assert loaded["v_kin"] == pytest.approx(-0.30)
        assert loaded["fallback_used"] is False


# ===========================================================================
# FD informational column (R1 issue #3: never used as a filter)
# ===========================================================================


class TestFDInformational:

    def test_fd_attached_central_difference(self):
        from scripts.studies.drivers.phase6b_v10a_v_sweep_diagnostic import (
            attach_fd_sensitivity,
        )
        records = [
            {"v_rhe": -0.10, "sigma_S_C_per_m2": -0.10, "R_net_unperturbed": 0.005},
            {"v_rhe": -0.20, "sigma_S_C_per_m2": -0.20, "R_net_unperturbed": 0.010},
            {"v_rhe": -0.30, "sigma_S_C_per_m2": -0.30, "R_net_unperturbed": 0.015},
        ]
        attach_fd_sensitivity(records)
        # Middle V: central difference (0.015 − 0.005) / (−0.30 − (−0.10))
        #         = 0.010 / (−0.20) = −0.05.
        assert records[1]["dRnet_dsigma_along_voltage"] == pytest.approx(-0.05)

    def test_fd_handles_degenerate_step(self):
        """When σ_S is identical at neighbours, FD should be None,
        not raise."""
        from scripts.studies.drivers.phase6b_v10a_v_sweep_diagnostic import (
            attach_fd_sensitivity,
        )
        records = [
            {"v_rhe": -0.10, "sigma_S_C_per_m2": -0.20, "R_net_unperturbed": 0.005},
            {"v_rhe": -0.20, "sigma_S_C_per_m2": -0.20, "R_net_unperturbed": 0.010},
        ]
        attach_fd_sensitivity(records)
        assert all(
            r["dRnet_dsigma_along_voltage"] is None for r in records
        )

    def test_fd_disagreement_is_NOT_a_filter(self, i_lim_4e):
        """R1 issue #3: even huge FD-vs-perturbation disagreement
        must NOT exclude a V from candidacy."""
        from scripts.studies.drivers.phase6b_v10a_v_sweep_diagnostic import (
            select_v_kin,
        )
        records = [_make_record(v_rhe=-0.30)]
        # Inject a wildly different FD value.
        records[0]["dRnet_dsigma_along_voltage"] = 1e6
        decision = select_v_kin(records, i_lim_4e_mA_cm2=i_lim_4e)
        # V still selected; path_mismatch_relative logged but no filter.
        assert decision.v_kin == pytest.approx(-0.30)
        per_v = decision.per_v_decisions[0]
        assert per_v["path_mismatch_relative"] is not None
        assert per_v["primary_valid"] is True


# ===========================================================================
# Log-step denominator (R3 nit #1: log(1+ε) − log(1−ε), not 2ε)
# ===========================================================================


class TestLogStepDenominator:

    def test_exact_log_form(self, i_lim_4e):
        """R3 nit #1: log-step denominator uses exact form."""
        from scripts.studies.drivers.phase6b_v10a_v_sweep_diagnostic import (
            select_v_kin,
        )
        records = [_make_record(v_rhe=-0.30, epsilon=0.05)]
        decision = select_v_kin(records, i_lim_4e_mA_cm2=i_lim_4e)
        rec = decision.per_v_decisions[0]
        # log(1.05/1.00) − log(0.95/1.00) ≈ 0.0488 − (−0.0513) ≈ 0.1001
        expected_log_step = math.log(1.05) - math.log(0.95)
        assert rec["log_step_denominator"] == pytest.approx(
            expected_log_step, rel=1e-12,
        )
        # The naive 2ε would give exactly 0.10 — confirm we are NOT
        # using that.
        assert rec["log_step_denominator"] != pytest.approx(0.10, abs=1e-6)


# ===========================================================================
# v10a' driver wiring — k0_r4e_factor flag (no Firedrake required)
# ===========================================================================


class TestK0R4eFactorDriver:
    """The ``--k0-r4e-factor`` flag must scale only the 4e-branch ``k0``
    in the SolverParams the driver feeds the solver.  The 2e branch and
    nested mutable structures (cathodic_conc_factors, stoichiometry)
    must be deep-copied so the module-level ``PARALLEL_2E_4E_REACTIONS_4SP``
    constant is never aliased / mutated across calls.
    """

    def test_factor_one_preserves_K0_HAT_R4E(self):
        from scripts.studies.drivers.phase6b_v10a_v_sweep_diagnostic import (
            _scale_k0_r4e_in_reactions,
        )
        from scripts._bv_common import (
            K0_HAT_R2E, K0_HAT_R4E, PARALLEL_2E_4E_REACTIONS_4SP,
        )
        rescaled = _scale_k0_r4e_in_reactions(
            PARALLEL_2E_4E_REACTIONS_4SP, factor=1.0,
        )
        assert rescaled[0]["k0"] == pytest.approx(float(K0_HAT_R2E))
        assert rescaled[1]["k0"] == pytest.approx(float(K0_HAT_R4E))

    def test_factor_1e_minus_18_scales_only_4e_branch(self):
        from scripts.studies.drivers.phase6b_v10a_v_sweep_diagnostic import (
            _scale_k0_r4e_in_reactions,
        )
        from scripts._bv_common import (
            K0_HAT_R2E, K0_HAT_R4E, PARALLEL_2E_4E_REACTIONS_4SP,
        )
        factor = 1e-18
        rescaled = _scale_k0_r4e_in_reactions(
            PARALLEL_2E_4E_REACTIONS_4SP, factor=factor,
        )
        # 2e untouched.
        assert rescaled[0]["k0"] == pytest.approx(float(K0_HAT_R2E))
        # 4e scaled.
        assert rescaled[1]["k0"] == pytest.approx(
            float(K0_HAT_R4E) * factor, rel=1e-12,
        )

    def test_factor_does_not_alias_module_constant(self):
        """Critique R3 #5 hygiene: rescaling once, then again with a
        different factor, must rescale from the ORIGINAL k0, not the
        previously-rescaled value.  Catches aliasing into the module
        constant."""
        from scripts.studies.drivers.phase6b_v10a_v_sweep_diagnostic import (
            _scale_k0_r4e_in_reactions,
        )
        from scripts._bv_common import (
            K0_HAT_R4E, PARALLEL_2E_4E_REACTIONS_4SP,
        )
        rescaled_1 = _scale_k0_r4e_in_reactions(
            PARALLEL_2E_4E_REACTIONS_4SP, factor=1e-18,
        )
        rescaled_2 = _scale_k0_r4e_in_reactions(
            PARALLEL_2E_4E_REACTIONS_4SP, factor=1e-10,
        )
        assert rescaled_1[1]["k0"] == pytest.approx(
            float(K0_HAT_R4E) * 1e-18, rel=1e-12,
        )
        assert rescaled_2[1]["k0"] == pytest.approx(
            float(K0_HAT_R4E) * 1e-10, rel=1e-12,
        )
        # Module constant is unchanged after both rescalings.
        assert PARALLEL_2E_4E_REACTIONS_4SP[1]["k0"] == pytest.approx(
            float(K0_HAT_R4E), rel=1e-12,
        )

    def test_nested_mutables_deep_copied(self):
        """The `cathodic_conc_factors` (list of dicts) and
        `stoichiometry` (list) must be deep-copied so a downstream
        mutation can't leak into the module constant."""
        from scripts.studies.drivers.phase6b_v10a_v_sweep_diagnostic import (
            _scale_k0_r4e_in_reactions,
        )
        from scripts._bv_common import PARALLEL_2E_4E_REACTIONS_4SP
        rescaled = _scale_k0_r4e_in_reactions(
            PARALLEL_2E_4E_REACTIONS_4SP, factor=1e-12,
        )
        for j in (0, 1):
            assert rescaled[j] is not PARALLEL_2E_4E_REACTIONS_4SP[j]
            if "stoichiometry" in PARALLEL_2E_4E_REACTIONS_4SP[j]:
                assert (
                    rescaled[j]["stoichiometry"]
                    is not PARALLEL_2E_4E_REACTIONS_4SP[j]["stoichiometry"]
                )
            if "cathodic_conc_factors" in PARALLEL_2E_4E_REACTIONS_4SP[j]:
                rescaled_ccfs = rescaled[j]["cathodic_conc_factors"]
                module_ccfs = PARALLEL_2E_4E_REACTIONS_4SP[j][
                    "cathodic_conc_factors"
                ]
                assert rescaled_ccfs is not module_ccfs
                # Each entry too — they're dicts.
                for r_ccf, m_ccf in zip(rescaled_ccfs, module_ccfs):
                    assert r_ccf is not m_ccf

    def test_cli_parses_scientific_notation(self):
        """``argparse`` ``type=float`` handles ``1e-18`` natively."""
        from scripts.studies.drivers.phase6b_v10a_v_sweep_diagnostic import (
            _parse_args,
        )
        args = _parse_args(["--k0-r4e-factor", "1e-18", "--no-plot"])
        assert args.k0_r4e_factor == pytest.approx(1e-18, rel=1e-12)
        # Default still works.
        args_default = _parse_args(["--no-plot"])
        assert args_default.k0_r4e_factor == pytest.approx(1.0)


if __name__ == "__main__":                                # pragma: no cover
    import sys as _sys
    _sys.exit(pytest.main([__file__, "-v"]))
