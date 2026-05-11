"""Phase 6β v10a — Phase A.2 driver unit tests.

Pure-Python tests on the helpers in
``scripts/studies/phase6b_v10a_phase_A2_v_kin.py``.  No Firedrake
required.

The plan (``~/.claude/plans/phase6b-v10a-phase-A2-v-kin.md``) was
hardened via GPT critique session 34 (4 rounds, APPROVED).  These
tests pin the invariants that came out of that critique so future
drift surfaces here rather than at a downstream ~30-min wall-time
run.

Invariants under test
---------------------

* ``classify_picard_status`` returns each of the 7 documented statuses
  on the right synthetic input (snes_failed, no_iters, single_iter,
  converged, early_break, converged_at_iter_cap, iter_cap_hit_unconverged).
* ``single_v_selectivity_gap_pp`` returns 0 inside the deck band and
  signed distance outside (positive when below, negative when above).
* ``classify_no_route_cause`` walks the 5 categories in order and
  picks the most-specific failure mode that applies.
* ``select_k_hyd_route`` enforces all five gates (theta, slope, transport,
  picard, mass-balance) and prefers the highest k_hyd that passes.
* ``_parse_args`` parses ``--k-hyd-grid``, ``--v-kin``, and
  ``--k0-r4e-factor`` correctly; defaults match the locked 10-point grid.
* ``test_exception_message_prefixes_match_solver`` (R4 #5) — solver
  source contains the exact three exception-text prefixes the driver
  decodes; drift in the solver surfaces in CI rather than at runtime.
"""
from __future__ import annotations

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


def _lam1(
    *,
    snes_converged: bool = True,
    picard_status: str = "converged",
    theta: float = 0.96,
    gamma: float = 0.0455,
    gamma_max: float = 0.047,
    k_des: float = 1.0,
    o2_flux_levich_ratio: float = 0.30,
    mass_balance_residual_rel: float = 1e-4,
    H2O2_selectivity_pct: float = 30.0,
    F0_decomposition: dict = None,
    R_2e_current_nondim: float = 0.5,
    R_4e_current_nondim: float = 0.5,
    extra: dict = None,
) -> dict:
    """Build a synthetic λ=1.0 rung dict that passes every gate by default.

    Override individual fields per test case.
    """
    d = {
        "lambda_hydrolysis": 1.0,
        "snes_converged": snes_converged,
        "picard_status": picard_status,
        "theta": theta,
        "gamma": gamma,
        "gamma_max": gamma_max,
        "k_des": k_des,
        "o2_flux_levich_ratio": o2_flux_levich_ratio,
        "mass_balance_residual_rel": mass_balance_residual_rel,
        "H2O2_selectivity_pct": H2O2_selectivity_pct,
        "R_2e_current_nondim": R_2e_current_nondim,
        "R_4e_current_nondim": R_4e_current_nondim,
        "F0_decomposition": F0_decomposition or {
            "amplification_from_singh": 1.0,
            "amplification_from_c_K": 1.7,
        },
    }
    if extra:
        d.update(extra)
    return d


def _per_k_hyd(
    *,
    k_hyd_target: float,
    lam1: dict,
    other_rungs: list = None,
    exception_phase: str = None,
) -> dict:
    """Build a synthetic per-k_hyd record."""
    rungs = list(other_rungs or [])
    rungs.append(lam1)
    return {
        "k_hyd_target": float(k_hyd_target),
        "ladder_converged": lam1.get("snes_converged", False),
        "exception_phase": exception_phase,
        "exception": None,
        "rungs": rungs,
        "partial_rungs": [],
    }


# ===========================================================================
# classify_picard_status
# ===========================================================================


class TestClassifyPicardStatus:
    """All 7 documented Picard statuses per plan §Implementation notes."""

    def test_snes_failed_overrides_everything(self):
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import (
            classify_picard_status,
        )
        # Even with a perfectly-converged history, snes=False → snes_failed.
        assert classify_picard_status(
            [0.04, 0.04, 0.04], snes_converged=False,
        ) == "snes_failed"

    def test_no_iters(self):
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import (
            classify_picard_status,
        )
        assert classify_picard_status([], snes_converged=True) == "no_iters"

    def test_single_iter(self):
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import (
            classify_picard_status,
        )
        assert classify_picard_status(
            [0.045], snes_converged=True,
        ) == "single_iter"

    def test_converged_two_iters_tight(self):
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import (
            classify_picard_status,
        )
        # last_rel = |0.045 - 0.045000001| / max(...) ≈ 2.2e-8 < 1e-4
        assert classify_picard_status(
            [0.045, 0.045000001], snes_converged=True,
        ) == "converged"

    def test_early_break_two_iters_loose(self):
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import (
            classify_picard_status,
        )
        # last_rel = |0.04 - 0.05| / 0.05 = 0.2 >= 1e-4, n<8 → early_break
        assert classify_picard_status(
            [0.04, 0.05], snes_converged=True,
        ) == "early_break"

    def test_converged_at_iter_cap(self):
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import (
            classify_picard_status,
        )
        # n=8 and last_rel<1e-4
        history = [
            0.04, 0.041, 0.0405, 0.04055, 0.04054,
            0.040545, 0.0405455, 0.04054555,
        ]
        # last_rel = |0.04054555 - 0.0405455| / 0.04054555 ≈ 1.2e-6 < 1e-4
        assert classify_picard_status(
            history, snes_converged=True,
        ) == "converged_at_iter_cap"

    def test_iter_cap_hit_unconverged(self):
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import (
            classify_picard_status,
        )
        # n=8, last two differ by >1e-4 relative.
        history = [0.04, 0.05, 0.04, 0.05, 0.04, 0.05, 0.04, 0.05]
        # last_rel = |0.05 - 0.04| / 0.05 = 0.2 ≥ 1e-4
        assert classify_picard_status(
            history, snes_converged=True,
        ) == "iter_cap_hit_unconverged"


# ===========================================================================
# single_v_selectivity_gap_pp
# ===========================================================================


class TestSingleVSelectivityGapPp:
    """R4 #1 — renamed from selectivity_gap_pp; signed distance to deck band."""

    def test_inside_band_returns_zero(self):
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import (
            single_v_selectivity_gap_pp,
        )
        assert single_v_selectivity_gap_pp(30.0) == 0.0
        assert single_v_selectivity_gap_pp(25.0) == 0.0  # boundary
        assert single_v_selectivity_gap_pp(50.0) == 0.0  # boundary

    def test_below_band_positive(self):
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import (
            single_v_selectivity_gap_pp,
        )
        assert single_v_selectivity_gap_pp(20.0) == pytest.approx(+5.0)
        assert single_v_selectivity_gap_pp(0.0) == pytest.approx(+25.0)

    def test_above_band_negative(self):
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import (
            single_v_selectivity_gap_pp,
        )
        assert single_v_selectivity_gap_pp(60.0) == pytest.approx(-10.0)
        assert single_v_selectivity_gap_pp(100.0) == pytest.approx(-50.0)

    def test_custom_band(self):
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import (
            single_v_selectivity_gap_pp,
        )
        assert single_v_selectivity_gap_pp(15.0, deck_band=(20.0, 40.0)) == (
            pytest.approx(+5.0)
        )


# ===========================================================================
# lambda1_record
# ===========================================================================


class TestLambda1Record:
    def test_finds_lambda_one_rung(self):
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import lambda1_record
        rec = {"rungs": [
            {"lambda_hydrolysis": 0.5, "marker": "mid"},
            {"lambda_hydrolysis": 1.0, "marker": "final"},
        ]}
        result = lambda1_record(rec)
        assert result is not None
        assert result["marker"] == "final"

    def test_returns_none_when_missing(self):
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import lambda1_record
        rec = {"rungs": [
            {"lambda_hydrolysis": 0.5},
            {"lambda_hydrolysis": 0.75},
        ]}
        assert lambda1_record(rec) is None

    def test_empty_rungs(self):
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import lambda1_record
        assert lambda1_record({"rungs": []}) is None


# ===========================================================================
# compute_mass_balance_residual_rel
# ===========================================================================


class TestMassBalanceResidualRel:
    def test_closed_form_identity_zero(self):
        """At SS the closed-form solver makes the residual identically 0."""
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import (
            compute_mass_balance_residual_rel,
        )
        # F0=1, λ=1, k_des=1, k_prot·c_H/δ=0.5, F0/Γ_max=10.
        # γ_ss = λ·F0 / [(1−λ) + λ·k_des + λ·denom_kprot + λ·F0/Γ_max]
        # With our shorthand denominator_kprot, F0_capped at θ:
        # We can construct a record where the identity holds exactly.
        rung = {
            "R_forward_capped": 0.9,          # F0*(1-θ) = 1*(1-0.1) = 0.9
            "denominator_kprot": 4.0,
            "gamma": 0.1,
            "k_des": 5.0,                     # 5 * 0.1 = 0.5
            "gamma_max": 1.0,
        }
        # 0.9 − 4*0.1 − 5*0.1 = 0.9 − 0.4 − 0.5 = 0.0
        residual = compute_mass_balance_residual_rel(rung)
        assert residual is not None
        assert residual < 1e-12

    def test_returns_none_when_missing_field(self):
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import (
            compute_mass_balance_residual_rel,
        )
        assert compute_mass_balance_residual_rel({}) is None
        assert compute_mass_balance_residual_rel(
            {"R_forward_capped": 0.1, "gamma": 0.01}
        ) is None  # missing denom_kprot + k_des

    def test_nonzero_residual_normalised(self):
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import (
            compute_mass_balance_residual_rel,
        )
        # F=1.0, denom_kprot*γ=0.1, k_des*γ=0.1, total subtracted = 0.2 →
        # residual_abs = 0.8.  Denom = max(1.0, k_des*γ_max=0.5) = 1.0 → 0.8.
        rung = {
            "R_forward_capped": 1.0,
            "denominator_kprot": 0.1,
            "gamma": 1.0,
            "k_des": 0.5,
            "gamma_max": 1.0,
        }
        # Wait — denominator_kprot is the COEFFICIENT, not the product.
        # 1.0 - 0.1*1.0 - 0.5*1.0 = 0.4 → abs=0.4; denom=1.0 → 0.4.
        residual = compute_mass_balance_residual_rel(rung)
        assert residual == pytest.approx(0.4, rel=1e-12)


# ===========================================================================
# classify_no_route_cause
# ===========================================================================


class TestClassifyNoRouteCause:
    """Order matters per R3 #6 + R4 #6 — most-specific failure wins."""

    def test_no_saturated_rung(self):
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import (
            classify_no_route_cause,
        )
        records = [
            _per_k_hyd(
                k_hyd_target=1e-5, lam1=_lam1(theta=0.1, gamma=0.005),
            ),
            _per_k_hyd(
                k_hyd_target=1e-4, lam1=_lam1(theta=0.4, gamma=0.02),
            ),
            _per_k_hyd(
                k_hyd_target=1e-3, lam1=_lam1(theta=0.85, gamma=0.04),
            ),  # below 0.9
        ]
        assert classify_no_route_cause(records) == "no_saturated_rung"

    def test_picard_failure_at_saturated(self):
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import (
            classify_no_route_cause,
        )
        records = [
            _per_k_hyd(
                k_hyd_target=1e-3, lam1=_lam1(theta=0.92, gamma=0.043),
            ),
            _per_k_hyd(
                k_hyd_target=1e-2,
                lam1=_lam1(
                    theta=0.98, gamma=0.046,
                    picard_status="iter_cap_hit_unconverged",
                ),
            ),
        ]
        # The 1e-2 saturated rung has bad picard; the 0.92 saturated rung is
        # picard_ok but the 1e-2 is not, so the saturated set has at least
        # one picard-ok entry (1e-3 at θ=0.92) — so picard_failure does NOT
        # fire.  Rework to fail picard on ALL saturated entries.
        records = [
            _per_k_hyd(
                k_hyd_target=1e-3,
                lam1=_lam1(
                    theta=0.92, gamma=0.043,
                    picard_status="iter_cap_hit_unconverged",
                ),
            ),
            _per_k_hyd(
                k_hyd_target=1e-2,
                lam1=_lam1(
                    theta=0.98, gamma=0.046,
                    picard_status="iter_cap_hit_unconverged",
                ),
            ),
        ]
        assert classify_no_route_cause(records) == "picard_failure"

    def test_mass_balance_failure(self):
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import (
            classify_no_route_cause,
        )
        records = [
            _per_k_hyd(
                k_hyd_target=1e-3, lam1=_lam1(
                    theta=0.92, gamma=0.043,
                    mass_balance_residual_rel=0.02,   # >5e-3
                ),
            ),
            _per_k_hyd(
                k_hyd_target=1e-2, lam1=_lam1(
                    theta=0.98, gamma=0.046,
                    mass_balance_residual_rel=0.02,   # >5e-3
                ),
            ),
        ]
        assert classify_no_route_cause(records) == "mass_balance_failure"

    def test_transport_only(self):
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import (
            classify_no_route_cause,
        )
        records = [
            _per_k_hyd(
                k_hyd_target=1e-3, lam1=_lam1(
                    theta=0.92, gamma=0.043,
                    o2_flux_levich_ratio=0.95,   # ≥0.9
                ),
            ),
            _per_k_hyd(
                k_hyd_target=1e-2, lam1=_lam1(
                    theta=0.98, gamma=0.046,
                    o2_flux_levich_ratio=0.97,   # ≥0.9
                ),
            ),
        ]
        assert classify_no_route_cause(records) == "transport_only"

    def test_grid_gap(self):
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import (
            classify_no_route_cause,
        )
        # All gates pass except the slope check — saturated rungs exist
        # but the slope between any pair is too steep (cap_coverage>0.9
        # but Γ still growing).  Single point: slope can't be computed,
        # so select_k_hyd_route returns None and the cause is grid_gap.
        records = [
            _per_k_hyd(
                k_hyd_target=1e-3, lam1=_lam1(
                    theta=0.92, gamma=0.043,
                    o2_flux_levich_ratio=0.30,
                ),
            ),
        ]
        # With only one entry, slope can't compute; classify_no_route_cause
        # should walk past the four earlier categories (saturated, picard,
        # mass_balance, transport are all fine) and return grid_gap.
        assert classify_no_route_cause(records) == "grid_gap"


# ===========================================================================
# select_k_hyd_route
# ===========================================================================


class TestSelectKHydRoute:
    def test_prefers_highest_passing_k_hyd(self):
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import (
            select_k_hyd_route,
        )
        # 1e-3: θ=0.92 (fails 0.95 strict gate)
        # 5e-3: θ=0.96, slope < 0.05 → eligible
        # 1e-2: θ=0.98, slope vs 5e-3 < 0.05 → eligible
        # Expect 1e-2 (highest passing).
        records = [
            _per_k_hyd(
                k_hyd_target=1e-3, lam1=_lam1(theta=0.92, gamma=0.0432),
            ),
            _per_k_hyd(
                k_hyd_target=5e-3, lam1=_lam1(theta=0.96, gamma=0.0451),
            ),
            _per_k_hyd(
                k_hyd_target=1e-2, lam1=_lam1(theta=0.98, gamma=0.04602),
            ),
        ]
        # gamma slope log: log(0.04602/0.0451)/log(1e-2/5e-3)=
        # log(1.0204)/log(2)=0.020/0.693≈0.0291 < 0.05 ✓
        assert select_k_hyd_route(records) == pytest.approx(1e-2)

    def test_no_candidate_at_theta_below_min(self):
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import (
            select_k_hyd_route,
        )
        records = [
            _per_k_hyd(
                k_hyd_target=1e-3, lam1=_lam1(theta=0.5, gamma=0.024),
            ),
        ]
        assert select_k_hyd_route(records) is None

    def test_filters_transport_limited(self):
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import (
            select_k_hyd_route,
        )
        # Saturated but transport-limited at the high end.
        records = [
            _per_k_hyd(
                k_hyd_target=5e-3, lam1=_lam1(
                    theta=0.96, gamma=0.0451, o2_flux_levich_ratio=0.7,
                ),
            ),
            _per_k_hyd(
                k_hyd_target=1e-2, lam1=_lam1(
                    theta=0.98, gamma=0.04602,
                    o2_flux_levich_ratio=0.95,   # transport-limited
                ),
            ),
        ]
        # 1e-2 fails transport; 5e-3 has no lower neighbour for slope check.
        assert select_k_hyd_route(records) is None


# ===========================================================================
# build_v10b_priorities_block
# ===========================================================================


class TestBuildV10bPrioritiesBlock:
    def test_route_exists_with_low_priority(self):
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import (
            build_v10b_priorities_block,
        )
        records = [
            _per_k_hyd(
                k_hyd_target=5e-3, lam1=_lam1(
                    theta=0.96, gamma=0.0451, H2O2_selectivity_pct=28.0,
                    F0_decomposition={"amplification_from_singh": 1.05},
                ),
            ),
            _per_k_hyd(
                k_hyd_target=1e-2, lam1=_lam1(
                    theta=0.98, gamma=0.04602, H2O2_selectivity_pct=27.0,
                    F0_decomposition={"amplification_from_singh": 1.10},
                ),
            ),
        ]
        block = build_v10b_priorities_block(records)
        assert block["k_hyd_route"] == pytest.approx(1e-2)
        assert block["single_v_selectivity_gap_pp"] == 0.0    # 27 inside band
        assert block["kdes_gammamax_priority"] == "low"
        assert block["max_amp_from_singh"] == pytest.approx(1.10)
        assert block["rHEl_recalibration_required"] is False
        assert block["no_route_cause"] is None

    def test_route_exists_with_high_priority(self):
        """|gap| > 10pp triggers HIGH priority."""
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import (
            build_v10b_priorities_block,
        )
        records = [
            _per_k_hyd(
                k_hyd_target=5e-3, lam1=_lam1(
                    theta=0.96, gamma=0.0451, H2O2_selectivity_pct=10.0,
                ),
            ),
            _per_k_hyd(
                k_hyd_target=1e-2, lam1=_lam1(
                    theta=0.98, gamma=0.04602, H2O2_selectivity_pct=12.0,
                ),
            ),
        ]
        block = build_v10b_priorities_block(records)
        assert block["k_hyd_route"] == pytest.approx(1e-2)
        # 12% is 13pp below band low (25%); |gap| > 10.
        assert block["single_v_selectivity_gap_pp"] == pytest.approx(13.0)
        assert block["kdes_gammamax_priority"] == "high"

    def test_no_route_classifies_cause(self):
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import (
            build_v10b_priorities_block,
        )
        records = [
            _per_k_hyd(
                k_hyd_target=1e-5, lam1=_lam1(theta=0.01, gamma=0.0005),
            ),
        ]
        block = build_v10b_priorities_block(records)
        assert block["k_hyd_route"] is None
        assert block["no_route_cause"] == "no_saturated_rung"


# ===========================================================================
# _parse_args / _parse_k_hyd_grid
# ===========================================================================


class TestParseArgs:
    def test_defaults(self):
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import (
            _parse_args, V_KIN_DEFAULT, K0_R4E_FACTOR_DEFAULT,
        )
        args = _parse_args(["--no-plot"])
        assert args.v_kin == pytest.approx(V_KIN_DEFAULT)
        assert args.k0_r4e_factor == pytest.approx(K0_R4E_FACTOR_DEFAULT)
        assert args.k_hyd_grid is None
        assert args.lambda_ladder is None
        assert args.with_perturbation is False

    def test_v_kin_override(self):
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import _parse_args
        args = _parse_args(["--v-kin", "-0.20", "--no-plot"])
        assert args.v_kin == pytest.approx(-0.20)

    def test_k0_r4e_factor_scientific(self):
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import _parse_args
        args = _parse_args(["--k0-r4e-factor", "1e-14", "--no-plot"])
        assert args.k0_r4e_factor == pytest.approx(1e-14)

    def test_k_hyd_grid_passthrough(self):
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import _parse_args
        args = _parse_args([
            "--k-hyd-grid", "1e-5,1e-4,1e-3", "--no-plot",
        ])
        assert args.k_hyd_grid == "1e-5,1e-4,1e-3"


class TestParseKHydGrid:
    def test_default_when_none(self):
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import (
            _parse_k_hyd_grid, K_HYD_GRID_DEFAULT,
        )
        assert _parse_k_hyd_grid(None) == K_HYD_GRID_DEFAULT
        assert _parse_k_hyd_grid("") == K_HYD_GRID_DEFAULT

    def test_default_grid_has_10_points(self):
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import (
            K_HYD_GRID_DEFAULT,
        )
        assert len(K_HYD_GRID_DEFAULT) == 10

    def test_default_grid_includes_baseline_kHyd(self):
        """1e-3 is the v10a' baseline — sanity audit can't run if missing."""
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import (
            K_HYD_GRID_DEFAULT,
        )
        assert any(abs(k - 1e-3) <= 1e-30 for k in K_HYD_GRID_DEFAULT)

    def test_parses_scientific_notation(self):
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import (
            _parse_k_hyd_grid,
        )
        out = _parse_k_hyd_grid("1e-5,3e-5,1e-4,1e-3,1e-2,1e-1")
        assert len(out) == 6
        assert out[0] == pytest.approx(1e-5)
        assert out[-1] == pytest.approx(1e-1)

    def test_whitespace_tolerant(self):
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import (
            _parse_k_hyd_grid,
        )
        out = _parse_k_hyd_grid(" 1e-5 , 1e-4 , 1e-3 ")
        assert out == (1e-5, 1e-4, 1e-3)


# ===========================================================================
# --lambda-ladder CLI (step 9.A)
# ===========================================================================


class TestParseLambdaLadder:
    """Step 9.A: --lambda-ladder CLI flag mirrors --k-hyd-grid validation.

    Defaults to LAMBDA_LADDER imported from v_sweep_diagnostic so absence
    of the flag reproduces the locked v10a behavior byte-equivalently.
    """

    def test_phase_A2_lambda_ladder_cli_default(self):
        """Flag absent -> falls back to imported LAMBDA_LADDER constant
        (0.0, 0.25, 0.50, 0.75, 1.0)."""
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import (
            _parse_args, _parse_lambda_ladder,
        )
        from scripts.studies.phase6b_v10a_v_sweep_diagnostic import (
            LAMBDA_LADDER,
        )
        args = _parse_args(["--no-plot"])
        assert args.lambda_ladder is None
        # Parse fallback path.
        ladder = _parse_lambda_ladder(args.lambda_ladder)
        assert ladder == tuple(float(x) for x in LAMBDA_LADDER)
        # Locked v10a default for the module constant.
        assert ladder == (0.0, 0.25, 0.50, 0.75, 1.0)

    def test_phase_A2_lambda_ladder_cli_custom(self):
        """--lambda-ladder 0.0,0.5,1.0 parses to the expected tuple."""
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import (
            _parse_args, _parse_lambda_ladder,
        )
        args = _parse_args(
            ["--lambda-ladder", "0.0,0.5,1.0", "--no-plot"],
        )
        assert args.lambda_ladder == "0.0,0.5,1.0"
        ladder = _parse_lambda_ladder(args.lambda_ladder)
        assert ladder == (0.0, 0.5, 1.0)

    def test_phase_A2_lambda_ladder_cli_b2_locked_grid(self):
        """Step 9 B.2 locked 10-point ladder parses correctly."""
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import (
            _parse_lambda_ladder,
        )
        raw = "0.0,0.10,0.25,0.40,0.50,0.60,0.75,0.85,0.95,1.0"
        ladder = _parse_lambda_ladder(raw)
        assert len(ladder) == 10
        assert ladder[0] == 0.0
        assert ladder[-1] == 1.0
        # Monotonic non-decreasing.
        for i in range(len(ladder) - 1):
            assert ladder[i] <= ladder[i + 1]

    def test_phase_A2_lambda_ladder_cli_rejects_non_monotonic(self):
        """Non-monotonic input raises SystemExit."""
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import (
            _parse_lambda_ladder,
        )
        with pytest.raises(SystemExit):
            _parse_lambda_ladder("0.0,0.5,0.3,1.0")

    def test_phase_A2_lambda_ladder_cli_rejects_off_endpoints(self):
        """Endpoints != 0.0 / 1.0 raise SystemExit."""
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import (
            _parse_lambda_ladder,
        )
        with pytest.raises(SystemExit):
            _parse_lambda_ladder("0.1,0.5,0.9")
        with pytest.raises(SystemExit):
            _parse_lambda_ladder("0.0,0.5,0.9")
        with pytest.raises(SystemExit):
            _parse_lambda_ladder("0.1,0.5,1.0")

    def test_phase_A2_lambda_ladder_cli_rejects_out_of_range(self):
        """Values outside [0, 1] raise SystemExit."""
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import (
            _parse_lambda_ladder,
        )
        with pytest.raises(SystemExit):
            _parse_lambda_ladder("0.0,1.5,1.0")
        with pytest.raises(SystemExit):
            _parse_lambda_ladder("-0.1,0.5,1.0")

    def test_phase_A2_lambda_ladder_cli_rejects_empty(self):
        """Empty value (just commas/whitespace) raises SystemExit."""
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import (
            _parse_lambda_ladder,
        )
        with pytest.raises(SystemExit):
            _parse_lambda_ladder(" , , ")


# ===========================================================================
# Exception-prefix drift guard (R4 #5)
# ===========================================================================


class TestExceptionMessagePrefixesMatchSolver:
    """R4 #5 — solver's LadderExhausted text must contain the three
    expected prefixes the driver decodes.  Drift here surfaces in CI
    rather than at runtime."""

    def test_solver_source_contains_all_three_prefixes(self):
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import (
            EXPECTED_LADDER_EXHAUSTED_PREFIXES,
        )
        solver_path = os.path.join(
            _ROOT, "Forward", "bv_solver", "anchor_continuation.py",
        )
        with open(solver_path, "r", encoding="utf-8") as f:
            source = f.read()
        for prefix in EXPECTED_LADDER_EXHAUSTED_PREFIXES:
            assert prefix in source, (
                f"Expected LadderExhausted prefix not found in solver "
                f"source: {prefix!r}.  If the solver's exception text "
                f"changed, update EXPECTED_LADDER_EXHAUSTED_PREFIXES + "
                f"the exception_phase classification in "
                f"_run_k_hyd_ramp() to match."
            )

    def test_solve_lambda_ramp_helper_raises_each_message(self):
        """Cross-check: assert the three prefixes appear in the
        solve_lambda_ramp_from_warm_start function body, not just
        somewhere else in the file."""
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import (
            EXPECTED_LADDER_EXHAUSTED_PREFIXES,
        )
        solver_path = os.path.join(
            _ROOT, "Forward", "bv_solver", "anchor_continuation.py",
        )
        with open(solver_path, "r", encoding="utf-8") as f:
            source = f.read()
        # Slice from def solve_lambda_ramp_from_warm_start to the end.
        marker = "def solve_lambda_ramp_from_warm_start("
        idx = source.find(marker)
        assert idx >= 0, (
            f"solve_lambda_ramp_from_warm_start not found in solver source"
        )
        body = source[idx:]
        # Two of the three prefixes are unique to this function; the third
        # (λ_hydrolysis ramp exhausted at λ=) may also appear in
        # solve_anchor_with_continuation.  Assert each prefix appears in
        # this function's body so a function-local refactor surfaces here.
        for prefix in EXPECTED_LADDER_EXHAUSTED_PREFIXES:
            assert prefix in body, (
                f"Prefix {prefix!r} not found in "
                f"solve_lambda_ramp_from_warm_start body"
            )


# ===========================================================================
# augment_rung_diagnostics
# ===========================================================================


class TestAugmentRungDiagnostics:
    def test_cd_mA_cm2_uses_negative_i_scale(self):
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import (
            augment_rung_diagnostics,
        )
        rung = {
            "cd_observable": 0.5,
            "R_2e_current_nondim": 0.5,
            "R_4e_current_nondim": 0.5,
        }
        out = augment_rung_diagnostics(
            rung, i_scale=4.0, i_lim_4e_mA_cm2=5.5,
            electrode_area_nondim=1.0, domain_height_hat=0.16,
            snes_converged=True, gamma_picard_history=[0.04],
            pc_mA_cm2=None,
        )
        # cd_mA_cm2 = -i_scale * cd_obs = -4 * 0.5 = -2.0
        assert out["cd_mA_cm2"] == pytest.approx(-2.0)
        assert out["current_filter_ratio"] == pytest.approx(2.0 / 5.5)

    def test_x_2e_signs(self):
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import (
            augment_rung_diagnostics,
        )
        rung = {
            "cd_observable": 1.0,
            "R_2e_current_nondim": 0.2,
            "R_4e_current_nondim": 0.8,
        }
        out = augment_rung_diagnostics(
            rung, i_scale=1.0, i_lim_4e_mA_cm2=5.5,
            electrode_area_nondim=1.0, domain_height_hat=0.16,
            snes_converged=True, gamma_picard_history=[0.04],
            pc_mA_cm2=None,
        )
        assert out["x_2e"] == pytest.approx(0.2)
        assert out["x_4e"] == pytest.approx(0.8)
        assert out["H2O2_selectivity_pct"] == pytest.approx(20.0)

    def test_does_not_mutate_input(self):
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import (
            augment_rung_diagnostics,
        )
        rung = {"cd_observable": 0.5, "R_2e_current_nondim": 0.5,
                "R_4e_current_nondim": 0.5}
        rung_copy = dict(rung)
        augment_rung_diagnostics(
            rung, i_scale=1.0, i_lim_4e_mA_cm2=5.5,
            electrode_area_nondim=1.0, domain_height_hat=0.16,
            snes_converged=True, gamma_picard_history=[0.04],
            pc_mA_cm2=None,
        )
        assert rung == rung_copy

    def test_picard_status_propagated(self):
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import (
            augment_rung_diagnostics,
        )
        out = augment_rung_diagnostics(
            {}, i_scale=1.0, i_lim_4e_mA_cm2=5.5,
            electrode_area_nondim=1.0, domain_height_hat=0.16,
            snes_converged=False, gamma_picard_history=[0.04, 0.05],
            pc_mA_cm2=None,
        )
        assert out["picard_status"] == "snes_failed"


if __name__ == "__main__":                                    # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
