"""Fast unit tests for Phase 6β step 6 plumbing-ablation driver helpers.

All tests in this module run without Firedrake; they exercise the
pure-Python helpers in
:mod:`scripts.studies.drivers.phase6b_step6_plumbing_ablation`.

Slow Firedrake-using tests live in
:mod:`tests.test_phase6b_step6_plumbing_ablation_slow` (R5 #2).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

import pytest

from scripts.studies.drivers.phase6b_step6_plumbing_ablation import (
    ABLATIONS_DEFAULT,
    DELTA_C_MAX,
    DELTA_C_MIN,
    REQUIRED_NUMERIC_KEYS,
    R_INJ_BRACKET_DEFAULT,
    SIGMA_SINGH_K_CU_DECK,
    SIGMA_SINGH_PLUMBING_SENTINEL,
    _build_ablation_sp_overrides,
    _compute_beta_K_Cu,
    _get_dotted,
    _override_to_signed_sigma_S,
    _parse_args,
    _parse_bracket,
    _select_r_inj_bracket,
    _verify_wiring_from_prepass,
    classify_ablation_status,
    classify_diagnostic_failure,
)


# ---------------------------------------------------------------------------
# _parse_args + bracket parsing
# ---------------------------------------------------------------------------


def test_parse_args_defaults():
    ns = _parse_args([])
    assert ns.v_kin == pytest.approx(-0.10)
    assert ns.k0_r4e_factor == pytest.approx(1e-14)
    assert ns.k_hyd == pytest.approx(1e-3)
    assert ns.ablations_list == ABLATIONS_DEFAULT


def test_parse_args_rejects_negative_sigma():
    with pytest.raises(SystemExit):
        _parse_args(["--sigma-singh-override", "-1.0"])


def test_parse_args_rejects_nan_sigma():
    with pytest.raises(SystemExit):
        _parse_args(["--sigma-singh-override", "nan"])


def test_parse_args_rejects_unknown_ablation():
    with pytest.raises(SystemExit):
        _parse_args(["--ablations", "A0,A4"])


def test_parse_args_accepts_custom_brackets():
    ns = _parse_args([
        "--r-inj-prepass-A1", "1e-3,1e-2",
        "--r-inj-prepass-A2", "5e-3,5e-2",
    ])
    # Brackets validated in _parse_args; full parse happens in main().
    assert ns.r_inj_prepass_A1 == "1e-3,1e-2"


def test_parse_bracket_default():
    assert _parse_bracket(None, R_INJ_BRACKET_DEFAULT) == R_INJ_BRACKET_DEFAULT


def test_parse_bracket_custom():
    assert _parse_bracket("1e-4, 2e-3, 5.0", R_INJ_BRACKET_DEFAULT) == (
        1e-4, 2e-3, 5.0,
    )


def test_parse_bracket_rejects_empty():
    with pytest.raises(ValueError):
        _parse_bracket(",,,", R_INJ_BRACKET_DEFAULT)


def test_parse_bracket_rejects_negative():
    with pytest.raises(ValueError):
        _parse_bracket("1e-3,-1.0", R_INJ_BRACKET_DEFAULT)


def test_parse_bracket_rejects_nonfinite():
    with pytest.raises(ValueError):
        _parse_bracket("1e-3,inf", R_INJ_BRACKET_DEFAULT)


# ---------------------------------------------------------------------------
# _override_to_signed_sigma_S round-trip
# ---------------------------------------------------------------------------


def test_override_to_signed_sigma_S_round_trip_at_sentinel():
    """``σ_signed → max(0, -σ_signed) · 6.2415e-6 → override`` round-trip."""
    override = float(SIGMA_SINGH_PLUMBING_SENTINEL)
    sigma_signed_C_m2 = _override_to_signed_sigma_S(override)
    # σ_signed should be negative (cathodic).
    assert sigma_signed_C_m2 < 0.0
    # The form-build code does:  σ_singh = max(0, -σ_signed) · 6.2415e-6
    factor = 1.0 / 1.602176634e-19 * 1e-24
    sigma_singh_recovered = max(0.0, -sigma_signed_C_m2) * factor
    assert sigma_singh_recovered == pytest.approx(override, rel=1e-12)


def test_override_to_signed_sigma_S_at_zero():
    """Zero override → zero signed σ → zero post-clamp σ_singh."""
    s = _override_to_signed_sigma_S(0.0)
    assert s == pytest.approx(0.0)


def test_override_to_signed_sigma_S_at_singh_K_cu_deck():
    """Reverse cross-check: K⁺/Cu Singh-Table-S3 σ ≈ 1.41e-5 counts/pm²
    corresponds to signed σ_S = -1.41e-5 / 6.2415e-6 ≈ -2.259 C/m².
    """
    s = _override_to_signed_sigma_S(SIGMA_SINGH_K_CU_DECK)
    expected = -SIGMA_SINGH_K_CU_DECK * (1.602176634e-19 / 1e-24)
    assert s == pytest.approx(expected, rel=1e-12)
    assert s == pytest.approx(-2.259, rel=1e-2)


# ---------------------------------------------------------------------------
# _build_ablation_sp_overrides
# ---------------------------------------------------------------------------


def test_overrides_A0_empty():
    assert _build_ablation_sp_overrides("A0") == {}


def test_overrides_A0b_empty():
    assert _build_ablation_sp_overrides("A0b") == {}


def test_overrides_A1():
    o = _build_ablation_sp_overrides("A1", r_inj=1e-3)
    assert o["apply_h_source"] is True
    assert o["apply_k_sink"] is False
    assert o["manufactured_R_inj"] == pytest.approx(1e-3)


def test_overrides_A2():
    o = _build_ablation_sp_overrides("A2", r_inj=5e-2)
    assert o["apply_h_source"] is False
    assert o["apply_k_sink"] is True
    assert o["manufactured_R_inj"] == pytest.approx(5e-2)


def test_overrides_A3_default():
    o = _build_ablation_sp_overrides("A3")
    assert o["override_sigma_singh_counts_pm2"] == pytest.approx(
        SIGMA_SINGH_PLUMBING_SENTINEL
    )


def test_overrides_A3_custom():
    o = _build_ablation_sp_overrides("A3", sigma_singh_override=0.05)
    assert o["override_sigma_singh_counts_pm2"] == pytest.approx(0.05)


def test_overrides_A1_rejects_missing_r_inj():
    with pytest.raises(ValueError):
        _build_ablation_sp_overrides("A1")


def test_overrides_A1_rejects_nan_r_inj():
    with pytest.raises(ValueError):
        _build_ablation_sp_overrides("A1", r_inj=float("nan"))


def test_overrides_A3_rejects_negative_sigma():
    with pytest.raises(ValueError):
        _build_ablation_sp_overrides("A3", sigma_singh_override=-0.01)


def test_overrides_rejects_unknown_ablation():
    with pytest.raises(ValueError):
        _build_ablation_sp_overrides("A99")


# ---------------------------------------------------------------------------
# _get_dotted + classify_diagnostic_failure
# ---------------------------------------------------------------------------


def test_get_dotted_top_level():
    rec = {"a": 1.0, "b": {"c": 2.0}}
    assert _get_dotted(rec, "a") == 1.0


def test_get_dotted_nested():
    rec = {"F0_decomposition": {"pka_factor_avg": 10.07}}
    assert _get_dotted(rec, "F0_decomposition.pka_factor_avg") == 10.07


def test_get_dotted_missing_returns_none():
    rec = {"F0_decomposition": {}}
    assert _get_dotted(rec, "F0_decomposition.pka_factor_avg") is None
    assert _get_dotted(rec, "missing.key") is None


def test_get_dotted_through_non_dict_returns_none():
    rec = {"a": 1.0}
    assert _get_dotted(rec, "a.b") is None


def test_classify_diagnostic_failure_a0_pass():
    rec: Dict[str, Any] = {
        k: 1.0 for k in REQUIRED_NUMERIC_KEYS["A0"]
    }
    assert classify_diagnostic_failure(rec, "A0") is None


def test_classify_diagnostic_failure_a0_missing():
    rec: Dict[str, Any] = {
        k: 1.0 for k in REQUIRED_NUMERIC_KEYS["A0"][:-1]
    }
    msg = classify_diagnostic_failure(rec, "A0")
    assert msg is not None and "is None" in msg


def test_classify_diagnostic_failure_a0_nan():
    rec: Dict[str, Any] = {
        k: 1.0 for k in REQUIRED_NUMERIC_KEYS["A0"]
    }
    rec["gamma"] = float("nan")
    msg = classify_diagnostic_failure(rec, "A0")
    assert msg is not None and "gamma" in msg


def test_classify_diagnostic_failure_a3_resolves_dotted():
    rec: Dict[str, Any] = {
        k: 1.0 for k in REQUIRED_NUMERIC_KEYS["A3"]
        if "." not in k
    }
    # Dotted keys
    rec["F0_decomposition"] = {
        "pka_factor_avg": 10.0,
        "amplification_from_singh": 9.9,
    }
    assert classify_diagnostic_failure(rec, "A3") is None


def test_classify_diagnostic_failure_a3_missing_dotted():
    rec: Dict[str, Any] = {
        k: 1.0 for k in REQUIRED_NUMERIC_KEYS["A3"]
        if "." not in k
    }
    rec["F0_decomposition"] = {"pka_factor_avg": 10.0}  # missing amp_from_singh
    msg = classify_diagnostic_failure(rec, "A3")
    assert msg is not None and "amplification_from_singh" in msg


def test_classify_diagnostic_failure_unknown_ablation():
    msg = classify_diagnostic_failure({}, "A99")
    assert msg is not None and "A99" in msg


# ---------------------------------------------------------------------------
# _select_r_inj_bracket
# ---------------------------------------------------------------------------


def _mk_prepass(R_inj, delta_abs, *, converged=True, positivity_ok=True):
    return {
        "R_inj": float(R_inj),
        "converged": bool(converged),
        "positivity_ok": bool(positivity_ok),
        "delta_c_abs_rel": float(delta_abs) if delta_abs is not None else None,
    }


def test_select_r_inj_picks_smallest_in_band():
    results = [
        _mk_prepass(1e-4, 0.01),    # under-shoot
        _mk_prepass(1e-3, 0.08),    # in band — pick this
        _mk_prepass(1e-2, 0.20),    # in band but larger
        _mk_prepass(1e-1, 0.40),    # over-shoot
    ]
    out = _select_r_inj_bracket(results)
    assert out["status"] == "selected"
    assert out["R_inj"] == pytest.approx(1e-3)


def test_select_r_inj_all_undershoot_signals_escalation():
    results = [
        _mk_prepass(1e-4, 0.005),
        _mk_prepass(1e-3, 0.01),
        _mk_prepass(1e-2, 0.03),
    ]
    out = _select_r_inj_bracket(results)
    assert out["status"] == "all_undershoot"
    assert out["R_inj"] is None


def test_select_r_inj_smallest_overshoots_inconclusive():
    results = [
        _mk_prepass(1e-4, 0.60),    # already too big at the floor
        _mk_prepass(1e-3, 0.80),
    ]
    out = _select_r_inj_bracket(results)
    assert out["status"] == "inconclusive_smallest_overshoots"
    assert out["R_inj"] is None


def test_select_r_inj_skips_non_converged():
    results = [
        _mk_prepass(1e-3, 0.10, converged=False),
        _mk_prepass(1e-2, 0.15),
    ]
    out = _select_r_inj_bracket(results)
    assert out["status"] == "selected"
    assert out["R_inj"] == pytest.approx(1e-2)


def test_select_r_inj_skips_positivity_failure():
    results = [
        _mk_prepass(1e-3, 0.10, positivity_ok=False),
        _mk_prepass(1e-2, 0.15),
    ]
    out = _select_r_inj_bracket(results)
    assert out["status"] == "selected"
    assert out["R_inj"] == pytest.approx(1e-2)


def test_select_r_inj_no_converged_returns_no_converged():
    results = [
        _mk_prepass(1e-3, 0.10, converged=False),
        _mk_prepass(1e-2, 0.15, converged=False),
    ]
    out = _select_r_inj_bracket(results)
    assert out["status"] == "no_converged"
    assert out["R_inj"] is None


# ---------------------------------------------------------------------------
# classify_ablation_status
# ---------------------------------------------------------------------------


def test_classify_status_all_pass():
    flags = {"gate1": True, "gate2": True, "gate3": True}
    assert classify_ablation_status(flags) == "pass"


def test_classify_status_one_fail():
    flags = {"gate1": True, "gate2": False}
    assert classify_ablation_status(flags) == "fail"


def test_classify_status_empty():
    assert classify_ablation_status({}) == "fail"


# ---------------------------------------------------------------------------
# _compute_beta_K_Cu sanity (matches CLAUDE.md hard rule #6 / A.2 verify)
# ---------------------------------------------------------------------------


def test_compute_beta_K_Cu_matches_a2_record():
    """β · σ_S_singh = ΔpKa.  At A.2 baseline σ_singh = 1.0704e-7
    counts/pm² (= -0.017149 C/m² · 6.2415e-6), ΔpKa = -4.88e-6.
    """
    beta = _compute_beta_K_Cu()
    # β ≈ -45.6
    assert beta == pytest.approx(-45.6, rel=1e-2)
    # ΔpKa at A.2 baseline
    sigma_signed_C_m2 = -0.017149
    factor = (1.0 / 1.602176634e-19) * 1e-24
    sigma_singh = max(0.0, -sigma_signed_C_m2) * factor
    delta_pka = beta * sigma_singh
    assert delta_pka == pytest.approx(-4.8817e-6, rel=1e-2)


def test_compute_beta_K_Cu_at_sentinel_predicts_amplification():
    """At σ_override = 0.022 counts/pm², pka_factor = 10^(-β·σ) ≈ 10.07."""
    beta = _compute_beta_K_Cu()
    pka_factor = math.pow(10.0, -beta * SIGMA_SINGH_PLUMBING_SENTINEL)
    assert pka_factor == pytest.approx(10.07, rel=1e-2)


# ---------------------------------------------------------------------------
# single_v_selectivity_gap_pp re-imported from A.2 driver for routing parity
# ---------------------------------------------------------------------------


def test_single_v_selectivity_gap_pp_inside_band():
    from scripts.studies.drivers.phase6b_v10a_phase_A2_v_kin import (
        single_v_selectivity_gap_pp,
    )
    assert single_v_selectivity_gap_pp(30.0) == pytest.approx(0.0)


def test_single_v_selectivity_gap_pp_below_band():
    from scripts.studies.drivers.phase6b_v10a_phase_A2_v_kin import (
        single_v_selectivity_gap_pp,
    )
    # band low = 25 → 30 - 19.91 = 5.09
    assert single_v_selectivity_gap_pp(19.91) == pytest.approx(5.09, rel=1e-4)


def test_single_v_selectivity_gap_pp_above_band():
    from scripts.studies.drivers.phase6b_v10a_phase_A2_v_kin import (
        single_v_selectivity_gap_pp,
    )
    # band high = 50 → 50 - 60 = -10
    assert single_v_selectivity_gap_pp(60.0) == pytest.approx(-10.0)


# ---------------------------------------------------------------------------
# _verify_wiring_from_prepass — the inconclusive-but-wiring-ok diagnosis
# ---------------------------------------------------------------------------


def _mk_prepass_signed(R_inj, delta_signed, *, converged=True):
    """Build a pre-pass row with both signed and abs delta."""
    abs_delta = abs(delta_signed) if delta_signed is not None else None
    return {
        "R_inj": float(R_inj),
        "converged": bool(converged),
        "delta_c_signed_rel": delta_signed,
        "delta_c_abs_rel": abs_delta,
    }


def test_wiring_verdict_a1_positive_sign_monotonic():
    """A1 sink-source wiring verified: signed Δc_H > 0 + monotonic |Δc|."""
    rows = [
        _mk_prepass_signed(1e-3, +1e-4),
        _mk_prepass_signed(1e-2, +1e-3),
        _mk_prepass_signed(1e-1, +1e-2),
        _mk_prepass_signed(1.0, +1e-1),
    ]
    v = _verify_wiring_from_prepass("A1", rows)
    assert v["sign_correct_at_largest_R_inj"] is True
    assert v["monotonic_in_R_inj"] is True
    assert v["largest_R_inj"] == pytest.approx(1.0)
    assert v["largest_signed_delta"] == pytest.approx(1e-1)


def test_wiring_verdict_a2_negative_sign_monotonic():
    """A2 sink-only wiring verified: signed Δc_K < 0 + monotonic |Δc|."""
    rows = [
        _mk_prepass_signed(1e-3, -1e-4),
        _mk_prepass_signed(1e-2, -1e-3),
        _mk_prepass_signed(1e-1, -1e-2),
        _mk_prepass_signed(1.0, -1e-1),
    ]
    v = _verify_wiring_from_prepass("A2", rows)
    assert v["sign_correct_at_largest_R_inj"] is True
    assert v["monotonic_in_R_inj"] is True


def test_wiring_verdict_a2_wrong_sign_flagged():
    """A2 with positive Δc_K (wrong sign) ⇒ wiring NOT verified."""
    rows = [
        _mk_prepass_signed(1e-3, +1e-4),    # K should drop, not rise
        _mk_prepass_signed(1e-2, +1e-3),
        _mk_prepass_signed(1e-1, +1e-2),
    ]
    v = _verify_wiring_from_prepass("A2", rows)
    assert v["sign_correct_at_largest_R_inj"] is False


def test_wiring_verdict_a2_non_monotonic_flagged():
    """Non-monotonic |Δc| growth ⇒ wiring NOT verified."""
    rows = [
        _mk_prepass_signed(1e-3, -1e-2),    # huge at small R_inj — weird
        _mk_prepass_signed(1e-2, -1e-3),
        _mk_prepass_signed(1e-1, -1e-2),
        _mk_prepass_signed(1.0, -5e-3),     # shrinks again — not monotonic
    ]
    v = _verify_wiring_from_prepass("A2", rows)
    assert v["sign_correct_at_largest_R_inj"] is True   # sign OK
    assert v["monotonic_in_R_inj"] is False              # but not monotonic


def test_wiring_verdict_no_converged():
    """Empty / no-converged input ⇒ defensive verdict."""
    v = _verify_wiring_from_prepass("A1", [])
    assert v["sign_correct_at_largest_R_inj"] is False
    assert v["monotonic_in_R_inj"] is False
    assert v["reason"] == "no_converged_prepass"


def test_wiring_verdict_at_step6_actual_a2_data():
    """The deck-baseline A2 pre-pass: c_K so Boltzmann-piled-up at V_kin
    that even R_inj=10 produces only ~0.5% Δc.  Sign correct + monotonic
    ⇒ wiring verified; magnitude unreachable at sentinel R_inj.

    Numbers cross-checked against the committed step 6 record
    (StudyResults/phase6b_step6_plumbing_ablation/ablation_matrix.json).
    """
    rows = [
        _mk_prepass_signed(1e-1, -3.685e-05),
        _mk_prepass_signed(1.0,  -4.907e-04),
        _mk_prepass_signed(2.0,  -9.949e-04),
        _mk_prepass_signed(5.0,  -2.508e-03),
        _mk_prepass_signed(10.0, -5.031e-03),
    ]
    v = _verify_wiring_from_prepass("A2", rows)
    assert v["sign_correct_at_largest_R_inj"] is True
    assert v["monotonic_in_R_inj"] is True
    assert v["largest_R_inj"] == pytest.approx(10.0)
    # |Δc| at R_inj=10 ≈ 0.5% — well under the 5% gate.
    assert v["largest_abs_delta"] < DELTA_C_MIN
