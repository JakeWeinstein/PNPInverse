"""Phase 6β step 10 Phase D — orchestrator pure-Python helper tests.

Covers the bracket-construction, loss-extraction, identifiability gate,
σ-mapping divergence, and outcome-verdict logic in
``scripts.studies.drivers.phase6b_step10_phase_D_orchestrate``.

All tests in this module are fast (no Firedrake).  Slow integration
tests for the full 10.B pipeline live in the orchestrator's CLI
``main()`` and are exercised manually as part of the Phase D run.
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


# ===================================================================
# Bracket construction
# ===================================================================


def test_stern_delta_beta_for_target_recovers_T():
    """Round-trip: Δ_β = (T - β·σ)/σ ⇒ T = (β + Δ_β)·σ."""
    from scripts.studies.drivers.phase6b_step10_phase_D_orchestrate import (
        stern_delta_beta_for_target,
    )
    beta_K_Cu = -45.608196
    sigma_max = 1e-7
    for T in (-5.0, -3.0, -1.0, -0.1, -0.01, -0.001, -1e-4):
        db = stern_delta_beta_for_target(
            target_dpka=T, beta_K_Cu=beta_K_Cu, sigma_max=sigma_max,
        )
        T_recovered = (beta_K_Cu + db) * sigma_max
        assert T_recovered == pytest.approx(T, rel=1e-9)


def test_ablation_delta_beta_for_target_recovers_T():
    from scripts.studies.drivers.phase6b_step10_phase_D_orchestrate import (
        ablation_delta_beta_for_target, ABLATION_SIGMA,
    )
    beta_K_Cu = -45.608196
    for T in (-14.9, -10.0, -8.0, -4.0, -1.0, -0.1):
        db = ablation_delta_beta_for_target(
            target_dpka=T, beta_K_Cu=beta_K_Cu, sigma=ABLATION_SIGMA,
        )
        T_recovered = (beta_K_Cu + db) * ABLATION_SIGMA
        assert T_recovered == pytest.approx(T, rel=1e-9)


def test_stern_delta_beta_for_target_rejects_zero_sigma():
    from scripts.studies.drivers.phase6b_step10_phase_D_orchestrate import (
        stern_delta_beta_for_target,
    )
    with pytest.raises(ZeroDivisionError):
        stern_delta_beta_for_target(
            target_dpka=-1.0, beta_K_Cu=-45.0, sigma_max=0.0,
        )


# ===================================================================
# Loss extraction
# ===================================================================


def test_loss_from_eval_finite_valid():
    from scripts.studies.drivers.phase6b_step10_phase_D_orchestrate import (
        loss_from_eval,
    )
    eval_data = {
        "n_gate_fail": 0,
        "sign_guard": {"status": "ok"},
        "aggregated_observables": {
            "max_H2O2_selectivity_in_window_pct": 45.0,
        },
    }
    loss, status = loss_from_eval(eval_data, deck_target_pct=50.0)
    assert loss == pytest.approx(5.0)
    assert status == "finite_valid"


def test_loss_from_eval_solve_failed():
    from scripts.studies.drivers.phase6b_step10_phase_D_orchestrate import (
        loss_from_eval,
    )
    eval_data = {
        "n_gate_fail": 3,
        "sign_guard": {"status": "ok"},
        "aggregated_observables": {
            "max_H2O2_selectivity_in_window_pct": 45.0,
        },
        "per_v_gate_results": [
            {"status": "newton_unconverged"},
        ],
    }
    loss, status = loss_from_eval(eval_data, deck_target_pct=50.0)
    assert math.isinf(loss)
    assert status == "solve_failed"


def test_loss_from_eval_pka_shift_overflow_priority():
    """When n_gate_fail > 0 AND any V status is pka_shift_overflow,
    the eval status is `pka_shift_overflow` (more specific than
    solve_failed).
    """
    from scripts.studies.drivers.phase6b_step10_phase_D_orchestrate import (
        loss_from_eval,
    )
    eval_data = {
        "n_gate_fail": 1,
        "sign_guard": {"status": "ok"},
        "aggregated_observables": {
            "max_H2O2_selectivity_in_window_pct": 45.0,
        },
        "per_v_gate_results": [
            {"status": "pka_shift_overflow"},
            {"status": "ok"},
        ],
    }
    loss, status = loss_from_eval(eval_data, deck_target_pct=50.0)
    assert math.isinf(loss)
    assert status == "pka_shift_overflow"


def test_loss_from_eval_sign_guard_violation_priority():
    """sign_guard_violation takes precedence over solve_failed."""
    from scripts.studies.drivers.phase6b_step10_phase_D_orchestrate import (
        loss_from_eval,
    )
    eval_data = {
        "n_gate_fail": 5,
        "sign_guard": {"status": "violation"},
        "aggregated_observables": {
            "max_H2O2_selectivity_in_window_pct": 45.0,
        },
    }
    loss, status = loss_from_eval(eval_data, deck_target_pct=50.0)
    assert math.isinf(loss)
    assert status == "sign_guard_violation"


def test_loss_from_eval_no_selectivity():
    from scripts.studies.drivers.phase6b_step10_phase_D_orchestrate import (
        loss_from_eval,
    )
    eval_data = {
        "n_gate_fail": 0,
        "sign_guard": {"status": "ok"},
        "aggregated_observables": {
            "max_H2O2_selectivity_in_window_pct": None,
        },
    }
    loss, status = loss_from_eval(eval_data, deck_target_pct=50.0)
    assert math.isinf(loss)
    assert status == "no_selectivity"


# ===================================================================
# D7 identifiability gate
# ===================================================================


def test_d7_identifiability_gate_passes_clean_unimodal():
    """Synthetic 5-point loss curve that's unimodal + above noise +
    above slope threshold + above range threshold."""
    from scripts.studies.drivers.phase6b_step10_phase_D_orchestrate import (
        d7_identifiability_gate,
    )
    # Build evals at Δ_β values forming a unimodal loss curve.
    losses = [10.0, 5.0, 1.0, 4.0, 9.0]  # min at index 2
    delta_betas = [-100.0, -50.0, 0.0, +50.0, +100.0]
    evals = [
        {"delta_beta_pm2": db, "loss": l, "status": "finite_valid"}
        for db, l in zip(delta_betas, losses)
    ]
    result = d7_identifiability_gate(
        evals, noise_std=0.1, sigma_max=1.0, beta_K_Cu=-45.608196,
    )
    assert result["overall_pass"] is True
    assert result["criteria"]["range"]["passes"] is True
    assert result["criteria"]["unimodality"]["passes"] is True


def test_d7_identifiability_gate_fails_flat_loss():
    """Flat loss curve (range < 1 pp²) → fails range criterion."""
    from scripts.studies.drivers.phase6b_step10_phase_D_orchestrate import (
        d7_identifiability_gate,
    )
    evals = [
        {"delta_beta_pm2": db, "loss": 5.0 + 0.1 * i,
         "status": "finite_valid"}
        for i, db in enumerate([-100.0, -50.0, 0.0, +50.0, +100.0])
    ]
    result = d7_identifiability_gate(
        evals, noise_std=0.0, sigma_max=1.0, beta_K_Cu=-45.608196,
    )
    assert result["overall_pass"] is False
    assert result["criteria"]["range"]["passes"] is False


def test_d7_identifiability_gate_fails_noise_floor():
    """Range > 1 pp² but < 3·noise_std → fails noise criterion."""
    from scripts.studies.drivers.phase6b_step10_phase_D_orchestrate import (
        d7_identifiability_gate,
    )
    losses = [10.0, 8.0, 7.5, 8.0, 10.0]  # delta = 2.5
    delta_betas = [-100.0, -50.0, 0.0, +50.0, +100.0]
    evals = [
        {"delta_beta_pm2": db, "loss": l, "status": "finite_valid"}
        for db, l in zip(delta_betas, losses)
    ]
    result = d7_identifiability_gate(
        evals, noise_std=2.0,  # 3*noise=6.0 > 2.5
        sigma_max=1.0, beta_K_Cu=-45.608196,
    )
    assert result["criteria"]["noise_floor"]["passes"] is False
    assert result["overall_pass"] is False


def test_d7_identifiability_gate_fails_multi_modal():
    """Bi-modal loss (2 interior minima) → fails unimodality."""
    from scripts.studies.drivers.phase6b_step10_phase_D_orchestrate import (
        d7_identifiability_gate,
    )
    losses = [10.0, 2.0, 8.0, 2.5, 10.0]  # 2 interior minima at idx 1, 3
    delta_betas = [-100.0, -50.0, 0.0, +50.0, +100.0]
    evals = [
        {"delta_beta_pm2": db, "loss": l, "status": "finite_valid"}
        for db, l in zip(delta_betas, losses)
    ]
    result = d7_identifiability_gate(
        evals, noise_std=0.0, sigma_max=1.0, beta_K_Cu=-45.608196,
    )
    assert result["criteria"]["unimodality"]["passes"] is False
    assert result["criteria"]["unimodality"]["interior_minima_count"] == 2
    assert result["overall_pass"] is False


def test_d7_identifiability_gate_too_few_evals():
    from scripts.studies.drivers.phase6b_step10_phase_D_orchestrate import (
        d7_identifiability_gate,
    )
    result = d7_identifiability_gate(
        [{"delta_beta_pm2": 0.0, "loss": 5.0, "status": "finite_valid"}],
        noise_std=0.0, sigma_max=1.0, beta_K_Cu=-45.608196,
    )
    assert result["overall_pass"] is False


# ===================================================================
# σ-mapping divergence
# ===================================================================


def test_sigma_divergence_within_threshold_ok():
    from scripts.studies.drivers.phase6b_step10_phase_D_orchestrate import (
        sigma_mapping_divergence,
    )
    div = sigma_mapping_divergence(
        delta_beta_fit_stern=10.0,
        delta_beta_fit_ablation=12.0,
    )
    # |10-12|/max(10,12) = 2/12 = 0.1667 < 0.30
    assert div["flag"] == "ok"
    assert div["rel_divergence"] == pytest.approx(2.0 / 12.0)


def test_sigma_divergence_above_threshold_flagged():
    from scripts.studies.drivers.phase6b_step10_phase_D_orchestrate import (
        sigma_mapping_divergence,
    )
    div = sigma_mapping_divergence(
        delta_beta_fit_stern=10.0,
        delta_beta_fit_ablation=1e6,  # huge difference (expected per Risk #4)
    )
    assert div["flag"] == "non_identifiable"


def test_sigma_divergence_zero_zero_returns_ok():
    from scripts.studies.drivers.phase6b_step10_phase_D_orchestrate import (
        sigma_mapping_divergence,
    )
    div = sigma_mapping_divergence(
        delta_beta_fit_stern=0.0, delta_beta_fit_ablation=0.0,
    )
    assert div["flag"] == "ok"


# ===================================================================
# Outcome verdict
# ===================================================================


def test_determine_outcome_A_locked_pass():
    from scripts.studies.drivers.phase6b_step10_phase_D_orchestrate import (
        determine_outcome,
    )
    evals = [
        {"delta_beta_pm2": 0.0, "loss": 3.0, "status": "finite_valid"},
        {"delta_beta_pm2": 50.0, "loss": 7.0, "status": "finite_valid"},
    ]
    out = determine_outcome(
        loss_finite_valid_stern=evals,
        deck_target_pct=50.0,
        d7_pass=True,
        sign_guard_status_at_min="ok",
    )
    assert out["verdict"] == "OUTCOME_A_LOCKED_PASS"
    assert out["primary_pass"] is True


def test_determine_outcome_B_falsified_primary():
    from scripts.studies.drivers.phase6b_step10_phase_D_orchestrate import (
        determine_outcome,
    )
    evals = [
        {"delta_beta_pm2": 0.0, "loss": 25.0, "status": "finite_valid"},
        {"delta_beta_pm2": 50.0, "loss": 30.0, "status": "finite_valid"},
    ]
    out = determine_outcome(
        loss_finite_valid_stern=evals,
        deck_target_pct=50.0,
        d7_pass=True,
        sign_guard_status_at_min="ok",
    )
    assert out["verdict"] == "OUTCOME_B_FALSIFIED_documented"
    assert out["primary_pass"] is False
    assert out["reason"] == "primary_gate_exceeded"


def test_determine_outcome_B_falsified_sign_guard():
    from scripts.studies.drivers.phase6b_step10_phase_D_orchestrate import (
        determine_outcome,
    )
    evals = [
        {"delta_beta_pm2": 0.0, "loss": 3.0, "status": "finite_valid"},
    ]
    out = determine_outcome(
        loss_finite_valid_stern=evals,
        deck_target_pct=50.0,
        d7_pass=True,
        sign_guard_status_at_min="violation",
    )
    assert out["verdict"] == "OUTCOME_B_FALSIFIED_documented"
    assert out["reason"] == "sign_guard_violation"


def test_determine_outcome_C_non_identifiable():
    from scripts.studies.drivers.phase6b_step10_phase_D_orchestrate import (
        determine_outcome,
    )
    out = determine_outcome(
        loss_finite_valid_stern=[],
        deck_target_pct=50.0,
        d7_pass=False,
        sign_guard_status_at_min="not_evaluated",
    )
    assert out["verdict"] == "OUTCOME_C_NON_IDENTIFIABLE_flagged"


# ===================================================================
# filter_finite_valid
# ===================================================================


def test_filter_finite_valid():
    from scripts.studies.drivers.phase6b_step10_phase_D_orchestrate import (
        filter_finite_valid,
    )
    evals = [
        {"delta_beta_pm2": 0.0, "loss": 3.0, "status": "finite_valid"},
        {"delta_beta_pm2": 1.0, "loss": float("inf"),
         "status": "solve_failed"},
        {"delta_beta_pm2": 2.0, "loss": float("inf"),
         "status": "pka_shift_overflow"},
        {"delta_beta_pm2": 3.0, "loss": 5.0, "status": "finite_valid"},
    ]
    finite = filter_finite_valid(evals)
    assert len(finite) == 2
    assert all(e["status"] == "finite_valid" for e in finite)
