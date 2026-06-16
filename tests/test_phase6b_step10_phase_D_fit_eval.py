"""Phase 6β step 10 — Phase D driver tests (D3 spec items #1–10).

Pure-Python tests for the Δ_β fit-evaluator driver: lazy Firedrake
imports, CLI parsing, V grid lock, observable mask, ring-onset
interpolation, selectivity formula, n_e formula, max-ring extraction,
unit conversions, NaN-safe aggregation.

Slow Firedrake-backed tests (per-eval forward solve) are NOT in this
file; see ``scripts/studies/phase6b_step10_phase_D_fit_eval.py``'s
``main()`` for the integration entry point.
"""
from __future__ import annotations

import io
import math
import os
import subprocess
import sys
import textwrap

import pytest


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ===================================================================
# D3 #1 — Module is import-clean (no Firedrake at top level)
# ===================================================================


def test_step10_fit_eval_module_firedrake_free():
    """The Phase D driver module must be importable without Firedrake.

    Verified by a clean-room subprocess.  Sub-imports of v10a' helpers
    that DO need Firedrake stay lazy (inside function bodies).
    """
    script = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {_ROOT!r})
        # Whitelist scipy/numpy that may be brought in by argparse defaults.
        import scripts.studies.drivers.phase6b_step10_phase_D_fit_eval as drv
        bad = sorted(
            m for m in sys.modules
            if m == "firedrake" or m.startswith("firedrake.")
            or m == "Forward.bv_solver"
            or m.startswith("Forward.bv_solver.")
        )
        assert not bad, f"driver import pulled Firedrake-side modules: {{bad!r}}"
        print("OK", len(drv.V_RHE_PRODUCTION_GRID))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, (
        f"Firedrake-free import failed:\nstdout: {result.stdout!r}\n"
        f"stderr: {result.stderr!r}"
    )
    assert result.stdout.startswith("OK"), result.stdout


# ===================================================================
# D3 #2 — CLI parses without crashing
# ===================================================================


def test_step10_fit_eval_cli_parses_minimum():
    from scripts.studies.drivers.phase6b_step10_phase_D_fit_eval import _parse_args

    args = _parse_args(["--delta-beta", "0.0"])
    assert args.delta_beta == 0.0
    assert args.sigma_mapping == "stern"
    assert args.mode == "production"


def test_step10_fit_eval_cli_parses_all_flags():
    from scripts.studies.drivers.phase6b_step10_phase_D_fit_eval import _parse_args

    args = _parse_args([
        "--delta-beta", "-12.345",
        "--sigma-mapping", "ablation_singh_0.141",
        "--out-subdir", "custom_dir",
        "--out-name", "custom.json",
        "--plot",
        "--mode", "a2_reproduction",
    ])
    assert args.delta_beta == -12.345
    assert args.sigma_mapping == "ablation_singh_0.141"
    assert args.out_subdir == "custom_dir"
    assert args.out_name == "custom.json"
    assert args.plot is True
    assert args.mode == "a2_reproduction"


def test_step10_fit_eval_cli_rejects_unknown_sigma_mapping():
    from scripts.studies.drivers.phase6b_step10_phase_D_fit_eval import _parse_args

    with pytest.raises(SystemExit):
        _parse_args(["--delta-beta", "0.0", "--sigma-mapping", "weird"])


# ===================================================================
# D3 #3 — V grid locked
# ===================================================================


def test_step10_fit_eval_v_grid_locked():
    """Production V grid: 24 unique points, V_KIN and +1.00 present."""
    from scripts.studies.drivers.phase6b_step10_phase_D_fit_eval import (
        V_RHE_PRODUCTION_GRID,
    )

    assert len(V_RHE_PRODUCTION_GRID) == 24
    assert -0.10 in V_RHE_PRODUCTION_GRID
    assert 1.00 in V_RHE_PRODUCTION_GRID
    # No duplicates (within float-precision tolerance).
    rounded = [round(v, 6) for v in V_RHE_PRODUCTION_GRID]
    assert len(set(rounded)) == 24


def test_step10_fit_eval_v_grid_a2_warm():
    """A.2 reproduction warm grid: 5 points, V_KIN at the end."""
    from scripts.studies.drivers.phase6b_step10_phase_D_fit_eval import A2_WARM_GRID

    assert len(A2_WARM_GRID) == 5
    assert A2_WARM_GRID[-1] == -0.10
    assert A2_WARM_GRID[0] == +0.55  # V_anchor


# ===================================================================
# D3 #4 — Observable mask excludes V_KIN
# ===================================================================


def test_step10_fit_eval_observable_mask_excludes_v_kin():
    from scripts.studies.drivers.phase6b_step10_phase_D_fit_eval import (
        in_observable_mask,
        V_KIN_BYTE_EQUIV_BASELINE,
    )

    assert not in_observable_mask(V_KIN_BYTE_EQUIV_BASELINE)
    assert not in_observable_mask(-0.10)
    assert in_observable_mask(-0.06)
    assert in_observable_mask(+1.00)
    assert in_observable_mask(+0.50)
    assert not in_observable_mask(+1.001)
    assert not in_observable_mask(-0.07)


# ===================================================================
# D3 #5 — Ring-onset interpolation
# ===================================================================


def test_step10_fit_eval_ring_onset_interp_synthetic_crossing():
    """Synthetic: gross_h2o2_current crosses 0.01 mA/cm² between
    V=+0.30 and V=+0.20.  Linear interp must reproduce the analytic
    crossing.
    """
    from scripts.studies.drivers.phase6b_step10_phase_D_fit_eval import (
        find_ring_onset_v,
    )

    v_values = [+0.50, +0.40, +0.30, +0.20, +0.10, +0.00]
    # Linear ramp: i(V) = -0.05 * (V - 0.50)  →  i(0.50)=0,
    # i(0.30)=0.01, i(0.20)=0.015, i(0.10)=0.02
    # Wait: I want a crossing inside the (0.30, 0.20) bracket.  Use:
    # i(V) at threshold 0.01: at V=0.30 → i=0.005, at V=0.20 → i=0.015
    ring_currents = [0.0, 0.0, 0.005, 0.015, 0.02, 0.025]
    onset = find_ring_onset_v(
        v_values=v_values, ring_currents_mA_cm2=ring_currents,
        threshold_mA_cm2=0.01,
    )
    # Descending sort: V=[0.50, 0.40, 0.30, 0.20, 0.10, 0.00].
    # First V where i ≥ 0.01: V=0.20 (i=0.015).
    # Crossing between (0.30, 0.005) and (0.20, 0.015):
    # frac = (0.01 - 0.005) / (0.015 - 0.005) = 0.5
    # v_cross = 0.30 + 0.5 * (0.20 - 0.30) = 0.30 - 0.05 = 0.25
    assert onset == pytest.approx(0.25, abs=1e-9)


def test_step10_fit_eval_ring_onset_no_crossing():
    """All currents below threshold → returns None."""
    from scripts.studies.drivers.phase6b_step10_phase_D_fit_eval import (
        find_ring_onset_v,
    )
    onset = find_ring_onset_v(
        v_values=[0.5, 0.3, 0.1],
        ring_currents_mA_cm2=[0.0, 0.001, 0.002],
        threshold_mA_cm2=0.01,
    )
    assert onset is None


def test_step10_fit_eval_ring_onset_already_above_threshold():
    """Most-anodic V already above threshold → returns that V."""
    from scripts.studies.drivers.phase6b_step10_phase_D_fit_eval import (
        find_ring_onset_v,
    )
    onset = find_ring_onset_v(
        v_values=[0.5, 0.3, 0.1],
        ring_currents_mA_cm2=[0.05, 0.10, 0.15],
        threshold_mA_cm2=0.01,
    )
    assert onset == pytest.approx(0.5, abs=1e-9)


def test_step10_fit_eval_adaptive_refinement_inserts_inside_bracket():
    """Adaptive refinement: 4 additional points spaced 0.01 V inside
    the bracket where the crossing happens.
    """
    from scripts.studies.drivers.phase6b_step10_phase_D_fit_eval import (
        adaptive_ring_onset_refinement_grid,
    )
    # Bracket (0.20, 0.30) (cathodic, anodic).  4 points at 0.01 V
    # spacing inside: 0.21, 0.22, 0.23, 0.24.
    v_values = [+0.50, +0.40, +0.30, +0.20, +0.10]
    ring_currents = [0.0, 0.0, 0.005, 0.015, 0.02]
    refine = adaptive_ring_onset_refinement_grid(
        v_values=v_values, ring_currents_mA_cm2=ring_currents,
        threshold_mA_cm2=0.01,
        n_refine=4, spacing_v=0.01,
    )
    assert refine == pytest.approx([0.21, 0.22, 0.23, 0.24], abs=1e-6)


# ===================================================================
# D3 #6 — Selectivity formula matches Ruggiero §2
# ===================================================================


def test_step10_fit_eval_selectivity_formula_matches_ruggiero():
    """``200 * (I_ring/N) / (|I_disk| + I_ring/N)``.  Synthetic check."""
    from scripts.studies.drivers.phase6b_step10_phase_D_fit_eval import (
        selectivity_h2o2_pct, N_COLLECTION,
    )
    # Pure-2e ORR limit: I_ring / N == |I_disk|, selectivity = 100%.
    cd = -1.0
    ring = abs(cd) * N_COLLECTION
    sel = selectivity_h2o2_pct(i_disk_mA_cm2=cd, i_ring_mA_cm2=ring)
    assert sel == pytest.approx(100.0)
    # Pure-4e ORR limit: I_ring = 0, selectivity = 0%.
    sel0 = selectivity_h2o2_pct(i_disk_mA_cm2=cd, i_ring_mA_cm2=0.0)
    assert sel0 == pytest.approx(0.0)
    # 50/50 case: I_ring / N = |I_disk|/3 ⇒ sel = 200·(1/3)/(1+1/3) = 50%
    sel50 = selectivity_h2o2_pct(
        i_disk_mA_cm2=cd, i_ring_mA_cm2=N_COLLECTION * abs(cd) / 3.0,
    )
    assert sel50 == pytest.approx(50.0)


def test_step10_fit_eval_selectivity_zero_currents_returns_zero():
    from scripts.studies.drivers.phase6b_step10_phase_D_fit_eval import (
        selectivity_h2o2_pct,
    )
    assert selectivity_h2o2_pct(i_disk_mA_cm2=0.0, i_ring_mA_cm2=0.0) == 0.0


def test_step10_fit_eval_selectivity_rejects_zero_n_collection():
    from scripts.studies.drivers.phase6b_step10_phase_D_fit_eval import (
        selectivity_h2o2_pct,
    )
    with pytest.raises(ValueError, match="n_collection must be > 0"):
        selectivity_h2o2_pct(
            i_disk_mA_cm2=-1.0, i_ring_mA_cm2=0.5, n_collection=0.0,
        )


# ===================================================================
# D3 #7 — n_e RRDE formula
# ===================================================================


def test_step10_fit_eval_n_e_rrde_synthetic():
    """``4 · |I_disk| / (|I_disk| + I_ring/N)``."""
    from scripts.studies.drivers.phase6b_step10_phase_D_fit_eval import (
        n_e_rrde, N_COLLECTION,
    )
    # Pure-2e: I_ring / N = |I_disk|, n_e = 4·|I_disk|/(2·|I_disk|) = 2.
    cd = -1.0
    n_e = n_e_rrde(
        i_disk_mA_cm2=cd, i_ring_mA_cm2=N_COLLECTION * abs(cd),
    )
    assert n_e == pytest.approx(2.0)
    # Pure-4e: I_ring = 0, n_e = 4.
    n_e_4 = n_e_rrde(i_disk_mA_cm2=cd, i_ring_mA_cm2=0.0)
    assert n_e_4 == pytest.approx(4.0)


def test_step10_fit_eval_n_e_zero_currents_returns_4():
    from scripts.studies.drivers.phase6b_step10_phase_D_fit_eval import n_e_rrde
    assert n_e_rrde(i_disk_mA_cm2=0.0, i_ring_mA_cm2=0.0) == 4.0


# ===================================================================
# D3 #8 — Max ring current extraction
# ===================================================================


def test_step10_fit_eval_max_ring_current_extraction():
    """Max ring current uses the **ring basis** field (matches deck's
    Brianna xlsx column 9 "Max Ring Current (mA/cm²)" convention).
    """
    from scripts.studies.drivers.phase6b_step10_phase_D_fit_eval import (
        aggregate_observables,
    )
    recs = [
        # V=-0.10 EXCLUDED (V_KIN); V=+1.05 EXCLUDED (out of mask)
        {"v_rhe": -0.10, "cd_mA_cm2": -2.0,
         "ring_current_ring_basis_mA_cm2": 0.336},  # = 1.5 * N
        {"v_rhe": +0.30, "cd_mA_cm2": -1.0,
         "ring_current_ring_basis_mA_cm2": 0.0672},  # = 0.3 * N
        {"v_rhe": +0.60, "cd_mA_cm2": -0.5,
         "ring_current_ring_basis_mA_cm2": 0.0896},  # = 0.4 * N
        {"v_rhe": +1.05, "cd_mA_cm2": -0.1,
         "ring_current_ring_basis_mA_cm2": 0.222},   # = 0.99 * N
    ]
    agg = aggregate_observables(recs)
    # Max ring (ring-basis) inside mask: max(0.0672, 0.0896) = 0.0896
    assert agg["max_ring_current_in_window_mA_cm2"] == pytest.approx(
        0.0896, abs=1e-6,
    )


# ===================================================================
# D3 #9 — Unit conversions signed identity
# ===================================================================


def test_step10_fit_eval_unit_conversions_signed_identity():
    """signed_sigma_C_m2 ↔ signed_sigma_counts_pm2 round-trip."""
    from scripts.studies.drivers.phase6b_step10_phase_D_fit_eval import (
        signed_sigma_C_m2_to_counts_pm2,
        cathodic_clamped_sigma_singh,
    )
    # Round-trip: -1.0 C/m² → counts_per_pm2 → invert.
    factor = signed_sigma_C_m2_to_counts_pm2(1.0)
    # Expected factor: 6.2415e-6 counts/pm² per C/m² (1/e * 1e-24)
    expected_factor = (1.0 / 1.602176634e-19) * 1.0e-24
    assert factor == pytest.approx(expected_factor, rel=1e-12)
    # Cathodic clamp: -0.02 C/m² (cathodic, σ_S<0) → positive Singh σ
    # = +0.02 * 6.2415e-6 = 1.2483e-7 counts/pm²
    sigma_sign_clamp = cathodic_clamped_sigma_singh(-0.02)
    assert sigma_sign_clamp == pytest.approx(0.02 * expected_factor)
    # Anodic clamp: +0.02 C/m² → 0
    sigma_anode = cathodic_clamped_sigma_singh(+0.02)
    assert sigma_anode == 0.0


# ===================================================================
# D3 #10 — NaN-safe aggregation (pure synthetic; production V-fail
# path does NOT skip — it invalidates via the per-V gate)
# ===================================================================


def test_step10_fit_eval_nan_aggregation_synthetic_skips_nans():
    """Pure-synthetic NaN-skip: aggregation must SKIP records with
    NaN currents inside the mask without raising.
    """
    from scripts.studies.drivers.phase6b_step10_phase_D_fit_eval import (
        aggregate_observables,
    )
    nan = float("nan")
    recs = [
        # V=+0.30: cd=-1, ring (ring-basis) = 0.3 * N (= 0.0672) → I_disk_2e = 0.3
        # selectivity = 200·0.3/(1 + 0.3) = 46.15%
        {"v_rhe": +0.30, "cd_mA_cm2": -1.0,
         "ring_current_ring_basis_mA_cm2": 0.0672},
        {"v_rhe": +0.60, "cd_mA_cm2": nan,
         "ring_current_ring_basis_mA_cm2": 0.0896},  # NaN cd → SKIP
        {"v_rhe": +0.90, "cd_mA_cm2": -0.5,
         "ring_current_ring_basis_mA_cm2": nan},      # NaN ring → SKIP for sel
    ]
    agg = aggregate_observables(recs)
    assert agg["n_records_in_mask"] == 3
    # Only V=+0.30 has both finite currents.
    assert agg["n_records_with_finite_currents"] == 1
    expected = 200.0 * (0.0672 / 0.224) / (1.0 + 0.0672 / 0.224)
    assert agg["max_H2O2_selectivity_in_window_pct"] == pytest.approx(
        expected, rel=1e-4,
    )


def test_step10_fit_eval_per_v_gate_status_ok():
    """Plan §D2 HARD per-V gate evaluator — the happy path."""
    from scripts.studies.drivers.phase6b_step10_phase_D_fit_eval import (
        per_v_gate_status,
    )
    rec = {
        "snes_converged": True, "picard_status": "converged",
        "mass_balance_residual_rel": 1e-6,
        "analytic_gamma_rel": 1e-6,
        "pka_shift_avg": -1.0,
    }
    passes, status = per_v_gate_status(rec)
    assert passes
    assert status == "ok"


def test_step10_fit_eval_per_v_gate_pka_shift_overflow():
    from scripts.studies.drivers.phase6b_step10_phase_D_fit_eval import (
        per_v_gate_status,
        GATE_PKA_SHIFT_OVERFLOW,
    )
    rec = {
        "snes_converged": True, "picard_status": "converged",
        "mass_balance_residual_rel": 1e-6,
        "analytic_gamma_rel": 1e-6,
        "pka_shift_avg": -(GATE_PKA_SHIFT_OVERFLOW + 1.0),
    }
    passes, status = per_v_gate_status(rec)
    assert not passes
    assert status == "pka_shift_overflow"


def test_step10_fit_eval_per_v_gate_mass_balance_fail():
    from scripts.studies.drivers.phase6b_step10_phase_D_fit_eval import (
        per_v_gate_status,
    )
    rec = {
        "snes_converged": True, "picard_status": "converged",
        "mass_balance_residual_rel": 0.1,
    }
    passes, status = per_v_gate_status(rec)
    assert not passes
    assert status == "mass_balance_high"


def test_step10_fit_eval_per_v_gate_newton_unconverged():
    from scripts.studies.drivers.phase6b_step10_phase_D_fit_eval import (
        per_v_gate_status,
    )
    passes, status = per_v_gate_status({"snes_converged": False})
    assert not passes
    assert status == "newton_unconverged"


# ===================================================================
# Predict pka_shift_max — pre-solve domain check (plan §D6)
# ===================================================================


def test_step10_fit_eval_predict_pka_shift_max():
    """``|pka_shift_max| = |β_K_Cu + Δ_β| · σ_max``."""
    from scripts.studies.drivers.phase6b_step10_phase_D_fit_eval import (
        predict_pka_shift_max,
    )
    # Δ_β = 0, σ_max = 1e-7 → |β_K_Cu| · σ_max ≈ 4.56e-6 (small)
    val = predict_pka_shift_max(
        beta_K_Cu=-45.608196, delta_beta=0.0, sigma_max=1e-7,
    )
    assert val == pytest.approx(45.608196 * 1e-7, rel=1e-9)
    # Δ_β = +1e8, σ_max = 1e-7 → ~10 (well below 15)
    val2 = predict_pka_shift_max(
        beta_K_Cu=-45.608196, delta_beta=1e8, sigma_max=1e-7,
    )
    assert val2 == pytest.approx(abs(-45.608196 + 1e8) * 1e-7, rel=1e-9)
