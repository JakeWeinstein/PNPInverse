"""Phase 6 beta v10b bracket + matrix driver fast-only unit tests.

Verifies that the v10b C_S bracket and Gamma_max x k_des matrix
drivers:

* Are importable without Firedrake (CLI / schema layer only at module
  top level).
* Parse the documented CLI flags.
* Enumerate the right target grid.
* Emit the right output schema.

All tests are fast (no Firedrake).  Slow integration runs land in the
solver-running ``main()`` invocation.
"""
from __future__ import annotations

import os
import subprocess
import sys
from typing import Any, Dict, List

import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ===================================================================
# Firedrake-free module top level (per Round 6 + Round 7 patch P38)
# ===================================================================


def _check_module_firedrake_free(module_dotted: str) -> None:
    """Import the module in a fresh subprocess and verify firedrake
    + Forward.bv_solver are NOT in sys.modules afterwards."""
    script = (
        "import sys; "
        "sys.path.insert(0, r'%s'); "
        "import %s; "  # noqa: S608 -- internal use only, no untrusted input
        "assert 'firedrake' not in sys.modules, "
        "    'module pulled firedrake into sys.modules'; "
        "assert 'Forward.bv_solver' not in sys.modules, "
        "    'module pulled Forward.bv_solver into sys.modules'; "
        "print('OK')"
    ) % (_ROOT, module_dotted)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, (
        f"firedrake-free import failed for {module_dotted}:\n"
        f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
    )


def test_phase6b_v10b_cs_bracket_module_firedrake_free():
    _check_module_firedrake_free(
        "scripts.studies.drivers.phase6b_v10b_cs_bracket"
    )


def test_phase6b_v10b_gamma_kdes_matrix_module_firedrake_free():
    _check_module_firedrake_free(
        "scripts.studies.drivers.phase6b_v10b_gamma_kdes_matrix"
    )


# ===================================================================
# C_S bracket driver CLI / target grid / schema
# ===================================================================


def test_phase6b_v10b_cs_bracket_cli_parses():
    from scripts.studies.drivers.phase6b_v10b_cs_bracket import _parse_args

    args = _parse_args([
        "--v-kin", "-0.10",
        "--v-anchor", "0.55",
        "--k0-r4e-factor", "1e-14",
        "--k-hyd", "1e-3",
        "--cs-bracket", "0.05,0.10,0.20,0.30",
        "--out-subdir", "phase6b_v10b_cs_bracket",
    ])
    assert args.v_kin == pytest.approx(-0.10, rel=1e-12)
    assert args.k_hyd == pytest.approx(1e-3, rel=1e-12)
    assert args.cs_bracket == "0.05,0.10,0.20,0.30"
    assert args.out_subdir == "phase6b_v10b_cs_bracket"


def test_phase6b_v10b_cs_bracket_target_grid():
    """The default C_S bracket is the 4-rung Pillai-safe-band set."""
    from scripts.studies.drivers.phase6b_v10b_cs_bracket import CS_BRACKET, _parse_cs_bracket

    assert CS_BRACKET == (0.05, 0.10, 0.20, 0.30)
    parsed = _parse_cs_bracket("0.05,0.10,0.20,0.30")
    assert parsed == (0.05, 0.10, 0.20, 0.30)


def test_phase6b_v10b_cs_bracket_rejects_invalid_entries():
    from scripts.studies.drivers.phase6b_v10b_cs_bracket import _parse_cs_bracket

    with pytest.raises(ValueError, match="positive"):
        _parse_cs_bracket("0.05,-0.10,0.20")
    with pytest.raises(ValueError, match="positive"):
        _parse_cs_bracket("0.05,0.0,0.20")
    with pytest.raises(ValueError, match="at least one entry"):
        _parse_cs_bracket("")


def test_phase6b_v10b_cs_bracket_output_schema():
    """The hard-gate evaluator emits all required schema keys."""
    from scripts.studies.drivers.phase6b_v10b_cs_bracket import _evaluate_hard_gates

    lam1: Dict[str, Any] = {
        "cd_mA_cm2": -3.12,
        "R_4e_current_nondim": 1e-9,
        "R_net": 0.045,
        "gamma": 0.045,
    }
    gates = _evaluate_hard_gates(
        lam1=lam1, analytic_gamma_clamped=0.0451,
    )
    required = {
        "cd_ok", "cd_mA_cm2", "r4_sign_ok", "r4_note",
        "R_4e_current_nondim", "r_net_ok", "r_net_note", "R_net",
        "analytic_solver_gamma_rel", "analytic_solver_gamma_ok",
        "gamma_max_rel_threshold", "pass",
    }
    assert required <= set(gates.keys())
    assert isinstance(gates["pass"], bool)


# ===================================================================
# Gamma_max x k_des matrix driver CLI / target grid / schema
# ===================================================================


def test_phase6b_v10b_gamma_kdes_matrix_cli_parses():
    from scripts.studies.drivers.phase6b_v10b_gamma_kdes_matrix import _parse_args

    args = _parse_args([
        "--v-kin", "-0.10",
        "--v-anchor", "0.55",
        "--k0-r4e-factor", "1e-14",
        "--k-des-bracket", "0.01,0.1,1.0,10.0,100.0",
        "--k-hyd-bracket", "1e-3,1e-1",
        "--gamma-max-ratios", "0.5,1.0,2.0",
        "--out-subdir", "phase6b_v10b_gamma_kdes_matrix",
    ])
    assert args.k_des_bracket == "0.01,0.1,1.0,10.0,100.0"
    assert args.k_hyd_bracket == "1e-3,1e-1"
    assert args.gamma_max_ratios == "0.5,1.0,2.0"


def test_phase6b_v10b_gamma_kdes_matrix_target_grid():
    """Per plan D7-D4: 3 Gamma_max x 5 k_des x 2 k_hyd = 30 rungs."""
    from scripts.studies.drivers.phase6b_v10b_gamma_kdes_matrix import (
        K_DES_BRACKET, K_HYD_BRACKET, GAMMA_MAX_RATIOS, _enumerate_rungs,
    )

    assert len(K_DES_BRACKET) == 5
    assert len(K_HYD_BRACKET) == 2
    assert len(GAMMA_MAX_RATIOS) == 3

    rungs = _enumerate_rungs(
        k_des_bracket=K_DES_BRACKET,
        k_hyd_bracket=K_HYD_BRACKET,
        gamma_max_values=(0.0235, 0.047, 0.094),
    )
    assert len(rungs) == 30, (
        f"D7-D4 must enumerate 30 rungs; got {len(rungs)}"
    )
    # First entry has the smallest gamma_max and the first k_des, k_hyd.
    g0, kd0, kh0 = rungs[0]
    assert g0 == pytest.approx(0.0235, rel=1e-12)
    assert kd0 == pytest.approx(0.01, rel=1e-12)
    assert kh0 == pytest.approx(1e-3, rel=1e-12)
    # All triples are unique.
    assert len(set(rungs)) == 30


def test_phase6b_v10b_gamma_kdes_matrix_output_schema():
    """The matrix hard-gate evaluator emits the same schema as the
    C_S bracket evaluator."""
    from scripts.studies.drivers.phase6b_v10b_gamma_kdes_matrix import (
        _evaluate_hard_gates,
    )
    lam1: Dict[str, Any] = {
        "cd_mA_cm2": -3.12,
        "R_4e_current_nondim": 1e-9,
        "R_net": 0.045,
        "gamma": 0.045,
    }
    gates = _evaluate_hard_gates(
        lam1=lam1, analytic_gamma_clamped=0.0451,
    )
    required = {
        "cd_ok", "cd_mA_cm2", "r4_sign_ok", "r4_note",
        "R_4e_current_nondim", "r_net_ok", "r_net_note", "R_net",
        "analytic_solver_gamma_rel", "analytic_solver_gamma_ok",
        "gamma_max_rel_threshold", "pass",
    }
    assert required <= set(gates.keys())


# ===================================================================
# Sign-convention regression (Round 2 issue #1 + Round 3 issue #8)
# ===================================================================


def test_hard_gates_reject_positive_cd_at_v_kin():
    """cd_mA_cm2 must be negative at V_kin (cathodic).  Positive cd
    fails the gate."""
    from scripts.studies.drivers.phase6b_v10b_cs_bracket import _evaluate_hard_gates

    lam1 = {
        "cd_mA_cm2": 0.5,                         # WRONG sign
        "R_4e_current_nondim": 1e-9,
        "R_net": 0.045,
        "gamma": 0.045,
    }
    gates = _evaluate_hard_gates(
        lam1=lam1, analytic_gamma_clamped=0.0451,
    )
    assert gates["pass"] is False
    assert gates["cd_ok"] is False


def test_hard_gates_reject_negative_r4_above_floor():
    """R_4e_current_nondim < 0 with magnitude above floor fails the
    sign gate."""
    from scripts.studies.drivers.phase6b_v10b_cs_bracket import _evaluate_hard_gates

    lam1 = {
        "cd_mA_cm2": -3.12,
        "R_4e_current_nondim": -1e-3,             # WRONG sign + above floor
        "R_net": 0.045,
        "gamma": 0.045,
    }
    gates = _evaluate_hard_gates(
        lam1=lam1, analytic_gamma_clamped=0.0451,
    )
    assert gates["pass"] is False
    assert gates["r4_sign_ok"] is False


def test_hard_gates_skip_r4_sign_below_floor():
    """R_4e_current_nondim with |value| below R_4E_SIGN_FLOOR is N/A
    (does not fail the gate just because magnitude is tiny)."""
    from scripts.studies.drivers.phase6b_v10b_cs_bracket import _evaluate_hard_gates

    lam1 = {
        "cd_mA_cm2": -3.12,
        "R_4e_current_nondim": -1e-9,             # below floor
        "R_net": 0.045,
        "gamma": 0.045,
    }
    gates = _evaluate_hard_gates(
        lam1=lam1, analytic_gamma_clamped=0.0451,
    )
    assert gates["r4_sign_ok"] is True
    assert gates["r4_note"] == "below_sign_floor"


def test_hard_gates_reject_negative_r_net():
    """R_net < 0 fails (R_net = k_des * Gamma should be non-negative
    by construction)."""
    from scripts.studies.drivers.phase6b_v10b_cs_bracket import _evaluate_hard_gates

    lam1 = {
        "cd_mA_cm2": -3.12,
        "R_4e_current_nondim": 1e-9,
        "R_net": -0.001,                          # WRONG sign
        "gamma": 0.045,
    }
    gates = _evaluate_hard_gates(
        lam1=lam1, analytic_gamma_clamped=0.0451,
    )
    assert gates["pass"] is False
    assert gates["r_net_ok"] is False


def test_hard_gates_reject_high_mass_balance_residual():
    """|Gamma_solver - Gamma_analytic| / max(...) > 5e-3 fails."""
    from scripts.studies.drivers.phase6b_v10b_cs_bracket import _evaluate_hard_gates

    lam1 = {
        "cd_mA_cm2": -3.12,
        "R_4e_current_nondim": 1e-9,
        "R_net": 0.045,
        "gamma": 0.040,                            # 11% off the analytic
    }
    gates = _evaluate_hard_gates(
        lam1=lam1, analytic_gamma_clamped=0.045,
    )
    assert gates["pass"] is False
    assert gates["analytic_solver_gamma_ok"] is False
    assert gates["analytic_solver_gamma_rel"] is not None
    assert gates["analytic_solver_gamma_rel"] > 5e-3
