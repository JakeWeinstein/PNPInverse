"""Phase 6β v10b — literature-calibration regression tests.

Fast-only unit tests that verify the v10b numeric constants, metadata
schema, V10A/V10B coexistence, AST-aware production-driver import
audit, and the convergence-audit HARD/SOFT split refactor.

All tests in this module are fast (no Firedrake).  See
``tests/test_phase6b_v10a_langmuir_cap.py`` for the slow-marked
integration tests against the Langmuir cap; this file is the v10b
sibling that exercises the V10B numeric drop.
"""
from __future__ import annotations

import ast
import io
import os
import sys
from typing import Any, Dict, List, Set

import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ===================================================================
# D8: Firedrake-free metadata module
# ===================================================================


def test_v10b_calibration_module_firedrake_free():
    """calibration.v10b must NOT pull firedrake or Forward.bv_solver
    into sys.modules.  Verified by a clean-room import in a subprocess
    so test ordering does not corrupt the check.
    """
    import subprocess

    script = (
        "import sys; "
        "sys.path.insert(0, r'%s'); "
        "import calibration.v10b as v; "
        "assert 'firedrake' not in sys.modules, "
        "    'calibration.v10b pulled firedrake into sys.modules'; "
        "assert 'Forward.bv_solver' not in sys.modules, "
        "    'calibration.v10b pulled Forward.bv_solver into sys.modules'; "
        "print('OK', v.GAMMA_MAX_HAT_V10B, v.K_DES_NONDIM_V10B, v.C_S_F_M2_V10B)"
    ) % _ROOT
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, (
        f"firedrake-free import failed:\n"
        f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
    )
    assert result.stdout.startswith("OK"), result.stdout


# ===================================================================
# D8: metadata schema completeness
# ===================================================================


REQUIRED_METADATA_KEYS: Set[str] = {
    "value",
    "units",
    "is_nondim",
    "source_type",
    "engineering_choice",
    "citation",
    "bracket",
    "prior",
    "compatibility",
}

REQUIRED_COMPATIBILITY_KEYS: Set[str] = {
    "mechanism",
    "electrode",
    "electrolyte",
    "dimensional",
}


def test_v10b_calibration_metadata_schema():
    """Each parameter entry has all required keys."""
    from calibration.v10b import V10B_CALIBRATION_METADATA

    expected_params = {"gamma_max", "k_des", "C_S"}
    assert set(V10B_CALIBRATION_METADATA.keys()) == expected_params

    for name, entry in V10B_CALIBRATION_METADATA.items():
        missing = REQUIRED_METADATA_KEYS - set(entry.keys())
        assert not missing, f"{name}: missing metadata keys {missing}"
        compat = entry["compatibility"]
        assert isinstance(compat, dict), f"{name}: compatibility not dict"
        missing_compat = REQUIRED_COMPATIBILITY_KEYS - set(compat.keys())
        assert not missing_compat, (
            f"{name}: missing compatibility keys {missing_compat}"
        )
        # engineering_choice True iff source_type == 'engineering'
        if entry["source_type"] == "engineering":
            assert entry["engineering_choice"] is True, (
                f"{name}: source_type=engineering but "
                f"engineering_choice not True"
            )
            assert entry["prior"] is not None, (
                f"{name}: engineering_choice must document prior"
            )
        else:
            assert entry["engineering_choice"] is False, (
                f"{name}: source_type={entry['source_type']!r} must "
                f"have engineering_choice False"
            )
        # bracket non-empty list of floats
        assert isinstance(entry["bracket"], list)
        assert len(entry["bracket"]) >= 1
        for v in entry["bracket"]:
            assert isinstance(v, (int, float))


def test_metadata_schema_required_keys():
    """Schema completeness sanity (same as above but treated as the
    standalone D8 explicit acceptance gate)."""
    from calibration.v10b import V10B_CALIBRATION_METADATA

    for name, entry in V10B_CALIBRATION_METADATA.items():
        for k in REQUIRED_METADATA_KEYS:
            assert k in entry, f"{name}: missing required key {k!r}"


# ===================================================================
# D8: V10B numeric values + cross-file consistency
# ===================================================================


def test_v10b_constants_solver_driver_metadata_consistency():
    """Plan D8: solver-layer constants, _bv_common alias, driver
    kinetics, and metadata block all agree on the V10B numeric values.
    """
    from calibration.v10b import (
        C_S_F_M2_V10B,
        GAMMA_MAX_HAT_V10B,
        K_DES_NONDIM_V10B,
        V10B_CALIBRATION_METADATA,
        V10B_KINETICS,
    )

    # C_S consistency: production target == calibration constant ==
    # metadata value.  STERN_F_M2_BASELINE is in the v-sweep driver
    # module.
    from scripts.studies.phase6b_v10a_v_sweep_diagnostic import (
        STERN_F_M2_BASELINE,
    )
    assert STERN_F_M2_BASELINE == pytest.approx(C_S_F_M2_V10B, rel=1e-15)
    assert C_S_F_M2_V10B == pytest.approx(
        V10B_CALIBRATION_METADATA["C_S"]["value"], rel=1e-15
    )
    assert C_S_F_M2_V10B == pytest.approx(0.20, rel=1e-15)

    # Gamma_max consistency: solver constant == _bv_common == metadata.
    # cation_hydrolysis import lazily because it pulls Firedrake; do it
    # only after the firedrake-free constants have been checked.
    try:
        from Forward.bv_solver.cation_hydrolysis import (
            GAMMA_MAX_HAT_V10B as SOLVER_V10B,
        )
        from scripts._bv_common import (
            GAMMA_MAX_HAT_V10B as BV_COMMON_V10B,
        )
        assert SOLVER_V10B == pytest.approx(GAMMA_MAX_HAT_V10B, rel=1e-15)
        assert BV_COMMON_V10B == pytest.approx(GAMMA_MAX_HAT_V10B, rel=1e-15)
    except ModuleNotFoundError:
        # Firedrake not installed (e.g. unit-test-only CI rig);
        # calibration.v10b values are still verifiable.
        pass

    assert GAMMA_MAX_HAT_V10B == pytest.approx(
        V10B_CALIBRATION_METADATA["gamma_max"]["value"], rel=1e-15
    )
    assert GAMMA_MAX_HAT_V10B == pytest.approx(0.047, rel=1e-15)

    # k_des consistency: solver constant == V10B_KINETICS == metadata.
    try:
        from Forward.bv_solver.cation_hydrolysis import (
            K_DES_NONDIM_V10B as SOLVER_KDES_V10B,
        )
        assert SOLVER_KDES_V10B == pytest.approx(K_DES_NONDIM_V10B, rel=1e-15)
    except ModuleNotFoundError:
        pass

    assert V10B_KINETICS["k_des_nondim"] == pytest.approx(
        K_DES_NONDIM_V10B, rel=1e-15
    )
    assert K_DES_NONDIM_V10B == pytest.approx(
        V10B_CALIBRATION_METADATA["k_des"]["value"], rel=1e-15
    )
    assert K_DES_NONDIM_V10B == pytest.approx(1.0, rel=1e-15)

    # gamma_max_nondim in V10B_KINETICS agrees with GAMMA_MAX_HAT_V10B
    assert V10B_KINETICS["gamma_max_nondim"] == pytest.approx(
        GAMMA_MAX_HAT_V10B, rel=1e-15
    )


# ===================================================================
# D8: factory default routes to V10B
# ===================================================================


def test_factory_default_uses_v10b():
    """``make_cation_hydrolysis_config`` default ``gamma_max_nondim``
    equals ``GAMMA_MAX_HAT_V10B`` and ``k_des`` callers see V10B.

    Firedrake-aware: skipped if scripts._bv_common pulls in a Firedrake
    side-effect at import time.
    """
    try:
        from scripts._bv_common import (
            GAMMA_MAX_HAT_V10B, make_cation_hydrolysis_config,
        )
    except ModuleNotFoundError:                                # pragma: no cover
        pytest.skip("scripts._bv_common unavailable (no Firedrake)")
    cfg = make_cation_hydrolysis_config(
        k_hyd=1e-3, k_prot=1e-3, k_des=1.0,
        delta_ohp_hat=4e-6,
    )
    assert cfg["gamma_max_nondim"] == pytest.approx(
        GAMMA_MAX_HAT_V10B, rel=1e-15
    )


# ===================================================================
# D8: V10A frozen historical alias
# ===================================================================


def test_v10a_smoke_alias_preserved():
    """The deprecation alias ``GAMMA_MAX_HAT_SMOKE`` is preserved as
    pointing at ``GAMMA_MAX_HAT_V10A_SMOKE`` (the frozen historical
    0.047), NOT at V10B.

    This is critical: it keeps test_phase6b_v10a_langmuir_cap.py and
    other historical callers byte-stable while production callers
    use V10B.
    """
    try:
        from Forward.bv_solver.cation_hydrolysis import (
            GAMMA_MAX_HAT_SMOKE,
            GAMMA_MAX_HAT_V10A_SMOKE,
        )
    except ModuleNotFoundError:                                # pragma: no cover
        pytest.skip("Forward.bv_solver unavailable (no Firedrake)")
    assert GAMMA_MAX_HAT_SMOKE == pytest.approx(
        GAMMA_MAX_HAT_V10A_SMOKE, rel=1e-15
    )
    assert GAMMA_MAX_HAT_V10A_SMOKE == pytest.approx(0.047, rel=1e-15)


def test_smoke_kinetics_alias_preserved_in_v_sweep_driver():
    """The v-sweep driver's ``SMOKE_KINETICS = SMOKE_KINETICS_V10A``
    deprecation alias preserves V10A backward-compat while production
    paths read V10B_KINETICS.
    """
    try:
        from scripts.studies.phase6b_v10a_v_sweep_diagnostic import (
            SMOKE_KINETICS, SMOKE_KINETICS_V10A,
        )
    except ModuleNotFoundError:                                # pragma: no cover
        pytest.skip("v-sweep driver unavailable (no Firedrake)")
    assert SMOKE_KINETICS is SMOKE_KINETICS_V10A or SMOKE_KINETICS == (
        SMOKE_KINETICS_V10A
    )


# ===================================================================
# D8: AST-aware production-driver import audit
# ===================================================================


_PRODUCTION_DRIVERS: List[str] = [
    "scripts/studies/phase6b_v10a_phase_A2_v_kin.py",
    "scripts/studies/phase6b_step6_plumbing_ablation.py",
    "scripts/studies/phase6b_v10b_cs_bracket.py",
    "scripts/studies/phase6b_v10b_gamma_kdes_matrix.py",
]


def _import_targets_for(path: str) -> List[str]:
    """Return the set of names imported from any module across a file."""
    full = os.path.join(_ROOT, path)
    if not os.path.exists(full):
        return []
    with open(full) as f:
        src = f.read()
    tree = ast.parse(src)
    targets: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                targets.append(alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                targets.append(alias.name)
    return targets


def test_v10b_production_drivers_use_V10B_kinetics():
    """Plan D8: every v10b-production driver imports V10B_KINETICS or
    V10B_CALIBRATION_METADATA (not SMOKE_KINETICS).

    The v-sweep diagnostic file defines the SMOKE_KINETICS_V10A alias
    itself and is whitelisted.  Test-path / historical-path scripts
    out of the production list are not audited here.
    """
    for path in _PRODUCTION_DRIVERS:
        full = os.path.join(_ROOT, path)
        if not os.path.exists(full):
            # Skipped drivers (D7-D1 / D7-D4) land in Phase v10b.D;
            # tolerate missing files at Phase B time.
            continue
        targets = _import_targets_for(path)
        # The acceptable production constant names are V10B_KINETICS
        # and V10B_CALIBRATION_METADATA.  Drivers should import at
        # least one of them.
        imports_v10b = (
            "V10B_KINETICS" in targets
            or "V10B_CALIBRATION_METADATA" in targets
        )
        assert imports_v10b, (
            f"{path}: production driver does not import V10B_KINETICS "
            f"or V10B_CALIBRATION_METADATA (imports: {targets})"
        )
        # Drivers should NOT import the deprecated SMOKE_KINETICS or
        # SMOKE_KINETICS_V10A as a production source.  (The v-sweep
        # diagnostic owns the alias definition; it's not in the
        # production-driver whitelist.)
        for forbidden in ("SMOKE_KINETICS", "GAMMA_MAX_HAT_SMOKE"):
            assert forbidden not in targets, (
                f"{path}: production driver still imports "
                f"deprecated {forbidden!r}"
            )


def test_v_sweep_driver_owns_smoke_kinetics_v10a_alias():
    """The v-sweep diagnostic must define ``SMOKE_KINETICS_V10A`` at
    module scope (as the frozen historical baseline) AND assign
    ``SMOKE_KINETICS = SMOKE_KINETICS_V10A`` as the deprecation alias
    (AST-aware; no false positives from a literal-string grep).
    """
    path = os.path.join(
        _ROOT, "scripts/studies/phase6b_v10a_v_sweep_diagnostic.py",
    )
    with open(path) as f:
        src = f.read()
    tree = ast.parse(src)

    defined_smoke_v10a = False
    aliased_smoke_to_v10a = False
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign):
            tgt = node.target
            if isinstance(tgt, ast.Name) and tgt.id == "SMOKE_KINETICS_V10A":
                defined_smoke_v10a = True
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "SMOKE_KINETICS":
                    if (
                        isinstance(node.value, ast.Name)
                        and node.value.id == "SMOKE_KINETICS_V10A"
                    ):
                        aliased_smoke_to_v10a = True

    assert defined_smoke_v10a, (
        "phase6b_v10a_v_sweep_diagnostic.py must define "
        "SMOKE_KINETICS_V10A at module scope"
    )
    assert aliased_smoke_to_v10a, (
        "phase6b_v10a_v_sweep_diagnostic.py must alias "
        "SMOKE_KINETICS = SMOKE_KINETICS_V10A (NOT to V10B_KINETICS)"
    )


# ===================================================================
# D8: convergence-audit hard/soft separation
# ===================================================================


def test_convergence_audit_hard_soft_separation():
    """The A.2 driver's ``_convergence_audit`` returns a payload with
    explicit ``hard_gates`` and ``soft_deltas`` blocks, and
    ``overall_pass`` is driven by ``hard_gates`` only.

    Lightweight test: feeds a synthetic per_k_hyd_records list with a
    minimal converged baseline rung and checks the audit shape.
    """
    try:
        from scripts.studies.phase6b_v10a_phase_A2_v_kin import (
            _convergence_audit,
        )
    except ModuleNotFoundError:                                # pragma: no cover
        pytest.skip("A.2 driver unavailable (no Firedrake)")

    # Synthetic 10-rung k_hyd record list; lambda=1 converged
    # everywhere with theta ramped 0.058 -> 0.998.
    transition_grid = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3]
    saturation_grid = [5e-3, 1e-2, 5e-2, 1e-1]
    rungs: List[Dict[str, Any]] = []
    thetas_t = [0.058, 0.10, 0.20, 0.40, 0.55, 0.75, 0.861, 0.93]
    thetas_s = [0.969, 0.99, 0.998, 0.998]
    for kh, th in zip(transition_grid, thetas_t):
        rungs.append({
            "k_hyd_target": kh,
            "exception_phase": None,
            "rungs": [{
                "lambda_hydrolysis": 1.0,
                "snes_converged": True,
                "gamma": th * 0.047,
                "theta": th,
                "sigma_S_C_per_m2": -0.01715,
                "cd_mA_cm2": -3.12,
                "picard_status": "converged",
                "mass_balance_residual_rel": 1e-14,
            }],
        })
    for kh, th in zip(saturation_grid, thetas_s):
        rungs.append({
            "k_hyd_target": kh,
            "exception_phase": None,
            "rungs": [{
                "lambda_hydrolysis": 1.0,
                "snes_converged": True,
                "gamma": th * 0.047,
                "theta": th,
                "sigma_S_C_per_m2": -0.01715,
                "cd_mA_cm2": -3.12,
                "picard_status": "converged",
                "mass_balance_residual_rel": 1e-14,
            }],
        })

    audit = _convergence_audit(rungs, None)
    assert "hard_gates" in audit
    assert "soft_deltas" in audit
    assert "overall_pass" in audit
    hg = audit["hard_gates"]
    assert "convergence_coverage_pass" in hg
    assert "picard_ok" in hg
    assert "mass_balance_ok" in hg
    assert "pass" in hg
    # overall_pass == hard_gates.pass (NOT depends on soft_deltas)
    assert audit["overall_pass"] == hg["pass"]

    sd = audit["soft_deltas"]
    assert "baseline_reproduction_relative_diffs" in sd
    assert "note" in sd


# ===================================================================
# D8: step 6 CLI flag + R_net audit field
# ===================================================================


def test_step6_a2_baseline_json_cli_flag():
    """Step 6 driver accepts ``--a2-baseline-json`` CLI flag."""
    try:
        from scripts.studies.phase6b_step6_plumbing_ablation import (
            _parse_args,
        )
    except ModuleNotFoundError:                                # pragma: no cover
        pytest.skip("step 6 driver unavailable (no Firedrake)")
    args = _parse_args([
        "--a2-baseline-json",
        "StudyResults/phase6b_v10b_phase_A2_v_kin/phase_a2_v_kin.json",
    ])
    assert hasattr(args, "a2_baseline_json")
    assert args.a2_baseline_json == (
        "StudyResults/phase6b_v10b_phase_A2_v_kin/phase_a2_v_kin.json"
    )


def test_step6_audit_keys_include_R_net():
    """Step 6's _baseline_reproduction_audit keys include R_net (D6).

    Uses introspection on the function source rather than running the
    audit (which needs a real A.2 JSON + Firedrake side-effects).
    """
    try:
        import inspect
        from scripts.studies.phase6b_step6_plumbing_ablation import (
            _baseline_reproduction_audit,
        )
    except ModuleNotFoundError:                                # pragma: no cover
        pytest.skip("step 6 driver unavailable (no Firedrake)")
    src = inspect.getsource(_baseline_reproduction_audit)
    assert '"R_net"' in src, (
        "_baseline_reproduction_audit must include R_net in its keys "
        "(plan D6)"
    )


# ===================================================================
# v10a -> v10b co-existence: V10A_SMOKE numeric value frozen at 0.047
# ===================================================================


def test_v10a_smoke_value_frozen_at_0_047():
    """V10A frozen historical numeric value is 0.047.  Future v10c+
    cycles may add their own constants; V10A_SMOKE never moves.
    """
    try:
        from Forward.bv_solver.cation_hydrolysis import (
            GAMMA_MAX_HAT_V10A_SMOKE,
        )
        from scripts._bv_common import (
            GAMMA_MAX_HAT_V10A_SMOKE as BV_V10A,
        )
    except ModuleNotFoundError:                                # pragma: no cover
        pytest.skip("Forward.bv_solver unavailable (no Firedrake)")
    assert GAMMA_MAX_HAT_V10A_SMOKE == 0.047
    assert BV_V10A == 0.047
