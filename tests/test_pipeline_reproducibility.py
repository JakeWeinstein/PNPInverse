"""Pipeline reproducibility tests for the v13 inference protocol.

This module implements the PIP requirements for Phase 5:

- **PIP-01** (TestSurrogateReproducibility): Surrogate-only inference (S1+S2)
  produces identical results across runs, verified at rel=1e-10 against
  saved JSON baselines.

- **PIP-02**: Regression baselines infrastructure -- ``--update-baselines``
  pytest flag regenerates baselines, missing baselines produce a clear error,
  and tolerance failures display a diff table.

Artifacts produced (under StudyResults/pipeline_reproducibility/):
    - regression_baselines.json

Run fast tests only::

    pytest tests/test_pipeline_reproducibility.py -x -v

Regenerate baselines::

    pytest tests/test_pipeline_reproducibility.py --update-baselines -x -v
"""

from __future__ import annotations

import json
import os
import subprocess as sp
import sys
from datetime import datetime, timezone

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pytest


# ===================================================================
# Path constants
# ===================================================================

_OUTPUT_DIR = os.path.join(_ROOT, "StudyResults", "pipeline_reproducibility")
_BASELINES_PATH = os.path.join(_OUTPUT_DIR, "regression_baselines.json")


# ===================================================================
# Baseline load / save / diff helpers
# ===================================================================

def _load_baselines(path: str) -> dict | None:
    """Load regression baselines from a JSON file.

    Returns None if the file does not exist.
    """
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def _save_baselines(path: str, data: dict) -> None:
    """Save regression baselines to a JSON file with metadata.

    Metadata includes generation timestamp, git commit hash, Python version,
    and NumPy version.  Parent directories are created if missing.
    """
    try:
        git_commit = (
            sp.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=_ROOT,
                stderr=sp.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (sp.CalledProcessError, FileNotFoundError):
        git_commit = "unknown"

    data["metadata"] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            data,
            f,
            indent=2,
            default=lambda o: o.tolist() if hasattr(o, "tolist") else o,
        )
    print(f"\n  Baselines saved to {path}")


def _format_diff_table(
    entries: list[tuple[str, float, float, float]],
) -> tuple[str, bool]:
    """Format a human-readable comparison table.

    Parameters
    ----------
    entries : list of (name, baseline_value, current_value, tolerance)
        Each entry is one parameter to compare.

    Returns
    -------
    (formatted_string, any_fail)
        The formatted table and whether any entry failed.
    """
    header = (
        f"{'Parameter':<15}| {'Baseline':>14} | {'Current':>14} | "
        f"{'Abs Diff':>10} | {'Rel Diff':>10} | {'Tol':>10} | {'Pass'}"
    )
    sep = "-" * len(header)
    lines = [header, sep]
    any_fail = False

    for name, baseline, current, tol in entries:
        abs_diff = abs(current - baseline)
        rel_diff = abs_diff / max(abs(baseline), 1e-30)
        passed = rel_diff <= tol
        if not passed:
            any_fail = True
        status = "PASS" if passed else "FAIL"
        lines.append(
            f"{name:<15}| {baseline:>14.6e} | {current:>14.6e} | "
            f"{abs_diff:>10.2e} | {rel_diff:>10.2e} | {tol:>10.2e} | {status}"
        )

    return "\n".join(lines), any_fail


def _assert_baselines(
    current: dict,
    section: str,
    tolerance: float,
    baselines: dict | None,
    update_baselines: bool,
    baselines_path: str,
) -> None:
    """Compare current values against baselines, or update them.

    Parameters
    ----------
    current : dict
        Mapping of parameter names to current values (float or list).
    section : str
        Section key in the baselines JSON (e.g. "surrogate_only").
    tolerance : float
        Relative tolerance for comparison.
    baselines : dict or None
        Loaded baselines dict, or None if file is missing.
    update_baselines : bool
        If True, merge current into baselines and save instead of comparing.
    baselines_path : str
        Path to the baselines JSON file.
    """
    if update_baselines:
        if baselines is None:
            baselines = {}
        baselines[section] = current
        _save_baselines(baselines_path, baselines)
        return

    if baselines is None:
        pytest.fail(
            "Baselines not found. Run with --update-baselines to generate.\n"
            f"  Expected path: {baselines_path}"
        )

    if section not in baselines:
        pytest.fail(
            f"Section '{section}' not found in baselines. "
            "Run with --update-baselines to regenerate."
        )

    saved = baselines[section]
    entries: list[tuple[str, float, float, float]] = []

    for key, cur_val in current.items():
        base_val = saved.get(key)
        if base_val is None:
            pytest.fail(
                f"Key '{key}' not found in baselines['{section}']. "
                "Run with --update-baselines to regenerate."
            )
        # Handle both scalar and list values
        if isinstance(cur_val, (list, np.ndarray)):
            cur_list = list(cur_val) if not isinstance(cur_val, list) else cur_val
            base_list = list(base_val) if not isinstance(base_val, list) else base_val
            for i, (c, b) in enumerate(zip(cur_list, base_list)):
                entries.append((f"{key}[{i}]", float(b), float(c), tolerance))
        else:
            entries.append((key, float(base_val), float(cur_val), tolerance))

    table, any_fail = _format_diff_table(entries)
    if any_fail:
        pytest.fail(f"Baseline regression detected (tol={tolerance}):\n\n{table}")


# ===================================================================
# Ensemble and surrogate data paths
# ===================================================================

_V11_DIR = os.path.join(_ROOT, "StudyResults", "surrogate_v11")
_ENSEMBLE_DIR = os.path.join(_V11_DIR, "nn_ensemble", "D3-deeper")


# ===================================================================
# Module-scoped fixtures
# ===================================================================

@pytest.fixture(scope="module")
def nn_ensemble_repro():
    """Load the D3-deeper NN ensemble for reproducibility tests.

    Skips if the ensemble directory is not present on disk.
    """
    if not os.path.isdir(_ENSEMBLE_DIR):
        pytest.skip("D3-deeper ensemble not found on disk")

    from Surrogate.ensemble import load_nn_ensemble

    return load_nn_ensemble(_ENSEMBLE_DIR, n_members=5, device="cpu")


@pytest.fixture(scope="module")
def surrogate_only_results(nn_ensemble_repro):
    """Run surrogate-only inference (S1+S2) with fixed seeds.

    Calls ``_run_surrogate_phases()`` from the v13 script with
    ``surr_strategy="joint"`` so only S1 (alpha-only) and S2 (joint
    4-param L-BFGS-B) are executed.  No Firedrake is imported.

    Returns the result dict from ``_run_surrogate_phases()``.
    """
    from types import SimpleNamespace

    from scripts.surrogate.Infer_BVMaster_charged_v13_ultimate import (
        _run_surrogate_phases,
    )
    from scripts._bv_common import K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2

    model = nn_ensemble_repro

    # True parameters
    true_k0_arr = np.array([K0_HAT_R1, K0_HAT_R2])
    true_alpha_arr = np.array([ALPHA_R1, ALPHA_R2])

    # Generate surrogate targets at true params
    pred = model.predict(K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2)
    target_cd = pred["current_density"]
    target_pc = pred["peroxide_current"]

    # Voltage grids (identical to v13 main())
    eta_symmetric = np.array([
        +5.0, +3.0, +1.0, -0.5,
        -1.0, -2.0, -3.0, -5.0, -8.0,
        -10.0, -15.0, -20.0,
    ])
    eta_shallow = np.array([
        -1.0, -2.0, -3.0, -4.0, -5.0, -6.5, -8.0,
        -10.0, -11.5, -13.0,
    ])
    all_eta = np.unique(np.concatenate([eta_symmetric, eta_shallow]))
    all_eta = np.sort(all_eta)[::-1]

    # Args namespace with joint strategy (S1+S2 only)
    args = SimpleNamespace(surr_strategy="joint")

    initial_k0_guess = [0.005, 0.0005]
    initial_alpha_guess = [0.4, 0.3]

    result = _run_surrogate_phases(
        surrogate=model,
        model_label="nn-ensemble",
        args=args,
        target_cd_surr=target_cd,
        target_pc_surr=target_pc,
        target_cd_full=target_cd,
        target_pc_full=target_pc,
        all_eta=all_eta,
        eta_shallow=eta_shallow,
        initial_k0_guess=initial_k0_guess,
        initial_alpha_guess=initial_alpha_guess,
        true_k0_arr=true_k0_arr,
        true_alpha_arr=true_alpha_arr,
        secondary_weight=1.0,
    )

    return result


# ===================================================================
# PIP-01: Surrogate reproducibility tests (fast, no Firedrake)
# ===================================================================

class TestSurrogateReproducibility:
    """PIP-01: Surrogate-only inference produces deterministic results.

    Runs S1 (alpha-only) + S2 (joint 4-param L-BFGS-B) and compares
    inferred parameters against saved baselines at rel=1e-10.  These
    outputs are float64 deterministic (no PDE solver involved).
    """

    def test_surrogate_parameters_reproducible(
        self, surrogate_only_results, update_baselines,
    ):
        """S1 alpha and S2 best k0/alpha match saved baselines at rel=1e-10."""
        r = surrogate_only_results
        baselines = _load_baselines(_BASELINES_PATH)

        current = {
            "s1_alpha": list(r["s1_alpha"]),
            "surr_best_k0": list(r["surr_best_k0"]),
            "surr_best_alpha": list(r["surr_best_alpha"]),
        }

        _assert_baselines(
            current=current,
            section="surrogate_only",
            tolerance=1e-10,
            baselines=baselines,
            update_baselines=update_baselines,
            baselines_path=_BASELINES_PATH,
        )

    def test_surrogate_loss_reproducible(
        self, surrogate_only_results, update_baselines,
    ):
        """S2 best loss matches saved baseline at rel=1e-10."""
        r = surrogate_only_results
        baselines = _load_baselines(_BASELINES_PATH)

        current = {
            "surr_best_loss": float(r["surr_best_loss"]),
        }

        _assert_baselines(
            current=current,
            section="surrogate_only_loss",
            tolerance=1e-10,
            baselines=baselines,
            update_baselines=update_baselines,
            baselines_path=_BASELINES_PATH,
        )
