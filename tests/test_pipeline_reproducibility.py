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
