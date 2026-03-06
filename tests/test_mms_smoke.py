"""MMS convergence smoke tests (Phase 1, Plan 02).

These tests confirm that after the MMS refactor to use production build_forms(),
convergence behavior is preserved. Full rate assertions belong in Phase 2.

Each test runs a small convergence study (3 mesh sizes), fits a log-log line
using scipy.stats.linregress, and asserts:
  - R-squared > 0.99 (good linear fit in log-log space)
  - Slope > 1.5 (sanity check: L2 rate should be ~2.0 for CG1)

Tests are skipped gracefully when Firedrake is not available.

Requirement: FWD-02 (MMS tests the production weak form).
"""

from __future__ import annotations

import os
import sys

import pytest
import numpy as np

# Ensure the PNPInverse root is importable
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from conftest import skip_without_firedrake


# ---------------------------------------------------------------------------
# Lazy import of MMS functions (Firedrake must be available at import time)
# ---------------------------------------------------------------------------

def _import_mms():
    """Import MMS functions lazily to avoid Firedrake import errors at collection."""
    from scripts.verification.mms_bv_convergence import (
        run_mms_single_species,
        run_mms_two_species,
        run_mms_charged,
    )
    return run_mms_single_species, run_mms_two_species, run_mms_charged


def _check_convergence(h_vals, err_vals, *, min_r_squared=0.99, min_slope=1.5, label=""):
    """Assert log-log convergence with R-squared and minimum slope.

    Parameters
    ----------
    h_vals : array-like
        Mesh sizes.
    err_vals : array-like
        Error norms corresponding to each mesh size.
    min_r_squared : float
        Minimum acceptable R-squared for log-log linear fit.
    min_slope : float
        Minimum acceptable slope (convergence rate sanity check).
    label : str
        Label for error messages.
    """
    from scipy.stats import linregress

    log_h = np.log(np.array(h_vals))
    log_err = np.log(np.array(err_vals))

    slope, intercept, r_value, p_value, std_err = linregress(log_h, log_err)
    r_squared = r_value ** 2

    assert r_squared > min_r_squared, (
        f"{label}: R^2 = {r_squared:.6f} < {min_r_squared} "
        f"(slope={slope:.3f}, std_err={std_err:.3f})"
    )
    assert slope > min_slope, (
        f"{label}: slope = {slope:.3f} < {min_slope} (expected ~2.0 for L2)"
    )

    return slope, r_squared


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

_SMOKE_MESH_SIZES = [8, 16, 32]


@skip_without_firedrake
@pytest.mark.firedrake
def test_mms_single_species_convergence_smoke():
    """Smoke test: single neutral species + BV, L2 convergence ~ O(h^2)."""
    run_mms_single_species, _, _ = _import_mms()

    results = run_mms_single_species(_SMOKE_MESH_SIZES, verbose=False)

    assert len(results["h"]) == len(_SMOKE_MESH_SIZES), (
        f"Expected {len(_SMOKE_MESH_SIZES)} mesh results, got {len(results['h'])}"
    )

    # Check concentration L2 convergence
    slope, r2 = _check_convergence(
        results["h"], results["c_L2"],
        label="single_species c_L2",
    )

    # Check potential L2 convergence
    _check_convergence(
        results["h"], results["phi_L2"],
        label="single_species phi_L2",
    )


@skip_without_firedrake
@pytest.mark.firedrake
def test_mms_two_species_convergence_smoke():
    """Smoke test: two neutral species + 2 BV reactions, L2 convergence ~ O(h^2)."""
    _, run_mms_two_species, _ = _import_mms()

    results = run_mms_two_species(_SMOKE_MESH_SIZES, verbose=False)

    assert len(results["h"]) == len(_SMOKE_MESH_SIZES), (
        f"Expected {len(_SMOKE_MESH_SIZES)} mesh results, got {len(results['h'])}"
    )

    # Check species 0 (O2) L2 convergence
    _check_convergence(
        results["h"], results["c0_L2"],
        label="two_species c0_L2 (O2)",
    )

    # Check species 1 (H2O2) L2 convergence
    _check_convergence(
        results["h"], results["c1_L2"],
        label="two_species c1_L2 (H2O2)",
    )

    # Check potential L2 convergence
    _check_convergence(
        results["h"], results["phi_L2"],
        label="two_species phi_L2",
    )


@skip_without_firedrake
@pytest.mark.firedrake
def test_mms_charged_convergence_smoke():
    """Smoke test: two charged species + Poisson + BV, L2 convergence ~ O(h^2)."""
    _, _, run_mms_charged = _import_mms()

    results = run_mms_charged(_SMOKE_MESH_SIZES, verbose=False)

    assert len(results["h"]) == len(_SMOKE_MESH_SIZES), (
        f"Expected {len(_SMOKE_MESH_SIZES)} mesh results, got {len(results['h'])}"
    )

    # Check cation (z=+1) L2 convergence
    _check_convergence(
        results["h"], results["c0_L2"],
        label="charged c0_L2 (cation)",
    )

    # Check anion (z=-1) L2 convergence
    _check_convergence(
        results["h"], results["c1_L2"],
        label="charged c1_L2 (anion)",
    )

    # Check potential L2 convergence
    _check_convergence(
        results["h"], results["phi_L2"],
        label="charged phi_L2",
    )
