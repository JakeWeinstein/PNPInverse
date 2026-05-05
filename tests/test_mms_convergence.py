"""3-species + Boltzmann ClO4- log-c MMS convergence test.

Publication-grade convergence proof for the production forward PDE
solver: 3 dynamic species (O2, H2O2, H+) with log-concentration primary
variables ``u_i = ln(c_i)``, ClO4- counterion handled via analytic
Boltzmann residual on Poisson, and the standard 2 BV reactions.

Convergence is verified via log-log linear regression (scipy.stats.linregress):
  - L2 error rate ~ O(h^2) for CG1 elements
  - H1 error rate ~ O(h^1) for CG1 elements
  - R-squared > 0.99 for all fits (asymptotic regime check)

GCI (Grid Convergence Index) is computed using the Roache 3-grid formula
with safety factor Fs=1.25, providing uncertainty quantification for the
finest-grid solution.

Artifacts (JSON + PNG) are saved to StudyResults/mms_convergence/.
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pytest

# Ensure the PNPInverse root is importable
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from conftest import skip_without_firedrake


# ---------------------------------------------------------------------------
# Lazy import of MMS functions (Firedrake must be available at import time)
# ---------------------------------------------------------------------------

def _import_mms_run():
    """Import run_mms lazily to avoid Firedrake import errors at collection."""
    from scripts.verification.mms_bv_3sp_logc_boltzmann import run_mms
    return run_mms


def _import_graded_verifier():
    """Import the graded-mesh verifier lazily."""
    from scripts.verification.mms_bv_3sp_logc_boltzmann import (
        verify_on_graded_production_mesh,
    )
    return verify_on_graded_production_mesh


# ---------------------------------------------------------------------------
# Field iteration constants
# ---------------------------------------------------------------------------

# u_i = ln(c_i) primary variables for 3 dynamic species + phi.  ClO4- is
# the analytic Boltzmann background, not a primary unknown.
FIELD_NAMES = ["u0", "u1", "u2", "phi"]
SPECIES_LABELS = ["O2 (u)", "H2O2 (u)", "H+ (u)", "phi"]


# ---------------------------------------------------------------------------
# Helper: convergence rate assertion via log-log regression
# ---------------------------------------------------------------------------

def assert_convergence_rate(
    h_vals,
    err_vals,
    expected_rate: float,
    rate_tol: float = 0.2,
    min_r_squared: float = 0.99,
    label: str = "",
) -> tuple[float, float]:
    """Assert log-log convergence rate and R-squared, return (slope, r_squared).

    Parameters
    ----------
    h_vals : array-like
        Mesh sizes (h = 1/N).
    err_vals : array-like
        Error norms corresponding to each mesh size.
    expected_rate : float
        Expected convergence rate (slope in log-log space).
    rate_tol : float
        Acceptable deviation from expected_rate (both sides).
    min_r_squared : float
        Minimum R-squared for the log-log linear fit.
    label : str
        Label for error messages.

    Returns
    -------
    tuple[float, float]
        (slope, r_squared) from the log-log regression.
    """
    from scipy.stats import linregress

    log_h = np.log(np.array(h_vals, dtype=float))
    log_err = np.log(np.array(err_vals, dtype=float))

    slope, _intercept, r_value, _p_value, std_err = linregress(log_h, log_err)
    r_squared = r_value ** 2

    assert r_squared > min_r_squared, (
        f"{label}: R^2 = {r_squared:.6f} < {min_r_squared} "
        f"(slope={slope:.4f}, std_err={std_err:.4f}). "
        f"Log-log fit is not sufficiently linear -- may not be in asymptotic regime."
    )
    assert slope >= expected_rate - rate_tol, (
        f"{label}: slope = {slope:.4f}, expected >= {expected_rate:.1f} - {rate_tol} "
        f"(R^2={r_squared:.6f}, std_err={std_err:.4f}). "
        f"Convergence rate is worse than theoretical minimum."
    )

    return slope, r_squared


# ---------------------------------------------------------------------------
# Helper: GCI (Grid Convergence Index) via Roache 3-grid formula
# ---------------------------------------------------------------------------

def compute_gci(
    errors: list[float],
    h_vals: list[float],
    observed_order: float,
    Fs: float = 1.25,
) -> list[dict]:
    """Compute GCI for each consecutive mesh pair using the Roache formula.

    Parameters
    ----------
    errors : list[float]
        Error norms from coarsest to finest mesh.
    h_vals : list[float]
        Mesh sizes from coarsest to finest.
    observed_order : float
        Observed convergence order (log-log slope).
    Fs : float
        Safety factor (1.25 for >= 3 grids per Roache).

    Returns
    -------
    list[dict]
        One dict per consecutive pair with keys: h_fine, h_coarse,
        refinement_ratio, error_fine, error_coarse, gci.
    """
    gci_results = []
    for i in range(1, len(errors)):
        e_coarse = errors[i - 1]
        e_fine = errors[i]
        h_coarse = h_vals[i - 1]
        h_fine = h_vals[i]
        r = h_coarse / h_fine  # refinement ratio

        if e_fine == 0.0:
            gci_val = float("nan")
        else:
            relative_error = abs(e_coarse - e_fine) / e_fine
            denom = r ** observed_order - 1.0
            gci_val = Fs * relative_error / denom if denom != 0.0 else float("nan")

        gci_results.append({
            "h_fine": h_fine,
            "h_coarse": h_coarse,
            "refinement_ratio": r,
            "error_fine": e_fine,
            "error_coarse": e_coarse,
            "gci": gci_val,
        })

    return gci_results


# ---------------------------------------------------------------------------
# Custom JSON encoder for numpy types
# ---------------------------------------------------------------------------

class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar types."""

    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

@skip_without_firedrake
@pytest.mark.slow
class TestMMSConvergence:
    """3-species + Boltzmann log-c MMS convergence tests."""

    MESH_SIZES = [8, 16, 32, 64]
    OUTPUT_DIR = os.path.join(_ROOT, "StudyResults", "mms_convergence")

    @pytest.fixture(scope="class")
    def mms_results(self):
        """Run the 3sp+Boltzmann log-c MMS convergence study once for the class."""
        run_mms = _import_mms_run()
        results = run_mms(self.MESH_SIZES, verbose=True)

        # Sanity: all mesh sizes should have produced results
        assert len(results["N"]) == len(self.MESH_SIZES), (
            f"Expected {len(self.MESH_SIZES)} mesh results, "
            f"got {len(results['N'])} (some solves may have failed)"
        )
        return results

    # ---- L2 convergence ----

    def test_l2_convergence_rates(self, mms_results):
        """Assert L2 convergence rate ~ 2.0 for all 5 fields (CG1)."""
        results_l2 = {}
        for field, label in zip(FIELD_NAMES, SPECIES_LABELS):
            slope, r_squared = assert_convergence_rate(
                mms_results["h"],
                mms_results[f"{field}_L2"],
                expected_rate=2.0,
                rate_tol=0.2,
                min_r_squared=0.99,
                label=f"{label} L2",
            )
            results_l2[field] = {"slope": slope, "r_squared": r_squared}

        # Store on class for artifact generation
        TestMMSConvergence._l2_results = results_l2

    # ---- H1 convergence ----

    def test_h1_convergence_rates(self, mms_results):
        """Assert H1 convergence rate ~ 1.0 for all 5 fields (CG1)."""
        results_h1 = {}
        for field, label in zip(FIELD_NAMES, SPECIES_LABELS):
            slope, r_squared = assert_convergence_rate(
                mms_results["h"],
                mms_results[f"{field}_H1"],
                expected_rate=1.0,
                rate_tol=0.2,
                min_r_squared=0.99,
                label=f"{label} H1",
            )
            results_h1[field] = {"slope": slope, "r_squared": r_squared}

        TestMMSConvergence._h1_results = results_h1

    # ---- GCI output ----

    def test_gci_output(self, mms_results):
        """Compute GCI via Roache formula (Fs=1.25) for all fields' L2 errors."""
        gci_all = {}
        for field, label in zip(FIELD_NAMES, SPECIES_LABELS):
            # Use L2 regression slope as observed order
            from scipy.stats import linregress

            log_h = np.log(np.array(mms_results["h"], dtype=float))
            log_err = np.log(np.array(mms_results[f"{field}_L2"], dtype=float))
            slope, _intercept, _r, _p, _se = linregress(log_h, log_err)

            gci_list = compute_gci(
                errors=mms_results[f"{field}_L2"],
                h_vals=mms_results["h"],
                observed_order=slope,
                Fs=1.25,
            )

            assert len(gci_list) == len(self.MESH_SIZES) - 1, (
                f"{label}: expected {len(self.MESH_SIZES) - 1} GCI entries, "
                f"got {len(gci_list)}"
            )

            for entry in gci_list:
                gci_val = entry["gci"]
                assert math.isfinite(gci_val), (
                    f"{label}: GCI is not finite: {gci_val}"
                )
                assert gci_val >= 0.0, (
                    f"{label}: GCI is negative: {gci_val}"
                )

            gci_all[field] = gci_list

        TestMMSConvergence._gci_results = gci_all

    # ---- Save artifacts ----

    def test_save_convergence_artifacts(self, mms_results):
        """Save convergence data JSON and PNG plot to StudyResults/mms_convergence/."""
        out_dir = self.OUTPUT_DIR
        os.makedirs(out_dir, exist_ok=True)

        # --- Build JSON structure ---
        # Recompute rates and GCI if earlier tests haven't run (pytest order)
        from scipy.stats import linregress

        fields_data = {}
        for field, label in zip(FIELD_NAMES, SPECIES_LABELS):
            # L2
            log_h = np.log(np.array(mms_results["h"], dtype=float))
            log_err_l2 = np.log(np.array(mms_results[f"{field}_L2"], dtype=float))
            slope_l2, _int, r_l2, _p, _se = linregress(log_h, log_err_l2)

            # H1
            log_err_h1 = np.log(np.array(mms_results[f"{field}_H1"], dtype=float))
            slope_h1, _int, r_h1, _p, _se = linregress(log_h, log_err_h1)

            # GCI on L2
            gci_list = compute_gci(
                errors=mms_results[f"{field}_L2"],
                h_vals=mms_results["h"],
                observed_order=slope_l2,
                Fs=1.25,
            )

            fields_data[field] = {
                "label": label,
                "L2_errors": mms_results[f"{field}_L2"],
                "H1_errors": mms_results[f"{field}_H1"],
                "L2_rate": float(slope_l2),
                "L2_r_squared": float(r_l2 ** 2),
                "H1_rate": float(slope_h1),
                "H1_r_squared": float(r_h1 ** 2),
                "gci": gci_list,
            }

        convergence_data = {
            "metadata": {
                "date": datetime.now(timezone.utc).isoformat(),
                "mesh_sizes": list(mms_results["N"]),
                "h_values": list(mms_results["h"]),
                "species": ["O2", "H2O2", "H+"],
                "boltzmann_counterion": "ClO4-",
                "primary_variable": "u_i = ln(c_i)",
                "Fs": 1.25,
                "description": (
                    "3-species + analytic Boltzmann ClO4- log-c MMS "
                    "convergence study with 2 BV reactions. Production "
                    "solver pipeline."
                ),
            },
            "fields": fields_data,
        }

        json_path = os.path.join(out_dir, "convergence_data.json")
        with open(json_path, "w") as f:
            json.dump(convergence_data, f, indent=2, cls=_NumpyEncoder)

        assert os.path.isfile(json_path), f"JSON not created: {json_path}"

        # --- Generate PNG ---
        from scripts.verification.mms_bv_3sp_logc_boltzmann import plot_convergence

        png_path = plot_convergence(mms_results, out_dir)

        # The function saves as mms_3sp_logc_boltzmann.png; copy to the
        # canonical name expected by the report generator.
        canonical_png = os.path.join(out_dir, "mms_convergence.png")
        if png_path != canonical_png:
            import shutil
            shutil.copy2(png_path, canonical_png)

        assert os.path.isfile(canonical_png), f"PNG not created: {canonical_png}"


# ---------------------------------------------------------------------------
# Production graded-mesh recovery test (single solve)
# ---------------------------------------------------------------------------

@skip_without_firedrake
@pytest.mark.slow
class TestMMSProductionGradedMesh:
    """Single-mesh MMS recovery on the production graded rectangle.

    Verifies the solver produces the manufactured solution within the
    expected discretization error of the actual production mesh
    (Nx=8, Ny=200, beta=3 -- mirrors plot_iv_curve_unified.py).  This is
    not an asymptotic convergence test; the thresholds are sized to the
    Nx=8-dominated x-direction discretization error (h_x^2 = 1/64) with
    safety margin.  Intended to fail loudly on any wiring bug that
    causes O(1) rather than O(h_x^2) errors.
    """

    # Empirically observed errors (May 2026 baseline):
    #   u0 (O2):   L2 = 1.6e-3,  H1 = 5.4e-2
    #   u1 (H2O2): L2 = 6.5e-4,  H1 = 5.5e-2
    #   u2 (H+):   L2 = 7.2e-4,  H1 = 5.5e-2
    #   phi:       L2 = 3.8e-4,  H1 = 4.0e-2
    # Thresholds set ~6x above observed to stay robust to compiler /
    # solver-version drift, while still ~100x below the O(1) errors a
    # wiring bug would produce.
    L2_THRESHOLD = 1e-2
    H1_THRESHOLD = 3e-1

    @pytest.fixture(scope="class")
    def graded_mesh_results(self):
        """Run a single solve on the production graded rectangle."""
        verify = _import_graded_verifier()
        return verify(verbose=True)

    def test_newton_converges(self, graded_mesh_results):
        """Newton must converge on the production mesh."""
        assert graded_mesh_results.get("newton_converged") is True, (
            f"Newton failed on production graded mesh: "
            f"{graded_mesh_results.get('newton_error', 'no error message')}"
        )
        assert graded_mesh_results["newton_iterations"] >= 0

    def test_l2_recovery(self, graded_mesh_results):
        """L2 errors of u_i and phi must be below the recovery threshold."""
        for field, label in zip(FIELD_NAMES, SPECIES_LABELS):
            err = graded_mesh_results[f"{field}_L2"]
            assert math.isfinite(err), f"{label}: L2 error is not finite ({err})"
            assert err < self.L2_THRESHOLD, (
                f"{label}: L2 error {err:.3e} >= threshold "
                f"{self.L2_THRESHOLD:.3e} -- solver may not be recovering "
                f"the manufactured solution"
            )

    def test_h1_recovery(self, graded_mesh_results):
        """H1 errors of u_i and phi must be below the recovery threshold."""
        for field, label in zip(FIELD_NAMES, SPECIES_LABELS):
            err = graded_mesh_results[f"{field}_H1"]
            assert math.isfinite(err), f"{label}: H1 error is not finite ({err})"
            assert err < self.H1_THRESHOLD, (
                f"{label}: H1 error {err:.3e} >= threshold "
                f"{self.H1_THRESHOLD:.3e}"
            )
