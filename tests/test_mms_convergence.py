"""4-species MMS convergence test with formal rate assertions and GCI output.

Tests FWD-01, FWD-03, FWD-05: automated, publication-grade convergence proof
for the forward PDE solver using the 4-species charged PNP + 2 BV reactions
case (O2, H2O2, H+, ClO4-) matching the v13 production config.

Convergence is verified via log-log linear regression (scipy.stats.linregress):
  - L2 error rate ~ O(h^2) for CG1 elements
  - H1 error rate ~ O(h^1) for CG1 elements
  - R-squared > 0.99 for all fits (asymptotic regime check)

GCI (Grid Convergence Index) is computed using the Roache 3-grid formula
with safety factor Fs=1.25, providing uncertainty quantification for the
finest-grid solution.

Artifacts (JSON + PNG) are saved to StudyResults/mms_convergence/ for
Phase 6 report generation.
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

def _import_mms_4species():
    """Import run_mms_4species lazily to avoid Firedrake import errors at collection."""
    from scripts.verification.mms_bv_4species import run_mms_4species
    return run_mms_4species


# ---------------------------------------------------------------------------
# Field iteration constants
# ---------------------------------------------------------------------------

FIELD_NAMES = ["c0", "c1", "c2", "c3", "phi"]
SPECIES_LABELS = ["O2", "H2O2", "H+", "ClO4-", "phi"]


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
    """4-species MMS convergence tests with rate assertions and GCI output."""

    MESH_SIZES = [8, 16, 32, 64]
    OUTPUT_DIR = os.path.join(_ROOT, "StudyResults", "mms_convergence")

    @pytest.fixture(scope="class")
    def mms_results(self):
        """Run the 4-species MMS convergence study once for the entire class."""
        run_mms_4species = _import_mms_4species()
        results = run_mms_4species(self.MESH_SIZES, verbose=True)

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
                "species": ["O2", "H2O2", "H+", "ClO4-"],
                "Fs": 1.25,
                "description": (
                    "4-species MMS convergence study (O2, H2O2, H+, ClO4-) "
                    "with 2 BV reactions. Production solver pipeline."
                ),
            },
            "fields": fields_data,
        }

        json_path = os.path.join(out_dir, "convergence_data.json")
        with open(json_path, "w") as f:
            json.dump(convergence_data, f, indent=2, cls=_NumpyEncoder)

        assert os.path.isfile(json_path), f"JSON not created: {json_path}"

        # --- Generate PNG ---
        # Use the plot_convergence function from mms_bv_4species
        from scripts.verification.mms_bv_4species import plot_convergence

        png_path = plot_convergence(mms_results, out_dir)

        # The function saves as mms_bv_4species.png; copy/rename to the
        # canonical name expected by the plan
        canonical_png = os.path.join(out_dir, "mms_convergence.png")
        if png_path != canonical_png:
            import shutil
            shutil.copy2(png_path, canonical_png)

        assert os.path.isfile(canonical_png), f"PNG not created: {canonical_png}"
