"""Tests for sensitivity visualization: sweep grid, Jacobian FD, parameter perturbation.

Validates the helper functions in scripts/studies/sensitivity_visualization.py
that support 1D parameter sweeps and Jacobian heatmap sensitivity analysis.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

# Add PNPInverse root to path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.studies.sensitivity_visualization import (
    build_sweep_factors,
    build_extended_voltage_grid,
    compute_jacobian_row,
    build_perturbed_params,
    write_metadata,
)


# ---- AUDT-04 metadata validation helper ----

AUDT04_REQUIRED_FIELDS = {
    "tool_name",
    "requirement",
    "justification_type",
    "reference",
    "rationale",
}


def validate_metadata(metadata: dict) -> bool:
    """Validate metadata dict passes AUDT-04 requirements."""
    return AUDT04_REQUIRED_FIELDS.issubset(set(metadata.keys()))


# ---- Test: build_sweep_factors ----

class TestBuildSweepFactors:
    def test_default_factors(self):
        factors = build_sweep_factors()
        assert factors == [0.5, 0.75, 1.0, 1.5, 2.0]

    def test_custom_factors(self):
        custom = (0.1, 1.0, 10.0)
        factors = build_sweep_factors(factors=custom)
        assert factors == [0.1, 1.0, 10.0]


# ---- Test: build_extended_voltage_grid ----

class TestBuildExtendedVoltageGrid:
    def test_descending_order(self):
        grid = build_extended_voltage_grid()
        # Must be sorted in descending order for warm-starting
        for i in range(len(grid) - 1):
            assert grid[i] > grid[i + 1], (
                f"Grid not descending at index {i}: {grid[i]} <= {grid[i+1]}"
            )

    def test_extends_beyond_v13(self):
        grid = build_extended_voltage_grid()
        # Must contain points beyond v13 default of -46.5
        assert grid.min() <= -60.0, (
            f"Grid minimum {grid.min()} does not extend to -60"
        )

    def test_contains_v13_cathodic_points(self):
        v13_cathodic = np.array([
            -1.0, -2.0, -3.0, -4.0, -5.0, -6.5, -8.0,
            -10.0, -13.0, -17.0, -22.0, -28.0,
            -35.0, -41.0, -46.5,
        ])
        grid = build_extended_voltage_grid()
        for v in v13_cathodic:
            assert v in grid, f"v13 point {v} missing from extended grid"

    def test_no_duplicates(self):
        grid = build_extended_voltage_grid()
        assert len(grid) == len(np.unique(grid)), "Grid contains duplicate values"

    def test_returns_numpy_array(self):
        grid = build_extended_voltage_grid()
        assert isinstance(grid, np.ndarray)


# ---- Test: compute_jacobian_row ----

class TestComputeJacobianRow:
    def test_quadratic_function(self):
        """f(x) = x[0]**2 + 2*x[1], gradient = [2*x[0], 2]."""
        def f(params):
            return params[0] ** 2 + 2 * params[1]

        params = np.array([3.0, 1.0])
        jac = compute_jacobian_row(f, params, h=1e-5)
        expected = np.array([6.0, 2.0])
        np.testing.assert_allclose(jac, expected, atol=1e-4)

    def test_default_step_size(self):
        """Verify h=1e-5 is the default step size (codebase convention)."""
        import inspect
        sig = inspect.signature(compute_jacobian_row)
        h_default = sig.parameters["h"].default
        assert h_default == 1e-5, f"Default h is {h_default}, expected 1e-5"

    def test_returns_correct_length(self):
        def f(params):
            return np.sum(params)

        params = np.array([1.0, 2.0, 3.0])
        jac = compute_jacobian_row(f, params)
        assert len(jac) == 3


# ---- Test: build_perturbed_params ----

class TestBuildPerturbedParams:
    TRUE_PARAMS = {
        "k0_1": 1e-3,
        "k0_2": 5e-5,
        "alpha_1": 0.627,
        "alpha_2": 0.5,
    }

    def test_perturb_k0_1(self):
        result = build_perturbed_params("k0_1", 2.0, self.TRUE_PARAMS)
        assert result["k0_1"] == pytest.approx(2e-3)
        assert result["k0_2"] == pytest.approx(5e-5)
        assert result["alpha_1"] == pytest.approx(0.627)
        assert result["alpha_2"] == pytest.approx(0.5)

    def test_perturb_alpha_2(self):
        result = build_perturbed_params("alpha_2", 0.5, self.TRUE_PARAMS)
        assert result["alpha_2"] == pytest.approx(0.25)
        assert result["k0_1"] == pytest.approx(1e-3)
        assert result["k0_2"] == pytest.approx(5e-5)
        assert result["alpha_1"] == pytest.approx(0.627)

    def test_does_not_mutate_original(self):
        original_copy = dict(self.TRUE_PARAMS)
        build_perturbed_params("k0_1", 2.0, self.TRUE_PARAMS)
        assert self.TRUE_PARAMS == original_copy


# ---- Test: metadata AUDT-04 ----

class TestMetadata:
    def test_metadata_passes_audt04(self, tmp_path):
        write_metadata(str(tmp_path))
        meta_path = tmp_path / "metadata.json"
        assert meta_path.exists(), "metadata.json not written"
        with open(meta_path) as f:
            metadata = json.load(f)
        assert validate_metadata(metadata), (
            f"Metadata missing AUDT-04 fields: "
            f"{AUDT04_REQUIRED_FIELDS - set(metadata.keys())}"
        )
        assert metadata["requirement"] == "DIAG-03"
