"""Tests for the secondary weight sweep infrastructure.

Tests cover:
- WeightSweepResult frozen dataclass
- compute_errors helper
- extract_training_bounds helper
- _SubsetSurrogateObjective with different weights
- format_results_table formatting
- save_results_csv output
- AlphaOnlySurrogateObjective weight propagation
- SurrogateObjective weight propagation
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile

import numpy as np
import pytest

# Ensure PNPInverse root is importable
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Surrogate.surrogate_model import BVSurrogateModel, SurrogateConfig
from Surrogate.objectives import (
    SurrogateObjective,
    AlphaOnlySurrogateObjective,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def fitted_surrogate() -> BVSurrogateModel:
    """Build and fit a small BVSurrogateModel for testing.

    Uses 10 training samples with 5 voltage points, simple linear data
    so that interpolation is well-conditioned.
    """
    rng = np.random.default_rng(42)
    config = SurrogateConfig(
        kernel="thin_plate_spline",
        degree=1,
        smoothing=1e-3,
        log_space_k0=True,
        normalize_inputs=True,
    )
    model = BVSurrogateModel(config)

    n_samples = 20
    n_eta = 5
    phi_applied = np.linspace(-10, 5, n_eta)

    # Parameters in physical space
    k0_1 = rng.uniform(0.001, 0.1, n_samples)
    k0_2 = rng.uniform(0.0001, 0.01, n_samples)
    alpha_1 = rng.uniform(0.3, 0.8, n_samples)
    alpha_2 = rng.uniform(0.3, 0.7, n_samples)
    params = np.column_stack([k0_1, k0_2, alpha_1, alpha_2])

    # Simple linear response: cd ~ -alpha_1 * phi, pc ~ -alpha_2 * phi * k0_2
    cd = np.outer(alpha_1, -phi_applied) * k0_1[:, None] * 10
    pc = np.outer(alpha_2, -phi_applied) * k0_2[:, None] * 100

    model.fit(params, cd, pc, phi_applied)
    return model


# ---------------------------------------------------------------------------
# Tests: WeightSweepResult
# ---------------------------------------------------------------------------

class TestWeightSweepResult:
    """Tests for the WeightSweepResult frozen dataclass."""

    def test_creation(self):
        from scripts.surrogate.sweep_secondary_weight import WeightSweepResult

        r = WeightSweepResult(
            weight=1.0,
            k0_1=0.01, k0_2=0.001,
            alpha_1=0.6, alpha_2=0.5,
            k0_1_err_pct=5.0, k0_2_err_pct=3.0,
            alpha_1_err_pct=2.0, alpha_2_err_pct=1.5,
            max_err_pct=5.0,
            loss=0.001, elapsed_s=1.5,
        )
        assert r.weight == 1.0
        assert r.max_err_pct == 5.0

    def test_frozen(self):
        from scripts.surrogate.sweep_secondary_weight import WeightSweepResult

        r = WeightSweepResult(
            weight=1.0,
            k0_1=0.01, k0_2=0.001,
            alpha_1=0.6, alpha_2=0.5,
            k0_1_err_pct=5.0, k0_2_err_pct=3.0,
            alpha_1_err_pct=2.0, alpha_2_err_pct=1.5,
            max_err_pct=5.0,
            loss=0.001, elapsed_s=1.5,
        )
        with pytest.raises(AttributeError):
            r.weight = 2.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Tests: compute_errors
# ---------------------------------------------------------------------------

class TestComputeErrors:
    """Tests for the compute_errors helper."""

    def test_perfect_match(self):
        from scripts.surrogate.sweep_secondary_weight import compute_errors

        k0 = np.array([0.01, 0.001])
        alpha = np.array([0.6, 0.5])
        k0_err, alpha_err = compute_errors(k0, alpha, k0, alpha)
        np.testing.assert_allclose(k0_err, 0.0, atol=1e-15)
        np.testing.assert_allclose(alpha_err, 0.0, atol=1e-15)

    def test_known_errors(self):
        from scripts.surrogate.sweep_secondary_weight import compute_errors

        k0 = np.array([0.011, 0.001])
        alpha = np.array([0.6, 0.55])
        true_k0 = np.array([0.01, 0.001])
        true_alpha = np.array([0.6, 0.5])

        k0_err, alpha_err = compute_errors(k0, alpha, true_k0, true_alpha)

        # k0_1 error: |0.011 - 0.01| / 0.01 = 0.1
        np.testing.assert_allclose(k0_err[0], 0.1, atol=1e-10)
        # k0_2 error: 0
        np.testing.assert_allclose(k0_err[1], 0.0, atol=1e-15)
        # alpha_1 error: 0
        np.testing.assert_allclose(alpha_err[0], 0.0, atol=1e-15)
        # alpha_2 error: |0.55 - 0.5| / 0.5 = 0.1
        np.testing.assert_allclose(alpha_err[1], 0.1, atol=1e-10)


# ---------------------------------------------------------------------------
# Tests: extract_training_bounds
# ---------------------------------------------------------------------------

class TestExtractTrainingBounds:
    """Tests for the extract_training_bounds helper."""

    def test_with_model_bounds(self, fitted_surrogate):
        from scripts.surrogate.sweep_secondary_weight import extract_training_bounds

        bounds = extract_training_bounds(fitted_surrogate)
        assert "k0_1" in bounds
        assert "k0_2" in bounds
        assert "alpha" in bounds
        # Each should be (lo, hi) tuple
        assert bounds["k0_1"][0] < bounds["k0_1"][1]
        assert bounds["k0_2"][0] < bounds["k0_2"][1]
        assert bounds["alpha"][0] < bounds["alpha"][1]

    def test_without_model_bounds(self):
        from scripts.surrogate.sweep_secondary_weight import extract_training_bounds

        model = BVSurrogateModel()
        model.training_bounds = None
        bounds = extract_training_bounds(model)
        assert "k0_1" in bounds
        assert "k0_2" in bounds
        assert "alpha" in bounds


# ---------------------------------------------------------------------------
# Tests: _SubsetSurrogateObjective with different weights
# ---------------------------------------------------------------------------

class TestSubsetSurrogateObjective:
    """Tests for the _SubsetSurrogateObjective class."""

    def test_weight_changes_objective(self, fitted_surrogate):
        from scripts.surrogate.sweep_secondary_weight import _SubsetSurrogateObjective

        n_eta = fitted_surrogate.n_eta
        # Create simple targets
        target_cd = np.ones(n_eta) * 0.5
        target_pc = np.ones(n_eta) * 0.3
        subset_idx = np.arange(n_eta)

        obj_w1 = _SubsetSurrogateObjective(
            surrogate=fitted_surrogate,
            target_cd=target_cd,
            target_pc=target_pc,
            subset_idx=subset_idx,
            secondary_weight=1.0,
        )
        obj_w10 = _SubsetSurrogateObjective(
            surrogate=fitted_surrogate,
            target_cd=target_cd,
            target_pc=target_pc,
            subset_idx=subset_idx,
            secondary_weight=10.0,
        )

        x = np.array([np.log10(0.01), np.log10(0.001), 0.6, 0.5])

        j_w1 = obj_w1.objective(x)
        j_w10 = obj_w10.objective(x)

        # Higher weight on peroxide should change the objective value
        assert j_w1 != j_w10

    def test_zero_weight_ignores_peroxide(self, fitted_surrogate):
        from scripts.surrogate.sweep_secondary_weight import _SubsetSurrogateObjective

        n_eta = fitted_surrogate.n_eta
        target_cd = np.ones(n_eta) * 0.5
        target_pc = np.ones(n_eta) * 0.3
        subset_idx = np.arange(n_eta)

        obj_w0 = _SubsetSurrogateObjective(
            surrogate=fitted_surrogate,
            target_cd=target_cd,
            target_pc=target_pc,
            subset_idx=subset_idx,
            secondary_weight=0.0,
        )

        # With zero weight, changing peroxide target should not matter
        obj_w0_diff_pc = _SubsetSurrogateObjective(
            surrogate=fitted_surrogate,
            target_cd=target_cd,
            target_pc=np.ones(n_eta) * 999.0,
            subset_idx=subset_idx,
            secondary_weight=0.0,
        )

        x = np.array([np.log10(0.01), np.log10(0.001), 0.6, 0.5])
        assert obj_w0.objective(x) == obj_w0_diff_pc.objective(x)

    def test_gradient_shape(self, fitted_surrogate):
        from scripts.surrogate.sweep_secondary_weight import _SubsetSurrogateObjective

        n_eta = fitted_surrogate.n_eta
        target_cd = np.ones(n_eta) * 0.5
        target_pc = np.ones(n_eta) * 0.3
        subset_idx = np.arange(n_eta)

        obj = _SubsetSurrogateObjective(
            surrogate=fitted_surrogate,
            target_cd=target_cd,
            target_pc=target_pc,
            subset_idx=subset_idx,
            secondary_weight=1.0,
        )

        x = np.array([np.log10(0.01), np.log10(0.001), 0.6, 0.5])
        grad = obj.gradient(x)
        assert grad.shape == (4,)

    def test_n_evals_counter(self, fitted_surrogate):
        from scripts.surrogate.sweep_secondary_weight import _SubsetSurrogateObjective

        n_eta = fitted_surrogate.n_eta
        target_cd = np.ones(n_eta) * 0.5
        target_pc = np.ones(n_eta) * 0.3
        subset_idx = np.arange(n_eta)

        obj = _SubsetSurrogateObjective(
            surrogate=fitted_surrogate,
            target_cd=target_cd,
            target_pc=target_pc,
            subset_idx=subset_idx,
            secondary_weight=1.0,
        )

        assert obj.n_evals == 0
        x = np.array([np.log10(0.01), np.log10(0.001), 0.6, 0.5])
        obj.objective(x)
        assert obj.n_evals == 1
        obj.objective(x)
        assert obj.n_evals == 2


# ---------------------------------------------------------------------------
# Tests: SurrogateObjective weight propagation
# ---------------------------------------------------------------------------

class TestSurrogateObjectiveWeight:
    """Tests for SurrogateObjective weight propagation."""

    def test_weight_affects_objective(self, fitted_surrogate):
        n_eta = fitted_surrogate.n_eta
        target_cd = np.ones(n_eta) * 0.5
        target_pc = np.ones(n_eta) * 0.3

        obj_w1 = SurrogateObjective(
            surrogate=fitted_surrogate,
            target_cd=target_cd,
            target_pc=target_pc,
            secondary_weight=1.0,
        )
        obj_w5 = SurrogateObjective(
            surrogate=fitted_surrogate,
            target_cd=target_cd,
            target_pc=target_pc,
            secondary_weight=5.0,
        )

        x = np.array([np.log10(0.01), np.log10(0.001), 0.6, 0.5])
        j_w1 = obj_w1.objective(x)
        j_w5 = obj_w5.objective(x)

        assert j_w1 != j_w5

    def test_default_weight_is_one(self, fitted_surrogate):
        n_eta = fitted_surrogate.n_eta
        target_cd = np.ones(n_eta) * 0.5
        target_pc = np.ones(n_eta) * 0.3

        obj = SurrogateObjective(
            surrogate=fitted_surrogate,
            target_cd=target_cd,
            target_pc=target_pc,
        )
        assert obj.secondary_weight == 1.0


# ---------------------------------------------------------------------------
# Tests: AlphaOnlySurrogateObjective weight propagation
# ---------------------------------------------------------------------------

class TestAlphaOnlyObjectiveWeight:
    """Tests for AlphaOnlySurrogateObjective weight propagation."""

    def test_weight_affects_objective(self, fitted_surrogate):
        n_eta = fitted_surrogate.n_eta
        target_cd = np.ones(n_eta) * 0.5
        target_pc = np.ones(n_eta) * 0.3

        obj_w1 = AlphaOnlySurrogateObjective(
            surrogate=fitted_surrogate,
            target_cd=target_cd,
            target_pc=target_pc,
            fixed_k0=[0.01, 0.001],
            secondary_weight=1.0,
        )
        obj_w10 = AlphaOnlySurrogateObjective(
            surrogate=fitted_surrogate,
            target_cd=target_cd,
            target_pc=target_pc,
            fixed_k0=[0.01, 0.001],
            secondary_weight=10.0,
        )

        x = np.array([0.6, 0.5])
        j_w1 = obj_w1.objective(x)
        j_w10 = obj_w10.objective(x)

        assert j_w1 != j_w10

    def test_default_weight_is_one(self, fitted_surrogate):
        n_eta = fitted_surrogate.n_eta
        target_cd = np.ones(n_eta) * 0.5
        target_pc = np.ones(n_eta) * 0.3

        obj = AlphaOnlySurrogateObjective(
            surrogate=fitted_surrogate,
            target_cd=target_cd,
            target_pc=target_pc,
            fixed_k0=[0.01, 0.001],
        )
        assert obj.secondary_weight == 1.0


# ---------------------------------------------------------------------------
# Tests: format_results_table
# ---------------------------------------------------------------------------

class TestFormatResultsTable:
    """Tests for the results table formatter."""

    def test_header_and_rows(self):
        from scripts.surrogate.sweep_secondary_weight import (
            WeightSweepResult, format_results_table,
        )

        results = [
            WeightSweepResult(
                weight=1.0,
                k0_1=0.01, k0_2=0.001,
                alpha_1=0.6, alpha_2=0.5,
                k0_1_err_pct=5.0, k0_2_err_pct=3.0,
                alpha_1_err_pct=2.0, alpha_2_err_pct=1.5,
                max_err_pct=5.0,
                loss=0.001, elapsed_s=1.5,
            ),
            WeightSweepResult(
                weight=2.0,
                k0_1=0.01, k0_2=0.001,
                alpha_1=0.6, alpha_2=0.5,
                k0_1_err_pct=4.0, k0_2_err_pct=2.0,
                alpha_1_err_pct=1.5, alpha_2_err_pct=1.0,
                max_err_pct=4.0,
                loss=0.002, elapsed_s=1.3,
            ),
        ]
        table = format_results_table(results)
        lines = table.split("\n")

        # Header + separator + 2 data rows
        assert len(lines) == 4
        assert "weight" in lines[0]
        assert "1.00" in lines[2]
        assert "2.00" in lines[3]


# ---------------------------------------------------------------------------
# Tests: save_results_csv
# ---------------------------------------------------------------------------

class TestSaveResultsCsv:
    """Tests for the CSV output function."""

    def test_csv_round_trip(self):
        from scripts.surrogate.sweep_secondary_weight import (
            WeightSweepResult, save_results_csv,
        )

        results = [
            WeightSweepResult(
                weight=0.5,
                k0_1=0.012, k0_2=0.0011,
                alpha_1=0.62, alpha_2=0.51,
                k0_1_err_pct=5.5, k0_2_err_pct=3.3,
                alpha_1_err_pct=2.2, alpha_2_err_pct=1.1,
                max_err_pct=5.5,
                loss=0.0015, elapsed_s=2.0,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "test_results.csv")
            save_results_csv(results, csv_path)

            assert os.path.exists(csv_path)

            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 1
            assert float(rows[0]["secondary_weight"]) == pytest.approx(0.5)
            assert float(rows[0]["max_err_pct"]) == pytest.approx(5.5, abs=0.01)

    def test_creates_directories(self):
        from scripts.surrogate.sweep_secondary_weight import (
            WeightSweepResult, save_results_csv,
        )

        results = [
            WeightSweepResult(
                weight=1.0,
                k0_1=0.01, k0_2=0.001,
                alpha_1=0.6, alpha_2=0.5,
                k0_1_err_pct=5.0, k0_2_err_pct=3.0,
                alpha_1_err_pct=2.0, alpha_2_err_pct=1.5,
                max_err_pct=5.0,
                loss=0.001, elapsed_s=1.5,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = os.path.join(tmpdir, "sub", "dir", "results.csv")
            save_results_csv(results, nested_path)
            assert os.path.exists(nested_path)


# ---------------------------------------------------------------------------
# Tests: subset_targets helper
# ---------------------------------------------------------------------------

class TestSubsetTargets:
    """Tests for the subset_targets helper."""

    def test_correct_subset(self):
        from scripts.surrogate.sweep_secondary_weight import subset_targets

        all_eta = np.array([5.0, 3.0, 1.0, -1.0, -3.0, -5.0])
        target_cd = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        target_pc = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        subset_eta = np.array([-1.0, -5.0])

        cd_sub, pc_sub, idx = subset_targets(target_cd, target_pc, all_eta, subset_eta)

        np.testing.assert_array_equal(idx, [3, 5])
        np.testing.assert_array_equal(cd_sub, [40.0, 60.0])
        np.testing.assert_array_equal(pc_sub, [4.0, 6.0])

    def test_empty_subset(self):
        from scripts.surrogate.sweep_secondary_weight import subset_targets

        all_eta = np.array([5.0, 3.0])
        target_cd = np.array([10.0, 20.0])
        target_pc = np.array([1.0, 2.0])
        subset_eta = np.array([-99.0])

        cd_sub, pc_sub, idx = subset_targets(target_cd, target_pc, all_eta, subset_eta)

        assert len(idx) == 0
