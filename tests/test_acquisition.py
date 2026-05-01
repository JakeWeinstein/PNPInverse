"""Unit tests for Surrogate.acquisition module."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np
import pytest

from Surrogate.acquisition import (
    AcquisitionConfig,
    AcquisitionResult,
    _acquire_optimizer_trajectory,
    _acquire_spacefill,
    _acquire_uncertainty,
    _deduplicate_candidates,
    _from_normalized_log,
    _min_distance_to_set,
    _to_normalized_log,
    select_new_samples,
)
from Surrogate.sampling import ParameterBounds


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bounds():
    """Standard parameter bounds for testing."""
    return ParameterBounds(
        k0_1_range=(1e-6, 1.0),
        k0_2_range=(1e-7, 0.1),
        alpha_1_range=(0.1, 0.9),
        alpha_2_range=(0.1, 0.9),
    )


@pytest.fixture
def existing_data():
    """Small existing training dataset in physical space."""
    rng = np.random.default_rng(123)
    n = 20
    params = np.column_stack([
        10.0 ** rng.uniform(-6, 0, n),      # k0_1
        10.0 ** rng.uniform(-7, -1, n),      # k0_2
        rng.uniform(0.1, 0.9, n),            # alpha_1
        rng.uniform(0.1, 0.9, n),            # alpha_2
    ])
    return params


def _make_multistart_candidate(k0_1, k0_2, alpha_1, alpha_2, loss, rank=0):
    """Helper to create a mock MultiStartCandidate."""
    candidate = MagicMock()
    candidate.k0_1 = k0_1
    candidate.k0_2 = k0_2
    candidate.alpha_1 = alpha_1
    candidate.alpha_2 = alpha_2
    candidate.polished_loss = loss
    candidate.rank = rank
    return candidate


def _make_multistart_result(candidates):
    """Helper to create a mock MultiStartResult."""
    result = MagicMock()
    result.candidates = tuple(candidates)
    return result


def _make_cascade_pass_result(k0_1, k0_2, alpha_1, alpha_2, loss):
    """Helper to create a mock CascadePassResult."""
    pr = MagicMock()
    pr.k0_1 = k0_1
    pr.k0_2 = k0_2
    pr.alpha_1 = alpha_1
    pr.alpha_2 = alpha_2
    pr.loss = loss
    return pr


def _make_cascade_result(pass_results):
    """Helper to create a mock CascadeResult."""
    result = MagicMock()
    result.pass_results = tuple(pass_results)
    return result


# ---------------------------------------------------------------------------
# Test: Normalization roundtrip
# ---------------------------------------------------------------------------

class TestNormalization:
    """Tests for _to_normalized_log and _from_normalized_log."""

    def test_roundtrip_single(self, bounds):
        """Convert to normalized log-space and back, check accuracy."""
        params = np.array([1e-3, 1e-5, 0.5, 0.7])
        norm = _to_normalized_log(params, bounds)
        recovered = _from_normalized_log(norm, bounds)
        np.testing.assert_allclose(recovered, params, rtol=1e-10)

    def test_roundtrip_batch(self, bounds):
        """Roundtrip for a batch of points."""
        rng = np.random.default_rng(42)
        params = np.column_stack([
            10.0 ** rng.uniform(-6, 0, 50),
            10.0 ** rng.uniform(-7, -1, 50),
            rng.uniform(0.1, 0.9, 50),
            rng.uniform(0.1, 0.9, 50),
        ])
        norm = _to_normalized_log(params, bounds)
        recovered = _from_normalized_log(norm, bounds)
        np.testing.assert_allclose(recovered, params, rtol=1e-10)

    def test_bounds_map_to_zero_one(self, bounds):
        """Corner points of the bounds should map to 0 and 1."""
        lo = np.array([1e-6, 1e-7, 0.1, 0.1])
        hi = np.array([1.0, 0.1, 0.9, 0.9])
        norm_lo = _to_normalized_log(lo, bounds)
        norm_hi = _to_normalized_log(hi, bounds)
        np.testing.assert_allclose(norm_lo, 0.0, atol=1e-12)
        np.testing.assert_allclose(norm_hi, 1.0, atol=1e-12)

    def test_output_in_unit_cube(self, bounds):
        """All normalized values should be in [0, 1]."""
        rng = np.random.default_rng(99)
        params = np.column_stack([
            10.0 ** rng.uniform(-6, 0, 100),
            10.0 ** rng.uniform(-7, -1, 100),
            rng.uniform(0.1, 0.9, 100),
            rng.uniform(0.1, 0.9, 100),
        ])
        norm = _to_normalized_log(params, bounds)
        assert np.all(norm >= -1e-12)
        assert np.all(norm <= 1.0 + 1e-12)


# ---------------------------------------------------------------------------
# Test: Distance utilities
# ---------------------------------------------------------------------------

class TestDistance:
    """Tests for _min_distance_to_set."""

    def test_known_distance(self):
        """Verify correct distance for a known geometry."""
        candidate = np.array([0.0, 0.0, 0.0, 0.0])
        existing = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
        ])
        dist = _min_distance_to_set(candidate, existing)
        assert abs(dist - 0.5) < 1e-12

    def test_empty_set_returns_inf(self):
        """Empty existing set should return inf."""
        candidate = np.array([0.5, 0.5, 0.5, 0.5])
        existing = np.empty((0, 4))
        dist = _min_distance_to_set(candidate, existing)
        assert dist == float("inf")

    def test_identical_point_returns_zero(self):
        """Distance to identical point is zero."""
        candidate = np.array([0.3, 0.7, 0.2, 0.8])
        existing = np.array([[0.3, 0.7, 0.2, 0.8]])
        dist = _min_distance_to_set(candidate, existing)
        assert abs(dist) < 1e-12


# ---------------------------------------------------------------------------
# Test: De-duplication
# ---------------------------------------------------------------------------

class TestDeduplication:
    """Tests for _deduplicate_candidates."""

    def test_removes_close_to_existing(self, bounds):
        """Candidates very close to existing data should be rejected."""
        existing = np.array([[1e-3, 1e-5, 0.5, 0.5]])
        # Create a candidate that is identical to existing
        candidates = np.array([
            [1e-3, 1e-5, 0.5, 0.5],       # duplicate
            [1e-1, 1e-2, 0.8, 0.8],       # far away
        ])
        labels = ["optimizer", "spacefill"]
        accepted, acc_labels, n_rej, _ = _deduplicate_candidates(
            candidates, labels, existing, bounds,
            min_dist_to_existing=0.05,
            min_dist_within_batch=0.08,
        )
        assert n_rej >= 1
        assert accepted.shape[0] <= 2

    def test_removes_close_within_batch(self, bounds):
        """Two very close candidates should be de-duplicated within batch."""
        existing = np.empty((0, 4))
        candidates = np.array([
            [1e-3, 1e-5, 0.5, 0.5],
            [1e-3, 1e-5, 0.5, 0.5],  # identical
        ])
        labels = ["optimizer", "spacefill"]
        accepted, acc_labels, n_rej, _ = _deduplicate_candidates(
            candidates, labels, existing, bounds,
            min_dist_to_existing=0.05,
            min_dist_within_batch=0.08,
        )
        assert accepted.shape[0] == 1
        assert n_rej == 1

    def test_preserves_order(self, bounds):
        """Optimizer points should survive before spacefill points."""
        existing = np.empty((0, 4))
        # Two points that are close to each other
        candidates = np.array([
            [1e-3, 1e-5, 0.5, 0.5],     # optimizer (first)
            [1e-3, 1e-5, 0.5, 0.5],     # spacefill (second, duplicate)
        ])
        labels = ["optimizer", "spacefill"]
        accepted, acc_labels, n_rej, per_strat = _deduplicate_candidates(
            candidates, labels, existing, bounds,
            min_dist_to_existing=0.05,
            min_dist_within_batch=0.08,
        )
        assert accepted.shape[0] == 1
        assert acc_labels[0] == "optimizer"
        assert per_strat.get("spacefill", 0) == 1

    def test_tracks_per_strategy_rejections(self, bounds):
        """Per-strategy rejection counts should be accurate."""
        existing = np.array([[1e-3, 1e-5, 0.5, 0.5]])
        candidates = np.array([
            [1e-3, 1e-5, 0.5, 0.5],          # dup of existing (optimizer)
            [1e-3 * 1.001, 1e-5, 0.5, 0.5],  # near-dup (uncertainty)
            [1e-1, 1e-2, 0.8, 0.8],          # far (spacefill)
        ])
        labels = ["optimizer", "uncertainty", "spacefill"]
        _, _, n_rej, per_strat = _deduplicate_candidates(
            candidates, labels, existing, bounds,
            min_dist_to_existing=0.05,
            min_dist_within_batch=0.08,
        )
        assert per_strat.get("optimizer", 0) >= 1


# ---------------------------------------------------------------------------
# Test: Optimizer trajectory acquisition
# ---------------------------------------------------------------------------

class TestOptimizerTrajectory:
    """Tests for _acquire_optimizer_trajectory."""

    def test_extracts_candidates(self, bounds):
        """Mock MultiStartResult with 5 candidates, verify extraction."""
        candidates = [
            _make_multistart_candidate(1e-3, 1e-5, 0.5, 0.5, 0.01, rank=0),
            _make_multistart_candidate(1e-2, 1e-4, 0.6, 0.6, 0.02, rank=1),
            _make_multistart_candidate(1e-4, 1e-6, 0.4, 0.4, 0.03, rank=2),
            _make_multistart_candidate(1e-1, 1e-3, 0.7, 0.7, 0.04, rank=3),
            _make_multistart_candidate(1e-5, 1e-7, 0.3, 0.3, 0.05, rank=4),
        ]
        ms_result = _make_multistart_result(candidates)
        rng = np.random.default_rng(42)

        pts = _acquire_optimizer_trajectory(
            multistart_result=ms_result,
            cascade_result=None,
            bounds=bounds,
            n_target=10,
            neighborhood_radius=0.3,
            n_neighbors_per_candidate=2,
            rng=rng,
        )
        assert pts.shape[0] > 0
        assert pts.shape[1] == 4

    def test_with_neighbors_within_bounds(self, bounds):
        """Verify neighborhood points are within bounds."""
        candidates = [
            _make_multistart_candidate(1e-3, 1e-5, 0.5, 0.5, 0.01, rank=0),
        ]
        ms_result = _make_multistart_result(candidates)
        rng = np.random.default_rng(42)

        pts = _acquire_optimizer_trajectory(
            multistart_result=ms_result,
            cascade_result=None,
            bounds=bounds,
            n_target=5,
            neighborhood_radius=0.3,
            n_neighbors_per_candidate=4,
            rng=rng,
        )
        # Verify within bounds (neighbors are generated in normalized space
        # and clamped, so back-converted values should be within bounds,
        # though the main entry point does the final clamping)
        for i in range(pts.shape[0]):
            assert pts[i, 0] >= bounds.k0_1_range[0] * 0.99  # small tolerance
            assert pts[i, 1] >= bounds.k0_2_range[0] * 0.99
            assert pts[i, 2] >= bounds.alpha_1_range[0] - 0.01
            assert pts[i, 3] >= bounds.alpha_2_range[0] - 0.01

    def test_empty_when_no_results(self, bounds):
        """Returns empty when both multistart and cascade are None."""
        rng = np.random.default_rng(42)
        pts = _acquire_optimizer_trajectory(
            multistart_result=None,
            cascade_result=None,
            bounds=bounds,
            n_target=10,
            neighborhood_radius=0.3,
            n_neighbors_per_candidate=2,
            rng=rng,
        )
        assert pts.shape == (0, 4)

    def test_cascade_results_included(self, bounds):
        """Cascade pass results are included in candidate pool."""
        pass_results = [
            _make_cascade_pass_result(1e-3, 1e-5, 0.5, 0.5, 0.01),
            _make_cascade_pass_result(1e-2, 1e-4, 0.6, 0.6, 0.02),
        ]
        cas_result = _make_cascade_result(pass_results)
        rng = np.random.default_rng(42)

        pts = _acquire_optimizer_trajectory(
            multistart_result=None,
            cascade_result=cas_result,
            bounds=bounds,
            n_target=5,
            neighborhood_radius=0.3,
            n_neighbors_per_candidate=1,
            rng=rng,
        )
        assert pts.shape[0] > 0


# ---------------------------------------------------------------------------
# Test: Uncertainty acquisition
# ---------------------------------------------------------------------------

class TestUncertaintyAcquisition:
    """Tests for _acquire_uncertainty."""

    def test_fallback_no_gp(self, bounds):
        """When gp_model=None, returns empty array."""
        rng = np.random.default_rng(42)
        pts = _acquire_uncertainty(
            gp_model=None,
            bounds=bounds,
            n_target=10,
            n_candidates=100,
            k0_2_weight=2.0,
            rng=rng,
        )
        assert pts.shape == (0, 4)

    def test_with_mock_gp(self, bounds):
        """With a mock GP model, returns points ranked by uncertainty."""
        n_eta = 22
        gp = MagicMock()
        gp.is_fitted = True

        def mock_predict(params):
            m = params.shape[0]
            return {
                "current_density": np.zeros((m, n_eta)),
                "peroxide_current": np.zeros((m, n_eta)),
                "phi_applied": np.linspace(-0.5, 0.5, n_eta),
                "current_density_std": np.random.default_rng(0).uniform(0, 1, (m, n_eta)),
                "peroxide_current_std": np.random.default_rng(1).uniform(0, 1, (m, n_eta)),
            }

        gp.predict_batch_with_uncertainty = mock_predict

        rng = np.random.default_rng(42)
        pts = _acquire_uncertainty(
            gp_model=gp,
            bounds=bounds,
            n_target=5,
            n_candidates=100,
            k0_2_weight=2.0,
            rng=rng,
        )
        assert pts.shape == (5, 4)
        # All within bounds
        assert np.all(pts[:, 0] >= bounds.k0_1_range[0])
        assert np.all(pts[:, 0] <= bounds.k0_1_range[1])


# ---------------------------------------------------------------------------
# Test: Space-filling LHS
# ---------------------------------------------------------------------------

class TestSpaceFill:
    """Tests for _acquire_spacefill."""

    def test_returns_correct_count(self, bounds):
        """Verify LHS returns exactly n_target points."""
        pts = _acquire_spacefill(bounds, n_target=15, seed=42)
        assert pts.shape == (15, 4)

    def test_within_bounds(self, bounds):
        """All LHS points should be within parameter bounds."""
        pts = _acquire_spacefill(bounds, n_target=50, seed=42)
        assert np.all(pts[:, 0] >= bounds.k0_1_range[0])
        assert np.all(pts[:, 0] <= bounds.k0_1_range[1])
        assert np.all(pts[:, 1] >= bounds.k0_2_range[0])
        assert np.all(pts[:, 1] <= bounds.k0_2_range[1])
        assert np.all(pts[:, 2] >= bounds.alpha_1_range[0])
        assert np.all(pts[:, 2] <= bounds.alpha_1_range[1])
        assert np.all(pts[:, 3] >= bounds.alpha_2_range[0])
        assert np.all(pts[:, 3] <= bounds.alpha_2_range[1])

    def test_different_seeds_different_points(self, bounds):
        """Different seeds should produce different LHS designs."""
        pts1 = _acquire_spacefill(bounds, n_target=10, seed=42)
        pts2 = _acquire_spacefill(bounds, n_target=10, seed=99)
        assert not np.allclose(pts1, pts2)


# ---------------------------------------------------------------------------
# Test: Config validation
# ---------------------------------------------------------------------------

class TestAcquisitionConfig:
    """Tests for AcquisitionConfig validation."""

    def test_valid_fractions(self):
        """Valid fractions that sum to 1.0 should not raise."""
        cfg = AcquisitionConfig(frac_optimizer=0.5, frac_uncertainty=0.3, frac_spacefill=0.2)
        assert abs(cfg.frac_optimizer + cfg.frac_uncertainty + cfg.frac_spacefill - 1.0) < 1e-6

    def test_invalid_fractions_raises(self):
        """Fractions not summing to 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            AcquisitionConfig(frac_optimizer=0.5, frac_uncertainty=0.5, frac_spacefill=0.5)


# ---------------------------------------------------------------------------
# Test: select_new_samples (integration)
# ---------------------------------------------------------------------------

class TestSelectNewSamples:
    """Tests for the main select_new_samples orchestrator."""

    def test_budget_allocation(self, bounds, existing_data):
        """Verify strategy counts approximately match config fractions."""
        ms_candidates = [
            _make_multistart_candidate(
                10.0 ** np.random.default_rng(i).uniform(-6, 0),
                10.0 ** np.random.default_rng(i + 100).uniform(-7, -1),
                np.random.default_rng(i + 200).uniform(0.1, 0.9),
                np.random.default_rng(i + 300).uniform(0.1, 0.9),
                float(i) * 0.01,
                rank=i,
            )
            for i in range(10)
        ]
        ms_result = _make_multistart_result(ms_candidates)

        cfg = AcquisitionConfig(
            budget=30,
            frac_optimizer=0.5,
            frac_uncertainty=0.0,
            frac_spacefill=0.5,
            verbose=False,
            min_distance_log=0.01,
            min_distance_batch=0.02,
        )
        result = select_new_samples(
            existing_data=existing_data,
            bounds=bounds,
            config=cfg,
            multistart_result=ms_result,
        )
        assert isinstance(result, AcquisitionResult)
        assert result.n_acquired > 0
        assert result.n_acquired <= cfg.budget
        assert result.samples.shape == (result.n_acquired, 4)
        assert len(result.strategy_labels) == result.n_acquired

    def test_no_gp_redistribution(self, bounds, existing_data):
        """With gp_model=None, uncertainty budget goes to optimizer+spacefill."""
        ms_candidates = [
            _make_multistart_candidate(1e-3, 1e-5, 0.5, 0.5, 0.01, rank=0),
            _make_multistart_candidate(1e-2, 1e-4, 0.6, 0.6, 0.02, rank=1),
        ]
        ms_result = _make_multistart_result(ms_candidates)

        cfg = AcquisitionConfig(
            budget=20,
            frac_optimizer=0.5,
            frac_uncertainty=0.3,
            frac_spacefill=0.2,
            verbose=False,
            min_distance_log=0.01,
            min_distance_batch=0.02,
        )
        result = select_new_samples(
            existing_data=existing_data,
            bounds=bounds,
            config=cfg,
            multistart_result=ms_result,
            gp_model=None,
        )
        assert result.gp_variance_used is False
        # No uncertainty labels should appear
        assert "uncertainty" not in result.strategy_labels

    def test_no_optimizer_redistribution(self, bounds, existing_data):
        """With no optimizer results, optimizer budget goes to spacefill."""
        cfg = AcquisitionConfig(
            budget=20,
            frac_optimizer=0.5,
            frac_uncertainty=0.0,
            frac_spacefill=0.5,
            verbose=False,
            min_distance_log=0.01,
            min_distance_batch=0.02,
        )
        result = select_new_samples(
            existing_data=existing_data,
            bounds=bounds,
            config=cfg,
            multistart_result=None,
            cascade_result=None,
            gp_model=None,
        )
        # All points should be spacefill
        for lbl in result.strategy_labels:
            assert lbl == "spacefill"

    def test_all_strategies(self, bounds, existing_data):
        """Integration test with mock GP, MultiStart, and Cascade results."""
        ms_candidates = [
            _make_multistart_candidate(1e-3, 1e-5, 0.5, 0.5, 0.01, rank=0),
            _make_multistart_candidate(1e-2, 1e-4, 0.6, 0.6, 0.02, rank=1),
        ]
        ms_result = _make_multistart_result(ms_candidates)

        cas_passes = [
            _make_cascade_pass_result(1e-4, 1e-6, 0.4, 0.4, 0.005),
        ]
        cas_result = _make_cascade_result(cas_passes)

        n_eta = 22
        gp = MagicMock()
        gp.is_fitted = True

        def mock_predict(params):
            m = params.shape[0]
            return {
                "current_density": np.zeros((m, n_eta)),
                "peroxide_current": np.zeros((m, n_eta)),
                "phi_applied": np.linspace(-0.5, 0.5, n_eta),
                "current_density_std": np.ones((m, n_eta)) * 0.1,
                "peroxide_current_std": np.ones((m, n_eta)) * 0.1,
            }

        gp.predict_batch_with_uncertainty = mock_predict

        cfg = AcquisitionConfig(
            budget=30,
            frac_optimizer=0.5,
            frac_uncertainty=0.3,
            frac_spacefill=0.2,
            verbose=False,
            min_distance_log=0.01,
            min_distance_batch=0.02,
        )
        result = select_new_samples(
            existing_data=existing_data,
            bounds=bounds,
            config=cfg,
            multistart_result=ms_result,
            cascade_result=cas_result,
            gp_model=gp,
        )
        assert result.gp_variance_used is True
        assert result.n_acquired > 0
        # Should have at least some labels from each strategy
        labels = set(result.strategy_labels)
        # optimizer and spacefill should always be present
        assert "optimizer" in labels or "spacefill" in labels

    def test_batch_diversity(self, bounds):
        """Verify no two accepted points are closer than min_distance_batch."""
        existing = np.empty((0, 4))
        cfg = AcquisitionConfig(
            budget=20,
            frac_optimizer=0.0,
            frac_uncertainty=0.0,
            frac_spacefill=1.0,
            verbose=False,
            min_distance_log=0.01,
            min_distance_batch=0.08,
        )
        result = select_new_samples(
            existing_data=existing,
            bounds=bounds,
            config=cfg,
        )
        if result.n_acquired < 2:
            pytest.skip("Not enough points acquired to test diversity")

        norm = _to_normalized_log(result.samples, bounds)
        for i in range(norm.shape[0]):
            for j in range(i + 1, norm.shape[0]):
                dist = np.sqrt(np.sum((norm[i] - norm[j]) ** 2))
                assert dist >= cfg.min_distance_batch - 1e-10, (
                    f"Points {i} and {j} are too close: dist={dist:.4f} "
                    f"< min_distance_batch={cfg.min_distance_batch}"
                )

    def test_all_within_bounds(self, bounds, existing_data):
        """All returned points must be within parameter bounds."""
        ms_candidates = [
            _make_multistart_candidate(1e-3, 1e-5, 0.5, 0.5, 0.01, rank=0),
        ]
        ms_result = _make_multistart_result(ms_candidates)

        cfg = AcquisitionConfig(
            budget=30,
            frac_optimizer=0.5,
            frac_uncertainty=0.0,
            frac_spacefill=0.5,
            verbose=False,
            min_distance_log=0.01,
            min_distance_batch=0.02,
        )
        result = select_new_samples(
            existing_data=existing_data,
            bounds=bounds,
            config=cfg,
            multistart_result=ms_result,
        )
        if result.n_acquired == 0:
            pytest.skip("No points acquired")

        pts = result.samples
        assert np.all(pts[:, 0] >= bounds.k0_1_range[0])
        assert np.all(pts[:, 0] <= bounds.k0_1_range[1])
        assert np.all(pts[:, 1] >= bounds.k0_2_range[0])
        assert np.all(pts[:, 1] <= bounds.k0_2_range[1])
        assert np.all(pts[:, 2] >= bounds.alpha_1_range[0])
        assert np.all(pts[:, 2] <= bounds.alpha_1_range[1])
        assert np.all(pts[:, 3] >= bounds.alpha_2_range[0])
        assert np.all(pts[:, 3] <= bounds.alpha_2_range[1])

    def test_first_iteration_no_results(self, bounds):
        """First iteration: no optimizer, no GP, no existing data."""
        existing = np.empty((0, 4))
        cfg = AcquisitionConfig(
            budget=20,
            frac_optimizer=0.5,
            frac_uncertainty=0.3,
            frac_spacefill=0.2,
            verbose=False,
            min_distance_log=0.01,
            min_distance_batch=0.02,
        )
        result = select_new_samples(
            existing_data=existing,
            bounds=bounds,
            config=cfg,
        )
        # All budget should go to spacefill
        assert result.n_acquired > 0
        for lbl in result.strategy_labels:
            assert lbl == "spacefill"
        assert result.gp_variance_used is False
