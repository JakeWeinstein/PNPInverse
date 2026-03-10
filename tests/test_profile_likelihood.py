"""Tests for profile likelihood grid construction and identifiability assessment.

Tests cover:
- build_profile_grid: correct range/spacing per parameter type
- assess_identifiability: parabolic, flat, one-sided profiles
- build_fixed_bounds: pin profiled parameter while keeping others free
"""

from __future__ import annotations

import numpy as np
import pytest

# Import functions from the profile likelihood script
from scripts.studies.profile_likelihood_pde import (
    build_profile_grid,
    assess_identifiability,
    build_fixed_bounds,
)


# ---------------------------------------------------------------------------
# build_profile_grid tests
# ---------------------------------------------------------------------------

class TestBuildProfileGrid:
    """Tests for grid construction per parameter type."""

    def test_k0_1_grid_log_spaced_30_points(self):
        """k0_1 grid: 30 log-spaced points from 0.01x to 100x true value."""
        true_val = 1.2632e-3
        grid = build_profile_grid("k0_1", true_val, n_points=30)
        assert len(grid) == 30
        assert grid[0] == pytest.approx(true_val * 0.01, rel=1e-6)
        assert grid[-1] == pytest.approx(true_val * 100, rel=1e-6)
        # Check log-spacing: ratios between consecutive points should be constant
        ratios = grid[1:] / grid[:-1]
        assert np.allclose(ratios, ratios[0], rtol=1e-6)

    def test_k0_2_grid_wider_range(self):
        """k0_2 grid: 30 log-spaced points from 0.001x to 1000x (wider for identifiability risk)."""
        true_val = 5.2632e-5
        grid = build_profile_grid("k0_2", true_val, n_points=30)
        assert len(grid) == 30
        assert grid[0] == pytest.approx(true_val * 0.001, rel=1e-6)
        assert grid[-1] == pytest.approx(true_val * 1000, rel=1e-6)
        # Check log-spacing
        ratios = grid[1:] / grid[:-1]
        assert np.allclose(ratios, ratios[0], rtol=1e-6)

    def test_alpha_1_grid_linear_spaced(self):
        """alpha_1 grid: 30 linearly-spaced points from 0.1 to 0.95."""
        true_val = 0.627
        grid = build_profile_grid("alpha_1", true_val, n_points=30)
        assert len(grid) == 30
        assert grid[0] == pytest.approx(0.1, rel=1e-6)
        assert grid[-1] == pytest.approx(0.95, rel=1e-6)
        # Check linear spacing: differences between consecutive points should be constant
        diffs = np.diff(grid)
        assert np.allclose(diffs, diffs[0], rtol=1e-6)

    def test_alpha_2_grid_linear_spaced(self):
        """alpha_2 grid: same linear spacing as alpha_1."""
        true_val = 0.5
        grid = build_profile_grid("alpha_2", true_val, n_points=30)
        assert len(grid) == 30
        assert grid[0] == pytest.approx(0.1, rel=1e-6)
        assert grid[-1] == pytest.approx(0.95, rel=1e-6)


# ---------------------------------------------------------------------------
# assess_identifiability tests
# ---------------------------------------------------------------------------

class TestAssessIdentifiability:
    """Tests for chi-squared identifiability assessment."""

    def test_parabolic_profile_identifiable(self):
        """Parabolic (U-shaped) profile -> identifiable, both sides bounded."""
        losses = np.array([100, 50, 20, 5, 0.1, 5, 20, 50, 100], dtype=float)
        result = assess_identifiability(
            profile_losses=losses,
            global_min_loss=0.1,
            n_obs=25,
            n_params=4,
        )
        assert result["identifiable"] is True
        assert result["left_bounded"] is True
        assert result["right_bounded"] is True
        assert "chi2_profile" in result
        assert len(result["chi2_profile"]) == len(losses)

    def test_flat_profile_not_identifiable(self):
        """Flat profile -> not identifiable (many parameter combos achieve same loss)."""
        losses = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=float)
        result = assess_identifiability(
            profile_losses=losses,
            global_min_loss=0.1,
            n_obs=25,
            n_params=4,
        )
        assert result["identifiable"] is False

    def test_one_sided_profile_partial(self):
        """One-sided profile: left bounded but right unbounded -> not identifiable."""
        losses = np.array([100, 50, 20, 5, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=float)
        result = assess_identifiability(
            profile_losses=losses,
            global_min_loss=0.1,
            n_obs=25,
            n_params=4,
        )
        assert result["left_bounded"] is True
        assert result["right_bounded"] is False
        assert result["identifiable"] is False


# ---------------------------------------------------------------------------
# build_fixed_bounds tests
# ---------------------------------------------------------------------------

class TestBuildFixedBounds:
    """Tests for bound construction with pinned profiled parameter."""

    def test_pin_k0_1(self):
        """Pinning k0_1 sets lower[0] == upper[0] == fixed_val, others unchanged."""
        default_bounds = {
            "k0_lower": [1e-8, 1e-8],
            "k0_upper": [100.0, 100.0],
            "alpha_lower": [0.05, 0.05],
            "alpha_upper": [0.95, 0.95],
        }
        fixed_val = 0.005
        result = build_fixed_bounds("k0_1", fixed_val, default_bounds)
        # k0_1 is component 0
        assert result["k0_lower"][0] == pytest.approx(fixed_val)
        assert result["k0_upper"][0] == pytest.approx(fixed_val)
        # k0_2 remains free
        assert result["k0_lower"][1] == pytest.approx(1e-8)
        assert result["k0_upper"][1] == pytest.approx(100.0)
        # alpha bounds unchanged
        assert result["alpha_lower"] == [0.05, 0.05]
        assert result["alpha_upper"] == [0.95, 0.95]

    def test_pin_alpha_2(self):
        """Pinning alpha_2 sets alpha_lower[1] == alpha_upper[1] == fixed_val."""
        default_bounds = {
            "k0_lower": [1e-8, 1e-8],
            "k0_upper": [100.0, 100.0],
            "alpha_lower": [0.05, 0.05],
            "alpha_upper": [0.95, 0.95],
        }
        fixed_val = 0.7
        result = build_fixed_bounds("alpha_2", fixed_val, default_bounds)
        # alpha_2 is component 1
        assert result["alpha_lower"][1] == pytest.approx(fixed_val)
        assert result["alpha_upper"][1] == pytest.approx(fixed_val)
        # alpha_1 remains free
        assert result["alpha_lower"][0] == pytest.approx(0.05)
        assert result["alpha_upper"][0] == pytest.approx(0.95)
        # k0 bounds unchanged
        assert result["k0_lower"] == [1e-8, 1e-8]
        assert result["k0_upper"] == [100.0, 100.0]

    def test_does_not_mutate_input(self):
        """build_fixed_bounds must not mutate the input default_bounds dict."""
        default_bounds = {
            "k0_lower": [1e-8, 1e-8],
            "k0_upper": [100.0, 100.0],
            "alpha_lower": [0.05, 0.05],
            "alpha_upper": [0.95, 0.95],
        }
        import copy
        original = copy.deepcopy(default_bounds)
        build_fixed_bounds("k0_1", 0.005, default_bounds)
        assert default_bounds == original
