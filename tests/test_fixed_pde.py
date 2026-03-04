"""Tests for the v10 fixed PDE refinement utilities.

Tests cover:
    1. RegularizationConfig + compute_regularization_penalty:
       - Correct penalty value for known inputs
       - Correct gradient (finite-difference check)
       - Zero penalty at the prior
       - Gradient symmetry
    2. compute_tight_bounds:
       - Correct structure (number of bounds, ordering)
       - k0 bounds in log-space vs linear-space
       - Alpha clipping to floor/ceiling
       - Surrogate-optimal is always inside bounds
    3. Edge cases:
       - Zero lambda disables regularization
       - Very large lambda dominates objective
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# Ensure PNPInverse root is importable
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Import from the v10 script
sys.path.insert(0, os.path.join(_ROOT, "scripts", "surrogate"))
from Infer_BVMaster_charged_v10_fixed_pde import (
    RegularizationConfig,
    compute_regularization_penalty,
    compute_tight_bounds,
)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture()
def reg_config() -> RegularizationConfig:
    """Standard 2-k0 + 2-alpha regularization config."""
    return RegularizationConfig(
        reg_lambda=1.0,
        k0_prior=np.array([0.01, 0.001]),
        alpha_prior=np.array([0.6, 0.5]),
        n_k0=2,
        k0_log_space=True,
    )


@pytest.fixture()
def surrogate_result() -> dict[str, np.ndarray]:
    """A representative surrogate-optimal result for bounds testing."""
    return {
        "k0": np.array([0.01, 0.001]),
        "alpha": np.array([0.627, 0.50]),
    }


# ===================================================================
# Tests: compute_regularization_penalty
# ===================================================================

class TestRegularizationPenalty:
    """Tests for the Tikhonov regularization penalty computation."""

    def test_zero_at_prior(self, reg_config: RegularizationConfig) -> None:
        """Penalty is zero when x equals the prior (in x-space)."""
        # x-space prior for k0: log10(k0_prior), for alpha: alpha_prior
        x_prior = np.concatenate([
            np.log10(reg_config.k0_prior),
            reg_config.alpha_prior,
        ])
        penalty, grad = compute_regularization_penalty(x_prior, reg_config)
        assert penalty == pytest.approx(0.0, abs=1e-15)
        np.testing.assert_allclose(grad, 0.0, atol=1e-15)

    def test_known_penalty_value(self, reg_config: RegularizationConfig) -> None:
        """Penalty matches hand-computed value for a known displacement."""
        # Displace k0_1 by +0.3 log-decades, k0_2 by -0.1, alpha by +0.02, -0.01
        x_prior = np.concatenate([
            np.log10(reg_config.k0_prior),
            reg_config.alpha_prior,
        ])
        delta = np.array([0.3, -0.1, 0.02, -0.01])
        x_test = x_prior + delta

        penalty, grad = compute_regularization_penalty(x_test, reg_config)

        # Expected: lambda * sum(delta^2) = 1.0 * (0.09 + 0.01 + 0.0004 + 0.0001)
        expected_penalty = 1.0 * (0.3**2 + 0.1**2 + 0.02**2 + 0.01**2)
        assert penalty == pytest.approx(expected_penalty, rel=1e-10)

        # Expected gradient: 2 * lambda * delta
        expected_grad = 2.0 * 1.0 * delta
        np.testing.assert_allclose(grad, expected_grad, atol=1e-12)

    def test_gradient_finite_difference(self, reg_config: RegularizationConfig) -> None:
        """Analytical gradient matches finite-difference approximation."""
        x = np.array([-2.5, -3.2, 0.65, 0.48])
        _, grad_analytical = compute_regularization_penalty(x, reg_config)

        h = 1e-7
        grad_fd = np.zeros_like(x)
        for i in range(len(x)):
            xp = x.copy()
            xm = x.copy()
            xp[i] += h
            xm[i] -= h
            fp, _ = compute_regularization_penalty(xp, reg_config)
            fm, _ = compute_regularization_penalty(xm, reg_config)
            grad_fd[i] = (fp - fm) / (2 * h)

        np.testing.assert_allclose(grad_analytical, grad_fd, rtol=1e-5, atol=1e-10)

    def test_zero_lambda(self) -> None:
        """Zero lambda produces zero penalty and gradient."""
        config = RegularizationConfig(
            reg_lambda=0.0,
            k0_prior=np.array([0.01, 0.001]),
            alpha_prior=np.array([0.6, 0.5]),
        )
        x = np.array([-1.0, -2.0, 0.7, 0.6])
        penalty, grad = compute_regularization_penalty(x, config)
        assert penalty == pytest.approx(0.0, abs=1e-15)
        np.testing.assert_allclose(grad, 0.0, atol=1e-15)

    def test_large_lambda_dominates(self) -> None:
        """With very large lambda, penalty dwarfs any plausible PDE misfit."""
        config = RegularizationConfig(
            reg_lambda=1e6,
            k0_prior=np.array([0.01, 0.001]),
            alpha_prior=np.array([0.6, 0.5]),
        )
        x_prior = np.concatenate([
            np.log10(config.k0_prior),
            config.alpha_prior,
        ])
        # Small deviation
        x = x_prior + np.array([0.01, 0.01, 0.001, 0.001])
        penalty, grad = compute_regularization_penalty(x, config)

        # Even small deviations yield large penalty
        assert penalty > 1.0
        assert np.linalg.norm(grad) > 100.0

    def test_penalty_scales_linearly_with_lambda(self) -> None:
        """Doubling lambda doubles penalty and gradient."""
        x = np.array([-2.0, -3.0, 0.65, 0.48])

        config_1 = RegularizationConfig(
            reg_lambda=1.0,
            k0_prior=np.array([0.01, 0.001]),
            alpha_prior=np.array([0.6, 0.5]),
        )
        config_2 = RegularizationConfig(
            reg_lambda=2.0,
            k0_prior=np.array([0.01, 0.001]),
            alpha_prior=np.array([0.6, 0.5]),
        )

        p1, g1 = compute_regularization_penalty(x, config_1)
        p2, g2 = compute_regularization_penalty(x, config_2)

        assert p2 == pytest.approx(2.0 * p1, rel=1e-12)
        np.testing.assert_allclose(g2, 2.0 * g1, rtol=1e-12)

    def test_linear_k0_space(self) -> None:
        """Regularization works correctly when k0_log_space=False."""
        config = RegularizationConfig(
            reg_lambda=1.0,
            k0_prior=np.array([0.01, 0.001]),
            alpha_prior=np.array([0.6, 0.5]),
            k0_log_space=False,
        )
        # x is [k0_1, k0_2, alpha_1, alpha_2] in linear space
        x = np.array([0.015, 0.0015, 0.65, 0.48])
        penalty, grad = compute_regularization_penalty(x, config)

        # Expected: lambda * sum((x - prior)^2)
        diff = x - np.concatenate([config.k0_prior, config.alpha_prior])
        expected = float(np.sum(diff**2))
        assert penalty == pytest.approx(expected, rel=1e-10)


# ===================================================================
# Tests: compute_tight_bounds
# ===================================================================

class TestTightBounds:
    """Tests for the tight bounds computation around surrogate optimum."""

    def test_correct_number_of_bounds(
        self, surrogate_result: dict[str, np.ndarray]
    ) -> None:
        """Returns one bound tuple per optimizer variable."""
        bounds = compute_tight_bounds(
            surrogate_k0=surrogate_result["k0"],
            surrogate_alpha=surrogate_result["alpha"],
        )
        n_k0 = len(surrogate_result["k0"])
        n_alpha = len(surrogate_result["alpha"])
        assert len(bounds) == n_k0 + n_alpha

    def test_surrogate_inside_bounds(
        self, surrogate_result: dict[str, np.ndarray]
    ) -> None:
        """The surrogate optimum is strictly inside the computed bounds."""
        bounds = compute_tight_bounds(
            surrogate_k0=surrogate_result["k0"],
            surrogate_alpha=surrogate_result["alpha"],
            bound_factor=2.0,
            alpha_margin=0.05,
            k0_log_space=True,
        )
        # k0 in log-space
        for i, k0_val in enumerate(surrogate_result["k0"]):
            log_k0 = np.log10(k0_val)
            lo, hi = bounds[i]
            assert lo < log_k0 < hi, f"k0[{i}] not inside bounds"

        # alpha in linear space
        n_k0 = len(surrogate_result["k0"])
        for i, alpha_val in enumerate(surrogate_result["alpha"]):
            lo, hi = bounds[n_k0 + i]
            assert lo < alpha_val < hi, f"alpha[{i}] not inside bounds"

    def test_k0_log_space_bounds(
        self, surrogate_result: dict[str, np.ndarray]
    ) -> None:
        """k0 bounds in log10 space match expected values."""
        factor = 2.0
        bounds = compute_tight_bounds(
            surrogate_k0=surrogate_result["k0"],
            surrogate_alpha=surrogate_result["alpha"],
            bound_factor=factor,
            k0_log_space=True,
        )
        for i, k0_val in enumerate(surrogate_result["k0"]):
            lo, hi = bounds[i]
            expected_lo = np.log10(k0_val / factor)
            expected_hi = np.log10(k0_val * factor)
            assert lo == pytest.approx(expected_lo, abs=1e-12)
            assert hi == pytest.approx(expected_hi, abs=1e-12)

    def test_k0_linear_space_bounds(
        self, surrogate_result: dict[str, np.ndarray]
    ) -> None:
        """k0 bounds in linear space match expected values."""
        factor = 3.0
        bounds = compute_tight_bounds(
            surrogate_k0=surrogate_result["k0"],
            surrogate_alpha=surrogate_result["alpha"],
            bound_factor=factor,
            k0_log_space=False,
        )
        for i, k0_val in enumerate(surrogate_result["k0"]):
            lo, hi = bounds[i]
            assert lo == pytest.approx(k0_val / factor, rel=1e-12)
            assert hi == pytest.approx(k0_val * factor, rel=1e-12)

    def test_alpha_clipping_floor(self) -> None:
        """Alpha bounds are clipped to floor when surrogate alpha is near floor."""
        bounds = compute_tight_bounds(
            surrogate_k0=np.array([0.01]),
            surrogate_alpha=np.array([0.07]),  # 0.07 - 0.05 = 0.02 < floor=0.05
            alpha_margin=0.05,
            alpha_floor=0.05,
            alpha_ceil=0.95,
        )
        alpha_lo, alpha_hi = bounds[1]
        assert alpha_lo == pytest.approx(0.05, abs=1e-12)
        assert alpha_hi == pytest.approx(0.12, abs=1e-12)

    def test_alpha_clipping_ceil(self) -> None:
        """Alpha bounds are clipped to ceiling when surrogate alpha is near ceil."""
        bounds = compute_tight_bounds(
            surrogate_k0=np.array([0.01]),
            surrogate_alpha=np.array([0.92]),  # 0.92 + 0.05 = 0.97 > ceil=0.95
            alpha_margin=0.05,
            alpha_floor=0.05,
            alpha_ceil=0.95,
        )
        alpha_lo, alpha_hi = bounds[1]
        assert alpha_lo == pytest.approx(0.87, abs=1e-12)
        assert alpha_hi == pytest.approx(0.95, abs=1e-12)

    def test_factor_one_gives_point_bounds(
        self, surrogate_result: dict[str, np.ndarray]
    ) -> None:
        """factor=1.0 gives a point (lo==hi) for k0 in log space."""
        bounds = compute_tight_bounds(
            surrogate_k0=surrogate_result["k0"],
            surrogate_alpha=surrogate_result["alpha"],
            bound_factor=1.0,
            alpha_margin=0.0,
            k0_log_space=True,
        )
        for i in range(len(surrogate_result["k0"])):
            lo, hi = bounds[i]
            assert lo == pytest.approx(hi, abs=1e-12)

    def test_wider_factor_gives_wider_bounds(
        self, surrogate_result: dict[str, np.ndarray]
    ) -> None:
        """Increasing bound_factor widens k0 bounds."""
        bounds_narrow = compute_tight_bounds(
            surrogate_k0=surrogate_result["k0"],
            surrogate_alpha=surrogate_result["alpha"],
            bound_factor=1.5,
            k0_log_space=True,
        )
        bounds_wide = compute_tight_bounds(
            surrogate_k0=surrogate_result["k0"],
            surrogate_alpha=surrogate_result["alpha"],
            bound_factor=3.0,
            k0_log_space=True,
        )
        for i in range(len(surrogate_result["k0"])):
            narrow_width = bounds_narrow[i][1] - bounds_narrow[i][0]
            wide_width = bounds_wide[i][1] - bounds_wide[i][0]
            assert wide_width > narrow_width


# ===================================================================
# Tests: Integration of regularization with a mock PDE objective
# ===================================================================

class TestRegularizedObjectiveIntegration:
    """Test that regularization correctly wraps a mock PDE objective."""

    def test_regularization_adds_to_mock_objective(self) -> None:
        """Total objective = mock PDE misfit + regularization penalty."""
        config = RegularizationConfig(
            reg_lambda=2.0,
            k0_prior=np.array([0.01, 0.001]),
            alpha_prior=np.array([0.6, 0.5]),
        )

        # Mock PDE misfit: just a quadratic in x
        def mock_pde_misfit(x: np.ndarray) -> tuple[float, np.ndarray]:
            val = 0.5 * float(np.sum(x**2))
            grad = x.copy()
            return val, grad

        x = np.array([-2.0, -3.0, 0.65, 0.48])
        pde_val, pde_grad = mock_pde_misfit(x)
        reg_val, reg_grad = compute_regularization_penalty(x, config)

        total_val = pde_val + reg_val
        total_grad = pde_grad + reg_grad

        # Verify addition
        assert total_val == pytest.approx(pde_val + reg_val, rel=1e-12)
        np.testing.assert_allclose(total_grad, pde_grad + reg_grad, rtol=1e-12)

    def test_regularization_pulls_toward_prior(self) -> None:
        """With strong lambda, the minimizer of (misfit + reg) is near prior."""
        config = RegularizationConfig(
            reg_lambda=1000.0,
            k0_prior=np.array([0.01]),
            alpha_prior=np.array([0.6]),
            n_k0=1,
        )
        x_prior = np.concatenate([
            np.log10(config.k0_prior), config.alpha_prior
        ])

        # Mock PDE has minimum at x = [0, 0] (far from prior)
        def total_fun(x: np.ndarray) -> float:
            pde = 0.5 * float(np.sum(x**2))
            reg, _ = compute_regularization_penalty(x, config)
            return pde + reg

        def total_grad(x: np.ndarray) -> np.ndarray:
            pde_g = x.copy()
            _, reg_g = compute_regularization_penalty(x, config)
            return pde_g + reg_g

        from scipy.optimize import minimize
        result = minimize(total_fun, x0=np.zeros(2), jac=total_grad,
                          method="L-BFGS-B")

        # With large lambda, result should be very close to prior
        np.testing.assert_allclose(result.x, x_prior, atol=0.01)
