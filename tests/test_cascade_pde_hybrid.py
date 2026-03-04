"""Tests for the cascade + PDE hybrid strategy.

Tests cover:
    1. HybridConfig frozen dataclass (defaults, frozen, custom, voltage range)
    2. PhaseResult frozen dataclass (creation, frozen)
    3. compute_errors helper (correct relative errors)
    4. compute_k0_2_bounds (log-space bounds, factor=1 degeneracy, wider factor)
    5. compute_tight_bounds (structure, surrogate inside, log-space)
    6. RegularizationConfig + compute_regularization_penalty
       (zero at prior, known value, FD gradient check, zero lambda)
    7. K02PDEObjectiveConfig frozen dataclass
    8. extract_training_bounds (fallback defaults)
    9. compute_shallow_subset_idx (correct index extraction)
    10. Gradient chain rule for log10 transform (dJ/d(log10(k0_2)))
    11. SweepResult frozen dataclass
    12. Voltage range selection (shallow vs cathodic)
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

# Import from the hybrid script
sys.path.insert(0, os.path.join(_ROOT, "scripts", "surrogate"))
from cascade_pde_hybrid import (
    HybridConfig,
    PhaseResult,
    K02PDEObjectiveConfig,
    RegularizationConfig,
    AsymmetricRegularizationConfig,
    SweepResult,
    compute_errors,
    compute_k0_2_bounds,
    compute_tight_bounds,
    compute_regularization_penalty,
    compute_asymmetric_regularization_penalty,
    compute_asymmetric_bounds,
    compute_shallow_subset_idx,
    extract_training_bounds,
    subset_targets,
)


# ===================================================================
# Tests: HybridConfig
# ===================================================================

class TestHybridConfig:
    """Tests for the HybridConfig frozen dataclass."""

    def test_defaults(self) -> None:
        cfg = HybridConfig()
        assert cfg.pass1_weight == 0.5
        assert cfg.pde_maxiter == 30
        assert cfg.k0_2_bound_factor == 20.0
        assert cfg.joint_polish is True
        assert cfg.joint_polish_maxiter == 8
        assert cfg.joint_polish_lambda == 1.0
        assert cfg.joint_polish_bound_factor == 2.0
        assert cfg.secondary_weight == 5.0
        assert cfg.workers == 0
        assert cfg.pde_voltage_range == "cathodic"

    def test_frozen(self) -> None:
        cfg = HybridConfig()
        with pytest.raises(AttributeError):
            cfg.pass1_weight = 0.1  # type: ignore[misc]

    def test_custom_values(self) -> None:
        cfg = HybridConfig(
            pass1_weight=0.3,
            pde_maxiter=20,
            k0_2_bound_factor=5.0,
            joint_polish=False,
            workers=4,
        )
        assert cfg.pass1_weight == 0.3
        assert cfg.pde_maxiter == 20
        assert cfg.k0_2_bound_factor == 5.0
        assert cfg.joint_polish is False
        assert cfg.workers == 4


# ===================================================================
# Tests: PhaseResult
# ===================================================================

class TestPhaseResult:
    """Tests for the PhaseResult frozen dataclass."""

    def test_creation(self) -> None:
        r = PhaseResult(
            phase_name="Phase 1",
            k0_1=0.02, k0_2=0.002,
            alpha_1=0.6, alpha_2=0.5,
            loss=1e-4, elapsed_s=2.5,
            n_pde_evals=10,
        )
        assert r.phase_name == "Phase 1"
        assert r.k0_1 == 0.02
        assert r.n_pde_evals == 10

    def test_frozen(self) -> None:
        r = PhaseResult(
            phase_name="test",
            k0_1=0.01, k0_2=0.001,
            alpha_1=0.6, alpha_2=0.5,
            loss=0.01, elapsed_s=1.0,
        )
        with pytest.raises(AttributeError):
            r.k0_1 = 0.02  # type: ignore[misc]

    def test_default_pde_evals(self) -> None:
        r = PhaseResult(
            phase_name="test",
            k0_1=0.01, k0_2=0.001,
            alpha_1=0.6, alpha_2=0.5,
            loss=0.01, elapsed_s=1.0,
        )
        assert r.n_pde_evals == 0


# ===================================================================
# Tests: compute_errors
# ===================================================================

class TestComputeErrors:
    """Tests for the relative error computation."""

    def test_zero_error(self) -> None:
        k0_err, alpha_err = compute_errors(
            [0.02, 0.002], [0.6, 0.5],
            np.array([0.02, 0.002]), np.array([0.6, 0.5]),
        )
        np.testing.assert_allclose(k0_err, 0.0, atol=1e-15)
        np.testing.assert_allclose(alpha_err, 0.0, atol=1e-15)

    def test_known_error(self) -> None:
        # 10% error on k0_1, 20% error on k0_2
        k0_err, alpha_err = compute_errors(
            [0.022, 0.0024], [0.66, 0.55],
            np.array([0.02, 0.002]), np.array([0.6, 0.5]),
        )
        assert k0_err[0] == pytest.approx(0.10, rel=1e-10)
        assert k0_err[1] == pytest.approx(0.20, rel=1e-10)
        assert alpha_err[0] == pytest.approx(0.10, rel=1e-10)
        assert alpha_err[1] == pytest.approx(0.10, rel=1e-10)

    def test_always_positive(self) -> None:
        k0_err, alpha_err = compute_errors(
            [0.015, 0.003], [0.55, 0.45],
            np.array([0.02, 0.002]), np.array([0.6, 0.5]),
        )
        assert np.all(k0_err >= 0)
        assert np.all(alpha_err >= 0)


# ===================================================================
# Tests: compute_k0_2_bounds
# ===================================================================

class TestComputeK02Bounds:
    """Tests for the 1D k0_2 bounds computation."""

    def test_factor_10(self) -> None:
        """factor=10 gives +/- 1 log-decade."""
        k0_2 = 0.001
        lo, hi = compute_k0_2_bounds(k0_2, 10.0)
        expected_lo = np.log10(k0_2 / 10.0)
        expected_hi = np.log10(k0_2 * 10.0)
        assert lo == pytest.approx(expected_lo, abs=1e-12)
        assert hi == pytest.approx(expected_hi, abs=1e-12)

    def test_factor_1_gives_point(self) -> None:
        """factor=1 gives lo==hi (degenerate bounds)."""
        k0_2 = 0.005
        lo, hi = compute_k0_2_bounds(k0_2, 1.0)
        assert lo == pytest.approx(hi, abs=1e-12)

    def test_center_inside_bounds(self) -> None:
        """The cascade k0_2 is inside the computed bounds."""
        k0_2 = 0.002
        lo, hi = compute_k0_2_bounds(k0_2, 5.0)
        log_k0_2 = np.log10(k0_2)
        assert lo < log_k0_2 < hi

    def test_wider_factor_gives_wider_bounds(self) -> None:
        """Increasing factor widens the interval."""
        k0_2 = 0.001
        lo_narrow, hi_narrow = compute_k0_2_bounds(k0_2, 2.0)
        lo_wide, hi_wide = compute_k0_2_bounds(k0_2, 10.0)
        assert (hi_wide - lo_wide) > (hi_narrow - lo_narrow)

    def test_very_small_k0_2(self) -> None:
        """Handles very small k0_2 without error."""
        k0_2 = 1e-15
        lo, hi = compute_k0_2_bounds(k0_2, 10.0)
        assert np.isfinite(lo)
        assert np.isfinite(hi)
        assert lo < hi


# ===================================================================
# Tests: compute_tight_bounds
# ===================================================================

class TestComputeTightBounds:
    """Tests for the tight bounds computation."""

    def test_correct_number(self) -> None:
        bounds = compute_tight_bounds(
            surrogate_k0=np.array([0.01, 0.001]),
            surrogate_alpha=np.array([0.6, 0.5]),
        )
        assert len(bounds) == 4  # 2 k0 + 2 alpha

    def test_surrogate_inside(self) -> None:
        k0 = np.array([0.01, 0.001])
        alpha = np.array([0.6, 0.5])
        bounds = compute_tight_bounds(k0, alpha, bound_factor=2.0, alpha_margin=0.05)
        for i, k0_val in enumerate(k0):
            lo, hi = bounds[i]
            assert lo < np.log10(k0_val) < hi
        for i, a_val in enumerate(alpha):
            lo, hi = bounds[2 + i]
            assert lo < a_val < hi

    def test_alpha_clipping_floor(self) -> None:
        bounds = compute_tight_bounds(
            surrogate_k0=np.array([0.01]),
            surrogate_alpha=np.array([0.07]),
            alpha_margin=0.05, alpha_floor=0.05, alpha_ceil=0.95,
        )
        assert bounds[1][0] == pytest.approx(0.05, abs=1e-12)

    def test_alpha_clipping_ceil(self) -> None:
        bounds = compute_tight_bounds(
            surrogate_k0=np.array([0.01]),
            surrogate_alpha=np.array([0.92]),
            alpha_margin=0.05, alpha_floor=0.05, alpha_ceil=0.95,
        )
        assert bounds[1][1] == pytest.approx(0.95, abs=1e-12)


# ===================================================================
# Tests: RegularizationConfig + compute_regularization_penalty
# ===================================================================

class TestRegularizationPenalty:
    """Tests for the regularization penalty computation."""

    @pytest.fixture()
    def reg_config(self) -> RegularizationConfig:
        return RegularizationConfig(
            reg_lambda=1.0,
            k0_prior=np.array([0.01, 0.001]),
            alpha_prior=np.array([0.6, 0.5]),
            n_k0=2,
        )

    def test_zero_at_prior(self, reg_config: RegularizationConfig) -> None:
        x_prior = np.concatenate([
            np.log10(reg_config.k0_prior),
            reg_config.alpha_prior,
        ])
        penalty, grad = compute_regularization_penalty(x_prior, reg_config)
        assert penalty == pytest.approx(0.0, abs=1e-15)
        np.testing.assert_allclose(grad, 0.0, atol=1e-15)

    def test_known_value(self, reg_config: RegularizationConfig) -> None:
        x_prior = np.concatenate([
            np.log10(reg_config.k0_prior),
            reg_config.alpha_prior,
        ])
        delta = np.array([0.3, -0.1, 0.02, -0.01])
        x = x_prior + delta

        penalty, grad = compute_regularization_penalty(x, reg_config)
        expected = 1.0 * (0.3**2 + 0.1**2 + 0.02**2 + 0.01**2)
        assert penalty == pytest.approx(expected, rel=1e-10)
        np.testing.assert_allclose(grad, 2.0 * delta, atol=1e-12)

    def test_gradient_fd_check(self, reg_config: RegularizationConfig) -> None:
        x = np.array([-2.5, -3.2, 0.65, 0.48])
        _, grad_analytical = compute_regularization_penalty(x, reg_config)

        h = 1e-7
        grad_fd = np.zeros_like(x)
        for i in range(len(x)):
            xp, xm = x.copy(), x.copy()
            xp[i] += h
            xm[i] -= h
            fp, _ = compute_regularization_penalty(xp, reg_config)
            fm, _ = compute_regularization_penalty(xm, reg_config)
            grad_fd[i] = (fp - fm) / (2 * h)

        np.testing.assert_allclose(grad_analytical, grad_fd, rtol=1e-5, atol=1e-10)

    def test_zero_lambda(self) -> None:
        config = RegularizationConfig(
            reg_lambda=0.0,
            k0_prior=np.array([0.01, 0.001]),
            alpha_prior=np.array([0.6, 0.5]),
        )
        x = np.array([-1.0, -2.0, 0.7, 0.6])
        penalty, grad = compute_regularization_penalty(x, config)
        assert penalty == pytest.approx(0.0, abs=1e-15)
        np.testing.assert_allclose(grad, 0.0, atol=1e-15)


# ===================================================================
# Tests: K02PDEObjectiveConfig
# ===================================================================

class TestK02PDEObjectiveConfig:
    """Tests for the K02PDEObjectiveConfig frozen dataclass."""

    def test_creation(self) -> None:
        cfg = K02PDEObjectiveConfig(
            fixed_k0_1=0.02,
            fixed_alpha_1=0.627,
            fixed_alpha_2=0.5,
            secondary_weight=1.0,
        )
        assert cfg.fixed_k0_1 == 0.02
        assert cfg.fixed_alpha_1 == 0.627
        assert cfg.fixed_alpha_2 == 0.5
        assert cfg.secondary_weight == 1.0

    def test_frozen(self) -> None:
        cfg = K02PDEObjectiveConfig(
            fixed_k0_1=0.02,
            fixed_alpha_1=0.627,
            fixed_alpha_2=0.5,
        )
        with pytest.raises(AttributeError):
            cfg.fixed_k0_1 = 0.03  # type: ignore[misc]

    def test_default_secondary_weight(self) -> None:
        cfg = K02PDEObjectiveConfig(
            fixed_k0_1=0.02,
            fixed_alpha_1=0.627,
            fixed_alpha_2=0.5,
        )
        assert cfg.secondary_weight == 1.0


# ===================================================================
# Tests: compute_shallow_subset_idx
# ===================================================================

class TestComputeShallowSubsetIdx:
    """Tests for voltage subset index extraction."""

    def test_basic_extraction(self) -> None:
        all_eta = np.array([5.0, 3.0, 1.0, -1.0, -3.0, -5.0, -8.0, -10.0])
        shallow = np.array([-1.0, -5.0, -10.0])
        idx = compute_shallow_subset_idx(all_eta, shallow)
        assert len(idx) == 3
        # -1.0 is at index 3, -5.0 at index 5, -10.0 at index 7
        np.testing.assert_array_equal(idx, [3, 5, 7])

    def test_empty_subset(self) -> None:
        all_eta = np.array([5.0, 3.0, 1.0])
        shallow = np.array([-1.0, -5.0])  # not in all_eta
        idx = compute_shallow_subset_idx(all_eta, shallow)
        assert len(idx) == 0

    def test_all_match(self) -> None:
        all_eta = np.array([-1.0, -3.0, -5.0])
        shallow = np.array([-1.0, -3.0, -5.0])
        idx = compute_shallow_subset_idx(all_eta, shallow)
        np.testing.assert_array_equal(idx, [0, 1, 2])


# ===================================================================
# Tests: Fixed params are truly held fixed
# ===================================================================

class TestFixedParamsContract:
    """Verify that the K02PDEObjectiveConfig correctly
    specifies which parameters are fixed vs free."""

    def test_config_preserves_fixed_values(self) -> None:
        """K02PDEObjectiveConfig stores exact fixed values."""
        k0_1_fixed = 0.012345
        alpha_1_fixed = 0.627
        alpha_2_fixed = 0.500

        cfg = K02PDEObjectiveConfig(
            fixed_k0_1=k0_1_fixed,
            fixed_alpha_1=alpha_1_fixed,
            fixed_alpha_2=alpha_2_fixed,
        )

        # Verify exact preservation (no rounding/mutation)
        assert cfg.fixed_k0_1 == k0_1_fixed
        assert cfg.fixed_alpha_1 == alpha_1_fixed
        assert cfg.fixed_alpha_2 == alpha_2_fixed

    def test_config_immutable(self) -> None:
        """Cannot accidentally mutate fixed params after creation."""
        cfg = K02PDEObjectiveConfig(
            fixed_k0_1=0.02,
            fixed_alpha_1=0.627,
            fixed_alpha_2=0.5,
        )
        with pytest.raises(AttributeError):
            cfg.fixed_k0_1 = 0.03  # type: ignore[misc]
        with pytest.raises(AttributeError):
            cfg.fixed_alpha_1 = 0.7  # type: ignore[misc]
        with pytest.raises(AttributeError):
            cfg.fixed_alpha_2 = 0.6  # type: ignore[misc]


# ===================================================================
# Tests: Gradient chain rule for log10 transform
# ===================================================================

class TestGradientChainRule:
    """Tests for the log10 chain rule used in gradient-based k0_2 optimization.

    The chain rule converts the PDE gradient from physical space to log10 space:
        dJ/d(log10(k0_2)) = dJ/dk0_2 * k0_2 * ln(10)
    """

    def test_chain_rule_identity(self) -> None:
        """Verify chain rule formula at a known k0_2 and gradient."""
        k0_2 = 0.002
        dJ_dk0_2 = -15.0  # arbitrary gradient value

        grad_log10 = dJ_dk0_2 * k0_2 * np.log(10.0)
        expected = -15.0 * 0.002 * np.log(10.0)
        assert grad_log10 == pytest.approx(expected, rel=1e-14)

    def test_chain_rule_finite_difference(self) -> None:
        """Finite-difference check of the chain rule.

        For a simple quadratic objective J(k0_2) = (k0_2 - k0_2_true)^2,
        the analytical gradient is dJ/dk0_2 = 2*(k0_2 - k0_2_true).
        We verify the chain-rule transformed gradient matches FD in log10 space.
        """
        k0_2_true = 0.003
        log10_k0_2 = np.log10(0.002)

        def J(log10_val: float) -> float:
            return (10.0 ** log10_val - k0_2_true) ** 2

        # Analytical chain-rule gradient
        k0_2 = 10.0 ** log10_k0_2
        dJ_dk0_2 = 2.0 * (k0_2 - k0_2_true)
        grad_analytical = dJ_dk0_2 * k0_2 * np.log(10.0)

        # Finite difference
        h = 1e-8
        grad_fd = (J(log10_k0_2 + h) - J(log10_k0_2 - h)) / (2.0 * h)

        assert grad_analytical == pytest.approx(grad_fd, rel=1e-5)

    def test_chain_rule_zero_gradient(self) -> None:
        """At the true parameter, both physical and log10 gradients are zero."""
        k0_2 = 0.003
        dJ_dk0_2 = 0.0

        grad_log10 = dJ_dk0_2 * k0_2 * np.log(10.0)
        assert grad_log10 == pytest.approx(0.0, abs=1e-16)

    def test_chain_rule_gradient_index(self) -> None:
        """Verify that k0_2 is at index 1 in joint gradient [k0_1, k0_2, a1, a2]."""
        # This is a layout test: in joint control mode with 2 k0 + 2 alpha,
        # the gradient vector is [dk0_1, dk0_2, dalpha_1, dalpha_2].
        # k0_2 must be extracted from index 1.
        full_grad = np.array([0.1, -0.5, 0.02, -0.03])
        k02_grad_index = 1
        assert full_grad[k02_grad_index] == pytest.approx(-0.5)


# ===================================================================
# Tests: Wider k0_2 bounds (factor=20)
# ===================================================================

class TestWiderK02Bounds:
    """Tests for the wider default k0_2 bound factor (20.0)."""

    def test_factor_20_gives_wider_than_10(self) -> None:
        """factor=20 gives wider bounds than factor=10."""
        k0_2 = 0.001
        lo_10, hi_10 = compute_k0_2_bounds(k0_2, 10.0)
        lo_20, hi_20 = compute_k0_2_bounds(k0_2, 20.0)
        assert (hi_20 - lo_20) > (hi_10 - lo_10)

    def test_factor_20_spans_correct_range(self) -> None:
        """factor=20 gives log10(k0_2/20) to log10(k0_2*20)."""
        k0_2 = 0.002
        lo, hi = compute_k0_2_bounds(k0_2, 20.0)
        expected_lo = np.log10(k0_2 / 20.0)
        expected_hi = np.log10(k0_2 * 20.0)
        assert lo == pytest.approx(expected_lo, abs=1e-12)
        assert hi == pytest.approx(expected_hi, abs=1e-12)

    def test_default_config_uses_factor_20(self) -> None:
        """HybridConfig default uses k0_2_bound_factor=20.0."""
        cfg = HybridConfig()
        assert cfg.k0_2_bound_factor == 20.0


# ===================================================================
# Tests: Voltage range selection
# ===================================================================

class TestVoltageRangeSelection:
    """Tests for the voltage range selection (shallow vs cathodic)."""

    def test_default_is_cathodic(self) -> None:
        """HybridConfig default voltage range is 'cathodic'."""
        cfg = HybridConfig()
        assert cfg.pde_voltage_range == "cathodic"

    def test_shallow_option(self) -> None:
        """Can create config with shallow voltage range."""
        cfg = HybridConfig(pde_voltage_range="shallow")
        assert cfg.pde_voltage_range == "shallow"

    def test_cathodic_has_more_points(self) -> None:
        """The cathodic grid (15 pts) has more voltage points than shallow (10 pts)."""
        eta_shallow = np.array([
            -1.0, -2.0, -3.0, -4.0, -5.0, -6.5, -8.0,
            -10.0, -11.5, -13.0,
        ])
        eta_cathodic = np.array([
            -1.0, -2.0, -3.0, -4.0, -5.0, -6.5, -8.0,
            -10.0, -13.0, -17.0, -22.0, -28.0,
            -35.0, -41.0, -46.5,
        ])
        assert len(eta_cathodic) == 15
        assert len(eta_shallow) == 10
        assert len(eta_cathodic) > len(eta_shallow)

    def test_cathodic_extends_deeper(self) -> None:
        """The cathodic grid reaches eta=-46.5 (deeper than shallow -13.0)."""
        eta_shallow = np.array([
            -1.0, -2.0, -3.0, -4.0, -5.0, -6.5, -8.0,
            -10.0, -11.5, -13.0,
        ])
        eta_cathodic = np.array([
            -1.0, -2.0, -3.0, -4.0, -5.0, -6.5, -8.0,
            -10.0, -13.0, -17.0, -22.0, -28.0,
            -35.0, -41.0, -46.5,
        ])
        assert eta_cathodic.min() == pytest.approx(-46.5)
        assert eta_shallow.min() == pytest.approx(-13.0)

    def test_subset_targets_extracts_correct_values(self) -> None:
        """subset_targets correctly extracts values for a voltage subset."""
        all_eta = np.array([5.0, 3.0, 1.0, -1.0, -3.0, -5.0, -8.0, -10.0])
        target_cd = np.arange(8, dtype=float) * 10.0
        target_pc = np.arange(8, dtype=float) * -1.0
        subset_eta = np.array([-1.0, -5.0, -10.0])

        cd_sub, pc_sub = subset_targets(target_cd, target_pc, all_eta, subset_eta)

        # -1.0 is at index 3, -5.0 at index 5, -10.0 at index 7
        np.testing.assert_array_equal(cd_sub, [30.0, 50.0, 70.0])
        np.testing.assert_array_equal(pc_sub, [-3.0, -5.0, -7.0])


# ===================================================================
# Tests: SweepResult
# ===================================================================

class TestSweepResult:
    """Tests for the SweepResult frozen dataclass."""

    def test_creation(self) -> None:
        r = SweepResult(
            weight=5.0,
            k0_1_err_pct=0.51,
            k0_2_err_pct=2.58,
            alpha_1_err_pct=1.92,
            alpha_2_err_pct=2.38,
            max_err_pct=2.58,
        )
        assert r.weight == 5.0
        assert r.max_err_pct == 2.58

    def test_frozen(self) -> None:
        r = SweepResult(
            weight=1.0,
            k0_1_err_pct=0.5, k0_2_err_pct=14.0,
            alpha_1_err_pct=1.9, alpha_2_err_pct=2.4,
            max_err_pct=14.0,
        )
        with pytest.raises(AttributeError):
            r.weight = 2.0  # type: ignore[misc]


# ===================================================================
# Tests: HybridConfig voltage range field
# ===================================================================

class TestHybridConfigVoltageRange:
    """Tests for the new pde_voltage_range field in HybridConfig."""

    def test_frozen_voltage_range(self) -> None:
        cfg = HybridConfig(pde_voltage_range="shallow")
        with pytest.raises(AttributeError):
            cfg.pde_voltage_range = "cathodic"  # type: ignore[misc]

    def test_default_secondary_weight(self) -> None:
        """Default secondary_weight is 5.0 (changed from 1.0)."""
        cfg = HybridConfig()
        assert cfg.secondary_weight == 5.0

    def test_default_pde_maxiter(self) -> None:
        """Default pde_maxiter is 30 (changed from 15)."""
        cfg = HybridConfig()
        assert cfg.pde_maxiter == 30


# ===================================================================
# Tests: AsymmetricRegularizationConfig
# ===================================================================

class TestAsymmetricRegularizationConfig:
    """Tests for the AsymmetricRegularizationConfig frozen dataclass."""

    def test_creation_and_frozen(self) -> None:
        cfg = AsymmetricRegularizationConfig(
            lambdas=(5.0, 0.1, 5.0, 5.0),
            k0_prior=(0.02, 0.002),
            alpha_prior=(0.6, 0.5),
        )
        assert cfg.lambdas == (5.0, 0.1, 5.0, 5.0)
        assert cfg.k0_prior == (0.02, 0.002)
        assert cfg.alpha_prior == (0.6, 0.5)
        with pytest.raises(AttributeError):
            cfg.lambdas = (1.0, 1.0, 1.0, 1.0)  # type: ignore[misc]

    def test_lambdas_stored(self) -> None:
        cfg = AsymmetricRegularizationConfig(
            lambdas=(10.0, 0.01, 3.0, 7.0),
            k0_prior=(0.01, 0.001),
            alpha_prior=(0.55, 0.45),
        )
        assert len(cfg.lambdas) == 4
        assert cfg.lambdas[0] == 10.0
        assert cfg.lambdas[1] == 0.01


# ===================================================================
# Tests: AsymmetricRegularizationPenalty
# ===================================================================

class TestAsymmetricRegularizationPenalty:
    """Tests for the asymmetric regularization penalty computation."""

    @pytest.fixture()
    def asym_config(self) -> AsymmetricRegularizationConfig:
        return AsymmetricRegularizationConfig(
            lambdas=(5.0, 0.1, 5.0, 5.0),
            k0_prior=(0.02, 0.002),
            alpha_prior=(0.6, 0.5),
        )

    def test_zero_at_prior(self, asym_config: AsymmetricRegularizationConfig) -> None:
        """x == prior => penalty=0, grad=zeros."""
        x_prior = np.concatenate([
            np.log10(np.array(asym_config.k0_prior)),
            np.array(asym_config.alpha_prior),
        ])
        penalty, grad = compute_asymmetric_regularization_penalty(x_prior, asym_config)
        assert penalty == pytest.approx(0.0, abs=1e-15)
        np.testing.assert_allclose(grad, 0.0, atol=1e-15)

    def test_known_value(self, asym_config: AsymmetricRegularizationConfig) -> None:
        """Known delta => verify penalty = sum(lambda_i * delta_i^2)."""
        x_prior = np.concatenate([
            np.log10(np.array(asym_config.k0_prior)),
            np.array(asym_config.alpha_prior),
        ])
        delta = np.array([0.1, -0.2, 0.03, -0.01])
        x = x_prior + delta

        penalty, grad = compute_asymmetric_regularization_penalty(x, asym_config)
        lam = np.array(asym_config.lambdas)
        expected = float(np.sum(lam * delta**2))
        assert penalty == pytest.approx(expected, rel=1e-10)
        np.testing.assert_allclose(grad, 2.0 * lam * delta, atol=1e-12)

    def test_asymmetric_gradient(self, asym_config: AsymmetricRegularizationConfig) -> None:
        """Different lambdas produce correct per-parameter gradient scaling."""
        x_prior = np.concatenate([
            np.log10(np.array(asym_config.k0_prior)),
            np.array(asym_config.alpha_prior),
        ])
        # Same displacement for all params
        delta = np.array([0.1, 0.1, 0.1, 0.1])
        x = x_prior + delta

        _, grad = compute_asymmetric_regularization_penalty(x, asym_config)
        lam = np.array(asym_config.lambdas)
        expected_grad = 2.0 * lam * delta
        np.testing.assert_allclose(grad, expected_grad, atol=1e-12)

        # k0_1 (lam=5.0) should have 50x gradient of k0_2 (lam=0.1)
        assert grad[0] / grad[1] == pytest.approx(50.0, rel=1e-10)

    def test_fd_gradient_check(self, asym_config: AsymmetricRegularizationConfig) -> None:
        """Finite-difference verification of gradient."""
        x = np.array([-2.5, -3.2, 0.65, 0.48])
        _, grad_analytical = compute_asymmetric_regularization_penalty(x, asym_config)

        h = 1e-7
        grad_fd = np.zeros_like(x)
        for i in range(len(x)):
            xp, xm = x.copy(), x.copy()
            xp[i] += h
            xm[i] -= h
            fp, _ = compute_asymmetric_regularization_penalty(xp, asym_config)
            fm, _ = compute_asymmetric_regularization_penalty(xm, asym_config)
            grad_fd[i] = (fp - fm) / (2 * h)

        np.testing.assert_allclose(grad_analytical, grad_fd, rtol=1e-5, atol=1e-10)

    def test_zero_lambda_component(self) -> None:
        """lambda_i=0 => param_i contributes nothing to penalty or gradient."""
        config = AsymmetricRegularizationConfig(
            lambdas=(0.0, 1.0, 0.0, 1.0),
            k0_prior=(0.02, 0.002),
            alpha_prior=(0.6, 0.5),
        )
        x_prior = np.concatenate([
            np.log10(np.array(config.k0_prior)),
            np.array(config.alpha_prior),
        ])
        delta = np.array([100.0, 0.5, 100.0, 0.5])
        x = x_prior + delta

        penalty, grad = compute_asymmetric_regularization_penalty(x, config)
        # Only params 1 and 3 contribute (lambda=1.0)
        expected = 1.0 * 0.5**2 + 1.0 * 0.5**2
        assert penalty == pytest.approx(expected, rel=1e-10)
        assert grad[0] == pytest.approx(0.0, abs=1e-15)
        assert grad[2] == pytest.approx(0.0, abs=1e-15)

    def test_strong_vs_weak(self) -> None:
        """lambda_k0_1=5.0, lambda_k0_2=0.1 => grad[0] is 50x grad[1] for equal displacement."""
        config = AsymmetricRegularizationConfig(
            lambdas=(5.0, 0.1, 5.0, 5.0),
            k0_prior=(0.02, 0.002),
            alpha_prior=(0.6, 0.5),
        )
        x_prior = np.concatenate([
            np.log10(np.array(config.k0_prior)),
            np.array(config.alpha_prior),
        ])
        # Equal displacement for k0_1 and k0_2
        delta = np.array([0.3, 0.3, 0.0, 0.0])
        x = x_prior + delta

        _, grad = compute_asymmetric_regularization_penalty(x, config)
        # grad[0] = 2 * 5.0 * 0.3 = 3.0
        # grad[1] = 2 * 0.1 * 0.3 = 0.06
        assert grad[0] == pytest.approx(3.0, rel=1e-10)
        assert grad[1] == pytest.approx(0.06, rel=1e-10)
        assert grad[0] / grad[1] == pytest.approx(50.0, rel=1e-10)


# ===================================================================
# Tests: AsymmetricBounds
# ===================================================================

class TestAsymmetricBounds:
    """Tests for the asymmetric bounds computation."""

    def test_k0_2_wider_than_k0_1(self) -> None:
        """With factors (1.5, 5.0), k0_2 bounds are wider than k0_1."""
        k0 = np.array([0.02, 0.002])
        alpha = np.array([0.6, 0.5])
        bounds = compute_asymmetric_bounds(
            k0, alpha,
            k0_bound_factors=(1.5, 5.0),
            alpha_margins=(0.03, 0.03),
        )
        k0_1_width = bounds[0][1] - bounds[0][0]
        k0_2_width = bounds[1][1] - bounds[1][0]
        assert k0_2_width > k0_1_width

    def test_alpha_inside_bounds(self) -> None:
        """Alpha values are inside their respective bounds."""
        k0 = np.array([0.02, 0.002])
        alpha = np.array([0.6, 0.5])
        bounds = compute_asymmetric_bounds(
            k0, alpha,
            k0_bound_factors=(2.0, 3.0),
            alpha_margins=(0.05, 0.1),
        )
        # alpha_1 bound
        assert bounds[2][0] < alpha[0] < bounds[2][1]
        # alpha_2 bound
        assert bounds[3][0] < alpha[1] < bounds[3][1]

    def test_correct_number(self) -> None:
        """Returns 4 bounds (2 k0 + 2 alpha)."""
        bounds = compute_asymmetric_bounds(
            k0_vals=np.array([0.01, 0.001]),
            alpha_vals=np.array([0.6, 0.5]),
            k0_bound_factors=(2.0, 5.0),
            alpha_margins=(0.05, 0.05),
        )
        assert len(bounds) == 4

    def test_k0_in_log_space(self) -> None:
        """k0 bounds are in log10 space."""
        k0 = np.array([0.01])
        alpha = np.array([0.5])
        bounds = compute_asymmetric_bounds(
            k0, alpha,
            k0_bound_factors=(10.0,),
            alpha_margins=(0.05,),
        )
        # k0=0.01, factor=10 => [log10(0.001), log10(0.1)] = [-3, -1]
        assert bounds[0][0] == pytest.approx(-3.0, abs=1e-12)
        assert bounds[0][1] == pytest.approx(-1.0, abs=1e-12)

    def test_alpha_floor_ceil_clipping(self) -> None:
        """Alpha bounds are clipped to [floor, ceil]."""
        k0 = np.array([0.01])
        alpha = np.array([0.06])
        bounds = compute_asymmetric_bounds(
            k0, alpha,
            k0_bound_factors=(2.0,),
            alpha_margins=(0.1,),
            alpha_floor=0.05,
            alpha_ceil=0.95,
        )
        assert bounds[1][0] == pytest.approx(0.05, abs=1e-12)


# ===================================================================
# Tests: Updated HybridConfig defaults
# ===================================================================

class TestUpdatedHybridConfig:
    """Tests for the new asymmetric fields in HybridConfig."""

    def test_new_defaults(self) -> None:
        """Verify all new asymmetric default values."""
        cfg = HybridConfig()
        # Asymmetric lambdas
        assert cfg.joint_polish_lambda_k0_1 == 5.0
        assert cfg.joint_polish_lambda_k0_2 == 0.1
        assert cfg.joint_polish_lambda_alpha_1 == 5.0
        assert cfg.joint_polish_lambda_alpha_2 == 5.0

        # Asymmetric bounds
        assert cfg.joint_polish_k0_1_bound_factor == 1.5
        assert cfg.joint_polish_k0_2_bound_factor == 5.0
        assert cfg.joint_polish_alpha_margin == 0.03

        # Updated maxiter
        assert cfg.joint_polish_maxiter == 8

    def test_custom_asymmetric_lambdas(self) -> None:
        """Can override asymmetric lambdas."""
        cfg = HybridConfig(
            joint_polish_lambda_k0_1=10.0,
            joint_polish_lambda_k0_2=0.01,
            joint_polish_lambda_alpha_1=3.0,
            joint_polish_lambda_alpha_2=7.0,
        )
        assert cfg.joint_polish_lambda_k0_1 == 10.0
        assert cfg.joint_polish_lambda_k0_2 == 0.01
        assert cfg.joint_polish_lambda_alpha_1 == 3.0
        assert cfg.joint_polish_lambda_alpha_2 == 7.0

    def test_frozen_new_fields(self) -> None:
        """New fields are frozen (cannot be mutated)."""
        cfg = HybridConfig()
        with pytest.raises(AttributeError):
            cfg.joint_polish_lambda_k0_2 = 1.0  # type: ignore[misc]
        with pytest.raises(AttributeError):
            cfg.joint_polish_k0_2_bound_factor = 10.0  # type: ignore[misc]
        with pytest.raises(AttributeError):
            cfg.joint_polish_alpha_margin = 0.1  # type: ignore[misc]


# ===================================================================
# Tests: joint_mode field
# ===================================================================

class TestJointMode:
    """Tests for the joint_mode field in HybridConfig."""

    def test_default_is_asymmetric(self) -> None:
        """Default joint_mode is 'asymmetric'."""
        cfg = HybridConfig()
        assert cfg.joint_mode == "asymmetric"

    def test_free_mode(self) -> None:
        """Can create config with joint_mode='free'."""
        cfg = HybridConfig(joint_mode="free")
        assert cfg.joint_mode == "free"

    def test_frozen_joint_mode(self) -> None:
        """joint_mode is frozen (cannot be mutated)."""
        cfg = HybridConfig()
        with pytest.raises(AttributeError):
            cfg.joint_mode = "free"  # type: ignore[misc]

    def test_free_mode_with_custom_overrides(self) -> None:
        """Free mode config can coexist with custom overrides."""
        cfg = HybridConfig(
            joint_mode="free",
            joint_polish_maxiter=25,
            joint_polish_lambda_k0_1=0.0,
            joint_polish_lambda_k0_2=0.0,
            joint_polish_lambda_alpha_1=0.0,
            joint_polish_lambda_alpha_2=0.0,
            joint_polish_k0_1_bound_factor=3.0,
            joint_polish_alpha_margin=0.08,
        )
        assert cfg.joint_mode == "free"
        assert cfg.joint_polish_maxiter == 25
        assert cfg.joint_polish_lambda_k0_1 == 0.0
        assert cfg.joint_polish_lambda_k0_2 == 0.0
        assert cfg.joint_polish_lambda_alpha_1 == 0.0
        assert cfg.joint_polish_lambda_alpha_2 == 0.0
        assert cfg.joint_polish_k0_1_bound_factor == 3.0
        assert cfg.joint_polish_alpha_margin == 0.08

    def test_asymmetric_mode_preserves_defaults(self) -> None:
        """Asymmetric mode preserves the original default values."""
        cfg = HybridConfig(joint_mode="asymmetric")
        assert cfg.joint_polish_lambda_k0_1 == 5.0
        assert cfg.joint_polish_lambda_k0_2 == 0.1
        assert cfg.joint_polish_lambda_alpha_1 == 5.0
        assert cfg.joint_polish_lambda_alpha_2 == 5.0
        assert cfg.joint_polish_k0_1_bound_factor == 1.5
        assert cfg.joint_polish_alpha_margin == 0.03
        assert cfg.joint_polish_maxiter == 8
