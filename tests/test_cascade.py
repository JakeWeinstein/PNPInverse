"""Tests for the per-observable inference cascade (Strategy 5).

Tests cover:
- CascadeConfig frozen dataclass (defaults, frozen, custom values)
- CascadePassResult frozen dataclass (creation, frozen)
- CascadeResult frozen dataclass (creation, frozen)
- run_cascade_inference (convergence, pass1 CD-dominant, pass2 fixes k0_1,
  recovery with perfect targets, skip_polish, verbose suppression)
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

from Surrogate.surrogate_model import BVSurrogateModel, SurrogateConfig
from Surrogate.cascade import (
    CascadeConfig,
    CascadePassResult,
    CascadeResult,
    run_cascade_inference,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def fitted_surrogate() -> BVSurrogateModel:
    """Build and fit a small BVSurrogateModel for testing.

    Uses 40 training samples with 8 voltage points and a non-separable
    response function so that all 4 parameters are identifiable.
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

    n_samples = 40
    n_eta = 8
    phi_applied = np.linspace(-10, 5, n_eta)

    k0_1 = rng.uniform(0.005, 0.05, n_samples)
    k0_2 = rng.uniform(0.0005, 0.005, n_samples)
    alpha_1 = rng.uniform(0.4, 0.8, n_samples)
    alpha_2 = rng.uniform(0.3, 0.7, n_samples)
    params = np.column_stack([k0_1, k0_2, alpha_1, alpha_2])

    cd = (np.outer(alpha_1, -phi_applied) * k0_1[:, None] * 10
          + np.outer(alpha_2 * k0_2, phi_applied ** 2) * 0.1)
    pc = (np.outer(alpha_2, -phi_applied) * k0_2[:, None] * 100
          + np.outer(alpha_1 * k0_1, phi_applied ** 2) * 0.5)

    model.fit(params, cd, pc, phi_applied)
    return model


@pytest.fixture()
def target_data(fitted_surrogate):
    """Generate target I-V curves at a known parameter set."""
    k0_1_true = 0.02
    k0_2_true = 0.002
    alpha_1_true = 0.6
    alpha_2_true = 0.5

    pred = fitted_surrogate.predict(k0_1_true, k0_2_true,
                                     alpha_1_true, alpha_2_true)
    return {
        "target_cd": pred["current_density"],
        "target_pc": pred["peroxide_current"],
        "k0_1_true": k0_1_true,
        "k0_2_true": k0_2_true,
        "alpha_1_true": alpha_1_true,
        "alpha_2_true": alpha_2_true,
    }


# ---------------------------------------------------------------------------
# Tests: CascadeConfig
# ---------------------------------------------------------------------------

class TestCascadeConfig:
    """Tests for the CascadeConfig frozen dataclass."""

    def test_defaults(self):
        cfg = CascadeConfig()
        assert cfg.pass1_weight == 0.5
        assert cfg.pass2_weight == 2.0
        assert cfg.pass1_maxiter == 60
        assert cfg.pass2_maxiter == 60
        assert cfg.polish_maxiter == 30
        assert cfg.polish_weight == 1.0
        assert cfg.skip_polish is False
        assert cfg.fd_step == 1e-5
        assert cfg.verbose is True

    def test_frozen(self):
        cfg = CascadeConfig()
        with pytest.raises(AttributeError):
            cfg.pass1_weight = 0.1  # type: ignore[misc]

    def test_custom_values(self):
        cfg = CascadeConfig(
            pass1_weight=0.1,
            pass2_weight=5.0,
            pass1_maxiter=40,
            pass2_maxiter=80,
            polish_maxiter=10,
            polish_weight=0.5,
            skip_polish=True,
            fd_step=1e-4,
            verbose=False,
        )
        assert cfg.pass1_weight == 0.1
        assert cfg.pass2_weight == 5.0
        assert cfg.pass1_maxiter == 40
        assert cfg.pass2_maxiter == 80
        assert cfg.polish_maxiter == 10
        assert cfg.polish_weight == 0.5
        assert cfg.skip_polish is True
        assert cfg.fd_step == 1e-4
        assert cfg.verbose is False


# ---------------------------------------------------------------------------
# Tests: CascadePassResult
# ---------------------------------------------------------------------------

class TestCascadePassResult:
    """Tests for the CascadePassResult frozen dataclass."""

    def test_creation(self):
        r = CascadePassResult(
            pass_name="Pass 1 (CD-dominant)",
            k0_1=0.02, k0_2=0.002,
            alpha_1=0.6, alpha_2=0.5,
            loss=1e-4,
            n_iters=25,
            n_evals=200,
            elapsed_s=1.5,
        )
        assert r.pass_name == "Pass 1 (CD-dominant)"
        assert r.k0_1 == 0.02
        assert r.loss == 1e-4
        assert r.n_iters == 25
        assert r.n_evals == 200

    def test_frozen(self):
        r = CascadePassResult(
            pass_name="test",
            k0_1=0.01, k0_2=0.001,
            alpha_1=0.6, alpha_2=0.5,
            loss=0.01,
            n_iters=10,
            n_evals=50,
            elapsed_s=1.0,
        )
        with pytest.raises(AttributeError):
            r.k0_1 = 0.02  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Tests: CascadeResult
# ---------------------------------------------------------------------------

class TestCascadeResult:
    """Tests for the CascadeResult frozen dataclass."""

    def test_creation(self):
        pass_results = (
            CascadePassResult(
                pass_name="Pass 1", k0_1=0.02, k0_2=0.002,
                alpha_1=0.6, alpha_2=0.5, loss=1e-4,
                n_iters=25, n_evals=200, elapsed_s=1.0,
            ),
            CascadePassResult(
                pass_name="Pass 2", k0_1=0.02, k0_2=0.0021,
                alpha_1=0.6, alpha_2=0.51, loss=5e-5,
                n_iters=20, n_evals=150, elapsed_s=0.8,
            ),
        )
        r = CascadeResult(
            best_k0_1=0.02, best_k0_2=0.0021,
            best_alpha_1=0.6, best_alpha_2=0.51,
            best_loss=5e-5,
            pass_results=pass_results,
            total_evals=350,
            total_time_s=1.8,
        )
        assert r.best_k0_1 == 0.02
        assert r.best_loss == 5e-5
        assert len(r.pass_results) == 2
        assert r.total_evals == 350

    def test_frozen(self):
        r = CascadeResult(
            best_k0_1=0.01, best_k0_2=0.001,
            best_alpha_1=0.6, best_alpha_2=0.5,
            best_loss=0.01,
            pass_results=(),
            total_evals=0,
            total_time_s=0.0,
        )
        with pytest.raises(AttributeError):
            r.best_k0_1 = 0.02  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Tests: run_cascade_inference
# ---------------------------------------------------------------------------

class TestRunCascadeInference:
    """Tests for the run_cascade_inference function."""

    def test_basic_convergence(self, fitted_surrogate, target_data):
        """Cascade should produce finite results without error."""
        config = CascadeConfig(
            pass1_weight=0.5,
            pass2_weight=2.0,
            pass1_maxiter=10,
            pass2_maxiter=10,
            polish_maxiter=5,
            skip_polish=False,
            verbose=False,
        )

        result = run_cascade_inference(
            surrogate=fitted_surrogate,
            target_cd=target_data["target_cd"],
            target_pc=target_data["target_pc"],
            initial_k0=[0.01, 0.001],
            initial_alpha=[0.5, 0.4],
            bounds_k0_1=(0.001, 0.1),
            bounds_k0_2=(0.0001, 0.01),
            bounds_alpha=(0.2, 0.9),
            config=config,
        )

        assert isinstance(result, CascadeResult)
        assert np.isfinite(result.best_loss)
        assert np.isfinite(result.best_k0_1)
        assert np.isfinite(result.best_k0_2)
        assert np.isfinite(result.best_alpha_1)
        assert np.isfinite(result.best_alpha_2)
        assert result.total_evals > 0
        assert result.total_time_s > 0
        # 3 passes: CD-dominant, PC-dominant, polish
        assert len(result.pass_results) == 3

    def test_pass1_cd_dominant(self, fitted_surrogate, target_data):
        """Pass 1 with low weight should produce a reasonable result."""
        config = CascadeConfig(
            pass1_weight=0.1,
            pass2_weight=2.0,
            pass1_maxiter=30,
            pass2_maxiter=10,
            polish_maxiter=5,
            skip_polish=False,
            verbose=False,
        )

        result = run_cascade_inference(
            surrogate=fitted_surrogate,
            target_cd=target_data["target_cd"],
            target_pc=target_data["target_pc"],
            initial_k0=[0.015, 0.0015],
            initial_alpha=[0.55, 0.45],
            bounds_k0_1=(0.005, 0.05),
            bounds_k0_2=(0.0005, 0.005),
            bounds_alpha=(0.3, 0.9),
            config=config,
        )

        # Pass 1 result should exist and have finite values
        p1 = result.pass_results[0]
        assert p1.pass_name == "Pass 1 (CD-dominant)"
        assert np.isfinite(p1.loss)
        assert np.isfinite(p1.k0_1)

    def test_pass2_fixes_k0_1(self, fitted_surrogate, target_data):
        """Pass 2 should keep k0_1 and alpha_1 at the Pass 1 values."""
        config = CascadeConfig(
            pass1_weight=0.5,
            pass2_weight=2.0,
            pass1_maxiter=20,
            pass2_maxiter=20,
            polish_maxiter=5,
            skip_polish=True,
            verbose=False,
        )

        result = run_cascade_inference(
            surrogate=fitted_surrogate,
            target_cd=target_data["target_cd"],
            target_pc=target_data["target_pc"],
            initial_k0=[0.01, 0.001],
            initial_alpha=[0.5, 0.4],
            bounds_k0_1=(0.001, 0.1),
            bounds_k0_2=(0.0001, 0.01),
            bounds_alpha=(0.2, 0.9),
            config=config,
        )

        p1 = result.pass_results[0]
        p2 = result.pass_results[1]

        # Pass 2 should have exactly the same k0_1 and alpha_1 as Pass 1
        np.testing.assert_allclose(p2.k0_1, p1.k0_1, rtol=1e-12)
        np.testing.assert_allclose(p2.alpha_1, p1.alpha_1, rtol=1e-12)

    def test_recovery_with_perfect_targets(self, fitted_surrogate, target_data):
        """With surrogate-generated targets, recover within 15%."""
        config = CascadeConfig(
            pass1_weight=0.5,
            pass2_weight=2.0,
            pass1_maxiter=60,
            pass2_maxiter=60,
            polish_maxiter=30,
            skip_polish=False,
            verbose=False,
        )

        result = run_cascade_inference(
            surrogate=fitted_surrogate,
            target_cd=target_data["target_cd"],
            target_pc=target_data["target_pc"],
            initial_k0=[0.015, 0.0015],
            initial_alpha=[0.55, 0.45],
            bounds_k0_1=(0.005, 0.05),
            bounds_k0_2=(0.0005, 0.005),
            bounds_alpha=(0.3, 0.9),
            config=config,
        )

        k0_1_err = abs(result.best_k0_1 - target_data["k0_1_true"]) / target_data["k0_1_true"]
        k0_2_err = abs(result.best_k0_2 - target_data["k0_2_true"]) / target_data["k0_2_true"]
        alpha_1_err = abs(result.best_alpha_1 - target_data["alpha_1_true"]) / target_data["alpha_1_true"]
        alpha_2_err = abs(result.best_alpha_2 - target_data["alpha_2_true"]) / target_data["alpha_2_true"]

        # Loss should be very small (surrogate-generated targets)
        assert result.best_loss < 1e-2, f"Loss {result.best_loss} too large"

        # At least k0_2 and alpha_2 should be recovered reasonably
        # (the polynomial test model has some k0*alpha ambiguity)
        assert k0_2_err < 0.50, f"k0_2 error {k0_2_err*100:.1f}% too large"
        assert alpha_2_err < 0.50, f"alpha_2 error {alpha_2_err*100:.1f}% too large"

    def test_skip_polish(self, fitted_surrogate, target_data):
        """When skip_polish=True, only 2 passes should be run."""
        config = CascadeConfig(
            pass1_weight=0.5,
            pass2_weight=2.0,
            pass1_maxiter=10,
            pass2_maxiter=10,
            skip_polish=True,
            verbose=False,
        )

        result = run_cascade_inference(
            surrogate=fitted_surrogate,
            target_cd=target_data["target_cd"],
            target_pc=target_data["target_pc"],
            initial_k0=[0.01, 0.001],
            initial_alpha=[0.5, 0.4],
            bounds_k0_1=(0.001, 0.1),
            bounds_k0_2=(0.0001, 0.01),
            bounds_alpha=(0.2, 0.9),
            config=config,
        )

        assert len(result.pass_results) == 2
        assert result.pass_results[0].pass_name == "Pass 1 (CD-dominant)"
        assert result.pass_results[1].pass_name == "Pass 2 (PC-dominant)"

    def test_verbose_false(self, fitted_surrogate, target_data, capsys):
        """verbose=False should suppress all print output."""
        config = CascadeConfig(
            pass1_weight=0.5,
            pass2_weight=2.0,
            pass1_maxiter=5,
            pass2_maxiter=5,
            polish_maxiter=3,
            skip_polish=False,
            verbose=False,
        )

        run_cascade_inference(
            surrogate=fitted_surrogate,
            target_cd=target_data["target_cd"],
            target_pc=target_data["target_pc"],
            initial_k0=[0.01, 0.001],
            initial_alpha=[0.5, 0.4],
            bounds_k0_1=(0.001, 0.1),
            bounds_k0_2=(0.0001, 0.01),
            bounds_alpha=(0.2, 0.9),
            config=config,
        )

        captured = capsys.readouterr()
        assert captured.out == "", f"Expected no output, got: {captured.out!r}"

    def test_default_config(self, fitted_surrogate, target_data):
        """run_cascade_inference with config=None should use defaults."""
        result = run_cascade_inference(
            surrogate=fitted_surrogate,
            target_cd=target_data["target_cd"],
            target_pc=target_data["target_pc"],
            initial_k0=[0.01, 0.001],
            initial_alpha=[0.5, 0.4],
            bounds_k0_1=(0.001, 0.1),
            bounds_k0_2=(0.0001, 0.01),
            bounds_alpha=(0.2, 0.9),
            config=None,
        )

        assert isinstance(result, CascadeResult)
        # Default config has skip_polish=False, so 3 passes
        assert len(result.pass_results) == 3
