"""Tests for Block Coordinate Descent (BCD) inference infrastructure.

Tests cover:
- ReactionBlockSurrogateObjective (reaction 0/1, invalid index, weight,
  fixed params, gradient shape, n_evals, consistency with full objective)
- BCDConfig frozen dataclass (defaults, frozen)
- BCDIterationResult frozen dataclass (creation, frozen)
- BCDResult frozen dataclass (creation, frozen, iteration_history tuple)
- run_block_coordinate_descent (convergence, recovery accuracy,
  early stopping, custom weights, verbose suppression)
"""

from __future__ import annotations

import io
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
from Surrogate.objectives import (
    ReactionBlockSurrogateObjective,
    SurrogateObjective,
)
from Surrogate.bcd import (
    BCDConfig,
    BCDIterationResult,
    BCDResult,
    run_block_coordinate_descent,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def fitted_surrogate() -> BVSurrogateModel:
    """Build and fit a small BVSurrogateModel for testing.

    Uses 40 training samples with 8 voltage points and a non-separable
    response function so that all 4 parameters are identifiable when
    fitting via block coordinate descent.

    The response mimics a simplified BV kinetic model where:
    - cd depends on k0_1, alpha_1, AND k0_2 (via selectivity coupling)
    - pc depends on k0_2, alpha_2, AND k0_1 (via total current coupling)
    This makes each block's sub-problem well-determined even when
    fixing the other reaction's parameters.
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

    # Parameters in physical space
    k0_1 = rng.uniform(0.005, 0.05, n_samples)
    k0_2 = rng.uniform(0.0005, 0.005, n_samples)
    alpha_1 = rng.uniform(0.4, 0.8, n_samples)
    alpha_2 = rng.uniform(0.3, 0.7, n_samples)
    params = np.column_stack([k0_1, k0_2, alpha_1, alpha_2])

    # Non-separable response (mimics coupled BV kinetics):
    # cd = -alpha_1 * phi * k0_1 * 10 + alpha_2 * k0_2 * phi^2 * 0.1
    # pc = -alpha_2 * phi * k0_2 * 100 + alpha_1 * k0_1 * phi^2 * 0.5
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

    pred = fitted_surrogate.predict(k0_1_true, k0_2_true, alpha_1_true, alpha_2_true)
    return {
        "target_cd": pred["current_density"],
        "target_pc": pred["peroxide_current"],
        "k0_1_true": k0_1_true,
        "k0_2_true": k0_2_true,
        "alpha_1_true": alpha_1_true,
        "alpha_2_true": alpha_2_true,
    }


# ---------------------------------------------------------------------------
# Tests: ReactionBlockSurrogateObjective
# ---------------------------------------------------------------------------

class TestReactionBlockSurrogateObjective:
    """Tests for the ReactionBlockSurrogateObjective class."""

    def test_reaction_0_free_params(self, fitted_surrogate, target_data):
        """Reaction 0 optimizes (k0_1, alpha_1), fixes (k0_2, alpha_2)."""
        obj = ReactionBlockSurrogateObjective(
            surrogate=fitted_surrogate,
            target_cd=target_data["target_cd"],
            target_pc=target_data["target_pc"],
            reaction_index=0,
            fixed_k0_other=target_data["k0_2_true"],
            fixed_alpha_other=target_data["alpha_2_true"],
        )

        # At the true params, objective should be near zero
        x_true = np.array([np.log10(target_data["k0_1_true"]),
                           target_data["alpha_1_true"]])
        j_true = obj.objective(x_true)
        assert j_true < 1e-6, f"Objective at true params should be ~0, got {j_true}"

    def test_reaction_1_free_params(self, fitted_surrogate, target_data):
        """Reaction 1 optimizes (k0_2, alpha_2), fixes (k0_1, alpha_1)."""
        obj = ReactionBlockSurrogateObjective(
            surrogate=fitted_surrogate,
            target_cd=target_data["target_cd"],
            target_pc=target_data["target_pc"],
            reaction_index=1,
            fixed_k0_other=target_data["k0_1_true"],
            fixed_alpha_other=target_data["alpha_1_true"],
        )

        x_true = np.array([np.log10(target_data["k0_2_true"]),
                           target_data["alpha_2_true"]])
        j_true = obj.objective(x_true)
        assert j_true < 1e-6, f"Objective at true params should be ~0, got {j_true}"

    def test_invalid_reaction_index_raises(self, fitted_surrogate):
        """reaction_index not in {0, 1} should raise ValueError."""
        n_eta = fitted_surrogate.n_eta
        with pytest.raises(ValueError, match="reaction_index must be 0 or 1"):
            ReactionBlockSurrogateObjective(
                surrogate=fitted_surrogate,
                target_cd=np.ones(n_eta),
                target_pc=np.ones(n_eta),
                reaction_index=2,
                fixed_k0_other=0.01,
                fixed_alpha_other=0.5,
            )

    def test_weight_affects_objective(self, fitted_surrogate):
        """Different secondary_weight should change the objective value."""
        n_eta = fitted_surrogate.n_eta
        target_cd = np.ones(n_eta) * 0.5
        target_pc = np.ones(n_eta) * 0.3

        obj_w1 = ReactionBlockSurrogateObjective(
            surrogate=fitted_surrogate,
            target_cd=target_cd,
            target_pc=target_pc,
            reaction_index=0,
            fixed_k0_other=0.001,
            fixed_alpha_other=0.5,
            secondary_weight=1.0,
        )
        obj_w10 = ReactionBlockSurrogateObjective(
            surrogate=fitted_surrogate,
            target_cd=target_cd,
            target_pc=target_pc,
            reaction_index=0,
            fixed_k0_other=0.001,
            fixed_alpha_other=0.5,
            secondary_weight=10.0,
        )

        x = np.array([np.log10(0.01), 0.6])
        j_w1 = obj_w1.objective(x)
        j_w10 = obj_w10.objective(x)
        assert j_w1 != j_w10

    def test_fixed_params_truly_fixed(self, fitted_surrogate, target_data):
        """Changing fixed_k0_other should change the objective for same x."""
        obj_a = ReactionBlockSurrogateObjective(
            surrogate=fitted_surrogate,
            target_cd=target_data["target_cd"],
            target_pc=target_data["target_pc"],
            reaction_index=0,
            fixed_k0_other=0.001,
            fixed_alpha_other=0.5,
        )
        obj_b = ReactionBlockSurrogateObjective(
            surrogate=fitted_surrogate,
            target_cd=target_data["target_cd"],
            target_pc=target_data["target_pc"],
            reaction_index=0,
            fixed_k0_other=0.005,
            fixed_alpha_other=0.5,
        )

        x = np.array([np.log10(0.02), 0.6])
        j_a = obj_a.objective(x)
        j_b = obj_b.objective(x)
        assert j_a != j_b, "Different fixed_k0_other should give different objectives"

    def test_gradient_shape(self, fitted_surrogate):
        """Gradient should be 2D (matching control vector)."""
        n_eta = fitted_surrogate.n_eta
        obj = ReactionBlockSurrogateObjective(
            surrogate=fitted_surrogate,
            target_cd=np.ones(n_eta) * 0.5,
            target_pc=np.ones(n_eta) * 0.3,
            reaction_index=0,
            fixed_k0_other=0.001,
            fixed_alpha_other=0.5,
        )

        x = np.array([np.log10(0.01), 0.6])
        grad = obj.gradient(x)
        assert grad.shape == (2,)

    def test_n_evals_counter(self, fitted_surrogate):
        """n_evals should increment correctly."""
        n_eta = fitted_surrogate.n_eta
        obj = ReactionBlockSurrogateObjective(
            surrogate=fitted_surrogate,
            target_cd=np.ones(n_eta) * 0.5,
            target_pc=np.ones(n_eta) * 0.3,
            reaction_index=1,
            fixed_k0_other=0.01,
            fixed_alpha_other=0.6,
        )

        assert obj.n_evals == 0
        x = np.array([np.log10(0.001), 0.5])
        obj.objective(x)
        assert obj.n_evals == 1
        obj.objective(x)
        assert obj.n_evals == 2

    def test_consistency_with_full_objective(self, fitted_surrogate, target_data):
        """Block objective at true params with correct fixed values
        should match the full SurrogateObjective value."""
        full_obj = SurrogateObjective(
            surrogate=fitted_surrogate,
            target_cd=target_data["target_cd"],
            target_pc=target_data["target_pc"],
            secondary_weight=1.0,
        )

        block_obj = ReactionBlockSurrogateObjective(
            surrogate=fitted_surrogate,
            target_cd=target_data["target_cd"],
            target_pc=target_data["target_pc"],
            reaction_index=0,
            fixed_k0_other=target_data["k0_2_true"],
            fixed_alpha_other=target_data["alpha_2_true"],
            secondary_weight=1.0,
        )

        x_full = np.array([
            np.log10(target_data["k0_1_true"]),
            np.log10(target_data["k0_2_true"]),
            target_data["alpha_1_true"],
            target_data["alpha_2_true"],
        ])
        x_block = np.array([
            np.log10(target_data["k0_1_true"]),
            target_data["alpha_1_true"],
        ])

        j_full = full_obj.objective(x_full)
        j_block = block_obj.objective(x_block)

        np.testing.assert_allclose(j_full, j_block, rtol=1e-10)

    def test_objective_and_gradient(self, fitted_surrogate):
        """objective_and_gradient should return consistent values."""
        n_eta = fitted_surrogate.n_eta
        obj = ReactionBlockSurrogateObjective(
            surrogate=fitted_surrogate,
            target_cd=np.ones(n_eta) * 0.5,
            target_pc=np.ones(n_eta) * 0.3,
            reaction_index=0,
            fixed_k0_other=0.001,
            fixed_alpha_other=0.5,
        )

        x = np.array([np.log10(0.01), 0.6])
        j, g = obj.objective_and_gradient(x)

        # objective_and_gradient should be consistent with separate calls
        j2 = obj.objective(x)
        g2 = obj.gradient(x)
        np.testing.assert_allclose(j, j2, rtol=1e-12)
        # g and g2 will differ slightly due to FD but should be close
        np.testing.assert_allclose(g, g2, rtol=1e-4)


# ---------------------------------------------------------------------------
# Tests: BCDConfig
# ---------------------------------------------------------------------------

class TestBCDConfig:
    """Tests for the BCDConfig frozen dataclass."""

    def test_defaults(self):
        cfg = BCDConfig()
        assert cfg.max_outer_iters == 10
        assert cfg.inner_maxiter == 30
        assert cfg.block_1_weight == 0.5
        assert cfg.block_2_weight == 2.0
        assert cfg.convergence_rtol == 1e-4
        assert cfg.convergence_atol_k0 == 1e-6
        assert cfg.convergence_atol_alpha == 1e-5
        assert cfg.fd_step == 1e-5
        assert cfg.verbose is True

    def test_frozen(self):
        cfg = BCDConfig()
        with pytest.raises(AttributeError):
            cfg.max_outer_iters = 20  # type: ignore[misc]

    def test_custom_values(self):
        cfg = BCDConfig(
            max_outer_iters=5,
            inner_maxiter=50,
            block_1_weight=0.3,
            block_2_weight=3.0,
        )
        assert cfg.max_outer_iters == 5
        assert cfg.inner_maxiter == 50
        assert cfg.block_1_weight == 0.3
        assert cfg.block_2_weight == 3.0


# ---------------------------------------------------------------------------
# Tests: BCDIterationResult
# ---------------------------------------------------------------------------

class TestBCDIterationResult:
    """Tests for the BCDIterationResult frozen dataclass."""

    def test_creation(self):
        r = BCDIterationResult(
            iteration=0,
            k0_1=0.01, k0_2=0.001,
            alpha_1=0.6, alpha_2=0.5,
            loss_after_block_1=0.01,
            loss_after_block_2=0.005,
            block_1_inner_iters=15,
            block_2_inner_iters=10,
            elapsed_s=1.5,
        )
        assert r.iteration == 0
        assert r.k0_1 == 0.01
        assert r.loss_after_block_2 == 0.005

    def test_frozen(self):
        r = BCDIterationResult(
            iteration=0,
            k0_1=0.01, k0_2=0.001,
            alpha_1=0.6, alpha_2=0.5,
            loss_after_block_1=0.01,
            loss_after_block_2=0.005,
            block_1_inner_iters=15,
            block_2_inner_iters=10,
            elapsed_s=1.5,
        )
        with pytest.raises(AttributeError):
            r.k0_1 = 0.02  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Tests: BCDResult
# ---------------------------------------------------------------------------

class TestBCDResult:
    """Tests for the BCDResult frozen dataclass."""

    def test_creation(self):
        history = (
            BCDIterationResult(
                iteration=0, k0_1=0.01, k0_2=0.001,
                alpha_1=0.6, alpha_2=0.5,
                loss_after_block_1=0.01, loss_after_block_2=0.005,
                block_1_inner_iters=15, block_2_inner_iters=10,
                elapsed_s=1.5,
            ),
        )
        r = BCDResult(
            k0_1=0.01, k0_2=0.001,
            alpha_1=0.6, alpha_2=0.5,
            final_loss=0.005,
            n_outer_iters=1,
            total_surrogate_evals=100,
            converged=True,
            convergence_reason="relative change below rtol",
            iteration_history=history,
            elapsed_s=1.5,
        )
        assert r.converged is True
        assert r.n_outer_iters == 1
        assert r.final_loss == 0.005

    def test_frozen(self):
        r = BCDResult(
            k0_1=0.01, k0_2=0.001,
            alpha_1=0.6, alpha_2=0.5,
            final_loss=0.005,
            n_outer_iters=1,
            total_surrogate_evals=100,
            converged=True,
            convergence_reason="test",
            iteration_history=(),
            elapsed_s=1.5,
        )
        with pytest.raises(AttributeError):
            r.k0_1 = 0.02  # type: ignore[misc]

    def test_iteration_history_is_tuple(self):
        r = BCDResult(
            k0_1=0.01, k0_2=0.001,
            alpha_1=0.6, alpha_2=0.5,
            final_loss=0.005,
            n_outer_iters=0,
            total_surrogate_evals=0,
            converged=False,
            convergence_reason="test",
            iteration_history=(),
            elapsed_s=0.0,
        )
        assert isinstance(r.iteration_history, tuple)


# ---------------------------------------------------------------------------
# Tests: run_block_coordinate_descent
# ---------------------------------------------------------------------------

class TestRunBlockCoordinateDescent:
    """Tests for the run_block_coordinate_descent function."""

    def test_basic_convergence(self, fitted_surrogate, target_data):
        """BCD should converge (or run to max iters) without error."""
        config = BCDConfig(
            max_outer_iters=3,
            inner_maxiter=10,
            block_1_weight=1.0,
            block_2_weight=1.0,
            verbose=False,
        )

        result = run_block_coordinate_descent(
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

        assert isinstance(result, BCDResult)
        assert result.n_outer_iters >= 1
        assert result.total_surrogate_evals > 0
        assert len(result.iteration_history) == result.n_outer_iters

    def test_loss_reduction(self, fitted_surrogate, target_data):
        """BCD should substantially reduce the loss from the initial guess.

        The simple polynomial test surrogate has inherent k0*alpha
        ambiguity (multiplicative coupling), so we test loss reduction
        rather than exact parameter recovery.  Exact recovery is tested
        against the real PDE-trained surrogate in the integration test.
        """
        config = BCDConfig(
            max_outer_iters=10,
            inner_maxiter=30,
            block_1_weight=1.0,
            block_2_weight=1.0,
            verbose=False,
        )

        initial_k0 = [0.015, 0.0015]
        initial_alpha = [0.55, 0.45]

        # Compute initial loss for comparison
        init_obj = SurrogateObjective(
            surrogate=fitted_surrogate,
            target_cd=target_data["target_cd"],
            target_pc=target_data["target_pc"],
            secondary_weight=1.0,
        )
        x_init = np.array([
            np.log10(initial_k0[0]),
            np.log10(initial_k0[1]),
            initial_alpha[0],
            initial_alpha[1],
        ])
        initial_loss = init_obj.objective(x_init)

        result = run_block_coordinate_descent(
            surrogate=fitted_surrogate,
            target_cd=target_data["target_cd"],
            target_pc=target_data["target_pc"],
            initial_k0=initial_k0,
            initial_alpha=initial_alpha,
            bounds_k0_1=(0.005, 0.05),
            bounds_k0_2=(0.0005, 0.005),
            bounds_alpha=(0.3, 0.9),
            config=config,
        )

        # BCD should reduce the loss by at least 50%
        assert result.final_loss < initial_loss * 0.5, (
            f"BCD final_loss={result.final_loss:.4e} did not improve "
            f"sufficiently from initial_loss={initial_loss:.4e}"
        )
        # k0_2 and alpha_2 should be recovered well (less ambiguous)
        k0_2_err = abs(result.k0_2 - target_data["k0_2_true"]) / target_data["k0_2_true"]
        alpha_2_err = abs(result.alpha_2 - target_data["alpha_2_true"]) / target_data["alpha_2_true"]
        assert k0_2_err < 0.5, f"k0_2 error {k0_2_err*100:.1f}% too large"
        assert alpha_2_err < 0.5, f"alpha_2 error {alpha_2_err*100:.1f}% too large"

    def test_convergence_stops_early(self, fitted_surrogate, target_data):
        """With tight convergence tolerances and many allowed iters,
        BCD should converge early (before max_outer_iters)."""
        config = BCDConfig(
            max_outer_iters=50,
            inner_maxiter=30,
            block_1_weight=1.0,
            block_2_weight=1.0,
            convergence_rtol=1e-3,
            convergence_atol_k0=1e-4,
            convergence_atol_alpha=1e-4,
            verbose=False,
        )

        result = run_block_coordinate_descent(
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

        # Should converge well before 50 iterations
        assert result.n_outer_iters < 50, (
            f"Expected early convergence, but ran {result.n_outer_iters} iters"
        )

    def test_custom_block_weights(self, fitted_surrogate, target_data):
        """Different block weights should produce a valid result."""
        config = BCDConfig(
            max_outer_iters=3,
            inner_maxiter=10,
            block_1_weight=0.5,
            block_2_weight=2.0,
            verbose=False,
        )

        result = run_block_coordinate_descent(
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

        assert isinstance(result, BCDResult)
        assert result.n_outer_iters >= 1

    def test_verbose_false_no_output(self, fitted_surrogate, target_data, capsys):
        """verbose=False should suppress all print output from BCD."""
        config = BCDConfig(
            max_outer_iters=2,
            inner_maxiter=5,
            verbose=False,
        )

        run_block_coordinate_descent(
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

    def test_verbose_true_produces_output(self, fitted_surrogate, target_data, capsys):
        """verbose=True should produce output containing BCD iter info."""
        config = BCDConfig(
            max_outer_iters=2,
            inner_maxiter=5,
            verbose=True,
        )

        run_block_coordinate_descent(
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
        assert "BCD iter" in captured.out

    def test_result_final_loss_uses_weight_one(self, fitted_surrogate, target_data):
        """final_loss in BCDResult should be computed with weight=1.0
        regardless of block weights."""
        config = BCDConfig(
            max_outer_iters=3,
            inner_maxiter=10,
            block_1_weight=0.1,
            block_2_weight=10.0,
            verbose=False,
        )

        result = run_block_coordinate_descent(
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

        # Verify by computing with SurrogateObjective at weight=1.0
        full_obj = SurrogateObjective(
            surrogate=fitted_surrogate,
            target_cd=target_data["target_cd"],
            target_pc=target_data["target_pc"],
            secondary_weight=1.0,
        )
        x_final = np.array([
            np.log10(result.k0_1),
            np.log10(result.k0_2),
            result.alpha_1,
            result.alpha_2,
        ])
        expected_loss = full_obj.objective(x_final)

        np.testing.assert_allclose(result.final_loss, expected_loss, rtol=1e-10)
