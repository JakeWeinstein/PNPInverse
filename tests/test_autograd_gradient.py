"""Verify autograd gradients match finite-difference gradients.

Tests that the PyTorch autograd gradient path (added in Phase 2e) produces
gradients consistent with central finite-difference approximations for all
objective classes and surrogate wrappers.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _TORCH_AVAILABLE, reason="PyTorch not available",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_nn_model():
    """Create a small fitted NNSurrogateModel on synthetic data.

    Uses a tiny network (hidden=16, 1 block) fit on 50 random samples
    to keep the test fast.  Accuracy is irrelevant -- we only need a
    differentiable forward pass that produces non-trivial gradients.
    """
    from Surrogate.nn_model import NNSurrogateModel

    rng = np.random.default_rng(12345)
    n_samples = 50
    n_eta = 22
    phi_applied = np.linspace(-0.8, 0.2, n_eta)

    # Random parameters in physical space
    k0_1 = 10.0 ** rng.uniform(-6, -1, size=n_samples)
    k0_2 = 10.0 ** rng.uniform(-8, -3, size=n_samples)
    alpha_1 = rng.uniform(0.1, 0.9, size=n_samples)
    alpha_2 = rng.uniform(0.1, 0.9, size=n_samples)
    params = np.column_stack([k0_1, k0_2, alpha_1, alpha_2])

    # Synthetic I-V curves (not physically meaningful)
    cd = rng.standard_normal((n_samples, n_eta)) * 0.01
    pc = rng.standard_normal((n_samples, n_eta)) * 0.001

    model = NNSurrogateModel(hidden=16, n_blocks=1, seed=42)
    model.fit(
        params, cd, pc, phi_applied,
        epochs=200, patience=200, verbose=False,
    )
    return model


@pytest.fixture
def dummy_ensemble(dummy_nn_model):
    """Create a 2-member ensemble from the dummy model."""
    from Surrogate.nn_model import NNSurrogateModel
    from Surrogate.ensemble import EnsembleMeanWrapper

    # Second member with different seed
    rng = np.random.default_rng(12345)
    n_samples = 50
    n_eta = 22
    phi_applied = np.linspace(-0.8, 0.2, n_eta)

    k0_1 = 10.0 ** rng.uniform(-6, -1, size=n_samples)
    k0_2 = 10.0 ** rng.uniform(-8, -3, size=n_samples)
    alpha_1 = rng.uniform(0.1, 0.9, size=n_samples)
    alpha_2 = rng.uniform(0.1, 0.9, size=n_samples)
    params = np.column_stack([k0_1, k0_2, alpha_1, alpha_2])

    cd = rng.standard_normal((n_samples, n_eta)) * 0.01
    pc = rng.standard_normal((n_samples, n_eta)) * 0.001

    model2 = NNSurrogateModel(hidden=16, n_blocks=1, seed=99)
    model2.fit(
        params, cd, pc, phi_applied,
        epochs=200, patience=200, verbose=False,
    )

    return EnsembleMeanWrapper([dummy_nn_model, model2])


def _fd_gradient(obj_fn, x, h=1e-5):
    """Central finite-difference gradient of a scalar function."""
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    for i in range(len(x)):
        xp = x.copy()
        xm = x.copy()
        xp[i] += h
        xm[i] -= h
        grad[i] = (obj_fn(xp) - obj_fn(xm)) / (2 * h)
    return grad


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestAutogradVsFD:
    """Compare autograd gradient to central FD at multiple random points."""

    def _assert_gradients_close(self, g_auto, g_fd, rtol=1e-3):
        """Assert relative error < rtol element-wise."""
        denom = np.maximum(np.abs(g_fd), 1e-12)
        rel_err = np.abs(g_auto - g_fd) / denom
        max_rel = np.max(rel_err)
        assert max_rel < rtol, (
            f"Max relative gradient error {max_rel:.6e} exceeds {rtol:.1e}\n"
            f"  autograd: {g_auto}\n"
            f"  FD:       {g_fd}\n"
            f"  rel_err:  {rel_err}"
        )

    def test_nn_model_predict_torch(self, dummy_nn_model):
        """predict_torch() returns a tensor with correct shape and grad support."""
        x = torch.tensor([-3.0, -5.0, 0.3, 0.5], dtype=torch.float64, requires_grad=True)
        y = dummy_nn_model.predict_torch(x)
        assert y.shape == (2 * dummy_nn_model.n_eta,)
        assert y.requires_grad
        # Backward should not raise
        y.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == (4,)

    def test_ensemble_predict_torch(self, dummy_ensemble):
        """Ensemble predict_torch() returns correct shape with grad."""
        x = torch.tensor([-3.0, -5.0, 0.3, 0.5], dtype=torch.float64, requires_grad=True)
        y = dummy_ensemble.predict_torch(x)
        assert y.shape == (2 * dummy_ensemble.n_eta,)
        assert y.requires_grad
        y.sum().backward()
        assert x.grad is not None

    def test_surrogate_objective_gradient_match(self, dummy_nn_model):
        """Autograd and FD gradients agree to < 0.1% for SurrogateObjective."""
        from Surrogate.objectives import SurrogateObjective

        n_eta = dummy_nn_model.n_eta
        rng = np.random.default_rng(777)

        target_cd = rng.standard_normal(n_eta) * 0.01
        target_pc = rng.standard_normal(n_eta) * 0.001

        obj = SurrogateObjective(
            dummy_nn_model, target_cd, target_pc, secondary_weight=1.0,
        )
        assert obj._use_autograd is True

        for _ in range(5):
            x = np.array([
                rng.uniform(-5, -2),
                rng.uniform(-7, -4),
                rng.uniform(0.2, 0.8),
                rng.uniform(0.2, 0.8),
            ])
            J_auto, g_auto = obj._autograd_objective_and_gradient(x)

            # Compute FD gradient using the numpy objective path
            g_fd = _fd_gradient(obj.objective, x, h=1e-5)

            self._assert_gradients_close(g_auto, g_fd)

    def test_ensemble_objective_gradient_match(self, dummy_ensemble):
        """Ensemble autograd gradient matches FD for SurrogateObjective."""
        from Surrogate.objectives import SurrogateObjective

        n_eta = dummy_ensemble.n_eta
        rng = np.random.default_rng(888)

        target_cd = rng.standard_normal(n_eta) * 0.01
        target_pc = rng.standard_normal(n_eta) * 0.001

        obj = SurrogateObjective(
            dummy_ensemble, target_cd, target_pc, secondary_weight=1.0,
        )
        assert obj._use_autograd is True

        for _ in range(3):
            x = np.array([
                rng.uniform(-5, -2),
                rng.uniform(-7, -4),
                rng.uniform(0.2, 0.8),
                rng.uniform(0.2, 0.8),
            ])
            _, g_auto = obj._autograd_objective_and_gradient(x)
            g_fd = _fd_gradient(obj.objective, x, h=1e-5)
            self._assert_gradients_close(g_auto, g_fd)

    def test_subset_objective_gradient_match(self, dummy_nn_model):
        """SubsetSurrogateObjective autograd gradient matches FD."""
        from Surrogate.objectives import SubsetSurrogateObjective

        n_eta = dummy_nn_model.n_eta
        rng = np.random.default_rng(999)

        # Use a subset of voltage indices
        subset_idx = np.array([2, 5, 8, 12, 15, 18], dtype=int)
        target_cd = rng.standard_normal(len(subset_idx)) * 0.01
        target_pc = rng.standard_normal(len(subset_idx)) * 0.001

        obj = SubsetSurrogateObjective(
            dummy_nn_model, target_cd, target_pc,
            subset_idx=subset_idx, secondary_weight=1.0,
        )
        assert obj._use_autograd is True

        for _ in range(5):
            x = np.array([
                rng.uniform(-5, -2),
                rng.uniform(-7, -4),
                rng.uniform(0.2, 0.8),
                rng.uniform(0.2, 0.8),
            ])
            _, g_auto = obj._autograd_objective_and_gradient(x)
            g_fd = _fd_gradient(obj.objective, x, h=1e-5)
            self._assert_gradients_close(g_auto, g_fd)

    def test_block_objective_gradient_match(self, dummy_nn_model):
        """ReactionBlockSurrogateObjective autograd gradient matches FD."""
        from Surrogate.objectives import ReactionBlockSurrogateObjective

        n_eta = dummy_nn_model.n_eta
        rng = np.random.default_rng(111)

        target_cd = rng.standard_normal(n_eta) * 0.01
        target_pc = rng.standard_normal(n_eta) * 0.001

        for reaction_index in (0, 1):
            obj = ReactionBlockSurrogateObjective(
                dummy_nn_model, target_cd, target_pc,
                reaction_index=reaction_index,
                fixed_k0_other=1e-4,
                fixed_alpha_other=0.5,
                secondary_weight=1.0,
            )
            assert obj._use_autograd is True

            for _ in range(5):
                x = np.array([
                    rng.uniform(-5, -2),
                    rng.uniform(0.2, 0.8),
                ])
                _, g_auto = obj._autograd_objective_and_gradient(x)
                g_fd = _fd_gradient(obj.objective, x, h=1e-5)
                self._assert_gradients_close(g_auto, g_fd)

    def test_alpha_only_objective_gradient_match(self, dummy_nn_model):
        """AlphaOnlySurrogateObjective autograd gradient matches FD."""
        from Surrogate.objectives import AlphaOnlySurrogateObjective

        n_eta = dummy_nn_model.n_eta
        rng = np.random.default_rng(222)

        target_cd = rng.standard_normal(n_eta) * 0.01
        target_pc = rng.standard_normal(n_eta) * 0.001

        obj = AlphaOnlySurrogateObjective(
            dummy_nn_model, target_cd, target_pc,
            fixed_k0=(1e-3, 1e-5),
            secondary_weight=1.0,
        )
        assert obj._use_autograd is True

        for _ in range(5):
            x = np.array([
                rng.uniform(0.2, 0.8),
                rng.uniform(0.2, 0.8),
            ])
            _, g_auto = obj._autograd_objective_and_gradient(x)
            g_fd = _fd_gradient(obj.objective, x, h=1e-5)
            self._assert_gradients_close(g_auto, g_fd)

    def test_fallback_to_fd_for_non_torch_surrogate(self):
        """Non-torch surrogates use FD without error and _use_autograd is False."""
        from Surrogate.objectives import SurrogateObjective

        class MockRBFSurrogate:
            """Mock surrogate without predict_torch (simulates RBF)."""

            def predict(self, k0_1, k0_2, alpha_1, alpha_2):
                n_eta = 22
                return {
                    "current_density": np.zeros(n_eta),
                    "peroxide_current": np.zeros(n_eta),
                    "phi_applied": np.linspace(-0.8, 0.2, n_eta),
                }

        mock = MockRBFSurrogate()
        target_cd = np.zeros(22)
        target_pc = np.zeros(22)

        obj = SurrogateObjective(mock, target_cd, target_pc)
        assert obj._use_autograd is False

        # Should work via FD without error
        x = np.array([-3.0, -5.0, 0.3, 0.5])
        J, g = obj.objective_and_gradient(x)
        assert np.isfinite(J)
        assert g.shape == (4,)

    def test_objective_and_gradient_consistency(self, dummy_nn_model):
        """objective_and_gradient returns same J as standalone objective."""
        from Surrogate.objectives import SurrogateObjective

        n_eta = dummy_nn_model.n_eta
        rng = np.random.default_rng(333)

        target_cd = rng.standard_normal(n_eta) * 0.01
        target_pc = rng.standard_normal(n_eta) * 0.001

        obj = SurrogateObjective(
            dummy_nn_model, target_cd, target_pc, secondary_weight=1.0,
        )

        x = np.array([-3.0, -5.0, 0.3, 0.5])

        # autograd path
        J_combined, _ = obj.objective_and_gradient(x)

        # numpy path (standalone objective)
        J_standalone = obj.objective(x)

        assert abs(J_combined - J_standalone) < 1e-10 * max(abs(J_combined), 1.0), (
            f"Objective mismatch: combined={J_combined}, standalone={J_standalone}"
        )
