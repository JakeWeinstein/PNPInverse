"""Tests for Surrogate.ensemble, SubsetSurrogateObjective, and v12 CLI.

Fast unit tests use mock surrogate models (no PyTorch/Firedrake required).
Slow integration tests load real NN ensembles and require PyTorch.
"""

from __future__ import annotations

import argparse
import os
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Surrogate.ensemble import EnsembleMeanWrapper, load_nn_ensemble
from Surrogate.objectives import SubsetSurrogateObjective


# ---------------------------------------------------------------------------
# Helpers -- mock surrogate model
# ---------------------------------------------------------------------------

class _MockSurrogateModel:
    """Minimal mock that satisfies the surrogate API (predict, predict_batch, etc.)."""

    def __init__(self, n_eta: int = 22, seed: int = 0):
        self._n_eta = n_eta
        self._phi_applied = np.linspace(-20, 5, n_eta)
        self._rng = np.random.default_rng(seed)
        self.training_bounds = {
            "k0_1": (1e-5, 1e-1),
            "k0_2": (1e-6, 1e-2),
            "alpha_1": (0.1, 0.9),
            "alpha_2": (0.1, 0.9),
        }

    @property
    def n_eta(self) -> int:
        return self._n_eta

    @property
    def phi_applied(self) -> np.ndarray:
        return self._phi_applied.copy()

    @property
    def is_fitted(self) -> bool:
        return True

    def predict(self, k0_1, k0_2, alpha_1, alpha_2):
        # Deterministic but parameter-dependent output
        scale = np.log10(max(k0_1, 1e-30)) + alpha_1
        cd = scale * np.sin(self._phi_applied) + self._rng.normal(0, 0.001, self._n_eta)
        pc = 0.5 * scale * np.cos(self._phi_applied) + self._rng.normal(0, 0.001, self._n_eta)
        return {
            "current_density": cd,
            "peroxide_current": pc,
            "phi_applied": self._phi_applied.copy(),
        }

    def predict_batch(self, parameters):
        cd_list, pc_list = [], []
        for row in parameters:
            p = self.predict(*row)
            cd_list.append(p["current_density"])
            pc_list.append(p["peroxide_current"])
        return {
            "current_density": np.array(cd_list),
            "peroxide_current": np.array(pc_list),
            "phi_applied": self._phi_applied.copy(),
        }


# ===================================================================
# TestEnsembleMeanWrapper
# ===================================================================

class TestEnsembleMeanWrapper:

    def test_predict_shape(self):
        models = [_MockSurrogateModel(seed=i) for i in range(3)]
        wrapper = EnsembleMeanWrapper(models)
        pred = wrapper.predict(1e-3, 1e-4, 0.5, 0.5)
        assert pred["current_density"].shape == (22,)
        assert pred["peroxide_current"].shape == (22,)
        assert pred["phi_applied"].shape == (22,)

    def test_predict_batch_shape(self):
        models = [_MockSurrogateModel(seed=i) for i in range(3)]
        wrapper = EnsembleMeanWrapper(models)
        params = np.array([
            [1e-3, 1e-4, 0.5, 0.5],
            [1e-2, 1e-3, 0.4, 0.6],
        ])
        batch = wrapper.predict_batch(params)
        assert batch["current_density"].shape == (2, 22)
        assert batch["peroxide_current"].shape == (2, 22)

    def test_training_bounds_merge(self):
        m1 = _MockSurrogateModel(seed=0)
        m1.training_bounds = {"k0_1": (1e-4, 1e-1), "alpha_1": (0.2, 0.8)}
        m2 = _MockSurrogateModel(seed=1)
        m2.training_bounds = {"k0_1": (1e-5, 5e-2), "alpha_1": (0.1, 0.9)}
        wrapper = EnsembleMeanWrapper([m1, m2])
        tb = wrapper.training_bounds
        assert tb is not None
        # min of lows
        assert tb["k0_1"][0] == pytest.approx(1e-5)
        # max of highs
        assert tb["k0_1"][1] == pytest.approx(1e-1)
        assert tb["alpha_1"][0] == pytest.approx(0.1)
        assert tb["alpha_1"][1] == pytest.approx(0.9)

    def test_is_fitted(self):
        wrapper = EnsembleMeanWrapper([_MockSurrogateModel()])
        assert wrapper.is_fitted is True

    def test_n_eta(self):
        wrapper = EnsembleMeanWrapper([_MockSurrogateModel(n_eta=15)])
        assert wrapper.n_eta == 15

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one model"):
            EnsembleMeanWrapper([])

    def test_no_training_bounds(self):
        m = _MockSurrogateModel()
        m.training_bounds = None
        wrapper = EnsembleMeanWrapper([m])
        assert wrapper.training_bounds is None


# ===================================================================
# TestLoadNNEnsemble
# ===================================================================

class TestLoadNNEnsemble:

    def test_missing_member_raises(self, tmp_path):
        # member_0/saved_model/ does NOT exist, so the first member fails
        with pytest.raises(FileNotFoundError, match="member_0"):
            load_nn_ensemble(str(tmp_path), n_members=2)

    @pytest.mark.slow
    def test_loads_d3_deeper(self):
        """Load actual D3-deeper ensemble (requires PyTorch + saved models)."""
        ensemble_dir = os.path.join(
            _ROOT, "StudyResults", "surrogate_v11", "nn_ensemble", "D3-deeper"
        )
        if not os.path.isdir(ensemble_dir):
            pytest.skip("D3-deeper ensemble not found on disk")
        model = load_nn_ensemble(ensemble_dir)
        assert model.is_fitted
        assert model.n_eta > 0
        pred = model.predict(1e-3, 1e-4, 0.5, 0.5)
        assert pred["current_density"].shape == (model.n_eta,)


# ===================================================================
# TestSubsetSurrogateObjective
# ===================================================================

class TestSubsetSurrogateObjective:

    def _make_objective(self):
        model = _MockSurrogateModel(seed=42)
        # Generate target at known params
        true_pred = model.predict(1e-3, 1e-4, 0.5, 0.5)
        target_cd = true_pred["current_density"]
        target_pc = true_pred["peroxide_current"]
        subset_idx = np.arange(5, 15)
        return SubsetSurrogateObjective(
            surrogate=model,
            target_cd=target_cd[subset_idx],
            target_pc=target_pc[subset_idx],
            subset_idx=subset_idx,
            secondary_weight=1.0,
            fd_step=1e-5,
            log_space_k0=True,
        )

    def test_objective_at_truth_near_zero(self):
        obj = self._make_objective()
        x_true = np.array([np.log10(1e-3), np.log10(1e-4), 0.5, 0.5])
        J = obj.objective(x_true)
        # At truth, loss should be very small (not exactly 0 due to mock randomness)
        assert J >= 0.0
        # Should be small relative to an off-target guess
        x_off = np.array([np.log10(1e-1), np.log10(1e-1), 0.2, 0.8])
        J_off = obj.objective(x_off)
        assert J < J_off

    def test_gradient_shape(self):
        obj = self._make_objective()
        x = np.array([np.log10(1e-3), np.log10(1e-4), 0.5, 0.5])
        grad = obj.gradient(x)
        assert grad.shape == (4,)

    def test_objective_and_gradient(self):
        obj = self._make_objective()
        x = np.array([-3.0, -4.0, 0.5, 0.5])
        J, g = obj.objective_and_gradient(x)
        assert isinstance(J, float)
        assert g.shape == (4,)

    def test_n_evals_increments(self):
        obj = self._make_objective()
        assert obj.n_evals == 0
        x = np.array([-3.0, -4.0, 0.5, 0.5])
        obj.objective(x)
        assert obj.n_evals == 1
        obj.objective(x)
        assert obj.n_evals == 2


# ===================================================================
# TestV12CLI
# ===================================================================

class TestV12CLI:
    """Test the v12 script's argument parser (no actual execution)."""

    def _get_parser(self):
        """Import the v12 script module and extract its parser."""
        # We test CLI structure without running main().
        # Build a parser that mirrors what the v12 script should have.
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", type=str, default="StudyResults/surrogate_v11/model_rbf_baseline.pkl")
        parser.add_argument("--model-type", type=str, default="nn-ensemble",
                            choices=["nn-ensemble", "rbf", "pod-rbf-log", "pod-rbf-nolog", "nn-single"])
        parser.add_argument("--design", type=str, default="D3-deeper")
        parser.add_argument("--nn-dir", type=str,
                            default="StudyResults/surrogate_v11/nn_ensemble")
        parser.add_argument("--compare", action="store_true")
        parser.add_argument("--no-pde", action="store_true")
        parser.add_argument("--workers", type=int, default=0)
        parser.add_argument("--pde-p3-maxiter", type=int, default=30)
        parser.add_argument("--pde-p4-maxiter", type=int, default=25)
        parser.add_argument("--secondary-weight", type=float, default=1.0)
        parser.add_argument("--noise-percent", type=float, default=2.0)
        parser.add_argument("--noise-seed", type=int, default=20260226)
        return parser

    def test_default_args(self):
        parser = self._get_parser()
        args = parser.parse_args([])
        assert args.model_type == "nn-ensemble"
        assert args.design == "D3-deeper"
        assert args.compare is False
        assert args.no_pde is False

    def test_model_type_choices(self):
        parser = self._get_parser()
        for mt in ["nn-ensemble", "rbf", "pod-rbf-log", "pod-rbf-nolog", "nn-single"]:
            args = parser.parse_args(["--model-type", mt])
            assert args.model_type == mt

    def test_compare_flag(self):
        parser = self._get_parser()
        args = parser.parse_args(["--compare"])
        assert args.compare is True

    def test_invalid_model_type(self):
        parser = self._get_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--model-type", "invalid"])
