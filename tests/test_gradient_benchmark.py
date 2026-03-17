"""Smoke tests for the gradient benchmark script.

Tests that benchmark functions return correctly structured output
using mock/synthetic surrogates -- does NOT require loading full
trained models.
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
# Helpers: tiny synthetic surrogates
# ---------------------------------------------------------------------------

def _make_dummy_ensemble():
    """Build a 2-member NN ensemble on synthetic data for testing."""
    from Surrogate.nn_model import NNSurrogateModel
    from Surrogate.ensemble import EnsembleMeanWrapper

    rng = np.random.default_rng(42)
    n_samples, n_eta = 50, 22
    phi = np.linspace(-0.8, 0.2, n_eta)

    k0_1 = 10.0 ** rng.uniform(-6, -1, size=n_samples)
    k0_2 = 10.0 ** rng.uniform(-8, -3, size=n_samples)
    a1 = rng.uniform(0.1, 0.9, size=n_samples)
    a2 = rng.uniform(0.1, 0.9, size=n_samples)
    params = np.column_stack([k0_1, k0_2, a1, a2])
    cd = rng.standard_normal((n_samples, n_eta)) * 0.01
    pc = rng.standard_normal((n_samples, n_eta)) * 0.001

    models = []
    for seed in (42, 99):
        m = NNSurrogateModel(hidden=16, n_blocks=1, seed=seed)
        m.fit(params, cd, pc, phi, epochs=100, patience=100, verbose=False)
        models.append(m)

    return EnsembleMeanWrapper(models)


@pytest.fixture(scope="module")
def dummy_ensemble():
    return _make_dummy_ensemble()


# ---------------------------------------------------------------------------
# Test 1: run_accuracy_benchmark returns a dict with keys per model
# ---------------------------------------------------------------------------

class TestAccuracyBenchmark:
    def test_returns_dict_with_model_keys(self, dummy_ensemble):
        from scripts.studies.gradient_benchmark import run_accuracy_benchmark

        # Provide the ensemble under a label
        models = {"nn_ensemble": {"model": dummy_ensemble, "has_autograd": True}}
        n_eta = dummy_ensemble.n_eta
        rng = np.random.default_rng(7)
        target_cd = rng.standard_normal(n_eta) * 0.01
        target_pc = rng.standard_normal(n_eta) * 0.001
        test_points = [np.array([-3.0, -5.0, 0.3, 0.5])]

        results = run_accuracy_benchmark(
            models, test_points, target_cd, target_pc, fd_steps=[1e-4],
        )

        assert isinstance(results, dict)
        assert "nn_ensemble" in results

    def test_accuracy_results_contain_relative_error(self, dummy_ensemble):
        from scripts.studies.gradient_benchmark import run_accuracy_benchmark

        models = {"nn_ensemble": {"model": dummy_ensemble, "has_autograd": True}}
        n_eta = dummy_ensemble.n_eta
        rng = np.random.default_rng(7)
        target_cd = rng.standard_normal(n_eta) * 0.01
        target_pc = rng.standard_normal(n_eta) * 0.001
        test_points = [np.array([-3.0, -5.0, 0.3, 0.5])]

        results = run_accuracy_benchmark(
            models, test_points, target_cd, target_pc, fd_steps=[1e-4],
        )

        model_res = results["nn_ensemble"]
        # Must have at least one method entry with relative_error
        assert len(model_res) > 0
        for method_result in model_res:
            assert "relative_error" in method_result
            assert "method" in method_result


# ---------------------------------------------------------------------------
# Test 2: run_speed_benchmark returns timing dict with ms_per_eval > 0
# ---------------------------------------------------------------------------

class TestSpeedBenchmark:
    def test_returns_timing_dict(self, dummy_ensemble):
        from scripts.studies.gradient_benchmark import run_speed_benchmark

        models = {"nn_ensemble": {"model": dummy_ensemble, "has_autograd": True}}
        n_eta = dummy_ensemble.n_eta
        rng = np.random.default_rng(7)
        target_cd = rng.standard_normal(n_eta) * 0.01
        target_pc = rng.standard_normal(n_eta) * 0.001
        x = np.array([-3.0, -5.0, 0.3, 0.5])

        results = run_speed_benchmark(
            models, x, target_cd, target_pc, n_iters=5, n_warmup=1,
        )

        assert isinstance(results, dict)
        assert "nn_ensemble" in results

        for entry in results["nn_ensemble"]:
            assert "ms_per_eval" in entry
            assert entry["ms_per_eval"] > 0
            assert "method" in entry
