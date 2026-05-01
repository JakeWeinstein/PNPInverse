"""Unit tests for the ISMO surrogate retraining pipeline.

Tests cover data merging, split index updates, type dispatch, quality
checks, and model-specific retraining for RBF, POD-RBF, and NN models.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from Surrogate.ismo_retrain import (
    ISMORetrainConfig,
    ISMORetrainResult,
    MergedData,
    _check_retrain_quality,
    _correct_weights_for_normalizer_shift,
    merge_training_data,
    retrain_rbf_baseline,
    retrain_pod_rbf,
    retrain_surrogate,
    update_split_indices,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_training_data():
    """Generate synthetic I-V curve data for testing."""
    rng = np.random.default_rng(42)
    N = 50
    n_eta = 22
    params = rng.uniform(size=(N, 4))
    params[:, 0] *= 1e-3   # k0_1 scale
    params[:, 1] *= 1e-4   # k0_2 scale
    params[:, 2] = 0.3 + 0.4 * params[:, 2]  # alpha_1 in [0.3, 0.7]
    params[:, 3] = 0.3 + 0.4 * params[:, 3]  # alpha_2 in [0.3, 0.7]
    phi = np.linspace(-1.5, 0.5, n_eta)
    cd = rng.standard_normal((N, n_eta)) * 0.1
    pc = rng.standard_normal((N, n_eta)) * 0.01
    return {
        "parameters": params,
        "current_density": cd,
        "peroxide_current": pc,
        "phi_applied": phi,
    }


@pytest.fixture
def new_training_data():
    """Generate synthetic new data for merging tests."""
    rng = np.random.default_rng(99)
    N = 10
    n_eta = 22
    params = rng.uniform(size=(N, 4))
    params[:, 0] *= 1e-3
    params[:, 1] *= 1e-4
    params[:, 2] = 0.3 + 0.4 * params[:, 2]
    params[:, 3] = 0.3 + 0.4 * params[:, 3]
    phi = np.linspace(-1.5, 0.5, n_eta)
    cd = rng.standard_normal((N, n_eta)) * 0.1
    pc = rng.standard_normal((N, n_eta)) * 0.01
    return {
        "parameters": params,
        "current_density": cd,
        "peroxide_current": pc,
        "phi_applied": phi,
    }


@pytest.fixture
def default_config():
    """Default retraining config."""
    return ISMORetrainConfig()


# ---------------------------------------------------------------------------
# Tests: data merging
# ---------------------------------------------------------------------------


class TestMergeTrainingData:

    def test_basic_merge(self, synthetic_training_data, new_training_data):
        """Merge 50 old + 10 new, verify shapes."""
        merged = merge_training_data(
            synthetic_training_data, new_training_data,
        )
        assert merged.parameters.shape == (60, 4)
        assert merged.current_density.shape == (60, 22)
        assert merged.peroxide_current.shape == (60, 22)
        assert merged.n_old == 50
        assert merged.n_new_valid == 10
        assert merged.n_duplicates_dropped == 0
        assert merged.n_nan_dropped == 0
        assert len(merged.new_indices) == 10
        np.testing.assert_array_equal(
            merged.new_indices, np.arange(50, 60),
        )

    def test_duplicates_dropped(self, synthetic_training_data):
        """Duplicate rows in new data are dropped (log-space detection)."""
        # Create new data where 3 rows are exact copies of existing rows
        new_data = {
            "parameters": np.vstack([
                synthetic_training_data["parameters"][:3],  # 3 duplicates
                np.array([[5e-4, 5e-5, 0.5, 0.5]] * 7),    # 7 unique
            ]),
            "current_density": np.vstack([
                synthetic_training_data["current_density"][:3],
                np.random.default_rng(77).standard_normal((7, 22)) * 0.1,
            ]),
            "peroxide_current": np.vstack([
                synthetic_training_data["peroxide_current"][:3],
                np.random.default_rng(78).standard_normal((7, 22)) * 0.01,
            ]),
            "phi_applied": synthetic_training_data["phi_applied"],
        }
        config = ISMORetrainConfig(duplicate_param_tol=1e-6)
        merged = merge_training_data(synthetic_training_data, new_data, config)
        assert merged.n_duplicates_dropped == 3
        assert merged.n_new_valid == 7
        assert merged.parameters.shape[0] == 57

    def test_nan_dropped(self, synthetic_training_data):
        """Rows with NaN in output are dropped."""
        rng = np.random.default_rng(88)
        N_new = 10
        n_eta = 22
        new_params = rng.uniform(size=(N_new, 4))
        new_params[:, 0] *= 1e-3
        new_params[:, 1] *= 1e-4
        new_cd = rng.standard_normal((N_new, n_eta)) * 0.1
        new_pc = rng.standard_normal((N_new, n_eta)) * 0.01

        # Inject NaN into 2 rows
        new_cd[2, 5] = np.nan
        new_cd[7, 0] = np.nan

        new_data = {
            "parameters": new_params,
            "current_density": new_cd,
            "peroxide_current": new_pc,
            "phi_applied": synthetic_training_data["phi_applied"],
        }
        merged = merge_training_data(synthetic_training_data, new_data)
        assert merged.n_nan_dropped == 2
        assert merged.n_new_valid == 8
        assert merged.parameters.shape[0] == 58

    def test_phi_mismatch(self, synthetic_training_data, new_training_data):
        """ValueError raised when phi_applied grids differ."""
        bad_new = dict(new_training_data)
        bad_new["phi_applied"] = np.linspace(-2.0, 1.0, 22)
        with pytest.raises(ValueError, match="phi_applied"):
            merge_training_data(synthetic_training_data, bad_new)

    def test_immutability(self, synthetic_training_data, new_training_data):
        """Input arrays are not modified by merge."""
        old_params_copy = synthetic_training_data["parameters"].copy()
        new_params_copy = new_training_data["parameters"].copy()
        old_cd_copy = synthetic_training_data["current_density"].copy()

        merge_training_data(synthetic_training_data, new_training_data)

        np.testing.assert_array_equal(
            synthetic_training_data["parameters"], old_params_copy,
        )
        np.testing.assert_array_equal(
            new_training_data["parameters"], new_params_copy,
        )
        np.testing.assert_array_equal(
            synthetic_training_data["current_density"], old_cd_copy,
        )


# ---------------------------------------------------------------------------
# Tests: split index updates
# ---------------------------------------------------------------------------


class TestUpdateSplitIndices:

    def test_basic_update(self):
        """New data indices appended to training set."""
        train_idx = np.arange(0, 40)
        test_idx = np.arange(40, 50)
        new_train, new_test = update_split_indices(train_idx, test_idx, 50, 10)

        assert len(new_train) == 50   # 40 old + 10 new
        assert len(new_test) == 10    # unchanged
        np.testing.assert_array_equal(new_train[:40], train_idx)
        np.testing.assert_array_equal(new_train[40:], np.arange(50, 60))
        np.testing.assert_array_equal(new_test, test_idx)

    def test_no_overlap(self):
        """Train and test indices do not overlap."""
        train_idx = np.arange(0, 40)
        test_idx = np.arange(40, 50)
        new_train, new_test = update_split_indices(train_idx, test_idx, 50, 10)
        overlap = np.intersect1d(new_train, new_test)
        assert len(overlap) == 0


# ---------------------------------------------------------------------------
# Tests: RBF baseline retraining
# ---------------------------------------------------------------------------


class TestRetrainRBFBaseline:

    def test_retrain_rbf(self, synthetic_training_data, new_training_data):
        """RBF baseline retrain returns a new fitted model."""
        from Surrogate.surrogate_model import BVSurrogateModel

        # Fit original model
        old_model = BVSurrogateModel()
        old_params = synthetic_training_data["parameters"]
        old_model.fit(
            old_params,
            synthetic_training_data["current_density"],
            synthetic_training_data["peroxide_current"],
            synthetic_training_data["phi_applied"],
        )

        # Merge and retrain
        config = ISMORetrainConfig(output_base_dir=tempfile.mkdtemp())
        merged = merge_training_data(
            synthetic_training_data, new_training_data, config,
        )
        train_idx = np.arange(0, 40)
        test_idx = np.arange(40, 50)
        new_train, new_test = update_split_indices(
            train_idx, test_idx, merged.n_old, merged.n_new_valid,
        )

        new_model = retrain_rbf_baseline(
            old_model, merged, new_train, new_test, config, iteration=1,
        )

        assert new_model is not old_model
        assert new_model.is_fitted
        # Predict should work
        result = new_model.predict_batch(old_params[:5])
        assert result["current_density"].shape == (5, 22)


# ---------------------------------------------------------------------------
# Tests: POD-RBF retraining
# ---------------------------------------------------------------------------


class TestRetrainPODRBF:

    def test_retrain_pod_rbf(self, synthetic_training_data, new_training_data):
        """POD-RBF retrain returns a new fitted model."""
        from Surrogate.pod_rbf_model import PODRBFSurrogateModel

        old_model = PODRBFSurrogateModel()
        old_model.fit(
            synthetic_training_data["parameters"],
            synthetic_training_data["current_density"],
            synthetic_training_data["peroxide_current"],
            synthetic_training_data["phi_applied"],
        )

        config = ISMORetrainConfig(output_base_dir=tempfile.mkdtemp())
        merged = merge_training_data(
            synthetic_training_data, new_training_data, config,
        )
        train_idx = np.arange(0, 40)
        test_idx = np.arange(40, 50)
        new_train, new_test = update_split_indices(
            train_idx, test_idx, merged.n_old, merged.n_new_valid,
        )

        new_model = retrain_pod_rbf(
            old_model, merged, new_train, new_test, config, iteration=1,
        )

        assert new_model is not old_model
        assert new_model.is_fitted
        assert new_model.n_modes > 0


# ---------------------------------------------------------------------------
# Tests: dispatch
# ---------------------------------------------------------------------------


class TestRetrainDispatch:

    def test_dispatch_rbf(self, synthetic_training_data, new_training_data):
        """retrain_surrogate dispatches correctly for BVSurrogateModel."""
        from Surrogate.surrogate_model import BVSurrogateModel

        old_model = BVSurrogateModel()
        old_model.fit(
            synthetic_training_data["parameters"],
            synthetic_training_data["current_density"],
            synthetic_training_data["peroxide_current"],
            synthetic_training_data["phi_applied"],
        )

        config = ISMORetrainConfig(output_base_dir=tempfile.mkdtemp())
        train_idx = np.arange(0, 40)
        test_idx = np.arange(40, 50)

        new_model = retrain_surrogate(
            old_model,
            new_training_data,
            synthetic_training_data,
            config,
            iteration=1,
            train_idx=train_idx,
            test_idx=test_idx,
        )

        assert isinstance(new_model, BVSurrogateModel)
        assert new_model is not old_model
        assert new_model.is_fitted

    def test_dispatch_unknown_type(
        self, synthetic_training_data, new_training_data,
    ):
        """TypeError raised for unknown surrogate type."""
        config = ISMORetrainConfig()
        train_idx = np.arange(0, 40)
        test_idx = np.arange(40, 50)

        with pytest.raises(TypeError, match="Unknown surrogate type"):
            retrain_surrogate(
                object(),  # not a surrogate
                new_training_data,
                synthetic_training_data,
                config,
                iteration=1,
                train_idx=train_idx,
                test_idx=test_idx,
            )

    def test_dispatch_does_not_mutate_original(
        self, synthetic_training_data, new_training_data,
    ):
        """Original model is not modified by retraining."""
        from Surrogate.surrogate_model import BVSurrogateModel

        old_model = BVSurrogateModel()
        old_model.fit(
            synthetic_training_data["parameters"],
            synthetic_training_data["current_density"],
            synthetic_training_data["peroxide_current"],
            synthetic_training_data["phi_applied"],
        )

        # Save original predictions
        test_params = synthetic_training_data["parameters"][:5]
        old_pred = old_model.predict_batch(test_params)
        old_cd = old_pred["current_density"].copy()

        config = ISMORetrainConfig(output_base_dir=tempfile.mkdtemp())
        train_idx = np.arange(0, 40)
        test_idx = np.arange(40, 50)

        retrain_surrogate(
            old_model,
            new_training_data,
            synthetic_training_data,
            config,
            iteration=1,
            train_idx=train_idx,
            test_idx=test_idx,
        )

        # Original predictions unchanged
        still_pred = old_model.predict_batch(test_params)
        np.testing.assert_array_equal(still_pred["current_density"], old_cd)


# ---------------------------------------------------------------------------
# Tests: quality check
# ---------------------------------------------------------------------------


class TestQualityCheck:

    def test_quality_check_passes(self, synthetic_training_data):
        """Quality check passes when new model is similar to old."""
        from Surrogate.surrogate_model import BVSurrogateModel

        model = BVSurrogateModel()
        model.fit(
            synthetic_training_data["parameters"],
            synthetic_training_data["current_density"],
            synthetic_training_data["peroxide_current"],
            synthetic_training_data["phi_applied"],
        )

        config = ISMORetrainConfig(max_degradation_ratio=2.0)
        test_params = synthetic_training_data["parameters"][:10]
        test_cd = synthetic_training_data["current_density"][:10]
        test_pc = synthetic_training_data["peroxide_current"][:10]

        # Comparing model to itself => ratio = 1.0 => passes
        passed, metrics = _check_retrain_quality(
            model, model, test_params, test_cd, test_pc, config, "test",
        )
        assert passed
        assert metrics["ratio"] == pytest.approx(1.0)

    def test_quality_check_detects_degradation(self, synthetic_training_data):
        """Quality check fails with very strict threshold."""
        from Surrogate.surrogate_model import BVSurrogateModel

        model = BVSurrogateModel()
        model.fit(
            synthetic_training_data["parameters"],
            synthetic_training_data["current_density"],
            synthetic_training_data["peroxide_current"],
            synthetic_training_data["phi_applied"],
        )

        # Use a ratio so strict even same model might fail with noise
        # Actually, comparing model to itself gives ratio=1.0.
        # Use threshold < 1.0 to force fail
        config = ISMORetrainConfig(max_degradation_ratio=0.5)
        test_params = synthetic_training_data["parameters"][:10]
        test_cd = synthetic_training_data["current_density"][:10]
        test_pc = synthetic_training_data["peroxide_current"][:10]

        passed, metrics = _check_retrain_quality(
            model, model, test_params, test_cd, test_pc, config, "test",
        )
        # ratio=1.0 > 0.5 => should FAIL
        assert not passed


# ---------------------------------------------------------------------------
# Tests: NN-specific (require torch)
# ---------------------------------------------------------------------------


class TestNNRetraining:

    @pytest.fixture(autouse=True)
    def _check_torch(self):
        pytest.importorskip("torch")

    def _make_small_nn(self, data, seed=0):
        """Fit a small NNSurrogateModel on synthetic data."""
        from Surrogate.nn_model import NNSurrogateModel

        model = NNSurrogateModel(hidden=16, n_blocks=1, seed=seed)
        model.fit(
            data["parameters"],
            data["current_density"],
            data["peroxide_current"],
            data["phi_applied"],
            epochs=50,
            patience=30,
            verbose=False,
        )
        return model

    def test_retrain_nn_single_member(
        self, synthetic_training_data, new_training_data,
    ):
        """Single NN member retrained via warm-start."""
        from Surrogate.ismo_retrain import _retrain_single_nn

        old_model = self._make_small_nn(synthetic_training_data)
        config = ISMORetrainConfig(
            nn_retrain_epochs=20,
            nn_retrain_patience=10,
            output_base_dir=tempfile.mkdtemp(),
        )
        merged = merge_training_data(
            synthetic_training_data, new_training_data, config,
        )
        train_idx = np.arange(0, 40)
        test_idx = np.arange(40, 50)
        new_train, new_test = update_split_indices(
            train_idx, test_idx, merged.n_old, merged.n_new_valid,
        )

        new_model = _retrain_single_nn(
            old_model, merged, new_train, new_test, config, iteration=1,
        )

        assert new_model is not old_model
        assert new_model.is_fitted
        result = new_model.predict_batch(
            synthetic_training_data["parameters"][:5],
        )
        assert result["current_density"].shape == (5, 22)

    def test_retrain_dispatch_nn_ensemble(
        self, synthetic_training_data, new_training_data,
    ):
        """Ensemble retrain via dispatch returns EnsembleMeanWrapper."""
        from Surrogate.ensemble import EnsembleMeanWrapper

        m0 = self._make_small_nn(synthetic_training_data, seed=0)
        m1 = self._make_small_nn(synthetic_training_data, seed=1)
        ensemble = EnsembleMeanWrapper([m0, m1])

        config = ISMORetrainConfig(
            nn_retrain_epochs=20,
            nn_retrain_patience=10,
            output_base_dir=tempfile.mkdtemp(),
        )
        train_idx = np.arange(0, 40)
        test_idx = np.arange(40, 50)

        new_ensemble = retrain_surrogate(
            ensemble,
            new_training_data,
            synthetic_training_data,
            config,
            iteration=1,
            train_idx=train_idx,
            test_idx=test_idx,
        )

        assert isinstance(new_ensemble, EnsembleMeanWrapper)
        assert len(new_ensemble.models) == 2
        assert new_ensemble.is_fitted

    def test_weight_correction_preserves_output(self, synthetic_training_data):
        """Analytical weight correction preserves model predictions."""
        import torch
        from Surrogate.nn_model import NNSurrogateModel, ZScoreNormalizer

        model = self._make_small_nn(synthetic_training_data)

        # Test points
        test_params = synthetic_training_data["parameters"][:10]
        pred_before = model.predict_batch(test_params)

        # Compute new normalizers from slightly perturbed data
        rng = np.random.default_rng(123)
        perturbed = synthetic_training_data.copy()
        extra = rng.uniform(size=(5, 4))
        extra[:, 0] *= 1e-3
        extra[:, 1] *= 1e-4
        extra[:, 2] = 0.3 + 0.4 * extra[:, 2]
        extra[:, 3] = 0.3 + 0.4 * extra[:, 3]
        all_params = np.vstack([
            synthetic_training_data["parameters"], extra,
        ])

        X_log = all_params.copy().astype(np.float64)
        X_log[:, 0] = np.log10(np.maximum(X_log[:, 0], 1e-30))
        X_log[:, 1] = np.log10(np.maximum(X_log[:, 1], 1e-30))
        new_input_norm = ZScoreNormalizer.from_data(X_log)

        all_cd = np.vstack([
            synthetic_training_data["current_density"],
            rng.standard_normal((5, 22)) * 0.1,
        ])
        all_pc = np.vstack([
            synthetic_training_data["peroxide_current"],
            rng.standard_normal((5, 22)) * 0.01,
        ])
        Y = np.concatenate([all_cd, all_pc], axis=1)
        new_output_norm = ZScoreNormalizer.from_data(Y)

        # Correct weights
        old_state = {
            k: v.cpu().clone() for k, v in model._model.state_dict().items()
        }
        corrected_state = _correct_weights_for_normalizer_shift(
            old_state,
            model._input_normalizer,
            new_input_norm,
            model._output_normalizer,
            new_output_norm,
        )

        # Build new model with corrected weights and new normalizers
        new_model = NNSurrogateModel(
            hidden=model._hidden,
            n_blocks=model._n_blocks,
            seed=model._seed,
        )
        new_model._n_eta = model._n_eta
        new_model._phi_applied = model._phi_applied.copy()
        new_model._input_normalizer = new_input_norm
        new_model._output_normalizer = new_output_norm
        new_model.training_bounds = model.training_bounds
        new_model._is_fitted = True

        from Surrogate.nn_model import ResNetMLP

        new_model._model = ResNetMLP(
            n_in=4,
            n_out=2 * model._n_eta,
            hidden=model._hidden,
            n_blocks=model._n_blocks,
        )
        new_model._model.load_state_dict(corrected_state)
        new_model._model.eval()

        pred_after = new_model.predict_batch(test_params)

        np.testing.assert_allclose(
            pred_after["current_density"],
            pred_before["current_density"],
            atol=1e-4,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            pred_after["peroxide_current"],
            pred_before["peroxide_current"],
            atol=1e-4,
            rtol=1e-4,
        )

    def test_nn_warm_start_save_load_roundtrip(
        self, synthetic_training_data, new_training_data,
    ):
        """Warm-start retrained model survives save/load round-trip."""
        from Surrogate.ismo_retrain import _retrain_single_nn
        from Surrogate.nn_model import NNSurrogateModel

        old_model = self._make_small_nn(synthetic_training_data)
        tmp_dir = tempfile.mkdtemp()
        config = ISMORetrainConfig(
            nn_retrain_epochs=20,
            nn_retrain_patience=10,
            output_base_dir=tmp_dir,
        )
        merged = merge_training_data(
            synthetic_training_data, new_training_data, config,
        )
        train_idx = np.arange(0, 40)
        test_idx = np.arange(40, 50)
        new_train, new_test = update_split_indices(
            train_idx, test_idx, merged.n_old, merged.n_new_valid,
        )

        retrained = _retrain_single_nn(
            old_model, merged, new_train, new_test, config, iteration=1,
        )

        # Save
        save_dir = os.path.join(tmp_dir, "roundtrip_test")
        retrained.save(save_dir)

        # Load
        loaded = NNSurrogateModel.load(save_dir)

        # Compare predictions
        test_params = synthetic_training_data["parameters"][:5]
        pred_retrained = retrained.predict_batch(test_params)
        pred_loaded = loaded.predict_batch(test_params)

        np.testing.assert_allclose(
            pred_loaded["current_density"],
            pred_retrained["current_density"],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            pred_loaded["peroxide_current"],
            pred_retrained["peroxide_current"],
            atol=1e-6,
        )

    def test_nn_ensemble_partial_failure_rollback(
        self, synthetic_training_data, new_training_data,
    ):
        """Failed member falls back to original; ensemble still returned."""
        from unittest.mock import patch, PropertyMock
        from Surrogate.ensemble import EnsembleMeanWrapper
        from Surrogate.ismo_retrain import retrain_nn_ensemble

        m0 = self._make_small_nn(synthetic_training_data, seed=0)
        m1 = self._make_small_nn(synthetic_training_data, seed=1)
        m2 = self._make_small_nn(synthetic_training_data, seed=2)
        ensemble = EnsembleMeanWrapper([m0, m1, m2])

        config = ISMORetrainConfig(
            nn_retrain_epochs=20,
            nn_retrain_patience=10,
            output_base_dir=tempfile.mkdtemp(),
        )
        merged = merge_training_data(
            synthetic_training_data, new_training_data, config,
        )
        train_idx = np.arange(0, 40)
        test_idx = np.arange(40, 50)
        new_train, new_test = update_split_indices(
            train_idx, test_idx, merged.n_old, merged.n_new_valid,
        )

        # Patch the second member's state_dict to raise an error
        original_state_dict = m1._model.state_dict

        call_count = [0]

        def mock_state_dict_for_m1(*args, **kwargs):
            # The retrain loop accesses state_dict on member._model
            # We want the second member (index 1) to fail
            call_count[0] += 1
            raise RuntimeError("Simulated failure for member 1")

        # Replace m1._model.state_dict method to force failure
        m1._model.state_dict = mock_state_dict_for_m1

        result = retrain_nn_ensemble(
            ensemble, merged, new_train, new_test, config, iteration=1,
        )

        # Restore
        m1._model.state_dict = original_state_dict

        assert isinstance(result, EnsembleMeanWrapper)
        assert len(result.models) == 3
        # The failed member should be the original m1
        assert result.models[1] is m1

    def test_warm_start_state_dict_in_fit(self, synthetic_training_data):
        """NNSurrogateModel.fit() accepts warm_start_state_dict."""
        import torch
        from Surrogate.nn_model import NNSurrogateModel

        model = self._make_small_nn(synthetic_training_data)
        state_dict = {
            k: v.cpu().clone() for k, v in model._model.state_dict().items()
        }

        # Create a new model and fit with warm start
        new_model = NNSurrogateModel(hidden=16, n_blocks=1, seed=99)
        new_model.fit(
            synthetic_training_data["parameters"],
            synthetic_training_data["current_density"],
            synthetic_training_data["peroxide_current"],
            synthetic_training_data["phi_applied"],
            epochs=10,
            patience=5,
            verbose=False,
            warm_start_state_dict=state_dict,
        )

        assert new_model.is_fitted
        result = new_model.predict_batch(
            synthetic_training_data["parameters"][:3],
        )
        assert result["current_density"].shape == (3, 22)


# ---------------------------------------------------------------------------
# Tests: config defaults
# ---------------------------------------------------------------------------


class TestConfig:

    def test_default_values(self):
        """ISMORetrainConfig has expected defaults."""
        config = ISMORetrainConfig()
        assert config.nn_retrain_epochs == 100
        assert config.nn_retrain_lr == 1e-4
        assert config.nn_retrain_patience == 50
        assert config.max_degradation_ratio == 1.10
        assert config.duplicate_param_tol == 1e-6
        assert config.quality_metric == "cd_mean_relative_error"

    def test_frozen(self):
        """ISMORetrainConfig is frozen (immutable)."""
        config = ISMORetrainConfig()
        with pytest.raises(AttributeError):
            config.nn_retrain_epochs = 999
