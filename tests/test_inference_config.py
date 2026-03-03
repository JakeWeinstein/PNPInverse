"""Unit tests for Inverse.inference_runner.config dataclasses.

These tests exercise pure-Python / NumPy logic and do NOT require Firedrake.
"""

from __future__ import annotations

import numpy as np
import pytest

from Inverse.inference_runner.config import (
    SyntheticData,
    InferenceRequest,
    RecoveryConfig,
    RecoveryAttempt,
    InferenceResult,
)


# ===================================================================
# SyntheticData
# ===================================================================

class TestSyntheticData:
    """Test SyntheticData construction and select_targets."""

    @pytest.fixture()
    def synth_data(self):
        clean_c = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
        clean_phi = np.array([0.1, 0.2, 0.3])
        noisy_c = [np.array([1.1, 2.1, 3.1]), np.array([4.1, 5.1, 6.1])]
        noisy_phi = np.array([0.11, 0.21, 0.31])
        return SyntheticData(
            clean_concentration_vectors=clean_c,
            clean_phi_vector=clean_phi,
            noisy_concentration_vectors=noisy_c,
            noisy_phi_vector=noisy_phi,
        )

    def test_construction(self, synth_data):
        assert len(synth_data.clean_concentration_vectors) == 2
        assert len(synth_data.noisy_concentration_vectors) == 2
        assert synth_data.clean_phi_vector.shape == (3,)
        assert synth_data.noisy_phi_vector.shape == (3,)

    def test_select_targets_clean(self, synth_data):
        c_list, phi = synth_data.select_targets(use_noisy_data=False)
        np.testing.assert_array_equal(c_list[0], np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(phi, np.array([0.1, 0.2, 0.3]))

    def test_select_targets_noisy(self, synth_data):
        c_list, phi = synth_data.select_targets(use_noisy_data=True)
        np.testing.assert_array_equal(c_list[0], np.array([1.1, 2.1, 3.1]))
        np.testing.assert_array_equal(phi, np.array([0.11, 0.21, 0.31]))

    def test_select_targets_returns_different_objects(self, synth_data):
        """Clean and noisy targets should be distinct arrays."""
        c_clean, phi_clean = synth_data.select_targets(use_noisy_data=False)
        c_noisy, phi_noisy = synth_data.select_targets(use_noisy_data=True)
        assert not np.array_equal(c_clean[0], c_noisy[0])
        assert not np.array_equal(phi_clean, phi_noisy)


# ===================================================================
# RecoveryConfig
# ===================================================================

class TestRecoveryConfig:
    """Test RecoveryConfig defaults and construction."""

    def test_default_values(self):
        cfg = RecoveryConfig()
        assert cfg.max_attempts == 15
        assert cfg.contraction_factor == pytest.approx(0.5)
        assert cfg.fallback_shrink_if_stuck == pytest.approx(0.15)
        assert cfg.max_it_only_attempts == 2
        assert cfg.anisotropy_only_attempts == 3
        assert cfg.tolerance_relax_attempts == 1
        assert cfg.anisotropy_target_ratio == pytest.approx(3.0)
        assert cfg.anisotropy_blend == pytest.approx(0.5)
        assert cfg.atol_relax_factor == pytest.approx(10.0)
        assert cfg.rtol_relax_factor == pytest.approx(10.0)
        assert cfg.ksp_rtol_relax_factor == pytest.approx(10.0)
        assert cfg.max_it_growth == pytest.approx(1.5)
        assert cfg.max_it_cap == 500
        assert cfg.line_search_schedule == ("bt", "l2", "cp", "basic")
        assert cfg.verbose is True

    def test_custom_values(self):
        cfg = RecoveryConfig(max_attempts=5, contraction_factor=0.3, verbose=False)
        assert cfg.max_attempts == 5
        assert cfg.contraction_factor == pytest.approx(0.3)
        assert cfg.verbose is False

    def test_line_search_schedule_is_tuple(self):
        cfg = RecoveryConfig()
        assert isinstance(cfg.line_search_schedule, tuple)


# ===================================================================
# InferenceRequest
# ===================================================================

class TestInferenceRequest:
    """Test InferenceRequest construction and defaults."""

    def _make_request(self, **overrides):
        defaults = dict(
            adapter=None,
            target=None,
            base_solver_params=[0] * 11,
            true_value=1.0,
            initial_guess=0.5,
        )
        defaults.update(overrides)
        return InferenceRequest(**defaults)

    def test_defaults(self):
        req = self._make_request()
        assert req.noise_percent == pytest.approx(10.0)
        assert req.seed is None
        assert req.optimizer_method == "L-BFGS-B"
        assert req.optimizer_options is None
        assert req.tolerance == pytest.approx(1e-8)
        assert req.bounds is None
        assert req.fit_to_noisy_data is True
        assert req.blob_initial_condition is True
        assert req.print_interval_data == 100
        assert req.print_interval_inverse == 100
        assert req.recovery_attempts is None

    def test_custom_values(self):
        req = self._make_request(
            noise_percent=5.0,
            seed=42,
            optimizer_method="Nelder-Mead",
            fit_to_noisy_data=False,
        )
        assert req.noise_percent == pytest.approx(5.0)
        assert req.seed == 42
        assert req.optimizer_method == "Nelder-Mead"
        assert req.fit_to_noisy_data is False

    def test_recovery_default_is_fresh_instance(self):
        req1 = self._make_request()
        req2 = self._make_request()
        # Each request should get its own RecoveryConfig instance
        assert req1.recovery is not req2.recovery
        assert isinstance(req1.recovery, RecoveryConfig)

    def test_recovery_custom(self):
        custom_recovery = RecoveryConfig(max_attempts=3, verbose=False)
        req = self._make_request(recovery=custom_recovery)
        assert req.recovery.max_attempts == 3
        assert req.recovery.verbose is False


# ===================================================================
# RecoveryAttempt
# ===================================================================

class TestRecoveryAttempt:
    """Test RecoveryAttempt dataclass."""

    def test_construction(self):
        att = RecoveryAttempt(
            attempt_index=0,
            phase="initial",
            solver_options={"snes_max_it": 50},
            start_guess=0.5,
            status="success",
            reason="converged",
        )
        assert att.attempt_index == 0
        assert att.phase == "initial"
        assert att.status == "success"
        assert att.best_objective_seen is None
        assert att.best_estimate_seen is None

    def test_optional_fields(self):
        att = RecoveryAttempt(
            attempt_index=2,
            phase="max_it",
            solver_options={},
            start_guess=[0.5, 0.6],
            status="failed",
            reason="SNES diverged",
            best_objective_seen=0.123,
            best_estimate_seen=[0.7, 0.8],
        )
        assert att.best_objective_seen == pytest.approx(0.123)
        assert att.best_estimate_seen == [0.7, 0.8]


# ===================================================================
# InferenceResult
# ===================================================================

class TestInferenceResult:
    """Test InferenceResult dataclass construction."""

    def test_construction(self):
        synth = SyntheticData(
            clean_concentration_vectors=[np.zeros(5)],
            clean_phi_vector=np.zeros(5),
            noisy_concentration_vectors=[np.ones(5) * 0.1],
            noisy_phi_vector=np.ones(5) * 0.01,
        )
        result = InferenceResult(
            target_key="diffusion",
            estimate=1.23,
            objective_value=0.001,
            true_solver_params=[0] * 11,
            inverse_solver_params=[0] * 11,
            synthetic_data=synth,
            optimized_controls=1.23,
            recovery_attempts=[],
        )
        assert result.target_key == "diffusion"
        assert result.estimate == pytest.approx(1.23)
        assert result.objective_value == pytest.approx(0.001)
        assert isinstance(result.synthetic_data, SyntheticData)
        assert result.recovery_attempts == []
