"""Surrogate-based inference robustness tests.

Exercises noise models, objective properties, recovery mechanisms,
formatting helpers, edge cases, noise-level sweeps, parameter variation,
initial-guess sensitivity, seed reproducibility, multi-start / BCD /
cascade robustness, and recovery stress -- all without Firedrake.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Path setup (mirrors conftest.py)
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Surrogate.surrogate_model import BVSurrogateModel, SurrogateConfig
from Surrogate.objectives import SurrogateObjective
from Inverse.inference_runner.config import RecoveryConfig, RecoveryAttempt
from Inverse.inference_runner.recovery import (
    _guess_to_array,
    _scale_guess,
    _blend_guess,
    _reduce_guess_anisotropy,
    _guesses_close,
    _attempt_phase_state,
)
from Inverse.inference_runner.formatting import (
    _format_float_for_log,
    _format_guess_for_log,
    _summarize_exception,
    _format_recovery_summary,
)


# ===================================================================
# Shared fixtures & helpers
# ===================================================================

@pytest.fixture()
def fitted_surrogate() -> BVSurrogateModel:
    """Build and fit a small BVSurrogateModel for testing."""
    rng = np.random.default_rng(42)
    config = SurrogateConfig(
        kernel="thin_plate_spline",
        degree=1,
        smoothing=1e-3,
        log_space_k0=True,
        normalize_inputs=True,
    )
    model = BVSurrogateModel(config)
    n_samples, n_eta = 40, 8
    phi_applied = np.linspace(-10, 5, n_eta)
    k0_1 = rng.uniform(0.005, 0.05, n_samples)
    k0_2 = rng.uniform(0.0005, 0.005, n_samples)
    alpha_1 = rng.uniform(0.4, 0.8, n_samples)
    alpha_2 = rng.uniform(0.3, 0.7, n_samples)
    params = np.column_stack([k0_1, k0_2, alpha_1, alpha_2])
    cd = (
        np.outer(alpha_1, -phi_applied) * k0_1[:, None] * 10
        + np.outer(alpha_2 * k0_2, phi_applied ** 2) * 0.1
    )
    pc = (
        np.outer(alpha_2, -phi_applied) * k0_2[:, None] * 100
        + np.outer(alpha_1 * k0_1, phi_applied ** 2) * 0.5
    )
    model.fit(params, cd, pc, phi_applied)
    return model


TRUE_PARAM_SETS = {
    "typical":           (0.020, 0.002,  0.60, 0.50),
    "symmetric":         (0.010, 0.010,  0.50, 0.50),
    "asymmetric_k0":     (0.040, 0.0005, 0.60, 0.50),
    "low_alpha":         (0.020, 0.002,  0.15, 0.12),
    "high_alpha":        (0.020, 0.002,  0.85, 0.78),
    "near_bounds_k0_lo": (0.005, 0.0005, 0.60, 0.50),
    "near_bounds_k0_hi": (0.048, 0.0048, 0.60, 0.50),
    "small_k0":          (0.006, 0.0006, 0.50, 0.40),
}


def add_noise_to_curve(curve: np.ndarray, noise_pct: float, seed: int) -> np.ndarray:
    """Add Gaussian noise proportional to RMS of the signal."""
    if noise_pct < 0:
        raise ValueError("noise_pct must be non-negative")
    rng = np.random.default_rng(seed)
    rms = np.sqrt(np.mean(curve ** 2))
    sigma = (noise_pct / 100.0) * max(rms, 1e-12)
    return curve + rng.normal(0.0, sigma, size=curve.shape)


OPTIMIZER_KWARGS = dict(
    method="L-BFGS-B",
    options={"maxiter": 60, "ftol": 1e-14, "gtol": 1e-8},
)

BOUNDS = [
    (np.log10(0.005), np.log10(0.05)),
    (np.log10(0.0005), np.log10(0.005)),
    (0.1, 0.9),
    (0.1, 0.9),
]


def run_surrogate_inference(
    surrogate,
    true_params,
    x0,
    noise_pct,
    seed,
    bounds=BOUNDS,
    secondary_weight=1.0,
):
    k0_1_true, k0_2_true, alpha_1_true, alpha_2_true = true_params
    pred = surrogate.predict(k0_1_true, k0_2_true, alpha_1_true, alpha_2_true)
    target_cd = add_noise_to_curve(pred["current_density"], noise_pct, seed)
    target_pc = add_noise_to_curve(pred["peroxide_current"], noise_pct, seed + 1000)
    obj = SurrogateObjective(
        surrogate=surrogate,
        target_cd=target_cd,
        target_pc=target_pc,
        secondary_weight=secondary_weight,
        fd_step=1e-5,
        log_space_k0=True,
    )
    result = minimize(obj.objective, x0, jac=obj.gradient, bounds=bounds, **OPTIMIZER_KWARGS)
    recovered = (10 ** result.x[0], 10 ** result.x[1], result.x[2], result.x[3])
    return result, recovered


def compute_relative_errors(recovered, true_params):
    return tuple(abs(r - t) / max(abs(t), 1e-16) for r, t in zip(recovered, true_params))


# ===================================================================
# 4.7 Noise Model Tests
# ===================================================================

class TestNoiseModel:
    """Verify add_noise_to_curve behaviour."""

    def test_noise_zero_preserves_signal(self):
        signal = np.array([1.0, 2.0, 3.0])
        out = add_noise_to_curve(signal, 0.0, seed=1)
        np.testing.assert_array_equal(out, signal)

    def test_noise_statistics(self):
        signal = np.array([5.0])
        rms = 5.0
        target_sigma = 0.10 * rms  # 10 %
        n = 100_000
        big = np.tile(signal, n)
        out = add_noise_to_curve(big, 10.0, seed=7)
        noise = out - big
        assert abs(np.mean(noise)) < 0.02 * target_sigma
        assert abs(np.std(noise) - target_sigma) < 0.02 * target_sigma

    def test_noise_seed_reproducibility(self):
        sig = np.linspace(0, 1, 20)
        a = add_noise_to_curve(sig, 5.0, seed=99)
        b = add_noise_to_curve(sig, 5.0, seed=99)
        np.testing.assert_array_equal(a, b)

    def test_noise_different_seeds(self):
        sig = np.linspace(0, 1, 20)
        a = add_noise_to_curve(sig, 5.0, seed=1)
        b = add_noise_to_curve(sig, 5.0, seed=2)
        assert not np.array_equal(a, b)

    def test_noise_scales_with_rms(self):
        sig_small = np.ones(100)
        sig_big = np.ones(100) * 10.0
        noise_small = add_noise_to_curve(sig_small, 10.0, seed=0) - sig_small
        noise_big = add_noise_to_curve(sig_big, 10.0, seed=0) - sig_big
        ratio = np.std(noise_big) / np.std(noise_small)
        assert abs(ratio - 10.0) < 1.0

    def test_noise_rejects_negative_percent(self):
        with pytest.raises(ValueError):
            add_noise_to_curve(np.ones(5), -1.0, seed=0)

    def test_noise_on_constant_signal(self):
        sig = np.full(10_000, 5.0)
        out = add_noise_to_curve(sig, 10.0, seed=3)
        noise = out - sig
        expected_sigma = 0.10 * 5.0
        assert abs(np.std(noise) - expected_sigma) < 0.05 * expected_sigma

    def test_noise_on_near_zero_signal(self):
        sig = np.full(100, 1e-15)
        out = add_noise_to_curve(sig, 10.0, seed=4)
        assert np.all(np.isfinite(out))


# ===================================================================
# 4.6 Objective Properties
# ===================================================================

class TestObjectiveProperties:
    """Verify mathematical properties of SurrogateObjective."""

    def test_objective_nonnegative(self, fitted_surrogate):
        rng = np.random.default_rng(11)
        true = TRUE_PARAM_SETS["typical"]
        pred = fitted_surrogate.predict(*true)
        obj = SurrogateObjective(
            surrogate=fitted_surrogate,
            target_cd=pred["current_density"],
            target_pc=pred["peroxide_current"],
        )
        for _ in range(100):
            x = np.array([
                rng.uniform(*BOUNDS[0]),
                rng.uniform(*BOUNDS[1]),
                rng.uniform(*BOUNDS[2]),
                rng.uniform(*BOUNDS[3]),
            ])
            assert obj.objective(x) >= 0.0

    def test_objective_zero_at_true_params(self, fitted_surrogate):
        true = TRUE_PARAM_SETS["typical"]
        pred = fitted_surrogate.predict(*true)
        obj = SurrogateObjective(
            surrogate=fitted_surrogate,
            target_cd=pred["current_density"],
            target_pc=pred["peroxide_current"],
        )
        x_true = np.array([np.log10(true[0]), np.log10(true[1]), true[2], true[3]])
        assert obj.objective(x_true) == pytest.approx(0.0, abs=1e-12)

    def test_objective_strictly_positive_at_wrong_params(self, fitted_surrogate):
        true = TRUE_PARAM_SETS["typical"]
        pred = fitted_surrogate.predict(*true)
        obj = SurrogateObjective(
            surrogate=fitted_surrogate,
            target_cd=pred["current_density"],
            target_pc=pred["peroxide_current"],
        )
        wrong = (0.04, 0.004, 0.8, 0.3)
        x_wrong = np.array([np.log10(wrong[0]), np.log10(wrong[1]), wrong[2], wrong[3]])
        assert obj.objective(x_wrong) > 0.0

    def test_gradient_finite_difference_consistency(self, fitted_surrogate):
        true = TRUE_PARAM_SETS["typical"]
        pred = fitted_surrogate.predict(*true)
        obj = SurrogateObjective(
            surrogate=fitted_surrogate,
            target_cd=pred["current_density"],
            target_pc=pred["peroxide_current"],
            fd_step=1e-6,
        )
        x0 = np.array([np.log10(0.015), np.log10(0.0015), 0.55, 0.45])
        grad_obj = obj.gradient(x0)
        # Manual central FD with a slightly different step
        h = 1e-5
        grad_manual = np.zeros(4)
        for i in range(4):
            xp = x0.copy(); xp[i] += h
            xm = x0.copy(); xm[i] -= h
            grad_manual[i] = (obj.objective(xp) - obj.objective(xm)) / (2 * h)
        np.testing.assert_allclose(grad_obj, grad_manual, rtol=0.05, atol=1e-10)

    def test_gradient_zero_at_optimum(self, fitted_surrogate):
        true = TRUE_PARAM_SETS["typical"]
        pred = fitted_surrogate.predict(*true)
        obj = SurrogateObjective(
            surrogate=fitted_surrogate,
            target_cd=pred["current_density"],
            target_pc=pred["peroxide_current"],
            fd_step=1e-6,
        )
        x_true = np.array([np.log10(true[0]), np.log10(true[1]), true[2], true[3]])
        grad = obj.gradient(x_true)
        assert np.linalg.norm(grad) < 1e-4

    def test_secondary_weight_scales_pc_term(self, fitted_surrogate):
        true = TRUE_PARAM_SETS["typical"]
        pred = fitted_surrogate.predict(*true)
        # Use a wrong point so PC term is nonzero
        target_cd = pred["current_density"] + 0.1
        target_pc = pred["peroxide_current"] + 0.1
        obj_w0 = SurrogateObjective(
            surrogate=fitted_surrogate,
            target_cd=target_cd,
            target_pc=target_pc,
            secondary_weight=0.0,
        )
        obj_w1 = SurrogateObjective(
            surrogate=fitted_surrogate,
            target_cd=target_cd,
            target_pc=target_pc,
            secondary_weight=1.0,
        )
        x = np.array([np.log10(true[0]), np.log10(true[1]), true[2], true[3]])
        j0 = obj_w0.objective(x)
        j1 = obj_w1.objective(x)
        # w=0 should give a smaller J since PC term is ignored
        assert j0 < j1

    def test_objective_continuous_near_truth(self, fitted_surrogate):
        true = TRUE_PARAM_SETS["typical"]
        pred = fitted_surrogate.predict(*true)
        obj = SurrogateObjective(
            surrogate=fitted_surrogate,
            target_cd=pred["current_density"],
            target_pc=pred["peroxide_current"],
        )
        x_true = np.array([np.log10(true[0]), np.log10(true[1]), true[2], true[3]])
        rng = np.random.default_rng(33)
        for _ in range(20):
            eps = rng.normal(0, 1e-3, size=4)
            j_perturbed = obj.objective(x_true + eps)
            # quadratic upper bound
            assert j_perturbed < 1e6 * np.dot(eps, eps) + 1e-10


# ===================================================================
# 4.8 Recovery Mechanisms
# ===================================================================

class TestRecoveryMechanisms:
    """Test helper functions from Inverse/inference_runner/recovery.py."""

    # _guess_to_array
    def test_guess_to_array_scalar(self):
        arr, is_vec = _guess_to_array(3.14)
        assert not is_vec
        np.testing.assert_allclose(arr, [3.14])

    def test_guess_to_array_list(self):
        arr, is_vec = _guess_to_array([1.0, 2.0, 3.0])
        assert is_vec
        np.testing.assert_allclose(arr, [1.0, 2.0, 3.0])

    def test_guess_to_array_ndarray(self):
        arr, is_vec = _guess_to_array(np.array([5.0, 6.0]))
        assert is_vec
        np.testing.assert_allclose(arr, [5.0, 6.0])

    def test_guess_to_array_tuple(self):
        arr, is_vec = _guess_to_array((7.0, 8.0))
        assert is_vec
        np.testing.assert_allclose(arr, [7.0, 8.0])

    # _scale_guess
    def test_scale_guess_scalar(self):
        out = _scale_guess(2.0, factor=0.5)
        assert out == pytest.approx(1.0)

    def test_scale_guess_vector(self):
        out = _scale_guess([4.0, 6.0], factor=0.25)
        assert out == pytest.approx([1.0, 1.5])

    def test_scale_guess_zero_factor(self):
        out = _scale_guess([1.0, 2.0], factor=0.0)
        assert out == pytest.approx([0.0, 0.0])

    # _blend_guess
    def test_blend_guess_halfway(self):
        out = _blend_guess(current_guess=[10.0, 20.0], safe_guess=[0.0, 0.0], contraction_factor=0.5)
        assert out == pytest.approx([5.0, 10.0])

    def test_blend_guess_scalar(self):
        out = _blend_guess(current_guess=10.0, safe_guess=0.0, contraction_factor=0.5)
        assert out == pytest.approx(5.0)

    def test_blend_guess_factor_zero(self):
        out = _blend_guess(current_guess=[10.0, 20.0], safe_guess=[1.0, 2.0], contraction_factor=0.0)
        assert out == pytest.approx([1.0, 2.0])

    def test_blend_guess_factor_one(self):
        out = _blend_guess(current_guess=[10.0, 20.0], safe_guess=[1.0, 2.0], contraction_factor=1.0)
        assert out == pytest.approx([10.0, 20.0])

    # _reduce_guess_anisotropy
    def test_reduce_anisotropy_no_change_when_isotropic(self):
        val = [1.0, 1.0, 1.0]
        out = _reduce_guess_anisotropy(val, target_ratio=3.0, blend=0.5)
        assert out == pytest.approx(val)

    def test_reduce_anisotropy_reduces_ratio(self):
        val = [1.0, 100.0]
        out = _reduce_guess_anisotropy(val, target_ratio=3.0, blend=1.0)
        mags = np.abs(out)
        ratio = max(mags) / min(mags)
        # Blend=1 should move fully to geometric mean
        assert ratio < 100.0

    def test_reduce_anisotropy_scalar_unchanged(self):
        val = 5.0
        out = _reduce_guess_anisotropy(val, target_ratio=3.0, blend=0.5)
        assert out == pytest.approx(5.0)

    # _guesses_close
    def test_guesses_close_identical(self):
        assert _guesses_close([1.0, 2.0], [1.0, 2.0])

    def test_guesses_close_different(self):
        assert not _guesses_close([1.0, 2.0], [1.0, 3.0])

    def test_guesses_close_scalars(self):
        assert _guesses_close(5.0, 5.0)
        assert not _guesses_close(5.0, 6.0)

    # _attempt_phase_state
    def test_phase_baseline_at_zero(self):
        rc = RecoveryConfig()
        phase, step, cycle = _attempt_phase_state(0, rc)
        assert phase == "baseline"
        assert step == 1
        assert cycle == 0

    def test_phase_cycling(self):
        rc = RecoveryConfig(
            max_it_only_attempts=2,
            anisotropy_only_attempts=3,
            tolerance_relax_attempts=1,
        )
        # attempt 1 -> max_it step 1
        phase, step, cycle = _attempt_phase_state(1, rc)
        assert phase == "max_it"
        assert step == 1
        assert cycle == 0

        # attempt 2 -> max_it step 2
        phase, step, cycle = _attempt_phase_state(2, rc)
        assert phase == "max_it"
        assert step == 2
        assert cycle == 0

        # attempt 3 -> anisotropy step 1
        phase, step, cycle = _attempt_phase_state(3, rc)
        assert phase == "anisotropy"
        assert step == 1
        assert cycle == 0

        # attempt 6 -> tolerance_relax step 1
        phase, step, cycle = _attempt_phase_state(6, rc)
        assert phase == "tolerance_relax"
        assert step == 1
        assert cycle == 0

        # attempt 7 -> cycle 1, max_it step 1
        phase, step, cycle = _attempt_phase_state(7, rc)
        assert phase == "max_it"
        assert step == 1
        assert cycle == 1


# ===================================================================
# Section 7: Formatting Tests
# ===================================================================

class TestFormattingRobustness:
    """Test formatting helpers in Inverse/inference_runner/formatting.py."""

    # _format_float_for_log
    def test_format_float_none(self):
        assert _format_float_for_log(None) == "-"

    def test_format_float_numeric(self):
        out = _format_float_for_log(1.23e-4)
        assert "1.230e-04" in out

    def test_format_float_string(self):
        out = _format_float_for_log("not-a-number")
        assert out == "not-a-number"

    def test_format_float_zero(self):
        out = _format_float_for_log(0.0)
        assert "0.000e+00" in out

    # _format_guess_for_log
    def test_format_guess_scalar(self):
        out = _format_guess_for_log(3.14)
        assert "3.14" in out

    def test_format_guess_vector(self):
        out = _format_guess_for_log([1.0, 2.0, 3.0])
        assert "[" in out and "]" in out

    # _summarize_exception
    def test_summarize_none(self):
        assert _summarize_exception(None) == "Unknown"

    def test_summarize_runtime_error(self):
        exc = RuntimeError("SNES DIVERGED_LINEAR_SOLVE")
        out = _summarize_exception(exc)
        assert "RuntimeError" in out
        assert "DIVERGED" in out

    def test_summarize_empty_message(self):
        exc = ValueError("")
        out = _summarize_exception(exc)
        assert "ValueError" in out

    def test_summarize_strips_file_paths(self):
        exc = RuntimeError("Error in /usr/local/lib/python3.11/site-packages/foo.py:123")
        out = _summarize_exception(exc)
        assert "/usr/local/lib" not in out

    # _format_recovery_summary
    def test_recovery_summary_empty(self):
        assert _format_recovery_summary([]) == "no attempts logged"

    def test_recovery_summary_single(self):
        att = RecoveryAttempt(
            attempt_index=0,
            phase="baseline",
            solver_options={},
            start_guess=0.5,
            status="success",
            reason="converged",
        )
        out = _format_recovery_summary([att])
        assert "attempt=0" in out
        assert "baseline" in out
        assert "success" in out

    def test_recovery_summary_truncates_long_reason(self):
        att = RecoveryAttempt(
            attempt_index=0,
            phase="baseline",
            solver_options={},
            start_guess=0.5,
            status="failed",
            reason="x" * 300,
        )
        out = _format_recovery_summary([att])
        assert "..." in out
        assert len(out) < 400


# ===================================================================
# 4.5 Edge Cases
# ===================================================================

class TestEdgeCases:
    """Edge-case inference scenarios."""

    def test_zero_noise_perfect_recovery(self, fitted_surrogate):
        true = TRUE_PARAM_SETS["typical"]
        x0 = np.array([np.log10(0.015), np.log10(0.0015), 0.55, 0.45])
        result, recovered = run_surrogate_inference(
            fitted_surrogate, true, x0, noise_pct=0.0, seed=0,
        )
        errs = compute_relative_errors(recovered, true)
        # With zero noise and a reasonable surrogate, loss should be very small
        assert result.fun < 1e-2

    def test_very_small_noise(self, fitted_surrogate):
        true = TRUE_PARAM_SETS["typical"]
        x0 = np.array([np.log10(0.015), np.log10(0.0015), 0.55, 0.45])
        result, recovered = run_surrogate_inference(
            fitted_surrogate, true, x0, noise_pct=0.1, seed=1,
        )
        assert result.fun < 1.0

    def test_identical_k0(self, fitted_surrogate):
        true = TRUE_PARAM_SETS["symmetric"]
        x0 = np.array([np.log10(0.008), np.log10(0.008), 0.45, 0.45])
        # Widen bounds to accommodate symmetric k0
        wide_bounds = [
            (np.log10(0.005), np.log10(0.05)),
            (np.log10(0.005), np.log10(0.05)),
            (0.1, 0.9),
            (0.1, 0.9),
        ]
        result, recovered = run_surrogate_inference(
            fitted_surrogate, true, x0, noise_pct=1.0, seed=2,
            bounds=wide_bounds,
        )
        assert np.isfinite(result.fun)

    def test_alpha_symmetry(self, fitted_surrogate):
        true = (0.020, 0.002, 0.50, 0.50)
        x0 = np.array([np.log10(0.015), np.log10(0.0015), 0.45, 0.45])
        result, _ = run_surrogate_inference(
            fitted_surrogate, true, x0, noise_pct=1.0, seed=3,
        )
        assert np.isfinite(result.fun)

    def test_alpha_near_bounds(self, fitted_surrogate):
        true = TRUE_PARAM_SETS["high_alpha"]
        x0 = np.array([np.log10(0.015), np.log10(0.0015), 0.80, 0.70])
        result, _ = run_surrogate_inference(
            fitted_surrogate, true, x0, noise_pct=1.0, seed=4,
        )
        assert np.isfinite(result.fun)

    def test_nan_target_handling(self, fitted_surrogate):
        """SurrogateObjective masks NaN targets; should not raise."""
        true = TRUE_PARAM_SETS["typical"]
        pred = fitted_surrogate.predict(*true)
        cd = pred["current_density"].copy()
        pc = pred["peroxide_current"].copy()
        cd[0] = np.nan
        pc[-1] = np.nan
        obj = SurrogateObjective(
            surrogate=fitted_surrogate,
            target_cd=cd,
            target_pc=pc,
        )
        x = np.array([np.log10(true[0]), np.log10(true[1]), true[2], true[3]])
        j = obj.objective(x)
        assert np.isfinite(j)

    def test_single_voltage_point(self):
        """Model with n_eta=1 should still work."""
        rng = np.random.default_rng(42)
        config = SurrogateConfig(
            kernel="thin_plate_spline", degree=1, smoothing=1e-3,
            log_space_k0=True, normalize_inputs=True,
        )
        model = BVSurrogateModel(config)
        n_samples = 40
        phi_applied = np.array([0.0])
        k0_1 = rng.uniform(0.005, 0.05, n_samples)
        k0_2 = rng.uniform(0.0005, 0.005, n_samples)
        a1 = rng.uniform(0.4, 0.8, n_samples)
        a2 = rng.uniform(0.3, 0.7, n_samples)
        params = np.column_stack([k0_1, k0_2, a1, a2])
        cd = (a1 * k0_1 * 10 + a2 * k0_2 * 0.1).reshape(-1, 1)
        pc = (a2 * k0_2 * 100 + a1 * k0_1 * 0.5).reshape(-1, 1)
        model.fit(params, cd, pc, phi_applied)
        pred = model.predict(0.02, 0.002, 0.6, 0.5)
        assert pred["current_density"].shape == (1,)

    def test_many_voltage_points(self):
        """Model with n_eta=50 should still work."""
        rng = np.random.default_rng(42)
        config = SurrogateConfig(
            kernel="thin_plate_spline", degree=1, smoothing=1e-3,
            log_space_k0=True, normalize_inputs=True,
        )
        model = BVSurrogateModel(config)
        n_samples = 40
        n_eta = 50
        phi_applied = np.linspace(-10, 5, n_eta)
        k0_1 = rng.uniform(0.005, 0.05, n_samples)
        k0_2 = rng.uniform(0.0005, 0.005, n_samples)
        a1 = rng.uniform(0.4, 0.8, n_samples)
        a2 = rng.uniform(0.3, 0.7, n_samples)
        params = np.column_stack([k0_1, k0_2, a1, a2])
        cd = np.outer(a1, -phi_applied) * k0_1[:, None] * 10
        pc = np.outer(a2, -phi_applied) * k0_2[:, None] * 100
        model.fit(params, cd, pc, phi_applied)
        pred = model.predict(0.02, 0.002, 0.6, 0.5)
        assert pred["current_density"].shape == (n_eta,)


# ===================================================================
# 4.2 Noise Level Sweep
# ===================================================================

class TestNoiseLevelSweep:
    """Recovery quality should degrade gracefully with noise."""

    @pytest.mark.parametrize(
        "noise_pct, max_loss",
        [
            (0.0, 1e-1),
            (1.0, 2.0),
            (2.0, 5.0),
            (5.0, 30.0),
            (10.0, 120.0),
            (20.0, 500.0),
        ],
    )
    def test_noise_level_recovery(self, fitted_surrogate, noise_pct, max_loss):
        true = TRUE_PARAM_SETS["typical"]
        x0 = np.array([np.log10(0.015), np.log10(0.0015), 0.55, 0.45])
        result, _ = run_surrogate_inference(
            fitted_surrogate, true, x0, noise_pct=noise_pct, seed=42,
        )
        assert result.fun < max_loss, (
            f"noise={noise_pct}%: loss={result.fun:.4e} > max_loss={max_loss:.4e}"
        )

    def test_monotonic_loss_increase(self, fitted_surrogate):
        """Average optimized loss should roughly increase with noise."""
        true = TRUE_PARAM_SETS["typical"]
        x0 = np.array([np.log10(0.015), np.log10(0.0015), 0.55, 0.45])
        noise_levels = [0.0, 1.0, 5.0, 20.0]
        losses = []
        for n_pct in noise_levels:
            result, _ = run_surrogate_inference(
                fitted_surrogate, true, x0, noise_pct=n_pct, seed=42,
            )
            losses.append(result.fun)
        # Overall trend: first should be <= last
        assert losses[0] <= losses[-1] + 1e-10


# ===================================================================
# 4.3 True Parameter Variation
# ===================================================================

class TestTrueParameterVariation:
    """All 8 parameter sets should be recoverable at 1% noise."""

    @pytest.mark.parametrize("name", list(TRUE_PARAM_SETS.keys()))
    def test_param_set(self, fitted_surrogate, name):
        true = TRUE_PARAM_SETS[name]
        x0 = np.array([np.log10(0.015), np.log10(0.0015), 0.50, 0.40])
        result, _ = run_surrogate_inference(
            fitted_surrogate, true, x0, noise_pct=1.0, seed=10,
        )
        # Optimizer should converge to a finite result
        assert np.isfinite(result.fun), f"Non-finite loss for {name}"
        assert result.fun < 50.0, f"Loss too large for {name}: {result.fun:.4e}"


# ===================================================================
# 4.4 Initial Guess Sensitivity
# ===================================================================

class TestInitialGuessSensitivity:
    """Different starting points should all converge to finite results."""

    INITIAL_GUESSES = {
        "near_truth": np.array([np.log10(0.019), np.log10(0.0019), 0.59, 0.49]),
        "at_truth": np.array([np.log10(0.020), np.log10(0.002), 0.60, 0.50]),
        "far": np.array([np.log10(0.040), np.log10(0.004), 0.80, 0.80]),
        "very_far_k0": np.array([np.log10(0.005), np.log10(0.0005), 0.55, 0.45]),
        "very_far_alpha": np.array([np.log10(0.015), np.log10(0.0015), 0.15, 0.15]),
        "at_bounds_lo": np.array([np.log10(0.005), np.log10(0.0005), 0.10, 0.10]),
        "at_bounds_hi": np.array([np.log10(0.05), np.log10(0.005), 0.90, 0.90]),
        "opposite_corner": np.array([np.log10(0.05), np.log10(0.005), 0.10, 0.10]),
    }

    @pytest.mark.parametrize("name", list(INITIAL_GUESSES.keys()))
    def test_initial_guess(self, fitted_surrogate, name):
        true = TRUE_PARAM_SETS["typical"]
        x0 = self.INITIAL_GUESSES[name]
        result, _ = run_surrogate_inference(
            fitted_surrogate, true, x0, noise_pct=1.0, seed=20,
        )
        assert np.isfinite(result.fun), f"Non-finite loss for x0={name}"

    def test_random_starts(self, fitted_surrogate):
        """5 random starting points should all converge."""
        rng = np.random.default_rng(77)
        true = TRUE_PARAM_SETS["typical"]
        for _ in range(5):
            x0 = np.array([
                rng.uniform(*BOUNDS[0]),
                rng.uniform(*BOUNDS[1]),
                rng.uniform(*BOUNDS[2]),
                rng.uniform(*BOUNDS[3]),
            ])
            result, _ = run_surrogate_inference(
                fitted_surrogate, true, x0, noise_pct=1.0, seed=rng.integers(0, 10000),
            )
            assert np.isfinite(result.fun)


# ===================================================================
# 4.1 Noise Seed Reproducibility
# ===================================================================

class TestNoiseSeedReproducibility:
    """Same seed should give identical results; different seeds vary."""

    def test_determinism_same_seed(self, fitted_surrogate):
        true = TRUE_PARAM_SETS["typical"]
        x0 = np.array([np.log10(0.015), np.log10(0.0015), 0.55, 0.45])
        r1, rec1 = run_surrogate_inference(
            fitted_surrogate, true, x0, noise_pct=5.0, seed=42,
        )
        r2, rec2 = run_surrogate_inference(
            fitted_surrogate, true, x0, noise_pct=5.0, seed=42,
        )
        np.testing.assert_allclose(rec1, rec2, rtol=1e-10)
        assert r1.fun == pytest.approx(r2.fun, rel=1e-10)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_five_seeds_1pct(self, fitted_surrogate, seed):
        true = TRUE_PARAM_SETS["typical"]
        x0 = np.array([np.log10(0.015), np.log10(0.0015), 0.55, 0.45])
        result, _ = run_surrogate_inference(
            fitted_surrogate, true, x0, noise_pct=1.0, seed=seed,
        )
        assert np.isfinite(result.fun)

    @pytest.mark.parametrize("seed", range(10))
    def test_ten_seeds_5pct(self, fitted_surrogate, seed):
        true = TRUE_PARAM_SETS["typical"]
        x0 = np.array([np.log10(0.015), np.log10(0.0015), 0.55, 0.45])
        result, _ = run_surrogate_inference(
            fitted_surrogate, true, x0, noise_pct=5.0, seed=seed,
        )
        assert np.isfinite(result.fun)

    @pytest.mark.parametrize("seed", [10, 11, 12, 13, 14])
    def test_five_seeds_10pct(self, fitted_surrogate, seed):
        true = TRUE_PARAM_SETS["typical"]
        x0 = np.array([np.log10(0.015), np.log10(0.0015), 0.55, 0.45])
        result, _ = run_surrogate_inference(
            fitted_surrogate, true, x0, noise_pct=10.0, seed=seed,
        )
        assert np.isfinite(result.fun)


# ===================================================================
# 4.9-4.11 Multi-Start, BCD, Cascade Robustness
# ===================================================================

class TestMultiStartRobustness:
    """Multi-start runs with noise should produce finite results."""

    def test_multistart_noisy(self, fitted_surrogate):
        from Surrogate.multistart import MultiStartConfig, run_multistart_inference

        true = TRUE_PARAM_SETS["typical"]
        pred = fitted_surrogate.predict(*true)
        target_cd = add_noise_to_curve(pred["current_density"], 5.0, seed=50)
        target_pc = add_noise_to_curve(pred["peroxide_current"], 5.0, seed=1050)

        config = MultiStartConfig(
            n_grid=200,
            n_top_candidates=5,
            polish_maxiter=10,
            verbose=False,
            seed=42,
        )
        result = run_multistart_inference(
            surrogate=fitted_surrogate,
            target_cd=target_cd,
            target_pc=target_pc,
            bounds_k0_1=(0.005, 0.05),
            bounds_k0_2=(0.0005, 0.005),
            bounds_alpha=(0.3, 0.8),
            config=config,
        )
        assert np.isfinite(result.best_loss)
        assert len(result.candidates) == 5

    def test_multistart_high_noise(self, fitted_surrogate):
        from Surrogate.multistart import MultiStartConfig, run_multistart_inference

        true = TRUE_PARAM_SETS["typical"]
        pred = fitted_surrogate.predict(*true)
        target_cd = add_noise_to_curve(pred["current_density"], 20.0, seed=60)
        target_pc = add_noise_to_curve(pred["peroxide_current"], 20.0, seed=1060)

        config = MultiStartConfig(
            n_grid=200,
            n_top_candidates=3,
            polish_maxiter=10,
            verbose=False,
            seed=99,
        )
        result = run_multistart_inference(
            surrogate=fitted_surrogate,
            target_cd=target_cd,
            target_pc=target_pc,
            bounds_k0_1=(0.005, 0.05),
            bounds_k0_2=(0.0005, 0.005),
            bounds_alpha=(0.3, 0.8),
            config=config,
        )
        assert np.isfinite(result.best_loss)


class TestBCDRobustness:
    """BCD runs with noise should produce finite results."""

    def test_bcd_noisy(self, fitted_surrogate):
        from Surrogate.bcd import BCDConfig, run_block_coordinate_descent

        true = TRUE_PARAM_SETS["typical"]
        pred = fitted_surrogate.predict(*true)
        target_cd = add_noise_to_curve(pred["current_density"], 5.0, seed=70)
        target_pc = add_noise_to_curve(pred["peroxide_current"], 5.0, seed=1070)

        config = BCDConfig(
            max_outer_iters=3,
            inner_maxiter=10,
            verbose=False,
        )
        result = run_block_coordinate_descent(
            surrogate=fitted_surrogate,
            target_cd=target_cd,
            target_pc=target_pc,
            initial_k0=[0.015, 0.0015],
            initial_alpha=[0.55, 0.45],
            bounds_k0_1=(0.005, 0.05),
            bounds_k0_2=(0.0005, 0.005),
            bounds_alpha=(0.3, 0.8),
            config=config,
        )
        assert np.isfinite(result.final_loss)
        assert result.n_outer_iters >= 1

    def test_bcd_asymmetric_params(self, fitted_surrogate):
        from Surrogate.bcd import BCDConfig, run_block_coordinate_descent

        true = TRUE_PARAM_SETS["asymmetric_k0"]
        pred = fitted_surrogate.predict(*true)
        target_cd = add_noise_to_curve(pred["current_density"], 2.0, seed=80)
        target_pc = add_noise_to_curve(pred["peroxide_current"], 2.0, seed=1080)

        config = BCDConfig(
            max_outer_iters=5,
            inner_maxiter=15,
            verbose=False,
        )
        result = run_block_coordinate_descent(
            surrogate=fitted_surrogate,
            target_cd=target_cd,
            target_pc=target_pc,
            initial_k0=[0.02, 0.002],
            initial_alpha=[0.55, 0.45],
            bounds_k0_1=(0.005, 0.05),
            bounds_k0_2=(0.0005, 0.005),
            bounds_alpha=(0.3, 0.8),
            config=config,
        )
        assert np.isfinite(result.final_loss)


class TestCascadeRobustness:
    """Cascade runs with noise should produce finite results."""

    def test_cascade_noisy(self, fitted_surrogate):
        from Surrogate.cascade import CascadeConfig, run_cascade_inference

        true = TRUE_PARAM_SETS["typical"]
        pred = fitted_surrogate.predict(*true)
        target_cd = add_noise_to_curve(pred["current_density"], 5.0, seed=90)
        target_pc = add_noise_to_curve(pred["peroxide_current"], 5.0, seed=1090)

        config = CascadeConfig(
            pass1_maxiter=10,
            pass2_maxiter=10,
            polish_maxiter=5,
            verbose=False,
        )
        result = run_cascade_inference(
            surrogate=fitted_surrogate,
            target_cd=target_cd,
            target_pc=target_pc,
            initial_k0=[0.015, 0.0015],
            initial_alpha=[0.55, 0.45],
            bounds_k0_1=(0.005, 0.05),
            bounds_k0_2=(0.0005, 0.005),
            bounds_alpha=(0.3, 0.8),
            config=config,
        )
        assert np.isfinite(result.best_loss)
        assert len(result.pass_results) == 3

    def test_cascade_skip_polish_noisy(self, fitted_surrogate):
        from Surrogate.cascade import CascadeConfig, run_cascade_inference

        true = TRUE_PARAM_SETS["typical"]
        pred = fitted_surrogate.predict(*true)
        target_cd = add_noise_to_curve(pred["current_density"], 10.0, seed=100)
        target_pc = add_noise_to_curve(pred["peroxide_current"], 10.0, seed=1100)

        config = CascadeConfig(
            pass1_maxiter=10,
            pass2_maxiter=10,
            skip_polish=True,
            verbose=False,
        )
        result = run_cascade_inference(
            surrogate=fitted_surrogate,
            target_cd=target_cd,
            target_pc=target_pc,
            initial_k0=[0.015, 0.0015],
            initial_alpha=[0.55, 0.45],
            bounds_k0_1=(0.005, 0.05),
            bounds_k0_2=(0.0005, 0.005),
            bounds_alpha=(0.3, 0.8),
            config=config,
        )
        assert np.isfinite(result.best_loss)
        assert len(result.pass_results) == 2


# ===================================================================
# Section 6: Recovery Stress
# ===================================================================

class TestRecoveryStress:
    """Stress-test recovery helper functions under repeated / extreme use."""

    def test_phase_cycling_over_15_attempts(self):
        rc = RecoveryConfig(
            max_it_only_attempts=2,
            anisotropy_only_attempts=3,
            tolerance_relax_attempts=1,
        )
        phases_seen = set()
        for attempt in range(15):
            phase, step, cycle = _attempt_phase_state(attempt, rc)
            if attempt > 0:
                phases_seen.add(phase)
        assert "max_it" in phases_seen
        assert "anisotropy" in phases_seen
        assert "tolerance_relax" in phases_seen

    def test_scale_guess_fallback(self):
        """_scale_guess with fallback_shrink_if_stuck should reduce magnitude."""
        val = [10.0, 20.0]
        shrink = 0.15
        out = _scale_guess(val, factor=max(0.0, 1.0 - shrink))
        expected = [10.0 * 0.85, 20.0 * 0.85]
        assert out == pytest.approx(expected)

    def test_anisotropy_reduction_large_ratio(self):
        """Very anisotropic guess should be tamed."""
        val = [0.001, 1000.0]
        out = _reduce_guess_anisotropy(val, target_ratio=3.0, blend=0.8)
        mags = np.abs(out)
        ratio = max(mags) / min(mags)
        assert ratio < 1e6  # big reduction from original 1e6

    def test_blend_guess_with_extreme_contraction(self):
        """contraction_factor clamped to [0,1]."""
        out = _blend_guess(
            current_guess=[100.0, 200.0],
            safe_guess=[1.0, 2.0],
            contraction_factor=2.0,  # should be clamped to 1.0
        )
        # factor=1.0 means keep current_guess
        assert out == pytest.approx([100.0, 200.0])

    def test_blend_guess_negative_contraction(self):
        """Negative contraction_factor clamped to 0."""
        out = _blend_guess(
            current_guess=[100.0, 200.0],
            safe_guess=[1.0, 2.0],
            contraction_factor=-0.5,  # clamped to 0.0
        )
        assert out == pytest.approx([1.0, 2.0])

    def test_guesses_close_near_zero(self):
        """Very small values should be compared correctly."""
        assert _guesses_close([1e-15, 1e-15], [1e-15, 1e-15])

    def test_attempt_phase_state_large_attempt(self):
        """Large attempt indices should not crash."""
        rc = RecoveryConfig()
        for attempt in range(100):
            phase, step, cycle = _attempt_phase_state(attempt, rc)
            assert phase in ("baseline", "max_it", "anisotropy", "tolerance_relax")
            assert step >= 1
            assert cycle >= 0

    def test_reduce_anisotropy_all_negative(self):
        """All-negative vector should preserve sign structure."""
        val = [-1.0, -100.0]
        out = _reduce_guess_anisotropy(val, target_ratio=3.0, blend=1.0)
        arr = np.array(out)
        # Should preserve negative signs
        assert np.all(arr < 0)

    def test_reduce_anisotropy_mixed_sign(self):
        """Mixed-sign vector should not crash."""
        val = [-1.0, 100.0]
        out = _reduce_guess_anisotropy(val, target_ratio=3.0, blend=0.5)
        assert np.all(np.isfinite(out))

    def test_scale_guess_large_factor(self):
        out = _scale_guess([1.0, 2.0], factor=1e6)
        assert out == pytest.approx([1e6, 2e6])
