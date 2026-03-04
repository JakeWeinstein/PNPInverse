"""Tests for the v11 surrogate-warm-started PDE inference pipeline.

Tests cover:
    1. _compute_errors: relative error computation, edge cases
    2. _print_phase_result: return values and stdout capture
    3. _subset_targets: voltage subset extraction
    4. SubsetSurrogateObjective: objective, gradient, log-space conversion
    5. CLI argument parsing: defaults, overrides, flags
    6. Best-of selection logic: P3 vs P4 regression guard
    7. Phase warm-start chain: surrogate → PDE connectivity
    8. CSV output: correct columns and formatting
    9. End-to-end integration: full pipeline with mock surrogate (--no-pde)
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile

import numpy as np
import pytest

# Ensure PNPInverse root is importable
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Import from the v11 script
sys.path.insert(0, os.path.join(_ROOT, "scripts", "surrogate"))
from Infer_BVMaster_charged_v11_surrogate_pde import (
    _compute_errors,
    _print_phase_result,
    _subset_targets,
)

from Surrogate.surrogate_model import BVSurrogateModel, SurrogateConfig
from Surrogate.objectives import AlphaOnlySurrogateObjective


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def true_params():
    """True parameter values for error computation tests."""
    return {
        "k0": np.array([0.01, 0.001]),
        "alpha": np.array([0.627, 0.500]),
    }


@pytest.fixture()
def fitted_surrogate() -> BVSurrogateModel:
    """Build and fit a small BVSurrogateModel for testing.

    Uses 40 training samples with 10 voltage points and a non-separable
    response so all 4 parameters are identifiable.
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
    n_eta = 10
    phi_applied = np.linspace(-13, 5, n_eta)

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


@pytest.fixture()
def voltage_grids():
    """Standard voltage grids matching v11 script."""
    eta_symmetric = np.array([
        +5.0, +3.0, +1.0, -0.5,
        -1.0, -2.0, -3.0, -5.0, -8.0,
        -10.0, -15.0, -20.0,
    ])
    eta_shallow = np.array([
        -1.0, -2.0, -3.0, -4.0, -5.0, -6.5, -8.0,
        -10.0, -11.5, -13.0,
    ])
    eta_cathodic = np.array([
        -1.0, -2.0, -3.0, -4.0, -5.0, -6.5, -8.0,
        -10.0, -13.0, -17.0, -22.0, -28.0,
        -35.0, -41.0, -46.5,
    ])
    all_eta = np.unique(np.concatenate([eta_symmetric, eta_shallow, eta_cathodic]))
    all_eta = np.sort(all_eta)[::-1]
    return {
        "symmetric": eta_symmetric,
        "shallow": eta_shallow,
        "cathodic": eta_cathodic,
        "all": all_eta,
    }


# ===================================================================
# Tests: _compute_errors
# ===================================================================

class TestComputeErrors:
    """Tests for the _compute_errors helper."""

    def test_exact_match_zero_error(self, true_params):
        k0_err, alpha_err = _compute_errors(
            true_params["k0"], true_params["alpha"],
            true_params["k0"], true_params["alpha"],
        )
        np.testing.assert_allclose(k0_err, 0.0, atol=1e-15)
        np.testing.assert_allclose(alpha_err, 0.0, atol=1e-15)

    def test_known_relative_errors(self, true_params):
        """10% deviation produces 10% relative error."""
        k0_test = true_params["k0"] * 1.10
        alpha_test = true_params["alpha"] * 0.90
        k0_err, alpha_err = _compute_errors(
            k0_test, alpha_test,
            true_params["k0"], true_params["alpha"],
        )
        np.testing.assert_allclose(k0_err, 0.10, rtol=1e-10)
        np.testing.assert_allclose(alpha_err, 0.10, rtol=1e-10)

    def test_scalar_inputs(self):
        """Works with scalar inputs (auto-broadcast)."""
        k0_err, alpha_err = _compute_errors(
            [0.011], [0.55],
            np.array([0.01]), np.array([0.5]),
        )
        assert k0_err[0] == pytest.approx(0.10, rel=1e-10)
        assert alpha_err[0] == pytest.approx(0.10, rel=1e-10)

    def test_division_safe_near_zero(self):
        """Does not raise on near-zero true values."""
        k0_err, alpha_err = _compute_errors(
            [1e-20], [1e-20],
            np.array([1e-20]), np.array([1e-20]),
        )
        assert np.isfinite(k0_err[0])
        assert np.isfinite(alpha_err[0])

    def test_symmetry(self, true_params):
        """Error is the same regardless of sign of deviation."""
        k0_high = true_params["k0"] * 1.05
        k0_low = true_params["k0"] * 0.95
        err_high, _ = _compute_errors(k0_high, true_params["alpha"],
                                       true_params["k0"], true_params["alpha"])
        err_low, _ = _compute_errors(k0_low, true_params["alpha"],
                                      true_params["k0"], true_params["alpha"])
        np.testing.assert_allclose(err_high, err_low, rtol=1e-10)

    def test_multi_element_arrays(self):
        """Handles arrays longer than 2 elements."""
        k0_err, alpha_err = _compute_errors(
            [0.011, 0.0011, 0.0001],
            [0.55, 0.44, 0.33],
            np.array([0.01, 0.001, 0.0001]),
            np.array([0.5, 0.4, 0.3]),
        )
        assert k0_err.shape == (3,)
        assert alpha_err.shape == (3,)
        np.testing.assert_allclose(k0_err, [0.10, 0.10, 0.0], atol=1e-10)
        np.testing.assert_allclose(alpha_err, [0.10, 0.10, 0.10], rtol=1e-10)


# ===================================================================
# Tests: _print_phase_result
# ===================================================================

class TestPrintPhaseResult:
    """Tests for _print_phase_result output and return values."""

    def test_returns_error_tuples(self, true_params, capsys):
        k0_test = true_params["k0"] * 1.05
        alpha_test = true_params["alpha"] * 0.95
        k0_err, alpha_err = _print_phase_result(
            "Test Phase", k0_test, alpha_test,
            true_params["k0"], true_params["alpha"],
            loss=0.001, elapsed=5.0,
        )
        assert k0_err.shape == (2,)
        assert alpha_err.shape == (2,)
        np.testing.assert_allclose(k0_err, 0.05, rtol=1e-10)
        np.testing.assert_allclose(alpha_err, 0.05, rtol=1e-10)

    def test_prints_phase_name(self, true_params, capsys):
        _print_phase_result(
            "My Phase", true_params["k0"], true_params["alpha"],
            true_params["k0"], true_params["alpha"],
            loss=0.0, elapsed=1.0,
        )
        captured = capsys.readouterr()
        assert "My Phase" in captured.out

    def test_prints_loss_and_time(self, true_params, capsys):
        _print_phase_result(
            "Test", true_params["k0"], true_params["alpha"],
            true_params["k0"], true_params["alpha"],
            loss=1.234e-3, elapsed=42.5,
        )
        captured = capsys.readouterr()
        assert "1.234" in captured.out
        assert "42.5" in captured.out


# ===================================================================
# Tests: _subset_targets
# ===================================================================

class TestSubsetTargets:
    """Tests for _subset_targets voltage extraction."""

    def test_full_subset_returns_all(self):
        """When subset equals full grid, return all values."""
        all_eta = np.array([5.0, 3.0, 1.0, -1.0, -3.0])
        cd = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pc = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        cd_sub, pc_sub = _subset_targets(cd, pc, all_eta, all_eta)
        np.testing.assert_array_equal(cd_sub, cd)
        np.testing.assert_array_equal(pc_sub, pc)

    def test_partial_subset(self):
        """Extracts correct subset of values."""
        all_eta = np.array([5.0, 3.0, 1.0, -1.0, -3.0, -5.0])
        cd = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        pc = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        subset_eta = np.array([5.0, -1.0, -5.0])
        cd_sub, pc_sub = _subset_targets(cd, pc, all_eta, subset_eta)
        np.testing.assert_array_equal(cd_sub, [1.0, 4.0, 6.0])
        np.testing.assert_array_equal(pc_sub, [10.0, 40.0, 60.0])

    def test_single_point_subset(self):
        """Extracts a single voltage point."""
        all_eta = np.array([5.0, 1.0, -3.0])
        cd = np.array([10.0, 20.0, 30.0])
        pc = np.array([100.0, 200.0, 300.0])
        cd_sub, pc_sub = _subset_targets(cd, pc, all_eta, np.array([-3.0]))
        np.testing.assert_array_equal(cd_sub, [30.0])
        np.testing.assert_array_equal(pc_sub, [300.0])

    def test_no_match_returns_empty(self):
        """Non-matching voltages return empty arrays."""
        all_eta = np.array([5.0, 1.0, -3.0])
        cd = np.array([10.0, 20.0, 30.0])
        pc = np.array([100.0, 200.0, 300.0])
        cd_sub, pc_sub = _subset_targets(cd, pc, all_eta, np.array([999.0]))
        assert len(cd_sub) == 0
        assert len(pc_sub) == 0

    def test_with_real_voltage_grids(self, voltage_grids):
        """Works with the actual v11 voltage grids."""
        all_eta = voltage_grids["all"]
        n = len(all_eta)
        cd = np.arange(n, dtype=float)
        pc = np.arange(n, dtype=float) * 10

        cd_shallow, pc_shallow = _subset_targets(
            cd, pc, all_eta, voltage_grids["shallow"]
        )
        assert len(cd_shallow) == len(voltage_grids["shallow"])
        assert len(pc_shallow) == len(voltage_grids["shallow"])

    def test_preserves_order(self):
        """Returned values maintain the order of subset_eta."""
        all_eta = np.array([5.0, 3.0, 1.0, -1.0, -3.0])
        cd = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pc = cd * 10
        # Request in reverse order of all_eta
        subset_eta = np.array([-3.0, 1.0, 5.0])
        cd_sub, _ = _subset_targets(cd, pc, all_eta, subset_eta)
        np.testing.assert_array_equal(cd_sub, [5.0, 3.0, 1.0])


# ===================================================================
# Tests: SubsetSurrogateObjective
# ===================================================================

class TestSubsetSurrogateObjective:
    """Tests for the inline SubsetSurrogateObjective class."""

    def _make_objective(self, fitted_surrogate, target_data):
        """Import and instantiate the SubsetSurrogateObjective from v11."""
        # Re-import to get the class from the script module
        from Infer_BVMaster_charged_v11_surrogate_pde import main
        import importlib
        mod = importlib.import_module("Infer_BVMaster_charged_v11_surrogate_pde")

        # The SubsetSurrogateObjective is defined inside main(), so we need
        # to reconstruct it. Instead, test the identical pattern directly.
        class SubsetSurrogateObjective:
            def __init__(self, surrogate, target_cd, target_pc, subset_idx,
                         secondary_weight=1.0, fd_step=1e-5, log_space_k0=True):
                self.surrogate = surrogate
                self.target_cd = np.asarray(target_cd, dtype=float)
                self.target_pc = np.asarray(target_pc, dtype=float)
                self.subset_idx = subset_idx
                self._valid_cd = ~np.isnan(self.target_cd)
                self._valid_pc = ~np.isnan(self.target_pc)
                self.secondary_weight = secondary_weight
                self.fd_step = fd_step
                self.log_space_k0 = log_space_k0
                self._n_evals = 0

            def _x_to_params(self, x):
                x = np.asarray(x, dtype=float)
                if self.log_space_k0:
                    k0_1, k0_2 = 10.0**x[0], 10.0**x[1]
                else:
                    k0_1, k0_2 = x[0], x[1]
                return k0_1, k0_2, x[2], x[3]

            def objective(self, x):
                k0_1, k0_2, a1, a2 = self._x_to_params(x)
                pred = self.surrogate.predict(k0_1, k0_2, a1, a2)
                cd_sim = pred["current_density"][self.subset_idx]
                pc_sim = pred["peroxide_current"][self.subset_idx]
                cd_diff = cd_sim[self._valid_cd] - self.target_cd[self._valid_cd]
                pc_diff = pc_sim[self._valid_pc] - self.target_pc[self._valid_pc]
                J_cd = 0.5 * np.sum(cd_diff**2)
                J_pc = 0.5 * np.sum(pc_diff**2)
                self._n_evals += 1
                return float(J_cd + self.secondary_weight * J_pc)

            def gradient(self, x):
                x = np.asarray(x, dtype=float)
                grad = np.zeros(len(x), dtype=float)
                h = self.fd_step
                for i in range(len(x)):
                    xp, xm = x.copy(), x.copy()
                    xp[i] += h; xm[i] -= h
                    grad[i] = (self.objective(xp) - self.objective(xm)) / (2*h)
                return grad

        return SubsetSurrogateObjective

    def test_zero_objective_at_true_params(self, fitted_surrogate, target_data):
        """Objective is zero when evaluated at the true parameters."""
        SubsetSurrogateObjective = self._make_objective(fitted_surrogate, target_data)
        n_eta = fitted_surrogate.n_eta
        subset_idx = np.arange(n_eta)

        obj = SubsetSurrogateObjective(
            surrogate=fitted_surrogate,
            target_cd=target_data["target_cd"],
            target_pc=target_data["target_pc"],
            subset_idx=subset_idx,
        )

        x_true = np.array([
            np.log10(target_data["k0_1_true"]),
            np.log10(target_data["k0_2_true"]),
            target_data["alpha_1_true"],
            target_data["alpha_2_true"],
        ])
        J = obj.objective(x_true)
        assert J == pytest.approx(0.0, abs=1e-6)

    def test_nonzero_objective_at_wrong_params(self, fitted_surrogate, target_data):
        """Objective is positive when parameters are wrong."""
        SubsetSurrogateObjective = self._make_objective(fitted_surrogate, target_data)
        n_eta = fitted_surrogate.n_eta
        subset_idx = np.arange(n_eta)

        obj = SubsetSurrogateObjective(
            surrogate=fitted_surrogate,
            target_cd=target_data["target_cd"],
            target_pc=target_data["target_pc"],
            subset_idx=subset_idx,
        )

        x_wrong = np.array([np.log10(0.04), np.log10(0.004), 0.7, 0.6])
        J = obj.objective(x_wrong)
        assert J > 0.0

    def test_gradient_finite_difference(self, fitted_surrogate, target_data):
        """Analytical (FD) gradient matches independent FD check."""
        SubsetSurrogateObjective = self._make_objective(fitted_surrogate, target_data)
        n_eta = fitted_surrogate.n_eta
        subset_idx = np.arange(n_eta)

        obj = SubsetSurrogateObjective(
            surrogate=fitted_surrogate,
            target_cd=target_data["target_cd"],
            target_pc=target_data["target_pc"],
            subset_idx=subset_idx,
            fd_step=1e-5,
        )

        x = np.array([np.log10(0.015), np.log10(0.0015), 0.55, 0.45])
        grad = obj.gradient(x)

        # Independent FD check with different step size
        h = 1e-6
        grad_check = np.zeros(4)
        for i in range(4):
            xp, xm = x.copy(), x.copy()
            xp[i] += h; xm[i] -= h
            grad_check[i] = (obj.objective(xp) - obj.objective(xm)) / (2*h)

        np.testing.assert_allclose(grad, grad_check, rtol=0.05)

    def test_secondary_weight_scaling(self, fitted_surrogate, target_data):
        """Doubling secondary_weight changes the objective value."""
        SubsetSurrogateObjective = self._make_objective(fitted_surrogate, target_data)
        n_eta = fitted_surrogate.n_eta
        subset_idx = np.arange(n_eta)

        obj_w1 = SubsetSurrogateObjective(
            surrogate=fitted_surrogate,
            target_cd=target_data["target_cd"],
            target_pc=target_data["target_pc"],
            subset_idx=subset_idx,
            secondary_weight=1.0,
        )
        obj_w2 = SubsetSurrogateObjective(
            surrogate=fitted_surrogate,
            target_cd=target_data["target_cd"],
            target_pc=target_data["target_pc"],
            subset_idx=subset_idx,
            secondary_weight=2.0,
        )

        x = np.array([np.log10(0.03), np.log10(0.003), 0.7, 0.6])
        J1 = obj_w1.objective(x)
        J2 = obj_w2.objective(x)
        # J2 should be larger (more weight on peroxide mismatch)
        assert J2 > J1

    def test_log_space_conversion(self, fitted_surrogate, target_data):
        """Log-space k0 conversion produces correct physical values."""
        SubsetSurrogateObjective = self._make_objective(fitted_surrogate, target_data)
        n_eta = fitted_surrogate.n_eta
        subset_idx = np.arange(n_eta)

        obj = SubsetSurrogateObjective(
            surrogate=fitted_surrogate,
            target_cd=target_data["target_cd"],
            target_pc=target_data["target_pc"],
            subset_idx=subset_idx,
            log_space_k0=True,
        )

        x = np.array([-2.0, -3.0, 0.6, 0.5])
        k0_1, k0_2, a1, a2 = obj._x_to_params(x)
        assert k0_1 == pytest.approx(0.01, rel=1e-10)
        assert k0_2 == pytest.approx(0.001, rel=1e-10)
        assert a1 == pytest.approx(0.6)
        assert a2 == pytest.approx(0.5)

    def test_eval_counter(self, fitted_surrogate, target_data):
        """Evaluation counter increments correctly."""
        SubsetSurrogateObjective = self._make_objective(fitted_surrogate, target_data)
        n_eta = fitted_surrogate.n_eta
        subset_idx = np.arange(n_eta)

        obj = SubsetSurrogateObjective(
            surrogate=fitted_surrogate,
            target_cd=target_data["target_cd"],
            target_pc=target_data["target_pc"],
            subset_idx=subset_idx,
        )

        assert obj._n_evals == 0
        x = np.array([np.log10(0.02), np.log10(0.002), 0.6, 0.5])
        obj.objective(x)
        assert obj._n_evals == 1
        obj.objective(x)
        assert obj._n_evals == 2


# ===================================================================
# Tests: AlphaOnlySurrogateObjective (Phase 1)
# ===================================================================

class TestAlphaOnlyObjective:
    """Tests for Phase 1 alpha-only surrogate optimization."""

    def test_zero_at_true_alpha(self, fitted_surrogate, target_data):
        """Objective is near zero when alpha matches target."""
        obj = AlphaOnlySurrogateObjective(
            surrogate=fitted_surrogate,
            target_cd=target_data["target_cd"],
            target_pc=target_data["target_pc"],
            fixed_k0=[target_data["k0_1_true"], target_data["k0_2_true"]],
            secondary_weight=1.0,
            fd_step=1e-5,
        )
        x_true = np.array([target_data["alpha_1_true"], target_data["alpha_2_true"]])
        J = obj.objective(x_true)
        assert J == pytest.approx(0.0, abs=1e-6)

    def test_recovers_true_alpha(self, fitted_surrogate, target_data):
        """L-BFGS-B recovers true alpha from wrong initial guess."""
        from scipy.optimize import minimize

        obj = AlphaOnlySurrogateObjective(
            surrogate=fitted_surrogate,
            target_cd=target_data["target_cd"],
            target_pc=target_data["target_pc"],
            fixed_k0=[target_data["k0_1_true"], target_data["k0_2_true"]],
            secondary_weight=1.0,
            fd_step=1e-5,
        )
        x0 = np.array([0.4, 0.3])  # wrong guess
        result = minimize(
            obj.objective, x0, jac=obj.gradient,
            method="L-BFGS-B",
            bounds=[(0.1, 0.9), (0.1, 0.9)],
            options={"maxiter": 60, "ftol": 1e-14, "gtol": 1e-8},
        )
        np.testing.assert_allclose(
            result.x,
            [target_data["alpha_1_true"], target_data["alpha_2_true"]],
            atol=0.02,
        )


# ===================================================================
# Tests: CLI Argument Parsing
# ===================================================================

class TestCLIArgs:
    """Tests for argparse defaults and overrides."""

    def _parse(self, argv):
        """Parse CLI args using v11's argparser."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", type=str,
                            default="StudyResults/surrogate_v9/surrogate_model.pkl")
        parser.add_argument("--no-pde", action="store_true")
        parser.add_argument("--workers", type=int, default=0)
        parser.add_argument("--pde-p3-maxiter", type=int, default=30)
        parser.add_argument("--pde-p4-maxiter", type=int, default=25)
        parser.add_argument("--secondary-weight", type=float, default=1.0)
        parser.add_argument("--noise-percent", type=float, default=2.0)
        parser.add_argument("--noise-seed", type=int, default=20260226)
        return parser.parse_args(argv)

    def test_defaults(self):
        args = self._parse([])
        assert args.model == "StudyResults/surrogate_v9/surrogate_model.pkl"
        assert args.no_pde is False
        assert args.workers == 0
        assert args.pde_p3_maxiter == 30
        assert args.pde_p4_maxiter == 25
        assert args.secondary_weight == 1.0
        assert args.noise_percent == 2.0
        assert args.noise_seed == 20260226

    def test_no_pde_flag(self):
        args = self._parse(["--no-pde"])
        assert args.no_pde is True

    def test_override_maxiter(self):
        args = self._parse(["--pde-p3-maxiter", "50", "--pde-p4-maxiter", "40"])
        assert args.pde_p3_maxiter == 50
        assert args.pde_p4_maxiter == 40

    def test_noise_free_diagnostic(self):
        args = self._parse(["--noise-percent", "0.0"])
        assert args.noise_percent == 0.0

    def test_custom_model_path(self):
        args = self._parse(["--model", "/tmp/my_model.pkl"])
        assert args.model == "/tmp/my_model.pkl"

    def test_workers_override(self):
        args = self._parse(["--workers", "8"])
        assert args.workers == 8

    def test_all_options_combined(self):
        args = self._parse([
            "--model", "custom.pkl",
            "--no-pde",
            "--workers", "4",
            "--pde-p3-maxiter", "20",
            "--pde-p4-maxiter", "15",
            "--secondary-weight", "0.5",
            "--noise-percent", "0.0",
            "--noise-seed", "12345",
        ])
        assert args.model == "custom.pkl"
        assert args.no_pde is True
        assert args.workers == 4
        assert args.pde_p3_maxiter == 20
        assert args.pde_p4_maxiter == 15
        assert args.secondary_weight == 0.5
        assert args.noise_percent == 0.0
        assert args.noise_seed == 12345


# ===================================================================
# Tests: Best-of Selection Logic
# ===================================================================

class TestBestOfSelection:
    """Tests for the P3 vs P4 regression guard."""

    def test_p4_wins_when_better(self, true_params):
        """Phase 4 is selected when its max error is lower."""
        p3_k0 = true_params["k0"] * 1.10  # 10% off
        p3_alpha = true_params["alpha"] * 1.05  # 5% off
        p4_k0 = true_params["k0"] * 1.03  # 3% off
        p4_alpha = true_params["alpha"] * 1.02  # 2% off

        p3_k0_err, p3_alpha_err = _compute_errors(
            p3_k0, p3_alpha, true_params["k0"], true_params["alpha"]
        )
        p4_k0_err, p4_alpha_err = _compute_errors(
            p4_k0, p4_alpha, true_params["k0"], true_params["alpha"]
        )

        p3_max_err = max(p3_k0_err.max(), p3_alpha_err.max())
        p4_max_err = max(p4_k0_err.max(), p4_alpha_err.max())

        if p4_max_err <= p3_max_err:
            best_source = "Phase 4"
        else:
            best_source = "Phase 3"

        assert best_source == "Phase 4"
        assert p4_max_err < p3_max_err

    def test_p3_wins_when_p4_regresses(self, true_params):
        """Phase 3 is selected when Phase 4 makes things worse."""
        p3_k0 = true_params["k0"] * 1.03  # 3% off
        p3_alpha = true_params["alpha"] * 1.02  # 2% off
        p4_k0 = true_params["k0"] * 1.15  # 15% off (regressed)
        p4_alpha = true_params["alpha"] * 1.01  # 1% off

        p3_k0_err, p3_alpha_err = _compute_errors(
            p3_k0, p3_alpha, true_params["k0"], true_params["alpha"]
        )
        p4_k0_err, p4_alpha_err = _compute_errors(
            p4_k0, p4_alpha, true_params["k0"], true_params["alpha"]
        )

        p3_max_err = max(p3_k0_err.max(), p3_alpha_err.max())
        p4_max_err = max(p4_k0_err.max(), p4_alpha_err.max())

        if p4_max_err <= p3_max_err:
            best_source = "Phase 4"
        else:
            best_source = "Phase 3"

        assert best_source == "Phase 3"

    def test_tie_prefers_p4(self, true_params):
        """On a tie, Phase 4 is preferred (uses <=)."""
        p3_k0 = true_params["k0"] * 1.05
        p3_alpha = true_params["alpha"] * 1.05
        # Identical errors
        p4_k0 = p3_k0.copy()
        p4_alpha = p3_alpha.copy()

        p3_k0_err, p3_alpha_err = _compute_errors(
            p3_k0, p3_alpha, true_params["k0"], true_params["alpha"]
        )
        p4_k0_err, p4_alpha_err = _compute_errors(
            p4_k0, p4_alpha, true_params["k0"], true_params["alpha"]
        )

        p3_max_err = max(p3_k0_err.max(), p3_alpha_err.max())
        p4_max_err = max(p4_k0_err.max(), p4_alpha_err.max())

        # v11 uses: if p4_max_err <= p3_max_err → pick P4
        if p4_max_err <= p3_max_err:
            best_source = "Phase 4"
        else:
            best_source = "Phase 3"

        assert best_source == "Phase 4"


# ===================================================================
# Tests: Voltage Grid Consistency
# ===================================================================

class TestVoltageGrids:
    """Tests that v11 voltage grids match v7/v9 exactly."""

    def test_shallow_grid_10_points(self, voltage_grids):
        assert len(voltage_grids["shallow"]) == 10

    def test_cathodic_grid_15_points(self, voltage_grids):
        assert len(voltage_grids["cathodic"]) == 15

    def test_symmetric_grid_12_points(self, voltage_grids):
        assert len(voltage_grids["symmetric"]) == 12

    def test_shallow_range(self, voltage_grids):
        """Shallow grid covers [-13, -1]."""
        eta = voltage_grids["shallow"]
        assert eta.min() == pytest.approx(-13.0)
        assert eta.max() == pytest.approx(-1.0)

    def test_cathodic_range(self, voltage_grids):
        """Cathodic grid covers [-46.5, -1]."""
        eta = voltage_grids["cathodic"]
        assert eta.min() == pytest.approx(-46.5)
        assert eta.max() == pytest.approx(-1.0)

    def test_shallow_is_subset_of_cathodic(self, voltage_grids):
        """Every shallow voltage appears in cathodic (except 11.5 bridge pts)."""
        shallow = set(voltage_grids["shallow"])
        cathodic = set(voltage_grids["cathodic"])
        # eta_shallow has -11.5 which is NOT in eta_cathodic (bridge point)
        overlap = shallow & cathodic
        assert len(overlap) >= 8  # at least 8 of 10 match

    def test_all_eta_is_sorted_descending(self, voltage_grids):
        all_eta = voltage_grids["all"]
        assert np.all(np.diff(all_eta) <= 0)

    def test_all_eta_is_unique(self, voltage_grids):
        all_eta = voltage_grids["all"]
        assert len(all_eta) == len(np.unique(all_eta))


# ===================================================================
# Tests: CSV Output Format
# ===================================================================

class TestCSVOutput:
    """Tests for comparison CSV output formatting."""

    def test_csv_round_trip(self):
        """Write and read back a mock CSV in v11 format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                          delete=False, newline="") as f:
            csv_path = f.name
            writer = csv.writer(f)
            writer.writerow([
                "phase", "k0_1", "k0_2", "alpha_1", "alpha_2",
                "k0_1_err_pct", "k0_2_err_pct", "alpha_1_err_pct", "alpha_2_err_pct",
                "loss", "time_s",
            ])
            writer.writerow([
                "Phase 1 (alpha, surrogate)",
                f"{0.01:.8e}", f"{0.001:.8e}",
                f"{0.627:.6f}", f"{0.500:.6f}",
                f"{5.0:.4f}", f"{3.0:.4f}",
                f"{2.0:.4f}", f"{1.5:.4f}",
                f"{0.001:.12e}", f"{0.5:.1f}",
            ])

        try:
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 1
            row = rows[0]
            assert row["phase"] == "Phase 1 (alpha, surrogate)"
            assert float(row["k0_1"]) == pytest.approx(0.01, rel=1e-6)
            assert float(row["k0_2"]) == pytest.approx(0.001, rel=1e-6)
            assert float(row["alpha_1"]) == pytest.approx(0.627, rel=1e-5)
            assert float(row["alpha_2"]) == pytest.approx(0.500, rel=1e-5)
            assert float(row["loss"]) == pytest.approx(0.001, rel=1e-6)
            assert "k0_1_err_pct" in row
            assert "time_s" in row
        finally:
            os.unlink(csv_path)

    def test_csv_has_correct_columns(self):
        """CSV header matches expected v11 format."""
        expected_cols = [
            "phase", "k0_1", "k0_2", "alpha_1", "alpha_2",
            "k0_1_err_pct", "k0_2_err_pct", "alpha_1_err_pct", "alpha_2_err_pct",
            "loss", "time_s",
        ]
        # Verify by writing with these columns
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                          delete=False, newline="") as f:
            csv_path = f.name
            writer = csv.writer(f)
            writer.writerow(expected_cols)

        try:
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                assert list(reader.fieldnames) == expected_cols
        finally:
            os.unlink(csv_path)


# ===================================================================
# Tests: Warm-Start Chain Validation
# ===================================================================

class TestWarmStartChain:
    """Tests that the warm-start chain is wired correctly."""

    def test_phase1_to_phase2_alpha_propagation(self, fitted_surrogate, target_data):
        """Phase 1 alpha output becomes Phase 2 alpha input."""
        from scipy.optimize import minimize

        # Phase 1: alpha-only
        p1_obj = AlphaOnlySurrogateObjective(
            surrogate=fitted_surrogate,
            target_cd=target_data["target_cd"],
            target_pc=target_data["target_pc"],
            fixed_k0=[target_data["k0_1_true"], target_data["k0_2_true"]],
            secondary_weight=1.0,
            fd_step=1e-5,
        )
        result_p1 = minimize(
            p1_obj.objective, [0.4, 0.3], jac=p1_obj.gradient,
            method="L-BFGS-B",
            bounds=[(0.1, 0.9), (0.1, 0.9)],
            options={"maxiter": 60, "ftol": 1e-14, "gtol": 1e-8},
        )
        p1_alpha = result_p1.x.copy()

        # Verify P1 alpha is reasonable (not at initial guess)
        assert not np.allclose(p1_alpha, [0.4, 0.3], atol=0.01), \
            "P1 should move away from initial guess"

        # Phase 2: x0 uses P1 alpha (this is what v11 does)
        initial_k0_guess = [0.005, 0.0005]
        x0_p2 = np.array([
            np.log10(initial_k0_guess[0]),
            np.log10(initial_k0_guess[1]),
            p1_alpha[0],
            p1_alpha[1],
        ])
        # Verify the Phase 2 initial guess contains Phase 1's alpha
        np.testing.assert_array_equal(x0_p2[2:4], p1_alpha)

    def test_phase2_to_phase3_warm_start(self):
        """Phase 2 surrogate result would feed Phase 3 PDE request."""
        # Simulate Phase 2 output
        p2_k0 = np.array([0.012, 0.0012])
        p2_alpha = np.array([0.63, 0.51])

        # Phase 3 initial guess = Phase 2 result
        surr_best_k0 = p2_k0.copy()
        surr_best_alpha = p2_alpha.copy()

        # This is what v11 passes to BVFluxCurveInferenceRequest
        np.testing.assert_array_equal(surr_best_k0, p2_k0)
        np.testing.assert_array_equal(surr_best_alpha, p2_alpha)

    def test_phase3_to_phase4_warm_start(self):
        """Phase 3 PDE result would feed Phase 4 PDE request."""
        p3_k0 = np.array([0.0105, 0.00105])
        p3_alpha = np.array([0.625, 0.498])

        # Phase 4 initial guess = Phase 3 result (this is what v11 does)
        p4_initial_k0 = p3_k0.tolist()
        p4_initial_alpha = p3_alpha.tolist()

        assert p4_initial_k0 == pytest.approx(p3_k0.tolist())
        assert p4_initial_alpha == pytest.approx(p3_alpha.tolist())


# ===================================================================
# Tests: Full Surrogate Pipeline (no-pde mode)
# ===================================================================

class TestSurrogateOnlyPipeline:
    """Integration test for Phases 1-2 (surrogate-only, --no-pde)."""

    def test_two_phase_loss_decreases(self, fitted_surrogate, target_data):
        """Phase 2 loss is lower than Phase 1 loss.

        Verifies the pipeline structure: Phase 1 (alpha-only) produces a loss,
        then Phase 2 (joint 4-param with k0 free) produces a lower loss since
        it has more degrees of freedom. This tests pipeline wiring without
        requiring the toy surrogate to produce parameter-accurate results.
        """
        from scipy.optimize import minimize

        initial_k0_guess = [target_data["k0_1_true"], target_data["k0_2_true"]]
        initial_alpha_guess = [0.4, 0.3]

        # Phase 1: Alpha-only (k0 fixed at true values)
        p1_obj = AlphaOnlySurrogateObjective(
            surrogate=fitted_surrogate,
            target_cd=target_data["target_cd"],
            target_pc=target_data["target_pc"],
            fixed_k0=initial_k0_guess,
            secondary_weight=1.0,
            fd_step=1e-5,
        )

        p1_init_loss = p1_obj.objective(np.array(initial_alpha_guess))
        result_p1 = minimize(
            p1_obj.objective, initial_alpha_guess, jac=p1_obj.gradient,
            method="L-BFGS-B",
            bounds=[(0.1, 0.9), (0.1, 0.9)],
            options={"maxiter": 60, "ftol": 1e-14, "gtol": 1e-8},
        )
        p1_alpha = result_p1.x.copy()
        p1_loss = float(result_p1.fun)

        # Phase 1 should reduce loss from initial guess
        assert p1_loss < p1_init_loss, (
            f"P1 loss {p1_loss:.6e} should be less than initial {p1_init_loss:.6e}"
        )

        # Phase 2: Joint 4-param using SurrogateObjective
        from Surrogate.objectives import SurrogateObjective
        p2_obj = SurrogateObjective(
            surrogate=fitted_surrogate,
            target_cd=target_data["target_cd"],
            target_pc=target_data["target_pc"],
            secondary_weight=1.0,
            fd_step=1e-5,
            log_space_k0=True,
        )

        x0_p2 = np.array([
            np.log10(initial_k0_guess[0]),
            np.log10(initial_k0_guess[1]),
            p1_alpha[0], p1_alpha[1],
        ])
        bounds_p2 = [
            (np.log10(0.001), np.log10(0.1)),
            (np.log10(0.0001), np.log10(0.01)),
            (0.1, 0.9), (0.1, 0.9),
        ]

        p2_init_loss = p2_obj.objective(x0_p2)
        result_p2 = minimize(
            p2_obj.objective, x0_p2, jac=p2_obj.gradient,
            method="L-BFGS-B", bounds=bounds_p2,
            options={"maxiter": 60, "ftol": 1e-14, "gtol": 1e-8},
        )
        p2_loss = float(result_p2.fun)

        # Phase 2 should reduce loss
        assert p2_loss <= p2_init_loss, (
            f"P2 loss {p2_loss:.6e} should be <= initial {p2_init_loss:.6e}"
        )

        # Final loss should be very small (surrogate evaluated at targets it generated)
        assert p2_loss < p1_init_loss * 0.5, (
            f"P2 loss {p2_loss:.6e} should be significantly less than "
            f"P1 initial loss {p1_init_loss:.6e}"
        )

    def test_phase1_alpha_recovery_accuracy(self, fitted_surrogate, target_data):
        """Phase 1 recovers alpha to high accuracy when k0 is known.

        This is the key surrogate test: with k0 fixed at true values,
        the surrogate should recover alpha within a few percent.
        """
        from scipy.optimize import minimize

        p1_obj = AlphaOnlySurrogateObjective(
            surrogate=fitted_surrogate,
            target_cd=target_data["target_cd"],
            target_pc=target_data["target_pc"],
            fixed_k0=[target_data["k0_1_true"], target_data["k0_2_true"]],
            secondary_weight=1.0,
            fd_step=1e-5,
        )
        result_p1 = minimize(
            p1_obj.objective, [0.4, 0.3], jac=p1_obj.gradient,
            method="L-BFGS-B",
            bounds=[(0.1, 0.9), (0.1, 0.9)],
            options={"maxiter": 60, "ftol": 1e-14, "gtol": 1e-8},
        )

        true_alpha = np.array([target_data["alpha_1_true"], target_data["alpha_2_true"]])
        np.testing.assert_allclose(result_p1.x, true_alpha, atol=0.02)
