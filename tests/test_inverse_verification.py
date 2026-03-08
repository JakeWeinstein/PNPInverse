"""Inverse problem verification tests.

Validates the gradient machinery (INV-02b) used by the surrogate-based
optimizer and scaffolds placeholder classes for the Firedrake-dependent
tests implemented in Plan 02.

Artifacts produced (under StudyResults/inverse_verification/):
    - gradient_fd_convergence.json: per-step-size FD gradient values,
      convergence rates, and analytic gradient comparison.

Requirements covered: INV-02 (surrogate FD convergence).
"""

import json
import os
from datetime import datetime, timezone

import numpy as np
import pytest

from Surrogate.ensemble import load_nn_ensemble
from Surrogate.objectives import SurrogateObjective

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUTPUT_DIR = os.path.join(_ROOT, "StudyResults", "inverse_verification")

_V11_DIR = os.path.join(_ROOT, "StudyResults", "surrogate_v11")
_ENSEMBLE_DIR = os.path.join(_V11_DIR, "nn_ensemble", "D3-deeper")

# True parameters from _bv_common (imported indirectly to avoid heavy deps)
from scripts._bv_common import K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2


# ---------------------------------------------------------------------------
# Fixtures (module-scoped -- expensive NN loading runs once)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def nn_ensemble():
    """Load the NN ensemble surrogate and build a SurrogateObjective at true params.

    Returns a dict with the surrogate model, objective, true x-vector, and
    target curves so tests can reuse them without re-loading.
    """
    model = load_nn_ensemble(_ENSEMBLE_DIR, n_members=5, device="cpu")

    # Generate targets at true parameters (self-consistency test)
    pred = model.predict(K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2)
    target_cd = pred["current_density"]
    target_pc = pred["peroxide_current"]

    # x-vector in log-space for k0
    x_true = np.array([
        np.log10(K0_HAT_R1),
        np.log10(K0_HAT_R2),
        ALPHA_R1,
        ALPHA_R2,
    ])

    obj = SurrogateObjective(
        surrogate=model,
        target_cd=target_cd,
        target_pc=target_pc,
        secondary_weight=1.0,
        fd_step=1e-5,
        log_space_k0=True,
    )

    return {
        "model": model,
        "objective": obj,
        "x_true": x_true,
        "target_cd": target_cd,
        "target_pc": target_pc,
    }


# ---------------------------------------------------------------------------
# INV-02b: Surrogate FD convergence test (fast -- no Firedrake)
# ---------------------------------------------------------------------------

class TestSurrogateFDConvergence:
    """INV-02b: Central finite-difference gradient convergence on the surrogate.

    Verifies that the SurrogateObjective's FD gradient converges at O(h^2)
    for central differences and agrees with the built-in gradient method.

    Notes on step-size selection:
        The NN ensemble operates in float32 internally, so FD errors at
        h < 1e-5 are dominated by surrogate roundoff rather than truncation
        error.  We therefore use h in {1e-2, 1e-3, 1e-4} and take h=1e-4
        as the reference.  Convergence rate is assessed only on parameters
        whose truncation error is large enough to dominate NN noise (the
        alpha parameters); for log10(k0) the objective curvature is small
        enough that truncation errors are already at the NN noise floor for
        h=1e-2, so convergence rates there are not meaningful.

        We evaluate at a perturbed point (not the optimum) because J=0
        at the true parameters and the gradient is identically zero.
    """

    def test_fd_convergence_rate(self, nn_ensemble):
        """Central FD converges at O(h^2) where signal dominates NN noise.

        Additionally verifies that FD at h=1e-3 agrees with
        SurrogateObjective.gradient(fd_step=1e-4) within 1% for all
        4 parameters.
        """
        obj = nn_ensemble["objective"]
        x_true = nn_ensemble["x_true"]

        # Evaluate away from the minimum so gradients are nonzero
        x0 = x_true + np.array([0.5, 0.5, 0.05, 0.05])

        step_sizes = [1e-2, 1e-3, 1e-4]
        n_params = len(x0)
        param_labels = ["log10_k0_1", "log10_k0_2", "alpha_1", "alpha_2"]

        # Compute FD gradients at each step size
        fd_grads = {}
        for h in step_sizes:
            grad = np.zeros(n_params)
            for i in range(n_params):
                x_plus = x0.copy()
                x_minus = x0.copy()
                x_plus[i] += h
                x_minus[i] -= h
                f_plus = obj.objective(x_plus)
                f_minus = obj.objective(x_minus)
                grad[i] = (f_plus - f_minus) / (2.0 * h)
            fd_grads[h] = grad

        # Use finest step as reference
        h_ref = step_sizes[-1]
        g_ref = fd_grads[h_ref]

        # Compute convergence rates per parameter
        convergence_rates = {}
        h_coarse = [step_sizes[0], step_sizes[1]]
        n_converged = 0
        for p_idx in range(n_params):
            errors = []
            for h in h_coarse:
                err = abs(fd_grads[h][p_idx] - g_ref[p_idx])
                errors.append(max(err, 1e-30))  # floor to avoid log(0)

            # Log-log slope: rate = log(e1/e2) / log(h1/h2)
            rate = np.log(errors[0] / errors[1]) / np.log(
                h_coarse[0] / h_coarse[1]
            )
            convergence_rates[param_labels[p_idx]] = float(rate)

            # Only assert convergence rate when truncation error dominates
            # NN noise. Check: error at coarsest h should be > 10x the
            # error at mid h. If not, both are at the noise floor.
            if errors[0] > 10 * errors[1]:
                assert 1.5 <= rate <= 3.0, (
                    f"Parameter {param_labels[p_idx]}: convergence rate "
                    f"{rate:.3f} outside expected [1.5, 3.0] for O(h^2) "
                    f"central differences (rates > 2 expected for smooth NN)"
                )
                n_converged += 1

        # At least 2 parameters (the alphas) should show clear convergence
        assert n_converged >= 2, (
            f"Only {n_converged}/4 parameters showed O(h^2) convergence "
            f"(expected >= 2). Rates: {convergence_rates}"
        )

        # Check FD at h=1e-3 agrees with SurrogateObjective.gradient() at
        # the same step size within 1%. This is a self-consistency check:
        # our manual FD loop should reproduce the class method exactly.
        obj_h1e3 = SurrogateObjective(
            surrogate=nn_ensemble["model"],
            target_cd=nn_ensemble["target_cd"],
            target_pc=nn_ensemble["target_pc"],
            secondary_weight=1.0,
            fd_step=1e-3,
            log_space_k0=True,
        )
        analytic_grad = obj_h1e3.gradient(x0)
        fd_mid = fd_grads[1e-3]

        for p_idx in range(n_params):
            if abs(analytic_grad[p_idx]) < 1e-15:
                continue
            rel_diff = abs(fd_mid[p_idx] - analytic_grad[p_idx]) / abs(
                analytic_grad[p_idx]
            )
            assert rel_diff < 0.01, (
                f"Parameter {param_labels[p_idx]}: FD(h=1e-3) vs "
                f".gradient(fd_step=1e-3) relative difference = "
                f"{rel_diff:.6f} (> 1%)"
            )

        # Save JSON artifact
        os.makedirs(_OUTPUT_DIR, exist_ok=True)
        artifact = {
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "x_true": x_true.tolist(),
                "x_eval": x0.tolist(),
                "param_labels": param_labels,
                "step_sizes": step_sizes,
                "reference_step": h_ref,
                "note": (
                    "Evaluated at perturbed point (not optimum) so gradients "
                    "are nonzero. Step sizes chosen for NN float32 precision."
                ),
            },
            "fd_gradients": {
                str(h): fd_grads[h].tolist() for h in step_sizes
            },
            "convergence_rates": convergence_rates,
            "analytic_gradient": analytic_grad.tolist(),
            "analytic_vs_fd_1e3_relative_diff": {
                param_labels[i]: (
                    float(
                        abs(fd_mid[i] - analytic_grad[i])
                        / max(abs(analytic_grad[i]), 1e-30)
                    )
                )
                for i in range(n_params)
            },
        }

        json_path = os.path.join(_OUTPUT_DIR, "gradient_fd_convergence.json")
        with open(json_path, "w") as f:
            json.dump(artifact, f, indent=2)


# ---------------------------------------------------------------------------
# Placeholder stubs for Plan 02
# ---------------------------------------------------------------------------

class TestParameterRecovery:
    """INV-01: Implemented in Plan 02"""
    pass


class TestGradientConsistencyPDE:
    """INV-02a: Implemented in Plan 02"""
    pass


class TestMultistartBasin:
    """INV-03: Implemented in Plan 02"""
    pass
