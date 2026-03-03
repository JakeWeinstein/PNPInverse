"""Adjoint-gradient joint inference with Tikhonov regularization.

Full 4-species charged system: O2, H2O2, H+, ClO4- (z=[0,0,+1,-1]).
Recovers [k0_1, k0_2, alpha_1, alpha_2] with regularization penalty:

    J_total = J_data + lambda * [sum((log10(k0) - log10(k0_prior))^2)
                                 + sum((alpha - alpha_prior)^2)]

This addresses parameter correlation between k0 and alpha by biasing toward
physically reasonable prior values. Tests multiple lambda values and compares
against unregularized (lambda=0) baseline.

Usage (from PNPInverse/ directory)::

    python scripts/inference/Infer_BVJoint_charged_regularized.py
"""

from __future__ import annotations

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts._bv_common import (
    setup_firedrake_env,
    K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2, I_SCALE,
    FOUR_SPECIES_CHARGED,
    SNES_OPTS_CHARGED,
    make_bv_solver_params,
    make_recovery_config,
    print_params_summary,
)
setup_firedrake_env()

import numpy as np

from FluxCurve import (
    BVFluxCurveInferenceRequest,
    run_bv_joint_flux_curve_inference,
)
from Forward.steady_state import SteadyStateConfig

print_params_summary()


# ---------------------------------------------------------------------------
# Single-lambda runner
# ---------------------------------------------------------------------------

def run_one_lambda(
    reg_lambda: float,
    *,
    output_subdir: str,
    eta_values: np.ndarray,
    true_k0: list,
    true_alpha: list,
    initial_k0_guess: list,
    initial_alpha_guess: list,
    k0_prior: list,
    alpha_prior: list,
) -> dict:
    """Run joint inference with a specific regularization lambda."""
    dt = 0.5
    max_ss_steps = 100
    t_end = dt * max_ss_steps
    base_sp = make_bv_solver_params(
        eta_hat=0.0, dt=dt, t_end=t_end,
        species=FOUR_SPECIES_CHARGED, snes_opts=SNES_OPTS_CHARGED,
    )

    steady = SteadyStateConfig(
        relative_tolerance=1e-4, absolute_tolerance=1e-8,
        consecutive_steps=4, max_steps=max_ss_steps,
        flux_observable="total_species", verbose=False,
    )

    observable_scale = -I_SCALE

    request = BVFluxCurveInferenceRequest(
        base_solver_params=base_sp,
        steady=steady,
        true_k0=true_k0,
        initial_guess=initial_k0_guess,
        phi_applied_values=eta_values.tolist(),
        target_csv_path=os.path.join(output_subdir, "phi_applied_vs_current_density_synthetic.csv"),
        output_dir=output_subdir,
        regenerate_target=True,
        target_noise_percent=2.0,
        target_seed=20260226,
        observable_mode="current_density",
        current_density_scale=observable_scale,
        observable_label="current density (mA/cm2)",
        observable_title=f"Regularized joint inference (lambda={reg_lambda})",
        control_mode="joint",
        k0_lower=1e-8, k0_upper=100.0,
        log_space=True,
        true_alpha=true_alpha,
        initial_alpha_guess=initial_alpha_guess,
        alpha_lower=0.05, alpha_upper=0.95,
        alpha_log_space=False,
        mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
        optimizer_method="L-BFGS-B",
        optimizer_tolerance=1e-12,
        optimizer_options={"maxiter": 30, "ftol": 1e-12, "gtol": 1e-6, "disp": True},
        max_iters=30, gtol=1e-6,
        fail_penalty=1e9,
        print_point_gradients=True,
        live_plot=False,
        live_plot_export_gif_path=os.path.join(output_subdir, "convergence.gif"),
        # Regularization
        regularization_lambda=reg_lambda,
        regularization_k0_prior=k0_prior,
        regularization_alpha_prior=alpha_prior,
        forward_recovery=make_recovery_config(),
    )

    return run_bv_joint_flux_curve_inference(request)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    eta_values = np.linspace(-1.0, -10.0, 10)

    true_k0 = [K0_HAT_R1, K0_HAT_R2]
    true_alpha = [ALPHA_R1, ALPHA_R2]

    # Deliberately wrong initial guesses
    initial_k0_guess = [0.005, 0.0005]
    initial_alpha_guess = [0.4, 0.3]

    # Prior values: slight perturbation from true (simulating imperfect prior knowledge)
    k0_prior = [K0_HAT_R1 * 1.5, K0_HAT_R2 * 2.0]   # within ~2x of true
    alpha_prior = [0.5, 0.4]                       # within 0.1 of true

    base_output = os.path.join("StudyResults", "bv_joint_regularized_charged")

    # Test lambda values including 0 (unregularized baseline)
    lambda_values = [0.0, 0.001, 0.01, 0.1, 1.0]

    all_results = []
    for lam in lambda_values:
        print(f"\n{'='*70}")
        print(f"  Regularization lambda = {lam}")
        print(f"{'='*70}")

        subdir = os.path.join(base_output, f"lambda_{lam:.4f}")
        try:
            result = run_one_lambda(
                lam,
                output_subdir=subdir,
                eta_values=eta_values,
                true_k0=true_k0,
                true_alpha=true_alpha,
                initial_k0_guess=initial_k0_guess,
                initial_alpha_guess=initial_alpha_guess,
                k0_prior=k0_prior,
                alpha_prior=alpha_prior,
            )
            all_results.append({"lambda": lam, "result": result, "error": None})
        except Exception as e:
            print(f"  FAILED: {e}")
            all_results.append({"lambda": lam, "result": None, "error": str(e)})

    # Summary table
    print(f"\n{'='*90}")
    print(f"  Regularization Study Summary")
    print(f"{'='*90}")
    print(f"{'lambda':>8} | {'k0_1 err':>10} {'k0_2 err':>10} {'a1 err':>10} {'a2 err':>10} | {'loss':>12}")
    print(f"{'-'*90}")

    true_k0_arr = np.asarray(true_k0)
    true_alpha_arr = np.asarray(true_alpha)

    for entry in all_results:
        lam = entry["lambda"]
        r = entry["result"]
        if r is None:
            print(f"{lam:>8.4f} | {'FAILED':>10} {'':>10} {'':>10} {'':>10} | {'N/A':>12}")
            continue
        best_k0 = np.asarray(r["best_k0"])
        best_alpha = np.asarray(r["best_alpha"])
        k0_err = np.abs(best_k0 - true_k0_arr) / np.maximum(np.abs(true_k0_arr), 1e-16)
        alpha_err = np.abs(best_alpha - true_alpha_arr) / np.maximum(np.abs(true_alpha_arr), 1e-16)
        print(f"{lam:>8.4f} | {k0_err[0]:>10.4f} {k0_err[1]:>10.4f} "
              f"{alpha_err[0]:>10.4f} {alpha_err[1]:>10.4f} | {r['best_loss']:>12.6e}")

    print(f"{'='*90}")
    print(f"  k0 prior (nondim): {k0_prior}")
    print(f"  alpha prior: {alpha_prior}")
    print(f"  True k0 (nondim): {true_k0}")
    print(f"  True alpha: {true_alpha}")

    # Verify lambda=0 reproduces unregularized results
    lam0 = next((e for e in all_results if e["lambda"] == 0.0), None)
    if lam0 and lam0["result"] is not None:
        print(f"\n  Verification: lambda=0 loss = {lam0['result']['best_loss']:.12e}")
        print(f"  (Should match unregularized baseline)")


if __name__ == "__main__":
    main()
