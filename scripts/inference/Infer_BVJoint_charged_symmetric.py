"""Joint k0+alpha inference -- SYMMETRIC voltage range (anodic + cathodic).

Full 4-species charged system: O2, H2O2, H+, ClO4- (z=[0,0,+1,-1]).
Symmetric range: 20 inference points spanning eta_hat in [-28, +5].

This script tests whether including anodic overpotentials improves
joint (k0, alpha) recovery by breaking the k0-alpha correlation.

At cathodic eta: I ~ k0 * exp(-alpha * eta)     -> Tafel slope = alpha
At anodic eta:   I ~ k0 * exp((1-alpha) * eta)  -> Tafel slope = (1-alpha)
Together: two independent constraints on alpha, breaking the k0-alpha correlation.

Usage (from PNPInverse/ directory)::

    python scripts/inference/Infer_BVJoint_charged_symmetric.py
"""

from __future__ import annotations

import csv
import time
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts._bv_common import (
    setup_firedrake_env,
    K0_HAT_R1,
    K0_HAT_R2,
    I_SCALE,
    ALPHA_R1,
    ALPHA_R2,
    FOUR_SPECIES_CHARGED,
    SNES_OPTS_CHARGED,
    make_bv_solver_params,
    make_recovery_config,
)
setup_firedrake_env()

import numpy as np

from FluxCurve import (
    BVFluxCurveInferenceRequest,
    run_bv_joint_flux_curve_inference,
)
from Forward.steady_state import SteadyStateConfig

def main() -> None:
    # Three voltage configurations for comparison
    configs = {
        "cathodic_only": {
            "eta": np.linspace(-1.0, -10.0, 10),
            "label": "Cathodic only [-1, -10], 10 points",
        },
        "symmetric_focused": {
            "eta": np.array([
                +5.0, +3.0, +1.0,           # anodic (3 pts)
                -0.5,                         # near-equilibrium
                -1.0, -2.0, -3.0,           # cathodic onset
                -5.0, -8.0,                  # transition
                -10.0, -15.0, -20.0,        # knee + plateau
            ]),
            "label": "Symmetric focused [-20, +5], 12 points",
        },
        "symmetric_full": {
            "eta": np.array([
                +5.0, +3.0, +2.0, +1.0, +0.5,
                -0.25, -0.5,
                -1.0, -1.5, -2.0, -3.0,
                -4.0, -5.0, -6.5, -8.0,
                -10.0, -13.0,
                -17.0, -22.0, -28.0,
            ]),
            "label": "Symmetric full [-28, +5], 20 points",
        },
    }

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

    true_k0 = [K0_HAT_R1, K0_HAT_R2]
    true_alpha = [ALPHA_R1, ALPHA_R2]
    initial_k0_guess = [0.005, 0.0005]
    initial_alpha_guess = [0.4, 0.3]
    observable_scale = -I_SCALE

    base_output = os.path.join("StudyResults", "bv_joint_inference_charged_symmetric")
    os.makedirs(base_output, exist_ok=True)

    all_results = {}

    for config_name, cfg in configs.items():
        eta_values = cfg["eta"]
        label = cfg["label"]

        print(f"\n{'='*70}")
        print(f"  Joint Inference: {label}")
        print(f"{'='*70}")

        out_dir = os.path.join(base_output, config_name)
        t0 = time.time()

        # Use bridge points for configurations with large gaps
        max_gap = 3.0 if eta_values.min() < -12 else 0.0

        request = BVFluxCurveInferenceRequest(
            base_solver_params=base_sp,
            steady=steady,
            true_k0=true_k0,
            initial_guess=initial_k0_guess,
            phi_applied_values=eta_values.tolist(),
            target_csv_path=os.path.join(out_dir, "target.csv"),
            output_dir=out_dir,
            regenerate_target=True,
            target_noise_percent=2.0,
            target_seed=20260226,
            observable_mode="current_density",
            current_density_scale=observable_scale,
            observable_label="current density (mA/cm2)",
            observable_title=f"Joint: {label}",
            control_mode="joint",
            k0_lower=1e-8, k0_upper=100.0,
            log_space=True,
            true_alpha=true_alpha,
            initial_alpha_guess=initial_alpha_guess,
            alpha_lower=0.05, alpha_upper=0.95,
            mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
            max_eta_gap=max_gap,
            optimizer_method="L-BFGS-B",
            optimizer_options={"maxiter": 40, "ftol": 1e-12, "gtol": 1e-6, "disp": True},
            max_iters=40,
            live_plot=False,
            forward_recovery=make_recovery_config(max_it_cap=600),
        )

        result = run_bv_joint_flux_curve_inference(request)
        elapsed = time.time() - t0

        best_k0 = np.asarray(result["best_k0"])
        best_alpha = np.asarray(result["best_alpha"])

        all_results[config_name] = {
            "k0": best_k0.tolist(),
            "alpha": best_alpha.tolist(),
            "loss": result["best_loss"],
            "time": elapsed,
            "label": label,
            "n_points": len(eta_values),
        }

        print(f"\n  Result: k0 = {best_k0.tolist()}, alpha = {best_alpha.tolist()}")
        print(f"  Time: {elapsed:.1f}s")

    # -----------------------------------------------------------------------
    # Comparison
    # -----------------------------------------------------------------------
    true_k0_arr = np.asarray(true_k0)
    true_alpha_arr = np.asarray(true_alpha)

    print(f"\n{'='*100}")
    print(f"  Joint Inference Comparison: Cathodic-Only vs Symmetric")
    print(f"{'='*100}")
    print(f"  True k0:    {true_k0}")
    print(f"  True alpha: {true_alpha}")
    print()

    print(f"{'Config':<25} | {'pts':>4} | {'k0_1 err':>10} {'k0_2 err':>10} "
          f"{'a1 err':>10} {'a2 err':>10} | {'loss':>12} | {'time':>6}")
    print(f"{'-'*100}")

    for config_name, r in all_results.items():
        k0_arr = np.asarray(r["k0"])
        alpha_arr = np.asarray(r["alpha"])
        k0_err = np.abs(k0_arr - true_k0_arr) / np.maximum(np.abs(true_k0_arr), 1e-16)
        alpha_err = np.abs(alpha_arr - true_alpha_arr) / np.maximum(np.abs(true_alpha_arr), 1e-16)
        print(f"{config_name:<25} | {r['n_points']:>4} | "
              f"{k0_err[0]:>10.4f} {k0_err[1]:>10.4f} "
              f"{alpha_err[0]:>10.4f} {alpha_err[1]:>10.4f} "
              f"| {r['loss']:>12.6e} | {r['time']:>5.0f}s")

    print(f"{'='*100}")

    # Save CSV
    csv_path = os.path.join(base_output, "joint_comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "config", "n_points", "k0_1", "k0_2", "alpha_1", "alpha_2",
            "k0_1_err", "k0_2_err", "alpha_1_err", "alpha_2_err",
            "loss", "time_s",
        ])
        for config_name, r in all_results.items():
            k0_arr = np.asarray(r["k0"])
            alpha_arr = np.asarray(r["alpha"])
            k0_err = np.abs(k0_arr - true_k0_arr) / np.maximum(np.abs(true_k0_arr), 1e-16)
            alpha_err = np.abs(alpha_arr - true_alpha_arr) / np.maximum(np.abs(true_alpha_arr), 1e-16)
            writer.writerow([
                config_name, r["n_points"],
                f"{k0_arr[0]:.8e}", f"{k0_arr[1]:.8e}",
                f"{alpha_arr[0]:.6f}", f"{alpha_arr[1]:.6f}",
                f"{k0_err[0]:.6f}", f"{k0_err[1]:.6f}",
                f"{alpha_err[0]:.6f}", f"{alpha_err[1]:.6f}",
                f"{r['loss']:.12e}", f"{r['time']:.1f}",
            ])
    print(f"\n[csv] Comparison saved -> {csv_path}")

    print(f"\n=== Joint Inference Comparison Complete ===")
    print(f"Output: {base_output}/")


if __name__ == "__main__":
    main()
