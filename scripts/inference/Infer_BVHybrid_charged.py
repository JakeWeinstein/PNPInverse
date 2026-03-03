"""Hybrid alpha+k0 inference -- best of both worlds.

Full 4-species charged system: O2, H2O2, H+, ClO4- (z=[0,0,+1,-1]).

Two-phase protocol:
    Phase A: Alpha-only inference on 12-point symmetric-focused range
             (anodic data gives the (1-alpha) Tafel slope independently).
             Expected: alpha_1 ~ 1.8% error.

    Phase B: k0-only inference on 15-point cathodic extended range
             (alpha fixed at Phase A result).
             Expected: k0_1 ~ 5% error.

Rationale: Phase 6 showed that symmetric data gives excellent alpha_1 (1.8%)
but k0 hit upper bounds. Meanwhile cathodic-only staged inference (Phase 5)
gives k0_1 = 4.9% when alpha is correct. The hybrid approach combines the
best alpha recovery (symmetric) with the best k0 recovery (cathodic).

Usage (from PNPInverse/ directory)::

    python scripts/inference/Infer_BVHybrid_charged.py
"""

from __future__ import annotations

import csv
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts._bv_common import (
    setup_firedrake_env,
    K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2, K_SCALE, I_SCALE,
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
    run_bv_k0_flux_curve_inference,
    run_bv_alpha_flux_curve_inference,
    run_bv_joint_flux_curve_inference,
)
from Forward.steady_state import SteadyStateConfig

print_params_summary()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ===================================================================
    # Phase A: 12-point symmetric-focused range for alpha inference
    # ===================================================================
    # This is the "focused" placement from the positive_eta_inference_plan.
    # Includes anodic points for the (1-alpha) Tafel slope and enough
    # cathodic points to constrain the alpha Tafel slope.
    eta_symmetric = np.array([
        +5.0, +3.0, +1.0,           # anodic (3 points) -- pure R1
        -0.5,                         # near-equilibrium (1 point)
        -1.0, -2.0, -3.0,           # cathodic onset (3 points)
        -5.0, -8.0,                  # transition (2 points)
        -10.0, -15.0, -20.0,        # knee + plateau (3 points)
    ])

    # Phase B: 15-point cathodic extended range for k0 inference
    # Best for k0 recovery (staged inference S2 gives k0_1 = 4.9% error).
    eta_cathodic = np.array([
        -1.0, -2.0, -3.0,           # onset
        -4.0, -5.0, -6.5, -8.0,     # transition
        -10.0, -13.0,                # knee
        -17.0, -22.0, -28.0,        # plateau (sparse)
        -35.0, -41.0, -46.5,        # deep plateau
    ])

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

    # Deliberately wrong initial guesses
    initial_k0_guess = [0.005, 0.0005]
    initial_alpha_guess = [0.4, 0.3]

    observable_scale = -I_SCALE
    base_output = os.path.join("StudyResults", "bv_hybrid_inference_charged")
    os.makedirs(base_output, exist_ok=True)

    true_k0_arr = np.asarray(true_k0)
    true_alpha_arr = np.asarray(true_alpha)

    print(f"\n{'='*70}")
    print(f"  HYBRID Alpha+k0 Inference")
    print(f"  Phase A: alpha from {len(eta_symmetric)}-pt symmetric range "
          f"[{eta_symmetric.min():+.1f}, {eta_symmetric.max():+.1f}]")
    print(f"  Phase B: k0 from {len(eta_cathodic)}-pt cathodic range "
          f"[{eta_cathodic.min():.1f}, {eta_cathodic.max():.1f}]")
    print(f"{'='*70}\n")

    t_total_start = time.time()

    # -------------------------------------------------------------------
    # Phase A: Alpha-only inference on symmetric range
    # -------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  Phase A: Alpha inference (symmetric {len(eta_symmetric)}-pt, k0 fixed at true)")
    print(f"  This exploits the anodic (1-alpha) Tafel slope for R1.")
    print(f"{'='*70}")
    t_a = time.time()

    phase_a_dir = os.path.join(base_output, "phase_a_alpha_symmetric")

    request_a = BVFluxCurveInferenceRequest(
        base_solver_params=base_sp,
        steady=steady,
        true_k0=true_k0,
        initial_guess=true_k0,  # k0 not optimized, but needed for target gen
        phi_applied_values=eta_symmetric.tolist(),
        target_csv_path=os.path.join(phase_a_dir, "target.csv"),
        output_dir=phase_a_dir,
        regenerate_target=True,
        target_noise_percent=2.0,
        target_seed=20260226,
        observable_mode="current_density",
        current_density_scale=observable_scale,
        observable_label="current density (mA/cm2)",
        observable_title="Phase A: Alpha from symmetric data",
        control_mode="alpha",
        fixed_k0=true_k0,
        true_alpha=true_alpha,
        initial_alpha_guess=initial_alpha_guess,
        alpha_lower=0.05, alpha_upper=0.95,
        mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
        max_eta_gap=3.0,
        optimizer_method="L-BFGS-B",
        optimizer_options={"maxiter": 40, "ftol": 1e-12, "gtol": 1e-6, "disp": True},
        max_iters=40,
        live_plot=False,
        forward_recovery=make_recovery_config(max_it_cap=600),
    )

    result_a = run_bv_alpha_flux_curve_inference(request_a)
    phase_a_alpha = np.asarray(result_a["best_alpha"])
    phase_a_time = time.time() - t_a

    alpha_err = np.abs(phase_a_alpha - true_alpha_arr) / np.maximum(np.abs(true_alpha_arr), 1e-16)
    print(f"\n  Phase A result:")
    print(f"    alpha_1 = {phase_a_alpha[0]:.6f}  (true {ALPHA_R1:.6f}, err {alpha_err[0]*100:.2f}%)")
    print(f"    alpha_2 = {phase_a_alpha[1]:.6f}  (true {ALPHA_R2:.6f}, err {alpha_err[1]*100:.2f}%)")
    print(f"    Time: {phase_a_time:.1f}s")

    # -------------------------------------------------------------------
    # Phase B: k0 inference on cathodic range, alpha fixed from Phase A
    # -------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  Phase B: k0 inference (cathodic {len(eta_cathodic)}-pt, "
          f"alpha fixed from Phase A)")
    print(f"  Alpha_1 fixed at {phase_a_alpha[0]:.6f}")
    print(f"{'='*70}")
    t_b = time.time()

    phase_b_dir = os.path.join(base_output, "phase_b_k0_cathodic")

    request_b = BVFluxCurveInferenceRequest(
        base_solver_params=base_sp,
        steady=steady,
        true_k0=true_k0,
        initial_guess=initial_k0_guess,
        phi_applied_values=eta_cathodic.tolist(),
        target_csv_path=os.path.join(phase_b_dir, "target.csv"),
        output_dir=phase_b_dir,
        regenerate_target=True,
        target_noise_percent=2.0,
        target_seed=20260226,
        observable_mode="current_density",
        current_density_scale=observable_scale,
        observable_label="current density (mA/cm2)",
        observable_title="Phase B: k0 from cathodic data (alpha from Phase A)",
        control_mode="k0",
        k0_lower=1e-8, k0_upper=100.0,
        log_space=True,
        # Fix alpha at Phase A result
        true_alpha=phase_a_alpha.tolist(),
        initial_alpha_guess=phase_a_alpha.tolist(),
        mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
        max_eta_gap=3.0,
        optimizer_method="L-BFGS-B",
        optimizer_options={"maxiter": 30, "ftol": 1e-12, "gtol": 1e-6, "disp": True},
        max_iters=30,
        live_plot=False,
        forward_recovery=make_recovery_config(max_it_cap=600),
    )

    result_b = run_bv_k0_flux_curve_inference(request_b)
    phase_b_k0 = np.asarray(result_b["best_k0"])
    phase_b_time = time.time() - t_b

    k0_err = np.abs(phase_b_k0 - true_k0_arr) / np.maximum(np.abs(true_k0_arr), 1e-16)
    print(f"\n  Phase B result:")
    print(f"    k0_1 = {phase_b_k0[0]:.6e}  (true {K0_HAT_R1:.6e}, err {k0_err[0]*100:.2f}%)")
    print(f"    k0_2 = {phase_b_k0[1]:.6e}  (true {K0_HAT_R2:.6e}, err {k0_err[1]*100:.2f}%)")
    print(f"    Time: {phase_b_time:.1f}s")

    # -------------------------------------------------------------------
    # Phase C (optional): Joint refinement with hybrid warm-start
    # -------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  Phase C: Joint refinement (cathodic {len(eta_cathodic)}-pt, "
          f"warm-started from A+B)")
    print(f"{'='*70}")
    t_c = time.time()

    phase_c_dir = os.path.join(base_output, "phase_c_joint_refinement")

    request_c = BVFluxCurveInferenceRequest(
        base_solver_params=base_sp,
        steady=steady,
        true_k0=true_k0,
        initial_guess=phase_b_k0.tolist(),
        phi_applied_values=eta_cathodic.tolist(),
        target_csv_path=os.path.join(phase_c_dir, "target.csv"),
        output_dir=phase_c_dir,
        regenerate_target=True,
        target_noise_percent=2.0,
        target_seed=20260226,
        observable_mode="current_density",
        current_density_scale=observable_scale,
        observable_label="current density (mA/cm2)",
        observable_title="Phase C: Joint refinement (hybrid warm-start)",
        control_mode="joint",
        k0_lower=1e-8, k0_upper=100.0,
        log_space=True,
        true_alpha=true_alpha,
        initial_alpha_guess=phase_a_alpha.tolist(),
        alpha_lower=0.05, alpha_upper=0.95,
        mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
        max_eta_gap=3.0,
        optimizer_method="L-BFGS-B",
        optimizer_options={"maxiter": 20, "ftol": 1e-12, "gtol": 1e-6, "disp": True},
        max_iters=20,
        live_plot=False,
        forward_recovery=make_recovery_config(max_it_cap=600),
    )

    result_c = run_bv_joint_flux_curve_inference(request_c)
    phase_c_k0 = np.asarray(result_c["best_k0"])
    phase_c_alpha = np.asarray(result_c["best_alpha"])
    phase_c_time = time.time() - t_c

    k0_err_c = np.abs(phase_c_k0 - true_k0_arr) / np.maximum(np.abs(true_k0_arr), 1e-16)
    alpha_err_c = np.abs(phase_c_alpha - true_alpha_arr) / np.maximum(np.abs(true_alpha_arr), 1e-16)
    print(f"\n  Phase C result:")
    print(f"    k0_1 = {phase_c_k0[0]:.6e}  (err {k0_err_c[0]*100:.2f}%)")
    print(f"    k0_2 = {phase_c_k0[1]:.6e}  (err {k0_err_c[1]*100:.2f}%)")
    print(f"    alpha_1 = {phase_c_alpha[0]:.6f}  (err {alpha_err_c[0]*100:.2f}%)")
    print(f"    alpha_2 = {phase_c_alpha[1]:.6f}  (err {alpha_err_c[1]*100:.2f}%)")
    print(f"    Time: {phase_c_time:.1f}s")

    total_time = time.time() - t_total_start

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print(f"\n{'='*90}")
    print(f"  HYBRID INFERENCE SUMMARY")
    print(f"{'='*90}")
    print(f"  True k0:    {true_k0}")
    print(f"  True alpha: {true_alpha}")
    print()

    phases = {
        "Phase A (alpha, sym 12pt)": {
            "k0": true_k0, "alpha": phase_a_alpha.tolist(),
            "loss": result_a["best_loss"], "time": phase_a_time,
        },
        "Phase B (k0, cat 15pt)": {
            "k0": phase_b_k0.tolist(), "alpha": phase_a_alpha.tolist(),
            "loss": result_b["best_loss"], "time": phase_b_time,
        },
        "Phase C (joint refine)": {
            "k0": phase_c_k0.tolist(), "alpha": phase_c_alpha.tolist(),
            "loss": result_c["best_loss"], "time": phase_c_time,
        },
    }

    print(f"{'Phase':<30} | {'k0_1 err':>10} {'k0_2 err':>10} {'a1 err':>10} {'a2 err':>10} "
          f"| {'loss':>12} | {'time':>6}")
    print(f"{'-'*90}")

    for name, ph in phases.items():
        k0_arr = np.asarray(ph["k0"])
        al_arr = np.asarray(ph["alpha"])
        k0_e = np.abs(k0_arr - true_k0_arr) / np.maximum(np.abs(true_k0_arr), 1e-16)
        al_e = np.abs(al_arr - true_alpha_arr) / np.maximum(np.abs(true_alpha_arr), 1e-16)
        print(f"{name:<30} | {k0_e[0]:>10.4f} {k0_e[1]:>10.4f} "
              f"{al_e[0]:>10.4f} {al_e[1]:>10.4f} "
              f"| {ph['loss']:>12.6e} | {ph['time']:>5.0f}s")

    print(f"{'-'*90}")
    print(f"  Total time (A+B+C): {total_time:.0f}s")
    print()

    # Comparison with prior best results
    print(f"  Comparison with prior methods (R1 only):")
    print(f"    Staged cathodic 15pt:  k0_1=4.9%, alpha_1=0.1%")
    print(f"    Joint symmetric 20pt:  k0_1=296%(bound), alpha_1=1.8%")
    print(f"    Hybrid A+B:            k0_1={k0_err[0]*100:.1f}%, "
          f"alpha_1={alpha_err[0]*100:.1f}%")
    print(f"    Hybrid A+B+C:          k0_1={k0_err_c[0]*100:.1f}%, "
          f"alpha_1={alpha_err_c[0]*100:.1f}%")
    print(f"{'='*90}")

    # Save comparison CSV
    csv_path = os.path.join(base_output, "hybrid_comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "phase", "k0_1", "k0_2", "alpha_1", "alpha_2",
            "k0_1_err", "k0_2_err", "alpha_1_err", "alpha_2_err",
            "loss", "time_s",
        ])
        for name, ph in phases.items():
            k0_arr = np.asarray(ph["k0"])
            al_arr = np.asarray(ph["alpha"])
            k0_e = np.abs(k0_arr - true_k0_arr) / np.maximum(np.abs(true_k0_arr), 1e-16)
            al_e = np.abs(al_arr - true_alpha_arr) / np.maximum(np.abs(true_alpha_arr), 1e-16)
            writer.writerow([
                name,
                f"{k0_arr[0]:.8e}", f"{k0_arr[1]:.8e}",
                f"{al_arr[0]:.6f}", f"{al_arr[1]:.6f}",
                f"{k0_e[0]:.6f}", f"{k0_e[1]:.6f}",
                f"{al_e[0]:.6f}", f"{al_e[1]:.6f}",
                f"{ph['loss']:.12e}", f"{ph['time']:.1f}",
            ])
    print(f"\n[csv] Comparison saved -> {csv_path}")

    print(f"\n=== Hybrid Inference Complete ===")
    print(f"Output: {base_output}/")


if __name__ == "__main__":
    main()
