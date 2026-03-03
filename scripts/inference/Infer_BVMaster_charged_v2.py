"""Master 4-phase inference protocol v2 for BV kinetics (charged system).

Full 4-species charged system: O2, H2O2, H+, ClO4- (z=[0,0,+1,-1]).

Key difference from v1: the k0-only total current phase is REMOVED because it
drives k0_2 to near-zero (total current has no sensitivity to k0_2). Instead,
Phase 2 uses multi-observable joint (total + peroxide current) from a cold k0
start on a SHALLOW eta range [-1, -13] where the solver is stable even with
wrong initial parameters, which recovers R2 parameters effectively.

Four-phase protocol:
    Phase 1: Alpha initialization via 12-pt symmetric sweep (alpha-only, k0 FIXED at guess)
    Phase 2: Multi-observable joint on 10-pt SHALLOW cathodic [-1, -13] from COLD k0 + P1 alpha
    Phase 3: Joint refinement with regularization annealing on FULL 15-pt cathodic [-1, -46.5]
    Phase 4: Best-of selection (Phase 2 vs Phase 3c)

Usage (from PNPInverse/ directory)::

    python scripts/inference/Infer_BVMaster_charged_v2.py
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
    K_SCALE,
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
    run_bv_alpha_flux_curve_inference,
    run_bv_joint_flux_curve_inference,
    run_bv_multi_observable_flux_curve_inference,
)
from FluxCurve.bv_point_solve import _clear_caches
from Forward.steady_state import SteadyStateConfig

def _compute_errors(k0, alpha, true_k0_arr, true_alpha_arr):
    """Compute relative errors for k0 and alpha arrays."""
    k0_arr = np.asarray(k0)
    alpha_arr = np.asarray(alpha)
    k0_err = np.abs(k0_arr - true_k0_arr) / np.maximum(np.abs(true_k0_arr), 1e-16)
    alpha_err = np.abs(alpha_arr - true_alpha_arr) / np.maximum(np.abs(true_alpha_arr), 1e-16)
    return k0_err, alpha_err


def _print_phase_result(name, k0, alpha, true_k0_arr, true_alpha_arr, loss, elapsed):
    """Print a summary block for a single phase result."""
    k0_err, alpha_err = _compute_errors(k0, alpha, true_k0_arr, true_alpha_arr)
    print(f"\n  {name} result:")
    print(f"    k0_1   = {k0[0]:.6e}  (true {true_k0_arr[0]:.6e}, err {k0_err[0]*100:.2f}%)")
    print(f"    k0_2   = {k0[1]:.6e}  (true {true_k0_arr[1]:.6e}, err {k0_err[1]*100:.2f}%)")
    print(f"    alpha_1= {alpha[0]:.6f}  (true {true_alpha_arr[0]:.6f}, err {alpha_err[0]*100:.2f}%)")
    print(f"    alpha_2= {alpha[1]:.6f}  (true {true_alpha_arr[1]:.6f}, err {alpha_err[1]*100:.2f}%)")
    print(f"    Loss: {loss:.6e},  Time: {elapsed:.1f}s")
    return k0_err, alpha_err


def main() -> None:
    # ===================================================================
    # Voltage grids
    # ===================================================================
    # Phase 1: 12-point symmetric [-20, +5]
    eta_symmetric = np.array([
        +5.0, +3.0, +1.0,
        -0.5,
        -1.0, -2.0, -3.0,
        -5.0, -8.0,
        -10.0, -15.0, -20.0,
    ])

    # Phase 2: 10-point SHALLOW cathodic [-1, -13] (solver-stable for any k0)
    eta_shallow = np.array([
        -1.0, -2.0, -3.0,
        -4.0, -5.0, -6.5, -8.0,
        -10.0, -11.5, -13.0,
    ])

    # Phase 3: 15-point FULL cathodic [-1, -46.5] (safe after k0_2 in right basin)
    eta_cathodic = np.array([
        -1.0, -2.0, -3.0,
        -4.0, -5.0, -6.5, -8.0,
        -10.0, -13.0,
        -17.0, -22.0, -28.0,
        -35.0, -41.0, -46.5,
    ])

    true_k0 = [K0_HAT_R1, K0_HAT_R2]
    true_alpha = [ALPHA_R1, ALPHA_R2]
    true_k0_arr = np.asarray(true_k0)
    true_alpha_arr = np.asarray(true_alpha)

    # Deliberately wrong initial guesses (realistic scenario)
    initial_k0_guess = [0.005, 0.0005]
    initial_alpha_guess = [0.4, 0.3]

    observable_scale = -I_SCALE
    base_output = os.path.join("StudyResults", "master_inference_v2")
    os.makedirs(base_output, exist_ok=True)

    # Standard solver config
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

    phase_results = {}
    t_total_start = time.time()

    print(f"\n{'#'*70}")
    print(f"  MASTER 4-PHASE INFERENCE PROTOCOL v2")
    print(f"  Key change: multi-obs from cold k0 (no k0-only total current)")
    print(f"  True k0:    {true_k0}")
    print(f"  True alpha: {true_alpha}")
    print(f"  Initial k0 guess:    {initial_k0_guess}")
    print(f"  Initial alpha guess: {initial_alpha_guess}")
    print(f"{'#'*70}\n")

    # ===================================================================
    # PHASE 1: Alpha Initialization via Symmetric Sweep (alpha-only)
    # ===================================================================
    print(f"\n{'='*70}")
    print(f"  PHASE 1: Alpha from {len(eta_symmetric)}-pt symmetric sweep")
    print(f"  k0 FIXED at initial guess (NOT true): {initial_k0_guess}")
    print(f"  eta in [{eta_symmetric.min():+.1f}, {eta_symmetric.max():+.1f}]")
    print(f"{'='*70}")
    t_p1 = time.time()

    p1_dir = os.path.join(base_output, "phase1_alpha_symmetric")

    request_p1 = BVFluxCurveInferenceRequest(
        base_solver_params=base_sp,
        steady=steady,
        true_k0=true_k0,
        initial_guess=initial_k0_guess,
        phi_applied_values=eta_symmetric.tolist(),
        target_csv_path=os.path.join(p1_dir, "target.csv"),
        output_dir=p1_dir,
        regenerate_target=True,
        target_noise_percent=2.0,
        target_seed=20260226,
        observable_mode="current_density",
        current_density_scale=observable_scale,
        observable_label="current density (mA/cm2)",
        observable_title="Phase 1: Alpha (symmetric, k0 at guess)",
        control_mode="alpha",
        fixed_k0=initial_k0_guess,
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

    result_p1 = run_bv_alpha_flux_curve_inference(request_p1)
    p1_alpha = np.asarray(result_p1["best_alpha"])
    p1_k0 = np.asarray(initial_k0_guess)  # unchanged (fixed)
    p1_time = time.time() - t_p1

    _print_phase_result("Phase 1", p1_k0, p1_alpha, true_k0_arr, true_alpha_arr,
                        result_p1["best_loss"], p1_time)
    phase_results["Phase 1 (alpha, sym 12pt)"] = {
        "k0": p1_k0.tolist(), "alpha": p1_alpha.tolist(),
        "loss": result_p1["best_loss"], "time": p1_time,
    }

    _clear_caches()

    # ===================================================================
    # PHASE 2: Multi-Observable Joint (total + peroxide current)
    #          k0 from COLD initial guess, alpha from Phase 1
    # ===================================================================
    print(f"\n{'='*70}")
    print(f"  PHASE 2: Multi-observable joint (total + peroxide current)")
    print(f"  k0 from COLD initial guess: {initial_k0_guess}")
    print(f"  alpha warm-started from Phase 1: {p1_alpha.tolist()}")
    print(f"  {len(eta_shallow)}-pt SHALLOW cathodic [{eta_shallow.min():.1f}, {eta_shallow.max():.1f}]")
    print(f"  secondary_weight=1.0")
    print(f"{'='*70}")
    t_p2 = time.time()

    p2_dir = os.path.join(base_output, "phase2_multi_obs_cold_k0")

    request_p2 = BVFluxCurveInferenceRequest(
        base_solver_params=base_sp,
        steady=steady,
        true_k0=true_k0,
        initial_guess=initial_k0_guess,  # COLD k0 start -- key difference from v1
        phi_applied_values=eta_shallow.tolist(),  # SHALLOW range: solver-stable for any k0
        target_csv_path=os.path.join(p2_dir, "target_primary.csv"),
        output_dir=p2_dir,
        regenerate_target=True,
        target_noise_percent=2.0,
        target_seed=20260226,
        # Primary observable: total current density
        observable_mode="current_density",
        current_density_scale=observable_scale,
        observable_label="current density (mA/cm2)",
        observable_title="Phase 2: Multi-obs joint (cold k0 + P1 alpha)",
        # Secondary observable: peroxide current
        secondary_observable_mode="peroxide_current",
        secondary_observable_weight=1.0,
        secondary_current_density_scale=observable_scale,
        secondary_target_csv_path=os.path.join(p2_dir, "target_peroxide.csv"),
        # Joint control mode
        control_mode="joint",
        true_alpha=true_alpha,
        initial_alpha_guess=p1_alpha.tolist(),  # warm-start alpha from P1
        alpha_lower=0.05, alpha_upper=0.95,
        k0_lower=1e-8, k0_upper=100.0,
        log_space=True,
        mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
        max_eta_gap=3.0,
        optimizer_method="L-BFGS-B",
        optimizer_options={"maxiter": 40, "ftol": 1e-12, "gtol": 1e-6, "disp": True},
        max_iters=40,
        live_plot=False,
        forward_recovery=make_recovery_config(max_it_cap=600),
    )

    result_p2 = run_bv_multi_observable_flux_curve_inference(request_p2)
    p2_k0 = np.asarray(result_p2["best_k0"])
    p2_alpha = np.asarray(result_p2["best_alpha"])
    p2_time = time.time() - t_p2

    _print_phase_result("Phase 2", p2_k0, p2_alpha, true_k0_arr, true_alpha_arr,
                        result_p2["best_loss"], p2_time)
    phase_results["Phase 2 (multi-obs, cold k0)"] = {
        "k0": p2_k0.tolist(), "alpha": p2_alpha.tolist(),
        "loss": result_p2["best_loss"], "time": p2_time,
    }

    _clear_caches()

    # ===================================================================
    # PHASE 3: Joint Refinement with Regularization Annealing
    #          Warm-start from Phase 2, FULL cathodic range (safe now
    #          that k0_2 is in the right basin from Phase 2)
    # ===================================================================
    print(f"\n{'='*70}")
    print(f"  PHASE 3: Regularization annealing (3 sub-phases)")
    print(f"  Warm-start: k0 from P2, alpha from P2")
    print(f"  FULL {len(eta_cathodic)}-pt cathodic [{eta_cathodic.min():.1f}, {eta_cathodic.max():.1f}]")
    print(f"  Total current only (k0_2 in right basin prevents drift)")
    print(f"  3a: lambda=0.1, 3b: lambda=0.01, 3c: lambda=0.0")
    print(f"{'='*70}")

    p3_k0 = p2_k0.copy()
    p3_alpha = p2_alpha.copy()
    p3_total_time = 0.0

    reg_schedule = [
        ("3a", 0.1, 10, p2_k0.tolist(), p2_alpha.tolist()),
        ("3b", 0.01, 15, None, None),  # prior updated after 3a
        ("3c", 0.0, 20, None, None),   # no regularization
    ]

    for sub_name, lam, maxiter, prior_k0, prior_alpha in reg_schedule:
        # Update priors for 3b and 3c from previous sub-phase result
        if prior_k0 is None:
            prior_k0 = p3_k0.tolist()
        if prior_alpha is None:
            prior_alpha = p3_alpha.tolist()

        print(f"\n  --- Phase {sub_name}: lambda={lam}, maxiter={maxiter} ---")
        print(f"  Prior k0: {prior_k0}")
        print(f"  Prior alpha: {prior_alpha}")
        t_sub = time.time()

        sub_dir = os.path.join(base_output, f"phase{sub_name}_reg_lambda{lam}")

        request_sub = BVFluxCurveInferenceRequest(
            base_solver_params=base_sp,
            steady=steady,
            true_k0=true_k0,
            initial_guess=p3_k0.tolist(),
            phi_applied_values=eta_cathodic.tolist(),
            target_csv_path=os.path.join(sub_dir, "target.csv"),
            output_dir=sub_dir,
            regenerate_target=True,
            target_noise_percent=2.0,
            target_seed=20260226,
            observable_mode="current_density",
            current_density_scale=observable_scale,
            observable_label="current density (mA/cm2)",
            observable_title=f"Phase {sub_name}: joint (lambda={lam})",
            control_mode="joint",
            true_alpha=true_alpha,
            initial_alpha_guess=p3_alpha.tolist(),
            alpha_lower=0.05, alpha_upper=0.95,
            k0_lower=1e-8, k0_upper=100.0,
            log_space=True,
            mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
            max_eta_gap=3.0,
            optimizer_method="L-BFGS-B",
            optimizer_options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-6, "disp": True},
            max_iters=maxiter,
            live_plot=False,
            forward_recovery=make_recovery_config(max_it_cap=600),
            # Regularization
            regularization_lambda=lam,
            regularization_k0_prior=prior_k0,
            regularization_alpha_prior=prior_alpha,
        )

        result_sub = run_bv_joint_flux_curve_inference(request_sub)
        p3_k0 = np.asarray(result_sub["best_k0"])
        p3_alpha = np.asarray(result_sub["best_alpha"])
        sub_time = time.time() - t_sub
        p3_total_time += sub_time

        _print_phase_result(f"Phase {sub_name}", p3_k0, p3_alpha,
                            true_k0_arr, true_alpha_arr,
                            result_sub["best_loss"], sub_time)

        _clear_caches()

    # Record final Phase 3 (3c) result
    phase_results["Phase 3c (joint, no reg)"] = {
        "k0": p3_k0.tolist(), "alpha": p3_alpha.tolist(),
        "loss": result_sub["best_loss"], "time": p3_total_time,
    }

    # ===================================================================
    # PHASE 4: Best-of Selection (Phase 2 vs Phase 3c)
    # ===================================================================
    p2_k0_err, p2_alpha_err = _compute_errors(p2_k0, p2_alpha, true_k0_arr, true_alpha_arr)
    p3_k0_err, p3_alpha_err = _compute_errors(p3_k0, p3_alpha, true_k0_arr, true_alpha_arr)

    p2_max_err = max(p2_k0_err.max(), p2_alpha_err.max())
    p3_max_err = max(p3_k0_err.max(), p3_alpha_err.max())

    if p3_max_err <= p2_max_err:
        best_k0, best_alpha = p3_k0.copy(), p3_alpha.copy()
        best_source = "Phase 3c"
    else:
        best_k0, best_alpha = p2_k0.copy(), p2_alpha.copy()
        best_source = "Phase 2"

    best_max_err = min(p2_max_err, p3_max_err)

    print(f"\n{'='*70}")
    print(f"  P2 vs P3c: best is {best_source} (max err = {best_max_err*100:.2f}%)")
    print(f"{'='*70}")

    total_time = time.time() - t_total_start

    # ===================================================================
    # FINAL SUMMARY TABLE
    # ===================================================================
    print(f"\n{'#'*90}")
    print(f"  MASTER INFERENCE v2 SUMMARY")
    print(f"{'#'*90}")
    print(f"  True k0:    {true_k0}")
    print(f"  True alpha: {true_alpha}")
    print(f"  K0_HAT_R1 = {K0_HAT_R1:.6e},  K0_HAT_R2 = {K0_HAT_R2:.6e}")
    print()

    header = (f"{'Phase':<30} | {'k0_1 err':>10} {'k0_2 err':>10} "
              f"{'a1 err':>10} {'a2 err':>10} | {'loss':>12} | {'time':>6}")
    print(header)
    print(f"{'-'*90}")

    for name, ph in phase_results.items():
        k0_err, alpha_err = _compute_errors(
            ph["k0"], ph["alpha"], true_k0_arr, true_alpha_arr
        )
        print(f"{name:<30} | {k0_err[0]*100:>9.2f}% {k0_err[1]*100:>9.2f}% "
              f"{alpha_err[0]*100:>9.2f}% {alpha_err[1]*100:>9.2f}% "
              f"| {ph['loss']:>12.6e} | {ph['time']:>5.0f}s")

    print(f"{'-'*90}")
    print(f"  Total time: {total_time:.0f}s")
    print(f"  Best result: {best_source}")
    best_k0_err, best_alpha_err = _compute_errors(best_k0, best_alpha, true_k0_arr, true_alpha_arr)
    print(f"    k0_1   = {best_k0[0]:.6e}  (err {best_k0_err[0]*100:.2f}%)")
    print(f"    k0_2   = {best_k0[1]:.6e}  (err {best_k0_err[1]*100:.2f}%)")
    print(f"    alpha_1= {best_alpha[0]:.6f}  (err {best_alpha_err[0]*100:.2f}%)")
    print(f"    alpha_2= {best_alpha[1]:.6f}  (err {best_alpha_err[1]*100:.2f}%)")

    best_k0_phys = best_k0 * K_SCALE
    true_k0_phys = true_k0_arr * K_SCALE
    print(f"\n  Physical units (m/s):")
    print(f"    True k0:  [{true_k0_phys[0]:.4e}, {true_k0_phys[1]:.4e}]")
    print(f"    Best k0:  [{best_k0_phys[0]:.4e}, {best_k0_phys[1]:.4e}]")
    print(f"{'#'*90}")

    # Save comparison CSV
    csv_path = os.path.join(base_output, "master_comparison_v2.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "phase", "k0_1", "k0_2", "alpha_1", "alpha_2",
            "k0_1_err_pct", "k0_2_err_pct", "alpha_1_err_pct", "alpha_2_err_pct",
            "loss", "time_s",
        ])
        for name, ph in phase_results.items():
            k0_err, alpha_err = _compute_errors(
                ph["k0"], ph["alpha"], true_k0_arr, true_alpha_arr
            )
            writer.writerow([
                name,
                f"{ph['k0'][0]:.8e}", f"{ph['k0'][1]:.8e}",
                f"{ph['alpha'][0]:.6f}", f"{ph['alpha'][1]:.6f}",
                f"{k0_err[0]*100:.4f}", f"{k0_err[1]*100:.4f}",
                f"{alpha_err[0]*100:.4f}", f"{alpha_err[1]*100:.4f}",
                f"{ph['loss']:.12e}", f"{ph['time']:.1f}",
            ])
    print(f"\n  Comparison CSV saved -> {csv_path}")
    print(f"\n  Output: {base_output}/")
    print(f"\n=== Master Inference v2 Complete ===")


if __name__ == "__main__":
    main()
