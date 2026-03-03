"""Staged inference -- EXTENDED voltage range.

Full 4-species charged system: O2, H2O2, H+, ClO4- (z=[0,0,+1,-1]).
Extended range: 15 inference points spanning eta_hat in [-1, -46.5].

Stages:
    Stage 1: Alpha inference (k0 fixed at true values) -> best_alpha_1
    Stage 2: k0 inference (alpha fixed at Stage 1 result) -> best_k0_2
    Stage 3: Joint refinement from Stage 1+2 warm start
    Stage 4: Direct joint inference (comparison baseline)

Usage (from PNPInverse/ directory)::

    python scripts/inference/Infer_BVStaged_charged_extended.py
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
    run_bv_k0_flux_curve_inference,
    run_bv_alpha_flux_curve_inference,
    run_bv_joint_flux_curve_inference,
)
from Forward.steady_state import SteadyStateConfig

def main() -> None:
    # Extended 15-point placement
    eta_values = np.array([
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

    initial_k0_guess = [0.005, 0.0005]
    initial_alpha_guess = [0.4, 0.3]

    observable_scale = -I_SCALE
    base_output = os.path.join("StudyResults", "bv_staged_inference_charged_extended")
    os.makedirs(base_output, exist_ok=True)

    stage_results = {}

    # ---- Stage 1: Alpha inference (k0 fixed) ----
    print(f"\n{'='*70}")
    print(f"  Stage 1: Alpha inference (k0 fixed at true values)")
    print(f"{'='*70}")
    t1 = time.time()

    stage1_dir = os.path.join(base_output, "stage1_alpha")

    request_s1 = BVFluxCurveInferenceRequest(
        base_solver_params=base_sp, steady=steady,
        true_k0=true_k0, initial_guess=true_k0,
        phi_applied_values=eta_values.tolist(),
        target_csv_path=os.path.join(stage1_dir, "target.csv"),
        output_dir=stage1_dir, regenerate_target=True,
        target_noise_percent=2.0, target_seed=20260226,
        observable_mode="current_density", current_density_scale=observable_scale,
        observable_label="current density (mA/cm2)",
        observable_title="Stage 1: Alpha inference (k0 fixed, extended)",
        control_mode="alpha", fixed_k0=true_k0,
        true_alpha=true_alpha, initial_alpha_guess=initial_alpha_guess,
        alpha_lower=0.05, alpha_upper=0.95,
        max_eta_gap=3.0,
        mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
        optimizer_method="L-BFGS-B",
        optimizer_options={"maxiter": 40, "ftol": 1e-12, "gtol": 1e-6, "disp": True},
        max_iters=40, live_plot=False, forward_recovery=make_recovery_config(max_it_cap=600),
    )

    result_s1 = run_bv_alpha_flux_curve_inference(request_s1)
    stage1_alpha = np.asarray(result_s1["best_alpha"])
    stage1_time = time.time() - t1
    print(f"\n  Stage 1 result: alpha = {stage1_alpha.tolist()}, time = {stage1_time:.1f}s")
    stage_results["stage1"] = {"alpha": stage1_alpha.tolist(), "k0": true_k0,
                                "loss": result_s1["best_loss"], "time": stage1_time}

    # ---- Stage 2: k0 inference (alpha fixed at Stage 1) ----
    print(f"\n{'='*70}")
    print(f"  Stage 2: k0 inference (alpha fixed at Stage 1 result)")
    print(f"{'='*70}")
    t2 = time.time()

    stage2_dir = os.path.join(base_output, "stage2_k0")

    request_s2 = BVFluxCurveInferenceRequest(
        base_solver_params=base_sp, steady=steady,
        true_k0=true_k0, initial_guess=initial_k0_guess,
        phi_applied_values=eta_values.tolist(),
        target_csv_path=os.path.join(stage2_dir, "target.csv"),
        output_dir=stage2_dir, regenerate_target=True,
        target_noise_percent=2.0, target_seed=20260226,
        observable_mode="current_density", current_density_scale=observable_scale,
        observable_label="current density (mA/cm2)",
        observable_title="Stage 2: k0 inference (alpha from Stage 1, extended)",
        control_mode="k0", k0_lower=1e-8, k0_upper=100.0, log_space=True,
        true_alpha=stage1_alpha.tolist(), initial_alpha_guess=stage1_alpha.tolist(),
        max_eta_gap=3.0,
        mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
        optimizer_method="L-BFGS-B",
        optimizer_options={"maxiter": 30, "ftol": 1e-12, "gtol": 1e-6, "disp": True},
        max_iters=30, live_plot=False, forward_recovery=make_recovery_config(max_it_cap=600),
    )

    result_s2 = run_bv_k0_flux_curve_inference(request_s2)
    stage2_k0 = np.asarray(result_s2["best_k0"])
    stage2_time = time.time() - t2
    print(f"\n  Stage 2 result: k0 = {stage2_k0.tolist()}, time = {stage2_time:.1f}s")
    stage_results["stage2"] = {"k0": stage2_k0.tolist(), "alpha": stage1_alpha.tolist(),
                                "loss": result_s2["best_loss"], "time": stage2_time}

    # ---- Stage 3: Joint refinement ----
    print(f"\n{'='*70}")
    print(f"  Stage 3: Joint refinement from Stage 1+2 warm start")
    print(f"{'='*70}")
    t3 = time.time()

    stage3_dir = os.path.join(base_output, "stage3_joint_warmstart")

    request_s3 = BVFluxCurveInferenceRequest(
        base_solver_params=base_sp, steady=steady,
        true_k0=true_k0, initial_guess=stage2_k0.tolist(),
        phi_applied_values=eta_values.tolist(),
        target_csv_path=os.path.join(stage3_dir, "target.csv"),
        output_dir=stage3_dir, regenerate_target=True,
        target_noise_percent=2.0, target_seed=20260226,
        observable_mode="current_density", current_density_scale=observable_scale,
        observable_label="current density (mA/cm2)",
        observable_title="Stage 3: Joint refinement (warm start, extended)",
        control_mode="joint", k0_lower=1e-8, k0_upper=100.0, log_space=True,
        true_alpha=true_alpha, initial_alpha_guess=stage1_alpha.tolist(),
        alpha_lower=0.05, alpha_upper=0.95,
        max_eta_gap=3.0,
        mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
        optimizer_method="L-BFGS-B",
        optimizer_options={"maxiter": 30, "ftol": 1e-12, "gtol": 1e-6, "disp": True},
        max_iters=30, live_plot=False, forward_recovery=make_recovery_config(max_it_cap=600),
    )

    result_s3 = run_bv_joint_flux_curve_inference(request_s3)
    stage3_k0 = np.asarray(result_s3["best_k0"])
    stage3_alpha = np.asarray(result_s3["best_alpha"])
    stage3_time = time.time() - t3
    print(f"\n  Stage 3 result: k0 = {stage3_k0.tolist()}, alpha = {stage3_alpha.tolist()}")
    stage_results["stage3"] = {"k0": stage3_k0.tolist(), "alpha": stage3_alpha.tolist(),
                                "loss": result_s3["best_loss"], "time": stage3_time}

    # ---- Stage 4: Direct joint inference ----
    print(f"\n{'='*70}")
    print(f"  Stage 4: Direct joint inference (baseline)")
    print(f"{'='*70}")
    t4 = time.time()

    stage4_dir = os.path.join(base_output, "stage4_direct_joint")

    request_s4 = BVFluxCurveInferenceRequest(
        base_solver_params=base_sp, steady=steady,
        true_k0=true_k0, initial_guess=initial_k0_guess,
        phi_applied_values=eta_values.tolist(),
        target_csv_path=os.path.join(stage4_dir, "target.csv"),
        output_dir=stage4_dir, regenerate_target=True,
        target_noise_percent=2.0, target_seed=20260226,
        observable_mode="current_density", current_density_scale=observable_scale,
        observable_label="current density (mA/cm2)",
        observable_title="Stage 4: Direct joint inference (extended)",
        control_mode="joint", k0_lower=1e-8, k0_upper=100.0, log_space=True,
        true_alpha=true_alpha, initial_alpha_guess=initial_alpha_guess,
        alpha_lower=0.05, alpha_upper=0.95,
        max_eta_gap=3.0,
        mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
        optimizer_method="L-BFGS-B",
        optimizer_options={"maxiter": 30, "ftol": 1e-12, "gtol": 1e-6, "disp": True},
        max_iters=30, live_plot=False, forward_recovery=make_recovery_config(max_it_cap=600),
    )

    result_s4 = run_bv_joint_flux_curve_inference(request_s4)
    stage4_k0 = np.asarray(result_s4["best_k0"])
    stage4_alpha = np.asarray(result_s4["best_alpha"])
    stage4_time = time.time() - t4
    stage_results["stage4"] = {"k0": stage4_k0.tolist(), "alpha": stage4_alpha.tolist(),
                                "loss": result_s4["best_loss"], "time": stage4_time}

    # ---- Summary ----
    true_k0_arr = np.asarray(true_k0)
    true_alpha_arr = np.asarray(true_alpha)
    total_staged_time = stage1_time + stage2_time + stage3_time

    print(f"\n{'='*90}")
    print(f"  Staged vs Direct (Extended Range)")
    print(f"{'='*90}")
    for stage_name, sr in stage_results.items():
        k0_arr = np.asarray(sr["k0"])
        alpha_arr = np.asarray(sr["alpha"])
        k0_err = np.abs(k0_arr - true_k0_arr) / np.maximum(np.abs(true_k0_arr), 1e-16)
        alpha_err = np.abs(alpha_arr - true_alpha_arr) / np.maximum(np.abs(true_alpha_arr), 1e-16)
        print(f"{stage_name:<20} | k0_err={k0_err.tolist()} alpha_err={alpha_err.tolist()} "
              f"| loss={sr['loss']:.6e} | {sr['time']:.0f}s")
    print(f"  Staged total: {total_staged_time:.0f}s, Direct: {stage4_time:.0f}s")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
