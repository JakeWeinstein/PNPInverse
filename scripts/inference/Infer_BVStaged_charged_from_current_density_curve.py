"""Staged inference: alpha -> k0 -> joint refinement vs direct joint.

Full 4-species charged system: O2, H2O2, H+, ClO4- (z=[0,0,+1,-1]).

Stages:
    Stage 1: Alpha inference (k0 fixed at true values) -> best_alpha_1
    Stage 2: k0 inference (alpha fixed at Stage 1 result) -> best_k0_2
    Stage 3: Joint refinement from Stage 1+2 warm start -> best_k0_3, best_alpha_3
    Stage 4: Direct joint inference (comparison baseline) -> best_k0_4, best_alpha_4

Compares staged (1+2+3) vs direct (4) in terms of parameter errors
and final objective values.

Usage (from PNPInverse/ directory)::

    python scripts/inference/Infer_BVStaged_charged_from_current_density_curve.py
"""

from __future__ import annotations

import csv
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PNPINVERSE_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _PNPINVERSE_ROOT not in sys.path:
    sys.path.insert(0, _PNPINVERSE_ROOT)

os.environ.setdefault("FIREDRAKE_TSFC_KERNEL_CACHE_DIR", "/tmp/firedrake-tsfc")
os.environ.setdefault("PYOP2_CACHE_DIR", "/tmp/pyop2")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from FluxCurve import (
    BVFluxCurveInferenceRequest,
    ForwardRecoveryConfig,
    run_bv_k0_flux_curve_inference,
    run_bv_alpha_flux_curve_inference,
    run_bv_joint_flux_curve_inference,
)
from Forward.params import SolverParams
from Forward.steady_state import SteadyStateConfig


# ---------------------------------------------------------------------------
# Physical constants and scales
# ---------------------------------------------------------------------------

F_CONST = 96485.3329
R_GAS = 8.31446
T_REF = 298.15
V_T = R_GAS * T_REF / F_CONST
N_ELECTRONS = 2

D_O2 = 1.9e-9;  C_O2 = 0.5
D_H2O2 = 1.6e-9; C_H2O2 = 0.0
D_HP = 9.311e-9; C_HP = 0.1
D_CLO4 = 1.792e-9; C_CLO4 = 0.1

K0_PHYS = 2.4e-8; ALPHA_1 = 0.627
K0_2_PHYS = 1e-9; ALPHA_2 = 0.5

L_REF = 1.0e-4; D_REF = D_O2; C_SCALE = C_O2; K_SCALE = D_REF / L_REF

D_O2_HAT = D_O2 / D_REF; D_H2O2_HAT = D_H2O2 / D_REF
D_HP_HAT = D_HP / D_REF; D_CLO4_HAT = D_CLO4 / D_REF
C_O2_HAT = C_O2 / C_SCALE; C_H2O2_HAT = C_H2O2 / C_SCALE
C_HP_HAT = C_HP / C_SCALE; C_CLO4_HAT = C_CLO4 / C_SCALE

K0_HAT = K0_PHYS / K_SCALE
K0_2_HAT = K0_2_PHYS / K_SCALE

I_SCALE = N_ELECTRONS * F_CONST * D_REF * C_SCALE / L_REF * 0.1

SNES_OPTS = {
    "snes_type":                 "newtonls",
    "snes_max_it":               300,
    "snes_atol":                 1e-7,
    "snes_rtol":                 1e-10,
    "snes_stol":                 1e-12,
    "snes_linesearch_type":      "l2",
    "snes_linesearch_maxlambda": 0.5,
    "snes_divergence_tolerance": 1e12,
    "ksp_type":                  "preonly",
    "pc_type":                   "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_mumps_icntl_8":         77,
    "mat_mumps_icntl_14":        80,
}


def _make_bv_solver_params(eta_hat: float, dt: float, t_end: float) -> SolverParams:
    """Build SolverParams for 4-species charged BV."""
    params = dict(SNES_OPTS)
    params["bv_convergence"] = {
        "clip_exponent": True, "exponent_clip": 50.0,
        "regularize_concentration": True, "conc_floor": 1e-12,
        "use_eta_in_bv": True,
    }
    params["nondim"] = {
        "enabled": True,
        "diffusivity_scale_m2_s": D_REF,
        "concentration_scale_mol_m3": C_SCALE,
        "length_scale_m": L_REF,
        "potential_scale_v": V_T,
        "kappa_inputs_are_dimensionless": True,
        "diffusivity_inputs_are_dimensionless": True,
        "concentration_inputs_are_dimensionless": True,
        "potential_inputs_are_dimensionless": True,
        "time_inputs_are_dimensionless": True,
    }
    params["bv_bc"] = {
        "reactions": [
            {
                "k0": K0_HAT, "alpha": ALPHA_1,
                "cathodic_species": 0, "anodic_species": 1,
                "c_ref": 1.0, "stoichiometry": [-1, +1, -2, 0],
                "n_electrons": 2, "reversible": True,
                "cathodic_conc_factors": [
                    {"species": 2, "power": 2, "c_ref_nondim": C_HP_HAT},
                ],
            },
            {
                "k0": K0_2_HAT, "alpha": ALPHA_2,
                "cathodic_species": 1, "anodic_species": None,
                "c_ref": 0.0, "stoichiometry": [0, -1, -2, 0],
                "n_electrons": 2, "reversible": False,
                "cathodic_conc_factors": [
                    {"species": 2, "power": 2, "c_ref_nondim": C_HP_HAT},
                ],
            },
        ],
        "k0": [K0_HAT] * 4, "alpha": [ALPHA_1] * 4,
        "stoichiometry": [-1, +1, -2, 0], "c_ref": [1.0] * 4,
        "E_eq_v": 0.0,
        "electrode_marker": 3, "concentration_marker": 4, "ground_marker": 4,
    }
    return SolverParams.from_list([
        4, 1, dt, t_end, [0, 0, 1, -1],
        [D_O2_HAT, D_H2O2_HAT, D_HP_HAT, D_CLO4_HAT],
        [0.01, 0.01, 0.01, 0.01],
        eta_hat,
        [C_O2_HAT, C_H2O2_HAT, C_HP_HAT, C_CLO4_HAT],
        0.0, params,
    ])


def _make_recovery():
    return ForwardRecoveryConfig(
        max_attempts=6, max_it_only_attempts=2,
        anisotropy_only_attempts=1, tolerance_relax_attempts=2,
        max_it_growth=1.5, max_it_cap=600,
    )


def main() -> None:
    eta_values = np.linspace(-1.0, -10.0, 10)
    dt = 0.5
    max_ss_steps = 100
    t_end = dt * max_ss_steps
    base_sp = _make_bv_solver_params(eta_hat=0.0, dt=dt, t_end=t_end)

    steady = SteadyStateConfig(
        relative_tolerance=1e-4, absolute_tolerance=1e-8,
        consecutive_steps=4, max_steps=max_ss_steps,
        flux_observable="total_species", verbose=False,
    )

    true_k0 = [K0_HAT, K0_2_HAT]
    true_alpha = [ALPHA_1, ALPHA_2]

    # Deliberately wrong initial guesses (same as direct joint script)
    initial_k0_guess = [0.005, 0.0005]
    initial_alpha_guess = [0.4, 0.3]

    observable_scale = -I_SCALE
    base_output = os.path.join("StudyResults", "bv_staged_inference_charged")
    os.makedirs(base_output, exist_ok=True)

    stage_results = {}

    # -----------------------------------------------------------------------
    # Stage 1: Alpha inference with k0 fixed at TRUE values
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  Stage 1: Alpha inference (k0 fixed at true values)")
    print(f"{'='*70}")
    t1 = time.time()

    stage1_dir = os.path.join(base_output, "stage1_alpha")

    request_s1 = BVFluxCurveInferenceRequest(
        base_solver_params=base_sp,
        steady=steady,
        true_k0=true_k0,
        initial_guess=true_k0,  # k0 are fixed, not optimized
        phi_applied_values=eta_values.tolist(),
        target_csv_path=os.path.join(stage1_dir, "target.csv"),
        output_dir=stage1_dir,
        regenerate_target=True,
        target_noise_percent=2.0,
        target_seed=20260226,
        observable_mode="current_density",
        current_density_scale=observable_scale,
        observable_label="current density (mA/cm2)",
        observable_title="Stage 1: Alpha inference (k0 fixed)",
        control_mode="alpha",
        fixed_k0=true_k0,
        true_alpha=true_alpha,
        initial_alpha_guess=initial_alpha_guess,
        alpha_lower=0.05, alpha_upper=0.95,
        mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
        optimizer_method="L-BFGS-B",
        optimizer_options={"maxiter": 40, "ftol": 1e-12, "gtol": 1e-6, "disp": True},
        max_iters=40,
        live_plot=False,
        forward_recovery=_make_recovery(),
    )

    result_s1 = run_bv_alpha_flux_curve_inference(request_s1)
    stage1_alpha = np.asarray(result_s1["best_alpha"])
    stage1_time = time.time() - t1

    print(f"\n  Stage 1 result: alpha = {stage1_alpha.tolist()}")
    print(f"  Time: {stage1_time:.1f}s")
    stage_results["stage1"] = {
        "alpha": stage1_alpha.tolist(),
        "k0": true_k0,
        "loss": result_s1["best_loss"],
        "time": stage1_time,
    }

    # -----------------------------------------------------------------------
    # Stage 2: k0 inference with alpha fixed at Stage 1 result
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  Stage 2: k0 inference (alpha fixed at Stage 1 result)")
    print(f"{'='*70}")
    t2 = time.time()

    stage2_dir = os.path.join(base_output, "stage2_k0")

    # Need to inject Stage 1 alpha into solver params for target generation
    # The bv_run pipeline uses true_k0 for target generation, so we just
    # need the correct initial k0 guess
    request_s2 = BVFluxCurveInferenceRequest(
        base_solver_params=base_sp,
        steady=steady,
        true_k0=true_k0,
        initial_guess=initial_k0_guess,
        phi_applied_values=eta_values.tolist(),
        target_csv_path=os.path.join(stage2_dir, "target.csv"),
        output_dir=stage2_dir,
        regenerate_target=True,
        target_noise_percent=2.0,
        target_seed=20260226,
        observable_mode="current_density",
        current_density_scale=observable_scale,
        observable_label="current density (mA/cm2)",
        observable_title="Stage 2: k0 inference (alpha from Stage 1)",
        control_mode="k0",
        k0_lower=1e-8, k0_upper=100.0,
        log_space=True,
        # Alpha is fixed at Stage 1 result (injected into solver params via true_alpha)
        true_alpha=stage1_alpha.tolist(),
        initial_alpha_guess=stage1_alpha.tolist(),
        mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
        optimizer_method="L-BFGS-B",
        optimizer_options={"maxiter": 30, "ftol": 1e-12, "gtol": 1e-6, "disp": True},
        max_iters=30,
        live_plot=False,
        forward_recovery=_make_recovery(),
    )

    result_s2 = run_bv_k0_flux_curve_inference(request_s2)
    stage2_k0 = np.asarray(result_s2["best_k0"])
    stage2_time = time.time() - t2

    print(f"\n  Stage 2 result: k0 = {stage2_k0.tolist()}")
    print(f"  Time: {stage2_time:.1f}s")
    stage_results["stage2"] = {
        "k0": stage2_k0.tolist(),
        "alpha": stage1_alpha.tolist(),
        "loss": result_s2["best_loss"],
        "time": stage2_time,
    }

    # -----------------------------------------------------------------------
    # Stage 3: Joint refinement from Stage 1+2 warm start
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  Stage 3: Joint refinement from Stage 1+2 warm start")
    print(f"{'='*70}")
    t3 = time.time()

    stage3_dir = os.path.join(base_output, "stage3_joint_warmstart")

    request_s3 = BVFluxCurveInferenceRequest(
        base_solver_params=base_sp,
        steady=steady,
        true_k0=true_k0,
        initial_guess=stage2_k0.tolist(),  # warm start from Stage 2
        phi_applied_values=eta_values.tolist(),
        target_csv_path=os.path.join(stage3_dir, "target.csv"),
        output_dir=stage3_dir,
        regenerate_target=True,
        target_noise_percent=2.0,
        target_seed=20260226,
        observable_mode="current_density",
        current_density_scale=observable_scale,
        observable_label="current density (mA/cm2)",
        observable_title="Stage 3: Joint refinement (warm start)",
        control_mode="joint",
        k0_lower=1e-8, k0_upper=100.0,
        log_space=True,
        true_alpha=true_alpha,
        initial_alpha_guess=stage1_alpha.tolist(),  # warm start from Stage 1
        alpha_lower=0.05, alpha_upper=0.95,
        mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
        optimizer_method="L-BFGS-B",
        optimizer_options={"maxiter": 30, "ftol": 1e-12, "gtol": 1e-6, "disp": True},
        max_iters=30,
        live_plot=False,
        forward_recovery=_make_recovery(),
    )

    result_s3 = run_bv_joint_flux_curve_inference(request_s3)
    stage3_k0 = np.asarray(result_s3["best_k0"])
    stage3_alpha = np.asarray(result_s3["best_alpha"])
    stage3_time = time.time() - t3

    print(f"\n  Stage 3 result: k0 = {stage3_k0.tolist()}, alpha = {stage3_alpha.tolist()}")
    print(f"  Time: {stage3_time:.1f}s")
    stage_results["stage3"] = {
        "k0": stage3_k0.tolist(),
        "alpha": stage3_alpha.tolist(),
        "loss": result_s3["best_loss"],
        "time": stage3_time,
    }

    # -----------------------------------------------------------------------
    # Stage 4: Direct joint inference (baseline comparison)
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  Stage 4: Direct joint inference (baseline)")
    print(f"{'='*70}")
    t4 = time.time()

    stage4_dir = os.path.join(base_output, "stage4_direct_joint")

    request_s4 = BVFluxCurveInferenceRequest(
        base_solver_params=base_sp,
        steady=steady,
        true_k0=true_k0,
        initial_guess=initial_k0_guess,  # same wrong initial as Stage 2
        phi_applied_values=eta_values.tolist(),
        target_csv_path=os.path.join(stage4_dir, "target.csv"),
        output_dir=stage4_dir,
        regenerate_target=True,
        target_noise_percent=2.0,
        target_seed=20260226,
        observable_mode="current_density",
        current_density_scale=observable_scale,
        observable_label="current density (mA/cm2)",
        observable_title="Stage 4: Direct joint inference",
        control_mode="joint",
        k0_lower=1e-8, k0_upper=100.0,
        log_space=True,
        true_alpha=true_alpha,
        initial_alpha_guess=initial_alpha_guess,  # same wrong initial
        alpha_lower=0.05, alpha_upper=0.95,
        mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
        optimizer_method="L-BFGS-B",
        optimizer_options={"maxiter": 30, "ftol": 1e-12, "gtol": 1e-6, "disp": True},
        max_iters=30,
        live_plot=False,
        forward_recovery=_make_recovery(),
    )

    result_s4 = run_bv_joint_flux_curve_inference(request_s4)
    stage4_k0 = np.asarray(result_s4["best_k0"])
    stage4_alpha = np.asarray(result_s4["best_alpha"])
    stage4_time = time.time() - t4

    print(f"\n  Stage 4 result: k0 = {stage4_k0.tolist()}, alpha = {stage4_alpha.tolist()}")
    print(f"  Time: {stage4_time:.1f}s")
    stage_results["stage4"] = {
        "k0": stage4_k0.tolist(),
        "alpha": stage4_alpha.tolist(),
        "loss": result_s4["best_loss"],
        "time": stage4_time,
    }

    # -----------------------------------------------------------------------
    # Comparison summary
    # -----------------------------------------------------------------------
    true_k0_arr = np.asarray(true_k0)
    true_alpha_arr = np.asarray(true_alpha)
    total_staged_time = stage1_time + stage2_time + stage3_time

    print(f"\n{'='*90}")
    print(f"  Staged vs Direct Inference Comparison")
    print(f"{'='*90}")
    print(f"  True k0:    {true_k0}")
    print(f"  True alpha: {true_alpha}")
    print()

    print(f"{'Stage':<20} | {'k0_1 err':>10} {'k0_2 err':>10} {'a1 err':>10} {'a2 err':>10} "
          f"| {'loss':>12} | {'time':>6}")
    print(f"{'-'*90}")

    for stage_name, sr in stage_results.items():
        k0_arr = np.asarray(sr["k0"])
        alpha_arr = np.asarray(sr["alpha"])
        k0_err = np.abs(k0_arr - true_k0_arr) / np.maximum(np.abs(true_k0_arr), 1e-16)
        alpha_err = np.abs(alpha_arr - true_alpha_arr) / np.maximum(np.abs(true_alpha_arr), 1e-16)
        print(f"{stage_name:<20} | {k0_err[0]:>10.4f} {k0_err[1]:>10.4f} "
              f"{alpha_err[0]:>10.4f} {alpha_err[1]:>10.4f} "
              f"| {sr['loss']:>12.6e} | {sr['time']:>5.0f}s")

    print(f"{'-'*90}")
    print(f"  Staged total time (S1+S2+S3): {total_staged_time:.0f}s")
    print(f"  Direct joint time (S4):       {stage4_time:.0f}s")
    print(f"{'='*90}")

    # Save comparison CSV
    csv_path = os.path.join(base_output, "staged_vs_direct_comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "stage", "k0_1", "k0_2", "alpha_1", "alpha_2",
            "k0_1_err", "k0_2_err", "alpha_1_err", "alpha_2_err",
            "loss", "time_s",
        ])
        for stage_name, sr in stage_results.items():
            k0_arr = np.asarray(sr["k0"])
            alpha_arr = np.asarray(sr["alpha"])
            k0_err = np.abs(k0_arr - true_k0_arr) / np.maximum(np.abs(true_k0_arr), 1e-16)
            alpha_err = np.abs(alpha_arr - true_alpha_arr) / np.maximum(np.abs(true_alpha_arr), 1e-16)
            writer.writerow([
                stage_name,
                f"{k0_arr[0]:.8e}", f"{k0_arr[1]:.8e}",
                f"{alpha_arr[0]:.6f}", f"{alpha_arr[1]:.6f}",
                f"{k0_err[0]:.6f}", f"{k0_err[1]:.6f}",
                f"{alpha_err[0]:.6f}", f"{alpha_err[1]:.6f}",
                f"{sr['loss']:.12e}", f"{sr['time']:.1f}",
            ])
    print(f"\n[csv] Comparison saved -> {csv_path}")

    # Agent log
    log_path = os.path.join(base_output, "agent_log.md")
    with open(log_path, "w") as f:
        f.write("# Staged Inference Study - Agent Log\n\n")
        f.write("## Approach\n")
        f.write("- Stage 1: Alpha inference with k0 fixed at true values\n")
        f.write("- Stage 2: k0 inference with alpha fixed at Stage 1 result\n")
        f.write("- Stage 3: Joint refinement from Stage 1+2 warm start\n")
        f.write("- Stage 4: Direct joint inference (comparison baseline)\n\n")
        f.write("## Results\n")
        for stage_name, sr in stage_results.items():
            f.write(f"### {stage_name}\n")
            f.write(f"- k0: {sr['k0']}\n")
            f.write(f"- alpha: {sr['alpha']}\n")
            f.write(f"- loss: {sr['loss']:.6e}\n")
            f.write(f"- time: {sr['time']:.1f}s\n\n")
        f.write(f"## Conclusion\n")
        f.write(f"Staged time: {total_staged_time:.0f}s vs Direct: {stage4_time:.0f}s\n")

    print(f"\n=== Staged Inference Study Complete ===")
    print(f"Output: {base_output}/")


if __name__ == "__main__":
    main()
