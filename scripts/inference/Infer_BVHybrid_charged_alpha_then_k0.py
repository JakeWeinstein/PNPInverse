"""Hybrid two-phase inference: alpha on SYMMETRIC range, then k0 on CATHODIC range.

Full 4-species charged system: O2, H2O2, H+, ClO4- (z=[0,0,+1,-1]).

Key insight:
    Phase A uses a SYMMETRIC voltage range (positive + negative eta) to recover alpha
    accurately, because including anodic overpotentials breaks the k0-alpha correlation:
        cathodic:  I ~ k0 * exp(-alpha * eta)       -> Tafel slope = alpha
        anodic:    I ~ k0 * exp((1-alpha) * eta)     -> Tafel slope = (1-alpha)
    Together these provide two independent constraints on alpha.

    Phase B uses an extended CATHODIC range with alpha FIXED at Phase A result to
    recover k0 accurately, since k0 controls the absolute magnitude once the slope
    (alpha) is known.

    Optional Phase C performs a short joint refinement from the Phase A+B warm start.

Uses speed improvements: multi-fidelity mesh coarsening + checkpoint warm-start.

Usage (from PNPInverse/ directory)::

    python scripts/inference/Infer_BVHybrid_charged_alpha_then_k0.py
"""

from __future__ import annotations

import copy
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
    _clear_caches,
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
    # ===================================================================
    # Voltage ranges
    # ===================================================================

    # Phase A: Symmetric focused range (best for alpha recovery)
    # Anodic points give (1-alpha) Tafel slope, cathodic give alpha slope.
    # Together they provide two independent constraints on alpha.
    eta_symmetric = np.array([
        +5.0, +3.0, +1.0,           # anodic (3 pts)
        -0.5,                         # near-equilibrium
        -1.0, -2.0, -3.0,            # cathodic onset
        -5.0, -8.0,                  # transition
        -10.0, -15.0, -20.0,         # knee + plateau
    ])

    # Phase B: Extended cathodic range (best for k0 recovery)
    # Once alpha is known, k0 controls the absolute magnitude; the
    # extended range provides more constraint on that magnitude.
    eta_cathodic = np.linspace(-1.0, -46.5, 15)

    # ===================================================================
    # Solver configuration
    # ===================================================================
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

    # Deliberately wrong initial guesses
    initial_k0_guess = [0.005, 0.0005]
    initial_alpha_guess = [0.4, 0.3]

    observable_scale = -I_SCALE
    base_output = os.path.join("StudyResults", "bv_hybrid_alpha_then_k0_charged")
    os.makedirs(base_output, exist_ok=True)

    phase_results = {}

    print(f"\n{'='*70}")
    print(f"  HYBRID Inference: Alpha (symmetric) -> k0 (cathodic)")
    print(f"{'='*70}")
    print(f"  Phase A voltage range: {len(eta_symmetric)} points, "
          f"eta in [{eta_symmetric.min():+.1f}, {eta_symmetric.max():+.1f}]")
    print(f"  Phase B voltage range: {len(eta_cathodic)} points, "
          f"eta in [{eta_cathodic.min():+.1f}, {eta_cathodic.max():+.1f}]")
    print(f"  True k0:    {true_k0}")
    print(f"  True alpha: {true_alpha}")
    print(f"  Speed improvements: multi-fidelity ON, checkpoint warm-start ON")
    print(f"{'='*70}\n")

    # ===================================================================
    # Phase A: Alpha inference on SYMMETRIC range (k0 fixed at true)
    # ===================================================================
    print(f"\n{'='*70}")
    print(f"  Phase A: Alpha inference on symmetric range (k0 fixed at true)")
    print(f"  {len(eta_symmetric)} points: eta in [{eta_symmetric.min():+.1f}, "
          f"{eta_symmetric.max():+.1f}]")
    print(f"  Anodic: {sum(eta_symmetric > 0)}, Near-eq: "
          f"{sum(np.abs(eta_symmetric) <= 0.5)}, Cathodic: {sum(eta_symmetric < -0.5)}")
    print(f"{'='*70}")
    t_A = time.time()

    phaseA_dir = os.path.join(base_output, "phaseA_alpha_symmetric")

    request_A = BVFluxCurveInferenceRequest(
        base_solver_params=base_sp,
        steady=steady,
        true_k0=true_k0,
        initial_guess=true_k0,  # k0 are fixed, not optimized
        phi_applied_values=eta_symmetric.tolist(),
        target_csv_path=os.path.join(phaseA_dir, "target.csv"),
        output_dir=phaseA_dir,
        regenerate_target=True,
        target_noise_percent=2.0,
        target_seed=20260226,
        observable_mode="current_density",
        current_density_scale=observable_scale,
        observable_label="current density (mA/cm2)",
        observable_title="Phase A: Alpha inference (symmetric, k0 fixed)",
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
        forward_recovery=_make_recovery(),
        # Speed improvements
        multifidelity_enabled=True,
        coarse_mesh_Nx=4,
        coarse_mesh_Ny=100,
        coarse_max_iters=5,
        use_checkpoint_warmstart=True,
    )

    result_A = run_bv_alpha_flux_curve_inference(request_A)
    recovered_alpha = np.asarray(result_A["best_alpha"])
    phaseA_time = time.time() - t_A

    print(f"\n  Phase A result: alpha = {recovered_alpha.tolist()}")
    print(f"  Phase A time: {phaseA_time:.1f}s")
    phase_results["phaseA"] = {
        "alpha": recovered_alpha.tolist(),
        "k0": true_k0,
        "loss": result_A["best_loss"],
        "time": phaseA_time,
    }

    # ===================================================================
    # Clear caches between phases (parameter values change)
    # ===================================================================
    _clear_caches()

    # ===================================================================
    # Phase B: k0 inference on CATHODIC range (alpha fixed at Phase A)
    # ===================================================================
    print(f"\n{'='*70}")
    print(f"  Phase B: k0 inference on cathodic range (alpha from Phase A)")
    print(f"  {len(eta_cathodic)} points: eta in [{eta_cathodic.min():+.1f}, "
          f"{eta_cathodic.max():+.1f}]")
    print(f"  Using recovered alpha = {recovered_alpha.tolist()}")
    print(f"{'='*70}")
    t_B = time.time()

    phaseB_dir = os.path.join(base_output, "phaseB_k0_cathodic")

    # Inject Phase A alpha into base_solver_params so the forward solver
    # uses the recovered alpha values (not the original true values)
    base_sp_B = copy.deepcopy(base_sp)
    bv_cfg = base_sp_B[10]["bv_bc"]
    for j, rxn in enumerate(bv_cfg["reactions"]):
        if j < len(recovered_alpha):
            rxn["alpha"] = float(recovered_alpha[j])

    request_B = BVFluxCurveInferenceRequest(
        base_solver_params=base_sp_B,
        steady=steady,
        true_k0=true_k0,
        initial_guess=initial_k0_guess,
        phi_applied_values=eta_cathodic.tolist(),
        target_csv_path=os.path.join(phaseB_dir, "target.csv"),
        output_dir=phaseB_dir,
        regenerate_target=True,
        target_noise_percent=2.0,
        target_seed=20260226,
        observable_mode="current_density",
        current_density_scale=observable_scale,
        observable_label="current density (mA/cm2)",
        observable_title="Phase B: k0 inference (cathodic, alpha from Phase A)",
        control_mode="k0",
        k0_lower=1e-8, k0_upper=100.0,
        log_space=True,
        # Alpha is fixed at Phase A result
        true_alpha=recovered_alpha.tolist(),
        initial_alpha_guess=recovered_alpha.tolist(),
        mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
        max_eta_gap=3.0,
        optimizer_method="L-BFGS-B",
        optimizer_options={"maxiter": 30, "ftol": 1e-12, "gtol": 1e-6, "disp": True},
        max_iters=30,
        live_plot=False,
        forward_recovery=_make_recovery(),
        # Speed improvements
        multifidelity_enabled=True,
        coarse_mesh_Nx=4,
        coarse_mesh_Ny=100,
        coarse_max_iters=5,
        use_checkpoint_warmstart=True,
    )

    result_B = run_bv_k0_flux_curve_inference(request_B)
    recovered_k0 = np.asarray(result_B["best_k0"])
    phaseB_time = time.time() - t_B

    print(f"\n  Phase B result: k0 = {recovered_k0.tolist()}")
    print(f"  Phase B time: {phaseB_time:.1f}s")
    phase_results["phaseB"] = {
        "k0": recovered_k0.tolist(),
        "alpha": recovered_alpha.tolist(),
        "loss": result_B["best_loss"],
        "time": phaseB_time,
    }

    # ===================================================================
    # Clear caches between phases
    # ===================================================================
    _clear_caches()

    # ===================================================================
    # Phase C (optional): Joint refinement from Phase A+B warm start
    # ===================================================================
    print(f"\n{'='*70}")
    print(f"  Phase C: Joint refinement from Phase A+B warm start")
    print(f"  Using recovered k0 = {recovered_k0.tolist()}")
    print(f"  Using recovered alpha = {recovered_alpha.tolist()}")
    print(f"{'='*70}")
    t_C = time.time()

    phaseC_dir = os.path.join(base_output, "phaseC_joint_refinement")

    # Use the full cathodic range for joint refinement (same as Phase B)
    request_C = BVFluxCurveInferenceRequest(
        base_solver_params=base_sp,  # original base params (true alpha)
        steady=steady,
        true_k0=true_k0,
        initial_guess=recovered_k0.tolist(),  # warm start from Phase B
        phi_applied_values=eta_cathodic.tolist(),
        target_csv_path=os.path.join(phaseC_dir, "target.csv"),
        output_dir=phaseC_dir,
        regenerate_target=True,
        target_noise_percent=2.0,
        target_seed=20260226,
        observable_mode="current_density",
        current_density_scale=observable_scale,
        observable_label="current density (mA/cm2)",
        observable_title="Phase C: Joint refinement (warm start from A+B)",
        control_mode="joint",
        k0_lower=1e-8, k0_upper=100.0,
        log_space=True,
        true_alpha=true_alpha,
        initial_alpha_guess=recovered_alpha.tolist(),  # warm start from Phase A
        alpha_lower=0.05, alpha_upper=0.95,
        mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
        max_eta_gap=3.0,
        optimizer_method="L-BFGS-B",
        optimizer_options={"maxiter": 15, "ftol": 1e-12, "gtol": 1e-6, "disp": True},
        max_iters=15,
        live_plot=False,
        forward_recovery=_make_recovery(),
        # Speed improvements
        multifidelity_enabled=True,
        coarse_mesh_Nx=4,
        coarse_mesh_Ny=100,
        coarse_max_iters=5,
        use_checkpoint_warmstart=True,
    )

    result_C = run_bv_joint_flux_curve_inference(request_C)
    refined_k0 = np.asarray(result_C["best_k0"])
    refined_alpha = np.asarray(result_C["best_alpha"])
    phaseC_time = time.time() - t_C

    print(f"\n  Phase C result: k0 = {refined_k0.tolist()}, alpha = {refined_alpha.tolist()}")
    print(f"  Phase C time: {phaseC_time:.1f}s")
    phase_results["phaseC"] = {
        "k0": refined_k0.tolist(),
        "alpha": refined_alpha.tolist(),
        "loss": result_C["best_loss"],
        "time": phaseC_time,
    }

    # ===================================================================
    # Comparison summary
    # ===================================================================
    true_k0_arr = np.asarray(true_k0)
    true_alpha_arr = np.asarray(true_alpha)
    total_time = phaseA_time + phaseB_time + phaseC_time

    print(f"\n{'='*90}")
    print(f"  HYBRID Inference Results: Alpha (symmetric) -> k0 (cathodic) -> Joint")
    print(f"{'='*90}")
    print(f"  True k0:    {true_k0}")
    print(f"  True alpha: {true_alpha}")
    print()

    print(f"{'Phase':<20} | {'k0_1':>12} {'k0_2':>12} {'a1':>8} {'a2':>8} "
          f"| {'k0_1 err':>10} {'k0_2 err':>10} {'a1 err':>10} {'a2 err':>10} "
          f"| {'loss':>12} | {'time':>6}")
    print(f"{'-'*130}")

    for phase_name, pr in phase_results.items():
        k0_arr = np.asarray(pr["k0"])
        alpha_arr = np.asarray(pr["alpha"])
        k0_err = np.abs(k0_arr - true_k0_arr) / np.maximum(np.abs(true_k0_arr), 1e-16)
        alpha_err = np.abs(alpha_arr - true_alpha_arr) / np.maximum(np.abs(true_alpha_arr), 1e-16)
        print(f"{phase_name:<20} | {k0_arr[0]:>12.6e} {k0_arr[1]:>12.6e} "
              f"{alpha_arr[0]:>8.4f} {alpha_arr[1]:>8.4f} "
              f"| {k0_err[0]:>10.4f} {k0_err[1]:>10.4f} "
              f"{alpha_err[0]:>10.4f} {alpha_err[1]:>10.4f} "
              f"| {pr['loss']:>12.6e} | {pr['time']:>5.0f}s")

    print(f"{'-'*130}")
    print(f"  Total time (A+B+C): {total_time:.0f}s")
    print(f"{'='*130}")

    # ===================================================================
    # Save comparison CSV
    # ===================================================================
    csv_path = os.path.join(base_output, "hybrid_comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "phase", "voltage_range", "control_mode",
            "k0_1", "k0_2", "alpha_1", "alpha_2",
            "k0_1_rel_err", "k0_2_rel_err", "alpha_1_rel_err", "alpha_2_rel_err",
            "loss", "time_s",
        ])
        range_labels = {
            "phaseA": "symmetric",
            "phaseB": "cathodic",
            "phaseC": "cathodic",
        }
        mode_labels = {
            "phaseA": "alpha",
            "phaseB": "k0",
            "phaseC": "joint",
        }
        for phase_name, pr in phase_results.items():
            k0_arr = np.asarray(pr["k0"])
            alpha_arr = np.asarray(pr["alpha"])
            k0_err = np.abs(k0_arr - true_k0_arr) / np.maximum(np.abs(true_k0_arr), 1e-16)
            alpha_err = np.abs(alpha_arr - true_alpha_arr) / np.maximum(np.abs(true_alpha_arr), 1e-16)
            writer.writerow([
                phase_name, range_labels.get(phase_name, ""),
                mode_labels.get(phase_name, ""),
                f"{k0_arr[0]:.8e}", f"{k0_arr[1]:.8e}",
                f"{alpha_arr[0]:.6f}", f"{alpha_arr[1]:.6f}",
                f"{k0_err[0]:.6f}", f"{k0_err[1]:.6f}",
                f"{alpha_err[0]:.6f}", f"{alpha_err[1]:.6f}",
                f"{pr['loss']:.12e}", f"{pr['time']:.1f}",
            ])
    print(f"\n[csv] Comparison saved -> {csv_path}")

    # ===================================================================
    # Save summary text
    # ===================================================================
    summary_path = os.path.join(base_output, "hybrid_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Hybrid Alpha+k0 Inference Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"True k0:    {true_k0}\n")
        f.write(f"True alpha: {true_alpha}\n\n")
        f.write(f"Phase A (alpha, symmetric): alpha = {recovered_alpha.tolist()}, "
                f"time = {phaseA_time:.1f}s\n")
        f.write(f"Phase B (k0, cathodic):     k0 = {recovered_k0.tolist()}, "
                f"time = {phaseB_time:.1f}s\n")
        f.write(f"Phase C (joint refinement): k0 = {refined_k0.tolist()}, "
                f"alpha = {refined_alpha.tolist()}, time = {phaseC_time:.1f}s\n\n")
        f.write(f"Total time: {total_time:.1f}s\n")

        # Per-phase errors
        for phase_name, pr in phase_results.items():
            k0_arr = np.asarray(pr["k0"])
            alpha_arr = np.asarray(pr["alpha"])
            k0_err = np.abs(k0_arr - true_k0_arr) / np.maximum(np.abs(true_k0_arr), 1e-16)
            alpha_err = np.abs(alpha_arr - true_alpha_arr) / np.maximum(np.abs(true_alpha_arr), 1e-16)
            f.write(f"\n{phase_name}:\n")
            f.write(f"  k0 rel errors:    [{k0_err[0]:.6f}, {k0_err[1]:.6f}]\n")
            f.write(f"  alpha rel errors: [{alpha_err[0]:.6f}, {alpha_err[1]:.6f}]\n")
            f.write(f"  loss: {pr['loss']:.6e}\n")
    print(f"[txt] Summary saved -> {summary_path}")

    print(f"\n=== Hybrid Alpha+k0 Inference Complete ===")
    print(f"Output: {base_output}/")


if __name__ == "__main__":
    main()
