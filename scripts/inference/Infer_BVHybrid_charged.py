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
        forward_recovery=_make_recovery(),
    )

    result_a = run_bv_alpha_flux_curve_inference(request_a)
    phase_a_alpha = np.asarray(result_a["best_alpha"])
    phase_a_time = time.time() - t_a

    alpha_err = np.abs(phase_a_alpha - true_alpha_arr) / np.maximum(np.abs(true_alpha_arr), 1e-16)
    print(f"\n  Phase A result:")
    print(f"    alpha_1 = {phase_a_alpha[0]:.6f}  (true {ALPHA_1:.6f}, err {alpha_err[0]*100:.2f}%)")
    print(f"    alpha_2 = {phase_a_alpha[1]:.6f}  (true {ALPHA_2:.6f}, err {alpha_err[1]*100:.2f}%)")
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
        forward_recovery=_make_recovery(),
    )

    result_b = run_bv_k0_flux_curve_inference(request_b)
    phase_b_k0 = np.asarray(result_b["best_k0"])
    phase_b_time = time.time() - t_b

    k0_err = np.abs(phase_b_k0 - true_k0_arr) / np.maximum(np.abs(true_k0_arr), 1e-16)
    print(f"\n  Phase B result:")
    print(f"    k0_1 = {phase_b_k0[0]:.6e}  (true {K0_HAT:.6e}, err {k0_err[0]*100:.2f}%)")
    print(f"    k0_2 = {phase_b_k0[1]:.6e}  (true {K0_2_HAT:.6e}, err {k0_err[1]*100:.2f}%)")
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
        forward_recovery=_make_recovery(),
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
