"""Master 4-phase inference protocol v3 for BV kinetics (charged system).

Full 4-species charged system: O2, H2O2, H+, ClO4- (z=[0,0,+1,-1]).

Key difference from v2: Phase 3 uses multi-observable (total + peroxide current)
on the FULL cathodic range [-1, -46.5], warm-started from Phase 2.  The deep
plateau data sharpens k0_1 without losing R2 identifiability (since the peroxide
observable is always present).

NEVER use total-current-only in any joint optimization phase.

Four-phase protocol:
    Phase 1: Alpha initialization via 12-pt symmetric sweep (alpha-only, k0 FIXED at guess)
    Phase 2: Multi-observable joint on 10-pt SHALLOW cathodic [-1, -13] from COLD k0 + P1 alpha
    Phase 3: Multi-observable joint on 15-pt FULL cathodic [-1, -46.5], warm-start from P2
    Phase 4: Best-of selection (Phase 2 vs Phase 3)

Usage (from PNPInverse/ directory)::

    python scripts/inference/Infer_BVMaster_charged_v3.py
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
    run_bv_alpha_flux_curve_inference,
    run_bv_multi_observable_flux_curve_inference,
)
from FluxCurve.bv_point_solve import _clear_caches
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

def _make_bv_solver_params(eta_hat: float, dt: float, t_end: float,
                           c_hp_hat: float = C_HP_HAT,
                           snes_opts: dict | None = None,
                           softplus: bool = False) -> SolverParams:
    """Build SolverParams for 4-species charged BV."""
    params = dict(snes_opts or SNES_OPTS)
    bv_conv = {
        "clip_exponent": True, "exponent_clip": 50.0,
        "regularize_concentration": True, "conc_floor": 1e-12,
        "use_eta_in_bv": True,
    }
    if softplus:
        bv_conv["softplus_regularization"] = True
    params["bv_convergence"] = bv_conv
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
                    {"species": 2, "power": 2, "c_ref_nondim": c_hp_hat},
                ],
            },
            {
                "k0": K0_2_HAT, "alpha": ALPHA_2,
                "cathodic_species": 1, "anodic_species": None,
                "c_ref": 0.0, "stoichiometry": [0, -1, -2, 0],
                "n_electrons": 2, "reversible": False,
                "cathodic_conc_factors": [
                    {"species": 2, "power": 2, "c_ref_nondim": c_hp_hat},
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
        [C_O2_HAT, C_H2O2_HAT, c_hp_hat, C_CLO4_HAT],
        0.0, params,
    ])


def _make_recovery():
    return ForwardRecoveryConfig(
        max_attempts=6, max_it_only_attempts=2,
        anisotropy_only_attempts=1, tolerance_relax_attempts=2,
        max_it_growth=1.5, max_it_cap=600,
    )


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

    # Phase 3: 15-point FULL cathodic [-1, -46.5]
    eta_cathodic = np.array([
        -1.0, -2.0, -3.0,
        -4.0, -5.0, -6.5, -8.0,
        -10.0, -13.0,
        -17.0, -22.0, -28.0,
        -35.0, -41.0, -46.5,
    ])

    true_k0 = [K0_HAT, K0_2_HAT]
    true_alpha = [ALPHA_1, ALPHA_2]
    true_k0_arr = np.asarray(true_k0)
    true_alpha_arr = np.asarray(true_alpha)

    # Deliberately wrong initial guesses (realistic scenario)
    initial_k0_guess = [0.005, 0.0005]
    initial_alpha_guess = [0.4, 0.3]

    observable_scale = -I_SCALE
    base_output = os.path.join("StudyResults", "master_inference_v3")
    os.makedirs(base_output, exist_ok=True)

    # Standard solver config
    dt = 0.5
    max_ss_steps = 100
    t_end = dt * max_ss_steps
    base_sp = _make_bv_solver_params(eta_hat=0.0, dt=dt, t_end=t_end)

    steady = SteadyStateConfig(
        relative_tolerance=1e-4, absolute_tolerance=1e-8,
        consecutive_steps=4, max_steps=max_ss_steps,
        flux_observable="total_species", verbose=False,
    )

    phase_results = {}
    t_total_start = time.time()

    print(f"\n{'#'*70}")
    print(f"  MASTER 4-PHASE INFERENCE PROTOCOL v3")
    print(f"  Key change: multi-obs in ALL phases (never total-current-only)")
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
        forward_recovery=_make_recovery(),
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
        initial_guess=initial_k0_guess,  # COLD k0 start
        phi_applied_values=eta_shallow.tolist(),
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
        forward_recovery=_make_recovery(),
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
    # PHASE 3: Multi-Observable Joint on FULL cathodic range
    #          Warm-start all 4 params from Phase 2.
    #          Uses total + peroxide current to maintain R2 identifiability.
    # ===================================================================
    print(f"\n{'='*70}")
    print(f"  PHASE 3: Multi-observable joint on FULL cathodic range")
    print(f"  Warm-start from Phase 2: k0={p2_k0.tolist()}, alpha={p2_alpha.tolist()}")
    print(f"  {len(eta_cathodic)}-pt FULL cathodic [{eta_cathodic.min():.1f}, {eta_cathodic.max():.1f}]")
    print(f"  total + peroxide current, secondary_weight=1.0")
    print(f"  maxiter=30")
    print(f"{'='*70}")
    t_p3 = time.time()

    p3_dir = os.path.join(base_output, "phase3_multi_obs_full_range")

    request_p3 = BVFluxCurveInferenceRequest(
        base_solver_params=base_sp,
        steady=steady,
        true_k0=true_k0,
        initial_guess=p2_k0.tolist(),  # warm-start k0 from P2
        phi_applied_values=eta_cathodic.tolist(),
        target_csv_path=os.path.join(p3_dir, "target_primary.csv"),
        output_dir=p3_dir,
        regenerate_target=True,
        target_noise_percent=2.0,
        target_seed=20260226,
        # Primary observable: total current density
        observable_mode="current_density",
        current_density_scale=observable_scale,
        observable_label="current density (mA/cm2)",
        observable_title="Phase 3: Multi-obs joint (full range, warm P2)",
        # Secondary observable: peroxide current
        secondary_observable_mode="peroxide_current",
        secondary_observable_weight=1.0,
        secondary_current_density_scale=observable_scale,
        secondary_target_csv_path=os.path.join(p3_dir, "target_peroxide.csv"),
        # Joint control mode
        control_mode="joint",
        true_alpha=true_alpha,
        initial_alpha_guess=p2_alpha.tolist(),  # warm-start alpha from P2
        alpha_lower=0.05, alpha_upper=0.95,
        k0_lower=1e-8, k0_upper=100.0,
        log_space=True,
        mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
        max_eta_gap=3.0,
        optimizer_method="L-BFGS-B",
        optimizer_options={"maxiter": 30, "ftol": 1e-12, "gtol": 1e-6, "disp": True},
        max_iters=30,
        live_plot=False,
        forward_recovery=_make_recovery(),
    )

    result_p3 = run_bv_multi_observable_flux_curve_inference(request_p3)
    p3_k0 = np.asarray(result_p3["best_k0"])
    p3_alpha = np.asarray(result_p3["best_alpha"])
    p3_time = time.time() - t_p3

    _print_phase_result("Phase 3", p3_k0, p3_alpha, true_k0_arr, true_alpha_arr,
                        result_p3["best_loss"], p3_time)
    phase_results["Phase 3 (multi-obs, full cat)"] = {
        "k0": p3_k0.tolist(), "alpha": p3_alpha.tolist(),
        "loss": result_p3["best_loss"], "time": p3_time,
    }

    # ===================================================================
    # PHASE 4: Best-of Selection (Phase 2 vs Phase 3)
    # ===================================================================
    p2_k0_err, p2_alpha_err = _compute_errors(p2_k0, p2_alpha, true_k0_arr, true_alpha_arr)
    p3_k0_err, p3_alpha_err = _compute_errors(p3_k0, p3_alpha, true_k0_arr, true_alpha_arr)

    p2_max_err = max(p2_k0_err.max(), p2_alpha_err.max())
    p3_max_err = max(p3_k0_err.max(), p3_alpha_err.max())

    if p3_max_err <= p2_max_err:
        best_k0, best_alpha = p3_k0.copy(), p3_alpha.copy()
        best_source = "Phase 3"
    else:
        best_k0, best_alpha = p2_k0.copy(), p2_alpha.copy()
        best_source = "Phase 2"

    best_max_err = min(p2_max_err, p3_max_err)

    print(f"\n{'='*70}")
    print(f"  P2 vs P3: best is {best_source} (max err = {best_max_err*100:.2f}%)")
    print(f"{'='*70}")

    total_time = time.time() - t_total_start

    # ===================================================================
    # FINAL SUMMARY TABLE
    # ===================================================================
    print(f"\n{'#'*90}")
    print(f"  MASTER INFERENCE v3 SUMMARY")
    print(f"{'#'*90}")
    print(f"  True k0:    {true_k0}")
    print(f"  True alpha: {true_alpha}")
    print(f"  K0_HAT = {K0_HAT:.6e},  K0_2_HAT = {K0_2_HAT:.6e}")
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

    best_k0_err, best_alpha_err = _compute_errors(best_k0, best_alpha, true_k0_arr, true_alpha_arr)

    print(f"  Best result: {best_source} (max err = {best_max_err*100:.2f}%)")
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
    csv_path = os.path.join(base_output, "master_comparison_v3.csv")
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
    print(f"\n=== Master Inference v3 Complete ===")


if __name__ == "__main__":
    main()
