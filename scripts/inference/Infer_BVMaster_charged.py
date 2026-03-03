"""Master 5-phase inference protocol for BV kinetics (charged system).

Full 4-species charged system: O2, H2O2, H+, ClO4- (z=[0,0,+1,-1]).

Five-phase protocol:
    Phase 1: Alpha initialization via 12-pt symmetric sweep (alpha-only, k0 FIXED at guess)
    Phase 2: k0 initialization via 15-pt cathodic sweep (k0-only, alpha fixed from P1)
    Phase 3: R2 recovery via multi-observable (total + peroxide current, joint k0+alpha)
    Phase 4: Joint refinement with regularization annealing (3 sub-phases)
    Phase 5: (Optional) Multi-pH validation if Phase 4 doesn't achieve <10% on all params

Usage (from PNPInverse/ directory)::

    python scripts/inference/Infer_BVMaster_charged.py
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
    run_bv_k0_flux_curve_inference,
    run_bv_joint_flux_curve_inference,
    run_bv_multi_observable_flux_curve_inference,
    run_bv_multi_ph_flux_curve_inference,
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

# Phase 5 uses tighter SNES settings for elevated-H+ robustness
SNES_OPTS_PHASE5 = dict(SNES_OPTS)
SNES_OPTS_PHASE5["snes_linesearch_maxlambda"] = 0.3


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

    # Phase 2/3/4: 15-point cathodic [-1, -46.5]
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
    base_output = os.path.join("StudyResults", "master_inference")
    os.makedirs(base_output, exist_ok=True)

    # Standard solver config (Phases 1-4)
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
    print(f"  MASTER 5-PHASE INFERENCE PROTOCOL")
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
        initial_guess=initial_k0_guess,  # needed for target gen (uses true_k0)
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
        fixed_k0=initial_k0_guess,  # Fix k0 at WRONG guess values
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

    # Clear caches between phases
    _clear_caches()

    # ===================================================================
    # PHASE 2: k0 Initialization via Cathodic Sweep (k0-only)
    # ===================================================================
    print(f"\n{'='*70}")
    print(f"  PHASE 2: k0 from {len(eta_cathodic)}-pt cathodic sweep")
    print(f"  alpha FIXED at Phase 1 result: {p1_alpha.tolist()}")
    print(f"  eta in [{eta_cathodic.min():.1f}, {eta_cathodic.max():.1f}]")
    print(f"{'='*70}")
    t_p2 = time.time()

    p2_dir = os.path.join(base_output, "phase2_k0_cathodic")

    request_p2 = BVFluxCurveInferenceRequest(
        base_solver_params=base_sp,
        steady=steady,
        true_k0=true_k0,
        initial_guess=initial_k0_guess,
        phi_applied_values=eta_cathodic.tolist(),
        target_csv_path=os.path.join(p2_dir, "target.csv"),
        output_dir=p2_dir,
        regenerate_target=True,
        target_noise_percent=2.0,
        target_seed=20260226,
        observable_mode="current_density",
        current_density_scale=observable_scale,
        observable_label="current density (mA/cm2)",
        observable_title="Phase 2: k0 (cathodic, alpha from P1)",
        control_mode="k0",
        k0_lower=1e-8, k0_upper=100.0,
        log_space=True,
        # Fix alpha at Phase 1 result
        true_alpha=p1_alpha.tolist(),
        initial_alpha_guess=p1_alpha.tolist(),
        mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
        max_eta_gap=3.0,
        optimizer_method="L-BFGS-B",
        optimizer_options={"maxiter": 30, "ftol": 1e-12, "gtol": 1e-6, "disp": True},
        max_iters=30,
        live_plot=False,
        forward_recovery=_make_recovery(),
    )

    result_p2 = run_bv_k0_flux_curve_inference(request_p2)
    p2_k0 = np.asarray(result_p2["best_k0"])
    p2_alpha = p1_alpha.copy()  # unchanged
    p2_time = time.time() - t_p2

    _print_phase_result("Phase 2", p2_k0, p2_alpha, true_k0_arr, true_alpha_arr,
                        result_p2["best_loss"], p2_time)
    phase_results["Phase 2 (k0, cat 15pt)"] = {
        "k0": p2_k0.tolist(), "alpha": p2_alpha.tolist(),
        "loss": result_p2["best_loss"], "time": p2_time,
    }

    _clear_caches()

    # ===================================================================
    # PHASE 3: R2 Recovery via Multi-Observable (joint k0+alpha)
    # ===================================================================
    print(f"\n{'='*70}")
    print(f"  PHASE 3: Multi-observable joint (total + peroxide current)")
    print(f"  Warm-start: k0 from P2, alpha from P1")
    print(f"  {len(eta_cathodic)}-pt cathodic, secondary_weight=1.0")
    print(f"{'='*70}")
    t_p3 = time.time()

    p3_dir = os.path.join(base_output, "phase3_multi_obs")

    request_p3 = BVFluxCurveInferenceRequest(
        base_solver_params=base_sp,
        steady=steady,
        true_k0=true_k0,
        initial_guess=p2_k0.tolist(),
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
        observable_title="Phase 3: Multi-obs joint fit",
        # Secondary observable: peroxide current
        secondary_observable_mode="peroxide_current",
        secondary_observable_weight=1.0,
        secondary_current_density_scale=observable_scale,
        secondary_target_csv_path=os.path.join(p3_dir, "target_peroxide.csv"),
        # Joint control mode
        control_mode="joint",
        true_alpha=true_alpha,
        initial_alpha_guess=p1_alpha.tolist(),
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

    result_p3 = run_bv_multi_observable_flux_curve_inference(request_p3)
    p3_k0 = np.asarray(result_p3["best_k0"])
    p3_alpha = np.asarray(result_p3["best_alpha"])
    p3_time = time.time() - t_p3

    _print_phase_result("Phase 3", p3_k0, p3_alpha, true_k0_arr, true_alpha_arr,
                        result_p3["best_loss"], p3_time)
    phase_results["Phase 3 (multi-obs joint)"] = {
        "k0": p3_k0.tolist(), "alpha": p3_alpha.tolist(),
        "loss": result_p3["best_loss"], "time": p3_time,
    }

    _clear_caches()

    # ===================================================================
    # PHASE 4: Joint Refinement with Regularization Annealing
    # ===================================================================
    print(f"\n{'='*70}")
    print(f"  PHASE 4: Regularization annealing (3 sub-phases)")
    print(f"  Warm-start: k0 from P3, alpha from P3")
    print(f"  4a: lambda=0.1, 4b: lambda=0.01, 4c: lambda=0.0")
    print(f"{'='*70}")

    p4_k0 = p3_k0.copy()
    p4_alpha = p3_alpha.copy()
    p4_total_time = 0.0

    reg_schedule = [
        ("4a", 0.1, 10, p2_k0.tolist(), p1_alpha.tolist()),
        ("4b", 0.01, 15, None, None),  # prior updated after 4a
        ("4c", 0.0, 20, None, None),   # no regularization
    ]

    for sub_name, lam, maxiter, prior_k0, prior_alpha in reg_schedule:
        # Update priors for 4b and 4c from previous sub-phase result
        if prior_k0 is None:
            prior_k0 = p4_k0.tolist()
        if prior_alpha is None:
            prior_alpha = p4_alpha.tolist()

        print(f"\n  --- Phase {sub_name}: lambda={lam}, maxiter={maxiter} ---")
        print(f"  Prior k0: {prior_k0}")
        print(f"  Prior alpha: {prior_alpha}")
        t_sub = time.time()

        sub_dir = os.path.join(base_output, f"phase4{sub_name[-1]}_reg_lambda{lam}")

        request_sub = BVFluxCurveInferenceRequest(
            base_solver_params=base_sp,
            steady=steady,
            true_k0=true_k0,
            initial_guess=p4_k0.tolist(),
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
            initial_alpha_guess=p4_alpha.tolist(),
            alpha_lower=0.05, alpha_upper=0.95,
            k0_lower=1e-8, k0_upper=100.0,
            log_space=True,
            mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
            max_eta_gap=3.0,
            optimizer_method="L-BFGS-B",
            optimizer_options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-6, "disp": True},
            max_iters=maxiter,
            live_plot=False,
            forward_recovery=_make_recovery(),
            # Regularization
            regularization_lambda=lam,
            regularization_k0_prior=prior_k0,
            regularization_alpha_prior=prior_alpha,
        )

        result_sub = run_bv_joint_flux_curve_inference(request_sub)
        p4_k0 = np.asarray(result_sub["best_k0"])
        p4_alpha = np.asarray(result_sub["best_alpha"])
        sub_time = time.time() - t_sub
        p4_total_time += sub_time

        _print_phase_result(f"Phase {sub_name}", p4_k0, p4_alpha,
                            true_k0_arr, true_alpha_arr,
                            result_sub["best_loss"], sub_time)

        _clear_caches()

    # Record final Phase 4 (4c) result
    phase_results["Phase 4c (joint, no reg)"] = {
        "k0": p4_k0.tolist(), "alpha": p4_alpha.tolist(),
        "loss": result_sub["best_loss"], "time": p4_total_time,
    }

    # ===================================================================
    # Decision: Phase 3 vs Phase 4c -- pick best overall
    # ===================================================================
    p3_k0_err, p3_alpha_err = _compute_errors(p3_k0, p3_alpha, true_k0_arr, true_alpha_arr)
    p4_k0_err, p4_alpha_err = _compute_errors(p4_k0, p4_alpha, true_k0_arr, true_alpha_arr)

    p3_max_err = max(p3_k0_err.max(), p3_alpha_err.max())
    p4_max_err = max(p4_k0_err.max(), p4_alpha_err.max())

    if p4_max_err <= p3_max_err:
        best_k0, best_alpha = p4_k0.copy(), p4_alpha.copy()
        best_source = "Phase 4c"
    else:
        best_k0, best_alpha = p3_k0.copy(), p3_alpha.copy()
        best_source = "Phase 3"

    best_max_err = min(p3_max_err, p4_max_err)

    print(f"\n{'='*70}")
    print(f"  P3 vs P4c: best is {best_source} (max err = {best_max_err*100:.2f}%)")
    print(f"{'='*70}")

    # ===================================================================
    # PHASE 5 (Optional): Multi-pH Validation
    # ===================================================================
    run_phase5 = best_max_err > 0.10
    if run_phase5:
        print(f"\n{'='*70}")
        print(f"  PHASE 5: Multi-pH validation (max err = {best_max_err*100:.1f}% > 10%)")
        print(f"  3 pH conditions, 19 eta points, Ny=300, beta=4.0")
        print(f"{'='*70}")
        t_p5 = time.time()

        # Phase 5 solver params with tighter settings
        dt_p5 = 0.1
        max_ss_steps_p5 = 500
        t_end_p5 = dt_p5 * max_ss_steps_p5
        base_sp_p5 = _make_bv_solver_params(
            eta_hat=0.0, dt=dt_p5, t_end=t_end_p5,
            snes_opts=SNES_OPTS_PHASE5, softplus=True,
        )

        steady_p5 = SteadyStateConfig(
            relative_tolerance=1e-4, absolute_tolerance=1e-8,
            consecutive_steps=4, max_steps=max_ss_steps_p5,
            flux_observable="total_species", verbose=False,
        )

        # 19-point denser eta grid
        eta_p5 = np.array([
            -1.0, -2.0, -3.0, -4.0, -5.0,
            -6.0, -7.0, -8.0, -9.0, -10.0,
            -12.0, -14.0, -17.0, -20.0,
            -24.0, -28.0, -32.0, -38.0, -46.5,
        ])

        p5_dir = os.path.join(base_output, "phase5_multi_ph")

        ph_conditions = [
            {
                "c_hp_hat": 0.1,
                "weight": 1.0,
                "c_hp_species_index": 2,
                "counterion_species_index": 3,
                "label": "c_H+=0.05 mol/m3 (easy)",
            },
            {
                "c_hp_hat": C_HP_HAT,
                "weight": 1.0,
                "c_hp_species_index": 2,
                "counterion_species_index": 3,
                "label": "c_H+=0.1 mol/m3 (baseline)",
            },
            {
                "c_hp_hat": 0.3,
                "weight": 1.0,
                "c_hp_species_index": 2,
                "counterion_species_index": 3,
                "label": "c_H+=0.15 mol/m3",
            },
        ]

        request_p5 = BVFluxCurveInferenceRequest(
            base_solver_params=base_sp_p5,
            steady=steady_p5,
            true_k0=true_k0,
            initial_guess=best_k0.tolist(),
            phi_applied_values=eta_p5.tolist(),
            target_csv_path=os.path.join(p5_dir, "target_base.csv"),
            output_dir=p5_dir,
            regenerate_target=True,
            target_noise_percent=2.0,
            target_seed=20260226,
            observable_mode="current_density",
            current_density_scale=observable_scale,
            observable_label="current density (mA/cm2)",
            observable_title="Phase 5: Multi-pH validation",
            control_mode="joint",
            true_alpha=true_alpha,
            initial_alpha_guess=best_alpha.tolist(),
            alpha_lower=0.05, alpha_upper=0.95,
            k0_lower=1e-8, k0_upper=100.0,
            log_space=True,
            mesh_Nx=8, mesh_Ny=300, mesh_beta=4.0,
            max_eta_gap=1.5,
            optimizer_method="L-BFGS-B",
            optimizer_options={"maxiter": 15, "ftol": 1e-12, "gtol": 1e-6, "disp": True},
            max_iters=15,
            live_plot=False,
            forward_recovery=_make_recovery(),
            multi_ph_conditions=ph_conditions,
        )

        result_p5 = run_bv_multi_ph_flux_curve_inference(request_p5)
        p5_k0 = np.asarray(result_p5["best_k0"])
        p5_alpha = np.asarray(result_p5["best_alpha"])
        p5_time = time.time() - t_p5

        _print_phase_result("Phase 5", p5_k0, p5_alpha, true_k0_arr, true_alpha_arr,
                            result_p5["best_loss"], p5_time)
        phase_results["Phase 5 (multi-pH)"] = {
            "k0": p5_k0.tolist(), "alpha": p5_alpha.tolist(),
            "loss": result_p5["best_loss"], "time": p5_time,
        }

        # Update best if Phase 5 improved
        p5_k0_err, p5_alpha_err = _compute_errors(p5_k0, p5_alpha, true_k0_arr, true_alpha_arr)
        p5_max_err = max(p5_k0_err.max(), p5_alpha_err.max())
        if p5_max_err < best_max_err:
            best_k0, best_alpha = p5_k0.copy(), p5_alpha.copy()
            best_source = "Phase 5"
            best_max_err = p5_max_err
    else:
        print(f"\n  Phase 5 SKIPPED (max err = {best_max_err*100:.1f}% <= 10%)")

    total_time = time.time() - t_total_start

    # ===================================================================
    # FINAL SUMMARY TABLE
    # ===================================================================
    print(f"\n{'#'*90}")
    print(f"  MASTER INFERENCE SUMMARY")
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
    csv_path = os.path.join(base_output, "master_comparison.csv")
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
    print(f"\n=== Master Inference Complete ===")


if __name__ == "__main__":
    main()
