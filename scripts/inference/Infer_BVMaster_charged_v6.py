"""Master inference protocol v6 for BV kinetics (charged system) -- MULTI-OBS PARALLEL.

Full 4-species charged system: O2, H2O2, H+, ClO4- (z=[0,0,+1,-1]).

Speed optimizations vs v5:
    1. MULTI-OBSERVABLE WORKERS (Strategy A): in multi-observable phases (P2, P3),
       each worker computes BOTH adjoint gradients (primary + secondary) after a
       single forward solve.  The "two-tape-pass" approach clears the adjoint tape
       and re-annotates from the converged IC for each observable, requiring only
       1 Newton iteration per tape pass.  This eliminates the ~330s sequential
       fallback for the secondary observable (~36% of v5 time in P2+P3).

    2. MORE WORKERS (Strategy D): default workers = min(n_points, cpu_count - 1).
       For 10-point P2 and 15-point P3 on a 10-core machine, this uses 9 workers
       instead of the previous 6.

    3. All v5 optimizations preserved: parallel fast path, persistent cache,
       SER adaptive dt, Jacobian lagging, predictor warm-starts, bridge points.

All other settings (voltage grids, optimizer config, convergence criteria)
are IDENTICAL to v5 to ensure accuracy comparison is fair.

Three-phase protocol:
    Phase 1: Alpha initialization via 12-pt symmetric sweep (alpha-only, k0 FIXED)
    Phase 2: Multi-observable joint on 10-pt SHALLOW cathodic [-1, -13]
             k0 from COLD guess, alpha warm-started from P1
    Phase 3: Multi-observable joint on 15-pt FULL cathodic [-1, -46.5]
             warm-start all 4 params from Phase 2

Usage (from PNPInverse/ directory)::

    python scripts/inference/Infer_BVMaster_charged_v6.py [--workers N]
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PNPINVERSE_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _PNPINVERSE_ROOT not in sys.path:
    sys.path.insert(0, _PNPINVERSE_ROOT)

from scripts._bv_common import (
    setup_firedrake_env,
    K0_HAT_R1, K0_HAT_R2, K_SCALE, I_SCALE,
    ALPHA_R1, ALPHA_R2,
    FOUR_SPECIES_CHARGED,
    SNES_OPTS_CHARGED,
    make_bv_solver_params,
    make_recovery_config,
    print_redimensionalized_results,
)
setup_firedrake_env()

# Backward-compat aliases used throughout this script
K0_HAT = K0_HAT_R1
K0_2_HAT = K0_HAT_R2
ALPHA_1 = ALPHA_R1
ALPHA_2 = ALPHA_R2

import numpy as np

from FluxCurve import (
    BVFluxCurveInferenceRequest,
    run_bv_alpha_flux_curve_inference,
    run_bv_multi_observable_flux_curve_inference,
)
from FluxCurve.bv_point_solve import (
    _clear_caches,
    set_parallel_pool,
    close_parallel_pool,
    _WARMSTART_MAX_STEPS,
    _SER_GROWTH_CAP,
    _SER_SHRINK,
    _SER_DT_MAX_RATIO,
)
from FluxCurve.bv_parallel import BVParallelPointConfig, BVPointSolvePool
from Forward.params import SolverParams
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


def _make_parallel_config(
    base_sp: SolverParams,
    steady: SteadyStateConfig,
    *,
    mesh_Nx: int,
    mesh_Ny: int,
    mesh_beta: float,
    observable_mode: str,
    observable_reaction_index,
    observable_scale: float,
    control_mode: str,
    n_controls: int,
    blob_initial_condition: bool = False,
    fail_penalty: float = 1e9,
    # Multi-observable fields (v6)
    secondary_observable_mode: str | None = None,
    secondary_observable_reaction_index: int | None = None,
    secondary_observable_scale: float | None = None,
) -> BVParallelPointConfig:
    """Build a BVParallelPointConfig from the script's solver params."""
    base_list = list(base_sp)
    return BVParallelPointConfig(
        base_solver_params=base_list,
        ss_relative_tolerance=float(steady.relative_tolerance),
        ss_absolute_tolerance=float(max(steady.absolute_tolerance, 1e-16)),
        ss_consecutive_steps=int(steady.consecutive_steps),
        ss_max_steps=int(steady.max_steps),
        mesh_Nx=mesh_Nx,
        mesh_Ny=mesh_Ny,
        mesh_beta=mesh_beta,
        blob_initial_condition=blob_initial_condition,
        fail_penalty=fail_penalty,
        warmstart_max_steps=_WARMSTART_MAX_STEPS,
        observable_mode=observable_mode,
        observable_reaction_index=observable_reaction_index,
        observable_scale=observable_scale,
        control_mode=control_mode,
        n_controls=n_controls,
        ser_growth_cap=_SER_GROWTH_CAP,
        ser_shrink=_SER_SHRINK,
        ser_dt_max_ratio=_SER_DT_MAX_RATIO,
        # Multi-observable fields (v6)
        secondary_observable_mode=secondary_observable_mode,
        secondary_observable_reaction_index=secondary_observable_reaction_index,
        secondary_observable_scale=secondary_observable_scale,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="BV Master Inference v6 (Multi-Obs Parallel)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of parallel workers (0=auto: min(n_points, cpu_count-1))")
    parser.add_argument("--no-parallel", action="store_true",
                        help="Disable parallel fast path (run as v4)")
    args = parser.parse_args()

    n_workers = args.workers
    use_parallel = not args.no_parallel

    # ===================================================================
    # Voltage grids (IDENTICAL to v4/v5)
    # ===================================================================
    # Phase 1: 12-point symmetric [-20, +5]
    eta_symmetric = np.array([
        +5.0, +3.0, +1.0,
        -0.5,
        -1.0, -2.0, -3.0,
        -5.0, -8.0,
        -10.0, -15.0, -20.0,
    ])

    # Phase 2: 10-point SHALLOW cathodic [-1, -13]
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

    # Deliberately wrong initial guesses (same as v4/v5)
    initial_k0_guess = [0.005, 0.0005]
    initial_alpha_guess = [0.4, 0.3]

    observable_scale = -I_SCALE
    base_output = os.path.join("StudyResults", "master_inference_v6")
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

    # Strategy D: auto-size workers based on phase point count
    def _auto_workers(n_points: int) -> int:
        if n_workers > 0:
            return n_workers
        return min(n_points, max(1, (os.cpu_count() or 4) - 1))

    parallel_tag = f"PARALLEL MULTI-OBS ({n_workers or 'auto'} workers)" if use_parallel else "SERIAL (parallel disabled)"

    print(f"\n{'#'*70}")
    print(f"  MASTER INFERENCE PROTOCOL v6 ({parallel_tag})")
    print(f"  Changes vs v5: multi-observable workers (Strategy A),")
    print(f"  auto-sized worker count (Strategy D)")
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

    # Set up parallel pool for Phase 1 (alpha-only, current_density observable)
    # Phase 1 is single-observable so no multi-obs needed
    if use_parallel:
        p1_n_workers = _auto_workers(len(eta_symmetric))
        p1_config = _make_parallel_config(
            base_sp, steady,
            mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
            observable_mode="current_density",
            observable_reaction_index=None,
            observable_scale=observable_scale,
            control_mode="alpha",
            n_controls=len(initial_alpha_guess),
        )
        p1_pool = BVPointSolvePool(p1_config, n_workers=p1_n_workers)
        set_parallel_pool(p1_pool)

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
        parallel_fast_path=use_parallel,
        parallel_workers=p1_n_workers if use_parallel else 0,
    )

    result_p1 = run_bv_alpha_flux_curve_inference(request_p1)
    p1_alpha = np.asarray(result_p1["best_alpha"])
    p1_k0 = np.asarray(initial_k0_guess)  # unchanged (fixed)
    p1_time = time.time() - t_p1

    if use_parallel:
        close_parallel_pool()

    _print_phase_result("Phase 1", p1_k0, p1_alpha, true_k0_arr, true_alpha_arr,
                        result_p1["best_loss"], p1_time)
    phase_results["Phase 1 (alpha, sym 12pt)"] = {
        "k0": p1_k0.tolist(), "alpha": p1_alpha.tolist(),
        "loss": result_p1["best_loss"], "time": p1_time,
    }

    _clear_caches()

    # ===================================================================
    # PHASE 2: Multi-Observable Joint (total + peroxide current)
    #          k0 from COLD initial guess, alpha warm-started from P1
    #
    # v6 KEY CHANGE: pool configured with secondary_observable_mode so
    # workers compute BOTH observables in a single task.
    # ===================================================================
    print(f"\n{'='*70}")
    print(f"  PHASE 2: Multi-observable joint (total + peroxide current)")
    print(f"  k0 from COLD initial guess: {initial_k0_guess}")
    print(f"  alpha warm-started from Phase 1: {p1_alpha.tolist()}")
    print(f"  {len(eta_shallow)}-pt SHALLOW cathodic [{eta_shallow.min():.1f}, {eta_shallow.max():.1f}]")
    print(f"  secondary_weight=1.0, maxiter=30, gtol=1e-6, ftol=1e-10")
    print(f"  [v6] Multi-obs workers: each worker computes 2 adjoints per forward solve")
    print(f"{'='*70}")
    t_p2 = time.time()

    n_joint_controls = len(initial_k0_guess) + len(p1_alpha)

    if use_parallel:
        p2_n_workers = _auto_workers(len(eta_shallow))
        p2_config = _make_parallel_config(
            base_sp, steady,
            mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
            observable_mode="current_density",
            observable_reaction_index=None,
            observable_scale=observable_scale,
            control_mode="joint",
            n_controls=n_joint_controls,
            # v6: multi-obs fields
            secondary_observable_mode="peroxide_current",
            secondary_observable_reaction_index=None,
            secondary_observable_scale=observable_scale,
        )
        p2_pool = BVPointSolvePool(p2_config, n_workers=p2_n_workers)
        set_parallel_pool(p2_pool)

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
        optimizer_options={"maxiter": 30, "ftol": 1e-10, "gtol": 1e-6, "disp": True},
        max_iters=30,
        live_plot=False,
        forward_recovery=make_recovery_config(max_it_cap=600),
        parallel_fast_path=use_parallel,
        parallel_workers=p2_n_workers if use_parallel else 0,
    )

    result_p2 = run_bv_multi_observable_flux_curve_inference(request_p2)
    p2_k0 = np.asarray(result_p2["best_k0"])
    p2_alpha = np.asarray(result_p2["best_alpha"])
    p2_time = time.time() - t_p2

    if use_parallel:
        close_parallel_pool()

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
    # ===================================================================
    print(f"\n{'='*70}")
    print(f"  PHASE 3: Multi-observable joint on FULL cathodic range")
    print(f"  Warm-start from Phase 2: k0={p2_k0.tolist()}, alpha={p2_alpha.tolist()}")
    print(f"  {len(eta_cathodic)}-pt FULL cathodic [{eta_cathodic.min():.1f}, {eta_cathodic.max():.1f}]")
    print(f"  total + peroxide current, secondary_weight=1.0")
    print(f"  maxiter=25, gtol=1e-6, ftol=1e-10")
    print(f"  [v6] Multi-obs workers: each worker computes 2 adjoints per forward solve")
    print(f"{'='*70}")
    t_p3 = time.time()

    n_joint_controls_p3 = len(p2_k0) + len(p2_alpha)

    if use_parallel:
        p3_n_workers = _auto_workers(len(eta_cathodic))
        p3_config = _make_parallel_config(
            base_sp, steady,
            mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
            observable_mode="current_density",
            observable_reaction_index=None,
            observable_scale=observable_scale,
            control_mode="joint",
            n_controls=n_joint_controls_p3,
            # v6: multi-obs fields
            secondary_observable_mode="peroxide_current",
            secondary_observable_reaction_index=None,
            secondary_observable_scale=observable_scale,
        )
        p3_pool = BVPointSolvePool(p3_config, n_workers=p3_n_workers)
        set_parallel_pool(p3_pool)

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
        optimizer_options={"maxiter": 25, "ftol": 1e-10, "gtol": 1e-6, "disp": True},
        max_iters=25,
        live_plot=False,
        forward_recovery=make_recovery_config(max_it_cap=600),
        parallel_fast_path=use_parallel,
        parallel_workers=p3_n_workers if use_parallel else 0,
    )

    result_p3 = run_bv_multi_observable_flux_curve_inference(request_p3)
    p3_k0 = np.asarray(result_p3["best_k0"])
    p3_alpha = np.asarray(result_p3["best_alpha"])
    p3_time = time.time() - t_p3

    if use_parallel:
        close_parallel_pool()

    _print_phase_result("Phase 3", p3_k0, p3_alpha, true_k0_arr, true_alpha_arr,
                        result_p3["best_loss"], p3_time)
    phase_results["Phase 3 (multi-obs, full cat)"] = {
        "k0": p3_k0.tolist(), "alpha": p3_alpha.tolist(),
        "loss": result_p3["best_loss"], "time": p3_time,
    }

    # ===================================================================
    # Best-of Selection (Phase 2 vs Phase 3)
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
    print(f"  MASTER INFERENCE v6 SUMMARY ({parallel_tag})")
    print(f"{'#'*90}")
    print(f"  True k0:    {true_k0}")
    print(f"  True alpha: {true_alpha}")
    print(f"  K0_HAT = {K0_HAT:.6e},  K0_2_HAT = {K0_2_HAT:.6e}")
    print()

    header = (f"{'Phase':<35} | {'k0_1 err':>10} {'k0_2 err':>10} "
              f"{'a1 err':>10} {'a2 err':>10} | {'loss':>12} | {'time':>6}")
    print(header)
    print(f"{'-'*95}")

    for name, ph in phase_results.items():
        k0_err, alpha_err = _compute_errors(
            ph["k0"], ph["alpha"], true_k0_arr, true_alpha_arr
        )
        print(f"{name:<35} | {k0_err[0]*100:>9.2f}% {k0_err[1]*100:>9.2f}% "
              f"{alpha_err[0]*100:>9.2f}% {alpha_err[1]*100:>9.2f}% "
              f"| {ph['loss']:>12.6e} | {ph['time']:>5.0f}s")

    print(f"{'-'*95}")
    print(f"  Total time: {total_time:.0f}s")

    best_k0_err, best_alpha_err = _compute_errors(best_k0, best_alpha, true_k0_arr, true_alpha_arr)

    print(f"  Best result: {best_source} (max err = {best_max_err*100:.2f}%)")
    print(f"    k0_1   = {best_k0[0]:.6e}  (err {best_k0_err[0]*100:.2f}%)")
    print(f"    k0_2   = {best_k0[1]:.6e}  (err {best_k0_err[1]*100:.2f}%)")
    print(f"    alpha_1= {best_alpha[0]:.6f}  (err {best_alpha_err[0]*100:.2f}%)")
    print(f"    alpha_2= {best_alpha[1]:.6f}  (err {best_alpha_err[1]*100:.2f}%)")

    print_redimensionalized_results(
        best_k0, true_k0_arr,
        best_alpha=best_alpha, true_alpha=true_alpha_arr,
    )
    print(f"{'#'*90}")

    # Save comparison CSV
    csv_path = os.path.join(base_output, "master_comparison_v6.csv")
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
    print(f"\n=== Master Inference v6 Complete ===")


if __name__ == "__main__":
    main()
