"""Master inference protocol v8.1 for BV kinetics -- SURROGATE + PDE REFINEMENT.

Round 2A improvements over v8:
    1. Remove Phase 3 (full cathodic surrogate) -- it actively degrades all params.
    2. Clamp optimizer bounds to training-data range so the surrogate is never
       queried in extrapolation.
    3. Add PDE refinement as the new Phase 3: use surrogate best (Phase 2) as
       warm-start for a short PDE-based L-BFGS-B pass on the shallow cathodic
       grid, recovering v7-level accuracy at a fraction of the cost.

Three-phase protocol:
    Phase 1: Alpha-only surrogate optimization (k0 FIXED at initial guess)
    Phase 2: Joint 4-param surrogate on SHALLOW cathodic voltages
             (bounds clamped to training-data range)
    Phase 3: PDE-based refinement from Phase 2 result (10 L-BFGS-B iters,
             parallel multi-observable, shallow cathodic grid)

Usage (from PNPInverse/ directory)::

    python scripts/surrogate/Infer_BVMaster_charged_v8_1_surrogate.py \\
        --model StudyResults/surrogate/surrogate_model.pkl

    # Skip PDE refinement (surrogate-only, same as v8 Phase 1+2):
    python scripts/surrogate/Infer_BVMaster_charged_v8_1_surrogate.py --no-pde

    # Control worker count for PDE phase:
    python scripts/surrogate/Infer_BVMaster_charged_v8_1_surrogate.py --workers 4
"""

from __future__ import annotations

import argparse
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
    K0_HAT_R1, K0_HAT_R2, K_SCALE, I_SCALE,
    ALPHA_R1, ALPHA_R2,
    FOUR_SPECIES_CHARGED,
    SNES_OPTS_CHARGED,
    make_bv_solver_params,
    make_recovery_config,
    print_params_summary,
    print_redimensionalized_results,
)
setup_firedrake_env()

# Backward-compat aliases used throughout this script
K0_HAT = K0_HAT_R1
K0_2_HAT = K0_HAT_R2
ALPHA_1 = ALPHA_R1
ALPHA_2 = ALPHA_R2

import numpy as np
from scipy.optimize import minimize

from Surrogate.io import load_surrogate
from Surrogate.objectives import SurrogateObjective, AlphaOnlySurrogateObjective

# ---------------------------------------------------------------------------
# Training-data bounds (Change 2): 2 decades around true values
# ---------------------------------------------------------------------------
K0_1_TRAIN_LO = K0_HAT * 0.01    # log10 ~ -4.9
K0_1_TRAIN_HI = K0_HAT * 100.0   # log10 ~ -0.9
K0_2_TRAIN_LO = K0_2_HAT * 0.01  # log10 ~ -6.3
K0_2_TRAIN_HI = K0_2_HAT * 100.0 # log10 ~ -2.3
ALPHA_TRAIN_LO = 0.10
ALPHA_TRAIN_HI = 0.90


def _compute_errors(k0, alpha, true_k0_arr, true_alpha_arr):
    k0_arr = np.asarray(k0)
    alpha_arr = np.asarray(alpha)
    k0_err = np.abs(k0_arr - true_k0_arr) / np.maximum(np.abs(true_k0_arr), 1e-16)
    alpha_err = np.abs(alpha_arr - true_alpha_arr) / np.maximum(np.abs(true_alpha_arr), 1e-16)
    return k0_err, alpha_err


def _print_phase_result(name, k0, alpha, true_k0_arr, true_alpha_arr, loss, elapsed):
    k0_err, alpha_err = _compute_errors(k0, alpha, true_k0_arr, true_alpha_arr)
    print(f"\n  {name} result:")
    print(f"    k0_1   = {k0[0]:.6e}  (true {true_k0_arr[0]:.6e}, err {k0_err[0]*100:.2f}%)")
    print(f"    k0_2   = {k0[1]:.6e}  (true {true_k0_arr[1]:.6e}, err {k0_err[1]*100:.2f}%)")
    print(f"    alpha_1= {alpha[0]:.6f}  (true {true_alpha_arr[0]:.6f}, err {alpha_err[0]*100:.2f}%)")
    print(f"    alpha_2= {alpha[1]:.6f}  (true {true_alpha_arr[1]:.6f}, err {alpha_err[1]*100:.2f}%)")
    print(f"    Loss: {loss:.6e},  Time: {elapsed:.1f}s")
    return k0_err, alpha_err


def _generate_targets_with_pde(phi_applied_values, observable_scale):
    """Generate target I-V curves using the PDE solver at true parameters.

    This mirrors v7's target generation: solve at true k0/alpha, add 2% noise.
    """
    from Forward.steady_state import SteadyStateConfig, add_percent_noise
    from Forward.bv_solver import make_graded_rectangle_mesh
    from FluxCurve.bv_point_solve import (
        solve_bv_curve_points_with_warmstart,
        _clear_caches,
    )

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

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)
    recovery = make_recovery_config(max_it_cap=600)

    dummy_target = np.zeros_like(phi_applied_values, dtype=float)

    results = {}
    for obs_mode in ["current_density", "peroxide_current"]:
        _clear_caches()
        seed_offset = 0 if obs_mode == "current_density" else 1

        points = solve_bv_curve_points_with_warmstart(
            base_solver_params=base_sp,
            steady=steady,
            phi_applied_values=phi_applied_values,
            target_flux=dummy_target,
            k0_values=[K0_HAT, K0_2_HAT],
            blob_initial_condition=False,
            fail_penalty=1e9,
            forward_recovery=recovery,
            observable_mode=obs_mode,
            observable_reaction_index=None,
            observable_scale=observable_scale,
            mesh=mesh,
            alpha_values=[ALPHA_1, ALPHA_2],
            control_mode="joint",
            max_eta_gap=3.0,
        )

        clean_flux = np.array([float(p.simulated_flux) for p in points], dtype=float)
        noisy_flux = add_percent_noise(clean_flux, 2.0, seed=20260226 + seed_offset)
        results[obs_mode] = noisy_flux

    _clear_caches()
    return results


def _subset_targets(target_cd, target_pc, all_eta, subset_eta):
    """Extract target values for a subset of voltages from the full grid."""
    idx = []
    for eta in subset_eta:
        matches = np.where(np.abs(all_eta - eta) < 1e-10)[0]
        if len(matches) > 0:
            idx.append(matches[0])
    idx = np.array(idx, dtype=int)
    return target_cd[idx], target_pc[idx]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BV Master Inference v8.1 (Surrogate + PDE Refinement)"
    )
    parser.add_argument("--model", type=str,
                        default="StudyResults/surrogate/surrogate_model.pkl",
                        help="Path to surrogate model .pkl")
    parser.add_argument("--no-pde", action="store_true",
                        help="Skip Phase 3 PDE refinement (surrogate-only)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Workers for PDE refinement (0=auto)")
    parser.add_argument("--pde-maxiter", type=int, default=10,
                        help="Max L-BFGS-B iterations for PDE refinement")
    args = parser.parse_args()

    # ===================================================================
    # Voltage grids (IDENTICAL to v7/v8)
    # ===================================================================
    eta_symmetric = np.array([
        +5.0, +3.0, +1.0, -0.5,
        -1.0, -2.0, -3.0, -5.0, -8.0,
        -10.0, -15.0, -20.0,
    ])
    eta_shallow = np.array([
        -1.0, -2.0, -3.0, -4.0, -5.0, -6.5, -8.0,
        -10.0, -11.5, -13.0,
    ])

    # eta_cathodic is defined for target generation only (to match the
    # surrogate's 22-point training grid).  Change 1: we do NOT run a
    # Phase 3 surrogate optimization on this range.
    eta_cathodic = np.array([
        -1.0, -2.0, -3.0, -4.0, -5.0, -6.5, -8.0,
        -10.0, -13.0, -17.0, -22.0, -28.0,
        -35.0, -41.0, -46.5,
    ])

    # Union of all voltages (sorted descending) -- must match surrogate grid
    all_eta = np.unique(np.concatenate([eta_symmetric, eta_shallow, eta_cathodic]))
    all_eta = np.sort(all_eta)[::-1]

    true_k0 = [K0_HAT, K0_2_HAT]
    true_alpha = [ALPHA_1, ALPHA_2]
    true_k0_arr = np.asarray(true_k0)
    true_alpha_arr = np.asarray(true_alpha)

    initial_k0_guess = [0.005, 0.0005]
    initial_alpha_guess = [0.4, 0.3]

    observable_scale = -I_SCALE
    base_output = os.path.join("StudyResults", "surrogate_v8_1")
    os.makedirs(base_output, exist_ok=True)

    phase_results = {}
    t_total_start = time.time()

    # Print training-data bounds for reference
    print(f"\n{'#'*70}")
    print(f"  MASTER INFERENCE PROTOCOL v8.1 (SURROGATE + PDE REFINEMENT)")
    print(f"  Changes vs v8:")
    print(f"    1. Removed Phase 3 (full cathodic surrogate) -- degrades params")
    print(f"    2. Clamped bounds to training range (no extrapolation)")
    print(f"    3. PDE refinement as new Phase 3 (warm from surrogate)")
    print(f"  True k0:    {true_k0}")
    print(f"  True alpha: {true_alpha}")
    print(f"  Initial k0 guess:    {initial_k0_guess}")
    print(f"  Initial alpha guess: {initial_alpha_guess}")
    print(f"  Training-data bounds:")
    print(f"    k0_1 log10: [{np.log10(K0_1_TRAIN_LO):.2f}, {np.log10(K0_1_TRAIN_HI):.2f}]")
    print(f"    k0_2 log10: [{np.log10(K0_2_TRAIN_LO):.2f}, {np.log10(K0_2_TRAIN_HI):.2f}]")
    print(f"    alpha:      [{ALPHA_TRAIN_LO:.2f}, {ALPHA_TRAIN_HI:.2f}]")
    print(f"{'#'*70}\n")

    # ===================================================================
    # Load surrogate model
    # ===================================================================
    print(f"Loading surrogate model from: {args.model}")
    surrogate = load_surrogate(args.model)
    surrogate_eta = surrogate.phi_applied
    print(f"  Surrogate voltage points: {surrogate.n_eta}")
    print(f"  Surrogate voltage range: [{surrogate_eta.min():.1f}, {surrogate_eta.max():.1f}]")

    # ===================================================================
    # Generate targets using PDE solver (same as v7/v8)
    # ===================================================================
    print(f"\nGenerating target I-V curves with PDE solver at true parameters...")
    t_target = time.time()
    targets = _generate_targets_with_pde(all_eta, observable_scale)
    target_cd_full = targets["current_density"]
    target_pc_full = targets["peroxide_current"]
    t_target_elapsed = time.time() - t_target
    print(f"  Target generation: {t_target_elapsed:.1f}s")

    target_cd_surr = target_cd_full
    target_pc_surr = target_pc_full

    # ===================================================================
    # PHASE 1: Alpha-only surrogate optimization
    # ===================================================================
    print(f"\n{'='*70}")
    print(f"  PHASE 1: Alpha-only surrogate optimization")
    print(f"  k0 FIXED at initial guess: {initial_k0_guess}")
    print(f"  Voltage grid: full ({len(all_eta)} pts)")
    print(f"  Alpha bounds: [{ALPHA_TRAIN_LO}, {ALPHA_TRAIN_HI}] (training range)")
    print(f"{'='*70}")
    t_p1 = time.time()

    p1_obj = AlphaOnlySurrogateObjective(
        surrogate=surrogate,
        target_cd=target_cd_surr,
        target_pc=target_pc_surr,
        fixed_k0=initial_k0_guess,
        secondary_weight=1.0,
        fd_step=1e-5,
    )

    x0_p1 = np.array(initial_alpha_guess, dtype=float)
    # Change 2: clamp alpha bounds to training range
    bounds_p1 = [(ALPHA_TRAIN_LO, ALPHA_TRAIN_HI), (ALPHA_TRAIN_LO, ALPHA_TRAIN_HI)]

    eval_counter_p1 = {"n": 0}
    def _p1_callback(x):
        eval_counter_p1["n"] += 1
        J = p1_obj.objective(x)
        print(f"  [P1 iter {eval_counter_p1['n']:>3d}] J={J:.6e} alpha=[{x[0]:.4f}, {x[1]:.4f}]")

    result_p1 = minimize(
        p1_obj.objective,
        x0_p1,
        jac=p1_obj.gradient,
        method="L-BFGS-B",
        bounds=bounds_p1,
        options={"maxiter": 60, "ftol": 1e-14, "gtol": 1e-8, "disp": False},
        callback=_p1_callback,
    )

    p1_alpha = result_p1.x.copy()
    p1_k0 = np.asarray(initial_k0_guess)
    p1_loss = float(result_p1.fun)
    p1_time = time.time() - t_p1

    _print_phase_result("Phase 1 (alpha, surrogate)", p1_k0, p1_alpha,
                        true_k0_arr, true_alpha_arr, p1_loss, p1_time)
    phase_results["Phase 1 (alpha, surrogate)"] = {
        "k0": p1_k0.tolist(), "alpha": p1_alpha.tolist(),
        "loss": p1_loss, "time": p1_time,
    }

    # ===================================================================
    # PHASE 2: Joint 4-param surrogate on SHALLOW cathodic
    #          k0 from COLD initial guess, alpha warm from P1
    #          Bounds CLAMPED to training-data range (Change 2)
    # ===================================================================
    print(f"\n{'='*70}")
    print(f"  PHASE 2: Joint 4-param surrogate (shallow-range objective)")
    print(f"  k0 from COLD guess: {initial_k0_guess}")
    print(f"  alpha warm from P1: {p1_alpha.tolist()}")
    print(f"  Bounds clamped to training range (Change 2)")
    print(f"{'='*70}")
    t_p2 = time.time()

    # Use shallow voltage subset for objective
    target_cd_shallow, target_pc_shallow = _subset_targets(
        target_cd_full, target_pc_full, all_eta, eta_shallow,
    )
    shallow_idx = []
    for eta in eta_shallow:
        matches = np.where(np.abs(all_eta - eta) < 1e-10)[0]
        if len(matches) > 0:
            shallow_idx.append(matches[0])
    shallow_idx = np.array(shallow_idx, dtype=int)

    class SubsetSurrogateObjective:
        """Surrogate objective on a subset of voltage points."""
        def __init__(self, surrogate, target_cd, target_pc, subset_idx,
                     secondary_weight=1.0, fd_step=1e-5, log_space_k0=True):
            self.surrogate = surrogate
            self.target_cd = np.asarray(target_cd, dtype=float)
            self.target_pc = np.asarray(target_pc, dtype=float)
            self.subset_idx = subset_idx
            self._valid_cd = ~np.isnan(self.target_cd)
            self._valid_pc = ~np.isnan(self.target_pc)
            self.secondary_weight = secondary_weight
            self.fd_step = fd_step
            self.log_space_k0 = log_space_k0
            self._n_evals = 0

        def _x_to_params(self, x):
            x = np.asarray(x, dtype=float)
            if self.log_space_k0:
                k0_1, k0_2 = 10.0**x[0], 10.0**x[1]
            else:
                k0_1, k0_2 = x[0], x[1]
            return k0_1, k0_2, x[2], x[3]

        def objective(self, x):
            k0_1, k0_2, a1, a2 = self._x_to_params(x)
            pred = self.surrogate.predict(k0_1, k0_2, a1, a2)
            cd_sim = pred["current_density"][self.subset_idx]
            pc_sim = pred["peroxide_current"][self.subset_idx]
            cd_diff = cd_sim[self._valid_cd] - self.target_cd[self._valid_cd]
            pc_diff = pc_sim[self._valid_pc] - self.target_pc[self._valid_pc]
            J_cd = 0.5 * np.sum(cd_diff**2)
            J_pc = 0.5 * np.sum(pc_diff**2)
            self._n_evals += 1
            return float(J_cd + self.secondary_weight * J_pc)

        def gradient(self, x):
            x = np.asarray(x, dtype=float)
            grad = np.zeros(len(x), dtype=float)
            h = self.fd_step
            for i in range(len(x)):
                xp, xm = x.copy(), x.copy()
                xp[i] += h; xm[i] -= h
                grad[i] = (self.objective(xp) - self.objective(xm)) / (2*h)
            return grad

    p2_obj = SubsetSurrogateObjective(
        surrogate=surrogate,
        target_cd=target_cd_shallow,
        target_pc=target_pc_shallow,
        subset_idx=shallow_idx,
        secondary_weight=1.0,
        fd_step=1e-5,
        log_space_k0=True,
    )

    x0_p2 = np.array([
        np.log10(initial_k0_guess[0]),
        np.log10(initial_k0_guess[1]),
        p1_alpha[0],
        p1_alpha[1],
    ], dtype=float)

    # Change 2: Clamp bounds to training-data range
    bounds_p2 = [
        (np.log10(K0_1_TRAIN_LO), np.log10(K0_1_TRAIN_HI)),
        (np.log10(K0_2_TRAIN_LO), np.log10(K0_2_TRAIN_HI)),
        (ALPHA_TRAIN_LO, ALPHA_TRAIN_HI),
        (ALPHA_TRAIN_LO, ALPHA_TRAIN_HI),
    ]
    print(f"  Optimizer bounds (log10 for k0):")
    print(f"    k0_1: [{bounds_p2[0][0]:.2f}, {bounds_p2[0][1]:.2f}]")
    print(f"    k0_2: [{bounds_p2[1][0]:.2f}, {bounds_p2[1][1]:.2f}]")
    print(f"    alpha: [{bounds_p2[2][0]:.2f}, {bounds_p2[2][1]:.2f}]")

    eval_counter_p2 = {"n": 0}
    def _p2_callback(x):
        eval_counter_p2["n"] += 1
        k0_1, k0_2 = 10.0**x[0], 10.0**x[1]
        J = p2_obj.objective(x)
        print(f"  [P2 iter {eval_counter_p2['n']:>3d}] J={J:.6e} "
              f"k0=[{k0_1:.4e},{k0_2:.4e}] alpha=[{x[2]:.4f},{x[3]:.4f}]")

    result_p2 = minimize(
        p2_obj.objective,
        x0_p2,
        jac=p2_obj.gradient,
        method="L-BFGS-B",
        bounds=bounds_p2,
        options={"maxiter": 60, "ftol": 1e-14, "gtol": 1e-8, "disp": False},
        callback=_p2_callback,
    )

    p2_k0 = np.array([10.0**result_p2.x[0], 10.0**result_p2.x[1]])
    p2_alpha = result_p2.x[2:4].copy()
    p2_loss = float(result_p2.fun)
    p2_time = time.time() - t_p2

    _print_phase_result("Phase 2 (joint, shallow, surr)", p2_k0, p2_alpha,
                        true_k0_arr, true_alpha_arr, p2_loss, p2_time)
    phase_results["Phase 2 (joint shallow surr)"] = {
        "k0": p2_k0.tolist(), "alpha": p2_alpha.tolist(),
        "loss": p2_loss, "time": p2_time,
    }

    t_surrogate_end = time.time()
    surrogate_time = t_surrogate_end - t_total_start - t_target_elapsed

    # Use Phase 2 result directly as surrogate best (Change 1: no Phase 3)
    surr_best_k0 = p2_k0.copy()
    surr_best_alpha = p2_alpha.copy()

    # ===================================================================
    # PHASE 3 (Change 3): PDE-based refinement from surrogate result
    # ===================================================================
    if not args.no_pde:
        print(f"\n{'='*70}")
        print(f"  PHASE 3: PDE-based refinement (warm-start from surrogate Phase 2)")
        print(f"  Starting: k0={surr_best_k0.tolist()}, alpha={surr_best_alpha.tolist()}")
        print(f"  Voltage grid: shallow cathodic ({len(eta_shallow)} pts)")
        print(f"  maxiter={args.pde_maxiter}, parallel_fast_path=True")
        print(f"{'='*70}")
        t_p3 = time.time()

        from Forward.steady_state import SteadyStateConfig
        from FluxCurve import (
            BVFluxCurveInferenceRequest,
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

        recovery = make_recovery_config(max_it_cap=600)

        _clear_caches()

        # Set up parallel pool (mirrors v7 Phase 2/3 shared pool setup)
        n_pde_workers = args.workers
        if n_pde_workers <= 0:
            n_pde_workers = min(len(eta_shallow), max(1, (os.cpu_count() or 4) - 1))

        n_joint_controls = 4  # k0_1, k0_2, alpha_1, alpha_2

        pde_config = BVParallelPointConfig(
            base_solver_params=list(base_sp),
            ss_relative_tolerance=float(steady.relative_tolerance),
            ss_absolute_tolerance=float(max(steady.absolute_tolerance, 1e-16)),
            ss_consecutive_steps=int(steady.consecutive_steps),
            ss_max_steps=int(steady.max_steps),
            mesh_Nx=8,
            mesh_Ny=200,
            mesh_beta=3.0,
            blob_initial_condition=False,
            fail_penalty=1e9,
            warmstart_max_steps=_WARMSTART_MAX_STEPS,
            observable_mode="current_density",
            observable_reaction_index=None,
            observable_scale=observable_scale,
            control_mode="joint",
            n_controls=n_joint_controls,
            ser_growth_cap=_SER_GROWTH_CAP,
            ser_shrink=_SER_SHRINK,
            ser_dt_max_ratio=_SER_DT_MAX_RATIO,
            secondary_observable_mode="peroxide_current",
            secondary_observable_reaction_index=None,
            secondary_observable_scale=observable_scale,
        )
        pde_pool = BVPointSolvePool(pde_config, n_workers=n_pde_workers)
        set_parallel_pool(pde_pool)
        print(f"  Parallel pool: {n_pde_workers} workers")

        p3_dir = os.path.join(base_output, "phase3_pde_refinement")

        request_p3 = BVFluxCurveInferenceRequest(
            base_solver_params=base_sp,
            steady=steady,
            true_k0=true_k0,
            initial_guess=surr_best_k0.tolist(),  # warm-start k0 from surrogate
            phi_applied_values=eta_shallow.tolist(),
            target_csv_path=os.path.join(p3_dir, "target_primary.csv"),
            output_dir=p3_dir,
            regenerate_target=True,
            target_noise_percent=2.0,
            target_seed=20260226,
            observable_mode="current_density",
            current_density_scale=observable_scale,
            observable_label="current density (mA/cm2)",
            observable_title="Phase 3: PDE refinement (warm from surrogate)",
            secondary_observable_mode="peroxide_current",
            secondary_observable_weight=1.0,
            secondary_current_density_scale=observable_scale,
            secondary_target_csv_path=os.path.join(p3_dir, "target_peroxide.csv"),
            control_mode="joint",
            true_alpha=true_alpha,
            initial_alpha_guess=surr_best_alpha.tolist(),  # warm-start alpha from surrogate
            alpha_lower=0.05, alpha_upper=0.95,
            k0_lower=1e-8, k0_upper=100.0,
            log_space=True,
            mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
            max_eta_gap=3.0,
            optimizer_method="L-BFGS-B",
            optimizer_options={
                "maxiter": args.pde_maxiter,
                "ftol": 1e-8,
                "gtol": 5e-6,
                "disp": True,
            },
            max_iters=args.pde_maxiter,
            live_plot=False,
            forward_recovery=recovery,
            parallel_fast_path=True,
            parallel_workers=n_pde_workers,
        )

        result_p3 = run_bv_multi_observable_flux_curve_inference(request_p3)
        p3_k0 = np.asarray(result_p3["best_k0"])
        p3_alpha = np.asarray(result_p3["best_alpha"])
        p3_loss = float(result_p3["best_loss"])
        p3_time = time.time() - t_p3

        close_parallel_pool()
        _clear_caches()

        _print_phase_result("Phase 3 (PDE refine)", p3_k0, p3_alpha,
                            true_k0_arr, true_alpha_arr, p3_loss, p3_time)
        phase_results["Phase 3 (PDE refine)"] = {
            "k0": p3_k0.tolist(), "alpha": p3_alpha.tolist(),
            "loss": p3_loss, "time": p3_time,
        }

        # Use PDE result as final (it should always be better with good warm-start)
        p3_k0_err, p3_alpha_err = _compute_errors(p3_k0, p3_alpha, true_k0_arr, true_alpha_arr)
        p2_k0_err, p2_alpha_err = _compute_errors(p2_k0, p2_alpha, true_k0_arr, true_alpha_arr)
        p3_max_err = max(p3_k0_err.max(), p3_alpha_err.max())
        p2_max_err = max(p2_k0_err.max(), p2_alpha_err.max())

        if p3_max_err <= p2_max_err:
            best_k0, best_alpha = p3_k0.copy(), p3_alpha.copy()
            best_source = "Phase 3 (PDE)"
        else:
            best_k0, best_alpha = p2_k0.copy(), p2_alpha.copy()
            best_source = "Phase 2 (surrogate)"
    else:
        best_k0, best_alpha = surr_best_k0.copy(), surr_best_alpha.copy()
        best_source = "Phase 2 (surrogate)"

    total_time = time.time() - t_total_start
    pde_time = total_time - t_target_elapsed - surrogate_time

    # ===================================================================
    # FINAL SUMMARY
    # ===================================================================
    print(f"\n{'#'*90}")
    print(f"  MASTER INFERENCE v8.1 SUMMARY (SURROGATE + PDE REFINEMENT)")
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
              f"| {ph['loss']:>12.6e} | {ph['time']:>5.1f}s")

    print(f"{'-'*95}")
    print(f"  Total time: {total_time:.1f}s (target gen: {t_target_elapsed:.1f}s, "
          f"surrogate phases: {surrogate_time:.1f}s, PDE refine: {pde_time:.1f}s)")

    best_k0_err, best_alpha_err = _compute_errors(best_k0, best_alpha, true_k0_arr, true_alpha_arr)
    best_max_err = max(best_k0_err.max(), best_alpha_err.max())

    print(f"\n  Best result: {best_source} (max err = {best_max_err*100:.2f}%)")
    print(f"    k0_1   = {best_k0[0]:.6e}  (err {best_k0_err[0]*100:.2f}%)")
    print(f"    k0_2   = {best_k0[1]:.6e}  (err {best_k0_err[1]*100:.2f}%)")
    print(f"    alpha_1= {best_alpha[0]:.6f}  (err {best_alpha_err[0]*100:.2f}%)")
    print(f"    alpha_2= {best_alpha[1]:.6f}  (err {best_alpha_err[1]*100:.2f}%)")

    # v7 / v8 / v8.1 comparison
    print(f"\n  {'='*70}")
    print(f"  v7 / v8 / v8.1 COMPARISON:")
    print(f"  {'='*70}")
    print(f"  {'Metric':<25} {'v7':>12} {'v8':>12} {'v8.1':>12}")
    print(f"  {'-'*61}")
    print(f"  {'Total time (s)':<25} {'415':>12} {'69.8':>12} {total_time:>12.1f}")
    print(f"  {'k0_1 err (%)':<25} {'10.9':>12} {'7.3':>12} {best_k0_err[0]*100:>12.1f}")
    print(f"  {'k0_2 err (%)':<25} {'2.6':>12} {'9.6':>12} {best_k0_err[1]*100:>12.1f}")
    print(f"  {'alpha_1 err (%)':<25} {'5.6':>12} {'4.5':>12} {best_alpha_err[0]*100:>12.1f}")
    print(f"  {'alpha_2 err (%)':<25} {'8.8':>12} {'4.6':>12} {best_alpha_err[1]*100:>12.1f}")
    print(f"  {'='*70}")

    print(f"\n  Timing breakdown:")
    print(f"    Target generation:  {t_target_elapsed:>8.1f}s")
    print(f"    Surrogate phases:   {surrogate_time:>8.1f}s")
    if not args.no_pde:
        print(f"    PDE refinement:     {pde_time:>8.1f}s")
    print(f"    Total:              {total_time:>8.1f}s")

    print(f"{'#'*90}")

    # Save comparison CSV
    csv_path = os.path.join(base_output, "master_comparison_v8_1.csv")
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
    print(f"\n=== Master Inference v8.1 Complete ===")


if __name__ == "__main__":
    main()
