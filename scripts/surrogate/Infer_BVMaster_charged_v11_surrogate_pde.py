"""Master inference protocol v11 for BV kinetics -- SURROGATE WARM-START + PDE REFINEMENT.

Four-phase pipeline that combines v9's fast surrogate warm-start with v7's
proven PDE refinement protocol:

    Phase 1: Surrogate alpha-only   (~0.1s) -- replaces v7's PDE Phase 1 (~160s)
    Phase 2: Surrogate joint 4-param (~0.1s) -- replaces v7's PDE Phase 2 (~170s)
    Phase 3: PDE joint on SHALLOW cathodic (v7 P2 settings, ~120s)
    Phase 4: PDE joint on FULL CATHODIC    (v7 P3 settings, ~120s)

The surrogate provides a warm start for the PDE phases, eliminating the
expensive cold-start PDE phases while preserving v7's full cathodic grid
and iteration budget for accurate refinement.

Usage (from PNPInverse/ directory)::

    python scripts/surrogate/Infer_BVMaster_charged_v11_surrogate_pde.py

    # Surrogate-only (skip PDE phases 3-4):
    python scripts/surrogate/Infer_BVMaster_charged_v11_surrogate_pde.py --no-pde

    # Control PDE workers and iterations:
    python scripts/surrogate/Infer_BVMaster_charged_v11_surrogate_pde.py \\
        --workers 4 --pde-p3-maxiter 30 --pde-p4-maxiter 25

    # Noise-free diagnostic:
    python scripts/surrogate/Infer_BVMaster_charged_v11_surrogate_pde.py \\
        --noise-percent 0.0
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
    K0_HAT_R1, K0_HAT_R2, I_SCALE,
    ALPHA_R1, ALPHA_R2,
    FOUR_SPECIES_CHARGED,
    SNES_OPTS_CHARGED,
    make_bv_solver_params,
    make_recovery_config,
    print_redimensionalized_results,
)
setup_firedrake_env()

# Backward-compat aliases
K0_HAT = K0_HAT_R1
K0_2_HAT = K0_HAT_R2
ALPHA_1 = ALPHA_R1
ALPHA_2 = ALPHA_R2

import numpy as np
from scipy.optimize import minimize

from Surrogate.io import load_surrogate
from Surrogate.objectives import AlphaOnlySurrogateObjective

# ---------------------------------------------------------------------------
# Fallback training-data bounds (used only if model lacks training_bounds)
# ---------------------------------------------------------------------------
K0_1_TRAIN_LO_DEFAULT = K0_HAT * 0.01
K0_1_TRAIN_HI_DEFAULT = K0_HAT * 100.0
K0_2_TRAIN_LO_DEFAULT = K0_2_HAT * 0.01
K0_2_TRAIN_HI_DEFAULT = K0_2_HAT * 100.0
ALPHA_TRAIN_LO_DEFAULT = 0.10
ALPHA_TRAIN_HI_DEFAULT = 0.90


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


def _generate_targets_with_pde(phi_applied_values, observable_scale, noise_percent, noise_seed):
    """Generate target I-V curves using the PDE solver at true parameters."""
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
        if noise_percent > 0:
            noisy_flux = add_percent_noise(clean_flux, noise_percent, seed=noise_seed + seed_offset)
        else:
            noisy_flux = clean_flux.copy()
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


def _make_parallel_config(base_sp, steady, *, mesh_Nx, mesh_Ny, mesh_beta,
                          observable_mode, observable_reaction_index,
                          observable_scale, control_mode, n_controls,
                          blob_initial_condition=False, fail_penalty=1e9,
                          secondary_observable_mode=None,
                          secondary_observable_reaction_index=None,
                          secondary_observable_scale=None):
    """Build a BVParallelPointConfig from solver params (mirrors v7)."""
    from FluxCurve.bv_point_solve import (
        _WARMSTART_MAX_STEPS,
        _SER_GROWTH_CAP,
        _SER_SHRINK,
        _SER_DT_MAX_RATIO,
    )
    from FluxCurve.bv_parallel import BVParallelPointConfig

    return BVParallelPointConfig(
        base_solver_params=list(base_sp),
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
        secondary_observable_mode=secondary_observable_mode,
        secondary_observable_reaction_index=secondary_observable_reaction_index,
        secondary_observable_scale=secondary_observable_scale,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BV Master Inference v11 (Surrogate Warm-Start + PDE Refinement)"
    )
    parser.add_argument("--model", type=str,
                        default="StudyResults/surrogate_v9/surrogate_model.pkl",
                        help="Path to surrogate model .pkl")
    parser.add_argument("--no-pde", action="store_true",
                        help="Skip PDE phases 3-4 (surrogate-only, for comparison)")
    parser.add_argument("--workers", type=int, default=0,
                        help="PDE parallel workers (0=auto)")
    parser.add_argument("--pde-p3-maxiter", type=int, default=30,
                        help="Phase 3 max L-BFGS-B iterations (default: 30)")
    parser.add_argument("--pde-p4-maxiter", type=int, default=25,
                        help="Phase 4 max L-BFGS-B iterations (default: 25)")
    parser.add_argument("--secondary-weight", type=float, default=1.0,
                        help="Weight on peroxide current observable (default: 1.0)")
    parser.add_argument("--noise-percent", type=float, default=2.0,
                        help="Target noise level (default: 2.0, use 0.0 for noise-free diagnostic)")
    parser.add_argument("--noise-seed", type=int, default=20260226,
                        help="Noise seed (default: 20260226)")
    args = parser.parse_args()

    # ===================================================================
    # Voltage grids (IDENTICAL to v7/v9)
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
    base_output = os.path.join("StudyResults", "master_inference_v11")
    os.makedirs(base_output, exist_ok=True)

    phase_results = {}
    t_total_start = time.time()

    print(f"\n{'#'*70}")
    print(f"  MASTER INFERENCE PROTOCOL v11 (SURROGATE WARM-START + PDE REFINEMENT)")
    print(f"  Phase 1-2: Surrogate (from v9)")
    print(f"  Phase 3-4: PDE refinement (from v7, warm-started by surrogate)")
    print(f"  True k0:    {true_k0}")
    print(f"  True alpha: {true_alpha}")
    print(f"  Initial k0 guess:    {initial_k0_guess}")
    print(f"  Initial alpha guess: {initial_alpha_guess}")
    print(f"  Secondary weight:    {args.secondary_weight}")
    print(f"  Noise: {args.noise_percent}% (seed={args.noise_seed})")
    print(f"{'#'*70}\n")

    # ===================================================================
    # Load surrogate model
    # ===================================================================
    print(f"Loading surrogate model from: {args.model}")
    surrogate = load_surrogate(args.model)
    surrogate_eta = surrogate.phi_applied
    print(f"  Surrogate voltage points: {surrogate.n_eta}")
    print(f"  Surrogate voltage range: [{surrogate_eta.min():.1f}, {surrogate_eta.max():.1f}]")

    # Dynamic bounds from model (for surrogate phases)
    if surrogate.training_bounds is not None:
        tb = surrogate.training_bounds
        K0_1_TRAIN_LO = tb["k0_1"][0]
        K0_1_TRAIN_HI = tb["k0_1"][1]
        K0_2_TRAIN_LO = tb["k0_2"][0]
        K0_2_TRAIN_HI = tb["k0_2"][1]
        ALPHA_TRAIN_LO = tb["alpha_1"][0]
        ALPHA_TRAIN_HI = tb["alpha_1"][1]
        ALPHA_TRAIN_LO = min(ALPHA_TRAIN_LO, tb["alpha_2"][0])
        ALPHA_TRAIN_HI = max(ALPHA_TRAIN_HI, tb["alpha_2"][1])
        print(f"  Using training bounds FROM MODEL:")
    else:
        K0_1_TRAIN_LO = K0_1_TRAIN_LO_DEFAULT
        K0_1_TRAIN_HI = K0_1_TRAIN_HI_DEFAULT
        K0_2_TRAIN_LO = K0_2_TRAIN_LO_DEFAULT
        K0_2_TRAIN_HI = K0_2_TRAIN_HI_DEFAULT
        ALPHA_TRAIN_LO = ALPHA_TRAIN_LO_DEFAULT
        ALPHA_TRAIN_HI = ALPHA_TRAIN_HI_DEFAULT
        print(f"  WARNING: model lacks training_bounds, using defaults:")

    print(f"    k0_1 log10: [{np.log10(max(K0_1_TRAIN_LO, 1e-30)):.2f}, {np.log10(K0_1_TRAIN_HI):.2f}]")
    print(f"    k0_2 log10: [{np.log10(max(K0_2_TRAIN_LO, 1e-30)):.2f}, {np.log10(K0_2_TRAIN_HI):.2f}]")
    print(f"    alpha:      [{ALPHA_TRAIN_LO:.4f}, {ALPHA_TRAIN_HI:.4f}]")

    # ===================================================================
    # Generate targets using PDE solver (same as v7/v9)
    # ===================================================================
    print(f"\nGenerating target I-V curves with PDE solver at true parameters...")
    t_target = time.time()
    targets = _generate_targets_with_pde(all_eta, observable_scale, args.noise_percent, args.noise_seed)
    target_cd_full = targets["current_density"]
    target_pc_full = targets["peroxide_current"]
    t_target_elapsed = time.time() - t_target
    print(f"  Target generation: {t_target_elapsed:.1f}s")

    target_cd_surr = target_cd_full
    target_pc_surr = target_pc_full

    # ===================================================================
    # PHASE 1: Surrogate alpha-only optimization (from v9)
    # ===================================================================
    print(f"\n{'='*70}")
    print(f"  PHASE 1: Surrogate alpha-only optimization")
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
        secondary_weight=args.secondary_weight,
        fd_step=1e-5,
    )

    x0_p1 = np.array(initial_alpha_guess, dtype=float)
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

    p1_dir = os.path.join(base_output, "phase1_surr_alpha")
    os.makedirs(p1_dir, exist_ok=True)

    _print_phase_result("Phase 1 (alpha, surrogate)", p1_k0, p1_alpha,
                        true_k0_arr, true_alpha_arr, p1_loss, p1_time)
    phase_results["Phase 1 (alpha, surrogate)"] = {
        "k0": p1_k0.tolist(), "alpha": p1_alpha.tolist(),
        "loss": p1_loss, "time": p1_time,
    }

    # ===================================================================
    # PHASE 2: Surrogate joint 4-param on SHALLOW cathodic (from v9)
    # ===================================================================
    print(f"\n{'='*70}")
    print(f"  PHASE 2: Surrogate joint 4-param (shallow-range objective)")
    print(f"  k0 from COLD guess: {initial_k0_guess}")
    print(f"  alpha warm from P1: {p1_alpha.tolist()}")
    print(f"  Bounds clamped to training range")
    print(f"{'='*70}")
    t_p2 = time.time()

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
        """Surrogate objective on a subset of voltage points (from v9)."""
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
        secondary_weight=args.secondary_weight,
        fd_step=1e-5,
        log_space_k0=True,
    )

    x0_p2 = np.array([
        np.log10(initial_k0_guess[0]),
        np.log10(initial_k0_guess[1]),
        p1_alpha[0],
        p1_alpha[1],
    ], dtype=float)

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

    p2_dir = os.path.join(base_output, "phase2_surr_joint")
    os.makedirs(p2_dir, exist_ok=True)

    _print_phase_result("Phase 2 (joint, shallow, surr)", p2_k0, p2_alpha,
                        true_k0_arr, true_alpha_arr, p2_loss, p2_time)
    phase_results["Phase 2 (joint shallow surr)"] = {
        "k0": p2_k0.tolist(), "alpha": p2_alpha.tolist(),
        "loss": p2_loss, "time": p2_time,
    }

    t_surrogate_end = time.time()
    surrogate_time = t_surrogate_end - t_total_start - t_target_elapsed

    # Surrogate best = Phase 2 result (warm start for PDE phases)
    surr_best_k0 = p2_k0.copy()
    surr_best_alpha = p2_alpha.copy()

    # ===================================================================
    # PDE PHASES 3-4 (from v7, warm-started by surrogate)
    # ===================================================================
    if not args.no_pde:
        from Forward.steady_state import SteadyStateConfig
        from FluxCurve import (
            BVFluxCurveInferenceRequest,
            run_bv_multi_observable_flux_curve_inference,
        )
        from FluxCurve.bv_point_solve import (
            _clear_caches,
            set_parallel_pool,
            close_parallel_pool,
        )
        from FluxCurve.bv_parallel import BVPointSolvePool

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

        # Auto-size workers
        n_pde_workers = args.workers
        if n_pde_workers <= 0:
            max_phase_points = max(len(eta_shallow), len(eta_cathodic))
            n_pde_workers = min(max_phase_points, max(1, (os.cpu_count() or 4) - 1))

        n_joint_controls = 4  # k0_1, k0_2, alpha_1, alpha_2

        # ---------------------------------------------------------------
        # Create shared parallel pool for Phases 3 & 4 (v7 optimization)
        # ---------------------------------------------------------------
        shared_config = _make_parallel_config(
            base_sp, steady,
            mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
            observable_mode="current_density",
            observable_reaction_index=None,
            observable_scale=observable_scale,
            control_mode="joint",
            n_controls=n_joint_controls,
            secondary_observable_mode="peroxide_current",
            secondary_observable_reaction_index=None,
            secondary_observable_scale=observable_scale,
        )
        shared_pool = BVPointSolvePool(shared_config, n_workers=n_pde_workers)
        set_parallel_pool(shared_pool)
        print(f"\n  [v11] Shared parallel pool created for P3+P4: {n_pde_workers} workers")

        # ===============================================================
        # PHASE 3: PDE Joint on SHALLOW cathodic (v7 Phase 2 settings)
        #          Warm-started from surrogate Phase 2
        # ===============================================================
        print(f"\n{'='*70}")
        print(f"  PHASE 3: PDE joint on SHALLOW cathodic (v7 P2 settings)")
        print(f"  Warm-start from surrogate: k0={surr_best_k0.tolist()}, alpha={surr_best_alpha.tolist()}")
        print(f"  {len(eta_shallow)}-pt shallow [{eta_shallow.min():.1f}, {eta_shallow.max():.1f}]")
        print(f"  maxiter={args.pde_p3_maxiter}, ftol=1e-8, gtol=5e-6")
        print(f"  Bounds: k0=[1e-8, 100], alpha=[0.05, 0.95] (WIDE, no regularization)")
        print(f"{'='*70}")
        t_p3 = time.time()

        _clear_caches()
        p3_dir = os.path.join(base_output, "phase3_pde_shallow")

        request_p3 = BVFluxCurveInferenceRequest(
            base_solver_params=base_sp,
            steady=steady,
            true_k0=true_k0,
            initial_guess=surr_best_k0.tolist(),  # warm-start k0 from surrogate
            phi_applied_values=eta_shallow.tolist(),
            target_csv_path=os.path.join(p3_dir, "target_primary.csv"),
            output_dir=p3_dir,
            regenerate_target=True,
            target_noise_percent=args.noise_percent,
            target_seed=args.noise_seed,
            observable_mode="current_density",
            current_density_scale=observable_scale,
            observable_label="current density (mA/cm2)",
            observable_title="Phase 3: PDE shallow (warm from surrogate)",
            secondary_observable_mode="peroxide_current",
            secondary_observable_weight=args.secondary_weight,
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
                "maxiter": args.pde_p3_maxiter,
                "ftol": 1e-8,
                "gtol": 5e-6,
                "disp": True,
            },
            max_iters=args.pde_p3_maxiter,
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

        _print_phase_result("Phase 3 (PDE shallow)", p3_k0, p3_alpha,
                            true_k0_arr, true_alpha_arr, p3_loss, p3_time)
        phase_results["Phase 3 (PDE shallow)"] = {
            "k0": p3_k0.tolist(), "alpha": p3_alpha.tolist(),
            "loss": p3_loss, "time": p3_time,
        }

        # Clear point caches between phases but keep pool alive (v7 pattern)
        _clear_caches()

        # ===============================================================
        # PHASE 4: PDE Joint on FULL CATHODIC (v7 Phase 3 settings)
        #          Warm-start from Phase 3
        # ===============================================================
        print(f"\n{'='*70}")
        print(f"  PHASE 4: PDE joint on FULL CATHODIC range (v7 P3 settings)")
        print(f"  Warm-start from Phase 3: k0={p3_k0.tolist()}, alpha={p3_alpha.tolist()}")
        print(f"  {len(eta_cathodic)}-pt full cathodic [{eta_cathodic.min():.1f}, {eta_cathodic.max():.1f}]")
        print(f"  maxiter={args.pde_p4_maxiter}, ftol=1e-8, gtol=5e-6")
        print(f"  Bounds: k0=[1e-8, 100], alpha=[0.05, 0.95] (WIDE, no regularization)")
        print(f"{'='*70}")
        t_p4 = time.time()

        p4_dir = os.path.join(base_output, "phase4_pde_full_cathodic")

        request_p4 = BVFluxCurveInferenceRequest(
            base_solver_params=base_sp,
            steady=steady,
            true_k0=true_k0,
            initial_guess=p3_k0.tolist(),  # warm-start k0 from Phase 3
            phi_applied_values=eta_cathodic.tolist(),
            target_csv_path=os.path.join(p4_dir, "target_primary.csv"),
            output_dir=p4_dir,
            regenerate_target=True,
            target_noise_percent=args.noise_percent,
            target_seed=args.noise_seed,
            observable_mode="current_density",
            current_density_scale=observable_scale,
            observable_label="current density (mA/cm2)",
            observable_title="Phase 4: PDE full cathodic (warm from P3)",
            secondary_observable_mode="peroxide_current",
            secondary_observable_weight=args.secondary_weight,
            secondary_current_density_scale=observable_scale,
            secondary_target_csv_path=os.path.join(p4_dir, "target_peroxide.csv"),
            control_mode="joint",
            true_alpha=true_alpha,
            initial_alpha_guess=p3_alpha.tolist(),  # warm-start alpha from Phase 3
            alpha_lower=0.05, alpha_upper=0.95,
            k0_lower=1e-8, k0_upper=100.0,
            log_space=True,
            mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
            max_eta_gap=3.0,
            optimizer_method="L-BFGS-B",
            optimizer_options={
                "maxiter": args.pde_p4_maxiter,
                "ftol": 1e-8,
                "gtol": 5e-6,
                "disp": True,
            },
            max_iters=args.pde_p4_maxiter,
            live_plot=False,
            forward_recovery=recovery,
            parallel_fast_path=True,
            parallel_workers=n_pde_workers,
        )

        result_p4 = run_bv_multi_observable_flux_curve_inference(request_p4)
        p4_k0 = np.asarray(result_p4["best_k0"])
        p4_alpha = np.asarray(result_p4["best_alpha"])
        p4_loss = float(result_p4["best_loss"])
        p4_time = time.time() - t_p4

        # Close shared pool (both P3 and P4 are done)
        close_parallel_pool()
        _clear_caches()
        print(f"  [v11] Shared parallel pool closed after P3+P4")

        _print_phase_result("Phase 4 (PDE full cathodic)", p4_k0, p4_alpha,
                            true_k0_arr, true_alpha_arr, p4_loss, p4_time)
        phase_results["Phase 4 (PDE full cathodic)"] = {
            "k0": p4_k0.tolist(), "alpha": p4_alpha.tolist(),
            "loss": p4_loss, "time": p4_time,
        }

        # ===============================================================
        # Best-of selection: Phase 3 vs Phase 4 (regression guard from v7)
        # ===============================================================
        p3_k0_err, p3_alpha_err = _compute_errors(p3_k0, p3_alpha, true_k0_arr, true_alpha_arr)
        p4_k0_err, p4_alpha_err = _compute_errors(p4_k0, p4_alpha, true_k0_arr, true_alpha_arr)

        p3_max_err = max(p3_k0_err.max(), p3_alpha_err.max())
        p4_max_err = max(p4_k0_err.max(), p4_alpha_err.max())

        if p4_max_err <= p3_max_err:
            best_k0, best_alpha = p4_k0.copy(), p4_alpha.copy()
            best_source = "Phase 4 (PDE full cathodic)"
        else:
            best_k0, best_alpha = p3_k0.copy(), p3_alpha.copy()
            best_source = "Phase 3 (PDE shallow)"

        best_max_err = min(p3_max_err, p4_max_err)

        print(f"\n{'='*70}")
        print(f"  P3 vs P4: best is {best_source} (max err = {best_max_err*100:.2f}%)")
        print(f"{'='*70}")
    else:
        best_k0, best_alpha = surr_best_k0.copy(), surr_best_alpha.copy()
        best_source = "Phase 2 (surrogate)"

    total_time = time.time() - t_total_start
    pde_time = total_time - t_target_elapsed - surrogate_time

    # ===================================================================
    # FINAL SUMMARY
    # ===================================================================
    print(f"\n{'#'*90}")
    print(f"  MASTER INFERENCE v11 SUMMARY (SURROGATE WARM-START + PDE REFINEMENT)")
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
          f"surrogate phases: {surrogate_time:.1f}s, PDE phases: {pde_time:.1f}s)")

    best_k0_err, best_alpha_err = _compute_errors(best_k0, best_alpha, true_k0_arr, true_alpha_arr)
    best_max_err = max(best_k0_err.max(), best_alpha_err.max())

    print(f"\n  Best result: {best_source} (max err = {best_max_err*100:.2f}%)")
    print(f"    k0_1   = {best_k0[0]:.6e}  (err {best_k0_err[0]*100:.2f}%)")
    print(f"    k0_2   = {best_k0[1]:.6e}  (err {best_k0_err[1]*100:.2f}%)")
    print(f"    alpha_1= {best_alpha[0]:.6f}  (err {best_alpha_err[0]*100:.2f}%)")
    print(f"    alpha_2= {best_alpha[1]:.6f}  (err {best_alpha_err[1]*100:.2f}%)")

    print_redimensionalized_results(
        best_k0, true_k0_arr,
        best_alpha=best_alpha, true_alpha=true_alpha_arr,
    )

    # v7 / v9 comparison
    print(f"\n  {'='*70}")
    print(f"  v7 / v9 / v11 COMPARISON:")
    print(f"  {'='*70}")
    print(f"  {'Metric':<25} {'v7':>12} {'v9':>12} {'v11':>12}")
    print(f"  {'-'*61}")
    print(f"  {'Total time (s)':<25} {'415':>12} {'--':>12} {total_time:>12.1f}")
    print(f"  {'k0_1 err (%)':<25} {'10.9':>12} {'8.76':>12} {best_k0_err[0]*100:>12.1f}")
    print(f"  {'k0_2 err (%)':<25} {'2.6':>12} {'--':>12} {best_k0_err[1]*100:>12.1f}")
    print(f"  {'alpha_1 err (%)':<25} {'5.6':>12} {'--':>12} {best_alpha_err[0]*100:>12.1f}")
    print(f"  {'alpha_2 err (%)':<25} {'8.8':>12} {'--':>12} {best_alpha_err[1]*100:>12.1f}")
    print(f"  {'='*70}")

    print(f"\n  Timing breakdown:")
    print(f"    Target generation:  {t_target_elapsed:>8.1f}s")
    print(f"    Surrogate phases:   {surrogate_time:>8.1f}s")
    if not args.no_pde:
        print(f"    PDE phases (3+4):   {pde_time:>8.1f}s")
    print(f"    Total:              {total_time:>8.1f}s")

    print(f"{'#'*90}")

    # Save comparison CSV
    csv_path = os.path.join(base_output, "master_comparison_v11.csv")
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
    print(f"\n=== Master Inference v11 Complete ===")


if __name__ == "__main__":
    main()
