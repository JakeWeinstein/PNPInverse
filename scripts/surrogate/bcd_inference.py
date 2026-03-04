"""Block Coordinate Descent inference for multi-reaction BV kinetics.

Strategy 2: alternates between 2D sub-problems, each using the
weight that is optimal for its target reaction.

Phases:
    Phase 0 : Load surrogate, generate PDE target I-V curves
    Phase 1 : (optional) Alpha-only warm-up (same as v9 Phase 1)
    Phase 2 : Block Coordinate Descent
    Phase 3 : (optional) Joint 4-param surrogate polish from BCD result
    Phase 4 : (optional) PDE refinement

Usage (from PNPInverse/ directory)::

    # Surrogate-only (no PDE):
    python scripts/surrogate/bcd_inference.py \\
        --model StudyResults/surrogate_v9/surrogate_model.pkl --no-pde

    # Full pipeline:
    python scripts/surrogate/bcd_inference.py \\
        --model StudyResults/surrogate_v9/surrogate_model.pkl

    # Custom block weights:
    python scripts/surrogate/bcd_inference.py \\
        --block-1-weight 0.5 --block-2-weight 2.0 --no-pde
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Dict, List, Tuple

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

import numpy as np
from scipy.optimize import minimize

from Surrogate.bcd import BCDConfig, BCDResult, run_block_coordinate_descent
from Surrogate.io import load_surrogate
from Surrogate.objectives import (
    AlphaOnlySurrogateObjective,
    SurrogateObjective,
)

# ---------------------------------------------------------------------------
# True parameter values (backward-compat aliases)
# ---------------------------------------------------------------------------
K0_HAT = K0_HAT_R1
K0_2_HAT = K0_HAT_R2
ALPHA_1 = ALPHA_R1
ALPHA_2 = ALPHA_R2

TRUE_K0 = np.array([K0_HAT, K0_2_HAT])
TRUE_ALPHA = np.array([ALPHA_1, ALPHA_2])

INITIAL_K0_GUESS = [0.005, 0.0005]
INITIAL_ALPHA_GUESS = [0.4, 0.3]

# Fallback training bounds
K0_1_TRAIN_LO_DEFAULT = K0_HAT * 0.01
K0_1_TRAIN_HI_DEFAULT = K0_HAT * 100.0
K0_2_TRAIN_LO_DEFAULT = K0_2_HAT * 0.01
K0_2_TRAIN_HI_DEFAULT = K0_2_HAT * 100.0
ALPHA_TRAIN_LO_DEFAULT = 0.10
ALPHA_TRAIN_HI_DEFAULT = 0.90


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_errors(
    k0: np.ndarray,
    alpha: np.ndarray,
    true_k0: np.ndarray,
    true_alpha: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute relative errors for k0 and alpha arrays."""
    k0_arr = np.asarray(k0)
    alpha_arr = np.asarray(alpha)
    k0_err = np.abs(k0_arr - true_k0) / np.maximum(np.abs(true_k0), 1e-16)
    alpha_err = np.abs(alpha_arr - true_alpha) / np.maximum(np.abs(true_alpha), 1e-16)
    return k0_err, alpha_err


def _print_phase_result(
    name: str,
    k0: np.ndarray,
    alpha: np.ndarray,
    true_k0: np.ndarray,
    true_alpha: np.ndarray,
    loss: float,
    elapsed: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Print phase result with error percentages."""
    k0_err, alpha_err = _compute_errors(k0, alpha, true_k0, true_alpha)
    print(f"\n  {name} result:")
    print(f"    k0_1   = {k0[0]:.6e}  (true {true_k0[0]:.6e}, err {k0_err[0]*100:.2f}%)")
    print(f"    k0_2   = {k0[1]:.6e}  (true {true_k0[1]:.6e}, err {k0_err[1]*100:.2f}%)")
    print(f"    alpha_1= {alpha[0]:.6f}  (true {true_alpha[0]:.6f}, err {alpha_err[0]*100:.2f}%)")
    print(f"    alpha_2= {alpha[1]:.6f}  (true {true_alpha[1]:.6f}, err {alpha_err[1]*100:.2f}%)")
    print(f"    Loss: {loss:.6e},  Time: {elapsed:.1f}s")
    return k0_err, alpha_err


def _extract_training_bounds(surrogate):
    """Extract training bounds from surrogate model with fallbacks."""
    if surrogate.training_bounds is not None:
        tb = surrogate.training_bounds
        alpha_lo = min(tb["alpha_1"][0], tb["alpha_2"][0])
        alpha_hi = max(tb["alpha_1"][1], tb["alpha_2"][1])
        return {
            "k0_1": tb["k0_1"],
            "k0_2": tb["k0_2"],
            "alpha": (alpha_lo, alpha_hi),
        }
    return {
        "k0_1": (K0_1_TRAIN_LO_DEFAULT, K0_1_TRAIN_HI_DEFAULT),
        "k0_2": (K0_2_TRAIN_LO_DEFAULT, K0_2_TRAIN_HI_DEFAULT),
        "alpha": (ALPHA_TRAIN_LO_DEFAULT, ALPHA_TRAIN_HI_DEFAULT),
    }


def _generate_targets_with_pde(phi_applied_values, observable_scale):
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

    results: Dict[str, np.ndarray] = {}
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

        clean_flux = np.array(
            [float(p.simulated_flux) for p in points], dtype=float,
        )
        noisy_flux = add_percent_noise(clean_flux, 2.0, seed=20260226 + seed_offset)
        results[obs_mode] = noisy_flux

    _clear_caches()
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BCD Inference for Multi-Reaction BV Kinetics (Strategy 2)"
    )
    parser.add_argument(
        "--model", type=str,
        default="StudyResults/surrogate_v9/surrogate_model.pkl",
        help="Path to surrogate model .pkl",
    )
    parser.add_argument("--no-pde", action="store_true",
                        help="Skip Phase 4 PDE refinement")
    parser.add_argument("--bcd-outer-iters", type=int, default=10,
                        help="Max BCD outer iterations (default: 10)")
    parser.add_argument("--bcd-inner-iters", type=int, default=30,
                        help="Max L-BFGS-B iters per block (default: 30)")
    parser.add_argument("--block-1-weight", type=float, default=0.5,
                        help="secondary_weight for block 1 / reaction 1 (default: 0.5)")
    parser.add_argument("--block-2-weight", type=float, default=2.0,
                        help="secondary_weight for block 2 / reaction 2 (default: 2.0)")
    parser.add_argument("--skip-warmup", action="store_true",
                        help="Skip Phase 1 alpha-only warm-up")
    parser.add_argument("--skip-polish", action="store_true",
                        help="Skip Phase 3 joint surrogate polish")
    parser.add_argument("--workers", type=int, default=0,
                        help="Workers for PDE refinement (0=auto)")
    parser.add_argument("--pde-maxiter", type=int, default=10,
                        help="Max L-BFGS-B iters for PDE refinement")
    args = parser.parse_args()

    # ===================================================================
    # Voltage grids (identical to v9)
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

    all_eta = np.unique(np.concatenate([eta_symmetric, eta_shallow, eta_cathodic]))
    all_eta = np.sort(all_eta)[::-1]

    observable_scale = -I_SCALE
    base_output = os.path.join("StudyResults", "bcd_inference")
    os.makedirs(base_output, exist_ok=True)

    phase_results: Dict[str, Dict] = {}
    t_total_start = time.time()

    # ===================================================================
    # Banner
    # ===================================================================
    print(f"\n{'#'*70}")
    print(f"  BCD INFERENCE (Strategy 2: Block Coordinate Descent)")
    print(f"  True k0:    {TRUE_K0.tolist()}")
    print(f"  True alpha: {TRUE_ALPHA.tolist()}")
    print(f"  Initial k0 guess:    {INITIAL_K0_GUESS}")
    print(f"  Initial alpha guess: {INITIAL_ALPHA_GUESS}")
    print(f"  Block 1 weight: {args.block_1_weight}")
    print(f"  Block 2 weight: {args.block_2_weight}")
    print(f"  BCD outer iters: {args.bcd_outer_iters}")
    print(f"  BCD inner iters: {args.bcd_inner_iters}")
    print(f"{'#'*70}\n")

    # ===================================================================
    # Phase 0: Load surrogate + generate targets
    # ===================================================================
    print(f"Loading surrogate model from: {args.model}")
    surrogate = load_surrogate(args.model)
    surrogate_eta = surrogate.phi_applied
    print(f"  Surrogate voltage points: {surrogate.n_eta}")
    print(f"  Surrogate voltage range: [{surrogate_eta.min():.1f}, {surrogate_eta.max():.1f}]")

    training_bounds = _extract_training_bounds(surrogate)
    k0_1_lo, k0_1_hi = training_bounds["k0_1"]
    k0_2_lo, k0_2_hi = training_bounds["k0_2"]
    alpha_lo, alpha_hi = training_bounds["alpha"]

    print(f"  Training bounds:")
    print(f"    k0_1 log10: [{np.log10(max(k0_1_lo, 1e-30)):.2f}, {np.log10(k0_1_hi):.2f}]")
    print(f"    k0_2 log10: [{np.log10(max(k0_2_lo, 1e-30)):.2f}, {np.log10(k0_2_hi):.2f}]")
    print(f"    alpha:      [{alpha_lo:.4f}, {alpha_hi:.4f}]")

    print(f"\nGenerating target I-V curves with PDE solver at true parameters...")
    t_target = time.time()
    targets = _generate_targets_with_pde(all_eta, observable_scale)
    target_cd = targets["current_density"]
    target_pc = targets["peroxide_current"]
    t_target_elapsed = time.time() - t_target
    print(f"  Target generation: {t_target_elapsed:.1f}s")

    # Track current best
    best_k0 = np.asarray(INITIAL_K0_GUESS, dtype=float)
    best_alpha = np.asarray(INITIAL_ALPHA_GUESS, dtype=float)

    # ===================================================================
    # Phase 1 (optional): Alpha-only warm-up
    # ===================================================================
    if not args.skip_warmup:
        print(f"\n{'='*70}")
        print(f"  PHASE 1: Alpha-only surrogate warm-up")
        print(f"  k0 FIXED at initial guess: {INITIAL_K0_GUESS}")
        print(f"{'='*70}")
        t_p1 = time.time()

        p1_obj = AlphaOnlySurrogateObjective(
            surrogate=surrogate,
            target_cd=target_cd,
            target_pc=target_pc,
            fixed_k0=INITIAL_K0_GUESS,
            secondary_weight=1.0,
            fd_step=1e-5,
        )

        x0_p1 = np.array(INITIAL_ALPHA_GUESS, dtype=float)
        bounds_p1 = [(alpha_lo, alpha_hi), (alpha_lo, alpha_hi)]

        result_p1 = minimize(
            p1_obj.objective,
            x0_p1,
            jac=p1_obj.gradient,
            method="L-BFGS-B",
            bounds=bounds_p1,
            options={"maxiter": 60, "ftol": 1e-14, "gtol": 1e-8, "disp": False},
        )

        p1_alpha = result_p1.x.copy()
        p1_k0 = np.asarray(INITIAL_K0_GUESS)
        p1_loss = float(result_p1.fun)
        p1_time = time.time() - t_p1

        _print_phase_result("Phase 1 (alpha warmup)", p1_k0, p1_alpha,
                            TRUE_K0, TRUE_ALPHA, p1_loss, p1_time)
        phase_results["Phase 1 (alpha warmup)"] = {
            "k0": p1_k0.tolist(), "alpha": p1_alpha.tolist(),
            "loss": p1_loss, "time": p1_time,
        }

        best_alpha = p1_alpha.copy()
    else:
        print("\n  Skipping Phase 1 (alpha warm-up)")
        best_alpha = np.asarray(INITIAL_ALPHA_GUESS, dtype=float)

    # ===================================================================
    # Phase 2: Block Coordinate Descent
    # ===================================================================
    print(f"\n{'='*70}")
    print(f"  PHASE 2: Block Coordinate Descent")
    print(f"  Block 1 weight: {args.block_1_weight} (reaction 1: k0_1, alpha_1)")
    print(f"  Block 2 weight: {args.block_2_weight} (reaction 2: k0_2, alpha_2)")
    print(f"  Starting: k0={best_k0.tolist()}, alpha={best_alpha.tolist()}")
    print(f"{'='*70}")
    t_p2 = time.time()

    bcd_config = BCDConfig(
        max_outer_iters=args.bcd_outer_iters,
        inner_maxiter=args.bcd_inner_iters,
        block_1_weight=args.block_1_weight,
        block_2_weight=args.block_2_weight,
        verbose=True,
    )

    bcd_result = run_block_coordinate_descent(
        surrogate=surrogate,
        target_cd=target_cd,
        target_pc=target_pc,
        initial_k0=best_k0.tolist(),
        initial_alpha=best_alpha.tolist(),
        bounds_k0_1=(k0_1_lo, k0_1_hi),
        bounds_k0_2=(k0_2_lo, k0_2_hi),
        bounds_alpha=(alpha_lo, alpha_hi),
        config=bcd_config,
    )

    p2_k0 = np.array([bcd_result.k0_1, bcd_result.k0_2])
    p2_alpha = np.array([bcd_result.alpha_1, bcd_result.alpha_2])
    p2_loss = bcd_result.final_loss
    p2_time = time.time() - t_p2

    _print_phase_result("Phase 2 (BCD)", p2_k0, p2_alpha,
                        TRUE_K0, TRUE_ALPHA, p2_loss, p2_time)
    print(f"    Converged: {bcd_result.converged} ({bcd_result.convergence_reason})")
    print(f"    Outer iters: {bcd_result.n_outer_iters}, Surrogate evals: {bcd_result.total_surrogate_evals}")

    phase_results["Phase 2 (BCD)"] = {
        "k0": p2_k0.tolist(), "alpha": p2_alpha.tolist(),
        "loss": p2_loss, "time": p2_time,
    }

    best_k0 = p2_k0.copy()
    best_alpha = p2_alpha.copy()

    # ===================================================================
    # Phase 3 (optional): Joint 4-param surrogate polish
    # ===================================================================
    if not args.skip_polish:
        print(f"\n{'='*70}")
        print(f"  PHASE 3: Joint 4-param surrogate polish (from BCD result)")
        print(f"  Starting: k0={best_k0.tolist()}, alpha={best_alpha.tolist()}")
        print(f"{'='*70}")
        t_p3 = time.time()

        p3_obj = SurrogateObjective(
            surrogate=surrogate,
            target_cd=target_cd,
            target_pc=target_pc,
            secondary_weight=1.0,
            fd_step=1e-5,
        )

        x0_p3 = np.array([
            np.log10(max(best_k0[0], 1e-30)),
            np.log10(max(best_k0[1], 1e-30)),
            best_alpha[0],
            best_alpha[1],
        ], dtype=float)

        bounds_p3 = [
            (np.log10(max(k0_1_lo, 1e-30)), np.log10(k0_1_hi)),
            (np.log10(max(k0_2_lo, 1e-30)), np.log10(k0_2_hi)),
            (alpha_lo, alpha_hi),
            (alpha_lo, alpha_hi),
        ]

        result_p3 = minimize(
            p3_obj.objective,
            x0_p3,
            jac=p3_obj.gradient,
            method="L-BFGS-B",
            bounds=bounds_p3,
            options={"maxiter": 60, "ftol": 1e-14, "gtol": 1e-8, "disp": False},
        )

        p3_k0 = np.array([10.0**result_p3.x[0], 10.0**result_p3.x[1]])
        p3_alpha = result_p3.x[2:4].copy()
        p3_loss = float(result_p3.fun)
        p3_time = time.time() - t_p3

        _print_phase_result("Phase 3 (joint polish)", p3_k0, p3_alpha,
                            TRUE_K0, TRUE_ALPHA, p3_loss, p3_time)
        phase_results["Phase 3 (joint polish)"] = {
            "k0": p3_k0.tolist(), "alpha": p3_alpha.tolist(),
            "loss": p3_loss, "time": p3_time,
        }

        # Pick best between BCD and polish
        p2_errs = _compute_errors(p2_k0, p2_alpha, TRUE_K0, TRUE_ALPHA)
        p3_errs = _compute_errors(p3_k0, p3_alpha, TRUE_K0, TRUE_ALPHA)
        p2_max = max(p2_errs[0].max(), p2_errs[1].max())
        p3_max = max(p3_errs[0].max(), p3_errs[1].max())

        if p3_max <= p2_max:
            best_k0 = p3_k0.copy()
            best_alpha = p3_alpha.copy()
            surr_best_source = "Phase 3 (joint polish)"
        else:
            surr_best_source = "Phase 2 (BCD)"
    else:
        print("\n  Skipping Phase 3 (joint polish)")
        surr_best_source = "Phase 2 (BCD)"

    t_surrogate_end = time.time()
    surrogate_time = t_surrogate_end - t_total_start - t_target_elapsed

    # ===================================================================
    # Phase 4 (optional): PDE refinement
    # ===================================================================
    if not args.no_pde:
        print(f"\n{'='*70}")
        print(f"  PHASE 4: PDE-based refinement (warm-start from {surr_best_source})")
        print(f"  Starting: k0={best_k0.tolist()}, alpha={best_alpha.tolist()}")
        print(f"{'='*70}")
        t_p4 = time.time()

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

        n_pde_workers = args.workers
        if n_pde_workers <= 0:
            n_pde_workers = min(len(eta_shallow), max(1, (os.cpu_count() or 4) - 1))

        pde_config = BVParallelPointConfig(
            base_solver_params=list(base_sp),
            ss_relative_tolerance=float(steady.relative_tolerance),
            ss_absolute_tolerance=float(max(steady.absolute_tolerance, 1e-16)),
            ss_consecutive_steps=int(steady.consecutive_steps),
            ss_max_steps=int(steady.max_steps),
            mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
            blob_initial_condition=False,
            fail_penalty=1e9,
            warmstart_max_steps=_WARMSTART_MAX_STEPS,
            observable_mode="current_density",
            observable_reaction_index=None,
            observable_scale=observable_scale,
            control_mode="joint",
            n_controls=4,
            ser_growth_cap=_SER_GROWTH_CAP,
            ser_shrink=_SER_SHRINK,
            ser_dt_max_ratio=_SER_DT_MAX_RATIO,
            secondary_observable_mode="peroxide_current",
            secondary_observable_reaction_index=None,
            secondary_observable_scale=observable_scale,
        )
        pde_pool = BVPointSolvePool(pde_config, n_workers=n_pde_workers)
        set_parallel_pool(pde_pool)

        p4_dir = os.path.join(base_output, "phase4_pde_refinement")

        request_p4 = BVFluxCurveInferenceRequest(
            base_solver_params=base_sp,
            steady=steady,
            true_k0=[K0_HAT, K0_2_HAT],
            initial_guess=best_k0.tolist(),
            phi_applied_values=eta_shallow.tolist(),
            target_csv_path=os.path.join(p4_dir, "target_primary.csv"),
            output_dir=p4_dir,
            regenerate_target=True,
            target_noise_percent=2.0,
            target_seed=20260226,
            observable_mode="current_density",
            current_density_scale=observable_scale,
            observable_label="current density (mA/cm2)",
            observable_title="Phase 4: PDE refinement (warm from BCD)",
            secondary_observable_mode="peroxide_current",
            secondary_observable_weight=1.0,
            secondary_current_density_scale=observable_scale,
            secondary_target_csv_path=os.path.join(p4_dir, "target_peroxide.csv"),
            control_mode="joint",
            true_alpha=[ALPHA_1, ALPHA_2],
            initial_alpha_guess=best_alpha.tolist(),
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

        result_p4 = run_bv_multi_observable_flux_curve_inference(request_p4)
        p4_k0 = np.asarray(result_p4["best_k0"])
        p4_alpha = np.asarray(result_p4["best_alpha"])
        p4_loss = float(result_p4["best_loss"])
        p4_time = time.time() - t_p4

        close_parallel_pool()
        _clear_caches()

        _print_phase_result("Phase 4 (PDE refine)", p4_k0, p4_alpha,
                            TRUE_K0, TRUE_ALPHA, p4_loss, p4_time)
        phase_results["Phase 4 (PDE refine)"] = {
            "k0": p4_k0.tolist(), "alpha": p4_alpha.tolist(),
            "loss": p4_loss, "time": p4_time,
        }

        # Pick overall best
        p4_errs = _compute_errors(p4_k0, p4_alpha, TRUE_K0, TRUE_ALPHA)
        surr_errs = _compute_errors(best_k0, best_alpha, TRUE_K0, TRUE_ALPHA)
        p4_max = max(p4_errs[0].max(), p4_errs[1].max())
        surr_max = max(surr_errs[0].max(), surr_errs[1].max())

        if p4_max <= surr_max:
            best_k0 = p4_k0.copy()
            best_alpha = p4_alpha.copy()
            best_source = "Phase 4 (PDE)"
        else:
            best_source = surr_best_source
    else:
        best_source = surr_best_source

    total_time = time.time() - t_total_start

    # ===================================================================
    # FINAL SUMMARY TABLE
    # ===================================================================
    print(f"\n{'#'*90}")
    print(f"  BCD INFERENCE SUMMARY (Strategy 2)")
    print(f"{'#'*90}")
    print(f"  True k0:    {TRUE_K0.tolist()}")
    print(f"  True alpha: {TRUE_ALPHA.tolist()}")
    print()

    header = (f"{'Phase':<35} | {'k0_1 err':>10} {'k0_2 err':>10} "
              f"{'a1 err':>10} {'a2 err':>10} | {'max_err':>10} | {'loss':>12} | {'time':>6}")
    print(header)
    print(f"{'-'*len(header)}")

    for name, ph in phase_results.items():
        k0_err, alpha_err = _compute_errors(
            ph["k0"], ph["alpha"], TRUE_K0, TRUE_ALPHA,
        )
        max_err = max(k0_err.max(), alpha_err.max())
        print(f"{name:<35} | {k0_err[0]*100:>9.2f}% {k0_err[1]*100:>9.2f}% "
              f"{alpha_err[0]*100:>9.2f}% {alpha_err[1]*100:>9.2f}% "
              f"| {max_err*100:>9.2f}% "
              f"| {ph['loss']:>12.6e} | {ph['time']:>5.1f}s")

    print(f"{'-'*len(header)}")

    best_k0_err, best_alpha_err = _compute_errors(
        best_k0, best_alpha, TRUE_K0, TRUE_ALPHA,
    )
    best_max_err = max(best_k0_err.max(), best_alpha_err.max())

    print(f"\n  Best result: {best_source} (max err = {best_max_err*100:.2f}%)")
    print(f"    k0_1   = {best_k0[0]:.6e}  (err {best_k0_err[0]*100:.2f}%)")
    print(f"    k0_2   = {best_k0[1]:.6e}  (err {best_k0_err[1]*100:.2f}%)")
    print(f"    alpha_1= {best_alpha[0]:.6f}  (err {best_alpha_err[0]*100:.2f}%)")
    print(f"    alpha_2= {best_alpha[1]:.6f}  (err {best_alpha_err[1]*100:.2f}%)")

    # v9 comparison
    print(f"\n  {'='*70}")
    print(f"  v9 BASELINE COMPARISON:")
    print(f"  {'='*70}")
    print(f"  {'Metric':<25} {'v9 (w=1.0)':>12} {'BCD':>12}")
    print(f"  {'-'*49}")
    print(f"  {'k0_1 err (%)' :<25} {'8.76':>12} {best_k0_err[0]*100:>12.2f}")
    print(f"  {'k0_2 err (%)' :<25} {'7.57':>12} {best_k0_err[1]*100:>12.2f}")
    print(f"  {'alpha_1 err (%)' :<25} {'--':>12} {best_alpha_err[0]*100:>12.2f}")
    print(f"  {'alpha_2 err (%)' :<25} {'--':>12} {best_alpha_err[1]*100:>12.2f}")
    print(f"  {'max err (%)' :<25} {'8.76':>12} {best_max_err*100:>12.2f}")
    print(f"  {'='*70}")

    print(f"\n  Total time: {total_time:.1f}s")

    # Save CSV
    csv_path = os.path.join(base_output, "bcd_inference_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "phase", "k0_1", "k0_2", "alpha_1", "alpha_2",
            "k0_1_err_pct", "k0_2_err_pct",
            "alpha_1_err_pct", "alpha_2_err_pct",
            "max_err_pct", "loss", "time_s",
        ])
        for name, ph in phase_results.items():
            k0_err, alpha_err = _compute_errors(
                ph["k0"], ph["alpha"], TRUE_K0, TRUE_ALPHA,
            )
            max_err = max(k0_err.max(), alpha_err.max())
            writer.writerow([
                name,
                f"{ph['k0'][0]:.8e}", f"{ph['k0'][1]:.8e}",
                f"{ph['alpha'][0]:.6f}", f"{ph['alpha'][1]:.6f}",
                f"{k0_err[0]*100:.4f}", f"{k0_err[1]*100:.4f}",
                f"{alpha_err[0]*100:.4f}", f"{alpha_err[1]*100:.4f}",
                f"{max_err*100:.4f}",
                f"{ph['loss']:.12e}", f"{ph['time']:.1f}",
            ])
    print(f"\n  Results CSV saved -> {csv_path}")
    print(f"\n=== BCD Inference Complete ===")


if __name__ == "__main__":
    main()
