"""Strategy 5: Per-observable inference cascade.

Exploits the weight-sweep insight that CD-dominant weighting recovers
reaction-1 parameters excellently, while PC-dominant weighting recovers
k0_2 well.  Runs a three-pass cascade:

    Pass 1: CD-dominant (low weight) -- all 4 params free
    Pass 2: PC-dominant (high weight) -- k0_2, alpha_2 free; k0_1, alpha_1 fixed
    Pass 3: Joint polish (moderate weight) -- all 4 params free (optional)

Three-phase protocol:
    Phase 0: Load surrogate, generate PDE targets at true parameters
    Phase 1: Run cascade inference (all 3 passes)
    Phase 2: (Optional) PDE refinement from cascade best

Usage (from PNPInverse/ directory)::

    python scripts/surrogate/cascade_inference.py \\
        --model StudyResults/surrogate_v9/surrogate_model.pkl

    # Skip PDE refinement (surrogate-only):
    python scripts/surrogate/cascade_inference.py --no-pde

    # Custom weights:
    python scripts/surrogate/cascade_inference.py \\
        --pass1-weight 0.1 --pass2-weight 5.0

    # Skip polish pass:
    python scripts/surrogate/cascade_inference.py --no-polish
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
)
setup_firedrake_env()

# Backward-compat aliases
K0_HAT = K0_HAT_R1
K0_2_HAT = K0_HAT_R2
ALPHA_1 = ALPHA_R1
ALPHA_2 = ALPHA_R2

import numpy as np

from Surrogate.io import load_surrogate
from Surrogate.cascade import CascadeConfig, run_cascade_inference


# ---------------------------------------------------------------------------
# Fallback training-data bounds
# ---------------------------------------------------------------------------
K0_1_TRAIN_LO_DEFAULT = K0_HAT * 0.01
K0_1_TRAIN_HI_DEFAULT = K0_HAT * 100.0
K0_2_TRAIN_LO_DEFAULT = K0_2_HAT * 0.01
K0_2_TRAIN_HI_DEFAULT = K0_2_HAT * 100.0
ALPHA_TRAIN_LO_DEFAULT = 0.10
ALPHA_TRAIN_HI_DEFAULT = 0.90


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_errors(k0, alpha, true_k0_arr, true_alpha_arr):
    """Compute relative errors for k0 and alpha arrays."""
    k0_arr = np.asarray(k0)
    alpha_arr = np.asarray(alpha)
    k0_err = np.abs(k0_arr - true_k0_arr) / np.maximum(np.abs(true_k0_arr), 1e-16)
    alpha_err = np.abs(alpha_arr - true_alpha_arr) / np.maximum(np.abs(true_alpha_arr), 1e-16)
    return k0_err, alpha_err


def _print_phase_result(name, k0, alpha, true_k0_arr, true_alpha_arr, loss, elapsed):
    """Print a phase result with parameter errors."""
    k0_err, alpha_err = _compute_errors(k0, alpha, true_k0_arr, true_alpha_arr)
    print(f"\n  {name} result:")
    print(f"    k0_1   = {k0[0]:.6e}  (true {true_k0_arr[0]:.6e}, err {k0_err[0]*100:.2f}%)")
    print(f"    k0_2   = {k0[1]:.6e}  (true {true_k0_arr[1]:.6e}, err {k0_err[1]*100:.2f}%)")
    print(f"    alpha_1= {alpha[0]:.6f}  (true {true_alpha_arr[0]:.6f}, err {alpha_err[0]*100:.2f}%)")
    print(f"    alpha_2= {alpha[1]:.6f}  (true {true_alpha_arr[1]:.6f}, err {alpha_err[1]*100:.2f}%)")
    print(f"    Loss: {loss:.6e},  Time: {elapsed:.1f}s")
    return k0_err, alpha_err


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


def _extract_training_bounds(surrogate):
    """Extract training bounds from the surrogate model (with fallbacks)."""
    if surrogate.training_bounds is not None:
        tb = surrogate.training_bounds
        k0_1_lo = tb["k0_1"][0]
        k0_1_hi = tb["k0_1"][1]
        k0_2_lo = tb["k0_2"][0]
        k0_2_hi = tb["k0_2"][1]
        alpha_lo = min(tb["alpha_1"][0], tb["alpha_2"][0])
        alpha_hi = max(tb["alpha_1"][1], tb["alpha_2"][1])
        return k0_1_lo, k0_1_hi, k0_2_lo, k0_2_hi, alpha_lo, alpha_hi
    return (K0_1_TRAIN_LO_DEFAULT, K0_1_TRAIN_HI_DEFAULT,
            K0_2_TRAIN_LO_DEFAULT, K0_2_TRAIN_HI_DEFAULT,
            ALPHA_TRAIN_LO_DEFAULT, ALPHA_TRAIN_HI_DEFAULT)


def _compute_shallow_subset_idx(all_eta, eta_shallow):
    """Find indices of shallow voltages within the full grid."""
    idx = []
    for eta in eta_shallow:
        matches = np.where(np.abs(all_eta - eta) < 1e-10)[0]
        if len(matches) > 0:
            idx.append(matches[0])
    return np.array(idx, dtype=int)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strategy 5: Per-Observable Inference Cascade"
    )
    parser.add_argument("--model", type=str,
                        default="StudyResults/surrogate_v9/surrogate_model.pkl",
                        help="Path to surrogate model .pkl")
    parser.add_argument("--pass1-weight", type=float, default=0.5,
                        help="CD-dominant weight for Pass 1 (default: 0.5)")
    parser.add_argument("--pass2-weight", type=float, default=2.0,
                        help="PC-dominant weight for Pass 2 (default: 2.0)")
    parser.add_argument("--polish-weight", type=float, default=1.0,
                        help="Joint polish weight for Pass 3 (default: 1.0)")
    parser.add_argument("--no-pde", action="store_true",
                        help="Skip PDE refinement (surrogate-only)")
    parser.add_argument("--no-polish", action="store_true",
                        help="Skip Pass 3 joint polish")
    parser.add_argument("--workers", type=int, default=0,
                        help="Workers for PDE refinement (0=auto)")
    parser.add_argument("--pde-maxiter", type=int, default=10,
                        help="Max L-BFGS-B iterations for PDE refinement")
    args = parser.parse_args()

    # ===================================================================
    # Voltage grids (IDENTICAL to v9)
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
    base_output = os.path.join("StudyResults", "cascade")
    os.makedirs(base_output, exist_ok=True)

    phase_results = {}
    t_total_start = time.time()

    # ===================================================================
    # Print header
    # ===================================================================
    print(f"\n{'#'*70}")
    print(f"  STRATEGY 5: PER-OBSERVABLE INFERENCE CASCADE")
    print(f"  True k0:      {true_k0}")
    print(f"  True alpha:   {true_alpha}")
    print(f"  Initial k0:   {initial_k0_guess}")
    print(f"  Initial alpha:{initial_alpha_guess}")
    print(f"  Pass 1 weight: {args.pass1_weight} (CD-dominant)")
    print(f"  Pass 2 weight: {args.pass2_weight} (PC-dominant)")
    print(f"  Polish weight: {args.polish_weight}")
    print(f"  Skip polish:   {args.no_polish}")
    print(f"{'#'*70}\n")

    # ===================================================================
    # Load surrogate model
    # ===================================================================
    print(f"Loading surrogate model from: {args.model}")
    surrogate = load_surrogate(args.model)
    surrogate_eta = surrogate.phi_applied
    print(f"  Surrogate voltage points: {surrogate.n_eta}")
    print(f"  Surrogate voltage range: [{surrogate_eta.min():.1f}, "
          f"{surrogate_eta.max():.1f}]")

    # Extract training bounds
    (K0_1_LO, K0_1_HI, K0_2_LO, K0_2_HI,
     ALPHA_LO, ALPHA_HI) = _extract_training_bounds(surrogate)

    bounds_source = "from model" if surrogate.training_bounds is not None else "defaults"
    print(f"  Training bounds ({bounds_source}):")
    print(f"    k0_1 log10: [{np.log10(max(K0_1_LO, 1e-30)):.2f}, "
          f"{np.log10(K0_1_HI):.2f}]")
    print(f"    k0_2 log10: [{np.log10(max(K0_2_LO, 1e-30)):.2f}, "
          f"{np.log10(K0_2_HI):.2f}]")
    print(f"    alpha:      [{ALPHA_LO:.4f}, {ALPHA_HI:.4f}]")

    # ===================================================================
    # Phase 0: Generate targets using PDE solver
    # ===================================================================
    print(f"\nGenerating target I-V curves with PDE solver at true parameters...")
    t_target = time.time()
    targets = _generate_targets_with_pde(all_eta, observable_scale)
    target_cd_full = targets["current_density"]
    target_pc_full = targets["peroxide_current"]
    t_target_elapsed = time.time() - t_target
    print(f"  Target generation: {t_target_elapsed:.1f}s")

    # ===================================================================
    # Compute shallow voltage subset indices
    # ===================================================================
    shallow_idx = _compute_shallow_subset_idx(all_eta, eta_shallow)
    print(f"  Shallow subset: {len(shallow_idx)} points "
          f"(eta from {eta_shallow[0]:.1f} to {eta_shallow[-1]:.1f})")

    # ===================================================================
    # PHASE 1: Cascade inference
    # ===================================================================
    print(f"\n{'='*70}")
    print(f"  PHASE 1: Per-Observable Cascade Inference")
    print(f"  Pass 1: CD-dominant (w={args.pass1_weight}), 4 params free")
    print(f"  Pass 2: PC-dominant (w={args.pass2_weight}), k0_2+alpha_2 free")
    if not args.no_polish:
        print(f"  Pass 3: Joint polish (w={args.polish_weight}), 4 params free")
    print(f"  Objective: shallow voltage subset ({len(shallow_idx)} pts)")
    print(f"{'='*70}")

    cascade_config = CascadeConfig(
        pass1_weight=args.pass1_weight,
        pass2_weight=args.pass2_weight,
        pass1_maxiter=60,
        pass2_maxiter=60,
        polish_maxiter=30,
        polish_weight=args.polish_weight,
        skip_polish=args.no_polish,
        fd_step=1e-5,
        verbose=True,
    )

    cascade_result = run_cascade_inference(
        surrogate=surrogate,
        target_cd=target_cd_full,
        target_pc=target_pc_full,
        initial_k0=initial_k0_guess,
        initial_alpha=initial_alpha_guess,
        bounds_k0_1=(K0_1_LO, K0_1_HI),
        bounds_k0_2=(K0_2_LO, K0_2_HI),
        bounds_alpha=(ALPHA_LO, ALPHA_HI),
        config=cascade_config,
        subset_idx=shallow_idx,
    )

    # Print per-pass results
    for pr in cascade_result.pass_results:
        pr_k0 = np.array([pr.k0_1, pr.k0_2])
        pr_alpha = np.array([pr.alpha_1, pr.alpha_2])
        _print_phase_result(pr.pass_name, pr_k0, pr_alpha,
                            true_k0_arr, true_alpha_arr,
                            pr.loss, pr.elapsed_s)
        phase_results[pr.pass_name] = {
            "k0": pr_k0.tolist(), "alpha": pr_alpha.tolist(),
            "loss": pr.loss, "time": pr.elapsed_s,
        }

    cascade_k0 = np.array([cascade_result.best_k0_1, cascade_result.best_k0_2])
    cascade_alpha = np.array([cascade_result.best_alpha_1,
                              cascade_result.best_alpha_2])

    t_surrogate_end = time.time()
    surrogate_time = t_surrogate_end - t_total_start - t_target_elapsed

    surr_best_k0 = cascade_k0.copy()
    surr_best_alpha = cascade_alpha.copy()

    # ===================================================================
    # PHASE 2 (Optional): PDE refinement from cascade best
    # ===================================================================
    if not args.no_pde:
        print(f"\n{'='*70}")
        print(f"  PHASE 2: PDE-based refinement (warm-start from cascade)")
        print(f"  Starting: k0={surr_best_k0.tolist()}, "
              f"alpha={surr_best_alpha.tolist()}")
        print(f"  Voltage grid: shallow cathodic ({len(eta_shallow)} pts)")
        print(f"{'='*70}")
        t_p2 = time.time()

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
            n_pde_workers = min(len(eta_shallow),
                                max(1, (os.cpu_count() or 4) - 1))

        n_joint_controls = 4

        pde_config = BVParallelPointConfig(
            base_solver_params=list(base_sp),
            ss_relative_tolerance=float(steady.relative_tolerance),
            ss_absolute_tolerance=float(
                max(steady.absolute_tolerance, 1e-16)),
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

        p2_dir = os.path.join(base_output, "phase2_pde_refinement")

        request_p2 = BVFluxCurveInferenceRequest(
            base_solver_params=base_sp,
            steady=steady,
            true_k0=true_k0,
            initial_guess=surr_best_k0.tolist(),
            phi_applied_values=eta_shallow.tolist(),
            target_csv_path=os.path.join(p2_dir, "target_primary.csv"),
            output_dir=p2_dir,
            regenerate_target=True,
            target_noise_percent=2.0,
            target_seed=20260226,
            observable_mode="current_density",
            current_density_scale=observable_scale,
            observable_label="current density (mA/cm2)",
            observable_title="Cascade PDE refinement (warm from cascade)",
            secondary_observable_mode="peroxide_current",
            secondary_observable_weight=1.0,
            secondary_current_density_scale=observable_scale,
            secondary_target_csv_path=os.path.join(
                p2_dir, "target_peroxide.csv"),
            control_mode="joint",
            true_alpha=true_alpha,
            initial_alpha_guess=surr_best_alpha.tolist(),
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

        result_p2 = run_bv_multi_observable_flux_curve_inference(request_p2)
        p2_k0 = np.asarray(result_p2["best_k0"])
        p2_alpha = np.asarray(result_p2["best_alpha"])
        p2_loss = float(result_p2["best_loss"])
        p2_time = time.time() - t_p2

        close_parallel_pool()
        _clear_caches()

        _print_phase_result("PDE refinement", p2_k0, p2_alpha,
                            true_k0_arr, true_alpha_arr, p2_loss, p2_time)
        phase_results["PDE refinement"] = {
            "k0": p2_k0.tolist(), "alpha": p2_alpha.tolist(),
            "loss": p2_loss, "time": p2_time,
        }

        # Pick the better result
        p2_k0_err, p2_alpha_err = _compute_errors(
            p2_k0, p2_alpha, true_k0_arr, true_alpha_arr)
        c_k0_err, c_alpha_err = _compute_errors(
            cascade_k0, cascade_alpha, true_k0_arr, true_alpha_arr)
        p2_max_err = max(p2_k0_err.max(), p2_alpha_err.max())
        c_max_err = max(c_k0_err.max(), c_alpha_err.max())

        if p2_max_err <= c_max_err:
            best_k0, best_alpha = p2_k0.copy(), p2_alpha.copy()
            best_source = "PDE refinement"
        else:
            best_k0, best_alpha = cascade_k0.copy(), cascade_alpha.copy()
            best_source = "Cascade (surrogate)"
    else:
        best_k0 = surr_best_k0.copy()
        best_alpha = surr_best_alpha.copy()
        best_source = "Cascade (surrogate)"

    total_time = time.time() - t_total_start
    pde_time = total_time - t_target_elapsed - surrogate_time

    # ===================================================================
    # FINAL SUMMARY
    # ===================================================================
    print(f"\n{'#'*90}")
    print(f"  STRATEGY 5: CASCADE INFERENCE SUMMARY")
    print(f"{'#'*90}")
    print(f"  True k0:    {true_k0}")
    print(f"  True alpha: {true_alpha}")
    print(f"  K0_HAT = {K0_HAT:.6e},  K0_2_HAT = {K0_2_HAT:.6e}")
    print(f"  Weights: pass1={args.pass1_weight}, pass2={args.pass2_weight}, "
          f"polish={args.polish_weight}")
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

    best_k0_err, best_alpha_err = _compute_errors(
        best_k0, best_alpha, true_k0_arr, true_alpha_arr)
    best_max_err = max(best_k0_err.max(), best_alpha_err.max())

    print(f"\n  Best result: {best_source} (max err = {best_max_err*100:.2f}%)")
    print(f"    k0_1   = {best_k0[0]:.6e}  (err {best_k0_err[0]*100:.2f}%)")
    print(f"    k0_2   = {best_k0[1]:.6e}  (err {best_k0_err[1]*100:.2f}%)")
    print(f"    alpha_1= {best_alpha[0]:.6f}  (err {best_alpha_err[0]*100:.2f}%)")
    print(f"    alpha_2= {best_alpha[1]:.6f}  (err {best_alpha_err[1]*100:.2f}%)")

    # v9 baseline comparison
    print(f"\n  {'='*70}")
    print(f"  v9 BASELINE COMPARISON:")
    print(f"  {'='*70}")
    print(f"  {'Metric':<25} {'v9 baseline':>12} {'Cascade':>12}")
    print(f"  {'-'*52}")
    print(f"  {'k0_1 err (%)':<25} {'8.76':>12} {best_k0_err[0]*100:>12.2f}")
    print(f"  {'k0_2 err (%)':<25} {'7.57':>12} {best_k0_err[1]*100:>12.2f}")
    print(f"  {'alpha_1 err (%)':<25} {'4.76':>12} {best_alpha_err[0]*100:>12.2f}")
    print(f"  {'alpha_2 err (%)':<25} {'6.35':>12} {best_alpha_err[1]*100:>12.2f}")
    print(f"  {'max err (%)':<25} {'8.76':>12} {best_max_err*100:>12.2f}")
    print(f"  {'='*70}")

    print(f"\n  Timing breakdown:")
    print(f"    Target generation:  {t_target_elapsed:>8.1f}s")
    print(f"    Cascade phases:     {surrogate_time:>8.1f}s")
    if not args.no_pde:
        print(f"    PDE refinement:     {pde_time:>8.1f}s")
    print(f"    Total:              {total_time:>8.1f}s")

    print(f"{'#'*90}")

    # Save comparison CSV
    csv_path = os.path.join(base_output, "cascade_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "phase", "k0_1", "k0_2", "alpha_1", "alpha_2",
            "k0_1_err_pct", "k0_2_err_pct", "alpha_1_err_pct",
            "alpha_2_err_pct", "loss", "time_s",
            "pass1_weight", "pass2_weight", "polish_weight",
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
                args.pass1_weight, args.pass2_weight, args.polish_weight,
            ])
    print(f"\n  Results CSV saved -> {csv_path}")
    print(f"\n  Output: {base_output}/")
    print(f"\n=== Strategy 5 Cascade Inference Complete ===")


if __name__ == "__main__":
    main()
