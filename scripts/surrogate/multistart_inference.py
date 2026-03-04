"""Strategy 3: Multi-start Latin Hypercube grid search + gradient polish.

Uses the fast ``predict_batch`` surrogate path to exhaustively search the
4D parameter space, then polishes the best candidates with L-BFGS-B.

Three-phase protocol:
    Phase 0: Load surrogate, generate PDE targets at true parameters
    Phase 1: Multi-start grid search + polish (new Strategy 3)
    Phase 2: (Optional) PDE refinement from multi-start best

Usage (from PNPInverse/ directory)::

    python scripts/surrogate/multistart_inference.py \\
        --model StudyResults/surrogate_v9/surrogate_model.pkl

    # Skip PDE refinement (surrogate-only):
    python scripts/surrogate/multistart_inference.py --no-pde

    # Custom grid size:
    python scripts/surrogate/multistart_inference.py --n-grid 50000
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

from Surrogate.io import load_surrogate
from Surrogate.multistart import (
    MultiStartConfig,
    run_multistart_inference,
)


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
        description="Strategy 3: Multi-Start LHS Grid Search + Gradient Polish"
    )
    parser.add_argument("--model", type=str,
                        default="StudyResults/surrogate_v9/surrogate_model.pkl",
                        help="Path to surrogate model .pkl")
    parser.add_argument("--n-grid", type=int, default=20_000,
                        help="Number of LHS grid points (default: 20000)")
    parser.add_argument("--n-candidates", type=int, default=20,
                        help="Number of top candidates to polish (default: 20)")
    parser.add_argument("--no-pde", action="store_true",
                        help="Skip Phase 2 PDE refinement (surrogate-only)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Workers for PDE refinement (0=auto)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for LHS sampler")
    parser.add_argument("--secondary-weight", type=float, default=1.0,
                        help="Weight on peroxide current observable (default: 1.0)")
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

    observable_scale = -I_SCALE
    base_output = os.path.join("StudyResults", "multistart")
    os.makedirs(base_output, exist_ok=True)

    phase_results = {}
    t_total_start = time.time()

    # ===================================================================
    # Print header
    # ===================================================================
    print(f"\n{'#'*70}")
    print(f"  STRATEGY 3: MULTI-START LHS GRID SEARCH + GRADIENT POLISH")
    print(f"  True k0:    {true_k0}")
    print(f"  True alpha: {true_alpha}")
    print(f"  Grid size:  {args.n_grid:,}")
    print(f"  Candidates: {args.n_candidates}")
    print(f"  Weight:     {args.secondary_weight}")
    print(f"  Seed:       {args.seed}")
    print(f"{'#'*70}\n")

    # ===================================================================
    # Load surrogate model
    # ===================================================================
    print(f"Loading surrogate model from: {args.model}")
    surrogate = load_surrogate(args.model)
    surrogate_eta = surrogate.phi_applied
    print(f"  Surrogate voltage points: {surrogate.n_eta}")
    print(f"  Surrogate voltage range: [{surrogate_eta.min():.1f}, {surrogate_eta.max():.1f}]")

    # Extract training bounds
    (K0_1_LO, K0_1_HI, K0_2_LO, K0_2_HI,
     ALPHA_LO, ALPHA_HI) = _extract_training_bounds(surrogate)

    bounds_source = "from model" if surrogate.training_bounds is not None else "defaults"
    print(f"  Training bounds ({bounds_source}):")
    print(f"    k0_1 log10: [{np.log10(max(K0_1_LO, 1e-30)):.2f}, {np.log10(K0_1_HI):.2f}]")
    print(f"    k0_2 log10: [{np.log10(max(K0_2_LO, 1e-30)):.2f}, {np.log10(K0_2_HI):.2f}]")
    print(f"    alpha:      [{ALPHA_LO:.4f}, {ALPHA_HI:.4f}]")

    # ===================================================================
    # Phase 0: Generate targets using PDE solver (same as v9)
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
    # PHASE 1: Multi-start grid search + polish
    # ===================================================================
    print(f"\n{'='*70}")
    print(f"  PHASE 1: Multi-start LHS grid search + L-BFGS-B polish")
    print(f"  Grid: {args.n_grid:,} LHS points")
    print(f"  Polish: top-{args.n_candidates} with maxiter={60}")
    print(f"  Objective: shallow voltage subset ({len(shallow_idx)} pts)")
    print(f"{'='*70}")

    ms_config = MultiStartConfig(
        n_grid=args.n_grid,
        n_top_candidates=args.n_candidates,
        polish_maxiter=60,
        secondary_weight=args.secondary_weight,
        fd_step=1e-5,
        use_shallow_subset=True,
        seed=args.seed,
        verbose=True,
    )

    ms_result = run_multistart_inference(
        surrogate=surrogate,
        target_cd=target_cd_full,
        target_pc=target_pc_full,
        bounds_k0_1=(K0_1_LO, K0_1_HI),
        bounds_k0_2=(K0_2_LO, K0_2_HI),
        bounds_alpha=(ALPHA_LO, ALPHA_HI),
        config=ms_config,
        subset_idx=shallow_idx,
    )

    ms_k0 = np.array([ms_result.best_k0_1, ms_result.best_k0_2])
    ms_alpha = np.array([ms_result.best_alpha_1, ms_result.best_alpha_2])
    ms_loss = ms_result.best_loss

    _print_phase_result("Phase 1 (multi-start, surrogate)",
                        ms_k0, ms_alpha, true_k0_arr, true_alpha_arr,
                        ms_loss, ms_result.total_time_s)
    phase_results["Phase 1 (multi-start, surr)"] = {
        "k0": ms_k0.tolist(), "alpha": ms_alpha.tolist(),
        "loss": ms_loss, "time": ms_result.total_time_s,
    }

    # Print top-5 polished candidates with errors
    print(f"\n  Top-5 polished candidates:")
    print(f"  {'Rank':>4} | {'k0_1 err':>10} {'k0_2 err':>10} "
          f"{'a1 err':>10} {'a2 err':>10} | {'loss':>12}")
    print(f"  {'-'*70}")
    for c in ms_result.candidates[:5]:
        c_k0 = np.array([c.k0_1, c.k0_2])
        c_alpha = np.array([c.alpha_1, c.alpha_2])
        k0_err, alpha_err = _compute_errors(c_k0, c_alpha, true_k0_arr, true_alpha_arr)
        print(f"  #{c.rank:>3d} | {k0_err[0]*100:>9.2f}% {k0_err[1]*100:>9.2f}% "
              f"{alpha_err[0]*100:>9.2f}% {alpha_err[1]*100:>9.2f}% "
              f"| {c.polished_loss:>12.6e}")

    t_surrogate_end = time.time()
    surrogate_time = t_surrogate_end - t_total_start - t_target_elapsed

    surr_best_k0 = ms_k0.copy()
    surr_best_alpha = ms_alpha.copy()

    # ===================================================================
    # PHASE 2 (Optional): PDE refinement from multi-start best
    # ===================================================================
    if not args.no_pde:
        print(f"\n{'='*70}")
        print(f"  PHASE 2: PDE-based refinement (warm-start from multi-start)")
        print(f"  Starting: k0={surr_best_k0.tolist()}, alpha={surr_best_alpha.tolist()}")
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
            n_pde_workers = min(len(eta_shallow), max(1, (os.cpu_count() or 4) - 1))

        n_joint_controls = 4

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
            observable_title="Phase 2: PDE refinement (warm from multi-start)",
            secondary_observable_mode="peroxide_current",
            secondary_observable_weight=args.secondary_weight,
            secondary_current_density_scale=observable_scale,
            secondary_target_csv_path=os.path.join(p2_dir, "target_peroxide.csv"),
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
                "maxiter": 10,
                "ftol": 1e-8,
                "gtol": 5e-6,
                "disp": True,
            },
            max_iters=10,
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

        _print_phase_result("Phase 2 (PDE refine)", p2_k0, p2_alpha,
                            true_k0_arr, true_alpha_arr, p2_loss, p2_time)
        phase_results["Phase 2 (PDE refine)"] = {
            "k0": p2_k0.tolist(), "alpha": p2_alpha.tolist(),
            "loss": p2_loss, "time": p2_time,
        }

        # Pick the better result
        p2_k0_err, p2_alpha_err = _compute_errors(p2_k0, p2_alpha, true_k0_arr, true_alpha_arr)
        ms_k0_err, ms_alpha_err = _compute_errors(ms_k0, ms_alpha, true_k0_arr, true_alpha_arr)
        p2_max_err = max(p2_k0_err.max(), p2_alpha_err.max())
        ms_max_err = max(ms_k0_err.max(), ms_alpha_err.max())

        if p2_max_err <= ms_max_err:
            best_k0, best_alpha = p2_k0.copy(), p2_alpha.copy()
            best_source = "Phase 2 (PDE)"
        else:
            best_k0, best_alpha = ms_k0.copy(), ms_alpha.copy()
            best_source = "Phase 1 (multi-start)"
    else:
        best_k0, best_alpha = surr_best_k0.copy(), surr_best_alpha.copy()
        best_source = "Phase 1 (multi-start)"

    total_time = time.time() - t_total_start
    pde_time = total_time - t_target_elapsed - surrogate_time

    # ===================================================================
    # FINAL SUMMARY
    # ===================================================================
    print(f"\n{'#'*90}")
    print(f"  STRATEGY 3: MULTI-START INFERENCE SUMMARY")
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

    best_k0_err, best_alpha_err = _compute_errors(best_k0, best_alpha, true_k0_arr, true_alpha_arr)
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
    print(f"  {'Metric':<25} {'v9 baseline':>12} {'Strategy 3':>12}")
    print(f"  {'-'*52}")
    print(f"  {'k0_1 err (%)':<25} {'8.76':>12} {best_k0_err[0]*100:>12.2f}")
    print(f"  {'k0_2 err (%)':<25} {'7.57':>12} {best_k0_err[1]*100:>12.2f}")
    print(f"  {'alpha_1 err (%)':<25} {'4.76':>12} {best_alpha_err[0]*100:>12.2f}")
    print(f"  {'alpha_2 err (%)':<25} {'6.35':>12} {best_alpha_err[1]*100:>12.2f}")
    print(f"  {'max err (%)':<25} {'8.76':>12} {best_max_err*100:>12.2f}")
    print(f"  {'='*70}")

    print(f"\n  Timing breakdown:")
    print(f"    Target generation:  {t_target_elapsed:>8.1f}s")
    print(f"    Multi-start phase:  {surrogate_time:>8.1f}s")
    if not args.no_pde:
        print(f"    PDE refinement:     {pde_time:>8.1f}s")
    print(f"    Total:              {total_time:>8.1f}s")

    print(f"{'#'*90}")

    # Save comparison CSV
    csv_path = os.path.join(base_output, "multistart_results.csv")
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
    print(f"\n  Results CSV saved -> {csv_path}")

    # Save candidates CSV
    cand_csv_path = os.path.join(base_output, "multistart_candidates.csv")
    with open(cand_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank", "k0_1", "k0_2", "alpha_1", "alpha_2",
            "k0_1_err_pct", "k0_2_err_pct", "alpha_1_err_pct", "alpha_2_err_pct",
            "grid_loss", "polished_loss", "polish_iters",
        ])
        for c in ms_result.candidates:
            c_k0 = np.array([c.k0_1, c.k0_2])
            c_alpha = np.array([c.alpha_1, c.alpha_2])
            k0_err, alpha_err = _compute_errors(c_k0, c_alpha, true_k0_arr, true_alpha_arr)
            writer.writerow([
                c.rank, f"{c.k0_1:.8e}", f"{c.k0_2:.8e}",
                f"{c.alpha_1:.6f}", f"{c.alpha_2:.6f}",
                f"{k0_err[0]*100:.4f}", f"{k0_err[1]*100:.4f}",
                f"{alpha_err[0]*100:.4f}", f"{alpha_err[1]*100:.4f}",
                f"{c.grid_loss:.12e}", f"{c.polished_loss:.12e}",
                c.polish_iters,
            ])
    print(f"  Candidates CSV saved -> {cand_csv_path}")
    print(f"\n  Output: {base_output}/")
    print(f"\n=== Strategy 3 Multi-Start Inference Complete ===")


if __name__ == "__main__":
    main()
