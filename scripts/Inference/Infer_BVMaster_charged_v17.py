"""Master inference protocol v17 — Physical E_eq + V vs RHE mapping.

Key changes from v16:
- Physical E_eq: E_eq_r1=0.68V, E_eq_r2=1.78V (vs RHE)
- Voltage grid in V vs RHE, mapped to phi_hat = V / V_T
- Restricted to reliable convergence window [-0.5V, +0.1V]
- PDE-only (cold start) since surrogates were trained with E_eq=0
- Single-phase PDE optimization (no surrogate warm-start)
- Dense voltage grid for better parameter resolution

Usage::
    python scripts/Inference/Infer_BVMaster_charged_v17.py
    python scripts/Inference/Infer_BVMaster_charged_v17.py --noise-percent 0
    python scripts/Inference/Infer_BVMaster_charged_v17.py --noise-percent 5 --pde-maxiter 40
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts._bv_common import (
    setup_firedrake_env,
    K0_HAT_R1, K0_HAT_R2, I_SCALE, V_T, K_SCALE,
    ALPHA_R1, ALPHA_R2,
    FOUR_SPECIES_CHARGED,
    SNES_OPTS_CHARGED,
    make_bv_solver_params,
    make_recovery_config,
    print_redimensionalized_results,
)
setup_firedrake_env()

import numpy as np

# ---------------------------------------------------------------------------
# Physical E_eq (vs RHE)
# ---------------------------------------------------------------------------
E_EQ_R1 = 0.68   # O2 + 2H+ + 2e- -> H2O2
E_EQ_R2 = 1.78   # H2O2 + 2H+ + 2e- -> 2H2O


def _compute_errors(k0, alpha, true_k0_arr, true_alpha_arr):
    k0_arr = np.asarray(k0)
    alpha_arr = np.asarray(alpha)
    k0_err = np.abs(k0_arr - true_k0_arr) / np.maximum(np.abs(true_k0_arr), 1e-16)
    alpha_err = np.abs(alpha_arr - true_alpha_arr) / np.maximum(np.abs(true_alpha_arr), 1e-16)
    return k0_err, alpha_err


def _print_result(name, k0, alpha, true_k0_arr, true_alpha_arr, loss, elapsed):
    k0_err, alpha_err = _compute_errors(k0, alpha, true_k0_arr, true_alpha_arr)
    print(f"\n  {name} result:")
    print(f"    k0_1   = {k0[0]:.6e}  (true {true_k0_arr[0]:.6e}, err {k0_err[0]*100:.2f}%)")
    print(f"    k0_2   = {k0[1]:.6e}  (true {true_k0_arr[1]:.6e}, err {k0_err[1]*100:.2f}%)")
    print(f"    alpha_1= {alpha[0]:.6f}  (true {true_alpha_arr[0]:.6f}, err {alpha_err[0]*100:.2f}%)")
    print(f"    alpha_2= {alpha[1]:.6f}  (true {true_alpha_arr[1]:.6f}, err {alpha_err[1]*100:.2f}%)")
    print(f"    Loss: {loss:.6e},  Time: {elapsed:.1f}s")
    return k0_err, alpha_err


# ---------------------------------------------------------------------------
# Target generation with physical E_eq
# ---------------------------------------------------------------------------
def _solve_clean_targets(phi_hat_values, observable_scale):
    """Generate noise-free targets using charge continuation with physical E_eq."""
    import firedrake as fd
    import pyadjoint as adj
    from Forward.bv_solver import make_graded_rectangle_mesh, solve_grid_with_charge_continuation
    from Forward.bv_solver.observables import _build_bv_observable_form

    # Tuned SNES for robustness with physical E_eq
    snes_opts = {
        "snes_type": "newtonls", "snes_max_it": 400,
        "snes_atol": 1e-7, "snes_rtol": 1e-10, "snes_stol": 1e-12,
        "snes_linesearch_type": "l2", "snes_linesearch_maxlambda": 0.3,
        "snes_divergence_tolerance": 1e10,
        "ksp_type": "preonly", "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_8": 77, "mat_mumps_icntl_14": 80,
    }

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)
    sp = make_bv_solver_params(
        eta_hat=0.0, dt=0.25, t_end=80.0,
        species=FOUR_SPECIES_CHARGED,
        snes_opts=snes_opts,
        E_eq_r1=E_EQ_R1,
        E_eq_r2=E_EQ_R2,
    )

    n_eta = len(phi_hat_values)
    clean_cd = np.full(n_eta, np.nan)
    clean_pc = np.full(n_eta, np.nan)

    print(f"  [target gen] Solving {n_eta} voltage points with E_eq=({E_EQ_R1}, {E_EQ_R2})")

    def _extract(orig_idx, phi_app, ctx):
        form_cd = _build_bv_observable_form(
            ctx, mode="current_density", reaction_index=None, scale=observable_scale)
        form_pc = _build_bv_observable_form(
            ctx, mode="peroxide_current", reaction_index=None, scale=observable_scale)
        clean_cd[orig_idx] = float(fd.assemble(form_cd))
        clean_pc[orig_idx] = float(fd.assemble(form_pc))

    with adj.stop_annotating():
        result = solve_grid_with_charge_continuation(
            sp, phi_applied_values=phi_hat_values,
            charge_steps=20, mesh=mesh,
            max_eta_gap=2.0, min_delta_z=0.002,
            per_point_callback=_extract,
        )

    n_full = sum(1 for pt in result.points.values() if pt.converged)
    n_vals = sum(1 for i in range(n_eta) if not np.isnan(clean_cd[i]))
    print(f"  [target gen] {n_full}/{n_eta} full-z, {n_vals}/{n_eta} with values")

    if n_full < n_eta:
        partial = [(pt.phi_applied, pt.achieved_z_factor)
                    for pt in result.points.values() if not pt.converged]
        for phi, z in sorted(partial):
            print(f"    PARTIAL: phi_hat={phi:.2f} (V_RHE={phi*V_T:.3f}V), z={z:.3f}")

    return clean_cd, clean_pc, n_full


def _generate_targets(phi_hat_values, observable_scale, noise_percent, noise_seed):
    """Generate targets with optional noise."""
    clean_cd, clean_pc, n_full = _solve_clean_targets(phi_hat_values, observable_scale)

    if noise_percent > 0:
        from Forward.steady_state import add_percent_noise
        noisy_cd = add_percent_noise(clean_cd, noise_percent, seed=noise_seed)
        noisy_pc = add_percent_noise(clean_pc, noise_percent, seed=noise_seed + 1)
    else:
        noisy_cd = clean_cd.copy()
        noisy_pc = clean_pc.copy()

    return {
        "current_density": noisy_cd,
        "peroxide_current": noisy_pc,
        "clean_cd": clean_cd,
        "clean_pc": clean_pc,
        "n_full_z": n_full,
    }


def main():
    parser = argparse.ArgumentParser(
        description="BV Master Inference v17 (Physical E_eq + V vs RHE)"
    )
    parser.add_argument("--pde-maxiter", type=int, default=30,
                        help="PDE max L-BFGS-B iterations (default: 30)")
    parser.add_argument("--pde-secondary-weight", type=float, default=1.0,
                        help="Weight on peroxide current")
    parser.add_argument("--noise-percent", type=float, default=2.0,
                        help="Target noise level (0.0 for noise-free)")
    parser.add_argument("--noise-seed", type=int, default=20260406,
                        help="Noise seed")
    parser.add_argument("--workers", type=int, default=0,
                        help="PDE parallel workers (0=auto)")
    parser.add_argument("--max-anodic-v", type=float, default=0.10,
                        help="Max anodic V_RHE to include (default: 0.10V, convergence limit)")
    parser.add_argument("--min-cathodic-v", type=float, default=-0.50,
                        help="Min cathodic V_RHE (default: -0.50V)")
    args = parser.parse_args()

    # ===================================================================
    # Voltage grid in V vs RHE → phi_hat
    # ===================================================================
    # Dense grid in the reliable convergence window
    v_rhe = np.array([
        -0.50, -0.45, -0.40, -0.35, -0.30, -0.25, -0.20,
        -0.15, -0.10, -0.05, 0.00, 0.05, 0.10,
    ])
    # Filter by user limits
    v_rhe = v_rhe[(v_rhe >= args.min_cathodic_v - 1e-6) &
                  (v_rhe <= args.max_anodic_v + 1e-6)]
    phi_hat = v_rhe / V_T
    phi_hat = np.sort(phi_hat)[::-1]  # Descending (most cathodic first for continuation)

    true_k0 = np.array([K0_HAT_R1, K0_HAT_R2])
    true_alpha = np.array([ALPHA_R1, ALPHA_R2])

    # Initial guesses (deliberately offset to test recovery)
    initial_k0 = np.array([K0_HAT_R1 * 2.0, K0_HAT_R2 * 0.5])
    initial_alpha = np.array([0.5, 0.4])

    observable_scale = -I_SCALE
    base_output = os.path.join("StudyResults", "master_inference_v17")
    os.makedirs(base_output, exist_ok=True)

    t_total_start = time.time()

    print(f"\n{'#'*70}")
    print(f"  MASTER INFERENCE PROTOCOL v17")
    print(f"  Physical E_eq + V vs RHE + PDE Cold Start")
    print(f"{'#'*70}")
    print(f"  E_eq_r1 = {E_EQ_R1}V, E_eq_r2 = {E_EQ_R2}V (vs RHE)")
    print(f"  V_T = {V_T:.6f}V")
    print(f"  True k0:    [{true_k0[0]:.6e}, {true_k0[1]:.6e}] (nondim)")
    print(f"  True k0:    [{true_k0[0]*K_SCALE:.4e}, {true_k0[1]*K_SCALE:.4e}] m/s")
    print(f"  True alpha: [{true_alpha[0]:.4f}, {true_alpha[1]:.4f}]")
    print(f"  Init k0:    [{initial_k0[0]:.6e}, {initial_k0[1]:.6e}]")
    print(f"  Init alpha: [{initial_alpha[0]:.4f}, {initial_alpha[1]:.4f}]")
    print(f"  V_RHE range: [{v_rhe.min():.3f}, {v_rhe.max():.3f}] ({len(v_rhe)} pts)")
    print(f"  phi_hat range: [{phi_hat.min():.1f}, {phi_hat.max():.1f}]")
    print(f"  Noise: {args.noise_percent}% (seed={args.noise_seed})")
    print(f"  PDE maxiter: {args.pde_maxiter}")
    print(f"  Secondary weight: {args.pde_secondary_weight}")
    print(f"{'#'*70}\n")

    # ===================================================================
    # Generate targets
    # ===================================================================
    print("Generating target I-V curves with physical E_eq...")
    t_target = time.time()
    targets = _generate_targets(phi_hat, observable_scale,
                                args.noise_percent, args.noise_seed)
    target_cd = targets["current_density"]
    target_pc = targets["peroxide_current"]
    t_target_elapsed = time.time() - t_target
    print(f"  Target generation: {t_target_elapsed:.1f}s, full-z: {targets['n_full_z']}/{len(phi_hat)}")

    # Quick summary of target data
    print(f"\n  Target summary (mA/cm² units):")
    for i, v in enumerate(v_rhe):
        pc_val = target_pc[i] if not np.isnan(target_pc[i]) else float('nan')
        cd_val = target_cd[i] if not np.isnan(target_cd[i]) else float('nan')
        print(f"    V={v:+.3f}V: cd={cd_val:+.4f}, pc={pc_val:+.4f}")

    # ===================================================================
    # PDE Inference (cold start)
    # ===================================================================
    from Forward.steady_state import SteadyStateConfig
    from FluxCurve import (
        BVFluxCurveInferenceRequest,
        run_bv_multi_observable_flux_curve_inference,
    )
    from FluxCurve.bv_point_solve import (
        _clear_caches, set_parallel_pool, close_parallel_pool,
    )
    from FluxCurve.bv_parallel import BVPointSolvePool

    dt = 0.25
    max_ss_steps = 320  # t_end = dt * max_ss_steps = 80
    t_end = dt * max_ss_steps

    # Base solver params WITH physical E_eq
    base_sp = make_bv_solver_params(
        eta_hat=0.0, dt=dt, t_end=t_end,
        species=FOUR_SPECIES_CHARGED,
        snes_opts={
            "snes_type": "newtonls", "snes_max_it": 400,
            "snes_atol": 1e-7, "snes_rtol": 1e-10, "snes_stol": 1e-12,
            "snes_linesearch_type": "l2", "snes_linesearch_maxlambda": 0.3,
            "snes_divergence_tolerance": 1e10,
            "ksp_type": "preonly", "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_8": 77, "mat_mumps_icntl_14": 80,
        },
        E_eq_r1=E_EQ_R1,
        E_eq_r2=E_EQ_R2,
    )

    steady = SteadyStateConfig(
        relative_tolerance=1e-4, absolute_tolerance=1e-8,
        consecutive_steps=4, max_steps=max_ss_steps,
        flux_observable="total_species", verbose=False,
    )

    recovery = make_recovery_config(max_it_cap=600)

    n_pde_workers = args.workers
    if n_pde_workers <= 0:
        n_pde_workers = min(len(phi_hat), max(1, (os.cpu_count() or 4) - 1))

    from FluxCurve.bv_point_solve import (
        _WARMSTART_MAX_STEPS, _SER_GROWTH_CAP, _SER_SHRINK, _SER_DT_MAX_RATIO,
    )
    from FluxCurve.bv_parallel import BVParallelPointConfig

    shared_config = BVParallelPointConfig(
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
    shared_pool = BVPointSolvePool(shared_config, n_workers=n_pde_workers)
    set_parallel_pool(shared_pool)
    print(f"\n  [v17] Parallel pool: {n_pde_workers} workers")

    # --- Pre-populate IC cache using charge continuation ---
    # CRITICAL: Without this, parallel workers fail because they can't solve
    # the charged-species problem from scratch at physical E_eq values.
    # The charge continuation ramps up the Poisson coupling (z-factor) gradually,
    # producing converged initial conditions that the workers can then use.
    print(f"\n  [v17] Pre-populating IC cache via charge continuation...")
    t_cache = time.time()
    from Forward.bv_solver import make_graded_rectangle_mesh, solve_grid_with_charge_continuation
    from FluxCurve.bv_point_solve import populate_cache_entry, mark_cache_populated_if_complete
    import pyadjoint as adj

    cache_mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)
    cache_sp = make_bv_solver_params(
        eta_hat=0.0, dt=0.25, t_end=80.0,
        species=FOUR_SPECIES_CHARGED,
        snes_opts={
            "snes_type": "newtonls", "snes_max_it": 400,
            "snes_atol": 1e-7, "snes_rtol": 1e-10, "snes_stol": 1e-12,
            "snes_linesearch_type": "l2", "snes_linesearch_maxlambda": 0.3,
            "snes_divergence_tolerance": 1e10,
            "ksp_type": "preonly", "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_8": 77, "mat_mumps_icntl_14": 80,
        },
        k0_hat_r1=float(initial_k0[0]),
        k0_hat_r2=float(initial_k0[1]),
        alpha_r1=float(initial_alpha[0]),
        alpha_r2=float(initial_alpha[1]),
        E_eq_r1=E_EQ_R1,
        E_eq_r2=E_EQ_R2,
    )

    with adj.stop_annotating():
        cache_result = solve_grid_with_charge_continuation(
            cache_sp,
            phi_applied_values=phi_hat,
            charge_steps=20,
            mesh=cache_mesh,
            max_eta_gap=2.0,
            min_delta_z=0.002,
        )

    for idx, pt in cache_result.points.items():
        populate_cache_entry(idx, pt.U_data, cache_result.mesh_dof_count)
    mark_cache_populated_if_complete(len(phi_hat))

    n_cached = sum(1 for pt in cache_result.points.values() if pt.converged)
    cache_time = time.time() - t_cache
    print(f"  [v17] IC cache: {n_cached}/{len(phi_hat)} full-z ({cache_time:.1f}s)")

    if not cache_result.all_converged():
        for p in cache_result.partial_points():
            print(f"    PARTIAL: phi_hat={p.phi_applied:.2f}, z={p.achieved_z_factor:.3f}")

    # --- PDE inference ---
    print(f"\n{'='*70}")
    print(f"  PDE Joint Inference (warm from IC cache)")
    print(f"  {len(phi_hat)} pts [{phi_hat.min():.1f}, {phi_hat.max():.1f}]")
    print(f"  maxiter={args.pde_maxiter}, secondary_weight={args.pde_secondary_weight}")
    print(f"{'='*70}")

    # NOTE: Do NOT call _clear_caches() here — the IC cache from charge
    # continuation is needed by the parallel workers during optimization.
    pde_dir = os.path.join(base_output, "PDE_joint")
    t_pde = time.time()

    request = BVFluxCurveInferenceRequest(
        base_solver_params=base_sp,
        steady=steady,
        true_k0=true_k0.tolist(),
        initial_guess=initial_k0.tolist(),
        phi_applied_values=phi_hat.tolist(),
        target_csv_path=os.path.join(pde_dir, "target_primary.csv"),
        output_dir=pde_dir,
        regenerate_target=True,
        target_noise_percent=args.noise_percent,
        target_seed=args.noise_seed,
        observable_mode="current_density",
        current_density_scale=observable_scale,
        observable_label="current density (mA/cm2)",
        observable_title="v17: PDE joint (physical E_eq, cold start)",
        secondary_observable_mode="peroxide_current",
        secondary_observable_weight=args.pde_secondary_weight,
        secondary_current_density_scale=observable_scale,
        secondary_target_csv_path=os.path.join(pde_dir, "target_peroxide.csv"),
        control_mode="joint",
        true_alpha=true_alpha.tolist(),
        initial_alpha_guess=initial_alpha.tolist(),
        alpha_lower=0.05, alpha_upper=0.95,
        k0_lower=1e-8, k0_upper=100.0,
        log_space=True,
        mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
        max_eta_gap=2.0,
        optimizer_method="L-BFGS-B",
        optimizer_options={
            "maxiter": args.pde_maxiter,
            "ftol": 1e-9,
            "gtol": 1e-6,
            "disp": True,
        },
        max_iters=args.pde_maxiter,
        live_plot=False,
        forward_recovery=recovery,
        parallel_fast_path=True,
        parallel_workers=n_pde_workers,
    )

    result = run_bv_multi_observable_flux_curve_inference(
        request,
        precomputed_targets={
            "primary": target_cd,
            "secondary": target_pc,
        },
    )

    best_k0 = np.asarray(result["best_k0"])
    best_alpha = np.asarray(result["best_alpha"])
    best_loss = float(result["best_loss"])
    pde_time = time.time() - t_pde

    close_parallel_pool()
    _clear_caches()

    _print_result("PDE joint", best_k0, best_alpha,
                  true_k0, true_alpha, best_loss, pde_time)

    # ===================================================================
    # SUMMARY
    # ===================================================================
    total_time = time.time() - t_total_start
    k0_err, alpha_err = _compute_errors(best_k0, best_alpha, true_k0, true_alpha)

    print(f"\n{'#'*70}")
    print(f"  v17 INFERENCE SUMMARY")
    print(f"{'#'*70}")
    print(f"  E_eq: ({E_EQ_R1}, {E_EQ_R2}) V vs RHE")
    print(f"  V_RHE range: [{v_rhe.min():.3f}, {v_rhe.max():.3f}] ({len(v_rhe)} pts)")
    print(f"  Noise: {args.noise_percent}%")
    print(f"")
    print(f"  k0_1:    {best_k0[0]:.6e}  (true {true_k0[0]:.6e}, err {k0_err[0]*100:.2f}%)")
    print(f"  k0_2:    {best_k0[1]:.6e}  (true {true_k0[1]:.6e}, err {k0_err[1]*100:.2f}%)")
    print(f"  alpha_1: {best_alpha[0]:.6f}  (true {true_alpha[0]:.6f}, err {alpha_err[0]*100:.2f}%)")
    print(f"  alpha_2: {best_alpha[1]:.6f}  (true {true_alpha[1]:.6f}, err {alpha_err[1]*100:.2f}%)")
    print(f"  Max err: {max(k0_err.max(), alpha_err.max())*100:.2f}%")
    print(f"  Loss:    {best_loss:.6e}")
    print(f"")
    print(f"  Total time: {total_time:.1f}s (target: {t_target_elapsed:.1f}s, PDE: {pde_time:.1f}s)")
    print(f"{'#'*70}")

    print_redimensionalized_results(
        best_k0, true_k0,
        best_alpha=best_alpha, true_alpha=true_alpha,
    )


if __name__ == "__main__":
    main()
