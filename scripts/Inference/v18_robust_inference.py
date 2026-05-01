"""V18 inference using stabilized z=1 solver for full onset coverage.

Identical to v17_robust_inference.py EXCEPT:
1. Monkey-patches build_forms to add ClO4- stabilization (d_art=0.001)
2. Voltage grid extends to +0.40V (onset region) for k0 identifiability

Usage::
    python scripts/Inference/v18_robust_inference.py
    python scripts/Inference/v18_robust_inference.py --offset 0.1
    python scripts/Inference/v18_robust_inference.py --maxiter 40
"""
from __future__ import annotations
import os, sys, time, argparse

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(line_buffering=True)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# -------------------------------------------------------
# Monkey-patch build_forms BEFORE any FluxCurve imports
# -------------------------------------------------------
from Forward.bv_solver import forms as _bv_forms_module
from Forward.bv_solver.stabilization import add_stabilization

_original_build_forms = _bv_forms_module.build_forms

def _stabilized_build_forms(ctx, solver_params):
    ctx = _original_build_forms(ctx, solver_params)
    ctx = add_stabilization(ctx, solver_params, d_art_scale=0.001, stabilized_species=[3])
    return ctx

_bv_forms_module.build_forms = _stabilized_build_forms
import Forward.bv_solver as _bv_pkg
_bv_pkg.build_forms = _stabilized_build_forms


def main():
    from scripts._bv_common import (
        setup_firedrake_env, K0_HAT_R1, K0_HAT_R2, I_SCALE, V_T,
        ALPHA_R1, ALPHA_R2, FOUR_SPECIES_CHARGED, make_bv_solver_params,
        make_recovery_config,
    )
    setup_firedrake_env()

    import numpy as np

    parser = argparse.ArgumentParser(description="V18 stabilized inference")
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--maxiter", type=int, default=25)
    parser.add_argument("--offset", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    E_EQ_R1, E_EQ_R2 = 0.68, 1.78

    # Onset-focused voltage grid: k0 is identifiable where current is
    # transitioning from kinetically-limited to transport-limited.
    # Exclude deeply cathodic points that dominate the sum-of-squares
    # and mask k0 sensitivity.
    V_RHE = np.array([
        0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
    ])
    PHI_HAT = np.sort(V_RHE / V_T)[::-1]

    true_k0 = np.array([K0_HAT_R1, K0_HAT_R2])
    true_alpha = np.array([ALPHA_R1, ALPHA_R2])
    off = args.offset
    init_k0 = true_k0 * (1.0 + off)
    init_alpha = true_alpha * (1.0 - off * 0.5)

    observable_scale = -I_SCALE
    base_output = os.path.join("StudyResults", "v18_robust_inference")
    os.makedirs(base_output, exist_ok=True)

    SNES_OPTS = {
        "snes_type": "newtonls", "snes_max_it": 400,
        "snes_atol": 1e-7, "snes_rtol": 1e-10, "snes_stol": 1e-12,
        "snes_linesearch_type": "l2", "snes_linesearch_maxlambda": 0.3,
        "snes_divergence_tolerance": 1e10,
        "ksp_type": "preonly", "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_8": 77, "mat_mumps_icntl_14": 80,
    }

    print(f"\n{'#'*70}")
    print(f"  V18 STABILIZED INFERENCE (ClO4 d_art=0.001)")
    print(f"{'#'*70}")
    print(f"  E_eq: ({E_EQ_R1}, {E_EQ_R2}) V vs RHE")
    print(f"  True k0:    {true_k0.tolist()}")
    print(f"  True alpha: {true_alpha.tolist()}")
    print(f"  Init k0:    {init_k0.tolist()} ({off*100:.0f}% offset)")
    print(f"  Init alpha: {init_alpha.tolist()}")
    print(f"  V_RHE: [{V_RHE.min():.2f}, {V_RHE.max():.2f}] ({len(V_RHE)} pts)")
    print(f"  Noise: {args.noise}%")
    print(f"{'#'*70}\n")

    # ------------------------------------------------------------------
    # Step 1: Generate targets at TRUE params
    # ------------------------------------------------------------------
    print("Step 1: Generating targets at true params (stabilized z=1)...")
    from Forward.bv_solver.robust_forward import solve_curve_robust

    sp_true = make_bv_solver_params(
        eta_hat=0.0, dt=0.25, t_end=80.0,
        species=FOUR_SPECIES_CHARGED, snes_opts=SNES_OPTS,
        E_eq_r1=E_EQ_R1, E_eq_r2=E_EQ_R2,
    )
    target_result = solve_curve_robust(
        sp_true, PHI_HAT, observable_scale,
        n_workers=args.workers or None, charge_steps=20,
    )
    target_cd = target_result.cd.copy()
    target_pc = target_result.pc.copy()

    if args.noise > 0:
        from Forward.steady_state import add_percent_noise
        target_cd = add_percent_noise(target_cd, args.noise, seed=args.seed)
        target_pc = add_percent_noise(target_pc, args.noise, seed=args.seed + 1)

    print(f"  Targets: {target_result.n_converged}/{target_result.n_total} converged")

    # Map phi_hat → V_RHE for printing (PHI_HAT is sorted descending)
    for i in range(len(PHI_HAT)):
        V_i = PHI_HAT[i] * V_T
        print(f"    phi={PHI_HAT[i]:+8.2f} (V≈{V_i:+.2f}): cd={target_cd[i]:10.6f}, pc={target_pc[i]:10.6f}")

    # ------------------------------------------------------------------
    # Step 2: Pre-populate IC cache at INITIAL GUESS params
    # ------------------------------------------------------------------
    print(f"\nStep 2: Populating IC cache at initial guess params...")
    from Forward.bv_solver.robust_forward import populate_ic_cache_robust

    sp_init = make_bv_solver_params(
        eta_hat=0.0, dt=0.25, t_end=80.0,
        species=FOUR_SPECIES_CHARGED, snes_opts=SNES_OPTS,
        k0_hat_r1=float(init_k0[0]), k0_hat_r2=float(init_k0[1]),
        alpha_r1=float(init_alpha[0]), alpha_r2=float(init_alpha[1]),
        E_eq_r1=E_EQ_R1, E_eq_r2=E_EQ_R2,
    )
    n_cached, cache_result = populate_ic_cache_robust(
        sp_init, PHI_HAT, observable_scale,
        n_workers=args.workers or None, charge_steps=20,
    )

    # ------------------------------------------------------------------
    # Step 3: Run FluxCurve adjoint inference
    # ------------------------------------------------------------------
    print(f"\nStep 3: Running adjoint inference with L-BFGS-B...")
    from Forward.steady_state import SteadyStateConfig
    from FluxCurve import (
        BVFluxCurveInferenceRequest,
        run_bv_multi_observable_flux_curve_inference,
    )
    from FluxCurve.bv_point_solve import set_parallel_pool, close_parallel_pool
    from FluxCurve.bv_parallel import BVPointSolvePool, BVParallelPointConfig
    from FluxCurve.bv_point_solve import (
        _WARMSTART_MAX_STEPS, _SER_GROWTH_CAP, _SER_SHRINK, _SER_DT_MAX_RATIO,
    )

    dt = 0.25
    max_ss = 320
    base_sp = make_bv_solver_params(
        eta_hat=0.0, dt=dt, t_end=dt * max_ss,
        species=FOUR_SPECIES_CHARGED, snes_opts=SNES_OPTS,
        E_eq_r1=E_EQ_R1, E_eq_r2=E_EQ_R2,
    )

    steady = SteadyStateConfig(
        relative_tolerance=1e-4, absolute_tolerance=1e-8,
        consecutive_steps=4, max_steps=max_ss,
        flux_observable="total_species", verbose=False,
    )
    recovery = make_recovery_config(max_it_cap=600)

    n_w = args.workers if args.workers > 0 else min(len(PHI_HAT), max(1, (os.cpu_count() or 4) - 1))

    cfg = BVParallelPointConfig(
        base_solver_params=list(base_sp),
        ss_relative_tolerance=float(steady.relative_tolerance),
        ss_absolute_tolerance=float(max(steady.absolute_tolerance, 1e-16)),
        ss_consecutive_steps=int(steady.consecutive_steps),
        ss_max_steps=int(steady.max_steps),
        mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
        blob_initial_condition=False, fail_penalty=1e9,
        warmstart_max_steps=_WARMSTART_MAX_STEPS,
        observable_mode="current_density",
        observable_reaction_index=None,
        observable_scale=observable_scale,
        control_mode="joint", n_controls=4,
        ser_growth_cap=_SER_GROWTH_CAP, ser_shrink=_SER_SHRINK,
        ser_dt_max_ratio=_SER_DT_MAX_RATIO,
        secondary_observable_mode="peroxide_current",
        secondary_observable_reaction_index=None,
        secondary_observable_scale=observable_scale,
    )

    pool = BVPointSolvePool(cfg, n_workers=n_w)
    set_parallel_pool(pool)

    pde_dir = os.path.join(base_output, "PDE_joint")
    os.makedirs(pde_dir, exist_ok=True)

    req = BVFluxCurveInferenceRequest(
        base_solver_params=base_sp,
        steady=steady,
        true_k0=true_k0.tolist(),
        initial_guess=init_k0.tolist(),
        phi_applied_values=PHI_HAT.tolist(),
        target_csv_path=os.path.join(pde_dir, "target_primary.csv"),
        output_dir=pde_dir,
        regenerate_target=True,
        target_noise_percent=args.noise,
        target_seed=args.seed,
        observable_mode="current_density",
        current_density_scale=observable_scale,
        observable_label="current density (mA/cm2)",
        observable_title="v18 stabilized: physical E_eq + onset",
        secondary_observable_mode="peroxide_current",
        secondary_observable_weight=1.0,
        secondary_current_density_scale=observable_scale,
        secondary_target_csv_path=os.path.join(pde_dir, "target_peroxide.csv"),
        control_mode="joint",
        true_alpha=true_alpha.tolist(),
        initial_alpha_guess=init_alpha.tolist(),
        alpha_lower=0.05, alpha_upper=0.95,
        k0_lower=1e-8, k0_upper=100.0,
        log_space=True,
        mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
        max_eta_gap=2.0,
        optimizer_method="L-BFGS-B",
        optimizer_options={
            "maxiter": args.maxiter,
            "ftol": 1e-10,
            "gtol": 1e-7,
            "disp": True,
        },
        max_iters=args.maxiter,
        live_plot=False,
        forward_recovery=recovery,
        parallel_fast_path=True,
        parallel_workers=n_w,
    )

    t_pde = time.time()
    result = run_bv_multi_observable_flux_curve_inference(
        req, precomputed_targets={"primary": target_cd, "secondary": target_pc},
    )
    pde_time = time.time() - t_pde

    close_parallel_pool()

    best_k0 = np.asarray(result["best_k0"])
    best_alpha = np.asarray(result["best_alpha"])
    k0_err = np.abs(best_k0 - true_k0) / true_k0
    alpha_err = np.abs(best_alpha - true_alpha) / true_alpha

    print(f"\n{'#'*70}")
    print(f"  V18 STABILIZED INFERENCE RESULT")
    print(f"{'#'*70}")
    print(f"  k0_1:    {best_k0[0]:.6e}  (true {true_k0[0]:.6e}, err {k0_err[0]*100:.1f}%)")
    print(f"  k0_2:    {best_k0[1]:.6e}  (true {true_k0[1]:.6e}, err {k0_err[1]*100:.1f}%)")
    print(f"  alpha_1: {best_alpha[0]:.4f}  (true {true_alpha[0]:.4f}, err {alpha_err[0]*100:.1f}%)")
    print(f"  alpha_2: {best_alpha[1]:.4f}  (true {true_alpha[1]:.4f}, err {alpha_err[1]*100:.1f}%)")
    print(f"  Max err: {max(k0_err.max(), alpha_err.max())*100:.1f}%")
    print(f"  Loss:    {result['best_loss']:.6e}")
    print(f"  PDE time: {pde_time:.1f}s")
    print(f"{'#'*70}")


if __name__ == "__main__":
    main()
