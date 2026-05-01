"""Quick test: can the v17 pipeline recover parameters from near-true initial guess?

Tests three scenarios:
1. From true params (should stay at true with loss~0)
2. From 20% offset (should recover)
3. From 2x offset (harder, tests basin of convergence)
"""
from __future__ import annotations
import os, sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(line_buffering=True)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts._bv_common import (
    setup_firedrake_env, K0_HAT_R1, K0_HAT_R2, I_SCALE, V_T,
    ALPHA_R1, ALPHA_R2, FOUR_SPECIES_CHARGED, make_bv_solver_params, make_recovery_config,
)
setup_firedrake_env()

import numpy as np
import time

E_EQ_R1, E_EQ_R2 = 0.68, 1.78

# Voltage grid (reliable convergence window)
V_RHE = np.array([-0.50, -0.40, -0.30, -0.20, -0.10, 0.00, 0.05, 0.10])
PHI_HAT = np.sort(V_RHE / V_T)[::-1]

true_k0 = np.array([K0_HAT_R1, K0_HAT_R2])
true_alpha = np.array([ALPHA_R1, ALPHA_R2])

observable_scale = -I_SCALE


def generate_targets():
    """Generate clean targets at true parameters."""
    import firedrake as fd
    import pyadjoint as adj
    from Forward.bv_solver import make_graded_rectangle_mesh, solve_grid_with_charge_continuation
    from Forward.bv_solver.observables import _build_bv_observable_form

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
        species=FOUR_SPECIES_CHARGED, snes_opts=snes_opts,
        E_eq_r1=E_EQ_R1, E_eq_r2=E_EQ_R2,
    )

    n = len(PHI_HAT)
    cd = np.full(n, np.nan)
    pc = np.full(n, np.nan)

    def _extract(idx, phi, ctx):
        cd[idx] = float(fd.assemble(_build_bv_observable_form(
            ctx, mode="current_density", reaction_index=None, scale=observable_scale)))
        pc[idx] = float(fd.assemble(_build_bv_observable_form(
            ctx, mode="peroxide_current", reaction_index=None, scale=observable_scale)))

    with adj.stop_annotating():
        solve_grid_with_charge_continuation(
            sp, phi_applied_values=PHI_HAT,
            charge_steps=20, mesh=mesh, max_eta_gap=2.0, min_delta_z=0.002,
            per_point_callback=_extract,
        )

    print(f"  Targets: {sum(~np.isnan(cd))}/{n} converged")
    return cd, pc


def run_inference(init_k0, init_alpha, label, target_cd, target_pc):
    """Run PDE inference from given initial guess."""
    from Forward.steady_state import SteadyStateConfig
    from FluxCurve import BVFluxCurveInferenceRequest, run_bv_multi_observable_flux_curve_inference
    from FluxCurve.bv_point_solve import _clear_caches, set_parallel_pool, close_parallel_pool
    from FluxCurve.bv_parallel import BVPointSolvePool, BVParallelPointConfig
    from FluxCurve.bv_point_solve import _WARMSTART_MAX_STEPS, _SER_GROWTH_CAP, _SER_SHRINK, _SER_DT_MAX_RATIO

    dt = 0.25
    max_ss = 320
    base_sp = make_bv_solver_params(
        eta_hat=0.0, dt=dt, t_end=dt * max_ss,
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
        E_eq_r1=E_EQ_R1, E_eq_r2=E_EQ_R2,
    )
    steady = SteadyStateConfig(
        relative_tolerance=1e-4, absolute_tolerance=1e-8,
        consecutive_steps=4, max_steps=max_ss,
        flux_observable="total_species", verbose=False,
    )
    recovery = make_recovery_config(max_it_cap=600)

    n_w = min(len(PHI_HAT), max(1, (os.cpu_count() or 4) - 1))
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

    _clear_caches()
    pool = BVPointSolvePool(cfg, n_workers=n_w)
    set_parallel_pool(pool)

    out_dir = os.path.join("StudyResults", "v17_recovery_test", label.replace(" ", "_"))
    os.makedirs(out_dir, exist_ok=True)

    req = BVFluxCurveInferenceRequest(
        base_solver_params=base_sp,
        steady=steady,
        true_k0=true_k0.tolist(),
        initial_guess=init_k0.tolist(),
        phi_applied_values=PHI_HAT.tolist(),
        target_csv_path=os.path.join(out_dir, "target_primary.csv"),
        output_dir=out_dir,
        regenerate_target=True,
        target_noise_percent=0,
        target_seed=42,
        observable_mode="current_density",
        current_density_scale=observable_scale,
        observable_label="current density (mA/cm2)",
        observable_title=f"Recovery test: {label}",
        secondary_observable_mode="peroxide_current",
        secondary_observable_weight=1.0,
        secondary_current_density_scale=observable_scale,
        secondary_target_csv_path=os.path.join(out_dir, "target_peroxide.csv"),
        control_mode="joint",
        true_alpha=true_alpha.tolist(),
        initial_alpha_guess=init_alpha.tolist(),
        alpha_lower=0.05, alpha_upper=0.95,
        k0_lower=1e-8, k0_upper=100.0,
        log_space=True,
        mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
        max_eta_gap=2.0,
        optimizer_method="L-BFGS-B",
        optimizer_options={"maxiter": 20, "ftol": 1e-10, "gtol": 1e-7, "disp": True},
        max_iters=20,
        live_plot=False,
        forward_recovery=recovery,
        parallel_fast_path=True,
        parallel_workers=n_w,
    )

    t0 = time.time()
    result = run_bv_multi_observable_flux_curve_inference(
        req, precomputed_targets={"primary": target_cd, "secondary": target_pc},
    )
    elapsed = time.time() - t0

    close_parallel_pool()
    _clear_caches()

    best_k0 = np.asarray(result["best_k0"])
    best_alpha = np.asarray(result["best_alpha"])
    k0_err = np.abs(best_k0 - true_k0) / true_k0
    alpha_err = np.abs(best_alpha - true_alpha) / true_alpha

    print(f"\n  [{label}] Results ({elapsed:.0f}s):")
    print(f"    k0_1:    {best_k0[0]:.6e} (err {k0_err[0]*100:.1f}%)")
    print(f"    k0_2:    {best_k0[1]:.6e} (err {k0_err[1]*100:.1f}%)")
    print(f"    alpha_1: {best_alpha[0]:.4f} (err {alpha_err[0]*100:.1f}%)")
    print(f"    alpha_2: {best_alpha[1]:.4f} (err {alpha_err[1]*100:.1f}%)")
    print(f"    Loss:    {result['best_loss']:.6e}")
    return result


def main():
    print("=== V17 RECOVERY TEST ===\n")
    print("Generating targets...")
    target_cd, target_pc = generate_targets()
    print(f"  cd range: [{np.nanmin(target_cd):.4f}, {np.nanmax(target_cd):.4f}]")
    print(f"  pc range: [{np.nanmin(target_pc):.4f}, {np.nanmax(target_pc):.4f}]")

    # Test 1: from true params (sanity check)
    print(f"\n--- Test 1: From TRUE parameters ---")
    run_inference(true_k0, true_alpha, "from_true", target_cd, target_pc)

    # Test 2: 20% offset
    print(f"\n--- Test 2: From 20% offset ---")
    run_inference(true_k0 * 1.2, true_alpha * 1.1, "20pct_offset", target_cd, target_pc)

    # Test 3: 2x offset
    print(f"\n--- Test 3: From 2x offset ---")
    run_inference(
        np.array([K0_HAT_R1 * 2, K0_HAT_R2 * 0.5]),
        np.array([0.5, 0.4]),
        "2x_offset", target_cd, target_pc,
    )


if __name__ == "__main__":
    main()
