"""V18 3-Species Boltzmann Inference: adjoint gradients with real physics.

Uses the 3-species model [O2, H2O2, H+] with Boltzmann ClO4- background.
No artificial diffusion. Full z=1 convergence to V=0.60V.
Monkey-patches build_forms to add the Boltzmann background term.
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
# 3-species constants (before any Firedrake imports)
# -------------------------------------------------------
from scripts._bv_common import (
    setup_firedrake_env, V_T, I_SCALE,
    K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2,
    D_O2_HAT, D_H2O2_HAT, D_HP_HAT,
    C_O2_HAT, C_H2O2_HAT, C_HP_HAT, C_CLO4_HAT,
    A_DEFAULT, N_ELECTRONS,
    _make_nondim_cfg, _make_bv_convergence_cfg,
    SNES_OPTS_CHARGED, make_recovery_config,
)
setup_firedrake_env()

E_EQ_R1, E_EQ_R2 = 0.68, 1.78

# -------------------------------------------------------
# Monkey-patch build_forms to add Boltzmann background
# -------------------------------------------------------
import firedrake as fd
from Forward.bv_solver import forms as _bv_forms_module

_original_build_forms = _bv_forms_module.build_forms

def _boltzmann_build_forms(ctx, solver_params):
    """Add Boltzmann ClO4- background to Poisson source after standard build."""
    ctx = _original_build_forms(ctx, solver_params)

    scaling = ctx["nondim"]
    mesh = ctx["mesh"]
    W = ctx["W"]
    U = ctx["U"]
    n = ctx["n_species"]

    phi = fd.split(U)[-1]
    w = fd.TestFunctions(W)[-1]
    dx = fd.Measure("dx", domain=mesh)

    charge_rhs = fd.Constant(float(scaling["charge_rhs_prefactor"]))
    c_clO4_bulk = fd.Constant(C_CLO4_HAT)

    # Boltzmann ClO4-: c = c_bulk * exp(-z*phi) = c_bulk * exp(phi) for z=-1
    phi_clipped = fd.min_value(fd.max_value(phi, fd.Constant(-50.0)), fd.Constant(50.0))
    c_clO4_boltzmann = c_clO4_bulk * fd.exp(phi_clipped)

    # z_ClO4 = -1: add -charge_rhs * (-1) * c_boltzmann * w = +charge_rhs * c_boltzmann * w
    ctx["F_res"] -= charge_rhs * fd.Constant(-1.0) * c_clO4_boltzmann * w * dx
    ctx["J_form"] = fd.derivative(ctx["F_res"], U)

    return ctx

_bv_forms_module.build_forms = _boltzmann_build_forms
import Forward.bv_solver as _bv_pkg
_bv_pkg.build_forms = _boltzmann_build_forms


def main():
    import numpy as np
    from Forward.params import SolverParams
    from Forward.bv_solver.robust_forward import solve_curve_robust, populate_ic_cache_robust
    from Forward.steady_state import SteadyStateConfig
    from FluxCurve import BVFluxCurveInferenceRequest, run_bv_multi_observable_flux_curve_inference
    from FluxCurve.bv_point_solve import set_parallel_pool, close_parallel_pool
    from FluxCurve.bv_parallel import BVPointSolvePool, BVParallelPointConfig
    from FluxCurve.bv_point_solve import (
        _WARMSTART_MAX_STEPS, _SER_GROWTH_CAP, _SER_SHRINK, _SER_DT_MAX_RATIO,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--maxiter", type=int, default=30)
    parser.add_argument("--offset", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    # Voltage grid: onset-focused (where k0 is identifiable)
    V_RHE = np.array([-0.10, 0.00, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50])
    PHI_HAT = np.sort(V_RHE / V_T)[::-1]

    true_k0 = np.array([K0_HAT_R1, K0_HAT_R2])
    true_alpha = np.array([ALPHA_R1, ALPHA_R2])
    off = args.offset
    init_k0 = true_k0 * (1.0 + off)
    init_alpha = true_alpha * (1.0 - off * 0.5)

    observable_scale = -I_SCALE

    def make_3sp(eta_hat=0.0, k0_r1=K0_HAT_R1, k0_r2=K0_HAT_R2,
                 alpha_r1=ALPHA_R1, alpha_r2=ALPHA_R2):
        params = dict(SNES_OPTS_CHARGED)
        params["bv_convergence"] = _make_bv_convergence_cfg()
        params["nondim"] = _make_nondim_cfg()
        r1 = {"k0": k0_r1, "alpha": alpha_r1, "cathodic_species": 0,
               "anodic_species": 1, "c_ref": 1.0, "stoichiometry": [-1, +1, -2],
               "n_electrons": N_ELECTRONS, "reversible": True, "E_eq_v": E_EQ_R1,
               "cathodic_conc_factors": [{"species": 2, "power": 2, "c_ref_nondim": C_HP_HAT}]}
        r2 = {"k0": k0_r2, "alpha": alpha_r2, "cathodic_species": 1,
               "anodic_species": None, "c_ref": 0.0, "stoichiometry": [0, -1, -2],
               "n_electrons": N_ELECTRONS, "reversible": False, "E_eq_v": E_EQ_R2,
               "cathodic_conc_factors": [{"species": 2, "power": 2, "c_ref_nondim": C_HP_HAT}]}
        params["bv_bc"] = {
            "reactions": [r1, r2],
            "k0": [k0_r1]*3, "alpha": [alpha_r1]*3,
            "stoichiometry": [-1, -1, -1], "c_ref": [1.0, 0.0, 1.0],
            "E_eq_v": 0.0, "electrode_marker": 3,
            "concentration_marker": 4, "ground_marker": 4,
        }
        return SolverParams.from_list([
            3, 1, 0.25, 80.0, [0, 0, 1],
            [D_O2_HAT, D_H2O2_HAT, D_HP_HAT],
            [A_DEFAULT]*3, eta_hat,
            [C_O2_HAT, C_H2O2_HAT, C_HP_HAT], 0.0, params,
        ])

    base_output = os.path.join("StudyResults", "v18_3sp_inference")
    os.makedirs(base_output, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"  V18 3-SPECIES BOLTZMANN INFERENCE")
    print(f"{'#'*70}")
    print(f"  True k0:    {true_k0.tolist()}")
    print(f"  True alpha: {true_alpha.tolist()}")
    print(f"  Init k0:    {init_k0.tolist()} ({off*100:.0f}% offset)")
    print(f"  Init alpha: {init_alpha.tolist()}")
    print(f"  V_RHE: [{V_RHE.min():.2f}, {V_RHE.max():.2f}] ({len(V_RHE)} pts)")
    print(f"{'#'*70}\n")

    # Step 1: targets
    print("Step 1: Generating targets...")
    sp_true = make_3sp()
    target_result = solve_curve_robust(
        sp_true, PHI_HAT, observable_scale,
        n_workers=args.workers or None, charge_steps=20,
    )
    target_cd = target_result.cd.copy()
    target_pc = target_result.pc.copy()
    print(f"  {target_result.n_converged}/{target_result.n_total} converged")
    for i in range(len(PHI_HAT)):
        V_i = PHI_HAT[i] * V_T
        print(f"    V≈{V_i:+.2f}: cd={target_cd[i]:.6f}, pc={target_pc[i]:.6f}")

    # Step 2: IC cache
    print(f"\nStep 2: IC cache at initial guess...")
    sp_init = make_3sp(k0_r1=float(init_k0[0]), k0_r2=float(init_k0[1]),
                       alpha_r1=float(init_alpha[0]), alpha_r2=float(init_alpha[1]))
    n_cached, _ = populate_ic_cache_robust(
        sp_init, PHI_HAT, observable_scale,
        n_workers=args.workers or None, charge_steps=20,
    )

    # Step 3: L-BFGS-B inference
    print(f"\nStep 3: L-BFGS-B adjoint inference...")
    dt = 0.25; max_ss = 320
    base_sp = make_3sp()

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
        observable_mode="current_density",
        current_density_scale=observable_scale,
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
        optimizer_options={"maxiter": args.maxiter, "ftol": 1e-12, "gtol": 1e-8, "disp": True},
        max_iters=args.maxiter,
        live_plot=False,
        forward_recovery=recovery,
        parallel_fast_path=True,
        parallel_workers=n_w,
    )

    t0 = time.time()
    result = run_bv_multi_observable_flux_curve_inference(
        req, precomputed_targets={"primary": target_cd, "secondary": target_pc},
    )
    pde_time = time.time() - t0
    close_parallel_pool()

    best_k0 = np.asarray(result["best_k0"])
    best_alpha = np.asarray(result["best_alpha"])
    k0_err = np.abs(best_k0 - true_k0) / true_k0
    alpha_err = np.abs(best_alpha - true_alpha) / true_alpha

    print(f"\n{'#'*70}")
    print(f"  V18 3-SPECIES BOLTZMANN INFERENCE RESULT")
    print(f"{'#'*70}")
    print(f"  k0_1:    {best_k0[0]:.6e}  (true {true_k0[0]:.6e}, err {k0_err[0]*100:.1f}%)")
    print(f"  k0_2:    {best_k0[1]:.6e}  (true {true_k0[1]:.6e}, err {k0_err[1]*100:.1f}%)")
    print(f"  alpha_1: {best_alpha[0]:.4f}  (true {true_alpha[0]:.4f}, err {alpha_err[0]*100:.1f}%)")
    print(f"  alpha_2: {best_alpha[1]:.4f}  (true {true_alpha[1]:.4f}, err {alpha_err[1]*100:.1f}%)")
    print(f"  Max err: {max(k0_err.max(), alpha_err.max())*100:.1f}%")
    print(f"  Loss:    {result['best_loss']:.6e}")
    print(f"  Time:    {pde_time:.1f}s")
    print(f"{'#'*70}")


if __name__ == "__main__":
    main()
