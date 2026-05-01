"""V17 inference with Nelder-Mead (derivative-free) optimizer.

L-BFGS-B fails because adjoint gradients require the solver to converge at
perturbed parameters. Nelder-Mead only needs function evaluations (no gradients)
and can tolerate occasional solver failures.

Strategy:
- Use charge continuation for EACH evaluation (expensive but robust)
- Optimize in log-space for k0, linear for alpha
- Use the full reliable voltage range [-0.5V, +0.1V]
- Penalty for unconverged points is finite but large

Usage:
    python scripts/Inference/v17_nelder_mead.py
    python scripts/Inference/v17_nelder_mead.py --noise 2.0 --maxiter 100
"""
from __future__ import annotations
import os, sys, time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(line_buffering=True)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts._bv_common import (
    setup_firedrake_env, K0_HAT_R1, K0_HAT_R2, I_SCALE, V_T, K_SCALE,
    ALPHA_R1, ALPHA_R2, FOUR_SPECIES_CHARGED, make_bv_solver_params,
)
setup_firedrake_env()

import numpy as np
from scipy.optimize import minimize

E_EQ_R1, E_EQ_R2 = 0.68, 1.78

# Voltage grid
V_RHE = np.array([
    -0.50, -0.40, -0.30, -0.20, -0.15, -0.10, -0.05, 0.00, 0.05, 0.10,
])
PHI_HAT = np.sort(V_RHE / V_T)[::-1]

true_k0 = np.array([K0_HAT_R1, K0_HAT_R2])
true_alpha = np.array([ALPHA_R1, ALPHA_R2])
observable_scale = -I_SCALE

# Tuned SNES
SNES_OPTS = {
    "snes_type": "newtonls", "snes_max_it": 400,
    "snes_atol": 1e-7, "snes_rtol": 1e-10, "snes_stol": 1e-12,
    "snes_linesearch_type": "l2", "snes_linesearch_maxlambda": 0.3,
    "snes_divergence_tolerance": 1e10,
    "ksp_type": "preonly", "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_mumps_icntl_8": 77, "mat_mumps_icntl_14": 80,
}


def forward_solve(k0_r1, k0_r2, alpha_r1, alpha_r2):
    """Solve forward problem using charge continuation. Returns (cd, pc) arrays."""
    import firedrake as fd
    import pyadjoint as adj
    from Forward.bv_solver import make_graded_rectangle_mesh, solve_grid_with_charge_continuation
    from Forward.bv_solver.observables import _build_bv_observable_form

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)
    sp = make_bv_solver_params(
        eta_hat=0.0, dt=0.25, t_end=80.0,
        species=FOUR_SPECIES_CHARGED, snes_opts=SNES_OPTS,
        k0_hat_r1=k0_r1, k0_hat_r2=k0_r2,
        alpha_r1=alpha_r1, alpha_r2=alpha_r2,
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
        result = solve_grid_with_charge_continuation(
            sp, phi_applied_values=PHI_HAT,
            charge_steps=15, mesh=mesh,
            max_eta_gap=2.0, min_delta_z=0.005,
            per_point_callback=_extract,
        )

    n_full = sum(1 for pt in result.points.values() if pt.converged)
    return cd, pc, n_full


def objective(x, target_cd, target_pc, eval_count):
    """Evaluate objective: MSE between model and target."""
    log_k0_1, log_k0_2, alpha_1, alpha_2 = x
    k0_1 = np.exp(log_k0_1)
    k0_2 = np.exp(log_k0_2)

    # Bounds check
    if alpha_1 < 0.05 or alpha_1 > 0.95 or alpha_2 < 0.05 or alpha_2 > 0.95:
        return 1e6
    if k0_1 < 1e-8 or k0_1 > 1e2 or k0_2 < 1e-10 or k0_2 > 1e0:
        return 1e6

    eval_count[0] += 1
    n_eval = eval_count[0]

    t0 = time.time()
    try:
        cd, pc, n_full = forward_solve(k0_1, k0_2, alpha_1, alpha_2)
    except Exception as exc:
        print(f"  [eval {n_eval:3d}] EXCEPTION: {exc}")
        return 1e6

    elapsed = time.time() - t0

    # Compute MSE on converged points
    valid = ~np.isnan(cd) & ~np.isnan(target_cd)
    if valid.sum() < 3:
        print(f"  [eval {n_eval:3d}] Too few converged points ({valid.sum()}), penalty")
        return 1e5

    mse_cd = np.mean((cd[valid] - target_cd[valid])**2)
    mse_pc = np.mean((pc[valid] - target_pc[valid])**2)
    loss = mse_cd + mse_pc

    # Penalty for unconverged points
    n_nan = (~valid).sum()
    loss += n_nan * 0.01  # Mild penalty

    k0_err = np.abs(np.array([k0_1, k0_2]) - true_k0) / true_k0
    a_err = np.abs(np.array([alpha_1, alpha_2]) - true_alpha) / true_alpha

    print(f"  [eval {n_eval:3d}] J={loss:.6e} "
          f"k0=[{k0_1:.3e},{k0_2:.3e}] a=[{alpha_1:.3f},{alpha_2:.3f}] "
          f"({n_full}/{len(PHI_HAT)} full-z) "
          f"k0_err=[{k0_err[0]*100:.1f}%,{k0_err[1]*100:.1f}%] "
          f"({elapsed:.0f}s)", flush=True)

    return loss


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--maxiter", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  V17 NELDER-MEAD INFERENCE")
    print(f"  {len(PHI_HAT)} voltage points, V_RHE [{V_RHE.min():.2f}, {V_RHE.max():.2f}]")
    print(f"  E_eq = ({E_EQ_R1}, {E_EQ_R2}) V")
    print(f"  Noise: {args.noise}%")
    print(f"{'='*70}\n")

    # Generate targets
    print("Generating targets at true parameters...")
    target_cd, target_pc, n_full = forward_solve(
        K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2)
    print(f"  {n_full}/{len(PHI_HAT)} full-z")
    print(f"  cd range: [{np.nanmin(target_cd):.4f}, {np.nanmax(target_cd):.4f}]")
    print(f"  pc range: [{np.nanmin(target_pc):.4f}, {np.nanmax(target_pc):.4f}]")

    if args.noise > 0:
        from Forward.steady_state import add_percent_noise
        target_cd = add_percent_noise(target_cd, args.noise, seed=args.seed)
        target_pc = add_percent_noise(target_pc, args.noise, seed=args.seed + 1)

    # Initial guess (2x offset)
    x0 = np.array([
        np.log(K0_HAT_R1 * 2),     # log(k0_1)
        np.log(K0_HAT_R2 * 0.5),   # log(k0_2)
        0.5,                         # alpha_1
        0.4,                         # alpha_2
    ])

    print(f"\nInitial guess:")
    print(f"  k0_1 = {np.exp(x0[0]):.6e} (true {K0_HAT_R1:.6e})")
    print(f"  k0_2 = {np.exp(x0[1]):.6e} (true {K0_HAT_R2:.6e})")
    print(f"  alpha_1 = {x0[2]:.4f} (true {ALPHA_R1:.4f})")
    print(f"  alpha_2 = {x0[3]:.4f} (true {ALPHA_R2:.4f})")

    eval_count = [0]
    t_start = time.time()

    result = minimize(
        objective,
        x0,
        args=(target_cd, target_pc, eval_count),
        method="Nelder-Mead",
        options={
            "maxiter": args.maxiter,
            "maxfev": args.maxiter * 3,
            "xatol": 1e-4,
            "fatol": 1e-8,
            "adaptive": True,
            "disp": True,
        },
    )

    elapsed = time.time() - t_start
    best_k0 = np.array([np.exp(result.x[0]), np.exp(result.x[1])])
    best_alpha = np.array([result.x[2], result.x[3]])
    k0_err = np.abs(best_k0 - true_k0) / true_k0
    alpha_err = np.abs(best_alpha - true_alpha) / true_alpha

    print(f"\n{'#'*70}")
    print(f"  NELDER-MEAD RESULT ({eval_count[0]} evals, {elapsed:.0f}s)")
    print(f"{'#'*70}")
    print(f"  k0_1:    {best_k0[0]:.6e}  (true {true_k0[0]:.6e}, err {k0_err[0]*100:.1f}%)")
    print(f"  k0_2:    {best_k0[1]:.6e}  (true {true_k0[1]:.6e}, err {k0_err[1]*100:.1f}%)")
    print(f"  alpha_1: {best_alpha[0]:.4f}  (true {true_alpha[0]:.4f}, err {alpha_err[0]*100:.1f}%)")
    print(f"  alpha_2: {best_alpha[1]:.4f}  (true {true_alpha[1]:.4f}, err {alpha_err[1]*100:.1f}%)")
    print(f"  Loss:    {result.fun:.6e}")
    print(f"  Max err: {max(k0_err.max(), alpha_err.max())*100:.1f}%")
    print(f"{'#'*70}")


if __name__ == "__main__":
    main()
