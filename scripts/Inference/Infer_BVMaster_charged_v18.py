"""V18 Inference: Parameter recovery using stabilized z=1 forward solver.

Key improvements over v17:
- Uses ClO4-only artificial diffusion (d_art=0.001) for z=1 convergence
- Covers onset region (0.1V - 0.7V) where alpha AND k0 are identifiable
- Full z=1 physics at all voltage points (no z=0 cheating)

Test: generate synthetic data at true params, recover from 20% offset.
"""
import sys, os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts._bv_common import (
    setup_firedrake_env, V_T, I_SCALE,
    FOUR_SPECIES_CHARGED, make_bv_solver_params,
    SNES_OPTS_CHARGED,
    K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2,
)
setup_firedrake_env()

import numpy as np
import time
import json

E_EQ_R1 = 0.68
E_EQ_R2 = 1.78

OUT_DIR = os.path.join(_ROOT, "StudyResults", "v18_inference")
os.makedirs(OUT_DIR, exist_ok=True)

# True parameters (nondimensional)
TRUE_PARAMS = {
    "k0_r1": K0_HAT_R1,
    "k0_r2": K0_HAT_R2,
    "alpha_r1": ALPHA_R1,
    "alpha_r2": ALPHA_R2,
}

# Voltage grid covering onset + cathodic
# Focus on onset region where k0/alpha are most identifiable
V_GRID = np.array([
    -0.20, -0.10, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25,
    0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65
])
PHI_GRID = V_GRID / V_T


def forward_solve(k0_r1, k0_r2, alpha_r1, alpha_r2, verbose=False):
    """Stabilized forward solve at the given parameters."""
    from Forward.bv_solver.stabilized_forward import solve_stabilized_curve

    sp = make_bv_solver_params(
        eta_hat=0.0,  # overridden per point
        dt=0.25, t_end=80.0,
        species=FOUR_SPECIES_CHARGED,
        snes_opts=SNES_OPTS_CHARGED,
        k0_hat_r1=k0_r1, k0_hat_r2=k0_r2,
        alpha_r1=alpha_r1, alpha_r2=alpha_r2,
        E_eq_r1=E_EQ_R1, E_eq_r2=E_EQ_R2,
    )

    result = solve_stabilized_curve(
        list(sp), PHI_GRID, -I_SCALE,
        d_art_scale=0.001,
        stabilized_species=[3],  # ClO4 only
        z_steps=20,
        verbose=verbose,
    )

    return result["cd"], result["pc"], result["z_achieved"]


def objective(params, target_cd, target_pc, weights=None):
    """Least-squares objective: J = Σ w_i * (cd_model - cd_target)² + (pc_model - pc_target)²"""
    k0_r1, k0_r2, alpha_r1, alpha_r2 = params

    # Bounds check
    if k0_r1 <= 0 or k0_r2 <= 0 or alpha_r1 <= 0.1 or alpha_r1 >= 0.95 or alpha_r2 <= 0.1 or alpha_r2 >= 0.95:
        return 1e10

    try:
        cd, pc, z_ach = forward_solve(k0_r1, k0_r2, alpha_r1, alpha_r2, verbose=False)
    except Exception as e:
        print(f"  Forward solve failed: {e}")
        return 1e8

    # Check convergence
    n_failed = np.sum(z_ach < 0.999)
    if n_failed > len(z_ach) * 0.3:
        print(f"  Too many failures: {n_failed}/{len(z_ach)}")
        return 1e7

    if weights is None:
        weights = np.ones(len(target_cd))

    # Only use points where model converged
    mask = z_ach >= 0.999
    residual_cd = (cd[mask] - target_cd[mask]) * weights[mask]
    residual_pc = (pc[mask] - target_pc[mask]) * weights[mask]

    J = float(np.sum(residual_cd**2) + np.sum(residual_pc**2))
    return J


def main():
    print("=" * 70)
    print("V18 INFERENCE: Stabilized z=1 Forward Solver")
    print("=" * 70)
    print(f"\nTrue parameters:")
    for k, v in TRUE_PARAMS.items():
        print(f"  {k}: {v:.6e}")
    print(f"\nVoltage grid: {len(V_GRID)} points from {V_GRID[0]:.2f}V to {V_GRID[-1]:.2f}V")

    # Step 1: Generate synthetic data
    print("\n--- Step 1: Generating synthetic data ---")
    t0 = time.time()
    target_cd, target_pc, target_z = forward_solve(
        TRUE_PARAMS["k0_r1"], TRUE_PARAMS["k0_r2"],
        TRUE_PARAMS["alpha_r1"], TRUE_PARAMS["alpha_r2"],
        verbose=True,
    )
    print(f"Data generated in {time.time()-t0:.1f}s")
    n_good = np.sum(target_z >= 0.999)
    print(f"Converged points: {n_good}/{len(V_GRID)}")

    # Save synthetic data
    np.savez(os.path.join(OUT_DIR, "synthetic_data.npz"),
             V_grid=V_GRID, phi_grid=PHI_GRID,
             target_cd=target_cd, target_pc=target_pc, target_z=target_z)

    # Print I-V curve
    print(f"\n{'V_RHE':>8} {'cd':>10} {'pc':>10} {'z':>6}")
    for i in range(len(V_GRID)):
        print(f"{V_GRID[i]:8.3f} {target_cd[i]:10.6f} {target_pc[i]:10.6f} {target_z[i]:6.3f}")

    # Step 2: Optimize from offset initial guess
    print("\n--- Step 2: Parameter recovery (20% offset) ---")
    offset = 0.20
    init_params = [
        TRUE_PARAMS["k0_r1"] * (1 + offset),
        TRUE_PARAMS["k0_r2"] * (1 + offset),
        TRUE_PARAMS["alpha_r1"] * (1 + offset),
        TRUE_PARAMS["alpha_r2"] * (1 + offset),
    ]
    print(f"Initial guess (20% offset):")
    for k, v, iv in zip(TRUE_PARAMS.keys(), TRUE_PARAMS.values(), init_params):
        print(f"  {k}: true={v:.6e}, init={iv:.6e}")

    # Use scipy optimizer
    from scipy.optimize import minimize

    eval_count = [0]
    best_loss = [1e10]
    best_params = [init_params[:]]

    def obj_with_log(p):
        eval_count[0] += 1
        J = objective(p, target_cd, target_pc)
        if J < best_loss[0]:
            best_loss[0] = J
            best_params[0] = list(p)
            errs = [(p[i]-list(TRUE_PARAMS.values())[i])/list(TRUE_PARAMS.values())[i]*100
                    for i in range(4)]
            print(f"  eval {eval_count[0]:3d}: J={J:.4e}, "
                  f"errs=[{errs[0]:.1f}%, {errs[1]:.1f}%, {errs[2]:.1f}%, {errs[3]:.1f}%]")
        return J

    # First: evaluate at initial guess
    print(f"\nEval at initial guess...")
    J_init = obj_with_log(init_params)
    print(f"  Initial loss: {J_init:.4e}")

    # Nelder-Mead (derivative-free, robust for this problem)
    print(f"\nStarting Nelder-Mead optimization...")
    t_opt = time.time()
    result = minimize(
        obj_with_log,
        init_params,
        method="Nelder-Mead",
        options={
            "maxiter": 100,
            "maxfev": 100,
            "xatol": 1e-8,
            "fatol": 1e-10,
            "adaptive": True,
        },
    )
    opt_time = time.time() - t_opt

    # Results
    print(f"\n{'='*70}")
    print(f"INFERENCE RESULTS")
    print(f"{'='*70}")
    print(f"Optimizer: {result.message}")
    print(f"Evaluations: {eval_count[0]}, time: {opt_time:.1f}s")
    print(f"Final loss: {best_loss[0]:.4e}")

    opt_params = best_params[0]
    true_vals = list(TRUE_PARAMS.values())
    names = list(TRUE_PARAMS.keys())

    print(f"\n{'Parameter':>12} {'True':>12} {'Recovered':>12} {'Error':>8}")
    print("-" * 50)
    for i in range(4):
        err = abs(opt_params[i] - true_vals[i]) / abs(true_vals[i]) * 100
        print(f"{names[i]:>12} {true_vals[i]:12.6e} {opt_params[i]:12.6e} {err:7.1f}%")

    # Save results
    results = {
        "true_params": TRUE_PARAMS,
        "init_params": {k: iv for k, iv in zip(names, init_params)},
        "recovered_params": {k: v for k, v in zip(names, opt_params)},
        "errors_pct": {k: abs(opt_params[i]-true_vals[i])/abs(true_vals[i])*100
                       for i, k in enumerate(names)},
        "loss_init": float(J_init),
        "loss_final": float(best_loss[0]),
        "n_evals": eval_count[0],
        "time_s": opt_time,
        "V_grid": V_GRID.tolist(),
    }
    with open(os.path.join(OUT_DIR, "inference_result.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {OUT_DIR}/inference_result.json")


if __name__ == "__main__":
    main()
