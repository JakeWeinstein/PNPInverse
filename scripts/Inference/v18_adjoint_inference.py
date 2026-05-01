"""V18 Adjoint Inference: Parameter recovery using stabilized z=1 + adjoint gradients.

Strategy:
1. Monkey-patch build_forms to include stabilization (adjoint-compatible)
2. Use the existing FluxCurve pipeline for IC cache + adjoint gradients
3. Voltage grid spans cathodic + onset for full identifiability

The stabilization adds D_art to ClO4- only, preventing co-ion negativity
while maintaining adjoint tape compatibility.
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

OUT_DIR = os.path.join(_ROOT, "StudyResults", "v18_adjoint_inference")
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------------
# Monkey-patch build_forms to include stabilization
# -------------------------------------------------------
from Forward.bv_solver import forms as _bv_forms_module
from Forward.bv_solver.stabilization import add_stabilization

_original_build_forms = _bv_forms_module.build_forms

def _stabilized_build_forms(ctx, solver_params):
    """Wrapper that adds ClO4- stabilization after standard form build."""
    ctx = _original_build_forms(ctx, solver_params)
    ctx = add_stabilization(ctx, solver_params, d_art_scale=0.001, stabilized_species=[3])
    return ctx

# Patch the module so all imports see the stabilized version
_bv_forms_module.build_forms = _stabilized_build_forms

# Also patch the bv_solver package-level import
import Forward.bv_solver as _bv_pkg
_bv_pkg.build_forms = _stabilized_build_forms

# -------------------------------------------------------
# Now import FluxCurve (which uses the patched build_forms)
# -------------------------------------------------------
from Forward.bv_solver.robust_forward import solve_curve_robust, populate_ic_cache_robust
from FluxCurve.bv_config import BVFluxCurveInferenceRequest
from FluxCurve.bv_curve_eval import evaluate_bv_curve_objective_and_gradient
from FluxCurve.bv_point_solve import _clear_caches
from FluxCurve.config import ForwardRecoveryConfig
from Forward.steady_state import SteadyStateConfig

# True parameters
TRUE = {"k0_r1": K0_HAT_R1, "k0_r2": K0_HAT_R2, "alpha_r1": ALPHA_R1, "alpha_r2": ALPHA_R2}

# Voltage grid: cathodic + onset (where both z=1 converge with stabilization)
V_GRID = np.array([-0.20, -0.10, 0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40])
PHI_GRID = V_GRID / V_T


def make_sp(k0_r1, k0_r2, alpha_r1, alpha_r2, eta_hat=0.0):
    """Create solver params at given BV parameters."""
    return make_bv_solver_params(
        eta_hat=eta_hat, dt=0.25, t_end=80.0,
        species=FOUR_SPECIES_CHARGED, snes_opts=SNES_OPTS_CHARGED,
        k0_hat_r1=k0_r1, k0_hat_r2=k0_r2,
        alpha_r1=alpha_r1, alpha_r2=alpha_r2,
        E_eq_r1=E_EQ_R1, E_eq_r2=E_EQ_R2,
    )


def generate_target_data(k0_r1, k0_r2, alpha_r1, alpha_r2):
    """Generate synthetic I-V curve using stabilized robust forward solver."""
    sp = make_sp(k0_r1, k0_r2, alpha_r1, alpha_r2)

    print("Generating target data with stabilized z=1 solver...")
    t0 = time.time()
    result = solve_curve_robust(
        list(sp), PHI_GRID, -I_SCALE,
        n_workers=1,  # sequential for reproducibility
        charge_steps=20,
        max_eta_gap=2.0,
    )
    elapsed = time.time() - t0

    print(f"Generated in {elapsed:.1f}s, {result.n_converged}/{result.n_total} converged")
    for i in range(len(V_GRID)):
        z = result.z_achieved[i]
        print(f"  V={V_GRID[i]:6.2f}: cd={result.cd[i]:10.6f}, z={z:.3f}")

    return result.cd, result.pc, result.z_achieved


def populate_cache(k0_r1, k0_r2, alpha_r1, alpha_r2):
    """Populate IC cache for the adjoint fast path."""
    sp = make_sp(k0_r1, k0_r2, alpha_r1, alpha_r2)
    _clear_caches()

    print("Populating IC cache...")
    t0 = time.time()
    n_cached = populate_ic_cache_robust(
        list(sp), PHI_GRID, n_workers=1, charge_steps=20,
    )
    print(f"IC cache populated: {n_cached} entries in {time.time()-t0:.1f}s")
    return n_cached


def run_inference(target_cd, init_k0_r1, init_k0_r2, init_alpha_r1, init_alpha_r2):
    """Run adjoint-based inference using FluxCurve pipeline."""
    from Forward.bv_solver import make_graded_rectangle_mesh

    sp = make_sp(init_k0_r1, init_k0_r2, init_alpha_r1, init_alpha_r2)

    # Populate IC cache at initial guess
    populate_cache(init_k0_r1, init_k0_r2, init_alpha_r1, init_alpha_r2)

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)

    request = BVFluxCurveInferenceRequest(
        base_solver_params=list(sp),
        steady=SteadyStateConfig(
            max_steps=20,
            consecutive_steps=4,
            relative_tolerance=1e-4,
            absolute_tolerance=1e-8,
        ),
        blob_initial_condition=False,
        fail_penalty=1e4,
        forward_recovery=ForwardRecoveryConfig(max_attempts=0),
        observable_mode="current_density",
        observable_reaction_index=None,
        current_density_scale=-I_SCALE,
        max_eta_gap=0.0,
    )

    k0_vals = np.array([init_k0_r1, init_k0_r2])
    alpha_vals = np.array([init_alpha_r1, init_alpha_r2])

    # Single evaluation: objective + gradient
    print("\nEvaluating objective + adjoint gradient at initial guess...")
    t0 = time.time()
    curve_result = evaluate_bv_curve_objective_and_gradient(
        request=request,
        phi_applied_values=PHI_GRID,
        target_flux=target_cd,
        k0_values=k0_vals,
        alpha_values=alpha_vals,
        mesh=mesh,
        control_mode="joint",
    )
    elapsed = time.time() - t0

    print(f"\nObjective: {curve_result.total_objective:.6e}")
    print(f"Gradient: {curve_result.total_gradient}")
    print(f"Converged points: {curve_result.n_points - curve_result.n_failed}/{curve_result.n_points}")
    print(f"Time: {elapsed:.1f}s")

    # Simple gradient descent iterations
    params = np.array([init_k0_r1, init_k0_r2, init_alpha_r1, init_alpha_r2])
    true_vals = np.array([TRUE["k0_r1"], TRUE["k0_r2"], TRUE["alpha_r1"], TRUE["alpha_r2"]])
    names = list(TRUE.keys())

    best_J = curve_result.total_objective
    best_params = params.copy()

    print(f"\n--- Gradient descent iterations ---")
    for it in range(20):
        grad = curve_result.total_gradient
        if np.linalg.norm(grad) < 1e-12:
            print(f"  iter {it}: gradient ~0, stopping")
            break

        # Normalized gradient descent step
        # Scale learning rate by parameter magnitude
        param_scales = np.abs(params) + 1e-15
        step = grad / param_scales  # normalize by param scale
        lr = 0.1 / (np.linalg.norm(step) + 1e-12)

        new_params = params - lr * grad
        # Clip to bounds
        new_params[0] = max(new_params[0], 1e-8)   # k0_r1
        new_params[1] = max(new_params[1], 1e-10)   # k0_r2
        new_params[2] = np.clip(new_params[2], 0.1, 0.95)  # alpha_r1
        new_params[3] = np.clip(new_params[3], 0.1, 0.95)  # alpha_r2

        # Re-populate cache at new params
        _clear_caches()
        populate_cache(new_params[0], new_params[1], new_params[2], new_params[3])

        # Re-evaluate
        sp_new = make_sp(new_params[0], new_params[1], new_params[2], new_params[3])
        request_new = BVFluxCurveInferenceRequest(
            base_solver_params=list(sp_new),
            steady=request.steady,
            blob_initial_condition=False,
            fail_penalty=1e4,
            forward_recovery=ForwardRecoveryConfig(max_attempts=0),
            observable_mode="current_density",
            observable_reaction_index=None,
            current_density_scale=-I_SCALE,
            max_eta_gap=0.0,
        )

        curve_result = evaluate_bv_curve_objective_and_gradient(
            request=request_new,
            phi_applied_values=PHI_GRID,
            target_flux=target_cd,
            k0_values=np.array([new_params[0], new_params[1]]),
            alpha_values=np.array([new_params[2], new_params[3]]),
            mesh=mesh,
            control_mode="joint",
        )

        J_new = curve_result.total_objective
        errs = np.abs(new_params - true_vals) / np.abs(true_vals) * 100
        print(f"  iter {it}: J={J_new:.4e}, errs=[{errs[0]:.1f}%, {errs[1]:.1f}%, "
              f"{errs[2]:.1f}%, {errs[3]:.1f}%], "
              f"grad_norm={np.linalg.norm(curve_result.total_gradient):.4e}")

        if J_new < best_J:
            best_J = J_new
            best_params = new_params.copy()
        params = new_params

    # Final report
    print(f"\n{'='*60}")
    print(f"INFERENCE RESULTS")
    print(f"{'='*60}")
    print(f"{'Param':>12} {'True':>12} {'Recovered':>12} {'Error':>8}")
    print("-" * 50)
    for i in range(4):
        err = abs(best_params[i] - true_vals[i]) / abs(true_vals[i]) * 100
        print(f"{names[i]:>12} {true_vals[i]:12.6e} {best_params[i]:12.6e} {err:7.1f}%")

    return best_params


def main():
    print("=" * 60)
    print("V18 ADJOINT INFERENCE WITH STABILIZED z=1 SOLVER")
    print("=" * 60)
    print(f"Voltage grid: {len(V_GRID)} points from {V_GRID[0]:.2f}V to {V_GRID[-1]:.2f}V")
    print(f"True: k0_r1={TRUE['k0_r1']:.4e}, k0_r2={TRUE['k0_r2']:.4e}, "
          f"α1={TRUE['alpha_r1']:.4f}, α2={TRUE['alpha_r2']:.4f}")

    # Step 1: Generate target data
    target_cd, target_pc, target_z = generate_target_data(
        TRUE["k0_r1"], TRUE["k0_r2"], TRUE["alpha_r1"], TRUE["alpha_r2"]
    )

    # Step 2: Run inference from 20% offset
    offset = 0.20
    init = [v * (1 + offset) for v in TRUE.values()]
    print(f"\nInitial guess (20% offset): {init}")

    t0 = time.time()
    result = run_inference(target_cd, *init)
    total_time = time.time() - t0
    print(f"\nTotal time: {total_time:.1f}s")

    # Save
    results = {
        "true_params": TRUE,
        "recovered": {k: float(result[i]) for i, k in enumerate(TRUE.keys())},
        "errors_pct": {k: float(abs(result[i] - list(TRUE.values())[i]) / abs(list(TRUE.values())[i]) * 100)
                       for i, k in enumerate(TRUE.keys())},
        "V_grid": V_GRID.tolist(),
        "target_cd": target_cd.tolist(),
        "total_time_s": total_time,
    }
    with open(os.path.join(OUT_DIR, "adjoint_inference_result.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
