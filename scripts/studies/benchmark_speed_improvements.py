"""Benchmark speed improvements for BV joint inference pipeline.

Runs the 10-point charged joint inference in 4 configurations:
  1. Baseline (no improvements)
  2. Multi-fidelity only
  3. Multi-fidelity + cached warm-start
  4. Multi-fidelity + cached warm-start + Gauss-Newton

Reports wall time, iterations, and final parameter errors for each.
Saves results to StudyResults/speed_benchmark/.

Usage (from PNPInverse/ directory)::

    python scripts/studies/benchmark_speed_improvements.py
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PNPINVERSE_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _PNPINVERSE_ROOT not in sys.path:
    sys.path.insert(0, _PNPINVERSE_ROOT)

os.environ.setdefault("FIREDRAKE_TSFC_KERNEL_CACHE_DIR", "/tmp/firedrake-tsfc")
os.environ.setdefault("PYOP2_CACHE_DIR", "/tmp/pyop2")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from FluxCurve import (
    BVFluxCurveInferenceRequest,
    ForwardRecoveryConfig,
    run_bv_joint_flux_curve_inference,
    _clear_caches,
)
from Forward.params import SolverParams
from Forward.steady_state import SteadyStateConfig


# ---------------------------------------------------------------------------
# Physical constants and scales (same as Infer_BVJoint_charged)
# ---------------------------------------------------------------------------

F_CONST = 96485.3329
R_GAS = 8.31446
T_REF = 298.15
V_T = R_GAS * T_REF / F_CONST
N_ELECTRONS = 2

D_O2 = 1.9e-9
C_O2 = 0.5
D_H2O2 = 1.6e-9
C_H2O2 = 0.0
D_HP = 9.311e-9
C_HP = 0.1
D_CLO4 = 1.792e-9
C_CLO4 = 0.1

K0_PHYS = 2.4e-8
ALPHA_1 = 0.627
K0_2_PHYS = 1e-9
ALPHA_2 = 0.5

L_REF = 1.0e-4
D_REF = D_O2
C_SCALE = C_O2
K_SCALE = D_REF / L_REF

D_O2_HAT = D_O2 / D_REF
D_H2O2_HAT = D_H2O2 / D_REF
D_HP_HAT = D_HP / D_REF
D_CLO4_HAT = D_CLO4 / D_REF

C_O2_HAT = C_O2 / C_SCALE
C_H2O2_HAT = C_H2O2 / C_SCALE
C_HP_HAT = C_HP / C_SCALE
C_CLO4_HAT = C_CLO4 / C_SCALE

K0_HAT = K0_PHYS / K_SCALE
K0_2_HAT = K0_2_PHYS / K_SCALE

I_SCALE = N_ELECTRONS * F_CONST * D_REF * C_SCALE / L_REF * 0.1


SNES_OPTS = {
    "snes_type": "newtonls",
    "snes_max_it": 300,
    "snes_atol": 1e-7,
    "snes_rtol": 1e-10,
    "snes_stol": 1e-12,
    "snes_linesearch_type": "l2",
    "snes_linesearch_maxlambda": 0.5,
    "snes_divergence_tolerance": 1e12,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_mumps_icntl_8": 77,
    "mat_mumps_icntl_14": 80,
}


def _make_bv_solver_params(eta_hat, dt, t_end):
    params = dict(SNES_OPTS)
    params["bv_convergence"] = {
        "clip_exponent": True,
        "exponent_clip": 50.0,
        "regularize_concentration": True,
        "conc_floor": 1e-12,
        "use_eta_in_bv": True,
    }
    params["nondim"] = {
        "enabled": True,
        "diffusivity_scale_m2_s": D_REF,
        "concentration_scale_mol_m3": C_SCALE,
        "length_scale_m": L_REF,
        "potential_scale_v": V_T,
        "kappa_inputs_are_dimensionless": True,
        "diffusivity_inputs_are_dimensionless": True,
        "concentration_inputs_are_dimensionless": True,
        "potential_inputs_are_dimensionless": True,
        "time_inputs_are_dimensionless": True,
    }
    params["bv_bc"] = {
        "reactions": [
            {
                "k0": K0_HAT,
                "alpha": ALPHA_1,
                "cathodic_species": 0,
                "anodic_species": 1,
                "c_ref": 1.0,
                "stoichiometry": [-1, +1, -2, 0],
                "n_electrons": 2,
                "reversible": True,
                "cathodic_conc_factors": [
                    {"species": 2, "power": 2, "c_ref_nondim": C_HP_HAT},
                ],
            },
            {
                "k0": K0_2_HAT,
                "alpha": ALPHA_2,
                "cathodic_species": 1,
                "anodic_species": None,
                "c_ref": 0.0,
                "stoichiometry": [0, -1, -2, 0],
                "n_electrons": 2,
                "reversible": False,
                "cathodic_conc_factors": [
                    {"species": 2, "power": 2, "c_ref_nondim": C_HP_HAT},
                ],
            },
        ],
        "k0": [K0_HAT] * 4,
        "alpha": [ALPHA_1] * 4,
        "stoichiometry": [-1, +1, -2, 0],
        "c_ref": [1.0] * 4,
        "E_eq_v": 0.0,
        "electrode_marker": 3,
        "concentration_marker": 4,
        "ground_marker": 4,
    }
    return SolverParams.from_list([
        4,
        1,
        dt,
        dt * 100,
        [0, 0, 1, -1],
        [D_O2_HAT, D_H2O2_HAT, D_HP_HAT, D_CLO4_HAT],
        [0.01, 0.01, 0.01, 0.01],
        eta_hat,
        [C_O2_HAT, C_H2O2_HAT, C_HP_HAT, C_CLO4_HAT],
        0.0,
        params,
    ])


def _build_base_request(output_dir, max_iters=10):
    """Build a baseline BVFluxCurveInferenceRequest for benchmarking."""
    eta_values = np.linspace(-1.0, -10.0, 10)

    true_k0 = [K0_HAT, K0_2_HAT]
    true_alpha = [ALPHA_1, ALPHA_2]
    initial_k0_guess = [0.005, 0.0005]
    initial_alpha_guess = [0.4, 0.3]

    dt = 0.5
    max_ss_steps = 100
    base_sp = _make_bv_solver_params(eta_hat=0.0, dt=dt, t_end=dt * max_ss_steps)

    steady = SteadyStateConfig(
        relative_tolerance=1e-4,
        absolute_tolerance=1e-8,
        consecutive_steps=4,
        max_steps=max_ss_steps,
        flux_observable="total_species",
        verbose=False,
        print_every=10,
    )

    observable_scale = -I_SCALE

    request = BVFluxCurveInferenceRequest(
        base_solver_params=base_sp,
        steady=steady,
        true_k0=true_k0,
        initial_guess=initial_k0_guess,
        phi_applied_values=eta_values.tolist(),
        target_csv_path=os.path.join(output_dir, "target.csv"),
        output_dir=output_dir,
        regenerate_target=False,  # reuse target across configs
        target_noise_percent=2.0,
        target_seed=20260226,
        observable_mode="current_density",
        current_density_scale=observable_scale,
        observable_label="current density (mA/cm2)",
        observable_title="Speed benchmark",
        control_mode="joint",
        k0_lower=1e-8,
        k0_upper=100.0,
        log_space=True,
        true_alpha=true_alpha,
        initial_alpha_guess=initial_alpha_guess,
        alpha_lower=0.05,
        alpha_upper=0.95,
        alpha_log_space=False,
        mesh_Nx=8,
        mesh_Ny=200,
        mesh_beta=3.0,
        optimizer_method="L-BFGS-B",
        optimizer_tolerance=1e-12,
        optimizer_options={
            "maxiter": max_iters,
            "ftol": 1e-12,
            "gtol": 1e-6,
            "disp": True,
        },
        max_iters=max_iters,
        gtol=1e-6,
        fail_penalty=1e9,
        print_point_gradients=True,
        blob_initial_condition=False,
        live_plot=False,
        live_plot_eval_lines=False,
        anisotropy_trigger_failed_points=4,
        anisotropy_trigger_failed_fraction=0.25,
        forward_recovery=ForwardRecoveryConfig(
            max_attempts=6,
            max_it_only_attempts=2,
            anisotropy_only_attempts=1,
            tolerance_relax_attempts=2,
            max_it_growth=1.5,
            max_it_cap=600,
            atol_relax_factor=10.0,
            rtol_relax_factor=10.0,
            ksp_rtol_relax_factor=10.0,
            line_search_schedule=("bt", "l2", "cp", "basic"),
            anisotropy_target_ratio=3.0,
            anisotropy_blend=0.5,
        ),
    )
    return request


def _compute_errors(result, true_k0, true_alpha):
    best_k0 = np.asarray(result["best_k0"], dtype=float)
    best_alpha = np.asarray(result.get("best_alpha", [0.0, 0.0]), dtype=float)
    true_k0_arr = np.asarray(true_k0, dtype=float)
    true_alpha_arr = np.asarray(true_alpha, dtype=float)

    k0_rel_err = np.abs(best_k0 - true_k0_arr) / np.maximum(np.abs(true_k0_arr), 1e-16)
    alpha_rel_err = np.abs(best_alpha - true_alpha_arr) / np.maximum(np.abs(true_alpha_arr), 1e-16)
    return {
        "k0_rel_error": k0_rel_err.tolist(),
        "alpha_rel_error": alpha_rel_err.tolist(),
        "best_k0": best_k0.tolist(),
        "best_alpha": best_alpha.tolist(),
        "best_loss": float(result["best_loss"]),
    }


def main():
    output_base = os.path.join("StudyResults", "speed_benchmark")
    os.makedirs(output_base, exist_ok=True)

    true_k0 = [K0_HAT, K0_2_HAT]
    true_alpha = [ALPHA_1, ALPHA_2]
    max_iters = 10

    configs = {
        "baseline": {
            "multifidelity_enabled": False,
            "use_checkpoint_warmstart": False,
            "optimizer_method": "L-BFGS-B",
        },
        "multifidelity_only": {
            "multifidelity_enabled": True,
            "coarse_mesh_Nx": 4,
            "coarse_mesh_Ny": 100,
            "coarse_max_iters": 3,
            "use_checkpoint_warmstart": False,
            "optimizer_method": "L-BFGS-B",
        },
        "multifidelity_cached": {
            "multifidelity_enabled": True,
            "coarse_mesh_Nx": 4,
            "coarse_mesh_Ny": 100,
            "coarse_max_iters": 3,
            "use_checkpoint_warmstart": True,
            "optimizer_method": "L-BFGS-B",
        },
        "multifidelity_cached_gn": {
            "multifidelity_enabled": True,
            "coarse_mesh_Nx": 4,
            "coarse_mesh_Ny": 100,
            "coarse_max_iters": 3,
            "use_checkpoint_warmstart": True,
            "optimizer_method": "gauss_newton",
        },
    }

    all_results = {}

    # Generate target data first (shared across all configs)
    first_request = _build_base_request(
        os.path.join(output_base, "baseline"), max_iters=max_iters
    )
    first_request.regenerate_target = True

    for name, overrides in configs.items():
        print(f"\n{'='*60}")
        print(f"  Configuration: {name}")
        print(f"{'='*60}\n")

        _clear_caches()

        config_dir = os.path.join(output_base, name)
        request = _build_base_request(config_dir, max_iters=max_iters)

        # Share target CSV from first config
        request.target_csv_path = first_request.target_csv_path
        request.regenerate_target = (name == "baseline")

        # Apply overrides
        for key, val in overrides.items():
            setattr(request, key, val)

        # For gauss_newton, also update optimizer_options
        if overrides.get("optimizer_method") == "gauss_newton":
            request.optimizer_options = {
                "maxiter": max_iters,
                "ftol": 1e-12,
                "gtol": 1e-6,
                "disp": True,
            }

        t0 = time.perf_counter()
        try:
            result = run_bv_joint_flux_curve_inference(request)
            t_elapsed = time.perf_counter() - t0
            errors = _compute_errors(result, true_k0, true_alpha)
            entry = {
                "config": name,
                "wall_time_s": round(t_elapsed, 1),
                "success": result.get("optimization_success", False),
                **errors,
                **{k: v for k, v in overrides.items()},
            }
        except Exception as exc:
            t_elapsed = time.perf_counter() - t0
            entry = {
                "config": name,
                "wall_time_s": round(t_elapsed, 1),
                "success": False,
                "error": str(exc),
                **{k: v for k, v in overrides.items()},
            }

        all_results[name] = entry
        print(f"\n[{name}] Wall time: {t_elapsed:.1f}s")
        print(f"[{name}] Result: {json.dumps(entry, indent=2, default=str)}")

    # Summary
    print(f"\n{'='*60}")
    print("  SPEED BENCHMARK SUMMARY")
    print(f"{'='*60}\n")

    summary_lines = []
    header = f"{'Config':<30} {'Time (s)':>10} {'Loss':>14} {'k0 err':>12} {'alpha err':>12}"
    print(header)
    summary_lines.append(header)
    print("-" * len(header))
    summary_lines.append("-" * len(header))

    baseline_time = all_results.get("baseline", {}).get("wall_time_s", 1.0)

    for name, entry in all_results.items():
        t = entry.get("wall_time_s", float("nan"))
        loss = entry.get("best_loss", float("nan"))
        k0_err = entry.get("k0_rel_error", [float("nan")])
        alpha_err = entry.get("alpha_rel_error", [float("nan")])
        speedup = baseline_time / max(t, 0.01)
        line = (
            f"{name:<30} {t:>10.1f} {loss:>14.6e} "
            f"{max(k0_err):>12.4f} {max(alpha_err):>12.4f} "
            f"({speedup:.2f}x)"
        )
        print(line)
        summary_lines.append(line)

    # Save summary
    summary_path = os.path.join(output_base, "benchmark_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"\nSaved summary: {summary_path}")

    # Save full results JSON
    results_path = os.path.join(output_base, "benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved results: {results_path}")


if __name__ == "__main__":
    main()
