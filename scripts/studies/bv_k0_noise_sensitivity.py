"""Noise sensitivity study for BV k0 inference.

Tests the BV k0 inference pipeline at 0%, 2%, 5%, and 10% noise levels
to characterize the identifiability of k0_1 (O2→H2O2) vs k0_2 (H2O2→H2O).

Usage (from PNPInverse/ directory)::

    python scripts/studies/bv_k0_noise_sensitivity.py
"""

from __future__ import annotations

import os
import sys
import csv
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
    run_bv_k0_flux_curve_inference,
)
from Forward.params import SolverParams
from Forward.steady_state import SteadyStateConfig


# ---------------------------------------------------------------------------
# Physical constants and scales (same as Infer_BVk0_from_current_density_curve.py)
# ---------------------------------------------------------------------------

F_CONST = 96485.3329
V_T = 0.025693
N_ELECTRONS = 2

D_O2 = 2.10e-9
C_BULK = 0.5
D_H2O2 = 1.60e-9

K0_PHYS = 2.4e-8
ALPHA_1 = 0.627
K0_2_PHYS = 1e-9
ALPHA_2 = 0.5

L_REF = 1.0e-4
D_REF = D_O2
K_SCALE = D_REF / L_REF

D_O2_HAT = D_O2 / D_REF
D_H2O2_HAT = D_H2O2 / D_REF
K0_HAT = K0_PHYS / K_SCALE
K0_2_HAT = K0_2_PHYS / K_SCALE

A_O2_HAT = 0.01
A_H2O2_HAT = 0.01

I_SCALE = N_ELECTRONS * F_CONST * D_REF * C_BULK / L_REF * 0.1


# ---------------------------------------------------------------------------
# SNES options
# ---------------------------------------------------------------------------

SNES_OPTS = {
    "snes_type":                 "newtonls",
    "snes_max_it":               200,
    "snes_atol":                 1e-7,
    "snes_rtol":                 1e-10,
    "snes_stol":                 1e-12,
    "snes_linesearch_type":      "l2",
    "snes_linesearch_maxlambda": 0.5,
    "snes_divergence_tolerance": 1e12,
    "ksp_type":                  "preonly",
    "pc_type":                   "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_mumps_icntl_8":         77,
    "mat_mumps_icntl_14":        80,
}


# ---------------------------------------------------------------------------
# Build SolverParams
# ---------------------------------------------------------------------------

def _make_bv_solver_params(eta_hat, dt, t_end):
    params = dict(SNES_OPTS)
    params["bv_convergence"] = {
        "clip_exponent":            True,
        "exponent_clip":            50.0,
        "regularize_concentration": True,
        "conc_floor":               1e-12,
        "use_eta_in_bv":            True,
    }
    params["nondim"] = {
        "enabled":                              True,
        "diffusivity_scale_m2_s":               D_O2,
        "concentration_scale_mol_m3":           C_BULK,
        "length_scale_m":                       L_REF,
        "potential_scale_v":                     V_T,
        "kappa_inputs_are_dimensionless":       True,
        "diffusivity_inputs_are_dimensionless": True,
        "concentration_inputs_are_dimensionless": True,
        "potential_inputs_are_dimensionless":   True,
        "time_inputs_are_dimensionless":        True,
    }
    params["bv_bc"] = {
        "reactions": [
            {
                "k0": K0_HAT,
                "alpha": ALPHA_1,
                "cathodic_species": 0,
                "anodic_species": 1,
                "c_ref": 1.0,
                "stoichiometry": [-1, +1],
                "n_electrons": 2,
                "reversible": True,
            },
            {
                "k0": K0_2_HAT,
                "alpha": ALPHA_2,
                "cathodic_species": 1,
                "anodic_species": None,
                "c_ref": 0.0,
                "stoichiometry": [0, -1],
                "n_electrons": 2,
                "reversible": False,
            },
        ],
        "k0": [K0_HAT, K0_2_HAT],
        "alpha": [ALPHA_1, ALPHA_2],
        "stoichiometry": [-1, -1],
        "c_ref": [1.0, 0.0],
        "E_eq_v": 0.0,
        "electrode_marker":      3,
        "concentration_marker":  4,
        "ground_marker":         4,
    }
    return SolverParams.from_list([
        2, 1, dt, t_end,
        [0, 0],
        [D_O2_HAT, D_H2O2_HAT],
        [A_O2_HAT, A_H2O2_HAT],
        eta_hat,
        [1.0, 0.0],
        0.0,
        params,
    ])


# ---------------------------------------------------------------------------
# Study configuration
# ---------------------------------------------------------------------------

NOISE_LEVELS = [0.0, 2.0, 5.0, 10.0]
N_SEEDS_PER_NOISE = 3  # multiple seeds per noise level for statistics


def main():
    eta_values = np.linspace(-1.0, -20.0, 15)
    true_k0 = [K0_HAT, K0_2_HAT]
    initial_guess = [0.01, 0.001]

    dt = 0.5
    max_ss_steps = 60
    t_end = dt * max_ss_steps

    base_sp = _make_bv_solver_params(eta_hat=0.0, dt=dt, t_end=t_end)

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
    study_dir = os.path.join("StudyResults", "bv_k0_noise_sensitivity")
    os.makedirs(study_dir, exist_ok=True)

    all_results = []

    for noise_pct in NOISE_LEVELS:
        n_seeds = 1 if noise_pct == 0.0 else N_SEEDS_PER_NOISE
        for seed_idx in range(n_seeds):
            seed = 20260304 + seed_idx * 1000

            run_label = f"noise{noise_pct:.0f}pct_seed{seed}"
            run_dir = os.path.join(study_dir, run_label)

            print(f"\n{'='*70}")
            print(f"Noise level: {noise_pct}%  Seed: {seed}  ({run_label})")
            print(f"{'='*70}")

            t0 = time.time()

            request = BVFluxCurveInferenceRequest(
                base_solver_params=base_sp,
                steady=steady,
                true_k0=true_k0,
                initial_guess=initial_guess,
                phi_applied_values=eta_values.tolist(),
                target_csv_path=os.path.join(
                    run_dir, "phi_applied_vs_current_density_synthetic.csv",
                ),
                output_dir=run_dir,
                regenerate_target=True,
                target_noise_percent=noise_pct,
                target_seed=seed,
                observable_mode="current_density",
                observable_reaction_index=None,
                current_density_scale=observable_scale,
                observable_label="current density (mA/cm²)",
                observable_title=f"BV k0 inference (noise={noise_pct}%)",
                k0_lower=1e-8,
                k0_upper=100.0,
                log_space=True,
                mesh_Nx=4,
                mesh_Ny=200,
                mesh_beta=3.0,
                optimizer_method="L-BFGS-B",
                optimizer_tolerance=1e-12,
                optimizer_options={
                    "maxiter": 50,
                    "ftol": 1e-12,
                    "gtol": 1e-6,
                    "disp": True,
                },
                max_iters=50,
                gtol=1e-6,
                fail_penalty=1e9,
                print_point_gradients=False,
                blob_initial_condition=False,
                live_plot=False,
                forward_recovery=ForwardRecoveryConfig(
                    max_attempts=6,
                    max_it_only_attempts=2,
                    anisotropy_only_attempts=1,
                    tolerance_relax_attempts=2,
                    max_it_growth=1.5,
                    max_it_cap=500,
                    atol_relax_factor=10.0,
                    rtol_relax_factor=10.0,
                    ksp_rtol_relax_factor=10.0,
                    line_search_schedule=("bt", "l2", "cp", "basic"),
                    anisotropy_target_ratio=3.0,
                    anisotropy_blend=0.5,
                ),
            )

            result = run_bv_k0_flux_curve_inference(request)
            elapsed = time.time() - t0

            best_k0 = np.asarray(result["best_k0"], dtype=float)
            true_k0_arr = np.asarray(true_k0, dtype=float)
            rel_err = np.abs(best_k0 - true_k0_arr) / np.maximum(np.abs(true_k0_arr), 1e-16)

            best_k0_phys = best_k0 * K_SCALE
            true_k0_phys = true_k0_arr * K_SCALE

            row = {
                "noise_percent": noise_pct,
                "seed": seed,
                "k0_1_true": true_k0_arr[0],
                "k0_2_true": true_k0_arr[1],
                "k0_1_est": best_k0[0],
                "k0_2_est": best_k0[1],
                "k0_1_true_phys": true_k0_phys[0],
                "k0_2_true_phys": true_k0_phys[1],
                "k0_1_est_phys": best_k0_phys[0],
                "k0_2_est_phys": best_k0_phys[1],
                "rel_err_k0_1": rel_err[0],
                "rel_err_k0_2": rel_err[1],
                "final_loss": result["best_loss"],
                "optimizer_success": result["optimization_success"],
                "elapsed_seconds": elapsed,
            }
            all_results.append(row)

            print(f"\n--- Result: noise={noise_pct}% seed={seed} ---")
            print(f"  k0_1: true={true_k0_phys[0]:.4e}  est={best_k0_phys[0]:.4e}  err={rel_err[0]:.4f}")
            print(f"  k0_2: true={true_k0_phys[1]:.4e}  est={best_k0_phys[1]:.4e}  err={rel_err[1]:.4f}")
            print(f"  loss={result['best_loss']:.6e}  time={elapsed:.1f}s")

    # Write summary CSV
    summary_path = os.path.join(study_dir, "noise_sensitivity_summary.csv")
    if all_results:
        keys = list(all_results[0].keys())
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in all_results:
                writer.writerow(row)
    print(f"\nSaved summary: {summary_path}")

    # Print summary table
    print("\n" + "=" * 90)
    print("NOISE SENSITIVITY SUMMARY")
    print("=" * 90)
    print(f"{'Noise %':>8} {'Seed':>10} {'k0_1 err':>10} {'k0_2 err':>10} {'Loss':>14} {'Time (s)':>10}")
    print("-" * 90)
    for row in all_results:
        print(
            f"{row['noise_percent']:8.1f} "
            f"{row['seed']:10d} "
            f"{row['rel_err_k0_1']:10.4f} "
            f"{row['rel_err_k0_2']:10.4f} "
            f"{row['final_loss']:14.6e} "
            f"{row['elapsed_seconds']:10.1f}"
        )
    print("-" * 90)

    # Aggregate by noise level
    print("\nAGGREGATE (mean ± std over seeds):")
    print(f"{'Noise %':>8} {'k0_1 err mean':>14} {'k0_1 err std':>14} {'k0_2 err mean':>14} {'k0_2 err std':>14}")
    print("-" * 80)
    for noise_pct in NOISE_LEVELS:
        rows_at_noise = [r for r in all_results if r["noise_percent"] == noise_pct]
        e1 = [r["rel_err_k0_1"] for r in rows_at_noise]
        e2 = [r["rel_err_k0_2"] for r in rows_at_noise]
        print(
            f"{noise_pct:8.1f} "
            f"{np.mean(e1):14.6f} "
            f"{np.std(e1):14.6f} "
            f"{np.mean(e2):14.6f} "
            f"{np.std(e2):14.6f}"
        )

    # Generate summary plot if matplotlib is available
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for i, (param_name, err_key) in enumerate([
            ("k0_1 (O₂→H₂O₂)", "rel_err_k0_1"),
            ("k0_2 (H₂O₂→H₂O)", "rel_err_k0_2"),
        ]):
            ax = axes[i]
            for noise_pct in NOISE_LEVELS:
                rows_at_noise = [r for r in all_results if r["noise_percent"] == noise_pct]
                errs = [r[err_key] for r in rows_at_noise]
                seeds = [r["seed"] for r in rows_at_noise]
                ax.scatter(
                    [noise_pct] * len(errs), errs,
                    marker="o", s=40, zorder=5,
                )
            # Mean line
            noise_arr = sorted(set(NOISE_LEVELS))
            means = []
            for n in noise_arr:
                rows_at_n = [r for r in all_results if r["noise_percent"] == n]
                means.append(np.mean([r[err_key] for r in rows_at_n]))
            ax.plot(noise_arr, means, "k--", linewidth=1.5, label="mean")

            ax.set_xlabel("Noise (%)")
            ax.set_ylabel("Relative Error")
            ax.set_title(f"Identifiability of {param_name}")
            ax.grid(True, alpha=0.25)
            ax.legend()

        plt.tight_layout()
        plot_path = os.path.join(study_dir, "noise_sensitivity_plot.png")
        plt.savefig(plot_path, dpi=160)
        plt.close()
        print(f"Saved plot: {plot_path}")
    except Exception as e:
        print(f"Could not generate plot: {e}")


if __name__ == "__main__":
    main()
