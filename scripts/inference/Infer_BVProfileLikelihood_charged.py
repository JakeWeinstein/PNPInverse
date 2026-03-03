"""Profile likelihood analysis for BV parameter identifiability.

For each parameter theta_i, fixes theta_i at a grid of values and
re-optimizes all other parameters. Plots the minimum objective vs theta_i.

Interpretation:
    - Flat profile  -> structural non-identifiability (many parameter combos
      achieve the same loss)
    - Parabolic profile -> well-identified parameter (unique minimum)
    - Asymmetric/skewed -> partial identifiability

Full 4-species charged system: O2, H2O2, H+, ClO4- (z=[0,0,+1,-1]).
Uses 10-point cathodic range for speed (~12s/eval, 20-point profile ~4 min).

Usage (from PNPInverse/ directory)::

    python scripts/inference/Infer_BVProfileLikelihood_charged.py
"""

from __future__ import annotations

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
    K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2, I_SCALE,
    FOUR_SPECIES_CHARGED,
    SNES_OPTS_CHARGED,
    make_bv_solver_params,
    make_recovery_config,
    print_params_summary,
)
setup_firedrake_env()

import numpy as np

from FluxCurve import (
    BVFluxCurveInferenceRequest,
    run_bv_k0_flux_curve_inference,
    run_bv_alpha_flux_curve_inference,
)
from Forward.params import SolverParams
from Forward.steady_state import SteadyStateConfig

print_params_summary()


# ---------------------------------------------------------------------------
# Profile runners
# ---------------------------------------------------------------------------

def _run_k0_profile_at_fixed_alpha(
    *,
    profile_param_name: str,
    profile_param_index: int,
    grid_values: np.ndarray,
    fixed_alpha: list,
    eta_values: np.ndarray,
    base_sp: SolverParams,
    steady: SteadyStateConfig,
    true_k0: list,
    observable_scale: float,
    output_dir: str,
) -> list:
    """Run k0 inference at each fixed alpha grid point.

    For alpha profiles: fix alpha_i at each grid value, optimize k0.
    Returns list of dicts with grid_value, best_loss, best_k0, etc.
    """
    results = []
    for gi, grid_val in enumerate(grid_values):
        print(f"\n  --- Profile point {gi+1}/{len(grid_values)}: "
              f"{profile_param_name}={grid_val:.4f} ---")

        alpha_fixed = list(fixed_alpha)
        alpha_fixed[profile_param_index] = float(grid_val)

        point_dir = os.path.join(output_dir, f"grid_{gi:03d}")

        request = BVFluxCurveInferenceRequest(
            base_solver_params=base_sp,
            steady=steady,
            true_k0=true_k0,
            initial_guess=[0.005, 0.0005],  # standard initial k0 guess
            phi_applied_values=eta_values.tolist(),
            target_csv_path=os.path.join(point_dir, "target.csv"),
            output_dir=point_dir,
            regenerate_target=True,
            target_noise_percent=2.0,
            target_seed=20260226,
            observable_mode="current_density",
            current_density_scale=observable_scale,
            observable_label="current density (mA/cm2)",
            observable_title=f"Profile: {profile_param_name}={grid_val:.4f}",
            control_mode="k0",
            k0_lower=1e-8, k0_upper=100.0,
            log_space=True,
            true_alpha=alpha_fixed,
            initial_alpha_guess=alpha_fixed,
            mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
            max_eta_gap=3.0,
            optimizer_method="L-BFGS-B",
            optimizer_options={"maxiter": 20, "ftol": 1e-12, "gtol": 1e-6, "disp": True},
            max_iters=20,
            live_plot=False,
            forward_recovery=make_recovery_config(max_it_cap=600),
        )

        result = run_bv_k0_flux_curve_inference(request)
        best_k0 = np.asarray(result["best_k0"])
        best_loss = float(result["best_loss"])

        results.append({
            "grid_value": float(grid_val),
            "best_loss": best_loss,
            "best_k0_1": float(best_k0[0]),
            "best_k0_2": float(best_k0[1]),
        })
        print(f"    loss={best_loss:.6e}, k0={best_k0.tolist()}")

    return results


def _run_alpha_profile_at_fixed_k0(
    *,
    profile_param_name: str,
    profile_param_index: int,
    grid_values: np.ndarray,
    fixed_k0: list,
    eta_values: np.ndarray,
    base_sp: SolverParams,
    steady: SteadyStateConfig,
    true_k0: list,
    true_alpha: list,
    observable_scale: float,
    output_dir: str,
) -> list:
    """Run alpha inference at each fixed k0 grid point.

    For k0 profiles: fix k0_i at each grid value, optimize alpha.
    Returns list of dicts.
    """
    results = []
    for gi, grid_val in enumerate(grid_values):
        print(f"\n  --- Profile point {gi+1}/{len(grid_values)}: "
              f"{profile_param_name}={grid_val:.6e} ---")

        k0_fixed = list(fixed_k0)
        k0_fixed[profile_param_index] = float(grid_val)

        point_dir = os.path.join(output_dir, f"grid_{gi:03d}")

        request = BVFluxCurveInferenceRequest(
            base_solver_params=base_sp,
            steady=steady,
            true_k0=true_k0,
            initial_guess=k0_fixed,
            phi_applied_values=eta_values.tolist(),
            target_csv_path=os.path.join(point_dir, "target.csv"),
            output_dir=point_dir,
            regenerate_target=True,
            target_noise_percent=2.0,
            target_seed=20260226,
            observable_mode="current_density",
            current_density_scale=observable_scale,
            observable_label="current density (mA/cm2)",
            observable_title=f"Profile: {profile_param_name}={grid_val:.6e}",
            control_mode="alpha",
            fixed_k0=k0_fixed,
            true_alpha=true_alpha,
            initial_alpha_guess=[0.4, 0.3],  # standard initial alpha guess
            alpha_lower=0.05, alpha_upper=0.95,
            mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
            max_eta_gap=3.0,
            optimizer_method="L-BFGS-B",
            optimizer_options={"maxiter": 20, "ftol": 1e-12, "gtol": 1e-6, "disp": True},
            max_iters=20,
            live_plot=False,
            forward_recovery=make_recovery_config(max_it_cap=600),
        )

        result = run_bv_alpha_flux_curve_inference(request)
        best_alpha = np.asarray(result["best_alpha"])
        best_loss = float(result["best_loss"])

        results.append({
            "grid_value": float(grid_val),
            "best_loss": best_loss,
            "best_alpha_1": float(best_alpha[0]),
            "best_alpha_2": float(best_alpha[1]),
        })
        print(f"    loss={best_loss:.6e}, alpha={best_alpha.tolist()}")

    return results


def _save_profile_csv(results: list, path: str, param_name: str):
    """Save profile results to CSV."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [param_name, "best_loss"] + [k for k in results[0] if k not in ("grid_value", "best_loss")]
        writer.writerow(header)
        for r in results:
            row = [r["grid_value"], r["best_loss"]]
            row += [r[k] for k in r if k not in ("grid_value", "best_loss")]
            writer.writerow(row)


def _plot_profile(results: list, param_name: str, true_value: float,
                  output_path: str, log_x: bool = False):
    """Plot profile likelihood curve."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print(f"  [skip] matplotlib not available, skipping plot for {param_name}")
        return

    grid_vals = [r["grid_value"] for r in results]
    losses = [r["best_loss"] for r in results]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    if log_x:
        ax.semilogx(grid_vals, losses, "o-", color="tab:blue", markersize=5)
    else:
        ax.plot(grid_vals, losses, "o-", color="tab:blue", markersize=5)
    ax.axvline(true_value, color="tab:red", linestyle="--", linewidth=1.5,
               label=f"true = {true_value:.4g}")
    ax.set_xlabel(param_name)
    ax.set_ylabel("min objective (other params optimized)")
    ax.set_title(f"Profile Likelihood: {param_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  [plot] {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # 10-point cathodic range for speed
    eta_values = np.linspace(-1.0, -10.0, 10)

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

    true_k0 = [K0_HAT_R1, K0_HAT_R2]
    true_alpha = [ALPHA_R1, ALPHA_R2]
    observable_scale = -I_SCALE

    base_output = os.path.join("StudyResults", "bv_profile_likelihood_charged")
    os.makedirs(base_output, exist_ok=True)

    n_grid = 15  # number of grid points per profile

    t_total = time.time()

    all_profiles = {}

    # -------------------------------------------------------------------
    # Profile 1: alpha_1 (fix alpha_1, optimize k0 with alpha_2 at true)
    # -------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  Profile: alpha_1 ({n_grid} grid points)")
    print(f"  Fix alpha_1 at grid, fix k0 at true, optimize remaining")
    print(f"{'='*70}")

    alpha1_grid = np.linspace(0.1, 0.9, n_grid)
    alpha1_dir = os.path.join(base_output, "profile_alpha1")
    os.makedirs(alpha1_dir, exist_ok=True)

    alpha1_results = _run_k0_profile_at_fixed_alpha(
        profile_param_name="alpha_1",
        profile_param_index=0,
        grid_values=alpha1_grid,
        fixed_alpha=true_alpha,
        eta_values=eta_values,
        base_sp=base_sp,
        steady=steady,
        true_k0=true_k0,
        observable_scale=observable_scale,
        output_dir=alpha1_dir,
    )
    _save_profile_csv(alpha1_results, os.path.join(alpha1_dir, "profile.csv"), "alpha_1")
    _plot_profile(alpha1_results, "alpha_1", ALPHA_R1,
                  os.path.join(alpha1_dir, "profile_alpha1.pdf"))
    all_profiles["alpha_1"] = alpha1_results

    # -------------------------------------------------------------------
    # Profile 2: alpha_2 (fix alpha_2, optimize k0 with alpha_1 at true)
    # -------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  Profile: alpha_2 ({n_grid} grid points)")
    print(f"{'='*70}")

    alpha2_grid = np.linspace(0.1, 0.9, n_grid)
    alpha2_dir = os.path.join(base_output, "profile_alpha2")
    os.makedirs(alpha2_dir, exist_ok=True)

    alpha2_results = _run_k0_profile_at_fixed_alpha(
        profile_param_name="alpha_2",
        profile_param_index=1,
        grid_values=alpha2_grid,
        fixed_alpha=true_alpha,
        eta_values=eta_values,
        base_sp=base_sp,
        steady=steady,
        true_k0=true_k0,
        observable_scale=observable_scale,
        output_dir=alpha2_dir,
    )
    _save_profile_csv(alpha2_results, os.path.join(alpha2_dir, "profile.csv"), "alpha_2")
    _plot_profile(alpha2_results, "alpha_2", ALPHA_R2,
                  os.path.join(alpha2_dir, "profile_alpha2.pdf"))
    all_profiles["alpha_2"] = alpha2_results

    # -------------------------------------------------------------------
    # Profile 3: k0_1 (fix k0_1, optimize alpha with k0_2 at true)
    # -------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  Profile: k0_1 ({n_grid} grid points)")
    print(f"{'='*70}")

    # Log-spaced grid around true k0_1
    k01_grid = np.geomspace(K0_HAT_R1 * 0.1, K0_HAT_R1 * 10.0, n_grid)
    k01_dir = os.path.join(base_output, "profile_k0_1")
    os.makedirs(k01_dir, exist_ok=True)

    k01_results = _run_alpha_profile_at_fixed_k0(
        profile_param_name="k0_1",
        profile_param_index=0,
        grid_values=k01_grid,
        fixed_k0=true_k0,
        eta_values=eta_values,
        base_sp=base_sp,
        steady=steady,
        true_k0=true_k0,
        true_alpha=true_alpha,
        observable_scale=observable_scale,
        output_dir=k01_dir,
    )
    _save_profile_csv(k01_results, os.path.join(k01_dir, "profile.csv"), "k0_1")
    _plot_profile(k01_results, "k0_1", K0_HAT_R1,
                  os.path.join(k01_dir, "profile_k0_1.pdf"), log_x=True)
    all_profiles["k0_1"] = k01_results

    # -------------------------------------------------------------------
    # Profile 4: k0_2 (fix k0_2, optimize alpha with k0_1 at true)
    # -------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  Profile: k0_2 ({n_grid} grid points)")
    print(f"{'='*70}")

    k02_grid = np.geomspace(K0_HAT_R2 * 0.01, K0_HAT_R2 * 100.0, n_grid)
    k02_dir = os.path.join(base_output, "profile_k0_2")
    os.makedirs(k02_dir, exist_ok=True)

    k02_results = _run_alpha_profile_at_fixed_k0(
        profile_param_name="k0_2",
        profile_param_index=1,
        grid_values=k02_grid,
        fixed_k0=true_k0,
        eta_values=eta_values,
        base_sp=base_sp,
        steady=steady,
        true_k0=true_k0,
        true_alpha=true_alpha,
        observable_scale=observable_scale,
        output_dir=k02_dir,
    )
    _save_profile_csv(k02_results, os.path.join(k02_dir, "profile.csv"), "k0_2")
    _plot_profile(k02_results, "k0_2", K0_HAT_R2,
                  os.path.join(k02_dir, "profile_k0_2.pdf"), log_x=True)
    all_profiles["k0_2"] = k02_results

    total_time = time.time() - t_total

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  PROFILE LIKELIHOOD SUMMARY")
    print(f"{'='*70}")

    for pname, pres in all_profiles.items():
        losses = [r["best_loss"] for r in pres]
        grid_vals = [r["grid_value"] for r in pres]
        min_loss = min(losses)
        min_idx = losses.index(min_loss)
        min_val = grid_vals[min_idx]
        loss_range = max(losses) - min_loss
        # Identify profile curvature near minimum
        if min_idx > 0 and min_idx < len(losses) - 1:
            curvature = (losses[min_idx-1] + losses[min_idx+1] - 2*min_loss)
        else:
            curvature = float("nan")

        if pname.startswith("k0"):
            true_val_str = f"{[K0_HAT_R1, K0_HAT_R2][int(pname[-1])-1]:.6e}"
            min_val_str = f"{min_val:.6e}"
        else:
            true_val_str = f"{[ALPHA_R1, ALPHA_R2][int(pname[-1])-1]:.4f}"
            min_val_str = f"{min_val:.4f}"

        flat_indicator = "FLAT" if loss_range < 0.1 * min_loss else "CURVED"
        print(f"  {pname:<10}: min_loss={min_loss:.6e} at {pname}={min_val_str} "
              f"(true={true_val_str}), range={loss_range:.2e}, curvature={curvature:.2e} "
              f"-> {flat_indicator}")

    print(f"\n  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Output: {base_output}/")

    # Save master CSV
    master_csv = os.path.join(base_output, "profile_summary.csv")
    with open(master_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter", "grid_value", "best_loss"])
        for pname, pres in all_profiles.items():
            for r in pres:
                writer.writerow([pname, r["grid_value"], r["best_loss"]])
    print(f"  [csv] Master summary -> {master_csv}")

    # -------------------------------------------------------------------
    # Combined plot (2x2 grid)
    # -------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        param_specs = [
            ("alpha_1", ALPHA_R1, False, axes[0, 0]),
            ("alpha_2", ALPHA_R2, False, axes[0, 1]),
            ("k0_1", K0_HAT_R1, True, axes[1, 0]),
            ("k0_2", K0_HAT_R2, True, axes[1, 1]),
        ]
        for pname, true_val, log_x, ax in param_specs:
            pres = all_profiles[pname]
            gv = [r["grid_value"] for r in pres]
            lo = [r["best_loss"] for r in pres]
            if log_x:
                ax.semilogx(gv, lo, "o-", markersize=4)
            else:
                ax.plot(gv, lo, "o-", markersize=4)
            ax.axvline(true_val, color="tab:red", ls="--", lw=1.5,
                       label=f"true={true_val:.4g}")
            ax.set_xlabel(pname)
            ax.set_ylabel("min J")
            ax.set_title(f"Profile: {pname}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle("Profile Likelihood -- BV Parameter Identifiability", fontsize=13)
        fig.tight_layout()
        combined_path = os.path.join(base_output, "profile_likelihood_combined.pdf")
        fig.savefig(combined_path, dpi=150)
        plt.close(fig)
        print(f"  [plot] Combined -> {combined_path}")
    except Exception as e:
        print(f"  [skip] Combined plot failed: {e}")

    print(f"\n=== Profile Likelihood Complete ===")


if __name__ == "__main__":
    main()
