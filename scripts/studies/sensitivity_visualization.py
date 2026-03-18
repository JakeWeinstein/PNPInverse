"""Parameter sensitivity visualization: 1D sweeps + Jacobian heatmap.

Produces 1D parameter sweeps showing total and peroxide current vs voltage
at multiplicative perturbation factors for each kinetic parameter (k0_1,
k0_2, alpha_1, alpha_2).  Also computes the Jacobian d(observable)/d(param)
via central finite differences at each voltage point, visualized as a heatmap.

Extended voltage range goes beyond the v13 default (-46.5) using warm-starting
and increased SNES iterations for convergence at extreme cathodic voltages.

Output: CSV data + PNG plots + JSON metadata in StudyResults/v14/sensitivity/

Usage (from PNPInverse/ directory)::

    python scripts/studies/sensitivity_visualization.py
    python scripts/studies/sensitivity_visualization.py --params k0_1 alpha_1
    python scripts/studies/sensitivity_visualization.py --voltage-min -75
    python scripts/studies/sensitivity_visualization.py --no-jacobian

Requirements: DIAG-03 (sensitivity analysis), AUDT-04 (tool justification).
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PNPINVERSE_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _PNPINVERSE_ROOT not in sys.path:
    sys.path.insert(0, _PNPINVERSE_ROOT)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SensitivityConfig:
    """Immutable configuration for sensitivity analysis."""

    sweep_factors: tuple = (0.5, 0.75, 1.0, 1.5, 2.0)
    params_to_sweep: tuple = ("k0_1", "k0_2", "alpha_1", "alpha_2")
    jacobian_h: float = 1e-5  # central FD step (codebase convention)
    output_dir: str = "StudyResults/v14/sensitivity"
    extended_voltage_min: float = -60.0
    snes_max_it: int = 400  # more iterations for extreme voltages
    dt: float = 0.25  # smaller dt for convergence at extended voltages
    max_ss_steps: int = 200  # compensate for smaller dt
    verbose: bool = True


# ---------------------------------------------------------------------------
# V13 cathodic voltage grid (reference)
# ---------------------------------------------------------------------------

_V13_CATHODIC = np.array([
    -1.0, -2.0, -3.0, -4.0, -5.0, -6.5, -8.0,
    -10.0, -13.0, -17.0, -22.0, -28.0,
    -35.0, -41.0, -46.5,
])


# ---------------------------------------------------------------------------
# Pure helper functions (no Firedrake dependency)
# ---------------------------------------------------------------------------

def build_sweep_factors(
    factors: Optional[tuple] = None,
) -> List[float]:
    """Return sweep multiplicative factors.

    Parameters
    ----------
    factors : tuple or None
        Custom factors. If None, returns default [0.5, 0.75, 1.0, 1.5, 2.0].

    Returns
    -------
    list[float]
    """
    if factors is not None:
        return list(factors)
    return [0.5, 0.75, 1.0, 1.5, 2.0]


def build_extended_voltage_grid(v_min: float = -60.0) -> np.ndarray:
    """Build extended voltage grid for sensitivity analysis.

    Starts with v13 cathodic points and extends beyond -46.5 to v_min.
    Also includes near-equilibrium anodic/symmetric points.
    Returns sorted descending (for warm-start continuation order).

    Parameters
    ----------
    v_min : float
        Most negative voltage to include. Default -60.0.

    Returns
    -------
    np.ndarray
        Unique voltage values sorted in descending order.
    """
    # Start with v13 cathodic grid
    points = list(_V13_CATHODIC)

    # Add extended cathodic points
    extended = [-50.0, -55.0, -60.0]
    if v_min <= -65.0:
        extended.append(-65.0)
    if v_min <= -70.0:
        extended.append(-70.0)
    if v_min <= -75.0:
        extended.append(-75.0)

    # Add anodic/near-equilibrium points
    anodic = [5.0, 3.0, 1.0, -0.5]

    all_points = points + extended + anodic
    # Filter to v_min
    all_points = [v for v in all_points if v >= v_min]

    # Unique and descending
    grid = np.array(sorted(set(all_points), reverse=True))
    return grid


def build_perturbed_params(
    param_name: str,
    factor: float,
    true_params: dict,
) -> dict:
    """Return a new params dict with one parameter perturbed by a factor.

    Does NOT mutate the original true_params (immutable pattern).

    Parameters
    ----------
    param_name : str
        Name of the parameter to perturb (e.g. "k0_1").
    factor : float
        Multiplicative factor to apply.
    true_params : dict
        Base parameter values.

    Returns
    -------
    dict
        New dict with param_name multiplied by factor.
    """
    result = dict(true_params)
    result[param_name] = true_params[param_name] * factor
    return result


def compute_jacobian_row(
    eval_func,
    base_params_array: np.ndarray,
    h: float = 1e-5,
) -> np.ndarray:
    """Compute gradient via central finite differences.

    Parameters
    ----------
    eval_func : callable
        Function mapping params array -> scalar observable value.
    base_params_array : np.ndarray
        Parameter values at which to evaluate the gradient.
    h : float
        Step size for central differences. Default 1e-5 (codebase convention).

    Returns
    -------
    np.ndarray
        Gradient array of length n_params.
    """
    n = len(base_params_array)
    gradient = np.zeros(n)
    for i in range(n):
        params_plus = base_params_array.copy()
        params_minus = base_params_array.copy()
        params_plus[i] += h
        params_minus[i] -= h
        gradient[i] = (eval_func(params_plus) - eval_func(params_minus)) / (2.0 * h)
    return gradient


# ---------------------------------------------------------------------------
# I-V curve evaluation (requires Firedrake)
# ---------------------------------------------------------------------------

def _import_firedrake_deps():
    """Lazily import Firedrake-dependent modules."""
    from scripts._bv_common import (
        setup_firedrake_env,
        K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2,
        I_SCALE,
        FOUR_SPECIES_CHARGED,
        SNES_OPTS_CHARGED,
        make_bv_solver_params,
        make_recovery_config,
    )
    setup_firedrake_env()

    from FluxCurve.bv_point_solve import (
        solve_bv_curve_points_with_warmstart,
        _clear_caches,
    )
    from Forward.steady_state import SteadyStateConfig

    return {
        "K0_HAT_R1": K0_HAT_R1,
        "K0_HAT_R2": K0_HAT_R2,
        "ALPHA_R1": ALPHA_R1,
        "ALPHA_R2": ALPHA_R2,
        "I_SCALE": I_SCALE,
        "FOUR_SPECIES_CHARGED": FOUR_SPECIES_CHARGED,
        "SNES_OPTS_CHARGED": SNES_OPTS_CHARGED,
        "make_bv_solver_params": make_bv_solver_params,
        "make_recovery_config": make_recovery_config,
        "solve_bv_curve_points_with_warmstart": solve_bv_curve_points_with_warmstart,
        "_clear_caches": _clear_caches,
        "SteadyStateConfig": SteadyStateConfig,
    }


def _get_true_params(deps: dict) -> dict:
    """Return true (baseline) parameter dict from codebase constants."""
    return {
        "k0_1": deps["K0_HAT_R1"],
        "k0_2": deps["K0_HAT_R2"],
        "alpha_1": deps["ALPHA_R1"],
        "alpha_2": deps["ALPHA_R2"],
    }


def evaluate_iv_curve(
    params: dict,
    voltage_grid: np.ndarray,
    config: SensitivityConfig,
    deps: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate I-V curve for given kinetic parameters.

    Uses solve_bv_curve_points_with_warmstart with extended SNES options.
    Handles solver failures by setting NaN for failed voltage points.

    Parameters
    ----------
    params : dict
        Kinetic parameters: k0_1, k0_2, alpha_1, alpha_2.
    voltage_grid : np.ndarray
        Voltage values (descending order for warm-starting).
    config : SensitivityConfig
        Configuration.
    deps : dict
        Firedrake dependency dict from _import_firedrake_deps().

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (total_flux, peroxide_flux) arrays, same length as voltage_grid.
    """
    # Build extended SNES options
    snes_opts = dict(deps["SNES_OPTS_CHARGED"])
    snes_opts["snes_max_it"] = config.snes_max_it

    # Build solver params at first voltage point (will be overridden per-point)
    base_sp = deps["make_bv_solver_params"](
        eta_hat=float(voltage_grid[0]),
        dt=config.dt,
        t_end=config.dt * config.max_ss_steps,
        species=deps["FOUR_SPECIES_CHARGED"],
        snes_opts=snes_opts,
        k0_hat_r1=float(params["k0_1"]),
        k0_hat_r2=float(params["k0_2"]),
        alpha_r1=float(params["alpha_1"]),
        alpha_r2=float(params["alpha_2"]),
    )

    steady = deps["SteadyStateConfig"](
        max_steps=config.max_ss_steps,
        absolute_tolerance=1e-7,
        relative_tolerance=1e-6,
        consecutive_steps=3,
    )

    recovery = deps["make_recovery_config"](max_it_cap=800)

    n_pts = len(voltage_grid)
    # Dummy target (not used for sensitivity -- we only need forward solve)
    dummy_target = np.zeros(n_pts)

    deps["_clear_caches"]()

    try:
        results = deps["solve_bv_curve_points_with_warmstart"](
            base_solver_params=base_sp,
            steady=steady,
            phi_applied_values=voltage_grid,
            target_flux=dummy_target,
            k0_values=[float(params["k0_1"]), float(params["k0_2"])],
            alpha_values=[float(params["alpha_1"]), float(params["alpha_2"])],
            blob_initial_condition=False,
            fail_penalty=0.0,
            forward_recovery=recovery,
            observable_mode="current_density",
            observable_reaction_index=None,
            observable_scale=float(-deps["I_SCALE"]),
            control_mode="joint",
        )

        total_flux = np.array([
            r.simulated_flux if r.converged else np.nan
            for r in results
        ])
        # For peroxide, re-run with observable_reaction_index=1
        deps["_clear_caches"]()
        results_peroxide = deps["solve_bv_curve_points_with_warmstart"](
            base_solver_params=base_sp,
            steady=steady,
            phi_applied_values=voltage_grid,
            target_flux=dummy_target,
            k0_values=[float(params["k0_1"]), float(params["k0_2"])],
            alpha_values=[float(params["alpha_1"]), float(params["alpha_2"])],
            blob_initial_condition=False,
            fail_penalty=0.0,
            forward_recovery=recovery,
            observable_mode="current_density",
            observable_reaction_index=1,
            observable_scale=float(-deps["I_SCALE"]),
            control_mode="joint",
        )
        peroxide_flux = np.array([
            r.simulated_flux if r.converged else np.nan
            for r in results_peroxide
        ])

    except Exception as exc:
        warnings.warn(f"I-V curve evaluation failed: {exc}")
        total_flux = np.full(n_pts, np.nan)
        peroxide_flux = np.full(n_pts, np.nan)

    # Log failed points
    n_failed = int(np.isnan(total_flux).sum())
    if n_failed > 0 and config.verbose:
        failed_voltages = voltage_grid[np.isnan(total_flux)]
        print(f"  [warning] {n_failed}/{n_pts} voltage points failed: "
              f"{failed_voltages.tolist()}")

    return total_flux, peroxide_flux


# ---------------------------------------------------------------------------
# Full Jacobian computation
# ---------------------------------------------------------------------------

def compute_full_jacobian(
    voltage_grid: np.ndarray,
    true_params: dict,
    config: SensitivityConfig,
    deps: dict,
) -> np.ndarray:
    """Compute Jacobian matrix d(observable)/d(parameter) at each voltage.

    Result shape: (2 * n_voltages, 4)
    First n_voltages rows: d(total_current)/d(param)
    Second n_voltages rows: d(peroxide_current)/d(param)

    Uses central finite differences with h=config.jacobian_h.

    Parameters
    ----------
    voltage_grid : np.ndarray
        Voltage values.
    true_params : dict
        Baseline parameter values.
    config : SensitivityConfig
        Configuration.
    deps : dict
        Firedrake dependency dict.

    Returns
    -------
    np.ndarray
        Jacobian matrix, shape (2 * n_voltages, n_params).
    """
    param_names = list(config.params_to_sweep)
    n_voltages = len(voltage_grid)
    n_params = len(param_names)
    h = config.jacobian_h

    # Evaluate baseline
    total_base, peroxide_base = evaluate_iv_curve(
        true_params, voltage_grid, config, deps
    )

    jacobian = np.zeros((2 * n_voltages, n_params))

    h_rel = h  # treat configured step as a relative perturbation fraction

    for j, pname in enumerate(param_names):
        if config.verbose:
            print(f"  [jacobian] Computing d/d({pname}) ...")

        # Use relative perturbation so small parameters aren't swamped
        p_val = true_params[pname]
        h_abs = h_rel * abs(p_val) if abs(p_val) > 1e-15 else h_rel

        params_plus = dict(true_params)
        params_plus[pname] = p_val + h_abs

        params_minus = dict(true_params)
        params_minus[pname] = p_val - h_abs

        total_plus, peroxide_plus = evaluate_iv_curve(
            params_plus, voltage_grid, config, deps
        )
        total_minus, peroxide_minus = evaluate_iv_curve(
            params_minus, voltage_grid, config, deps
        )

        # Central FD
        jacobian[:n_voltages, j] = (total_plus - total_minus) / (2.0 * h_abs)
        jacobian[n_voltages:, j] = (peroxide_plus - peroxide_minus) / (2.0 * h_abs)

    return jacobian


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def generate_sweep_plots(
    param_name: str,
    voltage_grid: np.ndarray,
    sweep_results: dict,
    config: SensitivityConfig,
) -> None:
    """Generate sweep plots: total and peroxide current vs voltage.

    Parameters
    ----------
    param_name : str
        Name of the swept parameter.
    voltage_grid : np.ndarray
        Voltage values.
    sweep_results : dict
        Maps factor -> (total_flux, peroxide_flux).
    config : SensitivityConfig
        Configuration.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for factor in sorted(sweep_results.keys()):
        total, peroxide = sweep_results[factor]
        label = f"{factor:.2f}x"
        lw = 2.0 if factor == 1.0 else 1.0
        ax1.plot(voltage_grid, total, label=label, linewidth=lw)
        ax2.plot(voltage_grid, peroxide, label=label, linewidth=lw)

    ax1.set_ylabel("Total Current (mA/cm2)")
    ax1.set_title(f"Sensitivity to {param_name}")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel("Peroxide Current (mA/cm2)")
    ax2.set_xlabel("Overpotential (eta)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(config.output_dir, exist_ok=True)
    out_path = os.path.join(config.output_dir, f"sweep_{param_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if config.verbose:
        print(f"  [plot] Saved {out_path}")


def generate_jacobian_heatmap(
    voltage_grid: np.ndarray,
    jacobian_matrix: np.ndarray,
    param_names: list,
    output_dir: str,
) -> None:
    """Generate Jacobian heatmap for total and peroxide current sensitivity.

    Parameters
    ----------
    voltage_grid : np.ndarray
        Voltage values.
    jacobian_matrix : np.ndarray
        Shape (2*n_voltages, n_params).
    param_names : list
        Parameter names.
    output_dir : str
        Output directory.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_voltages = len(voltage_grid)
    n_params = len(param_names)

    jac_total = jacobian_matrix[:n_voltages, :]
    jac_peroxide = jacobian_matrix[n_voltages:, :]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

    for ax, data, title in [
        (ax1, jac_total, "d(Total Current)/d(param)"),
        (ax2, jac_peroxide, "d(Peroxide Current)/d(param)"),
    ]:
        # Normalize per column to show relative sensitivity
        col_max = np.nanmax(np.abs(data), axis=0, keepdims=True)
        col_max = np.where(col_max == 0, 1.0, col_max)
        normalized = data / col_max

        im = ax.imshow(
            normalized,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-1, vmax=1,
            interpolation="nearest",
        )
        ax.set_xticks(range(n_params))
        ax.set_xticklabels(param_names, fontsize=9)
        ax.set_yticks(range(n_voltages))
        ax.set_yticklabels([f"{v:.1f}" for v in voltage_grid], fontsize=7)
        ax.set_ylabel("Overpotential (eta)")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.8, label="Normalized sensitivity")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "jacobian_heatmap.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved {out_path}")


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_sweep_csv(
    param_name: str,
    voltage_grid: np.ndarray,
    sweep_results: dict,
    output_dir: str,
) -> None:
    """Write sweep data to CSV.

    Columns: voltage, factor_0.5_total, factor_0.5_peroxide, ...
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"sweep_{param_name}.csv")

    factors_sorted = sorted(sweep_results.keys())
    header = ["voltage"]
    for f in factors_sorted:
        header.append(f"factor_{f}_total")
        header.append(f"factor_{f}_peroxide")

    with open(out_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for i, v in enumerate(voltage_grid):
            row = [v]
            for f in factors_sorted:
                total, peroxide = sweep_results[f]
                row.append(total[i])
                row.append(peroxide[i])
            writer.writerow(row)

    print(f"  [csv] Saved {out_path}")


def write_jacobian_csv(
    voltage_grid: np.ndarray,
    jacobian_matrix: np.ndarray,
    param_names: list,
    output_dir: str,
) -> None:
    """Write Jacobian data to CSV.

    Columns: voltage, d_total_d_k0_1, ..., d_peroxide_d_k0_1, ...
    """
    n_voltages = len(voltage_grid)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "jacobian_heatmap.csv")

    header = ["voltage"]
    for pname in param_names:
        header.append(f"d_total_d_{pname}")
    for pname in param_names:
        header.append(f"d_peroxide_d_{pname}")

    with open(out_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for i, v in enumerate(voltage_grid):
            row = [v]
            for j in range(len(param_names)):
                row.append(jacobian_matrix[i, j])
            for j in range(len(param_names)):
                row.append(jacobian_matrix[n_voltages + i, j])
            writer.writerow(row)

    print(f"  [csv] Saved {out_path}")


# ---------------------------------------------------------------------------
# Metadata (AUDT-04)
# ---------------------------------------------------------------------------

def write_metadata(output_dir: str) -> None:
    """Write AUDT-04 compliant JSON metadata sidecar.

    Parameters
    ----------
    output_dir : str
        Directory to write metadata.json into.
    """
    metadata = {
        "tool_name": "Parameter Sensitivity Visualization (Sweeps + Jacobian)",
        "requirement": "DIAG-03",
        "justification_type": "empirical",
        "reference": (
            "Sensitivity analysis via parameter perturbation and Jacobian "
            "evaluation is standard practice for identifying informative "
            "experimental conditions"
        ),
        "rationale": (
            "1D sweeps show observable response to parameter changes; "
            "Jacobian heatmap quantifies which voltage regions carry most "
            "information about each parameter, informing voltage selection "
            "for Phase 9"
        ),
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": "scripts/studies/sensitivity_visualization.py",
    }
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "metadata.json")
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  [meta] Saved {out_path}")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main() -> None:
    """Run sensitivity analysis with CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Parameter sensitivity visualization: 1D sweeps + Jacobian heatmap",
    )
    parser.add_argument(
        "--params",
        nargs="+",
        default=["k0_1", "k0_2", "alpha_1", "alpha_2"],
        help="Parameters to sweep (default: all 4)",
    )
    parser.add_argument(
        "--factors",
        type=str,
        default="0.5,0.75,1.0,1.5,2.0",
        help="Comma-separated multiplicative factors (default: 0.5,0.75,1.0,1.5,2.0)",
    )
    parser.add_argument(
        "--voltage-min",
        type=float,
        default=-60.0,
        help="Most negative voltage in extended grid (default: -60.0)",
    )
    parser.add_argument(
        "--no-jacobian",
        action="store_true",
        help="Skip Jacobian heatmap computation",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output (default: True)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="StudyResults/v14/sensitivity",
        help="Output directory (default: StudyResults/v14/sensitivity)",
    )
    args = parser.parse_args()

    # Parse factors
    factors = tuple(float(f) for f in args.factors.split(","))

    config = SensitivityConfig(
        sweep_factors=factors,
        params_to_sweep=tuple(args.params),
        output_dir=args.output_dir,
        extended_voltage_min=args.voltage_min,
        verbose=args.verbose,
    )

    print("=" * 60)
    print("Parameter Sensitivity Visualization")
    print("=" * 60)
    print(f"Parameters: {config.params_to_sweep}")
    print(f"Factors: {config.sweep_factors}")
    print(f"Voltage min: {config.extended_voltage_min}")
    print(f"Output: {config.output_dir}")
    print()

    # Import Firedrake dependencies
    deps = _import_firedrake_deps()
    true_params = _get_true_params(deps)

    # Build voltage grid
    voltage_grid = build_extended_voltage_grid(v_min=config.extended_voltage_min)
    print(f"Voltage grid: {len(voltage_grid)} points, "
          f"range [{voltage_grid.min():.1f}, {voltage_grid.max():.1f}]")
    print()

    # Build sweep factors
    sweep_factors = build_sweep_factors(factors=config.sweep_factors)

    # ---- 1D Parameter Sweeps ----
    for pname in config.params_to_sweep:
        print(f"\n--- Sweeping {pname} ---")
        sweep_results = {}

        for factor in sweep_factors:
            perturbed = build_perturbed_params(pname, factor, true_params)
            print(f"  factor={factor:.2f}x, {pname}={perturbed[pname]:.6e}")

            total, peroxide = evaluate_iv_curve(
                perturbed, voltage_grid, config, deps
            )
            sweep_results[factor] = (total, peroxide)

        # Generate plots and CSV
        generate_sweep_plots(pname, voltage_grid, sweep_results, config)
        write_sweep_csv(pname, voltage_grid, sweep_results, config.output_dir)

    # ---- Jacobian Heatmap ----
    if not args.no_jacobian:
        print("\n--- Computing Jacobian Heatmap ---")
        param_names = list(config.params_to_sweep)
        jacobian = compute_full_jacobian(
            voltage_grid, true_params, config, deps
        )
        generate_jacobian_heatmap(
            voltage_grid, jacobian, param_names, config.output_dir
        )
        write_jacobian_csv(
            voltage_grid, jacobian, param_names, config.output_dir
        )

    # ---- Metadata ----
    write_metadata(config.output_dir)

    print("\n" + "=" * 60)
    print("Sensitivity analysis complete.")
    print(f"Results in: {config.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
