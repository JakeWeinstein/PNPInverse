"""PDE-only profile likelihood analysis for practical identifiability.

For each kinetic parameter (k0_1, k0_2, alpha_1, alpha_2), fixes that
parameter at 30 grid points and jointly re-optimizes the remaining 3
parameters via L-BFGS-B.  The resulting profile curve is tested against
the chi-squared 95% CI threshold (delta-chi2 = 3.84) to determine
practical identifiability.

This is the gold-standard tool for practical identifiability assessment,
handling nonlinear parameter interactions that Fisher Information Matrix
approaches miss.

Output:
    - Per-parameter CSV: grid_value, loss, chi2_profile
    - Per-parameter PNG: chi2 profile plot with threshold line
    - identifiability_summary.csv: bounded/identifiable status per param
    - metadata.json: AUDT-04 justification sidecar

Usage (from PNPInverse/ directory)::

    python scripts/studies/profile_likelihood_pde.py
    python scripts/studies/profile_likelihood_pde.py --params k0_2 --n-points 50
    python scripts/studies/profile_likelihood_pde.py --help
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Environment setup (for standalone execution)
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DELTA_CHI2_95 = 3.84  # chi2(1) 95th percentile, 1 DOF

# Parameter name -> (type, component index)
# type: "k0" or "alpha"; component index within the k0 or alpha vector
PARAM_REGISTRY: Dict[str, Tuple[str, int]] = {
    "k0_1": ("k0", 0),
    "k0_2": ("k0", 1),
    "alpha_1": ("alpha", 0),
    "alpha_2": ("alpha", 1),
}


@dataclass(frozen=True)
class ProfileLikelihoodConfig:
    """Immutable configuration for profile likelihood analysis."""

    n_points: int = 30
    params_to_profile: Tuple[str, ...] = ("k0_1", "k0_2", "alpha_1", "alpha_2")
    delta_chi2: float = DELTA_CHI2_95
    output_dir: str = "StudyResults/v14/profile_likelihood"
    noise_percent: float = 2.0
    noise_seed: int = 0
    pde_maxiter: int = 25
    verbose: bool = True


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------

def build_profile_grid(
    param_name: str,
    true_value: float,
    n_points: int = 30,
) -> np.ndarray:
    """Build the profile grid for a given parameter.

    Parameters
    ----------
    param_name:
        One of "k0_1", "k0_2", "alpha_1", "alpha_2".
    true_value:
        True (reference) value of the parameter.
    n_points:
        Number of grid points.

    Returns
    -------
    np.ndarray
        Grid of parameter values to profile over.

    For k0_1: log-space from 0.01x to 100x true value.
    For k0_2: log-space from 0.001x to 1000x true value (wider range
              due to identifiability risk).
    For alpha_1, alpha_2: linear-space from 0.1 to 0.95 (physical bounds).
    """
    if param_name == "k0_1":
        return np.logspace(
            np.log10(true_value * 0.01),
            np.log10(true_value * 100),
            n_points,
        )
    elif param_name == "k0_2":
        return np.logspace(
            np.log10(true_value * 0.001),
            np.log10(true_value * 1000),
            n_points,
        )
    elif param_name in ("alpha_1", "alpha_2"):
        return np.linspace(0.1, 0.95, n_points)
    else:
        raise ValueError(
            f"Unknown parameter '{param_name}'. "
            f"Expected one of {list(PARAM_REGISTRY.keys())}."
        )


# ---------------------------------------------------------------------------
# Identifiability assessment
# ---------------------------------------------------------------------------

def assess_identifiability(
    profile_losses: np.ndarray,
    global_min_loss: float,
    n_obs: int,
    n_params: int,
    delta_chi2: float = DELTA_CHI2_95,
) -> Dict[str, Any]:
    """Assess practical identifiability from a profile likelihood curve.

    Parameters
    ----------
    profile_losses:
        Array of objective values at each profile grid point.
    global_min_loss:
        Global minimum objective (from unconstrained optimization).
    n_obs:
        Number of observations (data points).
    n_params:
        Number of free parameters in the model.
    delta_chi2:
        Chi-squared threshold for 95% CI (default 3.84 for 1 DOF).

    Returns
    -------
    dict with keys:
        identifiable: bool -- True if chi2 exceeds threshold on both sides
        left_bounded: bool -- True if chi2 exceeds threshold to the left
        right_bounded: bool -- True if chi2 exceeds threshold to the right
        chi2_profile: np.ndarray -- chi2 values at each grid point
    """
    sigma2_hat = global_min_loss / max(n_obs - n_params, 1)
    if sigma2_hat < 1e-30:
        # Degenerate case: essentially zero residual
        chi2_profile = np.zeros_like(profile_losses)
        return {
            "identifiable": False,
            "left_bounded": False,
            "right_bounded": False,
            "chi2_profile": chi2_profile,
        }

    chi2_profile = (profile_losses - global_min_loss) / sigma2_hat
    min_idx = int(np.argmin(chi2_profile))

    left_bounded = bool(
        np.any(chi2_profile[:min_idx] > delta_chi2)
    ) if min_idx > 0 else False

    right_bounded = bool(
        np.any(chi2_profile[min_idx + 1:] > delta_chi2)
    ) if min_idx < len(chi2_profile) - 1 else False

    identifiable = left_bounded and right_bounded

    return {
        "identifiable": identifiable,
        "left_bounded": left_bounded,
        "right_bounded": right_bounded,
        "chi2_profile": chi2_profile,
    }


# ---------------------------------------------------------------------------
# Bound construction for pinned parameter
# ---------------------------------------------------------------------------

def build_fixed_bounds(
    param_name: str,
    fixed_val: float,
    default_bounds: Dict[str, List[float]],
) -> Dict[str, List[float]]:
    """Build bounds with the profiled parameter pinned (lower == upper == fixed_val).

    Parameters
    ----------
    param_name:
        One of "k0_1", "k0_2", "alpha_1", "alpha_2".
    fixed_val:
        Value to pin the profiled parameter at.
    default_bounds:
        Dict with keys "k0_lower", "k0_upper", "alpha_lower", "alpha_upper",
        each a list of floats (one per component).

    Returns
    -------
    dict: Modified bounds (deep copy; input is not mutated).
    """
    result = copy.deepcopy(default_bounds)

    param_type, comp_idx = PARAM_REGISTRY[param_name]
    lower_key = f"{param_type}_lower"
    upper_key = f"{param_type}_upper"

    result[lower_key][comp_idx] = fixed_val
    result[upper_key][comp_idx] = fixed_val

    return result


# ---------------------------------------------------------------------------
# Profile runner (PDE optimization per grid point)
# ---------------------------------------------------------------------------

def run_profile_for_parameter(
    param_name: str,
    config: ProfileLikelihoodConfig,
    global_best: Dict[str, Any],
    default_bounds: Dict[str, List[float]],
) -> Dict[str, Any]:
    """Run full profile for one parameter.

    For each grid point, pins the profiled parameter and jointly
    re-optimizes the remaining 3 parameters via the PDE pipeline.

    Parameters
    ----------
    param_name:
        Parameter to profile.
    config:
        Profile likelihood configuration.
    global_best:
        Dict with "k0" (list), "alpha" (list), "loss" (float) from
        global unconstrained optimization.
    default_bounds:
        Default k0/alpha bounds for free parameters.

    Returns
    -------
    dict with:
        grid_values: np.ndarray
        profile_losses: np.ndarray
        identifiability: dict (from assess_identifiability)
    """
    from scripts._bv_common import (
        setup_firedrake_env,
        K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2,
        I_SCALE, FOUR_SPECIES_CHARGED, SNES_OPTS_CHARGED,
        make_bv_solver_params, make_recovery_config,
    )
    setup_firedrake_env()

    from FluxCurve import BVFluxCurveInferenceRequest
    from FluxCurve.bv_run.pipelines import run_bv_multi_observable_flux_curve_inference
    from Forward.steady_state import SteadyStateConfig

    param_type, comp_idx = PARAM_REGISTRY[param_name]
    true_k0 = [K0_HAT_R1, K0_HAT_R2]
    true_alpha = [ALPHA_R1, ALPHA_R2]
    true_value = true_k0[comp_idx] if param_type == "k0" else true_alpha[comp_idx]

    grid = build_profile_grid(param_name, true_value, n_points=config.n_points)

    # PDE solver setup
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
    eta_values = np.linspace(-1.0, -10.0, 10)
    observable_scale = -I_SCALE

    param_output_dir = os.path.join(config.output_dir, f"profile_{param_name}")
    os.makedirs(param_output_dir, exist_ok=True)

    profile_losses = np.full(len(grid), np.nan)

    for gi, grid_val in enumerate(grid):
        if config.verbose:
            print(f"  [{param_name}] point {gi + 1}/{len(grid)}: {grid_val:.6e}")

        pinned_bounds = build_fixed_bounds(param_name, float(grid_val), default_bounds)
        point_dir = os.path.join(param_output_dir, f"grid_{gi:03d}")
        os.makedirs(point_dir, exist_ok=True)

        # Initial guess for free parameters from global best
        init_k0 = list(global_best["k0"])
        init_alpha = list(global_best["alpha"])

        # Pin the profiled parameter in the initial guess too
        if param_type == "k0":
            init_k0[comp_idx] = float(grid_val)
        else:
            init_alpha[comp_idx] = float(grid_val)

        request = BVFluxCurveInferenceRequest(
            base_solver_params=base_sp,
            steady=steady,
            true_k0=true_k0,
            initial_guess=init_k0,
            phi_applied_values=eta_values.tolist(),
            target_csv_path=os.path.join(point_dir, "target_primary.csv"),
            output_dir=point_dir,
            regenerate_target=True,
            target_noise_percent=config.noise_percent,
            target_seed=config.noise_seed,
            observable_mode="current_density",
            current_density_scale=observable_scale,
            observable_label="current density (mA/cm2)",
            observable_title=f"Profile: {param_name}={grid_val:.4e}",
            control_mode="joint",
            k0_lower=pinned_bounds["k0_lower"][0] if len(pinned_bounds["k0_lower"]) == 1 else 1e-8,
            k0_upper=pinned_bounds["k0_upper"][0] if len(pinned_bounds["k0_upper"]) == 1 else 100.0,
            k0_lower_per_component=pinned_bounds["k0_lower"],
            k0_upper_per_component=pinned_bounds["k0_upper"],
            log_space=True,
            true_alpha=true_alpha,
            initial_alpha_guess=init_alpha,
            alpha_lower=pinned_bounds["alpha_lower"][0] if len(pinned_bounds["alpha_lower"]) == 1 else 0.05,
            alpha_upper=pinned_bounds["alpha_upper"][0] if len(pinned_bounds["alpha_upper"]) == 1 else 0.95,
            alpha_lower_per_component=pinned_bounds["alpha_lower"],
            alpha_upper_per_component=pinned_bounds["alpha_upper"],
            secondary_observable_mode="peroxide_current",
            secondary_observable_weight=1.0,
            secondary_target_csv_path=os.path.join(point_dir, "target_peroxide.csv"),
            mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
            max_eta_gap=3.0,
            optimizer_method="L-BFGS-B",
            optimizer_options={
                "maxiter": config.pde_maxiter,
                "ftol": 1e-12,
                "gtol": 1e-6,
                "disp": config.verbose,
            },
            max_iters=config.pde_maxiter,
            live_plot=False,
            forward_recovery=make_recovery_config(max_it_cap=600),
        )

        try:
            result = run_bv_multi_observable_flux_curve_inference(request)
            profile_losses[gi] = float(result["best_loss"])
            if config.verbose:
                print(f"    loss={profile_losses[gi]:.6e}")
        except Exception as exc:
            print(f"    [FAIL] {param_name} grid {gi}: {exc}")
            profile_losses[gi] = np.nan

    # Compute n_obs from eta_values * 2 observables
    n_obs = len(eta_values) * 2
    n_params = 4

    identifiability = assess_identifiability(
        profile_losses=profile_losses,
        global_min_loss=global_best["loss"],
        n_obs=n_obs,
        n_params=n_params,
        delta_chi2=config.delta_chi2,
    )

    return {
        "grid_values": grid,
        "profile_losses": profile_losses,
        "identifiability": identifiability,
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def generate_profile_plot(
    param_name: str,
    grid_values: np.ndarray,
    chi2_profile: np.ndarray,
    delta_chi2: float,
    true_value: float,
    output_dir: str,
) -> None:
    """Generate and save profile likelihood plot for one parameter."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  [skip] matplotlib not available, skipping plot for {param_name}")
        return

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    is_k0 = param_name.startswith("k0")
    plot_fn = ax.semilogx if is_k0 else ax.plot
    plot_fn(grid_values, chi2_profile, "o-", color="tab:blue", markersize=4,
            label="profile chi2")

    ax.axhline(delta_chi2, color="tab:orange", linestyle="--", linewidth=1.5,
               label=f"95% CI threshold ({delta_chi2})")
    ax.axvline(true_value, color="tab:red", linestyle="--", linewidth=1.5,
               label=f"true = {true_value:.4g}")

    # Mark where chi2 crosses threshold
    above = chi2_profile > delta_chi2
    for i in range(len(above) - 1):
        if above[i] != above[i + 1]:
            cross_x = 0.5 * (grid_values[i] + grid_values[i + 1])
            ax.axvline(cross_x, color="tab:green", linestyle=":", alpha=0.6)

    ax.set_xlabel(param_name)
    ax.set_ylabel("chi2 profile")
    ax.set_title(f"Profile Likelihood: {param_name}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, f"profile_{param_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [plot] {path}")


def write_profile_csv(
    param_name: str,
    grid_values: np.ndarray,
    profile_losses: np.ndarray,
    chi2_profile: np.ndarray,
    output_dir: str,
) -> str:
    """Write per-parameter profile CSV."""
    path = os.path.join(output_dir, f"profile_{param_name}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["grid_value", "loss", "chi2_profile"])
        for gv, loss, chi2 in zip(grid_values, profile_losses, chi2_profile):
            writer.writerow([gv, loss, chi2])
    print(f"  [csv] {path}")
    return path


def write_identifiability_summary(
    results: Dict[str, Dict[str, Any]],
    output_dir: str,
) -> str:
    """Write identifiability summary CSV across all profiled parameters.

    Columns: parameter, identifiable, left_bounded, right_bounded,
             min_loss, ci_lower, ci_upper
    """
    path = os.path.join(output_dir, "identifiability_summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "parameter", "identifiable", "left_bounded", "right_bounded",
            "min_loss", "ci_lower", "ci_upper",
        ])
        for pname, pdata in results.items():
            ident = pdata["identifiability"]
            grid = pdata["grid_values"]
            losses = pdata["profile_losses"]
            chi2 = ident["chi2_profile"]

            min_loss = float(np.nanmin(losses))

            # CI bounds: grid values where chi2 first/last crosses threshold
            below_thresh = chi2 <= DELTA_CHI2_95
            ci_lower = ""
            ci_upper = ""
            if np.any(below_thresh):
                ci_indices = np.where(below_thresh)[0]
                ci_lower = float(grid[ci_indices[0]])
                ci_upper = float(grid[ci_indices[-1]])

            writer.writerow([
                pname,
                ident["identifiable"],
                ident["left_bounded"],
                ident["right_bounded"],
                min_loss,
                ci_lower,
                ci_upper,
            ])
    print(f"  [csv] {path}")
    return path


def write_metadata(output_dir: str) -> str:
    """Write AUDT-04 JSON metadata sidecar."""
    metadata = {
        "tool_name": "PDE Profile Likelihood Identifiability Analysis",
        "requirement": "DIAG-02",
        "justification_type": "literature",
        "reference": (
            "Raue et al., Bioinformatics 2009 - Structural and practical "
            "identifiability analysis of partially observed dynamical models"
        ),
        "rationale": (
            "Profile likelihood is the gold-standard method for practical "
            "identifiability; it handles nonlinear parameter interactions "
            "that Fisher Information Matrix misses"
        ),
        "parameters_profiled": ["k0_1", "k0_2", "alpha_1", "alpha_2"],
        "statistical_test": "chi2(1) 95th percentile",
        "threshold": DELTA_CHI2_95,
        "created_by": "scripts/studies/profile_likelihood_pde.py",
    }
    path = os.path.join(output_dir, "metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"  [json] {path}")
    return path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run PDE-only profile likelihood analysis."""
    parser = argparse.ArgumentParser(
        description="PDE-only profile likelihood identifiability analysis"
    )
    parser.add_argument(
        "--params", nargs="+",
        default=["k0_1", "k0_2", "alpha_1", "alpha_2"],
        choices=["k0_1", "k0_2", "alpha_1", "alpha_2"],
        help="Parameters to profile (default: all 4)",
    )
    parser.add_argument(
        "--n-points", type=int, default=30,
        help="Number of grid points per profile (default: 30)",
    )
    parser.add_argument(
        "--noise-seed", type=int, default=0,
        help="Random seed for target noise (default: 0)",
    )
    parser.add_argument(
        "--pde-maxiter", type=int, default=25,
        help="L-BFGS-B max iterations per re-optimization (default: 25)",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="Verbose output (default: True)",
    )
    parser.add_argument(
        "--quiet", action="store_true", default=False,
        help="Suppress verbose output",
    )
    args = parser.parse_args()

    config = ProfileLikelihoodConfig(
        n_points=args.n_points,
        params_to_profile=tuple(args.params),
        noise_seed=args.noise_seed,
        pde_maxiter=args.pde_maxiter,
        verbose=not args.quiet,
    )

    # Lazy imports -- these require Firedrake
    from scripts._bv_common import (
        setup_firedrake_env,
        K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2,
        I_SCALE, FOUR_SPECIES_CHARGED, SNES_OPTS_CHARGED,
        make_bv_solver_params, make_recovery_config,
        print_params_summary,
    )
    setup_firedrake_env()

    from FluxCurve import BVFluxCurveInferenceRequest
    from FluxCurve.bv_run.pipelines import run_bv_multi_observable_flux_curve_inference
    from Forward.steady_state import SteadyStateConfig

    print_params_summary()

    # Default bounds for free parameters
    default_bounds: Dict[str, List[float]] = {
        "k0_lower": [1e-8, 1e-8],
        "k0_upper": [100.0, 100.0],
        "alpha_lower": [0.05, 0.05],
        "alpha_upper": [0.95, 0.95],
    }

    true_k0 = [K0_HAT_R1, K0_HAT_R2]
    true_alpha = [ALPHA_R1, ALPHA_R2]

    # ---------------------------------------------------------------
    # Step 1: Global optimization (all 4 parameters free)
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Step 1: Global PDE optimization (all parameters free)")
    print("=" * 70)

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
    eta_values = np.linspace(-1.0, -10.0, 10)
    observable_scale = -I_SCALE

    os.makedirs(config.output_dir, exist_ok=True)
    global_dir = os.path.join(config.output_dir, "global_optimization")
    os.makedirs(global_dir, exist_ok=True)

    global_request = BVFluxCurveInferenceRequest(
        base_solver_params=base_sp,
        steady=steady,
        true_k0=true_k0,
        initial_guess=[0.005, 0.0005],
        phi_applied_values=eta_values.tolist(),
        target_csv_path=os.path.join(global_dir, "target_primary.csv"),
        output_dir=global_dir,
        regenerate_target=True,
        target_noise_percent=config.noise_percent,
        target_seed=config.noise_seed,
        observable_mode="current_density",
        current_density_scale=observable_scale,
        observable_label="current density (mA/cm2)",
        observable_title="Global PDE optimization",
        control_mode="joint",
        k0_lower=1e-8, k0_upper=100.0,
        log_space=True,
        true_alpha=true_alpha,
        initial_alpha_guess=[0.4, 0.3],
        alpha_lower=0.05, alpha_upper=0.95,
        secondary_observable_mode="peroxide_current",
        secondary_observable_weight=1.0,
        secondary_target_csv_path=os.path.join(global_dir, "target_peroxide.csv"),
        mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
        max_eta_gap=3.0,
        optimizer_method="L-BFGS-B",
        optimizer_options={
            "maxiter": config.pde_maxiter,
            "ftol": 1e-12,
            "gtol": 1e-6,
            "disp": config.verbose,
        },
        max_iters=config.pde_maxiter,
        live_plot=False,
        forward_recovery=make_recovery_config(max_it_cap=600),
    )

    global_result = run_bv_multi_observable_flux_curve_inference(global_request)
    global_best = {
        "k0": list(global_result["best_k0"]),
        "alpha": list(global_result["best_alpha"]),
        "loss": float(global_result["best_loss"]),
    }
    print(f"\n  Global best: k0={global_best['k0']}, alpha={global_best['alpha']}")
    print(f"  Global min loss: {global_best['loss']:.6e}")

    # ---------------------------------------------------------------
    # Step 2: Profile each parameter
    # ---------------------------------------------------------------
    t_start = time.time()
    all_results: Dict[str, Dict[str, Any]] = {}

    for pname in config.params_to_profile:
        print(f"\n{'=' * 70}")
        print(f"  Profiling: {pname} ({config.n_points} grid points)")
        print(f"{'=' * 70}")

        pdata = run_profile_for_parameter(
            param_name=pname,
            config=config,
            global_best=global_best,
            default_bounds=default_bounds,
        )
        all_results[pname] = pdata

        # Write per-parameter outputs
        ident = pdata["identifiability"]
        write_profile_csv(
            pname, pdata["grid_values"], pdata["profile_losses"],
            ident["chi2_profile"], config.output_dir,
        )

        param_type, comp_idx = PARAM_REGISTRY[pname]
        true_val = true_k0[comp_idx] if param_type == "k0" else true_alpha[comp_idx]
        generate_profile_plot(
            pname, pdata["grid_values"], ident["chi2_profile"],
            config.delta_chi2, true_val, config.output_dir,
        )

        status = "IDENTIFIABLE" if ident["identifiable"] else "NOT IDENTIFIABLE"
        left = "bounded" if ident["left_bounded"] else "unbounded"
        right = "bounded" if ident["right_bounded"] else "unbounded"
        print(f"  --> {pname}: {status} (left={left}, right={right})")

    # ---------------------------------------------------------------
    # Step 3: Write summary outputs
    # ---------------------------------------------------------------
    write_identifiability_summary(all_results, config.output_dir)
    write_metadata(config.output_dir)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  PROFILE LIKELIHOOD COMPLETE")
    print(f"  Duration: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"  Output: {config.output_dir}/")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
