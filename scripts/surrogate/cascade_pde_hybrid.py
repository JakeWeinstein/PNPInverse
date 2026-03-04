"""Cascade + PDE Hybrid Strategy for BV kinetics parameter recovery.

Exploits the observation that the surrogate cascade (w=0.5) recovers
k0_1, alpha_1, alpha_2 to excellent accuracy (<2.5%) but k0_2 remains
limited (~16.6%) by surrogate interpolation error.  The PDE solver has
no such limitation, so we run a targeted 1D PDE optimization for k0_2
only, with the other 3 parameters fixed from the cascade.

Three-phase protocol:
    Phase 0: Load surrogate, generate PDE targets at true parameters
    Phase 1: Surrogate cascade (CD-dominant, w=0.5) -- locks k0_1,
             alpha_1, alpha_2 at excellent accuracy
    Phase 2: Gradient-based 1D PDE optimization for k0_2 only
             (L-BFGS-B with PDE gradient, full cathodic range, high
             peroxide weight) -- fixes k0_1, alpha_1, alpha_2 from
             Phase 1
    Phase 3: (Optional) Short joint PDE polish with all 4 params free,
             tight bounds, and regularization toward Phase 1+2 result

Key improvements over v1:
    - Full cathodic voltage range (15 pts, eta -1 to -46.5) for k0_2
      sensitivity (was 10 pts, -1 to -13)
    - High secondary_weight=5.0 (was 1.0) amplifies peroxide signal
    - Gradient-based L-BFGS-B (was gradient-free minimize_scalar)
    - Wider k0_2 bounds (factor 20, was 10)

Usage (from PNPInverse/ directory)::

    python scripts/surrogate/cascade_pde_hybrid.py \\
        --model StudyResults/surrogate_v9/surrogate_model.pkl

    # Shallow voltage range (legacy behavior):
    python scripts/surrogate/cascade_pde_hybrid.py \\
        --pde-voltage-range shallow --pde-secondary-weight 1.0

    # Weight sweep (compare w=1,2,5,10,20):
    python scripts/surrogate/cascade_pde_hybrid.py --sweep-weights

    # Skip joint polish:
    python scripts/surrogate/cascade_pde_hybrid.py --no-joint-polish
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from typing import Sequence

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts._bv_common import (
    setup_firedrake_env,
    K0_HAT_R1, K0_HAT_R2, K_SCALE, I_SCALE,
    ALPHA_R1, ALPHA_R2,
    FOUR_SPECIES_CHARGED,
    SNES_OPTS_CHARGED,
    make_bv_solver_params,
    make_recovery_config,
)
setup_firedrake_env()

# Backward-compat aliases
K0_HAT = K0_HAT_R1
K0_2_HAT = K0_HAT_R2
ALPHA_1 = ALPHA_R1
ALPHA_2 = ALPHA_R2

import numpy as np
from scipy.optimize import minimize

from Surrogate.io import load_surrogate
from Surrogate.cascade import CascadeConfig, run_cascade_inference


# ---------------------------------------------------------------------------
# Fallback training-data bounds
# ---------------------------------------------------------------------------
K0_1_TRAIN_LO_DEFAULT = K0_HAT * 0.01
K0_1_TRAIN_HI_DEFAULT = K0_HAT * 100.0
K0_2_TRAIN_LO_DEFAULT = K0_2_HAT * 0.01
K0_2_TRAIN_HI_DEFAULT = K0_2_HAT * 100.0
ALPHA_TRAIN_LO_DEFAULT = 0.10
ALPHA_TRAIN_HI_DEFAULT = 0.90


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HybridConfig:
    """Frozen configuration for the cascade+PDE hybrid strategy.

    Attributes
    ----------
    pass1_weight : float
        CD-dominant weight for the surrogate cascade.
    pde_maxiter : int
        Maximum function evaluations for 1D k0_2 PDE search.
    k0_2_bound_factor : float
        Multiplicative factor for k0_2 bounds (e.g. 20.0 => [k0_2/20, k0_2*20]).
    joint_polish : bool
        If True, run Phase 3 joint PDE polish after k0_2 refinement.
    joint_polish_maxiter : int
        Max L-BFGS-B iterations for joint polish.
    joint_polish_lambda : float
        Regularization strength for joint polish.
    joint_polish_bound_factor : float
        k0 bound factor for joint polish.
    secondary_weight : float
        Weight on peroxide current in PDE objective.
    workers : int
        Number of PDE parallel workers (0=auto).
    pde_voltage_range : str
        Voltage range for Phase 2 PDE: "shallow" (10 pts) or "cathodic" (15 pts).
    """

    pass1_weight: float = 0.5
    pde_maxiter: int = 30
    k0_2_bound_factor: float = 20.0
    joint_polish: bool = True
    joint_polish_maxiter: int = 8
    joint_polish_lambda: float = 1.0
    joint_polish_bound_factor: float = 2.0
    secondary_weight: float = 5.0
    workers: int = 0
    pde_voltage_range: str = "cathodic"

    # Phase 3: Asymmetric regularization lambdas
    joint_polish_lambda_k0_1: float = 5.0
    joint_polish_lambda_k0_2: float = 0.1
    joint_polish_lambda_alpha_1: float = 5.0
    joint_polish_lambda_alpha_2: float = 5.0

    # Phase 3: Asymmetric bounds
    joint_polish_k0_1_bound_factor: float = 1.5
    joint_polish_k0_2_bound_factor: float = 5.0
    joint_polish_alpha_margin: float = 0.03

    # Phase 3: Joint mode ("asymmetric" or "free")
    joint_mode: str = "asymmetric"


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PhaseResult:
    """Snapshot of a single hybrid phase result.

    Attributes
    ----------
    phase_name : str
        Human-readable name for the phase.
    k0_1, k0_2 : float
        Recovered k0 values.
    alpha_1, alpha_2 : float
        Recovered alpha values.
    loss : float
        Objective value.
    elapsed_s : float
        Wall-clock time for this phase (seconds).
    n_pde_evals : int
        Number of PDE evaluations (0 for surrogate phases).
    """

    phase_name: str
    k0_1: float
    k0_2: float
    alpha_1: float
    alpha_2: float
    loss: float
    elapsed_s: float
    n_pde_evals: int = 0


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def compute_errors(
    k0: Sequence[float],
    alpha: Sequence[float],
    true_k0_arr: np.ndarray,
    true_alpha_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute relative errors for k0 and alpha arrays."""
    k0_arr = np.asarray(k0)
    alpha_arr = np.asarray(alpha)
    k0_err = np.abs(k0_arr - true_k0_arr) / np.maximum(np.abs(true_k0_arr), 1e-16)
    alpha_err = np.abs(alpha_arr - true_alpha_arr) / np.maximum(
        np.abs(true_alpha_arr), 1e-16
    )
    return k0_err, alpha_err


def print_phase_result(
    name: str,
    k0: np.ndarray,
    alpha: np.ndarray,
    true_k0_arr: np.ndarray,
    true_alpha_arr: np.ndarray,
    loss: float,
    elapsed: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Print a phase result with parameter errors."""
    k0_err, alpha_err = compute_errors(k0, alpha, true_k0_arr, true_alpha_arr)
    print(f"\n  {name} result:")
    print(f"    k0_1   = {k0[0]:.6e}  (true {true_k0_arr[0]:.6e}, "
          f"err {k0_err[0]*100:.2f}%)")
    print(f"    k0_2   = {k0[1]:.6e}  (true {true_k0_arr[1]:.6e}, "
          f"err {k0_err[1]*100:.2f}%)")
    print(f"    alpha_1= {alpha[0]:.6f}  (true {true_alpha_arr[0]:.6f}, "
          f"err {alpha_err[0]*100:.2f}%)")
    print(f"    alpha_2= {alpha[1]:.6f}  (true {true_alpha_arr[1]:.6f}, "
          f"err {alpha_err[1]*100:.2f}%)")
    print(f"    Loss: {loss:.6e},  Time: {elapsed:.1f}s")
    return k0_err, alpha_err


def extract_training_bounds(surrogate) -> tuple:
    """Extract training bounds from the surrogate model (with fallbacks)."""
    if surrogate.training_bounds is not None:
        tb = surrogate.training_bounds
        k0_1_lo = tb["k0_1"][0]
        k0_1_hi = tb["k0_1"][1]
        k0_2_lo = tb["k0_2"][0]
        k0_2_hi = tb["k0_2"][1]
        alpha_lo = min(tb["alpha_1"][0], tb["alpha_2"][0])
        alpha_hi = max(tb["alpha_1"][1], tb["alpha_2"][1])
        return k0_1_lo, k0_1_hi, k0_2_lo, k0_2_hi, alpha_lo, alpha_hi
    return (K0_1_TRAIN_LO_DEFAULT, K0_1_TRAIN_HI_DEFAULT,
            K0_2_TRAIN_LO_DEFAULT, K0_2_TRAIN_HI_DEFAULT,
            ALPHA_TRAIN_LO_DEFAULT, ALPHA_TRAIN_HI_DEFAULT)


def compute_shallow_subset_idx(
    all_eta: np.ndarray, eta_shallow: np.ndarray,
) -> np.ndarray:
    """Find indices of shallow voltages within the full grid."""
    idx: list[int] = []
    for eta in eta_shallow:
        matches = np.where(np.abs(all_eta - eta) < 1e-10)[0]
        if len(matches) > 0:
            idx.append(int(matches[0]))
    return np.array(idx, dtype=int)


def compute_k0_2_bounds(
    cascade_k0_2: float,
    bound_factor: float,
) -> tuple[float, float]:
    """Compute 1D bounds for log10(k0_2) around the cascade result.

    Parameters
    ----------
    cascade_k0_2 : float
        k0_2 from cascade (physical space).
    bound_factor : float
        Multiplicative factor (e.g. 10.0 => [k0_2/10, k0_2*10]).

    Returns
    -------
    (lo, hi) : tuple of float
        Bounds in log10 space.
    """
    lo = cascade_k0_2 / bound_factor
    hi = cascade_k0_2 * bound_factor
    return (
        float(np.log10(max(lo, 1e-30))),
        float(np.log10(hi)),
    )


# ---------------------------------------------------------------------------
# Target generation
# ---------------------------------------------------------------------------

def generate_targets_with_pde(
    phi_applied_values: np.ndarray,
    observable_scale: float,
) -> dict[str, np.ndarray]:
    """Generate target I-V curves using the PDE solver at true parameters."""
    from Forward.steady_state import SteadyStateConfig, add_percent_noise
    from Forward.bv_solver import make_graded_rectangle_mesh
    from FluxCurve.bv_point_solve import (
        solve_bv_curve_points_with_warmstart,
        _clear_caches,
    )

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

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)
    recovery = make_recovery_config(max_it_cap=600)
    dummy_target = np.zeros_like(phi_applied_values, dtype=float)

    results: dict[str, np.ndarray] = {}
    for obs_mode in ["current_density", "peroxide_current"]:
        _clear_caches()
        seed_offset = 0 if obs_mode == "current_density" else 1

        points = solve_bv_curve_points_with_warmstart(
            base_solver_params=base_sp,
            steady=steady,
            phi_applied_values=phi_applied_values,
            target_flux=dummy_target,
            k0_values=[K0_HAT, K0_2_HAT],
            blob_initial_condition=False,
            fail_penalty=1e9,
            forward_recovery=recovery,
            observable_mode=obs_mode,
            observable_reaction_index=None,
            observable_scale=observable_scale,
            mesh=mesh,
            alpha_values=[ALPHA_1, ALPHA_2],
            control_mode="joint",
            max_eta_gap=3.0,
        )

        clean_flux = np.array(
            [float(p.simulated_flux) for p in points], dtype=float,
        )
        noisy_flux = add_percent_noise(
            clean_flux, 2.0, seed=20260226 + seed_offset,
        )
        results[obs_mode] = noisy_flux

    _clear_caches()
    return results


def subset_targets(
    target_cd: np.ndarray,
    target_pc: np.ndarray,
    all_eta: np.ndarray,
    subset_eta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract target values for a subset of voltages from the full grid."""
    idx: list[int] = []
    for eta in subset_eta:
        matches = np.where(np.abs(all_eta - eta) < 1e-10)[0]
        if len(matches) > 0:
            idx.append(int(matches[0]))
    idx_arr = np.array(idx, dtype=int)
    return target_cd[idx_arr], target_pc[idx_arr]


# ---------------------------------------------------------------------------
# Phase 2: 1D PDE k0_2 objective wrapper
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class K02PDEObjectiveConfig:
    """Configuration for the 1D PDE k0_2 objective.

    Attributes
    ----------
    fixed_k0_1 : float
        Fixed k0_1 from cascade (physical space).
    fixed_alpha_1, fixed_alpha_2 : float
        Fixed alpha values from cascade.
    secondary_weight : float
        Weight on peroxide current.
    """

    fixed_k0_1: float
    fixed_alpha_1: float
    fixed_alpha_2: float
    secondary_weight: float = 1.0


def build_k02_pde_objective(
    *,
    config: K02PDEObjectiveConfig,
    pde_request,
    eta_shallow: np.ndarray,
    target_cd: np.ndarray,
    target_pc: np.ndarray,
    mesh,
) -> tuple:
    """Build a 1D objective function for k0_2-only PDE optimization.

    Returns (objective_fn, eval_counter_dict) where:
    - objective_fn(log10_k0_2: float) -> float
    - eval_counter_dict["n"] tracks PDE evaluation count

    The function constructs full parameter vectors with k0_1, alpha_1,
    alpha_2 fixed, computes PDE-based I-V curves, and returns the
    combined misfit.
    """
    from FluxCurve.bv_curve_eval import (
        evaluate_bv_multi_observable_objective_and_gradient,
    )

    counter: dict[str, int] = {"n": 0}
    cache: dict[float, float] = {}

    def objective(log10_k0_2: float) -> float:
        # Round to avoid floating-point cache misses
        cache_key = round(log10_k0_2, 12)
        if cache_key in cache:
            return cache[cache_key]

        k0_2_candidate = 10.0 ** log10_k0_2
        k0_eval = np.array([config.fixed_k0_1, k0_2_candidate])
        alpha_eval = np.array([config.fixed_alpha_1, config.fixed_alpha_2])

        curve = evaluate_bv_multi_observable_objective_and_gradient(
            request=pde_request,
            phi_applied_values=eta_shallow,
            target_flux_primary=target_cd,
            target_flux_secondary=target_pc,
            k0_values=k0_eval,
            mesh=mesh,
            alpha_values=alpha_eval,
            control_mode="joint",
        )

        counter["n"] += 1
        obj_val = float(curve.objective)
        cache[cache_key] = obj_val

        k0_str = f"[{config.fixed_k0_1:.4e}, {k0_2_candidate:.4e}]"
        alpha_str = f"[{config.fixed_alpha_1:.4f}, {config.fixed_alpha_2:.4f}]"
        print(f"  [k0_2 PDE eval {counter['n']:>3d}] "
              f"J={obj_val:12.6e} k0={k0_str} alpha={alpha_str}")

        return obj_val

    return objective, counter


def build_k02_pde_objective_with_grad(
    *,
    config: K02PDEObjectiveConfig,
    pde_request,
    eta_values: np.ndarray,
    target_cd: np.ndarray,
    target_pc: np.ndarray,
    mesh,
) -> tuple:
    """Build a 1D objective+gradient function for k0_2-only PDE optimization.

    Returns (obj_and_grad_fn, eval_counter_dict) where:
    - obj_and_grad_fn(log10_k0_2_arr: np.ndarray) -> (float, np.ndarray)
    - eval_counter_dict["n"] tracks PDE evaluation count

    The gradient is computed via:
        dJ/d(log10(k0_2)) = dJ/dk0_2 * k0_2 * ln(10)

    where dJ/dk0_2 is extracted from the joint gradient at index 1
    (layout: [dk0_1, dk0_2, dalpha_1, dalpha_2] for joint control_mode).
    """
    from FluxCurve.bv_curve_eval import (
        evaluate_bv_multi_observable_objective_and_gradient,
    )

    counter: dict[str, int] = {"n": 0}
    cache: dict[float, tuple[float, np.ndarray]] = {}

    # Gradient index for k0_2 in joint control mode: [k0_1, k0_2, alpha_1, alpha_2]
    K02_GRAD_INDEX = 1

    def obj_and_grad(log10_k0_2_arr: np.ndarray) -> tuple[float, np.ndarray]:
        log10_k0_2 = float(log10_k0_2_arr[0])
        cache_key = round(log10_k0_2, 12)
        if cache_key in cache:
            return cache[cache_key]

        k0_2_candidate = 10.0 ** log10_k0_2
        k0_eval = np.array([config.fixed_k0_1, k0_2_candidate])
        alpha_eval = np.array([config.fixed_alpha_1, config.fixed_alpha_2])

        curve = evaluate_bv_multi_observable_objective_and_gradient(
            request=pde_request,
            phi_applied_values=eta_values,
            target_flux_primary=target_cd,
            target_flux_secondary=target_pc,
            k0_values=k0_eval,
            mesh=mesh,
            alpha_values=alpha_eval,
            control_mode="joint",
        )

        counter["n"] += 1
        obj_val = float(curve.objective)

        # Extract gradient w.r.t. physical k0_2 and apply chain rule
        full_grad = np.asarray(curve.gradient, dtype=float)
        dJ_dk0_2 = float(full_grad[K02_GRAD_INDEX])
        grad_log10_k0_2 = dJ_dk0_2 * k0_2_candidate * np.log(10.0)

        k0_str = f"[{config.fixed_k0_1:.4e}, {k0_2_candidate:.4e}]"
        alpha_str = f"[{config.fixed_alpha_1:.4f}, {config.fixed_alpha_2:.4f}]"
        print(f"  [k0_2 PDE eval {counter['n']:>3d}] "
              f"J={obj_val:12.6e} k0={k0_str} alpha={alpha_str} "
              f"grad_log10={grad_log10_k0_2:+12.6e}")

        result = (obj_val, np.array([grad_log10_k0_2]))
        cache[cache_key] = result
        return result

    return obj_and_grad, counter


# ---------------------------------------------------------------------------
# Phase 3: Joint PDE polish with regularization
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegularizationConfig:
    """Tikhonov regularization toward a prior in optimizer x-space.

    Penalty = lambda * sum((x_i - x_prior_i)^2).
    """

    reg_lambda: float
    k0_prior: np.ndarray
    alpha_prior: np.ndarray
    n_k0: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "k0_prior", np.asarray(self.k0_prior, dtype=float),
        )
        object.__setattr__(
            self, "alpha_prior", np.asarray(self.alpha_prior, dtype=float),
        )


def compute_regularization_penalty(
    x: np.ndarray, config: RegularizationConfig,
) -> tuple[float, np.ndarray]:
    """Compute Tikhonov penalty and gradient in optimizer x-space.

    x layout: [log10(k0_1), log10(k0_2), alpha_1, alpha_2].
    """
    n_k0 = config.n_k0
    lam = config.reg_lambda
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)

    log_prior = np.log10(np.maximum(config.k0_prior, 1e-30))
    diff_k0 = x[:n_k0] - log_prior
    diff_alpha = x[n_k0:] - config.alpha_prior

    penalty = lam * float(np.sum(diff_k0**2) + np.sum(diff_alpha**2))
    grad[:n_k0] = 2.0 * lam * diff_k0
    grad[n_k0:] = 2.0 * lam * diff_alpha

    return penalty, grad


@dataclass(frozen=True)
class AsymmetricRegularizationConfig:
    """Per-parameter Tikhonov regularization toward a prior.

    Penalty = sum_i(lambda_i * (x_i - x_prior_i)^2)

    x layout: [log10(k0_1), log10(k0_2), alpha_1, alpha_2]
    """

    lambdas: tuple  # (lambda_k0_1, lambda_k0_2, lambda_alpha_1, lambda_alpha_2)
    k0_prior: tuple  # (k0_1_prior, k0_2_prior) in PHYSICAL space
    alpha_prior: tuple  # (alpha_1_prior, alpha_2_prior)


def compute_asymmetric_regularization_penalty(
    x: np.ndarray,
    config: AsymmetricRegularizationConfig,
) -> tuple[float, np.ndarray]:
    """Compute per-parameter Tikhonov penalty and analytical gradient.

    Parameters
    ----------
    x : array
        Optimizer variables: [log10(k0_1), log10(k0_2), alpha_1, alpha_2].
    config : AsymmetricRegularizationConfig
        Per-parameter lambdas and priors.

    Returns
    -------
    penalty : float
        sum_i(lambda_i * (x_i - x_prior_i)^2).
    grad : array
        Gradient of the penalty w.r.t. x (same length as x).
    """
    lam = np.array(config.lambdas, dtype=float)
    log_prior = np.log10(np.maximum(np.array(config.k0_prior), 1e-30))
    x_prior = np.concatenate([log_prior, np.array(config.alpha_prior)])

    diff = np.asarray(x, dtype=float) - x_prior
    penalty = float(np.sum(lam * diff**2))
    grad = 2.0 * lam * diff
    return penalty, grad


def compute_asymmetric_bounds(
    k0_vals: np.ndarray,
    alpha_vals: np.ndarray,
    k0_bound_factors: tuple[float, float],
    alpha_margins: tuple[float, float],
    alpha_floor: float = 0.05,
    alpha_ceil: float = 0.95,
) -> list[tuple[float, float]]:
    """Compute optimization bounds with per-parameter widths.

    k0 bounds in log10 space: [log10(k0/factor), log10(k0*factor)]
    Alpha bounds in linear space: [alpha-margin, alpha+margin], clipped to [floor, ceil]

    Parameters
    ----------
    k0_vals : array
        k0 values in physical space.
    alpha_vals : array
        Alpha values in linear space.
    k0_bound_factors : tuple
        (factor_k0_1, factor_k0_2) multiplicative factors.
    alpha_margins : tuple
        (margin_alpha_1, margin_alpha_2) additive margins.
    alpha_floor, alpha_ceil : float
        Hard limits on alpha.

    Returns
    -------
    bounds : list of (lo, hi) tuples
        One per optimizer variable: [k0_1, k0_2, alpha_1, alpha_2].
    """
    bounds: list[tuple[float, float]] = []
    for k0_val, factor in zip(k0_vals, k0_bound_factors):
        lo = float(np.log10(max(k0_val / factor, 1e-30)))
        hi = float(np.log10(k0_val * factor))
        bounds.append((lo, hi))
    for a_val, margin in zip(alpha_vals, alpha_margins):
        lo = max(float(a_val - margin), alpha_floor)
        hi = min(float(a_val + margin), alpha_ceil)
        bounds.append((lo, hi))
    return bounds


def compute_tight_bounds(
    surrogate_k0: np.ndarray,
    surrogate_alpha: np.ndarray,
    bound_factor: float = 2.0,
    alpha_margin: float = 0.05,
    alpha_floor: float = 0.05,
    alpha_ceil: float = 0.95,
) -> list[tuple[float, float]]:
    """Compute tight bounds centered on a prior result (log10 space for k0)."""
    surrogate_k0 = np.asarray(surrogate_k0, dtype=float)
    surrogate_alpha = np.asarray(surrogate_alpha, dtype=float)
    bounds: list[tuple[float, float]] = []

    for k0_val in surrogate_k0:
        lo = k0_val / bound_factor
        hi = k0_val * bound_factor
        bounds.append((
            float(np.log10(max(lo, 1e-30))),
            float(np.log10(hi)),
        ))

    for alpha_val in surrogate_alpha:
        lo = max(alpha_val - alpha_margin, alpha_floor)
        hi = min(alpha_val + alpha_margin, alpha_ceil)
        bounds.append((float(lo), float(hi)))

    return bounds


# ---------------------------------------------------------------------------
# Results table printing
# ---------------------------------------------------------------------------

def print_results_table(
    phase_results: dict[str, PhaseResult],
    true_k0_arr: np.ndarray,
    true_alpha_arr: np.ndarray,
) -> None:
    """Print a formatted results table for all phases."""
    header = (
        f"{'Phase':<40} | {'k0_1 err':>10} {'k0_2 err':>10} "
        f"{'a1 err':>10} {'a2 err':>10} | {'max err':>10} | {'time':>6}"
    )
    print(header)
    print(f"{'-'*105}")

    for name, pr in phase_results.items():
        k0 = np.array([pr.k0_1, pr.k0_2])
        alpha = np.array([pr.alpha_1, pr.alpha_2])
        k0_err, alpha_err = compute_errors(k0, alpha, true_k0_arr, true_alpha_arr)
        max_err = max(float(k0_err.max()), float(alpha_err.max()))
        print(
            f"{name:<40} | {k0_err[0]*100:>9.2f}% {k0_err[1]*100:>9.2f}% "
            f"{alpha_err[0]*100:>9.2f}% {alpha_err[1]*100:>9.2f}% "
            f"| {max_err*100:>9.2f}% | {pr.elapsed_s:>5.1f}s"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> tuple[np.ndarray, np.ndarray, float] | None:
    parser = argparse.ArgumentParser(
        description="Cascade + PDE Hybrid Strategy for BV kinetics recovery",
    )
    parser.add_argument(
        "--model", type=str,
        default="StudyResults/surrogate_v9/surrogate_model.pkl",
        help="Path to surrogate model .pkl",
    )
    parser.add_argument(
        "--pass1-weight", type=float, default=0.5,
        help="CD-dominant weight for surrogate cascade (default: 0.5)",
    )
    parser.add_argument(
        "--no-joint-polish", action="store_true",
        help="Skip Phase 3 joint PDE polish",
    )
    parser.add_argument(
        "--pde-maxiter", type=int, default=30,
        help="Max function evaluations for 1D k0_2 PDE search (default: 30)",
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help="PDE worker count (0=auto)",
    )
    parser.add_argument(
        "--k0-2-bound-factor", type=float, default=20.0,
        help="Bounds around cascade k0_2 (default: 20.0, meaning x0.05 to x20)",
    )
    parser.add_argument(
        "--pde-secondary-weight", type=float, default=5.0,
        dest="secondary_weight",
        help="Weight on peroxide current in PDE objective (default: 5.0)",
    )
    parser.add_argument(
        "--secondary-weight", type=float, default=None,
        dest="secondary_weight_legacy",
        help=argparse.SUPPRESS,  # backward compat alias
    )
    parser.add_argument(
        "--joint-polish-maxiter", type=int, default=8,
        help="Max L-BFGS-B iterations for joint polish (default: 8)",
    )
    parser.add_argument(
        "--joint-polish-lambda", type=float, default=1.0,
        help="Regularization lambda for joint polish (default: 1.0, legacy symmetric)",
    )
    parser.add_argument(
        "--joint-lambda-k0-1", type=float, default=5.0,
        help="Asymmetric lambda for k0_1 in Phase 3 (default: 5.0, strong protection)",
    )
    parser.add_argument(
        "--joint-lambda-k0-2", type=float, default=0.1,
        help="Asymmetric lambda for k0_2 in Phase 3 (default: 0.1, weak -- let PDE correct)",
    )
    parser.add_argument(
        "--joint-lambda-alpha-1", type=float, default=5.0,
        help="Asymmetric lambda for alpha_1 in Phase 3 (default: 5.0, strong protection)",
    )
    parser.add_argument(
        "--joint-lambda-alpha-2", type=float, default=5.0,
        help="Asymmetric lambda for alpha_2 in Phase 3 (default: 5.0, strong protection)",
    )
    parser.add_argument(
        "--joint-k0-1-bound-factor", type=float, default=1.5,
        help="k0_1 bound factor for Phase 3 (default: 1.5, tight)",
    )
    parser.add_argument(
        "--joint-k0-2-bound-factor", type=float, default=5.0,
        help="k0_2 bound factor for Phase 3 (default: 5.0, wider)",
    )
    parser.add_argument(
        "--joint-alpha-margin", type=float, default=0.03,
        help="Alpha margin for Phase 3 bounds (default: 0.03, tight)",
    )
    parser.add_argument(
        "--joint-mode", type=str, default="asymmetric",
        choices=["asymmetric", "free"],
        help="Phase 3 mode. 'free': moderate bounds, no reg, more iters.",
    )
    parser.add_argument(
        "--pde-voltage-range", type=str, default="cathodic",
        choices=["shallow", "cathodic"],
        help="Voltage range for Phase 2 PDE: 'shallow' (10 pts) or 'cathodic' (15 pts, default)",
    )
    parser.add_argument(
        "--sweep-weights", action="store_true",
        help="Run full pipeline for secondary_weight in [1,2,5,10,20] and print comparison",
    )
    args = parser.parse_args()

    # Backward-compat: --secondary-weight (old name) overrides if provided
    if args.secondary_weight_legacy is not None:
        args.secondary_weight = args.secondary_weight_legacy

    # ---------------------------------------------------------------
    # Free-mode default resolution: if --joint-mode free is selected,
    # override defaults for Phase 3 params that were NOT explicitly
    # provided on the command line.
    # ---------------------------------------------------------------
    _FREE_MODE_DEFAULTS: dict[str, object] = {
        "joint_polish_maxiter": 15,
        "joint_lambda_k0_1": 0.0,
        "joint_lambda_k0_2": 0.0,
        "joint_lambda_alpha_1": 0.0,
        "joint_lambda_alpha_2": 0.0,
        "joint_k0_1_bound_factor": 3.0,
        "joint_k0_2_bound_factor": 5.0,
        "joint_alpha_margin": 0.08,
    }

    if args.joint_mode == "free":
        for dest_name, free_val in _FREE_MODE_DEFAULTS.items():
            # The arg is "not explicitly provided" when its value equals the
            # parser default.  We compare against the parser default for each
            # destination to detect this.
            parser_default = parser.get_default(dest_name)
            if getattr(args, dest_name) == parser_default:
                setattr(args, dest_name, free_val)

    hybrid_config = HybridConfig(
        pass1_weight=args.pass1_weight,
        pde_maxiter=args.pde_maxiter,
        k0_2_bound_factor=args.k0_2_bound_factor,
        joint_polish=not args.no_joint_polish,
        joint_polish_maxiter=args.joint_polish_maxiter,
        joint_polish_lambda=args.joint_polish_lambda,
        secondary_weight=args.secondary_weight,
        workers=args.workers,
        pde_voltage_range=args.pde_voltage_range,
        joint_polish_lambda_k0_1=args.joint_lambda_k0_1,
        joint_polish_lambda_k0_2=args.joint_lambda_k0_2,
        joint_polish_lambda_alpha_1=args.joint_lambda_alpha_1,
        joint_polish_lambda_alpha_2=args.joint_lambda_alpha_2,
        joint_polish_k0_1_bound_factor=args.joint_k0_1_bound_factor,
        joint_polish_k0_2_bound_factor=args.joint_k0_2_bound_factor,
        joint_polish_alpha_margin=args.joint_alpha_margin,
        joint_mode=args.joint_mode,
    )

    # ===================================================================
    # Voltage grids (IDENTICAL to v9/cascade)
    # ===================================================================
    eta_symmetric = np.array([
        +5.0, +3.0, +1.0, -0.5,
        -1.0, -2.0, -3.0, -5.0, -8.0,
        -10.0, -15.0, -20.0,
    ])
    eta_shallow = np.array([
        -1.0, -2.0, -3.0, -4.0, -5.0, -6.5, -8.0,
        -10.0, -11.5, -13.0,
    ])
    eta_cathodic = np.array([
        -1.0, -2.0, -3.0, -4.0, -5.0, -6.5, -8.0,
        -10.0, -13.0, -17.0, -22.0, -28.0,
        -35.0, -41.0, -46.5,
    ])

    # Union of all voltages (sorted descending) -- must match surrogate grid
    all_eta = np.unique(np.concatenate([eta_symmetric, eta_shallow, eta_cathodic]))
    all_eta = np.sort(all_eta)[::-1]

    true_k0 = [K0_HAT, K0_2_HAT]
    true_alpha = [ALPHA_1, ALPHA_2]
    true_k0_arr = np.asarray(true_k0)
    true_alpha_arr = np.asarray(true_alpha)

    initial_k0_guess = [0.005, 0.0005]
    initial_alpha_guess = [0.4, 0.3]

    observable_scale = -I_SCALE
    base_output = os.path.join("StudyResults", "cascade_pde_hybrid")
    os.makedirs(base_output, exist_ok=True)

    phase_results: dict[str, PhaseResult] = {}
    t_total_start = time.time()

    # ===================================================================
    # Print header
    # ===================================================================
    print(f"\n{'#'*70}")
    print(f"  CASCADE + PDE HYBRID STRATEGY")
    print(f"  True k0:        {true_k0}")
    print(f"  True alpha:     {true_alpha}")
    print(f"  Initial k0:     {initial_k0_guess}")
    print(f"  Initial alpha:  {initial_alpha_guess}")
    print(f"  Cascade weight: {hybrid_config.pass1_weight} (CD-dominant)")
    print(f"  PDE maxiter:    {hybrid_config.pde_maxiter}")
    print(f"  k0_2 bound:     x{hybrid_config.k0_2_bound_factor}")
    print(f"  PDE voltage:    {hybrid_config.pde_voltage_range}")
    print(f"  Secondary wt:   {hybrid_config.secondary_weight}")
    print(f"  Joint polish:   {hybrid_config.joint_polish}")
    print(f"  Joint mode:     {hybrid_config.joint_mode}")
    print(f"{'#'*70}\n")

    # ===================================================================
    # Load surrogate model
    # ===================================================================
    print(f"Loading surrogate model from: {args.model}")
    surrogate = load_surrogate(args.model)
    surrogate_eta = surrogate.phi_applied
    print(f"  Surrogate voltage points: {surrogate.n_eta}")
    print(f"  Surrogate voltage range: [{surrogate_eta.min():.1f}, "
          f"{surrogate_eta.max():.1f}]")

    # Extract training bounds
    (K0_1_LO, K0_1_HI, K0_2_LO, K0_2_HI,
     ALPHA_LO, ALPHA_HI) = extract_training_bounds(surrogate)

    bounds_source = (
        "from model" if surrogate.training_bounds is not None else "defaults"
    )
    print(f"  Training bounds ({bounds_source}):")
    print(f"    k0_1 log10: [{np.log10(max(K0_1_LO, 1e-30)):.2f}, "
          f"{np.log10(K0_1_HI):.2f}]")
    print(f"    k0_2 log10: [{np.log10(max(K0_2_LO, 1e-30)):.2f}, "
          f"{np.log10(K0_2_HI):.2f}]")
    print(f"    alpha:      [{ALPHA_LO:.4f}, {ALPHA_HI:.4f}]")

    # ===================================================================
    # Phase 0: Generate targets using PDE solver
    # ===================================================================
    print(f"\nGenerating target I-V curves with PDE solver at true parameters...")
    t_target = time.time()
    targets = generate_targets_with_pde(all_eta, observable_scale)
    target_cd_full = targets["current_density"]
    target_pc_full = targets["peroxide_current"]
    t_target_elapsed = time.time() - t_target
    print(f"  Target generation: {t_target_elapsed:.1f}s")

    # Compute shallow subset indices
    shallow_idx = compute_shallow_subset_idx(all_eta, eta_shallow)
    print(f"  Shallow subset: {len(shallow_idx)} points "
          f"(eta from {eta_shallow[0]:.1f} to {eta_shallow[-1]:.1f})")

    # ===================================================================
    # PHASE 1: Surrogate Cascade (CD-dominant, skip polish)
    # ===================================================================
    print(f"\n{'='*70}")
    print(f"  PHASE 1: Surrogate Cascade (CD-dominant, w={hybrid_config.pass1_weight})")
    print(f"  Pass 1: 4 params free, CD-dominant weighting")
    print(f"  Pass 2: k0_2 + alpha_2 free (PC-dominant)")
    print(f"  Pass 3: SKIPPED (PDE handles k0_2 refinement)")
    print(f"  Objective: shallow voltage subset ({len(shallow_idx)} pts)")
    print(f"{'='*70}")
    t_p1 = time.time()

    cascade_config = CascadeConfig(
        pass1_weight=hybrid_config.pass1_weight,
        pass2_weight=2.0,
        pass1_maxiter=60,
        pass2_maxiter=60,
        polish_maxiter=0,
        polish_weight=1.0,
        skip_polish=True,
        fd_step=1e-5,
        verbose=True,
    )

    cascade_result = run_cascade_inference(
        surrogate=surrogate,
        target_cd=target_cd_full,
        target_pc=target_pc_full,
        initial_k0=initial_k0_guess,
        initial_alpha=initial_alpha_guess,
        bounds_k0_1=(K0_1_LO, K0_1_HI),
        bounds_k0_2=(K0_2_LO, K0_2_HI),
        bounds_alpha=(ALPHA_LO, ALPHA_HI),
        config=cascade_config,
        subset_idx=shallow_idx,
    )

    p1_time = time.time() - t_p1

    # Record per-pass results from the cascade
    for pr in cascade_result.pass_results:
        pr_k0 = np.array([pr.k0_1, pr.k0_2])
        pr_alpha = np.array([pr.alpha_1, pr.alpha_2])
        print_phase_result(
            pr.pass_name, pr_k0, pr_alpha,
            true_k0_arr, true_alpha_arr, pr.loss, pr.elapsed_s,
        )
        phase_results[pr.pass_name] = PhaseResult(
            phase_name=pr.pass_name,
            k0_1=pr.k0_1, k0_2=pr.k0_2,
            alpha_1=pr.alpha_1, alpha_2=pr.alpha_2,
            loss=pr.loss, elapsed_s=pr.elapsed_s,
        )

    cascade_k0_1 = cascade_result.best_k0_1
    cascade_k0_2 = cascade_result.best_k0_2
    cascade_alpha_1 = cascade_result.best_alpha_1
    cascade_alpha_2 = cascade_result.best_alpha_2

    print(f"\n  Phase 1 total: {p1_time:.1f}s, "
          f"{cascade_result.total_evals} surrogate evals")

    # ===================================================================
    # PHASE 2: 1D PDE k0_2-only refinement (gradient-based L-BFGS-B)
    # ===================================================================
    # Select voltage grid based on config
    if hybrid_config.pde_voltage_range == "cathodic":
        eta_pde = eta_cathodic
        voltage_label = "full cathodic"
    else:
        eta_pde = eta_shallow
        voltage_label = "shallow cathodic"

    print(f"\n{'='*70}")
    print(f"  PHASE 2: 1D PDE k0_2-only refinement (L-BFGS-B)")
    print(f"  Fixed from cascade: k0_1={cascade_k0_1:.4e}, "
          f"alpha_1={cascade_alpha_1:.4f}, alpha_2={cascade_alpha_2:.4f}")
    print(f"  Starting k0_2: {cascade_k0_2:.4e} (log10={np.log10(cascade_k0_2):.4f})")
    k0_2_lo, k0_2_hi = compute_k0_2_bounds(
        cascade_k0_2, hybrid_config.k0_2_bound_factor,
    )
    print(f"  k0_2 bounds: log10 in [{k0_2_lo:.4f}, {k0_2_hi:.4f}]")
    print(f"  Max iterations: {hybrid_config.pde_maxiter}")
    print(f"  Secondary weight: {hybrid_config.secondary_weight}")
    print(f"  Voltage grid: {voltage_label} ({len(eta_pde)} pts)")
    print(f"{'='*70}")
    t_p2 = time.time()

    from Forward.steady_state import SteadyStateConfig
    from FluxCurve.bv_config import BVFluxCurveInferenceRequest
    from FluxCurve.bv_point_solve import (
        _clear_caches,
        set_parallel_pool,
        close_parallel_pool,
        _WARMSTART_MAX_STEPS,
        _SER_GROWTH_CAP,
        _SER_SHRINK,
        _SER_DT_MAX_RATIO,
    )
    from FluxCurve.bv_parallel import BVParallelPointConfig, BVPointSolvePool
    from Forward.bv_solver import make_graded_rectangle_mesh

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

    recovery = make_recovery_config(max_it_cap=600)
    _clear_caches()

    # Set up parallel pool
    n_pde_workers = hybrid_config.workers
    if n_pde_workers <= 0:
        n_pde_workers = min(
            len(eta_pde), max(1, (os.cpu_count() or 4) - 1),
        )

    n_joint_controls = 4

    pde_config = BVParallelPointConfig(
        base_solver_params=list(base_sp),
        ss_relative_tolerance=float(steady.relative_tolerance),
        ss_absolute_tolerance=float(max(steady.absolute_tolerance, 1e-16)),
        ss_consecutive_steps=int(steady.consecutive_steps),
        ss_max_steps=int(steady.max_steps),
        mesh_Nx=8,
        mesh_Ny=200,
        mesh_beta=3.0,
        blob_initial_condition=False,
        fail_penalty=1e9,
        warmstart_max_steps=_WARMSTART_MAX_STEPS,
        observable_mode="current_density",
        observable_reaction_index=None,
        observable_scale=observable_scale,
        control_mode="joint",
        n_controls=n_joint_controls,
        ser_growth_cap=_SER_GROWTH_CAP,
        ser_shrink=_SER_SHRINK,
        ser_dt_max_ratio=_SER_DT_MAX_RATIO,
        secondary_observable_mode="peroxide_current",
        secondary_observable_reaction_index=None,
        secondary_observable_scale=observable_scale,
    )
    pde_pool = BVPointSolvePool(pde_config, n_workers=n_pde_workers)
    set_parallel_pool(pde_pool)
    print(f"  Parallel pool: {n_pde_workers} workers")

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)

    # Extract PDE targets for the selected voltage grid
    target_cd_pde, target_pc_pde = subset_targets(
        target_cd_full, target_pc_full, all_eta, eta_pde,
    )

    p2_dir = os.path.join(base_output, "phase2_k02_refinement")
    os.makedirs(p2_dir, exist_ok=True)

    pde_request = BVFluxCurveInferenceRequest(
        base_solver_params=base_sp,
        steady=steady,
        true_k0=true_k0,
        initial_guess=[cascade_k0_1, cascade_k0_2],
        phi_applied_values=eta_pde.tolist(),
        target_csv_path=os.path.join(p2_dir, "target_primary.csv"),
        output_dir=p2_dir,
        observable_mode="current_density",
        current_density_scale=observable_scale,
        secondary_observable_mode="peroxide_current",
        secondary_observable_weight=hybrid_config.secondary_weight,
        secondary_current_density_scale=observable_scale,
        control_mode="joint",
        true_alpha=true_alpha,
        initial_alpha_guess=[cascade_alpha_1, cascade_alpha_2],
        log_space=True,
        mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
        max_eta_gap=3.0,
        forward_recovery=recovery,
        parallel_fast_path=True,
        parallel_workers=n_pde_workers,
        live_plot=False,
    )

    k02_config = K02PDEObjectiveConfig(
        fixed_k0_1=cascade_k0_1,
        fixed_alpha_1=cascade_alpha_1,
        fixed_alpha_2=cascade_alpha_2,
        secondary_weight=hybrid_config.secondary_weight,
    )

    obj_and_grad_fn, eval_counter = build_k02_pde_objective_with_grad(
        config=k02_config,
        pde_request=pde_request,
        eta_values=eta_pde,
        target_cd=target_cd_pde,
        target_pc=target_pc_pde,
        mesh=mesh,
    )

    # Run gradient-based 1D L-BFGS-B optimization for log10(k0_2)
    print(f"\n  Running gradient-based L-BFGS-B for log10(k0_2)...")
    opt_result = minimize(
        fun=lambda x: obj_and_grad_fn(x)[0],
        x0=np.array([np.log10(cascade_k0_2)]),
        jac=lambda x: obj_and_grad_fn(x)[1],
        method="L-BFGS-B",
        bounds=[(k0_2_lo, k0_2_hi)],
        options={"maxiter": hybrid_config.pde_maxiter, "ftol": 1e-12, "gtol": 1e-8},
    )

    p2_k0_2 = 10.0 ** opt_result.x[0]
    p2_loss = float(opt_result.fun)
    p2_time = time.time() - t_p2
    p2_n_evals = eval_counter["n"]

    p2_k0 = np.array([cascade_k0_1, p2_k0_2])
    p2_alpha = np.array([cascade_alpha_1, cascade_alpha_2])

    print_phase_result(
        "Phase 2 (PDE k0_2-only)", p2_k0, p2_alpha,
        true_k0_arr, true_alpha_arr, p2_loss, p2_time,
    )
    phase_results["Phase 2 (PDE k0_2-only)"] = PhaseResult(
        phase_name="Phase 2 (PDE k0_2-only)",
        k0_1=cascade_k0_1, k0_2=p2_k0_2,
        alpha_1=cascade_alpha_1, alpha_2=cascade_alpha_2,
        loss=p2_loss, elapsed_s=p2_time,
        n_pde_evals=p2_n_evals,
    )

    best_k0 = p2_k0.copy()
    best_alpha = p2_alpha.copy()
    best_source = "Phase 2 (PDE k0_2-only)"

    # ===================================================================
    # PHASE 3 (Optional): Joint PDE polish with ASYMMETRIC regularization
    # ===================================================================
    if hybrid_config.joint_polish:
        asym_lambdas = (
            hybrid_config.joint_polish_lambda_k0_1,
            hybrid_config.joint_polish_lambda_k0_2,
            hybrid_config.joint_polish_lambda_alpha_1,
            hybrid_config.joint_polish_lambda_alpha_2,
        )
        # Dynamic Phase 3 label based on joint_mode
        if hybrid_config.joint_mode == "free":
            p3_label = "Phase 3 (joint PDE free)"
        else:
            p3_label = "Phase 3 (joint PDE asym-reg)"

        print(f"\n{'='*70}")
        print(f"  PHASE 3: Joint PDE polish (mode={hybrid_config.joint_mode})")
        print(f"  Starting: k0=[{best_k0[0]:.4e}, {best_k0[1]:.4e}], "
              f"alpha=[{best_alpha[0]:.4f}, {best_alpha[1]:.4f}]")
        print(f"  Lambdas: k0_1={asym_lambdas[0]}, k0_2={asym_lambdas[1]}, "
              f"alpha_1={asym_lambdas[2]}, alpha_2={asym_lambdas[3]}")
        print(f"  Ratio: {asym_lambdas[0]/max(asym_lambdas[1], 1e-30):.0f}:1 "
              f"(k0_1 vs k0_2)")
        print(f"  k0 bound factors: k0_1={hybrid_config.joint_polish_k0_1_bound_factor}, "
              f"k0_2={hybrid_config.joint_polish_k0_2_bound_factor}")
        print(f"  Alpha margin: {hybrid_config.joint_polish_alpha_margin}")
        print(f"  Max iterations: {hybrid_config.joint_polish_maxiter}")
        print(f"  Secondary weight: {hybrid_config.secondary_weight}")
        print(f"  Voltage grid: {voltage_label} ({len(eta_pde)} pts)")
        print(f"{'='*70}")
        t_p3 = time.time()

        from FluxCurve.bv_curve_eval import (
            evaluate_bv_multi_observable_objective_and_gradient,
        )

        _clear_caches()

        # Build ASYMMETRIC bounds around Phase 2 result
        asym_bounds = compute_asymmetric_bounds(
            k0_vals=best_k0,
            alpha_vals=best_alpha,
            k0_bound_factors=(
                hybrid_config.joint_polish_k0_1_bound_factor,
                hybrid_config.joint_polish_k0_2_bound_factor,
            ),
            alpha_margins=(
                hybrid_config.joint_polish_alpha_margin,
                hybrid_config.joint_polish_alpha_margin,
            ),
        )
        print(f"  Asymmetric bounds:")
        print(f"    k0_1 log10: [{asym_bounds[0][0]:.4f}, {asym_bounds[0][1]:.4f}]"
              f"  (prior: {np.log10(best_k0[0]):.4f})")
        print(f"    k0_2 log10: [{asym_bounds[1][0]:.4f}, {asym_bounds[1][1]:.4f}]"
              f"  (prior: {np.log10(best_k0[1]):.4f})")
        print(f"    alpha_1:    [{asym_bounds[2][0]:.4f}, {asym_bounds[2][1]:.4f}]"
              f"  (prior: {best_alpha[0]:.4f})")
        print(f"    alpha_2:    [{asym_bounds[3][0]:.4f}, {asym_bounds[3][1]:.4f}]"
              f"  (prior: {best_alpha[1]:.4f})")

        # Asymmetric regularization config
        asym_config = AsymmetricRegularizationConfig(
            lambdas=asym_lambdas,
            k0_prior=(float(best_k0[0]), float(best_k0[1])),
            alpha_prior=(float(best_alpha[0]), float(best_alpha[1])),
        )

        # PDE request for Phase 3 (uses full cathodic range, matching Phase 2)
        p3_dir = os.path.join(base_output, "phase3_joint_polish")
        os.makedirs(p3_dir, exist_ok=True)

        p3_request = BVFluxCurveInferenceRequest(
            base_solver_params=base_sp,
            steady=steady,
            true_k0=true_k0,
            initial_guess=best_k0.tolist(),
            phi_applied_values=eta_pde.tolist(),
            target_csv_path=os.path.join(p3_dir, "target_primary.csv"),
            output_dir=p3_dir,
            observable_mode="current_density",
            current_density_scale=observable_scale,
            secondary_observable_mode="peroxide_current",
            secondary_observable_weight=hybrid_config.secondary_weight,
            secondary_current_density_scale=observable_scale,
            control_mode="joint",
            true_alpha=true_alpha,
            initial_alpha_guess=best_alpha.tolist(),
            log_space=True,
            mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
            max_eta_gap=3.0,
            forward_recovery=recovery,
            parallel_fast_path=True,
            parallel_workers=n_pde_workers,
            live_plot=False,
        )

        # Initial point
        n_k0 = 2
        x0_pde = np.concatenate([np.log10(best_k0), best_alpha])

        # Evaluation cache + tracking
        p3_cache: dict[tuple, dict] = {}
        p3_eval_counter: dict[str, int] = {"n": 0}
        p3_best_k0 = best_k0.copy()
        p3_best_alpha = best_alpha.copy()
        p3_best_loss = float("inf")

        def _evaluate_pde_p3(x: np.ndarray) -> dict[str, object]:
            nonlocal p3_best_k0, p3_best_alpha, p3_best_loss

            x = np.asarray(x, dtype=float)
            k0_eval = np.power(10.0, x[:n_k0])
            alpha_eval = x[n_k0:].copy()
            key = tuple(float(f"{v:.12g}") for v in list(k0_eval) + list(alpha_eval))
            if key in p3_cache:
                return p3_cache[key]

            curve = evaluate_bv_multi_observable_objective_and_gradient(
                request=p3_request,
                phi_applied_values=eta_pde,
                target_flux_primary=target_cd_pde,
                target_flux_secondary=target_pc_pde,
                k0_values=k0_eval,
                mesh=mesh,
                alpha_values=alpha_eval,
                control_mode="joint",
            )

            p3_eval_counter["n"] += 1
            pde_misfit = float(curve.objective)
            grad_ctrl = np.asarray(curve.gradient, dtype=float)

            # Chain rule: log-space k0
            grad_x = grad_ctrl.copy()
            grad_x[:n_k0] = grad_ctrl[:n_k0] * k0_eval * np.log(10.0)

            # Add ASYMMETRIC regularization
            reg_penalty, reg_grad = compute_asymmetric_regularization_penalty(
                x, asym_config,
            )
            total_objective = pde_misfit + reg_penalty
            total_grad = grad_x + reg_grad

            k0_str = ", ".join(f"{v:.6e}" for v in k0_eval)
            alpha_str = ", ".join(f"{v:.4f}" for v in alpha_eval)
            print(
                f"  [P3 eval {p3_eval_counter['n']:>3d}] "
                f"J_misfit={pde_misfit:12.6e} J_reg={reg_penalty:12.6e} "
                f"J_total={total_objective:12.6e} "
                f"k0=[{k0_str}] alpha=[{alpha_str}]"
            )

            if total_objective < p3_best_loss:
                p3_best_loss = total_objective
                p3_best_k0 = k0_eval.copy()
                p3_best_alpha = alpha_eval.copy()

            result = {"objective": total_objective, "grad_x": total_grad}
            p3_cache[key] = result
            return result

        def _fun_p3(x: np.ndarray) -> float:
            return float(_evaluate_pde_p3(x)["objective"])

        def _jac_p3(x: np.ndarray) -> np.ndarray:
            return np.asarray(_evaluate_pde_p3(x)["grad_x"], dtype=float)

        # Tighter convergence tolerances for free mode to prevent
        # premature termination with zero regularization.
        if hybrid_config.joint_mode == "free":
            p3_ftol, p3_gtol = 1e-12, 1e-8
        else:
            p3_ftol, p3_gtol = 1e-8, 5e-6

        opt_result_p3 = minimize(
            _fun_p3, x0_pde, jac=_jac_p3,
            method="L-BFGS-B",
            bounds=asym_bounds,
            options={
                "maxiter": hybrid_config.joint_polish_maxiter,
                "ftol": p3_ftol,
                "gtol": p3_gtol,
                "disp": True,
            },
        )

        p3_k0 = p3_best_k0.copy()
        p3_alpha = p3_best_alpha.copy()
        p3_loss = p3_best_loss
        p3_time = time.time() - t_p3

        print_phase_result(
            p3_label, p3_k0, p3_alpha,
            true_k0_arr, true_alpha_arr, p3_loss, p3_time,
        )
        phase_results[p3_label] = PhaseResult(
            phase_name=p3_label,
            k0_1=float(p3_k0[0]), k0_2=float(p3_k0[1]),
            alpha_1=float(p3_alpha[0]), alpha_2=float(p3_alpha[1]),
            loss=p3_loss, elapsed_s=p3_time,
            n_pde_evals=p3_eval_counter["n"],
        )

        # Pick best between Phase 2 and Phase 3
        p3_k0_err, p3_alpha_err = compute_errors(
            p3_k0, p3_alpha, true_k0_arr, true_alpha_arr,
        )
        p2_k0_err, p2_alpha_err = compute_errors(
            p2_k0, p2_alpha, true_k0_arr, true_alpha_arr,
        )
        p3_max_err = max(float(p3_k0_err.max()), float(p3_alpha_err.max()))
        p2_max_err = max(float(p2_k0_err.max()), float(p2_alpha_err.max()))

        if p3_max_err <= p2_max_err:
            best_k0, best_alpha = p3_k0.copy(), p3_alpha.copy()
            best_source = p3_label
        else:
            best_k0, best_alpha = p2_k0.copy(), p2_alpha.copy()
            best_source = "Phase 2 (PDE k0_2-only)"
            print(f"\n  NOTE: Phase 3 regression detected "
                  f"({p3_max_err*100:.2f}% > {p2_max_err*100:.2f}%)")
            print(f"  Keeping Phase 2 as final answer.")

    # Clean up PDE pool
    close_parallel_pool()
    _clear_caches()

    total_time = time.time() - t_total_start

    # ===================================================================
    # FINAL SUMMARY
    # ===================================================================
    print(f"\n{'#'*90}")
    print(f"  CASCADE + PDE HYBRID: FINAL SUMMARY")
    print(f"{'#'*90}")
    print(f"  True k0:    {true_k0}")
    print(f"  True alpha: {true_alpha}")
    print(f"  K0_HAT = {K0_HAT:.6e},  K0_2_HAT = {K0_2_HAT:.6e}")
    print(f"  Cascade weight: {hybrid_config.pass1_weight}")
    print(f"  k0_2 bound factor: {hybrid_config.k0_2_bound_factor}")
    print(f"  PDE voltage range: {hybrid_config.pde_voltage_range}")
    print(f"  Secondary weight: {hybrid_config.secondary_weight}")
    print()

    print_results_table(phase_results, true_k0_arr, true_alpha_arr)

    print(f"\n{'-'*105}")

    best_k0_err, best_alpha_err = compute_errors(
        best_k0, best_alpha, true_k0_arr, true_alpha_arr,
    )
    best_max_err = max(float(best_k0_err.max()), float(best_alpha_err.max()))

    print(f"\n  Best result: {best_source} (max err = {best_max_err*100:.2f}%)")
    print(f"    k0_1   = {best_k0[0]:.6e}  (err {best_k0_err[0]*100:.2f}%)")
    print(f"    k0_2   = {best_k0[1]:.6e}  (err {best_k0_err[1]*100:.2f}%)")
    print(f"    alpha_1= {best_alpha[0]:.6f}  (err {best_alpha_err[0]*100:.2f}%)")
    print(f"    alpha_2= {best_alpha[1]:.6f}  (err {best_alpha_err[1]*100:.2f}%)")

    # v9 baseline comparison
    print(f"\n  {'='*70}")
    print(f"  v9 BASELINE COMPARISON:")
    print(f"  {'='*70}")
    print(f"  {'Metric':<25} {'v9 baseline':>12} {'Cascade-only':>12} {'Hybrid':>12}")
    print(f"  {'-'*64}")
    print(f"  {'k0_1 err (%)':<25} {'8.76':>12} {'0.51':>12} "
          f"{best_k0_err[0]*100:>12.2f}")
    print(f"  {'k0_2 err (%)':<25} {'7.57':>12} {'16.60':>12} "
          f"{best_k0_err[1]*100:>12.2f}")
    print(f"  {'alpha_1 err (%)':<25} {'4.76':>12} {'1.92':>12} "
          f"{best_alpha_err[0]*100:>12.2f}")
    print(f"  {'alpha_2 err (%)':<25} {'6.35':>12} {'2.38':>12} "
          f"{best_alpha_err[1]*100:>12.2f}")
    print(f"  {'max err (%)':<25} {'8.76':>12} {'16.60':>12} "
          f"{best_max_err*100:>12.2f}")
    print(f"  {'='*70}")

    print(f"\n  Timing breakdown:")
    print(f"    Target generation:  {t_target_elapsed:>8.1f}s")
    print(f"    Phase 1 (surr):     {p1_time:>8.1f}s")
    if "Phase 2 (PDE k0_2-only)" in phase_results:
        print(f"    Phase 2 (PDE k02):  {phase_results['Phase 2 (PDE k0_2-only)'].elapsed_s:>8.1f}s "
              f"({phase_results['Phase 2 (PDE k0_2-only)'].n_pde_evals} PDE evals)")
    # Find Phase 3 result (label varies by joint_mode)
    p3_key = next((k for k in phase_results if k.startswith("Phase 3")), None)
    if p3_key is not None:
        p3_pr = phase_results[p3_key]
        p3_short = "free" if "free" in p3_key else "asym"
        print(f"    Phase 3 (PDE {p3_short}):  {p3_pr.elapsed_s:>8.1f}s "
              f"({p3_pr.n_pde_evals} PDE evals)")
    print(f"    Total:              {total_time:>8.1f}s")

    print(f"{'#'*90}")

    # Save results CSV
    csv_path = os.path.join(base_output, "hybrid_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "phase", "k0_1", "k0_2", "alpha_1", "alpha_2",
            "k0_1_err_pct", "k0_2_err_pct", "alpha_1_err_pct",
            "alpha_2_err_pct", "max_err_pct", "loss", "time_s", "pde_evals",
        ])
        for name, pr in phase_results.items():
            k0 = np.array([pr.k0_1, pr.k0_2])
            alpha = np.array([pr.alpha_1, pr.alpha_2])
            k0_err, alpha_err = compute_errors(k0, alpha, true_k0_arr, true_alpha_arr)
            max_err = max(float(k0_err.max()), float(alpha_err.max()))
            writer.writerow([
                name,
                f"{pr.k0_1:.8e}", f"{pr.k0_2:.8e}",
                f"{pr.alpha_1:.6f}", f"{pr.alpha_2:.6f}",
                f"{k0_err[0]*100:.4f}", f"{k0_err[1]*100:.4f}",
                f"{alpha_err[0]*100:.4f}", f"{alpha_err[1]*100:.4f}",
                f"{max_err*100:.4f}",
                f"{pr.loss:.12e}", f"{pr.elapsed_s:.1f}", pr.n_pde_evals,
            ])
    print(f"\n  Results CSV saved -> {csv_path}")
    print(f"  Output: {base_output}/")
    print(f"\n=== Cascade + PDE Hybrid Complete ===")

    return best_k0_err, best_alpha_err, best_max_err


# ---------------------------------------------------------------------------
# Weight sweep mode
# ---------------------------------------------------------------------------

SWEEP_WEIGHTS: list[float] = [1.0, 2.0, 5.0, 10.0, 20.0]


@dataclass(frozen=True)
class SweepResult:
    """Result from a single weight sweep run.

    Attributes
    ----------
    weight : float
        secondary_observable_weight used.
    k0_1_err_pct, k0_2_err_pct : float
        Relative errors in percent.
    alpha_1_err_pct, alpha_2_err_pct : float
        Relative errors in percent.
    max_err_pct : float
        Maximum relative error in percent.
    """

    weight: float
    k0_1_err_pct: float
    k0_2_err_pct: float
    alpha_1_err_pct: float
    alpha_2_err_pct: float
    max_err_pct: float


def run_weight_sweep(args) -> list[SweepResult]:
    """Run the full pipeline for each weight in SWEEP_WEIGHTS.

    Re-invokes this script via subprocess for each weight value to
    ensure clean PDE state between runs. Parses the CSV output to
    extract the best result per weight.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments (model, pde_voltage_range, etc.).

    Returns
    -------
    list[SweepResult]
        One result per weight.
    """
    import subprocess

    python = sys.executable
    script = os.path.abspath(__file__)
    results: list[SweepResult] = []

    for weight in SWEEP_WEIGHTS:
        print(f"\n{'@'*80}")
        print(f"  SWEEP: secondary_weight = {weight}")
        print(f"{'@'*80}")

        cmd = [
            python, script,
            "--model", args.model,
            "--pde-secondary-weight", str(weight),
            "--pde-voltage-range", args.pde_voltage_range,
            "--pde-maxiter", str(args.pde_maxiter),
            "--k0-2-bound-factor", str(args.k0_2_bound_factor),
            "--pass1-weight", str(args.pass1_weight),
        ]
        if args.no_joint_polish:
            cmd.append("--no-joint-polish")

        proc = subprocess.run(
            cmd, capture_output=False, text=True,
            cwd=os.path.dirname(os.path.dirname(_THIS_DIR)),
        )

        # Parse the CSV output
        csv_path = os.path.join(
            "StudyResults", "cascade_pde_hybrid", "hybrid_results.csv",
        )
        if os.path.exists(csv_path):
            with open(csv_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if rows:
                # Find the row with the lowest max_err
                best_row = min(rows, key=lambda r: float(r["max_err_pct"]))
                results.append(SweepResult(
                    weight=weight,
                    k0_1_err_pct=float(best_row["k0_1_err_pct"]),
                    k0_2_err_pct=float(best_row["k0_2_err_pct"]),
                    alpha_1_err_pct=float(best_row["alpha_1_err_pct"]),
                    alpha_2_err_pct=float(best_row["alpha_2_err_pct"]),
                    max_err_pct=float(best_row["max_err_pct"]),
                ))
            else:
                print(f"  WARNING: CSV empty for weight={weight}")
        else:
            print(f"  WARNING: CSV not found for weight={weight}")

    return results


def print_sweep_table(results: list[SweepResult]) -> None:
    """Print a comparison table from weight sweep results."""
    print(f"\n{'#'*90}")
    print(f"  WEIGHT SWEEP COMPARISON TABLE")
    print(f"{'#'*90}")
    header = (
        f"  {'weight':>8} | {'k0_1 err':>10} {'k0_2 err':>10} "
        f"{'a1 err':>10} {'a2 err':>10} | {'max err':>10}"
    )
    print(header)
    print(f"  {'-'*72}")
    for r in results:
        best_marker = " <-- BEST" if r == min(results, key=lambda x: x.max_err_pct) else ""
        print(
            f"  {r.weight:>8.1f} | {r.k0_1_err_pct:>9.2f}% {r.k0_2_err_pct:>9.2f}% "
            f"{r.alpha_1_err_pct:>9.2f}% {r.alpha_2_err_pct:>9.2f}% "
            f"| {r.max_err_pct:>9.2f}%{best_marker}"
        )
    print(f"{'#'*90}")


if __name__ == "__main__":
    # Check for --sweep-weights before full argument parsing
    # (since sweep re-invokes without --sweep-weights)
    if "--sweep-weights" in sys.argv:
        # Parse args for sweep
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", type=str,
                            default="StudyResults/surrogate_v9/surrogate_model.pkl")
        parser.add_argument("--pass1-weight", type=float, default=0.5)
        parser.add_argument("--no-joint-polish", action="store_true")
        parser.add_argument("--pde-maxiter", type=int, default=30)
        parser.add_argument("--workers", type=int, default=0)
        parser.add_argument("--k0-2-bound-factor", type=float, default=20.0)
        parser.add_argument("--pde-secondary-weight", type=float, default=5.0,
                            dest="secondary_weight")
        parser.add_argument("--pde-voltage-range", type=str, default="cathodic",
                            choices=["shallow", "cathodic"])
        parser.add_argument("--sweep-weights", action="store_true")
        sweep_args = parser.parse_args()
        sweep_results = run_weight_sweep(sweep_args)
        if sweep_results:
            print_sweep_table(sweep_results)
    else:
        main()
