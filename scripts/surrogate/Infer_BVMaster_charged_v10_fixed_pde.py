"""Master inference protocol v10 -- FIXED PDE REFINEMENT.

Improvements over v9 Phase 3:
    1. Tighter bounds around surrogate optimum (±factor in log-space for k0,
       ±0.05 for alpha) to prevent large deviations.
    2. Tikhonov regularization toward the surrogate result:
       J_total = J_pde + lambda * ||x - x_surr||^2
       (log-space for k0, linear for alpha).
    3. Reduced iteration limit (default 5) -- the PDE should make small
       corrections, not a full re-optimization.

Three-phase protocol:
    Phase 0-1: Surrogate optimization (alpha warmup + joint, identical to v9)
    Phase 2: Fixed PDE refinement with tighter bounds + regularization + early stop

Usage (from PNPInverse/ directory)::

    python scripts/surrogate/Infer_BVMaster_charged_v10_fixed_pde.py \\
        --model StudyResults/surrogate_v9/surrogate_model.pkl

    # Control PDE refinement:
    python scripts/surrogate/Infer_BVMaster_charged_v10_fixed_pde.py \\
        --pde-lambda 1.0 --pde-maxiter 5 --pde-bound-factor 2.0

    # Skip PDE:
    python scripts/surrogate/Infer_BVMaster_charged_v10_fixed_pde.py --no-pde
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
    print_params_summary,
    print_redimensionalized_results,
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
from Surrogate.objectives import SurrogateObjective, AlphaOnlySurrogateObjective

# ---------------------------------------------------------------------------
# Fallback training-data bounds
# ---------------------------------------------------------------------------
K0_1_TRAIN_LO_DEFAULT = K0_HAT * 0.01
K0_1_TRAIN_HI_DEFAULT = K0_HAT * 100.0
K0_2_TRAIN_LO_DEFAULT = K0_2_HAT * 0.01
K0_2_TRAIN_HI_DEFAULT = K0_2_HAT * 100.0
ALPHA_TRAIN_LO_DEFAULT = 0.10
ALPHA_TRAIN_HI_DEFAULT = 0.90


# ===================================================================
# Regularization wrapper for PDE refinement (Fix 2)
# ===================================================================

@dataclass(frozen=True)
class RegularizationConfig:
    """Configuration for trust-region penalty toward surrogate optimum.

    The penalty is computed in the optimizer's native x-space:
    - k0: log10-space, so penalty = lambda * (log10(k0) - log10(k0_surr))^2
    - alpha: linear space, so penalty = lambda * (alpha - alpha_surr)^2
    """

    reg_lambda: float
    k0_prior: np.ndarray      # surrogate-optimal k0 (physical space)
    alpha_prior: np.ndarray    # surrogate-optimal alpha (linear space)
    n_k0: int = 2
    k0_log_space: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "k0_prior", np.asarray(self.k0_prior, dtype=float))
        object.__setattr__(self, "alpha_prior", np.asarray(self.alpha_prior, dtype=float))


def compute_regularization_penalty(
    x: np.ndarray,
    config: RegularizationConfig,
) -> tuple[float, np.ndarray]:
    """Compute Tikhonov penalty and its gradient in optimizer x-space.

    Parameters
    ----------
    x : array
        Optimizer variables: [log10(k0_1), log10(k0_2), alpha_1, alpha_2]
        when k0_log_space=True.
    config : RegularizationConfig
        Regularization settings.

    Returns
    -------
    penalty : float
        lambda * sum((x_i - x_surr_i)^2) over all parameters.
    grad_penalty : array
        Gradient of the penalty w.r.t. x (same length as x).
    """
    n_k0 = config.n_k0
    lam = config.reg_lambda
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)

    penalty = 0.0

    # k0 part (log10 space)
    if config.k0_log_space:
        log_prior = np.log10(np.maximum(config.k0_prior, 1e-30))
        diff_k0 = x[:n_k0] - log_prior
    else:
        diff_k0 = x[:n_k0] - config.k0_prior

    penalty += lam * float(np.sum(diff_k0**2))
    grad[:n_k0] = 2.0 * lam * diff_k0

    # alpha part (linear space)
    diff_alpha = x[n_k0:] - config.alpha_prior
    penalty += lam * float(np.sum(diff_alpha**2))
    grad[n_k0:] = 2.0 * lam * diff_alpha

    return penalty, grad


# ===================================================================
# Tighter bounds computation (Fix 1)
# ===================================================================

def compute_tight_bounds(
    surrogate_k0: np.ndarray,
    surrogate_alpha: np.ndarray,
    bound_factor: float = 2.0,
    alpha_margin: float = 0.05,
    alpha_floor: float = 0.05,
    alpha_ceil: float = 0.95,
    k0_log_space: bool = True,
) -> list[tuple[float, float]]:
    """Compute tight optimization bounds centered on surrogate result.

    Parameters
    ----------
    surrogate_k0 : array
        Surrogate-optimal k0 values (physical space).
    surrogate_alpha : array
        Surrogate-optimal alpha values.
    bound_factor : float
        Multiplicative factor for k0 bounds (e.g. 2.0 means [k0/2, k0*2],
        which is ±0.3 log-decades).
    alpha_margin : float
        Additive margin for alpha bounds (e.g. ±0.05).
    alpha_floor, alpha_ceil : float
        Hard limits on alpha.
    k0_log_space : bool
        If True, return bounds in log10 space for k0.

    Returns
    -------
    bounds : list of (lo, hi) tuples
        One per optimizer variable: [k0_1, k0_2, ..., alpha_1, alpha_2, ...].
    """
    surrogate_k0 = np.asarray(surrogate_k0, dtype=float)
    surrogate_alpha = np.asarray(surrogate_alpha, dtype=float)
    bounds: list[tuple[float, float]] = []

    for k0_val in surrogate_k0:
        lo = k0_val / bound_factor
        hi = k0_val * bound_factor
        if k0_log_space:
            bounds.append((float(np.log10(max(lo, 1e-30))),
                           float(np.log10(hi))))
        else:
            bounds.append((float(lo), float(hi)))

    for alpha_val in surrogate_alpha:
        lo = max(alpha_val - alpha_margin, alpha_floor)
        hi = min(alpha_val + alpha_margin, alpha_ceil)
        bounds.append((float(lo), float(hi)))

    return bounds


# ===================================================================
# Utility functions (same as v9)
# ===================================================================

def _compute_errors(
    k0: Sequence[float],
    alpha: Sequence[float],
    true_k0_arr: np.ndarray,
    true_alpha_arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    k0_arr = np.asarray(k0)
    alpha_arr = np.asarray(alpha)
    k0_err = np.abs(k0_arr - true_k0_arr) / np.maximum(np.abs(true_k0_arr), 1e-16)
    alpha_err = np.abs(alpha_arr - true_alpha_arr) / np.maximum(np.abs(true_alpha_arr), 1e-16)
    return k0_err, alpha_err


def _print_phase_result(
    name: str,
    k0: np.ndarray,
    alpha: np.ndarray,
    true_k0_arr: np.ndarray,
    true_alpha_arr: np.ndarray,
    loss: float,
    elapsed: float,
) -> tuple[np.ndarray, np.ndarray]:
    k0_err, alpha_err = _compute_errors(k0, alpha, true_k0_arr, true_alpha_arr)
    print(f"\n  {name} result:")
    print(f"    k0_1   = {k0[0]:.6e}  (true {true_k0_arr[0]:.6e}, err {k0_err[0]*100:.2f}%)")
    print(f"    k0_2   = {k0[1]:.6e}  (true {true_k0_arr[1]:.6e}, err {k0_err[1]*100:.2f}%)")
    print(f"    alpha_1= {alpha[0]:.6f}  (true {true_alpha_arr[0]:.6f}, err {alpha_err[0]*100:.2f}%)")
    print(f"    alpha_2= {alpha[1]:.6f}  (true {true_alpha_arr[1]:.6f}, err {alpha_err[1]*100:.2f}%)")
    print(f"    Loss: {loss:.6e},  Time: {elapsed:.1f}s")
    return k0_err, alpha_err


def _generate_targets_with_pde(
    phi_applied_values: np.ndarray,
    observable_scale: float,
) -> dict[str, np.ndarray]:
    """Generate target I-V curves using PDE solver at true parameters."""
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

        clean_flux = np.array([float(p.simulated_flux) for p in points], dtype=float)
        noisy_flux = add_percent_noise(clean_flux, 2.0, seed=20260226 + seed_offset)
        results[obs_mode] = noisy_flux

    _clear_caches()
    return results


def _subset_targets(
    target_cd: np.ndarray,
    target_pc: np.ndarray,
    all_eta: np.ndarray,
    subset_eta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract target values for a subset of voltages from the full grid."""
    idx = []
    for eta in subset_eta:
        matches = np.where(np.abs(all_eta - eta) < 1e-10)[0]
        if len(matches) > 0:
            idx.append(matches[0])
    idx_arr = np.array(idx, dtype=int)
    return target_cd[idx_arr], target_pc[idx_arr]


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="BV Master Inference v10 (Fixed PDE Refinement)"
    )
    parser.add_argument("--model", type=str,
                        default="StudyResults/surrogate_v9/surrogate_model.pkl",
                        help="Path to surrogate model .pkl")
    parser.add_argument("--no-pde", action="store_true",
                        help="Skip PDE refinement (surrogate-only)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Workers for PDE refinement (0=auto)")
    parser.add_argument("--pde-maxiter", type=int, default=5,
                        help="Max L-BFGS-B iterations for PDE refinement (Fix 3)")
    parser.add_argument("--pde-lambda", type=float, default=1.0,
                        help="Regularization strength toward surrogate (Fix 2)")
    parser.add_argument("--pde-bound-factor", type=float, default=2.0,
                        help="k0 bound factor: [k0/f, k0*f] (Fix 1)")
    parser.add_argument("--secondary-weight", type=float, default=1.0,
                        help="Weight on peroxide current observable")
    args = parser.parse_args()

    # ===================================================================
    # Voltage grids (identical to v9)
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

    all_eta = np.unique(np.concatenate([eta_symmetric, eta_shallow, eta_cathodic]))
    all_eta = np.sort(all_eta)[::-1]

    true_k0 = [K0_HAT, K0_2_HAT]
    true_alpha = [ALPHA_1, ALPHA_2]
    true_k0_arr = np.asarray(true_k0)
    true_alpha_arr = np.asarray(true_alpha)

    initial_k0_guess = [0.005, 0.0005]
    initial_alpha_guess = [0.4, 0.3]

    observable_scale = -I_SCALE
    base_output = os.path.join("StudyResults", "surrogate_v10")
    os.makedirs(base_output, exist_ok=True)

    phase_results: dict[str, dict] = {}
    t_total_start = time.time()

    print(f"\n{'#'*70}")
    print(f"  MASTER INFERENCE PROTOCOL v10 (FIXED PDE REFINEMENT)")
    print(f"  Fixes: tighter bounds, regularization (lambda={args.pde_lambda}),")
    print(f"         early stopping (maxiter={args.pde_maxiter})")
    print(f"  True k0:    {true_k0}")
    print(f"  True alpha: {true_alpha}")
    print(f"  Initial k0 guess:    {initial_k0_guess}")
    print(f"  Initial alpha guess: {initial_alpha_guess}")
    print(f"  PDE bound factor:    {args.pde_bound_factor}")
    print(f"  PDE lambda:          {args.pde_lambda}")
    print(f"  PDE maxiter:         {args.pde_maxiter}")
    print(f"  Secondary weight:    {args.secondary_weight}")
    print(f"{'#'*70}\n")

    # ===================================================================
    # Load surrogate model
    # ===================================================================
    print(f"Loading surrogate model from: {args.model}")
    surrogate = load_surrogate(args.model)
    surrogate_eta = surrogate.phi_applied
    print(f"  Surrogate voltage points: {surrogate.n_eta}")
    print(f"  Surrogate voltage range: [{surrogate_eta.min():.1f}, {surrogate_eta.max():.1f}]")

    # Dynamic bounds from model
    if surrogate.training_bounds is not None:
        tb = surrogate.training_bounds
        K0_1_TRAIN_LO = tb["k0_1"][0]
        K0_1_TRAIN_HI = tb["k0_1"][1]
        K0_2_TRAIN_LO = tb["k0_2"][0]
        K0_2_TRAIN_HI = tb["k0_2"][1]
        ALPHA_TRAIN_LO = tb["alpha_1"][0]
        ALPHA_TRAIN_HI = tb["alpha_1"][1]
        ALPHA_TRAIN_LO = min(ALPHA_TRAIN_LO, tb["alpha_2"][0])
        ALPHA_TRAIN_HI = max(ALPHA_TRAIN_HI, tb["alpha_2"][1])
        print(f"  Using training bounds FROM MODEL:")
    else:
        K0_1_TRAIN_LO = K0_1_TRAIN_LO_DEFAULT
        K0_1_TRAIN_HI = K0_1_TRAIN_HI_DEFAULT
        K0_2_TRAIN_LO = K0_2_TRAIN_LO_DEFAULT
        K0_2_TRAIN_HI = K0_2_TRAIN_HI_DEFAULT
        ALPHA_TRAIN_LO = ALPHA_TRAIN_LO_DEFAULT
        ALPHA_TRAIN_HI = ALPHA_TRAIN_HI_DEFAULT
        print(f"  WARNING: model lacks training_bounds, using defaults:")

    print(f"    k0_1 log10: [{np.log10(max(K0_1_TRAIN_LO, 1e-30)):.2f}, "
          f"{np.log10(K0_1_TRAIN_HI):.2f}]")
    print(f"    k0_2 log10: [{np.log10(max(K0_2_TRAIN_LO, 1e-30)):.2f}, "
          f"{np.log10(K0_2_TRAIN_HI):.2f}]")
    print(f"    alpha:      [{ALPHA_TRAIN_LO:.4f}, {ALPHA_TRAIN_HI:.4f}]")

    # ===================================================================
    # Generate targets using PDE solver
    # ===================================================================
    print(f"\nGenerating target I-V curves with PDE solver at true parameters...")
    t_target = time.time()
    targets = _generate_targets_with_pde(all_eta, observable_scale)
    target_cd_full = targets["current_density"]
    target_pc_full = targets["peroxide_current"]
    t_target_elapsed = time.time() - t_target
    print(f"  Target generation: {t_target_elapsed:.1f}s")

    target_cd_surr = target_cd_full
    target_pc_surr = target_pc_full

    # ===================================================================
    # PHASE 1: Alpha-only surrogate optimization (identical to v9)
    # ===================================================================
    print(f"\n{'='*70}")
    print(f"  PHASE 1: Alpha-only surrogate optimization")
    print(f"  k0 FIXED at initial guess: {initial_k0_guess}")
    print(f"  Voltage grid: full ({len(all_eta)} pts)")
    print(f"  Alpha bounds: [{ALPHA_TRAIN_LO}, {ALPHA_TRAIN_HI}]")
    print(f"{'='*70}")
    t_p1 = time.time()

    p1_obj = AlphaOnlySurrogateObjective(
        surrogate=surrogate,
        target_cd=target_cd_surr,
        target_pc=target_pc_surr,
        fixed_k0=initial_k0_guess,
        secondary_weight=args.secondary_weight,
        fd_step=1e-5,
    )

    x0_p1 = np.array(initial_alpha_guess, dtype=float)
    bounds_p1 = [(ALPHA_TRAIN_LO, ALPHA_TRAIN_HI),
                 (ALPHA_TRAIN_LO, ALPHA_TRAIN_HI)]

    eval_counter_p1: dict[str, int] = {"n": 0}

    def _p1_callback(x: np.ndarray) -> None:
        eval_counter_p1["n"] += 1
        j_val = p1_obj.objective(x)
        print(f"  [P1 iter {eval_counter_p1['n']:>3d}] J={j_val:.6e} "
              f"alpha=[{x[0]:.4f}, {x[1]:.4f}]")

    result_p1 = minimize(
        p1_obj.objective,
        x0_p1,
        jac=p1_obj.gradient,
        method="L-BFGS-B",
        bounds=bounds_p1,
        options={"maxiter": 60, "ftol": 1e-14, "gtol": 1e-8, "disp": False},
        callback=_p1_callback,
    )

    p1_alpha = result_p1.x.copy()
    p1_k0 = np.asarray(initial_k0_guess)
    p1_loss = float(result_p1.fun)
    p1_time = time.time() - t_p1

    _print_phase_result("Phase 1 (alpha, surrogate)", p1_k0, p1_alpha,
                        true_k0_arr, true_alpha_arr, p1_loss, p1_time)
    phase_results["Phase 1 (alpha, surrogate)"] = {
        "k0": p1_k0.tolist(), "alpha": p1_alpha.tolist(),
        "loss": p1_loss, "time": p1_time,
    }

    # ===================================================================
    # PHASE 2: Joint 4-param surrogate on shallow cathodic (identical to v9)
    # ===================================================================
    print(f"\n{'='*70}")
    print(f"  PHASE 2: Joint 4-param surrogate (shallow-range objective)")
    print(f"  k0 from COLD guess: {initial_k0_guess}")
    print(f"  alpha warm from P1: {p1_alpha.tolist()}")
    print(f"{'='*70}")
    t_p2 = time.time()

    target_cd_shallow, target_pc_shallow = _subset_targets(
        target_cd_full, target_pc_full, all_eta, eta_shallow,
    )
    shallow_idx: list[int] = []
    for eta in eta_shallow:
        matches = np.where(np.abs(all_eta - eta) < 1e-10)[0]
        if len(matches) > 0:
            shallow_idx.append(int(matches[0]))
    shallow_idx_arr = np.array(shallow_idx, dtype=int)

    class SubsetSurrogateObjective:
        """Surrogate objective on a subset of voltage points."""

        def __init__(
            self,
            surrogate_model: object,
            target_cd: np.ndarray,
            target_pc: np.ndarray,
            subset_idx: np.ndarray,
            secondary_weight: float = 1.0,
            fd_step: float = 1e-5,
            log_space_k0: bool = True,
        ) -> None:
            self.surrogate = surrogate_model
            self.target_cd = np.asarray(target_cd, dtype=float)
            self.target_pc = np.asarray(target_pc, dtype=float)
            self.subset_idx = subset_idx
            self._valid_cd = ~np.isnan(self.target_cd)
            self._valid_pc = ~np.isnan(self.target_pc)
            self.secondary_weight = secondary_weight
            self.fd_step = fd_step
            self.log_space_k0 = log_space_k0
            self._n_evals = 0

        def _x_to_params(self, x: np.ndarray) -> tuple[float, float, float, float]:
            x = np.asarray(x, dtype=float)
            if self.log_space_k0:
                k0_1, k0_2 = 10.0**x[0], 10.0**x[1]
            else:
                k0_1, k0_2 = x[0], x[1]
            return k0_1, k0_2, x[2], x[3]

        def objective(self, x: np.ndarray) -> float:
            k0_1, k0_2, a1, a2 = self._x_to_params(x)
            pred = self.surrogate.predict(k0_1, k0_2, a1, a2)
            cd_sim = pred["current_density"][self.subset_idx]
            pc_sim = pred["peroxide_current"][self.subset_idx]
            cd_diff = cd_sim[self._valid_cd] - self.target_cd[self._valid_cd]
            pc_diff = pc_sim[self._valid_pc] - self.target_pc[self._valid_pc]
            j_cd = 0.5 * float(np.sum(cd_diff**2))
            j_pc = 0.5 * float(np.sum(pc_diff**2))
            self._n_evals += 1
            return j_cd + self.secondary_weight * j_pc

        def gradient(self, x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            grad = np.zeros(len(x), dtype=float)
            h = self.fd_step
            for i in range(len(x)):
                xp, xm = x.copy(), x.copy()
                xp[i] += h
                xm[i] -= h
                grad[i] = (self.objective(xp) - self.objective(xm)) / (2 * h)
            return grad

    p2_obj = SubsetSurrogateObjective(
        surrogate_model=surrogate,
        target_cd=target_cd_shallow,
        target_pc=target_pc_shallow,
        subset_idx=shallow_idx_arr,
        secondary_weight=args.secondary_weight,
        fd_step=1e-5,
        log_space_k0=True,
    )

    x0_p2 = np.array([
        np.log10(initial_k0_guess[0]),
        np.log10(initial_k0_guess[1]),
        p1_alpha[0],
        p1_alpha[1],
    ], dtype=float)

    bounds_p2 = [
        (np.log10(K0_1_TRAIN_LO), np.log10(K0_1_TRAIN_HI)),
        (np.log10(K0_2_TRAIN_LO), np.log10(K0_2_TRAIN_HI)),
        (ALPHA_TRAIN_LO, ALPHA_TRAIN_HI),
        (ALPHA_TRAIN_LO, ALPHA_TRAIN_HI),
    ]
    print(f"  Optimizer bounds (log10 for k0):")
    print(f"    k0_1: [{bounds_p2[0][0]:.2f}, {bounds_p2[0][1]:.2f}]")
    print(f"    k0_2: [{bounds_p2[1][0]:.2f}, {bounds_p2[1][1]:.2f}]")
    print(f"    alpha: [{bounds_p2[2][0]:.2f}, {bounds_p2[2][1]:.2f}]")

    eval_counter_p2: dict[str, int] = {"n": 0}

    def _p2_callback(x: np.ndarray) -> None:
        eval_counter_p2["n"] += 1
        k0_1, k0_2 = 10.0**x[0], 10.0**x[1]
        j_val = p2_obj.objective(x)
        print(f"  [P2 iter {eval_counter_p2['n']:>3d}] J={j_val:.6e} "
              f"k0=[{k0_1:.4e},{k0_2:.4e}] alpha=[{x[2]:.4f},{x[3]:.4f}]")

    result_p2 = minimize(
        p2_obj.objective,
        x0_p2,
        jac=p2_obj.gradient,
        method="L-BFGS-B",
        bounds=bounds_p2,
        options={"maxiter": 60, "ftol": 1e-14, "gtol": 1e-8, "disp": False},
        callback=_p2_callback,
    )

    p2_k0 = np.array([10.0**result_p2.x[0], 10.0**result_p2.x[1]])
    p2_alpha = result_p2.x[2:4].copy()
    p2_loss = float(result_p2.fun)
    p2_time = time.time() - t_p2

    _print_phase_result("Phase 2 (joint, shallow, surr)", p2_k0, p2_alpha,
                        true_k0_arr, true_alpha_arr, p2_loss, p2_time)
    phase_results["Phase 2 (joint shallow surr)"] = {
        "k0": p2_k0.tolist(), "alpha": p2_alpha.tolist(),
        "loss": p2_loss, "time": p2_time,
    }

    t_surrogate_end = time.time()
    surrogate_time = t_surrogate_end - t_total_start - t_target_elapsed

    surr_best_k0 = p2_k0.copy()
    surr_best_alpha = p2_alpha.copy()

    # ===================================================================
    # PHASE 3: FIXED PDE-based refinement (v10 improvements)
    # ===================================================================
    if not args.no_pde:
        print(f"\n{'='*70}")
        print(f"  PHASE 3: FIXED PDE refinement (v10)")
        print(f"  Starting: k0={surr_best_k0.tolist()}, alpha={surr_best_alpha.tolist()}")
        print(f"  Fix 1: Tight bounds (factor={args.pde_bound_factor})")
        print(f"  Fix 2: Regularization (lambda={args.pde_lambda})")
        print(f"  Fix 3: Early stopping (maxiter={args.pde_maxiter})")
        print(f"  Voltage grid: shallow cathodic ({len(eta_shallow)} pts)")
        print(f"{'='*70}")
        t_p3 = time.time()

        from Forward.steady_state import SteadyStateConfig
        from FluxCurve import BVFluxCurveInferenceRequest
        from FluxCurve.bv_curve_eval import (
            evaluate_bv_multi_observable_objective_and_gradient,
        )
        from FluxCurve.bv_point_solve import (
            _clear_caches,
            set_parallel_pool,
            close_parallel_pool,
            solve_bv_curve_points_with_warmstart,
            _WARMSTART_MAX_STEPS,
            _SER_GROWTH_CAP,
            _SER_SHRINK,
            _SER_DT_MAX_RATIO,
        )
        from FluxCurve.bv_parallel import BVParallelPointConfig, BVPointSolvePool
        from FluxCurve.recovery import clip_kappa
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
        n_pde_workers = args.workers
        if n_pde_workers <= 0:
            n_pde_workers = min(len(eta_shallow),
                                max(1, (os.cpu_count() or 4) - 1))

        n_k0 = 2
        n_alpha = 2
        n_joint_controls = n_k0 + n_alpha

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

        # Build PDE targets for the shallow grid (regenerate to match v9)
        p3_dir = os.path.join(base_output, "phase3_pde_refinement")
        os.makedirs(p3_dir, exist_ok=True)

        # Use shallow subset of the pre-generated targets
        target_cd_pde, target_pc_pde = _subset_targets(
            target_cd_full, target_pc_full, all_eta, eta_shallow,
        )

        # Build request object for evaluate_bv_multi_observable...
        pde_request = BVFluxCurveInferenceRequest(
            base_solver_params=base_sp,
            steady=steady,
            true_k0=true_k0,
            initial_guess=surr_best_k0.tolist(),
            phi_applied_values=eta_shallow.tolist(),
            target_csv_path=os.path.join(p3_dir, "target_primary.csv"),
            output_dir=p3_dir,
            observable_mode="current_density",
            current_density_scale=observable_scale,
            secondary_observable_mode="peroxide_current",
            secondary_observable_weight=args.secondary_weight,
            secondary_current_density_scale=observable_scale,
            control_mode="joint",
            true_alpha=true_alpha,
            initial_alpha_guess=surr_best_alpha.tolist(),
            log_space=True,
            mesh_Nx=8, mesh_Ny=200, mesh_beta=3.0,
            max_eta_gap=3.0,
            forward_recovery=recovery,
            parallel_fast_path=True,
            parallel_workers=n_pde_workers,
            live_plot=False,
        )

        # --- Fix 1: Compute tight bounds ---
        tight_bounds = compute_tight_bounds(
            surrogate_k0=surr_best_k0,
            surrogate_alpha=surr_best_alpha,
            bound_factor=args.pde_bound_factor,
            alpha_margin=0.05,
            k0_log_space=True,
        )
        print(f"\n  Tight bounds (v10 Fix 1):")
        print(f"    k0_1 log10: [{tight_bounds[0][0]:.4f}, {tight_bounds[0][1]:.4f}]"
              f"  (surr: {np.log10(surr_best_k0[0]):.4f})")
        print(f"    k0_2 log10: [{tight_bounds[1][0]:.4f}, {tight_bounds[1][1]:.4f}]"
              f"  (surr: {np.log10(surr_best_k0[1]):.4f})")
        print(f"    alpha_1:    [{tight_bounds[2][0]:.4f}, {tight_bounds[2][1]:.4f}]"
              f"  (surr: {surr_best_alpha[0]:.4f})")
        print(f"    alpha_2:    [{tight_bounds[3][0]:.4f}, {tight_bounds[3][1]:.4f}]"
              f"  (surr: {surr_best_alpha[1]:.4f})")

        # --- Fix 2: Regularization config ---
        reg_config = RegularizationConfig(
            reg_lambda=args.pde_lambda,
            k0_prior=surr_best_k0,
            alpha_prior=surr_best_alpha,
            n_k0=n_k0,
            k0_log_space=True,
        )
        print(f"\n  Regularization (v10 Fix 2): lambda={args.pde_lambda}")

        # Initial point in optimizer space
        k0_lo = np.array([tight_bounds[i][0] for i in range(n_k0)])
        k0_hi = np.array([tight_bounds[i][1] for i in range(n_k0)])
        alpha_lo = np.array([tight_bounds[n_k0 + i][0] for i in range(n_alpha)])
        alpha_hi = np.array([tight_bounds[n_k0 + i][1] for i in range(n_alpha)])

        x0_k0_log = np.log10(surr_best_k0)
        x0_alpha = surr_best_alpha.copy()
        x0_pde = np.concatenate([x0_k0_log, x0_alpha])

        # Evaluation caching and tracking
        pde_cache: dict[tuple, dict] = {}
        pde_eval_counter: dict[str, int] = {"n": 0}
        best_k0_pde = surr_best_k0.copy()
        best_alpha_pde = surr_best_alpha.copy()
        best_loss_pde = float("inf")

        def _x_to_params_pde(
            x: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray]:
            x = np.asarray(x, dtype=float)
            k0 = np.power(10.0, x[:n_k0])
            alpha = x[n_k0:].copy()
            return k0, alpha

        def _evaluate_pde(x: np.ndarray) -> dict[str, object]:
            nonlocal best_k0_pde, best_alpha_pde, best_loss_pde

            k0_eval, alpha_eval = _x_to_params_pde(x)
            key = tuple(float(f"{v:.12g}") for v in list(k0_eval) + list(alpha_eval))
            if key in pde_cache:
                return pde_cache[key]

            curve = evaluate_bv_multi_observable_objective_and_gradient(
                request=pde_request,
                phi_applied_values=eta_shallow,
                target_flux_primary=target_cd_pde,
                target_flux_secondary=target_pc_pde,
                k0_values=k0_eval,
                mesh=mesh,
                alpha_values=alpha_eval,
                control_mode="joint",
            )

            pde_eval_counter["n"] += 1
            pde_misfit = float(curve.objective)
            grad_ctrl = np.asarray(curve.gradient, dtype=float)

            # Chain rule: log-space k0
            grad_x = grad_ctrl.copy()
            grad_x[:n_k0] = grad_ctrl[:n_k0] * k0_eval * np.log(10.0)

            # Add regularization (Fix 2)
            reg_penalty, reg_grad = compute_regularization_penalty(x, reg_config)
            total_objective = pde_misfit + reg_penalty
            total_grad = grad_x + reg_grad

            k0_str = ", ".join(f"{v:.6e}" for v in k0_eval)
            alpha_str = ", ".join(f"{v:.4f}" for v in alpha_eval)
            print(f"  [pde eval {pde_eval_counter['n']:>3d}] "
                  f"J_misfit={pde_misfit:12.6e} J_reg={reg_penalty:12.6e} "
                  f"J_total={total_objective:12.6e} "
                  f"k0=[{k0_str}] alpha=[{alpha_str}] "
                  f"|grad|={np.linalg.norm(total_grad):.4e}")

            if total_objective < best_loss_pde:
                best_loss_pde = total_objective
                best_k0_pde = k0_eval.copy()
                best_alpha_pde = alpha_eval.copy()

            result = {"objective": total_objective, "grad_x": total_grad}
            pde_cache[key] = result
            return result

        def _fun_pde(x: np.ndarray) -> float:
            return float(_evaluate_pde(x)["objective"])

        def _jac_pde(x: np.ndarray) -> np.ndarray:
            return np.asarray(_evaluate_pde(x)["grad_x"], dtype=float)

        # --- Fix 3: Reduced iterations ---
        opt_result_pde = minimize(
            _fun_pde, x0_pde, jac=_jac_pde,
            method="L-BFGS-B",
            bounds=tight_bounds,
            options={
                "maxiter": args.pde_maxiter,
                "ftol": 1e-8,
                "gtol": 5e-6,
                "disp": True,
            },
        )

        p3_k0 = best_k0_pde.copy()
        p3_alpha = best_alpha_pde.copy()
        p3_loss = best_loss_pde
        p3_time = time.time() - t_p3

        close_parallel_pool()
        _clear_caches()

        _print_phase_result("Phase 3 (PDE fixed, v10)", p3_k0, p3_alpha,
                            true_k0_arr, true_alpha_arr, p3_loss, p3_time)
        phase_results["Phase 3 (PDE fixed, v10)"] = {
            "k0": p3_k0.tolist(), "alpha": p3_alpha.tolist(),
            "loss": p3_loss, "time": p3_time,
        }

        # Choose best between surrogate and PDE
        p3_k0_err, p3_alpha_err = _compute_errors(
            p3_k0, p3_alpha, true_k0_arr, true_alpha_arr)
        p2_k0_err, p2_alpha_err = _compute_errors(
            p2_k0, p2_alpha, true_k0_arr, true_alpha_arr)
        p3_max_err = max(float(p3_k0_err.max()), float(p3_alpha_err.max()))
        p2_max_err = max(float(p2_k0_err.max()), float(p2_alpha_err.max()))

        if p3_max_err <= p2_max_err:
            best_k0, best_alpha = p3_k0.copy(), p3_alpha.copy()
            best_source = "Phase 3 (PDE fixed)"
        else:
            best_k0, best_alpha = p2_k0.copy(), p2_alpha.copy()
            best_source = "Phase 2 (surrogate)"
            print(f"\n  WARNING: PDE refinement regressed "
                  f"(max_err {p3_max_err*100:.2f}% > {p2_max_err*100:.2f}%)")
            print(f"  Keeping surrogate result as final answer.")
    else:
        best_k0, best_alpha = surr_best_k0.copy(), surr_best_alpha.copy()
        best_source = "Phase 2 (surrogate)"

    total_time = time.time() - t_total_start
    pde_time = total_time - t_target_elapsed - surrogate_time

    # ===================================================================
    # FINAL SUMMARY
    # ===================================================================
    print(f"\n{'#'*90}")
    print(f"  MASTER INFERENCE v10 SUMMARY (FIXED PDE REFINEMENT)")
    print(f"{'#'*90}")
    print(f"  True k0:    {true_k0}")
    print(f"  True alpha: {true_alpha}")
    print(f"  K0_HAT = {K0_HAT:.6e},  K0_2_HAT = {K0_2_HAT:.6e}")
    print(f"  PDE fixes: bounds_factor={args.pde_bound_factor}, "
          f"lambda={args.pde_lambda}, maxiter={args.pde_maxiter}")
    print()

    header = (f"{'Phase':<35} | {'k0_1 err':>10} {'k0_2 err':>10} "
              f"{'a1 err':>10} {'a2 err':>10} | {'loss':>12} | {'time':>6}")
    print(header)
    print(f"{'-'*95}")

    for name, ph in phase_results.items():
        k0_err, alpha_err = _compute_errors(
            ph["k0"], ph["alpha"], true_k0_arr, true_alpha_arr
        )
        print(f"{name:<35} | {k0_err[0]*100:>9.2f}% {k0_err[1]*100:>9.2f}% "
              f"{alpha_err[0]*100:>9.2f}% {alpha_err[1]*100:>9.2f}% "
              f"| {ph['loss']:>12.6e} | {ph['time']:>5.1f}s")

    print(f"{'-'*95}")

    best_k0_err, best_alpha_err = _compute_errors(
        best_k0, best_alpha, true_k0_arr, true_alpha_arr)
    best_max_err = max(float(best_k0_err.max()), float(best_alpha_err.max()))

    print(f"\n  Best result: {best_source} (max err = {best_max_err*100:.2f}%)")
    print(f"    k0_1   = {best_k0[0]:.6e}  (err {best_k0_err[0]*100:.2f}%)")
    print(f"    k0_2   = {best_k0[1]:.6e}  (err {best_k0_err[1]*100:.2f}%)")
    print(f"    alpha_1= {best_alpha[0]:.6f}  (err {best_alpha_err[0]*100:.2f}%)")
    print(f"    alpha_2= {best_alpha[1]:.6f}  (err {best_alpha_err[1]*100:.2f}%)")

    # v9 comparison
    print(f"\n  {'='*70}")
    print(f"  v9 vs v10 COMPARISON:")
    print(f"  {'='*70}")
    print(f"  {'Metric':<25} {'v9 surr':>12} {'v9 PDE':>12} {'v10':>12}")
    print(f"  {'-'*61}")
    print(f"  {'k0_1 err (%)':<25} {'8.76':>12} {'16.78':>12} {best_k0_err[0]*100:>12.2f}")
    print(f"  {'k0_2 err (%)':<25} {'7.57':>12} {'10.70':>12} {best_k0_err[1]*100:>12.2f}")
    print(f"  {'alpha_1 err (%)':<25} {'4.76':>12} {'8.70':>12} {best_alpha_err[0]*100:>12.2f}")
    print(f"  {'alpha_2 err (%)':<25} {'6.35':>12} {'7.80':>12} {best_alpha_err[1]*100:>12.2f}")
    print(f"  {'max err (%)':<25} {'8.76':>12} {'16.78':>12} {best_max_err*100:>12.2f}")
    print(f"  {'='*70}")

    print(f"\n  Timing breakdown:")
    print(f"    Target generation:  {t_target_elapsed:>8.1f}s")
    print(f"    Surrogate phases:   {surrogate_time:>8.1f}s")
    if not args.no_pde:
        print(f"    PDE refinement:     {pde_time:>8.1f}s")
    print(f"    Total:              {total_time:>8.1f}s")

    print(f"{'#'*90}")

    # Save comparison CSV
    csv_path = os.path.join(base_output, "master_comparison_v10.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "phase", "k0_1", "k0_2", "alpha_1", "alpha_2",
            "k0_1_err_pct", "k0_2_err_pct", "alpha_1_err_pct", "alpha_2_err_pct",
            "loss", "time_s",
        ])
        for name, ph in phase_results.items():
            k0_err, alpha_err = _compute_errors(
                ph["k0"], ph["alpha"], true_k0_arr, true_alpha_arr
            )
            writer.writerow([
                name,
                f"{ph['k0'][0]:.8e}", f"{ph['k0'][1]:.8e}",
                f"{ph['alpha'][0]:.6f}", f"{ph['alpha'][1]:.6f}",
                f"{k0_err[0]*100:.4f}", f"{k0_err[1]*100:.4f}",
                f"{alpha_err[0]*100:.4f}", f"{alpha_err[1]*100:.4f}",
                f"{ph['loss']:.12e}", f"{ph['time']:.1f}",
            ])
    print(f"\n  Comparison CSV saved -> {csv_path}")
    print(f"\n  Output: {base_output}/")
    print(f"\n=== Master Inference v10 Complete ===")


if __name__ == "__main__":
    main()
