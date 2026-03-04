"""Sweep secondary_weight to find optimal peroxide current weighting.

Runs the Phase 1 (alpha-only) + Phase 2 (joint 4-param) surrogate
optimization pipeline for each weight value and reports parameter
recovery errors.

Usage (from PNPInverse/ directory)::

    python scripts/surrogate/sweep_secondary_weight.py
    python scripts/surrogate/sweep_secondary_weight.py --model path/to/model.pkl
    python scripts/surrogate/sweep_secondary_weight.py --weights 0.1 0.5 1.0 5.0
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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

import numpy as np
from scipy.optimize import minimize

from Surrogate.io import load_surrogate
from Surrogate.objectives import AlphaOnlySurrogateObjective
from Surrogate.surrogate_model import BVSurrogateModel


# ---------------------------------------------------------------------------
# True parameter values
# ---------------------------------------------------------------------------
K0_HAT = K0_HAT_R1
K0_2_HAT = K0_HAT_R2
ALPHA_1 = ALPHA_R1
ALPHA_2 = ALPHA_R2

TRUE_K0 = np.array([K0_HAT, K0_2_HAT])
TRUE_ALPHA = np.array([ALPHA_1, ALPHA_2])

# Default initial guesses (same as v9)
INITIAL_K0_GUESS = [0.005, 0.0005]
INITIAL_ALPHA_GUESS = [0.4, 0.3]

# Default weight values to sweep
DEFAULT_WEIGHTS = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

# Fallback training bounds (used only if model lacks them)
K0_1_TRAIN_LO_DEFAULT = K0_HAT * 0.01
K0_1_TRAIN_HI_DEFAULT = K0_HAT * 100.0
K0_2_TRAIN_LO_DEFAULT = K0_2_HAT * 0.01
K0_2_TRAIN_HI_DEFAULT = K0_2_HAT * 100.0
ALPHA_TRAIN_LO_DEFAULT = 0.10
ALPHA_TRAIN_HI_DEFAULT = 0.90


# ---------------------------------------------------------------------------
# Immutable result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WeightSweepResult:
    """Result of a single weight sweep trial."""

    weight: float
    k0_1: float
    k0_2: float
    alpha_1: float
    alpha_2: float
    k0_1_err_pct: float
    k0_2_err_pct: float
    alpha_1_err_pct: float
    alpha_2_err_pct: float
    max_err_pct: float
    loss: float
    elapsed_s: float


# ---------------------------------------------------------------------------
# Helper: compute relative errors
# ---------------------------------------------------------------------------

def compute_errors(
    k0: np.ndarray,
    alpha: np.ndarray,
    true_k0: np.ndarray,
    true_alpha: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute relative errors for k0 and alpha arrays."""
    k0_err = np.abs(k0 - true_k0) / np.maximum(np.abs(true_k0), 1e-16)
    alpha_err = np.abs(alpha - true_alpha) / np.maximum(np.abs(true_alpha), 1e-16)
    return k0_err, alpha_err


# ---------------------------------------------------------------------------
# Helper: extract training bounds from surrogate model
# ---------------------------------------------------------------------------

def extract_training_bounds(
    surrogate: BVSurrogateModel,
) -> Dict[str, Tuple[float, float]]:
    """Extract training bounds from the surrogate model, with fallbacks."""
    if surrogate.training_bounds is not None:
        tb = surrogate.training_bounds
        alpha_lo = min(tb["alpha_1"][0], tb["alpha_2"][0])
        alpha_hi = max(tb["alpha_1"][1], tb["alpha_2"][1])
        return {
            "k0_1": tb["k0_1"],
            "k0_2": tb["k0_2"],
            "alpha": (alpha_lo, alpha_hi),
        }
    return {
        "k0_1": (K0_1_TRAIN_LO_DEFAULT, K0_1_TRAIN_HI_DEFAULT),
        "k0_2": (K0_2_TRAIN_LO_DEFAULT, K0_2_TRAIN_HI_DEFAULT),
        "alpha": (ALPHA_TRAIN_LO_DEFAULT, ALPHA_TRAIN_HI_DEFAULT),
    }


# ---------------------------------------------------------------------------
# Helper: generate PDE targets (identical to v9)
# ---------------------------------------------------------------------------

def generate_targets_with_pde(
    phi_applied_values: np.ndarray,
    observable_scale: float,
) -> Dict[str, np.ndarray]:
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

    results: Dict[str, np.ndarray] = {}
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
        noisy_flux = add_percent_noise(clean_flux, 2.0, seed=20260226 + seed_offset)
        results[obs_mode] = noisy_flux

    _clear_caches()
    return results


# ---------------------------------------------------------------------------
# Helper: subset targets for shallow voltages
# ---------------------------------------------------------------------------

def subset_targets(
    target_cd: np.ndarray,
    target_pc: np.ndarray,
    all_eta: np.ndarray,
    subset_eta: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract target values and indices for a voltage subset.

    Returns
    -------
    (target_cd_subset, target_pc_subset, subset_indices)
    """
    idx = []
    for eta in subset_eta:
        matches = np.where(np.abs(all_eta - eta) < 1e-10)[0]
        if len(matches) > 0:
            idx.append(matches[0])
    idx_arr = np.array(idx, dtype=int)
    return target_cd[idx_arr], target_pc[idx_arr], idx_arr


# ---------------------------------------------------------------------------
# Core: run surrogate optimization for a single weight
# ---------------------------------------------------------------------------

def run_surrogate_optimization(
    surrogate: BVSurrogateModel,
    target_cd_full: np.ndarray,
    target_pc_full: np.ndarray,
    all_eta: np.ndarray,
    eta_shallow: np.ndarray,
    secondary_weight: float,
    bounds: Dict[str, Tuple[float, float]],
) -> WeightSweepResult:
    """Run Phase 1 + Phase 2 surrogate optimization at a given weight.

    Parameters
    ----------
    surrogate : BVSurrogateModel
        Fitted surrogate model.
    target_cd_full, target_pc_full : np.ndarray
        Full target I-V curves.
    all_eta : np.ndarray
        Full sorted voltage grid.
    eta_shallow : np.ndarray
        Shallow cathodic voltage subset.
    secondary_weight : float
        Weight on the peroxide current objective.
    bounds : dict
        Training bounds with keys "k0_1", "k0_2", "alpha".

    Returns
    -------
    WeightSweepResult
        Frozen result with all metrics.
    """
    t_start = time.time()

    alpha_lo, alpha_hi = bounds["alpha"]

    # ----- Phase 1: Alpha-only (k0 fixed at initial guess) -----
    p1_obj = AlphaOnlySurrogateObjective(
        surrogate=surrogate,
        target_cd=target_cd_full,
        target_pc=target_pc_full,
        fixed_k0=INITIAL_K0_GUESS,
        secondary_weight=secondary_weight,
        fd_step=1e-5,
    )

    x0_p1 = np.array(INITIAL_ALPHA_GUESS, dtype=float)
    bounds_p1 = [(alpha_lo, alpha_hi), (alpha_lo, alpha_hi)]

    result_p1 = minimize(
        p1_obj.objective,
        x0_p1,
        jac=p1_obj.gradient,
        method="L-BFGS-B",
        bounds=bounds_p1,
        options={"maxiter": 60, "ftol": 1e-14, "gtol": 1e-8, "disp": False},
    )

    p1_alpha = result_p1.x.copy()

    # ----- Phase 2: Joint 4-param on shallow voltages -----
    target_cd_shallow, target_pc_shallow, shallow_idx = subset_targets(
        target_cd_full, target_pc_full, all_eta, eta_shallow,
    )

    p2_obj = _SubsetSurrogateObjective(
        surrogate=surrogate,
        target_cd=target_cd_shallow,
        target_pc=target_pc_shallow,
        subset_idx=shallow_idx,
        secondary_weight=secondary_weight,
        fd_step=1e-5,
        log_space_k0=True,
    )

    x0_p2 = np.array([
        np.log10(INITIAL_K0_GUESS[0]),
        np.log10(INITIAL_K0_GUESS[1]),
        p1_alpha[0],
        p1_alpha[1],
    ], dtype=float)

    k0_1_lo, k0_1_hi = bounds["k0_1"]
    k0_2_lo, k0_2_hi = bounds["k0_2"]
    bounds_p2 = [
        (np.log10(k0_1_lo), np.log10(k0_1_hi)),
        (np.log10(k0_2_lo), np.log10(k0_2_hi)),
        (alpha_lo, alpha_hi),
        (alpha_lo, alpha_hi),
    ]

    result_p2 = minimize(
        p2_obj.objective,
        x0_p2,
        jac=p2_obj.gradient,
        method="L-BFGS-B",
        bounds=bounds_p2,
        options={"maxiter": 60, "ftol": 1e-14, "gtol": 1e-8, "disp": False},
    )

    p2_k0 = np.array([10.0**result_p2.x[0], 10.0**result_p2.x[1]])
    p2_alpha = result_p2.x[2:4].copy()
    p2_loss = float(result_p2.fun)

    elapsed = time.time() - t_start

    k0_err, alpha_err = compute_errors(p2_k0, p2_alpha, TRUE_K0, TRUE_ALPHA)

    return WeightSweepResult(
        weight=secondary_weight,
        k0_1=float(p2_k0[0]),
        k0_2=float(p2_k0[1]),
        alpha_1=float(p2_alpha[0]),
        alpha_2=float(p2_alpha[1]),
        k0_1_err_pct=float(k0_err[0] * 100),
        k0_2_err_pct=float(k0_err[1] * 100),
        alpha_1_err_pct=float(alpha_err[0] * 100),
        alpha_2_err_pct=float(alpha_err[1] * 100),
        max_err_pct=float(max(k0_err.max(), alpha_err.max()) * 100),
        loss=p2_loss,
        elapsed_s=elapsed,
    )


# ---------------------------------------------------------------------------
# Subset surrogate objective (same as v9 but standalone)
# ---------------------------------------------------------------------------

class _SubsetSurrogateObjective:
    """Surrogate objective on a subset of voltage points.

    Evaluates the combined (current_density + weighted peroxide_current)
    objective at a subset of the surrogate's voltage grid.
    """

    def __init__(
        self,
        surrogate: BVSurrogateModel,
        target_cd: np.ndarray,
        target_pc: np.ndarray,
        subset_idx: np.ndarray,
        secondary_weight: float = 1.0,
        fd_step: float = 1e-5,
        log_space_k0: bool = True,
    ):
        self.surrogate = surrogate
        self.target_cd = np.asarray(target_cd, dtype=float)
        self.target_pc = np.asarray(target_pc, dtype=float)
        self.subset_idx = subset_idx
        self._valid_cd = ~np.isnan(self.target_cd)
        self._valid_pc = ~np.isnan(self.target_pc)
        self.secondary_weight = secondary_weight
        self.fd_step = fd_step
        self.log_space_k0 = log_space_k0
        self._n_evals = 0

    def _x_to_params(
        self, x: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        """Convert optimizer x-vector to physical parameters."""
        x = np.asarray(x, dtype=float)
        if self.log_space_k0:
            k0_1, k0_2 = 10.0**x[0], 10.0**x[1]
        else:
            k0_1, k0_2 = float(x[0]), float(x[1])
        return k0_1, k0_2, float(x[2]), float(x[3])

    def objective(self, x: np.ndarray) -> float:
        """Evaluate combined objective at x."""
        k0_1, k0_2, a1, a2 = self._x_to_params(x)
        pred = self.surrogate.predict(k0_1, k0_2, a1, a2)
        cd_sim = pred["current_density"][self.subset_idx]
        pc_sim = pred["peroxide_current"][self.subset_idx]
        cd_diff = cd_sim[self._valid_cd] - self.target_cd[self._valid_cd]
        pc_diff = pc_sim[self._valid_pc] - self.target_pc[self._valid_pc]
        j_cd = 0.5 * np.sum(cd_diff**2)
        j_pc = 0.5 * np.sum(pc_diff**2)
        self._n_evals += 1
        return float(j_cd + self.secondary_weight * j_pc)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Central finite-difference gradient."""
        x = np.asarray(x, dtype=float)
        grad = np.zeros(len(x), dtype=float)
        h = self.fd_step
        for i in range(len(x)):
            x_plus, x_minus = x.copy(), x.copy()
            x_plus[i] += h
            x_minus[i] -= h
            grad[i] = (self.objective(x_plus) - self.objective(x_minus)) / (2 * h)
        return grad

    @property
    def n_evals(self) -> int:
        """Number of surrogate evaluations performed."""
        return self._n_evals


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_results_table(results: Sequence[WeightSweepResult]) -> str:
    """Format sweep results as a readable table.

    Returns
    -------
    str
        Formatted table string.
    """
    lines = []
    header = (
        f"{'weight':>8} | {'k0_1_err%':>10} {'k0_2_err%':>10} "
        f"{'a1_err%':>10} {'a2_err%':>10} | {'max_err%':>10} | "
        f"{'loss':>12} | {'time':>6}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for r in results:
        lines.append(
            f"{r.weight:>8.2f} | {r.k0_1_err_pct:>10.2f} {r.k0_2_err_pct:>10.2f} "
            f"{r.alpha_1_err_pct:>10.2f} {r.alpha_2_err_pct:>10.2f} | "
            f"{r.max_err_pct:>10.2f} | {r.loss:>12.6e} | {r.elapsed_s:>5.1f}s"
        )

    return "\n".join(lines)


def save_results_csv(
    results: Sequence[WeightSweepResult],
    output_path: str,
) -> None:
    """Save sweep results to a CSV file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "secondary_weight",
            "k0_1", "k0_2", "alpha_1", "alpha_2",
            "k0_1_err_pct", "k0_2_err_pct",
            "alpha_1_err_pct", "alpha_2_err_pct",
            "max_err_pct", "loss", "elapsed_s",
        ])
        for r in results:
            writer.writerow([
                f"{r.weight:.4f}",
                f"{r.k0_1:.8e}", f"{r.k0_2:.8e}",
                f"{r.alpha_1:.6f}", f"{r.alpha_2:.6f}",
                f"{r.k0_1_err_pct:.4f}", f"{r.k0_2_err_pct:.4f}",
                f"{r.alpha_1_err_pct:.4f}", f"{r.alpha_2_err_pct:.4f}",
                f"{r.max_err_pct:.4f}",
                f"{r.loss:.12e}", f"{r.elapsed_s:.2f}",
            ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep secondary_weight for optimal peroxide current weighting"
    )
    parser.add_argument(
        "--model", type=str,
        default="StudyResults/surrogate_v9/surrogate_model.pkl",
        help="Path to surrogate model .pkl",
    )
    parser.add_argument(
        "--weights", type=float, nargs="+",
        default=DEFAULT_WEIGHTS,
        help="Weight values to sweep (default: 0.1 0.25 0.5 1.0 2.0 5.0 10.0 20.0)",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="StudyResults/weight_sweep",
        help="Output directory for results",
    )
    args = parser.parse_args()

    # ----- Voltage grids (identical to v9) -----
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

    observable_scale = -I_SCALE

    # ----- Banner -----
    print(f"\n{'#'*70}")
    print(f"  SECONDARY WEIGHT SWEEP")
    print(f"  Weights: {args.weights}")
    print(f"  True k0:    {TRUE_K0.tolist()}")
    print(f"  True alpha: {TRUE_ALPHA.tolist()}")
    print(f"  Model:      {args.model}")
    print(f"{'#'*70}\n")

    # ----- Load surrogate -----
    print(f"Loading surrogate model from: {args.model}")
    surrogate = load_surrogate(args.model)
    training_bounds = extract_training_bounds(surrogate)
    print(f"  Training bounds: {training_bounds}")

    # ----- Generate PDE targets (once) -----
    print(f"\nGenerating target I-V curves with PDE solver at true parameters...")
    t_target = time.time()
    targets = generate_targets_with_pde(all_eta, observable_scale)
    target_cd_full = targets["current_density"]
    target_pc_full = targets["peroxide_current"]
    t_target_elapsed = time.time() - t_target
    print(f"  Target generation: {t_target_elapsed:.1f}s")

    # ----- Sweep -----
    results: List[WeightSweepResult] = []
    t_sweep_start = time.time()

    for i, weight in enumerate(args.weights):
        print(f"\n{'='*60}")
        print(f"  Weight {i+1}/{len(args.weights)}: secondary_weight = {weight}")
        print(f"{'='*60}")

        result = run_surrogate_optimization(
            surrogate=surrogate,
            target_cd_full=target_cd_full,
            target_pc_full=target_pc_full,
            all_eta=all_eta,
            eta_shallow=eta_shallow,
            secondary_weight=weight,
            bounds=training_bounds,
        )
        results.append(result)

        k0_err, alpha_err = compute_errors(
            np.array([result.k0_1, result.k0_2]),
            np.array([result.alpha_1, result.alpha_2]),
            TRUE_K0, TRUE_ALPHA,
        )
        print(f"  k0_1={result.k0_1:.6e} (err {result.k0_1_err_pct:.2f}%)")
        print(f"  k0_2={result.k0_2:.6e} (err {result.k0_2_err_pct:.2f}%)")
        print(f"  alpha_1={result.alpha_1:.6f} (err {result.alpha_1_err_pct:.2f}%)")
        print(f"  alpha_2={result.alpha_2:.6f} (err {result.alpha_2_err_pct:.2f}%)")
        print(f"  max_err={result.max_err_pct:.2f}%  loss={result.loss:.6e}  time={result.elapsed_s:.1f}s")

    sweep_time = time.time() - t_sweep_start

    # ----- Results table -----
    print(f"\n\n{'#'*80}")
    print(f"  WEIGHT SWEEP RESULTS")
    print(f"{'#'*80}\n")
    print(format_results_table(results))

    # ----- Identify optimal weight -----
    best = min(results, key=lambda r: r.max_err_pct)
    print(f"\n  OPTIMAL WEIGHT: {best.weight:.2f}")
    print(f"    max error = {best.max_err_pct:.2f}%")
    print(f"    k0_1 err  = {best.k0_1_err_pct:.2f}%")
    print(f"    k0_2 err  = {best.k0_2_err_pct:.2f}%")
    print(f"    alpha_1 err = {best.alpha_1_err_pct:.2f}%")
    print(f"    alpha_2 err = {best.alpha_2_err_pct:.2f}%")

    # ----- Save CSV -----
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "weight_sweep_results.csv")
    save_results_csv(results, csv_path)
    print(f"\n  Results CSV saved -> {csv_path}")

    # ----- Timing -----
    total_time = time.time() - t_target - t_target_elapsed + sweep_time + t_target_elapsed
    print(f"\n  Timing:")
    print(f"    Target generation: {t_target_elapsed:.1f}s")
    print(f"    Sweep ({len(args.weights)} weights): {sweep_time:.1f}s")
    print(f"    Total: {t_target_elapsed + sweep_time:.1f}s")
    print(f"\n=== Weight Sweep Complete ===")


if __name__ == "__main__":
    main()
