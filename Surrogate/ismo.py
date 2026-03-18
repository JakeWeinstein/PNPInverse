"""Iterative Surrogate Model Optimization (ISMO) core loop.

Wraps the existing multistart + cascade surrogate optimization pipeline in
an outer loop that iteratively:

1. Optimizes the current surrogate to find candidate parameters.
2. Evaluates those candidates with the true PDE solver.
3. Measures surrogate-vs-PDE discrepancy at the candidates.
4. Augments the training data with new (parameter, PDE output) pairs.
5. Retrains the surrogate on the expanded dataset.
6. Checks convergence and repeats or stops.

Budget: at most ``ISMOConfig.total_pde_budget`` new PDE solves total.

Reference: Lye, Mishra, Ray (2020) -- "Iterative surrogate model optimization."

Public API
----------
AcquisitionStrategy
    Enum for sample acquisition methods.
ISMOConfig
    Frozen configuration dataclass.
ISMOIteration
    Per-iteration result snapshot.
ISMOResult
    Final result container.
run_ismo
    Main entry point.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats.qmc import LatinHypercube

from Surrogate.cascade import CascadeConfig, CascadeResult, run_cascade_inference
from Surrogate.multistart import (
    MultiStartConfig,
    MultiStartResult,
    run_multistart_inference,
)
from Surrogate.objectives import SurrogateObjective, SubsetSurrogateObjective


# ---------------------------------------------------------------------------
# Acquisition strategy enum
# ---------------------------------------------------------------------------

class AcquisitionStrategy(str, Enum):
    """How to select new sample points for PDE evaluation.

    Attributes
    ----------
    OPTIMIZER_TRAJECTORY
        New sample points come from the top-K optimizer solutions
        (multistart candidates + cascade result).
    UNCERTAINTY
        Sample where the surrogate is most uncertain (NN ensemble
        inter-member std).  Falls back to OPTIMIZER_TRAJECTORY for
        surrogates without native uncertainty.
    HYBRID
        50/50 split between optimizer trajectory and uncertainty sampling.
    """

    OPTIMIZER_TRAJECTORY = "optimizer_trajectory"
    UNCERTAINTY = "uncertainty"
    HYBRID = "hybrid"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ISMOConfig:
    """Frozen configuration for the ISMO outer loop.

    Attributes
    ----------
    max_iterations : int
        Hard cap on ISMO outer iterations.
    samples_per_iteration : int
        Number of new PDE solves per iteration.
    total_pde_budget : int
        Absolute maximum PDE evaluations across all iterations.
    convergence_rtol : float
        Relative tolerance: ``|J_surr - J_pde| / max(|J_pde|, atol) < rtol``.
    convergence_atol : float
        Absolute tolerance floor (handles near-zero loss).
    surrogate_type : str
        One of ``"nn_ensemble"``, ``"pod_rbf_log"``, ``"pod_rbf_nolog"``,
        ``"rbf_baseline"``, ``"gp"``, ``"pce"``.
    acquisition_strategy : AcquisitionStrategy
        How to select new sample points.
    warm_start_retrain : bool
        If True, retrain NN from current weights; RBF/POD refit from scratch.
    retrain_epochs : int
        Number of epochs for warm-start NN retraining.
    multistart_config : MultiStartConfig or None
        Override for the inner multistart optimizer.
    cascade_config : CascadeConfig or None
        Override for the inner cascade optimizer.
    n_top_candidates_to_evaluate : int
        From the multistart top candidates, evaluate this many with the PDE.
    neighborhood_fraction : float
        Fraction of parameter range for sampling neighborhood.
    output_dir : str
        Directory for saving augmented data and retrained models.
    verbose : bool
        Print progress.
    """

    max_iterations: int = 5
    samples_per_iteration: int = 30
    total_pde_budget: int = 200
    convergence_rtol: float = 0.05
    convergence_atol: float = 1e-4
    surrogate_type: str = "nn_ensemble"
    acquisition_strategy: AcquisitionStrategy = AcquisitionStrategy.OPTIMIZER_TRAJECTORY
    warm_start_retrain: bool = False
    retrain_epochs: int = 200
    multistart_config: MultiStartConfig | None = None
    cascade_config: CascadeConfig | None = None
    n_top_candidates_to_evaluate: int = 10
    neighborhood_fraction: float = 0.1
    output_dir: str = "data/ismo_runs"
    verbose: bool = True


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ISMOIteration:
    """Snapshot of a single ISMO iteration.

    Attributes
    ----------
    iteration : int
        0-indexed iteration number.
    n_new_samples : int
        PDE evaluations added this iteration.
    n_total_training : int
        Total training set size after augmentation.
    surrogate_loss_at_best : float
        Surrogate objective at the best candidate (before retraining).
    pde_loss_at_best : float
        PDE objective at the same point.
    surrogate_pde_gap : float
        ``|surrogate_loss - pde_loss| / max(pde_loss, atol)``.
    convergence_metric : float
        Same as surrogate_pde_gap (alias for clarity).
    best_params : tuple
        Best parameter set ``(k0_1, k0_2, alpha_1, alpha_2)``.
    best_loss : float
        PDE loss at the best candidate (ground truth).
    candidate_pde_losses : tuple
        PDE losses at all evaluated candidates.
    candidate_surrogate_losses : tuple
        Corresponding surrogate losses.
    retrain_val_rmse_cd : float or None
        Validation RMSE for CD after retraining.
    retrain_val_rmse_pc : float or None
        Validation RMSE for PC after retraining.
    wall_time_s : float
        Wall clock time for this iteration.
    """

    iteration: int
    n_new_samples: int
    n_total_training: int
    surrogate_loss_at_best: float
    pde_loss_at_best: float
    surrogate_pde_gap: float
    convergence_metric: float
    best_params: tuple
    best_loss: float
    candidate_pde_losses: tuple
    candidate_surrogate_losses: tuple
    retrain_val_rmse_cd: float | None
    retrain_val_rmse_pc: float | None
    wall_time_s: float


@dataclass(frozen=True)
class ISMOResult:
    """Final result of the ISMO procedure.

    Attributes
    ----------
    converged : bool
        Whether the surrogate-PDE gap reached the convergence tolerance.
    termination_reason : str
        One of ``"converged"``, ``"budget_exhausted"``,
        ``"max_iterations"``, ``"no_improvement"``.
    n_iterations : int
        Number of completed ISMO iterations.
    total_pde_evals : int
        Total PDE evaluations performed.
    iteration_history : tuple
        Tuple of ``ISMOIteration`` snapshots.
    final_params : tuple
        Best parameter set ``(k0_1, k0_2, alpha_1, alpha_2)``.
    final_loss : float
        PDE loss at the best parameter set.
    final_surrogate_path : str or None
        Path to the saved retrained surrogate (if applicable).
    augmented_data_path : str or None
        Path to the final augmented training data.
    total_wall_time_s : float
        Total wall clock time.
    """

    converged: bool
    termination_reason: str
    n_iterations: int
    total_pde_evals: int
    iteration_history: tuple
    final_params: tuple
    final_loss: float
    final_surrogate_path: str | None
    augmented_data_path: str | None
    total_wall_time_s: float


# ---------------------------------------------------------------------------
# Private helpers: acquisition functions
# ---------------------------------------------------------------------------

def _lhs_in_neighborhood(
    center: np.ndarray,
    bounds_k0_1: Tuple[float, float],
    bounds_k0_2: Tuple[float, float],
    bounds_alpha: Tuple[float, float],
    neighborhood_fraction: float,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    """Generate LHS samples in a neighborhood of *center*.

    Parameters are in physical space: ``(k0_1, k0_2, alpha_1, alpha_2)``.
    k0 bounds are in physical space; sampling uses log10 internally.

    Returns
    -------
    np.ndarray of shape ``(n_samples, 4)``
        Samples in physical space.
    """
    # Full range in log/linear space
    log_k0_1_range = np.log10(bounds_k0_1[1]) - np.log10(bounds_k0_1[0])
    log_k0_2_range = np.log10(bounds_k0_2[1]) - np.log10(bounds_k0_2[0])
    alpha_range = bounds_alpha[1] - bounds_alpha[0]

    # Center in log/linear space
    center_log = np.array([
        np.log10(max(center[0], 1e-30)),
        np.log10(max(center[1], 1e-30)),
        center[2],
        center[3],
    ])

    # Neighborhood half-widths
    hw = neighborhood_fraction * np.array([
        log_k0_1_range,
        log_k0_2_range,
        alpha_range,
        alpha_range,
    ])

    # Clip neighborhood to global bounds
    lo = np.array([
        np.log10(bounds_k0_1[0]),
        np.log10(bounds_k0_2[0]),
        bounds_alpha[0],
        bounds_alpha[0],
    ])
    hi = np.array([
        np.log10(bounds_k0_1[1]),
        np.log10(bounds_k0_2[1]),
        bounds_alpha[1],
        bounds_alpha[1],
    ])
    nb_lo = np.maximum(center_log - hw, lo)
    nb_hi = np.minimum(center_log + hw, hi)

    # LHS in [0,1]^4 then scale
    sampler = LatinHypercube(d=4, seed=seed)
    unit = sampler.random(n=n_samples)
    samples_log = nb_lo + unit * (nb_hi - nb_lo)

    # Convert back to physical space
    physical = np.column_stack([
        10.0 ** samples_log[:, 0],
        10.0 ** samples_log[:, 1],
        samples_log[:, 2],
        samples_log[:, 3],
    ])
    return physical


def _acquire_optimizer_trajectory(
    multistart_result: MultiStartResult,
    cascade_result: CascadeResult | None,
    n_candidates: int,
    n_total: int,
    bounds_k0_1: Tuple[float, float],
    bounds_k0_2: Tuple[float, float],
    bounds_alpha: Tuple[float, float],
    neighborhood_fraction: float,
    seed: int,
) -> np.ndarray:
    """Select acquisition points from the optimizer trajectory.

    Collects top candidates from multistart, optionally prepends the cascade
    best, deduplicates, and fills remaining slots with LHS samples from
    a neighborhood around the best point.

    Returns
    -------
    np.ndarray of shape ``(n_total, 4)``
        Candidate parameter sets in physical space.
    """
    # Collect top candidates from multistart (sorted by polished_loss)
    candidates_list: List[np.ndarray] = []
    for c in multistart_result.candidates[:n_candidates]:
        candidates_list.append(np.array([c.k0_1, c.k0_2, c.alpha_1, c.alpha_2]))

    # Prepend cascade best if available
    if cascade_result is not None:
        cascade_pt = np.array([
            cascade_result.best_k0_1,
            cascade_result.best_k0_2,
            cascade_result.best_alpha_1,
            cascade_result.best_alpha_2,
        ])
        candidates_list.insert(0, cascade_pt)

    if not candidates_list:
        # Fallback: pure LHS
        return _lhs_in_neighborhood(
            center=np.array([
                np.sqrt(bounds_k0_1[0] * bounds_k0_1[1]),
                np.sqrt(bounds_k0_2[0] * bounds_k0_2[1]),
                0.5 * (bounds_alpha[0] + bounds_alpha[1]),
                0.5 * (bounds_alpha[0] + bounds_alpha[1]),
            ]),
            bounds_k0_1=bounds_k0_1,
            bounds_k0_2=bounds_k0_2,
            bounds_alpha=bounds_alpha,
            neighborhood_fraction=1.0,
            n_samples=n_total,
            seed=seed,
        )

    all_pts = np.array(candidates_list)

    # Deduplicate (within 1e-6 relative tolerance in log-k0 / linear-alpha)
    unique_pts: List[np.ndarray] = [all_pts[0]]
    for pt in all_pts[1:]:
        log_pt = np.array([
            np.log10(max(pt[0], 1e-30)),
            np.log10(max(pt[1], 1e-30)),
            pt[2],
            pt[3],
        ])
        is_dup = False
        for u in unique_pts:
            log_u = np.array([
                np.log10(max(u[0], 1e-30)),
                np.log10(max(u[1], 1e-30)),
                u[2],
                u[3],
            ])
            if np.allclose(log_pt, log_u, rtol=1e-6, atol=1e-10):
                is_dup = True
                break
        if not is_dup:
            unique_pts.append(pt)

    unique_arr = np.array(unique_pts[:n_total])

    # Fill remaining slots with LHS from neighborhood of best
    n_fill = n_total - len(unique_arr)
    if n_fill > 0:
        best = unique_arr[0]
        fill = _lhs_in_neighborhood(
            center=best,
            bounds_k0_1=bounds_k0_1,
            bounds_k0_2=bounds_k0_2,
            bounds_alpha=bounds_alpha,
            neighborhood_fraction=neighborhood_fraction,
            n_samples=n_fill,
            seed=seed,
        )
        result = np.vstack([unique_arr, fill])
    else:
        result = unique_arr

    return result[:n_total]


def _acquire_uncertainty(
    surrogate: Any,
    n_total: int,
    bounds_k0_1: Tuple[float, float],
    bounds_k0_2: Tuple[float, float],
    bounds_alpha: Tuple[float, float],
    best_params: np.ndarray,
    neighborhood_fraction: float,
    seed: int,
    # Fallback args for optimizer trajectory
    multistart_result: MultiStartResult | None = None,
    cascade_result: CascadeResult | None = None,
    n_candidates: int = 10,
) -> np.ndarray:
    """Select acquisition points by surrogate uncertainty.

    Generates 5000 LHS candidate points within a neighborhood and selects
    the top ``n_total`` by total uncertainty (cd_std + pc_std).

    Falls back to optimizer trajectory acquisition for surrogates without
    native uncertainty estimation.

    Returns
    -------
    np.ndarray of shape ``(n_total, 4)``
        Candidate parameter sets in physical space.
    """
    has_uncertainty = (
        hasattr(surrogate, "predict_with_uncertainty")
        or hasattr(surrogate, "predict_batch_with_uncertainty")
    )

    if not has_uncertainty:
        if multistart_result is None:
            # Pure LHS fallback
            return _lhs_in_neighborhood(
                center=best_params,
                bounds_k0_1=bounds_k0_1,
                bounds_k0_2=bounds_k0_2,
                bounds_alpha=bounds_alpha,
                neighborhood_fraction=neighborhood_fraction,
                n_samples=n_total,
                seed=seed,
            )
        return _acquire_optimizer_trajectory(
            multistart_result=multistart_result,
            cascade_result=cascade_result,
            n_candidates=n_candidates,
            n_total=n_total,
            bounds_k0_1=bounds_k0_1,
            bounds_k0_2=bounds_k0_2,
            bounds_alpha=bounds_alpha,
            neighborhood_fraction=neighborhood_fraction,
            seed=seed,
        )

    # Generate 5000 LHS candidate points in the neighborhood
    n_lhs = 5000
    lhs_pts = _lhs_in_neighborhood(
        center=best_params,
        bounds_k0_1=bounds_k0_1,
        bounds_k0_2=bounds_k0_2,
        bounds_alpha=bounds_alpha,
        neighborhood_fraction=neighborhood_fraction,
        n_samples=n_lhs,
        seed=seed,
    )

    # Compute uncertainty at each candidate
    uncertainties = np.zeros(n_lhs)
    for j in range(n_lhs):
        pt = lhs_pts[j]
        if hasattr(surrogate, "predict_with_uncertainty"):
            pred = surrogate.predict_with_uncertainty(pt[0], pt[1], pt[2], pt[3])
            uncertainties[j] = (
                np.mean(pred["current_density_std"])
                + np.mean(pred["peroxide_current_std"])
            )
        elif hasattr(surrogate, "predict_batch_with_uncertainty"):
            pred = surrogate.predict_batch_with_uncertainty(pt.reshape(1, 4))
            uncertainties[j] = (
                np.mean(pred["current_density_std"])
                + np.mean(pred["peroxide_current_std"])
            )

    # Select top n_total by uncertainty
    top_idx = np.argsort(uncertainties)[-n_total:]
    return lhs_pts[top_idx]


def _acquire_hybrid(
    surrogate: Any,
    multistart_result: MultiStartResult,
    cascade_result: CascadeResult | None,
    n_candidates: int,
    n_total: int,
    bounds_k0_1: Tuple[float, float],
    bounds_k0_2: Tuple[float, float],
    bounds_alpha: Tuple[float, float],
    best_params: np.ndarray,
    neighborhood_fraction: float,
    seed: int,
) -> np.ndarray:
    """50/50 split between optimizer trajectory and uncertainty sampling.

    Returns
    -------
    np.ndarray of shape ``(n_total, 4)``
        Candidate parameter sets in physical space.
    """
    n_traj = n_total // 2
    n_unc = n_total - n_traj

    pts_traj = _acquire_optimizer_trajectory(
        multistart_result=multistart_result,
        cascade_result=cascade_result,
        n_candidates=n_candidates,
        n_total=n_traj,
        bounds_k0_1=bounds_k0_1,
        bounds_k0_2=bounds_k0_2,
        bounds_alpha=bounds_alpha,
        neighborhood_fraction=neighborhood_fraction,
        seed=seed,
    )

    pts_unc = _acquire_uncertainty(
        surrogate=surrogate,
        n_total=n_unc,
        bounds_k0_1=bounds_k0_1,
        bounds_k0_2=bounds_k0_2,
        bounds_alpha=bounds_alpha,
        best_params=best_params,
        neighborhood_fraction=neighborhood_fraction,
        seed=seed + 1,
        multistart_result=multistart_result,
        cascade_result=cascade_result,
        n_candidates=n_candidates,
    )

    return np.vstack([pts_traj, pts_unc])


# ---------------------------------------------------------------------------
# Private helpers: PDE evaluation
# ---------------------------------------------------------------------------

def _compute_pde_loss(
    cd_sim: np.ndarray,
    pc_sim: np.ndarray,
    target_cd: np.ndarray,
    target_pc: np.ndarray,
    secondary_weight: float,
    subset_idx: np.ndarray | None,
) -> float:
    """Compute the PDE objective value (same formula as surrogate objective).

    Parameters
    ----------
    cd_sim, pc_sim : np.ndarray
        Simulated I-V curves from the PDE solver.
    target_cd, target_pc : np.ndarray
        Target I-V curves.
    secondary_weight : float
        Weight on the peroxide current term.
    subset_idx : np.ndarray or None
        If provided, restrict comparison to these voltage indices.

    Returns
    -------
    float
        Objective value.
    """
    if subset_idx is not None:
        cd_s = cd_sim[subset_idx]
        pc_s = pc_sim[subset_idx]
        cd_t = target_cd[subset_idx]
        pc_t = target_pc[subset_idx]
    else:
        cd_s = cd_sim
        pc_s = pc_sim
        cd_t = target_cd
        pc_t = target_pc

    valid_cd = ~np.isnan(cd_t) & ~np.isnan(cd_s)
    valid_pc = ~np.isnan(pc_t) & ~np.isnan(pc_s)

    cd_diff = cd_s[valid_cd] - cd_t[valid_cd]
    pc_diff = pc_s[valid_pc] - pc_t[valid_pc]

    j_cd = 0.5 * np.sum(cd_diff ** 2)
    j_pc = 0.5 * np.sum(pc_diff ** 2)

    return float(j_cd + secondary_weight * j_pc)


def _evaluate_candidates_pde(
    candidates: np.ndarray,
    phi_applied: np.ndarray,
    target_cd: np.ndarray,
    target_pc: np.ndarray,
    pde_solver_kwargs: dict,
    secondary_weight: float,
    subset_idx: np.ndarray | None,
    verbose: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[bool]]:
    """Evaluate candidate parameter sets with the true PDE solver.

    Uses subprocess-based parallel evaluation to avoid Firedrake+PyTorch
    conflicts (Firedrake's adjoint tape breaks if PyTorch has been imported
    in the same process).  Each worker spawns a fresh process with its own
    Firedrake environment.

    Parameters
    ----------
    candidates : np.ndarray of shape ``(M, 4)``
        Parameter sets ``(k0_1, k0_2, alpha_1, alpha_2)`` in physical space.
    phi_applied : np.ndarray
        Voltage grid.
    target_cd, target_pc : np.ndarray
        Target I-V curves (used for loss computation).
    pde_solver_kwargs : dict
        Must contain ``base_solver_params``, ``steady``, ``observable_scale``,
        and ``mesh_params`` (tuple of ``(Nx, Ny, beta)``).
    secondary_weight : float
        Weight on the peroxide current objective term.
    subset_idx : np.ndarray or None
        Voltage subset indices for loss computation.
    verbose : bool
        Print per-candidate timing.

    Returns
    -------
    cd_curves : np.ndarray of shape ``(M, n_eta)``
    pc_curves : np.ndarray of shape ``(M, n_eta)``
    pde_losses : np.ndarray of shape ``(M,)``
    converged_flags : list of bool
    """
    from Surrogate.training import generate_training_dataset_parallel

    m = len(candidates)
    n_eta = len(phi_applied)

    if verbose:
        print(
            f"  [ISMO PDE] Evaluating {m} candidates via subprocess pool...",
            flush=True,
        )

    # Use a temporary file for the parallel generator output.
    # Keep tmpdir alive until we extract the full per-candidate arrays
    # from the saved npz (the result dict only has valid-only arrays).
    import tempfile
    tmpdir = tempfile.mkdtemp()
    try:
        output_path = os.path.join(tmpdir, "ismo_pde_batch.npz")

        # Determine number of workers: use 4 for typical ISMO batches,
        # but cap at the number of candidates to avoid empty groups.
        n_workers = min(4, m)

        result = generate_training_dataset_parallel(
            parameter_samples=candidates,
            phi_applied_values=phi_applied,
            base_solver_params=pde_solver_kwargs["base_solver_params"],
            steady=pde_solver_kwargs["steady"],
            observable_scale=pde_solver_kwargs["observable_scale"],
            mesh_params=pde_solver_kwargs["mesh_params"],
            output_path=output_path,
            n_workers=n_workers,
            min_converged_fraction=0.5,
            verbose=verbose,
        )

        # Extract per-candidate results using the full arrays saved in the
        # npz.  generate_training_dataset_parallel processes candidates in
        # input order and stores all_converged / all_current_density /
        # all_peroxide_current indexed by the original sample index, so we
        # use index-based matching instead of fragile float comparison.
        saved = np.load(output_path, allow_pickle=True)
        all_converged_arr = saved["all_converged"].astype(bool)  # (m,)
        all_cd_saved = saved["all_current_density"]  # (m, n_eta)
        all_pc_saved = saved["all_peroxide_current"]  # (m, n_eta)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    cd_all = np.full((m, n_eta), np.nan)
    pc_all = np.full((m, n_eta), np.nan)
    pde_losses = np.full(m, np.inf)
    converged_flags: List[bool] = []

    # Use the per-index converged array directly.  Fall back to NaN-
    # checking on the output arrays if the converged flag is absent.
    for j in range(m):
        is_converged = bool(all_converged_arr[j])
        if not is_converged:
            # Secondary check: if the output row is not all-NaN the
            # solver produced data even though converged fraction was
            # below threshold -- still treat as unconverged.
            is_converged = False

        if is_converged:
            cd_all[j] = all_cd_saved[j]
            pc_all[j] = all_pc_saved[j]

        converged_flags.append(is_converged)

        if is_converged:
            pde_losses[j] = _compute_pde_loss(
                cd_all[j], pc_all[j], target_cd, target_pc,
                secondary_weight, subset_idx,
            )

    n_ok = sum(converged_flags)
    if verbose:
        print(
            f"  [ISMO PDE] Done: {n_ok}/{m} converged, "
            f"{m - n_ok} failed",
            flush=True,
        )
        # Print per-candidate summary
        for j in range(m):
            k0_1, k0_2, alpha_1, alpha_2 = candidates[j]
            status = "OK" if converged_flags[j] else "FAIL"
            print(
                f"  [ISMO PDE {j + 1}/{m}] "
                f"k0=({k0_1:.3e}, {k0_2:.3e}) "
                f"alpha=({alpha_1:.3f}, {alpha_2:.3f}) "
                f"loss={pde_losses[j]:.4e} [{status}]",
                flush=True,
            )

    return cd_all, pc_all, pde_losses, converged_flags


# ---------------------------------------------------------------------------
# Private helpers: surrogate loss computation
# ---------------------------------------------------------------------------

def _compute_surrogate_losses(
    surrogate: Any,
    candidates: np.ndarray,
    target_cd: np.ndarray,
    target_pc: np.ndarray,
    secondary_weight: float,
    subset_idx: np.ndarray | None,
) -> np.ndarray:
    """Compute surrogate objective at each candidate.

    Returns
    -------
    np.ndarray of shape ``(M,)``
        Surrogate objective values.
    """
    if subset_idx is not None:
        obj = SubsetSurrogateObjective(
            surrogate=surrogate,
            target_cd=target_cd,
            target_pc=target_pc,
            subset_idx=subset_idx,
            secondary_weight=secondary_weight,
        )
    else:
        obj = SurrogateObjective(
            surrogate=surrogate,
            target_cd=target_cd,
            target_pc=target_pc,
            secondary_weight=secondary_weight,
        )

    m = len(candidates)
    losses = np.zeros(m)
    for j in range(m):
        k0_1, k0_2, alpha_1, alpha_2 = candidates[j]
        # Objective expects log-space k0
        x = np.array([
            np.log10(max(k0_1, 1e-30)),
            np.log10(max(k0_2, 1e-30)),
            alpha_1,
            alpha_2,
        ])
        losses[j] = obj.objective(x)

    return losses


# ---------------------------------------------------------------------------
# Private helpers: training data augmentation
# ---------------------------------------------------------------------------

def _augment_training_data(
    existing_params: np.ndarray,
    existing_cd: np.ndarray,
    existing_pc: np.ndarray,
    new_params: np.ndarray,
    new_cd: np.ndarray,
    new_pc: np.ndarray,
    converged_mask: List[bool],
    output_path: str | None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Augment training data with new converged PDE evaluations.

    Filters to converged samples, deduplicates against existing data,
    and concatenates.  Never mutates input arrays.

    Parameters
    ----------
    existing_params : np.ndarray of shape ``(N, 4)``
    existing_cd : np.ndarray of shape ``(N, n_eta)``
    existing_pc : np.ndarray of shape ``(N, n_eta)``
    new_params : np.ndarray of shape ``(M, 4)``
    new_cd : np.ndarray of shape ``(M, n_eta)``
    new_pc : np.ndarray of shape ``(M, n_eta)``
    converged_mask : list of bool
        Which of the M new samples converged.
    output_path : str or None
        If provided, save augmented data as ``.npz``.

    Returns
    -------
    (augmented_params, augmented_cd, augmented_pc)
    """
    # Filter to converged
    mask = np.array(converged_mask, dtype=bool)
    if not np.any(mask):
        # No converged samples -- return existing data unchanged
        return (
            existing_params.copy(),
            existing_cd.copy(),
            existing_pc.copy(),
        )

    valid_params = new_params[mask]
    valid_cd = new_cd[mask]
    valid_pc = new_pc[mask]

    # Drop samples that still contain NaN in their I-V curves
    # (partial convergence can leave NaN at individual voltage points)
    nan_free = ~(np.any(np.isnan(valid_cd), axis=1) |
                 np.any(np.isnan(valid_pc), axis=1))
    if not np.all(nan_free):
        n_dropped = int((~nan_free).sum())
        valid_params = valid_params[nan_free]
        valid_cd = valid_cd[nan_free]
        valid_pc = valid_pc[nan_free]

    if len(valid_params) == 0:
        return (
            existing_params.copy(),
            existing_cd.copy(),
            existing_pc.copy(),
        )

    # Deduplicate against existing training data
    keep: List[int] = []
    for j in range(len(valid_params)):
        is_dup = False
        for k in range(len(existing_params)):
            if np.allclose(valid_params[j], existing_params[k], rtol=0, atol=1e-8):
                is_dup = True
                break
        if not is_dup:
            keep.append(j)

    if not keep:
        return (
            existing_params.copy(),
            existing_cd.copy(),
            existing_pc.copy(),
        )

    keep_arr = np.array(keep)
    aug_params = np.vstack([existing_params, valid_params[keep_arr]])
    aug_cd = np.vstack([existing_cd, valid_cd[keep_arr]])
    aug_pc = np.vstack([existing_pc, valid_pc[keep_arr]])

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez(
            output_path,
            parameters=aug_params,
            current_density=aug_cd,
            peroxide_current=aug_pc,
        )

    return aug_params, aug_cd, aug_pc


# ---------------------------------------------------------------------------
# Private helpers: convergence check
# ---------------------------------------------------------------------------

def _check_convergence(
    surrogate_loss: float,
    pde_loss: float,
    config: ISMOConfig,
) -> Tuple[bool, float]:
    """Check whether the surrogate-PDE gap meets the convergence criterion.

    Returns
    -------
    (converged, convergence_metric)
    """
    gap = abs(surrogate_loss - pde_loss)
    denominator = max(abs(pde_loss), config.convergence_atol)
    metric = gap / denominator
    converged = metric < config.convergence_rtol
    return converged, metric


# ---------------------------------------------------------------------------
# Private helpers: surrogate retraining
# ---------------------------------------------------------------------------

def _retrain_surrogate(
    surrogate: Any,
    surrogate_type: str,
    parameters: np.ndarray,
    current_density: np.ndarray,
    peroxide_current: np.ndarray,
    phi_applied: np.ndarray,
    config: ISMOConfig,
    iteration: int,
    output_dir: str,
) -> Tuple[Any, Dict[str, float | None]]:
    """Retrain the surrogate on augmented data.

    For NN ensemble: warm-start each member from current weights with
    reduced learning rate.  For RBF/POD-RBF/GP/PCE: refit from scratch.

    Parameters
    ----------
    surrogate : Any
        Current fitted surrogate model.
    surrogate_type : str
        Model type string (e.g. ``"nn_ensemble"``).
    parameters : np.ndarray of shape ``(N, 4)``
        Full augmented training parameters.
    current_density : np.ndarray of shape ``(N, n_eta)``
        Full augmented CD training data.
    peroxide_current : np.ndarray of shape ``(N, n_eta)``
        Full augmented PC training data.
    phi_applied : np.ndarray
        Voltage grid.
    config : ISMOConfig
        ISMO configuration.
    iteration : int
        Current iteration index.
    output_dir : str
        Base output directory.

    Returns
    -------
    (new_surrogate, retrain_metrics)
        retrain_metrics has keys ``"val_cd_rmse"`` and ``"val_pc_rmse"``
        (may be None).
    """
    iter_dir = os.path.join(output_dir, f"iter_{iteration}")
    os.makedirs(iter_dir, exist_ok=True)

    # 80/20 train/val split
    n = len(parameters)
    rng = np.random.default_rng(seed=42 + iteration)
    idx = rng.permutation(n)
    n_val = max(1, int(0.2 * n))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    train_params = parameters[train_idx]
    train_cd = current_density[train_idx]
    train_pc = peroxide_current[train_idx]
    val_params = parameters[val_idx]
    val_cd = current_density[val_idx]
    val_pc = peroxide_current[val_idx]

    metrics: Dict[str, float | None] = {"val_cd_rmse": None, "val_pc_rmse": None}

    if surrogate_type == "nn_ensemble":
        new_surrogate = _retrain_nn_ensemble(
            surrogate=surrogate,
            train_params=train_params,
            train_cd=train_cd,
            train_pc=train_pc,
            val_params=val_params,
            val_cd=val_cd,
            val_pc=val_pc,
            phi_applied=phi_applied,
            config=config,
            save_dir=os.path.join(iter_dir, "nn_ensemble"),
        )
        # Compute validation RMSE on retrained model
        val_pred = new_surrogate.predict_batch(val_params)
        metrics["val_cd_rmse"] = float(np.sqrt(
            np.mean((val_pred["current_density"] - val_cd) ** 2)
        ))
        metrics["val_pc_rmse"] = float(np.sqrt(
            np.mean((val_pred["peroxide_current"] - val_pc) ** 2)
        ))
    else:
        # RBF, POD-RBF, GP, PCE: refit from scratch on a deep copy
        # to avoid mutating the original surrogate object.
        import copy
        import pickle

        new_surrogate = copy.deepcopy(surrogate)
        new_surrogate.fit(
            train_params, train_cd, train_pc, phi_applied,
        )
        save_path = os.path.join(iter_dir, f"model_{surrogate_type}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(new_surrogate, f)
        # Compute validation RMSE if predict_batch is available
        if hasattr(new_surrogate, "predict_batch"):
            val_pred = new_surrogate.predict_batch(val_params)
            metrics["val_cd_rmse"] = float(np.sqrt(
                np.mean((val_pred["current_density"] - val_cd) ** 2)
            ))
            metrics["val_pc_rmse"] = float(np.sqrt(
                np.mean((val_pred["peroxide_current"] - val_pc) ** 2)
            ))

    return new_surrogate, metrics


def _retrain_nn_ensemble(
    surrogate: Any,
    train_params: np.ndarray,
    train_cd: np.ndarray,
    train_pc: np.ndarray,
    val_params: np.ndarray,
    val_cd: np.ndarray,
    val_pc: np.ndarray,
    phi_applied: np.ndarray,
    config: ISMOConfig,
    save_dir: str,
) -> Any:
    """Warm-start retrain each NN ensemble member and rebuild wrapper.

    Loads existing weights from each member, calls ``fit()`` with reduced
    learning rate and epochs, then rebuilds the ``EnsembleMeanWrapper``.

    Parameters
    ----------
    surrogate : EnsembleMeanWrapper
        Current fitted ensemble.
    train_params, train_cd, train_pc : np.ndarray
        Training data.
    val_params, val_cd, val_pc : np.ndarray
        Validation data.
    phi_applied : np.ndarray
        Voltage grid.
    config : ISMOConfig
        ISMO configuration (for ``retrain_epochs`` and ``warm_start_retrain``).
    save_dir : str
        Directory to save retrained members.

    Returns
    -------
    EnsembleMeanWrapper
        New ensemble with retrained members.
    """
    from Surrogate.ensemble import EnsembleMeanWrapper
    from Surrogate.nn_model import NNSurrogateModel

    retrained_models: List[Any] = []

    for idx, member in enumerate(surrogate.models):
        # Create a new model with the same architecture
        new_member = NNSurrogateModel(
            hidden=member._hidden,
            n_blocks=member._n_blocks,
            seed=member._seed,
            device=member._device,
        )

        if config.warm_start_retrain and member._model is not None:
            # Warm start: correct weights for normalizer shift before loading.
            # Without this correction the old state dict assumes the old
            # normalizer statistics, leading to a large transient error at
            # the start of fine-tuning.  Use the analytical correction from
            # ``ismo_retrain._correct_weights_for_normalizer_shift``.
            #
            # For production ISMO loops prefer
            # ``ismo_retrain.retrain_nn_ensemble`` which handles this
            # automatically with quality checks and fallback logic.
            from Surrogate.nn_model import ZScoreNormalizer
            from Surrogate.ismo_retrain import _correct_weights_for_normalizer_shift

            old_state = member._model.state_dict()

            # Compute new normalizers on the merged training data
            X_log = train_params.copy().astype(np.float64)
            X_log[:, 0] = np.log10(np.maximum(X_log[:, 0], 1e-30))
            X_log[:, 1] = np.log10(np.maximum(X_log[:, 1], 1e-30))
            new_input_norm = ZScoreNormalizer.from_data(X_log)

            Y_all = np.concatenate([train_cd, train_pc], axis=1)
            new_output_norm = ZScoreNormalizer.from_data(Y_all)

            corrected_state = _correct_weights_for_normalizer_shift(
                old_state,
                member._input_normalizer,
                new_input_norm,
                member._output_normalizer,
                new_output_norm,
            )

            new_member.fit(
                train_params,
                train_cd,
                train_pc,
                phi_applied,
                epochs=config.retrain_epochs,
                lr=1e-4,
                patience=50,
                val_parameters=val_params,
                val_cd=val_cd,
                val_pc=val_pc,
                verbose=False,
                warm_start_state_dict=corrected_state,
            )
        else:
            new_member.fit(
                train_params,
                train_cd,
                train_pc,
                phi_applied,
                epochs=config.retrain_epochs,
                lr=1e-3,
                patience=max(100, config.retrain_epochs // 5),
                val_parameters=val_params,
                val_cd=val_cd,
                val_pc=val_pc,
                verbose=config.verbose,
            )

        # Save retrained member
        member_dir = os.path.join(save_dir, f"member_{idx}", "saved_model")
        os.makedirs(member_dir, exist_ok=True)
        new_member.save(member_dir)

        retrained_models.append(new_member)

    return EnsembleMeanWrapper(retrained_models)


# ---------------------------------------------------------------------------
# Private helpers: result serialization
# ---------------------------------------------------------------------------

def _save_ismo_result(result: ISMOResult, output_dir: str) -> None:
    """Save ISMOResult to JSON for post-hoc analysis.

    Parameters
    ----------
    result : ISMOResult
        The completed ISMO result.
    output_dir : str
        Directory to save to.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "ismo_result.json")

    # Convert to JSON-serializable dict
    d = asdict(result)

    # Convert numpy types and tuples
    def _convert(obj: Any) -> Any:
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, tuple):
            return [_convert(x) for x in obj]
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(x) for x in obj]
        return obj

    d = _convert(d)

    with open(path, "w") as f:
        json.dump(d, f, indent=2)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_ismo(
    surrogate: Any,
    target_cd: np.ndarray,
    target_pc: np.ndarray,
    training_params: np.ndarray,
    training_cd: np.ndarray,
    training_pc: np.ndarray,
    phi_applied: np.ndarray,
    bounds_k0_1: Tuple[float, float],
    bounds_k0_2: Tuple[float, float],
    bounds_alpha: Tuple[float, float],
    pde_solver_kwargs: dict,
    config: ISMOConfig | None = None,
    subset_idx: np.ndarray | None = None,
) -> ISMOResult:
    """Run the Iterative Surrogate Model Optimization loop.

    Wraps the existing multistart + cascade pipeline in an outer loop that
    iteratively refines the surrogate by evaluating PDE at the optimizer's
    best candidates and retraining.

    Parameters
    ----------
    surrogate : Any
        Fitted surrogate model (NN ensemble, RBF, POD-RBF, GP, or PCE).
    target_cd : np.ndarray
        Target current density I-V curve.
    target_pc : np.ndarray
        Target peroxide current I-V curve.
    training_params : np.ndarray of shape ``(N, 4)``
        Current training parameter sets.
    training_cd : np.ndarray of shape ``(N, n_eta)``
        Current training CD data.
    training_pc : np.ndarray of shape ``(N, n_eta)``
        Current training PC data.
    phi_applied : np.ndarray
        Voltage grid.
    bounds_k0_1 : tuple of float
        ``(lo, hi)`` bounds for k0_1 in physical space.
    bounds_k0_2 : tuple of float
        ``(lo, hi)`` bounds for k0_2 in physical space.
    bounds_alpha : tuple of float
        ``(lo, hi)`` bounds for alpha (shared for both reactions).
    pde_solver_kwargs : dict
        Must contain ``base_solver_params``, ``steady``, ``observable_scale``,
        and optionally ``mesh``.
    config : ISMOConfig or None
        ISMO configuration.  If None, uses defaults.
    subset_idx : np.ndarray or None
        Voltage subset indices for objective computation.

    Returns
    -------
    ISMOResult
        Final result with convergence info, iteration history, and paths.
    """
    config = config or ISMOConfig()
    ms_config = config.multistart_config or MultiStartConfig()

    total_pde_evals = 0
    current_surrogate = surrogate
    current_params = training_params.copy()
    current_cd = training_cd.copy()
    current_pc = training_pc.copy()
    iterations: List[ISMOIteration] = []
    best_pde_loss_global = float("inf")
    best_params_global: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    stall_count = 0
    t_start = time.time()

    # Determine secondary_weight from multistart config
    secondary_weight = ms_config.secondary_weight

    if config.verbose:
        print(
            f"\n{'=' * 60}\n"
            f"ISMO: Starting iterative surrogate model optimization\n"
            f"  max_iterations={config.max_iterations}, "
            f"samples_per_iter={config.samples_per_iteration}, "
            f"budget={config.total_pde_budget}\n"
            f"  surrogate_type={config.surrogate_type}, "
            f"acquisition={config.acquisition_strategy.value}\n"
            f"  convergence_rtol={config.convergence_rtol}, "
            f"convergence_atol={config.convergence_atol}\n"
            f"{'=' * 60}",
            flush=True,
        )

    for i in range(config.max_iterations):
        t_iter = time.time()

        if config.verbose:
            print(
                f"\n--- ISMO Iteration {i} ---\n"
                f"  Training set size: {len(current_params)}\n"
                f"  PDE evals so far: {total_pde_evals}/{config.total_pde_budget}",
                flush=True,
            )

        # --- Step 1: Optimize current surrogate ---
        if config.verbose:
            print("  Step 1: Running multistart + cascade optimization...", flush=True)

        multistart_result = run_multistart_inference(
            current_surrogate,
            target_cd,
            target_pc,
            bounds_k0_1,
            bounds_k0_2,
            bounds_alpha,
            config=config.multistart_config,
            subset_idx=subset_idx,
        )

        cascade_result = run_cascade_inference(
            current_surrogate,
            target_cd,
            target_pc,
            initial_k0=[multistart_result.best_k0_1, multistart_result.best_k0_2],
            initial_alpha=[multistart_result.best_alpha_1, multistart_result.best_alpha_2],
            bounds_k0_1=bounds_k0_1,
            bounds_k0_2=bounds_k0_2,
            bounds_alpha=bounds_alpha,
            config=config.cascade_config,
            subset_idx=subset_idx,
        )

        # Best point from optimizer
        best_opt = np.array([
            cascade_result.best_k0_1,
            cascade_result.best_k0_2,
            cascade_result.best_alpha_1,
            cascade_result.best_alpha_2,
        ])

        # --- Step 2: Select acquisition points ---
        n_remaining_budget = config.total_pde_budget - total_pde_evals
        n_samples = min(config.samples_per_iteration, n_remaining_budget)
        if n_samples <= 0:
            if config.verbose:
                print("  Budget exhausted before acquisition.", flush=True)
            break

        if config.verbose:
            print(
                f"  Step 2: Acquiring {n_samples} new sample points "
                f"({config.acquisition_strategy.value})...",
                flush=True,
            )

        # Use seed + iteration to avoid duplicate LHS across iterations
        acq_seed = (ms_config.seed if ms_config.seed else 42) + i

        if config.acquisition_strategy == AcquisitionStrategy.OPTIMIZER_TRAJECTORY:
            candidates = _acquire_optimizer_trajectory(
                multistart_result=multistart_result,
                cascade_result=cascade_result,
                n_candidates=config.n_top_candidates_to_evaluate,
                n_total=n_samples,
                bounds_k0_1=bounds_k0_1,
                bounds_k0_2=bounds_k0_2,
                bounds_alpha=bounds_alpha,
                neighborhood_fraction=config.neighborhood_fraction,
                seed=acq_seed,
            )
        elif config.acquisition_strategy == AcquisitionStrategy.UNCERTAINTY:
            candidates = _acquire_uncertainty(
                surrogate=current_surrogate,
                n_total=n_samples,
                bounds_k0_1=bounds_k0_1,
                bounds_k0_2=bounds_k0_2,
                bounds_alpha=bounds_alpha,
                best_params=best_opt,
                neighborhood_fraction=config.neighborhood_fraction,
                seed=acq_seed,
                multistart_result=multistart_result,
                cascade_result=cascade_result,
                n_candidates=config.n_top_candidates_to_evaluate,
            )
        else:  # HYBRID
            candidates = _acquire_hybrid(
                surrogate=current_surrogate,
                multistart_result=multistart_result,
                cascade_result=cascade_result,
                n_candidates=config.n_top_candidates_to_evaluate,
                n_total=n_samples,
                bounds_k0_1=bounds_k0_1,
                bounds_k0_2=bounds_k0_2,
                bounds_alpha=bounds_alpha,
                best_params=best_opt,
                neighborhood_fraction=config.neighborhood_fraction,
                seed=acq_seed,
            )

        # --- Step 3: Evaluate candidates with PDE solver ---
        # Include the optimizer's best point (best_opt) in the candidate
        # list so its PDE loss is evaluated and convergence is checked at
        # the actual optimum, not just at acquisition points.
        best_opt_included = False
        if best_opt is not None:
            # Prepend best_opt if it's not already in candidates
            dists = np.linalg.norm(candidates - best_opt[np.newaxis, :], axis=1)
            if np.min(dists) > 1e-12:
                candidates = np.vstack([best_opt[np.newaxis, :], candidates])
                best_opt_included = True

        if config.verbose:
            extra = " (includes optimizer best)" if best_opt_included else ""
            print(
                f"  Step 3: Evaluating {len(candidates)} candidates with PDE solver{extra}...",
                flush=True,
            )

        cd_new, pc_new, pde_losses, converged_flags = _evaluate_candidates_pde(
            candidates=candidates,
            phi_applied=phi_applied,
            target_cd=target_cd,
            target_pc=target_pc,
            pde_solver_kwargs=pde_solver_kwargs,
            secondary_weight=secondary_weight,
            subset_idx=subset_idx,
            verbose=config.verbose,
        )
        total_pde_evals += len(candidates)

        # --- Step 4: Measure surrogate-PDE gap at best candidate ---
        surr_losses = _compute_surrogate_losses(
            current_surrogate, candidates, target_cd, target_pc,
            secondary_weight, subset_idx,
        )

        # Find best candidate by PDE loss (among converged)
        converged_mask_arr = np.array(converged_flags)
        if np.any(converged_mask_arr):
            # Only consider converged candidates for "best"
            masked_losses = np.where(converged_mask_arr, pde_losses, np.inf)
            best_idx = int(np.argmin(masked_losses))
        else:
            best_idx = int(np.argmin(pde_losses))

        converged_flag, gap = _check_convergence(
            surr_losses[best_idx], pde_losses[best_idx], config,
        )

        if config.verbose:
            print(
                f"  Step 4: Surrogate-PDE gap analysis\n"
                f"    Best candidate idx={best_idx}\n"
                f"    Surrogate loss = {surr_losses[best_idx]:.6e}\n"
                f"    PDE loss      = {pde_losses[best_idx]:.6e}\n"
                f"    Gap metric    = {gap:.4f} "
                f"(threshold={config.convergence_rtol})\n"
                f"    Converged     = {converged_flag}",
                flush=True,
            )

        # --- Step 5: Augment training data ---
        if config.verbose:
            n_conv = sum(converged_flags)
            print(
                f"  Step 5: Augmenting training data "
                f"({n_conv}/{len(candidates)} converged)...",
                flush=True,
            )

        aug_path = os.path.join(config.output_dir, f"iter_{i}", "augmented_data.npz")
        current_params, current_cd, current_pc = _augment_training_data(
            current_params, current_cd, current_pc,
            candidates, cd_new, pc_new, converged_flags,
            output_path=aug_path,
        )

        n_new_added = len(current_params) - (
            len(training_params) if i == 0
            else iterations[-1].n_total_training
        )

        if config.verbose:
            print(
                f"    New training set size: {len(current_params)} "
                f"(+{n_new_added})",
                flush=True,
            )

        # Handle all-candidates-failed: skip retraining
        if n_new_added == 0:
            if config.verbose:
                print(
                    "  WARNING: No new samples added (all candidates failed "
                    "or duplicated). Skipping retraining.",
                    flush=True,
                )

        # --- Step 6: Record iteration (before potential retraining) ---
        retrain_cd_rmse: float | None = None
        retrain_pc_rmse: float | None = None

        # --- Step 7: Check convergence (before retraining) ---
        if converged_flag:
            best_p = tuple(candidates[best_idx].tolist())
            iteration_result = ISMOIteration(
                iteration=i,
                n_new_samples=len(candidates),
                n_total_training=len(current_params),
                surrogate_loss_at_best=float(surr_losses[best_idx]),
                pde_loss_at_best=float(pde_losses[best_idx]),
                surrogate_pde_gap=float(abs(surr_losses[best_idx] - pde_losses[best_idx])),
                convergence_metric=float(gap),
                best_params=best_p,
                best_loss=float(pde_losses[best_idx]),
                candidate_pde_losses=tuple(pde_losses.tolist()),
                candidate_surrogate_losses=tuple(surr_losses.tolist()),
                retrain_val_rmse_cd=None,
                retrain_val_rmse_pc=None,
                wall_time_s=time.time() - t_iter,
            )
            iterations.append(iteration_result)

            if config.verbose:
                print(
                    f"\n  CONVERGED at iteration {i}! "
                    f"Gap={gap:.4f} < rtol={config.convergence_rtol}",
                    flush=True,
                )

            result = ISMOResult(
                converged=True,
                termination_reason="converged",
                n_iterations=i + 1,
                total_pde_evals=total_pde_evals,
                iteration_history=tuple(iterations),
                final_params=best_p,
                final_loss=float(pde_losses[best_idx]),
                final_surrogate_path=None,
                augmented_data_path=aug_path,
                total_wall_time_s=time.time() - t_start,
            )
            _save_ismo_result(result, config.output_dir)
            return result

        # --- Step 8: Retrain surrogate (if budget remains and new data added) ---
        if total_pde_evals < config.total_pde_budget and n_new_added > 0:
            if config.verbose:
                print(
                    f"  Step 8: Retraining surrogate ({config.surrogate_type})...",
                    flush=True,
                )

            current_surrogate, retrain_metrics = _retrain_surrogate(
                current_surrogate,
                config.surrogate_type,
                current_params,
                current_cd,
                current_pc,
                phi_applied,
                config,
                i,
                config.output_dir,
            )
            retrain_cd_rmse = retrain_metrics.get("val_cd_rmse")
            retrain_pc_rmse = retrain_metrics.get("val_pc_rmse")

            if config.verbose and retrain_cd_rmse is not None:
                print(
                    f"    Retrain val RMSE: CD={retrain_cd_rmse:.4e}, "
                    f"PC={retrain_pc_rmse:.4e}",
                    flush=True,
                )

        # --- Record iteration ---
        best_p = tuple(candidates[best_idx].tolist())
        iteration_result = ISMOIteration(
            iteration=i,
            n_new_samples=len(candidates),
            n_total_training=len(current_params),
            surrogate_loss_at_best=float(surr_losses[best_idx]),
            pde_loss_at_best=float(pde_losses[best_idx]),
            surrogate_pde_gap=float(abs(surr_losses[best_idx] - pde_losses[best_idx])),
            convergence_metric=float(gap),
            best_params=best_p,
            best_loss=float(pde_losses[best_idx]),
            candidate_pde_losses=tuple(pde_losses.tolist()),
            candidate_surrogate_losses=tuple(surr_losses.tolist()),
            retrain_val_rmse_cd=retrain_cd_rmse,
            retrain_val_rmse_pc=retrain_pc_rmse,
            wall_time_s=time.time() - t_iter,
        )
        iterations.append(iteration_result)

        # --- Step 9: No-improvement check ---
        if pde_losses[best_idx] < best_pde_loss_global:
            best_pde_loss_global = float(pde_losses[best_idx])
            best_params_global = best_p
            stall_count = 0
        else:
            stall_count += 1

        if stall_count >= 4:
            if config.verbose:
                print(
                    f"\n  No improvement for {stall_count} consecutive iterations. "
                    f"Stopping.",
                    flush=True,
                )
            result = ISMOResult(
                converged=False,
                termination_reason="no_improvement",
                n_iterations=i + 1,
                total_pde_evals=total_pde_evals,
                iteration_history=tuple(iterations),
                final_params=best_params_global,
                final_loss=best_pde_loss_global,
                final_surrogate_path=None,
                augmented_data_path=aug_path,
                total_wall_time_s=time.time() - t_start,
            )
            _save_ismo_result(result, config.output_dir)
            return result

        # --- Budget check at end of iteration ---
        if total_pde_evals >= config.total_pde_budget:
            if config.verbose:
                print(
                    f"\n  PDE budget exhausted ({total_pde_evals}/{config.total_pde_budget}).",
                    flush=True,
                )
            result = ISMOResult(
                converged=False,
                termination_reason="budget_exhausted",
                n_iterations=i + 1,
                total_pde_evals=total_pde_evals,
                iteration_history=tuple(iterations),
                final_params=best_params_global,
                final_loss=best_pde_loss_global,
                final_surrogate_path=None,
                augmented_data_path=aug_path,
                total_wall_time_s=time.time() - t_start,
            )
            _save_ismo_result(result, config.output_dir)
            return result

    # Reached max_iterations
    if config.verbose:
        print(
            f"\n  Reached max_iterations={config.max_iterations}.",
            flush=True,
        )

    aug_path = os.path.join(
        config.output_dir,
        f"iter_{config.max_iterations - 1}",
        "augmented_data.npz",
    )
    result = ISMOResult(
        converged=False,
        termination_reason="max_iterations",
        n_iterations=config.max_iterations,
        total_pde_evals=total_pde_evals,
        iteration_history=tuple(iterations),
        final_params=best_params_global,
        final_loss=best_pde_loss_global,
        final_surrogate_path=None,
        augmented_data_path=aug_path,
        total_wall_time_s=time.time() - t_start,
    )
    _save_ismo_result(result, config.output_dir)
    return result
