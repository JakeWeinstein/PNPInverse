"""Multi-start Latin Hypercube grid search with gradient-based polish.

Strategy 3 for BV kinetics parameter inference: exhaustively sample the
4D parameter space using a Latin Hypercube design, evaluate all points
via the fast ``predict_batch`` surrogate path, then polish the top
candidates with L-BFGS-B.

Public API
----------
MultiStartConfig
    Frozen configuration dataclass.
MultiStartCandidate
    Per-candidate result snapshot.
MultiStartResult
    Final result container.
run_multistart_inference
    Main entry point.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import LatinHypercube

from Surrogate.surrogate_model import BVSurrogateModel


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MultiStartConfig:
    """Frozen configuration for multi-start Latin Hypercube inference.

    Attributes
    ----------
    n_grid : int
        Number of Latin Hypercube sample points.
    n_top_candidates : int
        Number of best grid points to polish with L-BFGS-B.
    polish_maxiter : int
        Maximum L-BFGS-B iterations per polish run.
    secondary_weight : float
        Weight on the peroxide current objective term.
    fd_step : float
        Finite difference step size for gradient computation.
    use_shallow_subset : bool
        If True and ``subset_idx`` is provided, restrict the objective
        to the shallow voltage subset (matching v9 Phase 2).
    seed : int
        Random seed for the Latin Hypercube sampler.
    verbose : bool
        If True, print progress during grid evaluation and polishing.
    """

    n_grid: int = 20_000
    n_top_candidates: int = 20
    polish_maxiter: int = 60
    secondary_weight: float = 1.0
    fd_step: float = 1e-5
    use_shallow_subset: bool = True
    seed: int = 42
    verbose: bool = True


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MultiStartCandidate:
    """Snapshot of a single candidate from the multi-start procedure.

    Attributes
    ----------
    rank : int
        Rank in the grid-search ordering (0 = best grid point).
    k0_1, k0_2, alpha_1, alpha_2 : float
        Polished parameter values (physical space).
    grid_loss : float
        Objective value from the grid evaluation (before polish).
    polished_loss : float
        Objective value after L-BFGS-B polish.
    polish_iters : int
        Number of L-BFGS-B iterations used.
    polish_n_evals : int
        Number of surrogate evaluations during polish.
    """

    rank: int
    k0_1: float
    k0_2: float
    alpha_1: float
    alpha_2: float
    grid_loss: float
    polished_loss: float
    polish_iters: int
    polish_n_evals: int


@dataclass(frozen=True)
class MultiStartResult:
    """Final result of multi-start Latin Hypercube inference.

    Attributes
    ----------
    best_k0_1, best_k0_2, best_alpha_1, best_alpha_2 : float
        Best recovered parameter values (physical space).
    best_loss : float
        Objective value at the best polished candidate.
    best_candidate_rank : int
        Grid-search rank of the winning candidate.
    n_grid_points : int
        Total number of grid points evaluated.
    n_candidates_polished : int
        Number of candidates that were polished.
    candidates : tuple
        Tuple of ``MultiStartCandidate`` (sorted by polished_loss).
    grid_eval_time_s : float
        Wall-clock time for grid evaluation (seconds).
    polish_time_s : float
        Wall-clock time for polishing all candidates (seconds).
    total_time_s : float
        Total wall-clock time (seconds).
    """

    best_k0_1: float
    best_k0_2: float
    best_alpha_1: float
    best_alpha_2: float
    best_loss: float
    best_candidate_rank: int
    n_grid_points: int
    n_candidates_polished: int
    candidates: tuple
    grid_eval_time_s: float
    polish_time_s: float
    total_time_s: float


# ---------------------------------------------------------------------------
# Latin Hypercube sampling in mixed log/linear space
# ---------------------------------------------------------------------------

def _generate_lhs_grid(
    n_points: int,
    bounds_log_k0_1: Tuple[float, float],
    bounds_log_k0_2: Tuple[float, float],
    bounds_alpha: Tuple[float, float],
    seed: int,
) -> np.ndarray:
    """Generate LHS samples in 4D: log10(k0_1), log10(k0_2), alpha_1, alpha_2.

    Parameters
    ----------
    n_points : int
        Number of sample points.
    bounds_log_k0_1, bounds_log_k0_2 : (lo, hi)
        Bounds in log10 space for k0_1 and k0_2.
    bounds_alpha : (lo, hi)
        Bounds for alpha_1 and alpha_2 (linear space).
    seed : int
        Random seed.

    Returns
    -------
    np.ndarray of shape (n_points, 4)
        Columns: [k0_1, k0_2, alpha_1, alpha_2] in **physical** space.
    """
    sampler = LatinHypercube(d=4, seed=seed)
    unit_samples = sampler.random(n=n_points)  # (n_points, 4) in [0, 1]

    # Scale to bounds: dimensions 0,1 in log10 space, 2,3 in linear
    lo = np.array([
        bounds_log_k0_1[0], bounds_log_k0_2[0],
        bounds_alpha[0], bounds_alpha[0],
    ])
    hi = np.array([
        bounds_log_k0_1[1], bounds_log_k0_2[1],
        bounds_alpha[1], bounds_alpha[1],
    ])
    scaled = lo + unit_samples * (hi - lo)

    # Convert k0 from log10 to physical space
    params = np.empty_like(scaled)
    params[:, 0] = 10.0 ** scaled[:, 0]
    params[:, 1] = 10.0 ** scaled[:, 1]
    params[:, 2] = scaled[:, 2]
    params[:, 3] = scaled[:, 3]

    return params


# ---------------------------------------------------------------------------
# Vectorized batch objective evaluation
# ---------------------------------------------------------------------------

def _evaluate_grid_objectives(
    surrogate: BVSurrogateModel,
    params_grid: np.ndarray,
    target_cd: np.ndarray,
    target_pc: np.ndarray,
    secondary_weight: float,
    subset_idx: np.ndarray | None,
) -> np.ndarray:
    """Evaluate the objective for all grid points using predict_batch.

    Parameters
    ----------
    surrogate : BVSurrogateModel
        Fitted surrogate model.
    params_grid : np.ndarray of shape (M, 4)
        Grid parameter samples in physical space.
    target_cd, target_pc : np.ndarray
        Target I-V curves (full or subset).
    secondary_weight : float
        Weight on the peroxide current term.
    subset_idx : np.ndarray or None
        If provided, restrict predictions to these voltage indices.

    Returns
    -------
    np.ndarray of shape (M,)
        Objective values (NaN-safe).
    """
    pred = surrogate.predict_batch(params_grid)
    cd_pred = pred["current_density"]  # (M, n_eta)
    pc_pred = pred["peroxide_current"]  # (M, n_eta)

    # Apply subset selection to both predictions and targets
    if subset_idx is not None:
        cd_pred = cd_pred[:, subset_idx]
        pc_pred = pc_pred[:, subset_idx]
        target_cd = target_cd[subset_idx]
        target_pc = target_pc[subset_idx]

    # NaN masks for targets
    valid_cd = ~np.isnan(target_cd)
    valid_pc = ~np.isnan(target_pc)

    # Compute residuals (broadcast: (M, n_sub) - (n_sub,) -> (M, n_sub))
    cd_diff = cd_pred[:, valid_cd] - target_cd[valid_cd]
    pc_diff = pc_pred[:, valid_pc] - target_pc[valid_pc]

    # Handle NaN predictions gracefully
    cd_diff = np.where(np.isnan(cd_diff), 0.0, cd_diff)
    pc_diff = np.where(np.isnan(pc_diff), 0.0, pc_diff)

    j_cd = 0.5 * np.sum(cd_diff ** 2, axis=1)
    j_pc = 0.5 * np.sum(pc_diff ** 2, axis=1)

    objectives = j_cd + secondary_weight * j_pc

    # Mark points with any NaN prediction as infinite
    has_nan_cd = np.any(np.isnan(cd_pred), axis=1)
    has_nan_pc = np.any(np.isnan(pc_pred), axis=1)
    objectives = np.where(has_nan_cd | has_nan_pc, np.inf, objectives)

    return objectives


# ---------------------------------------------------------------------------
# Single-candidate polish with L-BFGS-B
# ---------------------------------------------------------------------------

def _polish_candidate(
    surrogate: BVSurrogateModel,
    x0_physical: np.ndarray,
    target_cd: np.ndarray,
    target_pc: np.ndarray,
    bounds_log: list[Tuple[float, float]],
    config: MultiStartConfig,
    subset_idx: np.ndarray | None,
) -> Tuple[np.ndarray, float, int, int]:
    """Polish a single candidate using L-BFGS-B.

    Parameters
    ----------
    surrogate : BVSurrogateModel
        Fitted surrogate.
    x0_physical : np.ndarray of shape (4,)
        [k0_1, k0_2, alpha_1, alpha_2] in physical space.
    target_cd, target_pc : np.ndarray
        Target I-V curves (full or subset-indexed).
    bounds_log : list of (lo, hi)
        Bounds in optimizer space [log10(k0_1), log10(k0_2), alpha_1, alpha_2].
    config : MultiStartConfig
        Configuration.
    subset_idx : np.ndarray or None
        Voltage subset indices.

    Returns
    -------
    (x_opt_physical, loss, n_iters, n_evals)
        Optimized params in physical space, loss, iteration count, eval count.
    """
    # Pre-compute subsetted targets and NaN masks (closure captures these)
    if subset_idx is not None:
        _tgt_cd = target_cd[subset_idx]
        _tgt_pc = target_pc[subset_idx]
    else:
        _tgt_cd = np.asarray(target_cd, dtype=float)
        _tgt_pc = np.asarray(target_pc, dtype=float)

    _valid_cd = ~np.isnan(_tgt_cd)
    _valid_pc = ~np.isnan(_tgt_pc)

    n_evals = 0

    def _objective(x: np.ndarray) -> float:
        nonlocal n_evals
        k0_1, k0_2 = 10.0 ** x[0], 10.0 ** x[1]
        alpha_1, alpha_2 = x[2], x[3]

        pred = surrogate.predict(k0_1, k0_2, alpha_1, alpha_2)
        cd_sim = pred["current_density"]
        pc_sim = pred["peroxide_current"]

        if subset_idx is not None:
            cd_sim = cd_sim[subset_idx]
            pc_sim = pc_sim[subset_idx]

        cd_diff = cd_sim[_valid_cd] - _tgt_cd[_valid_cd]
        pc_diff = pc_sim[_valid_pc] - _tgt_pc[_valid_pc]

        j = 0.5 * np.sum(cd_diff ** 2) + config.secondary_weight * 0.5 * np.sum(pc_diff ** 2)
        n_evals += 1
        return float(j)

    def _gradient(x: np.ndarray) -> np.ndarray:
        h = config.fd_step
        grad = np.zeros(4, dtype=float)
        for i in range(4):
            xp, xm = x.copy(), x.copy()
            xp[i] += h
            xm[i] -= h
            grad[i] = (_objective(xp) - _objective(xm)) / (2 * h)
        return grad

    # Convert initial point to optimizer space
    x0 = np.array([
        np.log10(max(x0_physical[0], 1e-30)),
        np.log10(max(x0_physical[1], 1e-30)),
        x0_physical[2],
        x0_physical[3],
    ], dtype=float)

    result = minimize(
        _objective,
        x0,
        jac=_gradient,
        method="L-BFGS-B",
        bounds=bounds_log,
        options={
            "maxiter": config.polish_maxiter,
            "ftol": 1e-14,
            "gtol": 1e-8,
        },
    )

    x_opt = result.x
    x_opt_physical = np.array([
        10.0 ** x_opt[0],
        10.0 ** x_opt[1],
        x_opt[2],
        x_opt[3],
    ])

    n_iters = result.get("nit", 0)
    return x_opt_physical, float(result.fun), int(n_iters), n_evals


# ---------------------------------------------------------------------------
# Main multi-start inference
# ---------------------------------------------------------------------------

def run_multistart_inference(
    surrogate: BVSurrogateModel,
    target_cd: np.ndarray,
    target_pc: np.ndarray,
    bounds_k0_1: Tuple[float, float],
    bounds_k0_2: Tuple[float, float],
    bounds_alpha: Tuple[float, float],
    config: MultiStartConfig | None = None,
    subset_idx: np.ndarray | None = None,
) -> MultiStartResult:
    """Run multi-start Latin Hypercube grid search with L-BFGS-B polish.

    Parameters
    ----------
    surrogate : BVSurrogateModel
        Fitted surrogate model.
    target_cd, target_pc : np.ndarray
        Target I-V curves.  If ``subset_idx`` is provided, these should
        be the full-grid targets (subsetting is applied internally).
    bounds_k0_1, bounds_k0_2 : (lo, hi)
        Bounds for k0 in **physical** space.
    bounds_alpha : (lo, hi)
        Bounds for alpha_1 and alpha_2.
    config : MultiStartConfig or None
        Algorithm configuration (defaults applied if None).
    subset_idx : np.ndarray or None
        Indices into the surrogate's voltage grid for the shallow subset.
        If provided and ``config.use_shallow_subset`` is True, the
        objective is restricted to these voltage points.

    Returns
    -------
    MultiStartResult
        Final result with all polished candidates.
    """
    if config is None:
        config = MultiStartConfig()

    t_start = time.time()

    # Resolve subset usage
    active_subset = subset_idx if (config.use_shallow_subset and subset_idx is not None) else None

    # Log-space bounds
    bounds_log_k0_1 = (np.log10(max(bounds_k0_1[0], 1e-30)), np.log10(bounds_k0_1[1]))
    bounds_log_k0_2 = (np.log10(max(bounds_k0_2[0], 1e-30)), np.log10(bounds_k0_2[1]))

    # -----------------------------------------------------------------------
    # Phase 1: Generate LHS grid and evaluate
    # -----------------------------------------------------------------------
    if config.verbose:
        print(f"\n  [MultiStart] Generating {config.n_grid:,} LHS samples (seed={config.seed})...")

    params_grid = _generate_lhs_grid(
        n_points=config.n_grid,
        bounds_log_k0_1=bounds_log_k0_1,
        bounds_log_k0_2=bounds_log_k0_2,
        bounds_alpha=bounds_alpha,
        seed=config.seed,
    )

    t_grid_start = time.time()

    objectives = _evaluate_grid_objectives(
        surrogate=surrogate,
        params_grid=params_grid,
        target_cd=target_cd,
        target_pc=target_pc,
        secondary_weight=config.secondary_weight,
        subset_idx=active_subset,
    )

    t_grid_end = time.time()
    grid_eval_time = t_grid_end - t_grid_start

    if config.verbose:
        n_valid = np.sum(np.isfinite(objectives))
        print(f"  [MultiStart] Grid evaluation: {grid_eval_time:.2f}s "
              f"({n_valid:,}/{config.n_grid:,} valid)")

    # Sort by objective, take top K
    sorted_idx = np.argsort(objectives)
    n_top = min(config.n_top_candidates, len(sorted_idx))
    top_indices = sorted_idx[:n_top]

    if config.verbose:
        print(f"  [MultiStart] Top-5 grid candidates:")
        for i, idx in enumerate(top_indices[:5]):
            p = params_grid[idx]
            print(f"    #{i}: k0=[{p[0]:.4e},{p[1]:.4e}] "
                  f"alpha=[{p[2]:.4f},{p[3]:.4f}] loss={objectives[idx]:.4e}")

    # -----------------------------------------------------------------------
    # Phase 2: Polish top candidates with L-BFGS-B
    # -----------------------------------------------------------------------
    if config.verbose:
        print(f"\n  [MultiStart] Polishing top-{n_top} candidates (maxiter={config.polish_maxiter})...")

    t_polish_start = time.time()

    bounds_log = [
        bounds_log_k0_1,
        bounds_log_k0_2,
        bounds_alpha,
        bounds_alpha,
    ]

    candidates: list[MultiStartCandidate] = []
    for rank, grid_idx in enumerate(top_indices):
        x0_phys = params_grid[grid_idx]
        grid_loss = float(objectives[grid_idx])

        x_opt, polished_loss, n_iters, n_evals = _polish_candidate(
            surrogate=surrogate,
            x0_physical=x0_phys,
            target_cd=target_cd,
            target_pc=target_pc,
            bounds_log=bounds_log,
            config=config,
            subset_idx=active_subset,
        )

        candidate = MultiStartCandidate(
            rank=rank,
            k0_1=float(x_opt[0]),
            k0_2=float(x_opt[1]),
            alpha_1=float(x_opt[2]),
            alpha_2=float(x_opt[3]),
            grid_loss=grid_loss,
            polished_loss=polished_loss,
            polish_iters=n_iters,
            polish_n_evals=n_evals,
        )
        candidates.append(candidate)

        if config.verbose:
            print(f"    Candidate #{rank}: grid={grid_loss:.4e} -> polished={polished_loss:.4e} "
                  f"k0=[{x_opt[0]:.4e},{x_opt[1]:.4e}] "
                  f"alpha=[{x_opt[2]:.4f},{x_opt[3]:.4f}] ({n_iters} iters)")

    t_polish_end = time.time()
    polish_time = t_polish_end - t_polish_start

    # Sort candidates by polished loss and pick the best
    candidates.sort(key=lambda c: c.polished_loss)
    best = candidates[0]

    total_time = time.time() - t_start

    if config.verbose:
        print(f"\n  [MultiStart] Best: k0=[{best.k0_1:.4e},{best.k0_2:.4e}] "
              f"alpha=[{best.alpha_1:.4f},{best.alpha_2:.4f}] "
              f"loss={best.polished_loss:.4e} (grid rank #{best.rank})")
        print(f"  [MultiStart] Time: grid={grid_eval_time:.2f}s, "
              f"polish={polish_time:.2f}s, total={total_time:.2f}s")

    return MultiStartResult(
        best_k0_1=best.k0_1,
        best_k0_2=best.k0_2,
        best_alpha_1=best.alpha_1,
        best_alpha_2=best.alpha_2,
        best_loss=best.polished_loss,
        best_candidate_rank=best.rank,
        n_grid_points=config.n_grid,
        n_candidates_polished=n_top,
        candidates=tuple(candidates),
        grid_eval_time_s=grid_eval_time,
        polish_time_s=polish_time,
        total_time_s=total_time,
    )
