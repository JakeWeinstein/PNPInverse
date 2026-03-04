"""Block Coordinate Descent (BCD) for multi-reaction BV kinetics inference.

Alternates between two 2D sub-problems, each optimizing a single
reaction's (k0, alpha) pair while holding the other reaction fixed.
Different secondary_weight values can be used per block, exploiting the
weight-sweep finding that low weight favours k0_1 recovery and high
weight favours k0_2 recovery.

Public API
----------
BCDConfig
    Frozen configuration dataclass.
BCDIterationResult
    Per-outer-iteration snapshot.
BCDResult
    Final result container.
run_block_coordinate_descent
    Main entry point.
"""

from __future__ import annotations

import io
import sys
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from Surrogate.objectives import (
    ReactionBlockSurrogateObjective,
    SurrogateObjective,
)
from Surrogate.surrogate_model import BVSurrogateModel


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BCDConfig:
    """Frozen configuration for Block Coordinate Descent.

    Attributes
    ----------
    max_outer_iters : int
        Maximum number of outer (alternating) iterations.
    inner_maxiter : int
        Maximum L-BFGS-B iterations per inner sub-problem.
    block_1_weight : float
        secondary_weight when optimizing reaction 1 (k0_1, alpha_1).
    block_2_weight : float
        secondary_weight when optimizing reaction 2 (k0_2, alpha_2).
    convergence_rtol : float
        Relative tolerance for parameter change (convergence check).
    convergence_atol_k0 : float
        Absolute tolerance for log10(k0) change (convergence check).
    convergence_atol_alpha : float
        Absolute tolerance for alpha change (convergence check).
    fd_step : float
        Finite difference step size for gradient computation.
    verbose : bool
        If True, print progress during optimization.
    """

    max_outer_iters: int = 10
    inner_maxiter: int = 30
    block_1_weight: float = 0.5
    block_2_weight: float = 2.0
    convergence_rtol: float = 1e-4
    convergence_atol_k0: float = 1e-6
    convergence_atol_alpha: float = 1e-5
    fd_step: float = 1e-5
    verbose: bool = True


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BCDIterationResult:
    """Snapshot of one outer BCD iteration.

    Attributes
    ----------
    iteration : int
        Outer iteration index (0-based).
    k0_1, k0_2 : float
        Recovered k0 values after this iteration.
    alpha_1, alpha_2 : float
        Recovered alpha values after this iteration.
    loss_after_block_1 : float
        Objective value after block 1 optimization.
    loss_after_block_2 : float
        Objective value after block 2 optimization.
    block_1_inner_iters : int
        Number of L-BFGS-B iterations used for block 1.
    block_2_inner_iters : int
        Number of L-BFGS-B iterations used for block 2.
    elapsed_s : float
        Wall-clock time for this outer iteration (seconds).
    """

    iteration: int
    k0_1: float
    k0_2: float
    alpha_1: float
    alpha_2: float
    loss_after_block_1: float
    loss_after_block_2: float
    block_1_inner_iters: int
    block_2_inner_iters: int
    elapsed_s: float


@dataclass(frozen=True)
class BCDResult:
    """Final result of Block Coordinate Descent.

    Attributes
    ----------
    k0_1, k0_2 : float
        Recovered k0 values.
    alpha_1, alpha_2 : float
        Recovered alpha values.
    final_loss : float
        Objective value at convergence (evaluated with weight=1.0).
    n_outer_iters : int
        Number of outer iterations performed.
    total_surrogate_evals : int
        Total surrogate evaluations across all blocks.
    converged : bool
        Whether the convergence criterion was met.
    convergence_reason : str
        Human-readable reason for termination.
    iteration_history : tuple of BCDIterationResult
        Per-iteration snapshots.
    elapsed_s : float
        Total wall-clock time (seconds).
    """

    k0_1: float
    k0_2: float
    alpha_1: float
    alpha_2: float
    final_loss: float
    n_outer_iters: int
    total_surrogate_evals: int
    converged: bool
    convergence_reason: str
    iteration_history: Tuple[BCDIterationResult, ...]
    elapsed_s: float


# ---------------------------------------------------------------------------
# Convergence check
# ---------------------------------------------------------------------------

def _check_convergence(
    prev: Tuple[float, float, float, float],
    curr: Tuple[float, float, float, float],
    config: BCDConfig,
) -> Tuple[bool, str]:
    """Check if parameters have converged between outer iterations.

    Parameters
    ----------
    prev, curr : (log10_k0_1, log10_k0_2, alpha_1, alpha_2)
        Previous and current parameter values (k0 in log10 space).
    config : BCDConfig
        Configuration with tolerances.

    Returns
    -------
    (converged, reason)
    """
    prev_arr = np.array(prev, dtype=float)
    curr_arr = np.array(curr, dtype=float)
    abs_change = np.abs(curr_arr - prev_arr)
    rel_change = abs_change / np.maximum(np.abs(prev_arr), 1e-30)

    # Absolute tolerance check: k0 (indices 0,1) and alpha (indices 2,3)
    k0_abs_ok = np.all(abs_change[:2] < config.convergence_atol_k0)
    alpha_abs_ok = np.all(abs_change[2:] < config.convergence_atol_alpha)

    # Relative tolerance check: all parameters
    rel_ok = np.all(rel_change < config.convergence_rtol)

    if rel_ok:
        return True, "relative change below rtol"
    if k0_abs_ok and alpha_abs_ok:
        return True, "absolute change below atol"
    return False, "not converged"


# ---------------------------------------------------------------------------
# Inner block optimization
# ---------------------------------------------------------------------------

def _optimize_block(
    surrogate: BVSurrogateModel,
    target_cd: np.ndarray,
    target_pc: np.ndarray,
    reaction_index: int,
    fixed_k0_other: float,
    fixed_alpha_other: float,
    x0: np.ndarray,
    bounds: Sequence[Tuple[float, float]],
    secondary_weight: float,
    inner_maxiter: int,
    fd_step: float,
) -> Tuple[np.ndarray, float, int, int]:
    """Optimize a single reaction block.

    Returns
    -------
    (x_opt, loss, n_iters, n_evals)
        Optimal control, loss value, L-BFGS-B iterations, surrogate evals.
    """
    obj = ReactionBlockSurrogateObjective(
        surrogate=surrogate,
        target_cd=target_cd,
        target_pc=target_pc,
        reaction_index=reaction_index,
        fixed_k0_other=fixed_k0_other,
        fixed_alpha_other=fixed_alpha_other,
        secondary_weight=secondary_weight,
        fd_step=fd_step,
    )

    result = minimize(
        obj.objective,
        x0,
        jac=obj.gradient,
        method="L-BFGS-B",
        bounds=list(bounds),
        options={
            "maxiter": inner_maxiter,
            "ftol": 1e-14,
            "gtol": 1e-8,
        },
    )

    n_iters = result.get("nit", 0)
    return result.x.copy(), float(result.fun), int(n_iters), obj.n_evals


# ---------------------------------------------------------------------------
# Main BCD algorithm
# ---------------------------------------------------------------------------

def run_block_coordinate_descent(
    surrogate: BVSurrogateModel,
    target_cd: np.ndarray,
    target_pc: np.ndarray,
    initial_k0: Sequence[float],
    initial_alpha: Sequence[float],
    bounds_k0_1: Tuple[float, float],
    bounds_k0_2: Tuple[float, float],
    bounds_alpha: Tuple[float, float],
    config: Optional[BCDConfig] = None,
) -> BCDResult:
    """Run Block Coordinate Descent for multi-reaction BV inference.

    Alternates between:
      - Block 1: optimize [log10(k0_1), alpha_1] with reaction 2 fixed
      - Block 2: optimize [log10(k0_2), alpha_2] with reaction 1 fixed

    Parameters
    ----------
    surrogate : BVSurrogateModel
        Fitted surrogate model.
    target_cd, target_pc : np.ndarray
        Target I-V curves.
    initial_k0 : sequence of 2 floats
        Initial [k0_1, k0_2] guesses.
    initial_alpha : sequence of 2 floats
        Initial [alpha_1, alpha_2] guesses.
    bounds_k0_1, bounds_k0_2 : (lo, hi)
        Bounds for k0_1 and k0_2 in *physical* space (will be log10'd).
    bounds_alpha : (lo, hi)
        Bounds for both alpha values.
    config : BCDConfig or None
        Algorithm configuration (defaults applied if None).

    Returns
    -------
    BCDResult
        Final result with iteration history.
    """
    if config is None:
        config = BCDConfig()

    t_start = time.time()

    # Current estimates
    k0_1 = float(initial_k0[0])
    k0_2 = float(initial_k0[1])
    alpha_1 = float(initial_alpha[0])
    alpha_2 = float(initial_alpha[1])

    # Bounds in log10 space for k0
    log_bounds_k0_1 = (np.log10(max(bounds_k0_1[0], 1e-30)),
                       np.log10(bounds_k0_1[1]))
    log_bounds_k0_2 = (np.log10(max(bounds_k0_2[0], 1e-30)),
                       np.log10(bounds_k0_2[1]))

    iteration_history: list[BCDIterationResult] = []
    total_evals = 0
    converged = False
    convergence_reason = "max_outer_iters reached"

    for outer in range(config.max_outer_iters):
        t_iter = time.time()

        prev_params = (np.log10(max(k0_1, 1e-30)),
                       np.log10(max(k0_2, 1e-30)),
                       alpha_1, alpha_2)

        # --- Block 1: optimize reaction 1 (k0_1, alpha_1), fix reaction 2 ---
        x0_b1 = np.array([np.log10(max(k0_1, 1e-30)), alpha_1], dtype=float)
        bounds_b1 = [log_bounds_k0_1, bounds_alpha]

        x_b1, loss_b1, iters_b1, evals_b1 = _optimize_block(
            surrogate=surrogate,
            target_cd=target_cd,
            target_pc=target_pc,
            reaction_index=0,
            fixed_k0_other=k0_2,
            fixed_alpha_other=alpha_2,
            x0=x0_b1,
            bounds=bounds_b1,
            secondary_weight=config.block_1_weight,
            inner_maxiter=config.inner_maxiter,
            fd_step=config.fd_step,
        )
        total_evals += evals_b1

        # Update reaction 1 parameters
        k0_1 = 10.0 ** x_b1[0]
        alpha_1 = float(x_b1[1])

        # --- Block 2: optimize reaction 2 (k0_2, alpha_2), fix reaction 1 ---
        x0_b2 = np.array([np.log10(max(k0_2, 1e-30)), alpha_2], dtype=float)
        bounds_b2 = [log_bounds_k0_2, bounds_alpha]

        x_b2, loss_b2, iters_b2, evals_b2 = _optimize_block(
            surrogate=surrogate,
            target_cd=target_cd,
            target_pc=target_pc,
            reaction_index=1,
            fixed_k0_other=k0_1,
            fixed_alpha_other=alpha_1,
            x0=x0_b2,
            bounds=bounds_b2,
            secondary_weight=config.block_2_weight,
            inner_maxiter=config.inner_maxiter,
            fd_step=config.fd_step,
        )
        total_evals += evals_b2

        # Update reaction 2 parameters
        k0_2 = 10.0 ** x_b2[0]
        alpha_2 = float(x_b2[1])

        iter_elapsed = time.time() - t_iter

        iteration_history.append(BCDIterationResult(
            iteration=outer,
            k0_1=k0_1,
            k0_2=k0_2,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            loss_after_block_1=loss_b1,
            loss_after_block_2=loss_b2,
            block_1_inner_iters=iters_b1,
            block_2_inner_iters=iters_b2,
            elapsed_s=iter_elapsed,
        ))

        if config.verbose:
            print(
                f"  [BCD iter {outer:>2d}] "
                f"k0=[{k0_1:.4e},{k0_2:.4e}] "
                f"alpha=[{alpha_1:.4f},{alpha_2:.4f}] "
                f"loss_b1={loss_b1:.4e} loss_b2={loss_b2:.4e} "
                f"({iter_elapsed:.2f}s)"
            )

        # --- Convergence check ---
        curr_params = (np.log10(max(k0_1, 1e-30)),
                       np.log10(max(k0_2, 1e-30)),
                       alpha_1, alpha_2)

        converged, convergence_reason = _check_convergence(
            prev_params, curr_params, config,
        )
        if converged:
            if config.verbose:
                print(f"  BCD converged at iter {outer}: {convergence_reason}")
            break

    # Compute final loss with weight=1.0 for fair comparison
    final_obj = SurrogateObjective(
        surrogate=surrogate,
        target_cd=target_cd,
        target_pc=target_pc,
        secondary_weight=1.0,
    )
    x_final = np.array([
        np.log10(max(k0_1, 1e-30)),
        np.log10(max(k0_2, 1e-30)),
        alpha_1,
        alpha_2,
    ])
    final_loss = final_obj.objective(x_final)
    total_evals += 1

    elapsed = time.time() - t_start

    return BCDResult(
        k0_1=k0_1,
        k0_2=k0_2,
        alpha_1=alpha_1,
        alpha_2=alpha_2,
        final_loss=final_loss,
        n_outer_iters=len(iteration_history),
        total_surrogate_evals=total_evals,
        converged=converged,
        convergence_reason=convergence_reason,
        iteration_history=tuple(iteration_history),
        elapsed_s=elapsed,
    )
