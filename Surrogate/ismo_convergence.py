"""ISMO convergence monitoring: criteria, checker, and diagnostic record.

Provides the ``ISMOConvergenceChecker`` class that tracks per-iteration
diagnostics, evaluates multi-criteria convergence (budget, stagnation,
agreement, stability), and produces JSON-serialisable summaries.

The ``ISMODiagnosticRecord`` dataclass extends 4-01's canonical
``ISMOIteration`` via composition -- it holds a reference to the core
record plus diagnostic-only fields (timing breakdowns, test-set metrics,
acquisition metadata).

Public API
----------
ISMOConvergenceCriteria
    Frozen configuration for convergence thresholds.
ISMODiagnosticRecord
    Per-iteration diagnostic wrapper around ``ISMOIteration``.
ISMOConvergenceChecker
    Stateful convergence evaluator with history tracking.
compute_surrogate_pde_agreement
    Point-wise surrogate-vs-PDE NRMSE at a single parameter point.
compute_parameter_stability
    Normalised log-space L2 distance between successive best params.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Guard import of 4-01 canonical dataclasses (may not exist yet)
# ---------------------------------------------------------------------------
try:
    from Surrogate.ismo import ISMOIteration, ISMOResult
except ImportError:
    ISMOIteration = None  # will be available when 4-01 is merged
    ISMOResult = None


# ---------------------------------------------------------------------------
# Convergence criteria
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ISMOConvergenceCriteria:
    """Frozen configuration controlling when the ISMO loop terminates.

    Attributes
    ----------
    agreement_tol : float
        Primary criterion -- surrogate-PDE NRMSE at the optimizer
        solution must fall below this threshold.
    stability_tol : float
        Secondary criterion -- normalised log-space L2 distance
        between successive best-parameter estimates.
    test_error_degradation_tol : float
        Maximum allowed relative increase in held-out test NRMSE
        before flagging degradation.
    max_pde_evals : int
        Hard budget cap on PDE evaluations across all iterations.
    max_iterations : int
        Maximum number of ISMO iterations.
    min_iterations : int
        Minimum iterations before convergence can be declared.
    stagnation_window : int
        Number of consecutive iterations without meaningful
        improvement before declaring stagnation.  Must be >= 2.
    min_useful_batch_size : int
        If remaining budget is less than this value after the
        agreement evaluation, the loop stops before acquisition.
    """

    agreement_tol: float = 0.01
    stability_tol: float = 0.01
    test_error_degradation_tol: float = 0.05
    max_pde_evals: int = 200
    max_iterations: int = 10
    min_iterations: int = 2
    stagnation_window: int = 3
    min_useful_batch_size: int = 5


# ---------------------------------------------------------------------------
# Diagnostic record (composition with 4-01's ISMOIteration)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ISMODiagnosticRecord:
    """Extends ISMOIteration with diagnostics-layer metadata.

    References the canonical ``ISMOIteration`` from 4-01 via composition.
    When 4-01 is not yet available, ``core`` may be set to ``None``.

    Attributes
    ----------
    core : ISMOIteration or None
        The canonical iteration record from 4-01.
    surrogate_pde_agreement : float
        NRMSE of surrogate vs PDE at the optimizer solution.
    parameter_stability : float
        L2 distance in normalised log-space from previous iteration.
    surrogate_test_nrmse_cd : float
        NRMSE on the held-out test set (current density).
    surrogate_test_nrmse_pc : float
        NRMSE on the held-out test set (peroxide current).
    objective_improvement : float
        Ratio ``prev_pde_obj / current_pde_obj`` (> 1 means improved).
        ``float('nan')`` for iteration 0.
    pde_solve_time_s : float
        Wall-clock time for PDE evaluations this iteration (seconds).
    retrain_time_s : float
        Wall-clock time for surrogate retraining (seconds).
    inference_time_s : float
        Wall-clock time for surrogate-based inference (seconds).
    acquisition_strategy : str
        Name of the acquisition strategy used (e.g. ``"trust_region"``).
    acquisition_details : tuple
        Tuple of ``(key, value)`` pairs for immutability in a frozen
        dataclass.
    n_pde_evals_this_iter : int
        Number of PDE evaluations consumed in this iteration
        (agreement eval + batch).
    iteration : int
        Zero-based iteration index.
    best_params : tuple
        Best parameter values as ``(k0_1, k0_2, alpha_1, alpha_2)``.
    n_total_training : int
        Total training-set size after augmentation this iteration.
    """

    core: Any  # ISMOIteration or None
    surrogate_pde_agreement: float
    parameter_stability: float
    surrogate_test_nrmse_cd: float
    surrogate_test_nrmse_pc: float
    objective_improvement: float
    pde_solve_time_s: float
    retrain_time_s: float
    inference_time_s: float
    acquisition_strategy: str
    acquisition_details: tuple
    n_pde_evals_this_iter: int
    iteration: int
    best_params: tuple
    n_total_training: int


# ---------------------------------------------------------------------------
# Convergence checker
# ---------------------------------------------------------------------------

class ISMOConvergenceChecker:
    """Stateful multi-criteria convergence evaluator for the ISMO loop.

    Parameters
    ----------
    criteria : ISMOConvergenceCriteria or None
        Convergence thresholds.  Defaults are applied when ``None``.

    Raises
    ------
    ValueError
        If ``criteria.stagnation_window < 2``.
    """

    def __init__(self, criteria: ISMOConvergenceCriteria | None = None) -> None:
        self.criteria = criteria or ISMOConvergenceCriteria()
        if self.criteria.stagnation_window < 2:
            raise ValueError(
                f"stagnation_window must be >= 2, got "
                f"{self.criteria.stagnation_window}"
            )
        self.history: List[ISMODiagnosticRecord] = []
        self._total_pde_evals: int = 0

    # ----- recording -----

    def record_iteration(self, record: ISMODiagnosticRecord) -> None:
        """Append a new iteration record and update cumulative PDE count."""
        self.history.append(record)
        self._total_pde_evals += record.n_pde_evals_this_iter

    # ----- budget helpers -----

    @property
    def total_pde_evals(self) -> int:
        """Total PDE evaluations consumed so far."""
        return self._total_pde_evals

    def is_budget_exhausted(self) -> bool:
        """True if the PDE evaluation budget is fully consumed."""
        return self._total_pde_evals >= self.criteria.max_pde_evals

    def remaining_budget(self) -> int:
        """Number of PDE evaluations remaining (clamped to >= 0)."""
        return max(0, self.criteria.max_pde_evals - self._total_pde_evals)

    # ----- convergence evaluation -----

    def check_convergence(self) -> Tuple[bool, str]:
        """Evaluate all stopping criteria against the recorded history.

        Returns
        -------
        (converged, reason) : (bool, str)
            ``converged`` is ``True`` when any stopping criterion fires.
            ``reason`` is a short human-readable label:

            * ``"budget_exhausted"``
            * ``"budget_too_low"``
            * ``"max_iterations"``
            * ``"converged"``
            * ``"stagnation"``
            * ``""`` -- none of the criteria are met.
        """
        c = self.criteria
        n = len(self.history)

        if n == 0:
            return False, ""

        latest = self.history[-1]

        # 1. Budget exceeded
        if self._total_pde_evals >= c.max_pde_evals:
            return True, "budget_exhausted"

        # 2. Budget too low for useful work
        remaining = self.remaining_budget()
        if remaining < c.min_useful_batch_size:
            return True, "budget_too_low"

        # 3. Max iterations reached
        if n >= c.max_iterations:
            return True, "max_iterations"

        # 4. Primary convergence: agreement AND stability both met
        if n >= c.min_iterations:
            if (
                latest.surrogate_pde_agreement < c.agreement_tol
                and latest.parameter_stability < c.stability_tol
            ):
                return True, "converged"

        # 5. Stagnation: last `stagnation_window` iterations show no
        #    meaningful improvement in agreement (< 5% relative decrease).
        w = c.stagnation_window
        if n >= max(w, c.min_iterations):
            window = self.history[-w:]
            agreements = [r.surrogate_pde_agreement for r in window]
            # Check if the best in the window is within 5% of the worst
            best_in_window = min(agreements)
            worst_in_window = max(agreements)
            if worst_in_window > 0:
                relative_improvement = (
                    (worst_in_window - best_in_window) / worst_in_window
                )
            else:
                relative_improvement = 0.0
            if relative_improvement < 0.05:
                return True, "stagnation"

        return False, ""

    # ----- summary -----

    def get_convergence_summary(self) -> Dict[str, Any]:
        """Return a JSON-serialisable summary of the convergence state."""
        n = len(self.history)
        converged, reason = self.check_convergence() if n > 0 else (False, "")

        summary: Dict[str, Any] = {
            "converged": converged,
            "termination_reason": reason,
            "n_iterations": n,
            "total_pde_evals": self._total_pde_evals,
            "remaining_budget": self.remaining_budget(),
            "criteria": {
                "agreement_tol": self.criteria.agreement_tol,
                "stability_tol": self.criteria.stability_tol,
                "test_error_degradation_tol": self.criteria.test_error_degradation_tol,
                "max_pde_evals": self.criteria.max_pde_evals,
                "max_iterations": self.criteria.max_iterations,
                "min_iterations": self.criteria.min_iterations,
                "stagnation_window": self.criteria.stagnation_window,
                "min_useful_batch_size": self.criteria.min_useful_batch_size,
            },
        }

        if n > 0:
            latest = self.history[-1]
            summary["latest"] = {
                "iteration": latest.iteration,
                "surrogate_pde_agreement": latest.surrogate_pde_agreement,
                "parameter_stability": latest.parameter_stability,
                "surrogate_test_nrmse_cd": latest.surrogate_test_nrmse_cd,
                "surrogate_test_nrmse_pc": latest.surrogate_test_nrmse_pc,
                "objective_improvement": (
                    None if math.isnan(latest.objective_improvement)
                    else latest.objective_improvement
                ),
                "best_params": list(latest.best_params),
                "n_total_training": latest.n_total_training,
            }

            summary["history"] = []
            for r in self.history:
                summary["history"].append({
                    "iteration": r.iteration,
                    "surrogate_pde_agreement": r.surrogate_pde_agreement,
                    "parameter_stability": r.parameter_stability,
                    "n_pde_evals": r.n_pde_evals_this_iter,
                    "best_params": list(r.best_params),
                })

        return summary


# ---------------------------------------------------------------------------
# Surrogate-PDE agreement metric
# ---------------------------------------------------------------------------

def compute_surrogate_pde_agreement(
    surrogate: Any,
    params_physical: np.ndarray,
    pde_solver_fn: Any,
    phi_applied: np.ndarray,
) -> Dict[str, Any]:
    """Evaluate surrogate and PDE at the same point, return NRMSE metrics.

    Parameters
    ----------
    surrogate : BVSurrogateModel
        Fitted surrogate model with ``predict()`` or ``predict_batch()``.
    params_physical : np.ndarray of shape (4,)
        ``[k0_1, k0_2, alpha_1, alpha_2]`` in physical space.
    pde_solver_fn : callable
        ``params -> {"current_density": ..., "peroxide_current": ...}``
        Costs exactly 1 PDE evaluation.
    phi_applied : np.ndarray
        Applied-voltage grid (used for NRMSE denominator).

    Returns
    -------
    dict
        ``'cd_nrmse'``, ``'pc_nrmse'``, ``'combined_nrmse'`` (max of
        cd and pc), plus the raw curves ``'cd_surrogate'``,
        ``'cd_pde'``, ``'pc_surrogate'``, ``'pc_pde'``.
    """
    k0_1, k0_2, alpha_1, alpha_2 = params_physical

    # Surrogate prediction
    pred = surrogate.predict(float(k0_1), float(k0_2), float(alpha_1), float(alpha_2))
    cd_surr = np.asarray(pred["current_density"], dtype=float)
    pc_surr = np.asarray(pred["peroxide_current"], dtype=float)

    # PDE truth
    pde_out = pde_solver_fn(params_physical)
    cd_pde = np.asarray(pde_out["current_density"], dtype=float)
    pc_pde = np.asarray(pde_out["peroxide_current"], dtype=float)

    # NRMSE: RMSE / range(pde_curve), with floor to avoid division by zero
    def _nrmse(predicted: np.ndarray, truth: np.ndarray) -> float:
        rmse = float(np.sqrt(np.mean((predicted - truth) ** 2)))
        denom = float(np.ptp(truth))
        if denom < 1e-12:
            denom = 1e-12
        return rmse / denom

    cd_nrmse = _nrmse(cd_surr, cd_pde)
    pc_nrmse = _nrmse(pc_surr, pc_pde)

    return {
        "cd_nrmse": cd_nrmse,
        "pc_nrmse": pc_nrmse,
        "combined_nrmse": max(cd_nrmse, pc_nrmse),
        "cd_surrogate": cd_surr,
        "cd_pde": cd_pde,
        "pc_surrogate": pc_surr,
        "pc_pde": pc_pde,
    }


# ---------------------------------------------------------------------------
# Parameter stability
# ---------------------------------------------------------------------------

def compute_parameter_stability(
    history: List[ISMODiagnosticRecord],
    current_params: np.ndarray,
    bounds: Any,
) -> float:
    """L2 distance in normalised log-space between current and previous best.

    Normalisation maps each dimension to [0, 1]:

    * k0 dimensions:  ``(log10(val) - log10(lo)) / (log10(hi) - log10(lo))``
    * alpha dimensions:  ``(val - lo) / (hi - lo)``

    Parameters
    ----------
    history : list of ISMODiagnosticRecord
        Previously recorded iterations (may be empty).
    current_params : np.ndarray of shape (4,)
        ``[k0_1, k0_2, alpha_1, alpha_2]`` in physical space.
    bounds : ParameterBounds or dict-like
        Must expose ``k0_1_range``, ``k0_2_range``, ``alpha_1_range``,
        ``alpha_2_range`` as ``(lo, hi)`` tuples, **or** be a dict with
        those keys.

    Returns
    -------
    float
        L2 distance in normalised space.  Returns 0.0 if there is no
        previous iteration.
    """
    if len(history) == 0:
        return 0.0

    prev_params = np.asarray(history[-1].best_params, dtype=float)

    # Extract bounds -- support both object attributes and dicts
    if isinstance(bounds, dict):
        k0_1_lo, k0_1_hi = bounds["k0_1_range"]
        k0_2_lo, k0_2_hi = bounds["k0_2_range"]
        a1_lo, a1_hi = bounds["alpha_1_range"]
        a2_lo, a2_hi = bounds["alpha_2_range"]
    else:
        k0_1_lo, k0_1_hi = bounds.k0_1_range
        k0_2_lo, k0_2_hi = bounds.k0_2_range
        a1_lo, a1_hi = bounds.alpha_1_range
        a2_lo, a2_hi = bounds.alpha_2_range

    def _norm_log(val: float, lo: float, hi: float) -> float:
        log_val = math.log10(max(val, 1e-30))
        log_lo = math.log10(max(lo, 1e-30))
        log_hi = math.log10(max(hi, 1e-30))
        denom = log_hi - log_lo
        if abs(denom) < 1e-30:
            return 0.0
        return (log_val - log_lo) / denom

    def _norm_lin(val: float, lo: float, hi: float) -> float:
        denom = hi - lo
        if abs(denom) < 1e-30:
            return 0.0
        return (val - lo) / denom

    cur_norm = np.array([
        _norm_log(current_params[0], k0_1_lo, k0_1_hi),
        _norm_log(current_params[1], k0_2_lo, k0_2_hi),
        _norm_lin(current_params[2], a1_lo, a1_hi),
        _norm_lin(current_params[3], a2_lo, a2_hi),
    ])

    prev_norm = np.array([
        _norm_log(prev_params[0], k0_1_lo, k0_1_hi),
        _norm_log(prev_params[1], k0_2_lo, k0_2_hi),
        _norm_lin(prev_params[2], a1_lo, a1_hi),
        _norm_lin(prev_params[3], a2_lo, a2_hi),
    ])

    return float(np.linalg.norm(cur_norm - prev_norm))


def compute_objective_improvement(
    history: List[ISMODiagnosticRecord],
    current_pde_loss: float,
) -> float:
    """Ratio of previous PDE loss to current (>1 means improvement).

    Returns ``float('nan')`` when there is no previous iteration.
    """
    if len(history) == 0:
        return float("nan")
    # Get previous PDE loss from the core record if available,
    # otherwise approximate from the agreement metric.
    prev = history[-1]
    if prev.core is not None and hasattr(prev.core, "pde_loss_at_best"):
        prev_loss = prev.core.pde_loss_at_best
    else:
        # Fallback: not directly comparable, return nan
        return float("nan")
    if current_pde_loss == 0.0:
        return float("inf")
    return prev_loss / current_pde_loss
