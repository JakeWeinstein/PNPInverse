"""Result dataclasses and serialization helpers for flux-curve inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import numpy as np


@dataclass
class RobinFluxCurveInferenceResult:
    """Inference outputs and generated artifact paths."""

    best_kappa: np.ndarray
    best_loss: float
    phi_applied_values: np.ndarray
    target_flux: np.ndarray
    best_simulated_flux: np.ndarray
    forward_failures_at_best: int
    fit_csv_path: str
    fit_plot_path: Optional[str]
    history_csv_path: str
    point_gradient_csv_path: str
    live_gif_path: Optional[str]
    optimization_success: bool
    optimization_message: str
    replay_rebuild_count: int
    replay_diag_rebuild_count: int
    replay_exception_rebuild_count: int


@dataclass
class PointAdjointResult:
    """Adjoint-evaluation result for one phi_applied point."""

    phi_applied: float
    target_flux: float
    simulated_flux: float
    objective: float
    gradient: np.ndarray
    converged: bool
    steps_taken: int
    reason: str = ""
    final_relative_change: Optional[float] = None
    final_absolute_change: Optional[float] = None
    diagnostics_valid: bool = True


@dataclass
class CurveAdjointResult:
    """Aggregated objective + gradient across the full phi_applied curve."""

    objective: float
    gradient: np.ndarray
    simulated_flux: np.ndarray
    points: List[PointAdjointResult]
    n_failed: int
    effective_kappa: np.ndarray
    used_anisotropy_recovery: bool = False
    used_replay_mode: bool = False


@dataclass
class _ReplayPointFunctional:
    """Persistent reduced-functional object for one phi_applied sweep point."""

    phi_applied: float
    tape: Any
    control_state: List[object]
    reduced_flux: Any
    reduced_flux_prev: Any
    reduced_state_delta_l2: Any
    reduced_state_norm_l2: Any
    steady_rel_tol: float
    steady_abs_tol: float
    steps_taken: int


@dataclass
class _ReplayBundle:
    """Collection of replay-ready point models for the full phi_applied sweep."""

    points: List[_ReplayPointFunctional]
    anchor_kappa: np.ndarray


def _point_result_to_payload(point: PointAdjointResult) -> Dict[str, object]:
    """Convert point result to a plain payload for inter-process transport."""
    return {
        "phi_applied": float(point.phi_applied),
        "target_flux": float(point.target_flux),
        "simulated_flux": float(point.simulated_flux),
        "objective": float(point.objective),
        "gradient": np.asarray(point.gradient, dtype=float).tolist(),
        "converged": bool(point.converged),
        "steps_taken": int(point.steps_taken),
        "reason": str(point.reason),
        "final_relative_change": (
            None if point.final_relative_change is None else float(point.final_relative_change)
        ),
        "final_absolute_change": (
            None if point.final_absolute_change is None else float(point.final_absolute_change)
        ),
        "diagnostics_valid": bool(point.diagnostics_valid),
    }


def _point_result_from_payload(payload: Mapping[str, object]) -> PointAdjointResult:
    """Reconstruct PointAdjointResult from plain payload."""
    gradient_raw = payload.get("gradient", [0.0, 0.0])
    gradient_arr = np.asarray(list(gradient_raw), dtype=float)
    return PointAdjointResult(
        phi_applied=float(payload.get("phi_applied", float("nan"))),
        target_flux=float(payload.get("target_flux", float("nan"))),
        simulated_flux=float(payload.get("simulated_flux", float("nan"))),
        objective=float(payload.get("objective", float("inf"))),
        gradient=gradient_arr,
        converged=bool(payload.get("converged", False)),
        steps_taken=int(payload.get("steps_taken", 0)),
        reason=str(payload.get("reason", "")),
        final_relative_change=(
            None
            if payload.get("final_relative_change", None) is None
            else float(payload.get("final_relative_change"))
        ),
        final_absolute_change=(
            None
            if payload.get("final_absolute_change", None) is None
            else float(payload.get("final_absolute_change"))
        ),
        diagnostics_valid=bool(payload.get("diagnostics_valid", False)),
    )
