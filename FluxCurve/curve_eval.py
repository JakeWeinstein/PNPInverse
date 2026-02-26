"""Curve-level objective and gradient evaluation across the phi_applied sweep."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from FluxCurve.config import RobinFluxCurveInferenceRequest
from FluxCurve.results import CurveAdjointResult, PointAdjointResult
from FluxCurve.recovery import _reduce_kappa_anisotropy
from FluxCurve.point_solve import _PointSolveExecutor, solve_point_objective_and_gradient
from Forward.steady_state import SteadyStateConfig, sweep_phi_applied_steady_flux, results_to_flux_array


def evaluate_curve_objective_and_gradient(
    *,
    request: RobinFluxCurveInferenceRequest,
    phi_applied_values: np.ndarray,
    target_flux: np.ndarray,
    kappa_values: np.ndarray,
    point_executor: Optional[_PointSolveExecutor] = None,
) -> CurveAdjointResult:
    """Evaluate curve objective + gradient, with anisotropy recovery fallback."""
    n_species = int(request.base_solver_params[0])

    def _evaluate_once(kappa_eval: np.ndarray) -> CurveAdjointResult:
        points: List[PointAdjointResult] = []
        simulated_flux = np.full(phi_applied_values.shape, np.nan, dtype=float)
        total_objective = 0.0
        total_gradient = np.zeros(n_species, dtype=float)
        n_failed = 0

        parallel_points: Optional[List[PointAdjointResult]] = None
        if point_executor is not None and bool(point_executor.enabled):
            parallel_points = point_executor.map_points(
                phi_applied_values=phi_applied_values,
                target_flux=target_flux,
                kappa_values=np.asarray(kappa_eval, dtype=float),
            )

        if parallel_points is not None:
            points = list(parallel_points)
        else:
            for phi_i, target_i in zip(phi_applied_values.tolist(), target_flux.tolist()):
                point = solve_point_objective_and_gradient(
                    base_solver_params=request.base_solver_params,
                    steady=request.steady,
                    phi_applied=float(phi_i),
                    target_flux=float(target_i),
                    kappa_values=kappa_eval.tolist(),
                    blob_initial_condition=bool(request.blob_initial_condition),
                    fail_penalty=float(request.fail_penalty),
                    forward_recovery=request.forward_recovery,
                    observable_mode=str(request.observable_mode),
                    observable_species_index=request.observable_species_index,
                    observable_scale=float(request.observable_scale),
                )
                points.append(point)

        for i, point in enumerate(points):
            simulated_flux[i] = point.simulated_flux
            total_objective += float(point.objective)
            if point.converged:
                total_gradient += point.gradient
            else:
                n_failed += 1

        return CurveAdjointResult(
            objective=float(total_objective),
            gradient=total_gradient,
            simulated_flux=simulated_flux,
            points=points,
            n_failed=n_failed,
            effective_kappa=np.asarray(kappa_eval, dtype=float).copy(),
            used_anisotropy_recovery=False,
        )

    primary = _evaluate_once(np.asarray(kappa_values, dtype=float))

    n_points = int(len(phi_applied_values))
    fail_threshold_by_points = max(1, int(request.anisotropy_trigger_failed_points))
    fail_threshold_by_fraction = int(
        np.ceil(
            max(0.0, float(request.anisotropy_trigger_failed_fraction)) * max(1, n_points)
        )
    )
    fail_trigger = max(fail_threshold_by_points, fail_threshold_by_fraction)

    if primary.n_failed < fail_trigger:
        return primary

    kappa_aniso = _reduce_kappa_anisotropy(
        np.asarray(kappa_values, dtype=float),
        target_ratio=float(request.forward_recovery.anisotropy_target_ratio),
        blend=float(request.forward_recovery.anisotropy_blend),
    )
    if np.allclose(kappa_aniso, np.asarray(kappa_values, dtype=float), rtol=1e-12, atol=1e-12):
        return primary

    print(
        "[recovery] many point failures detected "
        f"({primary.n_failed}/{n_points}); retrying with anisotropy-reduced kappa "
        f"[{kappa_aniso[0]:.6f}, {kappa_aniso[1]:.6f}]"
    )
    secondary = _evaluate_once(kappa_aniso)
    secondary.used_anisotropy_recovery = True

    # Prefer the evaluation with fewer failed points; tie-break on objective.
    if secondary.n_failed < primary.n_failed:
        return secondary
    if secondary.n_failed == primary.n_failed and secondary.objective < primary.objective:
        return secondary
    return primary


def evaluate_curve_loss_forward(
    *,
    base_solver_params: Sequence[object],
    steady: SteadyStateConfig,
    phi_applied_values: np.ndarray,
    target_flux: np.ndarray,
    kappa_values: np.ndarray,
    blob_initial_condition: bool,
    fail_penalty: float,
    observable_scale: float,
) -> Tuple[float, np.ndarray, int]:
    """Evaluate scalar curve loss (no derivatives), used by line search."""
    results = sweep_phi_applied_steady_flux(
        base_solver_params,
        phi_applied_values=phi_applied_values.tolist(),
        steady=steady,
        kappa_values=kappa_values.tolist(),
        blob_initial_condition=blob_initial_condition,
    )
    simulated_flux = float(observable_scale) * results_to_flux_array(results)
    n_failed = sum(0 if r.converged else 1 for r in results)

    if n_failed > 0 or np.any(~np.isfinite(simulated_flux)):
        return float(fail_penalty * max(1, n_failed)), simulated_flux, int(n_failed)

    residual = simulated_flux - target_flux
    loss = 0.5 * float(np.sum(residual * residual))
    return float(loss), simulated_flux, 0
