"""Parallel execution helpers for BV point solves."""

from __future__ import annotations

import time as _time
from typing import Any, Dict, List, Optional

import numpy as np

from .cache import _all_points_cache
from FluxCurve.results import PointAdjointResult


def _solve_cached_fast_path_parallel(
    *,
    n_points: int,
    n_controls: int,
    phi_applied_values: np.ndarray,
    target_flux: np.ndarray,
    k0_list: list,
    alpha_list: Optional[list],
    a_list: Optional[list],
    control_mode: str,
    fail_penalty: float,
    observable_mode: str,
    observable_reaction_index: Optional[int],
    observable_scale: float,
    parallel_pool: Any,
) -> Optional[List[PointAdjointResult]]:
    """Dispatch all cached-IC point solves to a parallel pool.

    Builds task tuples from ``_all_points_cache`` entries and submits them
    to the ``BVPointSolvePool``.  Returns a list of ``PointAdjointResult``
    in original point order, or ``None`` if the parallel path fails (caller
    should fall back to sequential).
    """
    t_par_start = _time.perf_counter()

    # Build task list
    tasks = []
    nan_results: Dict[int, PointAdjointResult] = {}

    for orig_idx in range(n_points):
        phi_applied_i = float(phi_applied_values[orig_idx])
        target_i = float(target_flux[orig_idx])

        if np.isnan(target_i):
            nan_results[orig_idx] = PointAdjointResult(
                phi_applied=phi_applied_i,
                target_flux=target_i,
                simulated_flux=float("nan"),
                objective=0.0,
                gradient=np.zeros(n_controls, dtype=float),
                converged=False,
                steps_taken=0,
                reason="target NaN (skipped)",
                final_relative_change=None,
                final_absolute_change=None,
                diagnostics_valid=False,
            )
            continue

        cached_U_data = _all_points_cache.get(orig_idx)
        if cached_U_data is None:
            return None  # missing cache entry, cannot parallelize

        # Convert cached_U_data (tuple of numpy arrays) to list for pickling
        cached_U_arrays = [arr.copy() for arr in cached_U_data]

        task = (
            orig_idx,
            phi_applied_i,
            target_i,
            k0_list,
            alpha_list,
            a_list,
            cached_U_arrays,
        )
        tasks.append(task)

    if not tasks:
        # All points were NaN -- return NaN results
        return [nan_results.get(i) for i in range(n_points)]

    # Submit to pool
    raw_results = parallel_pool.solve_points(tasks)
    if raw_results is None:
        return None  # pool failed

    # Convert raw results to PointAdjointResult and update cache
    results: List[Optional[PointAdjointResult]] = [None] * n_points

    # Fill NaN results first
    for idx, nr in nan_results.items():
        results[idx] = nr

    any_failed = False
    for raw in raw_results:
        idx = int(raw["point_index"])
        if not raw["success"]:
            any_failed = True
            break

        gradient_arr = np.asarray(raw["gradient"], dtype=float)
        results[idx] = PointAdjointResult(
            phi_applied=float(raw["phi_applied"]),
            target_flux=float(raw["target_flux"]),
            simulated_flux=float(raw["simulated_flux"]),
            objective=float(raw["objective"]),
            gradient=gradient_arr,
            converged=True,
            steps_taken=int(raw["steps_taken"]),
            reason="",
            final_relative_change=(
                float(raw["rel_metric"]) if raw["rel_metric"] is not None else None
            ),
            final_absolute_change=(
                float(raw["abs_metric"]) if raw["abs_metric"] is not None else None
            ),
            diagnostics_valid=True,
        )

        # Update the main-process cache with new converged states
        converged_arrays = raw.get("converged_U_arrays")
        if converged_arrays is not None:
            _all_points_cache[idx] = tuple(
                np.asarray(a, dtype=float) for a in converged_arrays
            )

    if any_failed:
        return None  # fall back to sequential

    t_par_elapsed = _time.perf_counter() - t_par_start

    # Print per-point timing from worker results
    for raw in raw_results:
        idx = int(raw["point_index"])
        phi_i = float(raw["phi_applied"])
        elapsed = raw.get("elapsed", 0.0)
        steps = int(raw["steps_taken"])
        print(
            f"  [timing] phi={phi_i:+8.4f} "
            f"total={elapsed:.2f}s "
            f"steps={steps} "
            f"(parallel-cached-IC)"
        )
    print(
        f"  [bv-parallel] {len(tasks)} points solved in {t_par_elapsed:.2f}s "
        f"wall-clock ({len(tasks)} tasks, "
        f"{parallel_pool.n_workers} workers)"
    )

    return [r for r in results]  # type: ignore[misc]


def _solve_cached_fast_path_parallel_multi_obs(
    *,
    n_points: int,
    n_controls: int,
    phi_applied_values: np.ndarray,
    target_flux_primary: np.ndarray,
    target_flux_secondary: np.ndarray,
    k0_list: list,
    alpha_list: Optional[list],
    a_list: Optional[list],
    control_mode: str,
    fail_penalty: float,
    observable_mode: str,
    observable_reaction_index: Optional[int],
    observable_scale: float,
    secondary_observable_mode: str,
    secondary_observable_scale: float,
    secondary_weight: float,
    parallel_pool: Any,
) -> Optional[Dict[str, object]]:
    """Dispatch multi-observable point solves to a parallel pool.

    Each worker computes BOTH primary and secondary adjoint gradients after
    a single forward solve.  Returns a dict with "primary_results" and
    "secondary_results" (each a list of PointAdjointResult), or None if
    the parallel path fails.
    """
    t_par_start = _time.perf_counter()

    # Build task list with multi-obs flag
    tasks = []
    nan_results_primary: Dict[int, PointAdjointResult] = {}
    nan_results_secondary: Dict[int, PointAdjointResult] = {}

    for orig_idx in range(n_points):
        phi_applied_i = float(phi_applied_values[orig_idx])
        target_i_primary = float(target_flux_primary[orig_idx])
        target_i_secondary = float(target_flux_secondary[orig_idx])

        if np.isnan(target_i_primary) and np.isnan(target_i_secondary):
            _nan_result = PointAdjointResult(
                phi_applied=phi_applied_i,
                target_flux=target_i_primary,
                simulated_flux=float("nan"),
                objective=0.0,
                gradient=np.zeros(n_controls, dtype=float),
                converged=False,
                steps_taken=0,
                reason="target NaN (skipped)",
                final_relative_change=None,
                final_absolute_change=None,
                diagnostics_valid=False,
            )
            nan_results_primary[orig_idx] = _nan_result
            nan_results_secondary[orig_idx] = PointAdjointResult(
                phi_applied=phi_applied_i,
                target_flux=target_i_secondary,
                simulated_flux=float("nan"),
                objective=0.0,
                gradient=np.zeros(n_controls, dtype=float),
                converged=False,
                steps_taken=0,
                reason="target NaN (skipped)",
                final_relative_change=None,
                final_absolute_change=None,
                diagnostics_valid=False,
            )
            continue

        cached_U_data = _all_points_cache.get(orig_idx)
        if cached_U_data is None:
            return None  # missing cache entry, cannot parallelize

        cached_U_arrays = [arr.copy() for arr in cached_U_data]

        # 9-element task tuple for multi-obs mode
        task = (
            orig_idx,
            phi_applied_i,
            target_i_primary,
            k0_list,
            alpha_list,
            a_list,
            cached_U_arrays,
            target_i_secondary,
            True,  # multi_obs_flag
        )
        tasks.append(task)

    if not tasks:
        return {
            "primary_results": [nan_results_primary.get(i) for i in range(n_points)],
            "secondary_results": [nan_results_secondary.get(i) for i in range(n_points)],
        }

    # Submit to pool
    raw_results = parallel_pool.solve_points(tasks)
    if raw_results is None:
        return None  # pool failed

    # Convert raw results to PointAdjointResult
    primary_results: List[Optional[PointAdjointResult]] = [None] * n_points
    secondary_results: List[Optional[PointAdjointResult]] = [None] * n_points

    # Fill NaN results
    for idx, nr in nan_results_primary.items():
        primary_results[idx] = nr
    for idx, nr in nan_results_secondary.items():
        secondary_results[idx] = nr

    any_failed = False
    for raw in raw_results:
        idx = int(raw["point_index"])
        if not raw["success"]:
            any_failed = True
            break

        # Primary result
        gradient_arr = np.asarray(raw["gradient"], dtype=float)
        primary_results[idx] = PointAdjointResult(
            phi_applied=float(raw["phi_applied"]),
            target_flux=float(raw["target_flux"]),
            simulated_flux=float(raw["simulated_flux"]),
            objective=float(raw["objective"]),
            gradient=gradient_arr,
            converged=True,
            steps_taken=int(raw["steps_taken"]),
            reason="",
            final_relative_change=(
                float(raw["rel_metric"]) if raw["rel_metric"] is not None else None
            ),
            final_absolute_change=(
                float(raw["abs_metric"]) if raw["abs_metric"] is not None else None
            ),
            diagnostics_valid=True,
        )

        # Secondary result
        sec_gradient_arr = np.asarray(raw["secondary_gradient"], dtype=float)
        sec_target = float(target_flux_secondary[idx])
        secondary_results[idx] = PointAdjointResult(
            phi_applied=float(raw["phi_applied"]),
            target_flux=sec_target,
            simulated_flux=float(raw["secondary_simulated_flux"]),
            objective=float(raw["secondary_objective"]),
            gradient=sec_gradient_arr,
            converged=True,
            steps_taken=int(raw["steps_taken"]),
            reason="",
            final_relative_change=(
                float(raw["rel_metric"]) if raw["rel_metric"] is not None else None
            ),
            final_absolute_change=(
                float(raw["abs_metric"]) if raw["abs_metric"] is not None else None
            ),
            diagnostics_valid=True,
        )

        # Update the main-process cache with new converged states
        converged_arrays = raw.get("converged_U_arrays")
        if converged_arrays is not None:
            _all_points_cache[idx] = tuple(
                np.asarray(a, dtype=float) for a in converged_arrays
            )

    if any_failed:
        return None  # fall back to sequential

    t_par_elapsed = _time.perf_counter() - t_par_start

    # Print per-point timing
    for raw in raw_results:
        idx = int(raw["point_index"])
        phi_i = float(raw["phi_applied"])
        elapsed = raw.get("elapsed", 0.0)
        steps = int(raw["steps_taken"])
        print(
            f"  [timing] phi={phi_i:+8.4f} "
            f"total={elapsed:.2f}s "
            f"steps={steps} "
            f"(parallel-multi-obs)"
        )
    print(
        f"  [bv-parallel-multi-obs] {len(tasks)} points solved in {t_par_elapsed:.2f}s "
        f"wall-clock ({len(tasks)} tasks, "
        f"{parallel_pool.n_workers} workers, 2 observables each)"
    )

    return {
        "primary_results": [r for r in primary_results],
        "secondary_results": [r for r in secondary_results],
    }
