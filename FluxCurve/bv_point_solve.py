"""Per-point BV adjoint solve with warm-start sequential sweep.

Unlike the Robin pipeline (independent per-point solves), the BV point
solver uses voltage continuation: points are solved sequentially from
smallest to largest |eta|, with each point's IC being the previous
point's converged solution.  This is essential for BV convergence at
large overpotentials.

When the phi_applied array contains both positive and negative values
(symmetric voltage range), a two-branch sweep is used: the negative
branch is swept outward from equilibrium first, then the positive branch.
This avoids interleaving cathodic and anodic solutions, which causes
SNES divergence at moderate |eta| due to inverted concentration profiles.
The only sign-change transition occurs near eta = 0, where solutions
are similar.  Bridge points fill any gap between branches.

Performance optimizations (2026-02-28):
- Mesh and function-space creation hoisted outside the per-point loop.
- Warm-started points use a reduced max_steps cap (default 20).
- Per-point timing instrumentation for profiling (setup/forward/adjoint).
- P1: Skip rf(control_state) tape replay; call rf.derivative() directly.
- P2: Cross-evaluation warm-start for the first point.
- P4: Quadratic/linear hybrid predictor step between consecutive voltage points.
- P5: PETSc Jacobian lagging (snes_lag_jacobian).
- P6: Adaptive pseudo-timestep (SER) via mutable fd.Constant dt.
- Bridge points: auto-insert forward-only points to span large eta gaps.
"""

from __future__ import annotations

import copy
import time as _time
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from FluxCurve.bv_config import BVFluxCurveInferenceRequest
from FluxCurve.config import ForwardRecoveryConfig
from FluxCurve.results import PointAdjointResult
from FluxCurve.recovery import _attempt_phase_state, _relax_solver_options_for_attempt
from FluxCurve.bv_observables import (
    _build_bv_observable_form,
    _build_bv_scalar_target_in_control_space,
    _bv_gradient_controls_to_array,
)
from Forward.steady_state import SteadyStateConfig, configure_bv_solver_params

# Default max pseudo-time steps for warm-started points.  When the initial
# guess comes from the previous voltage point's converged solution, only a
# few steps are needed to re-converge.  The full max_steps is still used
# for the first point (no warm-start) and for recovery attempts.
_WARMSTART_MAX_STEPS = 20

# P2: Cross-evaluation warm-start cache.
# Persists across calls to solve_bv_curve_points_with_warmstart so that
# the first point of eval N+1 uses eval N's converged solution as IC.
# Keyed by sorted_indices[0] (original index of the smallest-|eta| point).
_cross_eval_cache: Dict[int, tuple] = {}

# P7: Checkpoint-restart warm-start cache.
# After the first full sequential sweep, cache the converged solution for
# every point.  On subsequent evaluations, each point can be solved
# independently using its cached IC (no sequential sweep needed).
_all_points_cache: Dict[int, tuple] = {}
_cache_populated: bool = False

# P8: Module-level parallel pool for fast-path point solves.
# Set by set_parallel_pool() / cleared by close_parallel_pool().
# When not None and enabled, _solve_cached_fast_path will dispatch
# independent point solves to worker processes instead of looping.
_parallel_pool: Any = None

# P6: SER adaptive pseudo-timestep parameters.
_SER_GROWTH_CAP = 4.0    # max dt multiplier per step
_SER_SHRINK = 0.5        # dt multiplier when residual grows
_SER_DT_MAX_RATIO = 20.0  # max dt / dt_initial ratio

# Bridge point parameters.
_BRIDGE_MAX_STEPS = 40       # generous step cap for bridge points
_BRIDGE_SER_DT_MAX_RATIO = 50.0  # allow larger dt growth for bridges


def _clear_caches() -> None:
    """Clear all module-level caches.

    Call between multi-fidelity phases (coarse -> fine) to prevent stale
    solutions from a different mesh being reused.
    """
    global _cache_populated
    _cross_eval_cache.clear()
    _all_points_cache.clear()
    _cache_populated = False


def set_parallel_pool(pool: Any) -> None:
    """Set the module-level parallel pool for fast-path point solves."""
    global _parallel_pool
    _parallel_pool = pool


def close_parallel_pool() -> None:
    """Shutdown and clear the module-level parallel pool."""
    global _parallel_pool
    if _parallel_pool is not None:
        try:
            _parallel_pool.close()
        except Exception:
            pass
        _parallel_pool = None


def _apply_predictor(
    phi_applied_i: float,
    ctx_U,
    carry_U_data: tuple,
    predictor_prev: Optional[tuple],
    predictor_curr: Optional[tuple],
    predictor_prev2: Optional[tuple],
):
    """Apply quadratic/linear hybrid predictor to initial guess.

    Uses 3-point quadratic Lagrange interpolation when available and safe,
    otherwise falls back to 2-point linear extrapolation, otherwise simple
    warm-start copy.  Clamps concentrations to >= 1e-10 after prediction.

    After applying a predictor (quadratic or linear), validates the result:
    if any DOF deviates by more than 10x from the carry state, falls back
    to simple warm-start.  This prevents SNES divergence from aggressive
    extrapolation (which accounts for ~35% of point-solve retries).
    """
    _MAX_PREDICTOR_RATIO = 10.0  # max allowed deviation from carry state

    def _validate_and_maybe_revert():
        """Check predicted state; revert to carry if too far from carry."""
        for src, dst in zip(carry_U_data, ctx_U.dat):
            pred = dst.data_ro
            ref = np.maximum(np.abs(src), 1e-10)
            max_ratio = np.max(np.abs(pred - src) / ref)
            if max_ratio > _MAX_PREDICTOR_RATIO:
                # Prediction is too aggressive; revert to simple warm-start
                for s, d in zip(carry_U_data, ctx_U.dat):
                    d.data[:] = s
                return

    if predictor_prev is not None and predictor_curr is not None:
        eta_curr, U_curr_data = predictor_curr
        eta_prev, U_prev_data = predictor_prev

        # Try quadratic predictor if 3 points available
        if predictor_prev2 is not None:
            eta_prev2, U_prev2_data = predictor_prev2
            # Safety check: only use quadratic if extrapolation distance
            # is <= 2x the max prior gap
            extrap_dist = abs(phi_applied_i - eta_curr)
            max_prior_gap = max(abs(eta_curr - eta_prev), abs(eta_prev - eta_prev2))
            if max_prior_gap > 1e-14 and extrap_dist <= 2.0 * max_prior_gap:
                # Quadratic Lagrange interpolation
                eta_a, eta_b, eta_c = eta_prev2, eta_prev, eta_curr
                denom_a = (eta_a - eta_b) * (eta_a - eta_c)
                denom_b = (eta_b - eta_a) * (eta_b - eta_c)
                denom_c = (eta_c - eta_a) * (eta_c - eta_b)
                if abs(denom_a) > 1e-28 and abs(denom_b) > 1e-28 and abs(denom_c) > 1e-28:
                    L_a = ((phi_applied_i - eta_b) * (phi_applied_i - eta_c)) / denom_a
                    L_b = ((phi_applied_i - eta_a) * (phi_applied_i - eta_c)) / denom_b
                    L_c = ((phi_applied_i - eta_a) * (phi_applied_i - eta_b)) / denom_c
                    for u_a, u_b, u_c, dst in zip(
                        U_prev2_data, U_prev_data, U_curr_data, ctx_U.dat
                    ):
                        dst.data[:] = L_a * u_a + L_b * u_b + L_c * u_c
                    # Clamp concentrations to prevent negative values
                    for d in ctx_U.dat:
                        d.data[:] = np.maximum(d.data, 1e-10)
                    _validate_and_maybe_revert()
                    return
                # Fall through to linear if denominators are degenerate

        # Linear predictor from two prior solutions
        d_eta = eta_curr - eta_prev
        if abs(d_eta) > 1e-14:
            slope = (phi_applied_i - eta_curr) / d_eta
            for u_prev_arr, u_curr_arr, dst in zip(
                U_prev_data, U_curr_data, ctx_U.dat
            ):
                dst.data[:] = u_curr_arr + slope * (u_curr_arr - u_prev_arr)
            # Clamp concentrations to prevent negative values
            for d in ctx_U.dat:
                d.data[:] = np.maximum(d.data, 1e-10)
            _validate_and_maybe_revert()
            return

    # Simple warm-start (only one prior solution or degenerate)
    for src, dst in zip(carry_U_data, ctx_U.dat):
        dst.data[:] = src


def _solve_bridge_points(
    *,
    prev_solved_eta: float,
    next_eta: float,
    max_eta_gap: float,
    carry_U_data: tuple,
    base_solver_params: Sequence[object],
    k0_list: list,
    alpha_list: Optional[list],
    a_list: Optional[list],
    shared_mesh: Any,
    observable_mode: str,
    observable_reaction_index: Optional[int],
    observable_scale: float,
    rel_tol: float,
    abs_tol: float,
    predictor_prev: Optional[tuple],
    predictor_curr: Optional[tuple],
    predictor_prev2: Optional[tuple],
    blob_initial_condition: bool,
):
    """Solve bridge points between prev_solved_eta and next_eta (forward-only).

    Returns updated (carry_U_data, predictor_prev, predictor_curr, predictor_prev2,
    prev_solved_eta).
    """
    import firedrake as fd
    import firedrake.adjoint as adj
    from Forward.bv_solver import (
        build_context as bv_build_context,
        build_forms as bv_build_forms,
        set_initial_conditions as bv_set_initial_conditions,
    )

    gap = abs(next_eta - prev_solved_eta)
    if gap <= max_eta_gap:
        return carry_U_data, predictor_prev, predictor_curr, predictor_prev2, prev_solved_eta

    n_bridges = int(np.ceil(gap / max_eta_gap)) - 1
    bridge_etas = np.linspace(prev_solved_eta, next_eta, n_bridges + 2)[1:-1]

    for eta_b in bridge_etas:
        with adj.stop_annotating():
            bridge_params = configure_bv_solver_params(
                base_solver_params, phi_applied=eta_b,
                k0_values=k0_list, alpha_values=alpha_list, a_values=a_list,
            )
            bridge_ctx = bv_build_context(bridge_params, mesh=shared_mesh)
            bridge_ctx = bv_build_forms(bridge_ctx, bridge_params)
            bv_set_initial_conditions(bridge_ctx, bridge_params, blob=blob_initial_condition)

            # Simple warm-start from carry_U_data (no predictor for bridges
            # to avoid aggressive extrapolation causing SNES divergence)
            for src, dst in zip(carry_U_data, bridge_ctx["U"].dat):
                dst.data[:] = src
            bridge_ctx["U_prev"].assign(bridge_ctx["U"])

            # Build solver
            bridge_jac = fd.derivative(bridge_ctx["F_res"], bridge_ctx["U"])
            bridge_problem = fd.NonlinearVariationalProblem(
                bridge_ctx["F_res"], bridge_ctx["U"],
                bcs=bridge_ctx["bcs"], J=bridge_jac)

            bridge_solve_params = dict(bridge_params[10]) if isinstance(bridge_params[10], dict) else {}
            bridge_solve_params.setdefault("snes_lag_jacobian", 2)
            bridge_solve_params.setdefault("snes_lag_jacobian_persists", True)
            bridge_solver = fd.NonlinearVariationalSolver(
                bridge_problem, solver_parameters=bridge_solve_params)

            # Build observable for convergence check
            obs_form = _build_bv_observable_form(
                bridge_ctx, mode=observable_mode,
                reaction_index=observable_reaction_index,
                scale=float(observable_scale))

            # SER adaptive dt for bridge
            dt_const = bridge_ctx.get("dt_const")
            dt_initial = float(dt_const) if dt_const is not None else 1.0
            dt_current = dt_initial
            dt_max = dt_initial * _BRIDGE_SER_DT_MAX_RATIO

            prev_bridge_flux = None
            prev_delta = None
            steady_count = 0
            bridge_converged = False
            steps_taken = 0

            for step in range(1, _BRIDGE_MAX_STEPS + 1):
                steps_taken = step
                try:
                    bridge_solver.solve()
                except Exception:
                    break
                bridge_ctx["U_prev"].assign(bridge_ctx["U"])

                flux_val = float(fd.assemble(obs_form))

                if prev_bridge_flux is not None:
                    delta = abs(flux_val - prev_bridge_flux)
                    scale_val = max(abs(flux_val), abs(prev_bridge_flux), abs_tol)
                    rel_m = delta / scale_val
                    abs_m = delta
                    is_steady = (rel_m <= rel_tol) or (abs_m <= abs_tol)
                    steady_count = steady_count + 1 if is_steady else 0

                    # SER for bridge
                    if dt_const is not None and prev_delta is not None and delta > 0:
                        ratio = prev_delta / delta
                        if ratio > 1.0:
                            grow = min(ratio, _SER_GROWTH_CAP)
                            dt_current = min(dt_current * grow, dt_max)
                        else:
                            dt_current = max(dt_current * _SER_SHRINK, dt_initial)
                        dt_const.assign(dt_current)
                    prev_delta = delta
                else:
                    steady_count = 0

                prev_bridge_flux = flux_val
                if steady_count >= 4:
                    bridge_converged = True
                    break

            # Update carry_U_data from bridge solution
            carry_U_data = tuple(d.data_ro.copy() for d in bridge_ctx["U"].dat)

            # Update predictor state (3-point history)
            predictor_prev2 = predictor_prev
            predictor_prev = predictor_curr
            predictor_curr = (eta_b, carry_U_data)

            conv_tag = "converged" if bridge_converged else "NOT converged"
            print(
                f"  [bridge] eta={eta_b:+8.4f} steps={steps_taken} "
                f"({conv_tag}, forward-only)"
            )

        prev_solved_eta = eta_b

    return carry_U_data, predictor_prev, predictor_curr, predictor_prev2, prev_solved_eta


def _build_sweep_order(phi_applied_values: np.ndarray) -> np.ndarray:
    """Build sweep order for warm-start continuation with mixed-sign eta.

    When all phi_applied values share the same sign (or are zero), this
    reduces to the original ``np.argsort(np.abs(phi_applied_values))``
    (ascending |eta|).

    When both positive and negative values are present, a two-branch sweep
    is used:

    1. **Negative branch** (eta <= 0): sorted ascending in |eta|.
    2. **Positive branch** (eta > 0):  sorted ascending in eta.

    The negative branch is processed first (it typically contains the
    smallest-|eta| point, e.g. eta = -0.25).  The positive branch then
    starts from the smallest positive eta.  Between branches, the
    warm-start carries the state from the last negative point solved
    at the lowest |eta| end.  A bridge point through eta = 0 can be
    inserted by the existing bridge-point logic if ``max_eta_gap > 0``.

    If the smallest-|eta| point is positive, the positive branch goes
    first instead.

    Parameters
    ----------
    phi_applied_values : np.ndarray
        Array of dimensionless overpotentials.

    Returns
    -------
    np.ndarray
        Array of original indices giving the sweep order.
    """
    phi = np.asarray(phi_applied_values, dtype=float)
    n = len(phi)

    neg_mask = phi <= 0
    pos_mask = phi > 0

    has_neg = neg_mask.any()
    has_pos = pos_mask.any()

    if not has_neg or not has_pos:
        # Single-sign case: original behaviour (ascending |eta|)
        return np.argsort(np.abs(phi))

    # Two-branch case.
    # Indices in each branch, sorted outward from equilibrium.
    neg_indices = np.where(neg_mask)[0]
    neg_sorted = neg_indices[np.argsort(np.abs(phi[neg_indices]))]  # ascending |eta|

    pos_indices = np.where(pos_mask)[0]
    pos_sorted = pos_indices[np.argsort(phi[pos_indices])]  # ascending eta

    # Choose which branch goes first: the one containing the smallest |eta|.
    min_neg_abs = np.min(np.abs(phi[neg_indices]))
    min_pos_abs = np.min(np.abs(phi[pos_indices]))

    if min_neg_abs <= min_pos_abs:
        # Negative branch first (typical case: near-eq points are slightly negative)
        # Sweep negative outward, then positive outward.
        return np.concatenate([neg_sorted, pos_sorted])
    else:
        # Positive branch first (rare, but handles the case)
        return np.concatenate([pos_sorted, neg_sorted])


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


def _solve_cached_fast_path(
    *,
    n_points: int,
    n_controls: int,
    phi_applied_values: np.ndarray,
    target_flux: np.ndarray,
    base_solver_params: Sequence[object],
    k0_list: list,
    alpha_list: Optional[list],
    a_list: Optional[list],
    control_mode: str,
    shared_mesh: Any,
    blob_initial_condition: bool,
    fail_penalty: float,
    forward_recovery: ForwardRecoveryConfig,
    observable_mode: str,
    observable_reaction_index: Optional[int],
    observable_scale: float,
    abs_tol: float,
    rel_tol: float,
    max_steps: int,
    required_steady: int,
    max_attempts: int,
    mesh: Any,
    parallel_pool: Any = None,
) -> Optional[List[PointAdjointResult]]:
    """Solve all points independently using cached ICs (no sequential sweep).

    When *parallel_pool* is provided (a ``BVPointSolvePool`` instance), tasks
    are dispatched to worker processes for wall-clock parallelism.  If the
    parallel path fails or is unavailable, falls back to sequential.

    Returns None if any point fails SNES (caller should fall back to
    full sequential sweep).
    """
    # ---- Parallel fast path ----
    # Use explicit parameter if given, else fall back to module-level pool
    effective_pool = parallel_pool if parallel_pool is not None else _parallel_pool
    if effective_pool is not None and getattr(effective_pool, 'enabled', False):
        # Check config compatibility: the pool's worker config has a fixed
        # observable_mode/scale/control_mode/n_controls. If the current call
        # uses different settings (e.g. secondary observable in multi-obs
        # inference), skip parallel and use sequential fast path.
        pool_cfg = getattr(effective_pool, '_config', None)
        config_compatible = True
        if pool_cfg is not None:
            if str(pool_cfg.observable_mode) != str(observable_mode):
                config_compatible = False
            if abs(float(pool_cfg.observable_scale) - float(observable_scale)) > 1e-12:
                config_compatible = False
            if str(pool_cfg.control_mode) != str(control_mode):
                config_compatible = False
            if int(pool_cfg.n_controls) != int(n_controls):
                config_compatible = False

        if config_compatible:
            parallel_result = _solve_cached_fast_path_parallel(
                n_points=n_points,
                n_controls=n_controls,
                phi_applied_values=phi_applied_values,
                target_flux=target_flux,
                k0_list=k0_list,
                alpha_list=alpha_list,
                a_list=a_list,
                control_mode=control_mode,
                fail_penalty=fail_penalty,
                observable_mode=observable_mode,
                observable_reaction_index=observable_reaction_index,
                observable_scale=observable_scale,
                parallel_pool=effective_pool,
            )
            if parallel_result is not None:
                return parallel_result
            print("[bv-parallel] Parallel fast path failed; falling back to sequential")
        else:
            print(
                f"[bv-parallel] Config mismatch (pool: {pool_cfg.observable_mode}, "
                f"requested: {observable_mode}); using sequential fast path"
            )

    # ---- Sequential fast path (original) ----
    import firedrake as fd
    import firedrake.adjoint as adj
    from Forward.bv_solver import (
        build_context as bv_build_context,
        build_forms as bv_build_forms,
        set_initial_conditions as bv_set_initial_conditions,
    )
    from Forward.steady_state import configure_bv_solver_params

    results: List[Optional[PointAdjointResult]] = [None] * n_points

    for orig_idx in range(n_points):
        phi_applied_i = float(phi_applied_values[orig_idx])
        target_i = float(target_flux[orig_idx])

        if np.isnan(target_i):
            results[orig_idx] = PointAdjointResult(
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
            return None  # missing cache entry, fall back

        params = configure_bv_solver_params(
            base_solver_params,
            phi_applied=phi_applied_i,
            k0_values=k0_list,
            alpha_values=alpha_list,
            a_values=a_list,
        )

        t_start = _time.perf_counter()

        tape = adj.get_working_tape()
        tape.clear_tape()
        adj.continue_annotation()

        ctx = bv_build_context(params, mesh=shared_mesh)
        ctx = bv_build_forms(ctx, params)
        bv_set_initial_conditions(ctx, params, blob=blob_initial_condition)

        # Warm-start from cached IC
        for src, dst in zip(cached_U_data, ctx["U"].dat):
            dst.data[:] = src
        ctx["U_prev"].assign(ctx["U"])

        # Assign control values
        k0_funcs = list(ctx["bv_k0_funcs"])
        for j, k0_f in enumerate(k0_funcs):
            if j < len(k0_list):
                k0_f.assign(float(k0_list[j]))
        alpha_funcs = list(ctx.get("bv_alpha_funcs", []))
        if alpha_list is not None:
            for j, alpha_f in enumerate(alpha_funcs):
                if j < len(alpha_list):
                    alpha_f.assign(float(alpha_list[j]))
        steric_a_funcs = list(ctx.get("steric_a_funcs", []))
        if a_list is not None:
            for j, a_f in enumerate(steric_a_funcs):
                if j < len(a_list):
                    a_f.assign(float(a_list[j]))

        if control_mode == "k0":
            control_funcs = list(k0_funcs)
        elif control_mode == "alpha":
            control_funcs = list(alpha_funcs)
        elif control_mode == "joint":
            control_funcs = list(k0_funcs) + list(alpha_funcs)
        elif control_mode == "steric":
            control_funcs = list(steric_a_funcs)
        elif control_mode == "full":
            control_funcs = list(k0_funcs) + list(alpha_funcs) + list(steric_a_funcs)
        else:
            control_funcs = list(k0_funcs)

        U = ctx["U"]
        U_prev = ctx["U_prev"]
        F_res = ctx["F_res"]
        bcs = ctx["bcs"]

        jac = fd.derivative(F_res, U)
        problem = fd.NonlinearVariationalProblem(F_res, U, bcs=bcs, J=jac)
        solve_params = dict(params[10]) if isinstance(params[10], dict) else {}
        solve_params.setdefault("snes_lag_jacobian", 2)
        solve_params.setdefault("snes_lag_jacobian_persists", True)
        solver = fd.NonlinearVariationalSolver(problem, solver_parameters=solve_params)

        observable_form = _build_bv_observable_form(
            ctx, mode=observable_mode,
            reaction_index=observable_reaction_index,
            scale=float(observable_scale),
        )

        # SER adaptive dt
        dt_const = ctx.get("dt_const")
        dt_initial = float(dt_const) if dt_const is not None else 1.0
        dt_current = dt_initial
        dt_max = dt_initial * _SER_DT_MAX_RATIO

        prev_flux_val = None
        steady_count = 0
        rel_metric = None
        abs_metric = None
        simulated_flux = float("nan")
        steps_taken = 0
        failed = False
        prev_delta = None

        effective_max_steps = min(max_steps, _WARMSTART_MAX_STEPS)

        for step in range(1, effective_max_steps + 1):
            steps_taken = step
            try:
                solver.solve()
            except Exception:
                failed = True
                break
            U_prev.assign(U)

            with adj.stop_annotating():
                simulated_flux = float(fd.assemble(observable_form))

            if prev_flux_val is not None:
                delta = abs(simulated_flux - prev_flux_val)
                scale_val = max(abs(simulated_flux), abs(prev_flux_val), abs_tol)
                rel_metric = delta / scale_val
                abs_metric = delta
                is_steady = (rel_metric <= rel_tol) or (abs_metric <= abs_tol)
                steady_count = steady_count + 1 if is_steady else 0

                if dt_const is not None and prev_delta is not None and delta > 0:
                    ratio = prev_delta / delta
                    if ratio > 1.0:
                        grow = min(ratio, _SER_GROWTH_CAP)
                        dt_current = min(dt_current * grow, dt_max)
                    else:
                        dt_current = max(dt_current * _SER_SHRINK, dt_initial)
                    dt_const.assign(dt_current)
                prev_delta = delta
            else:
                steady_count = 0
            prev_flux_val = simulated_flux

            if steady_count >= required_steady:
                break

        if dt_const is not None:
            dt_const.assign(dt_initial)

        if failed or steady_count < required_steady:
            # This point failed with cached IC; fall back to full sweep
            return None

        # Compute adjoint gradient
        target_ctrl = _build_bv_scalar_target_in_control_space(
            ctx, target_i, name="target_flux_value",
            control_mode=control_mode,
        )
        target_scalar = fd.assemble(target_ctrl * fd.dx(domain=ctx["mesh"]))
        sim_scalar = fd.assemble(observable_form)
        point_objective = 0.5 * (sim_scalar - target_scalar) ** 2

        controls = [adj.Control(cf) for cf in control_funcs]
        rf = adj.ReducedFunctional(point_objective, controls)
        try:
            point_gradient = _bv_gradient_controls_to_array(
                rf.derivative(), n_controls
            )
        except Exception:
            return None  # adjoint failed, fall back

        # Update cache with new solution
        carry_U_data = tuple(d.data_ro.copy() for d in U.dat)
        _all_points_cache[orig_idx] = carry_U_data

        t_elapsed = _time.perf_counter() - t_start
        print(
            f"  [timing] phi={phi_applied_i:+8.4f} "
            f"total={t_elapsed:.2f}s "
            f"steps={steps_taken}/{effective_max_steps} "
            f"(cached-IC)"
        )

        results[orig_idx] = PointAdjointResult(
            phi_applied=phi_applied_i,
            target_flux=target_i,
            simulated_flux=float(sim_scalar),
            objective=float(point_objective),
            gradient=point_gradient,
            converged=True,
            steps_taken=steps_taken,
            reason="",
            final_relative_change=rel_metric,
            final_absolute_change=abs_metric,
            diagnostics_valid=True,
        )

    return [r for r in results]  # type: ignore[misc]


def solve_bv_curve_points_with_warmstart(
    *,
    base_solver_params: Sequence[object],
    steady: SteadyStateConfig,
    phi_applied_values: np.ndarray,
    target_flux: np.ndarray,
    k0_values: Sequence[float],
    blob_initial_condition: bool,
    fail_penalty: float,
    forward_recovery: ForwardRecoveryConfig,
    observable_mode: str,
    observable_reaction_index: Optional[int],
    observable_scale: float,
    mesh: Any = None,
    alpha_values: Optional[Sequence[float]] = None,
    control_mode: str = "k0",
    a_values: Optional[Sequence[float]] = None,
    max_eta_gap: float = 0.0,
    parallel_pool: Any = None,
) -> List[PointAdjointResult]:
    """Solve all phi_applied points sequentially with warm-start continuation.

    Points are solved outward from equilibrium, with each point's converged
    solution carried forward as the initial condition for the next.  When
    all points share the same sign, the order is ascending |phi_applied|.
    When both positive and negative values are present, a two-branch sweep
    is used: negative eta ascending in |eta|, then positive eta ascending.
    This prevents sign-change warm-start failures at moderate |eta|.
    Each point gets its own adjoint tape for dJ/d(control) computation.

    Parameters
    ----------
    base_solver_params :
        11-element list / SolverParams.
    steady :
        Steady-state convergence settings.
    phi_applied_values :
        Array of dimensionless overpotentials to sweep.
    target_flux :
        Array of target current-density values (same length as phi_applied_values).
    k0_values :
        Current k0 values to evaluate.
    mesh :
        Optional pre-built mesh (e.g. graded rectangle).
    alpha_values :
        Current alpha values (required when control_mode is "alpha" or "joint").
    control_mode : str
        ``"k0"``, ``"alpha"``, ``"joint"``, ``"steric"``, or ``"full"``.
    a_values :
        Current steric a values (required when control_mode is "steric" or "full").
    max_eta_gap : float
        Maximum allowed gap in eta between consecutive solved points.
        When > 0, auto-insert forward-only bridge points to span larger gaps.
        Default 0.0 disables bridge point insertion (backward compatible).

    Returns
    -------
    List[PointAdjointResult]
        One result per phi_applied point, in the ORIGINAL order of phi_applied_values.
    """
    global _cache_populated

    import firedrake as fd
    import firedrake.adjoint as adj

    k0_list = [float(v) for v in k0_values]
    alpha_list = [float(v) for v in alpha_values] if alpha_values is not None else None
    a_list = [float(v) for v in a_values] if a_values is not None else None

    if control_mode == "k0":
        n_controls = len(k0_list)
    elif control_mode == "alpha":
        if alpha_list is None:
            raise ValueError("alpha_values required when control_mode='alpha'")
        n_controls = len(alpha_list)
    elif control_mode == "joint":
        if alpha_list is None:
            raise ValueError("alpha_values required when control_mode='joint'")
        n_controls = len(k0_list) + len(alpha_list)
    elif control_mode == "steric":
        # Steric a_vals: n_species controls (one per species)
        n_species = int(base_solver_params[0])
        n_controls = n_species
    elif control_mode == "full":
        # Full: k0 + alpha + steric a
        if alpha_list is None:
            raise ValueError("alpha_values required when control_mode='full'")
        n_species = int(base_solver_params[0])
        n_controls = len(k0_list) + len(alpha_list) + n_species
    else:
        raise ValueError(f"Unknown control_mode '{control_mode}'")
    n_points = len(phi_applied_values)

    abs_tol = float(max(steady.absolute_tolerance, 1e-16))
    rel_tol = float(steady.relative_tolerance)
    max_steps = int(max(1, steady.max_steps))
    required_steady = int(max(1, steady.consecutive_steps))

    max_attempts = max(1, int(forward_recovery.max_attempts))

    # Sort points for warm-start continuation.
    #
    # When all phi_applied have the same sign (or are zero), sorting by |eta|
    # is optimal: we sweep outward from equilibrium monotonically.
    #
    # When phi_applied includes BOTH positive and negative values, naive
    # |eta| sorting interleaves them (e.g., -2, +2, +3, -3, ...).
    # Warm-starting from a cathodic solution to an anodic target (or vice
    # versa) causes SNES divergence because the concentration profiles are
    # inverted (e.g., O2 depleted vs accumulated at the electrode).
    #
    # Fix: "two-branch sweep".  Process negative-or-zero eta ascending in
    # |eta|, then positive eta ascending.  The only sign-change transition
    # happens between the smallest-|eta| negative point and the smallest
    # positive point, which are both near equilibrium and have similar
    # concentration profiles.
    sorted_indices = _build_sweep_order(phi_applied_values)
    results: List[Optional[PointAdjointResult]] = [None] * n_points

    # The first point's params set phi_applied for context construction.
    first_phi = float(phi_applied_values[sorted_indices[0]])

    from Forward.bv_solver import (
        build_context as bv_build_context,
        build_forms as bv_build_forms,
        set_initial_conditions as bv_set_initial_conditions,
    )

    # ---------------------------------------------------------------
    # ONE-TIME SETUP: build mesh and function spaces ONCE (Item 5).
    # These are not adjoint-tracked and do not need to be on the tape.
    # Only the Forms and control-Function assignments need taping.
    # ---------------------------------------------------------------
    first_params = configure_bv_solver_params(
        base_solver_params,
        phi_applied=first_phi,
        k0_values=k0_list,
        alpha_values=alpha_list,
        a_values=a_list,
    )
    with adj.stop_annotating():
        base_ctx = bv_build_context(first_params, mesh=mesh)
    # base_ctx now has: mesh, V_scalar, W, U (template), U_prev (template),
    # n_species.  These are reused for every point.
    shared_mesh = base_ctx["mesh"]
    shared_W = base_ctx["W"]

    # P7: Checkpoint-restart fast path.
    # If we have cached solutions for all points from a previous full sweep,
    # solve each point independently using its cached IC (no sequential sweep).
    # This is much faster because points can start from a nearby solution
    # rather than sweeping from equilibrium each time.
    if _cache_populated and len(_all_points_cache) >= n_points:
        _fast_path_ok = True
        for _i in range(n_points):
            if _i not in _all_points_cache:
                _fast_path_ok = False
                break
        if _fast_path_ok:
            print("[checkpoint] Using cached ICs for all points (fast path)")
            fast_results = _solve_cached_fast_path(
                n_points=n_points,
                n_controls=n_controls,
                phi_applied_values=phi_applied_values,
                target_flux=target_flux,
                base_solver_params=base_solver_params,
                k0_list=k0_list,
                alpha_list=alpha_list,
                a_list=a_list,
                control_mode=control_mode,
                shared_mesh=shared_mesh,
                blob_initial_condition=blob_initial_condition,
                fail_penalty=fail_penalty,
                forward_recovery=forward_recovery,
                observable_mode=observable_mode,
                observable_reaction_index=observable_reaction_index,
                observable_scale=observable_scale,
                abs_tol=abs_tol,
                rel_tol=rel_tol,
                max_steps=max_steps,
                required_steady=required_steady,
                max_attempts=max_attempts,
                mesh=mesh,
                parallel_pool=parallel_pool,
            )
            if fast_results is not None:
                return fast_results
            # Fast path failed on some point; fall through to full sequential sweep
            print("[checkpoint] Fast path failed; falling back to sequential sweep")

    # P2: Cross-evaluation warm-start.  If we have a cached solution
    # for the first point from a previous evaluation, use it instead of
    # cold-starting.  This is safe because adjacent optimizer iterations
    # have similar control values, so the old solution is a good IC.
    first_orig_idx = int(sorted_indices[0])
    carry_U_data = _cross_eval_cache.get(first_orig_idx, None)

    # Predictor step state: 3-point history for quadratic predictor.
    predictor_prev2: Optional[tuple] = None  # (eta, U_data) three points ago
    predictor_prev: Optional[tuple] = None   # (eta, U_data) two points ago
    predictor_curr: Optional[tuple] = None   # (eta, U_data) one point ago

    # Track previous solved eta for bridge point insertion
    prev_solved_eta: Optional[float] = None

    # Hub state: saved near-equilibrium warm-start for second branch.
    # When phi_applied contains both signs, the first branch sweeps outward
    # and saves the state at the smallest |eta| point.  When the second
    # branch starts, carry_U_data is restored to this hub state so the
    # warm-start transition is a small jump near eta = 0.
    hub_U_data: Optional[tuple] = None
    hub_eta: Optional[float] = None
    _first_branch_done = False  # set True after first point of second branch

    # Determine if we have a mixed-sign sweep (need hub logic)
    _has_mixed_signs = (
        (phi_applied_values <= 0).any() and (phi_applied_values > 0).any()
    )

    for sweep_idx, orig_idx in enumerate(sorted_indices):
        phi_applied_i = float(phi_applied_values[orig_idx])
        target_i = float(target_flux[orig_idx])

        # Skip points where the target is NaN (unconverged target data).
        # These contribute zero objective and zero gradient.  We still
        # solve them as forward-only bridge-style points to maintain
        # warm-start continuity, but do NOT record them as inference
        # points (no adjoint, no tape).
        if np.isnan(target_i):
            print(
                f"  [skip] phi={phi_applied_i:+8.4f} target=NaN "
                f"(target generation failed; skipping adjoint)"
            )
            # Still do a forward-only solve to maintain warm-start chain
            if carry_U_data is not None and max_eta_gap > 0 and prev_solved_eta is not None:
                (carry_U_data, predictor_prev, predictor_curr,
                 predictor_prev2, prev_solved_eta) = _solve_bridge_points(
                    prev_solved_eta=prev_solved_eta,
                    next_eta=phi_applied_i,
                    max_eta_gap=max_eta_gap,
                    carry_U_data=carry_U_data,
                    base_solver_params=base_solver_params,
                    k0_list=k0_list,
                    alpha_list=alpha_list,
                    a_list=a_list,
                    shared_mesh=shared_mesh,
                    observable_mode=observable_mode,
                    observable_reaction_index=observable_reaction_index,
                    observable_scale=observable_scale,
                    rel_tol=rel_tol,
                    abs_tol=abs_tol,
                    predictor_prev=predictor_prev,
                    predictor_curr=predictor_curr,
                    predictor_prev2=predictor_prev2,
                    blob_initial_condition=blob_initial_condition,
                )
            results[orig_idx] = PointAdjointResult(
                phi_applied=phi_applied_i,
                target_flux=target_i,
                simulated_flux=float("nan"),
                objective=0.0,  # zero contribution to loss
                gradient=np.zeros(n_controls, dtype=float),
                converged=False,
                steps_taken=0,
                reason="target NaN (skipped)",
                final_relative_change=None,
                final_absolute_change=None,
                diagnostics_valid=False,
            )
            continue

        # ---- Branch transition: restore hub state ----
        # When switching from negative to positive eta (or vice versa),
        # restore the near-equilibrium hub state instead of warm-starting
        # from the last point of the first branch (which may be far from
        # equilibrium, e.g. eta = -28).  Also clear predictor history.
        if (prev_solved_eta is not None
                and carry_U_data is not None
                and not _first_branch_done
                and _has_mixed_signs):
            prev_sign = prev_solved_eta > 0
            curr_sign = phi_applied_i > 0
            if prev_sign != curr_sign:
                _first_branch_done = True
                predictor_prev2 = None
                predictor_prev = None
                predictor_curr = None
                if hub_U_data is not None:
                    carry_U_data = hub_U_data
                    prev_solved_eta = hub_eta

        # ---- Save hub state at first converged point ----
        # The first point of the first branch is the closest to eta = 0
        # (by construction of _build_sweep_order).  Save it for the
        # branch transition.
        # (This is set after convergence below, at sweep_idx == 0.)

        # ---- Bridge point insertion (Phase 1) ----
        # If the gap from the previous solved eta exceeds max_eta_gap,
        # insert forward-only bridge points to carry the warm-start.
        if (max_eta_gap > 0
                and carry_U_data is not None
                and prev_solved_eta is not None):
            (carry_U_data, predictor_prev, predictor_curr,
             predictor_prev2, prev_solved_eta) = _solve_bridge_points(
                prev_solved_eta=prev_solved_eta,
                next_eta=phi_applied_i,
                max_eta_gap=max_eta_gap,
                carry_U_data=carry_U_data,
                base_solver_params=base_solver_params,
                k0_list=k0_list,
                alpha_list=alpha_list,
                a_list=a_list,
                shared_mesh=shared_mesh,
                observable_mode=observable_mode,
                observable_reaction_index=observable_reaction_index,
                observable_scale=observable_scale,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
                predictor_prev=predictor_prev,
                predictor_curr=predictor_curr,
                predictor_prev2=predictor_prev2,
                blob_initial_condition=blob_initial_condition,
            )

        # Use reduced max_steps for warm-started points (Item 2).
        is_warmstart = carry_U_data is not None
        effective_max_steps = min(max_steps, _WARMSTART_MAX_STEPS) if is_warmstart else max_steps

        point_result = None

        # Build the attempt sequence: 0=predictor warm-start, 1=simple warm-start,
        # 2..max_attempts+1 = recovery attempts (cold start with relaxed params).
        # This ensures we try simple warm-start before cold start.
        total_attempts = max_attempts + (1 if is_warmstart else 0)

        for attempt_raw in range(total_attempts):
            # Map attempt_raw to the recovery attempt index:
            # attempt_raw 0 = predictor warm-start (attempt=0)
            # attempt_raw 1 = simple warm-start if available (attempt=0, no predictor)
            # attempt_raw 2+ = recovery attempts (attempt=attempt_raw-1)
            if is_warmstart:
                if attempt_raw == 0:
                    attempt = 0
                    use_predictor = True
                elif attempt_raw == 1:
                    attempt = 0
                    use_predictor = False
                else:
                    attempt = attempt_raw - 1
                    use_predictor = False
            else:
                attempt = attempt_raw
                use_predictor = False

            # On recovery attempts (attempt > 0), revert to full max_steps.
            if attempt > 0:
                effective_max_steps = max_steps

            params = configure_bv_solver_params(
                base_solver_params,
                phi_applied=phi_applied_i,
                k0_values=k0_list,
                alpha_values=alpha_list,
                a_values=a_list,
            )
            if isinstance(params[10], dict):
                baseline_options = copy.deepcopy(params[10])
                phase, phase_step, _cycle = _attempt_phase_state(attempt, forward_recovery)
                _relax_solver_options_for_attempt(
                    params[10],
                    phase=phase,
                    phase_step=phase_step,
                    recovery=forward_recovery,
                    baseline_options=baseline_options,
                )

            # ---- Per-point setup (timed) ----
            t_setup_start = _time.perf_counter()

            # Clear tape and enable annotation for this point
            tape = adj.get_working_tape()
            tape.clear_tape()
            adj.continue_annotation()

            # Reuse the shared mesh and function space (Item 5).
            # build_context creates fresh U/U_prev Functions in shared_W.
            # build_forms creates the UFL residual (must be on the tape).
            ctx = bv_build_context(params, mesh=shared_mesh)
            ctx = bv_build_forms(ctx, params)
            bv_set_initial_conditions(ctx, params, blob=blob_initial_condition)

            # Warm-start: restore last converged state if available.
            if carry_U_data is not None and attempt == 0:
                if use_predictor:
                    _apply_predictor(
                        phi_applied_i, ctx["U"], carry_U_data,
                        predictor_prev, predictor_curr, predictor_prev2,
                    )
                else:
                    # Simple warm-start: direct copy without predictor
                    for src, dst in zip(carry_U_data, ctx["U"].dat):
                        dst.data[:] = src
                ctx["U_prev"].assign(ctx["U"])

            # Assign k0 values to the adjoint-tracked Functions
            k0_funcs = list(ctx["bv_k0_funcs"])
            for j, k0_f in enumerate(k0_funcs):
                if j < len(k0_list):
                    k0_f.assign(float(k0_list[j]))

            # Assign alpha values to the adjoint-tracked Functions
            alpha_funcs = list(ctx.get("bv_alpha_funcs", []))
            if alpha_list is not None:
                for j, alpha_f in enumerate(alpha_funcs):
                    if j < len(alpha_list):
                        alpha_f.assign(float(alpha_list[j]))

            # Assign steric a values to the adjoint-tracked Functions
            steric_a_funcs = list(ctx.get("steric_a_funcs", []))
            if a_list is not None:
                for j, a_f in enumerate(steric_a_funcs):
                    if j < len(a_list):
                        a_f.assign(float(a_list[j]))

            # Select control functions based on control_mode
            if control_mode == "k0":
                control_funcs = list(k0_funcs)
            elif control_mode == "alpha":
                control_funcs = list(alpha_funcs)
            elif control_mode == "joint":
                control_funcs = list(k0_funcs) + list(alpha_funcs)
            elif control_mode == "steric":
                control_funcs = list(steric_a_funcs)
            elif control_mode == "full":
                control_funcs = list(k0_funcs) + list(alpha_funcs) + list(steric_a_funcs)
            else:
                control_funcs = list(k0_funcs)

            U = ctx["U"]
            U_prev = ctx["U_prev"]
            F_res = ctx["F_res"]
            bcs = ctx["bcs"]

            jac = fd.derivative(F_res, U)
            problem = fd.NonlinearVariationalProblem(F_res, U, bcs=bcs, J=jac)

            # P5: Jacobian lagging -- reuse Jacobian/preconditioner for
            # 2 consecutive SNES solves before recomputing.  Cuts MUMPS
            # LU factorization frequency in half.  The "persists" flag
            # keeps the lag across PTC time steps (not just Newton iters).
            solve_params = dict(params[10]) if isinstance(params[10], dict) else {}
            solve_params.setdefault("snes_lag_jacobian", 2)
            solve_params.setdefault("snes_lag_jacobian_persists", True)
            solver = fd.NonlinearVariationalSolver(problem, solver_parameters=solve_params)

            observable_form = _build_bv_observable_form(
                ctx,
                mode=observable_mode,
                reaction_index=observable_reaction_index,
                scale=float(observable_scale),
            )

            t_setup_elapsed = _time.perf_counter() - t_setup_start

            # ---- Forward solve (timed) ----
            t_fwd_start = _time.perf_counter()

            # P6: Adaptive pseudo-timestep (SER).  Read initial dt from
            # the form's fd.Constant and grow/shrink based on convergence.
            dt_const = ctx.get("dt_const")
            dt_initial = float(dt_const) if dt_const is not None else 1.0
            dt_current = dt_initial
            dt_max = dt_initial * _SER_DT_MAX_RATIO

            prev_flux_val: Optional[float] = None
            steady_count = 0
            rel_metric: Optional[float] = None
            abs_metric: Optional[float] = None
            simulated_flux = float("nan")
            steps_taken = 0
            failed_by_exception = False
            prev_delta: Optional[float] = None

            for step in range(1, effective_max_steps + 1):
                steps_taken = step
                try:
                    solver.solve()
                except Exception as exc:
                    failed_by_exception = True
                    last_reason = f"{type(exc).__name__}: {exc}"
                    break

                U_prev.assign(U)

                # Non-annotated assembly for convergence check
                with adj.stop_annotating():
                    simulated_flux = float(fd.assemble(observable_form))

                if prev_flux_val is not None:
                    delta = abs(simulated_flux - prev_flux_val)
                    scale_val = max(abs(simulated_flux), abs(prev_flux_val), abs_tol)
                    rel_metric = delta / scale_val
                    abs_metric = delta
                    is_steady = (rel_metric <= rel_tol) or (abs_metric <= abs_tol)
                    steady_count = steady_count + 1 if is_steady else 0

                    # P6: SER timestep adaptation.
                    if dt_const is not None and prev_delta is not None and delta > 0:
                        ratio = prev_delta / delta
                        if ratio > 1.0:
                            # Residual decreasing: grow dt
                            grow = min(ratio, _SER_GROWTH_CAP)
                            dt_current = min(dt_current * grow, dt_max)
                        else:
                            # Residual increasing: shrink dt
                            dt_current = max(dt_current * _SER_SHRINK, dt_initial)
                        dt_const.assign(dt_current)

                    prev_delta = delta
                else:
                    steady_count = 0

                prev_flux_val = simulated_flux

                if steady_count >= required_steady:
                    break

            # P6: Reset dt to initial value for the next point (each
            # point should start with the base timestep).
            if dt_const is not None:
                dt_const.assign(dt_initial)

            t_fwd_elapsed = _time.perf_counter() - t_fwd_start

            # ---- Adjoint gradient (timed) ----
            t_adj_start = _time.perf_counter()

            if not failed_by_exception and steady_count >= required_steady:
                # Converged -- compute adjoint gradient
                target_ctrl = _build_bv_scalar_target_in_control_space(
                    ctx, target_i, name="target_flux_value",
                    control_mode=control_mode,
                )
                target_scalar = fd.assemble(
                    target_ctrl * fd.dx(domain=ctx["mesh"])
                )
                sim_scalar = fd.assemble(observable_form)
                point_objective = 0.5 * (sim_scalar - target_scalar) ** 2

                controls = [adj.Control(cf) for cf in control_funcs]
                rf = adj.ReducedFunctional(point_objective, controls)
                try:
                    # P1: Skip rf(control_state) -- avoid full tape replay.
                    # The tape already holds valid forward state from the
                    # solve loop above.  rf.derivative() only runs the
                    # backward adjoint sweep (tape.evaluate_adj), which
                    # does NOT re-execute the forward SNES solves.
                    point_objective_value = float(point_objective)
                    point_gradient = _bv_gradient_controls_to_array(
                        rf.derivative(), n_controls
                    )
                except Exception as rf_exc:
                    # Adjoint derivative failed (e.g. PETSc adjoint linear
                    # solve issue).  Treat as a failed point so the
                    # optimizer can continue.
                    print(
                        f"  [adjoint] derivative failed at "
                        f"phi={phi_applied_i:.4f}: {rf_exc}"
                    )
                    failed_by_exception = True
                    last_reason = f"adjoint derivative: {type(rf_exc).__name__}"

                if not failed_by_exception:
                    point_result = PointAdjointResult(
                        phi_applied=phi_applied_i,
                        target_flux=target_i,
                        simulated_flux=float(sim_scalar),
                        objective=point_objective_value,
                        gradient=point_gradient,
                        converged=True,
                        steps_taken=steps_taken,
                        reason="",
                        final_relative_change=rel_metric,
                        final_absolute_change=abs_metric,
                        diagnostics_valid=True,
                    )

                    # Snapshot U for warm-start of next point
                    carry_U_data = tuple(d.data_ro.copy() for d in U.dat)

                    # Update predictor state (3-point history)
                    predictor_prev2 = predictor_prev
                    predictor_prev = predictor_curr
                    predictor_curr = (phi_applied_i, carry_U_data)

                    # Track solved eta for bridge point insertion
                    prev_solved_eta = phi_applied_i

                    # P2: Cache the first point's solution for cross-eval
                    if sweep_idx == 0:
                        _cross_eval_cache[first_orig_idx] = carry_U_data

                    # P7: Cache this point's solution for checkpoint-restart
                    _all_points_cache[orig_idx] = carry_U_data

                    # Save hub state at the first converged point for
                    # branch-transition warm-start (mixed-sign sweep only).
                    if hub_U_data is None and _has_mixed_signs:
                        hub_U_data = carry_U_data
                        hub_eta = phi_applied_i

            t_adj_elapsed = _time.perf_counter() - t_adj_start

            # ---- Profiling output (Item 1) ----
            if is_warmstart and attempt == 0:
                ws_tag = "ws+pred" if use_predictor else "ws"
            else:
                ws_tag = "cold"
            print(
                f"  [timing] phi={phi_applied_i:+8.4f} "
                f"setup={t_setup_elapsed:.2f}s "
                f"fwd={t_fwd_elapsed:.2f}s "
                f"adj={t_adj_elapsed:.2f}s "
                f"steps={steps_taken}/{effective_max_steps} "
                f"({ws_tag})"
            )

            if point_result is not None:
                break  # success on this attempt

            # If this attempt failed, try next recovery phase
            if not failed_by_exception:
                last_reason = "steady-state criterion not satisfied before max_steps"

        if point_result is None:
            point_result = PointAdjointResult(
                phi_applied=phi_applied_i,
                target_flux=target_i,
                simulated_flux=simulated_flux,
                objective=float(fail_penalty),
                gradient=np.zeros(n_controls, dtype=float),
                converged=False,
                steps_taken=steps_taken,
                reason=last_reason if 'last_reason' in dir() else "all attempts failed",
                final_relative_change=None,
                final_absolute_change=None,
                diagnostics_valid=False,
            )
            # Even if this point failed, track its eta for bridge logic
            if prev_solved_eta is None:
                prev_solved_eta = phi_applied_i

        results[orig_idx] = point_result

    # P7: Mark cache as populated after a full sequential sweep.
    # All converged points have been cached in _all_points_cache above.
    n_cached = sum(1 for i in range(n_points) if i in _all_points_cache)
    if n_cached == n_points:
        _cache_populated = True

    return [r for r in results]  # type: ignore[misc]
