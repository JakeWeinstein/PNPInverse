"""Predictor/sweep ordering helpers for BV point solves."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from .cache import (
    _BRIDGE_MAX_STEPS,
    _BRIDGE_SER_DT_MAX_RATIO,
    _SER_GROWTH_CAP,
    _SER_SHRINK,
)
from FluxCurve.bv_observables import _build_bv_observable_form


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
    from Forward.steady_state import configure_bv_solver_params

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

            _bp_opts = bridge_params.solver_options if hasattr(bridge_params, 'solver_options') else bridge_params[10]
            bridge_solve_params = dict(_bp_opts) if isinstance(_bp_opts, dict) else {}
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
