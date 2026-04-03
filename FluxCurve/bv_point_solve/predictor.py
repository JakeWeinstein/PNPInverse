"""Predictor/sweep ordering helpers for BV point solves.

The core predictor (_apply_predictor) and sweep order (_build_sweep_order)
functions now live in Forward.bv_solver.sweep_order.  This module re-exports
them for backward compatibility and adds the bridge-point solver which depends
on FluxCurve-layer infrastructure.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from .cache import (
    _BRIDGE_MAX_STEPS,
    _BRIDGE_SER_DT_MAX_RATIO,
    _SER_GROWTH_CAP,
    _SER_SHRINK,
)
from Forward.bv_solver.observables import _build_bv_observable_form
from Forward.bv_solver.sweep_order import _apply_predictor, _build_sweep_order  # noqa: F401 — re-export


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
