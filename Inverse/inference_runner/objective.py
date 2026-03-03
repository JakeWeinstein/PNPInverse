"""Reduced functional construction and attempt monitoring."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np

from ..parameter_targets import ParameterTarget
from ..solver_interface import (
    ForwardSolverAdapter,
    deep_copy_solver_params,
)
from .data import _vector_to_function


def build_reduced_functional(
    *,
    adapter: ForwardSolverAdapter,
    target: ParameterTarget,
    solver_params: Sequence[Any],
    concentration_targets: Sequence[Sequence[float]],
    phi_target: Sequence[float],
    blob_initial_condition: bool = True,
    print_interval: int = 100,
    extra_eval_cb_pre: Optional[Any] = None,
    extra_eval_cb_post: Optional[Any] = None,
):
    """Build a Firedrake-adjoint reduced functional for a configured target."""
    import firedrake as fd
    import firedrake.adjoint as adj

    params_copy = deep_copy_solver_params(solver_params)
    n_species = int(params_copy[0])

    if len(concentration_targets) != n_species:
        raise ValueError(
            f"Expected {n_species} concentration targets, got {len(concentration_targets)}."
        )

    ctx = adapter.build_context(params_copy)
    ctx = adapter.build_forms(ctx, params_copy)

    c_target_fs = [
        _vector_to_function(ctx, c_vec, space_key="V_scalar")
        for c_vec in concentration_targets
    ]
    phi_target_f = _vector_to_function(ctx, phi_target, space_key="V_scalar")

    tape = adj.get_working_tape()
    tape.clear_tape()
    adj.continue_annotation()

    adapter.set_initial_conditions(ctx, params_copy, blob=blob_initial_condition)
    U_final = adapter.solve(ctx, params_copy, print_interval=print_interval)

    objective_terms = []
    if "concentration" in target.objective_fields:
        for i in range(n_species):
            diff_i = U_final.sub(i) - c_target_fs[i]
            objective_terms.append(fd.inner(diff_i, diff_i))

    if "phi" in target.objective_fields:
        phi_diff = U_final.sub(n_species) - phi_target_f
        objective_terms.append(fd.inner(phi_diff, phi_diff))

    if not objective_terms:
        raise ValueError(
            f"Target '{target.key}' must request at least one objective field."
        )

    Jobj = 0.5 * fd.assemble(sum(objective_terms) * fd.dx)

    control_functions = list(target.controls_from_context(ctx))
    if not control_functions:
        raise ValueError(f"Target '{target.key}' returned no optimization controls.")

    controls = [adj.Control(ctrl) for ctrl in control_functions]
    control_arg = controls[0] if len(controls) == 1 else controls

    rf_kwargs: Dict[str, Any] = {}
    target_pre = target.eval_cb_pre_factory(ctx) if target.eval_cb_pre_factory is not None else None
    target_post = (
        target.eval_cb_post_factory(ctx) if target.eval_cb_post_factory is not None else None
    )

    if target_pre is not None or extra_eval_cb_pre is not None:
        def chained_pre(m: Any) -> None:
            if target_pre is not None:
                target_pre(m)
            if extra_eval_cb_pre is not None:
                extra_eval_cb_pre(m)

        rf_kwargs["eval_cb_pre"] = chained_pre

    if target_post is not None or extra_eval_cb_post is not None:
        def chained_post(j: float, m: Any) -> None:
            if target_post is not None:
                target_post(j, m)
            if extra_eval_cb_post is not None:
                extra_eval_cb_post(j, m)

        rf_kwargs["eval_cb_post"] = chained_post

    rf = adj.ReducedFunctional(Jobj, control_arg, **rf_kwargs)
    return rf


@dataclass
class _AttemptMonitor:
    """Collect feasible points seen during one optimization attempt."""

    target: ParameterTarget
    best_objective: Optional[float] = None
    best_estimate: Optional[Any] = None
    last_estimate: Optional[Any] = None
    n_successful_evals: int = 0

    def eval_cb_post(self, j: float, m: Any) -> None:
        """Observe successful objective evaluations from the reduced functional."""
        try:
            estimate = copy.deepcopy(self.target.estimate_from_controls(m))
        except Exception:
            return

        self.last_estimate = estimate
        j_float = float(j)
        if np.isfinite(j_float) and (
            self.best_objective is None or j_float < self.best_objective
        ):
            self.best_objective = j_float
            self.best_estimate = copy.deepcopy(estimate)
        self.n_successful_evals += 1
