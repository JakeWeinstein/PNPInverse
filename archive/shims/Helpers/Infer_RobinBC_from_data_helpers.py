"""Compatibility helper for Robin-kappa inference using the unified engine.

This wrapper keeps legacy imports functional while routing through the new
unified inverse interface.
"""

from __future__ import annotations

from typing import Any, Sequence

from UnifiedInverse import (
    ForwardSolverAdapter,
    build_default_target_registry,
    build_reduced_functional,
)


def make_objective_and_grad(
    solver_params: Sequence[Any],
    c_targets: Sequence[Sequence[float]],
    phi_vec: Sequence[float],
    blob_ic: bool = True,
):
    """Build a reduced functional for inferring Robin transfer coefficient(s).

    Parameters
    ----------
    solver_params:
        Standard 11-entry solver parameter list configured for a Robin solver.
    c_targets:
        Final concentration vectors, one per species.
    phi_vec:
        Final electric-potential vector.
    blob_ic:
        Whether to initialize with the Gaussian blob concentration profile.

    Returns
    -------
    firedrake.adjoint.ReducedFunctional
        Objective functional parameterized by per-species ``kappa`` controls.
    """
    adapter = ForwardSolverAdapter.from_module_path(
        "Utils.robin_forsolve", solve_function_name="forsolve_robin"
    )
    target = build_default_target_registry()["robin_kappa"]

    return build_reduced_functional(
        adapter=adapter,
        target=target,
        solver_params=solver_params,
        concentration_targets=c_targets,
        phi_target=phi_vec,
        blob_initial_condition=blob_ic,
    )
