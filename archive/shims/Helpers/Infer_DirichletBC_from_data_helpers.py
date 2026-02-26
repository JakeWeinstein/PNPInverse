"""Compatibility helper for Dirichlet-phi0 inference using the unified engine.

This module preserves the previous helper import path while delegating to the
new modular interface in ``UnifiedInverse``.
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
    """Build a reduced functional for inferring Dirichlet boundary value ``phi0``.

    Parameters
    ----------
    solver_params:
        Standard 11-entry solver parameter list.
    c_targets:
        Concentration targets. Kept for compatibility with old signatures.
        The Dirichlet ``phi0`` target uses only ``phi_vec`` in the objective.
    phi_vec:
        Electric-potential target vector.
    blob_ic:
        Whether to initialize with the Gaussian blob concentration profile.

    Returns
    -------
    firedrake.adjoint.ReducedFunctional
        Objective functional parameterized by the scalar ``phi0`` control.
    """
    adapter = ForwardSolverAdapter.from_module_path("Utils.forsolve")
    target = build_default_target_registry()["dirichlet_phi0"]

    return build_reduced_functional(
        adapter=adapter,
        target=target,
        solver_params=solver_params,
        concentration_targets=c_targets,
        phi_target=phi_vec,
        blob_initial_condition=blob_ic,
    )
