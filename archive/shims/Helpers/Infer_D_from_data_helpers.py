"""Compatibility helper for diffusion inference using the unified engine.

This module keeps the historical import path used by study scripts while routing
all logic through ``UnifiedInverse``.
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
    blob_ic: bool = True,
):
    """Build a reduced functional for inferring diffusion coefficients ``D``.

    Parameters
    ----------
    solver_params:
        Standard 11-entry solver parameter list.
    c_targets:
        Final concentration vectors to match, one per species.
    blob_ic:
        Whether to initialize with the Gaussian blob concentration profile.

    Returns
    -------
    firedrake.adjoint.ReducedFunctional
        Objective functional parameterized by log-diffusion controls.
    """
    adapter = ForwardSolverAdapter.from_module_path("Utils.forsolve")
    target = build_default_target_registry()["diffusion"]

    # The diffusion objective uses only concentrations. A placeholder phi target
    # is supplied for interface consistency and is ignored by the target config.
    if not c_targets:
        raise ValueError("c_targets must include at least one concentration vector.")
    phi_placeholder = c_targets[0]

    return build_reduced_functional(
        adapter=adapter,
        target=target,
        solver_params=solver_params,
        concentration_targets=c_targets,
        phi_target=phi_placeholder,
        blob_initial_condition=blob_ic,
    )
