"""Pre-built objective factories for common inverse problems.

Each factory wires a ``ForwardSolverAdapter``, a ``ParameterTarget``, and the
``build_reduced_functional`` helper into a single call that returns a
``ReducedFunctional`` ready for use with ``scipy.optimize``.

These were previously in ``Helpers/Infer_{D,DirichletBC,RobinBC}_from_data_helpers.py``.
"""

from __future__ import annotations

from typing import Any, Sequence

from Inverse.solver_interface import ForwardSolverAdapter
from Inverse.parameter_targets import build_default_target_registry
from Inverse.inference_runner import build_reduced_functional


def make_diffusion_objective_and_grad(
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
    adapter = ForwardSolverAdapter.from_module_path("Forward.dirichlet_solver")
    target = build_default_target_registry()["diffusion"]

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


def make_dirichlet_phi0_objective_and_grad(
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
        Concentration targets.
    phi_vec:
        Electric-potential target vector.
    blob_ic:
        Whether to initialize with the Gaussian blob concentration profile.

    Returns
    -------
    firedrake.adjoint.ReducedFunctional
        Objective functional parameterized by the scalar ``phi0`` control.
    """
    adapter = ForwardSolverAdapter.from_module_path("Forward.dirichlet_solver")
    target = build_default_target_registry()["dirichlet_phi0"]

    return build_reduced_functional(
        adapter=adapter,
        target=target,
        solver_params=solver_params,
        concentration_targets=c_targets,
        phi_target=phi_vec,
        blob_initial_condition=blob_ic,
    )


def make_robin_kappa_objective_and_grad(
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
        "Forward.robin_solver", solve_function_name="forsolve_robin"
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
