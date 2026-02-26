"""Dirichlet-BC PNP forward solver.

Solves the Poisson-Nernst-Planck equations on a unit-square mesh with
Dirichlet boundary conditions on all boundaries.

This is the "basic" forward solver used by the diffusion-coefficient and
Dirichlet-phi0 inference workflows.  For Robin-BC problems (electrode
transfer coefficients) see :mod:`Forward.robin_solver`.

Public API
----------
build_context(solver_params) → dict
build_forms(ctx, solver_params) → dict
set_initial_conditions(ctx, solver_params, blob=True) → None
forsolve(ctx, solver_params, print_interval=100) → Function
"""

from __future__ import annotations

import numpy as np

import firedrake as fd
import firedrake.adjoint as adj

from Nondim.constants import FARADAY_CONSTANT, GAS_CONSTANT, DEFAULT_TEMPERATURE_K


def _as_species_list(values, n_species: int, name: str):
    """Normalize scalar-or-sequence into a per-species float list."""
    if np.isscalar(values):
        return [float(values) for _ in range(n_species)]
    try:
        vals = [float(v) for v in values]
    except TypeError as exc:
        raise ValueError(
            f"{name} must be a scalar or a sequence of length {n_species}"
        ) from exc
    if len(vals) != n_species:
        raise ValueError(
            f"{name} must have length n_species ({n_species}); got {len(vals)}"
        )
    return vals


def build_context(solver_params) -> dict:
    """Build mesh and function spaces for the Dirichlet PNP solver.

    Parameters
    ----------
    solver_params:
        11-entry list or :class:`Forward.params.SolverParams`.

    Returns
    -------
    dict
        Context with keys: mesh, V_scalar, W, U, U_prev, n_species.
    """
    try:
        n_species, order, dt, t_end, z_vals, D_vals, a_vals, phi_applied, c0, phi0, params = solver_params
    except Exception as exc:
        raise ValueError("forward_solver expects a list of 11 solver parameters") from exc

    if not (len(z_vals) == len(D_vals) == len(a_vals) == n_species):
        raise ValueError(
            f"z_vals, D_vals, and a_vals must all have length n_species ({n_species}); "
            f"got lengths {len(z_vals)}, {len(D_vals)}, {len(a_vals)}"
        )
    _as_species_list(c0, n_species, "c0")

    mesh = fd.UnitSquareMesh(32, 32)
    V_scalar = fd.FunctionSpace(mesh, "CG", order)
    W = fd.MixedFunctionSpace([V_scalar for _ in range(n_species)] + [V_scalar])
    U = fd.Function(W)
    U_prev = fd.Function(W)

    return {
        "mesh": mesh,
        "V_scalar": V_scalar,
        "W": W,
        "U": U,
        "U_prev": U_prev,
        "n_species": n_species,
    }


def build_forms(ctx: dict, solver_params) -> dict:
    """Assemble weak forms for the Dirichlet PNP problem.

    Note: The Poisson equation uses eps=1 (natural units).  This solver is
    intended for dimensionless or normalised model-space parameters.  For
    physical-units simulations with proper permittivity, use
    :mod:`Forward.robin_solver` with ``nondim.enabled=True``.

    Returns
    -------
    dict
        Updated context with keys: F_res, J_form, bcs, logD_funcs, D_consts,
        z_consts, phi0_func.
    """
    try:
        n_species, order, dt, t_end, z_vals, D_vals, a_vals, phi_applied, c0, phi0, params = solver_params
    except Exception as exc:
        raise ValueError("forward_solver expects a list of 11 solver parameters") from exc

    c0_vals = _as_species_list(c0, n_species, "c0")
    mesh = ctx["mesh"]
    W = ctx["W"]
    n = ctx["n_species"]

    F_over_RT = FARADAY_CONSTANT / (GAS_CONSTANT * DEFAULT_TEMPERATURE_K)

    R_space = fd.FunctionSpace(mesh, "R", 0)
    m = [fd.Function(R_space, name=f"logD{i}") for i in range(n)]
    for i in range(n):
        m[i].assign(np.log(float(D_vals[i])))
    D = [fd.exp(m[i]) for i in range(n)]

    z = [fd.Constant(int(z_vals[i])) for i in range(n)]

    U = ctx["U"]
    U_prev = ctx["U_prev"]
    ci = fd.split(U)[:-1]
    phi = fd.split(U)[-1]
    ci_prev = fd.split(U_prev)[:-1]
    v_tests = fd.TestFunctions(W)
    v_list = v_tests[:-1]
    w = v_tests[-1]

    # Steric chemical potential (Bikerman model): mu_steric = ln(1 - sum_j a_j c_j)
    # a_vals[i] is the dimensionless excluded-volume fraction per unit (model) concentration.
    # When all a_vals == 0 the steric term vanishes and the standard NP equation is recovered.
    a_vals_list = [float(v) for v in a_vals]
    steric_active = any(v != 0.0 for v in a_vals_list)
    if steric_active:
        a_consts = [fd.Constant(v) for v in a_vals_list]
        # Clamp packing fraction away from zero to avoid ln(0).
        packing = fd.max_value(
            fd.Constant(1.0) - sum(a_consts[j] * ci[j] for j in range(n)),
            fd.Constant(1e-8),
        )
        mu_steric = fd.ln(packing)

    F_res = 0
    for i in range(n):
        c = ci[i]
        c_old = ci_prev[i]
        v = v_list[i]
        drift = F_over_RT * z[i] * phi
        if steric_active:
            Jflux = D[i] * (fd.grad(c) + c * fd.grad(drift) + c * fd.grad(mu_steric))
        else:
            Jflux = D[i] * (fd.grad(c) + c * fd.grad(drift))
        F_res += ((c - c_old) / dt) * v * fd.dx + fd.dot(Jflux, fd.grad(v)) * fd.dx

    # Poisson with eps=1 (model-space / natural units).
    F_res += fd.Constant(1.0) * fd.dot(fd.grad(phi), fd.grad(w)) * fd.dx
    F_res -= sum(z[i] * FARADAY_CONSTANT * ci[i] * w for i in range(n)) * fd.dx

    phi0_func = fd.Function(R_space, name="phi0")
    phi0_func.assign(float(phi0))
    bc_phi = fd.DirichletBC(W.sub(n), phi0_func, 1)
    bc_ci = [fd.DirichletBC(W.sub(i), fd.Constant(c0_vals[i]), 3) for i in range(n)]
    bcs = bc_ci + [bc_phi]

    J_form = fd.derivative(F_res, U)
    ctx.update({
        "F_res": F_res,
        "J_form": J_form,
        "bcs": bcs,
        "logD_funcs": m,
        "D_consts": D,
        "z_consts": z,
        "phi0_func": phi0_func,
    })
    return ctx


def set_initial_conditions(ctx: dict, solver_params, blob: bool = True) -> None:
    """Set initial conditions on ``U_prev`` (and sync to ``U``).

    Parameters
    ----------
    blob:
        If True, initialise concentrations as bulk + Gaussian blob.
        If False, use uniform bulk concentrations.
    """
    try:
        n_species, order, dt, t_end, z_vals, D_vals, a_vals, phi_applied, c0, phi0, params = solver_params
    except Exception as exc:
        raise ValueError("forward_solver expects a list of 11 solver parameters") from exc

    c0_vals = _as_species_list(c0, n_species, "c0")
    mesh = ctx["mesh"]
    U_prev = ctx["U_prev"]
    n = ctx["n_species"]
    x, y = fd.SpatialCoordinate(mesh)

    if blob:
        A = fd.Constant(0.5)
        x0 = fd.Constant(0.5)
        y0 = fd.Constant(0.2)
        sigma = fd.Constant(0.08)
        gaussian_blob = A * fd.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        for i in range(n):
            U_prev.sub(i).interpolate(fd.Constant(c0_vals[i]) + gaussian_blob)
    else:
        for i in range(n):
            U_prev.sub(i).assign(fd.Constant(c0_vals[i]))

    U_prev.sub(n).assign(fd.Constant(0.0))
    ctx["U"].assign(U_prev)


def forsolve(ctx: dict, solver_params, print_interval: int = 100):
    """Run the Dirichlet PNP time-stepping loop.

    Returns
    -------
    Function
        Final solved state ``U``.
    """
    try:
        n_species, order, dt, t_end, z_vals, D_vals, a_vals, phi_applied, c0, phi0, params = solver_params
    except Exception as exc:
        raise ValueError("forward_solver expects a list of 11 solver parameters") from exc

    U = ctx["U"]
    U_prev = ctx["U_prev"]
    F_res = ctx["F_res"]
    bcs = ctx["bcs"]

    num_steps = int(t_end / dt)
    J = fd.derivative(F_res, U)
    problem = fd.NonlinearVariationalProblem(F_res, U, bcs=bcs, J=J)
    solver = fd.NonlinearVariationalSolver(problem, solver_parameters=params)

    for step in range(num_steps):
        if step % print_interval == 0:
            print("step", step)
        solver.solve()
        U_prev.assign(U)

    return U
