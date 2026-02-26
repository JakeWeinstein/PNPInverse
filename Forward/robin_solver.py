"""Robin-BC PNP forward solver with full nondimensionalization support.

This solver is used for Robin-kappa inference workflows where the electrode
boundary condition is::

    J_i · n  =  κ_i · (c_i − c∞_i)     at the electrode boundary

Nondimensionalization is handled by :func:`Nondim.transform.build_model_scaling`,
which is called inside :func:`build_forms`.  See that module for a full
derivation of the nondimensional PDE form and the physical interpretation of
each coefficient.

Public API
----------
build_context(solver_params) → dict
build_forms(ctx, solver_params) → dict
set_initial_conditions(ctx, solver_params, blob=False) → None
forsolve(ctx, solver_params, print_interval=100) → Function
forsolve_robin(ctx, solver_params, print_interval=100) → Function  (alias)
"""

from __future__ import annotations

import numpy as np

import firedrake as fd

from Nondim.transform import build_model_scaling, _get_robin_cfg


def build_context(solver_params) -> dict:
    """Build mesh and function spaces for the Robin PNP solver."""
    try:
        n_species, order, dt, t_end, z_vals, D_vals, a_vals, phi_applied, c0, phi0, params = solver_params
    except Exception as exc:
        raise ValueError("forward_solver expects a list of 11 solver parameters") from exc

    if not (len(z_vals) == len(D_vals) == len(a_vals) == n_species):
        raise ValueError(
            f"z_vals, D_vals, and a_vals must all have length n_species ({n_species}); "
            f"got lengths {len(z_vals)}, {len(D_vals)}, {len(a_vals)}"
        )

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
    """Assemble weak forms for the Robin PNP problem.

    Calls :func:`Nondim.transform.build_model_scaling` to convert physical
    inputs to model-space values, then assembles:

    * Nernst-Planck residuals with drift coefficient ``electromigration_prefactor``
    * Robin BC flux term ``κ_i · (c_i − c∞_i)`` at the electrode boundary
    * Poisson equation with coefficient ``poisson_coefficient``

    Returns
    -------
    dict
        Updated context. Key additions:
        F_res, J_form, bcs, logD_funcs, kappa_funcs, D_consts, z_consts,
        phi_applied_func, phi0_func, robin_settings, robin_settings_physical, nondim.
    """
    try:
        n_species, order, dt, t_end, z_vals, D_vals, a_vals, phi_applied, c0, phi0, params = solver_params
    except Exception as exc:
        raise ValueError("forward_solver expects a list of 11 solver parameters") from exc

    c0_raw = [float(v) for v in (
        [c0] * n_species if np.isscalar(c0) else list(c0)
    )][:n_species]
    mesh = ctx["mesh"]
    W = ctx["W"]
    n = ctx["n_species"]

    # Parse Robin settings (physical units) before scaling.
    robin_raw = _get_robin_cfg(params, n)

    # Apply nondimensionalization (or identity if disabled).
    scaling = build_model_scaling(
        params=params,
        n_species=n,
        dt=dt,
        t_end=t_end,
        D_vals=D_vals,
        c0_vals=c0_raw,
        phi_applied=phi_applied,
        phi0=phi0,
        robin=robin_raw,
    )

    # Model-space Robin settings (kappa and c_inf already scaled).
    robin_model = dict(robin_raw)
    robin_model["kappa_vals"] = [float(v) for v in scaling["kappa_model_vals"]]
    robin_model["c_inf_vals"] = [float(v) for v in scaling["c_inf_model_vals"]]

    electrode_marker = int(robin_model["electrode_marker"])
    concentration_marker = int(robin_model["concentration_marker"])
    ground_marker = int(robin_model["ground_marker"])
    c_inf_consts = [fd.Constant(v) for v in robin_model["c_inf_vals"]]
    ds = fd.Measure("ds", domain=mesh)

    # Scalar R-space for controls (adjoint tracks these).
    R_space = fd.FunctionSpace(mesh, "R", 0)

    # Log-diffusivity controls (log-space ensures positivity during optimisation).
    m = [fd.Function(R_space, name=f"logD{i}") for i in range(n)]
    for i in range(n):
        m[i].assign(np.log(float(scaling["D_model_vals"][i])))
    D = [fd.exp(m[i]) for i in range(n)]

    # Robin transfer coefficient controls.
    kappa_funcs = [fd.Function(R_space, name=f"kappa{i}") for i in range(n)]
    for i in range(n):
        kappa_funcs[i].assign(float(robin_model["kappa_vals"][i]))

    z = [fd.Constant(int(z_vals[i])) for i in range(n)]

    U = ctx["U"]
    U_prev = ctx["U_prev"]
    ci = fd.split(U)[:-1]
    phi = fd.split(U)[-1]
    ci_prev = fd.split(U_prev)[:-1]
    v_tests = fd.TestFunctions(W)
    v_list = v_tests[:-1]
    w = v_tests[-1]

    em = float(scaling["electromigration_prefactor"])
    dt_m = float(scaling["dt_model"])

    # Steric chemical potential (Bikerman model): mu_steric = ln(1 - sum_j a_j c_j)
    # In nondim mode, a_vals[i] should be supplied as the dimensionless excluded-volume
    # fraction per unit model concentration (a_phys * c_scale). When a_vals == 0 the
    # standard NP equation is recovered.
    a_vals_list = [float(v) for v in a_vals]
    steric_active = any(v != 0.0 for v in a_vals_list)
    if steric_active:
        a_consts = [fd.Constant(v) for v in a_vals_list]
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
        drift = em * z[i] * phi
        if steric_active:
            Jflux = D[i] * (fd.grad(c) + c * fd.grad(drift) + c * fd.grad(mu_steric))
        else:
            Jflux = D[i] * (fd.grad(c) + c * fd.grad(drift))
        F_res += ((c - c_old) / dt_m) * v * fd.dx + fd.dot(Jflux, fd.grad(v)) * fd.dx
        # Robin BC: J·n = κ(c - c∞)
        F_res += kappa_funcs[i] * (c - c_inf_consts[i]) * v * ds(electrode_marker)

    eps = fd.Constant(float(scaling["poisson_coefficient"]))
    F_res += eps * fd.dot(fd.grad(phi), fd.grad(w)) * fd.dx
    F_res -= (
        fd.Constant(float(scaling["charge_rhs_prefactor"]))
        * sum(z[i] * ci[i] * w for i in range(n))
        * fd.dx
    )

    phi_applied_func = fd.Function(R_space, name="phi_applied")
    phi_applied_func.assign(float(scaling["phi_applied_model"]))
    phi0_func = fd.Function(R_space, name="phi0")
    phi0_func.assign(float(scaling["phi0_model"]))

    bc_phi_electrode = fd.DirichletBC(W.sub(n), phi_applied_func, electrode_marker)
    bc_phi_ground = fd.DirichletBC(W.sub(n), fd.Constant(0.0), ground_marker)
    bc_ci = [
        fd.DirichletBC(
            W.sub(i), fd.Constant(float(scaling["c0_model_vals"][i])), concentration_marker
        )
        for i in range(n)
    ]
    bcs = bc_ci + [bc_phi_electrode, bc_phi_ground]

    J_form = fd.derivative(F_res, U)
    ctx.update({
        "F_res": F_res,
        "J_form": J_form,
        "bcs": bcs,
        "logD_funcs": m,
        "kappa_funcs": kappa_funcs,
        "D_consts": D,
        "z_consts": z,
        "phi_applied_func": phi_applied_func,
        "phi0_func": phi0_func,
        "robin_settings": robin_model,
        "robin_settings_physical": robin_raw,
        "nondim": scaling,
    })
    return ctx


def set_initial_conditions(ctx: dict, solver_params, blob: bool = True) -> None:
    """Set initial conditions: uniform concentrations + linear potential profile.

    The default ``blob=False`` (uniform IC) is closer to steady state for
    Robin-BC problems than the Gaussian-blob IC used by the Dirichlet solver.

    Parameters
    ----------
    blob:
        If True, add a Gaussian blob to the initial concentration field.
        If False (default for Robin), use uniform bulk concentrations only.
    """
    try:
        n_species, order, dt, t_end, z_vals, D_vals, a_vals, phi_applied, c0, phi0, params = solver_params
    except Exception as exc:
        raise ValueError("forward_solver expects a list of 11 solver parameters") from exc

    mesh = ctx["mesh"]
    U_prev = ctx["U_prev"]
    n = ctx["n_species"]
    scaling = ctx.get("nondim", {})

    c0_raw = [float(v) for v in ([c0] * n if np.isscalar(c0) else list(c0))][:n]
    c0_model = scaling.get("c0_model_vals", c0_raw)
    phi_applied_model = scaling.get("phi_applied_model", float(phi_applied))

    x, y = fd.SpatialCoordinate(mesh)

    if blob:
        A = fd.Constant(0.5)
        x0 = fd.Constant(0.5)
        y0 = fd.Constant(0.2)
        sigma = fd.Constant(0.08)
        gaussian_blob = A * fd.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        for i in range(n):
            U_prev.sub(i).interpolate(fd.Constant(float(c0_model[i])) + gaussian_blob)
    else:
        for i in range(n):
            U_prev.sub(i).assign(fd.Constant(float(c0_model[i])))

    # Linear potential profile: phi = phi_applied * (1 - y)
    U_prev.sub(n).interpolate(fd.Constant(float(phi_applied_model)) * (1.0 - y))
    ctx["U"].assign(U_prev)


def forsolve(ctx: dict, solver_params, print_interval: int = 100):
    """Run the Robin PNP time-stepping loop.

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
    scaling = ctx.get("nondim", {})

    dt_model = float(scaling.get("dt_model", dt))
    t_end_model = float(scaling.get("t_end_model", t_end))
    num_steps = int(t_end_model / dt_model)

    J = fd.derivative(F_res, U)
    problem = fd.NonlinearVariationalProblem(F_res, U, bcs=bcs, J=J)
    solver = fd.NonlinearVariationalSolver(problem, solver_parameters=params)

    for step in range(num_steps):
        if step % print_interval == 0:
            print("step", step)
        solver.solve()
        U_prev.assign(U)

    return U


def forsolve_robin(ctx: dict, solver_params, print_interval: int = 100):
    """Alias for :func:`forsolve` with an explicit Robin name for clarity."""
    return forsolve(ctx, solver_params, print_interval=print_interval)
