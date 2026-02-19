import firedrake as fd
import firedrake.adjoint as adj
import numpy as np


def _as_species_list(values, n_species, name):
    """Normalize scalar-or-sequence parameter into a per-species float list."""
    if np.isscalar(values):
        return [float(values) for _ in range(n_species)]
    try:
        vals = [float(v) for v in values]
    except TypeError as exc:
        raise ValueError(f"{name} must be a scalar or a sequence of length {n_species}") from exc
    if len(vals) != n_species:
        raise ValueError(f"{name} must have length n_species ({n_species}); got {len(vals)}")
    return vals


def _to_float(value, name):
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be float-like; got {value!r}") from exc


def _to_int(value, name):
    try:
        return int(value)
    except Exception as exc:
        raise ValueError(f"{name} must be int-like; got {value!r}") from exc


def _extract_robin_settings(params, n_species):
    robin_cfg = {}
    if isinstance(params, dict):
        maybe_robin_cfg = params.get("robin_bc", {})
        if isinstance(maybe_robin_cfg, dict):
            robin_cfg = maybe_robin_cfg

        kappa = robin_cfg.get("kappa", params.get("robin_kappa", 1.0))
        c_inf = robin_cfg.get("c_inf", params.get("robin_c_inf", 0.01))
        electrode_marker = robin_cfg.get(
            "electrode_marker", params.get("robin_electrode_marker", 1)
        )
        concentration_marker = robin_cfg.get(
            "concentration_marker", params.get("robin_concentration_marker", 3)
        )
        ground_marker = robin_cfg.get("ground_marker", params.get("robin_ground_marker", 3))
    else:
        kappa = 1.0
        c_inf = 0.01
        electrode_marker = 1
        concentration_marker = 3
        ground_marker = 3

    return {
        "kappa_vals": _as_species_list(kappa, n_species, "robin_kappa"),
        "c_inf_vals": _as_species_list(c_inf, n_species, "robin_c_inf"),
        "electrode_marker": _to_int(electrode_marker, "robin_electrode_marker"),
        "concentration_marker": _to_int(concentration_marker, "robin_concentration_marker"),
        "ground_marker": _to_int(ground_marker, "robin_ground_marker"),
    }


def build_context(solver_params):
    try:
        (
            n_species,
            order,
            dt,
            t_end,
            z_vals,
            D_vals,
            a_vals,
            phi_applied,
            c0,
            phi0,
            params,
        ) = solver_params
    except Exception as exc:
        raise ValueError("forward_solver expects a list of 11 solver parameters") from exc

    if not (len(z_vals) == len(D_vals) == len(a_vals) == n_species):
        raise ValueError(
            f"z_vals, D_vals, and a_vals must all have length n_species ({n_species}); "
            f"got lengths {len(z_vals)}, {len(D_vals)}, {len(a_vals)}"
        )
    _ = _as_species_list(c0, n_species, "c0")
    nx = 32
    ny = 32
    mesh = fd.UnitSquareMesh(nx, ny)
    V_scalar = fd.FunctionSpace(mesh, "CG", order)
    W = fd.MixedFunctionSpace([V_scalar for _ in range(n_species)] + [V_scalar])

    U = fd.Function(W)
    U_prev = fd.Function(W)

    return {"mesh": mesh, "V_scalar": V_scalar, "W": W, "U": U, "U_prev": U_prev, "n_species": n_species}


def build_forms(ctx, solver_params):
    try:
        (
            n_species,
            order,
            dt,
            t_end,
            z_vals,
            D_vals,
            a_vals,
            phi_applied,
            c0,
            phi0,
            params,
        ) = solver_params
    except Exception as exc:
        raise ValueError("forward_solver expects a list of 11 solver parameters") from exc

    c0_vals = _as_species_list(c0, n_species, "c0")

    mesh = ctx["mesh"]
    W = ctx["W"]
    n = ctx["n_species"]

    robin = _extract_robin_settings(params, n)
    c_inf_consts = [fd.Constant(v) for v in robin["c_inf_vals"]]
    electrode_marker = robin["electrode_marker"]
    concentration_marker = robin["concentration_marker"]
    ground_marker = robin["ground_marker"]
    ds = fd.Measure("ds", domain=mesh)

    F = 96485.3329
    gas_constant = 8.314462618
    temperature = 298.15
    F_over_RT = F / (gas_constant * temperature)

    # Keep scalar controls in an R-space so they can be optimized by firedrake-adjoint.
    R_space = fd.FunctionSpace(mesh, "R", 0)
    m = [fd.Function(R_space, name=f"logD{i}") for i in range(n)]
    # These are the parameters we're optimizing.
    # Use log because we don't want negative D-vals, so this ensures the optimizer keeps D positive.
    for i in range(n):
        m[i].assign(np.log(float(D_vals[i])))

    D = [fd.exp(m[i]) for i in range(n)]
    kappa_funcs = [fd.Function(R_space, name=f"kappa{i}") for i in range(n)]
    for i in range(n):
        kappa_funcs[i].assign(float(robin["kappa_vals"][i]))

    z = [fd.Constant(int(z_vals[i])) for i in range(n)]

    U = ctx["U"]
    U_prev = ctx["U_prev"]

    ci = fd.split(U)[:-1]
    phi = fd.split(U)[-1]
    ci_prev = fd.split(U_prev)[:-1]

    v_tests = fd.TestFunctions(W)
    v_list = v_tests[:-1]
    w = v_tests[-1]

    F_res = 0
    for i in range(n):
        c = ci[i]
        c_old = ci_prev[i]
        v = v_list[i]
        drift = F_over_RT * z[i] * phi
        Jflux = D[i] * (fd.grad(c) + c * fd.grad(drift))
        F_res += ((c - c_old) / dt) * v * fd.dx + fd.dot(Jflux, fd.grad(v)) * fd.dx

        # Robin BC from pnp_solver.py: J.n = kappa * (c - c_inf) at electrode boundary.
        F_res += kappa_funcs[i] * (c - c_inf_consts[i]) * v * ds(electrode_marker)

    eps = fd.Constant(1.0)
    F_res += eps * fd.dot(fd.grad(phi), fd.grad(w)) * fd.dx
    F_res -= sum(z[i] * F * ci[i] * w for i in range(n)) * fd.dx

    phi_applied_func = fd.Function(R_space, name="phi_applied")
    phi_applied_func.assign(_to_float(phi_applied, "phi_applied"))

    # Keep this key for interface compatibility with callers that expect ctx["phi0_func"].
    phi0_func = fd.Function(R_space, name="phi0")
    phi0_func.assign(_to_float(phi0, "phi0"))

    bc_phi_electrode = fd.DirichletBC(W.sub(n), phi_applied_func, electrode_marker)
    bc_phi_ground = fd.DirichletBC(W.sub(n), fd.Constant(0.0), ground_marker)
    bc_ci = [fd.DirichletBC(W.sub(i), fd.Constant(c0_vals[i]), concentration_marker) for i in range(n)]
    bcs = bc_ci + [bc_phi_electrode, bc_phi_ground]

    J_form = fd.derivative(F_res, U)

    ctx.update(
        {
            "F_res": F_res,
            "J_form": J_form,
            "bcs": bcs,
            "logD_funcs": m,
            "kappa_funcs": kappa_funcs,
            "D_consts": D,
            "z_consts": z,
            "phi_applied_func": phi_applied_func,
            "phi0_func": phi0_func,
            "robin_settings": robin,
        }
    )
    return ctx


def set_initial_conditions(ctx, solver_params, blob=True):
    try:
        (
            n_species,
            order,
            dt,
            t_end,
            z_vals,
            D_vals,
            a_vals,
            phi_applied,
            c0,
            phi0,
            params,
        ) = solver_params
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
            c_bulk_i = fd.Constant(c0_vals[i])
            U_prev.sub(i).interpolate(c_bulk_i + gaussian_blob)
    else:
        for i in range(n):
            U_prev.sub(i).assign(fd.Constant(c0_vals[i]))

    # Match pnp_solver.py Robin/BV initial condition style: linear potential profile.
    phi_applied_const = fd.Constant(_to_float(phi_applied, "phi_applied"))
    phi_init = phi_applied_const * (1.0 - y)
    U_prev.sub(n).interpolate(phi_init)

    ctx["U"].assign(U_prev)


def forsolve(ctx, solver_params, print_interval=100):
    from firedrake.adjoint import get_working_tape

    try:
        (
            n_species,
            order,
            dt,
            t_end,
            z_vals,
            D_vals,
            a_vals,
            phi_applied,
            c0,
            phi0,
            params,
        ) = solver_params
    except Exception as exc:
        raise ValueError("forward_solver expects a list of 11 solver parameters") from exc

    U = ctx["U"]
    U_prev = ctx["U_prev"]
    F_res = ctx["F_res"]
    J_form = ctx["J_form"]
    bcs = ctx["bcs"]

    num_steps = int(t_end / dt)

    J = fd.derivative(F_res, U)

    problem = fd.NonlinearVariationalProblem(F_res, U, bcs=bcs, J=J)
    # firedrake-adjoint hooks the standard solver when annotation is on.
    solver = fd.NonlinearVariationalSolver(problem, solver_parameters=params)

    for step in range(num_steps):
        if step % print_interval == 0:
            print("step", step)
        solver.solve()
        U_prev.assign(U)

    # Return the solved state (same data as U_prev after last assign) so
    # the objective directly depends on the solver output recorded on the tape.
    return U


def forsolve_robin(ctx, solver_params, print_interval=100):
    """Robin forward solve alias with explicit name for Robin inference scripts."""
    return forsolve(ctx, solver_params, print_interval=print_interval)
