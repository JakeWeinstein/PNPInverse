import firedrake as fd
import firedrake.adjoint as adj
import numpy as np

def build_context(solver_params):
    try:
        (n_species, order, dt, t_end, z_vals, D_vals,
         a_vals, phi_applied, c0, phi0, params) = solver_params
    except Exception as exc:
        raise ValueError("forward_solver expects a list of 11 solver parameters") from exc

    if not (len(z_vals) == len(D_vals) == len(a_vals) == n_species):
        raise ValueError(
            f"z_vals, D_vals, and a_vals must all have length n_species ({n_species}); "
            f"got lengths {len(z_vals)}, {len(D_vals)}, {len(a_vals)}"
        )
    nx=32
    ny=32
    mesh = fd.UnitSquareMesh(nx, ny)
    V_scalar = fd.FunctionSpace(mesh, "CG", order)
    W = fd.MixedFunctionSpace([V_scalar for _ in range(n_species)] + [V_scalar])

    U = fd.Function(W)
    U_prev = fd.Function(W)

    return {"mesh": mesh, "V_scalar": V_scalar, "W": W, "U": U, "U_prev": U_prev, "n_species": n_species}

def build_forms(ctx,solver_params):
    try:
        (n_species, order, dt, t_end, z_vals, D_vals,
         a_vals, phi_applied, c0, phi0, params) = solver_params
    except Exception as exc:
        raise ValueError("forward_solver expects a list of 11 solver parameters") from exc

    mesh = ctx["mesh"]
    W = ctx["W"]
    n = ctx["n_species"]
    V_scalar = ctx["V_scalar"]

    F = 96485.3329
    R = 8.314462618
    T = 298.15
    F_over_RT = F/(R*T)

    #Define 0 degree function space for D to live in because adjoint will only track functions
    R = fd.FunctionSpace(mesh, "R", 0)
    m = [fd.Function(R, name=f"logD{i}") for i in range(n)]
    #These are the parmeters we're optimizing
    #Use log because we don't want negative D-vals, so this ensures the optimizer keeps D positive
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

    F_res = 0
    for i in range(n):
        c = ci[i]; c_old = ci_prev[i]; v = v_list[i]
        drift = F_over_RT * z[i] * phi
        Jflux = D[i]*(fd.grad(c) + c*fd.grad(drift))
        F_res += ((c - c_old)/dt)*v*fd.dx + fd.dot(Jflux, fd.grad(v))*fd.dx

    eps = fd.Constant(1.0)
    F_res += eps*fd.dot(fd.grad(phi), fd.grad(w))*fd.dx
    F_res -= sum(z[i]*F*ci[i]*w for i in range(n))*fd.dx

    bc_phi = fd.DirichletBC(W.sub(n), fd.Constant(phi0), 1)
    bc_ci  = [fd.DirichletBC(W.sub(i), fd.Constant(c0), 3) for i in range(n)]
    bcs = bc_ci + [bc_phi]

    J_form = fd.derivative(F_res, U)

    ctx.update({"F_res": F_res, "J_form": J_form, "bcs": bcs,"logD_funcs": m, "D_consts": D, "z_consts": z})
    return ctx

def set_initial_conditions(ctx, solver_params, blob=True):
    try:
        (n_species, order, dt, t_end, z_vals, D_vals,
         a_vals, phi_applied, c0, phi0, params) = solver_params
    except Exception as exc:
        raise ValueError("forward_solver expects a list of 11 solver parameters") from exc
    
    mesh = ctx["mesh"]
    U_prev = ctx["U_prev"]
    n = ctx["n_species"]

    x, y = fd.SpatialCoordinate(mesh)
    c_bulk = fd.Constant(c0)

    if blob:
        A = fd.Constant(0.5); x0 = fd.Constant(0.5); y0 = fd.Constant(0.2); sigma = fd.Constant(0.08)
        gaussian = c_bulk + A*fd.exp(-((x-x0)**2 + (y-y0)**2)/(2*sigma**2))
        for i in range(n):
            U_prev.sub(i).interpolate(gaussian)
    else:
        for i in range(n):
            U_prev.sub(i).assign(c_bulk)

    U_prev.sub(n).assign(fd.Constant(0.0))
    ctx["U"].assign(U_prev)

def forsolve(ctx, solver_params, print_interval=100):
    from firedrake.adjoint import get_working_tape

    try:
        (n_species, order, dt, t_end, z_vals, D_vals,
         a_vals, phi_applied, c0, phi0, params) = solver_params
    except Exception as exc:
        raise ValueError("forward_solver expects a list of 11 solver parameters") from exc
    
    U = ctx["U"]
    U_prev = ctx["U_prev"]
    F_res = ctx["F_res"]
    J_form = ctx["J_form"]
    bcs = ctx["bcs"]

    num_steps = int(t_end/dt)

    J = fd.derivative(F_res, U)

    problem = fd.NonlinearVariationalProblem(F_res, U, bcs=bcs, J=J)
    # firedrake-adjoint hooks the standard solver when annotation is on
    solver = fd.NonlinearVariationalSolver(problem, solver_parameters=params)

    for step in range(num_steps):
        if step % print_interval == 0:
            print("step", step)
        solver.solve()
        # carry the solution forward in time
        U_prev.assign(U)

    # Return the solved state (same data as U_prev after last assign) so
    # the objective directly depends on the solver output recorded on the tape.
    return U
