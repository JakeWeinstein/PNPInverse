import numpy as np
import firedrake as fd
import firedrake.adjoint as adj

from Utils.robin_forsolve import (
    build_context,
    build_forms,
    forsolve_robin,
    set_initial_conditions,
)
from Utils.generate_noisy_data_robin import generate_noisy_data_robin


def m_to_string(m):
    outstr = ""
    for mval in m:
        outstr += str(mval.dat.data) + ", "
    return outstr[:-2]


def m_to_d_to_string(m):
    outstr = ""
    for mval in m:
        outstr += str(np.exp(mval.dat.data)) + ", "
    return outstr[:-2]


def vec_to_function(ctx, vec, *, space_key="V_scalar"):
    V = ctx[space_key]
    f = fd.Function(V)
    v = np.asarray(vec, dtype=float).ravel()
    if v.size != f.dat.data.size:
        raise ValueError(
            f"Target vector length {v.size} != DOFs {f.dat.data.size} for {space_key}"
        )
    f.dat.data[:] = v
    return f


def make_objective_and_grad_robin(solver_params, c_targets, blob_ic=True):
    ctx = build_context(solver_params)
    ctx = build_forms(ctx, solver_params)

    n_species = solver_params[0]
    if len(c_targets) != n_species:
        raise ValueError(f"Expected {n_species} target vectors, got {len(c_targets)}")

    logD_funcs = ctx["logD_funcs"]
    c_target_fs = [vec_to_function(ctx, c_target) for c_target in c_targets]

    tape = adj.get_working_tape()
    tape.clear_tape()
    adj.continue_annotation()

    set_initial_conditions(ctx, solver_params, blob=blob_ic)
    U_final = forsolve_robin(ctx, solver_params)

    Jobj = 0.5 * fd.assemble(
        sum(
            fd.inner(U_final.sub(i) - c_target_fs[i], U_final.sub(i) - c_target_fs[i])
            for i in range(n_species)
        )
        * fd.dx
    )

    def eval_cb(j, m):
        print(f"j = {j}, m = {m_to_string(m)}, D = {m_to_d_to_string(m)}")

    return adj.ReducedFunctional(
        Jobj,
        [adj.Control(logd_func) for logd_func in logD_funcs],
        eval_cb_post=eval_cb,
    )


params = {
    "snes_type": "newtonls",
    "snes_max_it": 100,
    "snes_atol": 1e-8,
    "snes_rtol": 1e-8,
    "snes_linesearch_type": "bt",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "robin_bc": {
        "kappa": [1.0, 1.0],
        "c_inf": [0.01, 0.01],
        "electrode_marker": 1,
        "concentration_marker": 3,
        "ground_marker": 3,
    },
}

D_true = [1.0, 1.0]
n_species = len(D_true)
z_vals = [1 if i % 2 == 0 else -1 for i in range(n_species)]
a_vals = [0.0 for _ in range(n_species)]
c0_vals = [0.1 for _ in range(n_species)]

#                   (n_species, order, dt, t_end, z_vals, D_vals, a_vals, phi_applied, c0_vals, phi0, params)
solver_params_gen = [n_species, 1, 1e-3, 0.1, z_vals, D_true, a_vals, 0.05, c0_vals, 5, params]

with adj.stop_annotating():
    data = generate_noisy_data_robin(solver_params_gen)

clean_c = list(data[:n_species])
phi_vec = data[n_species]

noisy_c = list(data[n_species + 1 : n_species * 2 + 1])
phi_noisy = data[-1]

theta0 = [10.0 for _ in range(n_species)]
solver_params = [n_species, 1, 1e-2, 0.1, z_vals, theta0, a_vals, 0.05, c0_vals, 5, params]

rf = make_objective_and_grad_robin(solver_params, clean_c)

m_vals = adj.minimize(rf, "BFGS", tol=1e-8, options={"disp": True})

mlst = [v.dat.data for _, v in enumerate(m_vals)]

# Reexponentiate because we solved for log(D)
dlst = np.exp(mlst)

print("D =", ", ".join(str(d) for d in dlst))
