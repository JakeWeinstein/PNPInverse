import numpy as np
import firedrake as fd
import firedrake.adjoint as adj

from Utils.robin_forsolve import build_context, build_forms, forsolve_robin, set_initial_conditions


def _controls_to_string(m):
    if not isinstance(m, (list, tuple)):
        m = [m]
    parts = []
    for i, control in enumerate(m):
        parts.append(f"kappa{i}={control.dat.data[0]}")
    return ", ".join(parts)


def vec_to_function(ctx, vec, *, space_key="V_scalar"):
    """
    Build a Firedrake Function on ctx[space_key] with DOF coefficients from vec.
    vec must have length == ctx[space_key].dim().
    """
    V = ctx[space_key]
    f = fd.Function(V)
    v = np.asarray(vec, dtype=float).ravel()

    if v.size != f.dat.data.size:
        raise ValueError(
            f"Target vector length {v.size} != DOFs {f.dat.data.size} for {space_key}"
        )

    f.dat.data[:] = v
    return f


def make_objective_and_grad(solver_params, c_targets, phi_vec, blob_ic=True):
    """
    Build a reduced functional to infer Robin kappa values from concentration
    and electric-potential data.

    Parameters
    ----------
    solver_params : list
        Forward solver parameters:
        [n_species, order, dt, t_end, z_vals, D_vals, a_vals, phi_applied, c0_vals, phi0, params]
    c_targets : sequence
        Sequence of target concentration vectors, one per species.
    phi_vec : array-like
        Target electric-potential coefficient vector.
    blob_ic : bool, optional
        Whether to use blob initial condition.
    """
    ctx = build_context(solver_params)
    ctx = build_forms(ctx, solver_params)

    n_species = solver_params[0]
    if len(c_targets) != n_species:
        raise ValueError(f"Expected {n_species} target vectors, got {len(c_targets)}")

    kappa_funcs = ctx["kappa_funcs"]
    c_target_fs = [vec_to_function(ctx, c_target) for c_target in c_targets]
    phi_target = vec_to_function(ctx, phi_vec)

    tape = adj.get_working_tape()
    tape.clear_tape()
    adj.continue_annotation()

    set_initial_conditions(ctx, solver_params, blob=blob_ic)
    U_final = forsolve_robin(ctx, solver_params)

    Jobj = 0.5 * fd.assemble(
        (
            sum(
                fd.inner(U_final.sub(i) - c_target_fs[i], U_final.sub(i) - c_target_fs[i])
                for i in range(n_species)
            )
            + fd.inner(U_final.sub(n_species) - phi_target, U_final.sub(n_species) - phi_target)
        )
        * fd.dx
    )

    def eval_cb_post(j, m):
        print(f"j = {j}, {_controls_to_string(m)}")

    rf = adj.ReducedFunctional(
        Jobj,
        [adj.Control(kappa_func) for kappa_func in kappa_funcs],
        eval_cb_post=eval_cb_post,
    )
    return rf
