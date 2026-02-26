import numpy as np
import firedrake as fd
import firedrake.adjoint as adj
from Utils.forsolve import *

def vec_to_function(ctx, vec, *, space_key="V_scalar"):
    """
    Build a Firedrake Function on ctx[space_key] with DOF coefficients from vec.
    vec must have length == ctx[space_key].dim().

    Needed to allow arbitrary data in objective function
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

def make_objective_and_grad(solver_params, c_targets,phi_vec, blob_ic=True):
    """
    Build the reduced functional for an arbitrary number of species.

    Parameters
    ----------
    solver_params : list
        Forward solver parameters (see generate_noisy_data).
    c_targets : sequence
        Sequence of target concentration vectors, one per species.
    blob_ic : bool, optional
        Whether to use blob initial condition.
    """

    ctx = build_context(solver_params)
    ctx = build_forms(ctx, solver_params)

    n_species = solver_params[0]
    if len(c_targets) != n_species:
        raise ValueError(f"Expected {n_species} target vectors, got {len(c_targets)}")

    phi0_func = ctx["phi0_func"]
    phi_target = vec_to_function(ctx, phi_vec)
    # Start from a clean tape
    tape = adj.get_working_tape()
    tape.clear_tape()
    adj.continue_annotation()

    set_initial_conditions(ctx, solver_params, blob=blob_ic)

    U_final = forsolve(ctx, solver_params)

    # L2 mismatch in electric potential
    Jobj = 0.5 * fd.assemble(
            fd.inner(U_final.sub(n_species) - phi_target, U_final.sub(n_species) - phi_target)
        * fd.dx
    )

    def eval_cb_pre(m):
        if isinstance(m, (list, tuple)):
            m = m[0]
        phi0_func.assign(m)

    # Called whenever adjoint is evaluated
    def eval_cb(j, m):
        print(f"j = {j}, phi0 = {m.dat.data}")

    rf = adj.ReducedFunctional(
        Jobj, 
        adj.Control(phi0_func),
        eval_cb_pre = eval_cb_pre,
        eval_cb_post = eval_cb
    )
    
    return rf
