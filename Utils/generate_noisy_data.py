import numpy as np
from Utils.forsolve import *


def generate_noisy_data(solver_params, noise_std=0.01, seed=None, print_interval=100):
    """
    Run the PNP forward solver and return final-step clean and noisy
    vectors for each species and phi.

    Parameters
    ----------
    solver_params : list
        [n_species, order, dt, t_end, z_vals, D_vals,
         a_vals, phi_applied, c0, phi0, params]
    noise_std : float, optional
        Standard deviation of additive Gaussian noise applied to each vector.
    seed : int, optional
        Seed for reproducible noise generation.
    print_interval : int, optional
        How often to print progress inside the forward solver.

    Returns
    -------
    tuple
        2*n_species + 2 arrays in order:
        (clean c_i for i=0..n-1, phi, noisy c_i for i=0..n-1, noisy phi)
    """
    rng = np.random.default_rng(seed)
    try:
        (n_species, order, dt, t_end, z_vals, D_vals,
         a_vals, phi_applied, c0, phi0, params) = solver_params
    except Exception as exc:
        raise ValueError("forward_solver expects a list of 11 solver parameters") from exc
    
    ctx = build_context(solver_params)
    ctx = build_forms(ctx, solver_params)
    set_initial_conditions(ctx, solver_params, blob=True)

    U_prev = forsolve(ctx, solver_params)

    phi_idx = n_species
    c_vecs = [np.array(U_prev.sub(i).dat.data_ro) for i in range(n_species)]
    phi_vec = np.array(U_prev.sub(phi_idx).dat.data_ro)

    c_noisy = [c_vec + rng.normal(0.0, noise_std, size=c_vec.shape) for c_vec in c_vecs]
    phi_noisy = phi_vec + rng.normal(0.0, noise_std, size=phi_vec.shape)

    return tuple(c_vecs + [phi_vec] + c_noisy + [phi_noisy])
