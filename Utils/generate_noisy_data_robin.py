import numpy as np

from Utils.robin_forsolve import build_context, build_forms, forsolve_robin, set_initial_conditions


def _add_percent_noise(vec, noise_percent, rng):
    """Add zero-mean Gaussian noise with sigma set as a percentage of RMS(vec)."""
    v = np.asarray(vec, dtype=float)
    pct = float(noise_percent)
    if pct < 0:
        raise ValueError(f"noise_percent must be non-negative; got {pct}")
    if pct == 0:
        return v.copy()

    rms = float(np.sqrt(np.mean(v * v)))
    sigma = (pct / 100.0) * max(rms, 1e-12)
    return v + rng.normal(0.0, sigma, size=v.shape)


def generate_noisy_data_robin(
    solver_params,
    noise_percent=10.0,
    seed=None,
    print_interval=100,
    noise_std=None,
):
    """
    Run the Robin-BC PNP forward solver and return final-step clean/noisy vectors.

    Parameters
    ----------
    solver_params : list
        [n_species, order, dt, t_end, z_vals, D_vals,
         a_vals, phi_applied, c0_vals, phi0, params]
    noise_percent : float, optional
        Percent noise level. Gaussian noise uses sigma =
        (noise_percent / 100) * RMS(field).
    noise_std : float, optional
        Deprecated compatibility alias for noise_percent.
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
    if noise_std is not None:
        noise_percent = float(noise_std)

    rng = np.random.default_rng(seed)
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

    ctx = build_context(solver_params)
    ctx = build_forms(ctx, solver_params)
    set_initial_conditions(ctx, solver_params, blob=True)

    U_final = forsolve_robin(ctx, solver_params, print_interval=print_interval)

    phi_idx = n_species
    c_vecs = [np.array(U_final.sub(i).dat.data_ro) for i in range(n_species)]
    phi_vec = np.array(U_final.sub(phi_idx).dat.data_ro)

    c_noisy = [_add_percent_noise(c_vec, noise_percent, rng) for c_vec in c_vecs]
    phi_noisy = _add_percent_noise(phi_vec, noise_percent, rng)

    return tuple(c_vecs + [phi_vec] + c_noisy + [phi_noisy])
