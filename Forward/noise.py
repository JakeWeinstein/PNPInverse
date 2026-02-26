"""Noise injection utilities for Dirichlet and Robin PNP forward solvers.

Merged from the original ``Utils/generate_noisy_data.py`` and
``Utils/generate_noisy_data_robin.py``.  Both functions share identical noise
logic; only the underlying forward solver differs.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from Forward.dirichlet_solver import (
    build_context as _d_build_context,
    build_forms as _d_build_forms,
    set_initial_conditions as _d_set_initial_conditions,
    forsolve as _d_forsolve,
)
from Forward.robin_solver import (
    build_context as _r_build_context,
    build_forms as _r_build_forms,
    set_initial_conditions as _r_set_initial_conditions,
    forsolve_robin as _r_forsolve,
)


def _add_percent_noise(
    vec: Sequence[float], noise_percent: float, rng: np.random.Generator
) -> np.ndarray:
    """Add zero-mean Gaussian noise with sigma = (noise_percent/100) * RMS(vec)."""
    v = np.asarray(vec, dtype=float)
    pct = float(noise_percent)
    if pct < 0:
        raise ValueError(f"noise_percent must be non-negative; got {pct}")
    if pct == 0:
        return v.copy()
    rms = float(np.sqrt(np.mean(v * v)))
    sigma = (pct / 100.0) * max(rms, 1e-12)
    return v + rng.normal(0.0, sigma, size=v.shape)


def generate_noisy_data(
    solver_params: Sequence[Any],
    noise_percent: float = 10.0,
    seed: Optional[int] = None,
    print_interval: int = 100,
    noise_std: Optional[float] = None,
) -> Tuple:
    """Run the Dirichlet PNP forward solver and return clean/noisy field vectors.

    Parameters
    ----------
    solver_params:
        Standard 11-entry solver parameter list.
    noise_percent:
        Gaussian noise level as percent of RMS(field).
    seed:
        RNG seed for reproducible noise.
    print_interval:
        Forward solver progress print frequency.
    noise_std:
        Deprecated alias for ``noise_percent``; kept for backward compat.

    Returns
    -------
    tuple
        ``2*n_species + 2`` arrays in order:
        ``(clean c_0, ..., clean c_{n-1}, clean phi,
          noisy c_0, ..., noisy c_{n-1}, noisy phi)``
    """
    if noise_std is not None:
        noise_percent = float(noise_std)

    rng = np.random.default_rng(seed)
    try:
        (n_species, order, dt, t_end, z_vals, D_vals,
         a_vals, phi_applied, c0, phi0, params) = solver_params
    except Exception as exc:
        raise ValueError("generate_noisy_data expects a list of 11 solver parameters") from exc

    ctx = _d_build_context(solver_params)
    ctx = _d_build_forms(ctx, solver_params)
    _d_set_initial_conditions(ctx, solver_params, blob=True)
    U_final = _d_forsolve(ctx, solver_params, print_interval=print_interval)

    c_vecs = [np.array(U_final.sub(i).dat.data_ro) for i in range(n_species)]
    phi_vec = np.array(U_final.sub(n_species).dat.data_ro)

    c_noisy = [_add_percent_noise(v, noise_percent, rng) for v in c_vecs]
    phi_noisy = _add_percent_noise(phi_vec, noise_percent, rng)

    return tuple(c_vecs + [phi_vec] + c_noisy + [phi_noisy])


def generate_noisy_data_robin(
    solver_params: Sequence[Any],
    noise_percent: float = 10.0,
    seed: Optional[int] = None,
    print_interval: int = 100,
    noise_std: Optional[float] = None,
) -> Tuple:
    """Run the Robin-BC PNP forward solver and return clean/noisy field vectors.

    Parameters
    ----------
    solver_params:
        Standard 11-entry solver parameter list (must include ``robin_bc``
        sub-dict in ``solver_params[10]``).
    noise_percent:
        Gaussian noise level as percent of RMS(field).
    seed:
        RNG seed for reproducible noise.
    print_interval:
        Forward solver progress print frequency.
    noise_std:
        Deprecated alias for ``noise_percent``; kept for backward compat.

    Returns
    -------
    tuple
        ``2*n_species + 2`` arrays in order:
        ``(clean c_0, ..., clean c_{n-1}, clean phi,
          noisy c_0, ..., noisy c_{n-1}, noisy phi)``
    """
    if noise_std is not None:
        noise_percent = float(noise_std)

    rng = np.random.default_rng(seed)
    try:
        (n_species, order, dt, t_end, z_vals, D_vals,
         a_vals, phi_applied, c0, phi0, params) = solver_params
    except Exception as exc:
        raise ValueError(
            "generate_noisy_data_robin expects a list of 11 solver parameters"
        ) from exc

    ctx = _r_build_context(solver_params)
    ctx = _r_build_forms(ctx, solver_params)
    _r_set_initial_conditions(ctx, solver_params, blob=True)
    U_final = _r_forsolve(ctx, solver_params, print_interval=print_interval)

    c_vecs = [np.array(U_final.sub(i).dat.data_ro) for i in range(n_species)]
    phi_vec = np.array(U_final.sub(n_species).dat.data_ro)

    c_noisy = [_add_percent_noise(v, noise_percent, rng) for v in c_vecs]
    phi_noisy = _add_percent_noise(phi_vec, noise_percent, rng)

    return tuple(c_vecs + [phi_vec] + c_noisy + [phi_noisy])
