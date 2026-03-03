"""Solver parameter construction and synthetic data generation."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from .config import SyntheticData
from ..parameter_targets import ParameterTarget
from ..solver_interface import (
    ForwardSolverAdapter,
    as_species_list,
    deep_copy_solver_params,
    extract_solution_vectors,
)
from Forward.params import SolverParams


def build_default_solver_params(
    *,
    n_species: int,
    order: int = 1,
    dt: float = 1e-2,
    t_end: float = 0.1,
    z_vals: Optional[Sequence[float]] = None,
    d_vals: Optional[Sequence[float]] = None,
    a_vals: Optional[Sequence[float]] = None,
    phi_applied: float = 0.05,
    c0_vals: Optional[Sequence[float]] = None,
    phi0: float = 1.0,
    solver_options: Optional[Dict[str, Any]] = None,
) -> SolverParams:
    """Build a standard 11-entry :class:`~Forward.params.SolverParams`.

    Returns a :class:`~Forward.params.SolverParams` instance which behaves
    identically to a plain list (full backward compat with index/unpack) while
    additionally exposing named attribute access::

        p = build_default_solver_params(n_species=2, ...)
        p.phi_applied   # same as p[7]
        p.solver_options  # same as p[10]

    Parameters
    ----------
    n_species:
        Number of ionic species.
    order:
        Finite-element polynomial order.
    dt:
        Time step.
    t_end:
        Final simulation time.
    z_vals:
        Per-species valences (defaults to alternating +1/-1).
    d_vals:
        Per-species diffusivities (defaults to 1.0).
    a_vals:
        Per-species activities (defaults to 0.0, currently unused).
    phi_applied:
        Applied boundary voltage.
    c0_vals:
        Initial bulk concentrations (defaults to 0.1).
    phi0:
        Reference/initial potential.
    solver_options:
        PETSc/SNES/Robin/nondim option dict.
    """
    n = int(n_species)
    if n <= 0:
        raise ValueError(f"n_species must be positive; got {n}.")

    z_default = [1 if i % 2 == 0 else -1 for i in range(n)]
    d_default = [1.0 for _ in range(n)]
    a_default = [0.0 for _ in range(n)]
    c0_default = [0.1 for _ in range(n)]

    z_values = as_species_list(z_vals if z_vals is not None else z_default, n, "z_vals")
    d_values = as_species_list(d_vals if d_vals is not None else d_default, n, "d_vals")
    a_values = as_species_list(a_vals if a_vals is not None else a_default, n, "a_vals")
    c0_values = as_species_list(c0_vals if c0_vals is not None else c0_default, n, "c0_vals")

    options = dict(solver_options or {})

    return SolverParams.from_list([
        n,
        int(order),
        float(dt),
        float(t_end),
        z_values,
        d_values,
        a_values,
        float(phi_applied),
        c0_values,
        float(phi0),
        options,
    ])


def generate_synthetic_data(
    adapter: ForwardSolverAdapter,
    solver_params: Sequence[Any],
    *,
    noise_percent: float,
    seed: Optional[int] = None,
    blob_initial_condition: bool = True,
    print_interval: int = 100,
) -> SyntheticData:
    """Generate clean/noisy final-time observations from a forward solve."""
    params_copy = deep_copy_solver_params(solver_params)
    n_species = int(params_copy[0])

    _, U_final = adapter.run_forward(
        params_copy,
        blob_initial_condition=blob_initial_condition,
        print_interval=print_interval,
    )

    clean_c, clean_phi = extract_solution_vectors(U_final, n_species)

    rng = np.random.default_rng(seed)
    noisy_c = [_add_percent_noise(vec, noise_percent, rng) for vec in clean_c]
    noisy_phi = _add_percent_noise(clean_phi, noise_percent, rng)

    return SyntheticData(
        clean_concentration_vectors=clean_c,
        clean_phi_vector=clean_phi,
        noisy_concentration_vectors=noisy_c,
        noisy_phi_vector=noisy_phi,
    )


def _add_percent_noise(vec: Sequence[float], noise_percent: float, rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian noise with sigma = ``noise_percent / 100 * RMS(vec)``."""
    v = np.asarray(vec, dtype=float)
    pct = float(noise_percent)
    if pct < 0:
        raise ValueError(f"noise_percent must be non-negative; got {pct}.")
    if pct == 0:
        return v.copy()
    rms = float(np.sqrt(np.mean(v * v)))
    sigma = (pct / 100.0) * max(rms, 1e-12)
    return v + rng.normal(0.0, sigma, size=v.shape)


def _vector_to_function(ctx: Dict[str, Any], vec: Sequence[float], *, space_key: str = "V_scalar"):
    """Create a Firedrake Function by directly setting coefficient vector values."""
    import firedrake as fd

    V = ctx[space_key]
    out = fd.Function(V)
    flat_vec = np.asarray(vec, dtype=float).ravel()

    if flat_vec.size != out.dat.data.size:
        raise ValueError(
            f"Target vector length {flat_vec.size} != DOFs {out.dat.data.size} for {space_key}."
        )

    out.dat.data[:] = flat_vec
    return out
