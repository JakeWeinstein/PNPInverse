"""Latin Hypercube Sampling for BV kinetics parameter space.

Generates well-distributed parameter samples for surrogate training data.
Supports log-space sampling for k0 (which spans orders of magnitude).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class ParameterBounds:
    """Bounds for BV kinetics parameters.

    k0 ranges are in dimensionless (k0_hat) space.  When ``log_space_k0=True``
    is passed to :func:`generate_lhs_samples`, k0 values are sampled
    uniformly in log10 space and then exponentiated.

    Attributes
    ----------
    k0_1_range : tuple of float
        (min, max) for first reaction rate constant k0_1.
    k0_2_range : tuple of float
        (min, max) for second reaction rate constant k0_2.
    alpha_1_range : tuple of float
        (min, max) for first transfer coefficient alpha_1.
    alpha_2_range : tuple of float
        (min, max) for second transfer coefficient alpha_2.
    """
    k0_1_range: Tuple[float, float] = (1e-6, 1.0)
    k0_2_range: Tuple[float, float] = (1e-7, 0.1)
    alpha_1_range: Tuple[float, float] = (0.1, 0.9)
    alpha_2_range: Tuple[float, float] = (0.1, 0.9)


def _lhs_unit_cube(n_samples: int, n_dims: int, rng: np.random.Generator) -> np.ndarray:
    """Generate Latin Hypercube samples on [0, 1]^n_dims.

    Uses the classic randomized LHS algorithm: each dimension is divided
    into n_samples equal strata, and a random permutation determines which
    stratum each sample falls in.  Within each stratum, a uniform random
    offset is added.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_dims : int
        Number of dimensions.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray of shape (n_samples, n_dims)
        Samples in [0, 1]^n_dims.
    """
    result = np.empty((n_samples, n_dims), dtype=float)
    for d in range(n_dims):
        perm = rng.permutation(n_samples)
        for i in range(n_samples):
            result[i, d] = (perm[i] + rng.uniform()) / n_samples
    return result


def generate_lhs_samples(
    bounds: ParameterBounds,
    n_samples: int,
    seed: int = 42,
    log_space_k0: bool = True,
) -> np.ndarray:
    """Generate Latin Hypercube samples in the BV parameter space.

    Parameters
    ----------
    bounds : ParameterBounds
        Parameter ranges for each dimension.
    n_samples : int
        Number of samples to generate.
    seed : int
        Random seed for reproducibility.
    log_space_k0 : bool
        If True, sample k0_1 and k0_2 uniformly in log10 space, then
        exponentiate.  This is recommended when k0 spans multiple
        orders of magnitude.

    Returns
    -------
    np.ndarray of shape (n_samples, 4)
        Columns: [k0_1, k0_2, alpha_1, alpha_2] in physical space.
    """
    rng = np.random.default_rng(seed)
    unit = _lhs_unit_cube(n_samples, 4, rng)

    samples = np.empty((n_samples, 4), dtype=float)

    if log_space_k0:
        log_k0_1_lo = np.log10(max(bounds.k0_1_range[0], 1e-30))
        log_k0_1_hi = np.log10(bounds.k0_1_range[1])
        log_k0_2_lo = np.log10(max(bounds.k0_2_range[0], 1e-30))
        log_k0_2_hi = np.log10(bounds.k0_2_range[1])
        samples[:, 0] = np.power(10.0, log_k0_1_lo + unit[:, 0] * (log_k0_1_hi - log_k0_1_lo))
        samples[:, 1] = np.power(10.0, log_k0_2_lo + unit[:, 1] * (log_k0_2_hi - log_k0_2_lo))
    else:
        samples[:, 0] = bounds.k0_1_range[0] + unit[:, 0] * (bounds.k0_1_range[1] - bounds.k0_1_range[0])
        samples[:, 1] = bounds.k0_2_range[0] + unit[:, 1] * (bounds.k0_2_range[1] - bounds.k0_2_range[0])

    samples[:, 2] = bounds.alpha_1_range[0] + unit[:, 2] * (bounds.alpha_1_range[1] - bounds.alpha_1_range[0])
    samples[:, 3] = bounds.alpha_2_range[0] + unit[:, 3] * (bounds.alpha_2_range[1] - bounds.alpha_2_range[0])

    return samples


def generate_multi_region_lhs_samples(
    wide_bounds: ParameterBounds,
    focused_bounds: ParameterBounds,
    n_base: int,
    n_focused: int,
    seed_base: int = 42,
    seed_focused: int = 99,
    log_space_k0: bool = True,
) -> np.ndarray:
    """Generate multi-region LHS samples: wide coverage + focused refinement.

    Generates ``n_base`` LHS samples from ``wide_bounds`` and ``n_focused``
    LHS samples from ``focused_bounds``, then concatenates them.  This
    ensures the surrogate has both broad coverage and high accuracy near
    the expected operating region.

    Parameters
    ----------
    wide_bounds : ParameterBounds
        Bounds for the wide/base coverage region.
    focused_bounds : ParameterBounds
        Tighter bounds around expected true values.
    n_base : int
        Number of wide-coverage samples.
    n_focused : int
        Number of focused-region samples.
    seed_base : int
        Random seed for the base LHS.
    seed_focused : int
        Random seed for the focused LHS.
    log_space_k0 : bool
        If True, sample k0 values in log10 space.

    Returns
    -------
    np.ndarray of shape (n_base + n_focused, 4)
        Concatenated samples.  Columns: [k0_1, k0_2, alpha_1, alpha_2].
    """
    samples_wide = generate_lhs_samples(
        wide_bounds, n_base, seed=seed_base, log_space_k0=log_space_k0,
    )
    samples_focused = generate_lhs_samples(
        focused_bounds, n_focused, seed=seed_focused, log_space_k0=log_space_k0,
    )
    return np.concatenate([samples_wide, samples_focused], axis=0)
