"""Sweep ordering and Lagrange predictor utilities for BV continuation solves.

Extracted from FluxCurve/bv_point_solve/predictor.py so that the Forward layer
can use them without importing from FluxCurve.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _apply_predictor(
    phi_applied_i: float,
    ctx_U,
    carry_U_data: tuple,
    predictor_prev: Optional[tuple],
    predictor_curr: Optional[tuple],
    predictor_prev2: Optional[tuple],
    n_species: int = 0,
):
    """Apply quadratic/linear hybrid predictor to initial guess.

    Uses 3-point quadratic Lagrange interpolation when available and safe,
    otherwise falls back to 2-point linear extrapolation, otherwise simple
    warm-start copy.  Clamps concentrations to >= 1e-10 after prediction.

    After applying a predictor (quadratic or linear), validates the result:
    if any DOF deviates by more than 10x from the carry state, falls back
    to simple warm-start.  This prevents SNES divergence from aggressive
    extrapolation (which accounts for ~35% of point-solve retries).
    """
    _MAX_PREDICTOR_RATIO = 10.0  # max allowed deviation from carry state

    def _validate_and_maybe_revert():
        """Check predicted state; revert to carry if too far from carry."""
        for src, dst in zip(carry_U_data, ctx_U.dat):
            pred = dst.data_ro
            ref = np.maximum(np.abs(src), 1e-10)
            max_ratio = np.max(np.abs(pred - src) / ref)
            if max_ratio > _MAX_PREDICTOR_RATIO:
                # Prediction is too aggressive; revert to simple warm-start
                for s, d in zip(carry_U_data, ctx_U.dat):
                    d.data[:] = s
                return

    if predictor_prev is not None and predictor_curr is not None:
        eta_curr, U_curr_data = predictor_curr
        eta_prev, U_prev_data = predictor_prev

        # Try quadratic predictor if 3 points available
        if predictor_prev2 is not None:
            eta_prev2, U_prev2_data = predictor_prev2
            # Safety check: only use quadratic if extrapolation distance
            # is <= 2x the max prior gap
            extrap_dist = abs(phi_applied_i - eta_curr)
            max_prior_gap = max(abs(eta_curr - eta_prev), abs(eta_prev - eta_prev2))
            if max_prior_gap > 1e-14 and extrap_dist <= 2.0 * max_prior_gap:
                # Quadratic Lagrange interpolation
                eta_a, eta_b, eta_c = eta_prev2, eta_prev, eta_curr
                denom_a = (eta_a - eta_b) * (eta_a - eta_c)
                denom_b = (eta_b - eta_a) * (eta_b - eta_c)
                denom_c = (eta_c - eta_a) * (eta_c - eta_b)
                if abs(denom_a) > 1e-28 and abs(denom_b) > 1e-28 and abs(denom_c) > 1e-28:
                    L_a = ((phi_applied_i - eta_b) * (phi_applied_i - eta_c)) / denom_a
                    L_b = ((phi_applied_i - eta_a) * (phi_applied_i - eta_c)) / denom_b
                    L_c = ((phi_applied_i - eta_a) * (phi_applied_i - eta_b)) / denom_c
                    for u_a, u_b, u_c, dst in zip(
                        U_prev2_data, U_prev_data, U_curr_data, ctx_U.dat
                    ):
                        dst.data[:] = L_a * u_a + L_b * u_b + L_c * u_c
                    # Clamp concentrations to prevent negative values
                    # (skip the last component -- electrical potential -- which can be negative)
                    for i, d in enumerate(ctx_U.dat):
                        if i < n_species:
                            d.data[:] = np.maximum(d.data, 1e-10)
                    _validate_and_maybe_revert()
                    return
                # Fall through to linear if denominators are degenerate

        # Linear predictor from two prior solutions
        d_eta = eta_curr - eta_prev
        if abs(d_eta) > 1e-14:
            slope = (phi_applied_i - eta_curr) / d_eta
            for u_prev_arr, u_curr_arr, dst in zip(
                U_prev_data, U_curr_data, ctx_U.dat
            ):
                dst.data[:] = u_curr_arr + slope * (u_curr_arr - u_prev_arr)
            # Clamp concentrations to prevent negative values
            # (skip the last component -- electrical potential -- which can be negative)
            for i, d in enumerate(ctx_U.dat):
                if i < n_species:
                    d.data[:] = np.maximum(d.data, 1e-10)
            _validate_and_maybe_revert()
            return

    # Simple warm-start (only one prior solution or degenerate)
    for src, dst in zip(carry_U_data, ctx_U.dat):
        dst.data[:] = src


def _build_sweep_order(phi_applied_values: np.ndarray) -> np.ndarray:
    """Build sweep order for warm-start continuation with mixed-sign eta.

    When all phi_applied values share the same sign (or are zero), this
    reduces to the original ``np.argsort(np.abs(phi_applied_values))``
    (ascending |eta|).

    When both positive and negative values are present, a two-branch sweep
    is used:

    1. **Negative branch** (eta <= 0): sorted ascending in |eta|.
    2. **Positive branch** (eta > 0):  sorted ascending in eta.

    The negative branch is processed first (it typically contains the
    smallest-|eta| point, e.g. eta = -0.25).  The positive branch then
    starts from the smallest positive eta.  Between branches, the
    warm-start carries the state from the last negative point solved
    at the lowest |eta| end.  A bridge point through eta = 0 can be
    inserted by the existing bridge-point logic if ``max_eta_gap > 0``.

    If the smallest-|eta| point is positive, the positive branch goes
    first instead.

    Parameters
    ----------
    phi_applied_values : np.ndarray
        Array of dimensionless overpotentials.

    Returns
    -------
    np.ndarray
        Array of original indices giving the sweep order.
    """
    phi = np.asarray(phi_applied_values, dtype=float)

    neg_mask = phi <= 0
    pos_mask = phi > 0

    has_neg = neg_mask.any()
    has_pos = pos_mask.any()

    if not has_neg or not has_pos:
        # Single-sign case: original behaviour (ascending |eta|)
        return np.argsort(np.abs(phi))

    # Two-branch case.
    # Indices in each branch, sorted outward from equilibrium.
    neg_indices = np.where(neg_mask)[0]
    neg_sorted = neg_indices[np.argsort(np.abs(phi[neg_indices]))]  # ascending |eta|

    pos_indices = np.where(pos_mask)[0]
    pos_sorted = pos_indices[np.argsort(phi[pos_indices])]  # ascending eta

    # Choose which branch goes first: the one containing the smallest |eta|.
    min_neg_abs = np.min(np.abs(phi[neg_indices]))
    min_pos_abs = np.min(np.abs(phi[pos_indices]))

    if min_neg_abs <= min_pos_abs:
        return np.concatenate([neg_sorted, pos_sorted])
    else:
        return np.concatenate([pos_sorted, neg_sorted])
