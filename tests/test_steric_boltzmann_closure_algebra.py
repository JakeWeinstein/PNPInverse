"""Pure-Python algebra tests for the steric-aware Boltzmann counterion closure.

The closure replaces the unbounded ideal Boltzmann profile

    c_-(x) = c_b * exp(-z * phi(x))

with the steric-aware steady-state algebraic reduction (handoff doc:
``docs/steric_analytic_clo4_reduction_handoff.md``):

    c_steric(x) = c_b * exp(phi) * (1 - A_dyn(x))
                  / (theta_b + a_b * c_b * exp(phi))

derived from the zero-flux condition `grad(mu_-) = 0` for an inert
counterion (z = -1) under the sign-corrected Bikerman chemical
potential `mu_i = ln(c_i) + z_i * phi - ln(theta)`.

These tests pin the math BEFORE any UFL code is written.  The reference
implementation `_c_steric_scalar` is a plain NumPy translation of the
closure formula; the production UFL build must produce numerically
matching values at the same (phi, A_dyn) arguments.

Properties asserted (one test each):

1. Bulk recovery — phi=0 with A_dyn at bulk gives c_steric == c_b.
2. Dilute limit  — a_b -> 0 collapses to the ideal Boltzmann c_b * exp(phi).
3. Saturation    — c_steric monotonically approaches (1-A_dyn)/a_b from below
                   as phi increases (within the symmetric phi clamp).
4. Packing       — 1 - A_dyn - a_b * c_steric stays strictly positive
                   over a wide (phi, A_dyn) grid.
5. theta_b > 0   — the validator rejects (a_b, c_b, dynamic-bulk fractions)
                   combinations that overpack the bulk.
"""
from __future__ import annotations

import math

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Reference implementation — plain NumPy translation of the UFL closure
# ---------------------------------------------------------------------------

def _phi_clamped(phi: float, phi_clamp: float) -> float:
    """Symmetric clamp matching ``Forward/bv_solver/boltzmann.py:114-117``."""
    return max(-phi_clamp, min(phi_clamp, phi))


def _c_steric_scalar(
    phi: float,
    A_dyn: float,
    *,
    c_b: float,
    a_b: float,
    theta_b: float,
    phi_clamp: float = 50.0,
) -> float:
    """Reference scalar implementation of the steric Boltzmann closure.

    Parameters
    ----------
    phi
        Local potential at the evaluation point (nondim).
    A_dyn
        Local dynamic-species packing fraction ``sum_j a_j * c_j(x)``.
    c_b
        Bulk concentration of the analytic counterion (nondim).
    a_b
        Steric size of the analytic counterion (nondim, Bikerman).
    theta_b
        Bulk packing fraction ``1 - sum_j a_j * c_b_j - a_b * c_b``.
        Must be positive; validate with ``_theta_b`` before calling.
    phi_clamp
        Symmetric phi clamp.  Same convention as ``boltzmann.py``.
    """
    q = math.exp(_phi_clamped(phi, phi_clamp))
    numerator = c_b * q * (1.0 - A_dyn)
    denominator = theta_b + a_b * c_b * q
    return numerator / denominator


def _theta_b(*, a_dyn_bulk: list[float], c0_dyn_bulk: list[float],
             a_b: float, c_b: float) -> float:
    """Compute the bulk packing fraction ``1 - sum_j a_j * c_b_j - a_b * c_b``.

    The caller is expected to validate ``> 0``; we return the raw value
    so the validator can produce a clearer error message.
    """
    if len(a_dyn_bulk) != len(c0_dyn_bulk):
        raise ValueError("a_dyn_bulk and c0_dyn_bulk must align")
    A_bulk_dyn = sum(a * c for a, c in zip(a_dyn_bulk, c0_dyn_bulk))
    return 1.0 - A_bulk_dyn - a_b * c_b


# ---------------------------------------------------------------------------
# Production parameters (mirror scripts/_bv_common.py constants)
# ---------------------------------------------------------------------------

A_DEFAULT = 0.01
C_O2_HAT = 1.0
C_H2O2_HAT = 0.0
C_HP_HAT = 0.2
C_CLO4_HAT = 0.2
H2O2_SEED_NONDIM = 1e-4

# Three dynamic species: O2, H2O2, H+
A_DYN_BULK_PROD = [A_DEFAULT, A_DEFAULT, A_DEFAULT]
C0_DYN_BULK_PROD = [C_O2_HAT, H2O2_SEED_NONDIM, C_HP_HAT]
A_DYN_BULK_SUM_PROD = sum(a * c for a, c in zip(A_DYN_BULK_PROD, C0_DYN_BULK_PROD))


# ---------------------------------------------------------------------------
# Property 1 — bulk recovery
# ---------------------------------------------------------------------------

def test_bulk_recovery_returns_c_bulk():
    """At phi=0 with A_dyn = sum_j a_j * c_b_j, c_steric must equal c_b."""
    theta_b = _theta_b(
        a_dyn_bulk=A_DYN_BULK_PROD,
        c0_dyn_bulk=C0_DYN_BULK_PROD,
        a_b=A_DEFAULT,
        c_b=C_CLO4_HAT,
    )
    assert theta_b > 0.0  # production setup must give positive bulk packing

    c = _c_steric_scalar(
        phi=0.0,
        A_dyn=A_DYN_BULK_SUM_PROD,
        c_b=C_CLO4_HAT,
        a_b=A_DEFAULT,
        theta_b=theta_b,
    )
    assert math.isclose(c, C_CLO4_HAT, rel_tol=1e-14, abs_tol=1e-14), (
        f"bulk recovery: expected {C_CLO4_HAT}, got {c}"
    )


# ---------------------------------------------------------------------------
# Property 2 — dilute limit (a_b -> 0)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("phi", [-2.0, -0.5, 0.0, 0.5, 2.0, 5.0])
def test_dilute_limit_recovers_ideal_boltzmann(phi):
    """As a_b -> 0 with A_dyn -> 0, c_steric / (c_b * exp(phi)) -> 1."""
    c_b = C_CLO4_HAT
    ideal = c_b * math.exp(phi)

    ratios = []
    for a_b in (1e-3, 1e-5, 1e-7, 1e-9):
        # Dynamic species also dilute -> A_dyn -> 0
        A_dyn = a_b * (C_O2_HAT + H2O2_SEED_NONDIM + C_HP_HAT)
        theta_b = 1.0 - A_dyn - a_b * c_b
        c = _c_steric_scalar(
            phi=phi, A_dyn=A_dyn,
            c_b=c_b, a_b=a_b, theta_b=theta_b,
        )
        ratios.append(c / ideal)

    # Ratios should converge monotonically to 1 as a_b decreases
    assert math.isclose(ratios[-1], 1.0, rel_tol=1e-7), (
        f"dilute limit at phi={phi}: ratios={ratios}, did not converge to 1"
    )
    # Convergence is monotone toward 1
    distances_from_one = [abs(r - 1.0) for r in ratios]
    assert distances_from_one == sorted(distances_from_one, reverse=True), (
        f"dilute limit at phi={phi}: distances {distances_from_one} not "
        "monotonically decreasing"
    )


# ---------------------------------------------------------------------------
# Property 3 — saturation at high phi (within phi clamp)
# ---------------------------------------------------------------------------

def test_saturation_at_high_phi_within_clamp():
    """c_steric -> (1 - A_dyn) / a_b from below as phi increases.

    Use phi_clamp=100 so a sweep up to phi=20 stays unclamped and we can
    observe monotone approach to the cap without interference from the
    clamp's saturation.
    """
    a_b = A_DEFAULT
    c_b = C_CLO4_HAT
    A_dyn = A_DYN_BULK_SUM_PROD
    theta_b = _theta_b(
        a_dyn_bulk=A_DYN_BULK_PROD,
        c0_dyn_bulk=C0_DYN_BULK_PROD,
        a_b=a_b, c_b=c_b,
    )
    cap = (1.0 - A_dyn) / a_b

    phi_grid = np.linspace(0.0, 20.0, 21)
    c_vals = [
        _c_steric_scalar(
            phi=phi, A_dyn=A_dyn, c_b=c_b, a_b=a_b,
            theta_b=theta_b, phi_clamp=100.0,
        )
        for phi in phi_grid
    ]

    # Monotone increasing
    diffs = np.diff(c_vals)
    assert (diffs > 0).all(), f"c_steric not monotonically increasing: diffs={diffs}"

    # Bounded above by cap, asymptotically approaching it
    assert all(c < cap for c in c_vals), (
        f"c_steric exceeded cap {cap}: max={max(c_vals)}"
    )
    # At phi=20, very close to cap (but strictly below)
    assert c_vals[-1] / cap > 0.999, (
        f"c_steric at phi=20 should be within 0.1% of cap; "
        f"got c={c_vals[-1]} cap={cap} ratio={c_vals[-1] / cap}"
    )


def test_clamp_caps_value_at_clamp_limit():
    """With phi_clamp=50, evaluating at phi=100 must equal phi=50 (clamp engaged)."""
    a_b = A_DEFAULT
    c_b = C_CLO4_HAT
    A_dyn = A_DYN_BULK_SUM_PROD
    theta_b = _theta_b(
        a_dyn_bulk=A_DYN_BULK_PROD,
        c0_dyn_bulk=C0_DYN_BULK_PROD,
        a_b=a_b, c_b=c_b,
    )
    c_at_100 = _c_steric_scalar(
        phi=100.0, A_dyn=A_dyn, c_b=c_b, a_b=a_b,
        theta_b=theta_b, phi_clamp=50.0,
    )
    c_at_50 = _c_steric_scalar(
        phi=50.0, A_dyn=A_dyn, c_b=c_b, a_b=a_b,
        theta_b=theta_b, phi_clamp=50.0,
    )
    assert math.isclose(c_at_100, c_at_50, rel_tol=1e-14)


def test_cathodic_depletion():
    """At deeply negative phi (within clamp), c_steric -> 0."""
    a_b = A_DEFAULT
    c_b = C_CLO4_HAT
    A_dyn = A_DYN_BULK_SUM_PROD
    theta_b = _theta_b(
        a_dyn_bulk=A_DYN_BULK_PROD,
        c0_dyn_bulk=C0_DYN_BULK_PROD,
        a_b=a_b, c_b=c_b,
    )
    c = _c_steric_scalar(
        phi=-30.0, A_dyn=A_dyn, c_b=c_b, a_b=a_b,
        theta_b=theta_b, phi_clamp=50.0,
    )
    assert 0.0 < c < 1e-10, f"cathodic depletion: expected ~0, got {c}"


# ---------------------------------------------------------------------------
# Property 4 — packing positivity over a (phi, A_dyn) grid
# ---------------------------------------------------------------------------

def test_packing_non_negative_over_grid():
    """Total packing 1 - A_dyn - a_b * c_steric stays NON-NEGATIVE for
    all physical (phi, A_dyn).  The mathematical limit is packing -> 0
    as c_steric -> (1 - A_dyn)/a_b at extreme phi; the production code
    clamps with ``packing_floor=1e-8`` to keep ``mu_steric = -ln(theta)``
    finite.  This test asserts no meaningful negative packing
    (floating-point cancellation can give ~0)."""
    a_b = A_DEFAULT
    c_b = C_CLO4_HAT

    phi_vals = np.linspace(-50.0, 50.0, 21)
    A_dyn_vals = np.linspace(0.0, 0.5, 11)

    theta_b = _theta_b(
        a_dyn_bulk=A_DYN_BULK_PROD,
        c0_dyn_bulk=C0_DYN_BULK_PROD,
        a_b=a_b, c_b=c_b,
    )

    # Allow floating-point underflow ~1e-12; closure is mathematically
    # >= 0 strictly for finite phi but cancels to ~0 at the cap.
    floor_tol = -1e-12
    for A_dyn in A_dyn_vals:
        for phi in phi_vals:
            c = _c_steric_scalar(
                phi=phi, A_dyn=A_dyn,
                c_b=c_b, a_b=a_b, theta_b=theta_b,
                phi_clamp=50.0,
            )
            packing = 1.0 - A_dyn - a_b * c
            assert packing >= floor_tol, (
                f"packing meaningfully negative at phi={phi:.2f}, "
                f"A_dyn={A_dyn:.3f}: packing={packing:.4e}, c={c:.4e}"
            )


def test_packing_approaches_zero_at_saturation():
    """At extreme phi within the clamp, c_steric -> (1 - A_dyn)/a_b and
    packing -> 0+ exactly.  Verify the asymptotic limit and monotone
    decrease.  This is the design property the production
    ``packing_floor`` clamp protects against in ``mu_steric = -ln(theta)``."""
    a_b = A_DEFAULT
    c_b = C_CLO4_HAT
    A_dyn = A_DYN_BULK_SUM_PROD

    theta_b = _theta_b(
        a_dyn_bulk=A_DYN_BULK_PROD,
        c0_dyn_bulk=C0_DYN_BULK_PROD,
        a_b=a_b, c_b=c_b,
    )

    packings = []
    for phi in (5.0, 10.0, 15.0, 20.0):
        c = _c_steric_scalar(
            phi=phi, A_dyn=A_dyn, c_b=c_b, a_b=a_b,
            theta_b=theta_b, phi_clamp=50.0,
        )
        packings.append(1.0 - A_dyn - a_b * c)

    # Strictly positive in this range (well below clamp saturation)
    assert all(p > 0 for p in packings), (
        f"packings should stay strictly positive at moderate phi; got {packings}"
    )
    # Monotone decreasing toward 0
    assert packings == sorted(packings, reverse=True), (
        f"packing should monotonically decrease toward 0; got {packings}"
    )
    # Last entry should be small (within ~1e-3 of zero)
    assert packings[-1] < 1e-3, (
        f"packing at phi=20 should be near 0; got {packings[-1]}"
    )


# ---------------------------------------------------------------------------
# Property 5 — theta_b validator
# ---------------------------------------------------------------------------

def test_theta_b_positive_for_production_setup():
    """The production preset (3 dynamic species + 1 ClO4 counterion) must
    have theta_b > 0 with a comfortable margin."""
    theta_b = _theta_b(
        a_dyn_bulk=A_DYN_BULK_PROD,
        c0_dyn_bulk=C0_DYN_BULK_PROD,
        a_b=A_DEFAULT,
        c_b=C_CLO4_HAT,
    )
    # 1 - 0.01*(1.0 + 1e-4 + 0.2) - 0.01*0.2 = 1 - 0.012001 - 0.002 = 0.985999
    assert theta_b > 0.98, f"expected theta_b ~0.986 for production; got {theta_b}"


def test_theta_b_negative_when_overpacked():
    """When dynamic species + counterion would overpack the bulk lattice,
    theta_b is non-positive — the validator caller must reject this config."""
    # Pathological: 100x larger steric size than production
    a_dyn_bulk = [1.0, 1.0, 1.0]
    c0_dyn_bulk = [C_O2_HAT, C_H2O2_HAT, C_HP_HAT]  # sum ~1.2
    a_b = 1.0
    c_b = C_CLO4_HAT
    theta_b = _theta_b(
        a_dyn_bulk=a_dyn_bulk,
        c0_dyn_bulk=c0_dyn_bulk,
        a_b=a_b, c_b=c_b,
    )
    # 1 - (1.0 + 0 + 0.2) - 1.0*0.2 = 1 - 1.2 - 0.2 = -0.4
    assert theta_b < 0.0, f"expected negative theta_b for overpacked config; got {theta_b}"


def test_theta_b_aligns_with_bulk_recovery():
    """The bulk-recovery property (test 1) is only valid when
    A_dyn(bulk) + a_b * c_b = 1 - theta_b.  This is a self-consistency
    check on the formula and the input parameters."""
    a_b = A_DEFAULT
    c_b = C_CLO4_HAT
    theta_b = _theta_b(
        a_dyn_bulk=A_DYN_BULK_PROD,
        c0_dyn_bulk=C0_DYN_BULK_PROD,
        a_b=a_b, c_b=c_b,
    )
    expected_one_minus_theta_b = A_DYN_BULK_SUM_PROD + a_b * c_b
    assert math.isclose(1.0 - theta_b, expected_one_minus_theta_b, rel_tol=1e-14)
