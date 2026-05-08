"""Smoke + algebra tests for ``Forward.bv_solver.multi_ion`` (plan §2.3).

Exercises the multi-ion shared-theta machinery on the Cs⁺ + SO₄²⁻
production target electrolyte:
  - bulk θ_b > 0
  - electroneutrality bisection at the bulk converges to phi_o ≈ 0
  - outer-region γ_s reduces to 1.0 when ψ_D = 0
  - λ_eff is finite and positive at small phi_o
  - 1:1 single-counterion ClO₄⁻ degenerates to the legacy formulas
"""
from __future__ import annotations

import math
import os
import sys

import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


@pytest.fixture
def csplus_so4_ctx():
    from scripts._bv_common import (
        DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,
        DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
        C_O2_HAT, H2O2_SEED_NONDIM, C_HP_HAT, A_DEFAULT,
    )
    from Forward.bv_solver.multi_ion import build_counterion_ctx
    counterions = [
        DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,
        DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
    ]
    a_dyn = [A_DEFAULT] * 3
    c_dyn_bulk = [C_O2_HAT, H2O2_SEED_NONDIM, C_HP_HAT]
    z_dyn = [0, 0, +1]
    return build_counterion_ctx(counterions, a_dyn, c_dyn_bulk, z_dyn)


def test_theta_b_positive_for_csplus_so4(csplus_so4_ctx):
    """Hard-sphere a_nondim from Cs⁺ r=2.2Å + SO₄ r=2.4Å keeps θ_b ≈ 0.99."""
    theta_b = csplus_so4_ctx["theta_b"]
    assert theta_b > 0.95
    assert theta_b < 1.0


def test_bulk_phi_o_solves_to_zero(csplus_so4_ctx):
    """At c_dyn = bulk and electroneutral counterions, phi_o = 0 satisfies
    ρ_total = 0 — the bisection should land on or near zero."""
    from Forward.bv_solver.multi_ion import (
        solve_outer_phi_multiion, _electroneutrality_residual,
    )
    c_dyn_bulk = csplus_so4_ctx["c_dyn_bulk"]
    phi_o = solve_outer_phi_multiion(ctx=csplus_so4_ctx, c_dyn_outer=c_dyn_bulk)
    rho = _electroneutrality_residual(
        phi=phi_o, ctx=csplus_so4_ctx, c_dyn=c_dyn_bulk
    )
    assert abs(rho) < 1e-9, f"phi_o={phi_o} gave rho_total={rho}"
    assert abs(phi_o) < 1e-3, f"bulk phi_o should be near 0, got {phi_o}"


def test_per_ion_outer_concs_reduce_to_bulk(csplus_so4_ctx):
    """At phi_o = 0 the closure gives c_k = c_b_k · (1-A_dyn)/(theta_b + sum a_k c_b_k).
    With sum(a_dyn*c_dyn) = A_dyn_bulk, the multi-ion closure reduces back
    to c_b_k by definition of theta_b."""
    from Forward.bv_solver.multi_ion import per_ion_outer_concs
    c_dyn_bulk = csplus_so4_ctx["c_dyn_bulk"]
    cs = per_ion_outer_concs(ctx=csplus_so4_ctx, c_dyn_outer=c_dyn_bulk, phi_o=0.0)
    # At bulk + phi_o=0, the closure must give back c_bulk for each ion.
    for ion, c_outer in zip(csplus_so4_ctx["ions"], cs):
        assert math.isclose(
            c_outer, ion["c_bulk_nondim"], rel_tol=1e-9, abs_tol=1e-12
        ), f"ion z={ion['z']}: c_outer={c_outer} != c_bulk={ion['c_bulk_nondim']}"


def test_gamma_s_unity_at_psi_d_zero(csplus_so4_ctx):
    """γ_s = 1 when ψ_D = 0 regardless of multi-ion sizes."""
    from Forward.bv_solver.multi_ion import (
        compute_surface_gamma_multiion, per_ion_outer_concs,
    )
    c_dyn_bulk = csplus_so4_ctx["c_dyn_bulk"]
    H_o = c_dyn_bulk[2]   # H+ outer = bulk
    cs = per_ion_outer_concs(ctx=csplus_so4_ctx, c_dyn_outer=c_dyn_bulk, phi_o=0.0)
    # Build per-ion dicts with c_outer (the gamma helper expects this).
    ions_with_outer = [
        {**ion, "c_outer": float(c_o)}
        for ion, c_o in zip(csplus_so4_ctx["ions"], cs)
    ]
    a_h = csplus_so4_ctx["a_dyn"][2]
    gamma = compute_surface_gamma_multiion(
        H_o=H_o, psi_D=0.0, a_h=a_h, ions=ions_with_outer
    )
    assert math.isclose(gamma, 1.0, rel_tol=1e-12)


def test_lambda_eff_finite_at_phi_o_zero(csplus_so4_ctx):
    """λ_eff is positive and finite at the bulk anchor."""
    from Forward.bv_solver.multi_ion import effective_debye_length_local
    c_dyn_bulk = csplus_so4_ctx["c_dyn_bulk"]
    poisson_coeff = 1.0e-6   # representative nondim ε for I~0.3M
    lam = effective_debye_length_local(
        phi_o=0.0, ctx=csplus_so4_ctx, c_dyn_outer=c_dyn_bulk,
        poisson_coeff=poisson_coeff,
    )
    assert math.isfinite(lam)
    assert lam > 0.0


def test_clo4_single_counterion_degenerates_to_log_clo4_ratio():
    """1:1 single-ClO₄⁻ counterion: bisection reproduces the legacy
    closed form ``phi_o = log(H_o / c_clo4_bulk)``."""
    from scripts._bv_common import (
        DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC,
        C_O2_HAT, H2O2_SEED_NONDIM, C_HP_HAT, A_DEFAULT, C_CLO4_HAT,
    )
    from Forward.bv_solver.multi_ion import (
        build_counterion_ctx, solve_outer_phi_multiion,
    )
    counterions = [DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC]
    a_dyn = [A_DEFAULT] * 3
    c_dyn_bulk = [C_O2_HAT, H2O2_SEED_NONDIM, C_HP_HAT]
    z_dyn = [0, 0, +1]
    ctx = build_counterion_ctx(counterions, a_dyn, c_dyn_bulk, z_dyn)
    # Pick H_o slightly enriched; legacy phi_o = log(H_o / c_clo4_bulk).
    H_o = 0.1234
    c_dyn_outer = [c_dyn_bulk[0], c_dyn_bulk[1], H_o]
    phi_o = solve_outer_phi_multiion(ctx=ctx, c_dyn_outer=c_dyn_outer)
    expected = math.log(H_o / float(C_CLO4_HAT))
    # Single steric counterion has a small nonlinear correction from the
    # Bikerman denominator vs the ideal log form; tolerate ~1% drift.
    assert abs(phi_o - expected) < 0.02 * abs(expected) + 1e-3, (
        f"phi_o={phi_o} vs legacy log(H_o/c_clo4)={expected}"
    )
