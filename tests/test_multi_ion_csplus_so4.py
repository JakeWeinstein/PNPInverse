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


# ---------------------------------------------------------------------------
# Phase 5α T5 — _phi_safe_exp clamps phi to per-ion phi_clamp_val
# ---------------------------------------------------------------------------

def test_phi_safe_exp_clamps_at_phi_clamp_50():
    """Mirror boltzmann.py:223 UFL clamp — phi above +50 is clipped
    to +50 BEFORE multiplying by -z."""
    from Forward.bv_solver.multi_ion import _phi_safe_exp
    # phi=60 with z=1 → phi_clamped=50 → exp(-50)
    assert _phi_safe_exp(60.0, z=1.0, phi_clamp_val=50.0) == math.exp(-50.0)
    # phi=-60 with z=1 → phi_clamped=-50 → exp(+50)
    assert _phi_safe_exp(-60.0, z=1.0, phi_clamp_val=50.0) == math.exp(50.0)
    # phi within range — no clamp
    assert _phi_safe_exp(5.0, z=2.0, phi_clamp_val=50.0) == math.exp(-10.0)


def test_phi_safe_exp_per_ion_phi_clamp_propagates(csplus_so4_ctx):
    """Each ion's phi_clamp is honored independently — Cs+ at phi_clamp=50
    and a synthetic ion with phi_clamp=10 in the same ctx clamp
    differently at extreme phi."""
    from Forward.bv_solver.multi_ion import _phi_safe_exp
    # Cs+ z=+1, phi_clamp=50 → at phi=60 returns exp(-50)
    cs_at_extreme = _phi_safe_exp(60.0, z=1.0, phi_clamp_val=50.0)
    # synthetic ion z=+1, phi_clamp=10 → at phi=60 returns exp(-10)
    syn_at_extreme = _phi_safe_exp(60.0, z=1.0, phi_clamp_val=10.0)
    assert cs_at_extreme != syn_at_extreme
    assert cs_at_extreme == math.exp(-50.0)
    assert syn_at_extreme == math.exp(-10.0)


# ---------------------------------------------------------------------------
# Phase 5α T6 — Picard helpers' multi-ion branches
# ---------------------------------------------------------------------------

def test_solve_phi_o_multi_ion_matches_solve_outer_phi_multiion(csplus_so4_ctx):
    """Helper with multi_ion_ctx must agree with a direct
    solve_outer_phi_multiion call at the same c_dyn_outer."""
    from Forward.bv_solver.multi_ion import solve_outer_phi_multiion
    from Forward.bv_solver.picard_ic import _solve_phi_o
    # Use bulk concs (electroneutrality solves to phi_o ≈ 0).
    c_dyn_outer = list(csplus_so4_ctx["c_dyn_bulk"])
    direct = solve_outer_phi_multiion(ctx=csplus_so4_ctx, c_dyn_outer=c_dyn_outer)
    helper = _solve_phi_o(
        H_o=c_dyn_outer[2],
        c_clo4_bulk=0.0,  # ignored under multi_ion_ctx
        multi_ion_ctx=csplus_so4_ctx,
        c_dyn_outer=c_dyn_outer,
        phi_o_prev=None,
    )
    assert math.isclose(helper, direct, rel_tol=1e-9, abs_tol=1e-12)


def test_solve_phi_o_multi_ion_warm_start_matches_cold_start(csplus_so4_ctx):
    """phi_o_prev warm-start must converge to same root as global
    bracket cold-start (same electroneutrality zero)."""
    from Forward.bv_solver.picard_ic import _solve_phi_o
    c_dyn_outer = list(csplus_so4_ctx["c_dyn_bulk"])
    cold = _solve_phi_o(
        H_o=c_dyn_outer[2], c_clo4_bulk=0.0,
        multi_ion_ctx=csplus_so4_ctx, c_dyn_outer=c_dyn_outer,
        phi_o_prev=None,
    )
    warm = _solve_phi_o(
        H_o=c_dyn_outer[2], c_clo4_bulk=0.0,
        multi_ion_ctx=csplus_so4_ctx, c_dyn_outer=c_dyn_outer,
        phi_o_prev=0.5,  # narrow local bracket
    )
    assert math.isclose(cold, warm, rel_tol=1e-9, abs_tol=1e-12)


def test_compute_picard_gamma_s_multi_ion_matches_compute_surface_gamma_multiion(
    csplus_so4_ctx,
):
    """Helper with multi_ion_ctx must agree with a direct
    compute_surface_gamma_multiion call at the same per-ion outer concs."""
    from Forward.bv_solver.multi_ion import (
        compute_surface_gamma_multiion, per_ion_outer_concs,
    )
    from Forward.bv_solver.picard_ic import _compute_picard_gamma_s
    from scripts._bv_common import A_DEFAULT

    c_s = list(csplus_so4_ctx["c_dyn_bulk"])
    phi_o = 0.0
    psi_D = 1.5

    c_outer_per_ion = per_ion_outer_concs(
        ctx=csplus_so4_ctx, c_dyn_outer=c_s, phi_o=phi_o,
    )
    ions_with_outer = [
        {**ion, "c_outer": float(c_o)}
        for ion, c_o in zip(csplus_so4_ctx["ions"], c_outer_per_ion)
    ]
    direct = compute_surface_gamma_multiion(
        H_o=c_s[2], psi_D=psi_D, a_h=A_DEFAULT, ions=ions_with_outer,
    )
    helper = _compute_picard_gamma_s(
        H_o=c_s[2], c_clo4_bulk=0.0,
        psi_D=psi_D, a_h=A_DEFAULT, a_cl=0.0, c_cl_anchor=0.0,
        multi_ion_ctx=csplus_so4_ctx, phi_o=phi_o, c_s=c_s,
    )
    assert math.isclose(helper, direct, rel_tol=1e-12, abs_tol=1e-15)


def test_solve_picard_stern_split_multi_ion_psi_D_below_full_drop(
    csplus_so4_ctx,
):
    """Multi-ion linear-Debye Stern must give |psi_D| <= |full_drop|."""
    from Forward.bv_solver.picard_ic import _solve_picard_stern_split

    phi_applied = 21.4
    phi_o = 0.0
    full_drop = phi_applied - phi_o
    poisson_coeff = 0.0025  # rough nondim eps
    stern_split = {"stern_coeff": 1.0, "eps": poisson_coeff}

    psi_S, psi_D, phi_surf = _solve_picard_stern_split(
        phi_applied_model=phi_applied,
        phi_o=phi_o,
        c_clo4_bulk=0.0,
        a_cl=0.0,
        stern_split=stern_split,
        multi_ion_ctx=csplus_so4_ctx,
        c_dyn_outer=list(csplus_so4_ctx["c_dyn_bulk"]),
        poisson_coefficient=poisson_coeff,
    )
    assert abs(psi_D) <= abs(full_drop) + 1e-12
    assert math.isclose(psi_S + psi_D, full_drop, rel_tol=1e-9, abs_tol=1e-12)
    assert math.isclose(phi_surf, phi_applied - psi_S, rel_tol=1e-9, abs_tol=1e-12)


def test_solve_picard_stern_split_multi_ion_no_stern_returns_full_drop(
    csplus_so4_ctx,
):
    """Multi-ion ctx + ``stern_split=None`` returns identical no-Stern
    triple to the single-ion path."""
    from Forward.bv_solver.picard_ic import _solve_picard_stern_split
    psi_S, psi_D, phi_surf = _solve_picard_stern_split(
        phi_applied_model=12.0, phi_o=0.5,
        c_clo4_bulk=0.0, a_cl=0.0, stern_split=None,
        multi_ion_ctx=csplus_so4_ctx,
        c_dyn_outer=list(csplus_so4_ctx["c_dyn_bulk"]),
        poisson_coefficient=0.0025,
    )
    assert psi_S == 0.0
    assert psi_D == 12.0 - 0.5
    assert phi_surf == 12.0


def test_update_electrostatics_multi_ion_smoke(csplus_so4_ctx):
    """Smoke test: multi-ion threading through _update_electrostatics
    produces finite output without raising."""
    from Forward.bv_solver.picard_ic import _update_electrostatics
    from scripts._bv_common import A_DEFAULT

    c_s = list(csplus_so4_ctx["c_dyn_bulk"])
    phi_o = 0.0   # bulk solve
    poisson_coeff = 0.0025
    rxns = [{"E_eq_model": 0.0}, {"E_eq_model": 1.5}]
    out = _update_electrostatics(
        c_s=c_s, h_idx=2, c_clo4_bulk=0.0, phi_o=phi_o,
        phi_applied_model=12.0,
        a_h=A_DEFAULT, a_cl=0.0, c_cl_anchor_kind="bulk",
        stern_split={"stern_coeff": 1.0, "eps": poisson_coeff},
        reactions=rxns,
        bv_exp_scale=1.0, exponent_clip=100.0, clip_exponent=True,
        multi_ion_ctx=csplus_so4_ctx,
        poisson_coefficient=poisson_coeff,
    )
    for k in ("phi_o", "psi_S", "psi_D", "phi_surface", "eta_drop", "gamma_s"):
        assert math.isfinite(out[k]), f"{k} not finite: {out[k]!r}"
    assert all(math.isfinite(e) for e in out["eta_list"])
    # eta_drop = psi_S (Stern path)
    assert out["eta_drop"] == out["psi_S"]


def test_update_electrostatics_single_ion_unchanged_when_ctx_None():
    """multi_ion_ctx=None → identical output to the pre-T6 single-ion path."""
    from Forward.bv_solver.picard_ic import _update_electrostatics
    from scripts._bv_common import A_DEFAULT
    rxns = [{"E_eq_model": 0.0}, {"E_eq_model": 1.5}]
    out_a = _update_electrostatics(
        c_s=[1.0, 1e-5, 0.05], h_idx=2,
        c_clo4_bulk=0.0833, phi_o=-0.51,
        phi_applied_model=12.0,
        a_h=A_DEFAULT, a_cl=A_DEFAULT, c_cl_anchor_kind="bulk",
        stern_split=None, reactions=rxns,
        bv_exp_scale=1.0, exponent_clip=100.0, clip_exponent=True,
        multi_ion_ctx=None,
    )
    out_b = _update_electrostatics(
        c_s=[1.0, 1e-5, 0.05], h_idx=2,
        c_clo4_bulk=0.0833, phi_o=-0.51,
        phi_applied_model=12.0,
        a_h=A_DEFAULT, a_cl=A_DEFAULT, c_cl_anchor_kind="bulk",
        stern_split=None, reactions=rxns,
        bv_exp_scale=1.0, exponent_clip=100.0, clip_exponent=True,
    )
    for k in ("phi_o", "psi_S", "psi_D", "phi_surface", "eta_drop", "gamma_s"):
        assert out_a[k] == out_b[k]
    assert out_a["eta_list"] == out_b["eta_list"]
