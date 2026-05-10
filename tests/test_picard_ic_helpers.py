"""Pure-Python scalar tests for ``Forward.bv_solver.picard_ic``.

No Firedrake imports.  Pattern follows
``tests/test_steric_boltzmann_closure_algebra.py`` and
``tests/test_steric_psi_profile.py``: reference scalar implementations
+ parametrized identity tests.

Coverage targets (one section per helper):

  1. ``compute_surface_gamma`` -- closed-form Bikerman activity.
  2. ``_factor_log_from_species_logs`` -- replaces ``_h_factor_log``.
     Includes regression vs legacy and γ-power identity tests
     (R1 cathodic γ³, R1 anodic γ¹, R2 cathodic γ³).
  3. ``solve_stern_split`` -- bisection on the Stern Robin closure.
  4. ``picard_outer_loop`` regression -- ideal-counterion + no-Stern
     should match a hand-coded legacy reference.

See:
  - ``docs/CHATGPT_HANDOFF_12_IC_PICARD_BUGS.md``
  - ``docs/CODEX_RESPONSE_TO_HANDOFF_13.md`` (γ-power identities)
"""
from __future__ import annotations

import math

import pytest

from Forward.bv_solver.picard_ic import (
    _compute_picard_gamma_s,
    _factor_log_from_species_logs,
    _safe_exp,
    _solve_phi_o,
    _solve_picard_stern_split,
    _update_electrostatics,
    compute_surface_gamma,
    compute_surface_slope_signed,
    picard_outer_loop,
    solve_stern_split,
)


# ---------------------------------------------------------------------------
# Production-stack constants (sourced from scripts/_bv_common.py)
# ---------------------------------------------------------------------------

from scripts._bv_common import A_DEFAULT, C_HP_HAT, C_CLO4_HAT

LAMBDA_D = 0.05         # representative production Debye length (nondim)
EPS = LAMBDA_D ** 2     # poisson_coefficient


# ---------------------------------------------------------------------------
# Section 1 -- compute_surface_gamma
# ---------------------------------------------------------------------------

def test_gamma_is_unity_at_zero_psi():
    """psi_D = 0 -> gamma_s = 1 for any (a_h, a_cl, anchors)."""
    g = compute_surface_gamma(
        H_o=C_HP_HAT, c_clo4_bulk=C_CLO4_HAT, psi_D=0.0,
        a_h=A_DEFAULT, a_cl=A_DEFAULT, c_cl_anchor=C_CLO4_HAT,
    )
    assert math.isclose(g, 1.0, abs_tol=1e-14)


def test_gamma_unity_when_a_h_a_cl_zero():
    """Ideal-counterion limit: a_h = a_cl = 0 -> gamma_s = 1 always."""
    for psi_D in (-30.0, -5.0, 0.0, 5.0, 30.0):
        g = compute_surface_gamma(
            H_o=C_HP_HAT, c_clo4_bulk=C_CLO4_HAT, psi_D=psi_D,
            a_h=0.0, a_cl=0.0, c_cl_anchor=C_CLO4_HAT,
        )
        assert math.isclose(g, 1.0, abs_tol=1e-14), (
            f"a_h=a_cl=0 should give gamma=1 at psi_D={psi_D}, got {g}"
        )


def test_gamma_saturation_at_high_positive_psi_D():
    """Large positive psi_D: counterion saturation,
    gamma_s ~ 1 / (a_cl * c_anchor * exp(+psi_D))."""
    psi_D = 20.0
    g = compute_surface_gamma(
        H_o=C_HP_HAT, c_clo4_bulk=C_CLO4_HAT, psi_D=psi_D,
        a_h=A_DEFAULT, a_cl=A_DEFAULT, c_cl_anchor=C_CLO4_HAT,
    )
    # Approximate cap: gamma_s ~ 1 / (a_cl * c_clo4_bulk * exp(psi_D)).
    expected = 1.0 / (A_DEFAULT * C_CLO4_HAT * math.exp(psi_D))
    # Within 1% of the asymptotic limit.
    assert g > 0.0
    assert abs(g - expected) / expected < 0.01, (
        f"gamma_s={g} vs expected ~{expected} at psi_D={psi_D}"
    )


def test_gamma_cathodic_proton_saturation():
    """Large negative psi_D: H+ saturation,
    gamma_s ~ 1 / (a_h * H_o * exp(-psi_D))."""
    psi_D = -20.0
    g = compute_surface_gamma(
        H_o=C_HP_HAT, c_clo4_bulk=C_CLO4_HAT, psi_D=psi_D,
        a_h=A_DEFAULT, a_cl=A_DEFAULT, c_cl_anchor=C_CLO4_HAT,
    )
    expected = 1.0 / (A_DEFAULT * C_HP_HAT * math.exp(-psi_D))
    assert g > 0.0
    assert abs(g - expected) / expected < 0.01, (
        f"gamma_s={g} vs expected ~{expected} at psi_D={psi_D}"
    )


def test_gamma_synthesised_4sp_anchor_uses_H_o():
    """For synthesised-4sp ClO4- the outer-region anchor is H_o, not c_clo4_bulk.
    Verify the gamma formula evaluates with c_cl_anchor = H_o."""
    H_o = 1e-3
    psi_D = 10.0
    g_4sp = compute_surface_gamma(
        H_o=H_o, c_clo4_bulk=C_CLO4_HAT, psi_D=psi_D,
        a_h=A_DEFAULT, a_cl=A_DEFAULT, c_cl_anchor=H_o,
    )
    g_3sp = compute_surface_gamma(
        H_o=H_o, c_clo4_bulk=C_CLO4_HAT, psi_D=psi_D,
        a_h=A_DEFAULT, a_cl=A_DEFAULT, c_cl_anchor=C_CLO4_HAT,
    )
    # 4sp anchor (H_o = 1e-3) is much smaller than 3sp anchor (C_CLO4_HAT = 0.2),
    # so gamma_s_4sp should be larger (less saturation in denominator).
    assert g_4sp > g_3sp, (
        f"4sp anchor=H_o gives gamma={g_4sp}; 3sp anchor=c_clo4_bulk gives "
        f"gamma={g_3sp}; expected 4sp > 3sp"
    )


# ---------------------------------------------------------------------------
# Section 2 -- _factor_log_from_species_logs (regression + γ-power identities)
# ---------------------------------------------------------------------------

def _legacy_h_factor_log(H_val: float, factors: list) -> float:
    """Reference: legacy _h_factor_log restricted to species==2 (H+) factors."""
    total = 0.0
    H_log = math.log(max(H_val, 1e-300))
    for f in factors:
        if int(f["species"]) != 2:
            continue
        power = float(f["power"])
        c_ref_log = math.log(max(float(f["c_ref_nondim"]), 1e-30))
        total += power * (H_log - c_ref_log)
    return total


def _orr_r1_factors():
    """ORR R1 cathodic_conc_factors: 2x H+ (species 2)."""
    return [
        {"species": 2, "power": 2.0, "c_ref_nondim": C_HP_HAT},
    ]


def _orr_r2_factors():
    """ORR R2 cathodic_conc_factors: 2x H+ (species 2)."""
    return [
        {"species": 2, "power": 2.0, "c_ref_nondim": C_HP_HAT},
    ]


@pytest.mark.parametrize("psi_D", [-5.0, -1.0, 0.0, 1.0, 5.0, 15.0])
def test_factor_log_regression_against_legacy(psi_D):
    """With gamma=1 and log_H_rxn = log(H_o)-psi_D, the new helper reproduces
    the legacy _h_factor_log(H_s, factors) where H_s = H_o*exp(-psi_D)."""
    H_o = C_HP_HAT
    H_s_legacy = max(H_o * math.exp(-psi_D), 1e-300)

    # gamma-free reaction-plane logs (a_h = a_cl = 0)
    log_O_rxn = math.log(1.0)        # not used by H+ factors
    log_P_rxn = math.log(1e-4)       # not used by H+ factors
    log_H_rxn = math.log(H_o) - psi_D  # gamma=1 path
    log_by_species = [log_O_rxn, log_P_rxn, log_H_rxn]

    for factors in (_orr_r1_factors(), _orr_r2_factors()):
        legacy = _legacy_h_factor_log(H_s_legacy, factors)
        new = _factor_log_from_species_logs(log_by_species, factors)
        assert math.isclose(legacy, new, rel_tol=1e-12, abs_tol=1e-12), (
            f"regression mismatch at psi_D={psi_D}: legacy={legacy}, new={new}"
        )


def test_factor_log_linearity_in_species_log_offset():
    """Adding a constant offset delta to every log_by_species entry shifts
    the result by delta * sum(power) over all factors used."""
    factors = _orr_r1_factors()
    log_by_species_base = [math.log(1.0), math.log(1e-4), math.log(C_HP_HAT)]
    log_by_species_shifted = [v + 0.5 for v in log_by_species_base]

    base = _factor_log_from_species_logs(log_by_species_base, factors)
    shifted = _factor_log_from_species_logs(log_by_species_shifted, factors)
    delta = shifted - base

    # All R1 factors are species==2 (power 2.0), so delta = 0.5 * 2.0 = 1.0
    expected_delta = sum(float(f["power"]) for f in factors) * 0.5
    assert math.isclose(delta, expected_delta, rel_tol=1e-12)


def _build_log_A1(*, k1, log_gamma, log_h_factor1, a1, n_e, eta1):
    """Helper: log_A1 in the gamma-aware formulation
    (matches picard_ic.picard_outer_loop construction)."""
    return math.log(k1) + log_gamma + log_h_factor1 - a1 * n_e * eta1


def _build_log_B1(*, k1, log_gamma, a1, n_e, eta1):
    return math.log(k1) + log_gamma + (1.0 - a1) * n_e * eta1


def test_gamma_power_identities_R1_cathodic_gamma3():
    """R1 cathodic carries gamma^(1+H_power) -- with H_power=2 in production
    that is gamma^3 (Codex's off-by-one guard)."""
    k1 = 1.2e-3
    a1 = 0.627
    n_e = 2.0
    eta1 = 5.0
    O_s = 1.0
    psi_D = 8.0
    H_o = C_HP_HAT

    factors = _orr_r1_factors()
    H_power = sum(int(f["power"]) for f in factors)  # = 2 in production

    # gamma=1 reference (a_h = a_cl = 0)
    log_O_rxn_g1 = math.log(O_s)
    log_H_rxn_g1 = math.log(H_o) - psi_D
    log_by_species_g1 = [log_O_rxn_g1, math.log(1e-4), log_H_rxn_g1]
    log_h_factor1_g1 = _factor_log_from_species_logs(log_by_species_g1, factors)
    log_A1_g1 = _build_log_A1(
        k1=k1, log_gamma=0.0, log_h_factor1=log_h_factor1_g1,
        a1=a1, n_e=n_e, eta1=eta1,
    )
    A1_g1 = _safe_exp(log_A1_g1)

    # gamma-aware
    gamma_s = compute_surface_gamma(
        H_o=H_o, c_clo4_bulk=C_CLO4_HAT, psi_D=psi_D,
        a_h=A_DEFAULT, a_cl=A_DEFAULT, c_cl_anchor=C_CLO4_HAT,
    )
    log_gamma = math.log(gamma_s)
    log_O_rxn_g = math.log(O_s) + log_gamma
    log_H_rxn_g = math.log(H_o) - psi_D + log_gamma
    log_by_species_g = [log_O_rxn_g, math.log(1e-4) + log_gamma, log_H_rxn_g]
    log_h_factor1_g = _factor_log_from_species_logs(log_by_species_g, factors)
    log_A1_g = _build_log_A1(
        k1=k1, log_gamma=log_gamma, log_h_factor1=log_h_factor1_g,
        a1=a1, n_e=n_e, eta1=eta1,
    )
    A1_g = _safe_exp(log_A1_g)

    # Ratio should be gamma^(1 + H_power) = gamma^3 in production
    ratio = A1_g / A1_g1
    expected = gamma_s ** (1 + H_power)
    assert math.isclose(ratio, expected, rel_tol=1e-10), (
        f"R1 cathodic gamma power: ratio={ratio}, expected gamma^{1+H_power}={expected} "
        f"(gamma_s={gamma_s}, H_power={H_power})"
    )


def test_gamma_power_identities_R1_anodic_gamma1():
    """R1 anodic carries gamma^1 (just the H2O2 reactant gamma shift)."""
    k1 = 1.2e-3
    a1 = 0.627
    n_e = 2.0
    eta1 = 5.0
    psi_D = 8.0
    H_o = C_HP_HAT

    log_B1_g1 = _build_log_B1(
        k1=k1, log_gamma=0.0, a1=a1, n_e=n_e, eta1=eta1,
    )
    B1_g1 = _safe_exp(log_B1_g1)

    gamma_s = compute_surface_gamma(
        H_o=H_o, c_clo4_bulk=C_CLO4_HAT, psi_D=psi_D,
        a_h=A_DEFAULT, a_cl=A_DEFAULT, c_cl_anchor=C_CLO4_HAT,
    )
    log_gamma = math.log(gamma_s)
    log_B1_g = _build_log_B1(
        k1=k1, log_gamma=log_gamma, a1=a1, n_e=n_e, eta1=eta1,
    )
    B1_g = _safe_exp(log_B1_g)

    ratio = B1_g / B1_g1
    assert math.isclose(ratio, gamma_s, rel_tol=1e-12), (
        f"R1 anodic gamma power: ratio={ratio}, expected gamma^1={gamma_s}"
    )


def test_gamma_power_identities_R2_cathodic_gamma3():
    """R2 cathodic carries gamma^(1+H_power) = gamma^3 (same as R1)."""
    k2 = 5.0e-5
    a2 = 0.5
    n_e = 2.0
    eta2 = 8.0
    P_s = 1e-4
    psi_D = 10.0
    H_o = C_HP_HAT

    factors = _orr_r2_factors()
    H_power = sum(int(f["power"]) for f in factors)  # = 2

    # gamma=1
    log_P_rxn_g1 = math.log(P_s)
    log_H_rxn_g1 = math.log(H_o) - psi_D
    log_by_species_g1 = [math.log(1.0), log_P_rxn_g1, log_H_rxn_g1]
    log_h_factor2_g1 = _factor_log_from_species_logs(log_by_species_g1, factors)
    log_A2_g1 = math.log(k2) + 0.0 + log_h_factor2_g1 - a2 * n_e * eta2
    A2_g1 = _safe_exp(log_A2_g1)

    # gamma-aware
    gamma_s = compute_surface_gamma(
        H_o=H_o, c_clo4_bulk=C_CLO4_HAT, psi_D=psi_D,
        a_h=A_DEFAULT, a_cl=A_DEFAULT, c_cl_anchor=C_CLO4_HAT,
    )
    log_gamma = math.log(gamma_s)
    log_P_rxn_g = log_P_rxn_g1 + log_gamma
    log_H_rxn_g = log_H_rxn_g1 + log_gamma
    log_by_species_g = [math.log(1.0) + log_gamma, log_P_rxn_g, log_H_rxn_g]
    log_h_factor2_g = _factor_log_from_species_logs(log_by_species_g, factors)
    log_A2_g = math.log(k2) + log_gamma + log_h_factor2_g - a2 * n_e * eta2
    A2_g = _safe_exp(log_A2_g)

    ratio = A2_g / A2_g1
    expected = gamma_s ** (1 + H_power)
    assert math.isclose(ratio, expected, rel_tol=1e-10), (
        f"R2 cathodic gamma power: ratio={ratio}, expected gamma^{1+H_power}={expected}"
    )


# ---------------------------------------------------------------------------
# Section 3 -- solve_stern_split
# ---------------------------------------------------------------------------

def test_stern_split_zero_capacitance_returns_no_split():
    """stern_coeff = 0 -> psi_S = 0, phi_surface = phi_applied (no-Stern)."""
    psi_S, psi_D, phi_surface = solve_stern_split(
        phi_applied_model=20.0, phi_o=-5.0,
        lambda_D=LAMBDA_D, c_clo4_bulk=C_CLO4_HAT, a_cl=A_DEFAULT,
        stern_coeff_nondim=0.0, eps_nondim=EPS,
    )
    assert psi_S == 0.0
    assert math.isclose(psi_D, 25.0, rel_tol=1e-12)  # full drop
    assert math.isclose(phi_surface, 20.0, rel_tol=1e-12)


def test_stern_split_huge_capacitance_collapses_psi_S():
    """stern_coeff -> infinity -> psi_S -> 0 (Stern layer transparent)."""
    psi_S, psi_D, phi_surface = solve_stern_split(
        phi_applied_model=20.0, phi_o=-5.0,
        lambda_D=LAMBDA_D, c_clo4_bulk=C_CLO4_HAT, a_cl=A_DEFAULT,
        stern_coeff_nondim=1e8, eps_nondim=EPS,
    )
    assert abs(psi_S) < 1e-3, f"huge stern_coeff should give psi_S~0, got {psi_S}"
    # psi_D should absorb the full drop
    assert abs(psi_D - 25.0) < 1e-3


def test_stern_split_finite_positive_psi_S_in_range():
    """Finite Stern, positive full_drop -> 0 < psi_S < full_drop, phi_surface < phi_applied."""
    phi_app = 20.0
    phi_o = -5.0
    psi_S, psi_D, phi_surface = solve_stern_split(
        phi_applied_model=phi_app, phi_o=phi_o,
        lambda_D=LAMBDA_D, c_clo4_bulk=C_CLO4_HAT, a_cl=A_DEFAULT,
        stern_coeff_nondim=2.0, eps_nondim=EPS,
    )
    full_drop = phi_app - phi_o
    assert 0.0 < psi_S < full_drop
    assert 0.0 < psi_D < full_drop
    assert phi_surface < phi_app
    assert math.isclose(psi_S + psi_D, full_drop, rel_tol=1e-9)


def test_stern_split_sign_symmetry():
    """Negation: solve(-phi_app, -phi_o) = -solve(+phi_app, +phi_o) componentwise."""
    psi_S_pos, psi_D_pos, phi_surface_pos = solve_stern_split(
        phi_applied_model=15.0, phi_o=-3.0,
        lambda_D=LAMBDA_D, c_clo4_bulk=C_CLO4_HAT, a_cl=A_DEFAULT,
        stern_coeff_nondim=2.0, eps_nondim=EPS,
    )
    psi_S_neg, psi_D_neg, phi_surface_neg = solve_stern_split(
        phi_applied_model=-15.0, phi_o=+3.0,
        lambda_D=LAMBDA_D, c_clo4_bulk=C_CLO4_HAT, a_cl=A_DEFAULT,
        stern_coeff_nondim=2.0, eps_nondim=EPS,
    )
    # Sign-flipped on all three components.
    assert math.isclose(psi_S_neg, -psi_S_pos, rel_tol=1e-9)
    assert math.isclose(psi_D_neg, -psi_D_pos, rel_tol=1e-9)
    assert math.isclose(phi_surface_neg, -phi_surface_pos, rel_tol=1e-9)


def test_stern_split_robin_identity_holds():
    """At the bisection solution, Robin closure
    | stern_coeff * psi_S - eps * surface_slope_signed(psi_D) | < tol."""
    phi_app = 15.0
    phi_o = -3.0
    stern_coeff = 2.0
    a_cl = A_DEFAULT
    psi_S, psi_D, _ = solve_stern_split(
        phi_applied_model=phi_app, phi_o=phi_o,
        lambda_D=LAMBDA_D, c_clo4_bulk=C_CLO4_HAT, a_cl=a_cl,
        stern_coeff_nondim=stern_coeff, eps_nondim=EPS,
    )
    nu_charged = 2.0 * a_cl * C_CLO4_HAT
    slope_signed = compute_surface_slope_signed(psi_D, LAMBDA_D, nu_charged)
    residual = stern_coeff * psi_S - EPS * slope_signed
    # Residual normalized by the bisection scale.
    scale = max(abs(stern_coeff * (phi_app - phi_o)), 1.0)
    assert abs(residual) / scale < 1e-10, (
        f"Robin identity violated: residual={residual:.3e}, scale={scale:.3e}"
    )


def test_stern_split_ideal_branch_uses_GC_when_a_cl_zero():
    """a_cl=0 -> nu_charged=0 -> GC slope (sinh-based), not BKSA log-formula."""
    phi_app = 5.0
    phi_o = -1.0
    psi_S, psi_D, _ = solve_stern_split(
        phi_applied_model=phi_app, phi_o=phi_o,
        lambda_D=LAMBDA_D, c_clo4_bulk=C_CLO4_HAT, a_cl=0.0,
        stern_coeff_nondim=2.0, eps_nondim=EPS,
    )
    # With a_cl=0, surface_slope = (2/lam_D) * sinh(psi_D/2)
    expected_slope = (2.0 / LAMBDA_D) * math.sinh(psi_D / 2.0)
    actual_slope = compute_surface_slope_signed(psi_D, LAMBDA_D, 0.0)
    assert math.isclose(expected_slope, actual_slope, rel_tol=1e-12)


def test_surface_slope_signed_zero_at_zero_psi():
    """surface_slope(psi_D=0) = 0."""
    s = compute_surface_slope_signed(0.0, LAMBDA_D, 2.0 * A_DEFAULT * C_CLO4_HAT)
    assert s == 0.0


def test_surface_slope_signed_sign_matches_psi_D():
    """sign(slope) = sign(psi_D)."""
    nu = 2.0 * A_DEFAULT * C_CLO4_HAT
    for psi_D in (-10.0, -1.0, 1.0, 10.0):
        s = compute_surface_slope_signed(psi_D, LAMBDA_D, nu)
        assert s * psi_D > 0.0, f"sign mismatch at psi_D={psi_D}: slope={s}"


# ---------------------------------------------------------------------------
# Section 4 -- picard_outer_loop regression vs legacy gamma-free reference
# ---------------------------------------------------------------------------

def _legacy_picard_reference(*, k1, k2, a1, a2, n_e, E1, E2,
                              h_factor1, h_factor2,
                              phi_applied_model, bv_exp_scale, exponent_clip,
                              clip_exponent,
                              H_b, O_b, P_b, D_O, D_P, D_H, P_FLOOR,
                              c_clo4_bulk, omega=0.5, max_iters=50, tol=1e-6):
    """Reference Python implementation of the legacy gamma-free Picard.

    Mirrors the in-tree code at ``forms_logc.py`` lines 718-814 prior to
    the refactor.  Used in the regression test below.
    """
    def _eta_clipped(E):
        eta = bv_exp_scale * (phi_applied_model - E)
        if clip_exponent:
            return max(min(eta, exponent_clip), -exponent_clip)
        return eta

    def _h_factor_log(H_val, factors):
        total = 0.0
        H_log = math.log(max(H_val, 1e-300))
        for f in factors:
            if int(f["species"]) != 2:
                continue
            power = float(f["power"])
            c_ref_log = math.log(max(float(f["c_ref_nondim"]), 1e-30))
            total += power * (H_log - c_ref_log)
        return total

    eta1 = _eta_clipped(E1)
    eta2 = _eta_clipped(E2)

    R1 = 0.0
    R2 = 0.0
    H_o = H_b
    phi_o = 0.0
    psi_D = phi_applied_model - phi_o
    O_s = O_b
    P_s = P_b

    for k in range(1, max_iters + 1):
        R1_old, R2_old = R1, R2

        H_s = max(H_o * _safe_exp(-psi_D), 1e-300)
        log_h_factor1 = _h_factor_log(H_s, h_factor1)
        log_h_factor2 = _h_factor_log(H_s, h_factor2)
        log_A1 = math.log(k1) + log_h_factor1 - a1 * n_e * eta1
        log_B1 = math.log(k1) + (1.0 - a1) * n_e * eta1
        log_A2 = math.log(k2) + log_h_factor2 - a2 * n_e * eta2
        A1 = _safe_exp(log_A1)
        B1 = _safe_exp(log_B1)
        A2 = _safe_exp(log_A2)

        m11 = 1.0 + A1 / D_O + B1 / D_P
        m12 = -B1 / D_P
        m21 = -A2 / D_P
        m22 = 1.0 + A2 / D_P
        rhs1 = A1 * O_b - B1 * P_b
        rhs2 = A2 * P_b
        det = m11 * m22 - m12 * m21
        if not math.isfinite(det) or abs(det) < 1e-300:
            return False, k, None
        R1_new = (m22 * rhs1 - m12 * rhs2) / det
        R2_new = (-m21 * rhs1 + m11 * rhs2) / det
        if not (math.isfinite(R1_new) and math.isfinite(R2_new)):
            return False, k, None

        R1 = (1.0 - omega) * R1 + omega * R1_new
        R2 = (1.0 - omega) * R2 + omega * R2_new

        O_s = max(O_b - R1 / D_O, 1e-300)
        P_s = max(P_b + (R1 - R2) / D_P, P_FLOOR)
        H_o = max(H_b - (R1 + R2) / D_H, 1e-300)
        phi_o = math.log(H_o / c_clo4_bulk)
        psi_D = phi_applied_model - phi_o

        denom1 = max(abs(R1), 1e-30)
        denom2 = max(abs(R2), 1e-30)
        delta = abs(R1 - R1_old) / denom1 + abs(R2 - R2_old) / denom2
        if delta < tol:
            return True, k, {
                "R1": R1, "R2": R2,
                "O_s": O_s, "P_s": P_s, "H_o": H_o,
                "phi_o": phi_o, "psi_D": psi_D,
            }

    return False, max_iters, None


# Production parameters (matching scripts/_bv_common.py)
_PICARD_REGRESSION_PARAMS = dict(
    k1=1.2e-3, k2=5.0e-5,
    a1=0.627, a2=0.5,
    n_e=2.0,
    E1=26.47,    # 0.68 V / V_T = 0.68 / 0.02569
    E2=69.27,    # 1.78 V / V_T
    h_factor1=_orr_r1_factors(),
    h_factor2=_orr_r2_factors(),
    bv_exp_scale=1.0,
    exponent_clip=100.0,
    clip_exponent=True,
    H_b=C_HP_HAT,
    O_b=1.0,
    P_b=1e-4,
    D_O=1.0, D_P=1.0, D_H=1.0,  # rough placeholder values for unit test
    P_FLOOR=1e-4,
    c_clo4_bulk=C_CLO4_HAT,
)


def test_picard_outer_loop_regression_matches_legacy_at_V05():
    """Ideal-counterion + no-Stern Picard reproduces the legacy gamma-free
    Picard at V_RHE = +0.5 V (phi_applied = +19.46 nondim units).

    Tolerance: R1, R2 (the boundary BV rates Picard solves for) match
    to 1e-12 -- the 2x2 algebra is byte-equivalent.  O_s, P_s, H_o
    (the surface concentrations) match to 1e-8 -- the new code
    recomputes these from the closed-form 2x2 fixed point at the end of
    the loop (P_s = (D_P*P_b + R1)/(D_P+A2); O_s analogously) instead
    of the loop's bulk-update ``P_b + (R1-R2)/D_P``, fixing a
    near-cancellation in the diffusion-limited regime that breaks
    rate-consistency for R2 at high V with Stern.  The two formulas
    agree at the fixed point in exact arithmetic; they differ at the
    1e-8 - 1e-9 level due to floating-point.
    """
    phi_applied_model = 0.5 / 0.02569  # ~+19.46

    ok_legacy, n_legacy, st_legacy = _legacy_picard_reference(
        phi_applied_model=phi_applied_model,
        **_PICARD_REGRESSION_PARAMS,
    )
    assert ok_legacy, "legacy reference must converge at V=+0.5 V"

    ok_new, _, n_new, st_new = picard_outer_loop(
        phi_applied_model=phi_applied_model,
        a_h=0.0, a_cl=0.0, c_cl_anchor_kind="bulk", stern_split=None,
        **_PICARD_REGRESSION_PARAMS,
    )
    assert ok_new, "new picard_outer_loop must converge at V=+0.5 V"

    # Same iteration count (deterministic; same scalar arithmetic in the loop).
    assert n_legacy == n_new, (
        f"iter count differs: legacy={n_legacy}, new={n_new}"
    )

    # R1, R2: byte-equivalent (the 2x2 algebra unchanged).
    for key in ("R1", "R2"):
        legacy_val = st_legacy[key]
        new_val = st_new[key]
        assert math.isclose(legacy_val, new_val, rel_tol=1e-12, abs_tol=1e-12), (
            f"{key} differs: legacy={legacy_val}, new={new_val}"
        )
    # H_o, phi_o, psi_D: byte-equivalent (loop bulk update path unchanged).
    for key in ("H_o", "phi_o", "psi_D"):
        legacy_val = st_legacy[key]
        new_val = st_new[key]
        assert math.isclose(legacy_val, new_val, rel_tol=1e-12, abs_tol=1e-12), (
            f"{key} differs: legacy={legacy_val}, new={new_val}"
        )
    # O_s, P_s: recomputed from closed form (fp noise allowed).
    for key in ("O_s", "P_s"):
        legacy_val = st_legacy[key]
        new_val = st_new[key]
        assert math.isclose(legacy_val, new_val, rel_tol=1e-8, abs_tol=1e-12), (
            f"{key} differs beyond fp tolerance: legacy={legacy_val}, new={new_val}"
        )


@pytest.mark.parametrize("V_RHE", [0.3, 0.4, 0.5, 0.6, 0.7])
def test_picard_outer_loop_regression_anodic_voltages(V_RHE):
    """Ideal-counterion + no-Stern Picard byte-equivalent across the
    converged-Picard anodic V grid (R1, R2, H_o, phi_o, psi_D match
    legacy at 1e-12; O_s, P_s match at 1e-8 -- see V05 test docstring)."""
    phi_applied_model = V_RHE / 0.02569

    ok_legacy, _, st_legacy = _legacy_picard_reference(
        phi_applied_model=phi_applied_model,
        **_PICARD_REGRESSION_PARAMS,
    )
    ok_new, _, _, st_new = picard_outer_loop(
        phi_applied_model=phi_applied_model,
        a_h=0.0, a_cl=0.0, c_cl_anchor_kind="bulk", stern_split=None,
        **_PICARD_REGRESSION_PARAMS,
    )
    assert ok_legacy == ok_new, (
        f"convergence status differs at V_RHE={V_RHE}: legacy={ok_legacy}, new={ok_new}"
    )
    if not ok_legacy:
        return  # both diverged; nothing more to check

    for key in ("R1", "R2", "H_o", "phi_o", "psi_D"):
        assert math.isclose(st_legacy[key], st_new[key],
                            rel_tol=1e-12, abs_tol=1e-12), (
            f"V_RHE={V_RHE}: {key} differs: {st_legacy[key]} vs {st_new[key]}"
        )
    for key in ("O_s", "P_s"):
        assert math.isclose(st_legacy[key], st_new[key],
                            rel_tol=1e-8, abs_tol=1e-12), (
            f"V_RHE={V_RHE}: {key} differs beyond fp tolerance: "
            f"{st_legacy[key]} vs {st_new[key]}"
        )


def test_picard_outer_loop_returns_consistent_eta_and_gamma():
    """Returned (eta1, eta2, gamma_s) must align with returned (psi_D, H_o)."""
    ok, _, _, st = picard_outer_loop(
        phi_applied_model=0.5 / 0.02569,
        a_h=0.0, a_cl=0.0, c_cl_anchor_kind="bulk", stern_split=None,
        **_PICARD_REGRESSION_PARAMS,
    )
    assert ok

    # No-Stern: eta_drop = phi_applied
    expected_eta1 = max(min(
        1.0 * (0.5/0.02569 - 26.47), 100.0), -100.0)
    expected_eta2 = max(min(
        1.0 * (0.5/0.02569 - 69.27), 100.0), -100.0)
    assert math.isclose(st["eta1"], expected_eta1, rel_tol=1e-12)
    assert math.isclose(st["eta2"], expected_eta2, rel_tol=1e-12)
    # No-Stern: phi_surface = phi_applied
    assert math.isclose(st["phi_surface"], 0.5/0.02569, rel_tol=1e-12)
    assert st["psi_S"] == 0.0
    # Ideal counterion (a_h = a_cl = 0): gamma_s = 1 always
    assert math.isclose(st["gamma_s"], 1.0, abs_tol=1e-14)


def test_picard_outer_loop_gamma_aware_changes_R_with_a_h_a_cl_nonzero():
    """With a_h, a_cl > 0 the Picard converges to different (R1, R2) than
    with a_h = a_cl = 0 -- this is Bug #2 (the gamma fix Phase C wires).

    This test does NOT verify correctness; it just verifies the new
    plumbing changes the answer when bikerman is enabled.
    """
    phi_applied_model = 0.5 / 0.02569

    ok_ideal, _, _, st_ideal = picard_outer_loop(
        phi_applied_model=phi_applied_model,
        a_h=0.0, a_cl=0.0, c_cl_anchor_kind="bulk", stern_split=None,
        **_PICARD_REGRESSION_PARAMS,
    )
    ok_bik, _, _, st_bik = picard_outer_loop(
        phi_applied_model=phi_applied_model,
        a_h=A_DEFAULT, a_cl=A_DEFAULT,
        c_cl_anchor_kind="bulk", stern_split=None,
        **_PICARD_REGRESSION_PARAMS,
    )
    if ok_ideal and ok_bik:
        # gamma_s != 1 with bikerman, so R1 should differ.
        assert not math.isclose(st_ideal["R1"], st_bik["R1"], rel_tol=1e-3), (
            "bikerman Picard gives same R1 as ideal -- gamma plumbing not active"
        )


# ---------------------------------------------------------------------------
# Phase 5α T1 — _solve_phi_o helper byte-equivalence
# ---------------------------------------------------------------------------

def test_solve_phi_o_single_ion_byte_equivalent_to_legacy_log():
    """Helper output must match ``log(H_o / c_clo4_bulk)`` at production
    parameter values where the floor is a no-op."""
    for H_o, c_b in [(0.1, 0.0833), (1e-5, 0.0833), (10.0, 1.0), (0.05, 0.0833)]:
        legacy = math.log(H_o / c_b)
        helper = _solve_phi_o(H_o=H_o, c_clo4_bulk=c_b)
        assert helper == legacy, (
            f"_solve_phi_o drift at H_o={H_o}, c_b={c_b}: "
            f"helper={helper!r} legacy={legacy!r}"
        )


def test_solve_phi_o_floors_protect_underflow():
    """When inputs collapse below 1e-300, helper floors them; legacy raw
    ``log(0)`` would raise. Exercises only the saturation branch — production
    inputs are always above 1e-30."""
    out = _solve_phi_o(H_o=0.0, c_clo4_bulk=0.0833)
    assert out == math.log(1e-300 / 0.0833)
    out2 = _solve_phi_o(H_o=0.05, c_clo4_bulk=0.0)
    assert out2 == math.log(0.05 / 1e-300)


# ---------------------------------------------------------------------------
# Phase 5α T2 — _solve_picard_stern_split helper byte-equivalence
# ---------------------------------------------------------------------------

def test_solve_picard_stern_split_no_stern_returns_full_drop_at_metal():
    """``stern_split=None`` -> ``(0.0, phi_applied - phi_o, phi_applied)``."""
    psi_S, psi_D, phi_surf = _solve_picard_stern_split(
        phi_applied_model=12.34,
        phi_o=0.5,
        c_clo4_bulk=0.0833,
        a_cl=0.0,
        stern_split=None,
    )
    assert psi_S == 0.0
    assert psi_D == 12.34 - 0.5
    assert phi_surf == 12.34


def test_solve_picard_stern_split_dispatches_to_solve_stern_split():
    """With a Stern dict, the helper must produce the same triple as a
    direct ``solve_stern_split`` call."""
    cfg = {"lambda_D": 0.05, "stern_coeff": 1.5, "eps": 0.0025}
    legacy = solve_stern_split(
        phi_applied_model=10.0,
        phi_o=1.5,
        lambda_D=cfg["lambda_D"],
        c_clo4_bulk=0.0833,
        a_cl=A_DEFAULT,
        stern_coeff_nondim=cfg["stern_coeff"],
        eps_nondim=cfg["eps"],
    )
    helper = _solve_picard_stern_split(
        phi_applied_model=10.0,
        phi_o=1.5,
        c_clo4_bulk=0.0833,
        a_cl=A_DEFAULT,
        stern_split=cfg,
    )
    assert helper == legacy


# ---------------------------------------------------------------------------
# Phase 5α T3 — _compute_picard_gamma_s helper byte-equivalence
# ---------------------------------------------------------------------------

def test_compute_picard_gamma_s_single_ion_matches_legacy():
    """Single-ion helper output must match a direct
    ``compute_surface_gamma`` call at every parameter point we exercise."""
    cases = [
        # (H_o, c_clo4_bulk, psi_D, a_h, a_cl, c_cl_anchor)
        (0.05, 0.0833, 0.0, 0.0, 0.0, 0.0833),     # ideal limit
        (0.05, 0.0833, 1.5, A_DEFAULT, A_DEFAULT, 0.0833),
        (1e-5, 0.0833, 10.87, A_DEFAULT, A_DEFAULT, 0.0833),  # high anodic
        (10.0, 0.0833, -3.0, A_DEFAULT, A_DEFAULT, 10.0),     # synthesised_4sp anchor
    ]
    for H_o, c_b, psi_D, a_h, a_cl, c_cl_anchor in cases:
        legacy = compute_surface_gamma(H_o, c_b, psi_D, a_h, a_cl, c_cl_anchor)
        helper = _compute_picard_gamma_s(
            H_o=H_o, c_clo4_bulk=c_b, psi_D=psi_D,
            a_h=a_h, a_cl=a_cl, c_cl_anchor=c_cl_anchor,
        )
        assert helper == legacy, (
            f"_compute_picard_gamma_s drift at H_o={H_o}, c_b={c_b}, "
            f"psi_D={psi_D}: helper={helper!r} legacy={legacy!r}"
        )


# ---------------------------------------------------------------------------
# Phase 5α T4 — _update_electrostatics consolidation + R3 sign correction
# ---------------------------------------------------------------------------

def _orr_reactions_for_eta():
    """Minimal 2-rxn list for ``_eta_list_from_drop`` exercises."""
    return [
        {"E_eq_model": 0.0},
        {"E_eq_model": 1.5},
    ]


def test_update_electrostatics_no_stern_eta_drop_is_phi_applied():
    """R3 sign correction (validated by GPT critique):
    no-Stern eta_drop = phi_applied_model (NOT phi_applied - phi_o)."""
    out = _update_electrostatics(
        c_s=[1.0, 1e-5, 0.05],
        h_idx=2,
        c_clo4_bulk=0.0833,
        phi_o=-0.51,
        phi_applied_model=21.4,
        a_h=A_DEFAULT,
        a_cl=A_DEFAULT,
        c_cl_anchor_kind="bulk",
        stern_split=None,
        reactions=_orr_reactions_for_eta(),
        bv_exp_scale=1.0,
        exponent_clip=100.0,
        clip_exponent=True,
    )
    assert out["eta_drop"] == 21.4   # NOT 21.4 - phi_o(=21.91)
    assert out["psi_S"] == 0.0
    assert out["psi_D"] == 21.4 - (-0.51)
    assert out["phi_surface"] == 21.4


def test_update_electrostatics_stern_eta_drop_is_psi_S():
    """Stern: eta_drop = psi_S, the Stern-layer drop."""
    cfg = {"lambda_D": 0.05, "stern_coeff": 1.5, "eps": 0.0025}
    out = _update_electrostatics(
        c_s=[1.0, 1e-5, 0.05],
        h_idx=2,
        c_clo4_bulk=0.0833,
        phi_o=-0.51,
        phi_applied_model=21.4,
        a_h=A_DEFAULT,
        a_cl=A_DEFAULT,
        c_cl_anchor_kind="bulk",
        stern_split=cfg,
        reactions=_orr_reactions_for_eta(),
        bv_exp_scale=1.0,
        exponent_clip=100.0,
        clip_exponent=True,
    )
    assert out["eta_drop"] == out["psi_S"]
    # |psi_S| < |full_drop| = 21.91 (Stern partitions)
    assert abs(out["psi_S"]) < abs(21.4 - (-0.51))


def test_update_electrostatics_eta_list_matches_direct_eta_list_from_drop():
    """The bundled eta_list must equal a direct _eta_list_from_drop call
    at the same eta_drop, for byte-equivalence of the in-loop block."""
    from Forward.bv_solver.picard_ic import _eta_list_from_drop
    rxns = _orr_reactions_for_eta()
    out = _update_electrostatics(
        c_s=[1.0, 1e-5, 0.05],
        h_idx=2,
        c_clo4_bulk=0.0833,
        phi_o=-0.51,
        phi_applied_model=12.0,
        a_h=A_DEFAULT,
        a_cl=A_DEFAULT,
        c_cl_anchor_kind="bulk",
        stern_split=None,
        reactions=rxns,
        bv_exp_scale=1.0,
        exponent_clip=100.0,
        clip_exponent=True,
    )
    expected = _eta_list_from_drop(
        eta_drop=12.0,
        reactions=rxns,
        bv_exp_scale=1.0,
        exponent_clip=100.0,
        clip_exponent=True,
    )
    assert out["eta_list"] == expected
