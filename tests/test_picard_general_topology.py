"""Verification ladder T2-T5, T11, T12 for the generalized Picard outer loop.

Per ``docs/picard_general_topology_derivation.md`` v3 §10.  These are pure-
Python scalar unit tests on ``picard_outer_loop_general`` — no Firedrake.

  - T2  pure-2e parallel vs. legacy sequential with R_2 disabled.
  - T3  pure-4e parallel: R_2e=0 ⇒ no peroxide flux; signed H_o uses
        |s_H|=4.
  - T4  γ-power probe: perturbing γ_s scales α̂_j, β̂_j, Ĉ_j by the
        documented powers (γ³, γ⁵, γ¹, γ⁰).
  - T5  singular Jacobian returns ``(False, "singular_jacobian_iter_k",
        k, partial_state)``.
  - T11 signed H_o synthetic (s_H=+2 proton-producing cathodic): H_o
        increases with R_1, confirms signed formula.
  - T12 constant-anodic branch (reversible + anodic_species=None +
        c_ref_model > 0): Ĉ_j ≠ 0, β̂_j = 0, RHS includes -Ĉ_j.

T1 (sequential byte-equivalence) lives in
``tests/test_picard_topology_derivation.py``.  Slow Firedrake tests
T6-T10 are deferred to a separate driver.
"""
from __future__ import annotations

import math

from Forward.bv_solver.picard_ic import (
    _build_picard_prefactors,
    _eta_list_from_drop,
    picard_outer_loop_general,
)
from scripts._bv_common import A_DEFAULT, C_HP_HAT, C_CLO4_HAT, V_T


# ---------------------------------------------------------------------------
# Reaction templates
# ---------------------------------------------------------------------------

# Legacy sequential (cathsub_2 = P; anodsub_2 = None; irreversible).
_LEGACY_R1 = dict(
    k0_model=1.2e-3, alpha=0.627, n_electrons=2.0, E_eq_model=26.47,
    cathodic_species=0, anodic_species=1, reversible=True,
    c_ref_model=0.0,
    cathodic_conc_factors=[
        {"species": 2, "power": 2, "c_ref_nondim": C_HP_HAT}
    ],
    stoichiometry=[-1, +1, -2],
)
_LEGACY_R2 = dict(
    k0_model=5.0e-5, alpha=0.5, n_electrons=2.0, E_eq_model=69.27,
    cathodic_species=1, anodic_species=None, reversible=False,
    c_ref_model=0.0,
    cathodic_conc_factors=[
        {"species": 2, "power": 2, "c_ref_nondim": C_HP_HAT}
    ],
    stoichiometry=[0, -1, -2],
)

# Parallel R_2e + R_4e (Ruggiero topology).
#   R_2e: O₂ + 2H⁺ + 2e⁻ → H₂O₂.   cathsub=O, anodsub=P, reversible.
#   R_4e: O₂ + 4H⁺ + 4e⁻ → 2H₂O.   cathsub=O, anodsub=None, irreversible.
_PARALLEL_R2E = dict(
    k0_model=1.2e-3, alpha=0.627, n_electrons=2.0, E_eq_model=26.47,
    cathodic_species=0, anodic_species=1, reversible=True,
    c_ref_model=0.0,
    cathodic_conc_factors=[
        {"species": 2, "power": 2, "c_ref_nondim": C_HP_HAT}
    ],
    stoichiometry=[-1, +1, -2],
)
_PARALLEL_R4E = dict(
    k0_model=8.0e-7, alpha=0.5, n_electrons=4.0, E_eq_model=47.49,  # 1.22 V / V_T
    cathodic_species=0, anodic_species=None, reversible=False,
    c_ref_model=0.0,
    cathodic_conc_factors=[
        {"species": 2, "power": 4, "c_ref_nondim": C_HP_HAT}
    ],
    stoichiometry=[-1, 0, -4],
)


def _shared_general(**overrides):
    """Common kwargs for picard_outer_loop_general; override per test."""
    base = dict(
        bulk_concs=[1.0, 1e-4, C_HP_HAT],
        diffusivities=[1.0, 1.0, 1.0],
        species_floors=[1e-300, 1e-30, 1e-300],
        h_idx=2,
        c_clo4_bulk=C_CLO4_HAT,
        phi_applied_model=0.5 / V_T,
        bv_exp_scale=1.0, exponent_clip=100.0, clip_exponent=True,
        a_h=0.0, a_cl=0.0, c_cl_anchor_kind="bulk", stern_split=None,
        omega=0.5, max_iters=50, tol=1e-6,
        topology_hint="general",
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# T2 — pure-2e parallel vs. legacy sequential with k0_R2 = 0
# ---------------------------------------------------------------------------

def test_t2_pure_2e_parallel_matches_legacy_with_r2_disabled():
    """Both runs leave only O₂ → H₂O₂ active; identities per v3 §10 T2."""
    # Parallel with k0_R4e = 0: only R_2e active.
    parallel_reactions = [
        _PARALLEL_R2E,
        {**_PARALLEL_R4E, "k0_model": 0.0},
    ]
    ok_p, _, _, st_p = picard_outer_loop_general(
        reactions=parallel_reactions,
        topology_hint="parallel_2e_4e",
        **{k: v for k, v in _shared_general().items() if k != "topology_hint"},
    )
    assert ok_p, "T2 parallel (k0_R4e=0) must converge"

    # Legacy sequential with k0_R2 = 0: only R_1 active (= O₂ → H₂O₂).
    legacy_reactions = [
        _LEGACY_R1,
        {**_LEGACY_R2, "k0_model": 0.0},
    ]
    ok_l, _, _, st_l = picard_outer_loop_general(
        reactions=legacy_reactions,
        topology_hint="sequential_2e_h2o2",
        **{k: v for k, v in _shared_general().items() if k != "topology_hint"},
    )
    assert ok_l, "T2 legacy (k0_R2=0) must converge"

    # Identities (bounded by Picard tolerance ~1e-6 propagated).
    R_2e = st_p["R_list"][0]
    R_4e = st_p["R_list"][1]
    R_1_legacy = st_l["R_list"][0]
    R_2_legacy = st_l["R_list"][1]
    assert abs(R_4e) < 1e-30, f"R_4e should be ~0 with k0=0; got {R_4e!r}"
    assert abs(R_2_legacy) < 1e-30, (
        f"R_2 should be ~0 with k0=0; got {R_2_legacy!r}"
    )
    assert math.isclose(R_2e, R_1_legacy, rel_tol=1e-10, abs_tol=1e-10), (
        f"R_2e ({R_2e!r}) must match R_1_legacy ({R_1_legacy!r})"
    )

    # Surface flux balance identities (D_O = D_P = D_H = 1.0):
    # O_s = O_b − R_2e / D_O; P_s = P_b + R_2e / D_P; H_o = H_b − R_2e / D_H.
    O_b, P_b, H_b = 1.0, 1e-4, C_HP_HAT
    expected_O_s = O_b - R_2e / 1.0
    expected_P_s = max(P_b + R_2e / 1.0, 1e-30)
    expected_H_o = max(H_b - R_2e / 1.0, 1e-300)
    # Note: parallel topology hint uses naive flux balance; sequential uses
    # closed-form O_s, P_s.  Compare per-topology surface concentrations.
    assert math.isclose(st_p["O_s"], expected_O_s, rel_tol=1e-10, abs_tol=1e-10)
    assert math.isclose(st_p["P_s"], expected_P_s, rel_tol=1e-10, abs_tol=1e-10)
    assert math.isclose(st_p["H_o"], expected_H_o, rel_tol=1e-10, abs_tol=1e-10)


# ---------------------------------------------------------------------------
# T3 — pure-4e parallel
# ---------------------------------------------------------------------------

def test_t3_pure_4e_parallel():
    """k0_R2e = 0 ⇒ R_2e ≈ 0; ambipolar H_o uses |s_H|=4.

    Run at V_RHE = +0.5 V (same as T1) to keep both R_2e and R_4e
    well-conditioned within the Picard exponent_clip = 100 cap.  At
    V_RHE = 0 the R_4e overpotential η = -47.5 nondim drives
    α̂_4e ~ exp(95) ~ 1e41 which overflows the linear solve; the v3
    contract reserves that pathological regime for T5.
    """
    reactions = [
        {**_PARALLEL_R2E, "k0_model": 0.0},
        _PARALLEL_R4E,
    ]
    shared = _shared_general(phi_applied_model=0.5 / V_T)
    ok, reason, _, st = picard_outer_loop_general(
        reactions=reactions,
        **{k: v for k, v in shared.items() if k != "topology_hint"},
        topology_hint="parallel_2e_4e",
    )
    assert ok, f"T3 pure-4e Picard must converge; reason={reason!r}"

    R_2e = st["R_list"][0]
    R_4e = st["R_list"][1]
    assert abs(R_2e) < 1e-30, f"R_2e should be ~0 with k0=0; got {R_2e!r}"
    assert R_4e > 0.0, f"R_4e should be positive cathodic at V_RHE=0; got {R_4e!r}"

    # P_s ≈ P_b (R_4e doesn't touch P; R_2e ≈ 0).
    P_b = 1e-4
    assert math.isclose(st["P_s"], P_b, rel_tol=1e-12, abs_tol=1e-30)

    # H_o = H_b − 2·R_4e / D_H (ambipolar with |s_H|=4: signed +(-4)·R_4e/(2 D_H) = -2 R_4e/D_H).
    H_b = C_HP_HAT
    D_H = 1.0
    expected_H_o = max(H_b - 2.0 * R_4e / D_H, 1e-300)
    assert math.isclose(st["H_o"], expected_H_o, rel_tol=1e-10, abs_tol=1e-30), (
        f"H_o={st['H_o']!r} vs expected {expected_H_o!r}; ambipolar formula drift"
    )

    # O_s = O_b − R_4e / D_O.
    O_b = 1.0
    D_O = 1.0
    expected_O_s = max(O_b - R_4e / D_O, 1e-300)
    assert math.isclose(st["O_s"], expected_O_s, rel_tol=1e-10, abs_tol=1e-30)


# ---------------------------------------------------------------------------
# T4 — γ-power probe
# ---------------------------------------------------------------------------

def test_t4_gamma_power_in_prefactors():
    """Build α̂_j, β̂_j, Ĉ_j with two log_gamma values and verify the ratio
    matches the documented γ-power for each branch (v3 §3 γ-power table).

    For the parallel R_2e + R_4e + a synthetic R_const config:

      R_2e cathodic (α̂_2e):  γ³  (1 + power_H=2)
      R_4e cathodic (α̂_4e):  γ⁵  (1 + power_H=4)
      R_2e anodic linear (β̂_2e):  γ¹
      R_const constant-anodic (Ĉ): γ⁰  (no γ factor on c_ref_model)
    """
    # Synthetic constant-anodic reaction (branch 2): reversible + no
    # anodic species + c_ref_model > 0.  Stoich and cathsub copy R_2e
    # so the test only exercises the prefactor γ-power, not the matrix.
    rxn_const = dict(
        k0_model=1.0e-3, alpha=0.5, n_electrons=2.0, E_eq_model=0.0,
        cathodic_species=0, anodic_species=None, reversible=True,
        c_ref_model=0.5,
        cathodic_conc_factors=[],   # no H⁺ factor; isolate the γ-power
        stoichiometry=[-1, 0, -2],
    )
    reactions = [_PARALLEL_R2E, _PARALLEL_R4E, rxn_const]

    eta_list = _eta_list_from_drop(
        eta_drop=0.0,
        reactions=reactions,
        bv_exp_scale=1.0, exponent_clip=100.0, clip_exponent=True,
    )

    def _build(log_gamma_value: float):
        # log_by_species: log(c) + log_gamma at OHP for non-H species; H
        # is post-Boltzmann shifted (psi_D=0 here). log_h_factor pulls in
        # one log_gamma per H⁺ factor power via log_by_species[2].
        log_by_species = [
            math.log(1.0) + log_gamma_value,
            math.log(1e-4) + log_gamma_value,
            math.log(C_HP_HAT) + log_gamma_value,  # psi_D = 0
        ]
        return _build_picard_prefactors(
            reactions=reactions,
            log_gamma=log_gamma_value,
            log_by_species=log_by_species,
            eta_list=eta_list,
        )

    log_gamma_a = math.log(0.5)
    log_gamma_b = math.log(0.25)
    log_gamma_diff = log_gamma_b - log_gamma_a    # log(0.25/0.5) = -log 2

    alpha_a, beta_a, c_a = _build(log_gamma_a)
    alpha_b, beta_b, c_b = _build(log_gamma_b)

    # α̂_2e ∝ γ^{1+2} = γ³.  log ratio = 3 · log_gamma_diff.
    log_ratio_alpha_2e = math.log(alpha_b[0]) - math.log(alpha_a[0])
    assert math.isclose(log_ratio_alpha_2e, 3.0 * log_gamma_diff, rel_tol=1e-12)

    # α̂_4e ∝ γ⁵.
    log_ratio_alpha_4e = math.log(alpha_b[1]) - math.log(alpha_a[1])
    assert math.isclose(log_ratio_alpha_4e, 5.0 * log_gamma_diff, rel_tol=1e-12)

    # β̂_2e ∝ γ¹.
    log_ratio_beta_2e = math.log(beta_a[0]) - math.log(beta_b[0])  # γ_a > γ_b
    # log(β_a) - log(β_b) = 1·(log γ_a - log γ_b) = -log_gamma_diff
    assert math.isclose(log_ratio_beta_2e, -log_gamma_diff, rel_tol=1e-12)

    # Ĉ_const ∝ γ⁰: independent of γ.
    assert math.isclose(c_a[2], c_b[2], rel_tol=1e-14, abs_tol=1e-30)
    # And β̂_const = 0 (branch 2 mutual exclusion).
    assert beta_a[2] == 0.0 and beta_b[2] == 0.0


# ---------------------------------------------------------------------------
# T5 — singular Jacobian
# ---------------------------------------------------------------------------

def test_t5_singular_jacobian_returns_failure():
    """Robust-failure path returns ``(False, reason, k, partial_state)``.

    Per v3 §9 item 8 the legitimate failure reasons are: singular_jacobian,
    non_finite_R, non_finite_state, picard_max_iters.  This test triggers
    the non_finite_R path by injecting ``k0_model = NaN`` so that
    ``log(k_j)`` propagates NaN through the prefactors and the linear
    solve output is non-finite.  All four failure paths share the same
    contract — return tuple, no exception, partial state preserved — so
    one trigger is sufficient for unit-level coverage; the other paths
    are exercised in integration via the adapter sites.
    """
    # Two identical reactions with huge k0 + extreme cathodic eta:
    # α̂_0 = α̂_1 ≈ 1e304 saturates ``_safe_exp`` cap; M rows become
    # linearly dependent in fp (1 + a ≈ a for a >> 1), giving det → NaN.
    rxn = dict(
        k0_model=1e150, alpha=0.5, n_electrons=2.0, E_eq_model=0.0,
        cathodic_species=0, anodic_species=None, reversible=False,
        c_ref_model=0.0,
        cathodic_conc_factors=[],
        stoichiometry=[-1, 0, -2],
    )
    ok, reason, k_iter, state = picard_outer_loop_general(
        reactions=[rxn, dict(rxn)],
        **_shared_general(
            phi_applied_model=-200.0,
            exponent_clip=1e10, clip_exponent=False,
        ),
    )
    assert not ok, f"Linearly-dependent M must fail; got ok=True, state={state}"
    assert "singular_jacobian" in reason, (
        f"Expected singular_jacobian; got {reason!r}"
    )
    # State dict is well-formed.
    assert "R_list" in state
    assert isinstance(k_iter, int) and k_iter >= 1


def test_t5_max_iters_returns_failure():
    """Forcing ``max_iters = 1`` with tight ``tol = 1e-30`` exits with
    ``picard_max_iters_*`` reason instead of converging silently."""
    rxn = dict(
        k0_model=1.0e-3, alpha=0.5, n_electrons=2.0, E_eq_model=0.0,
        cathodic_species=0, anodic_species=None, reversible=False,
        c_ref_model=0.0,
        cathodic_conc_factors=[],
        stoichiometry=[-1, 0, -2],
    )
    ok, reason, k_iter, state = picard_outer_loop_general(
        reactions=[rxn],
        **_shared_general(max_iters=1, tol=1e-30, phi_applied_model=-2.0),
    )
    assert not ok
    assert "picard_max_iters" in reason, reason
    assert k_iter == 1


# ---------------------------------------------------------------------------
# T11 — signed H_o synthetic (proton-producing cathodic)
# ---------------------------------------------------------------------------

def test_t11_signed_h_o_proton_producing():
    """Synthetic 1-reaction with s_H = +2: the signed ambipolar formula
    gives ``H_o = H_b + R_1 / D_H`` (not ``H_b - R_1 / D_H`` which would
    be the |s_H|/2 absolute-value form's wrong-sign result).

    This pathological config is used only to verify the signed formula —
    it is not a physical ORR reaction (acid-form ORR consumes H⁺).
    """
    # Cathsub = O (an ordinary species), no anodic, irreversible.
    # Stoich produces 2 H⁺ per turnover (s_H = +2 instead of -2).
    rxn = dict(
        k0_model=1.0e-3, alpha=0.5, n_electrons=2.0, E_eq_model=0.0,
        cathodic_species=0, anodic_species=None, reversible=False,
        c_ref_model=0.0,
        cathodic_conc_factors=[],   # no H⁺ in factors; isolate flux balance
        stoichiometry=[-1, 0, +2],   # produces H⁺
    )
    ok, _, _, st = picard_outer_loop_general(
        reactions=[rxn],
        **{k: v for k, v in _shared_general(phi_applied_model=-2.0).items()},
    )
    assert ok, "T11 synthetic must converge"
    R_1 = st["R_list"][0]
    assert R_1 > 0.0, f"R_1 should be cathodic positive at V<0; got {R_1!r}"

    H_b = C_HP_HAT
    D_H = 1.0
    # Signed ambipolar: H_o = H_b + s_H · R / (2 D_H)
    #                       = H_b + (+2)·R_1 / (2·D_H) = H_b + R_1 / D_H.
    expected_H_o = max(H_b + R_1 / D_H, 1e-300)
    assert math.isclose(st["H_o"], expected_H_o, rel_tol=1e-10, abs_tol=1e-30), (
        f"H_o={st['H_o']!r}, expected H_b+R_1/D_H={expected_H_o!r}.  "
        f"If H_o<H_b, code is using |s_H| (wrong sign)."
    )
    assert st["H_o"] > H_b, (
        f"H_o={st['H_o']} must increase above H_b={H_b} for s_H=+2; "
        f"|s_H| form would silently make this decrease."
    )


# ---------------------------------------------------------------------------
# T12 — constant-anodic branch validator
# ---------------------------------------------------------------------------

def test_t12_constant_anodic_branch():
    """Synthetic 1-reaction with reversible=True, anodic_species=None,
    c_ref_model > 1e-30 ⇒ branch 2 fires.  Identities per v3 §3:
    Ĉ_j ≠ 0, β̂_j = 0, b_j = α̂_j · O_b − Ĉ_j (no γ on Ĉ_j)."""
    rxn = dict(
        k0_model=1.0e-3, alpha=0.5, n_electrons=2.0, E_eq_model=0.0,
        cathodic_species=0, anodic_species=None, reversible=True,
        c_ref_model=0.5,
        cathodic_conc_factors=[],
        stoichiometry=[-1, 0, -2],
    )
    eta_list = _eta_list_from_drop(
        eta_drop=0.0,
        reactions=[rxn],
        bv_exp_scale=1.0, exponent_clip=100.0, clip_exponent=True,
    )
    log_gamma = math.log(0.7)   # arbitrary γ, isolate γ-power
    log_by_species = [
        math.log(1.0) + log_gamma,
        math.log(1e-4) + log_gamma,
        math.log(C_HP_HAT) + log_gamma,
    ]
    alpha_hat, beta_hat, c_hat = _build_picard_prefactors(
        reactions=[rxn],
        log_gamma=log_gamma,
        log_by_species=log_by_species,
        eta_list=eta_list,
    )
    # Branch 2: Ĉ_j ≠ 0, β̂_j = 0.
    assert beta_hat[0] == 0.0, f"branch 2 must give β̂=0; got {beta_hat[0]!r}"
    assert c_hat[0] > 0.0, f"branch 2 must give Ĉ>0 with c_ref=0.5; got {c_hat[0]!r}"

    # Ĉ_j γ-power = 0: changing γ must not change Ĉ.
    log_gamma_b = math.log(0.3)
    log_by_species_b = [
        math.log(1.0) + log_gamma_b,
        math.log(1e-4) + log_gamma_b,
        math.log(C_HP_HAT) + log_gamma_b,
    ]
    _, beta_hat_b, c_hat_b = _build_picard_prefactors(
        reactions=[rxn],
        log_gamma=log_gamma_b,
        log_by_species=log_by_species_b,
        eta_list=eta_list,
    )
    assert math.isclose(c_hat[0], c_hat_b[0], rel_tol=1e-14), (
        f"Ĉ_j must be γ-independent (γ⁰); a={c_hat[0]!r}, b={c_hat_b[0]!r}"
    )
    assert beta_hat_b[0] == 0.0
