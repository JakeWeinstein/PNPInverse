"""T1 — sequential 2x2 byte-equivalence reference for the generalized
Picard outer loop.

Per ``docs/picard_general_topology_derivation.md`` v3 §10 T1.

Locks the legacy 2-reaction sequential ORR Picard output at a fixed
input state to scalar values frozen at commit ``d8bf645``.  Any future
generalization of ``picard_outer_loop`` (Task #5 of the M3a.3 plan)
must preserve byte-equivalence to ≤ 1e-12 on each scalar when called
with the legacy 2-reaction list (or its post-refactor equivalent).

Companion: ``tests/test_picard_ic_helpers.py`` already exercises the
legacy-vs-current loop equivalence by hand-coding the legacy algorithm
in-test and comparing to the in-tree implementation.  This file
complements that with a hard-coded numerical reference — if both the
legacy reference and the in-tree implementation drift in tandem (a
silent regression), this file's frozen scalars catch it.

No Firedrake dependency.  Pure-Python scalar ``picard_outer_loop``.
"""
from __future__ import annotations

import math

from Forward.bv_solver.picard_ic import (
    picard_outer_loop,
    picard_outer_loop_general,
)
from scripts._bv_common import A_DEFAULT, C_HP_HAT, C_CLO4_HAT, V_T


# ---------------------------------------------------------------------------
# Fixed input state
# ---------------------------------------------------------------------------
#
# Production-like ORR sequential 2x2 at V_RHE = +0.5 V (eta1 = -7.009 nondim,
# eta2 = -49.81 nondim → R2 unclipped at exponent_clip=100; both rates active).
# Two scenarios:
#   T1a: ideal counterion (a_h = a_cl = 0) + no Stern.
#   T1b: bikerman counterion (a_h = a_cl = A_DEFAULT) + no Stern.
# ---------------------------------------------------------------------------

_V_RHE = 0.5
_PHI_APPLIED = _V_RHE / V_T  # ~19.460872

_SHARED_PARAMS = dict(
    H_b=C_HP_HAT, O_b=1.0, P_b=1e-4,
    D_O=1.0, D_P=1.0, D_H=1.0, P_FLOOR=1e-4,
    c_clo4_bulk=C_CLO4_HAT,
    k1=1.2e-3, k2=5.0e-5,
    a1=0.627, a2=0.5, n_e=2.0,
    E1=26.47, E2=69.27,
    h_factor1=[{"species": 2, "power": 2, "c_ref_nondim": C_HP_HAT}],
    h_factor2=[{"species": 2, "power": 2, "c_ref_nondim": C_HP_HAT}],
    bv_exp_scale=1.0, exponent_clip=100.0, clip_exponent=True,
    omega=0.5, max_iters=50, tol=1e-6,
)


# Frozen reference scalars (commit d8bf645).
# Captured by running ``picard_outer_loop`` directly with the inputs above.
_T1A_REFERENCE = {
    "n_iters": 21,
    "R1": -1.7546890734725322e-10,
    "R2": 7.271866006825849e-05,
    "O_s": 1.0000000006431884,
    "P_s": 0.0001,
    "H_o": 0.083260614848734,
    "phi_o": -0.0008730027712447712,
    "psi_D": 19.461745407900025,
    "psi_S": 0.0,
    "phi_surface": 19.460872405128782,
    "gamma_s": 1.0,
    "eta1": -7.009127594871217,
    "eta2": -49.809127594871214,
}

_T1B_REFERENCE = {
    "n_iters": 21,
    "R1": -2.7275201739286983e-15,
    "R2": 2.0398103428844732e-20,
    "O_s": 1.0000000000000027,
    "P_s": 0.0001,
    "H_o": 0.08333333333333608,
    "phi_o": 3.286260152890409e-14,
    "psi_D": 19.46087240512875,
    "psi_S": 0.0,
    "phi_surface": 19.460872405128782,
    "gamma_s": 4.240625385913346e-06,
    "eta1": -7.009127594871217,
    "eta2": -49.809127594871214,
}


def _assert_state_matches_reference(
    state: dict, n_iters: int, reference: dict, *, abs_tol: float, rel_tol: float
) -> None:
    assert n_iters == reference["n_iters"], (
        f"iteration count drift: got {n_iters}, reference {reference['n_iters']}"
    )
    for key, ref_val in reference.items():
        if key == "n_iters":
            continue
        got = state[key]
        # math.isclose handles abs_tol for near-zero values (R1 ~ 1e-15) and
        # rel_tol for O(1) values (psi_D, eta1, etc.) symmetrically.
        assert math.isclose(got, ref_val, abs_tol=abs_tol, rel_tol=rel_tol), (
            f"T1 byte-equivalence drift on {key}: "
            f"got {got!r}, reference {ref_val!r}, "
            f"abs_diff={abs(got - ref_val):.3e}"
        )


def test_t1a_sequential_byte_equivalence_ideal_counterion():
    """T1a — sequential 2x2 ORR with ideal counterion + no Stern.

    Reference frozen at commit ``d8bf645``.  Tolerance ≤ 1e-12 (abs and
    rel) — floating-point reordering only.  Picks up R2 ~ 7.3e-5 on
    the cathodic side (γ_s = 1 in the ideal limit means no activity
    suppression).
    """
    ok, reason, n_iters, state = picard_outer_loop(
        phi_applied_model=_PHI_APPLIED,
        a_h=0.0, a_cl=0.0, c_cl_anchor_kind="bulk", stern_split=None,
        **_SHARED_PARAMS,
    )
    assert ok, f"T1a Picard must converge; reason={reason!r}"
    _assert_state_matches_reference(
        state, n_iters, _T1A_REFERENCE, abs_tol=1e-12, rel_tol=1e-12,
    )


def test_t1b_sequential_byte_equivalence_bikerman_counterion():
    """T1b — sequential 2x2 ORR with bikerman counterion + no Stern.

    Reference frozen at commit ``d8bf645``.  Tolerance ≤ 1e-12 (abs and
    rel) — floating-point reordering only.  Picks up γ_s ~ 4.2e-6 from
    diffuse-layer counterion saturation, so the cathodic prefactors carry
    γ³ ~ 7.6e-17 (R1 cathodic) and rates collapse to O(1e-15).  This is
    physically expected at V_RHE = +0.5 V with the production a_h = a_cl
    = 0.01 — the test pins the exact floating-point output of the
    saturated regime, which is the most numerically delicate path the
    refactor must preserve.
    """
    ok, reason, n_iters, state = picard_outer_loop(
        phi_applied_model=_PHI_APPLIED,
        a_h=A_DEFAULT, a_cl=A_DEFAULT, c_cl_anchor_kind="bulk", stern_split=None,
        **_SHARED_PARAMS,
    )
    assert ok, f"T1b Picard must converge; reason={reason!r}"
    _assert_state_matches_reference(
        state, n_iters, _T1B_REFERENCE, abs_tol=1e-12, rel_tol=1e-12,
    )


# ---------------------------------------------------------------------------
# T1 byte-equivalence on the generalized N-reaction loop.
#
# Reproduces the legacy 2x2 sequential output exactly when called with the
# legacy reaction list and ``topology_hint='sequential_2e_h2o2'``.  This is
# the v3 §5/§9-item-1 byte-equivalence contract: the new loop must collapse
# to the legacy algebra at floating-point precision.
# ---------------------------------------------------------------------------

# Legacy 2-reaction sequential ORR per derivation v3 §5:
#   R1: O₂ + 2H⁺ + 2e⁻ → H₂O₂      (cathsub=O, anodsub=P, reversible)
#   R2: H₂O₂ + 2H⁺ + 2e⁻ → 2H₂O   (cathsub=P, anodsub=None, irreversible)
# Species ordering: 0=O, 1=P, 2=H.
_LEGACY_SEQUENTIAL_REACTIONS = [
    dict(
        k0_model=1.2e-3, alpha=0.627, n_electrons=2.0, E_eq_model=26.47,
        cathodic_species=0, anodic_species=1, reversible=True,
        c_ref_model=0.0,
        cathodic_conc_factors=[
            {"species": 2, "power": 2, "c_ref_nondim": C_HP_HAT}
        ],
        stoichiometry=[-1, +1, -2],
    ),
    dict(
        k0_model=5.0e-5, alpha=0.5, n_electrons=2.0, E_eq_model=69.27,
        cathodic_species=1, anodic_species=None, reversible=False,
        c_ref_model=0.0,
        cathodic_conc_factors=[
            {"species": 2, "power": 2, "c_ref_nondim": C_HP_HAT}
        ],
        stoichiometry=[0, -1, -2],
    ),
]

_GENERAL_SHARED_PARAMS = dict(
    bulk_concs=[1.0, 1e-4, C_HP_HAT],
    diffusivities=[1.0, 1.0, 1.0],
    species_floors=[1e-300, 1e-4, 1e-300],
    h_idx=2,
    c_clo4_bulk=C_CLO4_HAT,
    bv_exp_scale=1.0, exponent_clip=100.0, clip_exponent=True,
    omega=0.5, max_iters=50, tol=1e-6,
    topology_hint="sequential_2e_h2o2",
)


def test_t1a_general_matches_legacy_byte_equivalence_ideal():
    """``picard_outer_loop_general`` reproduces ``picard_outer_loop`` to
    ≤ 1e-12 on every scalar with ``topology_hint='sequential_2e_h2o2'``
    and the legacy 2-reaction list (ideal counterion, no Stern)."""
    ok_g, reason_g, n_g, st_g = picard_outer_loop_general(
        reactions=_LEGACY_SEQUENTIAL_REACTIONS,
        phi_applied_model=_PHI_APPLIED,
        a_h=0.0, a_cl=0.0, c_cl_anchor_kind="bulk", stern_split=None,
        **_GENERAL_SHARED_PARAMS,
    )
    assert ok_g, f"T1a general must converge; reason={reason_g!r}"
    _assert_state_matches_reference(
        st_g, n_g, _T1A_REFERENCE, abs_tol=1e-12, rel_tol=1e-12,
    )


def test_t1b_general_matches_legacy_byte_equivalence_bikerman():
    """As ``test_t1a_general_*`` but with the production bikerman
    (a_h = a_cl = A_DEFAULT) — exercises the γ-suppressed prefactor path
    (γ³ ~ 7.6e-17) which is the most numerically delicate."""
    ok_g, reason_g, n_g, st_g = picard_outer_loop_general(
        reactions=_LEGACY_SEQUENTIAL_REACTIONS,
        phi_applied_model=_PHI_APPLIED,
        a_h=A_DEFAULT, a_cl=A_DEFAULT, c_cl_anchor_kind="bulk", stern_split=None,
        **_GENERAL_SHARED_PARAMS,
    )
    assert ok_g, f"T1b general must converge; reason={reason_g!r}"
    _assert_state_matches_reference(
        st_g, n_g, _T1B_REFERENCE, abs_tol=1e-12, rel_tol=1e-12,
    )


def test_t1_general_h_plus_substrate_rejected():
    """v3 §9 item 11: configs with H⁺ as ``cathodic_species`` are rejected
    with reason ``h_plus_as_linear_substrate``."""
    bad_reactions = [
        dict(
            k0_model=1e-3, alpha=0.5, n_electrons=2.0, E_eq_model=0.0,
            cathodic_species=2, anodic_species=None, reversible=False,
            c_ref_model=0.0,
            cathodic_conc_factors=[],
            stoichiometry=[0, 0, -2],
        )
    ]
    ok, reason, _, _ = picard_outer_loop_general(
        reactions=bad_reactions,
        phi_applied_model=_PHI_APPLIED,
        a_h=0.0, a_cl=0.0, c_cl_anchor_kind="bulk", stern_split=None,
        **{**_GENERAL_SHARED_PARAMS, "topology_hint": "general"},
    )
    assert not ok
    assert "h_plus_as_linear_substrate" in reason, reason
