"""Phase 7.3 M3 — unit tests for the C1 (electrochemical H2O2 reduction) wiring.

C1 must be purely ADDITIVE (default routes unchanged byte-for-byte), carry the
deck-consistent H2O2/H2O formal potential, read SURFACE c_H (a cathodic_conc_
factor on the proton), receive the −V_OCP shift, and respect the single-
convention XOR (SHE frame + C1's c_H factor is a double-count → blocked).
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import scripts.studies.solver_demo_slide15_dual_pathway_cs as dp
from scripts._bv_common import (
    E_EQ_C1_V, E_EQ_R2E_V, E_EQ_R4E_V, K0_HAT_C1, make_c1_reaction,
)

V_OCP = 1.019


def _opts(routes="water", **kw):
    base = dict(
        routes=routes,
        k0_water_2e_factor=0.1, k0_water_4e_factor=1e-12,
        k0_acid_4e_factor=1e-15,
        alpha_water_2e=0.577, alpha_water_4e=0.305,
        l_eff_um=21.7, bulk_h_mol_m3=0.1,
        enable_water_ionization=True, coarse_grid=True,
        cation="k", v_ocp_rhe=V_OCP, v_grid_lo=None, v_grid_hi=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_c1_eeq_is_deck_consistent():
    # E°_C1 = 2·E_R4e − E_R2e (R2e + C1 ≡ R4e), ≈ 1.76 V H2O2/H2O couple.
    assert E_EQ_C1_V == pytest.approx(2 * E_EQ_R4E_V - E_EQ_R2E_V, abs=1e-12)
    assert 1.74 < E_EQ_C1_V < 1.79


def test_make_c1_schema():
    r = make_c1_reaction(k0_factor=2.0, alpha=0.5, h_order=1.5)
    assert r["cathodic_species"] == 1            # H2O2 consumed
    assert r["stoichiometry"] == [0, -1, -2]
    assert r["n_electrons"] == 2 and r["reversible"] is False
    assert r["consumes_h2o2"] is True and r["produces_h2o2"] is False
    assert r["k0"] == pytest.approx(2.0 * K0_HAT_C1)
    f = r["cathodic_conc_factors"]
    assert len(f) == 1 and f[0]["species"] == 2 and f[0]["power"] == 1.5


def test_c1_is_purely_additive():
    """Default routes (no c1) unchanged; adding c1 appends exactly one
    reaction and leaves the first four byte-identical."""
    base = dp._build_reactions(_opts(routes="water"))
    withc1 = dp._build_reactions(_opts(routes="water,c1", k0_c1_factor=1.0))
    assert len(base) == 4 and len(withc1) == 5
    assert base == withc1[:4]
    c1 = withc1[4]
    assert c1["label"] == "C1_h2o2_reduction"
    assert c1["k0"] > 0.0
    # C1 E_eq carries the −V_OCP shift like every other reaction.
    assert c1["E_eq_v"] == pytest.approx(E_EQ_C1_V - V_OCP, abs=1e-12)


def test_c1_not_disabled_by_acid_route_logic():
    """C1 is gated by the 'c1' token, NOT 'acid' — water,c1 (no acid) must
    keep C1 enabled."""
    rxns = dp._build_reactions(_opts(routes="water,c1", k0_c1_factor=1.0))
    c1 = [r for r in rxns if r.get("consumes_h2o2")][0]
    assert c1["k0"] > 0.0


def test_c1_flags_drive_params():
    rxns = dp._build_reactions(_opts(routes="water,c1", k0_c1_factor=3.0,
                                     alpha_c1=0.4, c1_h_order=2.0))
    c1 = [r for r in rxns if r.get("consumes_h2o2")][0]
    assert c1["k0"] == pytest.approx(3.0 * K0_HAT_C1)
    assert c1["alpha"] == 0.4
    assert c1["cathodic_conc_factors"][0]["power"] == 2.0


def test_xor_guard_blocks_she_plus_c1():
    """SHE frame + C1's c_H factor is a proton double-count → SystemExit."""
    with pytest.raises(SystemExit, match="double-counts"):
        dp._build_reactions(_opts(routes="water,c1", proton_frame="she",
                                  bulk_h_anchor_mol_m3=4.07e-4))


def test_routes_rejects_bogus_token():
    with pytest.raises(SystemExit, match="routes must be subset"):
        dp._build_reactions(_opts(routes="water,bogus"))
