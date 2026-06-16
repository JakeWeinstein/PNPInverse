"""Phase 7.3 P0.1 — single-convention frame byte-test (the falsification guard).

Fast, Firedrake-free unit tests on ``_build_reactions``: the SHE-anchored
formal-potential frame must reproduce the RHE-referenced pH-6.39 lock
BYTE-FOR-BYTE at the anchor condition (the central precondition of the
Phase 7.3 plan), be demonstrably LIVE off-anchor (correct Nernstian sign +
magnitude), enforce the proton-dependence XOR (formal-potential shift vs a
kinetic c_H factor — never both), and stay byte-equivalent for callers that
never opt in (the fit harness / pH-series).

Plan: ``~/.claude/plans/phase7p3-pH-coupled-orr-mechanism.md`` (P0.1).
The end-to-end SOLVER byte test lives in
``scripts/studies/phase7p3_p0_1_frame_byte_test.py`` (slow, Firedrake).
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import scripts.studies.drivers.solver_demo_slide15_dual_pathway_cs as dp

# pH-6.39 lock config (matches phase7p2_fit_dual_series_adjoint.py defaults).
BULK_H_PH639 = 4.07e-4   # mol/m³ -> pH 6.39 (the anchor)
BULK_H_PH4 = 0.1         # mol/m³ -> pH 4.0
BULK_H_PH10 = 1.0e-7     # mol/m³ -> pH 10.0
V_OCP_PH639 = 1.019


def _opts(*, bulk_h, proton_frame="rhe", routes="water",
          bulk_h_anchor=BULK_H_PH639):
    """Lock-config opts with a few overridable axes."""
    return SimpleNamespace(
        routes=routes,
        k0_water_2e_factor=10.0 ** -1.008699731156705,
        k0_water_4e_factor=10.0 ** -12.308926782786854,
        k0_acid_4e_factor=1e-15,
        alpha_water_2e=0.5770144526758703,
        alpha_water_4e=0.304853786169721,
        l_eff_um=21.7,
        bulk_h_mol_m3=bulk_h,
        enable_water_ionization=True,
        coarse_grid=True,
        cation="k",
        v_ocp_rhe=V_OCP_PH639,
        v_grid_lo=None, v_grid_hi=None,
        proton_frame=proton_frame,
        bulk_h_anchor_mol_m3=bulk_h_anchor,
    )


def test_she_shift_is_exactly_zero_at_anchor():
    """The whole gate: at the anchor condition the SHE offset is 0.0 EXACTLY
    (anchored on bulk c_H, not a rounded pH) -> byte-exact lock reproduction."""
    shift = dp._she_eeq_shift_v(_opts(bulk_h=BULK_H_PH639, proton_frame="she"))
    assert shift == 0.0
    assert dp._she_eeq_shift_v(_opts(bulk_h=BULK_H_PH639,
                                     proton_frame="rhe")) == 0.0


def test_reactions_byte_identical_at_anchor():
    """SHE frame == RHE frame, field-for-field, at the anchor (incl. exact
    E_eq_v floats). This is the P0.1 byte-for-byte guarantee."""
    rxn_rhe = dp._build_reactions(_opts(bulk_h=BULK_H_PH639, proton_frame="rhe"))
    rxn_she = dp._build_reactions(_opts(bulk_h=BULK_H_PH639, proton_frame="she"))
    assert len(rxn_rhe) == len(rxn_she) == 4
    for r_rhe, r_she in zip(rxn_rhe, rxn_she):
        assert r_rhe == r_she                     # full dict equality
        assert r_rhe["E_eq_v"] == r_she["E_eq_v"]  # exact float (redundant, explicit)


def test_default_frame_matches_explicit_rhe_and_missing_attr():
    """Callers that never set proton_frame (fit harness, pH-series) are
    byte-equivalent to explicit RHE."""
    opts_missing = _opts(bulk_h=BULK_H_PH639)
    del opts_missing.proton_frame
    del opts_missing.bulk_h_anchor_mol_m3
    rxn_default = dp._build_reactions(opts_missing)
    rxn_rhe = dp._build_reactions(_opts(bulk_h=BULK_H_PH639, proton_frame="rhe"))
    assert rxn_default == rxn_rhe


def test_frame_is_live_off_anchor_with_correct_nernst():
    """Off-anchor the SHE machinery DOES something: every reaction E_eq
    shifts by exactly S·(pH − pH_anchor), monotonic +S/pH, sign-correct."""
    s = dp._nernst_slope_v_per_ph()
    assert 0.058 < s < 0.060   # V_T·ln10 ≈ 0.05916 (the plan's 0.0592)

    rxn_rhe = dp._build_reactions(_opts(bulk_h=BULK_H_PH4, proton_frame="rhe"))
    rxn_she = dp._build_reactions(_opts(bulk_h=BULK_H_PH4, proton_frame="she"))
    ph4 = dp._ph_from_bulk_h(BULK_H_PH4)
    ph_anchor = dp._ph_from_bulk_h(BULK_H_PH639)
    expected = s * (ph4 - ph_anchor)            # ≈ -0.1414 V (pH4 < anchor)
    assert expected < -0.10
    for r_rhe, r_she in zip(rxn_rhe, rxn_she):
        assert r_she["E_eq_v"] - r_rhe["E_eq_v"] == pytest.approx(expected, abs=1e-12)

    # pH 10 (above anchor) -> positive shift (higher E_eq_RHE = earlier onset).
    shift10 = dp._she_eeq_shift_v(_opts(bulk_h=BULK_H_PH10, proton_frame="she"))
    assert shift10 > 0.0
    assert shift10 == pytest.approx(s * (dp._ph_from_bulk_h(BULK_H_PH10)
                                         - ph_anchor), abs=1e-12)


def test_xor_guard_blocks_double_count():
    """SHE frame + an enabled c_H-kinetic (acid) reaction must raise — the
    'formal-potential shift XOR kinetic c_H factor, never both' rule."""
    with pytest.raises(SystemExit, match="double-counts"):
        dp._build_reactions(_opts(bulk_h=BULK_H_PH4, proton_frame="she",
                                  routes="acid,water"))
    # water-only in the SHE frame is fine (water routes carry no c_H factor).
    dp._build_reactions(_opts(bulk_h=BULK_H_PH4, proton_frame="she",
                              routes="water"))


def test_invalid_frame_rejected():
    with pytest.raises(SystemExit, match="proton-frame"):
        dp._build_reactions(_opts(bulk_h=BULK_H_PH639, proton_frame="bogus"))
