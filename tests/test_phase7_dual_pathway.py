"""Phase 7 (v11) — dual-pathway (water-as-proton-donor) config tests.

Fast unit tests for the ``proton_donor`` reaction schema, role flags,
the shared reactions-vs-convergence cross-validation, the
``PARALLEL_2E_4E_DUAL_PATHWAY`` preset, and role-based observable index
resolution.  Slow Firedrake rate-law tests live in their own classes
below (Phase 2 of the plan).

Plan: ``~/.claude/plans/ancient-squishing-pond.md`` (GPT critique
session 41, APPROVED).
"""
from __future__ import annotations

import os
import sys

import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Forward.bv_solver.config import (
    _get_bv_reactions_cfg,
    _validate_reactions_vs_convergence,
)
from Forward.bv_solver.observables import _get_reaction_role_indices


def _params(reactions):
    return {"bv_bc": {"reactions": reactions}}


def _minimal_rxn(**overrides):
    rxn = {
        "k0": 1e-5,
        "alpha": 0.5,
        "cathodic_species": 0,
        "anodic_species": 1,
        "stoichiometry": [-1, +1, -2],
        "n_electrons": 2,
        "reversible": True,
        "E_eq_v": 0.695,
    }
    rxn.update(overrides)
    return rxn


# ===================================================================
# proton_donor parsing
# ===================================================================

class TestProtonDonorParsing:
    def test_default_is_hydronium(self):
        cfg = _get_bv_reactions_cfg(_params([_minimal_rxn()]), 3)
        assert cfg[0]["proton_donor"] == "hydronium"

    def test_water_normalizes_case_and_whitespace(self):
        rxn = _minimal_rxn(
            proton_donor="  WATER ", reversible=False, anodic_species=None,
        )
        cfg = _get_bv_reactions_cfg(_params([rxn]), 3)
        assert cfg[0]["proton_donor"] == "water"

    def test_typo_raises(self):
        rxn = _minimal_rxn(proton_donor="watter")
        with pytest.raises(ValueError, match="proton_donor"):
            _get_bv_reactions_cfg(_params([rxn]), 3)

    def test_key_survives_parser_round_trip(self):
        """Regression: the parser rebuilds dicts from a whitelist —
        proton_donor must be on it or it silently vanishes."""
        rxn = _minimal_rxn(
            proton_donor="water", reversible=False, anodic_species=None,
        )
        cfg = _get_bv_reactions_cfg(_params([rxn]), 3)
        assert "proton_donor" in cfg[0]
        assert "label" in cfg[0]
        assert "produces_h2o2" in cfg[0]
        assert "consumes_h2o2" in cfg[0]


# ===================================================================
# proton_donor validation rules
# ===================================================================

class TestProtonDonorValidation:
    def test_water_forbids_conc_factors(self):
        rxn = _minimal_rxn(
            proton_donor="water", reversible=False, anodic_species=None,
            cathodic_conc_factors=[
                {"species": 2, "power": 2, "c_ref_nondim": 1.0},
            ],
        )
        with pytest.raises(ValueError, match="cathodic_conc_factors"):
            _get_bv_reactions_cfg(_params([rxn]), 3)

    def test_water_requires_irreversible(self):
        rxn = _minimal_rxn(proton_donor="water", reversible=True)
        with pytest.raises(ValueError, match="reversible=False"):
            _get_bv_reactions_cfg(_params([rxn]), 3)


# ===================================================================
# role flags (produces_h2o2 / consumes_h2o2) vs stoichiometry
# ===================================================================

class TestRoleFlagValidation:
    def test_produces_requires_positive_stoich(self):
        rxn = _minimal_rxn(produces_h2o2=True, stoichiometry=[-1, 0, -4])
        with pytest.raises(ValueError, match="produces_h2o2"):
            _get_bv_reactions_cfg(_params([rxn]), 3)

    def test_consumes_requires_negative_stoich(self):
        rxn = _minimal_rxn(consumes_h2o2=True, stoichiometry=[-1, +1, -2])
        with pytest.raises(ValueError, match="consumes_h2o2"):
            _get_bv_reactions_cfg(_params([rxn]), 3)

    def test_both_flags_forbidden(self):
        rxn = _minimal_rxn(produces_h2o2=True, consumes_h2o2=True)
        with pytest.raises(ValueError, match="cannot both"):
            _get_bv_reactions_cfg(_params([rxn]), 3)

    def test_valid_producer_and_consumer_pass(self):
        producer = _minimal_rxn(produces_h2o2=True, label="prod")
        consumer = _minimal_rxn(
            consumes_h2o2=True, label="cons",
            stoichiometry=[0, -1, -2], cathodic_species=1,
            anodic_species=None, reversible=False,
        )
        cfg = _get_bv_reactions_cfg(_params([producer, consumer]), 3)
        assert cfg[0]["produces_h2o2"] and not cfg[0]["consumes_h2o2"]
        assert cfg[1]["consumes_h2o2"] and not cfg[1]["produces_h2o2"]
        assert [r["label"] for r in cfg] == ["prod", "cons"]


# ===================================================================
# shared cross-config validation (active water route needs Kw closure)
# ===================================================================

class TestCrossConfigValidation:
    def _water_rxn_cfg(self, k0):
        rxn = _minimal_rxn(
            proton_donor="water", reversible=False, anodic_species=None,
            k0=k0,
        )
        return _get_bv_reactions_cfg(_params([rxn]), 3)

    def test_active_water_route_without_kw_raises(self):
        cfg = self._water_rxn_cfg(1e-5)
        with pytest.raises(ValueError, match="enable_water_ionization"):
            _validate_reactions_vs_convergence(
                cfg, {"enable_water_ionization": False}
            )

    def test_active_water_route_with_kw_passes(self):
        cfg = self._water_rxn_cfg(1e-5)
        _validate_reactions_vs_convergence(
            cfg, {"enable_water_ionization": True}
        )

    def test_inactive_water_route_does_not_force_kw(self):
        """k0=0 water entries (ablation/provenance) must not require the
        Kw closure."""
        cfg = self._water_rxn_cfg(0.0)
        _validate_reactions_vs_convergence(
            cfg, {"enable_water_ionization": False}
        )

    def test_hydronium_routes_unaffected(self):
        cfg = _get_bv_reactions_cfg(_params([_minimal_rxn()]), 3)
        _validate_reactions_vs_convergence(cfg, {})
        _validate_reactions_vs_convergence(cfg, None)
        _validate_reactions_vs_convergence(None, {})


# ===================================================================
# PARALLEL_2E_4E_DUAL_PATHWAY preset
# ===================================================================

class TestDualPathwayPreset:
    @pytest.fixture(scope="class")
    def presets(self):
        from scripts._bv_common import (
            PARALLEL_2E_4E_DUAL_PATHWAY,
            PARALLEL_2E_4E_REACTIONS,
        )
        return PARALLEL_2E_4E_DUAL_PATHWAY, PARALLEL_2E_4E_REACTIONS

    def test_shape_and_labels(self, presets):
        dual, _ = presets
        assert len(dual) == 4
        assert [r["label"] for r in dual] == [
            "R2e_acid", "R2e_water", "R4e_acid", "R4e_water",
        ]

    def test_acid_entries_byte_equal_legacy_plus_new_keys(self, presets):
        dual, legacy = presets
        new_keys = {"label", "proton_donor", "produces_h2o2", "consumes_h2o2"}
        for dual_idx, legacy_idx in ((0, 0), (2, 1)):
            stripped = {
                k: v for k, v in dual[dual_idx].items() if k not in new_keys
            }
            assert stripped == legacy[legacy_idx]

    def test_water_entries_rate_law_shape(self, presets):
        dual, _ = presets
        for idx in (1, 3):
            rxn = dual[idx]
            assert rxn["proton_donor"] == "water"
            assert rxn["reversible"] is False
            assert rxn["cathodic_conc_factors"] == []
            assert rxn["anodic_species"] is None

    def test_stoichiometry_and_roles(self, presets):
        dual, _ = presets
        assert dual[1]["stoichiometry"] == [-1, +1, -2]
        assert dual[3]["stoichiometry"] == [-1, 0, -4]
        assert [bool(r.get("produces_h2o2")) for r in dual] == [
            True, True, False, False,
        ]

    def test_parses_clean_and_water_gate_fires(self, presets):
        dual, _ = presets
        cfg = _get_bv_reactions_cfg(_params(dual), 3)
        assert [r["proton_donor"] for r in cfg] == [
            "hydronium", "water", "hydronium", "water",
        ]
        with pytest.raises(ValueError, match="enable_water_ionization"):
            _validate_reactions_vs_convergence(
                cfg, {"enable_water_ionization": False}
            )
        _validate_reactions_vs_convergence(
            cfg, {"enable_water_ionization": True}
        )

    def test_e_eq_unshifted_in_preset(self, presets):
        """Presets must store UNSHIFTED E° vs RHE; the OCP shift is
        applied in exactly one place (the driver).  Guards against the
        double-shift bug class (critique R1#11)."""
        dual, _ = presets
        from scripts._bv_common import E_EQ_R2E_V, E_EQ_R4E_V
        assert dual[0]["E_eq_v"] == dual[1]["E_eq_v"] == E_EQ_R2E_V
        assert dual[2]["E_eq_v"] == dual[3]["E_eq_v"] == E_EQ_R4E_V
        assert E_EQ_R2E_V == pytest.approx(0.695)
        assert E_EQ_R4E_V == pytest.approx(1.23)


# ===================================================================
# default-off equivalence: legacy preset parses identically (mod new
# defaulted keys), and the convergence dict gains no new keys
# ===================================================================

class TestDefaultOffEquivalence:
    NEW_KEYS = {"proton_donor", "label", "produces_h2o2", "consumes_h2o2"}

    def test_legacy_preset_parse_unchanged_modulo_new_defaults(self):
        from scripts._bv_common import PARALLEL_2E_4E_REACTIONS
        cfg = _get_bv_reactions_cfg(_params(PARALLEL_2E_4E_REACTIONS), 3)
        for j, (parsed, raw) in enumerate(zip(cfg, PARALLEL_2E_4E_REACTIONS)):
            assert parsed["proton_donor"] == "hydronium"
            assert parsed["label"] == f"reaction_{j}"
            assert parsed["produces_h2o2"] is False
            assert parsed["consumes_h2o2"] is False
            legacy_part = {
                k: v for k, v in parsed.items() if k not in self.NEW_KEYS
            }
            # Same field set and values as the raw preset (parser casts
            # but does not add/drop legacy fields).
            assert set(legacy_part) == {
                "k0", "alpha", "cathodic_species", "anodic_species", "c_ref",
                "stoichiometry", "n_electrons", "reversible",
                "cathodic_conc_factors", "E_eq_v",
            }
            assert legacy_part["stoichiometry"] == list(raw["stoichiometry"])

    def test_convergence_cfg_untouched(self):
        from Forward.bv_solver.config import _get_bv_convergence_cfg
        cfg = _get_bv_convergence_cfg({"bv_convergence": {}})
        assert "proton_donor" not in cfg


# ===================================================================
# role-based observable index resolution (pure-dict ctx; no Firedrake)
# ===================================================================

class TestRoleIndexResolution:
    def _ctx(self, reactions):
        return {"nondim": {"bv_reactions": reactions}}

    def test_resolves_producers(self):
        ctx = self._ctx([
            {"produces_h2o2": True}, {"produces_h2o2": True}, {}, {},
        ])
        assert _get_reaction_role_indices(ctx, "produces_h2o2") == [0, 1]

    def test_stable_under_insertion(self):
        """R3 inserted mid-list must not change which reactions the
        peroxide observable sums (critique R2#4)."""
        base = [
            {"label": "R2e_acid", "produces_h2o2": True},
            {"label": "R2e_water", "produces_h2o2": True},
            {"label": "R4e_acid"},
            {"label": "R4e_water"},
        ]
        with_r3 = base[:2] + [{"label": "R3", "consumes_h2o2": True}] + base[2:]
        idx_base = _get_reaction_role_indices(
            self._ctx(base), "produces_h2o2")
        idx_r3 = _get_reaction_role_indices(
            self._ctx(with_r3), "produces_h2o2")
        labels_base = [base[i]["label"] for i in idx_base]
        labels_r3 = [with_r3[i]["label"] for i in idx_r3]
        assert labels_base == labels_r3 == ["R2e_acid", "R2e_water"]

    def test_missing_ctx_returns_none(self):
        assert _get_reaction_role_indices({}, "produces_h2o2") is None

    def test_empty_roles_returns_empty(self):
        assert _get_reaction_role_indices(self._ctx([{}]), "produces_h2o2") == []
