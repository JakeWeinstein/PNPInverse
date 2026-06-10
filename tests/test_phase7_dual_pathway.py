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


# ===================================================================
# Phase 2 — slow Firedrake tests
# ===================================================================

def _water_only_2e_reaction(k0):
    return {
        "k0": k0,
        "alpha": 0.627,
        "cathodic_species": 0,
        "anodic_species": None,
        "c_ref": 0.0,
        "stoichiometry": [-1, +1, -2],
        "n_electrons": 2,
        "reversible": False,
        "E_eq_v": 0.695,
        "cathodic_conc_factors": [],
        "label": "R2e_water",
        "proton_donor": "water",
        "produces_h2o2": True,
    }


def _acid_only_2e_irreversible(k0):
    """Acid-route 2e with c_H^2 factor, irreversible so the assembled
    rate scales EXACTLY as exp(2*delta) under a uniform u_H shift."""
    return {
        "k0": k0,
        "alpha": 0.627,
        "cathodic_species": 0,
        "anodic_species": None,
        "c_ref": 0.0,
        "stoichiometry": [-1, +1, -2],
        "n_electrons": 2,
        "reversible": False,
        "E_eq_v": 0.695,
        "cathodic_conc_factors": [
            {"species": 2, "power": 2, "c_ref_nondim": 1.0},
        ],
        "label": "R2e_acid_irrev",
        "proton_donor": "hydronium",
    }


@pytest.mark.slow
class TestWaterRouteRateLaw:
    """The water-route BV rate must be invariant to u_H; the acid route
    must scale as c_H^power.  Built on a coarse mesh at the linear-φ IC,
    assembled as the boundary rate integral."""

    def _build(self, reactions, *, enable_water_ionization):
        import firedrake as fd
        from scripts._bv_common import (
            DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC,
            THREE_SPECIES_LOGC_BOLTZMANN,
            make_bv_solver_params,
        )
        from Forward.bv_solver.dispatch import (
            build_context, build_forms, set_initial_conditions,
        )
        sp = make_bv_solver_params(
            eta_hat=0.0, dt=0.25, t_end=10.0,
            species=THREE_SPECIES_LOGC_BOLTZMANN,
            formulation="logc_muh",
            log_rate=True,
            bv_reactions=reactions,
            boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC],
            stern_capacitance_f_m2=0.10,
            initializer="linear_phi",
            enable_water_ionization=enable_water_ionization,
        )
        mesh = fd.UnitSquareMesh(8, 16)
        ctx = build_context(sp, mesh=mesh)
        ctx = build_forms(ctx, sp)
        set_initial_conditions(ctx, sp)
        return ctx

    def _boundary_rate(self, ctx):
        import firedrake as fd
        bv_cfg = ctx.get("bv_settings", {})
        marker = int(bv_cfg.get("electrode_marker", 1))
        ds = fd.Measure("ds", domain=ctx["mesh"])
        return float(fd.assemble(ctx["bv_rate_exprs"][0] * ds(marker)))

    def _perturb_h_component(self, ctx, delta):
        U = ctx["U"]
        h_sub = U.subfunctions[2]
        h_sub.assign(h_sub + delta)

    def test_water_route_rate_invariant_to_u_h(self):
        ctx = self._build(
            [_water_only_2e_reaction(1e-5)], enable_water_ionization=True,
        )
        r0 = self._boundary_rate(ctx)
        assert r0 != 0.0
        self._perturb_h_component(ctx, 0.3)
        r1 = self._boundary_rate(ctx)
        assert r1 == pytest.approx(r0, rel=1e-10)

    def test_acid_route_rate_scales_as_c_h_squared(self):
        ctx = self._build(
            [_acid_only_2e_irreversible(1e-5)], enable_water_ionization=False,
        )
        r0 = self._boundary_rate(ctx)
        assert r0 != 0.0
        delta = 0.3
        self._perturb_h_component(ctx, delta)
        r1 = self._boundary_rate(ctx)
        import math
        assert r1 / r0 == pytest.approx(math.exp(2 * delta), rel=1e-8)


@pytest.mark.slow
class TestDefaultOffResidualNormPhase7:
    """Legacy reactions parsed with defaulted phase-7 keys must build a
    residual byte-equivalent to the same reactions with the keys spelled
    out explicitly — i.e. the new schema fields are provenance-only on
    the hydronium path.  Both backends."""

    @pytest.mark.parametrize("formulation", ["logc", "logc_muh"])
    def test_residual_norm_matches(self, formulation):
        import firedrake as fd
        from scripts._bv_common import (
            DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC,
            PARALLEL_2E_4E_REACTIONS,
            THREE_SPECIES_LOGC_BOLTZMANN,
            make_bv_solver_params,
        )
        from Forward.bv_solver.dispatch import (
            build_context, build_forms, set_initial_conditions,
        )

        explicit = [
            {
                **rxn,
                "proton_donor": "hydronium",
                "label": f"reaction_{j}",
                "produces_h2o2": False,
                "consumes_h2o2": False,
            }
            for j, rxn in enumerate(PARALLEL_2E_4E_REACTIONS)
        ]

        norms = []
        for reactions in (PARALLEL_2E_4E_REACTIONS, explicit):
            sp = make_bv_solver_params(
                eta_hat=0.0, dt=0.25, t_end=10.0,
                species=THREE_SPECIES_LOGC_BOLTZMANN,
                formulation=formulation,
                log_rate=True,
                bv_reactions=reactions,
                boltzmann_counterions=[
                    DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC,
                ],
                stern_capacitance_f_m2=0.10,
                initializer="linear_phi",
            )
            mesh = fd.UnitSquareMesh(8, 16)
            ctx = build_context(sp, mesh=mesh)
            ctx = build_forms(ctx, sp)
            set_initial_conditions(ctx, sp)
            r = fd.assemble(ctx["F_res"])
            norm = float(fd.assemble(
                sum(fd.inner(s, s) * fd.dx
                    for s in fd.split(r.riesz_representation()))
            )) ** 0.5
            norms.append(norm)
        assert norms[1] == pytest.approx(norms[0], rel=1e-12, abs=1e-15)


@pytest.mark.slow
class TestWaterRouteValidationAtBuild:
    """Active water route + enable_water_ionization=False must raise at
    form build on BOTH backends (direct backend entry, not just the
    dispatcher — anchor_continuation imports the muh backend directly)."""

    @pytest.mark.parametrize("formulation", ["logc", "logc_muh"])
    def test_raises_without_kw(self, formulation):
        import firedrake as fd
        from scripts._bv_common import (
            DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC,
            THREE_SPECIES_LOGC_BOLTZMANN,
            make_bv_solver_params,
        )
        from Forward.bv_solver.dispatch import build_context, build_forms

        sp = make_bv_solver_params(
            eta_hat=0.0, dt=0.25, t_end=10.0,
            species=THREE_SPECIES_LOGC_BOLTZMANN,
            formulation=formulation,
            log_rate=True,
            bv_reactions=[_water_only_2e_reaction(1e-5)],
            boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC],
            stern_capacitance_f_m2=0.10,
            initializer="linear_phi",
            enable_water_ionization=False,
        )
        mesh = fd.UnitSquareMesh(8, 16)
        ctx = build_context(sp, mesh=mesh)
        with pytest.raises(ValueError, match="enable_water_ionization"):
            build_forms(ctx, sp)


@pytest.mark.slow
class TestWaterRouteEscapesLevichCap:
    """Headline physics test: with ONLY water routes active (acid k0=0)
    and the Kw closure on, the cathodic current must exceed 3x the H+
    Levich cap at L_eff = 15.4 um — finite current with zero H+-supply
    dependence — while staying at or below the O2 4e ceiling (no
    O2-free current).  Gate values (1600-rpm Levich-equivalent film):
    H+ cap 0.583 mA/cm2, O2-4e ceiling 5.71 mA/cm2."""

    H_CAP_MA_CM2 = 0.0898 * (100.0 / 15.4)        # 0.583
    O2_4E_CEILING_MA_CM2 = 5.71

    def test_water_only_current_escapes_h_cap(self):
        import numpy as np
        import firedrake as fd
        import firedrake.adjoint as adj
        from scripts._bv_common import (
            ALPHA_R4E,
            DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,
            DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
            I_SCALE,
            K0_HAT_R2E,
            K0_HAT_R4E,
            KW_HAT,
            SNES_OPTS_CHARGED,
            THREE_SPECIES_LOGC_BOLTZMANN,
            V_T,
            make_bv_solver_params,
        )

        snes_opts = {**SNES_OPTS_CHARGED}
        snes_opts.update({
            "snes_max_it":               400,
            "snes_atol":                 1e-7,
            "snes_rtol":                 1e-10,
            "snes_stol":                 1e-12,
            "snes_linesearch_type":      "l2",
            "snes_linesearch_maxlambda": 0.3,
            "snes_divergence_tolerance": 1e10,
        })
        from Forward.bv_solver import make_graded_rectangle_mesh
        from Forward.bv_solver.anchor_continuation import (
            solve_anchor_with_continuation,
        )
        from Forward.bv_solver.observables import _build_bv_observable_form

        # OCP-shifted convention (the proven 15.4-um configuration, run
        # phase7_p0_*): E° shifted by -0.903 V, anchor at V_solver = 0
        # (flat double layer).  Cold-anchoring at unshifted +0.55 V
        # (~ +21 V_T DL) diverges at the thin domain — verified.
        v_ocp = 0.903
        water_2e = _water_only_2e_reaction(float(K0_HAT_R2E))
        water_2e["E_eq_v"] = 0.695 - v_ocp
        water_4e = {
            "k0": float(K0_HAT_R4E),
            "alpha": float(ALPHA_R4E),
            "cathodic_species": 0,
            "anodic_species": None,
            "c_ref": 0.0,
            "stoichiometry": [-1, 0, -4],
            "n_electrons": 4,
            "reversible": False,
            "E_eq_v": 1.23 - v_ocp,
            "cathodic_conc_factors": [],
            "label": "R4e_water",
            "proton_donor": "water",
        }
        reactions = [water_2e, water_4e]

        sp = make_bv_solver_params(
            eta_hat=0.0, dt=0.25, t_end=80.0,
            species=THREE_SPECIES_LOGC_BOLTZMANN,
            snes_opts=snes_opts,
            formulation="logc_muh",
            log_rate=True,
            bv_reactions=reactions,
            boltzmann_counterions=[
                DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,
                DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
            ],
            multi_ion_enabled=True,
            stern_capacitance_f_m2=0.10,
            # linear_phi: the debye_boltzmann Picard IC mis-seeds
            # water-route reactions (diverges at the first reaction-free
            # rung — verified); at the V=0 flat-DL anchor linear_phi is
            # equivalent anyway.
            initializer="linear_phi",
            l_eff_m=15.4e-6,
            enable_water_ionization=True,
        )
        # Production pattern: cold-anchoring at deep cathodic V fails
        # (verified — first k0 rung diverges); anchor at the benign
        # anodic end and warm-walk the grid down to -0.30 V instead.
        from Forward.bv_solver.anchor_continuation import PreconvergedAnchor
        from Forward.bv_solver.grid_per_voltage import (
            snapshot_U, solve_grid_with_anchor,
        )

        anchor_v = 0.0  # deck +0.903 V: rest state, flat double layer
        sp_anchor = sp.with_phi_applied(anchor_v / float(V_T))
        # Ny=80 matches the proven 15.4-um demo-driver resolution; Ny=40
        # under-resolves the Debye layer at domain_height_hat=0.154 and
        # the reaction-free first rung already diverges.
        mesh = make_graded_rectangle_mesh(
            Nx=4, Ny=80, beta=3.0, domain_height_hat=0.154,
        )
        # Water-route anchor recipe (Phase 7 finding, 2026-06-10):
        # kw_eff_ladder=None — the ladder's Kw=0 floor is UNPHYSICAL for
        # water routes (H+-equivalent sink with no c_H damping and no
        # water reservoir -> mu_H blows up once k0 ramps).  Anchor at
        # FULL Kw instead and let the k0 AdaptiveLadder (with midpoint
        # insertion) walk into the alkaline boundary layer; the hard
        # band is k0_scale ~ 1e-8..1e-6 where the water demand first
        # crosses the O2 transport limit.
        k0_targets = {0: float(K0_HAT_R2E), 1: float(K0_HAT_R4E)}
        with adj.stop_annotating():
            result = solve_anchor_with_continuation(
                sp_anchor, mesh=mesh,
                k0_targets=k0_targets,
                initial_scales=(1e-12, 1e-9, 1e-6, 1e-3, 1.0),
                max_inserts_per_step=6,
                ic_at_target=True,
                kw_eff_ladder=None,
            )
        assert result.converged, f"anchor failed: {result.ladder_history!r}"

        anchor = PreconvergedAnchor(
            phi_applied_eta=anchor_v / float(V_T),
            U_snapshot=tuple(
                np.asarray(a).copy() for a in snapshot_U(result.ctx["U"])
            ),
            k0_targets=tuple(sorted(
                (int(j), float(k)) for j, k in k0_targets.items()
            )),
            mesh_dof_count=int(result.ctx["U"].function_space().dim()),
            ladder_history=tuple(
                (float(s), str(o)) for s, o in result.ladder_history
            ),
        )

        # Deck window 0.903 -> -0.30 in solver convention: 0 -> -1.203 V.
        v_grid = np.linspace(anchor_v, -1.203, 8)
        cd_at = {}

        def _grab(orig_idx, _phi_eta, ctx):
            f_cd = _build_bv_observable_form(
                ctx, mode="current_density", reaction_index=None,
                scale=-I_SCALE,
            )
            cd_at[orig_idx] = float(fd.assemble(f_cd))

        with adj.stop_annotating():
            grid_result = solve_grid_with_anchor(
                sp, anchor=anchor,
                phi_applied_values=v_grid / float(V_T),
                mesh=mesh,
                per_point_callback=_grab,
            )

        most_cathodic = len(v_grid) - 1
        assert grid_result.points[most_cathodic].converged, (
            f"grid walk failed to reach V={v_grid[-1]:+.2f}: "
            f"{ {i: p.converged for i, p in grid_result.points.items()} !r}"
        )
        cd = cd_at[most_cathodic]
        assert np.isfinite(cd) and cd < 0.0
        assert abs(cd) > 3.0 * self.H_CAP_MA_CM2, (
            f"|cd|={abs(cd):.4f} did not escape 3x H+ cap "
            f"({3 * self.H_CAP_MA_CM2:.3f} mA/cm2)"
        )
        assert abs(cd) <= 1.05 * self.O2_4E_CEILING_MA_CM2, (
            f"|cd|={abs(cd):.4f} exceeds O2 4e ceiling "
            f"({self.O2_4E_CEILING_MA_CM2} mA/cm2) — O2-free current?"
        )
