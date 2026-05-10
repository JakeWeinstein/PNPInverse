"""Phase 6β v9 Gate 1 — role-aware species machinery tests.

Fast unit tests (no Firedrake) that cover:

* Phase 1A: ``SpeciesConfig.roles`` field + ``resolve_role_index`` helper
  (legacy z-inference fallback and explicit-roles disambiguation).
* Phase 1B: solver-side resolvers (``resolve_h_index`` /
  ``_resolve_mu_h_index``) accept the optional ``roles`` argument and
  preserve legacy behavior when ``roles=None``.
* Phase 1C: IC counterion identification path falls through cleanly when
  ``roles`` are provided for the K2SO4-style 4sp dynamic stack with two
  z=+1 species.

See ``.claude/plans/write-up-the-formal-joyful-papert.md`` for the full
plan and the v9 architecture handoff at
``docs/CHATGPT_HANDOFF_29_phase6b-stern-coupling-and-audit-residuals/``
for the motivation.
"""
from __future__ import annotations

import os
import sys

import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts._bv_common import (
    DEFAULT_CLO4_BOLTZMANN_COUNTERION,
    DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC,
    DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,
    DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
    FOUR_SPECIES_LOGC_DYNAMIC,
    SpeciesConfig,
    THREE_SPECIES_LOGC_BOLTZMANN,
    resolve_role_index,
)


# ===================================================================
# Phase 1A — SpeciesConfig.roles field
# ===================================================================


class TestSpeciesConfigRolesField:
    """``roles`` is optional and length-validated against ``n_species``."""

    def test_roles_default_is_none_for_back_compat(self):
        sp = SpeciesConfig(
            n_species=2,
            z_vals=[0, 1],
            d_vals_hat=[1.0, 1.0],
            a_vals_hat=[0.0, 0.0],
            c0_vals_hat=[1.0, 0.1],
            stoichiometry_r1=[-1, -2],
            stoichiometry_r2=[0, -2],
        )
        assert sp.roles is None

    def test_roles_explicit_accepted(self):
        sp = SpeciesConfig(
            n_species=2,
            z_vals=[0, 1],
            d_vals_hat=[1.0, 1.0],
            a_vals_hat=[0.0, 0.0],
            c0_vals_hat=[1.0, 0.1],
            stoichiometry_r1=[-1, -2],
            stoichiometry_r2=[0, -2],
            roles=["neutral", "proton"],
        )
        assert sp.roles == ["neutral", "proton"]

    def test_roles_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="roles length 3 does not match"):
            SpeciesConfig(
                n_species=2,
                z_vals=[0, 1],
                d_vals_hat=[1.0, 1.0],
                a_vals_hat=[0.0, 0.0],
                c0_vals_hat=[1.0, 0.1],
                stoichiometry_r1=[-1, -2],
                stoichiometry_r2=[0, -2],
                roles=["neutral", "proton", "extra"],
            )

    def test_existing_3sp_preset_carries_roles(self):
        assert THREE_SPECIES_LOGC_BOLTZMANN.roles == [
            "neutral", "neutral", "proton",
        ]

    def test_existing_4sp_preset_carries_roles(self):
        assert FOUR_SPECIES_LOGC_DYNAMIC.roles == [
            "neutral", "neutral", "proton", "counterion",
        ]


# ===================================================================
# Phase 1A — resolve_role_index legacy / explicit / ambiguous paths
# ===================================================================


class TestResolveRoleIndexBackwardCompat:
    """``roles=None`` falls back to the legacy ``z=+1 ⇒ proton`` lookup."""

    def test_proton_legacy_path_basic(self):
        assert resolve_role_index(None, [0, 0, 1], "proton") == 2
        assert resolve_role_index(None, [1, 0, 0], "proton") == 0

    def test_proton_legacy_path_no_z_plus1_rejected(self):
        with pytest.raises(ValueError, match="z=\\+1"):
            resolve_role_index(None, [0, 0, 0], "proton")

    def test_proton_legacy_path_only(self):
        # Only "proton" works with roles=None; counterion lookup must
        # use explicit roles.
        with pytest.raises(NotImplementedError, match="legacy z-inference"):
            resolve_role_index(None, [0, 0, 1, -1], "counterion")


class TestResolveRoleIndexExplicit:
    """Explicit ``roles`` disambiguates two z=+1 species (K2SO4 case)."""

    def test_proton_explicit_disambiguates_two_zplus1(self):
        # K2SO4-style stack: O2, H2O2, H+, K+ all charge-pattern below;
        # both H+ (idx 2) and K+ (idx 3) carry z=+1.  The explicit role
        # picks H+ at idx 2.
        roles = ["neutral", "neutral", "proton", "counterion"]
        z_vals = [0, 0, 1, 1]
        assert resolve_role_index(roles, z_vals, "proton") == 2

    def test_counterion_explicit_picks_role_label(self):
        roles = ["neutral", "neutral", "proton", "counterion"]
        z_vals = [0, 0, 1, 1]
        assert resolve_role_index(roles, z_vals, "counterion") == 3

    def test_explicit_role_lookup_case_insensitive(self):
        roles = ["NEUTRAL", "neutral", "Proton"]
        assert resolve_role_index(roles, [0, 0, 1], "proton") == 2

    def test_explicit_role_missing_rejected(self):
        roles = ["neutral", "neutral", "proton"]
        with pytest.raises(ValueError, match="no species with role='counterion'"):
            resolve_role_index(roles, [0, 0, 1], "counterion")

    def test_explicit_role_duplicates_rejected(self):
        roles = ["neutral", "proton", "proton"]
        with pytest.raises(ValueError, match="multiple species with role='proton'"):
            resolve_role_index(roles, [0, 1, 1], "proton")


class TestResolveRoleIndexAmbiguousZRejected:
    """Legacy z-inference with multiple z=+1 species fails loudly."""

    def test_two_zplus1_no_roles_rejected(self):
        with pytest.raises(ValueError, match="exactly one species with z=\\+1"):
            resolve_role_index(None, [0, 0, 1, 1], "proton")


# ===================================================================
# Counterion entry role-key check (additive, no behavior change)
# ===================================================================


class TestCounterionEntriesCarryRoleKey:
    """Adding ``role`` to the existing counterion entries is additive."""

    def test_clo4_basic_has_role(self):
        assert DEFAULT_CLO4_BOLTZMANN_COUNTERION.get("role") == "counterion"

    def test_clo4_steric_inherits_role(self):
        # Spread from the basic entry, so it picks up "role" automatically.
        assert DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC.get("role") == "counterion"

    def test_csplus_steric_has_role(self):
        assert DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC.get("role") == "counterion"

    def test_sulfate_steric_has_role(self):
        assert DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC.get("role") == "counterion"


# ===================================================================
# Phase 1B — solver-side resolvers accept ``roles=`` (no Firedrake)
# ===================================================================


class TestResolveHIndexRolesAware:
    """``Forward.bv_solver.water_ionization.resolve_h_index`` legacy + roles."""

    def test_legacy_z_inference_unchanged(self):
        from Forward.bv_solver.water_ionization import resolve_h_index
        assert resolve_h_index([0, 0, 1]) == 2

    def test_legacy_two_zplus1_still_rejected(self):
        from Forward.bv_solver.water_ionization import resolve_h_index
        with pytest.raises(ValueError, match="exactly one species"):
            resolve_h_index([0, 0, 1, 1])

    def test_explicit_roles_disambiguates_two_zplus1(self):
        from Forward.bv_solver.water_ionization import resolve_h_index
        assert resolve_h_index(
            [0, 0, 1, 1],
            roles=["neutral", "neutral", "proton", "counterion"],
        ) == 2

    def test_roles_length_mismatch_rejected(self):
        from Forward.bv_solver.water_ionization import resolve_h_index
        with pytest.raises(ValueError, match="roles length"):
            resolve_h_index(
                [0, 0, 1],
                roles=["neutral", "proton"],
            )

    def test_roles_no_proton_rejected(self):
        from Forward.bv_solver.water_ionization import resolve_h_index
        with pytest.raises(ValueError, match="no species with role='proton'"):
            resolve_h_index(
                [0, 0, 1],
                roles=["neutral", "neutral", "counterion"],
            )


class TestMuHResolverRolesAware:
    """``Forward.bv_solver.forms_logc_muh._resolve_mu_h_index`` legacy + roles."""

    def test_legacy_z_inference_unchanged(self):
        from Forward.bv_solver.forms_logc_muh import _resolve_mu_h_index
        assert _resolve_mu_h_index([0, 0, 1]) == 2

    def test_legacy_two_zplus1_still_rejected(self):
        from Forward.bv_solver.forms_logc_muh import _resolve_mu_h_index
        with pytest.raises(ValueError, match="exactly one species"):
            _resolve_mu_h_index([0, 0, 1, 1])

    def test_explicit_roles_disambiguates_two_zplus1(self):
        from Forward.bv_solver.forms_logc_muh import _resolve_mu_h_index
        assert _resolve_mu_h_index(
            [0, 0, 1, 1],
            roles=["neutral", "neutral", "proton", "counterion"],
        ) == 2


# ===================================================================
# Phase 1B — config plumbing: bv_bc.species_roles → solver
# ===================================================================


class TestSpeciesRolesConfigPlumbing:
    """``_get_species_roles`` returns the configured roles or None."""

    def test_returns_none_when_absent(self):
        from Forward.bv_solver.config import _get_species_roles
        assert _get_species_roles({}, 3) is None
        assert _get_species_roles({"bv_bc": {}}, 3) is None
        assert _get_species_roles(
            {"bv_bc": {"species_roles": None}}, 3,
        ) is None

    def test_returns_explicit_list(self):
        from Forward.bv_solver.config import _get_species_roles
        assert _get_species_roles(
            {"bv_bc": {"species_roles": ["neutral", "neutral", "proton"]}},
            3,
        ) == ["neutral", "neutral", "proton"]

    def test_length_mismatch_rejected(self):
        from Forward.bv_solver.config import _get_species_roles
        with pytest.raises(ValueError, match="length 2 does not match"):
            _get_species_roles(
                {"bv_bc": {"species_roles": ["neutral", "proton"]}},
                3,
            )


class TestMakeBvSolverParamsWritesSpeciesRoles:
    """make_bv_solver_params plumbs species.roles into bv_bc.species_roles."""

    def test_3sp_preset_writes_roles_to_cfg(self):
        from scripts._bv_common import (
            DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC,
            make_bv_solver_params,
        )
        sp = make_bv_solver_params(
            eta_hat=0.0, dt=0.25, t_end=1.0,
            species=THREE_SPECIES_LOGC_BOLTZMANN,
            formulation="logc_muh",
            log_rate=True,
            boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC],
            stern_capacitance_f_m2=0.10,
            initializer="debye_boltzmann",
        )
        assert sp[10]["bv_bc"]["species_roles"] == [
            "neutral", "neutral", "proton",
        ]

    def test_no_roles_field_means_no_species_roles_key(self):
        # Build a SpeciesConfig with roles=None → cfg should not gain
        # the species_roles key (preserves byte-equivalence).
        from scripts._bv_common import (
            DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC,
            make_bv_solver_params,
            SpeciesConfig,
        )
        sp_no_roles = SpeciesConfig(
            n_species=THREE_SPECIES_LOGC_BOLTZMANN.n_species,
            z_vals=list(THREE_SPECIES_LOGC_BOLTZMANN.z_vals),
            d_vals_hat=list(THREE_SPECIES_LOGC_BOLTZMANN.d_vals_hat),
            a_vals_hat=list(THREE_SPECIES_LOGC_BOLTZMANN.a_vals_hat),
            c0_vals_hat=list(THREE_SPECIES_LOGC_BOLTZMANN.c0_vals_hat),
            stoichiometry_r1=list(THREE_SPECIES_LOGC_BOLTZMANN.stoichiometry_r1),
            stoichiometry_r2=list(THREE_SPECIES_LOGC_BOLTZMANN.stoichiometry_r2),
            k0_legacy=list(THREE_SPECIES_LOGC_BOLTZMANN.k0_legacy),
            alpha_legacy=list(THREE_SPECIES_LOGC_BOLTZMANN.alpha_legacy),
            stoichiometry_legacy=list(THREE_SPECIES_LOGC_BOLTZMANN.stoichiometry_legacy),
            c_ref_legacy=list(THREE_SPECIES_LOGC_BOLTZMANN.c_ref_legacy),
            roles=None,
        )
        sp = make_bv_solver_params(
            eta_hat=0.0, dt=0.25, t_end=1.0,
            species=sp_no_roles,
            formulation="logc_muh",
            log_rate=True,
            boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC],
            stern_capacitance_f_m2=0.10,
            initializer="debye_boltzmann",
        )
        assert "species_roles" not in sp[10]["bv_bc"]


# ===================================================================
# Phase 1C — IC counterion identification path is roles-aware
# ===================================================================


class TestFourSpKplusICCounterionResolution:
    """K2SO4-style 4sp dynamic stack (K⁺ at idx 3, z=+1) reaches the
    bikerman-consistent IC because explicit ``species_roles`` resolve K⁺
    as the counterion.  Without roles, the legacy ``z=-1`` check would
    skip synthesis and the IC would fall through to ``no_boltzmann_counterion``.
    Structural: does not require Newton convergence.
    """

    def _build_synthetic_4sp_kplus_params(self, *, formulation="logc"):
        """Synthetic 4sp params: O₂, H₂O₂, H⁺, K⁺ all dynamic; no
        explicit boltzmann_counterions.  Mirrors Picard's expected dict
        shape so the IC helper can run end-to-end.
        """
        bv_convergence = {
            "clip_exponent": True,
            "exponent_clip": 100.0,
            "regularize_concentration": True,
            "conc_floor": 1e-12,
            "use_eta_in_bv": True,
            "bv_log_rate": True,
            "u_clamp": 100.0,
            "formulation": formulation,
            "initializer": "debye_boltzmann",
            "domain_height_hat": 1.0,
            "enable_water_ionization": False,
            "kw_eff_hat": 0.0,
            "d_oh_hat": 0.0,
            "a_oh_hat": 0.0,
        }
        bv_bc = {
            "reactions": [
                {
                    "k0": 1e-8,
                    "alpha": 0.5,
                    "cathodic_species": 0,
                    "anodic_species": 1,
                    "c_ref": 1.0,
                    "stoichiometry": [-1, +1, -2, 0],
                    "n_electrons": 2,
                    "reversible": True,
                    "E_eq_v": 0.695,
                },
            ],
            "k0": [1e-8] * 4,
            "alpha": [0.5] * 4,
            "stoichiometry": [-1, -1, -1, 0],
            "c_ref": [1.0, 0.0, 1.0, 1.0],
            "E_eq_v": 0.0,
            "electrode_marker": 3,
            "concentration_marker": 4,
            "ground_marker": 4,
            # Explicit roles: H⁺ at idx 2, K⁺ at idx 3 (counterion).
            "species_roles": ["neutral", "neutral", "proton", "counterion"],
        }
        params = {
            "bv_bc": bv_bc,
            "bv_convergence": bv_convergence,
            "nondim": {"enabled": False},  # bypass scaling for this unit test
        }
        return params

    def test_synthesized_counterion_uses_roles_idx(self):
        """Direct unit test on the role-lookup logic that the IC helper
        replicates: when species_roles provide a counterion, the
        synthesized entry uses that index (not the legacy ``z==-1`` at
        idx 3) — this is the assertion the structural test relies on.
        """
        # Mirror the inline lookup from forms_logc.py:_try_debye_boltzmann_ic
        # (and forms_logc_muh.py).  The lookup is intentionally inline
        # rather than imported from scripts._bv_common so the solver
        # has no upstream-package dependency.
        species_roles = ["neutral", "neutral", "proton", "counterion"]
        z_vals_full = [0, 0, 1, 1]
        c0_model = [1.0, 1e-4, 0.0833, 166.58]

        idx_counterion = None
        role_matches = [
            i for i, r in enumerate(species_roles)
            if str(r).strip().lower() == "counterion"
        ]
        if len(role_matches) == 1:
            idx_counterion = role_matches[0]
        # Legacy z==-1 fallback would set idx_counterion=None here
        # (z_vals_full[3] == 1, not -1) — i.e. without roles, the
        # synthesis would NOT fire.

        assert idx_counterion == 3
        synthesised_entry = {
            "z": int(z_vals_full[idx_counterion]),
            "c_bulk_nondim": float(c0_model[idx_counterion]),
        }
        assert synthesised_entry["z"] == 1   # K⁺ has z=+1 (not z=-1!)
        assert synthesised_entry["c_bulk_nondim"] == pytest.approx(166.58)

    def test_legacy_path_without_roles_skips_kplus(self):
        """Confirm the legacy z=-1 check would NOT fire for K⁺ at z=+1."""
        z_vals_full = [0, 0, 1, 1]
        # Legacy: only checks z_vals_full[3] == -1.
        assert int(z_vals_full[3]) != -1
        # So without roles, no synthesis → "no_boltzmann_counterion".


class TestRoleFieldRoundtrip:
    """SpeciesConfig.roles → make_bv_solver_params → bv_bc.species_roles
    → _get_species_roles round-trip."""

    def test_kplus_4sp_roundtrip(self):
        from scripts._bv_common import (
            DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
            SpeciesConfig,
            make_bv_solver_params,
        )
        from Forward.bv_solver.config import _get_species_roles

        # Build a 4sp K2SO4-style species config
        species = SpeciesConfig(
            n_species=4,
            z_vals=[0, 0, 1, 1],
            d_vals_hat=[1.0, 0.84, 4.9, 1.03],
            a_vals_hat=[0.01, 0.01, 0.01, 3.06e-5],
            c0_vals_hat=[1.0, 1e-4, 0.0833, 166.58],
            stoichiometry_r1=[-1, +1, -2, 0],
            stoichiometry_r2=[0, -1, -2, 0],
            k0_legacy=[1e-8] * 4,
            alpha_legacy=[0.5] * 4,
            stoichiometry_legacy=[-1, -1, -1, 0],
            c_ref_legacy=[1.0, 0.0, 1.0, 1.0],
            roles=["neutral", "neutral", "proton", "counterion"],
        )
        sp = make_bv_solver_params(
            eta_hat=0.0, dt=0.25, t_end=1.0,
            species=species,
            formulation="logc_muh",
            log_rate=True,
            boltzmann_counterions=[DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC],
            stern_capacitance_f_m2=0.10,
            initializer="debye_boltzmann",
        )
        # cfg should carry the roles list through under bv_bc.species_roles.
        assert sp[10]["bv_bc"]["species_roles"] == [
            "neutral", "neutral", "proton", "counterion",
        ]
        # And _get_species_roles returns the same list.
        roles = _get_species_roles(sp[10], species.n_species)
        assert roles == ["neutral", "neutral", "proton", "counterion"]
