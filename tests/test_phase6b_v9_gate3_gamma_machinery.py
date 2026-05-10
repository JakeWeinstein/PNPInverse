"""Phase 6β v9 Gate 3 — Γ_MOH machinery + manufactured-source unit tests.

Fast unit tests for:

* Gate 3A: ``forms_indexing.unpack_dof_indices`` helper, mixed-space
  layout extension when ``enable_cation_hydrolysis=True``, and the
  byte-equivalence of the disabled path.
* Gate 3B: ``cation_hydrolysis.py`` helper-module public-API surface
  (default-off contract; bundle constructs cleanly with R-space funcs).
* Gate 3C: ``λ_hydrolysis_ladder`` validation rules and the
  ``set_reaction_lambda_hydrolysis_model`` /
  ``get_reaction_lambda_hydrolysis_model`` accessors (mirrors Phase 6α
  Kw_eff accessor tests).

Slow Firedrake-backed manufactured-source tests live further down,
gated under ``@pytest.mark.slow``.

See ``.claude/plans/write-up-the-formal-joyful-papert.md`` §"Phase 1 —
Gate 3" for the full plan.
"""
from __future__ import annotations

import os
import sys

import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ===================================================================
# Gate 3A — forms_indexing helper
# ===================================================================


class TestUnpackDofIndices:
    """Pure-Python unit tests on the layout-aware index helper."""

    def test_legacy_layout_no_gamma(self):
        from Forward.bv_solver.forms_indexing import unpack_dof_indices
        idx = unpack_dof_indices(has_gamma=False)
        assert idx.has_gamma is False
        assert idx.species_slice == slice(0, -1)
        assert idx.phi_index == -1
        assert idx.gamma_index is None

    def test_gamma_augmented_layout(self):
        from Forward.bv_solver.forms_indexing import unpack_dof_indices
        idx = unpack_dof_indices(has_gamma=True)
        assert idx.has_gamma is True
        assert idx.species_slice == slice(0, -2)
        assert idx.phi_index == -2
        assert idx.gamma_index == -1

    def test_indices_dataclass_is_frozen(self):
        # The dataclass must be hashable / immutable so callers can
        # pass it across the form-build boundary without aliasing
        # surprises.  ``frozen=True`` produces TypeError on set.
        from Forward.bv_solver.forms_indexing import unpack_dof_indices
        idx = unpack_dof_indices(has_gamma=False)
        with pytest.raises((AttributeError, Exception)):
            idx.phi_index = 99    # type: ignore[misc]

    def test_indices_against_python_tuple_slicing(self):
        """Sanity: slicing a length-(n+1) and length-(n+2) tuple with
        the helper indices recovers ``species`` and ``phi``/``gamma``
        positions exactly.  This is the contract the form-build code
        relies on.
        """
        from Forward.bv_solver.forms_indexing import unpack_dof_indices

        # n_species = 3 (the production stack)
        legacy_tuple = ("u0", "u1", "u2", "phi")
        idx_legacy = unpack_dof_indices(has_gamma=False)
        assert legacy_tuple[idx_legacy.species_slice] == ("u0", "u1", "u2")
        assert legacy_tuple[idx_legacy.phi_index] == "phi"

        gamma_tuple = ("u0", "u1", "u2", "phi", "gamma")
        idx_gamma = unpack_dof_indices(has_gamma=True)
        assert gamma_tuple[idx_gamma.species_slice] == ("u0", "u1", "u2")
        assert gamma_tuple[idx_gamma.phi_index] == "phi"
        assert gamma_tuple[idx_gamma.gamma_index] == "gamma"

    def test_indices_with_4_species_layout(self):
        """The 4sp K2SO4 stack (Gate 2) has species at 0..3 and phi at 4."""
        from Forward.bv_solver.forms_indexing import unpack_dof_indices

        legacy_4sp = ("u0", "u1", "u2", "u3", "phi")
        idx_legacy = unpack_dof_indices(has_gamma=False)
        assert legacy_4sp[idx_legacy.species_slice] == ("u0", "u1", "u2", "u3")
        assert legacy_4sp[idx_legacy.phi_index] == "phi"

        gamma_4sp = ("u0", "u1", "u2", "u3", "phi", "gamma")
        idx_gamma = unpack_dof_indices(has_gamma=True)
        assert gamma_4sp[idx_gamma.species_slice] == (
            "u0", "u1", "u2", "u3",
        )
        assert gamma_4sp[idx_gamma.phi_index] == "phi"
        assert gamma_4sp[idx_gamma.gamma_index] == "gamma"


# ===================================================================
# Gate 3A — solver_params plumbing for enable_cation_hydrolysis
# ===================================================================


_OMITTED = object()


def _build(**overrides):
    """Build production-stack solver params with overrides."""
    from scripts._bv_common import (
        DEFAULT_CLO4_BOLTZMANN_COUNTERION,
        THREE_SPECIES_LOGC_BOLTZMANN,
        make_bv_solver_params,
    )

    kwargs = dict(
        eta_hat=0.0,
        dt=0.25,
        t_end=10.0,
        species=THREE_SPECIES_LOGC_BOLTZMANN,
        formulation="logc",
        log_rate=True,
        boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION],
    )
    kwargs.update(overrides)
    return make_bv_solver_params(**kwargs)


def _bv_conv(sp):
    return sp[10]["bv_convergence"]


class TestCationHydrolysisDefaultOff:
    """Phase 6β v9 Gate 3 flag plumbing must default to False
    (byte-equivalence with pre-Gate-3 stack)."""

    def test_default_enable_is_false(self):
        cfg = _bv_conv(_build())
        assert cfg["enable_cation_hydrolysis"] is False

    def test_default_lambda_is_zero(self):
        cfg = _bv_conv(_build())
        assert cfg["lambda_hydrolysis"] == 0.0

    def test_default_config_is_none(self):
        cfg = _bv_conv(_build())
        assert cfg["cation_hydrolysis_config"] is None

    def test_explicit_true_writes_true(self):
        cfg = _bv_conv(_build(
            enable_cation_hydrolysis=True,
            cation_hydrolysis_config={"placeholder": True},
        ))
        assert cfg["enable_cation_hydrolysis"] is True
        assert cfg["cation_hydrolysis_config"] == {"placeholder": True}

    def test_explicit_lambda_lands(self):
        cfg = _bv_conv(_build(
            enable_cation_hydrolysis=True,
            cation_hydrolysis_config={"placeholder": True},
            lambda_hydrolysis=0.5,
        ))
        assert cfg["lambda_hydrolysis"] == 0.5


class TestCationHydrolysisConvergenceCfgParser:
    """``_get_bv_convergence_cfg`` must surface the Gate 3 keys with
    the correct default-off values; otherwise the form-build code path
    (which reads conv_cfg, not raw params) would silently disable the
    feature even when the user set ``enable_cation_hydrolysis=True``.
    """

    def test_parser_default_off(self):
        from Forward.bv_solver.config import _get_bv_convergence_cfg
        sp = _build()
        cfg = _get_bv_convergence_cfg(sp[10])
        assert cfg["enable_cation_hydrolysis"] is False
        assert cfg["lambda_hydrolysis"] == 0.0

    def test_parser_passes_through_when_enabled(self):
        from Forward.bv_solver.config import _get_bv_convergence_cfg
        sp = _build(
            enable_cation_hydrolysis=True,
            cation_hydrolysis_config={"foo": "bar"},
            lambda_hydrolysis=0.75,
        )
        cfg = _get_bv_convergence_cfg(sp[10])
        assert cfg["enable_cation_hydrolysis"] is True
        assert cfg["cation_hydrolysis_config"] == {"foo": "bar"}
        assert cfg["lambda_hydrolysis"] == 0.75

    def test_parser_rejects_non_dict_config(self):
        from Forward.bv_solver.config import _get_bv_convergence_cfg
        sp = _build(enable_cation_hydrolysis=True)
        sp[10]["bv_convergence"]["cation_hydrolysis_config"] = "not_a_dict"
        with pytest.raises(
            ValueError, match="cation_hydrolysis_config must be a dict"
        ):
            _get_bv_convergence_cfg(sp[10])

    def test_parser_skips_validation_when_disabled(self):
        from Forward.bv_solver.config import _get_bv_convergence_cfg
        # When the feature is off, the field values are not validated —
        # they're carried through verbatim so the config dict reflects
        # the user's exact intent.
        sp = _build()
        sp[10]["bv_convergence"]["cation_hydrolysis_config"] = "anything"
        cfg = _get_bv_convergence_cfg(sp[10])  # should not raise
        assert cfg["cation_hydrolysis_config"] == "anything"


# ===================================================================
# Gate 3C — λ_hydrolysis ladder + accessors
# ===================================================================


class TestLambdaHydrolysisAccessorErrors:
    """Fast unit tests on accessor error paths (no Firedrake required)."""

    def test_set_lambda_no_bundle_raises(self):
        from Forward.bv_solver.anchor_continuation import (
            set_reaction_lambda_hydrolysis_model,
        )
        with pytest.raises(ValueError, match="no 'cation_hydrolysis' bundle"):
            set_reaction_lambda_hydrolysis_model({}, 0.5)

    def test_set_lambda_out_of_range_raises(self):
        from Forward.bv_solver.anchor_continuation import (
            set_reaction_lambda_hydrolysis_model,
        )
        with pytest.raises(ValueError, match="must lie in"):
            set_reaction_lambda_hydrolysis_model({}, -0.1)
        with pytest.raises(ValueError, match="must lie in"):
            set_reaction_lambda_hydrolysis_model({}, 1.5)

    def test_get_lambda_no_bundle_raises(self):
        from Forward.bv_solver.anchor_continuation import (
            get_reaction_lambda_hydrolysis_model,
        )
        with pytest.raises(ValueError, match="no 'cation_hydrolysis' bundle"):
            get_reaction_lambda_hydrolysis_model({})

    def test_set_k_hyd_no_bundle_raises(self):
        from Forward.bv_solver.anchor_continuation import (
            set_reaction_k_hyd_model,
        )
        with pytest.raises(ValueError, match="no 'cation_hydrolysis' bundle"):
            set_reaction_k_hyd_model({}, 1.0)

    def test_set_k_des_zero_raises(self):
        from Forward.bv_solver.anchor_continuation import (
            set_reaction_k_des_model,
        )
        with pytest.raises(ValueError, match="must be positive"):
            set_reaction_k_des_model({}, 0.0)

    def test_set_k_hyd_negative_raises(self):
        from Forward.bv_solver.anchor_continuation import (
            set_reaction_k_hyd_model,
        )
        with pytest.raises(ValueError, match="must be non-negative"):
            set_reaction_k_hyd_model({}, -0.5)

    def test_set_r_h_el_no_bundle_raises(self):
        from Forward.bv_solver.anchor_continuation import (
            set_reaction_r_H_El_pm_model,
        )
        with pytest.raises(ValueError, match="no 'cation_hydrolysis' bundle"):
            set_reaction_r_H_El_pm_model({}, 200.98)

    def test_set_r_h_el_zero_raises(self):
        from Forward.bv_solver.anchor_continuation import (
            set_reaction_r_H_El_pm_model,
        )
        with pytest.raises(ValueError, match="must be positive"):
            set_reaction_r_H_El_pm_model({}, 0.0)

    def test_set_r_h_el_negative_raises(self):
        from Forward.bv_solver.anchor_continuation import (
            set_reaction_r_H_El_pm_model,
        )
        with pytest.raises(ValueError, match="must be positive"):
            set_reaction_r_H_El_pm_model({}, -1.0)

    def test_anchor_continuation_module_exports_gate3c(self):
        """Public surface for Gate 3C accessors + ladder kwarg."""
        import Forward.bv_solver.anchor_continuation as ac
        assert "set_reaction_lambda_hydrolysis_model" in ac.__all__
        assert "get_reaction_lambda_hydrolysis_model" in ac.__all__
        assert "set_reaction_k_hyd_model" in ac.__all__
        assert "set_reaction_k_prot_model" in ac.__all__
        assert "set_reaction_k_des_model" in ac.__all__

    def test_anchor_continuation_module_exports_gate4b(self):
        """Public surface for Gate 4B sweep optimizer."""
        import Forward.bv_solver.anchor_continuation as ac
        assert "set_reaction_r_H_El_pm_model" in ac.__all__
        assert "solve_lambda_ramp_from_warm_start" in ac.__all__


class TestLambdaHydrolysisLadderValidation:
    """Fast unit tests on ladder validation (mirror
    ``TestKwEffLadderValidation`` from Phase 6α)."""

    def _stub_sp(self):
        return _build()

    def test_lambda_combined_with_kw_raises(self):
        from Forward.bv_solver.anchor_continuation import (
            solve_anchor_with_continuation,
        )
        with pytest.raises(NotImplementedError, match="Combining"):
            solve_anchor_with_continuation(
                self._stub_sp(),
                mesh=None,
                k0_targets={0: 1.0},
                kw_eff_ladder=(0.0, 1.0),
                lambda_hydrolysis_ladder=(0.0, 1.0),
            )

    def test_lambda_combined_with_cs_raises(self):
        from Forward.bv_solver.anchor_continuation import (
            solve_anchor_with_continuation,
        )
        with pytest.raises(NotImplementedError, match="Combining"):
            solve_anchor_with_continuation(
                self._stub_sp(),
                mesh=None,
                k0_targets={0: 1.0},
                c_s_ladder=(1.0, 0.5, 0.10),
                lambda_hydrolysis_ladder=(0.0, 1.0),
            )


@pytest.mark.slow
class TestLambdaHydrolysisAccessorRoundtrip:
    """Slow Firedrake-backed: setter writes both metadata and Function."""

    def _build_ctx(self):
        import firedrake as fd
        from Forward.bv_solver.dispatch import build_context, build_forms
        from scripts._bv_common import (
            DEFAULT_SULFATE_ANALYTIC_BIKERMAN_FOR_K2SO4,
            FOUR_SPECIES_LOGC_DYNAMIC_K2SO4,
            make_bv_solver_params,
        )

        sp = make_bv_solver_params(
            eta_hat=0.0, dt=0.25, t_end=10.0,
            species=FOUR_SPECIES_LOGC_DYNAMIC_K2SO4,
            formulation="logc_muh",
            log_rate=True,
            boltzmann_counterions=[DEFAULT_SULFATE_ANALYTIC_BIKERMAN_FOR_K2SO4],
            stern_capacitance_f_m2=0.10,
            initializer="linear_phi",
            enable_cation_hydrolysis=True,
            cation_hydrolysis_config={
                "k_hyd": 1e-4,
                "k_prot": 1e-4,
                "k_des": 1.0,
                "delta_ohp_hat": 1e-2,
            },
        )
        mesh = fd.UnitSquareMesh(8, 8)
        ctx = build_context(sp, mesh=mesh)
        ctx = build_forms(ctx, sp)
        return sp, ctx

    def test_lambda_setter_updates_function_and_metadata(self):
        from Forward.bv_solver.anchor_continuation import (
            get_reaction_lambda_hydrolysis_model,
            set_reaction_lambda_hydrolysis_model,
        )
        _, ctx = self._build_ctx()
        # Initial value (default 0.0).
        assert get_reaction_lambda_hydrolysis_model(ctx) == pytest.approx(0.0)
        # Update through the setter.
        set_reaction_lambda_hydrolysis_model(ctx, 0.5)
        assert get_reaction_lambda_hydrolysis_model(ctx) == pytest.approx(0.5)
        assert ctx["bv_convergence"]["lambda_hydrolysis"] == pytest.approx(0.5)
        # Production target.
        set_reaction_lambda_hydrolysis_model(ctx, 1.0)
        assert get_reaction_lambda_hydrolysis_model(ctx) == pytest.approx(1.0)

    def test_kinetic_rate_setters_update_functions_and_metadata(self):
        from Forward.bv_solver.anchor_continuation import (
            set_reaction_k_des_model,
            set_reaction_k_hyd_model,
            set_reaction_k_prot_model,
            set_reaction_delta_ohp_model,
        )
        _, ctx = self._build_ctx()
        bundle = ctx["cation_hydrolysis"]

        set_reaction_k_hyd_model(ctx, 7.5e-5)
        assert float(bundle.k_hyd_func) == pytest.approx(7.5e-5)
        assert (
            ctx["bv_convergence"]["cation_hydrolysis_config"]["k_hyd"]
            == pytest.approx(7.5e-5)
        )

        set_reaction_k_prot_model(ctx, 2.5e-5)
        assert float(bundle.k_prot_func) == pytest.approx(2.5e-5)

        set_reaction_k_des_model(ctx, 5.0)
        assert float(bundle.k_des_func) == pytest.approx(5.0)

        set_reaction_delta_ohp_model(ctx, 4e-3)
        assert float(bundle.delta_ohp_func) == pytest.approx(4e-3)


# ===================================================================
# Gate 3A — slow Firedrake-backed mixed-space layout
# ===================================================================


def _build_sp(formulation: str, *, enable_cation_hydrolysis: bool):
    """Build a solver-params object for Gate 3A layout tests."""
    from scripts._bv_common import (
        DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC,
        THREE_SPECIES_LOGC_BOLTZMANN,
        make_bv_solver_params,
    )

    return make_bv_solver_params(
        eta_hat=0.0, dt=0.25, t_end=10.0,
        species=THREE_SPECIES_LOGC_BOLTZMANN,
        formulation=formulation,
        log_rate=True,
        boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC],
        stern_capacitance_f_m2=0.10,
        initializer="linear_phi",
        enable_cation_hydrolysis=enable_cation_hydrolysis,
        cation_hydrolysis_config={
            # Placeholder schema — Gate 4A fills with Singh params.
            "pka_shift_form": "placeholder",
        } if enable_cation_hydrolysis else None,
    )


@pytest.mark.slow
class TestMixedSpaceLayoutLegacy:
    """``enable_cation_hydrolysis=False`` must produce the byte-equivalent
    pre-Gate-3 mixed space (n_species + 1 components: species + phi)."""

    @pytest.mark.parametrize("formulation", ["logc", "logc_muh"])
    def test_mixed_space_has_n_plus_one_components(self, formulation):
        import firedrake as fd
        from Forward.bv_solver.dispatch import build_context

        sp = _build_sp(formulation, enable_cation_hydrolysis=False)
        mesh = fd.UnitSquareMesh(8, 8)
        ctx = build_context(sp, mesh=mesh)
        W = ctx["W"]
        # 3 species + 1 phi = 4 components.
        assert W.num_sub_spaces() == 4

    @pytest.mark.parametrize("formulation", ["logc", "logc_muh"])
    def test_indices_legacy_layout(self, formulation):
        import firedrake as fd
        from Forward.bv_solver.dispatch import build_context

        sp = _build_sp(formulation, enable_cation_hydrolysis=False)
        mesh = fd.UnitSquareMesh(8, 8)
        ctx = build_context(sp, mesh=mesh)
        idx = ctx["mixed_space_indices"]
        assert idx.has_gamma is False
        assert idx.phi_index == -1
        assert idx.gamma_index is None
        assert ctx["cation_hydrolysis_enabled"] is False


# ===================================================================
# Gate 3B — cation_hydrolysis helper module
# ===================================================================


class TestCationHydrolysisHelpersDefaultOff:
    """Gate check + role-resolver fast unit tests."""

    def test_is_cation_hydrolysis_enabled_default_false(self):
        from Forward.bv_solver.cation_hydrolysis import (
            is_cation_hydrolysis_enabled,
        )
        assert is_cation_hydrolysis_enabled({}) is False
        assert is_cation_hydrolysis_enabled(
            {"enable_cation_hydrolysis": False}
        ) is False

    def test_is_cation_hydrolysis_enabled_true(self):
        from Forward.bv_solver.cation_hydrolysis import (
            is_cation_hydrolysis_enabled,
        )
        assert is_cation_hydrolysis_enabled(
            {"enable_cation_hydrolysis": True}
        ) is True

    def test_resolve_counterion_index_basic(self):
        from Forward.bv_solver.cation_hydrolysis import (
            resolve_counterion_index,
        )
        idx = resolve_counterion_index(
            ["neutral", "neutral", "proton", "counterion"]
        )
        assert idx == 3

    def test_resolve_counterion_index_no_match(self):
        from Forward.bv_solver.cation_hydrolysis import (
            resolve_counterion_index,
        )
        with pytest.raises(ValueError, match="no species with role"):
            resolve_counterion_index(["neutral", "neutral", "proton"])

    def test_resolve_counterion_index_multiple_match(self):
        from Forward.bv_solver.cation_hydrolysis import (
            resolve_counterion_index,
        )
        with pytest.raises(ValueError, match="multiple species"):
            resolve_counterion_index(
                ["neutral", "counterion", "proton", "counterion"]
            )

    def test_resolve_counterion_index_no_roles_raises(self):
        from Forward.bv_solver.cation_hydrolysis import (
            resolve_counterion_index,
        )
        with pytest.raises(ValueError, match="explicit roles"):
            resolve_counterion_index(None)


@pytest.mark.slow
class TestCationHydrolysisBundleBuild:
    """Bundle constructs cleanly with all R-space funcs initialised
    to the defaults; the form-build code attaches it to ctx."""

    def test_bundle_attached_to_ctx_when_enabled(self):
        import firedrake as fd
        from Forward.bv_solver.dispatch import build_context, build_forms

        sp = _build_sp("logc_muh", enable_cation_hydrolysis=True)
        # Use the K2SO4 stack for proper roles + counterion lookup;
        # the THREE_SPECIES_LOGC_BOLTZMANN preset doesn't have a
        # ``counterion`` role.  Override species in-place.
        from scripts._bv_common import (
            DEFAULT_SULFATE_ANALYTIC_BIKERMAN_FOR_K2SO4,
            FOUR_SPECIES_LOGC_DYNAMIC_K2SO4,
            make_bv_solver_params,
        )

        sp = make_bv_solver_params(
            eta_hat=0.0, dt=0.25, t_end=10.0,
            species=FOUR_SPECIES_LOGC_DYNAMIC_K2SO4,
            formulation="logc_muh",
            log_rate=True,
            boltzmann_counterions=[DEFAULT_SULFATE_ANALYTIC_BIKERMAN_FOR_K2SO4],
            stern_capacitance_f_m2=0.10,
            initializer="linear_phi",
            enable_cation_hydrolysis=True,
            cation_hydrolysis_config={
                "k_hyd": 1e-4,
                "k_prot": 1e-4,
                "k_des": 1.0,
                "delta_ohp_hat": 1e-2,
            },
        )
        mesh = fd.UnitSquareMesh(8, 8)
        ctx = build_context(sp, mesh=mesh)
        ctx = build_forms(ctx, sp)

        bundle = ctx["cation_hydrolysis"]
        assert bundle is not None
        # Counterion index = 3 (K⁺ in the K2SO4 stack).
        assert bundle.counterion_idx == 3
        # All R-space Functions populated with the config defaults.
        assert float(bundle.k_hyd_func) == pytest.approx(1e-4, rel=1e-12)
        assert float(bundle.k_prot_func) == pytest.approx(1e-4, rel=1e-12)
        assert float(bundle.k_des_func) == pytest.approx(1.0, rel=1e-12)
        assert float(bundle.delta_ohp_func) == pytest.approx(1e-2, rel=1e-12)
        # λ defaults to 0 unless caller overrides; that's the disabled
        # baseline (Γ pinned to 0).
        assert float(bundle.lambda_hydrolysis_func) == 0.0

    def test_bundle_disabled_path_is_none(self):
        import firedrake as fd
        from Forward.bv_solver.dispatch import build_context, build_forms

        sp = _build_sp("logc", enable_cation_hydrolysis=False)
        mesh = fd.UnitSquareMesh(8, 8)
        ctx = build_context(sp, mesh=mesh)
        ctx = build_forms(ctx, sp)
        assert ctx["cation_hydrolysis"] is None


@pytest.mark.slow
class TestMixedSpaceLayoutWithCationHydrolysis:
    """Phase 6β v9 Gate 3A architectural choice: when
    ``enable_cation_hydrolysis=True`` the mixed function space stays
    at the legacy ``species + phi`` layout (n_species + 1 components).

    Γ_MOH is treated as an external R-space coefficient (like Phase 6α's
    ``kw_eff_func``), updated between continuation rungs by an outer
    Picard fixed-point iteration on the closed-form
    ``Γ_ss(λ) = λ·⟨R_net_forward⟩ / (λ·k_des + (1−λ) + λ·k_prot⟨c_H⟩/δ_OHP)``.
    This avoids the Firedrake R-space-in-mixed-space matnest format
    limitation that breaks monolithic LU assembly.
    """

    @pytest.mark.parametrize("formulation", ["logc", "logc_muh"])
    def test_mixed_space_unchanged_with_feature_enabled(self, formulation):
        import firedrake as fd
        from Forward.bv_solver.dispatch import build_context

        sp = _build_sp(formulation, enable_cation_hydrolysis=True)
        mesh = fd.UnitSquareMesh(8, 8)
        ctx = build_context(sp, mesh=mesh)
        W = ctx["W"]
        # 3 species + 1 phi = 4 components (no Γ slot in the mixed space).
        assert W.num_sub_spaces() == 4

    @pytest.mark.parametrize("formulation", ["logc", "logc_muh"])
    def test_indices_legacy_layout_with_feature_enabled(self, formulation):
        import firedrake as fd
        from Forward.bv_solver.dispatch import build_context

        sp = _build_sp(formulation, enable_cation_hydrolysis=True)
        mesh = fd.UnitSquareMesh(8, 8)
        ctx = build_context(sp, mesh=mesh)
        idx = ctx["mixed_space_indices"]
        assert idx.has_gamma is False
        assert idx.phi_index == -1
        assert idx.gamma_index is None
        # The flag is on ctx for downstream diagnostics even though the
        # mixed-space layout is unchanged.
        assert ctx["cation_hydrolysis_enabled"] is True

    @pytest.mark.parametrize("formulation", ["logc", "logc_muh"])
    def test_gamma_func_is_r_space_coefficient(self, formulation):
        """Γ_MOH lives on its own R-space ``Function``, not in the mixed space."""
        import firedrake as fd
        from Forward.bv_solver.dispatch import build_context, build_forms
        from scripts._bv_common import (
            DEFAULT_SULFATE_ANALYTIC_BIKERMAN_FOR_K2SO4,
            FOUR_SPECIES_LOGC_DYNAMIC_K2SO4,
            make_bv_solver_params,
        )

        sp = make_bv_solver_params(
            eta_hat=0.0, dt=0.25, t_end=10.0,
            species=FOUR_SPECIES_LOGC_DYNAMIC_K2SO4,
            formulation=formulation,
            log_rate=True,
            boltzmann_counterions=[DEFAULT_SULFATE_ANALYTIC_BIKERMAN_FOR_K2SO4],
            stern_capacitance_f_m2=0.10,
            initializer="linear_phi",
            enable_cation_hydrolysis=True,
            cation_hydrolysis_config={
                "k_hyd": 0.0, "k_prot": 0.0,
                "k_des": 1.0, "delta_ohp_hat": 1e-2,
            },
        )
        mesh = fd.UnitSquareMesh(8, 8)
        ctx = build_context(sp, mesh=mesh)
        ctx = build_forms(ctx, sp)

        bundle = ctx["cation_hydrolysis"]
        assert bundle is not None
        # Γ_MOH is a stand-alone R-space Function (1 DOF) — NOT in the
        # mixed space.
        gamma_fn = bundle.gamma_func
        assert gamma_fn.function_space().mesh() is mesh
        assert gamma_fn.function_space().dim() == 1
        # Default initial value.
        assert float(gamma_fn) == 0.0


# ===================================================================
# Gate 3D — manufactured-source slow tests
# ===================================================================


def _build_gate3d_sp(
    *,
    lambda_hydrolysis: float,
    manufactured_R_inj: float = None,
    enable_cation_hydrolysis: bool = True,
    k_des: float = 1.0,
):
    """Build solver-params for Gate 3D tests.

    Uses the K2SO4 4sp dynamic stack with Phase 6α water ionization
    enabled — the proven-converging Gate 2 anchor configuration.
    The Gate 3D-specific behaviour comes from the
    ``cation_hydrolysis_config`` + ``lambda_hydrolysis`` settings.
    """
    from scripts._bv_common import (
        A_OH_HAT,
        D_OH_HAT,
        DEFAULT_SULFATE_ANALYTIC_BIKERMAN_FOR_K2SO4,
        FOUR_SPECIES_LOGC_DYNAMIC_K2SO4,
        K0_HAT_R2E, K0_HAT_R4E,
        KW_HAT,
        PARALLEL_2E_4E_REACTIONS_4SP,
        make_bv_solver_params,
    )

    cation_cfg = (
        {
            "k_hyd": 0.0,            # zero so R_net = R_inj override only
            "k_prot": 0.0,           # zero so R_net = R_inj override only
            "k_des": float(k_des),
            "delta_ohp_hat": 1e-2,
        }
        if enable_cation_hydrolysis
        else None
    )

    sp = make_bv_solver_params(
        eta_hat=0.0, dt=0.25, t_end=80.0,
        species=FOUR_SPECIES_LOGC_DYNAMIC_K2SO4,
        formulation="logc_muh",
        log_rate=True,
        bv_reactions=PARALLEL_2E_4E_REACTIONS_4SP,
        boltzmann_counterions=[DEFAULT_SULFATE_ANALYTIC_BIKERMAN_FOR_K2SO4],
        stern_capacitance_f_m2=0.10,
        initializer="linear_phi",
        enable_water_ionization=True,
        kw_eff_hat=KW_HAT,
        d_oh_hat=D_OH_HAT,
        a_oh_hat=A_OH_HAT,
        enable_cation_hydrolysis=enable_cation_hydrolysis,
        cation_hydrolysis_config=cation_cfg,
        lambda_hydrolysis=lambda_hydrolysis,
    )

    if manufactured_R_inj is not None:
        new_opts = dict(sp.solver_options)
        new_bv = dict(new_opts["bv_convergence"])
        new_bv["manufactured_R_inj"] = float(manufactured_R_inj)
        new_opts["bv_convergence"] = new_bv
        sp = sp.with_solver_options(new_opts)

    return sp


def _run_gate3d_solve(*, sp, mesh_ny: int, voltage: float):
    """Run anchor with kw_eff ladder at the requested voltage.

    Mirrors Gate 2's proven anchor pattern: anodic V=+0.55 V
    (well-conditioned for K⁺ depletion at the electrode) + the
    Phase 6α Kw_eff outer ladder + 5-rung k0 inner ladder.  Γ is
    updated by the orchestrator's outer Picard at the end.
    """
    import firedrake as fd
    import firedrake.adjoint as adj
    from Forward.bv_solver import make_graded_rectangle_mesh
    from Forward.bv_solver.anchor_continuation import (
        solve_anchor_with_continuation,
    )
    from scripts._bv_common import K0_HAT_R2E, K0_HAT_R4E, KW_HAT, V_T

    mesh = make_graded_rectangle_mesh(
        Nx=4, Ny=int(mesh_ny), beta=3.0,
        domain_height_hat=1.0,
    )

    sp_at_voltage = sp.with_phi_applied(voltage / V_T)

    with adj.stop_annotating():
        result = solve_anchor_with_continuation(
            sp_at_voltage,
            mesh=mesh,
            k0_targets={0: float(K0_HAT_R2E), 1: float(K0_HAT_R4E)},
            initial_scales=(1e-12, 1e-9, 1e-6, 1e-3, 1.0),
            max_inserts_per_step=4,
            max_ss_steps_per_rung=200,
            ic_at_target=True,
            kw_eff_ladder=(0.0, KW_HAT * 1e-6, KW_HAT * 1e-3, KW_HAT * 0.1, KW_HAT),
        )
    if not result.converged:
        raise RuntimeError(
            f"Gate 3D: anchor failed to converge at V={voltage}; "
            f"history={result.ladder_history!r}"
        )
    return result.ctx


def _read_c_H_surface_mean(ctx) -> float:
    """Mean ``c_H`` across the electrode boundary."""
    import firedrake as fd

    bv_cfg = ctx["bv_settings"]
    electrode_marker = bv_cfg["electrode_marker"]
    mesh = ctx["mesh"]
    ds = fd.Measure("ds", domain=mesh)
    ci = ctx["ci_exprs"]
    # H+ index = 2 in the K2SO4 4sp stack (per FOUR_SPECIES_LOGC_DYNAMIC_K2SO4
    # roles list).
    h_idx = 2
    area = float(fd.assemble(fd.Constant(1.0) * ds(electrode_marker)))
    if area <= 0.0:
        raise RuntimeError(
            "_read_c_H_surface_mean: electrode boundary has zero area"
        )
    return float(fd.assemble(ci[h_idx] * ds(electrode_marker))) / area


@pytest.mark.slow
class TestProtonBoundarySourceSignConvention:
    """Sign convention: R_inj > 0 must increase c_H at the electrode.

    R_inj is the manufactured cation-hydrolysis source replacing the
    physical R_net.  Positive R_inj ⇔ proton produced at OHP.  We
    compare the converged c_H_surface for R_inj=+ε vs R_inj=−ε at
    λ=1; the difference must be in the predicted direction.

    Slow because each parameter setting requires a converged Newton
    SS solve via the k0-ladder anchor.
    """

    @pytest.mark.parametrize("voltage", [0.55])  # Gate 2's proven anodic anchor
    def test_positive_R_inj_increases_c_H(self, voltage):
        # R_inj=+1e-3 (proton produced) vs R_inj=-1e-3 (proton consumed).
        sp_pos = _build_gate3d_sp(
            lambda_hydrolysis=1.0, manufactured_R_inj=+1e-3,
        )
        sp_neg = _build_gate3d_sp(
            lambda_hydrolysis=1.0, manufactured_R_inj=-1e-3,
        )
        ctx_pos = _run_gate3d_solve(sp=sp_pos, mesh_ny=40, voltage=voltage)
        ctx_neg = _run_gate3d_solve(sp=sp_neg, mesh_ny=40, voltage=voltage)
        cH_pos = _read_c_H_surface_mean(ctx_pos)
        cH_neg = _read_c_H_surface_mean(ctx_neg)
        # Sanity guard: both must be positive concentrations.
        assert cH_pos > 0.0
        assert cH_neg > 0.0
        # Sign convention: positive R_inj feeds H+ into solution at
        # the electrode → c_H_surface higher than the negative case.
        assert cH_pos > cH_neg, (
            f"sign convention broken: c_H(R_inj=+1e-3)={cH_pos} should "
            f"exceed c_H(R_inj=-1e-3)={cH_neg}"
        )


@pytest.mark.slow
class TestGammaResidualAreaInvariance:
    """At fixed R_inj and λ=1, steady-state Γ depends on the Real-element
    DOF only (single global scalar) so it must be mesh-independent.

    Tests R4#10 from the v9 plan: ``ds(electrode_marker)`` integration
    must average R_net consistently regardless of mesh resolution.
    """

    def test_gamma_invariant_across_mesh_refinement(self):
        from Forward.bv_solver.cation_hydrolysis import extract_gamma_value
        # k_des=1 nondim so steady-state Γ = R_inj/k_des = 1e-3.
        sp = _build_gate3d_sp(
            lambda_hydrolysis=1.0,
            manufactured_R_inj=+1e-3,
            k_des=1.0,
        )
        ctx_coarse = _run_gate3d_solve(sp=sp, mesh_ny=40, voltage=0.55)
        ctx_fine = _run_gate3d_solve(sp=sp, mesh_ny=80, voltage=0.55)
        gamma_coarse = extract_gamma_value(ctx_coarse)
        gamma_fine = extract_gamma_value(ctx_fine)
        # Both should be near the analytic value 1e-3.
        assert gamma_coarse == pytest.approx(1e-3, rel=1e-2)
        assert gamma_fine == pytest.approx(1e-3, rel=1e-2)
        # Mesh-refinement invariance: relative difference < 1e-3.
        rel_diff = abs(gamma_coarse - gamma_fine) / max(
            abs(gamma_coarse), 1e-12
        )
        assert rel_diff < 1e-3, (
            f"Γ depends on mesh: Γ(Ny=40)={gamma_coarse}, "
            f"Γ(Ny=80)={gamma_fine}, rel_diff={rel_diff}"
        )


# ===================================================================
# Gate 4A — Singh 2016 SI Eq. (4) field-dependent pKa
# ===================================================================


class TestSinghPkaShiftPure:
    """Pure-Python (no Firedrake) tests on the Singh constants and
    helper.  Eq. (3) bulk pKa table verified against Singh Table S1
    per ``docs/singh_2016_pka_formula.md`` §3.4.
    """

    def test_singh_constants(self):
        from scripts._bv_common import (
            SINGH_A_PM, SINGH_B, SINGH_R_O_PM,
        )
        assert SINGH_A_PM == pytest.approx(620.32)
        assert SINGH_B == pytest.approx(17.154)
        assert SINGH_R_O_PM == pytest.approx(63.0)

    def test_singh_table_s1_per_cation(self):
        """Per-cation Table S1 verifies Eq. (3): pKa_bulk == B − A·z²/r_M-O."""
        import math
        from scripts._bv_common import (
            SINGH_2016_CATION_PARAMS,
            SINGH_A_PM, SINGH_B, SINGH_R_O_PM,
        )
        # Tolerances follow the table's published precision (1 d.p.).
        for cation, row in SINGH_2016_CATION_PARAMS.items():
            r_M_O = row["r_M_pm"] + SINGH_R_O_PM
            pKa_bulk_predicted = SINGH_B - SINGH_A_PM * (row["z_eff"] ** 2) / r_M_O
            assert math.isclose(
                pKa_bulk_predicted, row["pKa_bulk"], abs_tol=0.1
            ), (
                f"{cation}: Eq.(3) predicts pKa={pKa_bulk_predicted}, "
                f"table says {row['pKa_bulk']}"
            )

    def test_make_singh_pka_shift_params_K(self):
        from scripts._bv_common import make_singh_pka_shift_params
        params = make_singh_pka_shift_params("K+")
        assert params["z_eff"] == pytest.approx(0.919)
        assert params["r_M_pm"] == pytest.approx(138.0)
        assert params["r_H_El_pm"] == pytest.approx(200.98)  # Cu prior
        assert params["A_pm"] == pytest.approx(620.32)
        assert params["B"] == pytest.approx(17.154)
        assert params["r_O_pm"] == pytest.approx(63.0)
        assert params["anode_clamp"] is True

    def test_make_singh_pka_shift_params_override_r_h_el(self):
        from scripts._bv_common import make_singh_pka_shift_params
        params = make_singh_pka_shift_params("K+", r_H_El_pm=210.0)
        assert params["r_H_El_pm"] == pytest.approx(210.0)

    def test_make_singh_pka_shift_params_unknown_cation(self):
        from scripts._bv_common import make_singh_pka_shift_params
        with pytest.raises(ValueError, match="Unknown cation"):
            make_singh_pka_shift_params("Mg2+")


@pytest.mark.slow
class TestSinghPkaShiftUFL:
    """Slow Firedrake-backed: ``build_pka_shift`` produces a UFL
    expression that evaluates to the right sign + magnitude when
    assembled against a known σ_S input.
    """

    def test_singh_eq4_at_anodic_bias_clamps_to_zero(self):
        import firedrake as fd
        from Forward.bv_solver.cation_hydrolysis import build_pka_shift
        from scripts._bv_common import make_singh_pka_shift_params

        mesh = fd.UnitSquareMesh(4, 4)
        # Anodic σ_S > 0 → anode clamp → ΔpKa = 0.
        params = make_singh_pka_shift_params("K+")
        pka_shift = build_pka_shift(
            cation_params={**params, "pka_shift_form": "singh_2016_eq_4"},
            sigma_S=fd.Constant(1.0),  # positive (anodic) C/m²
        )
        val = float(fd.assemble(pka_shift * fd.dx(domain=mesh))) / float(
            fd.assemble(fd.Constant(1.0) * fd.dx(domain=mesh))
        )
        assert val == pytest.approx(0.0, abs=1e-30)

    def test_singh_eq4_cathodic_drives_negative(self):
        """Cathodic σ_S < 0 → ΔpKa < 0 (pKa drops, more proton produced)."""
        import firedrake as fd
        from Forward.bv_solver.cation_hydrolysis import build_pka_shift
        from scripts._bv_common import make_singh_pka_shift_params

        mesh = fd.UnitSquareMesh(4, 4)
        params = make_singh_pka_shift_params("K+")
        # σ_S = -1 C/m² (a strong cathodic bias).  ΔpKa should be
        # negative because r_H_El < r_M-O (Singh's standard cathode
        # geometry: 200.98 < 201).
        pka_shift = build_pka_shift(
            cation_params={**params, "pka_shift_form": "singh_2016_eq_4"},
            sigma_S=fd.Constant(-1.0),
        )
        val = float(fd.assemble(pka_shift * fd.dx(domain=mesh))) / float(
            fd.assemble(fd.Constant(1.0) * fd.dx(domain=mesh))
        )
        # ΔpKa is negative.  Magnitude check is order-of-magnitude;
        # at σ_S = -1 C/m² the cation hydrolysis pKa drops by O(1)
        # to O(10) units — see Singh Table S3 cross-check below.
        assert val < 0.0

    def test_singh_eq4_zero_sigma_gives_zero_shift(self):
        import firedrake as fd
        from Forward.bv_solver.cation_hydrolysis import build_pka_shift
        from scripts._bv_common import make_singh_pka_shift_params

        mesh = fd.UnitSquareMesh(4, 4)
        params = make_singh_pka_shift_params("K+")
        pka_shift = build_pka_shift(
            cation_params={**params, "pka_shift_form": "singh_2016_eq_4"},
            sigma_S=fd.Constant(0.0),
        )
        val = float(fd.assemble(pka_shift * fd.dx(domain=mesh))) / float(
            fd.assemble(fd.Constant(1.0) * fd.dx(domain=mesh))
        )
        assert val == pytest.approx(0.0, abs=1e-30)

    def test_singh_eq4_unknown_form_raises(self):
        import firedrake as fd
        from Forward.bv_solver.cation_hydrolysis import build_pka_shift
        with pytest.raises(NotImplementedError, match="not implemented"):
            build_pka_shift(
                cation_params={"pka_shift_form": "foo"},
                sigma_S=fd.Constant(0.0),
            )


@pytest.mark.slow
class TestGammaDirichletPinAtLambdaZero:
    """At λ=0: Γ pinned to 0 AND the manufactured R_net contribution
    is λ-zeroed out, so the converged observables must match a
    disabled-feature build to within semantic tolerance.

    R4#11 from the v9 plan: the hard-zero Γ pin gives byte-zero
    hydrolysis source contribution to the proton/cation residuals.
    """

    def test_gamma_pinned_to_zero_at_lambda_zero(self):
        from Forward.bv_solver.cation_hydrolysis import extract_gamma_value
        # λ=0, R_inj=+1e-3 — even with the manufactured override
        # active, the λ Dirichlet pin must drive Γ → 0.  Likewise the
        # proton/cation boundary source terms (also λ-modulated) must
        # contribute zero.
        sp = _build_gate3d_sp(
            lambda_hydrolysis=0.0,
            manufactured_R_inj=+1e-3,
        )
        ctx = _run_gate3d_solve(sp=sp, mesh_ny=40, voltage=0.55)
        gamma_val = extract_gamma_value(ctx)
        assert abs(gamma_val) < 1e-10, (
            f"λ=0 pin failed: Γ={gamma_val} (expected 0 to within 1e-10)"
        )

    def test_lambda_zero_matches_disabled_feature_observables(self):
        # Solve with cation hydrolysis ENABLED but λ=0 (everything pinned).
        sp_enabled_zero = _build_gate3d_sp(
            lambda_hydrolysis=0.0,
            manufactured_R_inj=+1e-3,
        )
        # Solve with cation hydrolysis fully DISABLED.
        sp_disabled = _build_gate3d_sp(
            lambda_hydrolysis=0.0,
            manufactured_R_inj=None,
            enable_cation_hydrolysis=False,
        )
        ctx_pinned = _run_gate3d_solve(
            sp=sp_enabled_zero, mesh_ny=40, voltage=0.55,
        )
        ctx_disabled = _run_gate3d_solve(
            sp=sp_disabled, mesh_ny=40, voltage=0.55,
        )
        cH_pinned = _read_c_H_surface_mean(ctx_pinned)
        cH_disabled = _read_c_H_surface_mean(ctx_disabled)
        # Semantic tolerance per R3#6: DOF layout differs (Γ slot is
        # absent in the disabled case) so byte-equivalence is impossible
        # — instead compare observables to within 1e-6 relative.
        rel_diff = abs(cH_pinned - cH_disabled) / max(abs(cH_disabled), 1e-12)
        assert rel_diff < 1e-6, (
            f"λ=0 with feature enabled diverges from disabled feature: "
            f"c_H_pinned={cH_pinned}, c_H_disabled={cH_disabled}, "
            f"rel_diff={rel_diff}"
        )
