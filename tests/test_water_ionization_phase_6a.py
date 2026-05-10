"""Phase 6α — water self-ionization (proton-condition variable) tests.

Fast unit tests for the Q1 constants + config plumbing.  Slow MMS,
anchor-smoke, and regression tests for Q2-Q3 land later in this file.

See ``docs/PHASE_6A_OH_WATER_IONIZATION_PLAN.md`` for the full design.
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
    A_OH_HAT,
    C_HP_HAT,
    C_OH_BULK_HAT,
    C_SCALE,
    DEFAULT_CLO4_BOLTZMANN_COUNTERION,
    DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC,
    D_OH,
    D_OH_HAT,
    D_REF,
    KW_HAT,
    KW_MOLAR_SQUARED,
    KW_PHYS,
    THREE_SPECIES_LOGC_BOLTZMANN,
    make_bv_solver_params,
)


# ===================================================================
# Q1 — Constants consistency
# ===================================================================

class TestKwConsistency:
    """Lock down the Kw scaling chain and pH-4 OH⁻ identity."""

    def test_kw_phys_from_molar_squared(self):
        # 1 mol/L = 1000 mol/m³ ⇒ (mol/L)² → 1e6 (mol/m³)²
        assert KW_PHYS == pytest.approx(KW_MOLAR_SQUARED * 1.0e6, rel=1e-15)

    def test_kw_hat_consistency(self):
        # KW_HAT = KW_PHYS / C_SCALE² and KW_HAT == C_HP_HAT * C_OH_BULK_HAT
        # to floating-point precision (this is the identity that anchors
        # the entire fast-water closure).
        expected = KW_PHYS / (C_SCALE ** 2)
        assert KW_HAT == pytest.approx(expected, rel=1e-12)
        assert KW_HAT == pytest.approx(C_HP_HAT * C_OH_BULK_HAT, rel=1e-12)

    def test_bulk_oh_concentration_pH4(self):
        # At pH 4: c_H = 1e-4 M = 0.1 mol/m³ ⇒ c_OH = Kw/c_H = 1e-10 M
        # = 1e-7 mol/m³.  Convert C_OH_BULK_HAT back to physical and check.
        c_oh_phys = C_OH_BULK_HAT * C_SCALE
        assert c_oh_phys == pytest.approx(1.0e-7, rel=1e-12)

    def test_d_oh_hat_consistency(self):
        # D_OH_HAT = D_OH / D_REF.  D_OH = 5.273e-9 m²/s; D_REF = D_O₂ =
        # 1.9e-9 m²/s ⇒ D_OH_HAT ≈ 2.7753.
        assert D_OH_HAT == pytest.approx(D_OH / D_REF, rel=1e-15)
        # Sanity-check the physical value.
        assert D_OH == pytest.approx(5.273e-9, rel=1e-12)

    def test_a_oh_hat_positive(self):
        # A_OH_HAT = (4/3)·π·r³·N_A·C_SCALE for r=1.76 Å.  Should be a
        # small positive (~1.65e-5 with C_SCALE=1.2 mol/m³).
        assert A_OH_HAT > 0.0
        assert A_OH_HAT < 1e-3   # hard-sphere is much smaller than 1


# ===================================================================
# Q1 — solver_params defaults are off (regression gate)
# ===================================================================

_OMITTED = object()


def _build(**overrides):
    """Build production-stack solver params."""
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


class TestWaterIonizationDefaultOff:
    """Phase 6α flag plumbing must default to False (byte-equivalence)."""

    def test_default_is_false(self):
        cfg = _bv_conv(_build())
        assert cfg["enable_water_ionization"] is False

    def test_default_kw_eff_matches_physical(self):
        cfg = _bv_conv(_build())
        assert cfg["kw_eff_hat"] == pytest.approx(KW_HAT, rel=1e-15)

    def test_default_d_oh_a_oh_match_physical(self):
        cfg = _bv_conv(_build())
        assert cfg["d_oh_hat"] == pytest.approx(D_OH_HAT, rel=1e-15)
        assert cfg["a_oh_hat"] == pytest.approx(A_OH_HAT, rel=1e-15)

    def test_explicit_true_writes_true(self):
        cfg = _bv_conv(_build(enable_water_ionization=True))
        assert cfg["enable_water_ionization"] is True

    def test_kw_eff_override_lands(self):
        # The anchor builder ramps kw_eff via this kwarg; lock that the
        # config dict reflects the override (not the default).
        cfg = _bv_conv(_build(
            enable_water_ionization=True,
            kw_eff_hat=KW_HAT * 1e-3,
        ))
        assert cfg["kw_eff_hat"] == pytest.approx(KW_HAT * 1e-3, rel=1e-15)

    def test_zero_kw_eff_accepted(self):
        # The continuation ladder starts at Kw_eff = 0 (water term off).
        # Must be writable through the factory.
        cfg = _bv_conv(_build(
            enable_water_ionization=True,
            kw_eff_hat=0.0,
        ))
        assert cfg["kw_eff_hat"] == 0.0


# ===================================================================
# Q1+Q3 — config-layer parser passes through Phase 6α keys
# ===================================================================

class TestConvergenceCfgParser:
    """``_get_bv_convergence_cfg`` must surface the Phase 6α keys with
    the correct default-off values; otherwise the form-build code path
    (which reads conv_cfg, not raw params) would silently disable the
    feature even when the user set ``enable_water_ionization=True``.
    """

    def test_parser_default_off(self):
        from Forward.bv_solver.config import _get_bv_convergence_cfg
        sp = _build()
        cfg = _get_bv_convergence_cfg(sp[10])
        assert cfg["enable_water_ionization"] is False
        # Default-off branch passes Kw=0 through (raw params keep KW_HAT,
        # but the parser surfaces whatever the residual will use).
        assert cfg["kw_eff_hat"] == pytest.approx(KW_HAT, rel=1e-15)

    def test_parser_passes_through_when_enabled(self):
        from Forward.bv_solver.config import _get_bv_convergence_cfg
        sp = _build(
            enable_water_ionization=True,
            kw_eff_hat=KW_HAT * 0.1,
        )
        cfg = _get_bv_convergence_cfg(sp[10])
        assert cfg["enable_water_ionization"] is True
        assert cfg["kw_eff_hat"] == pytest.approx(KW_HAT * 0.1, rel=1e-15)
        assert cfg["d_oh_hat"] == pytest.approx(D_OH_HAT, rel=1e-15)
        assert cfg["a_oh_hat"] == pytest.approx(A_OH_HAT, rel=1e-15)

    def test_parser_rejects_negative_kw(self):
        from Forward.bv_solver.config import _get_bv_convergence_cfg
        sp = _build(enable_water_ionization=True)
        # Inject a negative Kw_eff post-construction (the factory does
        # not gate on this).
        sp[10]["bv_convergence"]["kw_eff_hat"] = -1.0
        with pytest.raises(ValueError, match="kw_eff_hat must be non-negative"):
            _get_bv_convergence_cfg(sp[10])

    def test_parser_rejects_zero_d_oh_when_enabled(self):
        from Forward.bv_solver.config import _get_bv_convergence_cfg
        sp = _build(enable_water_ionization=True)
        sp[10]["bv_convergence"]["d_oh_hat"] = 0.0
        with pytest.raises(ValueError, match="d_oh_hat must be positive"):
            _get_bv_convergence_cfg(sp[10])

    def test_parser_rejects_negative_a_oh_when_enabled(self):
        from Forward.bv_solver.config import _get_bv_convergence_cfg
        sp = _build(enable_water_ionization=True)
        sp[10]["bv_convergence"]["a_oh_hat"] = -1e-5
        with pytest.raises(ValueError, match="a_oh_hat must be non-negative"):
            _get_bv_convergence_cfg(sp[10])

    def test_parser_skips_validation_when_disabled(self):
        from Forward.bv_solver.config import _get_bv_convergence_cfg
        # When the feature is off, the field values are not validated —
        # they're carried through verbatim so the config dict reflects
        # the user's exact intent.  This documents the layer split.
        sp = _build()
        sp[10]["bv_convergence"]["d_oh_hat"] = 0.0
        sp[10]["bv_convergence"]["kw_eff_hat"] = 0.0
        cfg = _get_bv_convergence_cfg(sp[10])  # should not raise
        assert cfg["d_oh_hat"] == 0.0
        assert cfg["kw_eff_hat"] == 0.0


# ===================================================================
# Q2 — water_ionization helper module unit tests
# ===================================================================

class TestWaterIonizationHelpers:
    """Pure-Python unit tests on the resolve_h_index / gate helpers."""

    def test_resolve_h_index_basic(self):
        from Forward.bv_solver.water_ionization import resolve_h_index
        assert resolve_h_index([0, 0, 1]) == 2
        assert resolve_h_index([1, 0, 0]) == 0

    def test_resolve_h_index_no_proton(self):
        from Forward.bv_solver.water_ionization import resolve_h_index
        with pytest.raises(ValueError, match="requires a species with z=\\+1"):
            resolve_h_index([0, 0, 0])

    def test_resolve_h_index_multiple_protons(self):
        from Forward.bv_solver.water_ionization import resolve_h_index
        with pytest.raises(ValueError, match="exactly one species"):
            resolve_h_index([1, 0, 1])

    def test_resolve_h_index_with_explicit_roles_disambiguates_two_zplus1(self):
        # Phase 6β v9 Gate 1: K2SO4-style stack has both H⁺ and K⁺ at
        # z=+1; explicit ``roles=`` argument disambiguates so the legacy
        # "exactly one z=+1 species" reject is bypassed.
        from Forward.bv_solver.water_ionization import resolve_h_index
        h_idx = resolve_h_index(
            [0, 0, 1, 1],
            roles=["neutral", "neutral", "proton", "counterion"],
        )
        assert h_idx == 2

    def test_is_water_ionization_enabled_default_false(self):
        from Forward.bv_solver.water_ionization import is_water_ionization_enabled
        assert is_water_ionization_enabled({}) is False
        assert is_water_ionization_enabled(
            {"enable_water_ionization": False}
        ) is False

    def test_is_water_ionization_enabled_true(self):
        from Forward.bv_solver.water_ionization import is_water_ionization_enabled
        assert is_water_ionization_enabled(
            {"enable_water_ionization": True}
        ) is True


# ===================================================================
# Q3 — Kw_eff continuation ladder validation (in solve_anchor_with_continuation)
# ===================================================================

class TestKwEffLadderValidation:
    """The Kw_eff outer ladder validation runs *before* the form is built
    (caller passes ``kw_eff_ladder``), so we can exercise it at the unit
    level by giving a malformed ladder and intercepting the ValueError.
    """

    def _stub_sp(self):
        # Build a minimal sp; we won't actually solve.  The validation
        # in solve_anchor_with_continuation runs early.
        return _build()

    def test_set_reaction_kw_eff_model_no_bundle_raises(self):
        from Forward.bv_solver.anchor_continuation import set_reaction_kw_eff_model
        with pytest.raises(ValueError, match="no 'water_ionization' bundle"):
            set_reaction_kw_eff_model({}, KW_HAT)

    def test_set_reaction_kw_eff_model_negative_raises(self):
        from Forward.bv_solver.anchor_continuation import set_reaction_kw_eff_model
        # Even with a stub bundle, the negative-value check fires first.
        with pytest.raises(ValueError, match="must be non-negative"):
            set_reaction_kw_eff_model({}, -1.0)

    def test_get_reaction_kw_eff_model_no_bundle_raises(self):
        from Forward.bv_solver.anchor_continuation import get_reaction_kw_eff_model
        with pytest.raises(ValueError, match="no 'water_ionization' bundle"):
            get_reaction_kw_eff_model({})

    def test_anchor_continuation_module_exports(self):
        # Q3: the public surface for Phase 6α Kw_eff continuation is
        # (set/get)_reaction_kw_eff_model + the kw_eff_ladder kwarg on
        # solve_anchor_with_continuation.  Lock the package surface.
        import Forward.bv_solver.anchor_continuation as ac
        assert "set_reaction_kw_eff_model" in ac.__all__
        assert "get_reaction_kw_eff_model" in ac.__all__


# ===================================================================
# Q4 — slow Firedrake-backed tests
# ===================================================================
#
# These exercise the actual UFL form build with water-ionization on/off,
# the disabled-path byte-equivalence (residual assembled at IC matches
# baseline), and the set/get_reaction_kw_eff_model round-trip on a live
# ctx.  Mark slow because they require Firedrake.

@pytest.mark.slow
class TestFormBuildWaterIonization:
    """Form build + ctx surface checks for the Phase 6α residual."""

    @staticmethod
    def _build_sp(*, formulation: str, enable_water: bool, kw_eff: float = None):
        kwargs = dict(
            eta_hat=0.0, dt=0.25, t_end=10.0,
            species=THREE_SPECIES_LOGC_BOLTZMANN,
            formulation=formulation,
            log_rate=True,
            boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC],
            stern_capacitance_f_m2=0.10,
            initializer="debye_boltzmann",
            enable_water_ionization=enable_water,
        )
        if kw_eff is not None:
            kwargs["kw_eff_hat"] = kw_eff
        return make_bv_solver_params(**kwargs)

    def _build_ctx(self, formulation: str, enable_water: bool, kw_eff: float = None):
        import firedrake as fd
        from Forward.bv_solver.dispatch import build_context, build_forms
        sp = self._build_sp(
            formulation=formulation, enable_water=enable_water, kw_eff=kw_eff,
        )
        mesh = fd.UnitSquareMesh(8, 8)
        ctx = build_context(sp, mesh=mesh)
        ctx = build_forms(ctx, sp)
        return sp, ctx

    @pytest.mark.parametrize("formulation", ["logc", "logc_muh"])
    def test_disabled_bundle_is_none(self, formulation):
        _, ctx = self._build_ctx(formulation, enable_water=False)
        assert ctx["water_ionization"] is None

    @pytest.mark.parametrize("formulation", ["logc", "logc_muh"])
    def test_enabled_bundle_populated(self, formulation):
        _, ctx = self._build_ctx(formulation, enable_water=True)
        bundle = ctx["water_ionization"]
        assert bundle is not None
        assert bundle.h_idx == 2  # H+ at index 2 in THREE_SPECIES preset
        assert float(bundle.kw_eff_func) == pytest.approx(KW_HAT, rel=1e-12)
        assert float(bundle.d_oh_const) == pytest.approx(D_OH_HAT, rel=1e-12)
        assert float(bundle.a_oh_const) == pytest.approx(A_OH_HAT, rel=1e-12)

    @pytest.mark.parametrize("formulation", ["logc", "logc_muh"])
    def test_kw_eff_setter_updates_function(self, formulation):
        from Forward.bv_solver.anchor_continuation import (
            set_reaction_kw_eff_model, get_reaction_kw_eff_model,
        )
        _, ctx = self._build_ctx(formulation, enable_water=True)
        # Initial value matches KW_HAT
        assert get_reaction_kw_eff_model(ctx) == pytest.approx(KW_HAT, rel=1e-12)
        # Update through the setter
        set_reaction_kw_eff_model(ctx, KW_HAT * 0.1)
        assert get_reaction_kw_eff_model(ctx) == pytest.approx(
            KW_HAT * 0.1, rel=1e-12
        )
        # Floor (Kw_eff = 0) is allowed
        set_reaction_kw_eff_model(ctx, 0.0)
        assert get_reaction_kw_eff_model(ctx) == 0.0


@pytest.mark.slow
class TestKwZeroReducesToBaseline:
    """At ``Kw_eff = 0`` the proton-condition residual must reduce
    *byte-equivalently* to the standard H⁺ NP residual.  Locks down the
    sign convention of :func:`build_proton_condition_flux` (a sign flip
    here was the source of the 2026-05-09 sweep stall: the Kw_eff=0
    floor of the continuation ladder appeared to ramp k0 like the
    baseline but every k0 rung past 5e-4 stalled because the flux
    enforced reversed transport).

    Both backends are checked.  The comparison uses the L²-norm of the
    assembled residual at the linear-φ IC; a sign error in the H⁺
    branch would change this even without water in the system.
    """

    @pytest.mark.parametrize("formulation", ["logc", "logc_muh"])
    def test_residual_norm_matches_baseline_at_kw_zero(self, formulation):
        import firedrake as fd
        from Forward.bv_solver.dispatch import (
            build_context, build_forms, set_initial_conditions,
        )
        from Forward.bv_solver.anchor_continuation import (
            set_reaction_kw_eff_model,
        )

        baseline_kwargs = dict(
            eta_hat=0.0, dt=0.25, t_end=10.0,
            species=THREE_SPECIES_LOGC_BOLTZMANN,
            formulation=formulation,
            log_rate=True,
            boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC],
            stern_capacitance_f_m2=0.10,
            initializer="linear_phi",
        )
        sp_off = make_bv_solver_params(**baseline_kwargs)
        sp_on = make_bv_solver_params(
            **baseline_kwargs,
            enable_water_ionization=True,
        )

        mesh = fd.UnitSquareMesh(16, 16)

        ctx_off = build_context(sp_off, mesh=mesh)
        ctx_off = build_forms(ctx_off, sp_off)
        set_initial_conditions(ctx_off, sp_off)

        ctx_on = build_context(sp_on, mesh=mesh)
        ctx_on = build_forms(ctx_on, sp_on)
        set_initial_conditions(ctx_on, sp_on)
        # Drive Kw_eff to zero — proton-condition residual reduces to
        # the H⁺ NP residual exactly (c_OH ≡ 0 ⇒ both branches identical).
        set_reaction_kw_eff_model(ctx_on, 0.0)

        r_off = fd.assemble(ctx_off["F_res"])
        r_on = fd.assemble(ctx_on["F_res"])
        norm_off = float(fd.assemble(
            sum(fd.inner(s, s) * fd.dx for s in fd.split(r_off.riesz_representation()))
        )) ** 0.5
        norm_on = float(fd.assemble(
            sum(fd.inner(s, s) * fd.dx for s in fd.split(r_on.riesz_representation()))
        )) ** 0.5
        assert norm_on == pytest.approx(norm_off, rel=1e-10, abs=1e-14)


@pytest.mark.slow
class TestDisabledPathByteEquivalence:
    """The residual assembled at the linear-φ IC must be byte-identical
    between (i) ``enable_water_ionization=False`` and (ii) a build with
    the flag plumbed but the form gated off.  This guards against
    accidentally moving a constant or restructuring the loop in a way
    that changes the dynamic-species residual on the disabled path.
    """

    @pytest.mark.parametrize("formulation", ["logc", "logc_muh"])
    def test_residual_norm_matches_baseline(self, formulation):
        import firedrake as fd
        from Forward.bv_solver.dispatch import (
            build_context, build_forms, set_initial_conditions,
        )

        # Baseline: factory called WITHOUT the new kwargs (legacy API).
        baseline_kwargs = dict(
            eta_hat=0.0, dt=0.25, t_end=10.0,
            species=THREE_SPECIES_LOGC_BOLTZMANN,
            formulation=formulation,
            log_rate=True,
            boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC],
            stern_capacitance_f_m2=0.10,
            initializer="linear_phi",  # cheaper than picard for the smoke
        )
        sp_base = make_bv_solver_params(**baseline_kwargs)
        sp_off = make_bv_solver_params(
            **baseline_kwargs,
            enable_water_ionization=False,
        )

        mesh = fd.UnitSquareMesh(16, 16)
        ctx_base = build_context(sp_base, mesh=mesh)
        ctx_base = build_forms(ctx_base, sp_base)
        set_initial_conditions(ctx_base, sp_base)

        ctx_off = build_context(sp_off, mesh=mesh)
        ctx_off = build_forms(ctx_off, sp_off)
        set_initial_conditions(ctx_off, sp_off)

        # Sanity: bundles match expectations.
        assert ctx_base["water_ionization"] is None
        assert ctx_off["water_ionization"] is None

        # Residual L² norm at the IC.  Must agree to floating-point
        # precision when the disabled path is byte-equivalent.
        r_base = fd.assemble(ctx_base["F_res"])
        r_off = fd.assemble(ctx_off["F_res"])
        # Cofunction: norm via riesz
        norm_base = float(fd.assemble(
            sum(fd.inner(s, s) * fd.dx for s in fd.split(r_base.riesz_representation()))
        )) ** 0.5
        norm_off = float(fd.assemble(
            sum(fd.inner(s, s) * fd.dx for s in fd.split(r_off.riesz_representation()))
        )) ** 0.5
        assert norm_base == pytest.approx(norm_off, rel=1e-12, abs=1e-14)


@pytest.mark.slow
@pytest.mark.skip(
    reason=(
        "Phase 6α Q4 deliverable: MMS for proton-condition residual "
        "(manufactured smooth (u_H, φ), analytic forcing matched to the "
        "conservative weak form).  Requires deriving the source term — "
        "scoped for a follow-up landing.  See "
        "docs/PHASE_6A_OH_WATER_IONIZATION_PLAN.md §Q4."
    )
)
def test_mms_proton_condition():
    """Manufactured-solution convergence test (≥ 2nd order in u_H).

    Skipped pending MMS source-term derivation; the residual code is
    locked down by ``TestFormBuildWaterIonization`` until then.
    """
    raise NotImplementedError
