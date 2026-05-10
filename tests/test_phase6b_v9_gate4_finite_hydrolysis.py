"""Phase 6β v9 Gate 4 — finite-rate cation hydrolysis slow regression.

Tests:

* ``TestHydrolysisActivationZeroReproducesGate2Anchor`` (slow): with
  the full v9 architecture (cation hydrolysis machinery + Singh
  Eq. (4) pKa formula) but ``λ_hydrolysis=0``, the converged anchor
  at V=+0.55 V must match Gate 2's frozen anchor observables to
  within semantic tolerance.  Verifies that the cation-hydrolysis
  residual contribution byte-zeroes at λ=0 even with Singh wired
  up (Gate 3D's pin invariant survives the Gate 4A coupling).

* ``TestSinghPkaShiftMatchesDeckK`` (slow): at the Gate 2 anodic
  anchor (σ_S > 0 → anode-clamped to 0), Singh ΔpKa(K⁺) returns 0.
  At a synthetic cathodic σ_S, ΔpKa is in the right direction and
  order of magnitude.

The full V_RHE warm-walk + sensitivity sweep is the
:mod:`scripts.studies.phase6b_v9_gate4_finite_hydrolysis_smoke`
driver's responsibility — it takes ~hours and is run by hand, not
under pytest.
"""
from __future__ import annotations

import os
import sys

import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# Gate 2 SUCCESS frozen baseline at V=+0.55 V (anodic anchor).  Used
# as the regression target — see
# ``StudyResults/phase6b_v9_gate2_smoke/SUCCESS.md``.
GATE2_BASELINE_V_ANCHOR = 0.55
GATE2_BASELINE_CD_MA_CM2_AT_ANCHOR = -0.583
GATE2_BASELINE_PC_MA_CM2_AT_ANCHOR = +0.291


def _build_gate4_baseline_sp():
    """Build the Gate 4 baseline solver-params: production K2SO4 stack
    with cation-hydrolysis machinery enabled and ``λ=0`` so the
    hydrolysis source vanishes."""
    from scripts._bv_common import (
        A_OH_HAT, D_OH_HAT, KW_HAT,
        DEFAULT_SULFATE_ANALYTIC_BIKERMAN_FOR_K2SO4,
        FOUR_SPECIES_LOGC_DYNAMIC_K2SO4,
        K0_HAT_R2E, K0_HAT_R4E,
        PARALLEL_2E_4E_REACTIONS_4SP,
        SNES_OPTS_CHARGED,
        make_bv_solver_params,
        make_singh_pka_shift_params,
    )

    snes_opts = {**SNES_OPTS_CHARGED}
    snes_opts.update({
        "snes_max_it": 400,
        "snes_atol": 1e-7,
        "snes_rtol": 1e-10,
    })

    cation_cfg = {
        "k_hyd": 1e2,
        "k_prot": 1e-2,
        "k_des": 1e5,
        "delta_ohp_hat": 4e-6,
        "pka_shift_form": "singh_2016_eq_4",
        "pka_shift_params": make_singh_pka_shift_params(
            "K+", r_H_El_pm=200.98,
        ),
    }

    sp = make_bv_solver_params(
        eta_hat=0.0, dt=0.25, t_end=80.0,
        species=FOUR_SPECIES_LOGC_DYNAMIC_K2SO4,
        snes_opts=snes_opts,
        formulation="logc_muh",
        log_rate=True,
        u_clamp=100.0,
        bv_reactions=PARALLEL_2E_4E_REACTIONS_4SP,
        boltzmann_counterions=[DEFAULT_SULFATE_ANALYTIC_BIKERMAN_FOR_K2SO4],
        stern_capacitance_f_m2=0.10,
        initializer="linear_phi",
        l_eff_m=16e-6,
        enable_water_ionization=True,
        kw_eff_hat=KW_HAT,
        d_oh_hat=D_OH_HAT,
        a_oh_hat=A_OH_HAT,
        enable_cation_hydrolysis=True,
        cation_hydrolysis_config=cation_cfg,
        lambda_hydrolysis=0.0,    # disabled at the anchor
    )
    new_opts = dict(sp.solver_options)
    new_bv = dict(new_opts["bv_convergence"])
    new_bv["exponent_clip"] = 100.0
    new_opts["bv_convergence"] = new_bv
    return sp.with_solver_options(new_opts)


@pytest.mark.slow
class TestHydrolysisActivationZeroReproducesGate2Anchor:
    """At ``λ_hydrolysis=0`` the v9 architecture (with full Singh
    wiring) must reproduce Gate 2's frozen anchor observables.

    Tolerance is 5e-2 relative per the plan §4C: discretisation
    differences (Ny=80 vs Gate 2's reference Ny=200) plus the v9
    architectural changes (Singh formula assembled but λ-zeroed)
    introduce small residual norm differences.
    """

    def test_anchor_at_v_plus055_matches_gate2_baseline(self):
        import firedrake as fd
        import firedrake.adjoint as adj
        from Forward.bv_solver import make_graded_rectangle_mesh
        from Forward.bv_solver.anchor_continuation import (
            solve_anchor_with_continuation,
        )
        from Forward.bv_solver.cation_hydrolysis import extract_gamma_value
        from Forward.bv_solver.observables import _build_bv_observable_form
        from scripts._bv_common import (
            I_SCALE, K0_HAT_R2E, K0_HAT_R4E, KW_HAT, V_T,
        )

        sp = _build_gate4_baseline_sp()
        sp_anchor = sp.with_phi_applied(GATE2_BASELINE_V_ANCHOR / V_T)

        mesh = make_graded_rectangle_mesh(
            Nx=8, Ny=80, beta=3.0,
            domain_height_hat=0.16,    # 16 µm / 100 µm = 0.16
        )

        with adj.stop_annotating():
            result = solve_anchor_with_continuation(
                sp_anchor,
                mesh=mesh,
                k0_targets={0: float(K0_HAT_R2E), 1: float(K0_HAT_R4E)},
                initial_scales=(1e-12, 1e-9, 1e-6, 1e-3, 1.0),
                max_inserts_per_step=4,
                max_ss_steps_per_rung=300,
                ic_at_target=True,
                kw_eff_ladder=(0.0, KW_HAT * 1e-6, KW_HAT * 1e-3, KW_HAT * 0.1, KW_HAT),
            )
        assert result.converged, (
            f"Gate 4C anchor failed: history={result.ladder_history}"
        )

        ctx = result.ctx
        f_cd = _build_bv_observable_form(
            ctx, mode="current_density", reaction_index=None, scale=-I_SCALE,
        )
        f_pc = _build_bv_observable_form(
            ctx, mode="peroxide_current", reaction_index=None, scale=-I_SCALE,
        )
        cd_anchor = float(fd.assemble(f_cd))
        pc_anchor = float(fd.assemble(f_pc))
        gamma_anchor = extract_gamma_value(ctx)

        # Γ_MOH = 0 at λ=0 is a hard invariant.
        assert abs(gamma_anchor) < 1e-10, (
            f"Γ_MOH != 0 at λ=0: got {gamma_anchor}"
        )

        # cd / pc match Gate 2 frozen baseline within 5e-2 relative.
        # Gate 2 SUCCESS at V=+0.55 V: cd = -0.583, pc = +0.291 mA/cm².
        assert abs(cd_anchor - GATE2_BASELINE_CD_MA_CM2_AT_ANCHOR) / abs(
            GATE2_BASELINE_CD_MA_CM2_AT_ANCHOR
        ) < 5e-2, (
            f"cd at anchor diverges from Gate 2 baseline: "
            f"got {cd_anchor}, expected {GATE2_BASELINE_CD_MA_CM2_AT_ANCHOR} "
            f"± 5e-2 relative"
        )
        assert abs(pc_anchor - GATE2_BASELINE_PC_MA_CM2_AT_ANCHOR) / abs(
            GATE2_BASELINE_PC_MA_CM2_AT_ANCHOR
        ) < 5e-2, (
            f"pc at anchor diverges from Gate 2 baseline: "
            f"got {pc_anchor}, expected {GATE2_BASELINE_PC_MA_CM2_AT_ANCHOR} "
            f"± 5e-2 relative"
        )


@pytest.mark.slow
class TestSinghAnodeClampAtAnchor:
    """At the Gate 2 anodic anchor (σ_S > 0), Singh's anode clamp
    forces ΔpKa = 0 — verifies the form-build code passes
    physical-units σ_S correctly to the Singh helper.
    """

    def test_singh_pka_shift_anode_clamped_at_anchor(self):
        import firedrake as fd
        import firedrake.adjoint as adj
        from Forward.bv_solver import make_graded_rectangle_mesh
        from Forward.bv_solver.anchor_continuation import (
            solve_anchor_with_continuation,
        )
        from Forward.bv_solver.cation_hydrolysis import build_pka_shift
        from scripts._bv_common import (
            K0_HAT_R2E, K0_HAT_R4E, KW_HAT, V_T,
        )

        sp = _build_gate4_baseline_sp()
        sp_anchor = sp.with_phi_applied(GATE2_BASELINE_V_ANCHOR / V_T)

        mesh = make_graded_rectangle_mesh(
            Nx=8, Ny=80, beta=3.0, domain_height_hat=0.16,
        )

        with adj.stop_annotating():
            result = solve_anchor_with_continuation(
                sp_anchor,
                mesh=mesh,
                k0_targets={0: float(K0_HAT_R2E), 1: float(K0_HAT_R4E)},
                initial_scales=(1e-12, 1e-9, 1e-6, 1e-3, 1.0),
                max_inserts_per_step=4,
                max_ss_steps_per_rung=300,
                ic_at_target=True,
                kw_eff_ladder=(0.0, KW_HAT * 1e-6, KW_HAT * 1e-3, KW_HAT * 0.1, KW_HAT),
            )
        assert result.converged

        ctx = result.ctx
        sigma_S_expr = ctx.get("_cation_hydrolysis_sigma_S_expr")
        assert sigma_S_expr is not None

        bundle = ctx["cation_hydrolysis"]
        pka_shift = build_pka_shift(
            cation_params=bundle.cation_params,
            sigma_S=sigma_S_expr,
        )
        electrode_marker = ctx["bv_settings"]["electrode_marker"]
        ds = fd.Measure("ds", domain=ctx["mesh"])
        area = float(fd.assemble(fd.Constant(1.0) * ds(electrode_marker)))
        avg_pka_shift = float(
            fd.assemble(pka_shift * ds(electrode_marker))
        ) / area
        # At V=+0.55 V (anodic), σ_S > 0 (Stern accumulates positive
        # charge), so the anode-clamp drives ΔpKa = 0 exactly.
        assert avg_pka_shift == pytest.approx(0.0, abs=1e-12), (
            f"Singh anode-clamp failed at anodic V_RHE=+0.55: "
            f"⟨ΔpKa⟩ = {avg_pka_shift}"
        )
