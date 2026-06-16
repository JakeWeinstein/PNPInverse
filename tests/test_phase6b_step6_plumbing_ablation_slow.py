"""Slow Firedrake-backed tests for Phase 6β step 6 plumbing-ablation
wiring.

These tests build small ctx objects to inspect the new ctx-stored UFL
expressions introduced by step 6 — see plan §"Solver-side wiring" for
the canonical artifacts (``_cation_hydrolysis_R_net_expr``,
``_cation_hydrolysis_pka_shift_expr``, ``_H_residual_term``, etc.).

The end-to-end byte-equivalence + A0b assembly + override-path tests
require a converged anchor + warm-walk and live in driver-level
slow tests; they are gated by ``RUN_SLOW_E2E`` env var since each
takes minutes (see ``test_A0_byte_equivalence_with_A2_baseline``).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List

import pytest


# ---------------------------------------------------------------------------
# Small-ctx fixture builder — mirrors the gate3 / langmuir-cap pattern.
# Uses pka_shift_form="singh_2016_eq_4" so pka_shift_expr is not a
# trivial Constant(0).  K⁺/Cu Singh parameters baked in.
# ---------------------------------------------------------------------------


def _build_step6_sp(
    *,
    apply_h_source: bool = True,
    apply_k_sink: bool = True,
    manufactured_R_inj=None,
    override_sigma_singh_counts_pm2=None,
):
    from scripts._bv_common import (
        DEFAULT_SULFATE_ANALYTIC_BIKERMAN_FOR_K2SO4,
        FOUR_SPECIES_LOGC_DYNAMIC_K2SO4,
        GAMMA_MAX_HAT_SMOKE,
        make_bv_solver_params,
        make_cation_hydrolysis_config,
    )

    cation_cfg = make_cation_hydrolysis_config(
        k_hyd=1e-3, k_prot=1e-3, k_des=1.0,
        delta_ohp_hat=4e-6,
        cation="K+",
        pka_shift_form="singh_2016_eq_4",
        gamma_max_nondim=GAMMA_MAX_HAT_SMOKE,
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
        cation_hydrolysis_config=cation_cfg,
    )

    new_opts = dict(sp.solver_options)
    new_bv = dict(new_opts.get("bv_convergence", {}))
    new_bv["apply_h_source"] = bool(apply_h_source)
    new_bv["apply_k_sink"] = bool(apply_k_sink)
    if manufactured_R_inj is not None:
        new_bv["manufactured_R_inj"] = float(manufactured_R_inj)
    if override_sigma_singh_counts_pm2 is not None:
        new_bv["override_sigma_singh_counts_pm2"] = float(
            override_sigma_singh_counts_pm2
        )
    new_opts["bv_convergence"] = new_bv
    return sp.with_solver_options(new_opts)


def _build_step6_ctx(sp):
    import firedrake as fd
    from Forward.bv_solver.dispatch import build_context, build_forms

    mesh = fd.UnitSquareMesh(8, 8)
    ctx = build_context(sp, mesh=mesh)
    ctx = build_forms(ctx, sp)
    return ctx


# ---------------------------------------------------------------------------
# Override-path coefficient independence (plan §test_override_pka_shift_
# independent_of_U)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestOverridePkaShiftIndependentOfU:
    """In override mode, the ctx-stored ``_cation_hydrolysis_pka_shift_expr``
    does NOT depend on the Newton primary ``ctx['U']`` — the override path
    feeds a Constant ``fake_signed_sigma_S`` into ``build_pka_shift``
    instead of the U-derived ``sigma_S_expr``.

    ``r_H_El_pm_func`` (a Function in R-space) IS expected to appear as a
    coefficient — that's the live-r_H_El hook.
    """

    def test_override_mode_U_not_in_pka_shift_coefficients(self):
        import firedrake as fd
        from ufl.algorithms import extract_coefficients

        sp = _build_step6_sp(override_sigma_singh_counts_pm2=0.022)
        ctx = _build_step6_ctx(sp)
        pka_shift = ctx["_cation_hydrolysis_pka_shift_expr"]
        assert pka_shift is not None
        coeffs = extract_coefficients(pka_shift)
        U = ctx["U"]
        for c in coeffs:
            assert c is not U, (
                "Override-mode pka_shift_expr must not reference "
                "ctx['U'] (the Newton primary); got coefficient "
                f"identical to U at {c!r}"
            )

    def test_default_mode_U_via_sigma_S_in_pka_shift(self):
        """Without override, pka_shift_expr depends on sigma_S_expr, which
        depends on (phi_applied - phi) — phi is a slot in U.  So U IS in
        coefficients in default mode (control case)."""
        import firedrake as fd
        from ufl.algorithms import extract_coefficients

        sp = _build_step6_sp()    # no override → default sigma_S path
        ctx = _build_step6_ctx(sp)
        pka_shift = ctx["_cation_hydrolysis_pka_shift_expr"]
        sigma_S_expr = ctx["_cation_hydrolysis_sigma_S_expr"]
        # σ_S in default mode uses phi (a slot of U).
        sigma_coeffs = extract_coefficients(sigma_S_expr)
        assert any(c is ctx["U"] for c in sigma_coeffs), (
            "Default-mode sigma_S_expr should reference ctx['U'] "
            f"(via the phi slot); got {sigma_coeffs!r}"
        )

    def test_override_mode_pka_sigma_S_is_constant_not_U(self):
        """The override-mode stored ``_pka_sigma_S_expr`` is the
        ``fake_signed_sigma_S`` Constant, not the U-derived form."""
        import firedrake as fd
        from ufl.algorithms import extract_coefficients

        sp = _build_step6_sp(override_sigma_singh_counts_pm2=0.022)
        ctx = _build_step6_ctx(sp)
        pka_sigma = ctx["_cation_hydrolysis_pka_sigma_S_expr"]
        coeffs = extract_coefficients(pka_sigma)
        for c in coeffs:
            assert c is not ctx["U"], (
                "Override-mode pka_sigma_S_expr must not reference U"
            )


# ---------------------------------------------------------------------------
# Stern σ_S remains live-from-U even when override is active (plan §
# test_stern_sigma_S_remains_solved_in_override_mode)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestSternSigmaSRemainsSolvedInOverrideMode:
    """The Stern-side ``ctx['_cation_hydrolysis_sigma_S_expr']`` still
    references ``ctx['U']`` even when the override is active — the
    override only fakes the Singh input, not the underlying Poisson-
    Stern computation of physical σ_S used elsewhere (e.g. diagnostics).
    """

    def test_sigma_S_expr_keeps_U(self):
        import firedrake as fd
        from ufl.algorithms import extract_coefficients

        sp = _build_step6_sp(override_sigma_singh_counts_pm2=0.022)
        ctx = _build_step6_ctx(sp)
        sigma_S_expr = ctx["_cation_hydrolysis_sigma_S_expr"]
        coeffs = extract_coefficients(sigma_S_expr)
        assert any(c is ctx["U"] for c in coeffs), (
            "Override mode should NOT zero out the physical "
            "_cation_hydrolysis_sigma_S_expr; it still feeds the "
            "diagnostics + Picard sigma_S boundary average."
        )


# ---------------------------------------------------------------------------
# H/K residual terms target the right slots (plan §
# test_H_K_residual_terms_target_correct_slots; R5 #2)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestHKResidualSlotWiring:
    """The ctx-stored ``_H_residual_term`` / ``_K_residual_term`` Forms
    are wired against ``v_list[h_idx_for_cation]`` and
    ``v_list[counterion_idx]`` respectively.

    Strategy (R5 #2): substitute the test function with a probe that is
    ``1.0`` in exactly one slot and ``0.0`` elsewhere via ``fd.action``.
    Slot-wiring is correct iff the form integrates to the expected
    R_net flux only when probed at its target slot, and to ``0`` for
    other slots.
    """

    def _make_probe(self, ctx, slot_idx):
        """Build a Function on ctx['U']'s mixed space with 1.0 in
        ``slot_idx`` (and 0.0 elsewhere)."""
        import firedrake as fd
        U = ctx["U"]
        W = U.function_space()
        probe = fd.Function(W)
        # Sub-function spaces are accessed by sub(idx).
        probe.sub(slot_idx).assign(fd.Constant(1.0))
        return probe

    def test_H_residual_targets_h_idx_only(self):
        import firedrake as fd
        from Forward.bv_solver.water_ionization import resolve_h_index

        sp = _build_step6_sp(
            manufactured_R_inj=1.0,   # makes R_net = Constant(1.0)
            apply_h_source=True, apply_k_sink=True,
        )
        ctx = _build_step6_ctx(sp)
        bundle = ctx["cation_hydrolysis"]
        h_idx = bundle.h_idx
        k_idx = bundle.counterion_idx
        H_term = ctx["_cation_hydrolysis_H_residual_term"]
        # Set λ to 1 so the residual flows non-trivially.
        bundle.lambda_hydrolysis_func.assign(fd.Constant(1.0))

        probe_h = self._make_probe(ctx, h_idx)
        val_h = float(fd.assemble(fd.action(H_term, probe_h)))
        probe_k = self._make_probe(ctx, k_idx)
        val_k = float(fd.assemble(fd.action(H_term, probe_k)))

        assert val_h != pytest.approx(0.0, abs=1e-30), (
            "H_residual_term must produce non-zero contribution when "
            f"probed at slot h_idx={h_idx}; got {val_h}"
        )
        assert val_k == pytest.approx(0.0, abs=1e-12), (
            "H_residual_term must NOT contribute to slot "
            f"counterion_idx={k_idx}; got {val_k} (residual wiring "
            "putting R_net into the wrong species slot)"
        )

    def test_K_residual_targets_counterion_idx_only(self):
        import firedrake as fd

        sp = _build_step6_sp(
            manufactured_R_inj=1.0,
            apply_h_source=True, apply_k_sink=True,
        )
        ctx = _build_step6_ctx(sp)
        bundle = ctx["cation_hydrolysis"]
        h_idx = bundle.h_idx
        k_idx = bundle.counterion_idx
        K_term = ctx["_cation_hydrolysis_K_residual_term"]
        bundle.lambda_hydrolysis_func.assign(fd.Constant(1.0))

        probe_k = self._make_probe(ctx, k_idx)
        val_k = float(fd.assemble(fd.action(K_term, probe_k)))
        probe_h = self._make_probe(ctx, h_idx)
        val_h = float(fd.assemble(fd.action(K_term, probe_h)))

        assert val_k != pytest.approx(0.0, abs=1e-30), (
            "K_residual_term must produce non-zero contribution when "
            f"probed at slot counterion_idx={k_idx}; got {val_k}"
        )
        assert val_h == pytest.approx(0.0, abs=1e-12), (
            "K_residual_term must NOT contribute to slot h_idx="
            f"{h_idx}; got {val_h} (sink wiring putting -R_net into the "
            "wrong species slot)"
        )

    def test_H_and_K_residuals_anti_symmetric(self):
        """The integrated H source and K sink scalar forms must be
        anti-symmetric: ``H_flux + K_flux ≈ 0`` (R_net injected as H
        is exactly the K consumed)."""
        import firedrake as fd

        sp = _build_step6_sp(
            manufactured_R_inj=1.0,
            apply_h_source=True, apply_k_sink=True,
        )
        ctx = _build_step6_ctx(sp)
        bundle = ctx["cation_hydrolysis"]
        bundle.lambda_hydrolysis_func.assign(fd.Constant(1.0))

        h_flux = float(
            fd.assemble(ctx["_cation_hydrolysis_H_flux_scalar_form"])
        )
        k_flux = float(
            fd.assemble(ctx["_cation_hydrolysis_K_flux_scalar_form"])
        )
        # Sum should be zero up to FP precision.
        assert abs(h_flux + k_flux) < 1e-12 * max(abs(h_flux), 1e-30), (
            f"H_flux + K_flux = {h_flux + k_flux}; expected 0 to FP "
            f"precision (H={h_flux}, K={k_flux})"
        )


# ---------------------------------------------------------------------------
# Override-path assembly — quantitative gate 1 (pka_factor ≈ 10^(-β·σ))
# at lab-Stern-σ-S level so a 4-step Newton at a small mesh suffices.
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestOverridePathAssembly:
    """With ``override_sigma_singh_counts_pm2 = 0.022`` and the K⁺/Cu
    Singh prior, the boundary-averaged ``pka_factor`` (= 10^(−ΔpKa))
    must equal the predicted 10.07 to within 5% — pinning the
    override-in-form path.

    This is a unit test of the **form-build path only**, not a full
    Newton solve.  We initialise U at the linear-phi IC, do not solve,
    and assemble the override-mode ``_cation_hydrolysis_pka_shift_expr``
    against the boundary measure.  Since the override path makes
    ``pka_shift_expr`` independent of U, the IC vs. converged-Newton
    state of U does not affect the assembled value.
    """

    def test_pka_factor_assembled_matches_prediction(self):
        import math

        import firedrake as fd

        override = 0.022
        sp = _build_step6_sp(override_sigma_singh_counts_pm2=override)
        ctx = _build_step6_ctx(sp)
        pka_shift = ctx["_cation_hydrolysis_pka_shift_expr"]
        electrode_marker = ctx["bv_settings"]["electrode_marker"]
        ds = fd.Measure("ds", domain=ctx["mesh"])
        area = float(fd.assemble(fd.Constant(1.0) * ds(electrode_marker)))
        pka_shift_avg = float(
            fd.assemble(pka_shift * ds(electrode_marker))
        ) / area
        pka_factor_avg = 10.0 ** (-pka_shift_avg)

        # β for K⁺/Cu = 2·A·z·r_H_El·G ≈ -45.61.
        r_M_O = 138.0 + 63.0
        r_H_El = 200.98
        beta = 2.0 * 620.32 * 0.919 * r_H_El * (
            1.0 - (r_M_O / r_H_El) ** 2
        )
        pka_factor_predicted = 10.0 ** (-beta * override)

        rel = abs(pka_factor_avg - pka_factor_predicted) / pka_factor_predicted
        assert rel < 0.05, (
            f"Override pka_factor {pka_factor_avg:.4g} mismatches "
            f"predicted {pka_factor_predicted:.4g} (rel={rel:.4g}); "
            "override is not reaching the assembled pka_shift_expr."
        )
        # Also pin the order of magnitude.
        assert 9.0 < pka_factor_avg < 12.0, (
            f"Expected pka_factor near 10.07; got {pka_factor_avg}"
        )

    def test_zero_override_recovers_unit_pka_factor(self):
        """Override σ = 0 should give ΔpKa = 0 ⇒ pka_factor = 1.0
        exactly — sanity check that the override path doesn't add
        spurious offsets."""
        import firedrake as fd

        sp = _build_step6_sp(override_sigma_singh_counts_pm2=0.0)
        ctx = _build_step6_ctx(sp)
        pka_shift = ctx["_cation_hydrolysis_pka_shift_expr"]
        electrode_marker = ctx["bv_settings"]["electrode_marker"]
        ds = fd.Measure("ds", domain=ctx["mesh"])
        area = float(fd.assemble(fd.Constant(1.0) * ds(electrode_marker)))
        pka_shift_avg = float(
            fd.assemble(pka_shift * ds(electrode_marker))
        ) / area
        assert pka_shift_avg == pytest.approx(0.0, abs=1e-30)


# ---------------------------------------------------------------------------
# A0 byte-equivalence end-to-end — gated by env var since it requires
# Pass 1 + Pass 2 solves (~minutes).
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("RUN_SLOW_E2E", "0") not in ("1", "true", "True"),
    reason=(
        "End-to-end byte-equivalence requires the full anchor+warm-walk "
        "pipeline (~minutes); set RUN_SLOW_E2E=1 to enable."
    ),
)
class TestA0ByteEquivalenceWithA2Baseline:
    """A0 at ``k_hyd=1e-3`` reproduces the committed A.2 record within
    the per-observable tiered tolerance (≤ 1e-6 rel for the
    investigate tier).
    """

    def test_a0_matches_committed_a2_record(self):
        import json

        from scripts.studies.drivers.phase6b_step6_plumbing_ablation import main

        rc = main(["--out-subdir", "phase6b_step6_test_e2e"])
        json_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "StudyResults", "phase6b_step6_test_e2e",
            "ablation_matrix.json",
        )
        assert os.path.exists(json_path), (
            f"Step 6 driver did not write {json_path} (rc={rc})"
        )
        with open(json_path) as f:
            payload = json.load(f)
        a0 = payload["ablation_records"]["A0"]
        audit = a0.get("baseline_reproduction_audit", {})
        assert audit.get("available"), audit
        assert audit.get("worst_tier") in ("pass", "investigate"), audit


# ---------------------------------------------------------------------------
# A0b assembly — at default A0 state, the four A0b gates pass.
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestA0bAssemblyAtA0State:
    """At a freshly-built A0 ctx (λ=0, no Newton solve), the four A0b
    gate quantities are computable and their consistency checks
    (gates 2-4) pass to FP precision.

    Gate 1 (mass-balance closure) is only meaningful at converged
    SS — at λ=0 Γ=0 and R_net=0, so the closure is trivially 0 ≈ 0.
    We assert that the assembled scalars are finite and that the
    anti-symmetry / aliasing identities (gates 2-4) hold by
    construction even at the trivial state.
    """

    def test_a0b_consistency_at_trivial_state(self):
        import firedrake as fd

        sp = _build_step6_sp()
        ctx = _build_step6_ctx(sp)
        bundle = ctx["cation_hydrolysis"]
        # Force lambda = 1 so the forms have a non-trivial coefficient.
        bundle.lambda_hydrolysis_func.assign(fd.Constant(1.0))
        # Forms are physical R_net (default).  At linear-phi IC Γ=0,
        # R_forward is k_hyd·c_M·pka_factor·1 (Langmuir factor =1 since
        # Γ=0); R_backward = 0.  So R_net > 0.  All consistency
        # identities still hold by construction.
        flux_form = ctx["_cation_hydrolysis_R_net_scalar_form"]
        H_form = ctx["_cation_hydrolysis_H_flux_scalar_form"]
        K_form = ctx["_cation_hydrolysis_K_flux_scalar_form"]
        f_flux = float(fd.assemble(flux_form))
        f_h = float(fd.assemble(H_form))
        f_k = float(fd.assemble(K_form))
        import math
        assert math.isfinite(f_flux)
        assert math.isfinite(f_h)
        assert math.isfinite(f_k)
        denom = max(abs(f_flux), 1e-30)
        # Gate 2: H_flux aliases R_net flux.
        assert abs(f_h - f_flux) / denom < 1e-12
        # Gate 3: K_flux is -R_net flux.
        assert abs(f_k - (-f_flux)) / denom < 1e-12
        # Gate 4: anti-symmetry of H + K.
        assert abs(f_h + f_k) / max(abs(f_h), 1e-30) < 1e-12


# ---------------------------------------------------------------------------
# Config-side validation (cross-validation rules)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestConfigCrossValidation:
    """The new flag cross-validation rules in
    :mod:`Forward.bv_solver.config` raise on the documented invalid
    combinations.
    """

    def test_half_physical_without_manufactured_rejected(self):
        sp = _build_step6_sp(apply_h_source=False, apply_k_sink=True)
        with pytest.raises(ValueError, match="manufactured_R_inj"):
            _build_step6_ctx(sp)

    def test_override_with_manufactured_rejected(self):
        sp = _build_step6_sp(
            manufactured_R_inj=1.0,
            override_sigma_singh_counts_pm2=0.022,
        )
        with pytest.raises(ValueError, match="mutually|at most|bypass"):
            _build_step6_ctx(sp)

    def test_negative_override_rejected(self):
        sp = _build_step6_sp(override_sigma_singh_counts_pm2=-0.01)
        with pytest.raises(ValueError, match="non-negative"):
            _build_step6_ctx(sp)

    def test_defaults_preserve_byte_equivalence(self):
        """No flags set ⇒ apply_h_source=True, apply_k_sink=True,
        override=None ⇒ ctx artifacts present but residual gates
        applied (byte-equivalent to v9/v10a/v10a'/A.2)."""
        import firedrake as fd

        sp = _build_step6_sp()
        ctx = _build_step6_ctx(sp)
        assert ctx.get("_cation_hydrolysis_R_net_expr") is not None
        assert ctx.get("_cation_hydrolysis_H_residual_term") is not None
        assert ctx.get("_cation_hydrolysis_K_residual_term") is not None
        # The artifacts exist; the residual still subtracts both.
        conv_cfg = ctx["bv_convergence"]
        assert conv_cfg["apply_h_source"] is True
        assert conv_cfg["apply_k_sink"] is True
        assert conv_cfg["override_sigma_singh_counts_pm2"] is None
        assert conv_cfg["manufactured_R_inj"] is None
