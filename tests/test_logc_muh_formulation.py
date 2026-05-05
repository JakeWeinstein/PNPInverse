"""Tests for the experimental ``formulation="logc_muh"`` branch.

The muh branch is being landed in phases (see
``~/.claude/plans/look-at-docs-electrochemical-potential-s-misty-trinket.md``).

Phase 1 (config validation + dispatcher routing) and Phase 2 (IC
reconstruction + transform-correctness residual-zero test) live in this
file.  Phase 4 will add overlap CD/PC tests vs the logc baseline.

Tests are split:

* Pure-Python (no Firedrake): config validation, dispatcher routing.
* Firedrake-required: IC field reconstruction, transform residual.
  Marked ``@pytest.mark.slow`` and gated on ``skip_without_firedrake``.
"""
from __future__ import annotations

import os
import sys
import warnings

import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from Forward.bv_solver.config import (
    _VALID_FORMULATIONS,
    _validate_formulation,
    _get_bv_convergence_cfg,
)
from Forward.bv_solver.dispatch import _read_formulation, _resolve_backend
from scripts._bv_common import (
    DEFAULT_CLO4_BOLTZMANN_COUNTERION,
    THREE_SPECIES_LOGC_BOLTZMANN,
    make_bv_solver_params,
)


# ============================================================================
# Phase 1: config validation
# ============================================================================

class TestConfigValidation:
    """``logc_muh`` must be a recognised formulation; mixed case is normalised."""

    def test_logc_muh_in_valid_formulations(self):
        assert "logc_muh" in _VALID_FORMULATIONS

    def test_validate_formulation_accepts_logc_muh(self):
        assert _validate_formulation("logc_muh") == "logc_muh"

    def test_validate_formulation_normalises_mixed_case(self):
        # Old docs use "logc_muH"; lowercase normalisation must accept it.
        assert _validate_formulation("logc_muH") == "logc_muh"
        assert _validate_formulation("LogC_MuH") == "logc_muh"

    def test_validate_formulation_rejects_typo(self):
        with pytest.raises(ValueError, match="formulation must be one of"):
            _validate_formulation("logc_mu_h")  # underscore typo

    def test_get_bv_convergence_stores_canonical(self):
        params = {"bv_convergence": {"formulation": "logc_muH"}}
        cfg = _get_bv_convergence_cfg(params)
        assert cfg["formulation"] == "logc_muh"


# ============================================================================
# Phase 1: dispatcher routing (no-op alias with UserWarning)
# ============================================================================

def _make_solver_params(formulation: str):
    """Build a production-stack SolverParams with the requested formulation."""
    return make_bv_solver_params(
        eta_hat=0.0,
        dt=0.25,
        t_end=10.0,
        species=THREE_SPECIES_LOGC_BOLTZMANN,
        formulation=formulation,
        log_rate=True,
        boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION],
    )


class TestDispatcherRouting:
    """``logc_muh`` routes to the muh backend; ``logc`` routes to logc."""

    def test_read_formulation_returns_logc_muh(self):
        sp = _make_solver_params("logc_muh")
        assert _read_formulation(sp) == "logc_muh"

    def test_read_formulation_normalises_mixed_case(self):
        # Round-trip: factory normalises on the way in, dispatcher
        # normalises on the way out — both must land on canonical.
        sp = _make_solver_params("logc_muH")
        assert _read_formulation(sp) == "logc_muh"

    def test_resolve_backend_returns_logc_muh(self):
        sp = _make_solver_params("logc_muh")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            backend = _resolve_backend(sp)
        assert backend == "logc_muh"
        # Phase 2: routing to the muh module is real, not a Phase 1
        # warn-and-fallback.  No UserWarning should fire.
        muh_warnings = [
            w for w in caught
            if issubclass(w.category, UserWarning) and "logc_muh" in str(w.message)
        ]
        assert muh_warnings == []

    def test_resolve_backend_returns_logc_for_default(self):
        sp = _make_solver_params("logc")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            backend = _resolve_backend(sp)
        assert backend == "logc"
        muh_warnings = [
            w for w in caught
            if issubclass(w.category, UserWarning) and "logc_muh" in str(w.message)
        ]
        assert muh_warnings == []

    def test_resolve_backend_handles_missing_config(self):
        # Empty list/tuple lookups must fall back to "logc" without raising.
        backend = _resolve_backend([])
        assert backend == "logc"


# ============================================================================
# Phase 1: factory wiring through make_bv_solver_params
# ============================================================================

class TestFactoryAcceptsLogcMuh:
    """``make_bv_solver_params(formulation="logc_muh")`` must not raise."""

    def test_factory_accepts_logc_muh(self):
        sp = _make_solver_params("logc_muh")
        assert sp[10]["bv_convergence"]["formulation"] == "logc_muh"

    def test_factory_accepts_mixed_case(self):
        sp = _make_solver_params("logc_muH")
        assert sp[10]["bv_convergence"]["formulation"] == "logc_muh"

    def test_factory_default_unchanged(self):
        # Production default remains "logc" — backward-compat for v13/v15/v16
        # inverse pipeline scripts.
        sp = make_bv_solver_params(
            eta_hat=0.0,
            dt=0.25,
            t_end=10.0,
            species=THREE_SPECIES_LOGC_BOLTZMANN,
            log_rate=True,
            boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION],
        )
        assert sp[10]["bv_convergence"]["formulation"] == "logc"


# ============================================================================
# Phase 2: Firedrake-backed IC + residual-transform tests
# ============================================================================

from conftest import skip_without_firedrake


def _build_muh_ctx(v_rhe: float, *, initializer: str = "linear_phi", mesh_n: int = 8):
    """Build a fresh muh ctx on UnitSquareMesh(mesh_n) for V_RHE=v_rhe.

    Returns (ctx, sp).  Mirrors ``_build_ctx_4sp`` in
    ``tests/test_initializer_debye_boltzmann_4sp.py``.
    """
    from scripts._bv_common import (
        setup_firedrake_env,
        V_T,
        K0_HAT_R1,
        K0_HAT_R2,
        ALPHA_R1,
        ALPHA_R2,
        SNES_OPTS_CHARGED,
    )
    setup_firedrake_env()

    import firedrake as fd  # noqa: E402
    from Forward.bv_solver import build_context, build_forms, set_initial_conditions

    mesh = fd.UnitSquareMesh(mesh_n, mesh_n)

    sp = make_bv_solver_params(
        eta_hat=v_rhe / V_T,
        dt=0.25,
        t_end=80.0,
        species=THREE_SPECIES_LOGC_BOLTZMANN,
        snes_opts=SNES_OPTS_CHARGED,
        formulation="logc_muh",
        log_rate=True,
        boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION],
        k0_hat_r1=K0_HAT_R1,
        k0_hat_r2=K0_HAT_R2,
        alpha_r1=ALPHA_R1,
        alpha_r2=ALPHA_R2,
        E_eq_r1=0.68,
        E_eq_r2=1.78,
        initializer=initializer,
    )
    ctx = build_context(sp, mesh=mesh)
    ctx = build_forms(ctx, sp)
    set_initial_conditions(ctx, sp)
    return ctx, sp


def _build_logc_ctx_matching(sp_muh, *, mesh_n: int = 8):
    """Build a logc ctx with identical params to ``sp_muh`` except formulation.

    Used for residual-transform comparison tests.  Returns (ctx, sp).
    """
    from scripts._bv_common import (
        setup_firedrake_env,
        V_T,
        K0_HAT_R1,
        K0_HAT_R2,
        ALPHA_R1,
        ALPHA_R2,
        SNES_OPTS_CHARGED,
    )
    setup_firedrake_env()

    import firedrake as fd  # noqa: E402
    from Forward.bv_solver import build_context, build_forms, set_initial_conditions

    mesh = fd.UnitSquareMesh(mesh_n, mesh_n)

    bv_conv = sp_muh[10]["bv_convergence"]
    initializer = bv_conv.get("initializer", "linear_phi")

    eta_hat = float(sp_muh[7])
    sp = make_bv_solver_params(
        eta_hat=eta_hat,
        dt=0.25,
        t_end=80.0,
        species=THREE_SPECIES_LOGC_BOLTZMANN,
        snes_opts=SNES_OPTS_CHARGED,
        formulation="logc",
        log_rate=True,
        boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION],
        k0_hat_r1=K0_HAT_R1,
        k0_hat_r2=K0_HAT_R2,
        alpha_r1=ALPHA_R1,
        alpha_r2=ALPHA_R2,
        E_eq_r1=0.68,
        E_eq_r2=1.78,
        initializer=initializer,
    )
    ctx = build_context(sp, mesh=mesh)
    ctx = build_forms(ctx, sp)
    set_initial_conditions(ctx, sp)
    return ctx, sp


@skip_without_firedrake
@pytest.mark.slow
class TestLinearPhiICReconstruction:
    """Linear-phi IC: reconstruct ``c_H = exp(mu_H - em*z_H*phi)`` and check
    it equals ``c_H_bulk`` pointwise.

    The IC is built so this holds by construction; the test guards against
    sign errors and unit-conversion typos in the ``em*z`` factor.
    """

    def test_reconstructs_bulk_cH(self):
        import numpy as np
        from scripts._bv_common import C_HP_HAT

        ctx, sp = _build_muh_ctx(0.0, initializer="linear_phi")
        U = ctx["U"]
        n = ctx["n_species"]
        scaling = ctx["nondim"]
        em = float(scaling["electromigration_prefactor"])
        z_h = float(sp[4][2])  # +1 for H+
        c_h_bulk = C_HP_HAT

        mu_h = U.dat[2].data_ro
        phi = U.dat[n].data_ro
        log_c_h = mu_h - em * z_h * phi
        c_h_reconstructed = np.exp(log_c_h)

        max_err = float(np.abs(c_h_reconstructed - c_h_bulk).max())
        rel_err = max_err / c_h_bulk
        assert rel_err < 1e-12, (
            f"linear_phi muh IC must reconstruct c_H_bulk={c_h_bulk:.4e} "
            f"pointwise; got max|c_H - c_H_bulk|/c_H_bulk = {rel_err:.4e}"
        )

    def test_reconstructs_at_anodic_voltage(self):
        # Same check at a non-trivial V where phi varies strongly.
        import numpy as np
        from scripts._bv_common import C_HP_HAT

        ctx, sp = _build_muh_ctx(0.5, initializer="linear_phi")
        U = ctx["U"]
        n = ctx["n_species"]
        scaling = ctx["nondim"]
        em = float(scaling["electromigration_prefactor"])
        z_h = float(sp[4][2])
        c_h_bulk = C_HP_HAT

        mu_h = U.dat[2].data_ro
        phi = U.dat[n].data_ro
        c_h_reconstructed = np.exp(mu_h - em * z_h * phi)
        rel_err = float(np.abs(c_h_reconstructed - c_h_bulk).max()) / c_h_bulk
        assert rel_err < 1e-10, (
            f"At V=+0.5 V the linear_phi muh IC must still reconstruct "
            f"c_H_bulk; rel_err={rel_err:.4e}"
        )


@skip_without_firedrake
@pytest.mark.slow
class TestDebyeBoltzmannICSmootherInMu:
    """Debye-Boltzmann IC: ``mu_H`` should be much smoother than ``u_H``
    in y at high anodic phi.

    This is the diagnostic signal for the analytic-cancellation property:
    ``mu_H_init = u_H_init + em*z_H*phi_init`` with ``em*z_H = 1`` reduces
    to ``2*ln(H_outer) - ln(c_clo4_bulk)`` -- the diffuse-layer ``psi``
    cancels.  ``u_H = ln(H_outer) - psi`` carries the full O(psi_D)
    Debye-layer variation.
    """

    def test_mu_h_range_much_smaller_than_u_h_range_at_v05(self):
        import numpy as np

        ctx, sp = _build_muh_ctx(0.5, initializer="debye_boltzmann")
        # If the IC fell back to linear_phi, this test is meaningless.
        assert ctx.get("initializer_fallback") is False, (
            f"Debye-Boltzmann IC must not fall back at V=+0.5 V; "
            f"reason={ctx.get('initializer_fallback_reason')!r}"
        )

        U = ctx["U"]
        n = ctx["n_species"]
        scaling = ctx["nondim"]
        em = float(scaling["electromigration_prefactor"])
        z_h = float(sp[4][2])

        mu_h = U.dat[2].data_ro
        phi = U.dat[n].data_ro
        u_h = mu_h - em * z_h * phi  # reconstructed log(c_H)

        mu_range = float(mu_h.max() - mu_h.min())
        u_range = float(u_h.max() - u_h.min())

        # u_H spans ~psi_D = 0.5/V_T = 19.5 from electrode to bulk; mu_H
        # should span much less (analytic Boltzmann cancellation).
        assert u_range > 5.0, (
            f"u_H range should reflect Debye-layer depletion; got {u_range:.4f}"
        )
        assert mu_range < 0.25 * u_range, (
            f"mu_H range ({mu_range:.4f}) should be << u_H range "
            f"({u_range:.4f}) at V=+0.5 V; the muh transform is supposed "
            f"to absorb the diffuse-layer variation."
        )


@skip_without_firedrake
@pytest.mark.slow
class TestTransformResidualEquivalence:
    """At a state that is the muh-transform of a logc IC state, the muh
    residual should equal the logc residual.

    This is the MMS substitute called out in the plan: rather than rebuild
    the manufactured-source machinery for muh (Phase 6, deferred),
    independently exercise the ``forms_logc_muh`` substitutions by checking
    they produce the same residual values as ``forms_logc`` at the
    corresponding state.

    Catches sign errors in ``em*z`` substitution that overlap-CD/PC tests
    would only catch at higher voltage.
    """

    def test_residual_equivalence_at_linear_phi_ic(self):
        import numpy as np
        import firedrake as fd

        # Build both contexts at V_RHE = -0.1 V (mild cathodic, both IC
        # and SS deep in the converged window for both formulations).
        ctx_muh, sp_muh = _build_muh_ctx(-0.1, initializer="linear_phi")
        ctx_logc, _ = _build_logc_ctx_matching(sp_muh)

        # Sanity: identical mesh layout, identical # unknowns.
        assert ctx_muh["U"].dat[0].data_ro.shape == ctx_logc["U"].dat[0].data_ro.shape

        # Both ICs should produce the *physically same* state (constant
        # bulk concentrations + linear phi profile).  Verify the muh ctx's
        # reconstruction matches the logc ctx's u_H pointwise.
        n = ctx_muh["n_species"]
        scaling = ctx_muh["nondim"]
        em = float(scaling["electromigration_prefactor"])
        z_h = float(sp_muh[4][2])

        u_h_logc = ctx_logc["U"].dat[2].data_ro
        u_h_muh_reconstructed = (
            ctx_muh["U"].dat[2].data_ro
            - em * z_h * ctx_muh["U"].dat[n].data_ro
        )
        max_state_diff = float(np.abs(u_h_logc - u_h_muh_reconstructed).max())
        assert max_state_diff < 1e-10, (
            f"Linear-phi ICs in logc and muh should produce identical "
            f"physical states; max|u_H_logc - u_H_muh_reconstructed|"
            f" = {max_state_diff:.4e}"
        )

        # Assemble both residuals.  Since both ICs are physically the same
        # state, the residuals (as vectors over test-function dofs) must
        # match.
        F_logc = fd.assemble(ctx_logc["F_res"])
        F_muh = fd.assemble(ctx_muh["F_res"])

        for i in range(n + 1):  # n species + 1 phi
            res_logc = F_logc.dat[i].data_ro
            res_muh = F_muh.dat[i].data_ro
            assert res_logc.shape == res_muh.shape
            max_diff = float(np.abs(res_logc - res_muh).max())
            scale = max(float(np.abs(res_logc).max()), 1.0)
            rel_diff = max_diff / scale
            # Tolerance: FE assembly differences across two independent
            # contexts give O(1e-12) noise.  A real sign error would show
            # up at ~1.0 magnitude.
            assert rel_diff < 1e-9, (
                f"Residual subspace {i}: max|F_logc - F_muh|/scale "
                f"= {rel_diff:.4e}; muh transform has a defect"
            )


# ============================================================================
# Phase 3: diagnostics consistency between logc and muh
# ============================================================================

@skip_without_firedrake
@pytest.mark.slow
class TestDiagnosticsConsistency:
    """``surface_field_means`` and ``collect_diagnostics`` must report the
    same physical quantities for matched logc and muh runs.

    The contract for downstream callers (study scripts, summary.md
    builders) is that ``u{i}_surface_mean`` is the surface mean of
    ``log(c_i)`` regardless of formulation.  For muh, the muh-aware
    ``ctx['u_exprs']`` reroute makes this automatic.
    """

    def test_surface_means_match_at_linear_phi_ic(self):
        import numpy as np
        from Forward.bv_solver.diagnostics import surface_field_means

        ctx_muh, sp_muh = _build_muh_ctx(-0.1, initializer="linear_phi")
        ctx_logc, _ = _build_logc_ctx_matching(sp_muh)

        means_logc = surface_field_means(ctx_logc)
        means_muh = surface_field_means(ctx_muh)

        # u{i}_surface_mean: same physical log(c_i) surface mean.
        for i in range(ctx_muh["n_species"]):
            key = f"u{i}_surface_mean"
            assert key in means_logc and key in means_muh
            diff = abs(means_logc[key] - means_muh[key])
            assert diff < 1e-9, (
                f"{key}: logc={means_logc[key]:.6e}, muh={means_muh[key]:.6e}, "
                f"diff={diff:.2e}"
            )

        # c{i}_surface_mean: derived from u_mean (Jensen-fault is shared
        # across formulations); rel diff should be tiny.
        for i in range(ctx_muh["n_species"]):
            key = f"c{i}_surface_mean"
            scale = max(abs(means_logc[key]), 1e-30)
            rel_diff = abs(means_logc[key] - means_muh[key]) / scale
            assert rel_diff < 1e-9, (
                f"{key}: logc={means_logc[key]:.6e}, muh={means_muh[key]:.6e}, "
                f"rel_diff={rel_diff:.2e}"
            )

        # phi_surface_mean: identical (same primary variable in both).
        diff = abs(means_logc["phi_surface_mean"] - means_muh["phi_surface_mean"])
        assert diff < 1e-9

    def test_muh_surface_mean_extra_keys(self):
        from Forward.bv_solver.diagnostics import surface_field_means

        ctx_muh, _ = _build_muh_ctx(-0.1, initializer="linear_phi")
        means_muh = surface_field_means(ctx_muh)

        # Muh-only key: mu{i}_surface_mean for the proton (i=2).
        assert "mu2_surface_mean" in means_muh
        # Sanity: mu_H_surf = u_H_surf + em*z_H*phi_surf.
        scaling = ctx_muh["nondim"]
        em = float(scaling["electromigration_prefactor"])
        # z_H = +1 for the production stack.
        expected_offset = em * 1.0 * means_muh["phi_surface_mean"]
        diff = abs(
            means_muh["mu2_surface_mean"]
            - (means_muh["u2_surface_mean"] + expected_offset)
        )
        assert diff < 1e-9

    def test_muh_surface_mean_logc_unchanged(self):
        from Forward.bv_solver.diagnostics import surface_field_means

        ctx_logc, _ = _build_logc_ctx_matching(
            _build_muh_ctx(-0.1, initializer="linear_phi")[1]
        )
        means_logc = surface_field_means(ctx_logc)

        # Logc must NOT have any mu* keys (no formulation regression).
        for k in means_logc:
            assert not k.startswith("mu"), (
                f"logc surface_field_means produced unexpected mu* key: {k}"
            )

    def test_collect_diagnostics_muh_consistent(self):
        """``collect_diagnostics`` reports ``min/max u{i}`` consistently as
        log(c_i) extremes, independent of formulation."""
        from Forward.bv_solver.diagnostics import collect_diagnostics

        ctx_muh, sp_muh = _build_muh_ctx(-0.1, initializer="linear_phi")
        ctx_logc, _ = _build_logc_ctx_matching(sp_muh)

        diag_muh = collect_diagnostics(ctx_muh, phase="cold_z_ramp")
        diag_logc = collect_diagnostics(ctx_logc, phase="cold_z_ramp")

        for i in range(ctx_muh["n_species"]):
            for key in (f"min_u{i}", f"max_u{i}"):
                assert key in diag_logc and key in diag_muh
                diff = abs(diag_logc[key] - diag_muh[key])
                assert diff < 1e-9, (
                    f"{key}: logc={diag_logc[key]:.4e}, muh={diag_muh[key]:.4e}"
                )

        # Muh adds raw mu_H min/max for species 2.
        assert "min_mu2" in diag_muh and "max_mu2" in diag_muh
        # Logc must not have these.
        assert "min_mu2" not in diag_logc
