"""Tests for the Jithin Eq 4.31 outer-Picard closure wrap.

Per APPROVED PLAN.md test §1-§16 (see
``.planning/jithin_picard_plan/PLAN.md`` and the 7-round GPT critique loop
at ``docs/handoffs/CHATGPT_HANDOFF_40_picard-closure-cliff/``).

Three tiers:

* **Tier A (no Firedrake)**: pure-Python math + helper tests.  Always run.
* **Tier B (Firedrake form-build only)**: tests that build a UFL form
  and inspect ctx state but do not invoke Newton/SS solves.  Skipped if
  Firedrake unavailable.
* **Tier C (Firedrake + Newton solves)**: integration tests that drive
  the Picard wrap end-to-end.  Marked ``slow``; require Firedrake.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from conftest import skip_without_firedrake


# ---------------------------------------------------------------------------
# Tier A: pure-Python math + helper tests
# ---------------------------------------------------------------------------

class TestPicardTargetMath:
    """test §5/§6/§9 (math fragments): compute_picard_target correctness."""

    def test_no_flux_returns_equilibrium(self):
        """R_O2 = 0 → ξ_target = c_b/θ_b (no-flux equilibrium)."""
        from Forward.bv_solver.closure_picard import compute_picard_target

        xi = compute_picard_target(
            c_b_hat=0.25,
            theta_b=0.94,
            theta_OHP=0.034,
            R_O2_hat=0.0,
            I_hat=1e-5,
            D_O2_hat=1.5e-9,
            xi_old=0.25 / 0.94,
        )
        assert abs(xi - 0.25 / 0.94) < 1e-12

    def test_levich_limit_ξ_zero(self):
        """As K → ∞, ξ_target → 0+ (rate-limited).  Closure self-consistent."""
        from Forward.bv_solver.closure_picard import compute_picard_target

        xi = compute_picard_target(
            c_b_hat=1.0,
            theta_b=1.0,
            theta_OHP=1.0,
            R_O2_hat=1e12,  # huge consumption
            I_hat=1.0,
            D_O2_hat=1e-9,
            xi_old=1.0,
        )
        # K_old = R/(θ·ξ) = 1e12; denom = 1 + 1e12·1·1/1e-9 = 1e21+1 ≈ 1e21
        assert 0.0 < xi < 1e-20

    def test_negative_R_O2_rejected(self):
        """test §9: R_O2_hat < 0 → ValueError."""
        from Forward.bv_solver.closure_picard import compute_picard_target

        with pytest.raises(ValueError, match="R_O2_hat = -0.1"):
            compute_picard_target(
                c_b_hat=1.0, theta_b=1.0, theta_OHP=1.0,
                R_O2_hat=-0.1, I_hat=1.0, D_O2_hat=1.0, xi_old=1.0,
            )

    def test_eq_b_linearized_self_consistent(self):
        """Eq B (semi-implicit Picard target) is the exact solution of the
        LINEARIZED equation with K_old fixed at xi_old.  Verifies:

          ξ_target · (1 + K_old · I · θ / D) = c_b / θ_b
          where K_old = R / (θ · xi_old)
        """
        from Forward.bv_solver.closure_picard import compute_picard_target

        c_b, θ_b, θ_OHP = 1.0, 1.0, 1.0
        R, I, D = 1e-5, 1.0, 1e-9
        xi_old = c_b / θ_b  # no-flux initial
        ξ_target = compute_picard_target(
            c_b_hat=c_b, theta_b=θ_b, theta_OHP=θ_OHP,
            R_O2_hat=R, I_hat=I, D_O2_hat=D, xi_old=xi_old,
        )
        K_old = R / (θ_OHP * xi_old)
        lhs = ξ_target * (1.0 + K_old * I * θ_OHP / D)
        rhs = c_b / θ_b
        assert abs(lhs - rhs) / rhs < 1e-12

    def test_continuum_fixed_point_in_kinetic_regime(self):
        """In the kinetic regime where R·I/D << c_b/θ_b, the Picard fixed
        point satisfies the continuum Eq A' residual ξ + R·I/D − c_b/θ_b ≈ 0
        within a few iterations (manual iteration here, no PDE).
        """
        from Forward.bv_solver.closure_picard import compute_picard_target

        c_b, θ_b, θ_OHP = 1.0, 1.0, 1.0
        R, I, D = 1e-12, 1.0, 1.0  # kinetic regime (R·I/D = 1e-12 << 1)
        xi = c_b / θ_b
        for _ in range(20):
            xi_new = compute_picard_target(
                c_b_hat=c_b, theta_b=θ_b, theta_OHP=θ_OHP,
                R_O2_hat=R, I_hat=I, D_O2_hat=D, xi_old=xi,
            )
            if abs(xi_new - xi) < 1e-15:
                break
            xi = xi_new
        # At converged fixed point, Eq A' residual ≈ 0
        residual = xi + R * I / D - c_b / θ_b
        assert abs(residual) < 1e-10


class TestComputeR_O2_PerSpecies:
    """test §6/§7 fragments: stoichiometry-weighted molar flux aggregation."""

    def test_single_reaction_single_o2(self):
        """R2e: O2 + 2H+ + 2e- → H2O2.  stoich[O2]=-1, so R_O2 = +R."""
        from Forward.bv_solver.closure_picard import compute_R_O2_per_species

        R_per_rxn = ((0, 1e-5),)
        cathodic_stoich = {0: {0: -1}}  # species 0, rxn 0, stoich -1
        result = compute_R_O2_per_species(
            R_per_reaction=R_per_rxn,
            cathodic_stoich=cathodic_stoich,
        )
        assert abs(result[0] - 1e-5) < 1e-18

    def test_parallel_2e_4e_shared_O2(self):
        """R2e + R4e both consume O2; R_O2 = R_2e + R_4e (both with stoich=-1)."""
        from Forward.bv_solver.closure_picard import compute_R_O2_per_species

        R_per_rxn = ((0, 2e-5), (1, 3e-5))  # R2e=2e-5, R4e=3e-5
        cathodic_stoich = {0: {0: -1, 1: -1}}  # O2 consumed by both
        result = compute_R_O2_per_species(
            R_per_reaction=R_per_rxn,
            cathodic_stoich=cathodic_stoich,
        )
        assert abs(result[0] - 5e-5) < 1e-18

    def test_different_stoich_per_reaction(self):
        """If R4e consumes 2 O2 (hypothetical), R_O2 = R_2e + 2·R_4e."""
        from Forward.bv_solver.closure_picard import compute_R_O2_per_species

        R_per_rxn = ((0, 1e-5), (1, 1e-5))
        cathodic_stoich = {0: {0: -1, 1: -2}}
        result = compute_R_O2_per_species(
            R_per_reaction=R_per_rxn,
            cathodic_stoich=cathodic_stoich,
        )
        assert abs(result[0] - 3e-5) < 1e-18


class TestNormalizeFactoryHelper:
    """test §15: _normalize_make_run_ss_factory(None) is make_run_ss."""

    def test_none_normalizes_to_bare(self):
        """test §15: identity check on the exact imported callable."""
        from Forward.bv_solver.grid_per_voltage import (
            make_run_ss,
            _normalize_make_run_ss_factory,
        )
        # Identity check (not just equality)
        assert _normalize_make_run_ss_factory(None) is make_run_ss

    def test_custom_factory_passes_through(self):
        """Custom factory passes through unchanged."""
        from Forward.bv_solver.grid_per_voltage import _normalize_make_run_ss_factory

        def my_factory(*args, **kwargs):
            return lambda max_steps: True

        assert _normalize_make_run_ss_factory(my_factory) is my_factory


class TestPicardConfig:
    """PicardConfig dataclass basic behavior."""

    def test_d_dict_roundtrip(self):
        from Forward.bv_solver.closure_picard import PicardConfig

        cfg = PicardConfig(
            D_per_species_hat=((0, 1.5e-9), (1, 1.6e-9)),
            electrode_marker=3,
            Lx_hat=1.0,
        )
        d = cfg.D_dict()
        assert d == {0: 1.5e-9, 1: 1.6e-9}

    def test_runtime_kwargs_complete(self):
        from Forward.bv_solver.closure_picard import PicardConfig

        cfg = PicardConfig(
            D_per_species_hat=((0, 1.5e-9),),
            electrode_marker=3,
            Lx_hat=1.0,
            damping_init=0.7,
            tol_residual=1e-5,
        )
        kw = cfg.runtime_kwargs()
        expected = {
            "max_picard_iters", "tol_residual", "tol_step", "tol_state",
            "damping_init", "damping_min", "max_damping_retries",
            "strict_floor", "floor_tol", "xi_floor",
        }
        assert set(kw.keys()) == expected
        assert kw["damping_init"] == 0.7
        assert kw["tol_residual"] == 1e-5

    def test_frozen_dataclass(self):
        from Forward.bv_solver.closure_picard import PicardConfig

        cfg = PicardConfig(
            D_per_species_hat=((0, 1.5e-9),),
            electrode_marker=3,
            Lx_hat=1.0,
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            cfg.electrode_marker = 5  # noqa: PLR0603


class TestSnapshotXi:
    """snapshot_xi / restore_xi: deterministic ordering, deep-copy semantics."""

    def test_deterministic_species_order(self):
        """snapshot_xi orders by species_idx ascending."""
        # Build fake xi_funcs without firedrake by mocking the .dat.data_ro
        # attribute.  snapshot_xi only reads from this.
        from Forward.bv_solver.closure_picard import snapshot_xi

        class FakeFunc:
            def __init__(self, vals):
                class Dat:
                    pass
                self.dat = Dat()
                self.dat.data_ro = np.asarray(vals, dtype=float)

        xi_funcs = {2: FakeFunc([1.0]), 0: FakeFunc([2.0]), 1: FakeFunc([3.0])}
        snap = snapshot_xi(xi_funcs)
        assert tuple(s for s, _ in snap) == (0, 1, 2)
        assert snap[0][1] == (2.0,)
        assert snap[1][1] == (3.0,)
        assert snap[2][1] == (1.0,)

    def test_restore_xi_writes_into_dat(self):
        from Forward.bv_solver.closure_picard import restore_xi

        class FakeFunc:
            def __init__(self, size):
                class Dat:
                    def __init__(self, size):
                        self.data = np.zeros(size, dtype=float)
                        self.data_ro = self.data
                self.dat = Dat(size)

        xi_funcs = {0: FakeFunc(1), 1: FakeFunc(1)}
        snap = ((0, (5.0,)), (1, (7.0,)))
        restore_xi(xi_funcs, snap)
        assert xi_funcs[0].dat.data[0] == 5.0
        assert xi_funcs[1].dat.data[0] == 7.0


# ---------------------------------------------------------------------------
# Tier B: Firedrake form-build only (no Newton solves)
# ---------------------------------------------------------------------------


@skip_without_firedrake
class TestFormBuildPicardMode:
    """test §2/§8/§16: form-build wiring for bv_picard_mode."""

    def _build_minimal_form_picard(self, picard_mode: bool = True):
        """Build a small 1-reaction logc_muh form with Picard mode set."""
        import firedrake as fd
        from Forward.bv_solver import make_graded_rectangle_mesh
        from Forward.bv_solver.dispatch import build_context, build_forms
        from scripts._bv_common import (
            C_SCALE,
            D_HP_HAT,
            DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,
            DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
            K0_HAT_R2E,
            SNES_OPTS_CHARGED,
            THREE_SPECIES_LOGC_BOLTZMANN,
            make_bv_solver_params,
            setup_firedrake_env,
        )
        import dataclasses

        setup_firedrake_env()

        species = dataclasses.replace(
            THREE_SPECIES_LOGC_BOLTZMANN,
            c0_vals_hat=[0.25 / C_SCALE, 1e-6 / C_SCALE, 10.0 / C_SCALE],
        )

        cs_entry = {
            **DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,
            "c_bulk_nondim": 190.0 / C_SCALE,
        }
        so4_entry = {
            **DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
            "c_bulk_nondim": 100.0 / C_SCALE,
        }

        rxns = [{
            "k0": float(K0_HAT_R2E) * 1e-25,
            "alpha": 1.0,
            "cathodic_species": 0,
            "anodic_species": None,
            "c_ref": 1.0,
            "stoichiometry": [-1, +1, -2],
            "n_electrons": 2,
            "reversible": False,
            "E_eq_v": 0.695,
        }]

        sp = make_bv_solver_params(
            eta_hat=0.0, dt=0.25, t_end=5.0,
            species=species,
            snes_opts=SNES_OPTS_CHARGED,
            formulation="logc_muh",
            log_rate=True,
            bv_reactions=rxns,
            boltzmann_counterions=[cs_entry, so4_entry],
            multi_ion_enabled=True,
            stern_capacitance_f_m2=0.20,
            initializer="debye_boltzmann",
            l_eff_m=10e-6,
        )
        if picard_mode:
            new_opts = dict(sp.solver_options)
            new_bv = dict(new_opts["bv_convergence"])
            new_bv["bv_jithin_closure_form"] = True
            new_bv["bv_picard_mode"] = True
            new_opts["bv_convergence"] = new_bv
            sp = sp.with_solver_options(new_opts)

        mesh = make_graded_rectangle_mesh(
            Nx=4, Ny=20, beta=3.0, domain_height_hat=10e-6 / 1e-4,  # 0.1 in nondim
        )
        ctx = build_context(sp, mesh=mesh)
        build_forms(ctx, sp)
        return ctx, sp

    def test_picard_mode_populates_ctx_keys(self):
        """test §16 inverse: bv_picard_mode=True populates all required ctx keys."""
        ctx, _ = self._build_minimal_form_picard(picard_mode=True)
        required = {
            "picard_log_xi_funcs",
            "packing_expr",
            "theta_inner_expr",
            "closure_theta_b",
            "closure_bulk_c_hat",
            "closure_cathodic_species_set",
            "closure_cathodic_stoich",
            "closure_packing_floor",
        }
        missing = required - set(ctx.keys())
        assert not missing, f"missing Picard ctx keys: {sorted(missing)}"
        # xi func count == cathodic species count
        assert len(ctx["picard_log_xi_funcs"]) == len(
            ctx["closure_cathodic_species_set"]
        )
        # For our 3-species setup, cathodic species 0 = O2 (z=0)
        assert 0 in ctx["closure_cathodic_species_set"]

    def test_non_picard_mode_does_not_touch_picard_keys(self):
        """test §16: bv_picard_mode=False (default) → no Picard-specific keys.

        ``packing_expr`` and ``theta_inner_expr`` are NOT in the asserted-
        absent set — they may be exposed by other diagnostics in the future
        without breaking the Picard-mode contract.
        """
        ctx, _ = self._build_minimal_form_picard(picard_mode=False)
        PICARD_ONLY = {
            "picard_log_xi_funcs",
            "closure_theta_b",
            "closure_bulk_c_hat",
            "closure_cathodic_species_set",
            "closure_cathodic_stoich",
            "closure_packing_floor",
            "_picard_run_ss_history",
        }
        leaked = PICARD_ONLY & set(ctx.keys())
        assert not leaked, f"non-Picard ctx leaked Picard-only keys: {sorted(leaked)}"

    def test_function_update_no_rebuild(self):
        """test §2: assemble(rate_form) reflects xi_func.assign() without
        rebuild — proves Picard updates propagate to the residual.
        """
        import firedrake as fd
        ctx, _ = self._build_minimal_form_picard(picard_mode=True)
        # Build BV rate observable
        from Forward.bv_solver.observables import _build_bv_observable_form
        rate_form = _build_bv_observable_form(
            ctx, mode="reaction", reaction_index=0, scale=1.0,
        )

        xi_func_0 = ctx["picard_log_xi_funcs"][0]
        # Set ξ = exp(-1) (small)
        xi_func_0.assign(float(-1.0))  # log(ξ) = -1
        R_a = float(fd.assemble(rate_form))
        # Set ξ smaller: exp(-3)
        xi_func_0.assign(float(-3.0))  # log(ξ) = -3
        R_b = float(fd.assemble(rate_form))
        # Rate scales linearly with ξ (k0·θ·ξ·H+_factor·exp), so:
        # R_b / R_a = exp(-3 + 1) = exp(-2) ≈ 0.1353
        if R_a == 0.0 and R_b == 0.0:
            pytest.skip("rate evaluated to 0 (likely far from active V)")
        ratio = R_b / R_a if R_a != 0 else float("inf")
        expected = math.exp(-2.0)
        # Tolerate 1e-9 relative
        assert abs(ratio - expected) / expected < 1e-9, (
            f"rate did not scale linearly with ξ: ratio={ratio}, expected={expected}"
        )


@skip_without_firedrake
class TestPicardConfigValidation:
    """test §8: form-build validation of Picard requirements."""

    def test_z_nonzero_cathodic_species_rejected(self):
        """test §8: z!=0 cathodic species in Picard mode → ValueError at form build."""
        # H+ has z=+1; if we make it a cathodic species in a reaction, build should fail.
        import dataclasses
        from Forward.bv_solver import make_graded_rectangle_mesh
        from Forward.bv_solver.dispatch import build_context, build_forms
        from scripts._bv_common import (
            C_SCALE,
            DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,
            DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
            K0_HAT_R2E,
            SNES_OPTS_CHARGED,
            THREE_SPECIES_LOGC_BOLTZMANN,
            make_bv_solver_params,
            setup_firedrake_env,
        )
        setup_firedrake_env()
        species = dataclasses.replace(
            THREE_SPECIES_LOGC_BOLTZMANN,
            c0_vals_hat=[0.25 / C_SCALE, 1e-6 / C_SCALE, 10.0 / C_SCALE],
        )
        cs_entry = {**DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,
                    "c_bulk_nondim": 190.0 / C_SCALE}
        so4_entry = {**DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
                     "c_bulk_nondim": 100.0 / C_SCALE}
        # Cathodic species 2 = H+ (z=+1) — Picard mode should reject this.
        rxns = [{
            "k0": float(K0_HAT_R2E) * 1e-25,
            "alpha": 1.0,
            "cathodic_species": 2,  # H+, z=+1
            "anodic_species": None,
            "c_ref": 1.0,
            "stoichiometry": [0, 0, -1],
            "n_electrons": 2,
            "reversible": False,
            "E_eq_v": 0.695,
        }]
        sp = make_bv_solver_params(
            eta_hat=0.0, dt=0.25, t_end=5.0,
            species=species,
            snes_opts=SNES_OPTS_CHARGED,
            formulation="logc_muh",
            log_rate=True,
            bv_reactions=rxns,
            boltzmann_counterions=[cs_entry, so4_entry],
            multi_ion_enabled=True,
            stern_capacitance_f_m2=0.20,
            initializer="debye_boltzmann",
            l_eff_m=10e-6,
        )
        new_opts = dict(sp.solver_options)
        new_bv = dict(new_opts["bv_convergence"])
        new_bv["bv_jithin_closure_form"] = True
        new_bv["bv_picard_mode"] = True
        new_opts["bv_convergence"] = new_bv
        sp = sp.with_solver_options(new_opts)

        mesh = make_graded_rectangle_mesh(
            Nx=4, Ny=20, beta=3.0, domain_height_hat=0.1,
        )
        from Forward.bv_solver.dispatch import build_context, build_forms
        ctx = build_context(sp, mesh=mesh)
        with pytest.raises(ValueError, match="requires z=0"):
            build_forms(ctx, sp)


# ---------------------------------------------------------------------------
# Tier C: full Firedrake Newton solves (slow integration tests)
# ---------------------------------------------------------------------------


@skip_without_firedrake
@pytest.mark.slow
class TestPicardIntegration:
    """test §1/§3/§4/§7/§10/§11/§12/§13/§14: full Newton/Picard integration."""

    @pytest.mark.xfail(
        reason="Full Picard integration tests are gated on the in-flight run "
               "verifying Picard converges in this codebase.  Will be filled "
               "in after the run completes if Picard converges; see "
               ".planning/jithin_picard_plan/PLAN.md for test specs."
    )
    def test_no_flux_disabled_reaction(self):
        """test §1: disabled reaction (R_j=0), 1 Picard iter, ξ unchanged."""
        raise NotImplementedError("see xfail message")

    @pytest.mark.xfail(reason="see xfail in test_no_flux_disabled_reaction")
    def test_weak_cathodic_low_rate_regression(self):
        """test §3: V=+0.60, cd matches v1 closure-exact within 1%."""
        raise NotImplementedError

    @pytest.mark.xfail(reason="see xfail in test_no_flux_disabled_reaction")
    def test_theta_unity_recovers_levich(self):
        """test §4: a_vals=[0,0,0] → standard Fick, plateau at Levich ±1%."""
        raise NotImplementedError

    @pytest.mark.xfail(reason="see xfail in test_no_flux_disabled_reaction")
    def test_parallel_2e_4e_shared_supply(self):
        """test §7: minimal 2-rxn config, both reactions share xi_func_O2."""
        raise NotImplementedError

    @pytest.mark.xfail(reason="see xfail in test_no_flux_disabled_reaction")
    def test_byte_equiv_omitted_vs_none(self):
        """test §10: solve_grid_with_anchor with factory omitted vs =None."""
        raise NotImplementedError

    @pytest.mark.xfail(reason="see xfail in test_no_flux_disabled_reaction")
    def test_byte_equiv_self_generated_baseline(self):
        """test §11: self-contained 3-V config; bare vs modified path equal."""
        raise NotImplementedError

    @pytest.mark.xfail(reason="see xfail in test_no_flux_disabled_reaction")
    def test_U_prev_restored_on_picard_rollback(self):
        """test §12: U_prev byte-equal to snapshot after forced rollback."""
        raise NotImplementedError

    @pytest.mark.xfail(reason="see xfail in test_no_flux_disabled_reaction")
    def test_picard_failure_restores_entry_state(self):
        """test §13: every False return of picard_run_ss restores (U, U_prev, xi)."""
        raise NotImplementedError

    @pytest.mark.xfail(reason="see xfail in test_no_flux_disabled_reaction")
    def test_picard_history_captures_all_attempts(self):
        """test §14: bisection scenario → multiple PicardResults; final accepted converges."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Smoke test: closure_picard module is importable + flagged constants exist
# ---------------------------------------------------------------------------


def test_closure_picard_module_smoke():
    """Module imports cleanly + key public surface is exposed."""
    from Forward.bv_solver import closure_picard

    expected_names = {
        "PicardConfig", "StateSnapshot", "PicardIterRecord", "PicardResult",
        "snapshot_state", "restore_state", "snapshot_xi", "restore_xi",
        "compute_picard_diagnostics", "compute_R_O2_per_species",
        "compute_picard_target",
        "make_picard_run_ss", "make_picard_run_ss_factory",
    }
    for name in expected_names:
        assert hasattr(closure_picard, name), f"missing public name: {name}"


def test_config_flag_default_off():
    """bv_picard_mode default False; bv_picard_strict_floor default True."""
    from Forward.bv_solver.config import _get_bv_convergence_cfg

    cfg = _get_bv_convergence_cfg({})
    assert cfg["bv_picard_mode"] is False
    assert cfg["bv_picard_strict_floor"] is True


def test_config_flag_requires_jithin_closure_form():
    """bv_picard_mode=True without bv_jithin_closure_form → ValueError."""
    from Forward.bv_solver.config import _get_bv_convergence_cfg

    with pytest.raises(ValueError, match="bv_jithin_closure_form"):
        _get_bv_convergence_cfg({"bv_convergence": {
            "bv_picard_mode": True,
            "bv_log_rate": True,
            "formulation": "logc_muh",
        }})


def test_config_flag_requires_log_rate():
    """bv_picard_mode=True without bv_log_rate → ValueError."""
    from Forward.bv_solver.config import _get_bv_convergence_cfg

    with pytest.raises(ValueError, match="bv_log_rate"):
        _get_bv_convergence_cfg({"bv_convergence": {
            "bv_picard_mode": True,
            "bv_jithin_closure_form": True,
            "formulation": "logc_muh",
        }})


def test_config_flag_requires_logc_muh_formulation():
    """bv_picard_mode=True with formulation!=logc_muh → ValueError."""
    from Forward.bv_solver.config import _get_bv_convergence_cfg

    with pytest.raises(ValueError, match="formulation='logc_muh'"):
        _get_bv_convergence_cfg({"bv_convergence": {
            "bv_picard_mode": True,
            "bv_jithin_closure_form": True,
            "bv_log_rate": True,
            "formulation": "logc",
        }})


def test_jithin_closure_mutual_exclusion_with_bv_steric_activity():
    """Pre-existing guard: bv_jithin_closure_form + bv_steric_activity are mutex.

    Cross-checked here because bv_picard_mode requires bv_jithin_closure_form
    and our v1 closure-substitute patch (which Picard mode wraps) imposes
    this mutex.  Catches accidental wiring that would re-enable both.
    """
    from Forward.bv_solver.config import _get_bv_convergence_cfg

    with pytest.raises(ValueError, match="mutually exclusive"):
        _get_bv_convergence_cfg({"bv_convergence": {
            "bv_jithin_closure_form": True,
            "bv_steric_activity": True,
        }})
