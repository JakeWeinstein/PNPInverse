"""Phase 7.2 Stage-3 observable contracts (session-43 plan).

Slow (Firedrake) tests:
  * form_cd (mode="current_density") equals the manual
    electron-weighted per-reaction sum — algebraic identity, checked
    at the IC state (no solve needed).
  * Isolated-reaction contracts: 2e-water-only => cd == pc;
    4e-water-only => pc == 0 and cd != 0.
  * Escape ≡ production: at a converged steady state on a coarse
    mesh, the outer-boundary H2O2 flux equals the reaction-sum
    production within 1%, and the −I_SCALE-scaled observable is
    cathodic-negative (R1#4 / R2#... boundary-flux sign test).

Run: pytest -m slow tests/test_phase7p2_observables.py -s -vv
"""
import math

import numpy as np
import pytest

pytestmark = pytest.mark.slow

COARSE = dict(Nx=4, Ny=24, beta=3.0, domain_height_hat=0.154)
ELECTRODE, BULK_TOP = 3, 4


def _dual_pathway_water_only(k2_factor=1e-5, k4_factor=1e-7):
    """Water-only list; factors MULTIPLY the preset k0 (harness
    convention) — 2.07e-4 / 2.90e-14 are the theta* factors."""
    from scripts._bv_common import PARALLEL_2E_4E_DUAL_PATHWAY
    rxns = []
    for r in PARALLEL_2E_4E_DUAL_PATHWAY:
        r = dict(r)
        if r["proton_donor"] == "water":
            f = k2_factor if r["n_electrons"] == 2 else k4_factor
            r["k0"] = float(r["k0"]) * float(f)
        else:
            r["k0"] = 0.0
        rxns.append(r)
    return rxns


def _build_ctx(reactions):
    import firedrake as fd
    from scripts._bv_common import (
        DEFAULT_KPLUS_BOLTZMANN_COUNTERION_STERIC,
        DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
        THREE_SPECIES_LOGC_BOLTZMANN,
        make_bv_solver_params,
    )
    from Forward.bv_solver import make_graded_rectangle_mesh
    from Forward.bv_solver.dispatch import (
        build_context, build_forms, set_initial_conditions,
    )
    sp = make_bv_solver_params(
        eta_hat=0.0, dt=0.25, t_end=10.0,
        species=THREE_SPECIES_LOGC_BOLTZMANN,
        formulation="logc_muh", log_rate=True,
        bv_reactions=reactions,
        boltzmann_counterions=[
            DEFAULT_KPLUS_BOLTZMANN_COUNTERION_STERIC,
            DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
        ],
        multi_ion_enabled=True,
        stern_capacitance_f_m2=0.10,
        initializer="linear_phi",
        l_eff_m=15.4e-6,
        enable_water_ionization=True,
    )
    mesh = make_graded_rectangle_mesh(**COARSE)
    ctx = build_context(sp, mesh=mesh)
    ctx = build_forms(ctx, sp)
    set_initial_conditions(ctx, sp)
    return ctx, sp


def _assemble(form):
    import firedrake as fd
    return float(fd.assemble(form))


class TestCdObservableContracts:
    def test_cd_equals_manual_electron_weighted_sum(self):
        import firedrake as fd
        from Forward.bv_solver.observables import (
            N_ELECTRONS_REF, _build_bv_observable_form,
        )
        from scripts._bv_common import I_SCALE

        ctx, _ = _build_ctx(_dual_pathway_water_only())
        form_cd = _build_bv_observable_form(
            ctx, mode="current_density", reaction_index=None,
            scale=-I_SCALE)
        ds = fd.Measure("ds", domain=ctx["mesh"])
        n_es = [int(r["n_electrons"])
                for r in ctx["nondim"]["bv_reactions"]]
        manual = sum(
            (-I_SCALE) * (n_e / N_ELECTRONS_REF) * _assemble(
                expr * ds(ELECTRODE))
            for n_e, expr in zip(n_es, ctx["bv_rate_exprs"]))
        cd = _assemble(form_cd)
        assert abs(cd - manual) <= max(1e-12, 1e-10 * abs(manual))

    def test_isolated_2e_cd_equals_pc(self):
        from Forward.bv_solver.observables import _build_bv_observable_form
        from scripts._bv_common import I_SCALE

        ctx, _ = _build_ctx(_dual_pathway_water_only(k2_factor=1e-2,
                                                     k4_factor=0.0))
        cd = _assemble(_build_bv_observable_form(
            ctx, mode="current_density", reaction_index=None,
            scale=-I_SCALE))
        pc = _assemble(_build_bv_observable_form(
            ctx, mode="reaction_sum", reaction_index=None,
            scale=-I_SCALE))
        assert cd != 0.0
        assert cd == pytest.approx(pc, rel=1e-10)

    def test_isolated_4e_pc_zero_cd_nonzero(self):
        from Forward.bv_solver.observables import _build_bv_observable_form
        from scripts._bv_common import I_SCALE

        ctx, _ = _build_ctx(_dual_pathway_water_only(k2_factor=0.0,
                                                     k4_factor=1e-2))
        cd = _assemble(_build_bv_observable_form(
            ctx, mode="current_density", reaction_index=None,
            scale=-I_SCALE))
        pc = _assemble(_build_bv_observable_form(
            ctx, mode="reaction_sum", reaction_index=None,
            scale=-I_SCALE))
        assert cd != 0.0
        assert pc == pytest.approx(0.0, abs=1e-14)


class TestEscapeEqualsProduction:
    """Steady state on the coarse mesh: outer-boundary H2O2 flux ==
    reaction-sum production (producing-only config) within 1%; the
    scaled observable matches the ring target's cathodic-negative
    sign convention."""

    H2O2_IDX = 1

    def test_escape_flux_matches_production(self):
        import firedrake as fd
        import firedrake.adjoint as adj
        from Forward.bv_solver import make_graded_rectangle_mesh
        from Forward.bv_solver.anchor_continuation import (
            solve_anchor_with_continuation,
        )
        from Forward.bv_solver.observables import _build_bv_observable_form
        from scripts._bv_common import (
            D_H2O2_HAT, I_SCALE, V_T,
            DEFAULT_KPLUS_BOLTZMANN_COUNTERION_STERIC,
            DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
            THREE_SPECIES_LOGC_BOLTZMANN,
            make_bv_solver_params,
        )

        from Forward.bv_solver.grid_per_voltage import (
            solve_grid_with_anchor,
        )
        from Forward.bv_solver.anchor_continuation import (
            PreconvergedAnchor,
        )
        from Forward.bv_solver.grid_per_voltage import snapshot_U

        # production-recipe pattern via the DRIVER's own config path
        # (_build_reactions + _make_sp: exponent_clip 100, u_clamp,
        # SNES opts, theta* alphas — zero config drift vs the
        # harness).  Anchor at REST, warm-walk one plateau point
        # (V_RHE 0.20 -> V_solver -0.819).
        from types import SimpleNamespace
        import scripts.studies.drivers.solver_demo_slide15_dual_pathway_cs as dp

        V_TARGET = -0.819
        opts = SimpleNamespace(
            routes="water",
            k0_water_2e_factor=2.073593e-4,
            k0_water_4e_factor=2.901518e-14,
            k0_acid_4e_factor=1e-15,
            alpha_water_2e=0.5499855248013807,
            alpha_water_4e=0.2854405505612447,
            l_eff_um=15.4, bulk_h_mol_m3=4.07e-4,
            enable_water_ionization=True, coarse_grid=True,
            cation="k", v_ocp_rhe=1.019,
            v_grid_lo=None, v_grid_hi=None,
        )
        rxns = dp._build_reactions(opts)
        sp, _ = dp._make_sp(opts, rxns,
                            stern_capacitance_f_m2=0.10,
                            initializer="linear_phi")
        mesh = make_graded_rectangle_mesh(**COARSE)
        k0_targets = {j: float(r["k0"]) for j, r in enumerate(rxns)
                      if float(r["k0"]) > 0.0}
        with adj.stop_annotating():
            ar = solve_anchor_with_continuation(
                sp.with_phi_applied(0.0), mesh=mesh,
                k0_targets=k0_targets,
                initial_scales=(1e-12, 1e-9, 1e-6, 1e-3, 1.0),
                max_inserts_per_step=6,
                ic_at_target=True, kw_eff_ladder=None,
            )
            assert ar.converged
            anchor = PreconvergedAnchor(
                phi_applied_eta=0.0,
                U_snapshot=tuple(
                    np.asarray(a).copy()
                    for a in snapshot_U(ar.ctx["U"])),
                k0_targets=tuple(sorted(
                    (int(j), float(k))
                    for j, k in k0_targets.items())),
                mesh_dof_count=int(ar.ctx["U"].function_space().dim()),
                ladder_history=tuple(
                    (float(s), str(o)) for s, o in ar.ladder_history),
            )
            grabbed = {}

            def _cb(orig_idx, _phi_eta, walk_ctx):
                grabbed["ctx"] = walk_ctx

            grid = solve_grid_with_anchor(
                sp, anchor=anchor,
                phi_applied_values=np.array([V_TARGET / float(V_T)]),
                mesh=mesh, n_substeps_warm=8, bisect_depth_warm=5,
                per_point_callback=_cb,
            )
        assert grid.points[0].converged
        ctx = grabbed["ctx"]
        # kill the transient at the TARGET V (harness pattern: phi
        # explicitly re-assigned; the walk ctx must not be trusted to
        # still sit at the target), dt -> inf, one more Newton solve
        import firedrake as _fd
        ctx["phi_applied_func"].assign(V_TARGET / float(V_T))
        ctx["dt_const"].assign(1.0e12)
        ctx["U_prev"].assign(ctx["U"])
        prob = _fd.NonlinearVariationalProblem(
            ctx["F_res"], ctx["U"], bcs=ctx["bcs"], J=ctx["J_form"])
        _fd.NonlinearVariationalSolver(prob).solve()

        ds = fd.Measure("ds", domain=ctx["mesh"])
        production_nondim = float(fd.assemble(
            sum(float(r["stoichiometry"][self.H2O2_IDX]) * expr
                for r, expr in zip(rxns, ctx["bv_rate_exprs"]))
            * ds(ELECTRODE)))
        assert production_nondim > 0.0

        # Discrete-consistent outer-boundary escape: pair the ACTUAL
        # residual 1-form with psi = y/H on the H2O2 test slot (zero
        # elsewhere).  psi vanishes at the electrode (kills the
        # source term) and is 1 at the top, so -F_res(psi) is the
        # outward escape INCLUDING every flux term the residual
        # carries (diffusion + Bikerman steric activity — the naive
        # -D dc/dn estimator misses the steric part by ~4.5% at this
        # state and raw boundary gradients of P1 fields are junk).
        import ufl
        W = ctx["U"].function_space()
        w = fd.Function(W)
        xx, yy = fd.SpatialCoordinate(ctx["mesh"])
        w.subfunctions[self.H2O2_IDX].interpolate(
            yy / COARSE["domain_height_hat"])
        v_test = ctx["F_res"].arguments()[0]
        escape_nondim = -float(fd.assemble(
            ufl.replace(ctx["F_res"], {v_test: w})))
        rel = abs(escape_nondim - production_nondim) / production_nondim
        assert rel <= 1e-4, (escape_nondim, production_nondim, rel)

        # sign test: the −I_SCALE-scaled production observable must be
        # cathodic-negative (same convention as the ring-derived target)
        pc = float(fd.assemble(_build_bv_observable_form(
            ctx, mode="reaction_sum", reaction_index=None,
            scale=-I_SCALE)))
        assert pc < 0.0
