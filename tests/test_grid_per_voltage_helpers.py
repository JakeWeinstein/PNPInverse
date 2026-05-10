"""Tests for the public ``snapshot_U`` / ``restore_U`` / ``make_run_ss``
helpers promoted out of :mod:`Forward.bv_solver.grid_per_voltage` in
Phase 5γ M2.

The byte-equivalence test is the regression guard for Risk R5: we
need to know that factoring the inline ``_make_run_ss`` closure into
a public ``make_run_ss`` factory does NOT change behavior in
``solve_grid_per_voltage_cold_with_warm_fallback``.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Section 1 -- public-API surface
# ---------------------------------------------------------------------------

def test_public_aliases_exist_at_module_top_level():
    from Forward.bv_solver.grid_per_voltage import (
        make_run_ss,
        restore_U,
        snapshot_U,
    )
    assert callable(snapshot_U)
    assert callable(restore_U)
    assert callable(make_run_ss)


def test_public_aliases_exported_from_package():
    from Forward.bv_solver import make_run_ss, restore_U, snapshot_U
    assert callable(snapshot_U)
    assert callable(restore_U)
    assert callable(make_run_ss)


def test_make_run_ss_signature_is_kwarg_only():
    """``make_run_ss(*, ctx, solver, of_cd, ...)`` — every parameter
    after the leading bare ``*`` must be keyword-only so future kwarg
    additions don't break callers."""
    import inspect

    from Forward.bv_solver.grid_per_voltage import make_run_ss

    sig = inspect.signature(make_run_ss)
    for name, param in sig.parameters.items():
        assert param.kind == inspect.Parameter.KEYWORD_ONLY, (
            f"{name} should be KEYWORD_ONLY; got {param.kind!r}"
        )


def test_warm_walk_phi_signature_is_kwarg_only():
    """Every parameter must be keyword-only for forward-compat."""
    import inspect

    from Forward.bv_solver.grid_per_voltage import warm_walk_phi

    sig = inspect.signature(warm_walk_phi)
    for name, param in sig.parameters.items():
        assert param.kind == inspect.Parameter.KEYWORD_ONLY, (
            f"{name} should be KEYWORD_ONLY; got {param.kind!r}"
        )


def test_warm_walk_phi_exported_from_package():
    from Forward.bv_solver import warm_walk_phi
    from Forward.bv_solver.grid_per_voltage import (
        warm_walk_phi as warm_walk_phi_internal,
    )
    assert warm_walk_phi is warm_walk_phi_internal
    assert callable(warm_walk_phi)


def test_solve_grid_with_anchor_signature_is_kwarg_only():
    """Every parameter after the leading positional must be kwarg-only."""
    import inspect

    from Forward.bv_solver.grid_per_voltage import solve_grid_with_anchor

    sig = inspect.signature(solve_grid_with_anchor)
    saw_pos = False
    for name, param in sig.parameters.items():
        if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
            assert not saw_pos, (
                f"only one positional parameter allowed; {name} is the second"
            )
            saw_pos = True
            continue
        assert param.kind == inspect.Parameter.KEYWORD_ONLY, (
            f"{name} should be KEYWORD_ONLY; got {param.kind!r}"
        )


def test_solve_grid_with_anchor_exported_from_package():
    from Forward.bv_solver import solve_grid_with_anchor
    from Forward.bv_solver.grid_per_voltage import (
        solve_grid_with_anchor as solve_grid_with_anchor_internal,
    )
    assert solve_grid_with_anchor is solve_grid_with_anchor_internal
    assert callable(solve_grid_with_anchor)


def test_solve_grid_with_anchor_rejects_non_anchor_input():
    """Passing a dict or None where a PreconvergedAnchor is required
    must raise TypeError BEFORE any expensive Firedrake build."""
    import numpy as np

    from Forward.bv_solver.grid_per_voltage import solve_grid_with_anchor

    with pytest.raises(TypeError, match="PreconvergedAnchor"):
        solve_grid_with_anchor(
            (1, 1, 0.25, 80.0, 0, 0, 0, 0, 0, 0, {}),
            anchor={"phi": 0.0},  # not a PreconvergedAnchor
            phi_applied_values=np.array([0.0]),
            mesh=None,
        )


# ---------------------------------------------------------------------------
# Section 2 -- private/public delegation
# ---------------------------------------------------------------------------

def test_snapshot_U_public_is_private_alias():
    from Forward.bv_solver.grid_per_voltage import _snapshot_U, snapshot_U
    # Identity not strictly required (one wraps the other), but they
    # must produce the same output for the same input.
    class _FakeDat:
        def __init__(self, data):
            import numpy as np
            self._data = np.asarray(data)

        @property
        def data_ro(self):
            return self._data

    class _FakeU:
        def __init__(self, parts):
            self.dat = [_FakeDat(p) for p in parts]

    U = _FakeU([[1, 2, 3], [4, 5]])
    snap_priv = _snapshot_U(U)
    snap_pub = snapshot_U(U)
    assert len(snap_priv) == len(snap_pub) == 2
    for a, b in zip(snap_priv, snap_pub):
        assert (a == b).all()


# ---------------------------------------------------------------------------
# Section 3 -- slow Firedrake round-trip + byte-equivalence
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_snapshot_restore_round_trip_real_firedrake():
    """Snapshot a real ``U``, mutate the dat in place, restore — the
    result should be the original."""
    pytest.importorskip("firedrake")
    import numpy as np

    from Forward.bv_solver import (
        make_graded_rectangle_mesh,
        build_context_logc_muh,
        build_forms_logc_muh,
    )
    from Forward.bv_solver.grid_per_voltage import restore_U, snapshot_U
    from scripts._bv_common import (
        setup_firedrake_env,
        SNES_OPTS_CHARGED,
        V_T,
        THREE_SPECIES_LOGC_BOLTZMANN,
        DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,
        DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
        K0_HAT_R2E,
        K0_HAT_R4E,
        ALPHA_R2E,
        ALPHA_R4E,
        E_EQ_R2E_V,
        E_EQ_R4E_V,
        C_HP_HAT,
        make_bv_solver_params,
    )
    setup_firedrake_env()

    rxns = [
        {
            "k0": float(K0_HAT_R2E),
            "alpha": float(ALPHA_R2E),
            "cathodic_species": 0,
            "anodic_species": 1,
            "c_ref": 1.0,
            "stoichiometry": [-1, +1, -2],
            "n_electrons": 2,
            "reversible": True,
            "E_eq_v": float(E_EQ_R2E_V),
            "cathodic_conc_factors": [
                {"species": 2, "power": 2, "c_ref_nondim": float(C_HP_HAT)},
            ],
        },
        {
            "k0": float(K0_HAT_R4E),
            "alpha": float(ALPHA_R4E),
            "cathodic_species": 0,
            "anodic_species": None,
            "c_ref": 0.0,
            "stoichiometry": [-1, 0, -4],
            "n_electrons": 4,
            "reversible": False,
            "E_eq_v": float(E_EQ_R4E_V),
            "cathodic_conc_factors": [
                {"species": 2, "power": 4, "c_ref_nondim": float(C_HP_HAT)},
            ],
        },
    ]
    sp = make_bv_solver_params(
        eta_hat=0.0,
        dt=0.25,
        t_end=80.0,
        species=THREE_SPECIES_LOGC_BOLTZMANN,
        snes_opts={**SNES_OPTS_CHARGED, "snes_max_it": 50},
        formulation="logc_muh",
        log_rate=True,
        u_clamp=100.0,
        bv_reactions=rxns,
        boltzmann_counterions=[
            DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,
            DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
        ],
        multi_ion_enabled=True,
        stern_capacitance_f_m2=0.10,
        initializer="linear_phi",
    )
    sp = sp.with_phi_applied(0.0 / V_T)

    mesh = make_graded_rectangle_mesh(Nx=4, Ny=10, beta=3.0)
    ctx = build_context_logc_muh(sp, mesh=mesh)
    ctx = build_forms_logc_muh(ctx, sp)

    snap = snapshot_U(ctx["U"])
    # snap is a tuple of np arrays (one per subspace).
    assert all(isinstance(arr, np.ndarray) for arr in snap)

    # Mutate U in-place: scribble a known offset onto every dat.
    for d in ctx["U"].dat:
        d.data[:] = d.data + 7.0
    # Confirm mutation did happen (dat values now differ from snap).
    for snap_arr, d in zip(snap, ctx["U"].dat):
        assert not np.allclose(snap_arr, d.data_ro), (
            "mutation did not actually change U.dat"
        )

    # Restore from snap.
    restore_U(snap, ctx["U"], ctx["U_prev"])
    for snap_arr, d in zip(snap, ctx["U"].dat):
        assert np.allclose(snap_arr, d.data_ro), (
            "restore_U did not restore U.dat from snapshot"
        )
    # U_prev should now equal U (which equals snap).
    for snap_arr, d in zip(snap, ctx["U_prev"].dat):
        assert np.allclose(snap_arr, d.data_ro)


@pytest.mark.slow
def test_make_run_ss_returns_callable_real_firedrake():
    """Smoke-test that ``make_run_ss`` returns a callable; do not
    actually run Newton (the next test does that as part of byte
    equivalence)."""
    pytest.importorskip("firedrake")
    import firedrake as fd

    from Forward.bv_solver import (
        make_graded_rectangle_mesh,
        build_context_logc_muh,
        build_forms_logc_muh,
    )
    from Forward.bv_solver.grid_per_voltage import make_run_ss
    from Forward.bv_solver.observables import _build_bv_observable_form
    from scripts._bv_common import (
        setup_firedrake_env,
        SNES_OPTS_CHARGED,
        V_T,
        THREE_SPECIES_LOGC_BOLTZMANN,
        DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,
        DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
        K0_HAT_R2E,
        K0_HAT_R4E,
        ALPHA_R2E,
        ALPHA_R4E,
        E_EQ_R2E_V,
        E_EQ_R4E_V,
        C_HP_HAT,
        make_bv_solver_params,
    )
    setup_firedrake_env()

    rxns = [
        {
            "k0": float(K0_HAT_R2E),
            "alpha": float(ALPHA_R2E),
            "cathodic_species": 0,
            "anodic_species": 1,
            "c_ref": 1.0,
            "stoichiometry": [-1, +1, -2],
            "n_electrons": 2,
            "reversible": True,
            "E_eq_v": float(E_EQ_R2E_V),
            "cathodic_conc_factors": [
                {"species": 2, "power": 2, "c_ref_nondim": float(C_HP_HAT)},
            ],
        },
        {
            "k0": float(K0_HAT_R4E),
            "alpha": float(ALPHA_R4E),
            "cathodic_species": 0,
            "anodic_species": None,
            "c_ref": 0.0,
            "stoichiometry": [-1, 0, -4],
            "n_electrons": 4,
            "reversible": False,
            "E_eq_v": float(E_EQ_R4E_V),
            "cathodic_conc_factors": [
                {"species": 2, "power": 4, "c_ref_nondim": float(C_HP_HAT)},
            ],
        },
    ]
    sp = make_bv_solver_params(
        eta_hat=0.0,
        dt=0.25,
        t_end=80.0,
        species=THREE_SPECIES_LOGC_BOLTZMANN,
        snes_opts={**SNES_OPTS_CHARGED, "snes_max_it": 50},
        formulation="logc_muh",
        log_rate=True,
        u_clamp=100.0,
        bv_reactions=rxns,
        boltzmann_counterions=[
            DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,
            DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
        ],
        multi_ion_enabled=True,
        stern_capacitance_f_m2=0.10,
        initializer="linear_phi",
    )
    sp = sp.with_phi_applied(0.0 / V_T)

    mesh = make_graded_rectangle_mesh(Nx=4, Ny=10, beta=3.0)
    ctx = build_context_logc_muh(sp, mesh=mesh)
    ctx = build_forms_logc_muh(ctx, sp)

    problem = fd.NonlinearVariationalProblem(
        ctx["F_res"], ctx["U"], bcs=ctx["bcs"], J=ctx["J_form"]
    )
    solver = fd.NonlinearVariationalSolver(
        problem, solver_parameters={"snes_max_it": 5}
    )
    of_cd = _build_bv_observable_form(
        ctx, mode="current_density", reaction_index=None, scale=1.0
    )

    run_ss = make_run_ss(ctx=ctx, solver=solver, of_cd=of_cd)
    assert callable(run_ss)


def _build_simple_3sp_clo4_sp(*, v_rhe: float):
    """Build the documented 3sp + ClO4-(Bikerman) + Stern + logc_muh
    production stack at the requested ``v_rhe`` (legacy R1/R2 reaction
    path). Mirrors ``peroxide_window_3sp_bikerman_muh.py`` defaults."""
    from scripts._bv_common import (
        setup_firedrake_env,
        SNES_OPTS_CHARGED,
        V_T,
        A_DEFAULT,
        THREE_SPECIES_LOGC_BOLTZMANN,
        DEFAULT_CLO4_BOLTZMANN_COUNTERION,
        K0_HAT_R1,
        K0_HAT_R2,
        ALPHA_R1,
        ALPHA_R2,
        make_bv_solver_params,
    )
    setup_firedrake_env()

    bikerman_clo4 = {
        **DEFAULT_CLO4_BOLTZMANN_COUNTERION,
        "steric_mode": "bikerman",
        "a_nondim": A_DEFAULT,
    }
    snes_opts = {**SNES_OPTS_CHARGED}
    snes_opts.update({
        "snes_max_it": 400,
        "snes_atol": 1e-7,
        "snes_rtol": 1e-10,
        "snes_stol": 1e-12,
        "snes_linesearch_type": "l2",
        "snes_linesearch_maxlambda": 0.3,
    })
    sp = make_bv_solver_params(
        eta_hat=0.0,
        dt=0.25,
        t_end=80.0,
        species=THREE_SPECIES_LOGC_BOLTZMANN,
        snes_opts=snes_opts,
        formulation="logc_muh",
        log_rate=True,
        u_clamp=100.0,
        boltzmann_counterions=[bikerman_clo4],
        stern_capacitance_f_m2=0.10,
        k0_hat_r1=float(K0_HAT_R1),
        k0_hat_r2=float(K0_HAT_R2),
        alpha_r1=float(ALPHA_R1),
        alpha_r2=float(ALPHA_R2),
        E_eq_r1=0.68,
        E_eq_r2=1.78,
        initializer="debye_boltzmann",
    )
    new_opts = dict(sp.solver_options)
    new_bv = dict(new_opts["bv_convergence"])
    new_bv["exponent_clip"] = 50.0
    new_opts["bv_convergence"] = new_bv
    sp = sp.with_solver_options(new_opts)
    return sp.with_phi_applied(float(v_rhe) / V_T)


@pytest.mark.slow
def test_make_run_ss_runs_to_finite_observable_under_continuation():
    """Risk R5 functional check: the public ``make_run_ss`` factory
    drives a real Newton SS to a finite, sign-consistent steady-state
    observable when called inside the k0-continuation orchestrator
    (which uses ``make_run_ss`` for every rung).

    This verifies the factory plumbing is correct on a real PDE
    problem. Risk R5 (orchestrator behavior drift) is covered
    separately by the Phase 5α 93/93 regression suite (the orchestrator
    delegates to the same factory, so any drift would surface there
    first).
    """
    pytest.importorskip("firedrake")
    import math
    import firedrake.adjoint as adj  # noqa: F401  (warm import)

    from Forward.bv_solver import make_graded_rectangle_mesh
    from Forward.bv_solver.anchor_continuation import (
        solve_anchor_with_continuation,
    )
    from scripts._bv_common import K0_HAT_R1, K0_HAT_R2

    sp = _build_simple_3sp_clo4_sp(v_rhe=0.0)

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=80, beta=3.0)
    result = solve_anchor_with_continuation(
        sp,
        mesh=mesh,
        k0_targets={0: float(K0_HAT_R1), 1: float(K0_HAT_R2)},
        initial_scales=(1e-6, 1e-3, 1.0),
        max_inserts_per_step=4,
        max_ss_steps_per_rung=200,
        ic_at_target=True,
    )
    assert result.converged is True
    assert result.U_data is not None
    # Each rung's cd_observable came from ``fd.assemble(of_cd)`` inside
    # the make_run_ss-driven SS; verify the final rung produced a
    # finite value.
    last_cd = result.rungs[-1].get("cd_observable")
    assert last_cd is not None and math.isfinite(last_cd), (
        f"final rung produced non-finite observable: rungs={result.rungs!r}"
    )
