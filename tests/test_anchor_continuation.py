"""Tests for ``Forward.bv_solver.anchor_continuation`` (Phase 5γ MVP).

Layout mirrors ``tests/test_picard_ic_helpers.py``: pure-Python tests
at the top (no Firedrake imports), with the Firedrake integration
smoke at the bottom under ``@pytest.mark.slow``.

Coverage:

  1. :func:`set_reaction_k0_model` / :func:`get_reaction_k0_model`:
     double-mutation contract, validation errors, no aliasing.
  2. :class:`AdaptiveLadder`: validation, normal progression, failure
     inserts geometric mean, max-inserts cap, history, no aliasing.
  3. :func:`solve_anchor_with_continuation`: smoke-only at
     ``V_RHE = 0`` with ``initial_scales = (1e-6, 1e-3, 1.0)``.
"""
from __future__ import annotations

import dataclasses
import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from Forward.bv_solver.anchor_continuation import (
    AdaptiveLadder,
    AnchorContinuationResult,
    LadderExhausted,
    NON_PETSC_KEYS,
    PreconvergedAnchor,
    extract_preconverged_anchor,
    get_reaction_k0_model,
    set_reaction_k0_model,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_ctx(n_reactions: int = 2, k0_initial: float = 1.0) -> dict:
    """Build a stand-in for a Firedrake context with the two layers
    that ``set_reaction_k0_model`` mutates."""
    rxns = [
        {"k0_model": float(k0_initial), "alpha": 0.5, "n_electrons": 2}
        for _ in range(n_reactions)
    ]
    funcs = [MagicMock(name=f"bv_k0_func_{j}") for j in range(n_reactions)]
    return {
        "nondim": {"bv_reactions": rxns},
        "bv_k0_funcs": funcs,
    }


# ---------------------------------------------------------------------------
# Section 1 -- set_reaction_k0_model / get_reaction_k0_model
# ---------------------------------------------------------------------------

def test_set_k0_mutates_both_dict_and_function():
    """The dict and the FE Function must both be updated. Picard reads
    one, the FE residual reads the other; they must agree."""
    ctx = _make_fake_ctx(n_reactions=2, k0_initial=1.0)
    set_reaction_k0_model(ctx, 0, 0.123)
    assert ctx["nondim"]["bv_reactions"][0]["k0_model"] == pytest.approx(0.123)
    ctx["bv_k0_funcs"][0].assign.assert_called_once_with(0.123)
    # Other reaction unchanged.
    assert ctx["nondim"]["bv_reactions"][1]["k0_model"] == pytest.approx(1.0)
    ctx["bv_k0_funcs"][1].assign.assert_not_called()


def test_set_k0_round_trip():
    ctx = _make_fake_ctx(n_reactions=2, k0_initial=1.0)
    set_reaction_k0_model(ctx, 1, 7.5e-4)
    assert get_reaction_k0_model(ctx, 1) == pytest.approx(7.5e-4)


def test_set_k0_zero_raises():
    ctx = _make_fake_ctx()
    with pytest.raises(ValueError, match="must be > 0"):
        set_reaction_k0_model(ctx, 0, 0.0)


def test_set_k0_negative_raises():
    ctx = _make_fake_ctx()
    with pytest.raises(ValueError, match="must be > 0"):
        set_reaction_k0_model(ctx, 0, -1.0)


def test_set_k0_index_out_of_range_raises():
    ctx = _make_fake_ctx(n_reactions=2)
    with pytest.raises(IndexError):
        set_reaction_k0_model(ctx, 2, 1e-3)
    with pytest.raises(IndexError):
        set_reaction_k0_model(ctx, -1, 1e-3)


def test_set_k0_does_not_alias_caller_dict():
    """If a caller stashed a reference to the old per-reaction dict,
    that reference should be unaffected by subsequent ``set_*`` calls
    (so callers can compare before/after states without surprises)."""
    ctx = _make_fake_ctx(n_reactions=2, k0_initial=1.0)
    captured = ctx["nondim"]["bv_reactions"][0]
    assert captured["k0_model"] == pytest.approx(1.0)

    set_reaction_k0_model(ctx, 0, 0.5)

    # Captured dict should still see the OLD value.
    assert captured["k0_model"] == pytest.approx(1.0)
    # Live ctx sees the NEW value.
    assert ctx["nondim"]["bv_reactions"][0]["k0_model"] == pytest.approx(0.5)


def test_set_k0_repeated_calls_use_latest_value():
    """Multiple ramp steps should chain: each call sees the latest dict
    state, FE Function gets each new value in turn."""
    ctx = _make_fake_ctx(n_reactions=1, k0_initial=1.0)
    for k in (1e-9, 1e-6, 1e-3, 1.0):
        set_reaction_k0_model(ctx, 0, k)
        assert get_reaction_k0_model(ctx, 0) == pytest.approx(k)
    # FE Function should have been assigned 4 times in order.
    calls = ctx["bv_k0_funcs"][0].assign.call_args_list
    assert [c.args[0] for c in calls] == [1e-9, 1e-6, 1e-3, 1.0]


# ---------------------------------------------------------------------------
# Section 2 -- AdaptiveLadder
# ---------------------------------------------------------------------------

def test_adaptive_ladder_initial_scales_validation():
    with pytest.raises(ValueError, match="non-empty"):
        AdaptiveLadder(initial_scales=())
    with pytest.raises(ValueError, match="must all be > 0"):
        AdaptiveLadder(initial_scales=(0.0, 1.0))
    with pytest.raises(ValueError, match="must all be > 0"):
        AdaptiveLadder(initial_scales=(-1.0, 1.0))
    with pytest.raises(ValueError, match="strictly monotonic"):
        AdaptiveLadder(initial_scales=(1e-3, 1e-3, 1.0))
    with pytest.raises(ValueError, match="strictly monotonic"):
        AdaptiveLadder(initial_scales=(1e-3, 1e-6, 1.0))
    with pytest.raises(ValueError, match="must end at 1.0"):
        AdaptiveLadder(initial_scales=(1e-3, 0.5))


def test_adaptive_ladder_max_inserts_validation():
    with pytest.raises(ValueError, match="must be >= 0"):
        AdaptiveLadder(initial_scales=(1e-3, 1.0), max_inserts_per_step=-1)


def test_adaptive_ladder_normal_progression():
    L = AdaptiveLadder(initial_scales=(1e-6, 1e-3, 1.0))
    assert not L.is_done()
    assert L.current_scale == pytest.approx(1e-6)
    assert L.previous_scale is None

    L.record_success()
    assert L.current_scale == pytest.approx(1e-3)
    assert L.previous_scale == pytest.approx(1e-6)

    L.record_success()
    assert L.current_scale == pytest.approx(1.0)

    L.record_success()
    assert L.is_done()
    with pytest.raises(IndexError):
        _ = L.current_scale


def test_adaptive_ladder_failure_inserts_geometric_mean():
    L = AdaptiveLadder(initial_scales=(1e-3, 1.0), max_inserts_per_step=4)
    L.record_success()                  # 1e-3 ok
    assert L.current_scale == pytest.approx(1.0)

    inserted = L.record_failure_and_insert()
    assert inserted is True
    expected_mid = math.sqrt(1e-3 * 1.0)  # ≈ 0.0316
    assert L.current_scale == pytest.approx(expected_mid)
    # 1.0 should still be in the planned sequence after the midpoint.
    planned = L.planned_scales()
    assert planned[-1] == pytest.approx(1.0)
    assert planned[-2] == pytest.approx(expected_mid)


def test_adaptive_ladder_no_previous_scale_failure_returns_false():
    """Failing the FIRST rung has no `previous_scale` to interpolate
    from; record_failure_and_insert should return False rather than
    invent a smaller floor."""
    L = AdaptiveLadder(initial_scales=(1e-12, 1.0), max_inserts_per_step=4)
    inserted = L.record_failure_and_insert()
    assert inserted is False


def test_adaptive_ladder_max_inserts_exhausted():
    L = AdaptiveLadder(initial_scales=(1e-3, 1.0), max_inserts_per_step=2)
    L.record_success()
    # Three consecutive failures: first two insert; third returns False.
    assert L.record_failure_and_insert() is True
    assert L.record_failure_and_insert() is True
    assert L.record_failure_and_insert() is False


def test_adaptive_ladder_success_resets_insert_counter():
    """After a success, the budget for the *next* rung's inserts should
    be reset to ``max_inserts_per_step``."""
    L = AdaptiveLadder(initial_scales=(1e-3, 1.0), max_inserts_per_step=2)
    L.record_success()
    L.record_failure_and_insert()              # insert #1 between 1e-3 and 1.0
    L.record_success()                         # midpoint succeeds
    # Now at original scale=1.0; failure here should be allowed to
    # insert two midpoints (between previous-success 0.0316 and 1.0).
    assert L.record_failure_and_insert() is True
    assert L.record_failure_and_insert() is True
    assert L.record_failure_and_insert() is False


def test_adaptive_ladder_history_records_ok_and_fail():
    L = AdaptiveLadder(initial_scales=(1e-3, 1.0), max_inserts_per_step=4)
    L.record_success()
    L.record_failure_and_insert()    # logs (1.0, "fail"), inserts midpoint
    L.record_success()               # logs (midpoint, "ok")
    L.record_success()               # logs (1.0, "ok")
    h = L.history()
    assert len(h) == 4
    assert h[0] == (pytest.approx(1e-3), "ok")
    assert h[1] == (pytest.approx(1.0), "fail")
    assert h[2][1] == "ok"            # midpoint
    assert h[3] == (pytest.approx(1.0), "ok")


def test_adaptive_ladder_history_has_no_aliasing():
    L = AdaptiveLadder(initial_scales=(1e-3, 1.0))
    L.record_success()
    h = L.history()
    h.append(("garbage", "ok"))      # mutate caller's copy
    h2 = L.history()
    assert ("garbage", "ok") not in h2


def test_adaptive_ladder_planned_scales_no_aliasing():
    L = AdaptiveLadder(initial_scales=(1e-3, 1.0))
    p = L.planned_scales()
    p.clear()
    assert L.planned_scales() == [pytest.approx(1e-3), pytest.approx(1.0)]


# ---------------------------------------------------------------------------
# Section 2b -- PreconvergedAnchor + extract_preconverged_anchor
# ---------------------------------------------------------------------------

def _make_anchor(
    *,
    phi_applied_eta: float = 0.0,
    n_subspaces: int = 2,
    k0_targets: tuple = ((0, 1.0), (1, 2.5)),
    mesh_dof_count: int = 100,
    ladder_history: tuple = ((1e-3, "ok"), (1.0, "ok")),
) -> PreconvergedAnchor:
    return PreconvergedAnchor(
        phi_applied_eta=phi_applied_eta,
        U_snapshot=tuple(np.arange(8, dtype=float) for _ in range(n_subspaces)),
        k0_targets=k0_targets,
        mesh_dof_count=mesh_dof_count,
        ladder_history=ladder_history,
    )


def _make_converged_result(
    *,
    n_subspaces: int = 2,
    ladder_history=None,
) -> AnchorContinuationResult:
    if ladder_history is None:
        ladder_history = [(1e-3, "ok"), (1.0, "ok")]
    return AnchorContinuationResult(
        converged=True,
        U_data=tuple(np.arange(8, dtype=float) for _ in range(n_subspaces)),
        ladder_history=list(ladder_history),
        rungs=[],
        ctx={},
    )


def test_preconverged_anchor_frozen():
    a = _make_anchor()
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.phi_applied_eta = 0.5  # type: ignore[misc]


def test_preconverged_anchor_validates_mesh_dof_count():
    with pytest.raises(ValueError, match="mesh_dof_count"):
        _make_anchor(mesh_dof_count=0)
    with pytest.raises(ValueError, match="mesh_dof_count"):
        _make_anchor(mesh_dof_count=-5)


def test_preconverged_anchor_validates_k0_positive():
    with pytest.raises(ValueError, match="must be > 0"):
        _make_anchor(k0_targets=((0, 0.0),))
    with pytest.raises(ValueError, match="must be > 0"):
        _make_anchor(k0_targets=((0, -1.0),))


def test_preconverged_anchor_validates_phi_finite():
    with pytest.raises(ValueError, match="phi_applied_eta"):
        _make_anchor(phi_applied_eta=float("nan"))
    with pytest.raises(ValueError, match="phi_applied_eta"):
        _make_anchor(phi_applied_eta=float("inf"))


def test_preconverged_anchor_k0_targets_dict_round_trip():
    a = _make_anchor(k0_targets=((0, 1.0), (1, 2.5)))
    d = a.k0_targets_dict()
    assert d == {0: 1.0, 1: 2.5}
    # Mutating the returned dict has no effect on the frozen tuple.
    d[0] = 999.0
    assert a.k0_targets_dict() == {0: 1.0, 1: 2.5}


def test_preconverged_anchor_validates_u_snapshot_non_empty():
    with pytest.raises(ValueError, match="U_snapshot"):
        PreconvergedAnchor(
            phi_applied_eta=0.0,
            U_snapshot=(),
            k0_targets=((0, 1.0),),
            mesh_dof_count=10,
            ladder_history=(),
        )


def test_preconverged_anchor_validates_u_snapshot_arrays():
    with pytest.raises(ValueError, match="numpy arrays"):
        PreconvergedAnchor(
            phi_applied_eta=0.0,
            U_snapshot=([1.0, 2.0],),  # list, not ndarray
            k0_targets=((0, 1.0),),
            mesh_dof_count=10,
            ladder_history=(),
        )


def test_extract_raises_on_unconverged_result():
    bad = AnchorContinuationResult(
        converged=False,
        U_data=None,
        ladder_history=[(1.0, "fail")],
        rungs=[],
        ctx={},
    )
    with pytest.raises(ValueError, match="result.converged is False"):
        extract_preconverged_anchor(
            bad,
            phi_applied_eta=0.0,
            k0_targets={0: 1.0},
            mesh_dof_count=10,
        )


def test_extract_raises_on_none_U_data():
    bad = AnchorContinuationResult(
        converged=True,
        U_data=None,
        ladder_history=[(1.0, "ok")],
        rungs=[],
        ctx={},
    )
    with pytest.raises(ValueError, match="U_data is None"):
        extract_preconverged_anchor(
            bad,
            phi_applied_eta=0.0,
            k0_targets={0: 1.0},
            mesh_dof_count=10,
        )


def test_extract_freezes_k0_targets():
    """Caller mutates input dict afterwards; anchor unchanged."""
    result = _make_converged_result()
    targets = {0: 1.0, 1: 2.5}
    a = extract_preconverged_anchor(
        result,
        phi_applied_eta=0.0,
        k0_targets=targets,
        mesh_dof_count=8,
    )
    targets[0] = 999.0
    targets[2] = 3.0
    assert a.k0_targets_dict() == {0: 1.0, 1: 2.5}


def test_extract_copies_U_snapshot():
    """Mutating result.U_data afterwards must not affect anchor."""
    result = _make_converged_result()
    a = extract_preconverged_anchor(
        result,
        phi_applied_eta=0.0,
        k0_targets={0: 1.0},
        mesh_dof_count=8,
    )
    # Mutate the original arrays in U_data.
    for arr in result.U_data:
        arr[:] = -42.0
    # Anchor's snapshot is independent.
    for arr in a.U_snapshot:
        assert np.allclose(arr, np.arange(8, dtype=float))


def test_extract_freezes_ladder_history():
    """Mutate result.ladder_history list; anchor's tuple unchanged."""
    result = _make_converged_result(
        ladder_history=[(1e-3, "ok"), (1.0, "ok")],
    )
    a = extract_preconverged_anchor(
        result,
        phi_applied_eta=0.0,
        k0_targets={0: 1.0},
        mesh_dof_count=8,
    )
    result.ladder_history.append((999.0, "garbage"))
    assert isinstance(a.ladder_history, tuple)
    assert (999.0, "garbage") not in a.ladder_history
    assert a.ladder_history == ((1e-3, "ok"), (1.0, "ok"))


def test_extract_sorts_k0_targets_by_index():
    """Output ordering of k0_targets is deterministic (sorted by index)."""
    result = _make_converged_result()
    a = extract_preconverged_anchor(
        result,
        phi_applied_eta=0.0,
        k0_targets={1: 2.5, 0: 1.0},
        mesh_dof_count=8,
    )
    assert a.k0_targets == ((0, 1.0), (1, 2.5))


def test_preconverged_anchor_exported_from_package():
    from Forward.bv_solver import (
        PreconvergedAnchor as PA,
        extract_preconverged_anchor as ex,
    )
    assert PA is PreconvergedAnchor
    assert ex is extract_preconverged_anchor


# ---------------------------------------------------------------------------
# Section 3 -- module-level constants
# ---------------------------------------------------------------------------

def test_non_petsc_keys_contains_expected_metadata_keys():
    """Drift detector for Risk R2: if a new metadata-only key is added
    to bv_solver/dispatch, this constant must be updated to keep it out
    of the PETSc options database."""
    expected = {"bv_bc", "bv_convergence", "nondim", "robin_bc"}
    assert NON_PETSC_KEYS == frozenset(expected)


# ---------------------------------------------------------------------------
# Section 4 -- slow Firedrake integration tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_set_k0_real_firedrake_round_trip():
    """End-to-end check that, after building a real ctx via the muh
    dispatcher, ``set_reaction_k0_model`` actually advances the FE
    Function value (as PETSc sees it via ``dat.data_ro``) AND the
    metadata dict."""
    pytest.importorskip("firedrake")
    import firedrake as fd
    import firedrake.adjoint as adj  # noqa: F401  (warm imports)

    from Forward.bv_solver import (
        make_graded_rectangle_mesh,
        build_context_logc_muh,
        build_forms_logc_muh,
    )
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
        initializer="linear_phi",   # linear-phi is enough; we don't run Newton
    )
    sp = sp.with_phi_applied(0.0 / V_T)

    mesh = make_graded_rectangle_mesh(Nx=4, Ny=20, beta=3.0)
    ctx = build_context_logc_muh(sp, mesh=mesh)
    ctx = build_forms_logc_muh(ctx, sp)

    # Sanity: both layers report the build-time k0.
    assert get_reaction_k0_model(ctx, 0) == pytest.approx(float(K0_HAT_R2E))
    fn0_before = float(ctx["bv_k0_funcs"][0].dat.data_ro[0])
    assert fn0_before == pytest.approx(float(K0_HAT_R2E))

    # Mutate to a new value via the helper.
    K_NEW = 1.234e-5
    set_reaction_k0_model(ctx, 0, K_NEW)

    assert get_reaction_k0_model(ctx, 0) == pytest.approx(K_NEW)
    fn0_after = float(ctx["bv_k0_funcs"][0].dat.data_ro[0])
    assert fn0_after == pytest.approx(K_NEW)

    # Reaction 1 untouched.
    assert get_reaction_k0_model(ctx, 1) == pytest.approx(float(K0_HAT_R4E))
    fn1_after = float(ctx["bv_k0_funcs"][1].dat.data_ro[0])
    assert fn1_after == pytest.approx(float(K0_HAT_R4E))


@pytest.mark.slow
def test_solve_anchor_with_continuation_smoke():
    """Smoke test for the orchestrator at V_RHE=0 V (easy regime).

    Uses the documented 3sp + ClO4- (Bikerman) + Stern + logc_muh
    production stack from ``CLAUDE.md`` (single counterion, legacy
    R1/R2 path). V_RHE=0 V is the easy regime — this test only
    exercises the orchestration plumbing (build IC + ramp k0 down +
    walk back up + record success), NOT the gate-failure recovery at
    +0.55 V (M5's anchor smoke driver does that).
    """
    pytest.importorskip("firedrake")
    import firedrake.adjoint as adj  # noqa: F401  (warm imports)

    from Forward.bv_solver import make_graded_rectangle_mesh
    from Forward.bv_solver.anchor_continuation import (
        solve_anchor_with_continuation,
    )
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
    new_bv["exponent_clip"] = 100.0
    new_opts["bv_convergence"] = new_bv
    sp = sp.with_solver_options(new_opts)
    sp = sp.with_phi_applied(0.0 / V_T)

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
    assert result.ladder_history[-1][0] == pytest.approx(1.0)
    assert result.ladder_history[-1][1] == "ok"
    assert result.rungs[-1]["snes_converged"] is True
