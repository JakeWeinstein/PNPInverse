"""Solver integration tests for the steric-aware analytic Boltzmann closure.

The closure is wired in ``Forward/bv_solver/forms_logc.py`` via
``build_steric_boltzmann_expressions`` and gated on
``boltzmann_counterions[j].steric_mode='bikerman'``.  Math is pinned in
``tests/test_steric_boltzmann_closure_algebra.py`` (fast, no Firedrake).

This file holds the slow Firedrake-backed gates:

* ``test_ideal_path_byte_identical`` — default ``steric_mode='ideal'``
  on the production no-Stern stack at V_RHE=+0.66 V must match the
  refreshed snapshot baseline in ``tests/test_stern_no_stern_snapshot.py``
  to ``rel_tol=1e-6``.  Confirms the wiring landing didn't perturb the
  legacy code path.
* ``test_bikerman_smoke_v0p3`` — 3sp + bikerman counterion at
  V_RHE=+0.3 V converges via the C+D orchestrator and the surface
  c_ClO4 stays bounded by the Bikerman cap 1/a_b.

The double-counting validator is exercised via a fast unit test (it
runs the validation branch before constructing UFL).

Marked ``slow`` for the solver-driven cases.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from conftest import skip_without_firedrake


# ---------------------------------------------------------------------------
# Constants mirrored from tests/test_stern_no_stern_snapshot.py:53-56
# ---------------------------------------------------------------------------

BASELINE_CD_MA_CM2 = 1.2968453558282709e-08
BASELINE_PC_MA_CM2 = 1.2969358369725412e-08
BASELINE_V_RHE = 0.66
REL_TOL = 1e-6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solve_3sp_grid(v_rhe_grid, *, counterion_entry, exponent_clip: float = 100.0,
                    mesh_ny: int = 100, initializer: str | None = None):
    """Solve a V_RHE grid through the C+D orchestrator with the
    specified analytic-Boltzmann counterion entry.  Returns
    ``(cd_array, pc_array, fields_per_v_list, converged_per_v_list)``.

    Mirrors the wiring used by ``tests/test_solver_equivalence.py``: no
    explicit ``initializer`` argument (lets the factory use its default
    ``linear_phi``), Ny=100 to keep the test budget reasonable, full
    multi-V grid so the warm-walk fallback has cold-success anchors.
    """
    from scripts._bv_common import (
        setup_firedrake_env,
        V_T, I_SCALE,
        K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2,
        THREE_SPECIES_LOGC_BOLTZMANN,
        SNES_OPTS_CHARGED,
        make_bv_solver_params,
    )
    setup_firedrake_env()

    import firedrake as fd
    import firedrake.adjoint as adj
    from Forward.bv_solver import (
        make_graded_rectangle_mesh,
        solve_grid_per_voltage_cold_with_warm_fallback,
    )
    from Forward.bv_solver.observables import _build_bv_observable_form
    from Forward.bv_solver.diagnostics import surface_field_means

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=int(mesh_ny), beta=3.0)

    snes_opts = {**SNES_OPTS_CHARGED}
    snes_opts.update({
        "snes_max_it":               400,
        "snes_atol":                 1e-7,
        "snes_rtol":                 1e-10,
        "snes_stol":                 1e-12,
        "snes_linesearch_type":      "l2",
        "snes_linesearch_maxlambda": 0.3,
        "snes_divergence_tolerance": 1e10,
    })

    factory_kwargs = dict(
        eta_hat=0.0, dt=0.25, t_end=80.0,
        species=THREE_SPECIES_LOGC_BOLTZMANN,
        snes_opts=snes_opts,
        formulation="logc", log_rate=True,
        u_clamp=100.0,
        boltzmann_counterions=[counterion_entry],
        k0_hat_r1=K0_HAT_R1, k0_hat_r2=K0_HAT_R2,
        alpha_r1=ALPHA_R1, alpha_r2=ALPHA_R2,
        E_eq_r1=0.68, E_eq_r2=1.78,
    )
    if initializer is not None:
        factory_kwargs["initializer"] = initializer
    sp = make_bv_solver_params(**factory_kwargs)
    new_opts = dict(sp.solver_options)
    new_bv = dict(new_opts["bv_convergence"])
    new_bv["exponent_clip"] = float(exponent_clip)
    new_opts["bv_convergence"] = new_bv
    sp = sp.with_solver_options(new_opts)

    NV = len(v_rhe_grid)
    cd_arr = np.full(NV, np.nan)
    pc_arr = np.full(NV, np.nan)
    fields_per_v: list[dict] = [{} for _ in range(NV)]

    def _grab(orig_idx, _phi_eta, ctx):
        f_cd = _build_bv_observable_form(
            ctx, mode="current_density", reaction_index=None, scale=-I_SCALE)
        f_pc = _build_bv_observable_form(
            ctx, mode="peroxide_current", reaction_index=None, scale=-I_SCALE)
        cd_arr[orig_idx] = float(fd.assemble(f_cd))
        pc_arr[orig_idx] = float(fd.assemble(f_pc))
        fields_per_v[orig_idx] = dict(surface_field_means(ctx))

    phi_hat_grid = np.array(v_rhe_grid, dtype=float) / V_T
    with adj.stop_annotating():
        result = solve_grid_per_voltage_cold_with_warm_fallback(
            sp,
            phi_applied_values=phi_hat_grid,
            mesh=mesh,
            max_z_steps=20,
            n_substeps_warm=4,
            bisect_depth_warm=3,
            per_point_callback=_grab,
        )

    converged_per_v = [bool(result.points[i].converged) for i in range(NV)]
    return cd_arr, pc_arr, fields_per_v, converged_per_v


def _solve_no_stern_at_v_for_baseline(v_rhe: float, *, counterion_entry,
                                      exponent_clip: float = 100.0):
    """Single-V cold solve via the production no-Stern stack with
    ``debye_boltzmann`` initializer at Ny=200.  Used for the byte-
    identity test against the existing snapshot at V_RHE=+0.66 V (where
    the direct z=1 path converges from the analytical IC).  Returns
    ``(cd, pc, fields, converged)``.
    """
    from scripts._bv_common import (
        setup_firedrake_env,
        V_T, I_SCALE,
        K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2,
        THREE_SPECIES_LOGC_BOLTZMANN,
        SNES_OPTS_CHARGED,
        make_bv_solver_params,
    )
    setup_firedrake_env()

    import firedrake as fd
    import firedrake.adjoint as adj
    from Forward.bv_solver import (
        make_graded_rectangle_mesh,
        solve_grid_per_voltage_cold_with_warm_fallback,
    )
    from Forward.bv_solver.observables import _build_bv_observable_form
    from Forward.bv_solver.diagnostics import surface_field_means

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)

    snes_opts = {**SNES_OPTS_CHARGED}
    snes_opts.update({
        "snes_max_it":               400,
        "snes_atol":                 1e-7,
        "snes_rtol":                 1e-10,
        "snes_stol":                 1e-12,
        "snes_linesearch_type":      "l2",
        "snes_linesearch_maxlambda": 0.3,
        "snes_divergence_tolerance": 1e10,
    })

    sp = make_bv_solver_params(
        eta_hat=0.0, dt=0.25, t_end=80.0,
        species=THREE_SPECIES_LOGC_BOLTZMANN,
        snes_opts=snes_opts,
        formulation="logc", log_rate=True,
        boltzmann_counterions=[counterion_entry],
        k0_hat_r1=K0_HAT_R1, k0_hat_r2=K0_HAT_R2,
        alpha_r1=ALPHA_R1, alpha_r2=ALPHA_R2,
        E_eq_r1=0.68, E_eq_r2=1.78,
        initializer="debye_boltzmann",
    )
    new_opts = dict(sp.solver_options)
    new_bv = dict(new_opts["bv_convergence"])
    new_bv["exponent_clip"] = float(exponent_clip)
    new_opts["bv_convergence"] = new_bv
    sp = sp.with_solver_options(new_opts)

    cd_arr = np.full(1, np.nan)
    pc_arr = np.full(1, np.nan)
    fields_holder: list[dict] = [{}]

    def _grab(orig_idx, _phi_eta, ctx):
        f_cd = _build_bv_observable_form(
            ctx, mode="current_density", reaction_index=None, scale=-I_SCALE)
        f_pc = _build_bv_observable_form(
            ctx, mode="peroxide_current", reaction_index=None, scale=-I_SCALE)
        cd_arr[orig_idx] = float(fd.assemble(f_cd))
        pc_arr[orig_idx] = float(fd.assemble(f_pc))
        fields_holder[0] = dict(surface_field_means(ctx))

    phi_hat_grid = np.array([v_rhe]) / V_T
    with adj.stop_annotating():
        result = solve_grid_per_voltage_cold_with_warm_fallback(
            sp,
            phi_applied_values=phi_hat_grid,
            mesh=mesh,
            max_z_steps=20,
            n_substeps_warm=8,
            bisect_depth_warm=5,
            per_point_callback=_grab,
        )
    return (
        float(cd_arr[0]),
        float(pc_arr[0]),
        fields_holder[0],
        bool(result.points[0].converged),
    )


# ---------------------------------------------------------------------------
# Fast unit test — double-counting validator
# ---------------------------------------------------------------------------

@skip_without_firedrake
def test_double_counting_rejected():
    """A bikerman counterion config that duplicates a dynamic species'
    (z, c_bulk) within ``rel_tol`` raises ``ValueError`` from the
    validation branch of ``build_steric_boltzmann_expressions`` before
    any UFL is constructed.

    No solver run; only the validation pre-check executes.
    """
    from Forward.bv_solver.boltzmann import build_steric_boltzmann_expressions

    bad_entry = {
        "z": -1,
        "c_bulk_nondim": 0.2,
        "phi_clamp": 50.0,
        "steric_mode": "bikerman",
        "a_nondim": 0.01,
    }
    params = {"bv_bc": {"boltzmann_counterions": [bad_entry]}}

    # Dynamic species that include (z=-1, c_bulk=0.2) — exact duplicate.
    z_dyn = [0, 0, 1, -1]
    c0_dyn = [1.0, 1e-4, 0.2, 0.2]
    a_dyn_floats = [0.01, 0.01, 0.01, 0.01]

    with pytest.raises(ValueError, match="duplicates dynamic species"):
        build_steric_boltzmann_expressions(
            ctx={},
            params=params,
            ci=[None] * 4,             # never touched — validation fires first
            a_dyn_funcs=[None] * 4,    # never touched
            a_dyn_floats=a_dyn_floats,
            c0_dyn=c0_dyn,
            z_dyn=z_dyn,
            phi=None,                  # never touched
            R_space=None,              # never touched
        )


@skip_without_firedrake
def test_theta_b_negative_rejected():
    """If the bikerman counterion + dynamic species overpack the bulk
    lattice, ``theta_b <= 0`` raises ``ValueError`` with a descriptive
    message naming the offending fractions."""
    from Forward.bv_solver.boltzmann import build_steric_boltzmann_expressions

    # Pathological: a_b=1.0 with c_b=0.2 and dynamic species also at
    # high a*c sums.  theta_b = 1 - 1.0 - 0.2 = -0.2 < 0.
    entry = {
        "z": -1,
        "c_bulk_nondim": 0.2,
        "phi_clamp": 50.0,
        "steric_mode": "bikerman",
        "a_nondim": 1.0,
    }
    params = {"bv_bc": {"boltzmann_counterions": [entry]}}
    a_dyn_floats = [1.0, 1.0, 1.0]
    c0_dyn = [1.0, 0.0, 0.2]
    z_dyn = [0, 0, 1]

    with pytest.raises(ValueError, match=r"theta_b > 0"):
        build_steric_boltzmann_expressions(
            ctx={}, params=params,
            ci=[None] * 3, a_dyn_funcs=[None] * 3,
            a_dyn_floats=a_dyn_floats,
            c0_dyn=c0_dyn, z_dyn=z_dyn,
            phi=None, R_space=None,
        )


@skip_without_firedrake
def test_multi_bikerman_double_counting_rejected():
    """Multi-counterion is now SUPPORTED via the shared-theta closure
    (plan §2.1).  Two bikerman entries with identical (z, c_bulk) still
    raise ``ValueError`` from the within-bikerman double-counting guard."""
    from Forward.bv_solver.boltzmann import build_steric_boltzmann_expressions

    e1 = {"z": -1, "c_bulk_nondim": 0.2, "phi_clamp": 50.0,
          "steric_mode": "bikerman", "a_nondim": 0.01}
    e2 = {"z": -1, "c_bulk_nondim": 0.2, "phi_clamp": 50.0,
          "steric_mode": "bikerman", "a_nondim": 0.005}
    params = {"bv_bc": {"boltzmann_counterions": [e1, e2]}}

    with pytest.raises(ValueError, match="duplicate"):
        build_steric_boltzmann_expressions(
            ctx={}, params=params,
            ci=[None] * 3, a_dyn_funcs=[None] * 3,
            a_dyn_floats=[0.01, 0.01, 0.01],
            c0_dyn=[1.0, 0.0, 0.2],
            z_dyn=[0, 0, 1],
            phi=None, R_space=None,
        )


@skip_without_firedrake
def test_no_bikerman_returns_empty_list():
    """If no entry has steric_mode='bikerman', the helper returns an
    empty list and the caller falls back to the legacy ideal path."""
    from Forward.bv_solver.boltzmann import build_steric_boltzmann_expressions

    params = {"bv_bc": {"boltzmann_counterions": [
        {"z": -1, "c_bulk_nondim": 0.2, "phi_clamp": 50.0},  # legacy default
    ]}}
    out = build_steric_boltzmann_expressions(
        ctx={}, params=params,
        ci=[None] * 3, a_dyn_funcs=[None] * 3,
        a_dyn_floats=[0.01, 0.01, 0.01],
        c0_dyn=[1.0, 0.0, 0.2],
        z_dyn=[0, 0, 1],
        phi=None, R_space=None,
    )
    assert out == []


# ---------------------------------------------------------------------------
# Slow regression — ideal path byte-identical
# ---------------------------------------------------------------------------

@skip_without_firedrake
@pytest.mark.slow
@pytest.mark.xfail(
    reason="Baseline predates C_O2=0.5→1.2 migration (M3a.2.1, 2026-05-07); "
           "regenerate by running once and updating BASELINE_CD/PC.",
    strict=True,
)
def test_ideal_path_byte_identical():
    """The default ``steric_mode='ideal'`` path on the 3sp+Boltzmann
    production stack at V_RHE=+0.66 V must reproduce the snapshot
    baseline from ``tests/test_stern_no_stern_snapshot.py`` to
    ``rel_tol=1e-6``.  This pins that the closure-helper landing
    cannot perturb the legacy code path.
    """
    from scripts._bv_common import DEFAULT_CLO4_BOLTZMANN_COUNTERION
    # Single-V cold solve at V=+0.66 V via the same setup as
    # tests/test_stern_no_stern_snapshot.py: debye_boltzmann IC, Ny=200,
    # ideal counterion entry.  The wiring landing must not perturb this
    # path even by 1e-6.
    cd, pc, _, converged = _solve_no_stern_at_v_for_baseline(
        v_rhe=BASELINE_V_RHE,
        counterion_entry=DEFAULT_CLO4_BOLTZMANN_COUNTERION,
        exponent_clip=100.0,
    )
    assert converged, (
        f"ideal-path solve at V_RHE={BASELINE_V_RHE} must converge"
    )
    assert cd == pytest.approx(BASELINE_CD_MA_CM2, rel=REL_TOL), (
        f"CD drifted from snapshot baseline: got {cd!r}, "
        f"expected {BASELINE_CD_MA_CM2!r} (rel_tol={REL_TOL}).  "
        f"The build_steric_boltzmann_expressions helper must be a no-op "
        f"when no counterion entry has steric_mode='bikerman'."
    )
    assert pc == pytest.approx(BASELINE_PC_MA_CM2, rel=REL_TOL), (
        f"PC drifted from snapshot baseline: got {pc!r}, "
        f"expected {BASELINE_PC_MA_CM2!r} (rel_tol={REL_TOL})."
    )


# ---------------------------------------------------------------------------
# Slow smoke — bikerman path converges and saturates at +0.3 V
# ---------------------------------------------------------------------------

@skip_without_firedrake
@pytest.mark.slow
def test_bikerman_smoke_cathodic_window():
    """3sp + bikerman ClO4- counterion converges on the cathodic V_RHE
    window via the production C+D orchestrator (linear_phi initializer +
    z-ramp + warm-walk fallback) and produces finite CD/PC at every
    converged voltage.

    Mirrors the wiring used by ``tests/test_solver_equivalence.py``: a
    multi-V grid, default ``linear_phi`` initializer (no
    ``debye_boltzmann`` analytical-IC complications — that IC's
    ``phi_init = ln(H_outer/c_ClO4_bulk) + psi`` is built from
    ideal-Boltzmann electroneutrality and saturates ``c_steric`` at the
    surface, blowing up ``mu_steric`` on the bikerman path; see plan
    risk register §1).

    Pass criteria:
        1. At least one voltage on the grid converges.
        2. At every converged voltage, CD and PC are finite.
        3. The diagnostic ``surface_counterion_within_steric`` (from
           ``collect_diagnostics``) does not spuriously flag at
           cathodic V where ``c_steric -> 0``.

    Anodic saturation validation is deferred to Commit 6 (equivalence
    vs 4sp dynamic on the cathodic overlap window).  Higher anodic V
    needs a bikerman-aware IC, which is out of scope for this plan.
    """
    from scripts._bv_common import DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC
    v_grid = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.05, 0.1]
    cd_arr, pc_arr, fields_per_v, converged_per_v = _solve_3sp_grid(
        v_grid,
        counterion_entry=DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC,
        exponent_clip=50.0,  # production default; clip=100 overflows cathodic R2
    )

    n_converged = sum(converged_per_v)
    assert n_converged >= 1, (
        f"3sp + bikerman ClO4- closure must converge at >= 1 voltage on "
        f"the cathodic window {v_grid}.  Convergence map: "
        f"{list(zip(v_grid, converged_per_v))}.  All voltages failing "
        f"means either the wiring is broken or the orchestrator's "
        f"linear_phi z-ramp can't bootstrap from a uniform-bulk IC for "
        f"the bikerman residual — investigate."
    )

    # Every converged voltage must produce finite CD and PC.
    for i, (v, conv) in enumerate(zip(v_grid, converged_per_v)):
        if not conv:
            continue
        assert np.isfinite(cd_arr[i]), (
            f"V_RHE={v}: converged but CD={cd_arr[i]} is non-finite — "
            f"bikerman closure failed silently in the observable extraction."
        )
        assert np.isfinite(pc_arr[i]), f"V_RHE={v}: PC={pc_arr[i]} non-finite"


@skip_without_firedrake
@pytest.mark.slow
def test_diagnostics_reports_bikerman_mode_at_converged_voltage():
    """``collect_diagnostics`` must report ``c_counterion0_steric_mode='bikerman'``
    for a bikerman counterion entry, and the per-counterion surface
    concentration field must be evaluated via the closure (not the
    unbounded ``c_b * exp(-z*phi)`` ideal expression).

    Runs the same cathodic V grid as the smoke test but uses the
    converged state at the first successful voltage to exercise the
    bikerman branch in ``collect_diagnostics`` (Commit 5).
    """
    from scripts._bv_common import (
        setup_firedrake_env,
        V_T,
        K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2,
        THREE_SPECIES_LOGC_BOLTZMANN,
        DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC,
        SNES_OPTS_CHARGED, make_bv_solver_params,
    )
    setup_firedrake_env()

    import firedrake.adjoint as adj
    from Forward.bv_solver import (
        make_graded_rectangle_mesh,
        solve_grid_per_voltage_cold_with_warm_fallback,
    )
    from Forward.bv_solver.diagnostics import collect_diagnostics

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=100, beta=3.0)
    snes_opts = {**SNES_OPTS_CHARGED}
    snes_opts.update({
        "snes_max_it": 400, "snes_atol": 1e-7, "snes_rtol": 1e-10,
        "snes_stol": 1e-12, "snes_linesearch_type": "l2",
        "snes_linesearch_maxlambda": 0.3, "snes_divergence_tolerance": 1e10,
    })

    sp = make_bv_solver_params(
        eta_hat=0.0, dt=0.25, t_end=80.0,
        species=THREE_SPECIES_LOGC_BOLTZMANN, snes_opts=snes_opts,
        formulation="logc", log_rate=True, u_clamp=100.0,
        boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC],
        k0_hat_r1=K0_HAT_R1, k0_hat_r2=K0_HAT_R2,
        alpha_r1=ALPHA_R1, alpha_r2=ALPHA_R2,
        E_eq_r1=0.68, E_eq_r2=1.78,
    )
    new_opts = dict(sp.solver_options)
    new_bv = dict(new_opts["bv_convergence"])
    new_bv["exponent_clip"] = 50.0
    new_opts["bv_convergence"] = new_bv
    sp = sp.with_solver_options(new_opts)

    v_grid = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0]
    diag_holder: dict[int, dict] = {}

    def _grab(orig_idx, _phi_eta, ctx):
        diag_holder[orig_idx] = collect_diagnostics(
            ctx, phase="cold", params=sp[10],
        )

    phi_hat_grid = np.array(v_grid, dtype=float) / V_T
    with adj.stop_annotating():
        result = solve_grid_per_voltage_cold_with_warm_fallback(
            sp, phi_applied_values=phi_hat_grid, mesh=mesh,
            max_z_steps=20, n_substeps_warm=4, bisect_depth_warm=3,
            per_point_callback=_grab,
        )

    converged_idx = [i for i in range(len(v_grid))
                     if result.points[i].converged and i in diag_holder]
    assert converged_idx, (
        f"No converged voltage produced diagnostics; cannot test bikerman "
        f"diagnostics path."
    )
    diag = diag_holder[converged_idx[0]]
    assert "c_counterion0_steric_mode" in diag, (
        f"diagnostics dict missing c_counterion0_steric_mode field; keys={list(diag)}"
    )
    assert diag["c_counterion0_steric_mode"] == "bikerman"
    assert "c_counterion0_surface_mean" in diag
    c_surf = diag["c_counterion0_surface_mean"]
    assert np.isfinite(c_surf), f"c_counterion0_surface_mean non-finite: {c_surf}"
    # Closure cap (1-A_dyn)/a_b ~ 99 nondim; at cathodic V c_steric -> 0
    # so we expect 0 < c_surf < 1/a_b * 1.1 (slack for floating point).
    assert 0.0 <= c_surf < 1.1 / 0.01, (
        f"c_counterion0_surface_mean = {c_surf} outside physical bounds "
        f"[0, 1/a_b)."
    )
    assert diag.get("surface_counterion_within_steric") is True, (
        f"surface_counterion_within_steric should be True for the bikerman "
        f"closure at cathodic V (where c_steric ~ 0); got "
        f"{diag.get('surface_counterion_within_steric')}"
    )


# ---------------------------------------------------------------------------
# Slow equivalence — 3sp+steric-Boltzmann vs 4sp dynamic on cathodic window
# ---------------------------------------------------------------------------

def _run_production_solve_for_equiv(*, species, boltzmann_counterions,
                                    V_RHE_grid, mesh_ny=100):
    """Mirror tests/test_solver_equivalence.py:_run_production_solve so
    the bikerman vs 4sp equivalence is comparable to the existing
    ideal vs 4sp equivalence baseline."""
    from scripts._bv_common import (
        setup_firedrake_env,
        V_T, I_SCALE,
        K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2,
        SNES_OPTS_CHARGED,
        make_bv_solver_params,
    )
    setup_firedrake_env()

    import firedrake as fd
    import firedrake.adjoint as adj
    from Forward.bv_solver import (
        make_graded_rectangle_mesh,
        solve_grid_per_voltage_cold_with_warm_fallback,
    )
    from Forward.bv_solver.observables import _build_bv_observable_form

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=int(mesh_ny), beta=3.0)
    snes_opts = {**SNES_OPTS_CHARGED}
    snes_opts.update({
        "snes_max_it": 400, "snes_atol": 1e-7, "snes_rtol": 1e-10,
        "snes_stol": 1e-12, "snes_linesearch_type": "l2",
        "snes_linesearch_maxlambda": 0.3, "snes_divergence_tolerance": 1e10,
    })

    sp = make_bv_solver_params(
        eta_hat=0.0, dt=0.25, t_end=80.0,
        species=species, snes_opts=snes_opts,
        formulation="logc", log_rate=True, u_clamp=100.0,
        boltzmann_counterions=boltzmann_counterions,
        k0_hat_r1=K0_HAT_R1, k0_hat_r2=K0_HAT_R2,
        alpha_r1=ALPHA_R1, alpha_r2=ALPHA_R2,
        E_eq_r1=0.68, E_eq_r2=1.78,
    )
    # Force clip=50 (production for cathodic V; the factory default
    # raised to 100 on 2026-05-04 overflows R2 cathodic on this grid).
    new_opts = dict(sp.solver_options)
    new_bv = dict(new_opts["bv_convergence"])
    new_bv["exponent_clip"] = 50.0
    new_opts["bv_convergence"] = new_bv
    sp = sp.with_solver_options(new_opts)

    V_RHE_grid = np.asarray(V_RHE_grid, dtype=float)
    NV = len(V_RHE_grid)
    cd = np.full(NV, np.nan)
    pc = np.full(NV, np.nan)

    def _grab(orig_idx, _phi_eta, ctx):
        f_cd = _build_bv_observable_form(
            ctx, mode="current_density", reaction_index=None, scale=-I_SCALE)
        f_pc = _build_bv_observable_form(
            ctx, mode="peroxide_current", reaction_index=None, scale=-I_SCALE)
        cd[orig_idx] = float(fd.assemble(f_cd))
        pc[orig_idx] = float(fd.assemble(f_pc))

    phi_hat_grid = V_RHE_grid / V_T
    with adj.stop_annotating():
        solve_grid_per_voltage_cold_with_warm_fallback(
            sp, phi_applied_values=phi_hat_grid, mesh=mesh,
            max_z_steps=20, n_substeps_warm=4, bisect_depth_warm=3,
            per_point_callback=_grab,
        )
    return {"V_RHE": V_RHE_grid, "cd_mA_cm2": cd, "pc_mA_cm2": pc}


@skip_without_firedrake
@pytest.mark.slow
def test_steric_boltzmann_equiv_to_4sp_dynamic():
    """The 3sp + bikerman ClO4- closure must agree with the 4sp dynamic
    formulation on the cathodic V_RHE window where both converge.

    Per the handoff (``docs/steric_analytic_clo4_reduction_handoff.md``
    §"Physical equivalence to dynamic 4sp"), the closure is the
    steady-state algebraic reduction of the dynamic ClO4- NP equation
    under the production BCs, so the discrete observables should match
    within discretization-error noise — comparable to or better than
    the existing ideal-Boltzmann ↔ 4sp equivalence test.

    Tolerance mirrors ``tests/test_solver_equivalence.py:195-196``
    (REL_TOL=5e-3, ABS_FLOOR=1e-4 mA/cm^2 hybrid).
    """
    from scripts._bv_common import (
        THREE_SPECIES_LOGC_BOLTZMANN,
        FOUR_SPECIES_LOGC_DYNAMIC,
        DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC,
    )

    V_GRID = np.array([-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.05, 0.1])
    REL_TOL = 5e-3
    ABS_FLOOR = 1e-4  # mA/cm^2

    r3_steric = _run_production_solve_for_equiv(
        species=THREE_SPECIES_LOGC_BOLTZMANN,
        boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC],
        V_RHE_grid=V_GRID,
    )
    r4_dynamic = _run_production_solve_for_equiv(
        species=FOUR_SPECIES_LOGC_DYNAMIC,
        boltzmann_counterions=None,
        V_RHE_grid=V_GRID,
    )

    common = ~(np.isnan(r3_steric["cd_mA_cm2"]) | np.isnan(r4_dynamic["cd_mA_cm2"]))
    n_common = int(common.sum())
    assert n_common >= 2, (
        f"3sp+steric and 4sp dynamic must converge at >=2 common voltages; "
        f"got {n_common}.  3sp+steric NaN at: "
        f"{V_GRID[np.isnan(r3_steric['cd_mA_cm2'])].tolist()}; "
        f"4sp dynamic NaN at: "
        f"{V_GRID[np.isnan(r4_dynamic['cd_mA_cm2'])].tolist()}"
    )

    def _hybrid_err(a, b):
        denom = np.maximum(np.maximum(np.abs(a), np.abs(b)), ABS_FLOOR)
        return np.abs(a - b) / denom

    cd_err = _hybrid_err(
        r3_steric["cd_mA_cm2"][common], r4_dynamic["cd_mA_cm2"][common])
    pc_err = _hybrid_err(
        r3_steric["pc_mA_cm2"][common], r4_dynamic["pc_mA_cm2"][common])

    cd_max = float(cd_err.max())
    pc_max = float(pc_err.max())

    assert cd_max < REL_TOL, (
        f"3sp+steric vs 4sp dynamic CD hybrid err exceeds {REL_TOL}: "
        f"got {cd_max} on common-converged grid "
        f"V_RHE={V_GRID[common].tolist()}\n"
        f"  3sp+steric CD: {r3_steric['cd_mA_cm2'][common].tolist()}\n"
        f"  4sp dynamic CD: {r4_dynamic['cd_mA_cm2'][common].tolist()}"
    )
    assert pc_max < REL_TOL, (
        f"3sp+steric vs 4sp dynamic PC hybrid err exceeds {REL_TOL}: "
        f"got {pc_max} on common-converged grid "
        f"V_RHE={V_GRID[common].tolist()}\n"
        f"  3sp+steric PC: {r3_steric['pc_mA_cm2'][common].tolist()}\n"
        f"  4sp dynamic PC: {r4_dynamic['pc_mA_cm2'][common].tolist()}"
    )


# ---------------------------------------------------------------------------
# Slow muh-formulation parity — same closure must work in formulation="logc_muh"
# ---------------------------------------------------------------------------

@skip_without_firedrake
@pytest.mark.slow
def test_bikerman_smoke_muh():
    """Same physics as ``test_bikerman_smoke_cathodic_window`` but via
    the experimental ``formulation='logc_muh'`` (Phase 2 hybrid: H+
    stored as electrochemical potential mu_H = u_H + em*z_H*phi).

    The bikerman closure must wire identically into both backends.
    """
    from scripts._bv_common import (
        setup_firedrake_env, V_T, I_SCALE,
        K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2,
        THREE_SPECIES_LOGC_BOLTZMANN,
        DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC,
        SNES_OPTS_CHARGED, make_bv_solver_params,
    )
    setup_firedrake_env()

    import firedrake as fd
    import firedrake.adjoint as adj
    from Forward.bv_solver import (
        make_graded_rectangle_mesh,
        solve_grid_per_voltage_cold_with_warm_fallback,
    )
    from Forward.bv_solver.observables import _build_bv_observable_form

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=100, beta=3.0)
    snes_opts = {**SNES_OPTS_CHARGED}
    snes_opts.update({
        "snes_max_it": 400, "snes_atol": 1e-7, "snes_rtol": 1e-10,
        "snes_stol": 1e-12, "snes_linesearch_type": "l2",
        "snes_linesearch_maxlambda": 0.3, "snes_divergence_tolerance": 1e10,
    })

    sp = make_bv_solver_params(
        eta_hat=0.0, dt=0.25, t_end=80.0,
        species=THREE_SPECIES_LOGC_BOLTZMANN, snes_opts=snes_opts,
        formulation="logc_muh", log_rate=True, u_clamp=100.0,
        boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC],
        k0_hat_r1=K0_HAT_R1, k0_hat_r2=K0_HAT_R2,
        alpha_r1=ALPHA_R1, alpha_r2=ALPHA_R2,
        E_eq_r1=0.68, E_eq_r2=1.78,
    )
    new_opts = dict(sp.solver_options)
    new_bv = dict(new_opts["bv_convergence"])
    new_bv["exponent_clip"] = 50.0
    new_opts["bv_convergence"] = new_bv
    sp = sp.with_solver_options(new_opts)

    v_grid = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.05, 0.1]
    NV = len(v_grid)
    cd_arr = np.full(NV, np.nan)
    pc_arr = np.full(NV, np.nan)

    def _grab(orig_idx, _phi_eta, ctx):
        f_cd = _build_bv_observable_form(
            ctx, mode="current_density", reaction_index=None, scale=-I_SCALE)
        f_pc = _build_bv_observable_form(
            ctx, mode="peroxide_current", reaction_index=None, scale=-I_SCALE)
        cd_arr[orig_idx] = float(fd.assemble(f_cd))
        pc_arr[orig_idx] = float(fd.assemble(f_pc))

    phi_hat_grid = np.array(v_grid, dtype=float) / V_T
    with adj.stop_annotating():
        result = solve_grid_per_voltage_cold_with_warm_fallback(
            sp, phi_applied_values=phi_hat_grid, mesh=mesh,
            max_z_steps=20, n_substeps_warm=4, bisect_depth_warm=3,
            per_point_callback=_grab,
        )

    points = result.points
    converged_per_v = [bool(points[i].converged) for i in range(len(v_grid))]
    n_converged = sum(converged_per_v)
    assert n_converged >= 1, (
        f"3sp + bikerman ClO4- (muh) must converge at >=1 V on the "
        f"cathodic window {v_grid}.  Convergence: "
        f"{list(zip(v_grid, converged_per_v))}"
    )
    for i, conv in enumerate(converged_per_v):
        if not conv:
            continue
        assert np.isfinite(cd_arr[i]), f"V_RHE={v_grid[i]} (muh): CD non-finite"
        assert np.isfinite(pc_arr[i]), f"V_RHE={v_grid[i]} (muh): PC non-finite"
