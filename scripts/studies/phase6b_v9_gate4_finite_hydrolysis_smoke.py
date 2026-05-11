"""Phase 6β v9 Gate 4 — finite-rate cation-hydrolysis smoke study.

Couples the v9 architecture: full Singh 2016 SI Eq. (4) field-dependent
pKa + finite-rate hydrolysis kinetics + Γ_MOH outer Picard.  Anchors
at the Gate 2 anodic V=+0.55 V (well-conditioned) with kw ladder, then
warm-walks down to the cathodic Gate 4 target at λ=0 (must reproduce
Gate 2 baseline cd/pc), then ramps λ from 0 → 1 at the Gate 4 target
voltage.

**Production-style flow** (mirrors Gate 2 SUCCESS recipe):

  1. Build the 4sp K2SO4 stack with ``enable_cation_hydrolysis=True``,
     ``enable_water_ionization=True``, ``initializer="linear_phi"``.
  2. ``solve_anchor_with_continuation`` at V=+0.55 V with k0 ladder +
     Phase 6α Kw_eff ladder + ``lambda_hydrolysis=0`` (no cation
     hydrolysis at the anchor).
  3. ``extract_preconverged_anchor`` from the converged state.
  4. ``solve_grid_with_anchor`` over V_RHE ∈ [+0.55, +0.50, ..., -0.40] V
     at λ=0.  Verifies the Gate 2 baseline reproduces.
  5. At V=-0.40 V, run an inner λ ramp 0 → 0.25 → 0.50 → 0.75 → 1.0
     by repeated ``solve_anchor_with_continuation`` calls with
     ``lambda_hydrolysis_ladder``.

Sweep dimensions (R5#5 wording guard: Gate 4 verdict is
architecture-only, not physics):

* ``r_H_El_pm ∈ {180, 195, 200.98, 215, 250}`` pm — load-bearing
  Cu→carbon transferability calibration.
* ``C_S ∈ {0.05, 0.10, 0.20}`` F/m² — Stern sensitivity.
* ``k_des ∈ {1e3, 1e5, 1e7}`` 1/s — desorption sensitivity at λ=1.

Use ``--quick`` to run just the baseline pass (single combination)
for end-to-end smoke verification.

Outputs:

* ``StudyResults/phase6b_v9_gate4_smoke/iv_curve.json``
* ``StudyResults/phase6b_v9_gate4_smoke/diagnostics.json``
* ``StudyResults/phase6b_v9_gate4_smoke/sensitivity_grid.json``

Usage::

    cd /Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse
    source ../venv-firedrake/bin/activate
    export MPLCONFIGDIR=/tmp XDG_CACHE_HOME=/tmp PYOP2_CACHE_DIR=/tmp/pyop2 \\
           FIREDRAKE_TSFC_KERNEL_CACHE_DIR=/tmp/firedrake-tsfc OMP_NUM_THREADS=1
    python -u scripts/studies/phase6b_v9_gate4_finite_hydrolysis_smoke.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(line_buffering=True)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

V_ANCHOR = 0.55
V_RHE_PRIMARY_FULL = (
    0.55, 0.50, 0.40, 0.30, 0.20, 0.10, 0.00,
    -0.10, -0.20, -0.30, -0.40,
)
V_RHE_PRIMARY = V_RHE_PRIMARY_FULL                 # mutated in main() if --voltage
V_LAMBDA_TARGET = -0.40                            # mutated in main() if --voltage

K0_INITIAL_SCALES = (1e-12, 1e-9, 1e-6, 1e-3, 1.0)

LAMBDA_LADDER_DEFAULT = (0.0, 0.25, 0.50, 0.75, 1.0)

R_H_EL_SWEEP_PM = (180.0, 195.0, 200.98, 215.0, 250.0)
C_S_SWEEP_F_M2 = (0.05, 0.10, 0.20)
# K_DES sweep in nondim units.  Plan §4B mentioned (1e3, 1e5, 1e7) 1/s
# in physical units; here we sweep around the smoke baseline of 1.0
# nondim (the smaller scale that Picard converges on for the K2SO4
# stack at V=−0.40 V; the physics calibration is 6β.2 work).
K_DES_SWEEP_NONDIM = (1e-1, 1e0, 1e1)

# Phase B: k_hyd ladder (Phase 6β v9 post-Gate-4 plan §B).  Spans 6 orders
# of magnitude from the smoke baseline up to where Picard is expected
# to break.  Stop at first rung where Picard fails or packing > 0.95.
K_HYD_RAMP_NONDIM = (1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3)

R_H_EL_BASELINE_PM = 200.98       # Singh Cu prior
C_S_BASELINE_F_M2 = 0.10
K_DES_BASELINE_NONDIM = 1.0
K_HYD_BASELINE_NONDIM = 1e-3      # smoke baseline; tame so Picard converges
K_PROT_BASELINE_NONDIM = 1e-3     # smoke baseline; tame for Picard

# Smoke study uses Ny=80 mesh (Gate 4B verdict criterion).
MESH_NX = 8
MESH_NY = 80
MESH_BETA = 3.0
L_EFF_M = 16e-6
EXPONENT_CLIP = 100.0
U_CLAMP = 100.0
INITIALIZER = "linear_phi"
FORMULATION = "logc_muh"
STERN_PROD_F_M2 = 0.10
ENABLE_WATER_IONIZATION = True

OUT_SUBDIR_DEFAULT = "phase6b_v9_gate4_smoke"
OUT_SUBDIR = OUT_SUBDIR_DEFAULT                    # mutated in main()


def _build_kw_ladder():
    from scripts._bv_common import KW_HAT
    return (0.0, KW_HAT * 1e-6, KW_HAT * 1e-3, KW_HAT * 0.1, KW_HAT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_sp(
    *,
    stern_capacitance_f_m2: float = STERN_PROD_F_M2,
    r_H_El_pm: float = R_H_EL_BASELINE_PM,
    k_des_nondim: float = K_DES_BASELINE_NONDIM,
    k_hyd_nondim: float = 1e-3,    # smoke baseline; tame so Newton converges
    k_prot_nondim: float = 1e-3,   # smoke baseline; tame for Picard
    lambda_hydrolysis: float = 0.0,
    delta_ohp_hat: float = 4e-6,    # 0.40 nm / L_REF=100 µm = 4e-6
    gamma_max_nondim: Optional[float] = None,
):
    from scripts._bv_common import (
        A_OH_HAT, D_OH_HAT, KW_HAT,
        DEFAULT_SULFATE_ANALYTIC_BIKERMAN_FOR_K2SO4,
        FOUR_SPECIES_LOGC_DYNAMIC_K2SO4,
        GAMMA_MAX_HAT_SMOKE,
        K0_HAT_R2E, K0_HAT_R4E,
        PARALLEL_2E_4E_REACTIONS_4SP,
        SNES_OPTS_CHARGED,
        make_bv_solver_params,
        make_cation_hydrolysis_config,
        setup_firedrake_env,
    )
    setup_firedrake_env()

    snes_opts = {**SNES_OPTS_CHARGED}
    snes_opts.update({
        "snes_max_it": 400,
        "snes_atol": 1e-7,
        "snes_rtol": 1e-10,
        "snes_stol": 1e-12,
        "snes_linesearch_type": "l2",
        "snes_linesearch_maxlambda": 0.3,
        "snes_divergence_tolerance": 1e10,
    })

    # Phase 6β v10a — explicit ``gamma_max_nondim`` makes the Langmuir
    # cap visible in the driver.  ``None`` falls back to the smoke
    # baseline (1 monolayer of MOH).
    gamma_max_effective = (
        GAMMA_MAX_HAT_SMOKE if gamma_max_nondim is None
        else float(gamma_max_nondim)
    )
    cation_cfg = make_cation_hydrolysis_config(
        k_hyd=float(k_hyd_nondim),
        k_prot=float(k_prot_nondim),
        k_des=float(k_des_nondim),
        delta_ohp_hat=float(delta_ohp_hat),
        cation="K+",
        r_H_El_pm=float(r_H_El_pm),
        pka_shift_form="singh_2016_eq_4",
        gamma_max_nondim=gamma_max_effective,
    )

    sp = make_bv_solver_params(
        eta_hat=0.0, dt=0.25, t_end=80.0,
        species=FOUR_SPECIES_LOGC_DYNAMIC_K2SO4,
        snes_opts=snes_opts,
        formulation=FORMULATION,
        log_rate=True,
        u_clamp=U_CLAMP,
        bv_reactions=PARALLEL_2E_4E_REACTIONS_4SP,
        boltzmann_counterions=[DEFAULT_SULFATE_ANALYTIC_BIKERMAN_FOR_K2SO4],
        stern_capacitance_f_m2=stern_capacitance_f_m2,
        initializer=INITIALIZER,
        l_eff_m=L_EFF_M,
        enable_water_ionization=ENABLE_WATER_IONIZATION,
        kw_eff_hat=KW_HAT,
        d_oh_hat=D_OH_HAT,
        a_oh_hat=A_OH_HAT,
        enable_cation_hydrolysis=True,
        cation_hydrolysis_config=cation_cfg,
        lambda_hydrolysis=lambda_hydrolysis,
    )
    new_opts = dict(sp.solver_options)
    new_bv = dict(new_opts["bv_convergence"])
    new_bv["exponent_clip"] = float(EXPONENT_CLIP)
    new_opts["bv_convergence"] = new_bv
    sp = sp.with_solver_options(new_opts)
    return sp


def _make_mesh():
    from Forward.bv_solver import make_graded_rectangle_mesh
    return make_graded_rectangle_mesh(
        Nx=MESH_NX, Ny=MESH_NY, beta=MESH_BETA,
        domain_height_hat=L_EFF_M / 1.0e-4,
    )


def _serialize(obj):
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        if not np.isfinite(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.integer, int, bool)):
        return obj if isinstance(obj, bool) else int(obj)
    if obj is None or isinstance(obj, str):
        return obj
    return str(obj)


def _anchor_then_walk_at_lambda_zero(
    *, sp, mesh, label: str,
) -> Dict[str, Any]:
    """Anchor at V_ANCHOR, warm-walk down through V_RHE_PRIMARY at λ=0.

    Returns a dict of per-V observables (cd, pc, gamma, etc.).
    """
    import firedrake as fd
    import firedrake.adjoint as adj
    from Forward.bv_solver import solve_grid_with_anchor
    from Forward.bv_solver.anchor_continuation import (
        extract_preconverged_anchor,
        solve_anchor_with_continuation,
        LadderExhausted,
    )
    from Forward.bv_solver.cation_hydrolysis import extract_gamma_value
    from Forward.bv_solver.observables import _build_bv_observable_form
    from scripts._bv_common import I_SCALE, K0_HAT_R2E, K0_HAT_R4E, V_T

    sp_anchor = sp.with_phi_applied(V_ANCHOR / V_T)

    NV = len(V_RHE_PRIMARY)
    cd = np.full(NV, np.nan)
    pc = np.full(NV, np.nan)
    gamma_per_v = np.full(NV, np.nan)

    def _grab(orig_idx, _phi_eta, ctx):
        f_cd = _build_bv_observable_form(
            ctx, mode="current_density", reaction_index=None, scale=-I_SCALE,
        )
        f_pc = _build_bv_observable_form(
            ctx, mode="peroxide_current", reaction_index=None, scale=-I_SCALE,
        )
        cd[orig_idx] = float(fd.assemble(f_cd))
        pc[orig_idx] = float(fd.assemble(f_pc))
        if ctx.get("cation_hydrolysis") is not None:
            gamma_per_v[orig_idx] = extract_gamma_value(ctx)

    print(
        f"[{label}] Step 1: anchor at V={V_ANCHOR:+.3f} V "
        f"(kw ladder, λ=0)", flush=True,
    )
    t0 = time.time()
    try:
        with adj.stop_annotating():
            anchor_result = solve_anchor_with_continuation(
                sp_anchor,
                mesh=mesh,
                k0_targets={0: float(K0_HAT_R2E), 1: float(K0_HAT_R4E)},
                initial_scales=K0_INITIAL_SCALES,
                max_inserts_per_step=4,
                max_ss_steps_per_rung=300,
                ic_at_target=True,
                kw_eff_ladder=_build_kw_ladder(),
            )
    except LadderExhausted as exc:
        anchor_result = None
        anchor_error = f"LadderExhausted: {exc}"
    else:
        anchor_error = None
    anchor_walltime = time.time() - t0

    if anchor_result is None or not anchor_result.converged:
        print(f"[{label}] anchor FAILED ({anchor_error})", flush=True)
        return {
            "label": label,
            "anchor_converged": False,
            "anchor_error": anchor_error,
            "wall_anchor": float(anchor_walltime),
            "cd_mA_cm2": [None] * NV,
            "pc_mA_cm2": [None] * NV,
            "gamma_per_v": [None] * NV,
        }

    print(
        f"[{label}] anchor done in {anchor_walltime:.1f}s; "
        f"warm-walking {NV} V_RHE points",
        flush=True,
    )
    anchor = extract_preconverged_anchor(
        anchor_result,
        phi_applied_eta=V_ANCHOR / V_T,
        k0_targets={0: float(K0_HAT_R2E), 1: float(K0_HAT_R4E)},
        mesh_dof_count=anchor_result.ctx["U"].function_space().dim(),
    )
    phi_grid = np.array(V_RHE_PRIMARY) / V_T
    t1 = time.time()
    with adj.stop_annotating():
        result = solve_grid_with_anchor(
            sp, mesh=mesh, anchor=anchor,
            phi_applied_values=phi_grid,
            per_point_callback=_grab,
        )
    walk_walltime = time.time() - t1
    return {
        "label": label,
        "anchor_converged": True,
        "anchor_error": None,
        "wall_anchor": float(anchor_walltime),
        "wall_walk": float(walk_walltime),
        "v_rhe": list(V_RHE_PRIMARY),
        "cd_mA_cm2": [
            float(x) if np.isfinite(x) else None for x in cd
        ],
        "pc_mA_cm2": [
            float(x) if np.isfinite(x) else None for x in pc
        ],
        "gamma_per_v": [
            float(x) if np.isfinite(x) else None for x in gamma_per_v
        ],
        "n_converged": int(sum(
            p.converged for p in result.points.values()
        )),
        "n_total": int(NV),
    }


def _lambda_ramp_at_voltage(
    *, sp_template, mesh, voltage: float, lambda_ladder, label: str,
) -> Dict[str, Any]:
    """Run an explicit λ ramp at fixed voltage via solve_anchor_with_continuation.

    Each rung corresponds to a successive λ in ``lambda_ladder``.  The
    orchestrator's outer Picard for Γ runs at every rung; observables
    + Γ are recorded.
    """
    import firedrake as fd
    import firedrake.adjoint as adj
    from Forward.bv_solver.anchor_continuation import (
        solve_anchor_with_continuation, LadderExhausted,
    )
    from Forward.bv_solver.cation_hydrolysis import extract_gamma_value
    from Forward.bv_solver.observables import _build_bv_observable_form
    from scripts._bv_common import I_SCALE, K0_HAT_R2E, K0_HAT_R4E, V_T

    sp = sp_template.with_phi_applied(voltage / V_T)

    print(
        f"[{label}] λ ramp at V={voltage:+.3f}: ladder={list(lambda_ladder)}",
        flush=True,
    )
    t0 = time.time()
    try:
        with adj.stop_annotating():
            result = solve_anchor_with_continuation(
                sp,
                mesh=mesh,
                k0_targets={0: float(K0_HAT_R2E), 1: float(K0_HAT_R4E)},
                initial_scales=K0_INITIAL_SCALES,
                max_inserts_per_step=4,
                max_ss_steps_per_rung=300,
                ic_at_target=True,
                lambda_hydrolysis_ladder=tuple(lambda_ladder),
            )
    except LadderExhausted as exc:
        wall = time.time() - t0
        print(f"[{label}] λ ramp FAILED ({exc})", flush=True)
        return {
            "label": label,
            "voltage": float(voltage),
            "lambda_ladder": list(lambda_ladder),
            "converged": False,
            "error": str(exc),
            "wall": float(wall),
        }
    wall = time.time() - t0

    ctx = result.ctx
    f_cd = _build_bv_observable_form(
        ctx, mode="current_density", reaction_index=None, scale=-I_SCALE,
    )
    f_pc = _build_bv_observable_form(
        ctx, mode="peroxide_current", reaction_index=None, scale=-I_SCALE,
    )
    cd_final = float(fd.assemble(f_cd))
    pc_final = float(fd.assemble(f_pc))
    gamma_final = extract_gamma_value(ctx) if ctx.get("cation_hydrolysis") else None

    return {
        "label": label,
        "voltage": float(voltage),
        "lambda_ladder": list(lambda_ladder),
        "converged": bool(result.converged),
        "error": None,
        "wall": float(wall),
        "cd_mA_cm2": cd_final,
        "pc_mA_cm2": pc_final,
        "gamma_final": gamma_final,
        "rungs": result.rungs,
    }


# ---------------------------------------------------------------------------
# Cache-aware sweep helpers (Phase 6β v9 Gate 4B optimization)
# ---------------------------------------------------------------------------


def _anchor_then_walk_at_lambda_zero_with_snapshot(
    *, sp, mesh, label: str,
):
    """Same as ``_anchor_then_walk_at_lambda_zero`` but ALSO returns the
    U-snapshot at V_LAMBDA_TARGET so subsequent λ-ramp sweeps can warm-
    start from it.
    """
    import firedrake as fd
    import firedrake.adjoint as adj
    from Forward.bv_solver import solve_grid_with_anchor
    from Forward.bv_solver.anchor_continuation import (
        extract_preconverged_anchor,
        solve_anchor_with_continuation,
        LadderExhausted,
    )
    from Forward.bv_solver.cation_hydrolysis import extract_gamma_value
    from Forward.bv_solver.observables import _build_bv_observable_form
    from scripts._bv_common import I_SCALE, K0_HAT_R2E, K0_HAT_R4E, V_T

    sp_anchor = sp.with_phi_applied(V_ANCHOR / V_T)

    NV = len(V_RHE_PRIMARY)
    cd = np.full(NV, np.nan)
    pc = np.full(NV, np.nan)
    gamma_per_v = np.full(NV, np.nan)

    def _grab(orig_idx, _phi_eta, ctx):
        f_cd = _build_bv_observable_form(
            ctx, mode="current_density", reaction_index=None, scale=-I_SCALE,
        )
        f_pc = _build_bv_observable_form(
            ctx, mode="peroxide_current", reaction_index=None, scale=-I_SCALE,
        )
        cd[orig_idx] = float(fd.assemble(f_cd))
        pc[orig_idx] = float(fd.assemble(f_pc))
        if ctx.get("cation_hydrolysis") is not None:
            gamma_per_v[orig_idx] = extract_gamma_value(ctx)

    print(
        f"[{label}] Step 1: anchor at V={V_ANCHOR:+.3f} V (kw ladder, λ=0)",
        flush=True,
    )
    t0 = time.time()
    try:
        with adj.stop_annotating():
            anchor_result = solve_anchor_with_continuation(
                sp_anchor,
                mesh=mesh,
                k0_targets={0: float(K0_HAT_R2E), 1: float(K0_HAT_R4E)},
                initial_scales=K0_INITIAL_SCALES,
                max_inserts_per_step=4,
                max_ss_steps_per_rung=300,
                ic_at_target=True,
                kw_eff_ladder=_build_kw_ladder(),
            )
    except LadderExhausted as exc:
        return None, {
            "label": label,
            "anchor_converged": False,
            "anchor_error": f"LadderExhausted: {exc}",
            "wall_anchor": float(time.time() - t0),
        }
    anchor_walltime = time.time() - t0
    if not anchor_result.converged:
        return None, {
            "label": label,
            "anchor_converged": False,
            "anchor_error": "anchor_result.converged=False",
            "wall_anchor": float(anchor_walltime),
        }

    print(
        f"[{label}] anchor done in {anchor_walltime:.1f}s; "
        f"warm-walking {NV} V_RHE points",
        flush=True,
    )
    anchor = extract_preconverged_anchor(
        anchor_result,
        phi_applied_eta=V_ANCHOR / V_T,
        k0_targets={0: float(K0_HAT_R2E), 1: float(K0_HAT_R4E)},
        mesh_dof_count=anchor_result.ctx["U"].function_space().dim(),
    )
    phi_grid = np.array(V_RHE_PRIMARY) / V_T
    t1 = time.time()
    with adj.stop_annotating():
        result = solve_grid_with_anchor(
            sp, mesh=mesh, anchor=anchor,
            phi_applied_values=phi_grid,
            per_point_callback=_grab,
        )
    walk_walltime = time.time() - t1

    # Find V_LAMBDA_TARGET in the grid and fetch its U_data snapshot.
    target_idx = None
    for i, v in enumerate(V_RHE_PRIMARY):
        if abs(float(v) - V_LAMBDA_TARGET) < 1e-6:
            target_idx = i
            break
    target_pt = result.points.get(target_idx) if target_idx is not None else None
    target_snapshot = (
        target_pt.U_data if target_pt and target_pt.converged else None
    )

    walk_summary = {
        "label": label,
        "anchor_converged": True,
        "anchor_error": None,
        "wall_anchor": float(anchor_walltime),
        "wall_walk": float(walk_walltime),
        "v_rhe": list(V_RHE_PRIMARY),
        "cd_mA_cm2": [
            float(x) if np.isfinite(x) else None for x in cd
        ],
        "pc_mA_cm2": [
            float(x) if np.isfinite(x) else None for x in pc
        ],
        "gamma_per_v": [
            float(x) if np.isfinite(x) else None for x in gamma_per_v
        ],
        "n_converged": int(sum(
            p.converged for p in result.points.values()
        )),
        "n_total": int(NV),
        "target_idx": target_idx,
        "target_converged": (
            bool(target_pt.converged) if target_pt else False
        ),
    }
    return target_snapshot, walk_summary


def _ramp_combination_from_cache(
    *, mesh, U_warmstart: tuple, label: str,
    r_H_El_pm: float, c_s_f_m2: float, k_des_nondim: float,
    k_hyd_nondim: float = K_HYD_BASELINE_NONDIM,
    k_prot_nondim: float = K_PROT_BASELINE_NONDIM,
    lambda_ladder, quick: bool,
) -> Dict[str, Any]:
    """Run the λ ramp for ONE combination, warm-starting from the cached snapshot.

    The expensive anchor + warm-walk is skipped — caller already
    produced ``U_warmstart`` at V_LAMBDA_TARGET.  Per-combination cost
    is dominated by the SS reconverge (~30 s) and the λ ramp Picard
    (~30–90 s), so total per combination is ~1–2 min vs ~12 min for
    the cold pipeline.
    """
    import firedrake as fd
    from Forward.bv_solver.anchor_continuation import (
        solve_lambda_ramp_from_warm_start, LadderExhausted,
    )
    from Forward.bv_solver.cation_hydrolysis import extract_gamma_value
    from Forward.bv_solver.observables import _build_bv_observable_form
    from scripts._bv_common import I_SCALE, K0_HAT_R2E, K0_HAT_R4E, V_T

    sp = _build_sp(
        stern_capacitance_f_m2=c_s_f_m2,
        r_H_El_pm=r_H_El_pm,
        k_des_nondim=k_des_nondim,
        k_hyd_nondim=k_hyd_nondim,
        k_prot_nondim=k_prot_nondim,
        lambda_hydrolysis=0.0,
    )
    sp_at_v = sp.with_phi_applied(V_LAMBDA_TARGET / V_T)

    print(
        f"[{label}] λ ramp at V={V_LAMBDA_TARGET:+.3f} V from cached "
        f"snapshot (r_H_El={r_H_El_pm:.2f} pm, C_S={c_s_f_m2:.3f}, "
        f"k_des={k_des_nondim:.0e}, k_hyd={k_hyd_nondim:.0e}, "
        f"k_prot={k_prot_nondim:.0e})",
        flush=True,
    )

    t0 = time.time()
    try:
        result = solve_lambda_ramp_from_warm_start(
            sp_at_v,
            mesh=mesh,
            U_warmstart=U_warmstart,
            k0_targets={0: float(K0_HAT_R2E), 1: float(K0_HAT_R4E)},
            lambda_hydrolysis_ladder=tuple(lambda_ladder),
            reconverge_at_ss=True,
            max_ss_steps_per_rung=200,
        )
    except LadderExhausted as exc:
        return {
            "label": label,
            "r_H_El_pm": float(r_H_El_pm),
            "c_s_f_m2": float(c_s_f_m2),
            "k_des_nondim": float(k_des_nondim),
            "k_hyd_nondim": float(k_hyd_nondim),
            "k_prot_nondim": float(k_prot_nondim),
            "converged": False,
            "error": f"LadderExhausted: {exc}",
            "wall": float(time.time() - t0),
        }
    wall = time.time() - t0

    ctx = result.ctx
    f_cd = _build_bv_observable_form(
        ctx, mode="current_density", reaction_index=None, scale=-I_SCALE,
    )
    f_pc = _build_bv_observable_form(
        ctx, mode="peroxide_current", reaction_index=None, scale=-I_SCALE,
    )
    return {
        "label": label,
        "r_H_El_pm": float(r_H_El_pm),
        "c_s_f_m2": float(c_s_f_m2),
        "k_des_nondim": float(k_des_nondim),
        "k_hyd_nondim": float(k_hyd_nondim),
        "k_prot_nondim": float(k_prot_nondim),
        "converged": bool(result.converged),
        "error": None,
        "wall": float(wall),
        "cd_mA_cm2": float(fd.assemble(f_cd)),
        "pc_mA_cm2": float(fd.assemble(f_pc)),
        "gamma_final": extract_gamma_value(ctx),
        "rungs": result.rungs,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main():
    global V_LAMBDA_TARGET, V_RHE_PRIMARY, OUT_SUBDIR
    parser = argparse.ArgumentParser(
        description="Phase 6β v9 Gate 4 finite-rate hydrolysis smoke study."
    )
    parser.add_argument(
        "--quick", action="store_true",
        help=(
            "Quick mode: single (r_H_El, C_S, k_des) baseline + minimal "
            "λ ramp ((0, 1)).  ~10-15 min wall."
        ),
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help=(
            "Disable the U-cache optimization — re-run anchor + walk "
            "for each combination.  ~12 min PER COMBO instead of "
            "~12 min once.  For debugging only."
        ),
    )
    parser.add_argument(
        "--voltage", type=float, default=None,
        help=(
            "Override V_LAMBDA_TARGET (V vs RHE).  Truncates V_RHE walk "
            "to terminate at the requested voltage.  Cache file and "
            "output dir are keyed by voltage so re-runs at different "
            "voltages don't collide.  Phase 6β v9 post-Gate-4 plan §A "
            "(observability) uses --voltage -0.20."
        ),
    )
    parser.add_argument(
        "--lambda-only", action="store_true",
        help=(
            "Skip the (r_H_El, C_S, k_des) sensitivity sweep — run "
            "only the baseline combination's λ ramp.  Equivalent to "
            "--quick.  Phase 6β v9 post-Gate-4 plan §A uses this."
        ),
    )
    parser.add_argument(
        "--k-hyd-ramp", action="store_true",
        help=(
            "Phase B: instead of the 9-combination (r_H_El, C_S, k_des) "
            "sensitivity sweep, ramp k_hyd over K_HYD_RAMP_NONDIM "
            "(1e-3 ... 1e3) at fixed (r_H_El=200.98 pm, C_S=0.10 F/m², "
            "k_des=1.0 nondim).  Stop at the first rung where Picard "
            "fails or packing > 0.95.  Output to "
            "StudyResults/phase6b_v9_k_hyd_ramp/iv_curve.json."
        ),
    )
    parser.add_argument(
        "--out-subdir", type=str, default=None,
        help=(
            "Override the StudyResults subdirectory name.  When unset, "
            "derived from flags: --voltage uses "
            "phase6b_v9_observability_v_<sign><abs>; --k-hyd-ramp uses "
            "phase6b_v9_k_hyd_ramp; default is phase6b_v9_gate4_smoke."
        ),
    )
    parser.add_argument(
        "--k-hyd", type=float, default=None,
        help=(
            "Override the baseline k_hyd_nondim used by the (non-k-hyd-ramp) "
            "sweeps.  Phase D uses this to apply Phase B's calibrated "
            "k_hyd before sweeping r_H_El."
        ),
    )
    parser.add_argument(
        "--k-prot", type=float, default=None,
        help=(
            "Override the baseline k_prot_nondim.  Pair with --k-hyd "
            "if Phase B's calibration also moved k_prot."
        ),
    )
    args = parser.parse_args()

    # ----- Apply voltage override (truncate V_RHE walk to end at request)
    if args.voltage is not None:
        V_LAMBDA_TARGET = float(args.voltage)
        if V_LAMBDA_TARGET >= V_RHE_PRIMARY_FULL[0] or V_LAMBDA_TARGET < min(V_RHE_PRIMARY_FULL):
            raise SystemExit(
                f"--voltage {V_LAMBDA_TARGET!r} must be in "
                f"({min(V_RHE_PRIMARY_FULL)}, {V_RHE_PRIMARY_FULL[0]}); "
                f"valid grid={V_RHE_PRIMARY_FULL}"
            )
        truncated = [v for v in V_RHE_PRIMARY_FULL if v >= V_LAMBDA_TARGET - 1e-9]
        # Snap user voltage to grid if not already there.
        if not any(abs(v - V_LAMBDA_TARGET) < 1e-9 for v in truncated):
            truncated.append(V_LAMBDA_TARGET)
        truncated.sort(reverse=True)
        V_RHE_PRIMARY = tuple(truncated)

    # ----- Derive output subdir from flags (precedence: --out-subdir > --k-hyd-ramp > --voltage > default)
    if args.out_subdir is not None:
        OUT_SUBDIR = args.out_subdir
    elif args.k_hyd_ramp:
        OUT_SUBDIR = "phase6b_v9_k_hyd_ramp"
    elif args.voltage is not None:
        sign = "minus" if V_LAMBDA_TARGET < 0 else "plus"
        abs_val = abs(V_LAMBDA_TARGET)
        whole = int(abs_val)
        frac = int(round((abs_val - whole) * 100))
        OUT_SUBDIR = f"phase6b_v9_observability_v_{sign}_{whole}_{frac:02d}"
    else:
        OUT_SUBDIR = OUT_SUBDIR_DEFAULT

    out_dir = os.path.join(_ROOT, "StudyResults", OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    mesh = _make_mesh()
    use_cache = not args.no_cache
    quick_mode = bool(args.quick or args.lambda_only or args.k_hyd_ramp)

    summary: Dict[str, Any] = {
        "stack": "phase6b_v9_gate4_finite_hydrolysis",
        "v_anchor": float(V_ANCHOR),
        "v_lambda_target": float(V_LAMBDA_TARGET),
        "v_rhe_grid_primary": list(V_RHE_PRIMARY),
        "lambda_ladder_default": list(LAMBDA_LADDER_DEFAULT),
        "k0_initial_scales": list(K0_INITIAL_SCALES),
        "kw_eff_ladder": list(_build_kw_ladder()),
        "mesh_ny": int(MESH_NY),
        "l_eff_m": float(L_EFF_M),
        "stern_baseline_f_m2": float(STERN_PROD_F_M2),
        "r_H_El_baseline_pm": float(R_H_EL_BASELINE_PM),
        "k_des_baseline_nondim": float(K_DES_BASELINE_NONDIM),
        "k_hyd_baseline_nondim": float(K_HYD_BASELINE_NONDIM),
        "k_prot_baseline_nondim": float(K_PROT_BASELINE_NONDIM),
        "quick_mode": bool(quick_mode),
        "use_cache": bool(use_cache),
        "args": {
            "quick": bool(args.quick),
            "lambda_only": bool(args.lambda_only),
            "k_hyd_ramp": bool(args.k_hyd_ramp),
            "voltage": (
                None if args.voltage is None else float(args.voltage)
            ),
            "out_subdir": args.out_subdir,
            "no_cache": bool(args.no_cache),
        },
        "baseline_walk": None,
        "combinations": [],
    }

    # Combination tuple: (label, r_H_El_pm, C_S_F_M2, k_des_nondim,
    #                    k_hyd_nondim, k_prot_nondim)
    k_hyd_baseline_use = (
        float(args.k_hyd) if args.k_hyd is not None else K_HYD_BASELINE_NONDIM
    )
    k_prot_baseline_use = (
        float(args.k_prot) if args.k_prot is not None else K_PROT_BASELINE_NONDIM
    )
    BASELINE_TUP = (
        R_H_EL_BASELINE_PM,
        C_S_BASELINE_F_M2,
        K_DES_BASELINE_NONDIM,
        k_hyd_baseline_use,
        k_prot_baseline_use,
    )

    if args.k_hyd_ramp:
        lambda_ladder = LAMBDA_LADDER_DEFAULT
        combinations = []
        for kh in K_HYD_RAMP_NONDIM:
            combinations.append((
                f"k_hyd={kh:.0e}",
                R_H_EL_BASELINE_PM, C_S_BASELINE_F_M2, K_DES_BASELINE_NONDIM,
                kh, K_PROT_BASELINE_NONDIM,
            ))
    elif quick_mode:
        # One baseline combination only.  Use the default 5-rung
        # λ ladder (gentle Picard transitions); a 2-rung ladder
        # ``(0.0, 1.0)`` jumps λ from 0 → 1 in one step which Picard
        # cannot rollback (AdaptiveLadder needs a previous_scale to
        # insert a midpoint).
        lambda_ladder = LAMBDA_LADDER_DEFAULT
        combinations = [("baseline", *BASELINE_TUP)]
    else:
        lambda_ladder = LAMBDA_LADDER_DEFAULT
        # Full sensitivity sweep — one axis at a time around the
        # baseline (otherwise the cartesian product is 5×3×3 = 45 runs).
        combinations = [("baseline", *BASELINE_TUP)]
        for r in R_H_EL_SWEEP_PM:
            if r != R_H_EL_BASELINE_PM:
                combinations.append((
                    f"r_H_El={r:.2f}pm",
                    r, C_S_BASELINE_F_M2, K_DES_BASELINE_NONDIM,
                    k_hyd_baseline_use, k_prot_baseline_use,
                ))
        for cs in C_S_SWEEP_F_M2:
            if cs != C_S_BASELINE_F_M2:
                combinations.append((
                    f"C_S={cs:.3f}",
                    R_H_EL_BASELINE_PM, cs, K_DES_BASELINE_NONDIM,
                    k_hyd_baseline_use, k_prot_baseline_use,
                ))
        for kd in K_DES_SWEEP_NONDIM:
            if kd != K_DES_BASELINE_NONDIM:
                combinations.append((
                    f"k_des={kd:.0e}",
                    R_H_EL_BASELINE_PM, C_S_BASELINE_F_M2, kd,
                    k_hyd_baseline_use, k_prot_baseline_use,
                ))

    # ----- Cache step: run baseline anchor + warm-walk ONCE.
    # Persist the U snapshot to disk so iteration on the λ ramp /
    # sensitivity sweep doesn't re-pay the ~9 min walk cost.
    # Cache filename keyed by V_LAMBDA_TARGET so re-runs at different
    # voltages don't collide with each other.
    snapshot_path = os.path.join(
        out_dir,
        f"u_warmstart_at_v_{V_LAMBDA_TARGET:+.3f}.npz",
    )
    # Backward-compat: if the legacy un-keyed snapshot exists and matches
    # V_LAMBDA_TARGET=-0.40 (the historical default), use it.
    legacy_path = os.path.join(out_dir, "u_warmstart_at_v_target.npz")
    if (
        not os.path.exists(snapshot_path)
        and os.path.exists(legacy_path)
        and abs(V_LAMBDA_TARGET - (-0.40)) < 1e-9
    ):
        snapshot_path = legacy_path
    cached_snapshot = None
    if use_cache:
        if os.path.exists(snapshot_path):
            print(
                f"\n[gate4] Loading cached U snapshot from {snapshot_path}\n",
                flush=True,
            )
            data = np.load(snapshot_path)
            n_arrays = int(data["n_arrays"])
            cached_snapshot = tuple(
                data[f"arr_{i}"] for i in range(n_arrays)
            )
            baseline_walk = {
                "loaded_from_disk": snapshot_path,
                "n_arrays": n_arrays,
            }
        else:
            print(
                "\n[gate4] Caching baseline anchor + warm-walk → "
                f"V={V_LAMBDA_TARGET:+.3f} V (~10–15 min)\n",
                flush=True,
            )
            try:
                cached_snapshot, baseline_walk = (
                    _anchor_then_walk_at_lambda_zero_with_snapshot(
                        sp=_build_sp(),  # baseline params for the cache walk
                        mesh=mesh,
                        label="baseline_cache",
                    )
                )
            except Exception as exc:
                print(
                    f"[gate4] cache step failed ({type(exc).__name__}: "
                    f"{exc}); falling back to per-combination cold path.",
                    flush=True,
                )
                cached_snapshot = None
                baseline_walk = {"error": f"{type(exc).__name__}: {exc}"}

            if cached_snapshot is not None:
                # Persist to disk so the next iteration can skip the walk.
                np.savez(
                    snapshot_path,
                    n_arrays=len(cached_snapshot),
                    **{f"arr_{i}": arr for i, arr in enumerate(cached_snapshot)},
                )
                print(
                    f"[gate4] Persisted U snapshot to {snapshot_path} "
                    f"({len(cached_snapshot)} arrays).",
                    flush=True,
                )

        summary["baseline_walk"] = baseline_walk
        with open(os.path.join(out_dir, "iv_curve.json"), "w") as f:
            json.dump(_serialize(summary), f, indent=2)

        if cached_snapshot is None:
            print(
                "[gate4] No usable cache snapshot; aborting sweep.",
                flush=True,
            )
            return 1

    # ----- Sweep step: λ ramp per combination.
    for label, r, cs, kd, kh, kp in combinations:
        print(
            f"\n[gate4] === {label}: r_H_El={r}, C_S={cs}, k_des={kd}, "
            f"k_hyd={kh}, k_prot={kp} ===\n",
            flush=True,
        )
        try:
            if use_cache:
                combo_result = _ramp_combination_from_cache(
                    mesh=mesh, U_warmstart=cached_snapshot, label=label,
                    r_H_El_pm=r, c_s_f_m2=cs, k_des_nondim=kd,
                    k_hyd_nondim=kh, k_prot_nondim=kp,
                    lambda_ladder=lambda_ladder, quick=quick_mode,
                )
            else:
                # Cold per-combination path (slow; for debugging only).
                walk_result = _anchor_then_walk_at_lambda_zero(
                    sp=_build_sp(
                        stern_capacitance_f_m2=cs,
                        r_H_El_pm=r,
                        k_des_nondim=kd,
                        k_hyd_nondim=kh,
                        k_prot_nondim=kp,
                    ),
                    mesh=mesh, label=label,
                )
                ramp_result = _lambda_ramp_at_voltage(
                    sp_template=_build_sp(
                        stern_capacitance_f_m2=cs,
                        r_H_El_pm=r,
                        k_des_nondim=kd,
                        k_hyd_nondim=kh,
                        k_prot_nondim=kp,
                    ),
                    mesh=mesh, voltage=V_LAMBDA_TARGET,
                    lambda_ladder=lambda_ladder,
                    label=f"{label}__lam_ramp",
                )
                combo_result = {
                    "label": label,
                    "r_H_El_pm": r, "c_s_f_m2": cs, "k_des_nondim": kd,
                    "k_hyd_nondim": kh, "k_prot_nondim": kp,
                    "walk_at_lambda_zero": walk_result,
                    "lambda_ramp": ramp_result,
                }
        except Exception as exc:
            combo_result = {
                "label": label,
                "r_H_El_pm": r, "c_s_f_m2": cs, "k_des_nondim": kd,
                "k_hyd_nondim": kh, "k_prot_nondim": kp,
                "error": f"{type(exc).__name__}: {exc}",
            }
        summary["combinations"].append(combo_result)
        with open(os.path.join(out_dir, "iv_curve.json"), "w") as f:
            json.dump(_serialize(summary), f, indent=2)

    print(f"\n[gate4] DONE.  {len(combinations)} combinations.", flush=True)
    print(f"[gate4]      use_cache={use_cache}", flush=True)
    print(f"[gate4]      out_dir={out_dir}", flush=True)
    print(f"[gate4]      v_lambda_target={V_LAMBDA_TARGET:+.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
