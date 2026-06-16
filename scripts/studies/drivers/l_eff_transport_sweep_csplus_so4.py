"""L_eff transport-domain sweep — falsifiable test of the H+ Levich-limit
hypothesis identified in
``docs/CHATGPT_REPLY_24_pass_a_kinetics_calibration.md``.

GPT's diagnosis: the model's cd plateau (~-0.0899 mA/cm²) is numerically
identical to the H+ electron-equivalent Levich limit at L_REF = 100 µm:

    F * D_H * C_H_bulk / L_REF * 0.1 = 0.0898 mA/cm²

The deck shows ~-0.18 mA/cm² (2x larger).  Sweeping a physical
transport-domain height ``L_eff_m`` with the global L_REF held fixed
should make the plateau scale linearly with 1/L_eff (Levich linearity).

Predictions from the plan (``.claude/plans/l-eff-transport-sweep.md``):

  1. ``|cd_plateau| ∝ 1/L_eff`` within ±15 % across {100, 66, 21, 16} µm.
  2. No peak appears at any L_eff on the V_RHE ∈ [-0.40, +0.55] V band
     (Levich diagnoses the magnitude only; the missing peak requires
     local-pH/buffer physics).
  3. Surface pH at the deepest cathodic V_RHE drops below 9 at the
     smallest L_eff (16 µm) — the H+ floor opens up enough to be
     deck-comparable.

Per-combo iv_curve.json schema mirrors
``mangan_full_grid_csplus_so4.py`` (cd, pc, j_ring, S_H2O2_percent,
n_e_rrde, surface_pH_proxy, c_H_surface_nondim, anchor history,
config block).  ``summary.json`` aggregates the 4 x 2 = 8 combos plus
overall wall time.

Companion scripts:

  - ``plot_l_eff_transport_sweep.py``  -- 4-panel overlay + Levich check
  - ``score_l_eff_sweep.py``           -- verdict.json with pass/fail per
                                         falsifiable prediction

Usage::

    python -u scripts/studies/l_eff_transport_sweep_csplus_so4.py
    python -u scripts/studies/l_eff_transport_sweep_csplus_so4.py \\
        --enable-water-ionization
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

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

# L_eff values: 100 µm = current default; 66/21/16 µm progressively expose
# the H+ Levich ceiling per GPT's diagnosis.  16 µm is the smallest the
# anchor smoke (tests/test_l_eff_smoke.py::test_anchor_at_l_eff_16um) was
# verified to converge at on the multi-ion + Stern stack.
L_EFF_VALUES_M = (100e-6, 66e-6, 21e-6, 16e-6)

# Two reference operating points from the prior K0_R4e/K0_R2e ratio sweep
# (memory: project_k0_r4e_ratio_regimes.md).  1e-30 ≈ "R_4e fully off".
K0_R4E_RATIOS = (1e-18, 1e-30)

# Page-15 V_RHE band (Mangan, deck-aligned) — same band as
# mangan_full_grid_csplus_so4.py but downsampled from 25 to 13 points
# (~0.079 V spacing) to keep the 4 x 2 = 8-combo wall time tractable.
# This is enough resolution to:
#   - score Levich linearity at the deepest cathodic V_RHE (4+ plateau pts);
#   - detect a peak near the deck +0.10 V feature (sample at +0.075 V);
#   - resolve the decay-to-zero band (+0.39, +0.47, +0.55).
# If a finer feature emerges and we need denser sampling, rerun a single
# (L_eff, ratio) combo on the 25-point grid.
V_RHE_GRID = tuple(np.linspace(-0.40, +0.55, 13).round(4).tolist())
ANCHOR_V_RHE = +0.55       # top of grid; weakest cathodic drive.

MESH_NX = 8
MESH_NY = 80
MESH_BETA = 3.0
EXPONENT_CLIP = 100.0
U_CLAMP = 100.0
N_SUBSTEPS_WARM = 8
BISECT_DEPTH_WARM = 5
INITIALIZER = "debye_boltzmann"
FORMULATION = "logc_muh"

INITIAL_SCALES = (1e-12, 1e-9, 1e-6, 1e-3, 1.0)
MAX_INSERTS_PER_STEP = 4
IC_AT_TARGET = True

# RRDE collection efficiency (Ruggiero §2 / Mangan deck baseline).
N_COLLECTION = 0.224
H_SPECIES_INDEX = 2  # H+ is species index 2 in THREE_SPECIES_LOGC_BOLTZMANN.

OUT_DIR = (
    Path(_ROOT) / "StudyResults" / "fast_realignment_2026-05-08"
    / "l_eff_transport_sweep"
)


# ---------------------------------------------------------------------------
# Per-combo solver-params builder
# ---------------------------------------------------------------------------


def _make_sp(*, l_eff_m: float, k0_r4e_factor: float, enable_water_ionization: bool = False):
    """Build SolverParams + k0_targets for a single (L_eff, ratio) combo.

    Mirrors ``mangan_full_grid_csplus_so4.py:_make_sp`` but threads the
    ``l_eff_m`` parameter through to ``make_bv_solver_params`` so the
    convergence-cfg block carries ``domain_height_hat = l_eff_m / L_REF``.
    The mesh y-extent is honored by ``make_graded_rectangle_mesh`` (per
    L_eff) in :func:`main`.

    Phase 6α: when ``enable_water_ionization`` is True, switches to the
    proton-condition residual on E = c_H − c_OH with the fast-equilibrium
    closure c_OH = K_w / c_H.  See
    ``docs/PHASE_6A_OH_WATER_IONIZATION_PLAN.md``.
    """
    from scripts._bv_common import (
        SNES_OPTS_CHARGED,
        THREE_SPECIES_LOGC_BOLTZMANN,
        DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,
        DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
        K0_HAT_R2E, K0_HAT_R4E,
        ALPHA_R2E, ALPHA_R4E,
        E_EQ_R2E_V, E_EQ_R4E_V,
        C_HP_HAT,
        make_bv_solver_params,
    )
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
    k0_r4e_target = float(K0_HAT_R4E) * float(k0_r4e_factor)
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
            "k0": k0_r4e_target,
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
        eta_hat=0.0, dt=0.25, t_end=80.0,
        species=THREE_SPECIES_LOGC_BOLTZMANN,
        snes_opts=snes_opts,
        formulation=FORMULATION, log_rate=True,
        u_clamp=U_CLAMP,
        bv_reactions=rxns,
        boltzmann_counterions=[
            DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,
            DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
        ],
        multi_ion_enabled=True,
        stern_capacitance_f_m2=0.10,
        initializer=INITIALIZER,
        l_eff_m=float(l_eff_m),
        enable_water_ionization=bool(enable_water_ionization),
    )
    new_opts = dict(sp.solver_options)
    new_bv = dict(new_opts["bv_convergence"])
    new_bv["exponent_clip"] = float(EXPONENT_CLIP)
    new_opts["bv_convergence"] = new_bv
    sp = sp.with_solver_options(new_opts)
    k0_targets = {0: float(K0_HAT_R2E), 1: k0_r4e_target}
    return sp, k0_targets


# ---------------------------------------------------------------------------
# Labels and directory layout
# ---------------------------------------------------------------------------


def _combo_label(l_eff_m: float, ratio: float) -> str:
    """Label like ``L100um_ratio_1e-18``."""
    l_um = round(l_eff_m * 1e6)
    return f"L{l_um}um_ratio_{ratio:g}"


def _config_dict(*, l_eff_m: float, ratio: float, enable_water_ionization: bool = False) -> dict:
    from scripts._bv_common import L_REF, KW_HAT
    return {
        "l_eff_m": float(l_eff_m),
        "domain_height_hat": float(l_eff_m) / float(L_REF),
        "ratio_K0_R4e_to_K0_R2e": float(ratio),
        "mesh_Nx": int(MESH_NX),
        "mesh_Ny": int(MESH_NY),
        "mesh_beta": float(MESH_BETA),
        "exponent_clip": float(EXPONENT_CLIP),
        "u_clamp": float(U_CLAMP),
        "stern_capacitance_f_m2": 0.10,
        "ic": INITIALIZER,
        "formulation": FORMULATION,
        "initial_scales": list(INITIAL_SCALES),
        "max_inserts_per_step": int(MAX_INSERTS_PER_STEP),
        "ic_at_target": bool(IC_AT_TARGET),
        "n_substeps_warm": int(N_SUBSTEPS_WARM),
        "bisect_depth_warm": int(BISECT_DEPTH_WARM),
        "v_rhe_grid_n": len(V_RHE_GRID),
        "v_rhe_grid_min": float(V_RHE_GRID[0]),
        "v_rhe_grid_max": float(V_RHE_GRID[-1]),
        "anchor_v_rhe": float(ANCHOR_V_RHE),
        "n_collection": float(N_COLLECTION),
        "enable_water_ionization": bool(enable_water_ionization),
        "kw_eff_hat_target": float(KW_HAT) if enable_water_ionization else 0.0,
    }


# ---------------------------------------------------------------------------
# Per-combo runner
# ---------------------------------------------------------------------------


def _run_one_combo(
    *, l_eff_m: float, ratio: float, mesh,
    enable_water_ionization: bool = False,
) -> dict:
    """Run anchor + grid walk for a single (L_eff, ratio).

    Returns a dict with the same schema as ``mangan_full_grid_csplus_so4.py``
    plus a top-level ``config`` block carrying ``l_eff_m`` and the
    ``domain_height_hat`` derived from it.

    Phase 6α: when ``enable_water_ionization`` is True, the anchor build
    threads a 5-rung Kw_eff continuation ladder
    ``(0, KW_HAT*1e-6, KW_HAT*1e-3, KW_HAT*0.1, KW_HAT)`` through
    ``solve_anchor_with_continuation`` so Newton ramps the water source
    on top of the existing k0 ladder.
    """
    from scripts._bv_common import V_T, I_SCALE, C_SCALE
    import firedrake as fd
    import firedrake.adjoint as adj
    from Forward.bv_solver import solve_grid_with_anchor
    from Forward.bv_solver.anchor_continuation import (
        LadderExhausted,
        extract_preconverged_anchor,
        solve_anchor_with_continuation,
    )
    from Forward.bv_solver.observables import _build_bv_observable_form
    from Forward.bv_solver.rrde_observables import assemble_rrde_observables

    sp, k0_targets = _make_sp(
        l_eff_m=l_eff_m, k0_r4e_factor=ratio,
        enable_water_ionization=enable_water_ionization,
    )
    sp_anchor = sp.with_phi_applied(float(ANCHOR_V_RHE) / V_T)

    print(f"\n----- L_eff = {l_eff_m * 1e6:.0f} µm, ratio = {ratio:g} -----")
    print(f"  domain_height_hat = "
          f"{sp.solver_options['bv_convergence']['domain_height_hat']:.4f}")
    print(f"  k0_R2e target = {k0_targets[0]:.3e}, "
          f"k0_R4e target = {k0_targets[1]:.3e}")
    if enable_water_ionization:
        from scripts._bv_common import KW_HAT
        print(f"  enable_water_ionization=True (Kw_eff target = {KW_HAT:.3e})")

    # Anchor build.
    t0 = time.time()
    anchor_converged = False
    anchor_ladder_history: list = []
    anchor_err: str | None = None
    anchor_dof = 0
    anchor_result = None
    if enable_water_ionization:
        from scripts._bv_common import KW_HAT
        kw_eff_ladder_arg = (
            0.0,
            KW_HAT * 1e-6,
            KW_HAT * 1e-3,
            KW_HAT * 0.1,
            KW_HAT,
        )
    else:
        kw_eff_ladder_arg = None

    # Per-rung progress callback so the operator sees per-Newton-solve
    # heartbeats during multi-minute anchor builds (Phase 6α adds up to
    # 5 Kw_eff rungs × 5 k0 rungs = 25 SS solves).  Stamps wallclock so
    # stalls stand out vs a healthy ramp.
    _t_combo = time.time()

    def _rung_print(scale, ok, ctx, rung_diag):
        elapsed = time.time() - _t_combo
        label = rung_diag.get("rung_label", "kw=default")
        cd = rung_diag.get("cd_observable")
        cd_str = f"  cd={cd:+.4f}" if cd is not None else ""
        flag = "ok" if ok else "FAIL"
        print(
            f"    [{elapsed:6.1f}s] {label}  k0_scale={scale:.3e}  "
            f"{flag}{cd_str}",
            flush=True,
        )

    try:
        with adj.stop_annotating():
            anchor_result = solve_anchor_with_continuation(
                sp_anchor,
                mesh=mesh,
                k0_targets=k0_targets,
                initial_scales=INITIAL_SCALES,
                max_inserts_per_step=MAX_INSERTS_PER_STEP,
                ic_at_target=IC_AT_TARGET,
                kw_eff_ladder=kw_eff_ladder_arg,
                rung_callback=_rung_print,
            )
        anchor_converged = bool(anchor_result.converged)
        anchor_ladder_history = list(anchor_result.ladder_history)
        if anchor_converged:
            anchor_dof = int(
                anchor_result.ctx["U"].function_space().dim()
            )
    except LadderExhausted as exc:
        anchor_err = f"LadderExhausted: {exc}"
        print(f"  anchor build FAILED: {anchor_err}", flush=True)
    except Exception as exc:
        anchor_err = f"{type(exc).__name__}: {exc}"
        print(f"  anchor build ERRORED: {anchor_err}", flush=True)
    anchor_wall = time.time() - t0

    NV = len(V_RHE_GRID)
    if not anchor_converged:
        return {
            "l_eff_m": float(l_eff_m),
            "ratio": float(ratio),
            "k0_targets": {str(j): float(v) for j, v in k0_targets.items()},
            "v_rhe": list(V_RHE_GRID),
            "phi_applied_hat": [float(v) / float(V_T) for v in V_RHE_GRID],
            "cd_mA_cm2": [None] * NV,
            "pc_mA_cm2": [None] * NV,
            "j_ring_mA_cm2": [None] * NV,
            "S_H2O2_percent": [None] * NV,
            "n_e_rrde": [None] * NV,
            "surface_pH_proxy": [None] * NV,
            "c_H_surface_nondim": [None] * NV,
            "N_collection_used": float(N_COLLECTION),
            "converged": [False] * NV,
            "method": ["anchor-build-failed"] * NV,
            "n_converged": 0,
            "n_total": NV,
            "anchor": {
                "v_rhe": float(ANCHOR_V_RHE),
                "converged": False,
                "ladder_history": [
                    [float(s), str(o)] for s, o in anchor_ladder_history
                ],
                "wall_seconds": float(anchor_wall),
                "error": anchor_err,
            },
            "config": _config_dict(
                l_eff_m=l_eff_m, ratio=ratio,
                enable_water_ionization=enable_water_ionization,
            ),
        }

    print(f"  anchor converged in {anchor_wall:.1f}s "
          f"(ladder={anchor_ladder_history!r})")

    anchor = extract_preconverged_anchor(
        anchor_result,
        phi_applied_eta=float(ANCHOR_V_RHE) / float(V_T),
        k0_targets=k0_targets,
        mesh_dof_count=anchor_dof,
    )

    # Grid walk + observable capture.
    cd_arr = np.full(NV, np.nan)
    pc_arr = np.full(NV, np.nan)

    def _grab(orig_idx: int, _phi_eta: float, ctx: dict) -> None:
        try:
            f_cd = _build_bv_observable_form(
                ctx, mode="current_density", reaction_index=None,
                scale=-I_SCALE,
            )
            cd_arr[orig_idx] = float(fd.assemble(f_cd))
        except Exception as exc:
            print(f"    cd capture failed @ idx={orig_idx}: "
                  f"{type(exc).__name__}: {exc}")
        try:
            f_pc = _build_bv_observable_form(
                ctx, mode="gross_h2o2_current", reaction_index=0,
                scale=-I_SCALE,
            )
            pc_arr[orig_idx] = float(fd.assemble(f_pc))
        except Exception as exc:
            print(f"    pc capture failed @ idx={orig_idx}: "
                  f"{type(exc).__name__}: {exc}")

    phi_grid_eta = np.array(V_RHE_GRID, dtype=float) / float(V_T)
    t0 = time.time()
    grid_result = solve_grid_with_anchor(
        sp,
        anchor=anchor,
        phi_applied_values=phi_grid_eta,
        mesh=mesh,
        n_substeps_warm=N_SUBSTEPS_WARM,
        bisect_depth_warm=BISECT_DEPTH_WARM,
        per_point_callback=_grab,
    )
    grid_wall = time.time() - t0

    converged = [bool(grid_result.points[i].converged) for i in range(NV)]
    method = [str(grid_result.points[i].method) for i in range(NV)]
    n_converged = sum(1 for c in converged if c)

    cd_json = [
        float(x) if (np.isfinite(x) and converged[i]) else None
        for i, x in enumerate(cd_arr)
    ]
    pc_json = [
        float(x) if (np.isfinite(x) and converged[i]) else None
        for i, x in enumerate(pc_arr)
    ]

    # Ring-side RRDE observables.
    j_ring_arr = np.full(NV, np.nan)
    s_h2o2_arr = np.full(NV, np.nan)
    n_e_arr = np.full(NV, np.nan)
    surface_pH_arr = np.full(NV, np.nan)
    c_H_surf_arr = np.full(NV, np.nan)
    h_diag_key = f"c{H_SPECIES_INDEX}_surface_mean"
    for i in range(NV):
        if not converged[i]:
            continue
        diag_i = grid_result.points[i].diagnostics or {}
        c_H_i = diag_i.get(h_diag_key)
        if c_H_i is None or not np.isfinite(c_H_i):
            continue
        c_H_surf_arr[i] = float(c_H_i)
        if not (np.isfinite(cd_arr[i]) and np.isfinite(pc_arr[i])):
            continue
        rrde = assemble_rrde_observables(
            j_disk=float(cd_arr[i]),
            j_h2o2_disk=float(pc_arr[i]),
            c_H_surface_nondim=float(c_H_i),
            C_scale_mol_m3=float(C_SCALE),
            N_collection=float(N_COLLECTION),
        )
        j_ring_arr[i] = rrde.j_ring_model
        s_h2o2_arr[i] = rrde.S_H2O2_percent
        n_e_arr[i] = rrde.n_e_rrde
        surface_pH_arr[i] = rrde.surface_pH_proxy

    def _to_json_list(arr):
        return [
            float(x) if (np.isfinite(x) and converged[i]) else None
            for i, x in enumerate(arr)
        ]

    print(f"  grid: {n_converged}/{NV} converged in {grid_wall:.1f}s")

    return {
        "l_eff_m": float(l_eff_m),
        "ratio": float(ratio),
        "k0_targets": {str(j): float(v) for j, v in k0_targets.items()},
        "v_rhe": list(V_RHE_GRID),
        "phi_applied_hat": [float(p) for p in phi_grid_eta.tolist()],
        "cd_mA_cm2": cd_json,
        "pc_mA_cm2": pc_json,
        "j_ring_mA_cm2": _to_json_list(j_ring_arr),
        "S_H2O2_percent": _to_json_list(s_h2o2_arr),
        "n_e_rrde": _to_json_list(n_e_arr),
        "surface_pH_proxy": _to_json_list(surface_pH_arr),
        "c_H_surface_nondim": _to_json_list(c_H_surf_arr),
        "N_collection_used": float(N_COLLECTION),
        "converged": converged,
        "method": method,
        "n_converged": int(n_converged),
        "n_total": int(NV),
        "anchor": {
            "v_rhe": float(ANCHOR_V_RHE),
            "phi_applied_eta": float(ANCHOR_V_RHE) / float(V_T),
            "converged": True,
            "ladder_history": [
                [float(s), str(o)] for s, o in anchor_ladder_history
            ],
            "wall_seconds": float(anchor_wall),
        },
        "grid_wall_seconds": float(grid_wall),
        "config": _config_dict(
            l_eff_m=l_eff_m, ratio=ratio,
            enable_water_ionization=enable_water_ionization,
        ),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import argparse

    from scripts._bv_common import setup_firedrake_env, L_REF
    setup_firedrake_env()

    from Forward.bv_solver import make_graded_rectangle_mesh

    parser = argparse.ArgumentParser(
        description=(
            "L_eff transport-domain sweep with optional Phase 6α water-"
            "self-ionization closure."
        )
    )
    parser.add_argument(
        "--enable-water-ionization",
        action="store_true",
        default=False,
        help=(
            "Enable the proton-condition residual on E = c_H − c_OH "
            "with the fast-equilibrium closure c_OH = K_w / c_H.  Adds "
            "a 5-rung Kw_eff continuation ladder to each anchor build "
            "(see docs/PHASE_6A_OH_WATER_IONIZATION_PLAN.md)."
        ),
    )
    args = parser.parse_args(argv)

    enable_water_ionization = bool(args.enable_water_ionization)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("  L_eff transport-domain sweep — H+ Levich-limit hypothesis")
    print("=" * 78)
    print(f"  L_eff values            = {[f'{v * 1e6:.0f} µm' for v in L_EFF_VALUES_M]!r}")
    print(f"  ratios                  = {list(K0_R4E_RATIOS)!r}")
    print(f"  V_RHE band              = [{V_RHE_GRID[0]:+.3f}, {V_RHE_GRID[-1]:+.3f}] V "
          f"({len(V_RHE_GRID)} points)")
    print(f"  anchor V_RHE            = {ANCHOR_V_RHE:+.3f} V")
    print(f"  mesh_Ny                 = {MESH_NY}")
    print(f"  enable_water_ionization = {enable_water_ionization}")
    print(f"  output                  = {OUT_DIR}")
    print(f"  combos to run           = {len(L_EFF_VALUES_M) * len(K0_R4E_RATIOS)}")

    sweep_t0 = time.time()
    summary_per_combo: list[dict] = []

    for l_eff_m in L_EFF_VALUES_M:
        # Each L_eff gets its own mesh; the rectangle's y-extent encodes
        # the physical transport-domain height in nondim units.
        domain_height_hat = float(l_eff_m) / float(L_REF)
        mesh = make_graded_rectangle_mesh(
            Nx=MESH_NX, Ny=MESH_NY, beta=MESH_BETA,
            domain_height_hat=domain_height_hat,
        )
        for ratio in K0_R4E_RATIOS:
            report = _run_one_combo(
                l_eff_m=l_eff_m, ratio=ratio, mesh=mesh,
                enable_water_ionization=enable_water_ionization,
            )
            label = _combo_label(l_eff_m, ratio)
            out_subdir = OUT_DIR / label
            out_subdir.mkdir(parents=True, exist_ok=True)
            with open(out_subdir / "iv_curve.json", "w") as f:
                json.dump(report, f, indent=2)
            summary_per_combo.append({
                "l_eff_m": float(l_eff_m),
                "l_eff_um": float(l_eff_m * 1e6),
                "ratio": float(ratio),
                "label": label,
                "anchor_converged": bool(report["anchor"]["converged"]),
                "n_converged": int(report["n_converged"]),
                "n_total": int(report["n_total"]),
            })

    sweep_wall = time.time() - sweep_t0

    summary = {
        "l_eff_values_m": [float(v) for v in L_EFF_VALUES_M],
        "ratios": [float(r) for r in K0_R4E_RATIOS],
        "anchor_v_rhe": float(ANCHOR_V_RHE),
        "v_rhe_grid": list(V_RHE_GRID),
        "per_combo": summary_per_combo,
        "wall_seconds": float(sweep_wall),
        "config_template": _config_dict(
            l_eff_m=L_EFF_VALUES_M[0], ratio=K0_R4E_RATIOS[0],
            enable_water_ionization=enable_water_ionization,
        ),
        "enable_water_ionization": bool(enable_water_ionization),
    }
    summary_path = OUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 78)
    print(f"  Sweep complete in {sweep_wall:.1f}s "
          f"({sweep_wall / 60:.1f} min)")
    print("=" * 78)
    for entry in summary_per_combo:
        flag = "ANCHOR-FAIL" if not entry["anchor_converged"] else "OK"
        print(
            f"    L_eff={entry['l_eff_um']:>4.0f} µm, "
            f"ratio={entry['ratio']:>10.3g}   "
            f"{entry['n_converged']}/{entry['n_total']}   {flag}"
        )
    print(f"  summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
