"""Deck-aligned full-grid driver — promising ratios on Mangan page-15 V_RHE band.

The K0_R4e/K0_R2e ratio sweep
(``k0_r4e_ratio_sweep_csplus_so4.py``) on V_RHE ∈ [+0.1, +0.8] V
identified two reference operating points (see
``project_k0_r4e_ratio_regimes.md`` memory):

* ``ratio = 1e-18`` — Butler shape + ~35-50% peroxide selectivity
  ("Mangan-like" candidate);
* ``ratio = 1e-24`` — R_4e fully shut off (pure-2e limit, identical to
  1e-30);

This driver re-runs *those two ratios* on the **deck-aligned page-15
V_RHE grid** ``linspace(-0.40, +0.55, 25)`` so the resulting curves
span the same V band that Mangan's experimental data covers. Anchor
remains at +0.55 V (top of grid); :func:`solve_grid_with_anchor` walks
both directions outward.

For each ratio, emits per-voltage iv_curve JSON; the companion plotter
(``plot_mangan_full_grid.py``) overlays cd & pc vs V_RHE alongside the
deck-aligned regime annotations.

Usage::

    python -u scripts/studies/mangan_full_grid_csplus_so4.py
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

# Mangan page-15 grid — same band as
# peroxide_window_3sp_parallel_2e_4e_csplus_so4.py PAGE_15_V_RHE_GRID.
V_RHE_GRID = tuple(np.linspace(-0.40, +0.55, 25).round(4).tolist())
ANCHOR_V_RHE = +0.55       # top of page-15 grid; weakest cathodic drive.

# Two reference operating points from the prior sweep.
K0_R4E_RATIOS = (1e-18, 1e-24)

MESH_NY = 80
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
    / "mangan_full_grid"
)


def _make_sp(*, k0_r4e_factor: float):
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
    )
    new_opts = dict(sp.solver_options)
    new_bv = dict(new_opts["bv_convergence"])
    new_bv["exponent_clip"] = float(EXPONENT_CLIP)
    new_opts["bv_convergence"] = new_bv
    sp = sp.with_solver_options(new_opts)
    k0_targets = {0: float(K0_HAT_R2E), 1: k0_r4e_target}
    return sp, k0_targets


def _ratio_label(ratio: float) -> str:
    return f"ratio_{ratio:g}"


def _run_one_ratio(ratio: float, *, mesh) -> dict:
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

    sp, k0_targets = _make_sp(k0_r4e_factor=ratio)
    sp_anchor = sp.with_phi_applied(float(ANCHOR_V_RHE) / V_T)

    print(f"\n----- ratio = {ratio:g} -----")
    print(f"  k0_R2e target = {k0_targets[0]:.3e}, "
          f"k0_R4e target = {k0_targets[1]:.3e}")

    # Anchor build.
    t0 = time.time()
    anchor_converged = False
    anchor_ladder_history: list = []
    anchor_err: str | None = None
    anchor_dof = 0
    anchor_result = None
    try:
        with adj.stop_annotating():
            anchor_result = solve_anchor_with_continuation(
                sp_anchor,
                mesh=mesh,
                k0_targets=k0_targets,
                initial_scales=INITIAL_SCALES,
                max_inserts_per_step=MAX_INSERTS_PER_STEP,
                ic_at_target=IC_AT_TARGET,
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
            "config": _config_dict(ratio),
        }

    print(f"  anchor converged in {anchor_wall:.1f}s "
          f"(ladder={anchor_ladder_history!r})")

    anchor = extract_preconverged_anchor(
        anchor_result,
        phi_applied_eta=float(ANCHOR_V_RHE) / float(V_T),
        k0_targets=k0_targets,
        mesh_dof_count=anchor_dof,
    )

    # Grid walk.
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

    # ----- 3. Post-process: ring-side RRDE observables -----
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
        "config": _config_dict(ratio),
    }


def _config_dict(ratio: float) -> dict:
    return {
        "ratio_K0_R4e_to_K0_R2e": float(ratio),
        "mesh_Ny": int(MESH_NY),
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
    }


def main() -> int:
    from scripts._bv_common import setup_firedrake_env
    setup_firedrake_env()

    from Forward.bv_solver import make_graded_rectangle_mesh

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"  Mangan-aligned full-grid driver — page-15 V_RHE band")
    print("=" * 78)
    print(f"  ratios       = {list(K0_R4E_RATIOS)!r}")
    print(f"  V_RHE band   = [{V_RHE_GRID[0]:+.3f}, {V_RHE_GRID[-1]:+.3f}] V "
          f"({len(V_RHE_GRID)} points)")
    print(f"  anchor V_RHE = {ANCHOR_V_RHE:+.3f} V")
    print(f"  mesh_Ny      = {MESH_NY}")
    print(f"  output       = {OUT_DIR}")

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=int(MESH_NY), beta=3.0)

    sweep_t0 = time.time()
    summary_per_ratio: list[dict] = []
    for ratio in K0_R4E_RATIOS:
        report = _run_one_ratio(ratio, mesh=mesh)
        out_subdir = OUT_DIR / _ratio_label(ratio)
        out_subdir.mkdir(parents=True, exist_ok=True)
        with open(out_subdir / "iv_curve.json", "w") as f:
            json.dump(report, f, indent=2)
        summary_per_ratio.append({
            "ratio": float(ratio),
            "label": _ratio_label(ratio),
            "anchor_converged": bool(report["anchor"]["converged"]),
            "n_converged": int(report["n_converged"]),
            "n_total": int(report["n_total"]),
        })

    sweep_wall = time.time() - sweep_t0

    summary = {
        "ratios": [float(r) for r in K0_R4E_RATIOS],
        "anchor_v_rhe": float(ANCHOR_V_RHE),
        "v_rhe_grid": list(V_RHE_GRID),
        "per_ratio": summary_per_ratio,
        "wall_seconds": float(sweep_wall),
        "config": _config_dict(0.0),
    }
    summary_path = OUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 78)
    print(f"  Sweep complete in {sweep_wall:.1f}s")
    print("=" * 78)
    for entry in summary_per_ratio:
        flag = "ANCHOR-FAIL" if not entry["anchor_converged"] else "OK"
        print(f"    ratio={entry['ratio']:>10.3g}   "
              f"{entry['n_converged']}/{entry['n_total']}   {flag}")
    print(f"  summary -> {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
