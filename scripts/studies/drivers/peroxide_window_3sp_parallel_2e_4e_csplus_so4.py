"""3sp + parallel-2e/4e + Cs⁺/SO₄²⁻ multi-ion driver (fast-realignment plan §3).

Production-target electrolyte and topology per Ruggiero 2022 §1-§2:
  electrolyte:  Cs₂SO₄ at I = 0.3 M (Cs⁺ = 199.9 mol/m³, SO₄²⁻ = 100 mol/m³)
                + H⁺ at pH 4 (= 0.1 mol/m³)
  topology:     parallel R_2e (E°=0.695 V) + R_4e (E°=1.23 V)
  catalyst:     CMK-3 (Mangan deck) / generic carbon (Ruggiero)

Anchored at V_RHE = +0.55 V (the weakest cathodic drive within the
page-15 grid; warm-start branch).

Four passes per call:
  (A) pure-2e:    k0_R4e = 0      (full grid; ≥ 15/25 to be "done")
  (B) pure-4e:    k0_R2e = 0      (anchor smoke; ≥ 15/25 to be "done")
  (C) mixed lit:  both k0 active  (single-V exploratory; not required for done)
  (D) mixed scan: k0_R4e ladder   (full grid at chosen ratio; ≥ 15/25)

Per the fast-realignment plan §3 "Important labeling caveat": Pass D
plots demonstrate that the multi-ion + parallel-topology machinery
produces inspectable observables.  They DO NOT represent a calibrated
agreement with Mangan deck page 15.  Calibration (K0_R4e, ALPHA_R4E,
Stern, cation radii) is M4-M6 work, deferred.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(line_buffering=True)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np


# Page-15 V_RHE grid, 25 points spanning [-0.40, +0.55].
PAGE_15_V_RHE_GRID = np.linspace(-0.40, +0.55, 25).round(4).tolist()
ANCHOR_V_RHE = +0.55                  # weakest cathodic drive
PASS_D_LADDER = [1e-12, 1e-15, 1e-18, 1e-21, 1e-24]  # plan §3
PASS_D_TEST_VOLTAGES = [-0.40, -0.20, 0.0, +0.20, +0.55]

MESH_NY = 200
EXPONENT_CLIP = 100.0
U_CLAMP = 100.0
N_SUBSTEPS_WARM = 8
BISECT_DEPTH_WARM = 5
INITIALIZER = "debye_boltzmann"
FORMULATION = "logc_muh"
OUT_SUBDIR = "fast_realignment_2026-05-08"

N_COLLECTION = 0.224
H_SPECIES_INDEX = 2


def _build_parallel_reactions(*, k0_factor_R2e: float, k0_factor_R4e: float):
    from scripts._bv_common import (
        K0_HAT_R2E, K0_HAT_R4E,
        ALPHA_R2E, ALPHA_R4E,
        E_EQ_R2E_V, E_EQ_R4E_V,
        C_HP_HAT,
    )
    return [
        {
            "k0": float(K0_HAT_R2E) * float(k0_factor_R2e),
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
            "k0": float(K0_HAT_R4E) * float(k0_factor_R4e),
            "alpha": float(ALPHA_R4E),
            "cathodic_species": 0,
            "anodic_species": None,
            "c_ref": 0.0,
            "stoichiometry": [-1,  0, -4],
            "n_electrons": 4,
            "reversible": False,
            "E_eq_v": float(E_EQ_R4E_V),
            "cathodic_conc_factors": [
                {"species": 2, "power": 4, "c_ref_nondim": float(C_HP_HAT)},
            ],
        },
    ]


def _make_sp(*, k0_r2e_factor: float, k0_r4e_factor: float):
    from scripts._bv_common import (
        SNES_OPTS_CHARGED,
        THREE_SPECIES_LOGC_BOLTZMANN,
        DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC,
        DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC,
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
    rxns = _build_parallel_reactions(
        k0_factor_R2e=k0_r2e_factor, k0_factor_R4e=k0_r4e_factor,
    )
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
        multi_ion_enabled=True,             # plan §2.2/§2.3 opt-in
        stern_capacitance_f_m2=0.10,
        initializer=INITIALIZER,
    )
    new_opts = dict(sp.solver_options)
    new_bv = dict(new_opts["bv_convergence"])
    new_bv["exponent_clip"] = float(EXPONENT_CLIP)
    new_opts["bv_convergence"] = new_bv
    return sp.with_solver_options(new_opts)


def _run_pass(label: str, sp, *, v_rhe_grid, anchor_v_rhe: float):
    from scripts._bv_common import setup_firedrake_env, V_T, I_SCALE, C_SCALE

    setup_firedrake_env()

    import firedrake as fd
    import firedrake.adjoint as adj
    from Forward.bv_solver import (
        make_graded_rectangle_mesh,
        solve_grid_per_voltage_cold_with_warm_fallback,
    )
    from Forward.bv_solver.observables import _build_bv_observable_form
    from Forward.bv_solver.rrde_observables import assemble_rrde_observables

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=int(MESH_NY), beta=3.0)

    NV = len(v_rhe_grid)
    cd = np.full(NV, np.nan)
    gross_R2e = np.full(NV, np.nan)
    gross_R4e = np.full(NV, np.nan)

    def _grab(orig_idx, _phi_eta, ctx):
        f_cd = _build_bv_observable_form(
            ctx, mode="current_density", reaction_index=None, scale=-I_SCALE)
        f_R2e = _build_bv_observable_form(
            ctx, mode="reaction", reaction_index=0, scale=-I_SCALE)
        f_R4e = _build_bv_observable_form(
            ctx, mode="reaction", reaction_index=1, scale=-I_SCALE)
        cd[orig_idx]        = float(fd.assemble(f_cd))
        gross_R2e[orig_idx] = float(fd.assemble(f_R2e))
        gross_R4e[orig_idx] = float(fd.assemble(f_R4e))

    phi_hat_grid = np.array(v_rhe_grid, dtype=float) / V_T
    t0 = time.time()
    with adj.stop_annotating():
        result = solve_grid_per_voltage_cold_with_warm_fallback(
            sp,
            phi_applied_values=phi_hat_grid,
            mesh=mesh,
            max_z_steps=20,
            n_substeps_warm=N_SUBSTEPS_WARM,
            bisect_depth_warm=BISECT_DEPTH_WARM,
            per_point_callback=_grab,
        )
    elapsed = time.time() - t0

    surface_pH_arr = np.full(NV, np.nan)
    j_ring_arr = np.full(NV, np.nan)
    s_h2o2_arr = np.full(NV, np.nan)
    n_e_arr = np.full(NV, np.nan)
    c_H_surf_arr = np.full(NV, np.nan)
    h_diag_key = f"c{H_SPECIES_INDEX}_surface_mean"
    for i in range(NV):
        diag_i = result.points[i].diagnostics or {}
        c_H_i = diag_i.get(h_diag_key)
        if c_H_i is None or not np.isfinite(c_H_i):
            continue
        c_H_surf_arr[i] = float(c_H_i)
        if not (np.isfinite(cd[i]) and np.isfinite(gross_R2e[i])):
            continue
        rrde = assemble_rrde_observables(
            j_disk=float(cd[i]),
            j_h2o2_disk=float(gross_R2e[i]),
            c_H_surface_nondim=float(c_H_i),
            C_scale_mol_m3=float(C_SCALE),
            N_collection=float(N_COLLECTION),
        )
        surface_pH_arr[i] = rrde.surface_pH_proxy
        j_ring_arr[i] = rrde.j_ring_model
        s_h2o2_arr[i] = rrde.S_H2O2_percent
        n_e_arr[i] = rrde.n_e_rrde

    def _to_json_list(arr):
        return [float(x) if np.isfinite(x) else None for x in arr]

    return {
        "label": label,
        "wall_seconds": float(elapsed),
        "v_rhe": list(v_rhe_grid),
        "phi_applied_hat": [float(x) for x in phi_hat_grid.tolist()],
        "anchor_v_rhe": float(anchor_v_rhe),
        "cd_mA_cm2": _to_json_list(cd),
        "gross_R2e_mA_cm2": _to_json_list(gross_R2e),
        "gross_R4e_mA_cm2": _to_json_list(gross_R4e),
        "surface_pH_proxy": _to_json_list(surface_pH_arr),
        "j_ring_mA_cm2": _to_json_list(j_ring_arr),
        "S_H2O2_percent": _to_json_list(s_h2o2_arr),
        "n_e_rrde": _to_json_list(n_e_arr),
        "c_H_surface_nondim": _to_json_list(c_H_surf_arr),
        "N_collection_used": float(N_COLLECTION),
        "converged": [bool(result.points[i].converged) for i in range(NV)],
        "method": [result.points[i].method for i in range(NV)],
        "z_achieved": [float(result.points[i].achieved_z_factor) for i in range(NV)],
        "diagnostics": [result.points[i].diagnostics for i in range(NV)],
        "n_converged": int(sum(result.points[i].converged for i in range(NV))),
        "n_total": int(NV),
    }


def _ratio_passes_smoke(report: dict, min_count: int = 5) -> bool:
    """A ladder ratio passes the Pass D smoke gate when ≥ min_count of the
    test-V points show non-zero R_2e AND R_4e contributions."""
    ok = 0
    for r2e, r4e in zip(report["gross_R2e_mA_cm2"], report["gross_R4e_mA_cm2"]):
        if r2e is None or r4e is None:
            continue
        if abs(r2e) > 0.0 and abs(r4e) > 0.0:
            ok += 1
    return ok >= min_count


def _write_pass_outputs(out_dir: str, label: str, report: dict):
    pass_dir = os.path.join(out_dir, label)
    os.makedirs(pass_dir, exist_ok=True)
    summary_path = os.path.join(pass_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {k: v for k, v in report.items() if k != "diagnostics"},
            f, indent=2,
        )
    diag_path = os.path.join(pass_dir, "diagnostics.json")
    with open(diag_path, "w") as f:
        json.dump(
            {"label": label, "v_rhe": report["v_rhe"],
             "diagnostics_at_v": report["diagnostics"]},
            f, indent=2, default=str,
        )
    return summary_path


def main():
    from scripts._bv_common import make_experiment_metadata, I_SCALE

    out_dir = os.path.join(_ROOT, "StudyResults", OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    experiment_metadata = make_experiment_metadata(
        catalyst="CMK-3",
        geometry="RRDE",
        pH_bulk=4.0,
        cation="Cs+",
        anion_model="sulfate_bikerman_multi_ion",
        rotation_rate_rpm=1600.0,
        L_eff_m=None,
        N_collection=N_COLLECTION,
        electrolyte_model="csplus_so4_multi_ion_bikerman",
        comparison_status="diagnostic_only",
        source_authority="Ruggiero_manuscript",
        target_curve="Mangan2025_deck_p15_H2O2_current_density_pH4_Csplus",
        acceptance_tier="trend",
    )
    metadata_dict = dataclasses.asdict(experiment_metadata)

    print("=" * 78)
    print("  Parallel 2e/4e + Cs⁺/SO₄²⁻ multi-ion driver — fast realignment")
    print("=" * 78)
    print(f"  V_RHE grid (n={len(PAGE_15_V_RHE_GRID)}): "
          f"{PAGE_15_V_RHE_GRID[0]:+.3f} .. {PAGE_15_V_RHE_GRID[-1]:+.3f} V")
    print(f"  anchor V_RHE = {ANCHOR_V_RHE:+.3f} V")
    print(f"  mesh_Ny       = {MESH_NY}")
    print(f"  exponent_clip = {EXPONENT_CLIP}")
    print(f"  initializer   = {INITIALIZER}")
    print(f"  formulation   = {FORMULATION}")
    print(f"  I_SCALE       = {I_SCALE:.4g} mA/cm²")
    print(f"  output        = {out_dir}")
    print()

    summary: dict[str, Any] = {
        "experiment_metadata": metadata_dict,
        "v_rhe_grid": PAGE_15_V_RHE_GRID,
        "anchor_v_rhe": ANCHOR_V_RHE,
        "mesh_Ny": int(MESH_NY),
        "exponent_clip": float(EXPONENT_CLIP),
        "u_clamp": float(U_CLAMP),
        "initializer": INITIALIZER,
        "formulation": FORMULATION,
        "i_scale_mA_cm2": float(I_SCALE),
        "passes": {},
    }
    t_start = time.time()

    # Pass A: pure-2e
    print("--- Pass A: pure-2e (k0_R4e disabled) ---")
    sp_A = _make_sp(k0_r2e_factor=1.0, k0_r4e_factor=0.0)
    rA = _run_pass("pass_A_pure_2e", sp_A,
                   v_rhe_grid=PAGE_15_V_RHE_GRID, anchor_v_rhe=ANCHOR_V_RHE)
    _write_pass_outputs(out_dir, "pass_A_pure_2e", rA)
    summary["passes"]["A"] = {
        "label": rA["label"], "wall_seconds": rA["wall_seconds"],
        "n_converged": rA["n_converged"], "n_total": rA["n_total"],
    }
    print(f"  pass A converged {rA['n_converged']}/{rA['n_total']} in {rA['wall_seconds']:.1f}s\n")

    # Pass B: pure-4e (anchor smoke only — ≥ 15/25 for "done")
    print("--- Pass B: pure-4e (k0_R2e disabled) ---")
    sp_B = _make_sp(k0_r2e_factor=0.0, k0_r4e_factor=1.0)
    rB = _run_pass("pass_B_pure_4e", sp_B,
                   v_rhe_grid=PAGE_15_V_RHE_GRID, anchor_v_rhe=ANCHOR_V_RHE)
    _write_pass_outputs(out_dir, "pass_B_pure_4e", rB)
    summary["passes"]["B"] = {
        "label": rB["label"], "wall_seconds": rB["wall_seconds"],
        "n_converged": rB["n_converged"], "n_total": rB["n_total"],
    }
    print(f"  pass B converged {rB['n_converged']}/{rB['n_total']} in {rB['wall_seconds']:.1f}s\n")

    # Pass C: mixed at literature K0 (single-V exploratory)
    print("--- Pass C: mixed @ literature K0 (single-V exploratory) ---")
    sp_C = _make_sp(k0_r2e_factor=1.0, k0_r4e_factor=1.0)
    rC = _run_pass("pass_C_mixed_lit", sp_C,
                   v_rhe_grid=[ANCHOR_V_RHE], anchor_v_rhe=ANCHOR_V_RHE)
    _write_pass_outputs(out_dir, "pass_C_mixed_lit", rC)
    summary["passes"]["C"] = {
        "label": rC["label"], "wall_seconds": rC["wall_seconds"],
        "n_converged": rC["n_converged"], "n_total": rC["n_total"],
    }
    print(f"  pass C converged {rC['n_converged']}/{rC['n_total']} in {rC['wall_seconds']:.1f}s\n")

    # Pass D: mixed with reduced K0_R4e (LADDER)
    print("--- Pass D: mixed K0_R4e ladder ---")
    summary["passes"]["D"] = {"ladder_results": [], "promoted_ratio": None}
    chosen_ratio: float | None = None
    chosen_full: dict | None = None
    for ratio in PASS_D_LADDER:
        print(f"  ladder probe k0_R4e×{ratio:g}")
        sp_D_test = _make_sp(k0_r2e_factor=1.0, k0_r4e_factor=ratio)
        rD_test = _run_pass(
            f"pass_D_mixed_ratio_{ratio:g}_smoke", sp_D_test,
            v_rhe_grid=PASS_D_TEST_VOLTAGES, anchor_v_rhe=ANCHOR_V_RHE,
        )
        passes = _ratio_passes_smoke(rD_test, min_count=3)
        summary["passes"]["D"]["ladder_results"].append({
            "ratio": float(ratio),
            "n_converged_smoke": rD_test["n_converged"],
            "n_total_smoke": rD_test["n_total"],
            "smoke_gate_passed": bool(passes),
        })
        print(f"    smoke: {rD_test['n_converged']}/{rD_test['n_total']} "
              f"converged, gate_passed={passes}")
        if passes:
            chosen_ratio = float(ratio)
            print(f"  promoting ratio={ratio:g} to full grid")
            sp_D_full = _make_sp(k0_r2e_factor=1.0, k0_r4e_factor=ratio)
            chosen_full = _run_pass(
                f"pass_D_mixed_ratio_{ratio:g}_full", sp_D_full,
                v_rhe_grid=PAGE_15_V_RHE_GRID, anchor_v_rhe=ANCHOR_V_RHE,
            )
            _write_pass_outputs(
                out_dir, f"pass_D_mixed_ratio_{ratio:g}_full", chosen_full,
            )
            summary["passes"]["D"]["promoted_ratio"] = float(ratio)
            summary["passes"]["D"]["promoted_n_converged"] = int(chosen_full["n_converged"])
            summary["passes"]["D"]["promoted_n_total"] = int(chosen_full["n_total"])
            break
    if chosen_ratio is None:
        print("  WARNING: no ladder ratio passed the smoke gate.  "
              "Pass D 'done' criterion not reachable via this ladder.")

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written -> {summary_path}")

    # Plain-text run summary with explicit calibration caveat (plan §3 R5).
    md_path = os.path.join(out_dir, "summary.md")
    with open(md_path, "w") as f:
        f.write("# Fast-Realignment Production Sweep — Summary\n\n")
        f.write(f"Date: 2026-05-08\n")
        f.write(f"Driver: `scripts/studies/peroxide_window_3sp_parallel_2e_4e_csplus_so4.py`\n")
        f.write(f"Branch: `fast-realignment-2026-05-08`\n\n")
        f.write("## Per-pass acceptance status\n\n")
        f.write(f"- Pass A pure-2e:    {rA['n_converged']}/{rA['n_total']} converged "
                f"({rA['wall_seconds']:.0f}s)\n")
        f.write(f"- Pass B pure-4e:    {rB['n_converged']}/{rB['n_total']} converged "
                f"({rB['wall_seconds']:.0f}s)\n")
        f.write(f"- Pass C mixed-lit:  {rC['n_converged']}/{rC['n_total']} (single-V exploratory)\n")
        f.write(f"- Pass D mixed-ratio: ")
        if chosen_full is not None:
            f.write(f"ratio={chosen_ratio:g} promoted to full grid; "
                    f"{chosen_full['n_converged']}/{chosen_full['n_total']} converged\n")
        else:
            f.write("no ratio passed smoke gate\n")
        f.write("\n")
        f.write("## Calibration caveat\n\n")
        f.write("Pass D plots demonstrate that the multi-ion + parallel-topology\n")
        f.write("machinery converges and produces inspectable observables.  They\n")
        f.write("DO NOT represent calibrated agreement with Mangan deck page 15.\n")
        f.write("Calibration of K0_R4e, ALPHA_R4E, Stern, cation hydrated radii,\n")
        f.write("etc., is M4-M6 work, deliberately deferred per the fast-realignment\n")
        f.write("plan (`docs/fast_realignment_plan_2026-05-08.md`).\n")
    print(f"Markdown summary -> {md_path}")

    elapsed = time.time() - t_start
    print()
    print("=" * 78)
    print(f"  Total wall time: {elapsed:.1f}s")
    print("=" * 78)


if __name__ == "__main__":
    main()
