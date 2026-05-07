"""Run C — Mangan deck page 15 comparison (Plan B "swirling-crunching-wren").

Compares the production 3sp+bikerman+muh+Stern stack against the
digitised page 15 H2O2 current density curve at pH 4 / Cs+. Voltage
grid is page-15-tailored (denser around peak, shoulder, onset).

Production stack (CLAUDE.md "Calling the production solver"):

    species              = THREE_SPECIES_LOGC_BOLTZMANN
    boltzmann_counterions= [DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC]
    formulation          = "logc_muh"
    log_rate             = True
    initializer          = "debye_boltzmann"
    stern_capacitance    = 0.10 F/m^2
    exponent_clip        = 100  (CLAUDE.md hard rule 2 — PC-trustworthy)

Output: StudyResults/mangan_p15_comparison/run_C/
  - iv_curve.json    (experiment_metadata + per-V cd/pc/RRDE observables)
  - diagnostics.json (per-V solver diagnostics)
  - comparison.png   (side-by-side experimental + model PC overlay)

See docs/m0_target_extraction.md for the M0 extraction outputs that
feed the experiment_metadata block.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from typing import Any, Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(line_buffering=True)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np


# ---------------------------------------------------------------------------
# Plan B locked decisions (B1, B6, B7, M0 extraction outputs)
# ---------------------------------------------------------------------------

TARGET_CURVE = "Mangan2025_deck_p15_H2O2_current_density_pH4_Csplus"
ACCEPTANCE_TIER = "semi_quant"

# Voltage grid: B6 says "Run the solver across V_RHE in [-0.40, +0.55] V to
# give context."  Denser around the digitised peak (~+0.10 V), shoulder
# (V_RHE in [+0.18, +0.30]), and onset (V_RHE ~ +0.42-0.45 V) so Run D can
# resolve the B7 tolerance bands.
V_RHE_GRID = (
    -0.40, -0.32, -0.25, -0.20, -0.15, -0.10, -0.05,
     0.00,  0.05,  0.08,  0.10,  0.12,  0.15,  0.18,
     0.22,  0.25,  0.28,  0.30,  0.35,  0.40,  0.42,
     0.45,  0.48,  0.50,  0.55,
)

EXPERIMENTAL_CSV = os.path.join(_ROOT, "data", "mangan_deck_p15_h2o2_current.csv")
OUT_SUBDIR = "mangan_p15_comparison/run_C"
PASS_LABEL = "3sp_bikerman_muh_stern_0p10_clip100"

# Production stack defaults (CLAUDE.md):
MESH_NY = 200
EXPONENT_CLIP = 100.0   # CLAUDE.md hard rule 2 — clip=100 only PC-trustworthy
U_CLAMP = 100.0
N_SUBSTEPS_WARM = 8
BISECT_DEPTH_WARM = 5
STERIC_CAP = 100.0
STERN_F_M2 = 0.10
INITIALIZER = "debye_boltzmann"
FORMULATION = "logc_muh"

# RRDE constants (B3): pH 4 + Cs+ -> Ruggiero 2022 cross-reference.
# See docs/m0_target_extraction.md B3 row.  N_collection is informational
# here (Run C compares peroxide_current directly against the digitised
# j_H2O2 from page 15; no ring/N machinery needed) but the M1 metadata
# block records it for downstream consumers.
N_COLLECTION = 0.224
ROTATION_RPM = 1600.0
H_SPECIES_INDEX = 2  # H+ in THREE_SPECIES_LOGC_BOLTZMANN


# ---------------------------------------------------------------------------
# Experimental CSV loader
# ---------------------------------------------------------------------------

def _load_experimental_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (V_RHE_V, j_H2O2_mA_cm2) numpy arrays from the digitised CSV.

    Lines starting with '#' are header metadata (provenance) and skipped.
    The first non-comment line is the column header.
    """
    v_list: list[float] = []
    j_list: list[float] = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        header_seen = False
        for row in reader:
            if not row:
                continue
            if row[0].startswith("#"):
                continue
            if not header_seen:
                header_seen = True
                continue
            v_list.append(float(row[0]))
            j_list.append(float(row[1]))
    return np.array(v_list, dtype=float), np.array(j_list, dtype=float)


# ---------------------------------------------------------------------------
# Solver pass
# ---------------------------------------------------------------------------

def _run_pass(*, v_rhe_grid: list[float]) -> dict[str, Any]:
    from scripts._bv_common import (
        setup_firedrake_env,
        V_T, I_SCALE, C_SCALE,
        K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2,
        THREE_SPECIES_LOGC_BOLTZMANN,
        DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC,
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
    from Forward.bv_solver.rrde_observables import assemble_rrde_observables

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=int(MESH_NY), beta=3.0)

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
        formulation=FORMULATION, log_rate=True,
        u_clamp=U_CLAMP,
        boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC],
        stern_capacitance_f_m2=STERN_F_M2,
        k0_hat_r1=K0_HAT_R1, k0_hat_r2=K0_HAT_R2,
        alpha_r1=ALPHA_R1,   alpha_r2=ALPHA_R2,
        E_eq_r1=0.68,        E_eq_r2=1.78,
        initializer=INITIALIZER,
    )
    new_opts = dict(sp.solver_options)
    new_bv = dict(new_opts["bv_convergence"])
    new_bv["exponent_clip"] = float(EXPONENT_CLIP)
    new_opts["bv_convergence"] = new_bv
    sp = sp.with_solver_options(new_opts)

    NV = len(v_rhe_grid)
    cd = np.full(NV, np.nan)
    pc = np.full(NV, np.nan)

    def _grab(orig_idx, _phi_eta, ctx):
        f_cd = _build_bv_observable_form(
            ctx, mode="current_density", reaction_index=None, scale=-I_SCALE)
        f_pc = _build_bv_observable_form(
            ctx, mode="peroxide_current", reaction_index=None, scale=-I_SCALE)
        cd[orig_idx] = float(fd.assemble(f_cd))
        pc[orig_idx] = float(fd.assemble(f_pc))

    phi_hat_grid = np.array(v_rhe_grid) / V_T
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

    # M1 RRDE post-processing (informational; not on the comparison surface
    # for page 15 since we compare pc directly against j_H2O2).
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
        if not (np.isfinite(cd[i]) and np.isfinite(pc[i])):
            continue
        rrde = assemble_rrde_observables(
            j_disk=float(cd[i]),
            j_h2o2_disk=float(pc[i]),
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
        "label": PASS_LABEL,
        "stern_f_m2": STERN_F_M2,
        "exponent_clip": EXPONENT_CLIP,
        "wall_seconds": float(elapsed),
        "v_rhe": list(v_rhe_grid),
        "phi_applied_hat": [float(x) for x in phi_hat_grid.tolist()],
        "cd_mA_cm2": _to_json_list(cd),
        "pc_mA_cm2": _to_json_list(pc),
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


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _make_plot(
    report: dict, exp_v: np.ndarray, exp_j: np.ndarray, png_path: str,
) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return f"matplotlib unavailable: {exc}"

    fig = plt.figure(figsize=(11, 9))
    gs = fig.add_gridspec(2, 1, hspace=0.30)
    ax_pc = fig.add_subplot(gs[0])
    ax_cd = fig.add_subplot(gs[1], sharex=ax_pc)

    v = np.asarray(report["v_rhe"])
    pc = np.array(
        [np.nan if x is None else x for x in report["pc_mA_cm2"]],
        dtype=float,
    )
    cd = np.array(
        [np.nan if x is None else x for x in report["cd_mA_cm2"]],
        dtype=float,
    )

    ax_pc.plot(
        exp_v, exp_j,
        marker="x", color="m", ls="-", lw=1.0, ms=5.0,
        alpha=0.8, label="experimental (deck p15, digitised)",
    )
    ax_pc.plot(
        v, pc,
        marker="o", color="C0", ls="-", lw=1.5, ms=6.0,
        label=f"model: {PASS_LABEL}",
    )
    ax_pc.axhline(0.0, color="k", lw=0.5, alpha=0.4)
    ax_pc.axvline(0.10, color="orange", ls=":", lw=0.8, alpha=0.5,
                  label="exp peak ~ +0.10 V")
    ax_pc.axvline(0.45, color="green", ls=":", lw=0.8, alpha=0.5,
                  label="exp onset-to-zero ~ +0.45 V")
    ax_pc.set_ylabel("Peroxide Current Density (mA/cm²)")
    ax_pc.set_title(
        f"Mangan deck p.15 H₂O₂ current density (pH 4, Cs⁺) vs model\n"
        f"(Ny={MESH_NY}, formulation={FORMULATION}, "
        f"clip={EXPONENT_CLIP:g}, Stern={STERN_F_M2:g} F/m², "
        f"initializer={INITIALIZER})"
    )
    ax_pc.grid(True, alpha=0.3)
    ax_pc.legend(fontsize=8, loc="best")

    ax_cd.plot(
        v, cd,
        marker="s", color="C3", ls="-", lw=1.5, ms=6.0,
        label=f"model: total CD",
    )
    ax_cd.axhline(0.0, color="k", lw=0.5, alpha=0.4)
    ax_cd.set_xlabel("V vs RHE (V)")
    ax_cd.set_ylabel("Total Current Density (mA/cm²)")
    ax_cd.grid(True, alpha=0.3)
    ax_cd.legend(fontsize=8, loc="best")

    plt.savefig(png_path, dpi=160, bbox_inches="tight")
    plt.close()
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import dataclasses
    from scripts._bv_common import make_experiment_metadata

    out_dir = os.path.join(_ROOT, "StudyResults", OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    # Plan B M0 extraction outputs -> experiment_metadata
    # See docs/m0_target_extraction.md for the per-quantity source authority.
    experiment_metadata = make_experiment_metadata(
        catalyst="CMK-3",
        geometry="RRDE",
        pH_bulk=4.0,
        cation="Cs+",
        anion_model="ClO4_protonic_surrogate",
        rotation_rate_rpm=ROTATION_RPM,
        L_eff_m=None,
        N_collection=N_COLLECTION,
        electrolyte_model="pH_countercharge_surrogate",
        comparison_status="deck_proxy",
        source_authority="Mangan2025_deck",
        target_curve=TARGET_CURVE,
        acceptance_tier=ACCEPTANCE_TIER,
    )
    metadata_dict = dataclasses.asdict(experiment_metadata)

    print("=" * 78)
    print("  Run C — Mangan deck p.15 H2O2 current density comparison")
    print("=" * 78)
    print(f"  target        = {TARGET_CURVE}")
    print(f"  V_RHE grid    = {len(V_RHE_GRID)} points in "
          f"[{min(V_RHE_GRID):+.2f}, {max(V_RHE_GRID):+.2f}] V")
    print(f"  pass          = {PASS_LABEL}")
    print(f"  mesh_Ny       = {MESH_NY}")
    print(f"  exponent_clip = {EXPONENT_CLIP}")
    print(f"  initializer   = {INITIALIZER}")
    print(f"  formulation   = {FORMULATION}")
    print(f"  Stern_C_F_m2  = {STERN_F_M2}")
    print(f"  N_collection  = {N_COLLECTION}")
    print(f"  comparison    = {experiment_metadata.comparison_status}")
    print(f"  source        = {experiment_metadata.source_authority}")
    print(f"  acceptance    = {experiment_metadata.acceptance_tier}")
    print(f"  output        = {out_dir}")
    print()

    print(f"--- pass: {PASS_LABEL} ---")
    t_start = time.time()
    report = _run_pass(v_rhe_grid=list(V_RHE_GRID))
    print(f"  converged {report['n_converged']}/{report['n_total']}  "
          f"in {report['wall_seconds']:.1f}s")
    for i, v in enumerate(V_RHE_GRID):
        ok = report["converged"][i]
        cd = report["cd_mA_cm2"][i]
        pc = report["pc_mA_cm2"][i]
        cd_s = f"{cd:+.3e}" if cd is not None else "(none)"
        pc_s = f"{pc:+.3e}" if pc is not None else "(none)"
        method_s = report["method"][i]
        print(f"    V={v:+.3f}  ok={ok}  cd={cd_s}  pc={pc_s}  method={method_s}")
    print()

    iv_path = os.path.join(out_dir, "iv_curve.json")
    with open(iv_path, "w") as f:
        json.dump({
            "experiment_metadata": metadata_dict,
            "v_rhe": list(V_RHE_GRID),
            "pass": PASS_LABEL,
            "stern_f_m2": float(STERN_F_M2),
            "mesh_Ny": int(MESH_NY),
            "exponent_clip": float(EXPONENT_CLIP),
            "u_clamp": float(U_CLAMP),
            "initializer": INITIALIZER,
            "formulation": FORMULATION,
            "steric_cap": float(STERIC_CAP),
            "experimental_csv": EXPERIMENTAL_CSV,
            "report": {k: v for k, v in report.items() if k != "diagnostics"},
        }, f, indent=2)
    print(f"  iv_curve.json    -> {iv_path}")

    diag_path = os.path.join(out_dir, "diagnostics.json")
    with open(diag_path, "w") as f:
        json.dump({
            "experiment_metadata": metadata_dict,
            "v_rhe": list(V_RHE_GRID),
            "label": report["label"],
            "stern_f_m2": report["stern_f_m2"],
            "diagnostics_at_v": report["diagnostics"],
        }, f, indent=2, default=str)
    print(f"  diagnostics.json -> {diag_path}")

    exp_v, exp_j = _load_experimental_csv(EXPERIMENTAL_CSV)
    png_path = os.path.join(out_dir, "comparison.png")
    err = _make_plot(report, exp_v, exp_j, png_path)
    if err is None:
        print(f"  comparison.png   -> {png_path}")
    else:
        print(f"  plot skipped: {err}")

    elapsed = time.time() - t_start
    print()
    print("=" * 78)
    print(f"  Total wall time: {elapsed:.1f}s")
    v_arr = report["v_rhe"]
    ok_arr = report["converged"]
    v_max = max((vv for vv, kk in zip(v_arr, ok_arr) if kk), default=None)
    v_max_s = f"{v_max:+.3f} V" if v_max is not None else "(none)"
    print(f"  highest converged V_RHE = {v_max_s}  "
          f"({report['n_converged']}/{report['n_total']})")
    print("=" * 78)


if __name__ == "__main__":
    main()
