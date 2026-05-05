"""Peroxide-window extension test: linear_phi vs debye_boltzmann initializer.

Goal: see whether the matched-asymptotic Debye-Boltzmann initial condition
unblocks ``V_RHE >= +0.68 V`` cold-start convergence in the production C+D
orchestrator.  Per ``docs/peroxide_window_investigation.md`` and
``docs/Peroxide Solver Convergence.md``, the existing linear-phi IC fails
at every voltage in the peroxide window (E_eq_R1 = +0.68 V).  The
analytical IC seeds Newton on the depleted-H+/enriched-counterion
manifold so the basin-entry transition does not have to be discovered
in a single Newton step.

Voltage grid (small, focused):

    V_TEST = [+0.66, +0.68, +0.70, +0.75, +0.80, +1.00]

Both initializers are run end-to-end through
``solve_grid_per_voltage_cold_with_warm_fallback`` at production
resolution (Ny=200, exponent_clip=100, n_substeps_warm=8,
bisect_depth_warm=5).  Outputs:

    StudyResults/peroxide_window_pb_init_test/iv_curve.json
        per-IC CD, PC, converged flag, method, Picard iters,
        surface c_ClO4 / c_H / c_H2O2.

    StudyResults/peroxide_window_pb_init_test/diagnostics.json
        full per-voltage diagnostics dump from Commit 1
        (max phi, surface fields, snes_reason, snes_iters,
        steric watch, fallback flags).

    StudyResults/peroxide_window_pb_init_test/comparison.png
        CD / PC vs V curves overlaid for both initializers.

Acceptable outcomes (per the plan §3 Commit 3 menu):

* **Best:** Newton converges at all V_TEST and surface c_ClO4 stays
  below the Bikerman steric cap (~100 nondim).  Unlikely without Stern.
* **Likely:** Newton converges but surface c_ClO4 exceeds the cap.
  IC + reduced model is a valid inverse-problem predictor; the converged
  observable is non-physical for the no-Stern model.
* **Worst:** Newton diverges at +0.68 V even with the new IC.  Stern
  test (skipped in this commit) becomes the necessary next step.

Run from PNPInverse/ with ../venv-firedrake/bin/activate active:

    python scripts/studies/peroxide_window_pb_init_test.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(line_buffering=True)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np


V_TEST = np.array([0.66, 0.68, 0.70, 0.75, 0.80, 1.00])
MESH_NY = 200
EXPONENT_CLIP = 100.0
N_SUBSTEPS_WARM = 8
BISECT_DEPTH_WARM = 5
STERIC_CAP = 100.0
INITIALIZERS = ("linear_phi", "debye_boltzmann")
OUT_SUBDIR = "peroxide_window_pb_init_test"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mesh-ny", type=int, default=MESH_NY,
        help=f"Graded mesh Ny (default {MESH_NY}).",
    )
    p.add_argument(
        "--clip", type=float, default=EXPONENT_CLIP,
        help=f"BV exponent_clip (default {EXPONENT_CLIP}).",
    )
    p.add_argument(
        "--initializers", type=str, default=",".join(INITIALIZERS),
        help=("Comma-separated list of initializers to compare "
              f"(default '{','.join(INITIALIZERS)}')."),
    )
    return p.parse_args()


def _run_one_initializer(initializer: str, *, mesh_ny: int, exponent_clip: float):
    """Cold + warm-walk solve at V_TEST with the given initializer flag.

    Returns a dict of arrays / per-voltage diagnostic records.
    """
    from scripts._bv_common import (
        setup_firedrake_env,
        V_T, I_SCALE,
        K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2,
        THREE_SPECIES_LOGC_BOLTZMANN,
        DEFAULT_CLO4_BOLTZMANN_COUNTERION,
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
        boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION],
        k0_hat_r1=K0_HAT_R1, k0_hat_r2=K0_HAT_R2,
        alpha_r1=ALPHA_R1, alpha_r2=ALPHA_R2,
        E_eq_r1=0.68, E_eq_r2=1.78,
        initializer=initializer,
    )
    new_opts = dict(sp.solver_options)
    new_bv = dict(new_opts["bv_convergence"])
    new_bv["exponent_clip"] = float(exponent_clip)
    new_opts["bv_convergence"] = new_bv
    sp = sp.with_solver_options(new_opts)

    NV = len(V_TEST)
    cd = np.full(NV, np.nan)
    pc = np.full(NV, np.nan)

    def _grab(orig_idx, _phi_eta, ctx):
        f_cd = _build_bv_observable_form(
            ctx, mode="current_density", reaction_index=None, scale=-I_SCALE)
        f_pc = _build_bv_observable_form(
            ctx, mode="peroxide_current", reaction_index=None, scale=-I_SCALE)
        cd[orig_idx] = float(fd.assemble(f_cd))
        pc[orig_idx] = float(fd.assemble(f_pc))

    phi_hat_grid = V_TEST / V_T
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

    converged_flags = [bool(result.points[i].converged) for i in range(NV)]
    methods = [result.points[i].method for i in range(NV)]
    z_achieved = [float(result.points[i].achieved_z_factor) for i in range(NV)]
    diagnostics_per_v = [result.points[i].diagnostics for i in range(NV)]

    return {
        "initializer": initializer,
        "wall_seconds": float(elapsed),
        "v_rhe": V_TEST.tolist(),
        "cd_mA_cm2": [float(x) if np.isfinite(x) else None for x in cd],
        "pc_mA_cm2": [float(x) if np.isfinite(x) else None for x in pc],
        "converged": converged_flags,
        "method": methods,
        "z_achieved": z_achieved,
        "diagnostics": diagnostics_per_v,
        "n_converged": int(sum(converged_flags)),
        "n_total": int(NV),
    }


def _summarize_outcome(reports: list[dict]) -> str:
    """Apply the Commit 3 success-criteria menu and return a label."""
    by_init = {r["initializer"]: r for r in reports}
    debye = by_init.get("debye_boltzmann")
    if debye is None:
        return "INDETERMINATE: no debye_boltzmann run"

    debye_idx_map = {round(v, 4): i for i, v in enumerate(V_TEST)}
    idx_068 = debye_idx_map.get(0.68)
    idx_070 = debye_idx_map.get(0.70)
    idx_080 = debye_idx_map.get(0.80)
    idx_100 = debye_idx_map.get(1.00)

    converged = debye["converged"]
    diags = debye["diagnostics"]

    def _within_steric(idx: int) -> bool | None:
        if idx is None or idx >= len(diags) or diags[idx] is None:
            return None
        d = diags[idx] or {}
        return bool(d.get("surface_counterion_within_steric", False))

    if idx_068 is None or not converged[idx_068]:
        return ("WORST: debye_boltzmann fails at V=+0.68 V; Stern test "
                "(skipped in this commit) is the necessary next step.")

    deeper_ok = any(
        i is not None and i < len(converged) and converged[i]
        for i in (idx_080, idx_100)
    )
    if idx_070 is None or not converged[idx_070] or not deeper_ok:
        return ("PARTIAL: V=+0.68 V converges but V=+0.70 V or all of "
                "{+0.80, +1.00} V do not.  IC unblocks the immediate "
                "wall but not the full peroxide window.")

    steric_ok = all(
        _within_steric(i) is True
        for i in (idx_068, idx_070, idx_080, idx_100)
        if i is not None and i < len(converged) and converged[i]
    )
    if steric_ok:
        return ("BEST: all converged peroxide-window points are within "
                "Bikerman steric cap.  IC is the production answer.")
    return ("LIKELY: peroxide-window points converge but surface c_ClO4 "
            "exceeds Bikerman cap (~100 nondim).  IC + reduced model is "
            "a valid inverse-problem predictor; converged observable is "
            "non-physical without Stern saturation.")


def _make_comparison_plot(reports: list[dict], png_path: str) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return f"matplotlib unavailable: {exc}"

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 1, hspace=0.3)
    ax_cd = fig.add_subplot(gs[0])
    ax_pc = fig.add_subplot(gs[1], sharex=ax_cd)

    markers = {"linear_phi": "o", "debye_boltzmann": "s"}
    colors = {"linear_phi": "#377eb8", "debye_boltzmann": "#e41a1c"}

    for report in reports:
        init = report["initializer"]
        v = np.asarray(report["v_rhe"])
        cd = np.array(
            [np.nan if x is None else x for x in report["cd_mA_cm2"]],
            dtype=float,
        )
        pc = np.array(
            [np.nan if x is None else x for x in report["pc_mA_cm2"]],
            dtype=float,
        )
        ax_cd.plot(v, cd, marker=markers[init], color=colors[init],
                   linestyle="-", label=f"CD, init={init}")
        ax_pc.plot(v, pc, marker=markers[init], color=colors[init],
                   linestyle="-", label=f"PC, init={init}")

    for ax in (ax_cd, ax_pc):
        ax.axvline(0.68, color="green", ls="--", lw=0.8, alpha=0.6,
                   label="E_eq_R1 = +0.68 V")
        ax.grid(True, alpha=0.3)

    ax_cd.set_ylabel("CD (mA/cm²)")
    ax_cd.set_title(
        f"Peroxide-window IC test  (Ny={MESH_NY}, clip={EXPONENT_CLIP:.0f}, "
        f"steric cap = {STERIC_CAP:.0f})\n"
        "Steric watch: in no-Stern mode, surface c_ClO4 = c_bulk·exp(phi_s) "
        "exceeds Bikerman cap past ~+0.16 V; converged values past that "
        "voltage are non-physical despite Newton convergence."
    )
    ax_cd.legend(fontsize=8, loc="best")
    ax_pc.set_yscale("symlog", linthresh=1e-6)
    ax_pc.set_xlabel("V vs RHE (V)")
    ax_pc.set_ylabel("PC (mA/cm²) [symlog]")
    ax_pc.legend(fontsize=8, loc="best")

    plt.savefig(png_path, dpi=160, bbox_inches="tight")
    plt.close()
    return None


def main() -> None:
    cli = _parse_args()
    inits = [s.strip() for s in cli.initializers.split(",") if s.strip()]
    OUT_DIR = os.path.join(_ROOT, "StudyResults", OUT_SUBDIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 78)
    print("  Peroxide-window IC test: linear_phi vs debye_boltzmann")
    print("=" * 78)
    print(f"  V_TEST              = {V_TEST.tolist()}")
    print(f"  initializers        = {inits}")
    print(f"  exponent_clip       = {cli.clip}")
    print(f"  n_substeps_warm     = {N_SUBSTEPS_WARM}")
    print(f"  bisect_depth_warm   = {BISECT_DEPTH_WARM}")
    print(f"  mesh_Ny             = {cli.mesh_ny}")
    print(f"  output              = {OUT_DIR}")
    print()

    reports: list[dict] = []
    for init in inits:
        print(f"--- pass: initializer={init} ---")
        report = _run_one_initializer(
            init, mesh_ny=cli.mesh_ny, exponent_clip=cli.clip,
        )
        reports.append(report)
        print(f"  converged {report['n_converged']}/{report['n_total']}  "
              f"in {report['wall_seconds']:.1f}s")
        for i, v in enumerate(V_TEST):
            ok = report["converged"][i]
            cd = report["cd_mA_cm2"][i]
            pc = report["pc_mA_cm2"][i]
            cd_s = f"{cd:+.4e}" if cd is not None else "(none)"
            pc_s = f"{pc:+.4e}" if pc is not None else "(none)"
            method = report["method"][i]
            d = report["diagnostics"][i] or {}
            steric = d.get("surface_counterion_within_steric", None)
            picard = d.get("picard_iters", None)
            fb = d.get("initializer_fallback", None)
            print(f"    V={v:+.3f}  ok={ok}  cd={cd_s}  pc={pc_s}  "
                  f"method={method}  steric_ok={steric}  "
                  f"picard={picard}  fallback={fb}")
        print()

    iv_path = os.path.join(OUT_DIR, "iv_curve.json")
    with open(iv_path, "w") as f:
        json.dump({
            "v_rhe": V_TEST.tolist(),
            "exponent_clip": float(cli.clip),
            "mesh_Ny": int(cli.mesh_ny),
            "n_substeps_warm": int(N_SUBSTEPS_WARM),
            "bisect_depth_warm": int(BISECT_DEPTH_WARM),
            "steric_cap": float(STERIC_CAP),
            "reports": [
                {k: v for k, v in r.items() if k != "diagnostics"}
                for r in reports
            ],
        }, f, indent=2)
    print(f"  iv_curve.json -> {iv_path}")

    diag_path = os.path.join(OUT_DIR, "diagnostics.json")
    with open(diag_path, "w") as f:
        json.dump({
            "v_rhe": V_TEST.tolist(),
            "reports": [
                {
                    "initializer": r["initializer"],
                    "diagnostics_at_v": r["diagnostics"],
                }
                for r in reports
            ],
        }, f, indent=2)
    print(f"  diagnostics.json -> {diag_path}")

    png_path = os.path.join(OUT_DIR, "comparison.png")
    err = _make_comparison_plot(reports, png_path)
    if err is None:
        print(f"  comparison.png -> {png_path}")
    else:
        print(f"  plot skipped: {err}")

    print()
    outcome = _summarize_outcome(reports)
    print("=" * 78)
    print(f"  OUTCOME: {outcome}")
    print("=" * 78)


if __name__ == "__main__":
    main()
