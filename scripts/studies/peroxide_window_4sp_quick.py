"""Quick test: 4sp dynamic ClO4- in the peroxide window, with/without Stern.

Companion to ``scripts/studies/peroxide_window_stern_test.py`` (3sp +
Boltzmann + Stern).  Hypothesis: dropping the analytic Boltzmann
counterion in favour of dynamic ClO4- as a 4th NP species, combined
with finite Stern capacitance, gives a fully physical solution in the
peroxide window — c_ClO4 capped by Bikerman steric (a=0.01 → cap ~100
nondim) at the surface, no exp() blow-up.

Three passes:

    (1)  4sp + linear_phi  + no Stern        — baseline
    (2)  4sp + linear_phi  + Stern C_S=0.10  — moderate compact-layer drop
    (3)  4sp + linear_phi  + Stern C_S=0.05  — small C_S, larger Stern drop

``debye_boltzmann`` initializer would silently fall back to ``linear_phi``
for 4sp dynamic (it requires a Boltzmann counterion config).  We pass
``linear_phi`` explicitly to avoid the indirection.

V_RHE grid: [0.60, 0.66, 0.68, 0.70, 0.75, 0.80, 1.00] (peroxide window).

Output: StudyResults/peroxide_window_4sp_quick/
"""
from __future__ import annotations

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


V_TEST = (0.60, 0.66, 0.68, 0.70, 0.75, 0.80, 1.00)
PASSES = (
    ("4sp_no_stern",  None),
    ("4sp_stern_0p10", 0.10),
    ("4sp_stern_0p05", 0.05),
)
MESH_NY = 200
EXPONENT_CLIP = 100.0
N_SUBSTEPS_WARM = 8
BISECT_DEPTH_WARM = 5
STERIC_CAP = 100.0
INITIALIZER = "linear_phi"
OUT_SUBDIR = "peroxide_window_4sp_quick"


def _run_one_pass(
    label: str,
    cs: Optional[float],
    *,
    v_rhe_grid,
    mesh_ny: int,
    exponent_clip: float,
) -> dict[str, Any]:
    from scripts._bv_common import (
        setup_firedrake_env,
        V_T, I_SCALE,
        K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2,
        FOUR_SPECIES_LOGC_DYNAMIC,
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
        species=FOUR_SPECIES_LOGC_DYNAMIC,
        snes_opts=snes_opts,
        formulation="logc", log_rate=True,
        boltzmann_counterions=None,            # ClO4- is dynamic, not analytic
        stern_capacitance_f_m2=cs,
        k0_hat_r1=K0_HAT_R1, k0_hat_r2=K0_HAT_R2,
        alpha_r1=ALPHA_R1, alpha_r2=ALPHA_R2,
        E_eq_r1=0.68, E_eq_r2=1.78,
        initializer=INITIALIZER,
    )
    new_opts = dict(sp.solver_options)
    new_bv = dict(new_opts["bv_convergence"])
    new_bv["exponent_clip"] = float(exponent_clip)
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

    converged_flags = [bool(result.points[i].converged) for i in range(NV)]
    methods = [result.points[i].method for i in range(NV)]
    z_achieved = [float(result.points[i].achieved_z_factor) for i in range(NV)]
    diagnostics_per_v = [result.points[i].diagnostics for i in range(NV)]

    return {
        "label": label,
        "cs_f_m2": cs,
        "wall_seconds": float(elapsed),
        "v_rhe": list(v_rhe_grid),
        "phi_applied_hat": [float(x) for x in phi_hat_grid.tolist()],
        "cd_mA_cm2": [float(x) if np.isfinite(x) else None for x in cd],
        "pc_mA_cm2": [float(x) if np.isfinite(x) else None for x in pc],
        "converged": converged_flags,
        "method": methods,
        "z_achieved": z_achieved,
        "diagnostics": diagnostics_per_v,
        "n_converged": int(sum(converged_flags)),
        "n_total": int(NV),
    }


def _make_plot(reports: list[dict], png_path: str) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return f"matplotlib unavailable: {exc}"

    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(3, 1, hspace=0.32)
    ax_cd = fig.add_subplot(gs[0])
    ax_pc = fig.add_subplot(gs[1], sharex=ax_cd)
    ax_clo4 = fig.add_subplot(gs[2], sharex=ax_cd)

    cmap = plt.get_cmap("viridis")
    colors = ["k"] + [cmap(0.3), cmap(0.7)]
    markers = ["o", "s", "^"]

    for idx, r in enumerate(reports):
        v = np.asarray(r["v_rhe"])
        cd = np.array([np.nan if x is None else x for x in r["cd_mA_cm2"]],
                      dtype=float)
        pc = np.array([np.nan if x is None else x for x in r["pc_mA_cm2"]],
                      dtype=float)
        # Surface c_ClO4 = c3_surface_mean (4sp: idx 3 is ClO4-)
        c_clo4 = []
        for d in r["diagnostics"]:
            d = d or {}
            v_clo4 = d.get("c3_surface_mean")
            c_clo4.append(np.nan if v_clo4 is None else v_clo4)
        c_clo4 = np.array(c_clo4, dtype=float)
        label = r["label"]
        ax_cd.plot(v, cd, marker=markers[idx], color=colors[idx], ls="-",
                   label=label)
        ax_pc.plot(v, pc, marker=markers[idx], color=colors[idx], ls="-",
                   label=label)
        ax_clo4.plot(v, c_clo4, marker=markers[idx], color=colors[idx], ls="-",
                     label=label)

    for ax in (ax_cd, ax_pc, ax_clo4):
        ax.axvline(0.68, color="green", ls="--", lw=0.8, alpha=0.6)
        ax.grid(True, alpha=0.3)

    ax_cd.set_ylabel("CD (mA/cm²)")
    ax_cd.set_title(
        "4sp dynamic ClO4- — peroxide-window quick test\n"
        f"(Ny={MESH_NY}, clip={EXPONENT_CLIP:.0f}, "
        f"initializer={INITIALIZER}, "
        "dashed line: E_eq_R1=+0.68 V)"
    )
    ax_cd.legend(fontsize=8, loc="best")

    ax_pc.set_ylabel("PC (mA/cm²) [symlog]")
    ax_pc.set_yscale("symlog", linthresh=1e-6)
    ax_pc.legend(fontsize=8, loc="best")

    ax_clo4.set_ylabel("Surface c_ClO4 (nondim)")
    ax_clo4.set_yscale("log")
    ax_clo4.axhline(STERIC_CAP, color="red", ls=":", lw=1.0, alpha=0.7,
                    label=f"Bikerman cap ~{STERIC_CAP:g}")
    ax_clo4.set_xlabel("V vs RHE (V)")
    ax_clo4.legend(fontsize=8, loc="best")

    plt.savefig(png_path, dpi=160, bbox_inches="tight")
    plt.close()
    return None


def main() -> None:
    out_dir = os.path.join(_ROOT, "StudyResults", OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 78)
    print("  Peroxide-window 4sp dynamic — quick test (no Boltzmann)")
    print("=" * 78)
    print(f"  V_RHE grid       = {list(V_TEST)}")
    print(f"  passes           = {[p[0] for p in PASSES]}")
    print(f"  mesh_Ny          = {MESH_NY}")
    print(f"  exponent_clip    = {EXPONENT_CLIP}")
    print(f"  initializer      = {INITIALIZER}")
    print(f"  steric (a=0.01)  = ON (cap ~{STERIC_CAP:g} nondim)")
    print(f"  output           = {out_dir}")
    print()

    reports: list[dict] = []
    t_start = time.time()
    for label, cs in PASSES:
        cs_str = "None" if cs is None else f"{cs:g}"
        print(f"--- pass: {label} (C_S={cs_str} F/m²) ---")
        r = _run_one_pass(label, cs, v_rhe_grid=list(V_TEST),
                          mesh_ny=MESH_NY, exponent_clip=EXPONENT_CLIP)
        reports.append(r)
        print(f"  converged {r['n_converged']}/{r['n_total']}  "
              f"in {r['wall_seconds']:.1f}s")
        for i, v in enumerate(V_TEST):
            ok = r["converged"][i]
            cd = r["cd_mA_cm2"][i]
            pc = r["pc_mA_cm2"][i]
            cd_s = f"{cd:+.3e}" if cd is not None else "(none)"
            pc_s = f"{pc:+.3e}" if pc is not None else "(none)"
            d = r["diagnostics"][i] or {}
            c_clo4 = d.get("c3_surface_mean")
            c_clo4_s = f"{c_clo4:.2e}" if c_clo4 is not None else "n/a"
            phi_s = d.get("phi_surface_mean")
            phi_app = r["phi_applied_hat"][i]
            drop_str = (f"{phi_app - phi_s:+.2f}"
                        if phi_s is not None else "n/a")
            print(f"    V={v:+.3f}  ok={ok}  cd={cd_s}  pc={pc_s}  "
                  f"method={r['method'][i]}  c_ClO4={c_clo4_s}  "
                  f"stern_drop={drop_str}")
        print()

    iv_path = os.path.join(out_dir, "iv_curve.json")
    with open(iv_path, "w") as f:
        json.dump({
            "v_rhe": list(V_TEST),
            "passes": [p[0] for p in PASSES],
            "cs_f_m2": [None if p[1] is None else float(p[1]) for p in PASSES],
            "mesh_Ny": int(MESH_NY),
            "exponent_clip": float(EXPONENT_CLIP),
            "initializer": INITIALIZER,
            "steric_cap": float(STERIC_CAP),
            "reports": [
                {k: v for k, v in r.items() if k != "diagnostics"}
                for r in reports
            ],
        }, f, indent=2)
    print(f"  iv_curve.json    -> {iv_path}")

    diag_path = os.path.join(out_dir, "diagnostics.json")
    with open(diag_path, "w") as f:
        json.dump({
            "v_rhe": list(V_TEST),
            "reports": [
                {
                    "label": r["label"],
                    "cs_f_m2": r["cs_f_m2"],
                    "diagnostics_at_v": r["diagnostics"],
                }
                for r in reports
            ],
        }, f, indent=2, default=str)
    print(f"  diagnostics.json -> {diag_path}")

    png_path = os.path.join(out_dir, "comparison.png")
    err = _make_plot(reports, png_path)
    if err is None:
        print(f"  comparison.png   -> {png_path}")
    else:
        print(f"  plot skipped: {err}")

    elapsed = time.time() - t_start
    print()
    print("=" * 78)
    print(f"  Total wall time: {elapsed:.1f}s")
    print(f"  Convergence summary:")
    for r in reports:
        print(f"    {r['label']:24s}  {r['n_converged']}/{r['n_total']}")
    print("=" * 78)


if __name__ == "__main__":
    main()
