"""V26 — Test strategy B (grid_charge_continuation) on full V_RHE grid at Ny=200.

Earlier we saw `solve_grid_with_charge_continuation` (Phase 1 V-sweep at z=0,
Phase 2 per-V z-ramp) fail to bridge V=0 -> V=-0.10 in the 3sp+Boltzmann
log-c stack at Ny=80.  The hypothesis was that the failure was a mesh
resolution artifact rather than a fundamental incompatibility.  This
script reruns at production Ny=200 on the full target grid V_RHE in
[-0.5, +0.6] V to see if B converges everywhere with the boltzmann_z_scale
patch applied.

Run::

    ../venv-firedrake/bin/python scripts/studies/v26_strategy_b_full_grid.py

Output::

    StudyResults/v26_strategy_b_full_grid/
        results.csv     — per-V CD/PC/z observables and converged flag
        results.json    — same data + summary
        summary.md      — human-readable summary
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


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--v-grid", nargs="+", type=float,
        default=[-0.50, -0.40, -0.30, -0.20, -0.10,
                 0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
        help="V_RHE points (V).  Default = full target grid from Apr 27 writeup.",
    )
    p.add_argument("--mesh-ny", type=int, default=200)
    p.add_argument("--out-subdir", type=str, default="v26_strategy_b_full_grid")
    return p.parse_args()


def main():
    args = _parse_args()

    from scripts._bv_common import (
        setup_firedrake_env, V_T, I_SCALE,
        K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2,
        THREE_SPECIES_LOGC_BOLTZMANN,
        DEFAULT_CLO4_BOLTZMANN_COUNTERION,
        make_bv_solver_params,
        SNES_OPTS_CHARGED,
    )
    setup_firedrake_env()

    E_EQ_R1, E_EQ_R2 = 0.68, 1.78

    import firedrake as fd
    import firedrake.adjoint as adj
    from Forward.bv_solver import (
        make_graded_rectangle_mesh,
        solve_grid_with_charge_continuation,
    )
    from Forward.bv_solver.observables import _build_bv_observable_form

    V_GRID = np.array(args.v_grid, dtype=float)
    NV = len(V_GRID)

    OUT_DIR = os.path.join(_ROOT, "StudyResults", args.out_subdir)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 72)
    print("V26: strategy B (grid_charge_continuation) at production resolution")
    print("=" * 72)
    print(f"  V_GRID:   {V_GRID.tolist()}")
    print(f"  mesh Ny:  {args.mesh_ny}")
    print(f"  Output:   {OUT_DIR}")
    print()

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=int(args.mesh_ny), beta=3.0)

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

    sp = make_bv_solver_params(
        eta_hat=0.0, dt=0.25, t_end=80.0,
        species=THREE_SPECIES_LOGC_BOLTZMANN,
        snes_opts=snes_opts,
        formulation="logc",
        log_rate=True,
        boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION],
        k0_hat_r1=K0_HAT_R1, k0_hat_r2=K0_HAT_R2,
        alpha_r1=ALPHA_R1, alpha_r2=ALPHA_R2,
        E_eq_r1=E_EQ_R1, E_eq_r2=E_EQ_R2,
    )

    cd = np.full(NV, np.nan)
    pc = np.full(NV, np.nan)
    z_factor = np.full(NV, np.nan)
    converged_flags = np.zeros(NV, dtype=bool)

    def _grab(orig_idx, _phi, ctx):
        f_cd = _build_bv_observable_form(
            ctx, mode="current_density", reaction_index=None, scale=-I_SCALE)
        f_pc = _build_bv_observable_form(
            ctx, mode="peroxide_current", reaction_index=None, scale=-I_SCALE)
        cd[orig_idx] = float(fd.assemble(f_cd))
        pc[orig_idx] = float(fd.assemble(f_pc))

    phi_hat_grid = V_GRID / V_T
    t_start = time.time()
    with adj.stop_annotating():
        res = solve_grid_with_charge_continuation(
            sp,
            phi_applied_values=phi_hat_grid,
            charge_steps=20,
            mesh=mesh,
            max_eta_gap=2.0,
            min_delta_z=0.002,
            per_point_callback=_grab,
        )
    wall = time.time() - t_start

    for idx, point in res.points.items():
        z_factor[idx] = point.achieved_z_factor
        converged_flags[idx] = bool(point.converged)

    n_converged = int(converged_flags.sum())
    print()
    print(f"Converged at {n_converged}/{NV} points in {wall:.1f}s")
    for i, V in enumerate(V_GRID):
        tag = "OK" if converged_flags[i] else f"FAIL z={z_factor[i]:.3f}"
        cd_str = f"{cd[i]:+.6e}" if not np.isnan(cd[i]) else "(none)"
        pc_str = f"{pc[i]:+.6e}" if not np.isnan(pc[i]) else "(none)"
        print(f"  V={V:+.3f}  cd={cd_str}  pc={pc_str}  z={z_factor[i]:.3f}  {tag}")

    # ----------------------- persist -----------------------
    rows = []
    for i, V in enumerate(V_GRID):
        rows.append({
            "V_RHE": float(V),
            "cd": (None if np.isnan(cd[i]) else float(cd[i])),
            "pc": (None if np.isnan(pc[i]) else float(pc[i])),
            "z_factor": float(z_factor[i]),
            "converged": bool(converged_flags[i]),
        })
    summary = {
        "n_converged": n_converged,
        "n_total": NV,
        "wall_seconds": wall,
        "rel_window_full": (n_converged == NV),
    }

    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump({"config": {
            "v_grid": V_GRID.tolist(),
            "mesh_ny": args.mesh_ny,
            "K0_HAT_R1": K0_HAT_R1, "K0_HAT_R2": K0_HAT_R2,
            "ALPHA_R1": ALPHA_R1, "ALPHA_R2": ALPHA_R2,
            "E_EQ_R1": E_EQ_R1, "E_EQ_R2": E_EQ_R2,
        }, "rows": rows, "summary": summary}, f, indent=2)

    with open(os.path.join(OUT_DIR, "results.csv"), "w") as f:
        f.write("V_RHE,cd,pc,z_factor,converged\n")
        for r in rows:
            cd_s = "" if r["cd"] is None else f"{r['cd']:.8e}"
            pc_s = "" if r["pc"] is None else f"{r['pc']:.8e}"
            f.write(f"{r['V_RHE']:.4f},{cd_s},{pc_s},"
                    f"{r['z_factor']:.4f},{r['converged']}\n")

    md = []
    md.append("# V26 — strategy B (grid_charge_continuation) at Ny=200\n")
    md.append("Tests whether `solve_grid_with_charge_continuation` (Phase 1 "
              "V-sweep at z=0, Phase 2 per-V z-ramp) reaches the full Apr 27 "
              "target grid V_RHE in [-0.50, +0.60] V at production "
              "resolution after the `boltzmann_z_scale` patch.\n")
    md.append(f"## Setup\n")
    md.append(f"- V_GRID = {V_GRID.tolist()}")
    md.append(f"- mesh: graded rectangle Nx=8, Ny={args.mesh_ny}, beta=3.0")
    md.append(f"- TRUE: K0_HAT_R1={K0_HAT_R1:.6e}, K0_HAT_R2={K0_HAT_R2:.6e}, "
              f"α1={ALPHA_R1}, α2={ALPHA_R2}\n")
    md.append("## Per-voltage results\n")
    md.append("| V_RHE (V) | cd | pc | z_factor | converged |")
    md.append("|---:|---:|---:|---:|---|")
    for r in rows:
        cd_s = "(none)" if r["cd"] is None else f"{r['cd']:+.4e}"
        pc_s = "(none)" if r["pc"] is None else f"{r['pc']:+.4e}"
        md.append(f"| {r['V_RHE']:+.3f} | {cd_s} | {pc_s} | "
                  f"{r['z_factor']:.4f} | {r['converged']} |")
    md.append("")
    md.append("## Aggregate\n")
    md.append(f"- converged: {n_converged}/{NV}")
    md.append(f"- wall: {wall:.1f}s")
    md.append(f"- full target grid covered: {n_converged == NV}")
    md.append("")

    with open(os.path.join(OUT_DIR, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    print()
    print(f"Saved: {OUT_DIR}/{{summary.md, results.csv, results.json}}")


if __name__ == "__main__":
    main()
