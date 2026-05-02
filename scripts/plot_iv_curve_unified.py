"""Minimal I-V plot via the unified BV-PNP forward-solver API.

Generates CD (total current density) and PC (peroxide current) at the
TRUE kinetic parameters across V_RHE in [-0.5, +0.6] V using the
production 3sp + analytic Boltzmann counterion + log-c + log-rate BV
stack from writeups/WeekOfApr27/PNP Inverse Solver Revised.tex.

This script is intentionally linear and minimal so the whole forward
pipeline can be read top-to-bottom: param construction, mesh,
orchestrator, observable extraction, plot.

Run from the PNPInverse directory::

    ../venv-firedrake/bin/python scripts/plot_iv_curve_unified.py

Output (under StudyResults/iv_curve_unified/):

    iv_curve.png       — CD and PC vs V_RHE (mA/cm^2)
    iv_curve.csv       — V_RHE, cd_mA_cm2, pc_mA_cm2, z_factor, method
    iv_curve.json      — same data + config used to generate it
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(line_buffering=True)

# Standard PNPInverse path setup (script lives at scripts/, root is one up).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np


# ---------------------------------------------------------------------------
# Inputs (everything that controls the curve lives here)
# ---------------------------------------------------------------------------

# Full target grid from the Apr 27 writeup.
V_RHE_GRID = np.array([
    -0.50, -0.40, -0.30, -0.20, -0.10,
    0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60,
])
MESH_NY = 200          # production resolution (matches v18, v24, writeup)
OUT_SUBDIR = "iv_curve_unified"


def main():
    # -----------------------------------------------------------------------
    # 1. Constants, scales, and the production-stack factory
    #
    # Everything physical (D, c_bulk, V_T, I_SCALE, K0_HAT_*, ALPHA_*) is
    # centralised in scripts/_bv_common.py — single source of truth shared
    # with v13/v15/v16/v18/v24 etc.
    # -----------------------------------------------------------------------
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

    # E_eq for the two reactions (V vs RHE).
    E_EQ_R1, E_EQ_R2 = 0.68, 1.78

    # -----------------------------------------------------------------------
    # 2. Forward-solver entry points
    #
    # The dispatcher in Forward.bv_solver routes build_context/build_forms/
    # set_initial_conditions to forms_logc.py (vs forms.py) based on
    # params['bv_convergence']['formulation'].
    #
    # solve_grid_per_voltage_cold_with_warm_fallback is the C+D orchestrator
    # documented in docs/bv_solver_unified_api.md: per-V cold + internal
    # z-ramp, then warm-walk from cold anchors with paf substepping +
    # bisection for the cathodic / anodic edges that don't cold-converge.
    # -----------------------------------------------------------------------
    import firedrake as fd
    import firedrake.adjoint as adj
    from Forward.bv_solver import (
        make_graded_rectangle_mesh,
        solve_grid_per_voltage_cold_with_warm_fallback,
    )
    from Forward.bv_solver.observables import _build_bv_observable_form

    OUT_DIR = os.path.join(_ROOT, "StudyResults", OUT_SUBDIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 72)
    print("I-V curve via unified BV-PNP API")
    print("=" * 72)
    print(f"  V_RHE grid:  {V_RHE_GRID.tolist()}")
    print(f"  mesh Ny:     {MESH_NY}")
    print(f"  TRUE k0/α:   k0_R1={K0_HAT_R1:.6e}, k0_R2={K0_HAT_R2:.6e}, "
          f"α1={ALPHA_R1}, α2={ALPHA_R2}")
    print(f"  E_eq R1/R2:  {E_EQ_R1} / {E_EQ_R2} V (RHE)")
    print(f"  Output:      {OUT_DIR}")
    print()

    # -----------------------------------------------------------------------
    # 3. Mesh (graded rectangle, Nx=8, Ny=200, beta=3 — production default)
    # -----------------------------------------------------------------------
    mesh = make_graded_rectangle_mesh(Nx=8, Ny=int(MESH_NY), beta=3.0)

    # -----------------------------------------------------------------------
    # 4. Solver params for the production stack
    #
    # The three new flags drive the dispatcher and the residual:
    #     formulation="logc"                 -> forms_logc.py backend
    #     log_rate=True                      -> bv_log_rate in residual
    #     boltzmann_counterions=[ClO4-]      -> Boltzmann residual in Poisson
    # -----------------------------------------------------------------------
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
        formulation="logc",
        log_rate=True,
        boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION],
        k0_hat_r1=K0_HAT_R1, k0_hat_r2=K0_HAT_R2,
        alpha_r1=ALPHA_R1,   alpha_r2=ALPHA_R2,
        E_eq_r1=E_EQ_R1,     E_eq_r2=E_EQ_R2,
    )

    # -----------------------------------------------------------------------
    # 5. Observable callback
    #
    # The orchestrator invokes this once per converged voltage with the
    # live ctx, so we can assemble CD and PC before the next voltage
    # rebuilds the forms.  Sign convention: CD < 0 for cathodic ORR
    # currents because we multiply the assembled rate by -I_SCALE
    # (the writeup and the inverse scripts use the same sign).
    # -----------------------------------------------------------------------
    NV = len(V_RHE_GRID)
    cd_nondim = np.full(NV, np.nan)
    pc_nondim = np.full(NV, np.nan)
    z_achieved = np.full(NV, np.nan)
    method_at_v = ["MISSING"] * NV

    def _grab_observables(orig_idx, _phi_eta, ctx):
        f_cd = _build_bv_observable_form(
            ctx, mode="current_density",  reaction_index=None, scale=-I_SCALE)
        f_pc = _build_bv_observable_form(
            ctx, mode="peroxide_current", reaction_index=None, scale=-I_SCALE)
        cd_nondim[orig_idx] = float(fd.assemble(f_cd))
        pc_nondim[orig_idx] = float(fd.assemble(f_pc))

    # -----------------------------------------------------------------------
    # 6. Run the orchestrator
    # -----------------------------------------------------------------------
    phi_hat_grid = V_RHE_GRID / V_T
    t_start = time.time()
    with adj.stop_annotating():
        result = solve_grid_per_voltage_cold_with_warm_fallback(
            sp,
            phi_applied_values=phi_hat_grid,
            mesh=mesh,
            max_z_steps=20,
            n_substeps_warm=4,
            bisect_depth_warm=3,
            per_point_callback=_grab_observables,
        )
    wall = time.time() - t_start

    for idx, point in result.points.items():
        z_achieved[idx] = point.achieved_z_factor
        method_at_v[idx] = point.method

    n_ok = int(np.sum(~np.isnan(cd_nondim)))
    print()
    print(f"Converged at {n_ok}/{NV} points in {wall:.1f}s")

    # -----------------------------------------------------------------------
    # 7. Persist (CSV + JSON) and plot
    #
    # The observable_form already includes the -I_SCALE factor, so cd_nondim
    # is already in mA/cm^2 — no further conversion needed.
    # -----------------------------------------------------------------------
    cd_ma_cm2 = cd_nondim   # observable already includes -I_SCALE
    pc_ma_cm2 = pc_nondim

    rows = []
    for i, V in enumerate(V_RHE_GRID):
        rows.append({
            "V_RHE":      float(V),
            "cd_mA_cm2":  None if np.isnan(cd_ma_cm2[i]) else float(cd_ma_cm2[i]),
            "pc_mA_cm2":  None if np.isnan(pc_ma_cm2[i]) else float(pc_ma_cm2[i]),
            "z_factor":   float(z_achieved[i]),
            "method":     method_at_v[i],
        })

    with open(os.path.join(OUT_DIR, "iv_curve.csv"), "w") as f:
        f.write("V_RHE,cd_mA_cm2,pc_mA_cm2,z_factor,method\n")
        for r in rows:
            cd_s = "" if r["cd_mA_cm2"] is None else f"{r['cd_mA_cm2']:.8e}"
            pc_s = "" if r["pc_mA_cm2"] is None else f"{r['pc_mA_cm2']:.8e}"
            f.write(
                f"{r['V_RHE']:.4f},{cd_s},{pc_s},"
                f"{r['z_factor']:.4f},{r['method']}\n"
            )

    with open(os.path.join(OUT_DIR, "iv_curve.json"), "w") as f:
        json.dump({
            "config": {
                "v_rhe_grid":  V_RHE_GRID.tolist(),
                "mesh_ny":     MESH_NY,
                "K0_HAT_R1":   K0_HAT_R1, "K0_HAT_R2": K0_HAT_R2,
                "ALPHA_R1":    ALPHA_R1,  "ALPHA_R2":  ALPHA_R2,
                "E_EQ_R1":     E_EQ_R1,   "E_EQ_R2":   E_EQ_R2,
                "I_SCALE":     I_SCALE,   "V_T":       V_T,
                "formulation": "logc",
                "log_rate":    True,
                "boltzmann_counterion": DEFAULT_CLO4_BOLTZMANN_COUNTERION,
                "wall_seconds": wall,
            },
            "rows": rows,
        }, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.0), sharex=True)

        ax = axes[0]
        ax.plot(V_RHE_GRID, cd_ma_cm2, "b-o", markersize=6, label="CD (3sp+B logc)")
        ax.set_ylabel("Total current density (mA/cm²)")
        ax.set_title(
            f"I-V curve at TRUE parameters via unified API\n"
            f"(formulation=logc, log_rate=True, Boltzmann ClO4-, Ny={MESH_NY})"
        )
        ax.grid(True, alpha=0.3); ax.legend()

        ax = axes[1]
        ax.plot(V_RHE_GRID, pc_ma_cm2, "r-s", markersize=6, label="PC (3sp+B logc)")
        ax.set_xlabel("V vs RHE (V)")
        ax.set_ylabel("Peroxide current density (mA/cm²)")
        ax.grid(True, alpha=0.3); ax.legend()

        plt.tight_layout()
        plot_path = os.path.join(OUT_DIR, "iv_curve.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  plot: {plot_path}")
    except Exception as exc:
        print(f"  plot skipped: {exc}")

    # -----------------------------------------------------------------------
    # 8. Summary
    # -----------------------------------------------------------------------
    print()
    for r in rows:
        cd_s = "(none)" if r["cd_mA_cm2"] is None else f"{r['cd_mA_cm2']:+.6e}"
        pc_s = "(none)" if r["pc_mA_cm2"] is None else f"{r['pc_mA_cm2']:+.6e}"
        print(
            f"  V={r['V_RHE']:+.3f}  cd={cd_s}  pc={pc_s}  "
            f"z={r['z_factor']:.3f}  {r['method']}"
        )

    print()
    print(f"Saved: {OUT_DIR}/{{iv_curve.png, iv_curve.csv, iv_curve.json}}")


if __name__ == "__main__":
    main()
