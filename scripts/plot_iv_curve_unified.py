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
import argparse

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
DEFAULT_V_RHE_GRID = np.array([
    -0.50, -0.40, -0.30, -0.20, -0.10,
    0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60,
])
DEFAULT_MESH_NY = 200  # production resolution (matches v18, v24, writeup)
DEFAULT_OUT_SUBDIR = "iv_curve_unified"


def _parse_v_grid(text: str | None) -> np.ndarray:
    if text is None or not text.strip():
        return DEFAULT_V_RHE_GRID.copy()
    values = [float(tok.strip()) for tok in text.split(",") if tok.strip()]
    if not values:
        raise ValueError("--v-list did not contain any voltages")
    return np.array(values, dtype=float)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--v-list",
        default=None,
        help="Comma-separated V_RHE grid in volts. Defaults to the production grid.",
    )
    p.add_argument(
        "--mesh-ny",
        type=int,
        default=DEFAULT_MESH_NY,
        help=f"Graded mesh Ny (default {DEFAULT_MESH_NY}).",
    )
    p.add_argument(
        "--formulation",
        choices=("logc", "logc_muh"),
        default="logc",
        help="Weak-form backend (default logc).",
    )
    p.add_argument(
        "--counterion-mode",
        choices=("ideal", "steric"),
        default="ideal",
        help="Analytic ClO4- counterion closure (default ideal).",
    )
    p.add_argument(
        "--initializer",
        choices=("linear_phi", "debye_boltzmann"),
        default="linear_phi",
        help="Initial condition used by the dispatcher (default linear_phi).",
    )
    p.add_argument(
        "--exponent-clip",
        type=float,
        default=None,
        help="Override bv_convergence.exponent_clip. Defaults to the factory value.",
    )
    p.add_argument(
        "--stern-capacitance",
        type=float,
        default=None,
        help="Finite Stern capacitance in F/m^2. Defaults to no Stern layer.",
    )
    p.add_argument(
        "--out-subdir",
        default=DEFAULT_OUT_SUBDIR,
        help=f"StudyResults subdirectory (default {DEFAULT_OUT_SUBDIR}).",
    )
    args, passthrough = p.parse_known_args()
    sys.argv = [sys.argv[0], *passthrough]
    return args


def main():
    args = _parse_args()
    v_rhe_grid = _parse_v_grid(args.v_list)

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
        DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC,
        SNES_OPTS_CHARGED,
        make_bv_solver_params,
    )
    setup_firedrake_env()
    counterion = (
        DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC
        if args.counterion_mode == "steric"
        else DEFAULT_CLO4_BOLTZMANN_COUNTERION
    )

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

    OUT_DIR = os.path.join(_ROOT, "StudyResults", args.out_subdir)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 72)
    print("I-V curve via unified BV-PNP API")
    print("=" * 72)
    print(f"  V_RHE grid:  {v_rhe_grid.tolist()}")
    print(f"  mesh Ny:     {args.mesh_ny}")
    print(f"  formulation: {args.formulation}")
    print(f"  counterion:  {args.counterion_mode}")
    print(f"  initializer: {args.initializer}")
    print(f"  exp clip:    {args.exponent_clip if args.exponent_clip is not None else '(factory default)'}")
    print(f"  Stern C:     {args.stern_capacitance if args.stern_capacitance is not None else '(none)'}")
    print(f"  TRUE k0/α:   k0_R1={K0_HAT_R1:.6e}, k0_R2={K0_HAT_R2:.6e}, "
          f"α1={ALPHA_R1}, α2={ALPHA_R2}")
    print(f"  E_eq R1/R2:  {E_EQ_R1} / {E_EQ_R2} V (RHE)")
    print(f"  Output:      {OUT_DIR}")
    print()

    # -----------------------------------------------------------------------
    # 3. Mesh (graded rectangle, Nx=8, Ny=200, beta=3 — production default)
    # -----------------------------------------------------------------------
    mesh = make_graded_rectangle_mesh(Nx=8, Ny=int(args.mesh_ny), beta=3.0)

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
        formulation=args.formulation,
        log_rate=True,
        boltzmann_counterions=[counterion],
        k0_hat_r1=K0_HAT_R1, k0_hat_r2=K0_HAT_R2,
        alpha_r1=ALPHA_R1,   alpha_r2=ALPHA_R2,
        E_eq_r1=E_EQ_R1,     E_eq_r2=E_EQ_R2,
        initializer=args.initializer,
        stern_capacitance_f_m2=args.stern_capacitance,
    )
    if args.exponent_clip is not None:
        new_opts = dict(sp.solver_options)
        new_bv = dict(new_opts["bv_convergence"])
        new_bv["exponent_clip"] = float(args.exponent_clip)
        new_opts["bv_convergence"] = new_bv
        sp = sp.with_solver_options(new_opts)

    # -----------------------------------------------------------------------
    # 5. Observable callback
    #
    # The orchestrator invokes this once per converged voltage with the
    # live ctx, so we can assemble CD and PC before the next voltage
    # rebuilds the forms.  Sign convention: CD < 0 for cathodic ORR
    # currents because we multiply the assembled rate by -I_SCALE
    # (the writeup and the inverse scripts use the same sign).
    # -----------------------------------------------------------------------
    NV = len(v_rhe_grid)
    cd_nondim = np.full(NV, np.nan)
    pc_nondim = np.full(NV, np.nan)
    z_achieved = np.full(NV, np.nan)
    method_at_v = ["MISSING"] * NV
    diagnostics_at_v = [None] * NV

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
    phi_hat_grid = v_rhe_grid / V_T
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
        diagnostics_at_v[idx] = point.diagnostics

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
    for i, V in enumerate(v_rhe_grid):
        diag = diagnostics_at_v[i] or {}
        rows.append({
            "V_RHE":      float(V),
            "cd_mA_cm2":  None if np.isnan(cd_ma_cm2[i]) else float(cd_ma_cm2[i]),
            "pc_mA_cm2":  None if np.isnan(pc_ma_cm2[i]) else float(pc_ma_cm2[i]),
            "z_factor":   float(z_achieved[i]),
            "method":     method_at_v[i],
            "initializer_fallback": diag.get("initializer_fallback"),
            "initializer_fallback_reason": diag.get("initializer_fallback_reason"),
            "initializer_picard_iters": diag.get("initializer_picard_iters"),
        })

    with open(os.path.join(OUT_DIR, "iv_curve.csv"), "w") as f:
        f.write(
            "V_RHE,cd_mA_cm2,pc_mA_cm2,z_factor,method,"
            "initializer_fallback,initializer_picard_iters,"
            "initializer_fallback_reason\n"
        )
        for r in rows:
            cd_s = "" if r["cd_mA_cm2"] is None else f"{r['cd_mA_cm2']:.8e}"
            pc_s = "" if r["pc_mA_cm2"] is None else f"{r['pc_mA_cm2']:.8e}"
            fallback_s = "" if r["initializer_fallback"] is None else str(r["initializer_fallback"])
            picard_s = "" if r["initializer_picard_iters"] is None else str(r["initializer_picard_iters"])
            reason_s = "" if r["initializer_fallback_reason"] is None else str(r["initializer_fallback_reason"])
            f.write(
                f"{r['V_RHE']:.4f},{cd_s},{pc_s},"
                f"{r['z_factor']:.4f},{r['method']},"
                f"{fallback_s},{picard_s},{reason_s}\n"
            )

    with open(os.path.join(OUT_DIR, "iv_curve.json"), "w") as f:
        json.dump({
            "config": {
                "v_rhe_grid":  v_rhe_grid.tolist(),
                "mesh_ny":     args.mesh_ny,
                "K0_HAT_R1":   K0_HAT_R1, "K0_HAT_R2": K0_HAT_R2,
                "ALPHA_R1":    ALPHA_R1,  "ALPHA_R2":  ALPHA_R2,
                "E_EQ_R1":     E_EQ_R1,   "E_EQ_R2":   E_EQ_R2,
                "I_SCALE":     I_SCALE,   "V_T":       V_T,
                "formulation": args.formulation,
                "log_rate":    True,
                "counterion_mode": args.counterion_mode,
                "boltzmann_counterion": counterion,
                "initializer": args.initializer,
                "exponent_clip": args.exponent_clip,
                "stern_capacitance_f_m2": args.stern_capacitance,
                "wall_seconds": wall,
            },
            "rows": rows,
            "diagnostics": diagnostics_at_v,
        }, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.0), sharex=True)

        ax = axes[0]
        ax.plot(v_rhe_grid, cd_ma_cm2, "b-o", markersize=6,
                label=f"CD (3sp+B {args.formulation})")
        ax.set_ylabel("Total current density (mA/cm²)")
        ax.set_title(
            f"I-V curve at TRUE parameters via unified API\n"
            f"(formulation={args.formulation}, log_rate=True, "
            f"{args.counterion_mode} Boltzmann ClO4-, "
            f"initializer={args.initializer}, "
            f"clip={args.exponent_clip if args.exponent_clip is not None else 'default'}, "
            f"Stern={args.stern_capacitance if args.stern_capacitance is not None else 'none'}, "
            f"Ny={args.mesh_ny})"
        )
        ax.grid(True, alpha=0.3); ax.legend()

        ax = axes[1]
        ax.plot(v_rhe_grid, pc_ma_cm2, "r-s", markersize=6,
                label=f"PC (3sp+B {args.formulation})")
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
        ic_s = ""
        if r["initializer_fallback"] is not None:
            ic_s = f"  ic_fallback={r['initializer_fallback']}"
        print(
            f"  V={r['V_RHE']:+.3f}  cd={cd_s}  pc={pc_s}  "
            f"z={r['z_factor']:.3f}  {r['method']}{ic_s}"
        )

    print()
    print(f"Saved: {OUT_DIR}/{{iv_curve.png, iv_curve.csv, iv_curve.json}}")


if __name__ == "__main__":
    main()
