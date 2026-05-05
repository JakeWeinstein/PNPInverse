"""Extend the I-V curve into the peroxide window past V_RHE = +0.6 V.

Goal: see what PC actually does as V crosses E_eq_R1 = +0.68 V, where R1
changes sign (cathodic ORR → anodic peroxide oxidation), and into the
"peroxide window" V_RHE ∈ (E_eq_R1, E_eq_R2) = (+0.68, +1.78) V where
both reactions consume H₂O₂.

Knobs vs `scripts/plot_iv_curve_unified.py`:
  - V grid extended to +1.20 V with fine 0.02 V spacing across the
    E_eq_R1 transition (so warm-walk has small enough steps).
  - exponent_clip = 100 so R1 anodic exponent isn't throttled when it
    reverses (also keeps clip=100 results we already validated as the
    cathodic-half reference).
  - n_substeps_warm = 8, bisect_depth_warm = 5 (vs production 4 / 3) so
    warm-walk has more headroom to bridge qualitative SS changes.

Output: same per-clip iv_curve.{csv,json,png} format as the production
driver, in StudyResults/peroxide_window_extension/.

Run from PNPInverse/ with ../venv-firedrake/bin/activate active::

    python scripts/studies/peroxide_window_extension.py
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


# ---------------------------------------------------------------------------
# Extended V grid
# ---------------------------------------------------------------------------
#  Production cathodic (kept):                 -0.50 ... +0.60 (13 points)
#  Fine bridge across E_eq_R1 = +0.68:         +0.62, +0.64, +0.66, +0.68,
#                                              +0.70, +0.72  (6 points)
#  Peroxide window coarse:                     +0.75, +0.80, +0.85, +0.90,
#                                              +0.95, +1.00, +1.05, +1.10,
#                                              +1.15, +1.20 (10 points)
#  Total: 29 points.
V_RHE_GRID = np.array([
    -0.50, -0.40, -0.30, -0.20, -0.10,
    0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60,
    0.62, 0.64, 0.66, 0.68, 0.70, 0.72,
    0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20,
])

MESH_NY = 200
EXPONENT_CLIP = 100.0
N_SUBSTEPS_WARM = 8
BISECT_DEPTH_WARM = 5

OUT_SUBDIR = "peroxide_window_extension"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--diagnostics-json", type=str, default=None,
        help=("If set, dump per-voltage diagnostics (max phi, surface "
              "concentrations, SNES reason/iters, steric watch) to this "
              "path as JSON. Best-effort on the failure path."),
    )
    return p.parse_args()


def main() -> None:
    cli = _parse_args()
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

    E_EQ_R1, E_EQ_R2 = 0.68, 1.78

    import firedrake as fd
    import firedrake.adjoint as adj
    from Forward.bv_solver import (
        make_graded_rectangle_mesh,
        solve_grid_per_voltage_cold_with_warm_fallback,
    )
    from Forward.bv_solver.observables import _build_bv_observable_form

    OUT_DIR = os.path.join(_ROOT, "StudyResults", OUT_SUBDIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 78)
    print("  Peroxide-window extension: I-V across E_eq_R1 = +0.68 V")
    print("=" * 78)
    print(f"  V_RHE grid (n={len(V_RHE_GRID)}): {V_RHE_GRID.tolist()}")
    print(f"  exponent_clip:        {EXPONENT_CLIP}")
    print(f"  n_substeps_warm:      {N_SUBSTEPS_WARM}")
    print(f"  bisect_depth_warm:    {BISECT_DEPTH_WARM}")
    print(f"  mesh_Ny:              {MESH_NY}")
    print(f"  output:               {OUT_DIR}")
    print()

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
        formulation="logc",
        log_rate=True,
        boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION],
        k0_hat_r1=K0_HAT_R1, k0_hat_r2=K0_HAT_R2,
        alpha_r1=ALPHA_R1,   alpha_r2=ALPHA_R2,
        E_eq_r1=E_EQ_R1,     E_eq_r2=E_EQ_R2,
    )
    # Override exponent_clip via the frozen-dataclass copy helper.
    new_opts = dict(sp.solver_options)
    new_bv = dict(new_opts["bv_convergence"])
    new_bv["exponent_clip"] = float(EXPONENT_CLIP)
    new_opts["bv_convergence"] = new_bv
    sp = sp.with_solver_options(new_opts)
    print(f"  exponent_clip override applied: "
          f"{sp.solver_options['bv_convergence']['exponent_clip']}")
    print()

    n_species = THREE_SPECIES_LOGC_BOLTZMANN.n_species
    NV = len(V_RHE_GRID)
    cd_nondim = np.full(NV, np.nan)
    pc_nondim = np.full(NV, np.nan)
    z_achieved = np.full(NV, np.nan)
    method_at_v = ["MISSING"] * NV
    surface_fields_at_v: list[dict | None] = [None] * NV

    def _surface_field_means(ctx) -> dict:
        bv_settings = ctx["bv_settings"]
        elec_marker = int(bv_settings["electrode_marker"])
        ds_e = fd.ds(domain=ctx["mesh"], subdomain_id=elec_marker)
        area = float(fd.assemble(fd.Constant(1.0) * ds_e))
        out: dict = {"electrode_area_nondim": area}
        U = ctx["U"]
        for i in range(n_species):
            u_mean = float(fd.assemble(U.sub(i) * ds_e)) / area
            out[f"u{i}_surface_mean"] = u_mean
            try:
                out[f"c{i}_surface_mean"] = float(np.exp(u_mean))
            except OverflowError:
                out[f"c{i}_surface_mean"] = float("inf")
        out["phi_surface_mean"] = float(
            fd.assemble(U.sub(n_species) * ds_e)
        ) / area
        return out

    def _grab_observables(orig_idx, _phi_eta, ctx):
        f_cd = _build_bv_observable_form(
            ctx, mode="current_density", reaction_index=None, scale=-I_SCALE)
        f_pc = _build_bv_observable_form(
            ctx, mode="peroxide_current", reaction_index=None, scale=-I_SCALE)
        cd_nondim[orig_idx] = float(fd.assemble(f_cd))
        pc_nondim[orig_idx] = float(fd.assemble(f_pc))
        try:
            surface_fields_at_v[orig_idx] = _surface_field_means(ctx)
        except Exception as exc:
            surface_fields_at_v[orig_idx] = {
                "error": f"{type(exc).__name__}: {exc}"
            }

    phi_hat_grid = V_RHE_GRID / V_T
    t_start = time.time()
    with adj.stop_annotating():
        result = solve_grid_per_voltage_cold_with_warm_fallback(
            sp,
            phi_applied_values=phi_hat_grid,
            mesh=mesh,
            max_z_steps=20,
            n_substeps_warm=N_SUBSTEPS_WARM,
            bisect_depth_warm=BISECT_DEPTH_WARM,
            per_point_callback=_grab_observables,
        )
    wall = time.time() - t_start

    for idx, point in result.points.items():
        z_achieved[idx] = point.achieved_z_factor
        method_at_v[idx] = point.method

    n_ok = int(np.sum(~np.isnan(cd_nondim)))
    print()
    print(f"Converged at {n_ok}/{NV} points in {wall:.1f}s")

    # Save CSV + JSON
    csv_path = os.path.join(OUT_DIR, "iv_curve.csv")
    with open(csv_path, "w") as f:
        f.write("V_RHE,cd_mA_cm2,pc_mA_cm2,z_factor,method\n")
        for v, cd, pc, z, m in zip(
            V_RHE_GRID, cd_nondim, pc_nondim, z_achieved, method_at_v
        ):
            cd_s = "" if not np.isfinite(cd) else f"{cd:.8e}"
            pc_s = "" if not np.isfinite(pc) else f"{pc:.8e}"
            f.write(f"{v:.4f},{cd_s},{pc_s},{z:.4f},{m}\n")

    out_json = {
        "exponent_clip": float(EXPONENT_CLIP),
        "n_substeps_warm": int(N_SUBSTEPS_WARM),
        "bisect_depth_warm": int(BISECT_DEPTH_WARM),
        "v_rhe": V_RHE_GRID.tolist(),
        "cd_mA_cm2": [float(x) if np.isfinite(x) else None for x in cd_nondim],
        "pc_mA_cm2": [float(x) if np.isfinite(x) else None for x in pc_nondim],
        "z_factor": [float(x) for x in z_achieved],
        "method_at_v": method_at_v,
        "surface_fields": surface_fields_at_v,
        "n_converged": n_ok, "n_total": NV,
        "wall_seconds": float(wall),
        "config": {
            "mesh_Nx": 8, "mesh_Ny": int(MESH_NY), "mesh_beta": 3.0,
            "k0_hat_r1": float(K0_HAT_R1), "k0_hat_r2": float(K0_HAT_R2),
            "alpha_r1": float(ALPHA_R1), "alpha_r2": float(ALPHA_R2),
            "E_eq_r1": E_EQ_R1, "E_eq_r2": E_EQ_R2,
            "V_T": float(V_T), "I_SCALE": float(I_SCALE),
        },
    }
    json_path = os.path.join(OUT_DIR, "iv_curve.json")
    with open(json_path, "w") as f:
        json.dump(out_json, f, indent=2)

    print(f"  CSV  -> {csv_path}")
    print(f"  JSON -> {json_path}")

    if cli.diagnostics_json:
        diag_records: list[dict | None] = [None] * NV
        for idx, point in result.points.items():
            diag_records[idx] = {
                "orig_idx": idx,
                "v_rhe": float(V_RHE_GRID[idx]),
                "phi_applied_eta": float(point.phi_applied),
                "converged": bool(point.converged),
                "method": point.method,
                "achieved_z_factor": float(point.achieved_z_factor),
                "diagnostics": point.diagnostics,
            }
        diag_out = {
            "v_rhe": V_RHE_GRID.tolist(),
            "diagnostics_at_v": diag_records,
            "exponent_clip": float(EXPONENT_CLIP),
            "mesh_Ny": int(MESH_NY),
            "n_substeps_warm": int(N_SUBSTEPS_WARM),
            "bisect_depth_warm": int(BISECT_DEPTH_WARM),
        }
        diag_path = cli.diagnostics_json
        os.makedirs(os.path.dirname(diag_path) or ".", exist_ok=True)
        with open(diag_path, "w") as f:
            json.dump(diag_out, f, indent=2)
        print(f"  DIAG -> {diag_path}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(11, 11))
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.3)
        ax_cd = fig.add_subplot(gs[0])
        ax_pc = fig.add_subplot(gs[1], sharex=ax_cd)
        ax_pc_zoom = fig.add_subplot(gs[2], sharex=ax_cd)

        # CD
        ax_cd.plot(V_RHE_GRID, cd_nondim, "b-o", markersize=5,
                   label=f"CD (clip={int(EXPONENT_CLIP)})")
        ax_cd.axvline(E_EQ_R1, color="green", ls="--", lw=0.8, alpha=0.6,
                      label=f"E_eq_R1 = +{E_EQ_R1} V")
        ax_cd.set_ylabel("CD (mA/cm²)")
        ax_cd.set_title(
            f"I-V across the peroxide window  "
            f"(clip={int(EXPONENT_CLIP)}, n_subw={N_SUBSTEPS_WARM}, "
            f"bisect={BISECT_DEPTH_WARM})"
        )
        ax_cd.grid(True, alpha=0.3)
        ax_cd.legend(fontsize=9)

        # PC symlog
        ax_pc.plot(V_RHE_GRID, pc_nondim, "r-s", markersize=5,
                   label=f"PC (clip={int(EXPONENT_CLIP)})")
        ax_pc.set_yscale("symlog", linthresh=1e-6)
        ax_pc.axvline(E_EQ_R1, color="green", ls="--", lw=0.8, alpha=0.6)
        ax_pc.axhline(0, color="k", ls=":", lw=0.6)
        ax_pc.set_ylabel("PC (mA/cm²) [symlog]")
        ax_pc.set_title("PC, symlog (linthresh=1e-6) — full dynamic range")
        ax_pc.grid(True, which="both", alpha=0.3)
        ax_pc.legend(fontsize=9)

        # PC linear zoom
        ax_pc_zoom.plot(V_RHE_GRID, pc_nondim, "r-s", markersize=5)
        ax_pc_zoom.axvline(E_EQ_R1, color="green", ls="--", lw=0.8, alpha=0.6)
        ax_pc_zoom.axhline(0, color="k", ls=":", lw=0.6)
        ax_pc_zoom.set_xlabel("V vs RHE (V)")
        ax_pc_zoom.set_ylabel("PC (mA/cm²) [linear]")
        ax_pc_zoom.set_title(
            "PC linear scale — see fine structure across E_eq_R1"
        )
        ax_pc_zoom.grid(True, alpha=0.3)

        png = os.path.join(OUT_DIR, "iv_curve.png")
        plt.savefig(png, dpi=160, bbox_inches="tight")
        plt.close()
        print(f"  PNG  -> {png}")
    except Exception as exc:
        print(f"  plot skipped: {exc}")

    # Per-row summary
    print()
    print(f"{'V_RHE':>7s}  {'CD':>14s}  {'PC':>14s}  {'z':>6s}  method")
    for v, cd, pc, z, m in zip(
        V_RHE_GRID, cd_nondim, pc_nondim, z_achieved, method_at_v
    ):
        cd_s = "(none)" if not np.isfinite(cd) else f"{cd:+.6e}"
        pc_s = "(none)" if not np.isfinite(pc) else f"{pc:+.6e}"
        print(f"{v:>+7.3f}  {cd_s:>14s}  {pc_s:>14s}  {z:>6.3f}  {m}")


if __name__ == "__main__":
    main()
