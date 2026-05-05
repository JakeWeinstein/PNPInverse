"""Clip-threshold sensitivity study for the production C+D forward solver.

Runs the production I-V curve at multiple ``exponent_clip`` values and
compares CD(V), PC(V), and surface concentrations. Answers: does the
η-clip distort the SS observable in the V_RHE range where it applies?

The production default is ``exponent_clip = 50`` (R2 cathodic clips at
V_RHE < +0.495 V). The production V grid runs to V_RHE = -0.5 V where
the *unclipped* η_R2 = -88.7, so:

  - ``exponent_clip = 50``:  entire production grid V < +0.495 V is
                              clipped for R2 (production default)
  - ``exponent_clip = 100``: entire production grid V > -0.79 V is
                              UNCLIPPED for R2 (and R1, always)
  - ``exponent_clip = 200``: a fortiori unclipped

If observable values agree across thresholds, the clip is observable-
neutral in the production grid. If they differ, the magnitude of the
difference quantifies the clip-induced distortion of the qualitative
physical behaviour.

Implementation: this script mirrors ``scripts/plot_iv_curve_unified.py``
but parametrises the clip threshold via SolverParams.with_solver_options
and also captures surface concentrations + electrode phi per voltage.

Run from PNPInverse/ with ``../venv-firedrake/bin/activate``::

    python scripts/studies/clip_threshold_sensitivity.py
    python scripts/studies/clip_threshold_sensitivity.py --clips 50 100
    python scripts/studies/clip_threshold_sensitivity.py \
        --clips 50 100 200 --out-subdir clip_threshold_v1
"""
from __future__ import annotations

import argparse
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

# ---------------------------------------------------------------------------
# Production V grid (mirrors plot_iv_curve_unified.py)
# ---------------------------------------------------------------------------
V_RHE_GRID = np.array([
    -0.50, -0.40, -0.30, -0.20, -0.10,
    0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60,
])
MESH_NY = 200


def _override_exponent_clip(sp, clip: float):
    """Return a copy of ``sp`` with ``bv_convergence.exponent_clip`` overridden."""
    new_opts = dict(sp.solver_options)
    new_bv = dict(new_opts["bv_convergence"])
    new_bv["exponent_clip"] = float(clip)
    new_opts["bv_convergence"] = new_bv
    return sp.with_solver_options(new_opts)


def _surface_field_means(ctx, n_species: int) -> dict:
    """Compute mean of u_i (= ln c_i) and phi over the electrode boundary.

    Uses the electrode marker from ``ctx['bv_settings']`` and weighted
    boundary assemble. Returns nondim values; converted to c_i via exp.
    """
    import firedrake as fd

    mesh = ctx["mesh"]
    bv = ctx["bv_settings"]
    elec_marker = int(bv["electrode_marker"])
    ds_e = fd.ds(domain=mesh, subdomain_id=elec_marker)
    area = float(fd.assemble(fd.Constant(1.0) * ds_e))

    out: dict = {"electrode_area_nondim": area}
    U = ctx["U"]
    for i in range(n_species):
        u_mean = float(fd.assemble(U.sub(i) * ds_e)) / area
        out[f"u{i}_surface_mean"] = u_mean
        # Cap the exp to avoid overflow in case u_mean is huge (it shouldn't be)
        try:
            out[f"c{i}_surface_mean"] = float(np.exp(u_mean))
        except OverflowError:
            out[f"c{i}_surface_mean"] = float("inf")
    phi_mean = float(fd.assemble(U.sub(n_species) * ds_e)) / area
    out["phi_surface_mean"] = phi_mean
    return out


def run_iv_curve(clip: float, out_dir: str, *, verbose: bool = True) -> dict:
    """Run the full production I-V curve at the given exponent_clip.

    Mirrors ``plot_iv_curve_unified.py`` but writes to ``out_dir`` and
    captures surface fields per voltage in the JSON.
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

    E_EQ_R1, E_EQ_R2 = 0.68, 1.78

    import firedrake as fd
    import firedrake.adjoint as adj
    from Forward.bv_solver import (
        make_graded_rectangle_mesh,
        solve_grid_per_voltage_cold_with_warm_fallback,
    )
    from Forward.bv_solver.observables import _build_bv_observable_form

    os.makedirs(out_dir, exist_ok=True)

    if verbose:
        print()
        print("=" * 72)
        print(f"  I-V curve at exponent_clip = {clip}")
        print("=" * 72)
        print(f"  V_RHE grid:  {V_RHE_GRID.tolist()}")
        print(f"  mesh Ny:     {MESH_NY}")
        print(f"  output:      {out_dir}")

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
    sp = _override_exponent_clip(sp, clip)
    if verbose:
        print(f"  exponent_clip override applied: "
              f"{sp.solver_options['bv_convergence']['exponent_clip']}")

    n_species = THREE_SPECIES_LOGC_BOLTZMANN.n_species
    NV = len(V_RHE_GRID)
    cd_nondim = np.full(NV, np.nan)
    pc_nondim = np.full(NV, np.nan)
    z_achieved = np.full(NV, np.nan)
    method_at_v = ["MISSING"] * NV
    surface_fields_at_v: list[dict | None] = [None] * NV

    def _grab_observables(orig_idx, _phi_eta, ctx):
        f_cd = _build_bv_observable_form(
            ctx, mode="current_density", reaction_index=None, scale=-I_SCALE)
        f_pc = _build_bv_observable_form(
            ctx, mode="peroxide_current", reaction_index=None, scale=-I_SCALE)
        cd_nondim[orig_idx] = float(fd.assemble(f_cd))
        pc_nondim[orig_idx] = float(fd.assemble(f_pc))
        try:
            surface_fields_at_v[orig_idx] = _surface_field_means(ctx, n_species)
        except Exception as exc:
            surface_fields_at_v[orig_idx] = {"error": f"{type(exc).__name__}: {exc}"}

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
    if verbose:
        print()
        print(f"  Converged at {n_ok}/{NV} points in {wall:.1f}s")

    # CSV (mirrors plot_iv_curve_unified.py)
    csv_path = os.path.join(out_dir, "iv_curve.csv")
    with open(csv_path, "w") as f:
        f.write("V_RHE,cd_mA_cm2,pc_mA_cm2,z_factor,method\n")
        for v, cd, pc, z, m in zip(V_RHE_GRID, cd_nondim, pc_nondim, z_achieved, method_at_v):
            cd_s = "" if not np.isfinite(cd) else f"{cd:.8e}"
            pc_s = "" if not np.isfinite(pc) else f"{pc:.8e}"
            f.write(f"{v:.4f},{cd_s},{pc_s},{z:.4f},{m}\n")
    if verbose:
        print(f"  CSV  -> {csv_path}")

    # JSON (with surface fields)
    out_json = {
        "exponent_clip": float(clip),
        "v_rhe": V_RHE_GRID.tolist(),
        "cd_mA_cm2": [float(x) if np.isfinite(x) else None for x in cd_nondim],
        "pc_mA_cm2": [float(x) if np.isfinite(x) else None for x in pc_nondim],
        "z_factor": [float(x) for x in z_achieved],
        "method_at_v": method_at_v,
        "surface_fields": surface_fields_at_v,
        "n_converged": n_ok,
        "n_total": NV,
        "wall_seconds": float(wall),
        "config": {
            "mesh_Nx": 8, "mesh_Ny": int(MESH_NY), "mesh_beta": 3.0,
            "k0_hat_r1": float(K0_HAT_R1), "k0_hat_r2": float(K0_HAT_R2),
            "alpha_r1": float(ALPHA_R1),   "alpha_r2": float(ALPHA_R2),
            "E_eq_r1": E_EQ_R1, "E_eq_r2": E_EQ_R2,
            "V_T": float(V_T), "I_SCALE": float(I_SCALE),
        },
    }
    json_path = os.path.join(out_dir, "iv_curve.json")
    with open(json_path, "w") as f:
        json.dump(out_json, f, indent=2)
    if verbose:
        print(f"  JSON -> {json_path}")
    return out_json


# ---------------------------------------------------------------------------
# Comparison plotting
# ---------------------------------------------------------------------------
def plot_comparison(results: dict[float, dict], out_path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    clips = sorted(results.keys())
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][:len(clips)]

    # --- (0,0): CD vs V ---
    ax = axes[0, 0]
    for clip, color in zip(clips, colors):
        r = results[clip]
        v = np.array(r["v_rhe"])
        cd = np.array([x if x is not None else np.nan for x in r["cd_mA_cm2"]])
        ax.plot(v, cd, "o-", color=color, label=f"clip={clip}", linewidth=1.5, markersize=5)
    ax.axvline(0.495, color="red", linestyle="--", linewidth=0.8, alpha=0.6,
               label="R2 unclip @ clip=50")
    ax.set_xlabel("$V_{RHE}$ [V]")
    ax.set_ylabel("CD [mA/cm²]")
    ax.set_title("Total current density vs voltage")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- (0,1): PC vs V ---
    ax = axes[0, 1]
    for clip, color in zip(clips, colors):
        r = results[clip]
        v = np.array(r["v_rhe"])
        pc = np.array([x if x is not None else np.nan for x in r["pc_mA_cm2"]])
        ax.plot(v, pc, "o-", color=color, label=f"clip={clip}", linewidth=1.5, markersize=5)
    ax.axvline(0.495, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("$V_{RHE}$ [V]")
    ax.set_ylabel("PC [mA/cm²]")
    ax.set_title("Peroxide current vs voltage")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- (1,0): Relative differences in CD vs baseline (smallest clip) ---
    ax = axes[1, 0]
    base_clip = min(clips)
    base_cd = np.array([x if x is not None else np.nan for x in results[base_clip]["cd_mA_cm2"]])
    for clip, color in zip(clips, colors):
        if clip == base_clip:
            continue
        cd = np.array([x if x is not None else np.nan for x in results[clip]["cd_mA_cm2"]])
        rel = (cd - base_cd) / np.where(np.abs(base_cd) > 1e-20, base_cd, 1e-20)
        v = np.array(results[clip]["v_rhe"])
        ax.plot(v, rel, "o-", color=color, label=f"(clip={clip} − {base_clip})/baseline",
                linewidth=1.5, markersize=5)
    ax.axhline(0, color="k", linestyle=":", linewidth=0.8)
    ax.axvline(0.495, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("$V_{RHE}$ [V]")
    ax.set_ylabel("ΔCD / CD_base")
    ax.set_title(f"Relative CD shift vs clip={base_clip}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- (1,1): Surface c_H2O2 across thresholds (log scale) ---
    ax = axes[1, 1]
    for clip, color in zip(clips, colors):
        r = results[clip]
        v = np.array(r["v_rhe"])
        sf = r["surface_fields"]
        c_h2o2 = []
        for entry in sf:
            if entry is None or "c1_surface_mean" not in entry:
                c_h2o2.append(np.nan)
            else:
                c_h2o2.append(entry["c1_surface_mean"])
        c_h2o2 = np.array(c_h2o2)
        c_h2o2 = np.where(c_h2o2 > 1e-300, c_h2o2, np.nan)  # log safety
        ax.semilogy(v, c_h2o2, "o-", color=color, label=f"clip={clip}",
                    linewidth=1.5, markersize=5)
    ax.axvline(0.495, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("$V_{RHE}$ [V]")
    ax.set_ylabel("$c_{H_2O_2}$ at electrode (nondim)")
    ax.set_title("Surface H₂O₂ concentration (internal field)")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        "Clip-threshold sensitivity: do SS observables shift when the η-clip "
        "is widened?",
        fontsize=12,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def print_comparison_table(results: dict[float, dict]) -> None:
    clips = sorted(results.keys())
    base_clip = min(clips)
    print()
    print("=" * 100)
    print(f"  Comparison table (baseline clip = {base_clip})")
    print("=" * 100)

    header = f"  {'V_RHE':>7s} "
    for clip in clips:
        header += f" {f'CD@{clip}':>14s}"
    for clip in clips:
        if clip != base_clip:
            header += f" {f'ΔCD/CDb@{clip}':>15s}"
    print(header)
    print("-" * len(header))

    base_cd = np.array(
        [x if x is not None else np.nan for x in results[base_clip]["cd_mA_cm2"]]
    )
    v_grid = np.array(results[base_clip]["v_rhe"])

    for i, v in enumerate(v_grid):
        row = f"  {v:>+7.3f} "
        for clip in clips:
            cd = results[clip]["cd_mA_cm2"][i]
            cd_s = f"{cd:+.4e}" if cd is not None and np.isfinite(cd) else "       nan    "
            row += f" {cd_s:>14s}"
        for clip in clips:
            if clip == base_clip:
                continue
            cd = results[clip]["cd_mA_cm2"][i]
            if cd is None or not np.isfinite(cd) or not np.isfinite(base_cd[i]) or abs(base_cd[i]) < 1e-20:
                row += f" {'nan':>15s}"
            else:
                rel = (cd - base_cd[i]) / base_cd[i]
                row += f" {rel:>+15.3e}"
        print(row)
    print("=" * 100)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--clips", type=float, nargs="+", default=[50.0, 100.0],
        help="exponent_clip values to sweep (default: [50, 100]).",
    )
    parser.add_argument(
        "--out-subdir", default="clip_threshold_sensitivity",
        help="subdir under StudyResults/ for outputs.",
    )
    args = parser.parse_args()

    base_out = os.path.join(_ROOT, "StudyResults", args.out_subdir)
    os.makedirs(base_out, exist_ok=True)

    all_results: dict[float, dict] = {}
    for clip in args.clips:
        out_dir = os.path.join(base_out, f"clip_{int(clip)}")
        result = run_iv_curve(clip, out_dir, verbose=True)
        all_results[float(clip)] = result

    # Aggregate plot
    plot_path = os.path.join(base_out, "comparison.png")
    plot_comparison(all_results, plot_path)
    print(f"\n[OK] Comparison plot -> {plot_path}")

    # Print table
    print_comparison_table(all_results)

    # Summary JSON
    summary_path = os.path.join(base_out, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {str(k): v for k, v in all_results.items()},
            f, indent=2,
        )
    print(f"\n[OK] Summary JSON -> {summary_path}")


if __name__ == "__main__":
    main()
