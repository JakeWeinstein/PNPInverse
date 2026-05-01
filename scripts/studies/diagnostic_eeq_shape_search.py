"""Shape search: find parameters that produce the experimental peroxide curve.

The experimental curve (DataPlot.png, pH 4 Cs+) shows:
- Onset ~0.3V vs RHE
- Peak (most negative) ~-0.35 mA/cm² near 0V
- Dip and recovery at more cathodic potentials
- Plateau ~-0.15 to -0.2 mA/cm² at -0.5V

Key insight: the "dip" comes from H2O2 consumption by reaction 2.
To get this shape, we need sufficient k0_r2 relative to k0_r1.

We sweep k0_r2 (and optionally L_ref) to find the curve shape match.

Usage:
    python scripts/studies/diagnostic_eeq_shape_search.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(line_buffering=True)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

from scripts._bv_common import V_T, I_SCALE, K0_HAT_R1, K0_HAT_R2, K_SCALE

E_EQ_R1 = 0.68  # V vs RHE
E_EQ_R2 = 1.78  # V vs RHE

# Dense voltage grid in the interesting region
V_RHE = np.sort(np.array([
    -0.50, -0.45, -0.40, -0.35, -0.30, -0.25, -0.20, -0.15,
    -0.10, -0.05, 0.00, 0.05, 0.10, 0.15, 0.20, 0.25,
    0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70,
]))[::-1]  # Descending


def _worker_solve(case: dict) -> dict:
    """Solve one parameter set."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ.setdefault("FIREDRAKE_TSFC_KERNEL_CACHE_DIR", "/tmp/firedrake-tsfc")
    os.environ.setdefault("PYOP2_CACHE_DIR", "/tmp/pyop2")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")

    from scripts._bv_common import (
        setup_firedrake_env, I_SCALE as I_SC,
        FOUR_SPECIES_CHARGED, make_bv_solver_params, V_T as VT,
    )
    setup_firedrake_env()

    import firedrake as fd
    import pyadjoint as adj
    from Forward.bv_solver import make_graded_rectangle_mesh, solve_grid_with_charge_continuation
    from Forward.bv_solver.observables import _build_bv_observable_form

    label = case["label"]
    v_rhe = np.array(case["v_rhe"])
    phi_hat = v_rhe / VT

    observable_scale = -I_SC
    n_pts = len(v_rhe)
    cd = np.full(n_pts, np.nan)
    pc = np.full(n_pts, np.nan)

    print(f"  [{label}] Starting {n_pts} points, phi_hat: [{phi_hat.min():.1f}, {phi_hat.max():.1f}]", flush=True)

    snes_opts = {
        "snes_type": "newtonls", "snes_max_it": 400,
        "snes_atol": 1e-7, "snes_rtol": 1e-10, "snes_stol": 1e-12,
        "snes_linesearch_type": "l2", "snes_linesearch_maxlambda": 0.3,
        "snes_divergence_tolerance": 1e10,
        "ksp_type": "preonly", "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_8": 77, "mat_mumps_icntl_14": 80,
    }

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)
    sp = make_bv_solver_params(
        eta_hat=0.0, dt=0.25, t_end=80.0,
        species=FOUR_SPECIES_CHARGED,
        snes_opts=snes_opts,
        k0_hat_r1=case["k0_hat_r1"],
        k0_hat_r2=case["k0_hat_r2"],
        alpha_r1=case.get("alpha_r1", 0.627),
        alpha_r2=case.get("alpha_r2", 0.5),
        E_eq_r1=case["E_eq_r1"],
        E_eq_r2=case["E_eq_r2"],
    )

    def _extract(orig_idx, phi_app, ctx):
        form_cd = _build_bv_observable_form(
            ctx, mode="current_density", reaction_index=None, scale=observable_scale,
        )
        form_pc = _build_bv_observable_form(
            ctx, mode="peroxide_current", reaction_index=None, scale=observable_scale,
        )
        cd[orig_idx] = float(fd.assemble(form_cd))
        pc[orig_idx] = float(fd.assemble(form_pc))

    with adj.stop_annotating():
        solve_grid_with_charge_continuation(
            sp, phi_applied_values=phi_hat,
            charge_steps=20, mesh=mesh,
            max_eta_gap=2.0, min_delta_z=0.002,
            per_point_callback=_extract,
        )

    n_conv = sum(1 for i in range(n_pts) if not np.isnan(pc[i]))
    print(f"  [{label}] Done, converged {n_conv}/{n_pts}", flush=True)

    # NOTE: cd and pc are already in mA/cm² (because observable_scale = -I_SCALE)
    return {
        "label": label, "v_rhe": v_rhe.tolist(),
        "cd": cd.tolist(), "pc": pc.tolist(),
        "n_conv": n_conv, "n_total": n_pts,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  SHAPE SEARCH: Finding experimental curve match")
    print(f"{'='*70}")
    print(f"  E_eq_r1 = {E_EQ_R1}V, E_eq_r2 = {E_EQ_R2}V")
    print(f"  k0_r1 (default) = {K0_HAT_R1:.6e} (nondim) = {K0_HAT_R1*K_SCALE:.3e} m/s")
    print(f"  k0_r2 (default) = {K0_HAT_R2:.6e} (nondim) = {K0_HAT_R2*K_SCALE:.3e} m/s")
    print(f"  I_SCALE = {I_SCALE:.4f} mA/cm²")
    print(f"  Voltage grid: {len(V_RHE)} points [{V_RHE.min():.2f}, {V_RHE.max():.2f}] V vs RHE")
    print(f"{'='*70}\n")

    # Sweep k0_r2 multipliers to find the "dip and recovery" shape
    k0_r2_multipliers = [1, 10, 50, 200, 1000]

    cases = []
    for mult in k0_r2_multipliers:
        k0_r2_val = K0_HAT_R2 * mult
        cases.append({
            "label": f"k0_r2 x{mult} ({k0_r2_val*K_SCALE:.2e} m/s)",
            "v_rhe": V_RHE.tolist(),
            "k0_hat_r1": K0_HAT_R1,
            "k0_hat_r2": k0_r2_val,
            "E_eq_r1": E_EQ_R1,
            "E_eq_r2": E_EQ_R2,
        })

    n_workers = min(args.max_workers, len(cases))
    print(f"Running {len(cases)} cases with {n_workers} workers\n")

    ctx = mp.get_context("spawn")
    results = {}

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        futures = {pool.submit(_worker_solve, c): c["label"] for c in cases}
        for future in as_completed(futures):
            label = futures[future]
            try:
                res = future.result()
                results[res["label"]] = res
                print(f"  Collected: {label} ({res['n_conv']}/{res['n_total']})")
            except Exception as exc:
                print(f"  FAILED {label}: {exc}")
                import traceback
                traceback.print_exc()

    if not results:
        print("No results!")
        return

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = os.path.join(_ROOT, "StudyResults", "diagnostic_eeq_sweep")
    os.makedirs(out_dir, exist_ok=True)

    ordered = [c["label"] for c in cases]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(cases)))

    # --- Peroxide current vs V_RHE ---
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, label in enumerate(ordered):
        if label in results:
            r = results[label]
            v = np.array(r["v_rhe"])
            pc = np.array(r["pc"])  # Already in mA/cm²
            sort_idx = np.argsort(v)
            mask = ~np.isnan(pc[sort_idx])
            ax.plot(v[sort_idx][mask], pc[sort_idx][mask], "o-",
                    color=colors[i], linewidth=2, markersize=4,
                    label=f"{label} ({r['n_conv']}/{r['n_total']})")
    ax.set_xlabel("Applied Voltage (V vs RHE)", fontsize=13)
    ax.set_ylabel("Peroxide Current Density (mA/cm²)", fontsize=13)
    ax.set_title("Shape Search: Peroxide Current vs k0_r2", fontsize=14)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(E_EQ_R1, color="red", linewidth=0.8, linestyle=":", alpha=0.5, label=f"E_eq_r1={E_EQ_R1}V")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, "shape_search_peroxide.png")
    fig.savefig(p, dpi=150)
    print(f"\nSaved: {p}")
    plt.close(fig)

    # --- Total current vs V_RHE ---
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, label in enumerate(ordered):
        if label in results:
            r = results[label]
            v = np.array(r["v_rhe"])
            cd = np.array(r["cd"])
            sort_idx = np.argsort(v)
            mask = ~np.isnan(cd[sort_idx])
            ax.plot(v[sort_idx][mask], cd[sort_idx][mask], "o-",
                    color=colors[i], linewidth=2, markersize=4,
                    label=f"{label} ({r['n_conv']}/{r['n_total']})")
    ax.set_xlabel("Applied Voltage (V vs RHE)", fontsize=13)
    ax.set_ylabel("Total Current Density (mA/cm²)", fontsize=13)
    ax.set_title("Shape Search: Total Current vs k0_r2", fontsize=14)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, "shape_search_total_current.png")
    fig.savefig(p, dpi=150)
    print(f"Saved: {p}")
    plt.close(fig)

    # --- Save CSV data ---
    for label in ordered:
        if label in results:
            r = results[label]
            safe = label.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
            csv_p = os.path.join(out_dir, f"shape_{safe}.csv")
            with open(csv_p, "w") as f:
                f.write("v_rhe,current_density_mAcm2,peroxide_current_mAcm2\n")
                for j in range(len(r["v_rhe"])):
                    f.write(f"{r['v_rhe'][j]:.6f},{r['cd'][j]:.8e},{r['pc'][j]:.8e}\n")

    # --- Print shape metrics ---
    print(f"\n{'='*70}")
    print(f"  SHAPE METRICS")
    print(f"{'='*70}")
    for label in ordered:
        if label in results:
            r = results[label]
            v = np.array(r["v_rhe"])
            pc = np.array(r["pc"])
            sort_idx = np.argsort(v)
            v_s = v[sort_idx]
            pc_s = pc[sort_idx]
            mask = ~np.isnan(pc_s)
            if mask.sum() == 0:
                continue
            v_m = v_s[mask]
            pc_m = pc_s[mask]

            # Onset: first V where |pc| > 0.01 * max|pc|
            pc_max = np.max(np.abs(pc_m))
            onset_mask = np.abs(pc_m) > 0.01 * pc_max
            onset_v = v_m[onset_mask][-1] if onset_mask.any() else float('nan')

            # Peak: most negative pc
            peak_idx = np.argmin(pc_m)
            peak_v = v_m[peak_idx]
            peak_pc = pc_m[peak_idx]

            # Value at -0.5V
            cathodic_mask = np.abs(v_m - (-0.5)) < 0.02
            pc_at_minus05 = pc_m[cathodic_mask][0] if cathodic_mask.any() else float('nan')

            # "Dip ratio": |pc at peak| / |pc at -0.5V| > 1 means dip exists
            dip_ratio = abs(peak_pc) / max(abs(pc_at_minus05), 1e-16) if not np.isnan(pc_at_minus05) else float('nan')

            print(f"\n  {label}:")
            print(f"    Onset: {onset_v:.3f} V vs RHE")
            print(f"    Peak: {peak_pc:.4f} mA/cm² at {peak_v:.3f} V")
            print(f"    pc at -0.5V: {pc_at_minus05:.4f} mA/cm²")
            print(f"    Dip ratio: {dip_ratio:.2f} (>1 = dip exists)")

    print(f"\nAll plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
