"""Shape match v2: focus on cathodic regime where solver is robust.

Insights from v1:
- Physical E_eq correctly places onset near 0.68V
- Positive-eta points suffer partial z-convergence (unreliable above ~0.2V)
- Need to extend further cathodic to see "dip and recovery"
- k0_r2 alone doesn't create the dip shape

Strategy: focus on cathodic range (-1V to 0.3V) and vary both k0_r2 AND
the ratio k0_r1/k0_r2, plus try different alpha values. The paper also
shows L_ref matters - we keep L_ref=100µm as that's fixed infrastructure,
but note L_ref adjustments shift the transport limit proportionally.

Key experimental features to match (DataPlot.png):
- Onset ~0.3V vs RHE (cathodic current begins)
- Peak ~-0.35 mA/cm² near 0V
- Recovery to ~-0.15 mA/cm² at -0.5V
- Transport limit at deep cathodic
"""
from __future__ import annotations
import os, sys

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

E_EQ_R1 = 0.68
E_EQ_R2 = 1.78

# Dense cathodic grid where solver converges well
V_RHE = np.sort(np.array([
    -1.00, -0.90, -0.80, -0.70, -0.60,
    -0.50, -0.45, -0.40, -0.35, -0.30, -0.25, -0.20,
    -0.15, -0.10, -0.05, 0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
]))[::-1]


def _worker(case: dict) -> dict:
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
    z_achieved = np.full(n_pts, np.nan)

    print(f"  [{label}] {n_pts} pts, phi_hat: [{phi_hat.min():.1f}, {phi_hat.max():.1f}]", flush=True)

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
            ctx, mode="current_density", reaction_index=None, scale=observable_scale)
        form_pc = _build_bv_observable_form(
            ctx, mode="peroxide_current", reaction_index=None, scale=observable_scale)
        cd[orig_idx] = float(fd.assemble(form_cd))
        pc[orig_idx] = float(fd.assemble(form_pc))

    with adj.stop_annotating():
        result = solve_grid_with_charge_continuation(
            sp, phi_applied_values=phi_hat,
            charge_steps=20, mesh=mesh,
            max_eta_gap=2.0, min_delta_z=0.002,
            per_point_callback=_extract,
        )

    # Track z-convergence per point
    for idx, pt in result.points.items():
        z_achieved[idx] = pt.achieved_z_factor

    n_full = sum(1 for pt in result.points.values() if pt.converged)
    n_vals = sum(1 for i in range(n_pts) if not np.isnan(pc[i]))
    print(f"  [{label}] {n_full}/{n_pts} full-z, {n_vals}/{n_pts} with values", flush=True)

    return {
        "label": label, "v_rhe": v_rhe.tolist(),
        "cd": cd.tolist(), "pc": pc.tolist(),
        "z_achieved": z_achieved.tolist(),
        "n_full_z": n_full, "n_vals": n_vals, "n_total": n_pts,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  SHAPE MATCH v2: Cathodic Focus + Extended Range")
    print(f"{'='*70}")
    print(f"  V grid: {len(V_RHE)} pts [{V_RHE.min():.2f}, {V_RHE.max():.2f}] V vs RHE")
    print(f"  phi_hat range: [{(V_RHE/V_T).min():.1f}, {(V_RHE/V_T).max():.1f}]")
    print(f"{'='*70}\n")

    # Cases: vary k0_r2 and k0_r1 together
    cases = [
        # Baseline
        {"label": "baseline",
         "k0_hat_r1": K0_HAT_R1, "k0_hat_r2": K0_HAT_R2,
         "E_eq_r1": E_EQ_R1, "E_eq_r2": E_EQ_R2, "v_rhe": V_RHE.tolist()},
        # Increase k0_r2 significantly
        {"label": "k0_r2 x100",
         "k0_hat_r1": K0_HAT_R1, "k0_hat_r2": K0_HAT_R2 * 100,
         "E_eq_r1": E_EQ_R1, "E_eq_r2": E_EQ_R2, "v_rhe": V_RHE.tolist()},
        # Increase both k0 to get higher currents
        {"label": "both x5, r2 x500",
         "k0_hat_r1": K0_HAT_R1 * 5, "k0_hat_r2": K0_HAT_R2 * 500,
         "E_eq_r1": E_EQ_R1, "E_eq_r2": E_EQ_R2, "v_rhe": V_RHE.tolist()},
        # Lower E_eq_r2 to bring rxn 2 equilibrium closer
        {"label": "E_eq_r2=1.2V",
         "k0_hat_r1": K0_HAT_R1, "k0_hat_r2": K0_HAT_R2 * 100,
         "E_eq_r1": E_EQ_R1, "E_eq_r2": 1.20, "v_rhe": V_RHE.tolist()},
        # Higher alpha_r2 for steeper rxn 2 onset
        {"label": "alpha_r2=0.8, k0_r2 x100",
         "k0_hat_r1": K0_HAT_R1, "k0_hat_r2": K0_HAT_R2 * 100,
         "alpha_r2": 0.8, "E_eq_r1": E_EQ_R1, "E_eq_r2": E_EQ_R2,
         "v_rhe": V_RHE.tolist()},
        # The "dip" needs rxn2 to activate at a DIFFERENT voltage than rxn1
        # Try E_eq_r2 much lower so rxn 2 starts earlier
        {"label": "E_eq_r2=0.4V, k0_r2 x100",
         "k0_hat_r1": K0_HAT_R1, "k0_hat_r2": K0_HAT_R2 * 100,
         "E_eq_r1": E_EQ_R1, "E_eq_r2": 0.40, "v_rhe": V_RHE.tolist()},
    ]

    n_workers = min(args.max_workers, len(cases))
    print(f"Running {len(cases)} cases with {n_workers} workers\n")

    ctx = mp.get_context("spawn")
    results = {}

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        futures = {pool.submit(_worker, c): c["label"] for c in cases}
        for future in as_completed(futures):
            label = futures[future]
            try:
                res = future.result()
                results[res["label"]] = res
                print(f"  OK: {label} ({res['n_full_z']}/{res['n_total']} full-z)")
            except Exception as exc:
                print(f"  FAIL: {label}: {exc}")
                import traceback; traceback.print_exc()

    if not results:
        return

    # -----------------------------------------------------------------------
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = os.path.join(_ROOT, "StudyResults", "shape_match_v2")
    os.makedirs(out_dir, exist_ok=True)

    ordered = [c["label"] for c in cases]
    colors = plt.cm.tab10(np.linspace(0, 1, len(cases)))

    # --- Peroxide current (mA/cm²) vs V_RHE, full-z points only ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax_idx, (ax, title_suffix, filter_z) in enumerate(zip(
        axes, ["(all points)", "(full z=1.0 only)"], [False, True]
    )):
        for i, label in enumerate(ordered):
            if label not in results:
                continue
            r = results[label]
            v = np.array(r["v_rhe"])
            pc = np.array(r["pc"])
            z = np.array(r["z_achieved"])
            sort_idx = np.argsort(v)

            if filter_z:
                mask = (~np.isnan(pc[sort_idx])) & (z[sort_idx] >= 0.999)
            else:
                mask = ~np.isnan(pc[sort_idx])

            if mask.sum() == 0:
                continue
            ax.plot(v[sort_idx][mask], pc[sort_idx][mask], "o-",
                    color=colors[i], linewidth=2, markersize=4,
                    label=f"{label} ({mask.sum()}/{len(v)})")

        ax.set_xlabel("Applied Voltage (V vs RHE)", fontsize=12)
        ax.set_ylabel("Peroxide Current (mA/cm²)", fontsize=12)
        ax.set_title(f"Peroxide Current {title_suffix}", fontsize=13)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.axvline(E_EQ_R1, color="red", linewidth=0.8, linestyle=":", alpha=0.5)
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    p = os.path.join(out_dir, "peroxide_shape_match_v2.png")
    fig.savefig(p, dpi=150)
    print(f"\nSaved: {p}")
    plt.close(fig)

    # --- Z-convergence diagnostic ---
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, label in enumerate(ordered):
        if label not in results:
            continue
        r = results[label]
        v = np.array(r["v_rhe"])
        z = np.array(r["z_achieved"])
        sort_idx = np.argsort(v)
        ax.plot(v[sort_idx], z[sort_idx], "o-", color=colors[i],
                linewidth=1.5, markersize=4, label=label)
    ax.set_xlabel("Applied Voltage (V vs RHE)", fontsize=12)
    ax.set_ylabel("Achieved z-factor", fontsize=12)
    ax.set_title("Charge Continuation z-Convergence vs Voltage", fontsize=13)
    ax.axhline(1.0, color="green", linewidth=1, linestyle="--", alpha=0.5, label="z=1 (full)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.1)
    fig.tight_layout()
    p = os.path.join(out_dir, "z_convergence_map.png")
    fig.savefig(p, dpi=150)
    print(f"Saved: {p}")
    plt.close(fig)

    # --- Shape metrics ---
    print(f"\n{'='*70}")
    print(f"  SHAPE METRICS (full-z points only)")
    print(f"{'='*70}")
    for label in ordered:
        if label not in results:
            continue
        r = results[label]
        v = np.array(r["v_rhe"])
        pc = np.array(r["pc"])
        z = np.array(r["z_achieved"])
        sort_idx = np.argsort(v)
        v_s, pc_s, z_s = v[sort_idx], pc[sort_idx], z[sort_idx]

        # Only use full-z points
        full_mask = (z_s >= 0.999) & ~np.isnan(pc_s)
        if full_mask.sum() == 0:
            print(f"\n  {label}: NO full-z points!")
            continue
        v_f, pc_f = v_s[full_mask], pc_s[full_mask]

        peak_idx = np.argmin(pc_f)
        peak_v, peak_pc = v_f[peak_idx], pc_f[peak_idx]
        most_cathodic_pc = pc_f[0]  # most negative V
        most_cathodic_v = v_f[0]
        dip = abs(peak_pc) / max(abs(most_cathodic_pc), 1e-16) if most_cathodic_pc != 0 else 0

        print(f"\n  {label} ({full_mask.sum()} full-z pts):")
        print(f"    Full-z V range: [{v_f.min():.3f}, {v_f.max():.3f}] V")
        print(f"    Peak pc: {peak_pc:.4f} mA/cm² at {peak_v:.3f} V")
        print(f"    pc at {most_cathodic_v:.2f}V: {most_cathodic_pc:.4f} mA/cm²")
        print(f"    Dip ratio: {dip:.2f} (>1 means peak is more negative than deep cathodic)")

    print(f"\nAll saved to: {out_dir}")


if __name__ == "__main__":
    main()
