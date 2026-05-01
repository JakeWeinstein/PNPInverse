"""Diagnostic: forward sweep in V vs RHE with physical E_eq values.

Maps the experimental voltage range (-0.5V to 1.25V vs RHE) through the
nondimensionalization and runs forward solves with:
  - E_eq_r1 = 0.68V  (O2 + 2H+ + 2e- -> H2O2, standard)
  - E_eq_r2 = 1.78V  (H2O2 + 2H+ + 2e- -> 2H2O, standard)

Compares with E_eq = 0,0 baseline to show the effect.

Key insight: phi_applied_hat = V_applied / V_T, and the solver computes
eta_j = phi_applied_hat - E_eq_j / V_T for each reaction.

Usage:
    python scripts/studies/diagnostic_eeq_voltage_sweep.py
    python scripts/studies/diagnostic_eeq_voltage_sweep.py --cases all
    python scripts/studies/diagnostic_eeq_voltage_sweep.py --cases physical --workers 4
"""
from __future__ import annotations

import os
import sys
import argparse

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

sys.stdout.reconfigure(line_buffering=True)

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
from scripts._bv_common import V_T, I_SCALE

# Standard equilibrium potentials vs RHE
E_EQ_R1_PHYS = 0.68   # V, O2 -> H2O2
E_EQ_R2_PHYS = 1.78   # V, H2O2 -> H2O


def v_rhe_to_phi_hat(v_rhe: np.ndarray) -> np.ndarray:
    """Convert V vs RHE to dimensionless phi_applied_hat = V / V_T."""
    return v_rhe / V_T


def phi_hat_to_v_rhe(phi_hat: np.ndarray) -> np.ndarray:
    """Convert dimensionless phi_applied_hat back to V vs RHE."""
    return phi_hat * V_T


# ---------------------------------------------------------------------------
# Voltage grids in V vs RHE (matching experimental range)
# ---------------------------------------------------------------------------
def make_voltage_grid_physical(n_points: int = 30) -> np.ndarray:
    """Create a voltage grid in V vs RHE spanning the experimental range.

    Experimental data spans roughly -0.5V to 1.25V vs RHE.
    Denser sampling near onset region (0.1V to 0.5V).
    """
    # Coarse grid across full range
    v_coarse = np.linspace(-0.5, 1.25, 15)
    # Dense grid near onset (where curve shape changes rapidly)
    v_onset = np.linspace(0.0, 0.6, 12)
    # Dense grid in transport-limited regime
    v_cathodic = np.linspace(-0.5, -0.1, 6)

    v_all = np.unique(np.concatenate([v_coarse, v_onset, v_cathodic]))
    return np.sort(v_all)[::-1]  # descending (most cathodic first)


def make_voltage_grid_cathodic_only() -> np.ndarray:
    """Cathodic-only grid matching the regime where the solver is robust."""
    v_rhe = np.array([
        -0.50, -0.40, -0.30, -0.20, -0.15, -0.10, -0.05,
        0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
    ])
    return np.sort(v_rhe)[::-1]


# ---------------------------------------------------------------------------
# Worker: solve one case in a spawned process
# ---------------------------------------------------------------------------
def _worker_solve(case: dict) -> dict:
    """Solve one case in a spawned worker process."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ.setdefault("FIREDRAKE_TSFC_KERNEL_CACHE_DIR", "/tmp/firedrake-tsfc")
    os.environ.setdefault("PYOP2_CACHE_DIR", "/tmp/pyop2")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")

    from scripts._bv_common import (
        setup_firedrake_env, I_SCALE,
        FOUR_SPECIES_CHARGED, make_bv_solver_params,
        SNES_OPTS_CHARGED,
    )
    setup_firedrake_env()

    import firedrake as fd
    import pyadjoint as adj
    from Forward.bv_solver import make_graded_rectangle_mesh, solve_grid_with_charge_continuation
    from Forward.bv_solver.observables import _build_bv_observable_form

    label = case["label"]
    v_rhe = np.array(case["v_rhe"])
    phi_hat = v_rhe / case["V_T"]  # Convert to dimensionless

    observable_scale = -I_SCALE
    n_pts = len(phi_hat)
    cd = np.full(n_pts, np.nan)
    pc = np.full(n_pts, np.nan)

    print(f"  [{label}] Starting {n_pts} voltage points...", flush=True)
    print(f"  [{label}] V_RHE range: [{v_rhe.min():.3f}, {v_rhe.max():.3f}] V", flush=True)
    print(f"  [{label}] phi_hat range: [{phi_hat.min():.1f}, {phi_hat.max():.1f}]", flush=True)
    print(f"  [{label}] E_eq_r1={case['E_eq_r1']:.3f}V, E_eq_r2={case['E_eq_r2']:.3f}V", flush=True)

    # With physical E_eq, overpotentials for rxn 1:
    eta_r1 = phi_hat - case["E_eq_r1"] / case["V_T"]
    print(f"  [{label}] eta_r1 range: [{eta_r1.min():.1f}, {eta_r1.max():.1f}]", flush=True)

    # Tuned SNES for robustness
    snes_opts = {
        "snes_type":                 "newtonls",
        "snes_max_it":               400,
        "snes_atol":                 1e-7,
        "snes_rtol":                 1e-10,
        "snes_stol":                 1e-12,
        "snes_linesearch_type":      "l2",
        "snes_linesearch_maxlambda": 0.3,
        "snes_divergence_tolerance": 1e10,
        "ksp_type":                  "preonly",
        "pc_type":                   "lu",
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_8":         77,
        "mat_mumps_icntl_14":        80,
    }

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)
    from scripts._bv_common import K0_HAT_R1, K0_HAT_R2
    sp_kwargs = dict(
        eta_hat=0.0, dt=0.25, t_end=80.0,
        species=FOUR_SPECIES_CHARGED,
        snes_opts=snes_opts,
        k0_hat_r1=case.get("k0_hat_r1", K0_HAT_R1),
        k0_hat_r2=case.get("k0_hat_r2", K0_HAT_R2),
        E_eq_r1=case["E_eq_r1"],
        E_eq_r2=case["E_eq_r2"],
    )
    sp = make_bv_solver_params(**sp_kwargs)

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
            sp,
            phi_applied_values=phi_hat,
            charge_steps=20,
            mesh=mesh,
            max_eta_gap=2.0,
            min_delta_z=0.002,
            per_point_callback=_extract,
        )

    n_conv = sum(1 for i in range(n_pts) if not np.isnan(cd[i]))
    print(f"  [{label}] Done, converged {n_conv}/{n_pts}", flush=True)

    return {
        "label": label,
        "v_rhe": v_rhe.tolist(),
        "phi_hat": phi_hat.tolist(),
        "cd": cd.tolist(),
        "pc": pc.tolist(),
        "n_conv": n_conv,
        "n_total": n_pts,
    }


def main():
    parser = argparse.ArgumentParser(description="Diagnostic E_eq voltage sweep")
    parser.add_argument("--cases", type=str, default="physical",
                        choices=["physical", "baseline", "both", "all"],
                        help="Which cases to run")
    parser.add_argument("--grid", type=str, default="cathodic",
                        choices=["full", "cathodic"],
                        help="Voltage grid: 'full' (-0.5 to 1.25V) or 'cathodic' (-0.5 to 0.3V)")
    parser.add_argument("--workers", type=int, default=0,
                        help="Max parallel workers (0=auto)")
    args = parser.parse_args()

    from scripts._bv_common import K0_HAT_R1, K0_HAT_R2

    if args.grid == "full":
        v_rhe = make_voltage_grid_physical()
    else:
        v_rhe = make_voltage_grid_cathodic_only()

    print(f"\n{'='*70}")
    print(f"  DIAGNOSTIC: E_eq Voltage Sweep")
    print(f"{'='*70}")
    print(f"  V_T = {V_T:.6f} V")
    print(f"  I_SCALE = {I_SCALE:.4f} mA/cm²")
    print(f"  E_eq_r1 (physical) = {E_EQ_R1_PHYS} V vs RHE")
    print(f"  E_eq_r2 (physical) = {E_EQ_R2_PHYS} V vs RHE")
    print(f"  E_eq_r1 / V_T = {E_EQ_R1_PHYS/V_T:.1f}")
    print(f"  E_eq_r2 / V_T = {E_EQ_R2_PHYS/V_T:.1f}")
    print(f"  Grid: {args.grid} ({len(v_rhe)} points)")
    print(f"  V_RHE range: [{v_rhe.min():.3f}, {v_rhe.max():.3f}] V")
    print(f"  phi_hat range: [{v_rhe_to_phi_hat(v_rhe).min():.1f}, {v_rhe_to_phi_hat(v_rhe).max():.1f}]")
    print(f"{'='*70}\n")

    cases = []
    if args.cases in ("baseline", "both", "all"):
        cases.append({
            "label": "E_eq=0,0 (baseline)",
            "v_rhe": v_rhe.tolist(),
            "V_T": V_T,
            "E_eq_r1": 0.0,
            "E_eq_r2": 0.0,
        })
    if args.cases in ("physical", "both", "all"):
        cases.append({
            "label": f"E_eq={E_EQ_R1_PHYS},{E_EQ_R2_PHYS} (physical)",
            "v_rhe": v_rhe.tolist(),
            "V_T": V_T,
            "E_eq_r1": E_EQ_R1_PHYS,
            "E_eq_r2": E_EQ_R2_PHYS,
        })
    if args.cases == "all":
        # Also test with E_eq_r1 only (r2 stays at 0)
        cases.append({
            "label": f"E_eq={E_EQ_R1_PHYS},0 (r1 only)",
            "v_rhe": v_rhe.tolist(),
            "V_T": V_T,
            "E_eq_r1": E_EQ_R1_PHYS,
            "E_eq_r2": 0.0,
        })
        # Higher k0_r2 with physical E_eq
        cases.append({
            "label": f"physical + 50x k0_r2",
            "v_rhe": v_rhe.tolist(),
            "V_T": V_T,
            "E_eq_r1": E_EQ_R1_PHYS,
            "E_eq_r2": E_EQ_R2_PHYS,
            "k0_hat_r2": K0_HAT_R2 * 50,
        })

    n_workers = args.workers if args.workers > 0 else min(len(cases), max(1, (os.cpu_count() or 4) - 1))
    print(f"Launching {len(cases)} cases with {n_workers} workers\n")

    ctx = mp.get_context("spawn")
    results = {}

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        futures = {pool.submit(_worker_solve, c): c["label"] for c in cases}
        for future in as_completed(futures):
            label = futures[future]
            try:
                res = future.result()
                results[res["label"]] = res
                print(f"  Collected: {label} ({res['n_conv']}/{res['n_total']} converged)")
            except Exception as exc:
                print(f"  FAILED {label}: {exc}")
                import traceback
                traceback.print_exc()

    if not results:
        print("No results collected!")
        return

    # -----------------------------------------------------------------------
    # Plot results
    # -----------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = os.path.join(_ROOT, "StudyResults", "diagnostic_eeq_sweep")
    os.makedirs(out_dir, exist_ok=True)

    ordered_labels = [c["label"] for c in cases]
    colors = ["steelblue", "darkorange", "crimson", "purple", "green"]

    # --- Plot 1: Peroxide current vs V_RHE (dimensional) ---
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, label in enumerate(ordered_labels):
        if label in results:
            r = results[label]
            v = np.array(r["v_rhe"])
            pc = np.array(r["pc"])
            # Convert from scaled to mA/cm²
            pc_ma = pc * I_SCALE  # observable_scale was -I_SCALE, so pc is in mA/cm² units
            sort_idx = np.argsort(v)
            mask = ~np.isnan(pc[sort_idx])
            ax.plot(v[sort_idx][mask], pc_ma[sort_idx][mask], "o-",
                    color=colors[i % len(colors)],
                    linewidth=2, markersize=4, label=f"{label} ({r['n_conv']}/{r['n_total']})")
    ax.set_xlabel("Applied Voltage (V vs RHE)", fontsize=13)
    ax.set_ylabel("Peroxide Current Density (mA/cm²)", fontsize=13)
    ax.set_title("Peroxide Current vs Applied Voltage (V vs RHE)", fontsize=14)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(E_EQ_R1_PHYS, color="red", linewidth=0.8, linestyle=":", alpha=0.5, label=f"E_eq_r1={E_EQ_R1_PHYS}V")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, "peroxide_vs_v_rhe.png")
    fig.savefig(p, dpi=150)
    print(f"\nSaved: {p}")
    plt.close(fig)

    # --- Plot 2: Total current vs V_RHE ---
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, label in enumerate(ordered_labels):
        if label in results:
            r = results[label]
            v = np.array(r["v_rhe"])
            cd = np.array(r["cd"])
            cd_ma = cd * I_SCALE
            sort_idx = np.argsort(v)
            mask = ~np.isnan(cd[sort_idx])
            ax.plot(v[sort_idx][mask], cd_ma[sort_idx][mask], "o-",
                    color=colors[i % len(colors)],
                    linewidth=2, markersize=4, label=f"{label} ({r['n_conv']}/{r['n_total']})")
    ax.set_xlabel("Applied Voltage (V vs RHE)", fontsize=13)
    ax.set_ylabel("Total Current Density (mA/cm²)", fontsize=13)
    ax.set_title("Total Current vs Applied Voltage (V vs RHE)", fontsize=14)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(E_EQ_R1_PHYS, color="red", linewidth=0.8, linestyle=":", alpha=0.5, label=f"E_eq_r1={E_EQ_R1_PHYS}V")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, "total_current_vs_v_rhe.png")
    fig.savefig(p, dpi=150)
    print(f"Saved: {p}")
    plt.close(fig)

    # --- Plot 3: Peroxide current vs dimensionless eta_r1 ---
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, label in enumerate(ordered_labels):
        if label in results:
            r = results[label]
            phi_hat = np.array(r["phi_hat"])
            # Compute eta_r1 for this case
            case = cases[i] if i < len(cases) else cases[0]
            eta_r1 = phi_hat - case["E_eq_r1"] / V_T
            pc = np.array(r["pc"])
            sort_idx = np.argsort(eta_r1)
            mask = ~np.isnan(pc[sort_idx])
            ax.plot(eta_r1[sort_idx][mask], pc[sort_idx][mask], "o-",
                    color=colors[i % len(colors)],
                    linewidth=2, markersize=4, label=label)
    ax.set_xlabel(r"Dimensionless overpotential $\hat{\eta}_{r1}$ (Rxn 1)", fontsize=13)
    ax.set_ylabel("Peroxide current (scaled)", fontsize=13)
    ax.set_title("Peroxide Current vs Rxn 1 Overpotential", fontsize=14)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="red", linewidth=0.8, linestyle=":", alpha=0.5, label="η_r1 = 0 (equilibrium)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, "peroxide_vs_eta_r1.png")
    fig.savefig(p, dpi=150)
    print(f"Saved: {p}")
    plt.close(fig)

    # --- Convergence summary ---
    print(f"\n{'='*70}")
    print(f"  CONVERGENCE SUMMARY")
    print(f"{'='*70}")
    for label in ordered_labels:
        if label in results:
            r = results[label]
            v = np.array(r["v_rhe"])
            pc = np.array(r["pc"])
            failed_idx = np.where(np.isnan(pc))[0]
            print(f"  {label}: {r['n_conv']}/{r['n_total']} converged")
            if len(failed_idx) > 0:
                failed_v = v[failed_idx]
                print(f"    Failed at V_RHE: {failed_v.tolist()}")
                failed_phi = np.array(r["phi_hat"])[failed_idx]
                print(f"    Failed at phi_hat: {[f'{x:.1f}' for x in failed_phi]}")

    # --- Save raw data ---
    for label in ordered_labels:
        if label in results:
            r = results[label]
            safe_label = label.replace(" ", "_").replace("=", "").replace(",", "_").replace("(", "").replace(")", "")
            csv_path = os.path.join(out_dir, f"data_{safe_label}.csv")
            with open(csv_path, "w") as f:
                f.write("v_rhe,phi_hat,current_density,peroxide_current\n")
                for j in range(len(r["v_rhe"])):
                    f.write(f"{r['v_rhe'][j]:.6f},{r['phi_hat'][j]:.4f},{r['cd'][j]:.8e},{r['pc'][j]:.8e}\n")
            print(f"  Data saved: {csv_path}")

    print(f"\nAll plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
