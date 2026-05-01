"""Sweep k0_2 values in parallel and plot peroxide + total current."""
from __future__ import annotations
import os, sys, multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.stdout.reconfigure(line_buffering=True)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

# Voltage grid (cathodic only — positive voltages dropped)
PHI_APPLIED = np.sort(np.array([
    -46.5, -41.0, -35.0, -28.0, -22.0, -20.0, -17.0, -15.0,
    -13.0, -11.5, -10.0, -8.0, -6.5, -5.0, -4.0, -3.0,
    -2.0, -1.0, -0.5,
]))[::-1]  # descending for solver


def _worker_solve(mult: float) -> dict:
    """Solve one k0_2 multiplier in a spawned worker process."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ.setdefault("FIREDRAKE_TSFC_KERNEL_CACHE_DIR", "/tmp/firedrake-tsfc")
    os.environ.setdefault("PYOP2_CACHE_DIR", "/tmp/pyop2")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")

    from scripts._bv_common import (
        setup_firedrake_env,
        K0_HAT_R2, I_SCALE,
        FOUR_SPECIES_CHARGED, SNES_OPTS_CHARGED,
        make_bv_solver_params,
    )
    setup_firedrake_env()

    import firedrake as fd
    import pyadjoint as adj
    from Forward.bv_solver import make_graded_rectangle_mesh, solve_grid_with_charge_continuation
    from Forward.bv_solver.observables import _build_bv_observable_form

    k02_val = K0_HAT_R2 * mult
    observable_scale = -I_SCALE
    n_eta = len(PHI_APPLIED)
    cd = np.full(n_eta, np.nan)
    pc = np.full(n_eta, np.nan)

    print(f"  [worker {mult}x] Starting solve, k0_2={k02_val:.4e}", flush=True)

    mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)
    sp = make_bv_solver_params(
        eta_hat=0.0, dt=0.5, t_end=50.0,
        species=FOUR_SPECIES_CHARGED,
        snes_opts=SNES_OPTS_CHARGED,
        k0_hat_r2=k02_val,
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
            sp,
            phi_applied_values=PHI_APPLIED,
            charge_steps=10,
            mesh=mesh,
            max_eta_gap=3.0,
            per_point_callback=_extract,
        )

    n_conv = sum(1 for i in range(n_eta) if not np.isnan(cd[i]))
    print(f"  [worker {mult}x] Done, converged {n_conv}/{n_eta}", flush=True)
    return {"mult": mult, "cd": cd.tolist(), "pc": pc.tolist()}


def main():
    # Extended sweep: 1x through 500x
    k02_multipliers = [1, 10, 50, 100, 200, 500]

    n_workers = min(len(k02_multipliers), max(1, (os.cpu_count() or 4) - 1))
    print(f"Launching {len(k02_multipliers)} solves across {n_workers} workers")

    ctx = mp.get_context("spawn")
    results = {}

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        futures = {pool.submit(_worker_solve, m): m for m in k02_multipliers}
        for future in as_completed(futures):
            mult = futures[future]
            try:
                res = future.result()
                results[res["mult"]] = {
                    "cd": np.array(res["cd"]),
                    "pc": np.array(res["pc"]),
                }
                print(f"  Collected result for {mult}x")
            except Exception as exc:
                print(f"  FAILED {mult}x: {exc}")

    # Import matplotlib after all Firedrake work is done
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sort_idx = np.argsort(PHI_APPLIED)
    eta_plot = PHI_APPLIED[sort_idx]

    out_dir = os.path.join(_ROOT, "StudyResults", "k02_sweep_plots")
    os.makedirs(out_dir, exist_ok=True)

    cmap = plt.cm.viridis
    n = len(k02_multipliers)
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    # --- Plot 1: Peroxide current (full range) ---
    fig, ax = plt.subplots(figsize=(11, 7))
    for i, mult in enumerate(sorted(results.keys())):
        r = results[mult]
        from scripts._bv_common import K0_HAT_R2
        label = f"{mult}x (k0_2={K0_HAT_R2 * mult:.3e})"
        ax.plot(eta_plot, r["pc"][sort_idx], "o-", color=colors[i],
                linewidth=2, markersize=5, label=label)
    ax.set_xlabel(r"Dimensionless overpotential $\hat{\eta}$", fontsize=13)
    ax.set_ylabel("Peroxide current (scaled)", fontsize=13)
    ax.set_title("Peroxide Current vs k0_2 Multiplier (Extended)", fontsize=14)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.legend(fontsize=9, title="k0_2 value")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path1 = os.path.join(out_dir, "peroxide_k02_sweep_ext.png")
    fig.savefig(path1, dpi=150)
    print(f"\nSaved: {path1}")
    plt.close(fig)

    # --- Plot 2: Total current density ---
    fig, ax = plt.subplots(figsize=(11, 7))
    for i, mult in enumerate(sorted(results.keys())):
        r = results[mult]
        label = f"{mult}x (k0_2={K0_HAT_R2 * mult:.3e})"
        ax.plot(eta_plot, r["cd"][sort_idx], "o-", color=colors[i],
                linewidth=2, markersize=5, label=label)
    ax.set_xlabel(r"Dimensionless overpotential $\hat{\eta}$", fontsize=13)
    ax.set_ylabel("Total current density (scaled)", fontsize=13)
    ax.set_title("Total Current Density vs k0_2 Multiplier (Extended)", fontsize=14)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.legend(fontsize=9, title="k0_2 value")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path2 = os.path.join(out_dir, "total_current_k02_sweep_ext.png")
    fig.savefig(path2, dpi=150)
    print(f"Saved: {path2}")
    plt.close(fig)

    # --- Plot 3: Zoomed peroxide ---
    fig, ax = plt.subplots(figsize=(11, 7))
    mask = (eta_plot >= -25) & (eta_plot <= 5)
    for i, mult in enumerate(sorted(results.keys())):
        r = results[mult]
        label = f"{mult}x (k0_2={K0_HAT_R2 * mult:.3e})"
        ax.plot(eta_plot[mask], r["pc"][sort_idx][mask], "o-", color=colors[i],
                linewidth=2, markersize=6, label=label)
    ax.set_xlabel(r"Dimensionless overpotential $\hat{\eta}$", fontsize=13)
    ax.set_ylabel("Peroxide current (scaled)", fontsize=13)
    ax.set_title("Peroxide Current (Zoomed: -25 to +5)", fontsize=14)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.legend(fontsize=9, title="k0_2 value")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path3 = os.path.join(out_dir, "peroxide_k02_sweep_ext_zoomed.png")
    fig.savefig(path3, dpi=150)
    print(f"Saved: {path3}")
    plt.close(fig)

    print(f"\nAll plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
