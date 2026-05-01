"""Test per-reaction E_eq with tuned solver parameters for convergence."""
from __future__ import annotations
import os, sys

sys.stdout.reconfigure(line_buffering=True)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

# Cathodic-only voltage grid
PHI_APPLIED = np.sort(np.array([
    -46.5, -41.0, -35.0, -28.0, -22.0, -20.0, -17.0, -15.0,
    -13.0, -11.5, -10.0, -8.0, -6.5, -5.0, -4.0, -3.0,
    -2.0, -1.0, -0.5,
]))[::-1]


def _worker_solve(case: dict) -> dict:
    """Solve one case in a spawned worker."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ.setdefault("FIREDRAKE_TSFC_KERNEL_CACHE_DIR", "/tmp/firedrake-tsfc")
    os.environ.setdefault("PYOP2_CACHE_DIR", "/tmp/pyop2")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")

    from scripts._bv_common import (
        setup_firedrake_env, I_SCALE,
        FOUR_SPECIES_CHARGED, make_bv_solver_params,
    )
    setup_firedrake_env()

    import firedrake as fd
    import pyadjoint as adj
    from Forward.bv_solver import make_graded_rectangle_mesh, solve_grid_with_charge_continuation
    from Forward.bv_solver.observables import _build_bv_observable_form

    label = case["label"]
    observable_scale = -I_SCALE
    n_eta = len(PHI_APPLIED)
    cd = np.full(n_eta, np.nan)
    pc = np.full(n_eta, np.nan)

    print(f"  [worker {label}] Starting...", flush=True)

    # Tuned SNES opts for large E_eq split:
    # - more iterations (400)
    # - stricter line search (maxlambda 0.3)
    # - tighter divergence tolerance
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

    # Smaller dt + longer t_end for robustness at stiff points
    mesh = make_graded_rectangle_mesh(Nx=8, Ny=200, beta=3.0)
    sp = make_bv_solver_params(
        eta_hat=0.0, dt=0.25, t_end=80.0,
        species=FOUR_SPECIES_CHARGED,
        snes_opts=snes_opts,
        k0_hat_r2=case["k0_hat_r2"],
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
            sp,
            phi_applied_values=PHI_APPLIED,
            charge_steps=20,       # finer z-ramp steps
            mesh=mesh,
            max_eta_gap=2.0,       # tighter bridge spacing
            min_delta_z=0.002,     # finer z subdivision
            per_point_callback=_extract,
        )

    n_conv = sum(1 for i in range(n_eta) if not np.isnan(cd[i]))
    print(f"  [worker {label}] Done, converged {n_conv}/{n_eta}", flush=True)
    return {"label": label, "cd": cd.tolist(), "pc": pc.tolist()}


def main():
    from scripts._bv_common import K0_HAT_R2

    cases = [
        {
            "label": "baseline (E_eq=0,0)",
            "k0_hat_r2": K0_HAT_R2,
            "E_eq_r1": 0.0,
            "E_eq_r2": 0.0,
        },
        {
            "label": "split E_eq (0.68V, 1.78V)",
            "k0_hat_r2": K0_HAT_R2,
            "E_eq_r1": 0.68,
            "E_eq_r2": 1.78,
        },
        {
            "label": "split E_eq + 10x k0_2",
            "k0_hat_r2": K0_HAT_R2 * 10,
            "E_eq_r1": 0.68,
            "E_eq_r2": 1.78,
        },
        {
            "label": "split E_eq + 50x k0_2",
            "k0_hat_r2": K0_HAT_R2 * 50,
            "E_eq_r1": 0.68,
            "E_eq_r2": 1.78,
        },
    ]

    n_workers = len(cases)
    print(f"Launching {len(cases)} cases across {n_workers} workers")
    print(f"Tuned params: dt=0.25, t_end=80, snes_max_it=400, charge_steps=20, "
          f"max_eta_gap=2.0, min_delta_z=0.002")

    ctx = mp.get_context("spawn")
    results = {}

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        futures = {pool.submit(_worker_solve, c): c["label"] for c in cases}
        for future in as_completed(futures):
            label = futures[future]
            try:
                res = future.result()
                results[res["label"]] = {
                    "cd": np.array(res["cd"]),
                    "pc": np.array(res["pc"]),
                }
                print(f"  Collected: {label}")
            except Exception as exc:
                print(f"  FAILED {label}: {exc}")
                import traceback
                traceback.print_exc()

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sort_idx = np.argsort(PHI_APPLIED)
    eta_plot = PHI_APPLIED[sort_idx]

    out_dir = os.path.join(_ROOT, "StudyResults", "per_reaction_eeq")
    os.makedirs(out_dir, exist_ok=True)

    ordered_labels = [c["label"] for c in cases]
    colors = ["steelblue", "darkorange", "crimson", "purple"]

    # Peroxide current
    fig, ax = plt.subplots(figsize=(11, 7))
    for i, label in enumerate(ordered_labels):
        if label in results:
            r = results[label]
            ax.plot(eta_plot, r["pc"][sort_idx], "o-", color=colors[i],
                    linewidth=2, markersize=5, label=label)
    ax.set_xlabel(r"Dimensionless overpotential $\hat{\eta}$", fontsize=13)
    ax.set_ylabel("Peroxide current (scaled)", fontsize=13)
    ax.set_title("Peroxide Current: Per-Reaction E_eq (Tuned Solver)", fontsize=14)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, "peroxide_per_rxn_eeq_tuned.png")
    fig.savefig(p, dpi=150)
    print(f"\nSaved: {p}")
    plt.close(fig)

    # Total current
    fig, ax = plt.subplots(figsize=(11, 7))
    for i, label in enumerate(ordered_labels):
        if label in results:
            r = results[label]
            ax.plot(eta_plot, r["cd"][sort_idx], "o-", color=colors[i],
                    linewidth=2, markersize=5, label=label)
    ax.set_xlabel(r"Dimensionless overpotential $\hat{\eta}$", fontsize=13)
    ax.set_ylabel("Total current density (scaled)", fontsize=13)
    ax.set_title("Total Current: Per-Reaction E_eq (Tuned Solver)", fontsize=14)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, "total_current_per_rxn_eeq_tuned.png")
    fig.savefig(p, dpi=150)
    print(f"Saved: {p}")
    plt.close(fig)

    print(f"\nAll plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
