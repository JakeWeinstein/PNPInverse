"""Test script: overlay steady-state Robin observable curves for multiple kappa values.

This script runs repeated phi_applied sweeps, one per kappa pair, and plots all
phi_applied-vs-observable curves on the same graph for visual comparison.

Run examples:
    python Studies/Test_RobinFlux_kappa_overlay.py
    python Studies/Test_RobinFlux_kappa_overlay.py --kappa-list "0.8,0.8;2,2;0.8,2"
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Sequence

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PNPINVERSE_ROOT = os.path.dirname(_THIS_DIR)
if _PNPINVERSE_ROOT not in sys.path:
    sys.path.insert(0, _PNPINVERSE_ROOT)

# Keep Firedrake cache paths writable in sandboxed/restricted environments.
os.environ.setdefault("FIREDRAKE_TSFC_KERNEL_CACHE_DIR", "/tmp/firedrake-tsfc")
os.environ.setdefault("PYOP2_CACHE_DIR", "/tmp/pyop2")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from UnifiedInverse import build_default_solver_params
from Utils.robin_flux_experiment import (
    SteadyStateConfig,
    sweep_phi_applied_steady_flux,
    write_phi_applied_flux_csv,
)


def build_solver_options() -> dict:
    """Return PETSc/SNES options used for Robin flux studies."""
    return {
        "snes_type": "newtonls",
        "snes_max_it": 100,
        "snes_atol": 1e-8,
        "snes_rtol": 1e-8,
        "snes_linesearch_type": "bt",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "robin_bc": {
            "kappa": [0.8, 0.8],
            "c_inf": [0.01, 0.01],
            "electrode_marker": 1,
            "concentration_marker": 3,
            "ground_marker": 3,
        },
    }


def parse_kappa_list(raw: str) -> List[List[float]]:
    """Parse kappa list string of form 'a,b;c,d;...'.

    Each semicolon-separated entry is one 2-species kappa pair.
    """
    entries = [chunk.strip() for chunk in raw.split(";") if chunk.strip()]
    if not entries:
        raise ValueError("kappa list is empty. Example: '0.8,0.8;2,2;0.8,2'")

    parsed: List[List[float]] = []
    for entry in entries:
        parts = [p.strip() for p in entry.split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError(
                f"Invalid kappa pair '{entry}'. Each pair must have two values."
            )
        pair = [float(parts[0]), float(parts[1])]
        if pair[0] <= 0.0 or pair[1] <= 0.0:
            raise ValueError(f"Invalid kappa pair '{entry}'. Values must be > 0.")
        parsed.append(pair)
    return parsed


def format_kappa_label(kappa: Sequence[float]) -> str:
    """Return compact legend label for one kappa pair."""
    return f"kappa=[{kappa[0]:.3f}, {kappa[1]:.3f}]"


def kappa_file_suffix(kappa: Sequence[float]) -> str:
    """Create filesystem-safe suffix encoding one kappa pair."""
    def _clean(v: float) -> str:
        return f"{v:.4f}".replace("-", "m").replace(".", "p")

    return f"kappa_{_clean(float(kappa[0]))}_{_clean(float(kappa[1]))}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay phi_applied-vs-steady-state observable curves for multiple Robin kappa pairs."
        )
    )
    parser.add_argument(
        "--kappa-list",
        type=str,
        default="0.8,0.8;2.0,2.0;0.8,2.0;2.0,0.8",
        help="Semicolon-separated kappa pairs: 'k11,k12;k21,k22;...'",
    )
    parser.add_argument("--phi-min", type=float, default=0.0)
    parser.add_argument("--phi-max", type=float, default=0.04)
    parser.add_argument("--n-points", type=int, default=15)
    parser.add_argument("--dt", type=float, default=1e-1)
    parser.add_argument(
        "--observable-mode",
        type=str,
        default="total_species",
        choices=["total_species", "total_charge", "charge_proxy_no_f", "species"],
        help=(
            "Scalar observable assembled from species boundary fluxes. "
            "'charge_proxy_no_f' is charge-weighted flux without Faraday scaling."
        ),
    )
    parser.add_argument(
        "--species-index",
        type=int,
        default=None,
        help="Required when --observable-mode=species.",
    )
    parser.add_argument(
        "--observable-scale",
        type=float,
        default=1.0,
        help="Post-assembly multiplicative scale applied to plotted values.",
    )
    parser.add_argument(
        "--y-label",
        type=str,
        default="steady-state flux (observable)",
        help="Y-axis label for plot.",
    )
    parser.add_argument(
        "--plot-title",
        type=str,
        default="Robin Boundary: phi_applied vs Flux for Multiple Kappa Pairs",
        help="Plot title.",
    )
    parser.add_argument("--steady-max-steps", type=int, default=120)
    parser.add_argument("--steady-rel-tol", type=float, default=5e-4)
    parser.add_argument("--steady-abs-tol", type=float, default=1e-7)
    parser.add_argument("--steady-consecutive", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("StudyResults", "robin_flux_experiment"),
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="phi_applied_vs_steady_flux_kappa_overlay",
    )
    args = parser.parse_args()

    kappa_pairs = parse_kappa_list(args.kappa_list)
    phi_applied_values = np.linspace(float(args.phi_min), float(args.phi_max), int(args.n_points))

    base_solver_params = build_default_solver_params(
        n_species=2,
        order=1,
        dt=float(args.dt),
        t_end=20.0,
        z_vals=[1, -1],
        d_vals=[1.0, 1.0],
        a_vals=[0.0, 0.0],
        phi_applied=0.05,
        c0_vals=[0.1, 0.1],
        phi0=0.05,
        solver_options=build_solver_options(),
    )

    steady = SteadyStateConfig(
        relative_tolerance=float(args.steady_rel_tol),
        absolute_tolerance=float(args.steady_abs_tol),
        consecutive_steps=int(args.steady_consecutive),
        max_steps=int(args.steady_max_steps),
        flux_observable=str(args.observable_mode),
        species_index=args.species_index,
        verbose=False,
        print_every=10,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    print("=== Robin Flux Kappa Overlay Test ===")
    print(
        f"phi_applied range: [{phi_applied_values.min():.4f}, {phi_applied_values.max():.4f}] "
        f"with {len(phi_applied_values)} points"
    )
    print(
        f"observable mode: {args.observable_mode}, "
        f"species_index={args.species_index}, scale={float(args.observable_scale):.8g}"
    )
    print(f"kappa pairs: {kappa_pairs}")

    overlay_data = []
    for kappa in kappa_pairs:
        results = sweep_phi_applied_steady_flux(
            base_solver_params,
            phi_applied_values=phi_applied_values.tolist(),
            steady=steady,
            kappa_values=kappa,
            blob_initial_condition=True,
        )
        converged = np.asarray([bool(r.converged) for r in results], dtype=bool)
        observable_vals = np.asarray(
            [float(args.observable_scale) * float(r.observed_flux) if r.converged else np.nan for r in results],
            dtype=float,
        )

        suffix = kappa_file_suffix(kappa)
        csv_path = os.path.join(args.output_dir, f"{args.output_prefix}_{suffix}.csv")
        write_phi_applied_flux_csv(csv_path, results, noisy_flux=None)

        n_ok = int(np.sum(converged))
        n_total = int(converged.size)
        print(
            f"{format_kappa_label(kappa):>24}  converged points: {n_ok:>2d}/{n_total:<2d}  "
            f"csv: {csv_path}"
        )

        overlay_data.append(
            {
                "kappa": list(kappa),
                "converged": converged,
                "observable": observable_vals,
            }
        )

    if plt is None:
        print("matplotlib not available; skipping overlay plot.")
        return

    fig_path = os.path.join(args.output_dir, f"{args.output_prefix}.png")
    plt.figure(figsize=(8, 4.75))
    for item in overlay_data:
        kappa = item["kappa"]
        converged = item["converged"]
        observable_vals = item["observable"]
        label = format_kappa_label(kappa)

        plt.plot(
            phi_applied_values,
            observable_vals,
            marker="o",
            linewidth=2,
            label=label,
        )
        # Mark non-converged points at y=0 with x-marker so failures are visible.
        failed_idx = np.where(~converged)[0]
        if failed_idx.size > 0:
            plt.scatter(
                phi_applied_values[failed_idx],
                np.zeros_like(failed_idx, dtype=float),
                marker="x",
                s=55,
                linewidths=1.5,
                label=f"{label} (non-converged)",
            )

    plt.xlabel("applied voltage phi_applied")
    plt.ylabel(str(args.y_label))
    plt.title(str(args.plot_title))
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=170)
    plt.close()
    print(f"Saved overlay plot: {fig_path}")


if __name__ == "__main__":
    main()
