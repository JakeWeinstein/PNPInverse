"""Generate phi_applied-vs-steady-state-flux data for Robin boundary experiments.

This script mirrors an experimental voltage sweep:
1) apply each phi_applied value,
2) solve until steady-state flux criterion is met,
3) record steady-state boundary flux.

Run:
    python Studies/Generate_RobinFlux_vs_phi0_data.py
"""

from __future__ import annotations

import os
import sys

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
    add_percent_noise,
    all_results_converged,
    results_to_flux_array,
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


def main() -> None:
    dt = 1e-1
    # Fast run in a known-stable region.
    phi_applied_values = np.linspace(0.0, 0.04, 15)
    true_kappa = [0.8, 0.8]
    flux_noise_percent = 2.0
    seed = 20260220

    base_solver_params = build_default_solver_params(
        n_species=2,
        order=1,
        dt=dt,
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
        relative_tolerance=5e-4,
        absolute_tolerance=1e-7,
        consecutive_steps=4,
        max_steps=120,
        flux_observable="total_species",
        verbose=False,
        print_every=10,
    )

    print("=== Generate phi_applied vs Steady-State Flux Data ===")
    print(
        f"phi_applied range: [{phi_applied_values.min():.4f}, {phi_applied_values.max():.4f}]"
    )
    print(f"true kappa: {true_kappa}")
    print(f"flux noise percent: {flux_noise_percent}")

    # Target curve used for synthetic dataset.
    results = sweep_phi_applied_steady_flux(
        base_solver_params,
        phi_applied_values=phi_applied_values.tolist(),
        steady=steady,
        kappa_values=true_kappa,
        blob_initial_condition=False,
    )
    if not all_results_converged(results):
        failed = [f"{r.phi_applied:.6f}" for r in results if not r.converged]
        raise RuntimeError(
            "Data generation sweep has non-converged points; adjust phi_applied range, dt, "
            f"or solver options. Failed phi_applied values: {failed}"
        )

    clean_flux = results_to_flux_array(results)
    noisy_flux = add_percent_noise(clean_flux, flux_noise_percent, seed=seed)

    out_dir = os.path.join("StudyResults", "robin_flux_experiment")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "phi_applied_vs_steady_flux_synthetic.csv")
    fig_path = os.path.join(out_dir, "phi_applied_vs_steady_flux_synthetic.png")

    write_phi_applied_flux_csv(csv_path, results, noisy_flux=noisy_flux)
    print(f"Saved dataset CSV: {csv_path}")

    if plt is None:
        print("matplotlib not available; skipping curve plot generation.")
    else:
        # Plot clean + noisy curve for quick visual check.
        plt.figure(figsize=(7, 4))
        plt.plot(
            phi_applied_values,
            clean_flux,
            marker="o",
            linewidth=2,
            label=f"clean (kappa={true_kappa})",
        )
        plt.plot(
            phi_applied_values,
            noisy_flux,
            marker="s",
            linewidth=2,
            label=f"noisy (kappa={true_kappa})",
            alpha=0.9,
        )
        plt.xlabel("applied voltage phi_applied")
        plt.ylabel("steady-state flux (observable)")
        plt.title(f"Robin Boundary: phi_applied vs Flux (kappa={true_kappa})")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path, dpi=160)
        plt.close()
        print(f"Saved curve plot: {fig_path}")

    n_fail = sum(0 if r.converged else 1 for r in results)
    print(f"Steady-state solve failures in sweep: {n_fail}/{len(results)}")


if __name__ == "__main__":
    main()
