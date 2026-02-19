"""Generate report-ready current-density-proxy plots for Robin kappa studies.

This script runs three noise cases (0%, 2.5%, 5%) for the current-density
observable and writes:
1) synthetic data-generation plots, and
2) best-fit inference plots,
matching the plot structure used in the weekly report.
"""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass
from typing import List

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
import matplotlib.pyplot as plt

from Helpers.Infer_RobinKappa_from_flux_curve_helpers import (
    ForwardRecoveryConfig,
    RobinFluxCurveInferenceRequest,
    run_robin_kappa_flux_curve_inference,
)
from UnifiedInverse import build_default_solver_params
from Utils.robin_flux_experiment import SteadyStateConfig


@dataclass
class CaseConfig:
    noise_percent: float
    seed: int
    tag: str


def build_solver_options() -> dict:
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


def read_clean_noisy_current_density_csv(path: str, scale: float):
    phi = []
    clean = []
    noisy = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            phi.append(float(row["phi_applied"]))
            clean.append(scale * float(row["flux_clean"]))
            noisy_val = row.get("flux_noisy", "")
            if noisy_val is None or str(noisy_val).strip() == "":
                noisy.append(scale * float(row["flux_clean"]))
            else:
                noisy.append(scale * float(noisy_val))
    return np.asarray(phi, dtype=float), np.asarray(clean, dtype=float), np.asarray(noisy, dtype=float)


def main() -> None:
    base_solver_params = build_default_solver_params(
        n_species=2,
        order=1,
        dt=1e-1,
        t_end=20.0,
        z_vals=[1, -1],
        d_vals=[1.0, 1.0],
        a_vals=[0.0, 0.0],
        phi_applied=0.05,
        c0_vals=[0.1, 0.1],
        phi0=0.05,
        solver_options=build_solver_options(),
    )
    # Provisional unit convention for report plots:
    # use charge-weighted flux proxy without Faraday scaling and keep
    # scale=1 so values stay near the flux-space magnitude range.
    current_density_scale = 1.0

    steady = SteadyStateConfig(
        relative_tolerance=5e-4,
        absolute_tolerance=1e-7,
        consecutive_steps=4,
        max_steps=120,
        flux_observable="charge_proxy_no_f",
        verbose=False,
        print_every=10,
    )

    cases: List[CaseConfig] = [
        CaseConfig(noise_percent=0.0, seed=20260220, tag="0pct"),
        CaseConfig(noise_percent=2.5, seed=20260222, tag="2p5pct"),
        CaseConfig(noise_percent=5.0, seed=20260221, tag="5pct"),
    ]

    assets_dir = os.path.join("writeups", "assets")
    os.makedirs(assets_dir, exist_ok=True)

    summary_rows = []

    for case in cases:
        case_dir = os.path.join(assets_dir, f"robin_current_density_case_{case.tag}")
        os.makedirs(case_dir, exist_ok=True)

        target_csv_path = os.path.join(
            case_dir, "phi_applied_vs_steady_current_density_synthetic.csv"
        )
        request = RobinFluxCurveInferenceRequest(
            base_solver_params=base_solver_params,
            steady=steady,
            true_value=[1.0, 2.0],
            initial_guess=[5.0, 5.0],
            phi_applied_values=np.linspace(0.0, 0.04, 15),
            target_csv_path=target_csv_path,
            output_dir=case_dir,
            regenerate_target=True,
            target_noise_percent=float(case.noise_percent),
            target_seed=int(case.seed),
            observable_mode="charge_proxy_no_f",
            observable_species_index=None,
            observable_scale=current_density_scale,
            observable_label="charge-weighted boundary flux proxy (a.u.)",
            observable_title="Robin kappa inference from charge-weighted flux proxy",
            kappa_lower=1e-6,
            kappa_upper=20.0,
            optimizer_method="L-BFGS-B",
            optimizer_tolerance=None,
            optimizer_options={
                "maxiter": 20,
                "ftol": 1e-8,
                "gtol": 1e-4,
                "disp": True,
            },
            max_iters=8,
            gtol=1e-4,
            fail_penalty=1e9,
            print_point_gradients=False,
            blob_initial_condition=True,
            live_plot=False,
            live_plot_eval_lines=False,
            anisotropy_trigger_failed_points=4,
            anisotropy_trigger_failed_fraction=0.25,
            forward_recovery=ForwardRecoveryConfig(
                max_attempts=8,
                max_it_only_attempts=2,
                anisotropy_only_attempts=1,
                tolerance_relax_attempts=2,
                max_it_growth=1.5,
                max_it_cap=500,
                atol_relax_factor=10.0,
                rtol_relax_factor=10.0,
                ksp_rtol_relax_factor=10.0,
                line_search_schedule=("bt", "l2", "cp", "basic"),
                anisotropy_target_ratio=3.0,
                anisotropy_blend=0.5,
            ),
        )

        result = run_robin_kappa_flux_curve_inference(request)

        phi_vals, clean_vals, noisy_vals = read_clean_noisy_current_density_csv(
            target_csv_path, current_density_scale
        )

        gen_plot_path = os.path.join(
            assets_dir, f"robin_current_density_generation_{case.tag}.png"
        )
        plt.figure(figsize=(7, 4))
        plt.plot(phi_vals, clean_vals, "o-", linewidth=2, label="clean synthetic proxy")
        plt.plot(phi_vals, noisy_vals, "s--", linewidth=1.8, label=f"noisy ({case.noise_percent:.1f}%)")
        plt.xlabel("applied voltage phi_applied")
        plt.ylabel("charge-weighted boundary flux proxy (a.u.)")
        plt.title(f"Current-Density Proxy Data Generation ({case.noise_percent:.1f}% noise)")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(gen_plot_path, dpi=170)
        plt.close()

        fit_plot_path = os.path.join(
            assets_dir, f"robin_kappa_current_density_inference_{case.tag}.png"
        )
        plt.figure(figsize=(7, 4))
        plt.plot(
            result.phi_applied_values,
            result.target_flux,
            marker="o",
            linewidth=2,
            label="target proxy",
        )
        plt.plot(
            result.phi_applied_values,
            result.best_simulated_flux,
            marker="s",
            linewidth=2,
            label="best-fit simulated proxy",
        )
        plt.xlabel("applied voltage phi_applied")
        plt.ylabel("charge-weighted boundary flux proxy (a.u.)")
        plt.title(f"Robin Kappa Inference from Current-Density Proxy ({case.noise_percent:.1f}% noise)")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fit_plot_path, dpi=170)
        plt.close()

        summary_rows.append(
            {
                "case_tag": case.tag,
                "noise_percent": case.noise_percent,
                "seed": case.seed,
                "best_kappa0": float(result.best_kappa[0]),
                "best_kappa1": float(result.best_kappa[1]),
                "best_loss": float(result.best_loss),
                "gen_plot": gen_plot_path,
                "fit_plot": fit_plot_path,
            }
        )

        print(
            f"[{case.tag}] best_kappa={result.best_kappa.tolist()} "
            f"best_loss={result.best_loss:.6e}"
        )
        print(f"[{case.tag}] saved: {gen_plot_path}")
        print(f"[{case.tag}] saved: {fit_plot_path}")

    summary_path = os.path.join(assets_dir, "robin_current_density_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_tag",
                "noise_percent",
                "seed",
                "best_kappa0",
                "best_kappa1",
                "best_loss",
                "gen_plot",
                "fit_plot",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    print(f"Saved summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
