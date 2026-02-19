"""Benchmark runtime impact of Robin-kappa initial-condition choice.

This script runs one no-noise Robin-kappa curve-inference problem twice:
1) blob initial condition
2) non-blob (uniform/linear) initial condition

Both runs use the same optimization settings so runtime and convergence can be
compared directly.
"""

from __future__ import annotations

import copy
import csv
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PNPINVERSE_ROOT = os.path.dirname(_THIS_DIR)
if _PNPINVERSE_ROOT not in sys.path:
    sys.path.insert(0, _PNPINVERSE_ROOT)

from Helpers.Infer_RobinKappa_from_flux_curve_helpers import (
    ForwardRecoveryConfig,
    RobinFluxCurveInferenceRequest,
    RobinFluxCurveInferenceResult,
    run_robin_kappa_flux_curve_inference,
)
from UnifiedInverse import build_default_solver_params
from Utils.robin_flux_experiment import SteadyStateConfig


@dataclass(frozen=True)
class CaseConfig:
    """Configuration for one benchmark case."""

    label: str
    blob_initial_condition: bool


def build_solver_options() -> Dict[str, Any]:
    """Return PETSc/SNES options used across both benchmark cases."""
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


def build_request(case: CaseConfig, output_root: str) -> RobinFluxCurveInferenceRequest:
    """Build a no-noise inference request for one IC case."""
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

    steady = SteadyStateConfig(
        relative_tolerance=5e-4,
        absolute_tolerance=1e-7,
        consecutive_steps=4,
        max_steps=120,
        flux_observable="total_species",
        verbose=False,
        print_every=10,
    )

    case_dir = os.path.join(output_root, case.label)
    return RobinFluxCurveInferenceRequest(
        base_solver_params=base_solver_params,
        steady=steady,
        true_value=[1.0, 2.0],
        initial_guess=[5.0, 5.0],
        phi_applied_values=np.linspace(0.0, 0.04, 15),
        target_csv_path=os.path.join(case_dir, "target_no_noise.csv"),
        output_dir=case_dir,
        regenerate_target=True,
        target_noise_percent=0.0,
        target_seed=20260222,
        kappa_lower=1e-6,
        kappa_upper=20.0,
        optimizer_method="L-BFGS-B",
        optimizer_tolerance=1e-12,
        optimizer_options={
            "maxiter": 80,
            "ftol": 1e-12,
            "gtol": 1e-8,
            "disp": True,
        },
        max_iters=8,
        gtol=1e-4,
        fail_penalty=1e9,
        print_point_gradients=False,
        blob_initial_condition=bool(case.blob_initial_condition),
        live_plot=False,
        live_plot_pause_seconds=0.001,
        live_plot_eval_lines=False,
        live_plot_eval_line_alpha=0.30,
        live_plot_eval_max_lines=120,
        live_plot_export_gif_path=None,
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


def run_case(case: CaseConfig, output_root: str) -> Dict[str, Any]:
    """Run one inference case and return metrics for comparison."""
    request = build_request(case, output_root)
    request_runtime = copy.deepcopy(request)
    t0 = time.perf_counter()
    result: RobinFluxCurveInferenceResult = run_robin_kappa_flux_curve_inference(request_runtime)
    runtime_s = time.perf_counter() - t0
    return {
        "case": case.label,
        "blob_initial_condition": bool(case.blob_initial_condition),
        "runtime_seconds": float(runtime_s),
        "best_kappa0": float(result.best_kappa[0]),
        "best_kappa1": float(result.best_kappa[1]),
        "best_loss": float(result.best_loss),
        "optimization_success": bool(result.optimization_success),
        "optimization_message": str(result.optimization_message),
    }


def write_summary(path: str, rows: List[Dict[str, Any]]) -> None:
    """Write comparison rows to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "case",
        "blob_initial_condition",
        "runtime_seconds",
        "best_kappa0",
        "best_kappa1",
        "best_loss",
        "optimization_success",
        "optimization_message",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    """Run the no-noise IC runtime benchmark and print summary."""
    output_root = os.path.join(
        "StudyResults",
        "robin_flux_experiment",
        "ic_runtime_benchmark_no_noise",
    )
    os.makedirs(output_root, exist_ok=True)

    cases = [
        CaseConfig(label="blob_ic", blob_initial_condition=True),
        CaseConfig(label="flat_ic", blob_initial_condition=False),
    ]

    rows: List[Dict[str, Any]] = []
    for case in cases:
        print("\n" + "=" * 90)
        print(
            f"Running case={case.label:>8s}  "
            f"blob_initial_condition={str(case.blob_initial_condition):>5s}"
        )
        print("=" * 90)
        row = run_case(case, output_root)
        rows.append(row)
        print(
            "Case summary: "
            f"runtime={row['runtime_seconds']:.3f}s  "
            f"kappa=[{row['best_kappa0']:.6f}, {row['best_kappa1']:.6f}]  "
            f"loss={row['best_loss']:.8f}  "
            f"success={row['optimization_success']}"
        )

    csv_path = os.path.join(output_root, "ic_runtime_comparison_no_noise.csv")
    write_summary(csv_path, rows)

    by_case = {str(r["case"]): r for r in rows}
    if "blob_ic" in by_case and "flat_ic" in by_case:
        blob_t = float(by_case["blob_ic"]["runtime_seconds"])
        flat_t = float(by_case["flat_ic"]["runtime_seconds"])
        speedup = blob_t / flat_t if flat_t > 0.0 else float("nan")
        gain_pct = (1.0 - (flat_t / blob_t)) * 100.0 if blob_t > 0.0 else float("nan")
        print("\n" + "-" * 90)
        print(
            "Runtime comparison (blob -> flat): "
            f"{blob_t:.3f}s -> {flat_t:.3f}s  "
            f"(speedup={speedup:.4f}x, gain={gain_pct:.2f}%)"
        )
        print("-" * 90)
    print(f"\nSaved benchmark summary CSV: {csv_path}")


if __name__ == "__main__":
    main()
