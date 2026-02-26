"""Quick benchmark: replay on vs off for no-noise current-density inference.

This script compares two otherwise-identical runs:
1) replay enabled
2) replay disabled

It records runtime, peak memory (RSS), optimization status, and fitted kappa.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PNPINVERSE_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _PNPINVERSE_ROOT not in sys.path:
    sys.path.insert(0, _PNPINVERSE_ROOT)

from FluxCurve import (
    ForwardRecoveryConfig,
    RobinFluxCurveInferenceRequest,
    run_robin_kappa_flux_curve_inference,
)
from Inverse import build_default_solver_params
from Nondim.compat import build_physical_scales_dict as build_physical_scales, build_solver_options
from Forward.steady_state import (
    SteadyStateConfig,
    add_percent_noise,
    all_results_converged,
    results_to_flux_array,
    sweep_phi_applied_steady_flux,
    write_phi_applied_flux_csv,
)


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration shared by both replay benchmark cases."""

    true_kappa: Sequence[float]
    initial_guess: Sequence[float]
    phi_applied_values: np.ndarray
    target_seed: int
    target_noise_percent: float


def build_base_solver_params(scales: Dict[str, Any]) -> Sequence[object]:
    """Create baseline solver params for current-density runs."""
    return build_default_solver_params(
        n_species=2,
        order=1,
        dt=1e-1,
        t_end=20.0,
        z_vals=[1, -1],
        d_vals=[float(scales["d_species_m2_s"][0]), float(scales["d_species_m2_s"][1])],
        a_vals=[0.0, 0.0],
        phi_applied=0.05,
        c0_vals=[
            float(scales["bulk_concentration_mol_m3"]),
            float(scales["bulk_concentration_mol_m3"]),
        ],
        phi0=0.05,
        solver_options=build_solver_options(scales),
    )


def build_steady_config() -> SteadyStateConfig:
    """Steady-state configuration used in the benchmark."""
    return SteadyStateConfig(
        relative_tolerance=5e-4,
        absolute_tolerance=1e-7,
        consecutive_steps=4,
        max_steps=120,
        flux_observable="total_charge",
        verbose=False,
        print_every=10,
    )


def prepare_target_csv(
    *,
    target_csv_path: str,
    base_solver_params: Sequence[object],
    steady: SteadyStateConfig,
    phi_applied_values: np.ndarray,
    true_kappa: Sequence[float],
    noise_percent: float,
    seed: int,
) -> None:
    """Generate one fixed target CSV shared by replay-on/off runs."""
    target_results = sweep_phi_applied_steady_flux(
        base_solver_params,
        phi_applied_values=phi_applied_values.tolist(),
        steady=steady,
        kappa_values=[float(v) for v in true_kappa],
        blob_initial_condition=False,
    )
    if not all_results_converged(target_results):
        failed = [f"{r.phi_applied:.6f}" for r in target_results if not r.converged]
        raise RuntimeError(
            "Target generation failed for some phi_applied values. "
            f"Failed values: {failed}"
        )
    clean = results_to_flux_array(target_results)
    noisy = add_percent_noise(clean, float(noise_percent), seed=int(seed))
    os.makedirs(os.path.dirname(target_csv_path), exist_ok=True)
    write_phi_applied_flux_csv(target_csv_path, target_results, noisy_flux=noisy)


def peak_rss_mb() -> float:
    """Return peak RSS in MB for the current process."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        # macOS reports bytes.
        return float(usage) / (1024.0 * 1024.0)
    # Linux reports KB.
    return float(usage) / 1024.0


def run_single_case(
    *,
    replay_enabled: bool,
    target_csv_path: str,
    output_dir: str,
    config: BenchmarkConfig,
    parallel_points_enabled: bool,
    parallel_point_workers: int,
    parallel_start_method: str = "spawn",
) -> Dict[str, Any]:
    """Run one no-noise current-density inference case and collect metrics."""
    scales = build_physical_scales()
    base_solver_params = build_base_solver_params(scales)
    steady = build_steady_config()

    request = RobinFluxCurveInferenceRequest(
        base_solver_params=base_solver_params,
        steady=steady,
        true_value=[float(v) for v in config.true_kappa],
        initial_guess=[float(v) for v in config.initial_guess],
        phi_applied_values=np.asarray(config.phi_applied_values, dtype=float),
        target_csv_path=target_csv_path,
        output_dir=output_dir,
        regenerate_target=False,
        target_noise_percent=float(config.target_noise_percent),
        target_seed=int(config.target_seed),
        observable_mode="total_charge",
        observable_species_index=None,
        observable_scale=float(scales["molar_flux_scale_mol_m2_s"]),
        observable_label="steady current density (A/m^2)",
        observable_title="Replay benchmark (current density)",
        kappa_lower=1e-6,
        kappa_upper=20.0,
        optimizer_method="L-BFGS-B",
        optimizer_tolerance=1e-12,
        optimizer_options={
            "maxiter": 20,
            "ftol": 1e-12,
            "gtol": 1e-8,
            "disp": True,
        },
        max_iters=8,
        gtol=1e-4,
        fail_penalty=1e9,
        print_point_gradients=False,
        blob_initial_condition=False,
        live_plot=False,
        live_plot_eval_lines=False,
        live_plot_export_gif_path=None,
        replay_mode_enabled=bool(replay_enabled),
        replay_reenable_after_successes=1,
        parallel_point_solves_enabled=bool(parallel_points_enabled),
        parallel_point_workers=int(max(1, parallel_point_workers)),
        parallel_point_min_points=4,
        parallel_start_method=str(parallel_start_method),
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

    t0 = time.perf_counter()
    result = run_robin_kappa_flux_curve_inference(request)
    elapsed = time.perf_counter() - t0

    kappa_scale_m_s = float(scales["kappa_scale_m_s"])

    out = {
        "replay_enabled": int(bool(replay_enabled)),
        "parallel_point_solves_enabled": int(bool(parallel_points_enabled)),
        "parallel_point_workers": int(max(1, parallel_point_workers)),
        "runtime_seconds": float(elapsed),
        "peak_rss_mb": float(peak_rss_mb()),
        "best_kappa0": float(result.best_kappa[0]),
        "best_kappa1": float(result.best_kappa[1]),
        "best_kappa0_m_s": kappa_scale_m_s * float(result.best_kappa[0]),
        "best_kappa1_m_s": kappa_scale_m_s * float(result.best_kappa[1]),
        "best_loss": float(result.best_loss),
        "optimization_success": int(bool(result.optimization_success)),
        "optimization_message": str(result.optimization_message),
        "fit_csv_path": str(result.fit_csv_path),
        "fit_plot_path": "" if result.fit_plot_path is None else str(result.fit_plot_path),
    }
    return out


def _stream_and_capture(cmd: List[str], *, cwd: str) -> Dict[str, Any]:
    """Run subprocess, stream logs, and parse RESULT_JSON payload."""
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    payload: Dict[str, Any] = {}
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        if line.startswith("RESULT_JSON:"):
            payload = json.loads(line.split("RESULT_JSON:", 1)[1].strip())
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"Benchmark subprocess failed with exit code {ret}.")
    if not payload:
        raise RuntimeError("Benchmark subprocess did not emit RESULT_JSON payload.")
    return payload


def write_summary_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    """Write replay benchmark summary to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = [
        "replay_enabled",
        "parallel_point_solves_enabled",
        "parallel_point_workers",
        "runtime_seconds",
        "peak_rss_mb",
        "best_kappa0",
        "best_kappa1",
        "best_kappa0_m_s",
        "best_kappa1_m_s",
        "best_loss",
        "optimization_success",
        "optimization_message",
        "fit_csv_path",
        "fit_plot_path",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser()
    p.add_argument("--single", action="store_true", help="Run one case and emit RESULT_JSON.")
    p.add_argument("--replay", type=int, default=1, help="1 to enable replay, 0 to disable.")
    p.add_argument(
        "--parallel-points",
        type=int,
        default=0,
        help="1 to enable process-parallel point solves, 0 for serial.",
    )
    p.add_argument(
        "--parallel-workers",
        type=int,
        default=4,
        help="Worker count when --parallel-points=1.",
    )
    p.add_argument(
        "--parallel-start-method",
        type=str,
        default="spawn",
        help="multiprocessing start method (spawn/forkserver).",
    )
    p.add_argument(
        "--target-csv",
        type=str,
        default="StudyResults/robin_current_density_experiment/replay_benchmark_target_no_noise.csv",
    )
    p.add_argument(
        "--output-root",
        type=str,
        default="StudyResults/robin_current_density_experiment/replay_benchmark_no_noise",
    )
    return p.parse_args()


def main() -> None:
    """Entry point for replay benchmark."""
    args = parse_args()
    config = BenchmarkConfig(
        true_kappa=[1.0, 2.0],
        initial_guess=[5.0, 5.0],
        phi_applied_values=np.linspace(0.0, 0.04, 15),
        target_seed=20260222,
        target_noise_percent=0.0,
    )

    if args.single:
        replay_enabled = bool(int(args.replay))
        parallel_points_enabled = bool(int(args.parallel_points))
        case_name = "replay_on" if replay_enabled else "replay_off"
        output_dir = os.path.join(str(args.output_root), case_name)
        os.makedirs(output_dir, exist_ok=True)
        result = run_single_case(
            replay_enabled=replay_enabled,
            target_csv_path=str(args.target_csv),
            output_dir=output_dir,
            config=config,
            parallel_points_enabled=parallel_points_enabled,
            parallel_point_workers=int(args.parallel_workers),
            parallel_start_method=str(args.parallel_start_method),
        )
        print("RESULT_JSON:" + json.dumps(result, sort_keys=True))
        return

    os.makedirs(os.path.dirname(str(args.target_csv)), exist_ok=True)
    scales = build_physical_scales()
    base_solver_params = build_base_solver_params(scales)
    steady = build_steady_config()

    print("Preparing shared no-noise current-density target curve (A/m^2 scaling)...")
    prepare_target_csv(
        target_csv_path=str(args.target_csv),
        base_solver_params=base_solver_params,
        steady=steady,
        phi_applied_values=np.asarray(config.phi_applied_values, dtype=float),
        true_kappa=config.true_kappa,
        noise_percent=float(config.target_noise_percent),
        seed=int(config.target_seed),
    )
    print(f"Target ready: {args.target_csv}")

    rows: List[Dict[str, Any]] = []
    cwd = os.getcwd()
    for replay_flag in (1, 0):
        case = "replay_on" if replay_flag == 1 else "replay_off"
        print("\n" + "=" * 92)
        print(f"Running case: {case}")
        print("=" * 92)
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--single",
            "--replay",
            str(replay_flag),
            "--parallel-points",
            "0",
            "--parallel-workers",
            str(int(args.parallel_workers)),
            "--parallel-start-method",
            str(args.parallel_start_method),
            "--target-csv",
            str(args.target_csv),
            "--output-root",
            str(args.output_root),
        ]
        row = _stream_and_capture(cmd, cwd=cwd)
        rows.append(row)

    summary_csv = os.path.join(str(args.output_root), "replay_runtime_memory_summary.csv")
    write_summary_csv(summary_csv, rows)

    by_flag = {int(r["replay_enabled"]): r for r in rows}
    if 1 in by_flag and 0 in by_flag:
        t_on = float(by_flag[1]["runtime_seconds"])
        t_off = float(by_flag[0]["runtime_seconds"])
        m_on = float(by_flag[1]["peak_rss_mb"])
        m_off = float(by_flag[0]["peak_rss_mb"])
        speedup = t_off / t_on if t_on > 0.0 else float("nan")
        mem_ratio = m_on / m_off if m_off > 0.0 else float("nan")
        print("\n" + "-" * 92)
        print(
            f"Replay ON runtime:  {t_on:.3f} s | peak RSS: {m_on:.1f} MB | "
            f"kappa=[{float(by_flag[1]['best_kappa0']):.6f}, {float(by_flag[1]['best_kappa1']):.6f}] | "
            f"loss={float(by_flag[1]['best_loss']):.6e}"
        )
        print(
            f"Replay OFF runtime: {t_off:.3f} s | peak RSS: {m_off:.1f} MB | "
            f"kappa=[{float(by_flag[0]['best_kappa0']):.6f}, {float(by_flag[0]['best_kappa1']):.6f}] | "
            f"loss={float(by_flag[0]['best_loss']):.6e}"
        )
        print(
            f"Speedup (OFF/ON): {speedup:.3f}x | Replay-on memory ratio (ON/OFF): {mem_ratio:.3f}x"
        )
        print("-" * 92)

    print(f"Saved summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
