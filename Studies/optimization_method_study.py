#!/usr/bin/env python3
import argparse
import base64
import contextlib
import csv
import io
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PNPINVERSE_ROOT = SCRIPT_DIR.parent
if str(PNPINVERSE_ROOT) not in sys.path:
    sys.path.insert(0, str(PNPINVERSE_ROOT))

# Keep Firedrake caches writable in sandboxed environments.
os.environ.setdefault("FIREDRAKE_TSFC_KERNEL_CACHE_DIR", "/tmp/firedrake-tsfc")
os.environ.setdefault("PYOP2_CACHE_DIR", "/tmp/pyop2")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("OMP_NUM_THREADS", "1")

METHODS = ["BFGS", "L-BFGS-B", "CG", "SLSQP", "TNC", "Newton-CG"]
BOUNDED_METHODS = {"L-BFGS-B", "SLSQP", "TNC"}
# Percent noise levels (sigma = pct/100 * RMS(field)).
NOISE_STDS = [0.0, 5.0, 20.0]
BASE_SEED = 20260211
NOISE_ZERO_SEEDS = [BASE_SEED]
NOISY_SEEDS = [BASE_SEED, BASE_SEED + 1, BASE_SEED + 2]

D_TRUE_VALUES = [
    [1.0, 3.0],
    [1.0, 1.0],
    [0.5, 2.0],
]
D_GUESSES = [
    [0.3, 0.3],
    [10.0, 10.0],
]

PHI_TRUE_VALUES = [0.5, 1.5]
PHI_GUESSES = [0.2, 10.0]


def build_common_solver_options():
    return {
        "snes_type": "newtonls",
        "snes_max_it": 80,
        "snes_atol": 1e-8,
        "snes_rtol": 1e-8,
        "snes_linesearch_type": "bt",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }


def rss_to_mib(ru_maxrss: float) -> float:
    # On macOS ru_maxrss is bytes; on Linux it's KiB.
    if ru_maxrss > 10_000_000:
        return float(ru_maxrss) / (1024.0 * 1024.0)
    return float(ru_maxrss) / 1024.0


def tail_lines(text: str, n: int = 12) -> str:
    lines = text.strip().splitlines()
    if not lines:
        return ""
    return "\n".join(lines[-n:])


def compact_log(text: str) -> str:
    return text.replace("\r", "").replace("\n", "\\n")


def as_float_list(vals):
    return [float(v) for v in vals]


def seeds_for_noise(noise_std: float):
    if abs(float(noise_std)) < 1e-15:
        return NOISE_ZERO_SEEDS
    return NOISY_SEEDS


def run_single(config):
    import firedrake.adjoint as adj

    from Helpers.Infer_D_from_data_helpers import make_objective_and_grad as make_d_objective
    from Helpers.Infer_DirichletBC_from_data_helpers import make_objective_and_grad as make_phi_objective
    from Utils.generate_noisy_data import generate_noisy_data

    method = config["method"]
    problem = config["problem"]
    noise_std = float(config["noise_std"])
    seed = int(config["seed"])

    params = build_common_solver_options()
    n_species = 2
    z_vals = [1, -1]
    a_vals = [0.0, 0.0]
    c0_vals = [0.1, 0.1]

    # Use identical temporal discretization across study cases.
    dt = 2e-2
    t_end = 0.1

    run_log = io.StringIO()
    start = time.perf_counter()
    success = False
    failure_reason = ""
    objective_value = None
    est_value = None
    err_abs = None
    err_rel = None

    try:
        with contextlib.redirect_stdout(run_log), contextlib.redirect_stderr(run_log):
            if problem == "infer_D":
                d_true = as_float_list(config["d_true"])
                d_guess = as_float_list(config["d_guess"])
                phi0_fixed = 1.0

                solver_params_gen = [
                    n_species,
                    1,
                    dt,
                    t_end,
                    z_vals,
                    d_true,
                    a_vals,
                    0.05,
                    c0_vals,
                    phi0_fixed,
                    params,
                ]
                with adj.stop_annotating():
                    data = generate_noisy_data(solver_params_gen, noise_percent=noise_std, seed=seed)

                noisy_c = list(data[n_species + 1 : 2 * n_species + 1])

                solver_params_inv = [
                    n_species,
                    1,
                    dt,
                    t_end,
                    z_vals,
                    d_guess,
                    a_vals,
                    0.05,
                    c0_vals,
                    phi0_fixed,
                    params,
                ]
                rf = make_d_objective(solver_params_inv, noisy_c)
                opt_kwargs = {
                    "tol": 1e-8,
                    "options": {"disp": False, "maxiter": 60},
                }
                opt_controls = adj.minimize(rf, method, **opt_kwargs)
                d_est = [float(np.exp(v.dat.data[0])) for v in opt_controls]
                d_true_arr = np.array(d_true)
                d_est_arr = np.array(d_est)
                objective_value = float(rf(opt_controls))
                est_value = d_est
                err_abs = float(np.linalg.norm(d_est_arr - d_true_arr, ord=2))
                denom = float(np.linalg.norm(d_true_arr, ord=2))
                err_rel = float(err_abs / denom) if denom > 0 else float("nan")
                success = True

            elif problem == "infer_phi0":
                d_fixed = [1.0, 3.0]
                phi_true = float(config["phi_true"])
                phi_guess = float(config["phi_guess"])

                solver_params_gen = [
                    n_species,
                    1,
                    dt,
                    t_end,
                    z_vals,
                    d_fixed,
                    a_vals,
                    0.05,
                    c0_vals,
                    phi_true,
                    params,
                ]
                with adj.stop_annotating():
                    data = generate_noisy_data(solver_params_gen, noise_percent=noise_std, seed=seed)

                noisy_c = list(data[n_species + 1 : 2 * n_species + 1])
                noisy_phi = data[-1]

                solver_params_inv = [
                    n_species,
                    1,
                    dt,
                    t_end,
                    z_vals,
                    d_fixed,
                    a_vals,
                    0.05,
                    c0_vals,
                    phi_guess,
                    params,
                ]
                rf = make_phi_objective(solver_params_inv, noisy_c, noisy_phi)

                opt_kwargs = {
                    "tol": 1e-8,
                    "options": {"disp": False, "maxiter": 60},
                }
                if method in BOUNDED_METHODS:
                    opt_kwargs["bounds"] = (1e-8, None)

                opt_control = adj.minimize(rf, method, **opt_kwargs)
                phi_est = float(opt_control.dat.data[0])
                objective_value = float(rf(opt_control))
                est_value = phi_est
                err_abs = abs(phi_est - phi_true)
                err_rel = abs(phi_est - phi_true) / abs(phi_true) if abs(phi_true) > 0 else float("nan")
                success = True
            else:
                raise ValueError(f"Unknown problem '{problem}'")
    except Exception as exc:
        failure_reason = f"{type(exc).__name__}: {exc}"

    elapsed = time.perf_counter() - start
    peak_mib = rss_to_mib(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    return {
        "problem": problem,
        "method": method,
        "noise_std": noise_std,
        "seed": seed,
        "d_true": config.get("d_true"),
        "d_guess": config.get("d_guess"),
        "phi_true": config.get("phi_true"),
        "phi_guess": config.get("phi_guess"),
        "success": success,
        "failure_reason": failure_reason,
        "time_s": float(elapsed),
        "peak_rss_mib": float(peak_mib),
        "objective": objective_value,
        "estimate": est_value,
        "error_abs": err_abs,
        "error_rel": err_rel,
        "log_tail": compact_log(tail_lines(run_log.getvalue())),
    }


def build_configs():
    configs = []
    for noise_std in NOISE_STDS:
        for seed in seeds_for_noise(noise_std):
            for method in METHODS:
                for d_true in D_TRUE_VALUES:
                    for d_guess in D_GUESSES:
                        configs.append(
                            {
                                "problem": "infer_D",
                                "method": method,
                                "noise_std": noise_std,
                                "seed": int(seed),
                                "d_true": d_true,
                                "d_guess": d_guess,
                            }
                        )
                for phi_true in PHI_TRUE_VALUES:
                    for phi_guess in PHI_GUESSES:
                        configs.append(
                            {
                                "problem": "infer_phi0",
                                "method": method,
                                "noise_std": noise_std,
                                "seed": int(seed),
                                "phi_true": phi_true,
                                "phi_guess": phi_guess,
                            }
                        )
    return configs


def median_or_nan(values):
    vals = [v for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return float("nan")
    return float(statistics.median(vals))


def is_converged_run(row):
    # Converged runs are the only runs that should contribute to median metrics.
    return bool(row.get("success")) and not (row.get("failure_reason") or "").strip()


def aggregate_summary(results):
    groups = {}
    for r in results:
        key = (r["problem"], r["method"], r["noise_std"])
        groups.setdefault(key, []).append(r)

    summary = []
    for key in sorted(groups):
        rows = groups[key]
        converged = [r for r in rows if is_converged_run(r)]
        summary.append(
            {
                "problem": key[0],
                "method": key[1],
                "noise_std": key[2],
                "n_runs": len(rows),
                "n_success": len(converged),
                "success_rate": len(converged) / len(rows),
                "median_time_s": median_or_nan([r["time_s"] for r in converged]),
                "median_peak_rss_mib": median_or_nan([r["peak_rss_mib"] for r in converged]),
                "median_error_rel": median_or_nan([r["error_rel"] for r in converged]),
                "median_objective": median_or_nan([r["objective"] for r in converged]),
            }
        )
    return summary


def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def format_float(x):
    if x is None:
        return ""
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return "nan"
    return f"{x:.4g}" if isinstance(x, float) else str(x)


def write_markdown(path: Path, summary_rows, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    fail_rows = [r for r in results if not r["success"]]

    lines = []
    lines.append("# Optimization Method Study")
    lines.append("")
    no_noise_seeds = sorted({int(r["seed"]) for r in results if abs(float(r["noise_std"])) < 1e-15})
    noisy_seeds = sorted({int(r["seed"]) for r in results if float(r["noise_std"]) > 0.0})
    lines.append(f"Noise std = 0 seeds: {no_noise_seeds}")
    lines.append(f"Noise std > 0 seeds: {noisy_seeds}")
    lines.append("")
    lines.append(f"Total runs: {len(results)}")
    lines.append(f"Failed runs: {len(fail_rows)}")
    lines.append("Median time/memory/error statistics use converged runs only.")
    lines.append("")

    lines.append("## Summary by Problem / Method / Noise")
    lines.append("")
    lines.append("| problem | method | noise_std | success_rate | n_success/n_runs | median_time_s | median_peak_rss_mib | median_rel_error | median_objective |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r["problem"]),
                    str(r["method"]),
                    format_float(r["noise_std"]),
                    format_float(r["success_rate"]),
                    f'{r["n_success"]}/{r["n_runs"]}',
                    format_float(r["median_time_s"]),
                    format_float(r["median_peak_rss_mib"]),
                    format_float(r["median_error_rel"]),
                    format_float(r["median_objective"]),
                ]
            )
            + " |"
        )

    if fail_rows:
        lines.append("")
        lines.append("## Failure Log")
        lines.append("")
        lines.append("| problem | method | noise_std | case | reason |")
        lines.append("|---|---|---:|---|---|")
        for r in fail_rows:
            case = (
                f"d_true={r['d_true']}, d_guess={r['d_guess']}"
                if r["problem"] == "infer_D"
                else f"phi_true={r['phi_true']}, phi_guess={r['phi_guess']}"
            )
            reason = (r["failure_reason"] or "").replace("|", "/")[:140]
            lines.append(
                f"| {r['problem']} | {r['method']} | {format_float(r['noise_std'])} | {case} | {reason} |"
            )

    path.write_text("\n".join(lines) + "\n")


def run_parent(output_dir: Path, timeout_s: int):
    configs = build_configs()
    results = []

    script = Path(__file__).resolve()
    total = len(configs)

    # Warm-up runs so cache/JIT overhead is not attributed to first measured case.
    warmups = [configs[0], next(c for c in configs if c["problem"] == "infer_phi0")]
    for warm in warmups:
        payload = base64.b64encode(json.dumps(warm).encode("utf-8")).decode("ascii")
        subprocess.run(
            [sys.executable, str(script), "--worker", payload],
            cwd=str(script.parent),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )

    for i, cfg in enumerate(configs, start=1):
        case_label = (
            f"d_true={cfg['d_true']} d_guess={cfg['d_guess']}"
            if cfg["problem"] == "infer_D"
            else f"phi_true={cfg['phi_true']} phi_guess={cfg['phi_guess']}"
        )
        print(
            f"[{i:03d}/{total}] {cfg['problem']} method={cfg['method']} noise={cfg['noise_std']} seed={cfg['seed']} {case_label}",
            flush=True,
        )
        payload = base64.b64encode(json.dumps(cfg).encode("utf-8")).decode("ascii")
        try:
            proc = subprocess.run(
                [sys.executable, str(script), "--worker", payload],
                cwd=str(script.parent),
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            marker = "RESULT_JSON:"
            result_line = None
            for line in proc.stdout.splitlines()[::-1]:
                if line.startswith(marker):
                    result_line = line[len(marker):]
                    break
            if result_line is None:
                results.append(
                    {
                        "problem": cfg["problem"],
                        "method": cfg["method"],
                        "noise_std": cfg["noise_std"],
                        "seed": cfg["seed"],
                        "d_true": cfg.get("d_true"),
                        "d_guess": cfg.get("d_guess"),
                        "phi_true": cfg.get("phi_true"),
                        "phi_guess": cfg.get("phi_guess"),
                        "success": False,
                        "failure_reason": "No RESULT_JSON marker from worker",
                        "time_s": float("nan"),
                        "peak_rss_mib": float("nan"),
                        "objective": None,
                        "estimate": None,
                        "error_abs": None,
                        "error_rel": None,
                        "log_tail": compact_log(tail_lines(proc.stdout + "\n" + proc.stderr)),
                    }
                )
            else:
                results.append(json.loads(result_line))
        except subprocess.TimeoutExpired as exc:
            results.append(
                {
                    "problem": cfg["problem"],
                    "method": cfg["method"],
                    "noise_std": cfg["noise_std"],
                    "seed": cfg["seed"],
                    "d_true": cfg.get("d_true"),
                    "d_guess": cfg.get("d_guess"),
                    "phi_true": cfg.get("phi_true"),
                    "phi_guess": cfg.get("phi_guess"),
                    "success": False,
                    "failure_reason": f"TimeoutExpired after {timeout_s}s",
                    "time_s": float(timeout_s),
                    "peak_rss_mib": float("nan"),
                    "objective": None,
                    "estimate": None,
                    "error_abs": None,
                    "error_rel": None,
                    "log_tail": compact_log(tail_lines((exc.stdout or "") + "\n" + (exc.stderr or ""))),
                }
            )

    summary = aggregate_summary(results)

    csv_path = output_dir / "opt_method_study_results.csv"
    summary_csv_path = output_dir / "opt_method_study_summary.csv"
    md_path = output_dir / "opt_method_study_summary.md"

    write_csv(csv_path, results)
    write_csv(summary_csv_path, summary)
    write_markdown(md_path, summary, results)

    print("\nWrote:")
    print(f"- {csv_path}")
    print(f"- {summary_csv_path}")
    print(f"- {md_path}")


def main():
    parser = argparse.ArgumentParser(description="PNP inverse optimization method study")
    parser.add_argument("--worker", type=str, help="base64 encoded single-run config")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("StudyResults") / "optimization_methods",
        help="Directory for study outputs",
    )
    parser.add_argument("--timeout-s", type=int, default=300, help="Per-run timeout in seconds")
    args = parser.parse_args()

    if args.worker:
        cfg = json.loads(base64.b64decode(args.worker.encode("ascii")).decode("utf-8"))
        result = run_single(cfg)
        print("RESULT_JSON:" + json.dumps(result))
        return

    run_parent(args.output_dir, args.timeout_s)


if __name__ == "__main__":
    main()
