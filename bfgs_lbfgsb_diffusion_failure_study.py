#!/usr/bin/env python3
import argparse
import base64
import contextlib
import csv
import io
import json
import math
import os
import re
import resource
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

# Keep Firedrake caches writable in sandboxed environments.
os.environ.setdefault("FIREDRAKE_TSFC_KERNEL_CACHE_DIR", "/tmp/firedrake-tsfc")
os.environ.setdefault("PYOP2_CACHE_DIR", "/tmp/pyop2")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("OMP_NUM_THREADS", "1")

METHODS = ["BFGS", "L-BFGS-B"]
NOISE_STDS = [0.0, 0.005, 0.02]

# Intentionally different from prior studies.
NOISELESS_SEED = 20270400
NOISY_SEEDS = [20270401 + i for i in range(10)]

D_TRUE_VALUES = [
    [1.0, 3.0],
    [0.5, 2.0],
]
D_GUESSES = [
    [0.3, 0.3],
    [10.0, 10.0],
]

D_PAIR_RE = re.compile(r"D\s*=\s*\[([^\]]+)\],\s*\[([^\]]+)\]")


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


def to_jsonable(value):
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value)
    return value


def compact_log(text: str) -> str:
    return text.replace("\r", "").replace("\n", "\\n")


def tail_lines(text: str, n: int = 20) -> str:
    lines = text.strip().splitlines()
    if not lines:
        return ""
    return "\n".join(lines[-n:])


def parse_d_pairs(text: str):
    pairs = []
    for m in D_PAIR_RE.finditer(text or ""):
        try:
            d1 = float(m.group(1))
            d2 = float(m.group(2))
            pairs.append([d1, d2])
        except Exception:
            continue
    return pairs


def min_component(pair):
    if not pair or len(pair) != 2:
        return float("nan")
    return float(min(pair[0], pair[1]))


def as_float_list(vals):
    return [float(v) for v in vals]


def seeds_for_noise(noise_std: float):
    if abs(float(noise_std)) < 1e-15:
        return [NOISELESS_SEED]
    return list(NOISY_SEEDS)


def run_single(config):
    import firedrake.adjoint as adj

    from Helpers.Infer_D_from_data_helpers import make_objective_and_grad as make_d_objective
    from Utils.generate_noisy_data import generate_noisy_data

    method = config["method"]
    noise_std = float(config["noise_std"])
    seed = int(config["seed"])
    d_true = as_float_list(config["d_true"])
    d_guess = as_float_list(config["d_guess"])

    params = build_common_solver_options()
    n_species = 2
    z_vals = [1, -1]
    a_vals = [0.0, 0.0]
    c0_vals = [0.1, 0.1]
    phi0_fixed = 1.0
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
                data = generate_noisy_data(solver_params_gen, noise_std=noise_std, seed=seed)

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
                "options": {"disp": False, "maxiter": 80},
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
    except Exception as exc:
        failure_reason = f"{type(exc).__name__}: {exc}"

    elapsed = time.perf_counter() - start
    peak_mib = rss_to_mib(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    log_text = run_log.getvalue()
    d_pairs = parse_d_pairs(log_text)
    min_pair = min(d_pairs, key=min_component) if d_pairs else None
    last_pair = d_pairs[-1] if d_pairs else None

    return {
        "problem": "infer_D",
        "method": method,
        "noise_std": noise_std,
        "seed": seed,
        "d_true": d_true,
        "d_guess": d_guess,
        "success": success,
        "failure_reason": failure_reason,
        "time_s": float(elapsed),
        "peak_rss_mib": float(peak_mib),
        "objective": objective_value,
        "estimate": est_value,
        "error_abs": err_abs,
        "error_rel": err_rel,
        "n_logged_d_points": len(d_pairs),
        "min_logged_d_pair": min_pair,
        "last_logged_d_pair": last_pair,
        "min_logged_d_component": min_component(min_pair) if min_pair else float("nan"),
        "log_tail": compact_log(tail_lines(log_text)),
    }


def build_configs():
    cfgs = []
    for noise_std in NOISE_STDS:
        for seed in seeds_for_noise(noise_std):
            for method in METHODS:
                for d_true in D_TRUE_VALUES:
                    for d_guess in D_GUESSES:
                        cfgs.append(
                            {
                                "method": method,
                                "noise_std": noise_std,
                                "seed": int(seed),
                                "d_true": d_true,
                                "d_guess": d_guess,
                            }
                        )
    return cfgs


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
        key = (r["method"], r["noise_std"])
        groups.setdefault(key, []).append(r)

    summary = []
    for key in sorted(groups):
        rows = groups[key]
        succ = [r for r in rows if is_converged_run(r)]
        summary.append(
            {
                "problem": "infer_D",
                "method": key[0],
                "noise_std": key[1],
                "n_runs": len(rows),
                "n_success": len(succ),
                "success_rate": len(succ) / len(rows) if rows else float("nan"),
                "median_time_s": median_or_nan([r["time_s"] for r in succ]),
                "median_peak_rss_mib": median_or_nan([r["peak_rss_mib"] for r in succ]),
                "median_error_rel": median_or_nan([r["error_rel"] for r in succ]),
                "median_objective": median_or_nan([r["objective"] for r in succ]),
            }
        )
    return summary


def short_reason(text):
    txt = (text or "").strip()
    if not txt:
        return "Unknown"
    if "DIVERGED_LINE_SEARCH" in txt:
        return "DIVERGED_LINE_SEARCH"
    if "TimeoutExpired" in txt:
        return "TimeoutExpired"
    return txt.splitlines()[0]


def build_failure_cases(results):
    rows = []
    fails = [r for r in results if not r["success"]]
    for r in sorted(fails, key=lambda x: (x["method"], x["noise_std"], x["seed"], str(x["d_true"]), str(x["d_guess"]))):
        rows.append(
            {
                "method": r["method"],
                "noise_std": r["noise_std"],
                "seed": r["seed"],
                "d_true": r["d_true"],
                "d_guess": r["d_guess"],
                "n_logged_d_points": r["n_logged_d_points"],
                "min_logged_d_pair": r["min_logged_d_pair"],
                "last_logged_d_pair": r["last_logged_d_pair"],
                "min_logged_d_component": r["min_logged_d_component"],
                "failure_reason_short": short_reason(r["failure_reason"]),
            }
        )
    return rows


def method_seed_stability(results):
    out = []
    noisy = [r for r in results if r["noise_std"] > 0.0]
    for method in METHODS:
        rows = [r for r in noisy if r["method"] == method]
        fails = [r for r in rows if not r["success"]]
        seeds_tested = sorted({int(r["seed"]) for r in rows})
        seeds_failed = sorted({int(r["seed"]) for r in fails})
        out.append(
            {
                "method": method,
                "n_runs": len(rows),
                "n_fail": len(fails),
                "failure_rate": (len(fails) / len(rows)) if rows else float("nan"),
                "seeds_tested": seeds_tested,
                "seeds_failed": seeds_failed,
            }
        )
    return out


def safe_fmt(x):
    if x is None:
        return "nan"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "nan"
    if isinstance(x, float):
        return f"{x:.4g}"
    return str(x)


def commonality_analysis(results):
    fail_rows = [r for r in results if not r["success"]]
    succ_rows = [r for r in results if r["success"]]

    out = {
        "n_fail": len(fail_rows),
        "fail_by_method": Counter(r["method"] for r in fail_rows),
        "fail_by_noise": Counter(r["noise_std"] for r in fail_rows),
        "fail_by_guess": Counter(tuple(r["d_guess"]) for r in fail_rows),
        "fail_by_true": Counter(tuple(r["d_true"]) for r in fail_rows),
    }

    fail_min_comp = [r["min_logged_d_component"] for r in fail_rows if np.isfinite(r["min_logged_d_component"])]
    succ_min_comp = [r["min_logged_d_component"] for r in succ_rows if np.isfinite(r["min_logged_d_component"])]

    out["fail_min_comp"] = {
        "n": len(fail_min_comp),
        "min": min(fail_min_comp) if fail_min_comp else float("nan"),
        "median": median_or_nan(fail_min_comp),
        "max": max(fail_min_comp) if fail_min_comp else float("nan"),
    }
    out["succ_min_comp"] = {
        "n": len(succ_min_comp),
        "min": min(succ_min_comp) if succ_min_comp else float("nan"),
        "median": median_or_nan(succ_min_comp),
        "max": max(succ_min_comp) if succ_min_comp else float("nan"),
    }

    thresholds = [0.05, 0.1, 0.2, 0.5]
    by_method = {}
    for method in METHODS:
        m_fail = [r for r in fail_rows if r["method"] == method and np.isfinite(r["min_logged_d_component"])]
        m_succ = [r for r in succ_rows if r["method"] == method and np.isfinite(r["min_logged_d_component"])]
        by_method[method] = {
            "n_fail": len(m_fail),
            "n_succ": len(m_succ),
            "median_fail_min_component": median_or_nan([r["min_logged_d_component"] for r in m_fail]),
            "median_succ_min_component": median_or_nan([r["min_logged_d_component"] for r in m_succ]),
            "frac_fail_below": {
                t: (sum(r["min_logged_d_component"] < t for r in m_fail) / len(m_fail) if m_fail else float("nan"))
                for t in thresholds
            },
            "frac_succ_below": {
                t: (sum(r["min_logged_d_component"] < t for r in m_succ) / len(m_succ) if m_succ else float("nan"))
                for t in thresholds
            },
        }
    out["by_method"] = by_method

    return out


def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: to_jsonable(v) for k, v in row.items()})


def write_markdown(path: Path, summary_rows, results, failure_rows, seed_stability, analysis):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# BFGS vs L-BFGS-B Diffusion Failure Study")
    lines.append("")
    lines.append("## Study Design")
    lines.append("- Problem: `infer_D` only")
    lines.append("- Methods: `BFGS`, `L-BFGS-B`")
    lines.append(f"- Noise levels: `{NOISE_STDS}`")
    lines.append(f"- Seed for no-noise (`noise_std=0`): `{NOISELESS_SEED}`")
    lines.append(f"- Seeds for noisy runs (`noise_std>0`): `{NOISY_SEEDS}`")
    lines.append("- Cases per run: `d_true in {[1,3],[0.5,2]}`, `d_guess in {[0.3,0.3],[10,10]}`")
    lines.append("")
    lines.append(f"Total runs: **{len(results)}**")
    lines.append(f"Failed runs: **{sum(1 for r in results if not r['success'])}**")
    lines.append("Median time/memory/error statistics use converged runs only.")
    lines.append("")

    lines.append("## Summary by Method and Noise")
    lines.append("")
    lines.append("| method | noise_std | success_rate | n_success/n_runs | median_time_s | median_peak_rss_mib | median_rel_error | median_objective |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in sorted(summary_rows, key=lambda x: (x["method"], x["noise_std"])):
        lines.append(
            f"| {r['method']} | {safe_fmt(r['noise_std'])} | {safe_fmt(r['success_rate'])} | {r['n_success']}/{r['n_runs']} | "
            f"{safe_fmt(r['median_time_s'])} | {safe_fmt(r['median_peak_rss_mib'])} | {safe_fmt(r['median_error_rel'])} | {safe_fmt(r['median_objective'])} |"
        )
    lines.append("")

    lines.append("## Across-Seed Stability for Noisy Cases (`noise_std > 0`)")
    lines.append("")
    lines.append("| method | failures/runs | failure_rate | seeds_with_failures |")
    lines.append("|---|---:|---:|---|")
    for r in seed_stability:
        seeds_failed = "none" if not r["seeds_failed"] else str(r["seeds_failed"])
        lines.append(
            f"| {r['method']} | {r['n_fail']}/{r['n_runs']} | {safe_fmt(r['failure_rate'])} | {seeds_failed} |"
        )
    lines.append("")

    lines.append("## Failure Cases and Logged D Values")
    lines.append("")
    if not failure_rows:
        lines.append("No failures were observed.")
    else:
        lines.append("| method | noise_std | seed | d_true | d_guess | min_logged_d_pair | last_logged_d_pair | min_logged_d_component | reason |")
        lines.append("|---|---:|---:|---|---|---|---|---:|---|")
        for r in failure_rows:
            lines.append(
                f"| {r['method']} | {safe_fmt(r['noise_std'])} | {r['seed']} | {r['d_true']} | {r['d_guess']} | "
                f"{r['min_logged_d_pair']} | {r['last_logged_d_pair']} | {safe_fmt(r['min_logged_d_component'])} | {r['failure_reason_short']} |"
            )
    lines.append("")

    lines.append("## What Failed D Values Have in Common")
    lines.append("")
    lines.append(f"- Failures by method: `{dict(analysis['fail_by_method'])}`")
    lines.append(f"- Failures by noise: `{dict(analysis['fail_by_noise'])}`")
    lines.append(f"- Failures by initial guess: `{dict(analysis['fail_by_guess'])}`")
    lines.append(f"- Failures by true D: `{dict(analysis['fail_by_true'])}`")
    lines.append(
        f"- Logged minimum D component in failed runs: min={safe_fmt(analysis['fail_min_comp']['min'])}, "
        f"median={safe_fmt(analysis['fail_min_comp']['median'])}, max={safe_fmt(analysis['fail_min_comp']['max'])}"
    )
    lines.append(
        f"- Logged minimum D component in successful runs: min={safe_fmt(analysis['succ_min_comp']['min'])}, "
        f"median={safe_fmt(analysis['succ_min_comp']['median'])}, max={safe_fmt(analysis['succ_min_comp']['max'])}"
    )
    lines.append("")

    lines.append("### Method-Level Comparison")
    lines.append("")
    lines.append("| method | median min-D (fail) | median min-D (success) | fail frac(min-D<0.1) | success frac(min-D<0.1) |")
    lines.append("|---|---:|---:|---:|---:|")
    for method in METHODS:
        m = analysis["by_method"][method]
        lines.append(
            f"| {method} | {safe_fmt(m['median_fail_min_component'])} | {safe_fmt(m['median_succ_min_component'])} | "
            f"{safe_fmt(m['frac_fail_below'][0.1])} | {safe_fmt(m['frac_succ_below'][0.1])} |"
        )
    lines.append("")

    lines.append("### Interpretation")
    lines.append("")
    lines.append(
        "- Forward failures are consistent with trial iterates driving one or both diffusion coefficients to very small values, which can increase stiffness and hurt nonlinear line-search robustness."
    )
    lines.append(
        "- If failures cluster under high initial guesses (e.g. `[10, 10]`), that suggests aggressive early steps in log-D space are part of the mechanism."
    )
    lines.append(
        "- The comparison against successful runs shows whether failed trajectories visit a low-D region that successful trajectories usually avoid."
    )
    lines.append("")

    path.write_text("\n".join(lines) + "\n")


def run_parent(output_dir: Path, timeout_s: int):
    configs = build_configs()
    results = []
    script = Path(__file__).resolve()
    total = len(configs)

    # Warmup one case per method.
    warmups = [configs[0], next(c for c in configs if c["method"] == "L-BFGS-B")]
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
        print(
            f"[{i:03d}/{total}] method={cfg['method']} noise={cfg['noise_std']} seed={cfg['seed']} "
            f"d_true={cfg['d_true']} d_guess={cfg['d_guess']}",
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
                        "problem": "infer_D",
                        "method": cfg["method"],
                        "noise_std": cfg["noise_std"],
                        "seed": cfg["seed"],
                        "d_true": cfg["d_true"],
                        "d_guess": cfg["d_guess"],
                        "success": False,
                        "failure_reason": "No RESULT_JSON marker from worker",
                        "time_s": float("nan"),
                        "peak_rss_mib": float("nan"),
                        "objective": None,
                        "estimate": None,
                        "error_abs": None,
                        "error_rel": None,
                        "n_logged_d_points": 0,
                        "min_logged_d_pair": None,
                        "last_logged_d_pair": None,
                        "min_logged_d_component": float("nan"),
                        "log_tail": compact_log(tail_lines(proc.stdout + "\n" + proc.stderr)),
                    }
                )
            else:
                results.append(json.loads(result_line))
        except subprocess.TimeoutExpired as exc:
            results.append(
                {
                    "problem": "infer_D",
                    "method": cfg["method"],
                    "noise_std": cfg["noise_std"],
                    "seed": cfg["seed"],
                    "d_true": cfg["d_true"],
                    "d_guess": cfg["d_guess"],
                    "success": False,
                    "failure_reason": f"TimeoutExpired after {timeout_s}s",
                    "time_s": float(timeout_s),
                    "peak_rss_mib": float("nan"),
                    "objective": None,
                    "estimate": None,
                    "error_abs": None,
                    "error_rel": None,
                    "n_logged_d_points": 0,
                    "min_logged_d_pair": None,
                    "last_logged_d_pair": None,
                    "min_logged_d_component": float("nan"),
                    "log_tail": compact_log(tail_lines((exc.stdout or "") + "\n" + (exc.stderr or ""))),
                }
            )

    summary_rows = aggregate_summary(results)
    failure_rows = build_failure_cases(results)
    seed_stability = method_seed_stability(results)
    analysis = commonality_analysis(results)

    output_dir.mkdir(parents=True, exist_ok=True)
    results_csv = output_dir / "bfgs_lbfgsb_diffusion_results.csv"
    summary_csv = output_dir / "bfgs_lbfgsb_diffusion_summary.csv"
    failures_csv = output_dir / "bfgs_lbfgsb_diffusion_failure_cases.csv"
    analysis_md = output_dir / "bfgs_lbfgsb_diffusion_failure_analysis.md"

    write_csv(results_csv, results)
    write_csv(summary_csv, summary_rows)
    write_csv(failures_csv, failure_rows)
    write_markdown(analysis_md, summary_rows, results, failure_rows, seed_stability, analysis)

    print("\nWrote:")
    print(f"- {results_csv}")
    print(f"- {summary_csv}")
    print(f"- {failures_csv}")
    print(f"- {analysis_md}")


def main():
    parser = argparse.ArgumentParser(description="BFGS vs L-BFGS-B diffusion failure study")
    parser.add_argument("--worker", type=str, help="base64 encoded single-run config")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("StudyResults") / "bfgs_lbfgsb_diffusion_failure_study",
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
