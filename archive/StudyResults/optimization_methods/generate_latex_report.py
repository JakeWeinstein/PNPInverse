#!/usr/bin/env python3
import ast
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import numpy as np

METHODS = ["BFGS", "L-BFGS-B", "CG", "SLSQP", "TNC", "Newton-CG"]
NOISES = [0.0, 0.005, 0.02]
PROBLEMS = ["infer_D", "infer_phi0"]


def to_float(value):
    try:
        return float(value)
    except Exception:
        return float("nan")


def median(values):
    vals = [v for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return float("nan")
    return float(np.median(vals))


def is_converged_run(row):
    return bool(row.get("success")) and not (row.get("failure_reason") or "").strip()


def load_summary(path):
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["noise_std"] = float(row["noise_std"])
            row["n_runs"] = int(row["n_runs"])
            row["n_success"] = int(row["n_success"])
            row["success_rate"] = float(row["success_rate"])
            row["median_time_s"] = to_float(row["median_time_s"])
            row["median_peak_rss_mib"] = to_float(row["median_peak_rss_mib"])
            row["median_error_rel"] = to_float(row["median_error_rel"])
            row["median_objective"] = to_float(row["median_objective"])
            rows.append(row)
    return rows


def load_results(path):
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["noise_std"] = float(row["noise_std"])
            row["seed"] = int(row["seed"])
            row["success"] = row["success"] == "True"
            row["time_s"] = to_float(row["time_s"])
            row["peak_rss_mib"] = to_float(row["peak_rss_mib"])
            row["error_rel"] = to_float(row["error_rel"])
            rows.append(row)
    return rows


def build_summary_map(summary_rows):
    data = {}
    for row in summary_rows:
        key = (row["problem"], row["method"], row["noise_std"])
        data[key] = row
    return data


def annotation_style(im, value):
    if not np.isfinite(value):
        return "black", [patheffects.withStroke(linewidth=1.5, foreground="white")]
    r, g, b, _ = im.cmap(im.norm(value))
    # Relative luminance for contrast-aware text color selection.
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    if luminance > 0.55:
        return "black", [patheffects.withStroke(linewidth=1.5, foreground="white")]
    return "white", [patheffects.withStroke(linewidth=1.5, foreground="black")]


def draw_heatmap(ax, mat, title, vmin=None, vmax=None, cmap="viridis", fmt="{:.3g}"):
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(range(len(NOISES)))
    ax.set_xticklabels([str(n) for n in NOISES])
    ax.set_xlabel("noise_std")
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels(METHODS)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            color, effects = annotation_style(im, val)
            if np.isfinite(val):
                text = ax.text(j, i, fmt.format(val), ha="center", va="center", color=color, fontsize=8)
                text.set_path_effects(effects)
            else:
                text = ax.text(j, i, "nan", ha="center", va="center", color=color, fontsize=8)
                text.set_path_effects(effects)
    return im


def make_heatmaps(summary_map, outdir):
    outdir.mkdir(parents=True, exist_ok=True)

    metrics = [
        (
            "success_rate",
            "Success Rate",
            "higher is better",
            (0.0, 1.0),
            "viridis",
            "{:.2f}",
            "success_rate_heatmap.png",
        ),
        (
            "median_time_s",
            "Median Time (s)",
            "lower is better",
            (None, None),
            "magma",
            "{:.2f}",
            "median_time_heatmap.png",
        ),
        (
            "median_peak_rss_mib",
            "Median Peak RSS (MiB)",
            "lower is better",
            (None, None),
            "cividis",
            "{:.1f}",
            "median_memory_heatmap.png",
        ),
    ]

    # Linear-scale heatmaps
    for key, title, direction, bounds, cmap, fmt, fname in metrics:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        for ax, problem in zip(axes, PROBLEMS):
            mat = np.full((len(METHODS), len(NOISES)), np.nan)
            for i, method in enumerate(METHODS):
                for j, noise in enumerate(NOISES):
                    row = summary_map.get((problem, method, noise))
                    if row is not None:
                        mat[i, j] = row[key]
            vmin, vmax = bounds
            im = draw_heatmap(
                ax,
                mat,
                f"{problem}: {title} ({direction})",
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                fmt=fmt,
            )
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(direction)
        fig.savefig(outdir / fname, dpi=180)
        plt.close(fig)

    # Log-scale heatmap for relative error
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, problem in zip(axes, PROBLEMS):
        mat = np.full((len(METHODS), len(NOISES)), np.nan)
        for i, method in enumerate(METHODS):
            for j, noise in enumerate(NOISES):
                row = summary_map.get((problem, method, noise))
                if row is not None:
                    mat[i, j] = row["median_error_rel"]
        mat_log = np.where(mat > 0, np.log10(mat), np.nan)
        im = draw_heatmap(
            ax,
            mat_log,
            f"{problem}: log10(Median Relative Error) (lower is better)",
            cmap="plasma",
            fmt="{:.2f}",
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("lower is better")
    fig.savefig(outdir / "median_rel_error_log10_heatmap.png", dpi=180)
    plt.close(fig)


def make_failure_plot(results_rows, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    fails = [r for r in results_rows if not is_converged_run(r)]
    counts = Counter((r["problem"], r["method"]) for r in fails)
    by_problem = {p: [counts[(p, m)] for m in METHODS] for p in PROBLEMS}

    x = np.arange(len(METHODS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.bar(x - width / 2, by_problem["infer_D"], width, label="infer_D")
    ax.bar(x + width / 2, by_problem["infer_phi0"], width, label="infer_phi0")
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS, rotation=20, ha="right")
    ax.set_ylabel("Failure Count (lower is better)")
    ax.set_title("Failure Counts by Method and Problem (lower is better)")
    ax.legend()
    fig.savefig(outdir / "failure_counts.png", dpi=180)
    plt.close(fig)


def infer_d_case_failure_stats(results_rows):
    infer_d = [r for r in results_rows if r["problem"] == "infer_D"]
    symmetric = [r for r in infer_d if is_symmetric_d_true(r.get("d_true"))]
    nonsymmetric = [r for r in infer_d if not is_symmetric_d_true(r.get("d_true"))]

    def summarize(rows):
        fails = [r for r in rows if not is_converged_run(r)]
        return {
            "n_runs": len(rows),
            "n_fail": len(fails),
            "fail_rate": (len(fails) / len(rows)) if rows else float("nan"),
            "reasons": Counter(reason_label(r.get("failure_reason", "")) for r in fails),
        }

    by_method = []
    for method in METHODS:
        sel = [r for r in symmetric if r["method"] == method]
        fail = [r for r in sel if not is_converged_run(r)]
        by_method.append(
            {
                "method": method,
                "n_runs": len(sel),
                "n_fail": len(fail),
                "fail_rate": (len(fail) / len(sel)) if sel else float("nan"),
            }
        )

    return {
        "symmetric": summarize(symmetric),
        "nonsymmetric": summarize(nonsymmetric),
        "by_method": by_method,
        "rows_symmetric": symmetric,
    }


def make_symmetric_case_plots(results_rows, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    infer_d_sym = [
        r for r in results_rows if r["problem"] == "infer_D" and is_symmetric_d_true(r.get("d_true"))
    ]
    infer_d_non = [
        r for r in results_rows if r["problem"] == "infer_D" and not is_symmetric_d_true(r.get("d_true"))
    ]

    # Heatmap: symmetric case success rate by method/noise.
    mat = np.full((len(METHODS), len(NOISES)), np.nan)
    for i, method in enumerate(METHODS):
        for j, noise in enumerate(NOISES):
            sel = [r for r in infer_d_sym if r["method"] == method and float(r["noise_std"]) == float(noise)]
            if not sel:
                continue
            n_success = sum(1 for r in sel if is_converged_run(r))
            mat[i, j] = n_success / len(sel)

    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    im = draw_heatmap(
        ax,
        mat,
        r"infer_D (D_true=[1,1]): Success Rate (higher is better)",
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
        fmt="{:.2f}",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("higher is better")
    fig.savefig(outdir / "infer_d_symmetric_success_rate_heatmap.png", dpi=180)
    plt.close(fig)

    # Bar chart: failure rate by method, symmetric vs non-symmetric cases.
    sym_rate = []
    non_rate = []
    for method in METHODS:
        sym_rows = [r for r in infer_d_sym if r["method"] == method]
        non_rows = [r for r in infer_d_non if r["method"] == method]
        sym_fail = sum(1 for r in sym_rows if not is_converged_run(r))
        non_fail = sum(1 for r in non_rows if not is_converged_run(r))
        sym_rate.append((100.0 * sym_fail / len(sym_rows)) if sym_rows else float("nan"))
        non_rate.append((100.0 * non_fail / len(non_rows)) if non_rows else float("nan"))

    x = np.arange(len(METHODS))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.2), constrained_layout=True)
    ax.bar(x - width / 2, sym_rate, width, label=r"D_true=[1,1]")
    ax.bar(x + width / 2, non_rate, width, label=r"D_true$\neq$[1,1]")
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS, rotation=20, ha="right")
    ax.set_ylabel("Failure Rate (%) (lower is better)")
    ax.set_title(r"infer_D Failure Rate by Method: Symmetric vs Non-Symmetric D_true")
    ax.legend()
    fig.savefig(outdir / "infer_d_failure_rate_symmetric_vs_nonsymmetric.png", dpi=180)
    plt.close(fig)


def aggregate_across_noise(results_rows):
    groups = defaultdict(list)
    for row in results_rows:
        groups[(row["problem"], row["method"])].append(row)

    agg = []
    for (problem, method), rows in sorted(groups.items()):
        succ = [r for r in rows if is_converged_run(r)]
        agg.append(
            {
                "problem": problem,
                "method": method,
                "n_success": len(succ),
                "n_runs": len(rows),
                "success_rate": len(succ) / len(rows) if rows else float("nan"),
                "median_time_s": median([r["time_s"] for r in succ]),
                "median_peak_rss_mib": median([r["peak_rss_mib"] for r in succ]),
                "median_error_rel": median([r["error_rel"] for r in succ]),
            }
        )
    return agg


def phi_estimate_stats(results_rows):
    vals = []
    for r in results_rows:
        if r["problem"] != "infer_phi0" or not r["success"]:
            continue
        est_str = (r.get("estimate") or "").strip()
        if not est_str:
            continue
        if est_str.startswith("["):
            est = ast.literal_eval(est_str)
            if isinstance(est, list) and est:
                vals.append(float(est[0]))
        else:
            vals.append(float(est_str))
    if not vals:
        return {"count": 0, "min": float("nan"), "max": float("nan"), "n_negative": 0}
    return {"count": len(vals), "min": min(vals), "max": max(vals), "n_negative": sum(v < 0 for v in vals)}


def failure_analysis(results_rows):
    fails = [r for r in results_rows if not is_converged_run(r)]
    reasons = Counter(reason_label(r.get("failure_reason", "")) for r in fails)
    return fails, reasons


def reason_label(reason):
    text = (reason or "").strip()
    if not text:
        return "Unknown"
    if "DIVERGED_LINE_SEARCH" in text:
        return "DIVERGED_LINE_SEARCH"
    if "SciPyConvergenceError" in text:
        return "SciPyConvergenceError"
    if "TimeoutExpired" in text:
        return "TimeoutExpired"
    return text.splitlines()[0]


def seed_policy_summary(results_rows):
    return {
        "noise0": sorted({int(r["seed"]) for r in results_rows if abs(float(r["noise_std"])) < 1e-15}),
        "noisy": sorted({int(r["seed"]) for r in results_rows if float(r["noise_std"]) > 0.0}),
    }


def method_seed_failure_summary(results_rows):
    rows = [r for r in results_rows if float(r["noise_std"]) > 0.0]
    out = []
    for problem in PROBLEMS:
        for method in METHODS:
            sel = [r for r in rows if r["problem"] == problem and r["method"] == method]
            if not sel:
                continue
            seeds = sorted({int(r["seed"]) for r in sel})
            fail_rows = [r for r in sel if not is_converged_run(r)]
            fail_seeds = sorted({int(r["seed"]) for r in fail_rows})
            out.append(
                {
                    "problem": problem,
                    "method": method,
                    "n_runs": len(sel),
                    "n_fail": len(fail_rows),
                    "seeds_tested": seeds,
                    "seeds_failed": fail_seeds,
                    "n_seeds_tested": len(seeds),
                    "n_seeds_failed": len(fail_seeds),
                }
            )
    return out


def latex_escape(text):
    return (
        str(text)
        .replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("~", "\\textasciitilde{}")
        .replace("^", "\\textasciicircum{}")
    )


def fmt(x, digits=3):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "nan"
    if isinstance(x, float):
        return f"{x:.{digits}g}"
    return str(x)


def format_vec_text(value, digits=4):
    if value is None:
        return "n/a"
    txt = str(value).strip()
    if not txt:
        return "n/a"
    try:
        parsed = ast.literal_eval(txt)
    except Exception:
        return txt
    if isinstance(parsed, (list, tuple)):
        try:
            return "[" + ", ".join(f"{float(v):.{digits}g}" for v in parsed) + "]"
        except Exception:
            return txt
    try:
        return f"{float(parsed):.{digits}g}"
    except Exception:
        return txt


def parse_vec2(value):
    txt = str(value).strip()
    if not txt:
        return None
    try:
        parsed = ast.literal_eval(txt)
    except Exception:
        return None
    if not isinstance(parsed, (list, tuple)) or len(parsed) != 2:
        return None
    try:
        return (float(parsed[0]), float(parsed[1]))
    except Exception:
        return None


def is_symmetric_d_true(value, target=(1.0, 1.0), tol=1e-12):
    vec = parse_vec2(value)
    if vec is None:
        return False
    return abs(vec[0] - target[0]) <= tol and abs(vec[1] - target[1]) <= tol


def parse_logged_d_pairs(log_tail):
    if not log_tail:
        return []
    pairs = []
    pattern = re.compile(r"D\s*=\s*\[([^\]]+)\],\s*\[([^\]]+)\]")
    for match in pattern.finditer(log_tail):
        try:
            d1 = float(match.group(1))
            d2 = float(match.group(2))
            pairs.append((d1, d2))
        except Exception:
            continue
    return pairs


def lbfgsb_failure_diagnostics(results_rows):
    bfgs_map = {}
    for row in results_rows:
        if row["problem"] != "infer_D" or row["method"] != "BFGS":
            continue
        key = (row["noise_std"], row.get("seed"), row.get("d_true"), row.get("d_guess"))
        bfgs_map[key] = row

    cases = []
    for row in results_rows:
        if row["problem"] != "infer_D" or row["method"] != "L-BFGS-B" or is_converged_run(row):
            continue
        key = (row["noise_std"], row.get("seed"), row.get("d_true"), row.get("d_guess"))
        matched_bfgs = bfgs_map.get(key)
        d_pairs = parse_logged_d_pairs(row.get("log_tail", ""))
        min_pair = min(d_pairs, key=lambda x: min(x[0], x[1])) if d_pairs else None
        last_pair = d_pairs[-1] if d_pairs else None
        reason_first = (row.get("failure_reason", "") or "").splitlines()[0]
        iters_match = re.search(r"after\s+(\d+)\s+nonlinear iterations", row.get("failure_reason", ""))
        nonlinear_iters = int(iters_match.group(1)) if iters_match else None

        cases.append(
            {
                "noise_std": row["noise_std"],
                "seed": int(row["seed"]),
                "d_true": format_vec_text(row.get("d_true")),
                "d_guess": format_vec_text(row.get("d_guess")),
                "reason_first": reason_first,
                "nonlinear_iters": nonlinear_iters,
                "min_pair": min_pair,
                "last_pair": last_pair,
                "bfgs_success": bool(matched_bfgs and is_converged_run(matched_bfgs)),
                "bfgs_estimate": format_vec_text(matched_bfgs.get("estimate") if matched_bfgs else None),
            }
        )

    infer_d_rows = [r for r in results_rows if r["problem"] == "infer_D"]
    return {
        "lbfgsb_fail_count": sum(
            1 for r in infer_d_rows if r["method"] == "L-BFGS-B" and not is_converged_run(r)
        ),
        "bfgs_fail_count": sum(1 for r in infer_d_rows if r["method"] == "BFGS" and not is_converged_run(r)),
        "cases": sorted(cases, key=lambda c: (c["noise_std"], c["seed"])),
    }


def build_latex_report(
    base_dir,
    agg_rows,
    fails,
    reasons,
    phi_stats,
    lbfgsb_diag,
    seed_info,
    method_seed_stats,
    infer_d_case_stats,
    total_runs,
):
    # Split aggregate table by problem for readability
    agg_d = [r for r in agg_rows if r["problem"] == "infer_D"]
    agg_phi = [r for r in agg_rows if r["problem"] == "infer_phi0"]

    dominant_reason, dominant_count = ("None", 0)
    if reasons:
        dominant_reason, dominant_count = reasons.most_common(1)[0]
    fail_counts_by_problem = Counter(r["problem"] for r in fails)

    lines = []
    lines.append(r"\documentclass[11pt]{article}")
    lines.append(r"\usepackage[margin=1in]{geometry}")
    lines.append(r"\usepackage{booktabs}")
    lines.append(r"\usepackage{graphicx}")
    lines.append(r"\usepackage{float}")
    lines.append(r"\usepackage{siunitx}")
    lines.append(r"\usepackage{longtable}")
    lines.append(r"\usepackage{array}")
    lines.append(r"\usepackage{setspace}")
    lines.append(r"\setlength{\parindent}{0pt}")
    lines.append(r"\sisetup{round-mode=places,round-precision=4}")
    lines.append(r"\title{PNP Inverse Optimization Method Study}")
    lines.append(r"\author{Automated Benchmark Report}")
    lines.append(r"\date{\today}")
    lines.append(r"\begin{document}")
    lines.append(r"\onehalfspacing")
    lines.append(r"\maketitle")
    lines.append("")
    lines.append(r"\section*{What We Ran}")
    lines.append(r"\paragraph{Goal}\quad\\")
    lines.append(
        r"We're comparing optimization methods for two inverse problems: inferring diffusion coefficients ($D$) and inferring the Dirichlet BC potential ($\phi_0$)."
    )
    lines.append(r"")
    lines.append(r"\paragraph{Setup}\quad\\")
    lines.append(
        r"Methods: \texttt{BFGS}, \texttt{L-BFGS-B}, \texttt{CG}, \texttt{SLSQP}, \texttt{TNC}, \texttt{Newton-CG}."
    )
    lines.append(r"Noise levels: $\sigma \in \{0, 0.005, 0.02\}$.")
    lines.append(
        f"Seeds used for $\\sigma=0$: {latex_escape(seed_info['noise0'])}; seeds used for $\\sigma>0$: {latex_escape(seed_info['noisy'])}."
    )
    lines.append(f"Total runs: {total_runs}.")
    lines.append(r"\vspace{1em}")
    lines.append(r"\hrule")
    lines.append(r"\vspace{1em}")
    lines.append(r"\paragraph{How to read the plots}\quad\\")
    lines.append(
        r"Each plot title states the optimization direction explicitly (\textit{higher is better} or \textit{lower is better})."
    )
    lines.append("")

    lines.append(r"\section*{Method Comparisons}")
    lines.append(r"\begin{figure}[H]")
    lines.append(r"\centering")
    lines.append(r"\includegraphics[width=0.95\textwidth]{success_rate_heatmap.png}")
    lines.append(r"\caption{Success-rate heatmaps by problem, method, and noise level (higher is better).}")
    lines.append(r"\end{figure}")

    lines.append(r"\begin{figure}[H]")
    lines.append(r"\centering")
    lines.append(r"\includegraphics[width=0.95\textwidth]{median_time_heatmap.png}")
    lines.append(r"\caption{Median wall-clock time (seconds) for successful runs (lower is better).}")
    lines.append(r"\end{figure}")

    lines.append(r"\begin{figure}[H]")
    lines.append(r"\centering")
    lines.append(r"\includegraphics[width=0.95\textwidth]{median_memory_heatmap.png}")
    lines.append(r"\caption{Median peak RSS memory (MiB) for successful runs (lower is better).}")
    lines.append(r"\end{figure}")

    lines.append(r"\begin{figure}[H]")
    lines.append(r"\centering")
    lines.append(r"\includegraphics[width=0.95\textwidth]{median_rel_error_log10_heatmap.png}")
    lines.append(r"\caption{$\log_{10}$ median relative error by method and noise (lower is better).}")
    lines.append(r"\end{figure}")

    lines.append(r"\begin{figure}[H]")
    lines.append(r"\centering")
    lines.append(r"\includegraphics[width=0.8\textwidth]{failure_counts.png}")
    lines.append(r"\caption{Failure counts by method and problem (lower is better).}")
    lines.append(r"\end{figure}")

    lines.append(r"\begin{figure}[H]")
    lines.append(r"\centering")
    lines.append(r"\includegraphics[width=0.78\textwidth]{infer_d_symmetric_success_rate_heatmap.png}")
    lines.append(r"\caption{Symmetric diffusion truth case ($D_{\mathrm{true}}=[1,1]$): success rate by method and noise (higher is better).}")
    lines.append(r"\end{figure}")

    lines.append(r"\begin{figure}[H]")
    lines.append(r"\centering")
    lines.append(r"\includegraphics[width=0.9\textwidth]{infer_d_failure_rate_symmetric_vs_nonsymmetric.png}")
    lines.append(r"\caption{infer\_D failure-rate comparison by method: symmetric case vs all non-symmetric cases (lower is better).}")
    lines.append(r"\end{figure}")
    lines.append(r"\vspace{1em}")
    lines.append(r"\hrule")
    lines.append(r"\vspace{1em}")

    lines.append(r"\section*{Aggregated Results Across Noise Levels}")
    lines.append(r"\subsection*{Infer D}")
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Method & Success (\%) & Median Time (s) & Median RSS (MiB) & Median RelErr \\")
    lines.append(r"\midrule")
    for r in agg_d:
        lines.append(
            f"{latex_escape(r['method'])} & {fmt(100.0 * r['success_rate'])} & {fmt(r['median_time_s'])} & {fmt(r['median_peak_rss_mib'])} & {fmt(r['median_error_rel'])} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append(r"\par\medskip")

    lines.append(r"\subsection*{Infer $\phi_0$}")
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Method & Success (\%) & Median Time (s) & Median RSS (MiB) & Median RelErr \\")
    lines.append(r"\midrule")
    for r in agg_phi:
        lines.append(
            f"{latex_escape(r['method'])} & {fmt(100.0 * r['success_rate'])} & {fmt(r['median_time_s'])} & {fmt(r['median_peak_rss_mib'])} & {fmt(r['median_error_rel'])} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append(r"\par\medskip")

    lines.append(r"\section*{Crash Analysis}")
    lines.append(r"\paragraph{What failed?}\quad\\")
    lines.append(f"Observed failures: {len(fails)} total.")
    lines.append(
        f"\\texttt{{infer\\_D}} failures: {fail_counts_by_problem.get('infer_D', 0)}; "
        f"\\texttt{{infer\\_phi0}} failures: {fail_counts_by_problem.get('infer_phi0', 0)}."
    )
    lines.append(r"")
    lines.append(
        r"The dominant failure reason was Firedrake nonlinear forward-solve divergence: \texttt{DIVERGED\_LINE\_SEARCH}."
    )
    lines.append(
        f"Most common recorded reason count: {dominant_count} ({latex_escape(dominant_reason)})."
    )
    lines.append(r"")
    lines.append(r"\paragraph{Was this caused by unconstrained $\phi_0 < 0$?}\quad\\")
    lines.append(r"No.")
    lines.append(
        r"Evidence: all failures occurred in the diffusion-inference problem, not the BC-inference problem; the BC runs completed successfully for all methods/noise levels."
    )
    lines.append(
        f"Across successful BC runs, estimated $\\phi_0$ ranged from {phi_stats['min']:.6g} to {phi_stats['max']:.6g} with {phi_stats['n_negative']} negative estimates."
    )
    lines.append(
        r"For diffusion inference, controls are parameterized as $\log D$ and exponentiated in the forward model, so $D$ remains strictly positive by construction."
    )
    lines.append(
        r"The crashes are therefore consistent with unstable trial iterates that make the PDE nonlinear solve hard for Newton line search, not sign violations of $\phi_0$."
    )
    lines.append("")
    lines.append(r"\subsection*{Symmetric Diffusion Case: $D_{\mathrm{true}}=[1,1]$}")
    sym = infer_d_case_stats["symmetric"]
    non = infer_d_case_stats["nonsymmetric"]
    sym_reason, sym_reason_count = ("None", 0)
    if sym["reasons"]:
        sym_reason, sym_reason_count = sym["reasons"].most_common(1)[0]
    lines.append(
        f"Symmetric-case failures: {sym['n_fail']}/{sym['n_runs']} "
        f"({fmt(100.0 * sym['fail_rate'])}\\%). "
        f"Non-symmetric failures: {non['n_fail']}/{non['n_runs']} "
        f"({fmt(100.0 * non['fail_rate'])}\\%)."
    )
    lines.append(
        f"Dominant symmetric-case failure reason: {latex_escape(sym_reason)} "
        f"({sym_reason_count} runs)."
    )
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lrrr}")
    lines.append(r"\toprule")
    lines.append(r"Method & Failures / Runs & Failure Rate (\%) & Comment \\")
    lines.append(r"\midrule")
    for row in infer_d_case_stats["by_method"]:
        comment = "no failures" if row["n_fail"] == 0 else "unstable in symmetric case"
        lines.append(
            f"{latex_escape(row['method'])} & {row['n_fail']}/{row['n_runs']} & {fmt(100.0 * row['fail_rate'])} & {latex_escape(comment)} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append(r"\par\medskip")
    lines.append(r"\subsection*{Why L-BFGS-B Failed While BFGS Did Not (\texttt{infer\_D})}")
    lines.append(
        f"In \\texttt{{infer\\_D}}, \\texttt{{L-BFGS-B}} had {lbfgsb_diag['lbfgsb_fail_count']} forward-solve failures, while \\texttt{{BFGS}} had {lbfgsb_diag['bfgs_fail_count']}."
    )
    failed_d_guesses = sorted({c["d_guess"] for c in lbfgsb_diag["cases"]})
    failed_d_trues = sorted({c["d_true"] for c in lbfgsb_diag["cases"]})
    failed_d_guesses_txt = ", ".join(failed_d_guesses) if failed_d_guesses else "none"
    failed_d_trues_txt = ", ".join(failed_d_trues) if failed_d_trues else "none"
    lines.append(
        f"Failed \\texttt{{L-BFGS-B}} cases used initial guesses {latex_escape(failed_d_guesses_txt)} and true-$D$ cases {latex_escape(failed_d_trues_txt)}."
    )
    lines.append(
        r"In failed runs, logged trial iterates show sharp drops in one or both diffusivities before SNES terminated with \texttt{DIVERGED\_LINE\_SEARCH}."
    )
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{rrlllll}")
    lines.append(r"\toprule")
    lines.append(
        r"$\sigma$ & Seed & Initial guess $D_0$ & SNES iters & Min logged trial $D$ & Last logged $D$ & Matched BFGS estimate \\"
    )
    lines.append(r"\midrule")
    for case in lbfgsb_diag["cases"]:
        min_pair = "n/a"
        if case["min_pair"] is not None:
            min_pair = f"[{case['min_pair'][0]:.4g}, {case['min_pair'][1]:.4g}]"
        last_pair = "n/a"
        if case["last_pair"] is not None:
            last_pair = f"[{case['last_pair'][0]:.4g}, {case['last_pair'][1]:.4g}]"
        iters_txt = str(case["nonlinear_iters"]) if case["nonlinear_iters"] is not None else "n/a"
        bfgs_txt = case["bfgs_estimate"] if case["bfgs_success"] else "failed"
        lines.append(
            f"{fmt(case['noise_std'])} & {int(case['seed'])} & {latex_escape(case['d_guess'])} & {latex_escape(iters_txt)} & {latex_escape(min_pair)} & {latex_escape(last_pair)} & {latex_escape(bfgs_txt)} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")
    lines.append(r"\par\medskip")
    lines.append(
        r"This points to line-search/forward-solve fragility from aggressive trial steps in log-diffusivity space, not a $\phi_0<0$ issue."
    )
    lines.append("")
    lines.append(r"\subsection*{Across-Seed Stability ($\sigma>0$)}")
    lines.append(
        r"To test whether failures are random or method-specific, the noisy cases were repeated across multiple seeds and aggregated by method."
    )
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lrrp{0.36\textwidth}}")
    lines.append(r"\toprule")
    lines.append(r"Method (\texttt{infer\_D}) & Failures / Runs & Failure Rate (\%) & Seeds with failures \\")
    lines.append(r"\midrule")
    infer_d_seed_stats = [r for r in method_seed_stats if r["problem"] == "infer_D"]
    for row in infer_d_seed_stats:
        rate = 100.0 * (row["n_fail"] / row["n_runs"]) if row["n_runs"] > 0 else float("nan")
        seeds_failed_txt = "none" if not row["seeds_failed"] else str(row["seeds_failed"])
        lines.append(
            f"{latex_escape(row['method'])} & {row['n_fail']}/{row['n_runs']} & {fmt(rate)} & {latex_escape(seeds_failed_txt)} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append(r"\par\medskip")
    if infer_d_seed_stats:
        methods_with_fail = [r["method"] for r in infer_d_seed_stats if r["n_fail"] > 0]
        methods_with_fail_txt = ", ".join(methods_with_fail) if methods_with_fail else "none"
        lines.append(
            f"Methods with any noisy-case failures in \\texttt{{infer\\_D}}: {latex_escape(methods_with_fail_txt)}."
        )
    lines.append(
        r"If the same method fails across multiple seeds while others remain stable, that is evidence of method-specific robustness differences rather than pure random noise effects."
    )
    lines.append(r"\vspace{1em}")
    lines.append(r"\hrule")
    lines.append(r"\vspace{1em}")

    lines.append(r"\section*{Practical Recommendations}")
    lines.append(r"\paragraph{Takeaways}\quad\\")
    lines.append(r"\begin{itemize}")
    lines.append(r"\item For \texttt{infer\_phi0}: \texttt{SLSQP} or \texttt{L-BFGS-B} gave the best speed with full robustness.")
    lines.append(r"\item For \texttt{infer\_D}: \texttt{BFGS} had best reliability (100\% success) and good accuracy; \texttt{L-BFGS-B} was faster but had occasional forward-solve failures.")
    lines.append(r"\item \texttt{TNC} and \texttt{Newton-CG} were less robust for \texttt{infer\_D} under this setup.")
    lines.append(r"\end{itemize}")

    lines.append(r"\end{document}")
    out_tex = base_dir / "opt_method_study_report.tex"
    out_tex.write_text("\n".join(lines) + "\n")
    return out_tex


def main():
    base_dir = Path(__file__).resolve().parent
    summary_csv = base_dir / "opt_method_study_summary.csv"
    results_csv = base_dir / "opt_method_study_results.csv"

    summary_rows = load_summary(summary_csv)
    results_rows = load_results(results_csv)
    summary_map = build_summary_map(summary_rows)

    make_heatmaps(summary_map, base_dir)
    make_failure_plot(results_rows, base_dir)
    make_symmetric_case_plots(results_rows, base_dir)

    agg_rows = aggregate_across_noise(results_rows)
    fails, reasons = failure_analysis(results_rows)
    phi_stats = phi_estimate_stats(results_rows)
    lbfgsb_diag = lbfgsb_failure_diagnostics(results_rows)
    seed_info = seed_policy_summary(results_rows)
    method_seed_stats = method_seed_failure_summary(results_rows)
    infer_d_case_stats = infer_d_case_failure_stats(results_rows)

    out_tex = build_latex_report(
        base_dir,
        agg_rows,
        fails,
        reasons,
        phi_stats,
        lbfgsb_diag,
        seed_info,
        method_seed_stats,
        infer_d_case_stats,
        total_runs=len(results_rows),
    )
    print(f"Wrote report tex: {out_tex}")
    print("Wrote figures:")
    for name in [
        "success_rate_heatmap.png",
        "median_time_heatmap.png",
        "median_memory_heatmap.png",
        "median_rel_error_log10_heatmap.png",
        "failure_counts.png",
        "infer_d_symmetric_success_rate_heatmap.png",
        "infer_d_failure_rate_symmetric_vs_nonsymmetric.png",
    ]:
        print(f"- {base_dir / name}")


if __name__ == "__main__":
    main()
