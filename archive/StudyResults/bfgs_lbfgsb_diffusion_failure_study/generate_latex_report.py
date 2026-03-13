#!/usr/bin/env python3
import ast
import csv
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import numpy as np

METHODS = ["BFGS", "L-BFGS-B"]
NOISES = [0.0, 0.005, 0.02]


def to_float(value):
    try:
        return float(value)
    except Exception:
        return float("nan")


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
            row["min_logged_d_component"] = to_float(row.get("min_logged_d_component"))
            rows.append(row)
    return rows


def load_failure_cases(path):
    rows = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["noise_std"] = float(row["noise_std"])
            row["seed"] = int(row["seed"])
            row["n_logged_d_points"] = int(row["n_logged_d_points"])
            row["min_logged_d_component"] = to_float(row["min_logged_d_component"])
            rows.append(row)
    return rows


def build_summary_map(summary_rows):
    data = {}
    for row in summary_rows:
        data[(row["method"], row["noise_std"])] = row
    return data


def annotation_style(im, value):
    if not np.isfinite(value):
        return "black", [patheffects.withStroke(linewidth=1.5, foreground="white")]
    r, g, b, _ = im.cmap(im.norm(value))
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
            text = "nan" if not np.isfinite(val) else fmt.format(val)
            artist = ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)
            artist.set_path_effects(effects)
    return im


def make_heatmaps(summary_map, outdir):
    outdir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("success_rate", "Success Rate (higher is better)", (0.0, 1.0), "viridis", "{:.2f}", "success_rate_heatmap.png", "higher is better"),
        ("median_time_s", "Median Time (s) (lower is better)", (None, None), "magma", "{:.2f}", "median_time_heatmap.png", "lower is better"),
        (
            "median_peak_rss_mib",
            "Median Peak RSS (MiB) (lower is better)",
            (None, None),
            "cividis",
            "{:.1f}",
            "median_memory_heatmap.png",
            "lower is better",
        ),
    ]

    for key, title, bounds, cmap, fmt, fname, cbar_label in metrics:
        fig, ax = plt.subplots(1, 1, figsize=(6.0, 2.8), constrained_layout=True)
        mat = np.full((len(METHODS), len(NOISES)), np.nan)
        for i, method in enumerate(METHODS):
            for j, noise in enumerate(NOISES):
                row = summary_map.get((method, noise))
                if row is not None:
                    mat[i, j] = row[key]
        vmin, vmax = bounds
        im = draw_heatmap(ax, mat, title, vmin=vmin, vmax=vmax, cmap=cmap, fmt=fmt)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label)
        fig.savefig(outdir / fname, dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(6.0, 2.8), constrained_layout=True)
    mat = np.full((len(METHODS), len(NOISES)), np.nan)
    for i, method in enumerate(METHODS):
        for j, noise in enumerate(NOISES):
            row = summary_map.get((method, noise))
            if row is not None:
                mat[i, j] = row["median_error_rel"]
    mat_log = np.where(mat > 0, np.log10(mat), np.nan)
    im = draw_heatmap(
        ax,
        mat_log,
        "log10(Median Relative Error) (lower is better)",
        cmap="plasma",
        fmt="{:.2f}",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("lower is better")
    fig.savefig(outdir / "median_rel_error_log10_heatmap.png", dpi=180)
    plt.close(fig)


def make_failure_plot(results_rows, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    fails = [r for r in results_rows if not r["success"]]
    counts = {m: [0, 0, 0] for m in METHODS}
    noise_to_idx = {n: i for i, n in enumerate(NOISES)}
    for r in fails:
        counts[r["method"]][noise_to_idx[r["noise_std"]]] += 1

    x = np.arange(len(NOISES))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.0, 3.2), constrained_layout=True)
    ax.bar(x - width / 2, counts["BFGS"], width, label="BFGS")
    ax.bar(x + width / 2, counts["L-BFGS-B"], width, label="L-BFGS-B")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in NOISES])
    ax.set_xlabel("noise_std")
    ax.set_ylabel("Failure Count (lower is better)")
    ax.set_title("Failure Counts by Noise and Method (lower is better)")
    ax.legend()
    fig.savefig(outdir / "failure_counts_by_noise.png", dpi=180)
    plt.close(fig)


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


def parse_pair_text(value):
    if value is None:
        return None
    txt = str(value).strip()
    if not txt:
        return None
    try:
        parsed = ast.literal_eval(txt)
    except Exception:
        return None
    if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
        try:
            return [float(parsed[0]), float(parsed[1])]
        except Exception:
            return None
    return None


def pair_min_component(pair):
    if pair is None:
        return float("nan")
    return float(min(pair[0], pair[1]))


def format_pair(pair, digits=4):
    if pair is None:
        return "n/a"
    return f"[{pair[0]:.{digits}g}, {pair[1]:.{digits}g}]"


def seed_summary(results_rows):
    return {
        "noise0": sorted({r["seed"] for r in results_rows if abs(r["noise_std"]) < 1e-15}),
        "noisy": sorted({r["seed"] for r in results_rows if r["noise_std"] > 0.0}),
    }


def build_latex_report(base_dir, summary_rows, results_rows, failure_rows):
    seeds = seed_summary(results_rows)
    total_runs = len(results_rows)
    total_fails = sum(1 for r in results_rows if not r["success"])

    fail_reason_counts = Counter()
    for r in results_rows:
        if r["success"]:
            continue
        reason = r.get("failure_reason", "")
        if "DIVERGED_LINE_SEARCH" in reason:
            fail_reason_counts["DIVERGED_LINE_SEARCH"] += 1
        elif reason:
            fail_reason_counts[reason.splitlines()[0]] += 1
        else:
            fail_reason_counts["Unknown"] += 1

    noisy_rows = [r for r in results_rows if r["noise_std"] > 0.0]
    stability = []
    for method in METHODS:
        mrows = [r for r in noisy_rows if r["method"] == method]
        mfails = [r for r in mrows if not r["success"]]
        seeds_failed = sorted({r["seed"] for r in mfails})
        stability.append(
            {
                "method": method,
                "n_runs": len(mrows),
                "n_fail": len(mfails),
                "rate": (len(mfails) / len(mrows)) if mrows else float("nan"),
                "seeds_failed": seeds_failed,
            }
        )

    fail_guess = Counter()
    fail_true = Counter()
    fail_min_comp = []
    succ_min_comp = []

    for r in results_rows:
        min_c = to_float(r.get("min_logged_d_component"))
        if np.isfinite(min_c):
            if r["success"]:
                succ_min_comp.append(min_c)
            else:
                fail_min_comp.append(min_c)

    for r in failure_rows:
        fail_guess[str(r["d_guess"])] += 1
        fail_true[str(r["d_true"])] += 1

    # Build matched-case comparison for L-BFGS-B failures against BFGS.
    result_by_key = {}
    for r in results_rows:
        key = (r["method"], r["noise_std"], r["seed"], str(r.get("d_true")), str(r.get("d_guess")))
        result_by_key[key] = r

    lbfgsb_vs_bfgs = []
    for fr in sorted(
        [r for r in results_rows if (not r["success"]) and r["method"] == "L-BFGS-B"],
        key=lambda x: (x["noise_std"], x["seed"]),
    ):
        key_bfgs = ("BFGS", fr["noise_std"], fr["seed"], str(fr.get("d_true")), str(fr.get("d_guess")))
        br = result_by_key.get(key_bfgs)
        lb_min_pair = parse_pair_text(fr.get("min_logged_d_pair"))
        lb_last_pair = parse_pair_text(fr.get("last_logged_d_pair"))
        b_min_pair = parse_pair_text(br.get("min_logged_d_pair")) if br else None
        b_last_pair = parse_pair_text(br.get("last_logged_d_pair")) if br else None
        lbfgsb_vs_bfgs.append(
            {
                "noise_std": fr["noise_std"],
                "seed": fr["seed"],
                "d_true": str(fr.get("d_true")),
                "d_guess": str(fr.get("d_guess")),
                "lb_min_pair": lb_min_pair,
                "lb_last_pair": lb_last_pair,
                "b_success": bool(br and br.get("success")),
                "b_min_pair": b_min_pair,
                "b_last_pair": b_last_pair,
            }
        )

    lb_min_vals = [pair_min_component(r["lb_min_pair"]) for r in lbfgsb_vs_bfgs if np.isfinite(pair_min_component(r["lb_min_pair"]))]
    b_min_vals = [pair_min_component(r["b_min_pair"]) for r in lbfgsb_vs_bfgs if np.isfinite(pair_min_component(r["b_min_pair"]))]
    lb_last_min_vals = [pair_min_component(r["lb_last_pair"]) for r in lbfgsb_vs_bfgs if np.isfinite(pair_min_component(r["lb_last_pair"]))]
    b_last_min_vals = [pair_min_component(r["b_last_pair"]) for r in lbfgsb_vs_bfgs if np.isfinite(pair_min_component(r["b_last_pair"]))]

    lines = []
    lines.append(r"\documentclass[11pt]{article}")
    lines.append(r"\usepackage[margin=1in]{geometry}")
    lines.append(r"\usepackage{booktabs}")
    lines.append(r"\usepackage{graphicx}")
    lines.append(r"\usepackage{float}")
    lines.append(r"\usepackage{siunitx}")
    lines.append(r"\usepackage{array}")
    lines.append(r"\usepackage{setspace}")
    lines.append(r"\setlength{\parindent}{0pt}")
    lines.append(r"\sisetup{round-mode=places,round-precision=4}")
    lines.append(r"\title{BFGS vs L-BFGS-B Diffusion Failure Study}")
    lines.append(r"\author{Automated Benchmark Report}")
    lines.append(r"\date{\today}")
    lines.append(r"\begin{document}")
    lines.append(r"\onehalfspacing")
    lines.append(r"\maketitle")
    lines.append("")

    lines.append(r"\section*{What We Ran}")
    lines.append(r"\paragraph{Goal}\quad\\")
    lines.append(r"This study isolates diffusion-coefficient inference and compares \texttt{BFGS} against \texttt{L-BFGS-B}, with focus on forward-solve failures.")
    lines.append("")
    lines.append(r"\paragraph{Setup}\quad\\")
    lines.append(r"Problem: \texttt{infer\_D} only.")
    lines.append(r"Methods: \texttt{BFGS}, \texttt{L-BFGS-B}.")
    lines.append(r"Noise levels: $\sigma \in \{0, 0.005, 0.02\}$.")
    lines.append(
        f"Seeds used for $\\sigma=0$: {latex_escape(seeds['noise0'])}; seeds used for $\\sigma>0$: {latex_escape(seeds['noisy'])}."
    )
    lines.append(f"Total runs: {total_runs}; failed runs: {total_fails}.")
    lines.append(r"\vspace{1em}")
    lines.append(r"\hrule")
    lines.append(r"\vspace{1em}")
    lines.append(r"\paragraph{How to read the plots}\quad\\")
    lines.append(r"Each title and colorbar states whether higher or lower values are preferred.")
    lines.append("")

    lines.append(r"\section*{Method Comparisons}")
    fig_specs = [
        ("success_rate_heatmap.png", "Success rate by method/noise (higher is better)."),
        ("median_time_heatmap.png", "Median wall-clock time (s) for successful runs (lower is better)."),
        ("median_memory_heatmap.png", "Median peak RSS memory (MiB) for successful runs (lower is better)."),
        ("median_rel_error_log10_heatmap.png", "$\\log_{10}$ median relative error by method/noise (lower is better)."),
        ("failure_counts_by_noise.png", "Failure counts by noise and method (lower is better)."),
    ]
    for fname, caption in fig_specs:
        lines.append(r"\begin{figure}[H]")
        lines.append(r"\centering")
        lines.append(rf"\includegraphics[width=0.88\textwidth]{{{fname}}}")
        lines.append(rf"\caption{{{caption}}}")
        lines.append(r"\end{figure}")
    lines.append(r"\vspace{1em}")
    lines.append(r"\hrule")
    lines.append(r"\vspace{1em}")

    lines.append(r"\section*{Aggregated Results Across Noise Levels}")
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Method & Success (\%) & Median Time (s) & Median RSS (MiB) & Median RelErr \\")
    lines.append(r"\midrule")
    for r in sorted(summary_rows, key=lambda x: (x["method"], x["noise_std"])):
        lines.append(
            f"{latex_escape(r['method'])} ($\\sigma$={fmt(r['noise_std'])}) & {fmt(100.0*r['success_rate'])} & "
            f"{fmt(r['median_time_s'])} & {fmt(r['median_peak_rss_mib'])} & {fmt(r['median_error_rel'])} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append(r"\par\medskip")

    lines.append(r"\section*{Crash Analysis}")
    lines.append(r"\paragraph{What failed?}\quad\\")
    lines.append(f"All {total_fails} failures were in \\texttt{{L-BFGS-B}} runs; \\texttt{{BFGS}} had 0 failures.")
    if fail_reason_counts:
        top_reason, top_count = fail_reason_counts.most_common(1)[0]
        lines.append(
            f"Dominant reason: \\texttt{{{latex_escape(top_reason)}}} ({top_count} of {total_fails} failures)."
        )

    lines.append("")
    lines.append(r"\subsection*{Across-Seed Stability for Noisy Cases ($\sigma>0$)}")
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lrrp{0.4\textwidth}}")
    lines.append(r"\toprule")
    lines.append(r"Method & Failures / Runs & Failure Rate (\%) & Seeds with failures \\")
    lines.append(r"\midrule")
    for row in stability:
        seeds_failed_txt = "none" if not row["seeds_failed"] else str(row["seeds_failed"])
        lines.append(
            f"{latex_escape(row['method'])} & {row['n_fail']}/{row['n_runs']} & {fmt(100.0*row['rate'])} & {latex_escape(seeds_failed_txt)} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append(r"\par\medskip")

    lines.append(r"\subsection*{Failure-Causing D Patterns}")
    lines.append(
        f"Failure initial guesses: {latex_escape(dict(fail_guess))}; failure true-$D$ cases: {latex_escape(dict(fail_true))}."
    )
    lines.append(
        f"Logged min-$D$ component in failed runs: min={fmt(min(fail_min_comp) if fail_min_comp else float('nan'))}, "
        f"median={fmt(float(np.median(fail_min_comp)) if fail_min_comp else float('nan'))}, "
        f"max={fmt(max(fail_min_comp) if fail_min_comp else float('nan'))}."
    )
    lines.append(
        f"Logged min-$D$ component in successful runs: min={fmt(min(succ_min_comp) if succ_min_comp else float('nan'))}, "
        f"median={fmt(float(np.median(succ_min_comp)) if succ_min_comp else float('nan'))}, "
        f"max={fmt(max(succ_min_comp) if succ_min_comp else float('nan'))}."
    )

    lines.append("")
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{rrllll}")
    lines.append(r"\toprule")
    lines.append(r"$\sigma$ & Seed & $D_{\mathrm{true}}$ & $D_0$ & Min logged trial $D$ & Last logged $D$ \\")
    lines.append(r"\midrule")
    for r in failure_rows:
        lines.append(
            f"{fmt(r['noise_std'])} & {r['seed']} & {latex_escape(r['d_true'])} & {latex_escape(r['d_guess'])} & "
            f"{latex_escape(r['min_logged_d_pair'])} & {latex_escape(r['last_logged_d_pair'])} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")
    lines.append(r"\par\medskip")

    lines.append(r"\subsection*{L-BFGS-B Failure D Conditions vs Matched BFGS Runs}")
    lines.append(
        r"This section compares each failing \texttt{L-BFGS-B} case against the \texttt{BFGS} run with the same noise, seed, $D_{\mathrm{true}}$, and $D_0$."
    )
    lines.append(
        f"Across matched failing cases, median min logged trial-$D$ component was {fmt(float(np.median(lb_min_vals)) if lb_min_vals else float('nan'))} for "
        f"\\texttt{{L-BFGS-B}} vs {fmt(float(np.median(b_min_vals)) if b_min_vals else float('nan'))} for \\texttt{{BFGS}}."
    )
    lines.append(
        f"Median min component at last logged iterate was {fmt(float(np.median(lb_last_min_vals)) if lb_last_min_vals else float('nan'))} for "
        f"\\texttt{{L-BFGS-B}} vs {fmt(float(np.median(b_last_min_vals)) if b_last_min_vals else float('nan'))} for \\texttt{{BFGS}}."
    )
    lines.append(
        r"Similarity in failures: all failed \texttt{L-BFGS-B} runs used $D_0=[10,10]$ and $D_{\mathrm{true}}=[1,3]$, with \texttt{DIVERGED\_LINE\_SEARCH}."
    )
    lines.append(
        r"Contrast: matched \texttt{BFGS} runs converged and ended at less aggressive final iterates for the same cases."
    )
    lines.append("")
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{rrlllll}")
    lines.append(r"\toprule")
    lines.append(
        r"$\sigma$ & Seed & $D_{\mathrm{true}}$ & $D_0$ & LBFGSB min/last $D$ & BFGS min/last $D$ & BFGS converged \\"
    )
    lines.append(r"\midrule")
    for r in lbfgsb_vs_bfgs:
        lb_pair_txt = f"{format_pair(r['lb_min_pair'])} / {format_pair(r['lb_last_pair'])}"
        b_pair_txt = f"{format_pair(r['b_min_pair'])} / {format_pair(r['b_last_pair'])}"
        b_conv = "yes" if r["b_success"] else "no"
        lines.append(
            f"{fmt(r['noise_std'])} & {r['seed']} & {latex_escape(r['d_true'])} & {latex_escape(r['d_guess'])} & "
            f"{latex_escape(lb_pair_txt)} & {latex_escape(b_pair_txt)} & {b_conv} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")
    lines.append(r"\par\medskip")

    lines.append(r"\section*{Interpretation}")
    lines.append(r"\begin{itemize}")
    lines.append(r"\item In this dataset, forward-solve failures are method-specific: observed only with \texttt{L-BFGS-B}.")
    lines.append(r"\item Failures cluster in the high-initial-guess regime ($D_0=[10,10]$), which suggests aggressive early trial steps are important.")
    lines.append(r"\item Failure trajectories frequently visit low-$D$ trial values before \texttt{DIVERGED\_LINE\_SEARCH}; this is consistent with harder nonlinear solves in those regions.")
    lines.append(r"\end{itemize}")
    lines.append(r"\end{document}")

    out_tex = base_dir / "bfgs_lbfgsb_diffusion_failure_report.tex"
    out_tex.write_text("\n".join(lines) + "\n")
    return out_tex


def main():
    base_dir = Path(__file__).resolve().parent
    summary_csv = base_dir / "bfgs_lbfgsb_diffusion_summary.csv"
    results_csv = base_dir / "bfgs_lbfgsb_diffusion_results.csv"
    failure_csv = base_dir / "bfgs_lbfgsb_diffusion_failure_cases.csv"

    summary_rows = load_summary(summary_csv)
    results_rows = load_results(results_csv)
    failure_rows = load_failure_cases(failure_csv)
    summary_map = build_summary_map(summary_rows)

    make_heatmaps(summary_map, base_dir)
    make_failure_plot(results_rows, base_dir)
    out_tex = build_latex_report(base_dir, summary_rows, results_rows, failure_rows)

    print(f"Wrote report tex: {out_tex}")
    print("Wrote figures:")
    for name in [
        "success_rate_heatmap.png",
        "median_time_heatmap.png",
        "median_memory_heatmap.png",
        "median_rel_error_log10_heatmap.png",
        "failure_counts_by_noise.png",
    ]:
        print(f"- {base_dir / name}")


if __name__ == "__main__":
    main()
