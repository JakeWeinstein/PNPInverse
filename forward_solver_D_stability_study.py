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
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

# Keep Firedrake caches writable in sandboxed environments.
os.environ.setdefault("FIREDRAKE_TSFC_KERNEL_CACHE_DIR", "/tmp/firedrake-tsfc")
os.environ.setdefault("PYOP2_CACHE_DIR", "/tmp/pyop2")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np

ITER_RE = re.compile(r"after\s+(\d+)\s+nonlinear iterations")


def build_common_solver_options():
    # Match inverse-study setup for apples-to-apples forward-solver behavior.
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


def compact_log(text: str) -> str:
    return text.replace("\r", "").replace("\n", "\\n")


def tail_lines(text: str, n: int = 25) -> str:
    lines = text.strip().splitlines()
    if not lines:
        return ""
    return "\n".join(lines[-n:])


def build_configs_default():
    # Original coarse global sweep + denser strip around observed L-BFGS-B region.
    d0_global = np.geomspace(0.01, 3.0, 12)
    d0_focus = np.linspace(0.02, 0.35, 10)
    d1_global = np.geomspace(0.2, 3.5, 12)
    d1_focus = np.linspace(0.3, 1.3, 10)

    d0_vals = np.unique(np.round(np.concatenate([d0_global, d0_focus]), 6)).tolist()
    d1_vals = np.unique(np.round(np.concatenate([d1_global, d1_focus]), 6)).tolist()

    cfgs = []
    for d0 in d0_vals:
        for d1 in d1_vals:
            cfgs.append({"d0": float(d0), "d1": float(d1)})

    meta = {
        "mode": "default",
        "label": "Default grid",
        "description": "Coarse global sweep with an extra LBFGS-B focus strip.",
        "n_points": len(cfgs),
        "composition": (
            f"{len(d0_vals)} unique D0 values x {len(d1_vals)} unique D1 values "
            f"(cross-product grid)"
        ),
    }
    return cfgs, meta


def build_configs_anisotropy_dense(anis_ratio_threshold: float):
    # Higher-density sampling concentrated in high-anisotropy regions.
    points = set()
    counts = Counter()

    def add_cartesian(d0_vals, d1_vals, bucket, require_ratio=False):
        for d0 in d0_vals:
            for d1 in d1_vals:
                ratio = max(d0, d1) / min(d0, d1)
                if require_ratio and ratio < anis_ratio_threshold:
                    continue
                key = (round(float(d0), 6), round(float(d1), 6))
                if key not in points:
                    points.add(key)
                    counts[bucket] += 1

    # Coarse global backbone for full-domain coverage.
    d0_global = np.geomspace(0.01, 3.0, 18)
    d1_global = np.geomspace(0.2, 3.5, 18)
    add_cartesian(d0_global, d1_global, "global_coarse", require_ratio=False)

    # Mid-range focus from prior inverse-study behavior.
    d0_focus = np.linspace(0.02, 0.35, 14)
    d1_focus = np.linspace(0.3, 1.3, 14)
    add_cartesian(d0_focus, d1_focus, "lbfgsb_focus", require_ratio=False)

    # Dense anisotropy bands (low/high and high/low corners).
    d0_low = np.geomspace(0.01, 0.35, 20)
    d1_high = np.geomspace(1.0, 3.5, 22)
    add_cartesian(d0_low, d1_high, "anis_lowD0_highD1", require_ratio=True)

    d0_high = np.geomspace(1.0, 3.0, 22)
    d1_low = np.geomspace(0.2, 0.7, 20)
    add_cartesian(d0_high, d1_low, "anis_highD0_lowD1", require_ratio=True)

    # Extra density in extreme corners where failures are often concentrated.
    d0_low_ext = np.geomspace(0.01, 0.12, 16)
    d1_high_ext = np.geomspace(2.0, 3.5, 18)
    add_cartesian(d0_low_ext, d1_high_ext, "anis_extreme_lowD0_highD1", require_ratio=True)

    d0_high_ext = np.geomspace(2.0, 3.0, 18)
    d1_low_ext = np.geomspace(0.2, 0.35, 16)
    add_cartesian(d0_high_ext, d1_low_ext, "anis_extreme_highD0_lowD1", require_ratio=True)

    cfgs = [{"d0": d0, "d1": d1} for d0, d1 in sorted(points)]
    composition = (
        f"global_coarse={counts['global_coarse']}, "
        f"lbfgsb_focus={counts['lbfgsb_focus']}, "
        f"anis_lowD0_highD1={counts['anis_lowD0_highD1']}, "
        f"anis_highD0_lowD1={counts['anis_highD0_lowD1']}, "
        f"anis_extreme_lowD0_highD1={counts['anis_extreme_lowD0_highD1']}, "
        f"anis_extreme_highD0_lowD1={counts['anis_extreme_highD0_lowD1']}"
    )
    meta = {
        "mode": "anisotropy_dense",
        "label": "Anisotropy-dense grid",
        "description": (
            "Coarse global grid with dense sampling concentrated where "
            f"max(D0,D1)/min(D0,D1) >= {anis_ratio_threshold:g}."
        ),
        "anis_ratio_threshold": float(anis_ratio_threshold),
        "n_points": len(cfgs),
        "composition": composition,
    }
    return cfgs, meta


def build_configs(study_mode="default", anis_ratio_threshold=8.0):
    if study_mode == "default":
        return build_configs_default()
    if study_mode == "anisotropy_dense":
        return build_configs_anisotropy_dense(anis_ratio_threshold=anis_ratio_threshold)
    raise ValueError(f"Unsupported study_mode '{study_mode}'")


def parse_fail_reason(reason):
    txt = (reason or "").strip()
    if not txt:
        return "Unknown"
    if "DIVERGED_LINE_SEARCH" in txt:
        return "DIVERGED_LINE_SEARCH"
    if "DIVERGED_MAX_IT" in txt:
        return "DIVERGED_MAX_IT"
    return txt.splitlines()[0]


def is_converged_run(row):
    return bool(row.get("success")) and not (row.get("failure_reason") or "").strip()


def run_single(config):
    from Utils.forsolve import build_context, build_forms, forsolve, set_initial_conditions

    d0 = float(config["d0"])
    d1 = float(config["d1"])

    # Same parameterization used in inverse studies.
    n_species = 2
    solver_params = [
        n_species,              # n_species
        1,                      # order
        2e-2,                   # dt
        0.1,                    # t_end
        [1, -1],                # z_vals
        [d0, d1],               # D_vals
        [0.0, 0.0],             # a_vals
        0.05,                   # phi_applied (unused in current weak form, kept consistent)
        [0.1, 0.1],             # c0 per species
        1.0,                    # phi0
        build_common_solver_options(),
    ]

    run_log = io.StringIO()
    start = time.perf_counter()
    success = False
    failure_reason = ""
    nonlinear_iters = None

    try:
        with contextlib.redirect_stdout(run_log), contextlib.redirect_stderr(run_log):
            ctx = build_context(solver_params)
            ctx = build_forms(ctx, solver_params)
            set_initial_conditions(ctx, solver_params, blob=True)
            forsolve(ctx, solver_params, print_interval=1000)
            success = True
    except Exception as exc:
        failure_reason = f"{type(exc).__name__}: {exc}"
        m = ITER_RE.search(failure_reason)
        if m:
            nonlinear_iters = int(m.group(1))

    elapsed = time.perf_counter() - start
    return {
        "d0": d0,
        "d1": d1,
        "success": success,
        "time_s": float(elapsed),
        "failure_reason": failure_reason,
        "failure_reason_short": parse_fail_reason(failure_reason),
        "nonlinear_iters": nonlinear_iters,
        "log_tail": compact_log(tail_lines(run_log.getvalue())),
    }


def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def safe_median(values):
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return float("nan")
    return float(statistics.median(vals))


def safe_min(values):
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return float("nan")
    return float(min(vals))


def safe_max(values):
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return float("nan")
    return float(max(vals))


def fmt(x, digits=4):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "nan"
    if isinstance(x, float):
        return f"{x:.{digits}g}"
    return str(x)


def latex_escape(text: str) -> str:
    return str(text).replace("_", r"\_")


def analyze_results(results):
    succ = [r for r in results if is_converged_run(r)]
    fail = [r for r in results if not is_converged_run(r)]

    fail_reasons = Counter(r["failure_reason_short"] for r in fail)
    fail_d0 = [r["d0"] for r in fail]
    fail_d1 = [r["d1"] for r in fail]

    succ_min_d = [min(r["d0"], r["d1"]) for r in succ]
    fail_min_d = [min(r["d0"], r["d1"]) for r in fail]
    succ_ratio = [max(r["d0"], r["d1"]) / min(r["d0"], r["d1"]) for r in succ if min(r["d0"], r["d1"]) > 0]
    fail_ratio = [max(r["d0"], r["d1"]) / min(r["d0"], r["d1"]) for r in fail if min(r["d0"], r["d1"]) > 0]

    # Region motivated by LBFGS-B trial points from previous studies.
    focus_fail = [
        r for r in results
        if (0.02 <= r["d0"] <= 0.35) and (0.3 <= r["d1"] <= 1.3)
    ]
    focus_fail_count = sum(not is_converged_run(r) for r in focus_fail)

    def count_and_rate(rows):
        n = len(rows)
        n_fail = sum(not is_converged_run(r) for r in rows)
        rate = (n_fail / n) if n else float("nan")
        return n, n_fail, rate

    regime_min_d_low = [r for r in results if min(r["d0"], r["d1"]) <= 0.1]
    regime_ratio_high = [r for r in results if (max(r["d0"], r["d1"]) / min(r["d0"], r["d1"])) >= 10.0]
    regime_low_and_imbalanced = [
        r for r in results
        if (min(r["d0"], r["d1"]) <= 0.1) and ((max(r["d0"], r["d1"]) / min(r["d0"], r["d1"])) >= 10.0)
    ]
    regime_extreme_corner = [r for r in results if (max(r["d0"], r["d1"]) >= 2.5) and (min(r["d0"], r["d1"]) <= 0.1)]
    regime_low_d0_high_d1 = [r for r in results if (r["d0"] <= 0.1) and (r["d1"] >= 1.3)]
    regime_high_d0_low_d1 = [r for r in results if (r["d0"] >= 1.0) and (r["d1"] <= 0.35)]

    n_min_d_low, nfail_min_d_low, rate_min_d_low = count_and_rate(regime_min_d_low)
    n_ratio_high, nfail_ratio_high, rate_ratio_high = count_and_rate(regime_ratio_high)
    n_low_imb, nfail_low_imb, rate_low_imb = count_and_rate(regime_low_and_imbalanced)
    n_extreme_corner, nfail_extreme_corner, rate_extreme_corner = count_and_rate(regime_extreme_corner)
    n_low_d0_high_d1, nfail_low_d0_high_d1, rate_low_d0_high_d1 = count_and_rate(regime_low_d0_high_d1)
    n_high_d0_low_d1, nfail_high_d0_low_d1, rate_high_d0_low_d1 = count_and_rate(regime_high_d0_low_d1)

    return {
        "n_total": len(results),
        "n_success": len(succ),
        "n_fail": len(fail),
        "success_rate": (len(succ) / len(results)) if results else float("nan"),
        "fail_reasons": dict(fail_reasons),
        "succ_time_median": safe_median([r["time_s"] for r in succ]),
        "succ_time_min": safe_min([r["time_s"] for r in succ]),
        "succ_time_max": safe_max([r["time_s"] for r in succ]),
        "fail_d0_min": safe_min(fail_d0),
        "fail_d0_max": safe_max(fail_d0),
        "fail_d1_min": safe_min(fail_d1),
        "fail_d1_max": safe_max(fail_d1),
        "succ_min_d_median": safe_median(succ_min_d),
        "fail_min_d_median": safe_median(fail_min_d),
        "succ_ratio_median": safe_median(succ_ratio),
        "fail_ratio_median": safe_median(fail_ratio),
        "focus_region_n": len(focus_fail),
        "focus_region_fail_n": focus_fail_count,
        "focus_region_fail_rate": (focus_fail_count / len(focus_fail)) if focus_fail else float("nan"),
        "regime_min_d_low_n": n_min_d_low,
        "regime_min_d_low_fail_n": nfail_min_d_low,
        "regime_min_d_low_fail_rate": rate_min_d_low,
        "regime_ratio_high_n": n_ratio_high,
        "regime_ratio_high_fail_n": nfail_ratio_high,
        "regime_ratio_high_fail_rate": rate_ratio_high,
        "regime_low_imb_n": n_low_imb,
        "regime_low_imb_fail_n": nfail_low_imb,
        "regime_low_imb_fail_rate": rate_low_imb,
        "regime_extreme_corner_n": n_extreme_corner,
        "regime_extreme_corner_fail_n": nfail_extreme_corner,
        "regime_extreme_corner_fail_rate": rate_extreme_corner,
        "regime_low_d0_high_d1_n": n_low_d0_high_d1,
        "regime_low_d0_high_d1_fail_n": nfail_low_d0_high_d1,
        "regime_low_d0_high_d1_fail_rate": rate_low_d0_high_d1,
        "regime_high_d0_low_d1_n": n_high_d0_low_d1,
        "regime_high_d0_low_d1_fail_n": nfail_high_d0_low_d1,
        "regime_high_d0_low_d1_fail_rate": rate_high_d0_low_d1,
    }


def make_plot(results, out_png):
    succ = [r for r in results if is_converged_run(r)]
    fail = [r for r in results if not is_converged_run(r)]
    all_d0 = [r["d0"] for r in results if r.get("d0") and r["d0"] > 0]
    all_d1 = [r["d1"] for r in results if r.get("d1") and r["d1"] > 0]

    fig, ax = plt.subplots(figsize=(7.0, 5.5), constrained_layout=True)

    if succ:
        sc = ax.scatter(
            [r["d0"] for r in succ],
            [r["d1"] for r in succ],
            c=[r["time_s"] for r in succ],
            cmap="viridis",
            marker="s",
            s=54,
            linewidths=0.0,
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Convergence time (s), lower is better")

    if fail:
        ax.scatter(
            [r["d0"] for r in fail],
            [r["d1"] for r in fail],
            marker="x",
            s=55,
            linewidths=1.3,
            color="red",
            label="Did not converge",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    line_added = False
    if all_d0 and all_d1:
        # Draw the diagonal only over the overlapping axis span so it does not expand axis limits.
        d_min = max(min(all_d0), min(all_d1))
        d_max = min(max(all_d0), max(all_d1))
        if d_min < d_max:
            ax.plot([d_min, d_max], [d_min, d_max], "--", color="black", linewidth=0.9, alpha=0.6, label="D_0 = D_1")
            line_added = True
    if fail or line_added:
        ax.legend(loc="upper right")
    ax.set_xlabel("D_0")
    ax.set_ylabel("D_1")
    ax.set_title("Forward Solver Stability Map: Convergence Time and Failures")
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def write_tex_report(out_tex: Path, analysis, results, out_png_name, study_meta=None):
    fail_rows = [r for r in results if not is_converged_run(r)]
    fail_rows = sorted(fail_rows, key=lambda r: (r["d0"], r["d1"]))
    rep_fail_rows = fail_rows[:10]
    fail_reason_items = [f"{latex_escape(k)}: {v}" for k, v in analysis["fail_reasons"].items()]
    fail_reason_text = ", ".join(fail_reason_items) if fail_reason_items else "none"

    lines = []
    lines.append(r"\documentclass[11pt]{article}")
    lines.append(r"\usepackage[margin=1in]{geometry}")
    lines.append(r"\usepackage{booktabs}")
    lines.append(r"\usepackage{graphicx}")
    lines.append(r"\usepackage{float}")
    lines.append(r"\usepackage{siunitx}")
    lines.append(r"\usepackage{setspace}")
    lines.append(r"\setlength{\parindent}{0pt}")
    lines.append(r"\sisetup{round-mode=places,round-precision=4}")
    lines.append(r"\title{Forward Solver D-Stability Study}")
    lines.append(r"\author{Automated Benchmark Report}")
    lines.append(r"\date{\today}")
    lines.append(r"\begin{document}")
    lines.append(r"\onehalfspacing")
    lines.append(r"\maketitle")
    lines.append("")

    lines.append(r"\section*{What We Ran}")
    lines.append(r"\paragraph{Goal}\quad\\")
    lines.append(
        r"This study maps where the forward PNP solve converges or fails as $(D_0,D_1)$ varies."
    )
    lines.append("")
    lines.append(r"\paragraph{Setup}\quad\\")
    if study_meta is None:
        study_meta = {
            "label": "Default grid",
            "description": "Coarse global sweep with an extra LBFGS-B focus strip.",
            "n_points": len(results),
            "composition": "n/a",
        }
    lines.append(
        rf"Study mode: {latex_escape(study_meta.get('label', 'n/a'))}. "
        rf"Grid points: {int(study_meta.get('n_points', len(results)))}. "
        rf"{latex_escape(study_meta.get('description', ''))} "
        rf"Composition: {latex_escape(study_meta.get('composition', 'n/a'))}. "
        r"Non-$D$ parameters match the inverse studies: $n_{\mathrm{species}}=2$, $dt=0.02$, "
        r"$t_{\mathrm{end}}=0.1$, $z=[1,-1]$, $a=[0,0]$, $c_0=[0.1,0.1]$, $\phi_0=1.0$, "
        r"SNES newtonls + LU/MUMPS."
    )
    lines.append("")
    lines.append(r"\vspace{1em}")
    lines.append(r"\hrule")
    lines.append(r"\vspace{1em}")
    lines.append(r"\paragraph{How to read the map}\quad\\")
    lines.append(
        r"Colored cells are converged solves with color = wall time (seconds, lower is better); "
        r"red x marks failed solves. "
        r"The dashed diagonal marks $D_0 = D_1$."
    )

    lines.append(r"\section*{Summary Metrics}")
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{lr}")
    lines.append(r"\toprule")
    lines.append(r"Metric & Value \\")
    lines.append(r"\midrule")
    lines.append(rf"Total $(D_0,D_1)$ points & {analysis['n_total']} \\")
    lines.append(rf"Converged points & {analysis['n_success']} \\")
    lines.append(rf"Failed points & {analysis['n_fail']} \\")
    lines.append(rf"Success rate & {fmt(100.0*analysis['success_rate'], 5)}\% \\")
    lines.append(rf"Median converged solve time (s) & {fmt(analysis['succ_time_median'], 5)} \\")
    lines.append(rf"Failure reasons & {fail_reason_text} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")

    lines.append(r"\section*{D-Stability Map}")
    lines.append(r"\begin{figure}[H]")
    lines.append(r"\centering")
    lines.append(rf"\includegraphics[width=0.9\textwidth]{{{out_png_name}}}")
    lines.append(r"\caption{Forward-solver behavior in $(D_0,D_1)$ space. Dashed line marks $D_0 = D_1$. Converged time is lower-is-better.}")
    lines.append(r"\end{figure}")
    lines.append("")

    lines.append(r"\section*{Failure Pattern Summary}")
    lines.append(
        rf"Failures span $D_0\in[{fmt(analysis['fail_d0_min'])},{fmt(analysis['fail_d0_max'])}]$ and "
        rf"$D_1\in[{fmt(analysis['fail_d1_min'])},{fmt(analysis['fail_d1_max'])}]$, "
        rf"but cluster most strongly in low-and-imbalanced diffusion regimes."
    )
    lines.append(
        rf"Median $\min(D_0,D_1)$ is {fmt(analysis['fail_min_d_median'])} for failed points "
        rf"vs {fmt(analysis['succ_min_d_median'])} for converged points; "
        rf"median anisotropy $\max/\min$ is {fmt(analysis['fail_ratio_median'])} vs {fmt(analysis['succ_ratio_median'])}."
    )
    lines.append(
        r"\begin{table}[H]"
    )
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lrr}")
    lines.append(r"\toprule")
    lines.append(r"Region & Failures / Points & Failure Rate (\%) \\")
    lines.append(r"\midrule")
    lines.append(
        rf"LBFGS-B focus: $D_0\in[0.02,0.35], D_1\in[0.3,1.3]$ & "
        rf"{analysis['focus_region_fail_n']} / {analysis['focus_region_n']} & "
        rf"{fmt(100.0*analysis['focus_region_fail_rate'])} \\"
    )
    lines.append(
        rf"High anisotropy: $\max(D_0,D_1)/\min(D_0,D_1)\ge 10$ & "
        rf"{analysis['regime_ratio_high_fail_n']} / {analysis['regime_ratio_high_n']} & "
        rf"{fmt(100.0*analysis['regime_ratio_high_fail_rate'])} \\"
    )
    lines.append(
        rf"Extreme corner: $\min(D)\le 0.1$, $\max(D)\ge 2.5$ & "
        rf"{analysis['regime_extreme_corner_fail_n']} / {analysis['regime_extreme_corner_n']} & "
        rf"{fmt(100.0*analysis['regime_extreme_corner_fail_rate'])} \\"
    )
    lines.append(
        rf"Low-$D_0$/high-$D_1$: $D_0\le 0.1$, $D_1\ge 1.3$ & "
        rf"{analysis['regime_low_d0_high_d1_fail_n']} / {analysis['regime_low_d0_high_d1_n']} & "
        rf"{fmt(100.0*analysis['regime_low_d0_high_d1_fail_rate'])} \\"
    )
    lines.append(
        rf"High-$D_0$/low-$D_1$: $D_0\ge 1.0$, $D_1\le 0.35$ & "
        rf"{analysis['regime_high_d0_low_d1_fail_n']} / {analysis['regime_high_d0_low_d1_n']} & "
        rf"{fmt(100.0*analysis['regime_high_d0_low_d1_fail_rate'])} \\"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    lines.append(r"\section*{Representative Failed Points}")
    if not rep_fail_rows:
        lines.append(r"No failed forward solves were observed.")
    else:
        lines.append(r"\begin{table}[H]")
        lines.append(r"\centering")
        lines.append(r"\small")
        lines.append(r"\begin{tabular}{rrrl}")
        lines.append(r"\toprule")
        lines.append(r"$D_0$ & $D_1$ & Time (s) & Failure reason \\")
        lines.append(r"\midrule")
        for r in rep_fail_rows:
            lines.append(
                rf"{fmt(r['d0'])} & {fmt(r['d1'])} & {fmt(r['time_s'])} & {r['failure_reason_short'].replace('_', r'\_')} \\"
            )
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
    lines.append(
        r"Full failed-point list is available in "
        r"\texttt{forward\_solver\_d\_stability\_results.csv}."
    )

    lines.append(r"\section*{Interpretation}")
    lines.append(r"\begin{itemize}")
    lines.append(
        r"\item The dominant failure mode is \texttt{DIVERGED\_LINE\_SEARCH}, indicating globalization difficulty in SNES Newton steps."
    )
    lines.append(
        r"\item Failure probability rises sharply when one diffusion coefficient is small and the other is large (strong imbalance)."
    )
    lines.append(
        r"\item This explains the inverse-study behavior: methods that step through highly imbalanced $D$ trial points are more likely to trigger forward-solve failures."
    )
    lines.append(r"\end{itemize}")

    lines.append(r"\end{document}")
    out_tex.write_text("\n".join(lines) + "\n")


def run_parent(output_dir: Path, timeout_s: int, study_mode: str, anis_ratio_threshold: float):
    cfgs, study_meta = build_configs(study_mode=study_mode, anis_ratio_threshold=anis_ratio_threshold)
    total = len(cfgs)
    script = Path(__file__).resolve()
    results = []

    print(f"Sweeping {total} forward solves ({study_meta['label']})", flush=True)
    print(f"Composition: {study_meta['composition']}", flush=True)

    # Warmup one interior point.
    warm = cfgs[min(len(cfgs) // 2, len(cfgs) - 1)]
    payload = base64.b64encode(json.dumps(warm).encode("utf-8")).decode("ascii")
    subprocess.run(
        [sys.executable, str(script), "--worker", payload],
        cwd=str(script.parent),
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )

    for i, cfg in enumerate(cfgs, start=1):
        print(f"[{i:03d}/{total}] D0={cfg['d0']:.6g} D1={cfg['d1']:.6g}", flush=True)
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
                        "d0": cfg["d0"],
                        "d1": cfg["d1"],
                        "success": False,
                        "time_s": float("nan"),
                        "failure_reason": "No RESULT_JSON marker from worker",
                        "failure_reason_short": "HarnessError",
                        "nonlinear_iters": None,
                        "log_tail": compact_log(tail_lines(proc.stdout + "\n" + proc.stderr)),
                    }
                )
            else:
                results.append(json.loads(result_line))
        except subprocess.TimeoutExpired as exc:
            results.append(
                {
                    "d0": cfg["d0"],
                    "d1": cfg["d1"],
                    "success": False,
                    "time_s": float(timeout_s),
                    "failure_reason": f"TimeoutExpired after {timeout_s}s",
                    "failure_reason_short": "TimeoutExpired",
                    "nonlinear_iters": None,
                    "log_tail": compact_log(tail_lines((exc.stdout or "") + "\n" + (exc.stderr or ""))),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "forward_solver_d_stability_results.csv"
    out_png = output_dir / "forward_solver_d_stability_map.png"
    out_tex = output_dir / "forward_solver_d_stability_report.tex"

    write_csv(out_csv, results)
    make_plot(results, out_png)
    analysis = analyze_results(results)
    write_tex_report(out_tex, analysis, results, out_png.name, study_meta=study_meta)

    print("\nWrote:")
    print(f"- {out_csv}")
    print(f"- {out_png}")
    print(f"- {out_tex}")


def main():
    parser = argparse.ArgumentParser(description="Forward solver stability study over D0/D1 grid")
    parser.add_argument("--worker", type=str, help="base64 encoded single-run config")
    default_output = Path("StudyResults") / "forward_solver_D_stability"
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="Directory for outputs",
    )
    parser.add_argument("--timeout-s", type=int, default=120, help="Per-run timeout in seconds")
    parser.add_argument(
        "--study-mode",
        choices=["default", "anisotropy_dense"],
        default="default",
        help="Grid design mode for D0/D1 sampling",
    )
    parser.add_argument(
        "--anis-ratio-threshold",
        type=float,
        default=8.0,
        help="Anisotropy threshold max(D0,D1)/min(D0,D1) used in anisotropy_dense mode",
    )
    args = parser.parse_args()

    if args.worker:
        cfg = json.loads(base64.b64decode(args.worker.encode("ascii")).decode("utf-8"))
        result = run_single(cfg)
        print("RESULT_JSON:" + json.dumps(result))
        return

    if args.study_mode == "anisotropy_dense" and args.output_dir == default_output:
        args.output_dir = Path("StudyResults") / "forward_solver_D_stability_anisotropy_dense"

    run_parent(
        output_dir=args.output_dir,
        timeout_s=args.timeout_s,
        study_mode=args.study_mode,
        anis_ratio_threshold=args.anis_ratio_threshold,
    )


if __name__ == "__main__":
    main()
