#!/usr/bin/env python3
"""Generate publication-quality figures and LaTeX tables for the V&V report.

Reads all data from StudyResults/ JSON/CSV files and produces:
  - 3 PDF figures in writeups/vv_report/figures/
  - 5 LaTeX table snippets in writeups/vv_report/tables/

Dependencies: matplotlib, numpy (+ stdlib json, csv, pathlib, sys)
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent  # project root
STUDY = ROOT / "StudyResults"
FIGDIR = Path(__file__).resolve().parent / "figures"
TABDIR = Path(__file__).resolve().parent / "tables"

# ---------------------------------------------------------------------------
# Publication styling
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.figsize": (6.5, 4.5),
})

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_mms_convergence():
    """Load MMS convergence data."""
    path = STUDY / "mms_convergence" / "convergence_data.json"
    with open(path) as f:
        return json.load(f)


def load_surrogate_fidelity():
    """Load surrogate fidelity summary."""
    path = STUDY / "surrogate_fidelity" / "fidelity_summary.json"
    with open(path) as f:
        return json.load(f)


def load_parameter_recovery():
    """Load parameter recovery summary."""
    path = STUDY / "inverse_verification" / "parameter_recovery_summary.json"
    with open(path) as f:
        return json.load(f)


def load_gradient_fd_convergence():
    """Load surrogate gradient FD convergence data."""
    path = STUDY / "inverse_verification" / "gradient_fd_convergence.json"
    with open(path) as f:
        return json.load(f)


def load_gradient_pde_consistency():
    """Load PDE gradient consistency data."""
    path = STUDY / "inverse_verification" / "gradient_pde_consistency.json"
    with open(path) as f:
        return json.load(f)


def load_multistart_basin():
    """Load multistart basin data."""
    path = STUDY / "inverse_verification" / "multistart_basin.json"
    with open(path) as f:
        return json.load(f)


def load_worst_case_iv():
    """Load worst-case I-V overlay CSV."""
    path = STUDY / "master_inference_v13" / "P2_pde_full_cathodic" / "multi_obs_fit.csv"
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def load_ranking_report():
    """Load surrogate ranking report from Phase 3 benchmark."""
    path = STUDY / "surrogate_fidelity" / "ranking_report.json"
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Figure generators
# ---------------------------------------------------------------------------

FIELD_COLORS = {
    "c0": "#1f77b4",  # O2
    "c1": "#ff7f0e",  # H2O2
    "c2": "#2ca02c",  # H+
    "c3": "#d62728",  # ClO4-
    "phi": "#9467bd",  # phi
}

FIELD_MARKERS = {
    "c0": "o",
    "c1": "s",
    "c2": "^",
    "c3": "D",
    "phi": "v",
}


def make_mms_convergence_figure(data):
    """Figure 1: MMS convergence (two-panel log-log)."""
    h_values = np.array(data["metadata"]["h_values"])
    fields = data["fields"]

    fig, (ax_l2, ax_h1) = plt.subplots(1, 2, figsize=(12, 5))

    for key in ["c0", "c1", "c2", "c3", "phi"]:
        fdata = fields[key]
        label_name = fdata["label"]
        color = FIELD_COLORS[key]
        marker = FIELD_MARKERS[key]

        # L2 panel
        l2_err = np.array(fdata["L2_errors"])
        l2_rate = fdata["L2_rate"]
        ax_l2.loglog(
            h_values, l2_err,
            marker=marker, color=color, linewidth=1.5, markersize=6,
            label=f"{label_name} (rate={l2_rate:.2f})",
        )

        # H1 panel
        h1_err = np.array(fdata["H1_errors"])
        h1_rate = fdata["H1_rate"]
        ax_h1.loglog(
            h_values, h1_err,
            marker=marker, color=color, linewidth=1.5, markersize=6,
            label=f"{label_name} (rate={h1_rate:.2f})",
        )

    # Reference slopes
    h_ref = np.array([h_values[0], h_values[-1]])

    # O(h^2) reference on L2 panel
    scale_l2 = 0.5 * float(np.max([np.max(fields[k]["L2_errors"]) for k in fields]))
    ref_l2 = scale_l2 * (h_ref / h_ref[0]) ** 2
    ax_l2.loglog(h_ref, ref_l2, "k--", linewidth=1.0, alpha=0.6, label=r"$O(h^2)$ ref")

    # O(h) reference on H1 panel
    scale_h1 = 0.5 * float(np.max([np.max(fields[k]["H1_errors"]) for k in fields]))
    ref_h1 = scale_h1 * (h_ref / h_ref[0]) ** 1
    ax_h1.loglog(h_ref, ref_h1, "k--", linewidth=1.0, alpha=0.6, label=r"$O(h)$ ref")

    for ax, ylabel, title in [
        (ax_l2, r"$L^2$ error", r"$L^2$ Error Convergence"),
        (ax_h1, r"$H^1$ error", r"$H^1$ Error Convergence"),
    ]:
        ax.set_xlabel(r"Mesh spacing $h$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = FIGDIR / "mms_convergence.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def make_parameter_recovery_figure(data):
    """Figure 2: Parameter recovery median max error vs noise level."""
    results = data["results"]
    noise_levels = []
    median_errors = []
    gate_thresholds = []
    informational_flags = []

    for noise_str in ["0.0", "1.0", "2.0", "5.0"]:
        r = results[noise_str]
        noise_levels.append(float(noise_str))
        median_errors.append(r["median_max_relative_error"])
        gate_thresholds.append(r["gate_threshold"])
        informational_flags.append(r.get("informational", False))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    # Use evenly-spaced categorical positions so 0/1/2/5 don't distort spacing
    x_positions = list(range(len(noise_levels)))
    x_labels = ["0%", "1%", "2%", "5%"]

    # Plot markers
    for i, (xp, me, info) in enumerate(zip(x_positions, median_errors, informational_flags)):
        marker_style = "X" if info else "o"
        color = "gray" if info else "#1f77b4"
        ax.plot(xp, me, marker=marker_style, markersize=10, color=color,
                zorder=5, linestyle="none")
        if info:
            ax.annotate("informational", (xp, me),
                        textcoords="offset points", xytext=(10, -12),
                        fontsize=8, fontstyle="italic", color="gray")

    # Connect non-informational points with a line
    non_info_x = [xp for xp, inf in zip(x_positions, informational_flags) if not inf]
    non_info_y = [me for me, inf in zip(median_errors, informational_flags) if not inf]
    ax.plot(non_info_x, non_info_y, "-", color="#1f77b4", linewidth=1.2, alpha=0.6)

    # Gate threshold lines — span only the relevant x range
    unique_gates = sorted(set(gate_thresholds))
    gate_colors = ["#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]
    for j, gt in enumerate(unique_gates):
        ax.axhline(y=gt, linestyle="--", linewidth=1.0, alpha=0.5,
                   color=gate_colors[j % len(gate_colors)],
                   label=f"Gate = {gt:.2f}")

    # Surrogate bias floor annotation
    bias = data.get("surrogate_bias", 0.107)
    ax.axhline(y=bias, linestyle=":", linewidth=1.0, alpha=0.5, color="black",
               label=f"Surrogate bias floor ({bias:.1%})")

    ax.set_xlabel("Noise level")
    ax.set_ylabel("Median max relative error")
    ax.set_title("Parameter Recovery vs Noise Level")
    ax.set_yscale("log")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = FIGDIR / "parameter_recovery.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def make_worst_case_iv_figure(iv_rows):
    """Figure 3: Worst-case I-V overlay."""
    phi = [r["phi_applied"] for r in iv_rows]
    target_pri = [r["target_primary"] for r in iv_rows]
    sim_pri = [r["simulated_primary"] for r in iv_rows]
    target_sec = [r["target_secondary"] for r in iv_rows]
    sim_sec = [r["simulated_secondary"] for r in iv_rows]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    ax.plot(phi, target_pri, "o-", color="#1f77b4", markersize=5, linewidth=1.5,
            label="Total Current Density (PDE target)")
    ax.plot(phi, sim_pri, "x--", color="#1f77b4", markersize=6, linewidth=1.2,
            label="Total Current Density (Surrogate fit)")
    ax.plot(phi, target_sec, "o-", color="#ff7f0e", markersize=5, linewidth=1.5,
            label="Peroxide Current Density (PDE target)")
    ax.plot(phi, sim_sec, "x--", color="#ff7f0e", markersize=6, linewidth=1.2,
            label="Peroxide Current Density (Surrogate fit)")

    ax.set_xlabel(r"Applied potential $\phi_{\mathrm{applied}}$ (V)")
    ax.set_ylabel(r"Current density (A/cm$^2$)")
    ax.set_title("Worst-Case I--V Overlay (P2 Full Cathodic)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = FIGDIR / "worst_case_iv.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Table generators
# ---------------------------------------------------------------------------

def make_mms_rates_table(data):
    """Table 1: MMS convergence rates."""
    fields = data["fields"]
    lines = [
        r"\begin{tabular}{l S S S S S}",
        r"\toprule",
        r"Species & {$L^2$ Rate} & {$L^2$ $R^2$} & {$H^1$ Rate} & {$H^1$ $R^2$} & {GCI (finest)} \\",
        r"\midrule",
    ]

    for key in ["c0", "c1", "c2", "c3", "phi"]:
        f = fields[key]
        label = f["label"]
        gci_finest = f["gci"][-1]["gci"]
        lines.append(
            f"{label} & {f['L2_rate']:.3f} & {f['L2_r_squared']:.7f} "
            f"& {f['H1_rate']:.3f} & {f['H1_r_squared']:.7f} "
            f"& {gci_finest:.3f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    out = TABDIR / "mms_rates.tex"
    out.write_text("\n".join(lines) + "\n")
    return out


def make_surrogate_fidelity_table(data):
    """Table 2: Surrogate fidelity NRMSE statistics (all 6 models)."""
    models = data["models"]
    name_map = {
        "nn_ensemble": "NN Ensemble",
        "rbf_baseline": "RBF Baseline",
        "pod_rbf_log": "POD-RBF (log)",
        "pod_rbf_nolog": "POD-RBF (no-log)",
        "gp_fixed": "GP (GPyTorch)",
        "pce": "PCE (ChaosPy)",
    }

    # Ordered list of model keys to display
    model_order = [
        "nn_ensemble", "rbf_baseline", "pod_rbf_log",
        "pod_rbf_nolog", "gp_fixed", "pce",
    ]

    lines = [
        r"\begin{tabular}{l S S S S S}",
        r"\toprule",
        r"Model & {CD Median} & {CD 95th} & {CD Max} & {PC Median} & {PC 95th} \\",
        r"\midrule",
    ]

    for mkey in model_order:
        if mkey not in models:
            continue
        m = models[mkey]
        name = name_map.get(mkey, mkey)
        lines.append(
            f"{name} & {m['cd_median_nrmse']:.4f} & {m['cd_95th_nrmse']:.4f} "
            f"& {m['cd_max_nrmse']:.4f} & {m['pc_median_nrmse']:.4f} "
            f"& {m['pc_95th_nrmse']:.4f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\par\smallskip")
    lines.append(r"{\footnotesize Note: PC NRMSE inflated by near-zero-range samples; median is the robust statistic.}")

    out = TABDIR / "surrogate_fidelity.tex"
    out.write_text("\n".join(lines) + "\n")
    return out


def make_parameter_recovery_table(data):
    """Table 3: Parameter recovery per noise level."""
    results = data["results"]
    bias = data.get("surrogate_bias", 0.107)

    lines = [
        r"\begin{tabular}{l S S l}",
        r"\toprule",
        r"Noise Level & {Median Max Error} & {Gate Threshold} & Status \\",
        r"\midrule",
    ]

    for noise_str, noise_label in [("0.0", "0\\%"), ("1.0", "1\\%"),
                                    ("2.0", "2\\%"), ("5.0", "5\\%")]:
        r = results[noise_str]
        status = "Pass" if r["pass"] else "Fail"
        if r.get("informational", False):
            row = (
                f"{noise_label}\\textsuperscript{{a}} "
                f"& {{{r['median_max_relative_error']:.4f}}} "
                f"& {{{r['gate_threshold']:.2f}}} & {status}"
            )
        else:
            row = (
                f"{noise_label} & {r['median_max_relative_error']:.4f} "
                f"& {r['gate_threshold']:.2f} & {status}"
            )
        lines.append(row + r" \\")

    lines.append(r"\bottomrule")
    lines.append(
        r"\multicolumn{4}{l}{\footnotesize\textsuperscript{a} "
        r"Informational --- exceeds surrogate approximation limits.} \\"
    )
    bias_pct = bias * 100
    lines.append(
        f"\\multicolumn{{4}}{{l}}{{\\footnotesize "
        f"Surrogate bias floor at 0\\% noise: {bias_pct:.1f}\\% relative error.}} \\\\"
    )
    lines.append(r"\end{tabular}")

    out = TABDIR / "parameter_recovery.tex"
    out.write_text("\n".join(lines) + "\n")
    return out


def make_gradient_consistency_table(fd_data, pde_data):
    """Table 4: Gradient consistency (surrogate FD + PDE FD)."""
    lines = [
        r"\begin{tabular}{l l S}",
        r"\toprule",
        r"\multicolumn{3}{l}{\textbf{Surrogate FD Convergence}} \\",
        r"\midrule",
        r"& Parameter & {FD Conv.\ Rate} \\",
        r"\cmidrule{2-3}",
    ]

    surr_rates = fd_data["convergence_rates"]
    for param, rate in surr_rates.items():
        lines.append(f"& {param.replace('_', r'\_')} & {rate:.3f} \\\\")

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{3}{l}{\textbf{PDE FD Convergence (+10\% point)}} \\")
    lines.append(r"\midrule")
    lines.append(r"& Parameter & {FD Conv.\ Rate} \\")
    lines.append(r"\cmidrule{2-3}")

    pde_plus10 = pde_data["evaluation_points"]["+10%"]["convergence_rates"]
    for param, rate in pde_plus10.items():
        lines.append(f"& {param.replace('_', r'\_')} & {rate:.3f} \\\\")

    lines.append(r"\midrule")
    lines.append(
        r"\multicolumn{3}{l}{\footnotesize "
        r"Analytic vs FD relative diff = 0.0 for all params (exact match at $h=10^{-3}$).} \\"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    out = TABDIR / "gradient_consistency.tex"
    out.write_text("\n".join(lines) + "\n")
    return out


def make_surrogate_comparison_figure(ranking_data, fidelity_data):
    """Figure 4: Multi-panel surrogate comparison across all 6 models.

    Panel A (top-left): Grouped bar chart of CD and PC median NRMSE.
    Panel B (top-right): Grouped bar chart of 5 normalized dimension scores (top 3 models).
    Panel C (bottom): Horizontal bar chart of composite scores, sorted best-to-worst.
    """
    models_data = ranking_data["models"]
    # Sort by rank
    ranked_keys = sorted(models_data.keys(), key=lambda m: models_data[m]["rank"])
    display_names = [models_data[m]["display_name"] for m in ranked_keys]
    short_names = [
        n.replace(" (GPyTorch)", "").replace(" (ChaosPy)", "").replace(" (D1)", "")
        for n in display_names
    ]

    # Color palette: highlight selected model(s)
    primary = ranking_data["recommendation"]["primary_surrogate"]
    colors = []
    for m in ranked_keys:
        if m == primary:
            colors.append("#2ca02c")  # green for selected
        else:
            colors.append("#1f77b4")  # blue for others

    fig = plt.figure(figsize=(13, 9))

    # --- Panel A: CD + PC median NRMSE grouped bar ---
    ax_a = fig.add_subplot(2, 2, 1)
    x = np.arange(len(ranked_keys))
    width = 0.35

    cd_vals = [models_data[m]["prediction_accuracy"]["cd_median_nrmse"] for m in ranked_keys]
    pc_vals = [models_data[m]["prediction_accuracy"]["pc_median_nrmse"] for m in ranked_keys]

    bars_cd = ax_a.bar(x - width / 2, cd_vals, width, label="CD Median NRMSE",
                       color="#1f77b4", alpha=0.8)
    bars_pc = ax_a.bar(x + width / 2, pc_vals, width, label="PC Median NRMSE",
                       color="#ff7f0e", alpha=0.8)

    ax_a.set_yscale("log")
    ax_a.set_xlabel("Model")
    ax_a.set_ylabel("Median NRMSE")
    ax_a.set_title("(A) Prediction Accuracy")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(short_names, rotation=30, ha="right", fontsize=8)
    ax_a.legend(fontsize=8, loc="upper right")
    ax_a.grid(True, alpha=0.3, axis="y")

    # --- Panel B: Dimension scores for top 3 models (grouped bar) ---
    ax_b = fig.add_subplot(2, 2, 2)
    top3_keys = ranked_keys[:3]
    top3_names = short_names[:3]
    dimensions = [
        ("Inverse\nRecovery", "inverse_recovery"),
        ("Prediction\nAccuracy", "prediction_accuracy"),
        ("k02\nPerformance", "k02_performance"),
        ("Speed", "speed"),
        ("Gradient\nQuality", "gradient_quality"),
    ]

    x_dims = np.arange(len(dimensions))
    n_top = len(top3_keys)
    bar_w = 0.25
    top3_colors = ["#2ca02c", "#1f77b4", "#ff7f0e"]
    hatches = ["", "//", ".."]

    for i, mkey in enumerate(top3_keys):
        scores = [models_data[mkey][dim_key]["score_norm"] for _, dim_key in dimensions]
        ax_b.bar(
            x_dims + (i - 1) * bar_w, scores, bar_w,
            label=top3_names[i], color=top3_colors[i], alpha=0.8,
            hatch=hatches[i], edgecolor="white", linewidth=0.5,
        )

    ax_b.set_xlabel("Dimension")
    ax_b.set_ylabel("Normalized Score (0=best, 1=worst)")
    ax_b.set_title("(B) Top-3 Model Profiles")
    ax_b.set_xticks(x_dims)
    ax_b.set_xticklabels([d[0] for d in dimensions], fontsize=8)
    ax_b.legend(fontsize=8, loc="upper right")
    ax_b.set_ylim(0, 1.1)
    ax_b.grid(True, alpha=0.3, axis="y")

    # --- Panel C: Composite scores horizontal bar (bottom, spanning full width) ---
    ax_c = fig.add_subplot(2, 1, 2)
    composite_scores = [models_data[m]["composite_score"] for m in ranked_keys]
    # Reverse for horizontal bar (best at top)
    y_pos = np.arange(len(ranked_keys))

    bars = ax_c.barh(y_pos, composite_scores, color=colors, alpha=0.85, edgecolor="gray")

    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels(
        [f"#{models_data[m]['rank']}  {short_names[i]}" for i, m in enumerate(ranked_keys)],
        fontsize=9,
    )
    ax_c.set_xlabel("Composite Score (lower = better)")
    ax_c.set_title("(C) Overall Ranking")
    ax_c.grid(True, alpha=0.3, axis="x")
    ax_c.invert_yaxis()

    # Annotate bars with score values
    for bar, score in zip(bars, composite_scores):
        ax_c.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}", va="center", fontsize=8,
        )

    fig.tight_layout()
    out = FIGDIR / "surrogate_comparison.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def make_summary_table():
    """Table 5: Synthesis summary across all V&V layers."""
    rows = [
        ("Forward Solver", "MMS Convergence",
         r"$L^2 \sim O(h^2)$, $H^1 \sim O(h)$, $R^2 > 0.999$", "Pass"),
        ("Forward Solver", "GCI Uncertainty",
         r"Safety factor $\sim 1.25$ at finest mesh", "Pass"),
        ("Surrogate", "Hold-out Validation",
         r"CD median NRMSE $< 0.013$ (best of 6 models)", "Pass"),
        ("Inverse", "Parameter Recovery (0--2\\% noise)",
         "Median max error $<$ gate", "Pass"),
        ("Inverse", "Gradient Consistency",
         r"FD convergence $O(h^2)$ at +10\%", "Pass"),
        ("Inverse", "Multistart Basin",
         r"CV $< 0.002$, NRMSE $< 0.004$", "Pass"),
        ("Pipeline", "Reproducibility",
         "Bitwise-identical outputs", "Pass"),
    ]

    lines = [
        r"\begin{tabular}{l l l c}",
        r"\toprule",
        r"Layer & Test & Key Result & Status \\",
        r"\midrule",
    ]

    for layer, test, result, status in rows:
        lines.append(f"{layer} & {test} & {result} & {status} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    out = TABDIR / "summary.tex"
    out.write_text("\n".join(lines) + "\n")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Generate all figures and tables."""
    FIGDIR.mkdir(parents=True, exist_ok=True)
    TABDIR.mkdir(parents=True, exist_ok=True)

    generated = []
    errors = []

    # --- Figures ---
    try:
        mms_data = load_mms_convergence()
        p = make_mms_convergence_figure(mms_data)
        generated.append(str(p))
        print(f"  [OK] {p.name}")
    except Exception as e:
        errors.append(f"MMS convergence figure: {e}")
        print(f"  [FAIL] mms_convergence.pdf: {e}")

    try:
        pr_data = load_parameter_recovery()
        p = make_parameter_recovery_figure(pr_data)
        generated.append(str(p))
        print(f"  [OK] {p.name}")
    except Exception as e:
        errors.append(f"Parameter recovery figure: {e}")
        print(f"  [FAIL] parameter_recovery.pdf: {e}")

    try:
        iv_rows = load_worst_case_iv()
        p = make_worst_case_iv_figure(iv_rows)
        generated.append(str(p))
        print(f"  [OK] {p.name}")
    except Exception as e:
        errors.append(f"Worst-case IV figure: {e}")
        print(f"  [FAIL] worst_case_iv.pdf: {e}")

    try:
        ranking_data = load_ranking_report()
        sf_data_fig = load_surrogate_fidelity()
        p = make_surrogate_comparison_figure(ranking_data, sf_data_fig)
        generated.append(str(p))
        print(f"  [OK] {p.name}")
    except Exception as e:
        errors.append(f"Surrogate comparison figure: {e}")
        print(f"  [FAIL] surrogate_comparison.pdf: {e}")

    # --- Tables ---
    try:
        if "mms_data" not in dir():
            mms_data = load_mms_convergence()
        p = make_mms_rates_table(mms_data)
        generated.append(str(p))
        print(f"  [OK] {p.name}")
    except Exception as e:
        errors.append(f"MMS rates table: {e}")
        print(f"  [FAIL] mms_rates.tex: {e}")

    try:
        sf_data = load_surrogate_fidelity()
        p = make_surrogate_fidelity_table(sf_data)
        generated.append(str(p))
        print(f"  [OK] {p.name}")
    except Exception as e:
        errors.append(f"Surrogate fidelity table: {e}")
        print(f"  [FAIL] surrogate_fidelity.tex: {e}")

    try:
        if "pr_data" not in dir():
            pr_data = load_parameter_recovery()
        p = make_parameter_recovery_table(pr_data)
        generated.append(str(p))
        print(f"  [OK] {p.name}")
    except Exception as e:
        errors.append(f"Parameter recovery table: {e}")
        print(f"  [FAIL] parameter_recovery.tex: {e}")

    try:
        fd_data = load_gradient_fd_convergence()
        pde_data = load_gradient_pde_consistency()
        p = make_gradient_consistency_table(fd_data, pde_data)
        generated.append(str(p))
        print(f"  [OK] {p.name}")
    except Exception as e:
        errors.append(f"Gradient consistency table: {e}")
        print(f"  [FAIL] gradient_consistency.tex: {e}")

    try:
        p = make_summary_table()
        generated.append(str(p))
        print(f"  [OK] {p.name}")
    except Exception as e:
        errors.append(f"Summary table: {e}")
        print(f"  [FAIL] summary.tex: {e}")

    # --- Summary ---
    print(f"\nGenerated {len(generated)} files.")
    if errors:
        print(f"Errors ({len(errors)}):")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("All figures and tables generated successfully.")
        sys.exit(0)


if __name__ == "__main__":
    main()
