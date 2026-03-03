#!/usr/bin/env python3
"""Generate all figures for the Week of March 4 writeup.

Run from the PNPInverse/ root directory:
    python writeups/WeekOfMar4/generate_plots.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STUDY = os.path.join(ROOT, "StudyResults")
FIGDIR = os.path.join(ROOT, "writeups", "WeekOfMar4", "figures")
os.makedirs(FIGDIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.figsize": (6, 4),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": False,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
})

COLORS = plt.cm.tab10.colors

generated = []
skipped = []


def _safe_read(path):
    """Read a CSV, returning None with a warning if missing."""
    if not os.path.isfile(path):
        warnings.warn(f"Missing: {path}")
        return None
    return pd.read_csv(path)


# =========================================================================
# Plot 1: Forward I-V Curve Shape
# =========================================================================
def plot1_iv_curve_shape():
    path = os.path.join(STUDY, "bv_iv_curve_charged", "bv_iv_curve_charged.csv")
    df = _safe_read(path)
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4))
    eta = df["eta_hat"].values
    I_total = df["I_total_mA_cm2"].values

    ax.plot(eta, I_total, "k-", lw=1.8, label="Total current")

    # Annotate regions
    ax.axvspan(eta.min(), -2, alpha=0.07, color=COLORS[0], label="Onset")
    ax.axvspan(-2, -6, alpha=0.07, color=COLORS[1], label="Transition")
    ax.axvspan(-6, -12, alpha=0.07, color=COLORS[2], label="Knee")
    ax.axvspan(-12, eta.max(), alpha=0.07, color=COLORS[3], label="Plateau")

    ax.set_xlabel(r"$\hat\eta$ (dimensionless overpotential)")
    ax.set_ylabel(r"$I_{\mathrm{total}}$ (mA cm$^{-2}$)")
    ax.set_title("Forward I--V Curve: 4-Species Charged BV Model")
    ax.legend(loc="lower left", framealpha=0.9)

    # Secondary x-axis: V_RHE
    ax2 = ax.twiny()
    vrhe = df["V_RHE"].values
    ax2.set_xlim(ax.get_xlim())
    # Place a few ticks
    tick_etas = np.array([0, -5, -10, -15, -20, -30, -40])
    tick_etas = tick_etas[(tick_etas >= eta.min()) & (tick_etas <= eta.max())]
    # Map eta_hat -> V_RHE: V_RHE = 0.695 + eta_hat * (kT/e) ~ 0.695 + eta_hat*0.02569
    # But we have actual data, use linear interp
    ax2.set_xticks(tick_etas)
    ax2.set_xticklabels([f"{0.695 + e * 0.02569:.2f}" for e in tick_etas])
    ax2.set_xlabel(r"$V_{\mathrm{RHE}}$ (V)")

    plt.tight_layout()
    out = os.path.join(FIGDIR, "iv_curve_shape.pdf")
    fig.savefig(out)
    plt.close(fig)
    generated.append("iv_curve_shape.pdf")


# =========================================================================
# Plot 2: Regularized Inference -- Target vs Fit (10-point)
# =========================================================================
def plot2_regularized_fit_10pt():
    lam_dirs = ["lambda_0.0000", "lambda_0.0010"]
    lam_labels = [r"$\lambda=0$", r"$\lambda=10^{-3}$"]
    base = os.path.join(STUDY, "bv_joint_regularized_charged")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for i, (ld, ll) in enumerate(zip(lam_dirs, lam_labels)):
        fit = _safe_read(os.path.join(base, ld, "phi_applied_vs_current_density_fit.csv"))
        syn = _safe_read(os.path.join(base, ld, "phi_applied_vs_current_density_synthetic.csv"))
        if fit is None:
            continue
        ax = axes[i]
        ax.scatter(fit["phi_applied"], fit["target_current_density"],
                   c=COLORS[0], s=40, zorder=5, label="Target (noisy)", edgecolors="k", linewidths=0.4)
        ax.plot(fit["phi_applied"], fit["simulated_current_density"],
                "-", c=COLORS[1], lw=2, label="Fit")
        if syn is not None:
            ax.plot(syn["phi_applied"], syn["flux_clean"],
                    "--", c="gray", lw=1, alpha=0.6, label="Clean truth")
        ax.set_xlabel(r"$\hat\eta$")
        ax.set_title(f"Regularized Joint, {ll}")
        ax.legend(fontsize=8)
    axes[0].set_ylabel(r"Current density (dimless)")
    plt.tight_layout()
    out = os.path.join(FIGDIR, "regularized_fit_10pt.pdf")
    fig.savefig(out)
    plt.close(fig)
    generated.append("regularized_fit_10pt.pdf")


# =========================================================================
# Plot 3: Staged Inference -- Stage Comparison (10-point)
# =========================================================================
def plot3_staged_fit_10pt():
    base = os.path.join(STUDY, "bv_staged_inference_charged")
    stages = [
        ("stage1_alpha", "S1: Alpha only"),
        ("stage2_k0", "S2: k0 only"),
        ("stage3_joint_warmstart", "S3: Joint warm-start"),
        ("stage4_direct_joint", "S4: Direct joint"),
    ]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    # Plot target from stage1
    syn = _safe_read(os.path.join(base, "stage1_alpha", "target.csv"))
    if syn is not None:
        ax.scatter(syn["phi_applied"], syn["flux_noisy"],
                   c="k", s=50, zorder=10, marker="x", label="Target (noisy)", linewidths=1.2)
    for i, (sd, sl) in enumerate(stages):
        fit = _safe_read(os.path.join(base, sd, "phi_applied_vs_current_density_fit.csv"))
        if fit is None:
            continue
        ax.plot(fit["phi_applied"], fit["simulated_current_density"],
                "-o", c=COLORS[i], lw=1.5, ms=4, label=sl)
    ax.set_xlabel(r"$\hat\eta$")
    ax.set_ylabel(r"Current density (dimless)")
    ax.set_title("Staged Inference: 10-Point Range")
    ax.legend(fontsize=8, loc="lower left")
    plt.tight_layout()
    out = os.path.join(FIGDIR, "staged_fit_10pt.pdf")
    fig.savefig(out)
    plt.close(fig)
    generated.append("staged_fit_10pt.pdf")


# =========================================================================
# Plot 4: Peroxide Current -- Target vs Fit
# =========================================================================
def plot4_peroxide_fit_10pt():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    configs = [
        ("bv_k0_peroxide_current_charged", r"$k_0$-only (peroxide obs.)"),
        ("bv_joint_peroxide_current_charged", r"Joint $(k_0,\alpha)$ (peroxide obs.)"),
    ]
    for i, (dirname, title) in enumerate(configs):
        fit = _safe_read(os.path.join(STUDY, dirname, "phi_applied_vs_current_density_fit.csv"))
        syn = _safe_read(os.path.join(STUDY, dirname, "phi_applied_vs_peroxide_current_synthetic.csv"))
        if fit is None:
            continue
        ax = axes[i]
        ax.scatter(fit["phi_applied"], fit["target_current_density"],
                   c=COLORS[0], s=40, zorder=5, label="Target (noisy)", edgecolors="k", linewidths=0.4)
        ax.plot(fit["phi_applied"], fit["simulated_current_density"],
                "-", c=COLORS[1], lw=2, label="Fit")
        if syn is not None:
            ax.plot(syn["phi_applied"], syn["flux_clean"],
                    "--", c="gray", lw=1, alpha=0.6, label="Clean truth")
        ax.set_xlabel(r"$\hat\eta$")
        ax.set_title(title)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(r"Current density (dimless)")
    plt.tight_layout()
    out = os.path.join(FIGDIR, "peroxide_fit_10pt.pdf")
    fig.savefig(out)
    plt.close(fig)
    generated.append("peroxide_fit_10pt.pdf")


# =========================================================================
# Plot 5: Extended Range -- Staged Inference (15-point)
# =========================================================================
def plot5_staged_fit_15pt():
    base = os.path.join(STUDY, "bv_staged_inference_charged_extended")
    stages = [
        ("stage1_alpha", "S1: Alpha only"),
        ("stage2_k0", "S2: k0 only"),
        ("stage3_joint_warmstart", "S3: Joint warm-start"),
        ("stage4_direct_joint", "S4: Direct joint"),
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    syn = _safe_read(os.path.join(base, "stage1_alpha", "target.csv"))
    if syn is not None:
        ax.scatter(syn["phi_applied"], syn["flux_noisy"],
                   c="k", s=50, zorder=10, marker="x", label="Target (noisy)", linewidths=1.2)

    for i, (sd, sl) in enumerate(stages):
        fit = _safe_read(os.path.join(base, sd, "phi_applied_vs_current_density_fit.csv"))
        if fit is None:
            continue
        ax.plot(fit["phi_applied"], fit["simulated_current_density"],
                "-o", c=COLORS[i], lw=1.5, ms=4, label=sl)

    # Shade bridge region
    ax.axvspan(-46.5, -13, alpha=0.06, color="gray", label=r"Bridge region ($\hat\eta<-13$)")

    ax.set_xlabel(r"$\hat\eta$")
    ax.set_ylabel(r"Current density (dimless)")
    ax.set_title("Staged Inference: 15-Point Extended Range")
    ax.legend(fontsize=8, loc="lower left")
    plt.tight_layout()
    out = os.path.join(FIGDIR, "staged_fit_15pt.pdf")
    fig.savefig(out)
    plt.close(fig)
    generated.append("staged_fit_15pt.pdf")


# =========================================================================
# Plot 6: Extended Range -- Regularized (15-point)
# =========================================================================
def plot6_regularized_fit_15pt():
    lam_dirs = ["lambda_0.0000", "lambda_0.0010"]
    lam_labels = [r"$\lambda=0$", r"$\lambda=10^{-3}$"]
    base = os.path.join(STUDY, "bv_joint_regularized_charged_extended")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for i, (ld, ll) in enumerate(zip(lam_dirs, lam_labels)):
        fit = _safe_read(os.path.join(base, ld, "phi_applied_vs_current_density_fit.csv"))
        syn = _safe_read(os.path.join(base, ld, "phi_applied_vs_current_density_synthetic.csv"))
        if fit is None:
            continue
        ax = axes[i]
        ax.scatter(fit["phi_applied"], fit["target_current_density"],
                   c=COLORS[0], s=40, zorder=5, label="Target (noisy)", edgecolors="k", linewidths=0.4)
        ax.plot(fit["phi_applied"], fit["simulated_current_density"],
                "-", c=COLORS[1], lw=2, label="Fit")
        if syn is not None:
            ax.plot(syn["phi_applied"], syn["flux_clean"],
                    "--", c="gray", lw=1, alpha=0.6, label="Clean truth")
        ax.set_xlabel(r"$\hat\eta$")
        ax.set_title(f"Regularized Joint (Extended), {ll}")
        ax.legend(fontsize=8)
    axes[0].set_ylabel(r"Current density (dimless)")
    plt.tight_layout()
    out = os.path.join(FIGDIR, "regularized_fit_15pt.pdf")
    fig.savefig(out)
    plt.close(fig)
    generated.append("regularized_fit_15pt.pdf")


# =========================================================================
# Plot 7: Steric Inference -- Target vs Fit (10pt and 15pt)
# =========================================================================
def plot7_steric_fit_comparison():
    configs = [
        ("bv_steric_charged", "Steric-Only: 10-Point"),
        ("bv_steric_charged_extended", "Steric-Only: 15-Point"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for i, (dirname, title) in enumerate(configs):
        fit = _safe_read(os.path.join(STUDY, dirname, "phi_applied_vs_current_density_fit.csv"))
        syn = _safe_read(os.path.join(STUDY, dirname, "phi_applied_vs_current_density_synthetic.csv"))
        if fit is None:
            continue
        ax = axes[i]
        ax.scatter(fit["phi_applied"], fit["target_current_density"],
                   c=COLORS[0], s=40, zorder=5, label="Target (noisy)", edgecolors="k", linewidths=0.4)
        ax.plot(fit["phi_applied"], fit["simulated_current_density"],
                "-", c=COLORS[1], lw=2, label="Fit")
        if syn is not None:
            ax.plot(syn["phi_applied"], syn["flux_clean"],
                    "--", c="gray", lw=1, alpha=0.6, label="Clean truth")
        ax.set_xlabel(r"$\hat\eta$")
        ax.set_title(title)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(r"Current density (dimless)")
    plt.tight_layout()
    out = os.path.join(FIGDIR, "steric_fit_comparison.pdf")
    fig.savefig(out)
    plt.close(fig)
    generated.append("steric_fit_comparison.pdf")


# =========================================================================
# Plot 8: Full (8-param) Inference -- Target vs Fit
# =========================================================================
def plot8_full_fit_comparison():
    configs = [
        ("bv_full_charged", "Full (8-param): 10-Point"),
        ("bv_full_charged_extended", "Full (8-param): 15-Point"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for i, (dirname, title) in enumerate(configs):
        fit = _safe_read(os.path.join(STUDY, dirname, "phi_applied_vs_current_density_fit.csv"))
        syn = _safe_read(os.path.join(STUDY, dirname, "phi_applied_vs_current_density_synthetic.csv"))
        if fit is None:
            continue
        ax = axes[i]
        ax.scatter(fit["phi_applied"], fit["target_current_density"],
                   c=COLORS[0], s=40, zorder=5, label="Target (noisy)", edgecolors="k", linewidths=0.4)
        ax.plot(fit["phi_applied"], fit["simulated_current_density"],
                "-", c=COLORS[1], lw=2, label="Fit")
        if syn is not None:
            ax.plot(syn["phi_applied"], syn["flux_clean"],
                    "--", c="gray", lw=1, alpha=0.6, label="Clean truth")
        ax.set_xlabel(r"$\hat\eta$")
        ax.set_title(title)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(r"Current density (dimless)")
    plt.tight_layout()
    out = os.path.join(FIGDIR, "full_fit_comparison.pdf")
    fig.savefig(out)
    plt.close(fig)
    generated.append("full_fit_comparison.pdf")


# =========================================================================
# Plot 9: Parameter Recovery Bar Chart -- R1 Parameters
# =========================================================================
def plot9_r1_recovery():
    # Hardcoded from phase3/5 summaries
    methods = [
        "Reg. ($\\lambda$=0)",
        "Reg. ($\\lambda$=1e-3)",
        "Staged S2/S3",
        "Direct Joint",
        "k0 Peroxide",
        "Joint Peroxide",
    ]
    k0_1_10pt  = [199.5, 50.2, 19.2, 221.3, 5.4,  18.3]
    k0_1_15pt  = [210.6, 51.1,  4.9, 210.8, 4.1, 280.2]
    al_1_10pt  = [74.4,  21.7,  5.8,  75.4, np.nan,  5.4]
    al_1_15pt  = [72.3,  22.3,  0.1,  72.3, np.nan, 10.8]

    x = np.arange(len(methods))
    w = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # k0_1
    ax = axes[0]
    ax.bar(x - w, k0_1_10pt, 2*w, color=COLORS[0], alpha=0.8, label="10-pt")
    ax.bar(x + w, k0_1_15pt, 2*w, color=COLORS[1], alpha=0.8, label="15-pt")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Relative Error (%)")
    ax.set_title(r"$k_{0,1}$ Recovery Error")
    ax.legend()
    ax.set_ylim(0, max(max(k0_1_10pt), max(k0_1_15pt)) * 1.15)

    # alpha_1
    ax = axes[1]
    al_1_10pt_plot = [v if not np.isnan(v) else 0 for v in al_1_10pt]
    al_1_15pt_plot = [v if not np.isnan(v) else 0 for v in al_1_15pt]
    bars1 = ax.bar(x - w, al_1_10pt_plot, 2*w, color=COLORS[0], alpha=0.8, label="10-pt")
    bars2 = ax.bar(x + w, al_1_15pt_plot, 2*w, color=COLORS[1], alpha=0.8, label="15-pt")
    # Mark N/A bars
    for j in range(len(methods)):
        if np.isnan(al_1_10pt[j]):
            ax.text(x[j] - w, 1, "N/A", ha="center", va="bottom", fontsize=7, color="gray")
        if np.isnan(al_1_15pt[j]):
            ax.text(x[j] + w, 1, "N/A", ha="center", va="bottom", fontsize=7, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Relative Error (%)")
    ax.set_title(r"$\alpha_1$ Recovery Error")
    ax.legend()
    ax.set_ylim(0, max(max(al_1_10pt_plot), max(al_1_15pt_plot)) * 1.15)

    plt.tight_layout()
    out = os.path.join(FIGDIR, "r1_recovery_comparison.pdf")
    fig.savefig(out)
    plt.close(fig)
    generated.append("r1_recovery_comparison.pdf")


# =========================================================================
# Plot 10: Parameter Recovery Bar Chart -- R2 Parameters
# =========================================================================
def plot10_r2_recovery():
    methods = [
        "Reg. ($\\lambda$=0)",
        "Reg. ($\\lambda$=1e-3)",
        "Staged S2/S3",
        "Direct Joint",
        "k0 Peroxide",
        "Joint Peroxide",
    ]
    k0_2_10pt  = [377.3, 100.7, 96.9, 270.2, 99.98, 96.5]
    k0_2_15pt  = [139.6, 100.6, 98.5, 137.0, 99.98, 760.0]
    al_2_10pt  = [76.9, 19.4, 66.9, 90.0, np.nan, 90.0]
    al_2_15pt  = [90.0, 19.4, 72.3, 90.0, np.nan, 84.0]

    x = np.arange(len(methods))
    w = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.bar(x - w, k0_2_10pt, 2*w, color=COLORS[0], alpha=0.8, label="10-pt")
    ax.bar(x + w, k0_2_15pt, 2*w, color=COLORS[1], alpha=0.8, label="15-pt")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Relative Error (%)")
    ax.set_title(r"$k_{0,2}$ Recovery Error")
    ax.legend()
    ax.set_ylim(0, max(max(k0_2_10pt), max(k0_2_15pt)) * 1.15)

    ax = axes[1]
    al_2_10pt_plot = [v if not np.isnan(v) else 0 for v in al_2_10pt]
    al_2_15pt_plot = [v if not np.isnan(v) else 0 for v in al_2_15pt]
    ax.bar(x - w, al_2_10pt_plot, 2*w, color=COLORS[0], alpha=0.8, label="10-pt")
    ax.bar(x + w, al_2_15pt_plot, 2*w, color=COLORS[1], alpha=0.8, label="15-pt")
    for j in range(len(methods)):
        if np.isnan(al_2_10pt[j]):
            ax.text(x[j] - w, 1, "N/A", ha="center", va="bottom", fontsize=7, color="gray")
        if np.isnan(al_2_15pt[j]):
            ax.text(x[j] + w, 1, "N/A", ha="center", va="bottom", fontsize=7, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Relative Error (%)")
    ax.set_title(r"$\alpha_2$ Recovery Error")
    ax.legend()
    ax.set_ylim(0, max(max(al_2_10pt_plot), max(al_2_15pt_plot)) * 1.15)

    plt.tight_layout()
    out = os.path.join(FIGDIR, "r2_recovery_comparison.pdf")
    fig.savefig(out)
    plt.close(fig)
    generated.append("r2_recovery_comparison.pdf")


# =========================================================================
# Plot 11: Steric Parameter Recovery Bar Chart
# =========================================================================
def plot11_steric_recovery():
    species = [r"$a_{\mathrm{O_2}}$", r"$a_{\mathrm{H_2O_2}}$",
               r"$a_{\mathrm{H^+}}$", r"$a_{\mathrm{ClO_4^-}}$"]
    # From phase4/5 summaries
    steric_10  = [82.4,  98.0,  23.1,  98.0]
    steric_15  = [200.0, 164.6, 57.1,  98.0]
    full_10    = [35.4,  80.7,  31.2,   6.6]
    full_15    = [71.3,  44.9,  84.1,  98.0]

    x = np.arange(len(species))
    w = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    ax = axes[0]
    ax.bar(x - w, steric_10, 2*w, color=COLORS[0], alpha=0.8, label="10-pt")
    ax.bar(x + w, steric_15, 2*w, color=COLORS[1], alpha=0.8, label="15-pt")
    ax.set_xticks(x)
    ax.set_xticklabels(species, fontsize=10)
    ax.set_ylabel("Relative Error (%)")
    ax.set_title("Steric-Only Inference")
    ax.legend()
    ax.set_ylim(0, 230)

    ax = axes[1]
    ax.bar(x - w, full_10, 2*w, color=COLORS[0], alpha=0.8, label="10-pt")
    ax.bar(x + w, full_15, 2*w, color=COLORS[1], alpha=0.8, label="15-pt")
    ax.set_xticks(x)
    ax.set_xticklabels(species, fontsize=10)
    ax.set_ylabel("Relative Error (%)")
    ax.set_title("Full (8-param) Inference")
    ax.legend()
    ax.set_ylim(0, 120)

    plt.tight_layout()
    out = os.path.join(FIGDIR, "steric_recovery_comparison.pdf")
    fig.savefig(out)
    plt.close(fig)
    generated.append("steric_recovery_comparison.pdf")


# =========================================================================
# Plot 12: Optimization Convergence -- Loss vs Iteration
# =========================================================================
def plot12_convergence_staged():
    fig, ax = plt.subplots(figsize=(6, 4))
    configs = [
        (os.path.join(STUDY, "bv_staged_inference_charged", "stage2_k0",
                       "bv_k0_optimization_history.csv"),
         "Staged S2 (10-pt)"),
        (os.path.join(STUDY, "bv_staged_inference_charged_extended", "stage2_k0",
                       "bv_k0_optimization_history.csv"),
         "Staged S2 (15-pt)"),
    ]
    for i, (path, label) in enumerate(configs):
        df = _safe_read(path)
        if df is None:
            continue
        # Get unique iterations
        iters = df.groupby("iteration").first().reset_index()
        ax.semilogy(iters["iteration"], iters["objective"],
                    "-o", c=COLORS[i], ms=4, label=label)
    ax.set_xlabel("L-BFGS-B Iteration")
    ax.set_ylabel("Objective (log scale)")
    ax.set_title("Convergence: Staged k0 Inference")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(FIGDIR, "convergence_staged.pdf")
    fig.savefig(out)
    plt.close(fig)
    generated.append("convergence_staged.pdf")


# =========================================================================
# Plot 13: Regularization Sensitivity -- Error vs Lambda
# =========================================================================
def plot13_regularization_sensitivity():
    lambdas = [0.0, 0.001, 0.01, 0.1, 1.0]
    # From phase3 summary
    k0_1_err_10 = [199.5, 50.2, 49.6, 49.9, 50.0]
    al_1_err_10 = [74.4,  21.7, 20.7, 20.3, 20.3]
    # From phase5 summary
    k0_1_err_15 = [210.6, 51.1, 49.6, 49.9, 50.0]
    al_1_err_15 = [72.3,  22.3, 20.8, 20.3, 20.3]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.semilogx([max(l, 1e-5) for l in lambdas], k0_1_err_10,
                "-o", c=COLORS[0], label="10-pt")
    ax.semilogx([max(l, 1e-5) for l in lambdas], k0_1_err_15,
                "-s", c=COLORS[1], label="15-pt")
    # Mark lambda=0 separately
    ax.plot(1e-5, k0_1_err_10[0], "o", c=COLORS[0], ms=8, mfc="none", mew=2)
    ax.plot(1e-5, k0_1_err_15[0], "s", c=COLORS[1], ms=8, mfc="none", mew=2)
    ax.annotate(r"$\lambda=0$", (1e-5, k0_1_err_10[0]), textcoords="offset points",
                xytext=(10, 5), fontsize=8)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Relative Error (%)")
    ax.set_title(r"$k_{0,1}$ Error vs $\lambda$")
    ax.legend()

    ax = axes[1]
    ax.semilogx([max(l, 1e-5) for l in lambdas], al_1_err_10,
                "-o", c=COLORS[0], label="10-pt")
    ax.semilogx([max(l, 1e-5) for l in lambdas], al_1_err_15,
                "-s", c=COLORS[1], label="15-pt")
    ax.plot(1e-5, al_1_err_10[0], "o", c=COLORS[0], ms=8, mfc="none", mew=2)
    ax.plot(1e-5, al_1_err_15[0], "s", c=COLORS[1], ms=8, mfc="none", mew=2)
    ax.annotate(r"$\lambda=0$", (1e-5, al_1_err_10[0]), textcoords="offset points",
                xytext=(10, 5), fontsize=8)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Relative Error (%)")
    ax.set_title(r"$\alpha_1$ Error vs $\lambda$")
    ax.legend()

    plt.tight_layout()
    out = os.path.join(FIGDIR, "regularization_sensitivity.pdf")
    fig.savefig(out)
    plt.close(fig)
    generated.append("regularization_sensitivity.pdf")


# =========================================================================
# Plot 14: Extended vs Original -- I-V Curve Overlay
# =========================================================================
def plot14_voltage_range_comparison():
    # 10-point target
    syn10 = _safe_read(os.path.join(STUDY, "bv_joint_regularized_charged",
                                     "lambda_0.0000", "phi_applied_vs_current_density_synthetic.csv"))
    # 15-point target
    syn15 = _safe_read(os.path.join(STUDY, "bv_joint_regularized_charged_extended",
                                     "lambda_0.0000", "phi_applied_vs_current_density_synthetic.csv"))
    if syn10 is None and syn15 is None:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))

    if syn10 is not None:
        ax.scatter(syn10["phi_applied"], syn10["flux_noisy"],
                   c=COLORS[0], s=60, zorder=6, marker="o", label="10-pt (inference)",
                   edgecolors="k", linewidths=0.4)
        ax.plot(syn10["phi_applied"], syn10["flux_clean"],
                "--", c=COLORS[0], lw=1, alpha=0.5)

    if syn15 is not None:
        # Mark points that are also in 10pt vs new
        etas_10 = set(syn10["phi_applied"].values) if syn10 is not None else set()
        etas_15 = syn15["phi_applied"].values
        mask_shared = np.array([e in etas_10 for e in etas_15])
        mask_new = ~mask_shared

        ax.scatter(etas_15[mask_new], syn15["flux_noisy"].values[mask_new],
                   c=COLORS[1], s=60, zorder=6, marker="s", label="Extended (new points)",
                   edgecolors="k", linewidths=0.4)
        ax.plot(syn15["phi_applied"], syn15["flux_clean"],
                "-", c=COLORS[1], lw=1.2, alpha=0.5, label="Clean truth (15-pt)")

    # Forward curve for reference
    fwd = _safe_read(os.path.join(STUDY, "bv_iv_curve_charged", "bv_iv_curve_charged.csv"))
    if fwd is not None:
        ax.plot(fwd["eta_hat"], fwd["I_total_mA_cm2"] / (-0.1837),  # approximate scale
                ":", c="gray", lw=0.8, alpha=0.4)

    # Shade bridge region
    ax.axvspan(-46.5, -13, alpha=0.06, color="gray")
    ax.annotate("Bridge region", xy=(-30, -0.18), fontsize=8, color="gray",
                ha="center")

    ax.set_xlabel(r"$\hat\eta$")
    ax.set_ylabel(r"Current density (dimless)")
    ax.set_title("Voltage Range: 10-Point vs 15-Point Extended")
    ax.legend(fontsize=8, loc="lower left")
    plt.tight_layout()
    out = os.path.join(FIGDIR, "voltage_range_comparison.pdf")
    fig.savefig(out)
    plt.close(fig)
    generated.append("voltage_range_comparison.pdf")


# =========================================================================
# Plot 15: Symmetric Voltage Range -- Target I-V Curve
# =========================================================================
def plot15_symmetric_target_iv():
    """Show the 20-point symmetric target alongside cathodic-only targets."""
    sym_target = _safe_read(os.path.join(
        STUDY, "bv_staged_inference_charged_symmetric", "stage1_alpha", "target.csv"))
    cat10_target = _safe_read(os.path.join(
        STUDY, "bv_staged_inference_charged", "stage1_alpha", "target.csv"))
    cat15_target = _safe_read(os.path.join(
        STUDY, "bv_staged_inference_charged_extended", "stage1_alpha", "target.csv"))

    fig, ax = plt.subplots(figsize=(7, 4.5))

    if cat10_target is not None:
        ax.scatter(cat10_target["phi_applied"], cat10_target["flux_noisy"],
                   c=COLORS[0], s=50, marker="o", zorder=5,
                   label="Cathodic 10-pt", edgecolors="k", linewidths=0.4)
    if cat15_target is not None:
        mask_new = ~cat15_target["phi_applied"].isin(
            cat10_target["phi_applied"] if cat10_target is not None else [])
        ax.scatter(cat15_target.loc[mask_new, "phi_applied"],
                   cat15_target.loc[mask_new, "flux_noisy"],
                   c=COLORS[1], s=50, marker="s", zorder=5,
                   label="Extended 15-pt (new)", edgecolors="k", linewidths=0.4)
    if sym_target is not None:
        eta = sym_target["phi_applied"].values
        flux = sym_target["flux_noisy"].values
        anodic = eta > 0
        near_eq = (eta >= -0.5) & (eta <= 0)
        cathodic = eta < -0.5
        # Only plot the anodic and near-equilibrium points that are unique
        mask_sym_new = anodic | near_eq
        finite_mask = np.isfinite(flux)
        mask_plot = mask_sym_new & finite_mask
        ax.scatter(eta[mask_plot], flux[mask_plot],
                   c=COLORS[3], s=70, marker="D", zorder=6,
                   label="Symmetric (anodic/eq.)", edgecolors="k", linewidths=0.5)
        # Mark NaN point (eta=+5)
        nan_mask = mask_sym_new & ~finite_mask
        if nan_mask.any():
            ax.scatter(eta[nan_mask], np.zeros(nan_mask.sum()),
                       c="red", s=80, marker="X", zorder=7,
                       label=r"Failed ($\hat\eta=+5$)")

    # Reference forward curve
    fwd = _safe_read(os.path.join(STUDY, "bv_iv_curve_charged", "bv_iv_curve_charged.csv"))
    if fwd is not None:
        ax.plot(fwd["eta_hat"], fwd["I_total_mA_cm2"] / (-0.1837),
                ":", c="gray", lw=0.8, alpha=0.4, label="Forward ref.")

    ax.axvline(0, color="gray", ls="--", lw=0.6, alpha=0.5)
    ax.axvspan(0, 6, alpha=0.05, color=COLORS[3])
    ax.annotate("Anodic", xy=(2.5, 0.003), fontsize=8, color=COLORS[3], ha="center")
    ax.axvspan(-46.5, -13, alpha=0.04, color="gray")
    ax.annotate("Bridge region", xy=(-30, -0.18), fontsize=8, color="gray", ha="center")

    ax.set_xlabel(r"$\hat\eta$ (dimensionless overpotential)")
    ax.set_ylabel(r"Current density (dimless)")
    ax.set_title("Symmetric Voltage Range: 20-Point Target I--V Curve")
    ax.legend(fontsize=7, loc="lower left", ncol=2)
    plt.tight_layout()
    out = os.path.join(FIGDIR, "symmetric_target_iv.pdf")
    fig.savefig(out)
    plt.close(fig)
    generated.append("symmetric_target_iv.pdf")


# =========================================================================
# Plot 16: Alpha_1 Recovery Across All Phases
# =========================================================================
def plot16_alpha1_all_phases():
    """Bar chart showing alpha_1 recovery across all phases and methods."""
    methods = [
        "Staged S2/S3",
        "Direct Joint",
        "Reg. ($\\lambda$=1e-3)",
    ]
    # alpha_1 errors from each phase (best stage for staged)
    al1_10pt = [5.8,  75.4,  21.7]
    al1_15pt = [0.1,  72.3,  22.3]
    al1_sym  = [9.5,   1.8,  np.nan]  # Reg not run with symmetric

    x = np.arange(len(methods))
    w = 0.22

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - w, al1_10pt, w, color=COLORS[0], alpha=0.85, label="Cathodic 10-pt")
    ax.bar(x, al1_15pt, w, color=COLORS[1], alpha=0.85, label="Cathodic 15-pt")
    bars3 = ax.bar(x + w, [v if not np.isnan(v) else 0 for v in al1_sym],
                   w, color=COLORS[3], alpha=0.85, label="Symmetric 20-pt")
    # Mark N/A
    for j in range(len(methods)):
        if np.isnan(al1_sym[j]):
            ax.text(x[j] + w, 0.5, "N/A", ha="center", va="bottom",
                    fontsize=7, color="gray")
    # Annotate the breakthrough
    ax.annotate("1.8%\n(38x better)",
                xy=(x[1] + w, al1_sym[1]),
                xytext=(x[1] + w + 0.35, 30),
                fontsize=8, color=COLORS[3],
                arrowprops=dict(arrowstyle="->", color=COLORS[3], lw=1.2),
                ha="center")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel(r"$\alpha_1$ Relative Error (%)")
    ax.set_title(r"$\alpha_1$ Recovery: Symmetric Range Breakthrough")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 85)
    plt.tight_layout()
    out = os.path.join(FIGDIR, "alpha1_all_phases.pdf")
    fig.savefig(out)
    plt.close(fig)
    generated.append("alpha1_all_phases.pdf")


# =========================================================================
# Plot 17: Symmetric Joint Inference -- Cathodic vs Symmetric
# =========================================================================
def plot17_symmetric_joint_comparison():
    """Side-by-side: alpha vs k0 errors for cathodic-only, symmetric-focused, symmetric-full."""
    comp = _safe_read(os.path.join(
        STUDY, "bv_joint_inference_charged_symmetric", "joint_comparison.csv"))
    if comp is None:
        return

    configs = comp["config"].values
    labels = ["Cathodic 10pt", "Symmetric 12pt", "Symmetric 20pt"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    x = np.arange(len(configs))
    w = 0.3

    # Left: alpha errors
    ax = axes[0]
    al1 = comp["alpha_1_err"].values * 100
    al2 = comp["alpha_2_err"].values * 100
    ax.bar(x - w/2, al1, w, color=COLORS[0], alpha=0.85, label=r"$\alpha_1$")
    ax.bar(x + w/2, al2, w, color=COLORS[2], alpha=0.85, label=r"$\alpha_2$")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Relative Error (%)")
    ax.set_title(r"$\alpha$ Recovery: Joint Inference")
    ax.legend(fontsize=8)
    # Annotate the improvement
    ax.annotate(f"{al1[0]:.1f}%", (x[0] - w/2, al1[0]), ha="center",
                va="bottom", fontsize=7, fontweight="bold")
    ax.annotate(f"{al1[2]:.1f}%", (x[2] - w/2, al1[2] + 1), ha="center",
                va="bottom", fontsize=7, fontweight="bold", color=COLORS[0])

    # Right: k0 errors (capped for readability)
    ax = axes[1]
    k0_1 = comp["k0_1_err"].values * 100
    k0_2 = comp["k0_2_err"].values * 100
    ax.bar(x - w/2, k0_1, w, color=COLORS[1], alpha=0.85, label=r"$k_{0,1}$")
    ax.bar(x + w/2, k0_2, w, color=COLORS[3], alpha=0.85, label=r"$k_{0,2}$")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Relative Error (%)")
    ax.set_title(r"$k_0$ Recovery: Joint Inference")
    ax.legend(fontsize=8)
    # Mark "hit bound" on symmetric bars
    for j in [1, 2]:
        if k0_1[j] > 250:
            ax.annotate("bound", (x[j] - w/2, k0_1[j]),
                        ha="center", va="bottom", fontsize=6, color="red", rotation=90)
        if k0_2[j] > 250:
            ax.annotate("bound", (x[j] + w/2, k0_2[j]),
                        ha="center", va="bottom", fontsize=6, color="red", rotation=90)

    plt.tight_layout()
    out = os.path.join(FIGDIR, "symmetric_joint_comparison.pdf")
    fig.savefig(out)
    plt.close(fig)
    generated.append("symmetric_joint_comparison.pdf")


# =========================================================================
# Main
# =========================================================================
if __name__ == "__main__":
    print(f"Study root: {STUDY}")
    print(f"Output dir: {FIGDIR}")
    print()

    plot_funcs = [
        ("Plot  1: Forward I-V Curve Shape", plot1_iv_curve_shape),
        ("Plot  2: Regularized Fit (10-pt)", plot2_regularized_fit_10pt),
        ("Plot  3: Staged Fit (10-pt)", plot3_staged_fit_10pt),
        ("Plot  4: Peroxide Fit (10-pt)", plot4_peroxide_fit_10pt),
        ("Plot  5: Staged Fit (15-pt)", plot5_staged_fit_15pt),
        ("Plot  6: Regularized Fit (15-pt)", plot6_regularized_fit_15pt),
        ("Plot  7: Steric Fit Comparison", plot7_steric_fit_comparison),
        ("Plot  8: Full Fit Comparison", plot8_full_fit_comparison),
        ("Plot  9: R1 Recovery Comparison", plot9_r1_recovery),
        ("Plot 10: R2 Recovery Comparison", plot10_r2_recovery),
        ("Plot 11: Steric Recovery Comparison", plot11_steric_recovery),
        ("Plot 12: Convergence (Staged)", plot12_convergence_staged),
        ("Plot 13: Regularization Sensitivity", plot13_regularization_sensitivity),
        ("Plot 14: Voltage Range Comparison", plot14_voltage_range_comparison),
        ("Plot 15: Symmetric Target I-V", plot15_symmetric_target_iv),
        ("Plot 16: Alpha1 Across All Phases", plot16_alpha1_all_phases),
        ("Plot 17: Symmetric Joint Comparison", plot17_symmetric_joint_comparison),
    ]

    for name, func in plot_funcs:
        try:
            func()
            print(f"  [OK] {name}")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            skipped.append(name)

    print()
    print(f"Generated: {len(generated)} plots")
    for f in generated:
        print(f"  {f}")
    if skipped:
        print(f"\nSkipped/Failed: {len(skipped)}")
        for s in skipped:
            print(f"  {s}")
    print("\nDone.")
