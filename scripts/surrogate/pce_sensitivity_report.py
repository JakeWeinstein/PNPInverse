"""Generate sensitivity analysis plots from a fitted PCE surrogate model.

Loads a fitted PCE model and produces:
1. Voltage-resolved Sobol index line plots (S_i vs voltage for each param)
2. Stacked bar chart of variance decomposition at each voltage
3. LaTeX table export for the V&V report

Usage (from PNPInverse/ directory)::

    python scripts/surrogate/pce_sensitivity_report.py
    python scripts/surrogate/pce_sensitivity_report.py \
        --model data/surrogate_models/pce/pce_model.pkl \
        --output-dir StudyResults/surrogate_fidelity/
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PNPINVERSE_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _PNPINVERSE_ROOT not in sys.path:
    sys.path.insert(0, _PNPINVERSE_ROOT)

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Surrogate.pce_model import PCESurrogateModel


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

_PARAM_COLORS = {
    "log10_k0_1": "#1f77b4",
    "log10_k0_2": "#ff7f0e",
    "alpha_1": "#2ca02c",
    "alpha_2": "#d62728",
    "k0_1": "#1f77b4",
    "k0_2": "#ff7f0e",
}

_PARAM_LABELS = {
    "log10_k0_1": r"$\log_{10} k_0^{(1)}$",
    "log10_k0_2": r"$\log_{10} k_0^{(2)}$",
    "alpha_1": r"$\alpha_1$",
    "alpha_2": r"$\alpha_2$",
    "k0_1": r"$k_0^{(1)}$",
    "k0_2": r"$k_0^{(2)}$",
}


def plot_sobol_line(
    sobol: dict,
    phi_applied: np.ndarray,
    output_key: str,
    output_label: str,
    output_dir: str,
) -> str:
    """Plot first-order and total-order Sobol indices vs voltage.

    Parameters
    ----------
    sobol : dict
        Sobol index structure from PCESurrogateModel.
    phi_applied : np.ndarray (n_eta,)
        Voltage grid.
    output_key : str
        'current_density' or 'peroxide_current'.
    output_label : str
        Human-readable label for the output.
    output_dir : str
        Directory to save the plot.

    Returns
    -------
    str
        Path to saved figure.
    """
    data = sobol[output_key]
    first_order = data["first_order"]  # (4, n_eta)
    total_order = data["total_order"]  # (4, n_eta)
    param_names = sobol["parameter_names"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # First-order
    ax = axes[0]
    for i, name in enumerate(param_names):
        color = _PARAM_COLORS.get(name, None)
        label = _PARAM_LABELS.get(name, name)
        ax.plot(phi_applied, first_order[i, :], label=label, color=color, linewidth=1.5)
    ax.set_xlabel("Applied Potential (V)")
    ax.set_ylabel("Sobol Index")
    ax.set_title(f"{output_label} -- First-Order $S_i$")
    ax.legend(loc="best", fontsize=9)
    ax.set_ylim(bottom=-0.05, top=1.05)
    ax.grid(True, alpha=0.3)

    # Total-order
    ax = axes[1]
    for i, name in enumerate(param_names):
        color = _PARAM_COLORS.get(name, None)
        label = _PARAM_LABELS.get(name, name)
        ax.plot(phi_applied, total_order[i, :], label=label, color=color, linewidth=1.5)
    ax.set_xlabel("Applied Potential (V)")
    ax.set_title(f"{output_label} -- Total-Order $S_{{T,i}}$")
    ax.legend(loc="best", fontsize=9)
    ax.set_ylim(bottom=-0.05, top=1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    tag = "cd" if "current" in output_key else "pc"
    path = os.path.join(output_dir, f"pce_sobol_{tag}_vs_voltage.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_variance_decomposition(
    sobol: dict,
    phi_applied: np.ndarray,
    output_key: str,
    output_label: str,
    output_dir: str,
) -> str:
    """Stacked bar chart of variance decomposition at each voltage.

    Parameters
    ----------
    sobol : dict
    phi_applied : np.ndarray
    output_key : str
    output_label : str
    output_dir : str

    Returns
    -------
    str
        Path to saved figure.
    """
    data = sobol[output_key]
    first_order = data["first_order"]  # (4, n_eta)
    param_names = sobol["parameter_names"]
    n_eta = first_order.shape[1]

    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(n_eta)
    bottom = np.zeros(n_eta)

    for i, name in enumerate(param_names):
        color = _PARAM_COLORS.get(name, None)
        label = _PARAM_LABELS.get(name, name)
        vals = np.clip(first_order[i, :], 0, None)  # Clip negatives for display
        ax.bar(x, vals, bottom=bottom, label=label, color=color, width=0.8)
        bottom += vals

    # Interaction remainder (1 - sum of first-order)
    remainder = 1.0 - bottom
    remainder = np.clip(remainder, 0, None)
    ax.bar(x, remainder, bottom=bottom, label="Interactions", color="#7f7f7f",
           width=0.8, alpha=0.5)

    # Label x-axis with voltages
    tick_step = max(1, n_eta // 10)
    tick_positions = list(range(0, n_eta, tick_step))
    tick_labels = [f"{phi_applied[k]:.2f}" for k in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, fontsize=8)

    ax.set_xlabel("Applied Potential (V)")
    ax.set_ylabel("Variance Fraction")
    ax.set_title(f"{output_label} -- Variance Decomposition (First-Order)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    tag = "cd" if "current" in output_key else "pc"
    path = os.path.join(output_dir, f"pce_variance_decomposition_{tag}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def export_latex_table(sobol: dict, output_dir: str) -> str:
    """Export mean Sobol indices as a LaTeX table.

    Parameters
    ----------
    sobol : dict
    output_dir : str

    Returns
    -------
    str
        Path to saved .tex file.
    """
    param_names = sobol["parameter_names"]

    lines = [
        r"\begin{tabular}{l c c c c}",
        r"\toprule",
        r"Parameter & \multicolumn{2}{c}{Current Density} & \multicolumn{2}{c}{Peroxide Current} \\",
        r" & $S_i$ & $S_{T,i}$ & $S_i$ & $S_{T,i}$ \\",
        r"\midrule",
    ]

    cd_data = sobol["current_density"]
    pc_data = sobol["peroxide_current"]

    label_map = {
        "log10_k0_1": r"$\log_{10} k_0^{(1)}$",
        "log10_k0_2": r"$\log_{10} k_0^{(2)}$",
        "alpha_1": r"$\alpha_1$",
        "alpha_2": r"$\alpha_2$",
        "k0_1": r"$k_0^{(1)}$",
        "k0_2": r"$k_0^{(2)}$",
    }

    for i, name in enumerate(param_names):
        label = label_map.get(name, name)
        cd_s = cd_data["mean_first_order"][i]
        cd_st = cd_data["mean_total_order"][i]
        pc_s = pc_data["mean_first_order"][i]
        pc_st = pc_data["mean_total_order"][i]
        lines.append(
            f"{label} & {cd_s:.4f} & {cd_st:.4f} & {pc_s:.4f} & {pc_st:.4f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
    ])

    path = os.path.join(output_dir, "pce_sobol_table.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PCE sensitivity analysis plots and tables",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="data/surrogate_models/pce/pce_model.pkl",
        help="Path to fitted PCE model pickle",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="StudyResults/surrogate_fidelity",
        help="Directory to save plots and tables",
    )
    args = parser.parse_args()

    print(f"\n{'#'*70}")
    print(f"  PCE SENSITIVITY REPORT GENERATOR")
    print(f"  Model      : {args.model}")
    print(f"  Output dir : {args.output_dir}")
    print(f"{'#'*70}\n")

    # Load model
    model = PCESurrogateModel.load(args.model)
    os.makedirs(args.output_dir, exist_ok=True)

    phi_applied = model.phi_applied
    sobol = model._sobol_indices

    if sobol is None:
        print("  Sobol indices not found in model. Recomputing...")
        sobol = model.compute_sobol_indices(verbose=True)

    # Generate line plots
    for output_key, output_label in [
        ("current_density", "Current Density"),
        ("peroxide_current", "Peroxide Current"),
    ]:
        plot_sobol_line(sobol, phi_applied, output_key, output_label, args.output_dir)
        plot_variance_decomposition(
            sobol, phi_applied, output_key, output_label, args.output_dir
        )

    # Export LaTeX table
    export_latex_table(sobol, args.output_dir)

    # Print sensitivity report
    model.print_sensitivity_report()

    # Print key finding
    cd_data = sobol["current_density"]
    k0_2_first = cd_data["mean_first_order"][1]
    k0_2_total = cd_data["mean_total_order"][1]
    print(f"\n  Summary: k0_2 explains {k0_2_first*100:.1f}% first-order "
          f"and {k0_2_total*100:.1f}% total-order of CD variance.\n")

    print(f"{'#'*70}")
    print(f"  REPORT COMPLETE")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
