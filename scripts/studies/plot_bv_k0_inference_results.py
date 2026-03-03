"""Generate plots and convergence GIF for BV k0 inference results.

Reads the CSV outputs from StudyResults/bv_k0_inference/ and produces:
  1. Static plot: target data (noisy) + clean curve
  2. Static plot: target data vs initial guess vs final fit
  3. Convergence GIF: animated L-BFGS-B progress

Usage (from PNPInverse/ directory)::

    python scripts/studies/plot_bv_k0_inference_results.py
"""

from __future__ import annotations

import csv
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

from FluxCurve.plot import export_live_fit_gif

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

STUDY_DIR = os.path.join("StudyResults", "bv_k0_inference")
TARGET_CSV = os.path.join(STUDY_DIR, "phi_applied_vs_current_density_synthetic.csv")
FIT_CSV = os.path.join(STUDY_DIR, "phi_applied_vs_current_density_fit.csv")
HISTORY_CSV = os.path.join(STUDY_DIR, "bv_k0_optimization_history.csv")
POINT_CSV = os.path.join(STUDY_DIR, "bv_k0_point_gradients.csv")

# Physical scales (same as inference script)
F_CONST = 96485.3329
V_T = 0.025693
D_O2 = 2.10e-9
C_BULK = 0.5
L_REF = 1.0e-4
D_REF = D_O2
K_SCALE = D_REF / L_REF
N_ELECTRONS = 2
I_SCALE = N_ELECTRONS * F_CONST * D_REF * C_BULK / L_REF * 0.1

# True and initial k0 (dimensionless)
K0_HAT_TRUE = [2.4e-8 / K_SCALE, 1e-9 / K_SCALE]
K0_HAT_INIT = [0.01, 0.001]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _eta_to_V_RHE(eta_hat):
    """Convert dimensionless overpotential to V vs RHE."""
    return eta_hat * V_T + 0.695


# ---------------------------------------------------------------------------
# Plot 1: Target data
# ---------------------------------------------------------------------------

def plot_target_data(out_path):
    """Plot clean curve + noisy target data."""
    rows = _read_csv(TARGET_CSV)
    phi = np.array([float(r["phi_applied"]) for r in rows])
    flux_clean = np.array([float(r["flux_clean"]) for r in rows])
    flux_noisy = np.array([float(r["flux_noisy"]) for r in rows])

    # Convert to V_RHE
    v_rhe = _eta_to_V_RHE(phi)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(v_rhe, flux_clean, "k-", linewidth=1.5, label="clean (no noise)", zorder=3)
    ax.plot(
        v_rhe, flux_noisy, "o", color="#1f77b4", markersize=6,
        markeredgecolor="white", markeredgewidth=0.5,
        label="noisy target (2% noise)", zorder=4,
    )
    ax.set_xlabel("$V$ vs RHE (V)", fontsize=12)
    ax.set_ylabel("Current density (dimensionless)", fontsize=12)
    ax.set_title("Synthetic BV target I\u2013V curve", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: Target vs initial guess vs final fit
# ---------------------------------------------------------------------------

def plot_target_vs_fit(out_path):
    """Plot target data, initial guess curve, and final optimized fit."""
    target_rows = _read_csv(TARGET_CSV)
    fit_rows = _read_csv(FIT_CSV)
    history_rows = _read_csv(HISTORY_CSV)
    point_rows = _read_csv(POINT_CSV)

    phi = np.array([float(r["phi_applied"]) for r in target_rows])
    flux_noisy = np.array([float(r["flux_noisy"]) for r in target_rows])
    v_rhe = _eta_to_V_RHE(phi)

    # Final fit from fit CSV
    fit_sim = np.array([float(r["simulated_current_density"]) for r in fit_rows])

    # Initial guess curve: evaluation 1 from point gradients
    init_curve = np.full_like(phi, np.nan)
    for r in point_rows:
        if int(r["evaluation"]) == 1:
            idx = int(r["point_index"])
            if 0 <= idx < len(init_curve):
                init_curve[idx] = float(r["simulated_flux"])

    # Get final k0 values
    last_history = history_rows[-1]
    k0_1_est = float(last_history["k0_0"])
    k0_2_est = float(last_history["k0_1"])
    k0_1_est_phys = k0_1_est * K_SCALE
    k0_2_est_phys = k0_2_est * K_SCALE

    fig, ax = plt.subplots(figsize=(8, 5))

    # Target data
    ax.plot(
        v_rhe, flux_noisy, "o", color="#1f77b4", markersize=7,
        markeredgecolor="white", markeredgewidth=0.6,
        label="target data (2% noise)", zorder=5,
    )

    # Initial guess
    ax.plot(
        v_rhe, init_curve, "D--", color="#d62728", linewidth=1.5,
        markersize=5, markeredgecolor="white", markeredgewidth=0.4,
        label=f"initial guess ($\\hat{{k}}_0$ = [{K0_HAT_INIT[0]}, {K0_HAT_INIT[1]}])",
        zorder=3,
    )

    # Final fit
    ax.plot(
        v_rhe, fit_sim, "s-", color="#2ca02c", linewidth=2, markersize=5,
        markeredgecolor="white", markeredgewidth=0.4,
        label=(
            f"L-BFGS-B fit "
            f"($k_{{0,1}}$={k0_1_est_phys:.2e}, $k_{{0,2}}$={k0_2_est_phys:.2e} m/s)"
        ),
        zorder=4,
    )

    ax.set_xlabel("$V$ vs RHE (V)", fontsize=12)
    ax.set_ylabel("Current density (dimensionless)", fontsize=12)
    ax.set_title(
        "BV $k_0$ inference: target vs initial guess vs optimized fit",
        fontsize=13,
    )
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# GIF: Convergence animation
# ---------------------------------------------------------------------------

def generate_convergence_gif(out_path):
    """Generate convergence GIF using export_live_fit_gif."""
    target_rows = _read_csv(TARGET_CSV)
    history_rows = _read_csv(HISTORY_CSV)
    point_rows = _read_csv(POINT_CSV)

    phi = np.array([float(r["phi_applied"]) for r in target_rows])
    flux_noisy = np.array([float(r["flux_noisy"]) for r in target_rows])

    # Remap BV history column names to what export_live_fit_gif expects
    mapped_history = []
    for r in history_rows:
        mapped_history.append({
            "evaluation": r["evaluation"],
            "objective": r["objective"],
            "n_failed_points": r["n_failed_points"],
            "kappa0": r["k0_0"],
            "kappa1": r["k0_1"],
        })

    result = export_live_fit_gif(
        path=out_path,
        phi_applied_values=phi,
        target_flux=flux_noisy,
        history_rows=mapped_history,
        point_rows=point_rows,
        seconds=6.0,
        n_frames=7,  # one frame per evaluation (7 evals total)
        dpi=160,
        y_label="current density (dimensionless)",
        title="BV $k_0$ inference convergence",
    )

    if result:
        print(f"Saved: {result}")
    else:
        print("GIF generation failed (PIL may not be available)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(STUDY_DIR, exist_ok=True)

    plot_target_data(os.path.join(STUDY_DIR, "target_data.png"))
    plot_target_vs_fit(os.path.join(STUDY_DIR, "target_vs_initial_vs_fit.png"))
    generate_convergence_gif(os.path.join(STUDY_DIR, "bv_k0_convergence.gif"))


if __name__ == "__main__":
    main()
