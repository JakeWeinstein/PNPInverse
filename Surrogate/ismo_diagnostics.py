"""ISMO per-iteration diagnostics: CSV logging, scatter plots, convergence curves.

Provides the ``ISMODiagnostics`` class for recording each iteration to CSV,
generating scatter plots (surrogate vs PDE objectives), I-V comparison
overlays at the optimizer's best point, and saving per-iteration state to
NPZ files.

Standalone functions ``plot_convergence_curves`` and
``plot_k0_2_recovery_comparison`` produce summary figures after the ISMO
loop completes.

Public API
----------
ISMODiagnostics
    Per-iteration logging and plotting.
plot_convergence_curves
    2x2 summary figure of ISMO convergence history.
plot_k0_2_recovery_comparison
    Bar chart comparing k0_2 recovery error between Phase 3 and ISMO.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from Surrogate.ismo_convergence import ISMODiagnosticRecord


# ---------------------------------------------------------------------------
# Per-iteration diagnostics
# ---------------------------------------------------------------------------

class ISMODiagnostics:
    """Per-iteration ISMO logging, plotting, and state persistence.

    Parameters
    ----------
    output_dir : str
        Root output directory.  Created on first use if it does not exist.
    """

    _CSV_HEADER = [
        "iteration",
        "surrogate_pde_agreement",
        "parameter_stability",
        "surrogate_test_nrmse_cd",
        "surrogate_test_nrmse_pc",
        "objective_improvement",
        "pde_solve_time_s",
        "retrain_time_s",
        "inference_time_s",
        "acquisition_strategy",
        "n_pde_evals_this_iter",
        "n_total_training",
        "best_k0_1",
        "best_k0_2",
        "best_alpha_1",
        "best_alpha_2",
    ]

    def __init__(self, output_dir: str = "StudyResults/ismo") -> None:
        self.output_dir = output_dir
        self._csv_path = os.path.join(output_dir, "ismo_iteration_log.csv")
        self._header_written = False

    # ------------------------------------------------------------------ #
    # CSV logging
    # ------------------------------------------------------------------ #

    def log_iteration(self, record: ISMODiagnosticRecord) -> None:
        """Append one row to the iteration CSV log.

        The CSV is created (with header) on the first call.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        write_header = not self._header_written and not os.path.exists(self._csv_path)
        with open(self._csv_path, "a", newline="") as fh:
            writer = csv.writer(fh)
            if write_header:
                writer.writerow(self._CSV_HEADER)
            self._header_written = True

            bp = record.best_params
            obj_imp = (
                "" if math.isnan(record.objective_improvement)
                else f"{record.objective_improvement:.6f}"
            )

            writer.writerow([
                record.iteration,
                f"{record.surrogate_pde_agreement:.6e}",
                f"{record.parameter_stability:.6e}",
                f"{record.surrogate_test_nrmse_cd:.6e}",
                f"{record.surrogate_test_nrmse_pc:.6e}",
                obj_imp,
                f"{record.pde_solve_time_s:.2f}",
                f"{record.retrain_time_s:.2f}",
                f"{record.inference_time_s:.2f}",
                record.acquisition_strategy,
                record.n_pde_evals_this_iter,
                record.n_total_training,
                f"{bp[0]:.6e}",
                f"{bp[1]:.6e}",
                f"{bp[2]:.6f}",
                f"{bp[3]:.6f}",
            ])

        print(
            f"  [Diagnostics] Iteration {record.iteration} logged to "
            f"{self._csv_path}",
            flush=True,
        )

    # ------------------------------------------------------------------ #
    # Scatter plot: surrogate vs PDE objectives at candidate points
    # ------------------------------------------------------------------ #

    def plot_surrogate_vs_pde_scatter(
        self,
        iteration: int,
        candidate_params: np.ndarray,
        surrogate_objectives: np.ndarray,
        pde_objectives: np.ndarray,
    ) -> None:
        """Scatter plot of surrogate objective vs PDE objective.

        Saved to ``{output_dir}/iter_{iteration:02d}_surrogate_vs_pde.png``.
        Includes a 1:1 reference line, R-squared annotation, and NRMSE.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(self.output_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(pde_objectives, surrogate_objectives, s=20, alpha=0.7)

        lo = min(pde_objectives.min(), surrogate_objectives.min())
        hi = max(pde_objectives.max(), surrogate_objectives.max())
        margin = 0.05 * (hi - lo) if hi > lo else 0.1
        ax.plot(
            [lo - margin, hi + margin],
            [lo - margin, hi + margin],
            "k--",
            linewidth=1,
            label="1:1",
        )

        # R-squared
        ss_res = np.sum((surrogate_objectives - pde_objectives) ** 2)
        ss_tot = np.sum((pde_objectives - np.mean(pde_objectives)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        # NRMSE
        rmse = float(np.sqrt(np.mean((surrogate_objectives - pde_objectives) ** 2)))
        denom = float(np.ptp(pde_objectives)) if np.ptp(pde_objectives) > 0 else 1.0
        nrmse = rmse / denom

        ax.set_xlabel("PDE objective")
        ax.set_ylabel("Surrogate objective")
        ax.set_title(f"Iteration {iteration}: Surrogate vs PDE")
        ax.annotate(
            f"R² = {r2:.4f}\nNRMSE = {nrmse:.4f}",
            xy=(0.05, 0.92),
            xycoords="axes fraction",
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )
        ax.legend(loc="lower right")
        fig.tight_layout()

        path = os.path.join(
            self.output_dir, f"iter_{iteration:02d}_surrogate_vs_pde.png"
        )
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  [Diagnostics] Saved scatter plot: {path}", flush=True)

    # ------------------------------------------------------------------ #
    # I-V comparison at optimizer best
    # ------------------------------------------------------------------ #

    def plot_iv_comparison_at_best(
        self,
        iteration: int,
        phi_applied: np.ndarray,
        surrogate_cd: np.ndarray,
        pde_cd: np.ndarray,
        surrogate_pc: np.ndarray,
        pde_pc: np.ndarray,
    ) -> None:
        """Two-panel I-V overlay (surrogate vs PDE) at the best point.

        Saved to ``{output_dir}/iter_{iteration:02d}_iv_comparison.png``.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(self.output_dir, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Current density
        ax1.plot(phi_applied, pde_cd, "k-", linewidth=1.5, label="PDE")
        ax1.plot(phi_applied, surrogate_cd, "r--", linewidth=1.5, label="Surrogate")
        ax1.set_xlabel("Applied potential (V)")
        ax1.set_ylabel("Current density")
        ax1.set_title(f"Iter {iteration}: Current Density")
        ax1.legend()

        # Peroxide current
        ax2.plot(phi_applied, pde_pc, "k-", linewidth=1.5, label="PDE")
        ax2.plot(phi_applied, surrogate_pc, "r--", linewidth=1.5, label="Surrogate")
        ax2.set_xlabel("Applied potential (V)")
        ax2.set_ylabel("Peroxide current")
        ax2.set_title(f"Iter {iteration}: Peroxide Current")
        ax2.legend()

        fig.suptitle(f"ISMO Iteration {iteration}: Surrogate vs PDE at Best Point")
        fig.tight_layout()

        path = os.path.join(
            self.output_dir, f"iter_{iteration:02d}_iv_comparison.png"
        )
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  [Diagnostics] Saved I-V comparison: {path}", flush=True)

    # ------------------------------------------------------------------ #
    # Per-iteration state persistence
    # ------------------------------------------------------------------ #

    def save_iteration_state(
        self,
        iteration: int,
        record: ISMODiagnosticRecord,
        new_params: np.ndarray,
        new_cd: np.ndarray,
        new_pc: np.ndarray,
    ) -> None:
        """Save per-iteration NPZ with new training points and metadata.

        Saved to ``{output_dir}/iter_{iteration:02d}_state.npz``.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        path = os.path.join(
            self.output_dir, f"iter_{iteration:02d}_state.npz"
        )
        np.savez_compressed(
            path,
            new_params=new_params,
            new_cd=new_cd,
            new_pc=new_pc,
            best_params=np.asarray(record.best_params),
            surrogate_pde_agreement=np.float64(record.surrogate_pde_agreement),
            parameter_stability=np.float64(record.parameter_stability),
            n_total_training=np.int64(record.n_total_training),
            iteration=np.int64(record.iteration),
        )
        print(f"  [Diagnostics] Saved iteration state: {path}", flush=True)


# ---------------------------------------------------------------------------
# Post-ISMO convergence curves
# ---------------------------------------------------------------------------

def plot_convergence_curves(
    history: List[ISMODiagnosticRecord],
    output_path: str = "StudyResults/ismo/convergence_curves.png",
    criteria: Any = None,
) -> None:
    """Generate a 2x2 summary figure after the ISMO loop completes.

    Subplots:
        1. Top-left: Surrogate-PDE agreement vs iteration (log-scale y).
        2. Top-right: Parameter estimates vs iteration.
        3. Bottom-left: Training-set size vs iteration (bar chart).
        4. Bottom-right: Parameter stability vs iteration.

    Parameters
    ----------
    history : list of ISMODiagnosticRecord
        Full ISMO iteration history.
    output_path : str
        Where to save the PNG.
    criteria : ISMOConvergenceCriteria or None
        If provided, convergence thresholds are drawn as dashed lines.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(history) == 0:
        print("  [Diagnostics] No history to plot.", flush=True)
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    iters = [r.iteration for r in history]
    agreements = [r.surrogate_pde_agreement for r in history]
    stabilities = [r.parameter_stability for r in history]
    n_training = [r.n_total_training for r in history]
    best_params_arr = np.array([list(r.best_params) for r in history])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # ---- Top-left: Agreement ----
    ax = axes[0, 0]
    colors = []
    if criteria is not None:
        colors = [
            "green" if a < criteria.agreement_tol else "red" for a in agreements
        ]
    else:
        colors = ["steelblue"] * len(agreements)
    ax.scatter(iters, agreements, c=colors, zorder=3)
    ax.plot(iters, agreements, "k-", alpha=0.4, zorder=2)
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Surrogate-PDE Agreement (NRMSE)")
    ax.set_title("Convergence: Surrogate-PDE Agreement")
    if criteria is not None:
        ax.axhline(
            criteria.agreement_tol, color="gray", linestyle="--", linewidth=1,
            label=f"tol={criteria.agreement_tol}",
        )
        ax.legend()

    # ---- Top-right: Parameter estimates ----
    ax = axes[0, 1]
    if best_params_arr.shape[0] > 0:
        ax.semilogy(iters, best_params_arr[:, 0], "o-", label="k0_1")
        ax.semilogy(iters, best_params_arr[:, 1], "s-", label="k0_2")
        ax2 = ax.twinx()
        ax2.plot(iters, best_params_arr[:, 2], "^--", color="C2", label="alpha_1")
        ax2.plot(iters, best_params_arr[:, 3], "v--", color="C3", label="alpha_2")
        ax2.set_ylabel("alpha values")
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("k0 values (log scale)")
    ax.set_title("Parameter Estimates vs Iteration")

    # ---- Bottom-left: Training set size ----
    ax = axes[1, 0]
    ax.bar(iters, n_training, color="steelblue", alpha=0.8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total training samples")
    ax.set_title("Training Set Growth")

    # ---- Bottom-right: Stability ----
    ax = axes[1, 1]
    ax.plot(iters, stabilities, "o-", color="darkorange")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Parameter Stability (L2 norm)")
    ax.set_title("Parameter Stability vs Iteration")
    if criteria is not None:
        ax.axhline(
            criteria.stability_tol, color="gray", linestyle="--", linewidth=1,
            label=f"tol={criteria.stability_tol}",
        )
        ax.legend()

    fig.suptitle("ISMO Convergence Summary", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  [Diagnostics] Saved convergence curves: {output_path}", flush=True)


# ---------------------------------------------------------------------------
# k0_2 recovery comparison
# ---------------------------------------------------------------------------

def plot_k0_2_recovery_comparison(
    phase3_errors: Dict[str, float],
    ismo_errors: Dict[str, float],
    output_path: str = "StudyResults/ismo/k0_2_improvement.png",
) -> None:
    """Bar chart comparing k0_2 recovery error: Phase 3 vs ISMO.

    Parameters
    ----------
    phase3_errors : dict
        ``{"0pct_noise": float, "1pct_noise": float}`` -- Phase 3
        k0_2 relative errors.
    ismo_errors : dict
        Same structure for ISMO-refined surrogate.
    output_path : str
        Where to save the PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    labels = ["0% noise", "1% noise"]
    p3_vals = [
        phase3_errors.get("0pct_noise", 0.0),
        phase3_errors.get("1pct_noise", 0.0),
    ]
    ismo_vals = [
        ismo_errors.get("0pct_noise", 0.0),
        ismo_errors.get("1pct_noise", 0.0),
    ]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    bars1 = ax.bar(x - width / 2, p3_vals, width, label="Phase 3 Baseline", color="C0")
    bars2 = ax.bar(x + width / 2, ismo_vals, width, label="ISMO Refined", color="C1")

    # Target line: 50% of Phase 3 0% noise error
    target = p3_vals[0] * 0.5
    ax.axhline(target, color="red", linestyle="--", linewidth=1.5, label=f"50% target ({target:.1f}%)")

    ax.set_ylabel("k0_2 Relative Error (%)")
    ax.set_title("k0_2 Recovery: Phase 3 vs ISMO")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Annotate bar values
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(
            f"{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(
            f"{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  [Diagnostics] Saved k0_2 comparison: {output_path}", flush=True)
