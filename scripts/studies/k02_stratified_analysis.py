"""k0_2-stratified error analysis across all surrogate models.

Computes per-model, per-k0_2-bin error metrics (mean, median, p95, max NRMSE)
for both current density (CD) and peroxide current (PC) outputs.  Optionally
correlates GP predicted uncertainty with actual error per k0_2 bin, and
cross-references PCE Sobol sensitivity indices for k0_2.

Outputs:
    - k02_stratified_errors.json   (complete results dict)
    - k02_error_vs_value.png       (scatter + binned-summary plot)
    - k02_error_heatmap.png        (model x bin worst-case heatmap)
    - k02_bin_table.csv            (flat tabular summary)

Usage (from PNPInverse/ directory)::

    python scripts/studies/k02_stratified_analysis.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PNPINVERSE_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _PNPINVERSE_ROOT not in sys.path:
    sys.path.insert(0, _PNPINVERSE_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BIN_EDGES_LOG10 = [-7, -6, -5, -4, -3, -2, -1]
BIN_LABELS = [
    "[1e-7,1e-6)",
    "[1e-6,1e-5)",
    "[1e-5,1e-4)",
    "[1e-4,1e-3)",
    "[1e-3,1e-2)",
    "[1e-2,1e-1)",
]

MODEL_PREFIXES = {
    "nn_ensemble": "nn_ensemble",
    "rbf_baseline": "rbf_baseline",
    "pod_rbf_log": "pod_rbf_log",
    "pod_rbf_nolog": "pod_rbf_nolog",
}

MODEL_COLORS = {
    "nn_ensemble": "#1f77b4",     # blue
    "rbf_baseline": "#ff7f0e",    # orange
    "pod_rbf_log": "#2ca02c",     # green
    "pod_rbf_nolog": "#d62728",   # red
}

OUTPUT_TYPES = ["cd", "pc"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assign_bins(k0_2_values: np.ndarray) -> np.ndarray:
    """Return integer bin indices (0-based) for each k0_2 value.

    Values outside [1e-7, 0.1) are clipped to the nearest bin.
    """
    log_vals = np.log10(np.clip(k0_2_values, 1e-20, None))
    # np.digitize with right=False: bin i holds values where edges[i] <= val < edges[i+1]
    indices = np.digitize(log_vals, BIN_EDGES_LOG10) - 1
    return np.clip(indices, 0, len(BIN_LABELS) - 1)


def compute_bin_metrics(values: np.ndarray) -> dict:
    """Compute summary statistics for a 1-D array of NRMSE values."""
    if len(values) == 0:
        return {"mean": None, "median": None, "p95": None, "max": None, "count": 0}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "count": int(len(values)),
    }


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

def load_errors(csv_path: str) -> pd.DataFrame:
    """Load per_sample_errors.csv and validate columns."""
    df = pd.read_csv(csv_path)
    required = ["k0_2"]
    for model in MODEL_PREFIXES:
        for out in OUTPUT_TYPES:
            col = f"{model}_{out}_nrmse"
            required.append(col)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")
    logger.info("Loaded %d test samples from %s", len(df), csv_path)
    return df


# ---------------------------------------------------------------------------
# 3. Per-bin error metrics
# ---------------------------------------------------------------------------

def compute_stratified_errors(df: pd.DataFrame) -> dict:
    """Compute per-model, per-output, per-bin error metrics."""
    bin_idx = assign_bins(df["k0_2"].values)
    results = {}
    for model in MODEL_PREFIXES:
        results[model] = {}
        for out in OUTPUT_TYPES:
            col = f"{model}_{out}_nrmse"
            per_bin = {}
            for bi, label in enumerate(BIN_LABELS):
                mask = bin_idx == bi
                per_bin[label] = compute_bin_metrics(df.loc[mask, col].values)
            results[model][out] = per_bin
    return results


# ---------------------------------------------------------------------------
# 4. GP uncertainty correlation
# ---------------------------------------------------------------------------

def gp_uq_correlation(
    gp_model_dir: str,
    test_data_path: str,
    split_indices_path: str,
    df: pd.DataFrame,
) -> dict | None:
    """Compute Spearman rank correlation between GP std and actual error per bin.

    Returns None if the GP model cannot be loaded.
    """
    try:
        from Surrogate.gp_model import GPSurrogateModel
    except ImportError:
        logger.warning("Cannot import GPSurrogateModel; skipping GP UQ correlation.")
        return None

    # Try gp_fixed first, fall back to gp
    model_dir = gp_model_dir
    if not os.path.isdir(model_dir):
        alt = os.path.join(os.path.dirname(model_dir), "gp")
        if os.path.isdir(alt):
            model_dir = alt
        else:
            logger.warning("GP model directory not found: %s", gp_model_dir)
            return None

    try:
        gp = GPSurrogateModel.load(model_dir, device="cpu")
    except Exception as exc:
        logger.warning("Failed to load GP model from %s: %s", model_dir, exc)
        return None

    # Load test data
    try:
        merged = np.load(test_data_path)
        splits = np.load(split_indices_path)
        test_idx = splits["test_idx"]
        params = merged["parameters"][test_idx]  # (N_test, 4)
    except Exception as exc:
        logger.warning("Failed to load test data: %s", exc)
        return None

    try:
        pred = gp.predict_batch_with_uncertainty(params)
        cd_std = pred["current_density_std"]   # (N, 22)
        pc_std = pred["peroxide_current_std"]  # (N, 22)
        # Average std across voltage points for each sample
        cd_std_mean = np.mean(cd_std, axis=1)
        pc_std_mean = np.mean(pc_std, axis=1)
    except Exception as exc:
        logger.warning("GP predict_batch_with_uncertainty failed: %s", exc)
        return None

    # Match samples: df may be a subset reordered; we need to align
    # The simplest approach: use test_idx ordering from split_indices
    # and match to df by sample_idx
    if "sample_idx" in df.columns:
        df_idx_set = set(df["sample_idx"].values)
        # Build a mapping from sample_idx -> row in predictions
        pred_sample_idx = test_idx
        common = sorted(df_idx_set & set(pred_sample_idx))
        if len(common) < 10:
            logger.warning("Too few matching samples between GP predictions and CSV (%d)", len(common))
            return None

        # Build lookup arrays
        pred_pos = {sid: i for i, sid in enumerate(pred_sample_idx)}
        df_pos = {sid: i for i, sid in enumerate(df["sample_idx"].values)}

        cd_std_aligned = np.array([cd_std_mean[pred_pos[s]] for s in common])
        pc_std_aligned = np.array([pc_std_mean[pred_pos[s]] for s in common])

        # Get actual NRMSE - use nn_ensemble as reference (or average across models)
        cd_nrmse_aligned = np.array([
            df.iloc[df_pos[s]]["nn_ensemble_cd_nrmse"] for s in common
        ])
        pc_nrmse_aligned = np.array([
            df.iloc[df_pos[s]]["nn_ensemble_pc_nrmse"] for s in common
        ])
        k02_aligned = np.array([df.iloc[df_pos[s]]["k0_2"] for s in common])
    else:
        # Assume same ordering
        n = min(len(df), len(cd_std_mean))
        cd_std_aligned = cd_std_mean[:n]
        pc_std_aligned = pc_std_mean[:n]
        cd_nrmse_aligned = df["nn_ensemble_cd_nrmse"].values[:n]
        pc_nrmse_aligned = df["nn_ensemble_pc_nrmse"].values[:n]
        k02_aligned = df["k0_2"].values[:n]

    # Overall correlation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cd_rho_all, cd_p_all = stats.spearmanr(cd_std_aligned, cd_nrmse_aligned)
        pc_rho_all, pc_p_all = stats.spearmanr(pc_std_aligned, pc_nrmse_aligned)

    # Per-bin correlation
    bin_idx = assign_bins(k02_aligned)
    per_bin = {}
    for bi, label in enumerate(BIN_LABELS):
        mask = bin_idx == bi
        if mask.sum() < 5:
            per_bin[label] = {
                "cd_spearman_rho": None, "cd_p": None,
                "pc_spearman_rho": None, "pc_p": None,
                "count": int(mask.sum()),
            }
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cr, cp = stats.spearmanr(cd_std_aligned[mask], cd_nrmse_aligned[mask])
            pr, pp = stats.spearmanr(pc_std_aligned[mask], pc_nrmse_aligned[mask])
        per_bin[label] = {
            "cd_spearman_rho": float(cr) if np.isfinite(cr) else None,
            "cd_p": float(cp) if np.isfinite(cp) else None,
            "pc_spearman_rho": float(pr) if np.isfinite(pr) else None,
            "pc_p": float(pp) if np.isfinite(pp) else None,
            "count": int(mask.sum()),
        }

    return {
        "per_bin": per_bin,
        "overall": {
            "cd_rho": float(cd_rho_all) if np.isfinite(cd_rho_all) else None,
            "cd_p": float(cd_p_all) if np.isfinite(cd_p_all) else None,
            "pc_rho": float(pc_rho_all) if np.isfinite(pc_rho_all) else None,
            "pc_p": float(pc_p_all) if np.isfinite(pc_p_all) else None,
        },
    }


# ---------------------------------------------------------------------------
# 5. PCE Sobol cross-reference
# ---------------------------------------------------------------------------

def load_pce_sobol(sobol_json_path: str) -> dict | None:
    """Extract k0_2 Sobol indices from the PCE analysis."""
    if not os.path.isfile(sobol_json_path):
        logger.warning("Sobol JSON not found: %s", sobol_json_path)
        return None

    with open(sobol_json_path) as f:
        sobol = json.load(f)

    # k0_2 is index 1 in parameter_names
    k02_idx = 1
    cd_first = sobol["current_density"]["first_order"][k02_idx]
    pc_first = sobol["peroxide_current"]["first_order"][k02_idx]
    cd_mean = sobol["current_density"]["mean_first_order"][k02_idx]
    pc_mean = sobol["peroxide_current"]["mean_first_order"][k02_idx]

    return {
        "cd_mean_first_order": float(cd_mean),
        "pc_mean_first_order": float(pc_mean),
        "cd_per_voltage": [float(v) for v in cd_first],
        "pc_per_voltage": [float(v) for v in pc_first],
    }


# ---------------------------------------------------------------------------
# 6a. Generate JSON output
# ---------------------------------------------------------------------------

def build_results_json(
    model_metrics: dict,
    gp_uq: dict | None,
    pce_sobol: dict | None,
    coverage_json_path: str,
) -> dict:
    """Assemble the complete results dictionary."""
    # Load training coverage for reference
    training_coverage = None
    if os.path.isfile(coverage_json_path):
        with open(coverage_json_path) as f:
            cov = json.load(f)
        training_coverage = cov.get("k02_per_decade")

    return {
        "k02_bins": BIN_LABELS,
        "models": model_metrics,
        "gp_uq_correlation": gp_uq,
        "pce_sobol_k02": pce_sobol,
        "training_coverage": training_coverage,
    }


# ---------------------------------------------------------------------------
# 6b. Error-vs-k0_2 scatter/line plot
# ---------------------------------------------------------------------------

def plot_error_vs_k02(df: pd.DataFrame, output_path: str) -> None:
    """Two-panel scatter + binned median with IQR error bars."""
    bin_idx = assign_bins(df["k0_2"].values)
    log_k02 = np.log10(np.clip(df["k0_2"].values, 1e-20, None))
    bin_centers = np.array([-6.5, -5.5, -4.5, -3.5, -2.5, -1.5])

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    for ax_idx, out_label in enumerate(["cd", "pc"]):
        ax = axes[ax_idx]
        out_name = "Current Density" if out_label == "cd" else "Peroxide Current"

        for model, color in MODEL_COLORS.items():
            col = f"{model}_{out_label}_nrmse"
            vals = df[col].values

            # Scatter
            ax.scatter(
                log_k02, vals,
                c=color, alpha=0.15, s=12, edgecolors="none",
                label=f"{model} (scatter)" if ax_idx == 0 else None,
            )

            # Binned median + IQR
            medians, q25s, q75s = [], [], []
            for bi in range(len(BIN_LABELS)):
                mask = bin_idx == bi
                bv = vals[mask]
                if len(bv) == 0:
                    medians.append(np.nan)
                    q25s.append(np.nan)
                    q75s.append(np.nan)
                else:
                    medians.append(np.median(bv))
                    q25s.append(np.percentile(bv, 25))
                    q75s.append(np.percentile(bv, 75))
            medians = np.array(medians)
            q25s = np.array(q25s)
            q75s = np.array(q75s)

            ax.errorbar(
                bin_centers, medians,
                yerr=[medians - q25s, q75s - medians],
                color=color, marker="o", linewidth=2, capsize=4,
                label=model,
            )

        # Vertical bin edges
        for edge in BIN_EDGES_LOG10:
            ax.axvline(edge, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)

        ax.set_yscale("log")
        ax.set_ylabel(f"{out_name} NRMSE")
        ax.set_title(f"{out_name}")
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.2)

    axes[1].set_xlabel("log10(k0_2)")
    fig.suptitle("Surrogate Error vs k0_2 Value", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved error-vs-k0_2 plot: %s", output_path)


# ---------------------------------------------------------------------------
# 6c. Heatmap
# ---------------------------------------------------------------------------

def plot_heatmap(model_metrics: dict, output_path: str) -> None:
    """Side-by-side heatmaps of worst-case (max) NRMSE per model x bin."""
    model_names = list(MODEL_PREFIXES.keys())
    n_models = len(model_names)
    n_bins = len(BIN_LABELS)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    for ax_idx, out_label in enumerate(["cd", "pc"]):
        ax = axes[ax_idx]
        out_name = "Current Density" if out_label == "cd" else "Peroxide Current"

        data = np.zeros((n_models, n_bins))
        for mi, model in enumerate(model_names):
            for bi, label in enumerate(BIN_LABELS):
                val = model_metrics[model][out_label][label]["max"]
                data[mi, bi] = val if val is not None else np.nan

        # Replace zeros with small epsilon for log scale
        data_plot = np.where(data > 0, data, 1e-10)

        vmin = max(np.nanmin(data_plot), 1e-6)
        vmax = np.nanmax(data_plot)
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

        im = ax.imshow(data_plot, norm=norm, cmap="YlOrRd", aspect="auto")

        # Annotate cells
        for mi in range(n_models):
            for bi in range(n_bins):
                val = data[mi, bi]
                if np.isnan(val):
                    txt = "N/A"
                else:
                    txt = f"{val:.1e}"
                text_color = "white" if data_plot[mi, bi] > np.sqrt(vmin * vmax) else "black"
                ax.text(bi, mi, txt, ha="center", va="center", fontsize=7, color=text_color)

        ax.set_xticks(range(n_bins))
        ax.set_xticklabels(BIN_LABELS, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(n_models))
        ax.set_yticklabels(model_names, fontsize=8)
        ax.set_title(f"{out_name} (Max NRMSE)", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8, label="Max NRMSE")

    fig.suptitle("Worst-Case Error by Model and k0_2 Range", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved heatmap: %s", output_path)


# ---------------------------------------------------------------------------
# 6d. CSV table
# ---------------------------------------------------------------------------

def write_bin_table(model_metrics: dict, output_path: str) -> None:
    """Write flat CSV with one row per model-output-bin combination."""
    rows = []
    for model in MODEL_PREFIXES:
        for out in OUTPUT_TYPES:
            for label in BIN_LABELS:
                m = model_metrics[model][out][label]
                rows.append({
                    "model": model,
                    "output": out,
                    "bin": label,
                    "mean_nrmse": m["mean"],
                    "median_nrmse": m["median"],
                    "p95_nrmse": m["p95"],
                    "max_nrmse": m["max"],
                    "count": m["count"],
                })
    pd.DataFrame(rows).to_csv(output_path, index=False)
    logger.info("Saved bin table: %s (%d rows)", output_path, len(rows))


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="k0_2-stratified surrogate error analysis")
    p.add_argument(
        "--errors-csv",
        default=os.path.join(_PNPINVERSE_ROOT, "StudyResults/surrogate_fidelity/per_sample_errors.csv"),
    )
    p.add_argument(
        "--sobol-json",
        default=os.path.join(_PNPINVERSE_ROOT, "StudyResults/surrogate_fidelity/pce_sobol_indices.json"),
    )
    p.add_argument(
        "--coverage-json",
        default=os.path.join(_PNPINVERSE_ROOT, "StudyResults/training_data_audit/coverage_metrics.json"),
    )
    p.add_argument(
        "--output-dir",
        default=os.path.join(_PNPINVERSE_ROOT, "StudyResults/surrogate_fidelity/"),
    )
    p.add_argument(
        "--gp-model-dir",
        default=os.path.join(_PNPINVERSE_ROOT, "data/surrogate_models/gp_fixed/"),
    )
    p.add_argument(
        "--test-data",
        default=os.path.join(_PNPINVERSE_ROOT, "data/surrogate_models/training_data_merged.npz"),
    )
    p.add_argument(
        "--split-indices",
        default=os.path.join(_PNPINVERSE_ROOT, "data/surrogate_models/split_indices.npz"),
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load data
    df = load_errors(args.errors_csv)

    # 3. Compute per-bin error metrics
    model_metrics = compute_stratified_errors(df)

    # 4. GP uncertainty correlation
    gp_uq = gp_uq_correlation(
        args.gp_model_dir, args.test_data, args.split_indices, df,
    )
    if gp_uq is not None:
        logger.info("GP UQ correlation computed (overall CD rho=%.3f, PC rho=%.3f)",
                     gp_uq["overall"]["cd_rho"] or 0.0,
                     gp_uq["overall"]["pc_rho"] or 0.0)
    else:
        logger.info("GP UQ correlation skipped.")

    # 5. PCE Sobol cross-reference
    pce_sobol = load_pce_sobol(args.sobol_json)
    if pce_sobol:
        logger.info("PCE Sobol k0_2: CD mean=%.4f, PC mean=%.4f",
                     pce_sobol["cd_mean_first_order"],
                     pce_sobol["pc_mean_first_order"])

    # 6a. JSON output
    results = build_results_json(model_metrics, gp_uq, pce_sobol, args.coverage_json)
    json_path = os.path.join(args.output_dir, "k02_stratified_errors.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved JSON: %s", json_path)

    # 6b. Error-vs-k0_2 plot
    plot_error_vs_k02(df, os.path.join(args.output_dir, "k02_error_vs_value.png"))

    # 6c. Heatmap
    plot_heatmap(model_metrics, os.path.join(args.output_dir, "k02_error_heatmap.png"))

    # 6d. CSV table
    write_bin_table(model_metrics, os.path.join(args.output_dir, "k02_bin_table.csv"))

    logger.info("k0_2 stratified analysis complete.")


if __name__ == "__main__":
    main()
