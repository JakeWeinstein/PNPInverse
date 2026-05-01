"""Training Data Audit for Phase 1: Coverage, Density, and Gap Analysis.

Determines whether the 3,194-sample training set has coverage gaps, biases,
or insufficient density -- especially in k0_2 -- that explain surrogate bias.
Produces a go/no-go decision on data augmentation.

Go/no-go criteria:
  Augment if max-empty-ball > 0.15 OR any k0_2 log-decade has < 100 samples
  OR weak-signal k0_2 region has < 200 samples.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.spatial import KDTree
from scipy.stats import gaussian_kde, spearmanr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_NPZ = ROOT / "data" / "surrogate_models" / "training_data_merged.npz"
ERROR_CSV = ROOT / "StudyResults" / "surrogate_fidelity" / "per_sample_errors.csv"
OUT_DIR = ROOT / "StudyResults" / "training_data_audit"
NN_ENSEMBLE_DIR = ROOT / "data" / "surrogate_models" / "nn_ensemble" / "D3-deeper"

# Parameter names and bounds
PARAM_NAMES = ["k0_1", "k0_2", "alpha_1", "alpha_2"]
PARAM_BOUNDS = {
    "k0_1": (1e-6, 1.0),
    "k0_2": (1e-7, 0.1),
    "alpha_1": (0.1, 0.9),
    "alpha_2": (0.1, 0.9),
}
LOG_PARAMS = {"k0_1", "k0_2"}

# v11 focused bounds (from scripts/surrogate/overnight_train_v11.py)
V11_FOCUSED_BOUNDS = {
    "k0_1": (1e-4, 1e-1),
    "k0_2": (1e-5, 1e-2),
    "alpha_1": (0.2, 0.7),
    "alpha_2": (0.2, 0.7),
}


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def normalize_params(params: np.ndarray) -> np.ndarray:
    """Normalize 4D parameters to [0, 1]: log10 for k0, linear for alpha."""
    normed = np.empty_like(params)
    for i, name in enumerate(PARAM_NAMES):
        lo, hi = PARAM_BOUNDS[name]
        if name in LOG_PARAMS:
            normed[:, i] = (np.log10(params[:, i]) - np.log10(lo)) / (
                np.log10(hi) - np.log10(lo)
            )
        else:
            normed[:, i] = (params[:, i] - lo) / (hi - lo)
    return normed


def to_analysis_space(params: np.ndarray) -> np.ndarray:
    """Transform to analysis space: log10 for k0, linear for alpha."""
    result = params.copy()
    result[:, 0] = np.log10(params[:, 0])
    result[:, 1] = np.log10(params[:, 1])
    return result


# ---------------------------------------------------------------------------
# Step 1: Load and characterize
# ---------------------------------------------------------------------------

def load_and_characterize() -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Load training data and print summary statistics.

    Returns (all_params, v9_params, v11_params, n_v9).
    """
    data = np.load(DATA_NPZ, allow_pickle=True)
    params = data["parameters"]
    n_v9 = int(data["n_v9"])
    n_new = int(data["n_new"])

    v9 = params[:n_v9]
    v11 = params[n_v9:]

    print("=" * 60)
    print("STEP 1: Training Data Summary")
    print("=" * 60)
    print(f"Total samples : {len(params)}")
    print(f"v9 (seed=42)  : {n_v9}")
    print(f"v11 (new)     : {n_new}")
    print()

    for i, name in enumerate(PARAM_NAMES):
        col = params[:, i]
        print(f"  {name:10s}  min={col.min():.3e}  max={col.max():.3e}  "
              f"mean={col.mean():.3e}  std={col.std():.3e}")
        if name in LOG_PARAMS:
            log_col = np.log10(col)
            print(f"  {'(log10)':10s}  min={log_col.min():.3f}  max={log_col.max():.3f}  "
                  f"mean={log_col.mean():.3f}  std={log_col.std():.3f}")

    # Check bounds
    for i, name in enumerate(PARAM_NAMES):
        lo, hi = PARAM_BOUNDS[name]
        oob = np.sum((params[:, i] < lo) | (params[:, i] > hi))
        if oob > 0:
            print(f"  WARNING: {oob} samples outside declared bounds for {name}")

    print()
    return params, v9, v11, n_v9


# ---------------------------------------------------------------------------
# Step 2: 1D marginal histograms
# ---------------------------------------------------------------------------

def plot_marginals(params: np.ndarray, v9: np.ndarray, v11: np.ndarray) -> None:
    """Create 2x2 marginal histograms, one per parameter."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for idx, (name, ax) in enumerate(zip(PARAM_NAMES, axes)):
        lo, hi = PARAM_BOUNDS[name]

        if name in LOG_PARAMS:
            v9_vals = np.log10(v9[:, idx])
            v11_vals = np.log10(v11[:, idx])
            bins = np.linspace(np.log10(lo), np.log10(hi), 21)
            xlabel = f"log10({name})"
        else:
            v9_vals = v9[:, idx]
            v11_vals = v11[:, idx]
            bins = np.linspace(lo, hi, 21)
            xlabel = name

        total = len(v9_vals) + len(v11_vals)
        ideal = total / 20

        ax.hist(v9_vals, bins=bins, alpha=0.5, color="tab:blue", label="v9")
        ax.hist(v11_vals, bins=bins, alpha=0.5, color="tab:orange", label="v11")
        ax.axhline(ideal, color="gray", ls="--", lw=1, label=f"ideal={ideal:.0f}")

        # Count combined per-bin and flag sparse bins
        combined = np.concatenate([v9_vals, v11_vals])
        counts, _ = np.histogram(combined, bins=bins)
        for b_i, cnt in enumerate(counts):
            if cnt < 10:
                mid = 0.5 * (bins[b_i] + bins[b_i + 1])
                ax.annotate(
                    f"{cnt}",
                    (mid, cnt),
                    ha="center", va="bottom",
                    color="red", fontweight="bold", fontsize=9,
                )

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_title(f"Marginal: {name}")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "marginal_histograms.png", dpi=150)
    plt.close(fig)
    print("Saved marginal_histograms.png")


# ---------------------------------------------------------------------------
# Step 3: 2D pairwise scatter plots
# ---------------------------------------------------------------------------

def plot_pairwise(params: np.ndarray, v9: np.ndarray, v11: np.ndarray) -> None:
    """6-panel pairwise scatter with density coloring and bin grid overlay."""
    pairs = []
    for i in range(4):
        for j in range(i + 1, 4):
            pairs.append((i, j))

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.ravel()

    for panel, ((i, j), ax) in enumerate(zip(pairs, axes)):
        ni, nj = PARAM_NAMES[i], PARAM_NAMES[j]

        def _vals(arr, idx):
            if PARAM_NAMES[idx] in LOG_PARAMS:
                return np.log10(arr[:, idx])
            return arr[:, idx]

        def _bounds(idx):
            lo, hi = PARAM_BOUNDS[PARAM_NAMES[idx]]
            if PARAM_NAMES[idx] in LOG_PARAMS:
                return np.log10(lo), np.log10(hi)
            return lo, hi

        xi_all = _vals(params, i)
        xj_all = _vals(params, j)

        # KDE density
        try:
            xy = np.vstack([xi_all, xj_all])
            kde = gaussian_kde(xy)
            density = kde(xy)
        except Exception:
            density = np.ones(len(xi_all))

        ax.scatter(
            _vals(v9, i), _vals(v9, j),
            c=density[: len(v9)], cmap="viridis", s=8, alpha=0.6,
            marker="o", edgecolors="none", label="v9",
        )
        ax.scatter(
            _vals(v11, i), _vals(v11, j),
            c=density[len(v9):], cmap="viridis", s=8, alpha=0.6,
            marker="^", edgecolors="none", label="v11",
        )

        # 10x10 bin grid
        ilo, ihi = _bounds(i)
        jlo, jhi = _bounds(j)
        ibins = np.linspace(ilo, ihi, 11)
        jbins = np.linspace(jlo, jhi, 11)
        H, _, _ = np.histogram2d(xi_all, xj_all, bins=[ibins, jbins])

        for bi in range(10):
            for bj in range(10):
                mid_x = 0.5 * (ibins[bi] + ibins[bi + 1])
                mid_y = 0.5 * (jbins[bj] + jbins[bj + 1])
                cnt = int(H[bi, bj])
                if cnt == 0:
                    ax.plot(mid_x, mid_y, "rx", ms=6, mew=1.5)
                elif cnt <= 4:
                    ax.plot(
                        mid_x, mid_y, "s", ms=8,
                        mfc="none", mec="gold", mew=1.5,
                    )

        lbl_i = f"log10({ni})" if ni in LOG_PARAMS else ni
        lbl_j = f"log10({nj})" if nj in LOG_PARAMS else nj
        ax.set_xlabel(lbl_i)
        ax.set_ylabel(lbl_j)
        ax.set_title(f"{ni} vs {nj}")
        ax.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "pairwise_scatter.png", dpi=150)
    plt.close(fig)
    print("Saved pairwise_scatter.png")


# ---------------------------------------------------------------------------
# Step 4: Coverage heatmaps for k0_2
# ---------------------------------------------------------------------------

def compute_coverage_heatmaps(
    params: np.ndarray,
) -> Dict:
    """Bin k0_2 (and cross-params) and count samples per cell."""
    metrics: Dict = {}

    log_k02 = np.log10(params[:, 1])
    log_k01 = np.log10(params[:, 0])
    alpha1 = params[:, 2]
    alpha2 = params[:, 3]

    # --- 4a: k0_2 marginal density (12 bins, [-7, -1]) ---
    k02_bins = np.linspace(-7, -1, 13)
    k02_counts, _ = np.histogram(log_k02, bins=k02_bins)

    # Samples per log-decade (6 decades, 2 bins each)
    decade_labels = []
    decade_counts = []
    for d in range(6):
        lo = -7 + d
        hi = lo + 1
        mask = (log_k02 >= lo) & (log_k02 < hi)
        cnt = int(mask.sum())
        decade_labels.append(f"[1e{lo}, 1e{hi})")
        decade_counts.append(cnt)

    metrics["k02_marginal_bins"] = k02_counts.tolist()
    metrics["k02_per_decade"] = dict(zip(decade_labels, decade_counts))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(12), k02_counts, color="steelblue")
    ax.set_xticks(range(12))
    ax.set_xticklabels([f"{k02_bins[i]:.1f}" for i in range(12)], fontsize=8)
    ax.axhline(100, color="red", ls="--", label="100 threshold")
    ax.set_xlabel("log10(k0_2) bin left edge")
    ax.set_ylabel("Count")
    ax.set_title("k0_2 Marginal Density (12 bins)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "coverage_k02_marginal.png", dpi=150)
    plt.close(fig)

    # --- 4b-d: 2D heatmaps ---
    heatmap_specs = [
        ("k01", log_k01, np.linspace(-6, 0, 13), "log10(k0_1)"),
        ("alpha1", alpha1, np.linspace(0.1, 0.9, 9), "alpha_1"),
        ("alpha2", alpha2, np.linspace(0.1, 0.9, 9), "alpha_2"),
    ]

    for label, vals, other_bins, other_label in heatmap_specs:
        H, xedges, yedges = np.histogram2d(log_k02, vals, bins=[k02_bins, other_bins])
        n_empty = int((H == 0).sum())
        n_sparse = int((H < 5).sum())

        metrics[f"k02_vs_{label}"] = {
            "n_empty_bins": n_empty,
            "n_sparse_bins_lt5": n_sparse,
            "min_count": int(H.min()),
            "max_count": int(H.max()),
            "total_bins": int(H.size),
        }

        fig, ax = plt.subplots(figsize=(10, 7))
        im = ax.imshow(
            H.T, origin="lower", aspect="auto",
            extent=[k02_bins[0], k02_bins[-1], other_bins[0], other_bins[-1]],
            cmap="YlOrRd",
        )
        # Annotate counts
        for bi in range(H.shape[0]):
            for bj in range(H.shape[1]):
                mid_x = 0.5 * (k02_bins[bi] + k02_bins[bi + 1])
                mid_y = 0.5 * (other_bins[bj] + other_bins[bj + 1])
                cnt = int(H[bi, bj])
                color = "red" if cnt < 5 else ("white" if cnt > H.max() / 2 else "black")
                ax.text(mid_x, mid_y, str(cnt), ha="center", va="center",
                        fontsize=6, color=color, fontweight="bold" if cnt < 5 else "normal")

        plt.colorbar(im, ax=ax, label="Sample count")
        ax.set_xlabel("log10(k0_2)")
        ax.set_ylabel(other_label)
        ax.set_title(f"Coverage: k0_2 vs {label}  (empty={n_empty}, sparse<5={n_sparse})")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"coverage_k02_{label}.png", dpi=150)
        plt.close(fig)

    print("Saved coverage heatmaps")
    print(f"  k0_2 per-decade counts: {dict(zip(decade_labels, decade_counts))}")
    return metrics


# ---------------------------------------------------------------------------
# Step 5: Correlate surrogate error with sample density
# ---------------------------------------------------------------------------

def error_vs_density(
    train_params: np.ndarray, metrics: Dict,
) -> Dict:
    """Scatter local training density vs test-sample NRMSE."""
    import csv

    # Load test errors
    rows = []
    with open(ERROR_CSV) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    test_params = np.array([
        [float(r["k0_1"]), float(r["k0_2"]), float(r["alpha_1"]), float(r["alpha_2"])]
        for r in rows
    ])
    nn_cd_nrmse = np.array([float(r["cd_nrmse"]) for r in rows])
    nn_pc_nrmse = np.array([float(r["pc_nrmse"]) for r in rows])

    # Normalize ALL parameters to [0,1] (matching Step 8)
    train_normed = normalize_params(train_params)
    test_normed = normalize_params(test_params)

    tree = KDTree(train_normed)

    radii = [0.25, 0.5, 1.0]
    density_results = {}

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    for r_idx, radius in enumerate(radii):
        counts = tree.query_ball_point(test_normed, r=radius, return_length=True)
        counts = np.array(counts, dtype=float)

        rho_cd, p_cd = spearmanr(counts, nn_cd_nrmse)
        rho_pc, p_pc = spearmanr(counts, nn_pc_nrmse)

        density_results[f"r={radius}"] = {
            "spearman_cd": {"rho": float(rho_cd), "p": float(p_cd)},
            "spearman_pc": {"rho": float(rho_pc), "p": float(p_pc)},
            "mean_density": float(counts.mean()),
            "min_density": int(counts.min()),
            "max_density": int(counts.max()),
        }

        for obs_idx, (nrmse, label) in enumerate(
            [(nn_cd_nrmse, "CD"), (nn_pc_nrmse, "PC")]
        ):
            ax = axes[obs_idx, r_idx]
            ax.scatter(counts, nrmse, s=10, alpha=0.5, c="steelblue")
            ax.set_xlabel(f"Local density (r={radius})")
            ax.set_ylabel(f"NRMSE ({label})")
            rho = rho_cd if label == "CD" else rho_pc
            p = p_cd if label == "CD" else p_pc
            ax.set_title(f"{label}, r={radius}: rho={rho:.3f}, p={p:.2e}")

    fig.suptitle("Surrogate Error vs Training Sample Density (nn_ensemble)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "error_vs_density.png", dpi=150)
    plt.close(fig)

    # Top 5% worst samples analysis (using r=0.5)
    counts_05 = tree.query_ball_point(test_normed, r=0.5, return_length=True)
    counts_05 = np.array(counts_05, dtype=float)

    for label, nrmse in [("cd", nn_cd_nrmse), ("pc", nn_pc_nrmse)]:
        cutoff = np.percentile(nrmse, 95)
        worst_mask = nrmse >= cutoff
        worst_density = counts_05[worst_mask]
        rest_density = counts_05[~worst_mask]
        density_results[f"top5pct_worst_{label}"] = {
            "n_worst": int(worst_mask.sum()),
            "worst_mean_density": float(worst_density.mean()),
            "rest_mean_density": float(rest_density.mean()),
            "worst_in_low_density": int((worst_density < np.median(counts_05)).sum()),
        }

    metrics["error_density"] = density_results
    print("Saved error_vs_density.png")
    print(f"  Spearman (r=0.5): CD rho={density_results['r=0.5']['spearman_cd']['rho']:.3f}, "
          f"PC rho={density_results['r=0.5']['spearman_pc']['rho']:.3f}")
    return metrics


# ---------------------------------------------------------------------------
# Step 6: k0_2 sensitivity scan via nn_ensemble
# ---------------------------------------------------------------------------

def k02_sensitivity_scan(metrics: Dict) -> Dict:
    """Sweep k0_2 through nn_ensemble to find weak-signal sub-ranges."""
    try:
        sys.path.insert(0, str(ROOT))
        from Surrogate.ensemble import load_nn_ensemble

        model = load_nn_ensemble(str(NN_ENSEMBLE_DIR), n_members=5, device="cpu")
    except Exception as e:
        print(f"  WARNING: Could not load nn_ensemble: {e}")
        print("  Skipping sensitivity scan.")
        metrics["sensitivity_scan"] = {"status": "skipped", "reason": str(e)}
        return metrics

    # Midpoints
    k01_mid = np.sqrt(1e-6 * 1.0)  # geometric mean
    a1_mid = 0.5
    a2_mid = 0.5

    k02_sweep = np.logspace(-7, -1, 50)
    params_sweep = np.column_stack([
        np.full(50, k01_mid),
        k02_sweep,
        np.full(50, a1_mid),
        np.full(50, a2_mid),
    ])

    result = model.predict_batch(params_sweep)
    pc_curves = result["peroxide_current"]  # (50, 22)
    cd_curves = result["current_density"]

    # Signal range: max - min across voltage for each k0_2
    pc_range = pc_curves.max(axis=1) - pc_curves.min(axis=1)
    cd_range = cd_curves.max(axis=1) - cd_curves.min(axis=1)

    # Finite-difference sensitivity: |dPC/d(log k0_2)|
    log_k02 = np.log10(k02_sweep)
    pc_sensitivity = np.zeros(50)
    for i in range(50):
        k02_plus = k02_sweep[i] * 1.01
        k02_minus = k02_sweep[i] * 0.99
        p_plus = model.predict(k01_mid, k02_plus, a1_mid, a2_mid)
        p_minus = model.predict(k01_mid, k02_minus, a1_mid, a2_mid)
        dpc = np.linalg.norm(p_plus["peroxide_current"] - p_minus["peroxide_current"])
        dlog = np.log10(k02_plus) - np.log10(k02_minus)
        pc_sensitivity[i] = dpc / dlog if dlog > 0 else 0.0

    # Identify weak-signal region: where pc_range < 10% of max pc_range
    threshold = 0.1 * pc_range.max()
    weak_mask = pc_range < threshold
    if weak_mask.any():
        weak_lo = log_k02[weak_mask].min()
        weak_hi = log_k02[weak_mask].max()
        weak_region = (float(weak_lo), float(weak_hi))
    else:
        weak_region = None

    metrics["sensitivity_scan"] = {
        "status": "completed",
        "weak_signal_region_log10": weak_region,
        "pc_range_min": float(pc_range.min()),
        "pc_range_max": float(pc_range.max()),
    }

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    ax = axes[0]
    ax.plot(log_k02, pc_range, "b-o", ms=3, label="PC signal range")
    ax.axhline(threshold, color="red", ls="--", label=f"10% of max = {threshold:.2e}")
    if weak_region:
        ax.axvspan(weak_region[0], weak_region[1], alpha=0.2, color="red",
                   label=f"Weak: [{weak_region[0]:.1f}, {weak_region[1]:.1f}]")
    ax.set_xlabel("log10(k0_2)")
    ax.set_ylabel("PC signal range (max-min over voltage)")
    ax.set_title("Peroxide Current Signal Strength vs k0_2")
    ax.legend()

    ax = axes[1]
    ax.plot(log_k02, cd_range, "g-o", ms=3, label="CD signal range")
    ax.set_xlabel("log10(k0_2)")
    ax.set_ylabel("CD signal range")
    ax.set_title("Current Density Signal Strength vs k0_2")
    ax.legend()

    ax = axes[2]
    ax.plot(log_k02, pc_sensitivity, "r-o", ms=3, label="|dPC/d(log k0_2)|")
    ax.set_xlabel("log10(k0_2)")
    ax.set_ylabel("Sensitivity")
    ax.set_title("Local Sensitivity of PC to k0_2")
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUT_DIR / "k02_signal_strength.png", dpi=150)
    plt.close(fig)
    print(f"Saved k02_signal_strength.png  (weak region: {weak_region})")
    return metrics


# ---------------------------------------------------------------------------
# Step 7: Convergence failure analysis
# ---------------------------------------------------------------------------

def convergence_failure_analysis(
    params: np.ndarray, v11: np.ndarray, n_v9: int, metrics: Dict,
) -> Dict:
    """Attempt to reconstruct failed v11 samples by regenerating the design."""
    try:
        sys.path.insert(0, str(ROOT))
        from Surrogate.sampling import ParameterBounds, generate_multi_region_lhs_samples

        wide_bounds = ParameterBounds(
            k0_1_range=(1e-6, 1.0),
            k0_2_range=(1e-7, 0.1),
            alpha_1_range=(0.1, 0.9),
            alpha_2_range=(0.1, 0.9),
        )
        focused_bounds = ParameterBounds(
            k0_1_range=(1e-4, 1e-1),
            k0_2_range=(1e-5, 1e-2),
            alpha_1_range=(0.2, 0.7),
            alpha_2_range=(0.2, 0.7),
        )

        intended = generate_multi_region_lhs_samples(
            wide_bounds=wide_bounds,
            focused_bounds=focused_bounds,
            n_base=2000,
            n_focused=1000,
            seed_base=200,
            seed_focused=300,
            log_space_k0=True,
        )

        n_intended = len(intended)
        n_converged = len(v11)
        n_failed = n_intended - n_converged

        print(f"\nStep 7: Convergence Failure Analysis")
        print(f"  Intended v11 samples: {n_intended}")
        print(f"  Converged v11 samples: {n_converged}")
        print(f"  Failed: {n_failed} ({100*n_failed/n_intended:.1f}%)")

        # Match converged samples to intended to find failures
        # Use KD-tree matching in normalized space
        intended_normed = normalize_params(intended)
        converged_normed = normalize_params(v11)

        tree = KDTree(converged_normed)
        dists, _ = tree.query(intended_normed)
        # Samples with dist > small threshold are likely failures
        threshold = 0.01
        failed_mask = dists > threshold
        n_matched_failures = failed_mask.sum()
        failed_params = intended[failed_mask]

        print(f"  Matched failures: {n_matched_failures}")

        if n_matched_failures > 10:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            for ax_idx, (j, jname) in enumerate(
                [(0, "k0_1"), (2, "alpha_1"), (3, "alpha_2")]
            ):
                xi = np.log10(intended[:, 1])
                xj = np.log10(intended[:, j]) if jname in LOG_PARAMS else intended[:, j]

                xi_fail = np.log10(failed_params[:, 1])
                xj_fail = (np.log10(failed_params[:, j])
                           if jname in LOG_PARAMS else failed_params[:, j])

                ax = axes[ax_idx]
                ax.scatter(xi[~failed_mask], xj[~failed_mask],
                           s=4, alpha=0.3, c="steelblue", label="converged")
                ax.scatter(xi_fail, xj_fail,
                           s=12, alpha=0.8, c="red", marker="x", label="failed")
                ax.set_xlabel("log10(k0_2)")
                lbl = f"log10({jname})" if jname in LOG_PARAMS else jname
                ax.set_ylabel(lbl)
                ax.set_title(f"k0_2 vs {jname}")
                ax.legend(fontsize=8)

            fig.suptitle(f"Convergence Failures ({n_matched_failures} of {n_intended})")
            fig.tight_layout()
            fig.savefig(OUT_DIR / "convergence_failures.png", dpi=150)
            plt.close(fig)
            print("  Saved convergence_failures.png")

        # Where do failures cluster in k0_2 space?
        if n_matched_failures > 0:
            fail_log_k02 = np.log10(failed_params[:, 1])
            fail_per_decade = {}
            for d in range(6):
                lo = -7 + d
                hi = lo + 1
                cnt = int(((fail_log_k02 >= lo) & (fail_log_k02 < hi)).sum())
                fail_per_decade[f"[1e{lo}, 1e{hi})"] = cnt
            metrics["convergence_failures"] = {
                "n_intended": n_intended,
                "n_converged": n_converged,
                "n_failed": int(n_matched_failures),
                "failure_rate": float(n_matched_failures / n_intended),
                "failures_per_decade": fail_per_decade,
            }
        else:
            metrics["convergence_failures"] = {
                "status": "no_failures_detected",
                "n_intended": n_intended,
                "n_converged": n_converged,
            }

    except Exception as e:
        print(f"  WARNING: Could not reconstruct v11 design: {e}")
        metrics["convergence_failures"] = {"status": "infeasible", "reason": str(e)}

    return metrics


# ---------------------------------------------------------------------------
# Step 8: Max-empty-ball radius
# ---------------------------------------------------------------------------

def max_empty_ball(params: np.ndarray, metrics: Dict) -> Dict:
    """Compute the largest empty ball in normalized [0,1]^4 parameter space."""
    normed = normalize_params(params)
    tree = KDTree(normed)

    rng = np.random.default_rng(12345)
    n_candidates = 50_000
    candidates = rng.uniform(0, 1, size=(n_candidates, 4))

    dists, idxs = tree.query(candidates)
    max_idx = np.argmax(dists)
    max_radius = float(dists[max_idx])
    max_loc_normed = candidates[max_idx]

    # Convert back to physical space
    phys = np.empty(4)
    for i, name in enumerate(PARAM_NAMES):
        lo, hi = PARAM_BOUNDS[name]
        if name in LOG_PARAMS:
            phys[i] = 10.0 ** (np.log10(lo) + max_loc_normed[i] * (np.log10(hi) - np.log10(lo)))
        else:
            phys[i] = lo + max_loc_normed[i] * (hi - lo)

    metrics["max_empty_ball"] = {
        "radius": max_radius,
        "location_normed": max_loc_normed.tolist(),
        "location_physical": {
            name: float(phys[i]) for i, name in enumerate(PARAM_NAMES)
        },
        "n_candidates": n_candidates,
    }

    print(f"\nStep 8: Max-empty-ball radius = {max_radius:.4f}")
    print(f"  Location: {dict(zip(PARAM_NAMES, phys))}")
    if max_radius > 0.15:
        print("  *** EXCEEDS 0.15 THRESHOLD -- significant gap ***")
    else:
        print("  Within acceptable range (< 0.15)")

    return metrics


# ---------------------------------------------------------------------------
# Step 9: Generate diagnostic report
# ---------------------------------------------------------------------------

def generate_report(
    params: np.ndarray, v9: np.ndarray, v11: np.ndarray,
    n_v9: int, metrics: Dict,
) -> None:
    """Compile all findings into REPORT.md with go/no-go decision."""

    lines = ["# Training Data Audit Report", ""]
    lines.append(f"**Date:** 2026-03-16")
    lines.append(f"**Total samples:** {len(params)}")
    lines.append(f"**v9 batch:** {n_v9}  |  **v11 batch:** {len(v11)}")
    lines.append("")

    # --- 1. Summary Stats ---
    lines.append("## 1. Summary Statistics")
    lines.append("")
    lines.append("| Parameter | Min | Max | Mean | Std |")
    lines.append("|-----------|-----|-----|------|-----|")
    for i, name in enumerate(PARAM_NAMES):
        col = params[:, i]
        lines.append(f"| {name} | {col.min():.3e} | {col.max():.3e} | "
                     f"{col.mean():.3e} | {col.std():.3e} |")
    lines.append("")

    # --- 2. Marginal Distributions ---
    lines.append("## 2. Marginal Distributions")
    lines.append("")
    lines.append("See `marginal_histograms.png`.")
    lines.append("")

    # --- 3. Coverage Analysis ---
    lines.append("## 3. Coverage Analysis (k0_2)")
    lines.append("")
    lines.append("### Per-decade sample counts")
    lines.append("")
    lines.append("| Decade | Count | Status |")
    lines.append("|--------|-------|--------|")
    k02_decades = metrics.get("k02_per_decade", {})
    decade_under100 = []
    for decade, cnt in k02_decades.items():
        status = "OK" if cnt >= 100 else "**UNDER 100**"
        if cnt < 100:
            decade_under100.append((decade, cnt))
        lines.append(f"| {decade} | {cnt} | {status} |")
    lines.append("")

    for label in ["k01", "alpha1", "alpha2"]:
        key = f"k02_vs_{label}"
        if key in metrics:
            info = metrics[key]
            lines.append(f"### k0_2 vs {label}")
            lines.append(f"- Empty bins: {info['n_empty_bins']} / {info['total_bins']}")
            lines.append(f"- Sparse bins (<5): {info['n_sparse_bins_lt5']}")
            lines.append(f"- Min count: {info['min_count']}, Max count: {info['max_count']}")
            lines.append("")

    # --- 4. Error-density correlation ---
    lines.append("## 4. Error-Density Correlation")
    lines.append("")
    ed = metrics.get("error_density", {})
    for r_key in ["r=0.25", "r=0.5", "r=1.0"]:
        if r_key in ed:
            info = ed[r_key]
            lines.append(f"### {r_key}")
            lines.append(f"- CD: Spearman rho={info['spearman_cd']['rho']:.3f}, "
                         f"p={info['spearman_cd']['p']:.2e}")
            lines.append(f"- PC: Spearman rho={info['spearman_pc']['rho']:.3f}, "
                         f"p={info['spearman_pc']['p']:.2e}")
            lines.append("")

    for label in ["cd", "pc"]:
        key = f"top5pct_worst_{label}"
        if key in ed:
            info = ed[key]
            lines.append(f"### Top 5% worst {label.upper()} samples")
            lines.append(f"- N worst: {info['n_worst']}")
            lines.append(f"- Worst mean density: {info['worst_mean_density']:.1f}")
            lines.append(f"- Rest mean density: {info['rest_mean_density']:.1f}")
            lines.append(f"- Worst in low-density half: {info['worst_in_low_density']}")
            lines.append("")

    # --- 5. Sensitivity Analysis ---
    lines.append("## 5. Sensitivity Analysis")
    lines.append("")
    ss = metrics.get("sensitivity_scan", {})
    if ss.get("status") == "completed":
        wr = ss.get("weak_signal_region_log10")
        if wr:
            lines.append(f"- Weak-signal k0_2 region: log10 in [{wr[0]:.1f}, {wr[1]:.1f}]")
            lines.append(f"  (physical: [{10**wr[0]:.1e}, {10**wr[1]:.1e}])")
            # Count samples in weak region
            log_k02 = np.log10(params[:, 1])
            n_weak = int(((log_k02 >= wr[0]) & (log_k02 <= wr[1])).sum())
            lines.append(f"- Samples in weak-signal region: {n_weak}")
            metrics["weak_signal_sample_count"] = n_weak
        else:
            lines.append("- No weak-signal region identified (all k0_2 values produce signal)")
            metrics["weak_signal_sample_count"] = None
        lines.append(f"- PC signal range: [{ss['pc_range_min']:.2e}, {ss['pc_range_max']:.2e}]")
    else:
        lines.append(f"- Sensitivity scan skipped: {ss.get('reason', 'unknown')}")
        metrics["weak_signal_sample_count"] = None
    lines.append("")

    # --- 6. Max-empty-ball ---
    lines.append("## 6. Max-Empty-Ball")
    lines.append("")
    meb = metrics.get("max_empty_ball", {})
    radius = meb.get("radius", 0)
    lines.append(f"- Radius: {radius:.4f}")
    lines.append(f"- Location: {meb.get('location_physical', {})}")
    lines.append(f"- Threshold: 0.15")
    lines.append(f"- Status: {'**EXCEEDS THRESHOLD**' if radius > 0.15 else 'OK'}")
    lines.append("")

    # --- 7. Convergence Failures ---
    lines.append("## 7. Convergence Failures")
    lines.append("")
    cf = metrics.get("convergence_failures", {})
    if cf.get("status") == "infeasible":
        lines.append(f"Analysis infeasible: {cf.get('reason')}")
    elif cf.get("status") == "no_failures_detected":
        lines.append("No convergence failures detected (all intended samples matched).")
    else:
        lines.append(f"- Intended: {cf.get('n_intended', '?')}")
        lines.append(f"- Converged: {cf.get('n_converged', '?')}")
        lines.append(f"- Failed: {cf.get('n_failed', '?')} "
                     f"({100*cf.get('failure_rate', 0):.1f}%)")
        fpd = cf.get("failures_per_decade", {})
        if fpd:
            lines.append("")
            lines.append("| Decade | Failures |")
            lines.append("|--------|----------|")
            for decade, cnt in fpd.items():
                lines.append(f"| {decade} | {cnt} |")
    lines.append("")

    # --- 8. Go/No-Go Decision ---
    lines.append("## 8. Go/No-Go Decision")
    lines.append("")

    reasons_to_augment = []

    # Criterion 1: max-empty-ball > 0.15
    if radius > 0.15:
        reasons_to_augment.append(f"Max-empty-ball radius ({radius:.4f}) exceeds 0.15")

    # Criterion 2: any k0_2 log-decade has < 100 samples
    for decade, cnt in k02_decades.items():
        if cnt < 100:
            reasons_to_augment.append(f"k0_2 decade {decade} has only {cnt} samples (< 100)")

    # Criterion 3: weak-signal region has < 200 samples
    weak_count = metrics.get("weak_signal_sample_count")
    if weak_count is not None and weak_count < 200:
        reasons_to_augment.append(
            f"Weak-signal k0_2 region has only {weak_count} samples (< 200)"
        )

    if reasons_to_augment:
        lines.append("### DECISION: AUGMENTATION NEEDED")
        lines.append("")
        lines.append("Reasons:")
        for r in reasons_to_augment:
            lines.append(f"- {r}")
        lines.append("")

        # Augmentation plan
        lines.append("### Augmentation Plan")
        lines.append("")

        # Identify gap regions from coverage analysis
        lines.append("**Strategy:** Targeted LHS in under-sampled regions")
        lines.append("")

        # Target 1000-3000 samples depending on gaps
        n_target = min(3000, max(1000, sum(
            max(0, 100 - cnt) * 10 for cnt in k02_decades.values()
        )))
        lines.append(f"- **New samples:** ~{n_target}")
        lines.append(f"- **Estimated runtime:** ~{n_target * 3 / 3600:.1f} hours "
                     f"(at ~3s/sample)")
        lines.append("")

        # Focused bounds for gap regions
        if decade_under100:
            lines.append("**Focused sampling bounds for under-sampled decades:**")
            lines.append("")
            for decade, cnt in decade_under100:
                # Parse decade string like "[1e-7, 1e-6)"
                parts = decade.replace("[", "").replace(")", "").split(", ")
                lo_str, hi_str = parts[0], parts[1]
                lo_val = float(lo_str)
                hi_val = float(hi_str)
                lines.append(f"- {decade}: k0_2 in [{lo_val:.0e}, {hi_val:.0e}], "
                             f"need ~{max(0, 100 - cnt) * 3} additional samples")
            lines.append("")

        wr = ss.get("weak_signal_region_log10")
        if wr and weak_count is not None and weak_count < 200:
            lines.append("**Additional focused sampling for weak-signal region:**")
            lines.append(f"- k0_2 in [{10**wr[0]:.1e}, {10**wr[1]:.1e}]")
            lines.append(f"- Need ~{200 - weak_count} additional samples")
            lines.append("")

        lines.append("**Sampling function:** `generate_multi_region_lhs_samples()` "
                     "with focused bounds on identified gap regions")
        lines.append("**Full parameter bounds (wide):** Default ParameterBounds")
    else:
        lines.append("### DECISION: PROCEED WITH EXISTING DATA")
        lines.append("")
        lines.append("All criteria met:")
        lines.append(f"- Max-empty-ball radius ({radius:.4f}) <= 0.15")
        lines.append("- All k0_2 log-decades have >= 100 samples")
        if weak_count is not None:
            lines.append(f"- Weak-signal region has {weak_count} samples (>= 200)")

    lines.append("")

    report_path = OUT_DIR / "REPORT.md"
    report_path.write_text("\n".join(lines))
    print(f"\nSaved REPORT.md")

    # Also save full metrics JSON
    # Convert any numpy types
    def _jsonify(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    json_str = json.dumps(metrics, indent=2, default=_jsonify)
    (OUT_DIR / "coverage_metrics.json").write_text(json_str)
    print("Saved coverage_metrics.json")

    # Print decision
    print("\n" + "=" * 60)
    if reasons_to_augment:
        print("GO/NO-GO: *** AUGMENTATION NEEDED ***")
        for r in reasons_to_augment:
            print(f"  - {r}")
    else:
        print("GO/NO-GO: PROCEED WITH EXISTING DATA")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1
    params, v9, v11, n_v9 = load_and_characterize()

    # Steps 2-3 (independent)
    plot_marginals(params, v9, v11)
    plot_pairwise(params, v9, v11)

    # Step 4
    metrics = compute_coverage_heatmaps(params)

    # Step 5
    metrics = error_vs_density(params, metrics)

    # Step 6
    metrics = k02_sensitivity_scan(metrics)

    # Step 7
    metrics = convergence_failure_analysis(params, v11, n_v9, metrics)

    # Step 8
    metrics = max_empty_ball(params, metrics)

    # Step 9
    generate_report(params, v9, v11, n_v9, metrics)


if __name__ == "__main__":
    main()
