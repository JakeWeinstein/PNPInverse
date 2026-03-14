"""Multi-seed wrapper for v13 pipeline robustness assessment.

Runs the v13 master inference pipeline across multiple noise seeds at a
specified noise level, then aggregates per-parameter recovery errors and
generates summary statistics and visualizations.

This establishes the v13 performance baseline (DIAG-01) that all later
diagnostic phases compare against.

Usage::

    # Full 20-seed run at 2% noise (~2-3 hours)
    python scripts/studies/run_multi_seed_v13.py

    # Quick 3-seed test
    python scripts/studies/run_multi_seed_v13.py --num-seeds 3

    # Custom noise level
    python scripts/studies/run_multi_seed_v13.py --noise-percent 5.0
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


def _log(tag: str, msg: str) -> None:
    """Print tagged log message following codebase convention."""
    print(f"[{tag}] {msg}")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MultiSeedConfig:
    """Configuration for multi-seed v13 robustness assessment."""

    num_seeds: int = 20
    noise_percent: float = 2.0
    seed_start: int = 0
    v13_script_path: str = field(
        default_factory=lambda: os.path.join(
            _THIS_DIR, "..", "Inference",
            "Infer_BVMaster_charged_v13_ultimate.py",
        )
    )
    output_dir: str = field(
        default_factory=lambda: os.path.join(_ROOT, "StudyResults", "v14", "multi_seed")
    )
    timeout_per_seed: int = 900


# ---------------------------------------------------------------------------
# Error-percentage column names
# ---------------------------------------------------------------------------

ERR_COLS = ["k0_1_err_pct", "k0_2_err_pct", "alpha_1_err_pct", "alpha_2_err_pct"]
PARAM_COLS = ["k0_1", "k0_2", "alpha_1", "alpha_2"]


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

def parse_v13_csv(csv_path: str) -> dict | None:
    """Extract the P2 (full-cathodic PDE) row from a v13 output CSV.

    Parameters
    ----------
    csv_path:
        Path to ``master_comparison_v13.csv``.

    Returns
    -------
    dict or None
        Dict with keys matching CSV columns for the P2 row, or None if no
        P2 row is found.
    """
    if not os.path.isfile(csv_path):
        _log("PARSE", f"CSV not found: {csv_path}")
        return None

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            phase = row.get("phase", "")
            if phase.startswith("P2"):
                # Convert numeric fields to float
                result = dict(row)
                for col in ERR_COLS + PARAM_COLS + ["loss", "time_s"]:
                    if col in result:
                        result[col] = float(result[col])
                return result

    _log("PARSE", f"No P2 row found in {csv_path}")
    return None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_seed_results(results: list[dict]) -> dict:
    """Compute per-parameter summary statistics across seeds.

    Parameters
    ----------
    results:
        List of P2 result dicts (one per seed), each containing the
        ``*_err_pct`` keys as floats.

    Returns
    -------
    dict
        Nested dict: ``{err_col: {median, p25, p75, max}}``.
    """
    stats: dict[str, dict[str, float]] = {}
    for col in ERR_COLS:
        values = np.array([r[col] for r in results], dtype=np.float64)
        stats[col] = {
            "median": float(np.median(values)),
            "p25": float(np.percentile(values, 25)),
            "p75": float(np.percentile(values, 75)),
            "max": float(np.max(values)),
        }
    return stats


# ---------------------------------------------------------------------------
# Single seed runner
# ---------------------------------------------------------------------------

def run_single_seed(seed: int, config: MultiSeedConfig) -> dict | None:
    """Run the v13 pipeline for a single noise seed.

    Parameters
    ----------
    seed:
        Noise seed to pass via ``--noise-seed``.
    config:
        Multi-seed configuration.

    Returns
    -------
    dict or None
        Parsed P2 result dict, or None on failure.
    """
    cmd = [
        sys.executable,
        config.v13_script_path,
        "--noise-seed", str(seed),
        "--noise-percent", str(config.noise_percent),
    ]
    _log("SEED", f"Running seed {seed}: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=_ROOT,
            capture_output=True,
            text=True,
            timeout=config.timeout_per_seed,
        )
    except subprocess.TimeoutExpired:
        _log("SEED", f"Seed {seed} timed out after {config.timeout_per_seed}s")
        return None

    if result.returncode != 0:
        _log("SEED", f"Seed {seed} failed (rc={result.returncode})")
        _log("STDERR", result.stderr[-500:] if result.stderr else "(empty)")
        return None

    # Parse the output CSV
    csv_path = os.path.join(
        _ROOT, "StudyResults", "master_inference_v13", "master_comparison_v13.csv"
    )
    p2_result = parse_v13_csv(csv_path)
    if p2_result is not None:
        p2_result["seed"] = seed
    return p2_result


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def generate_plots(seed_results: list[dict], output_dir: str) -> None:
    """Generate box plots and per-seed scatter plots of error percentages.

    Parameters
    ----------
    seed_results:
        List of P2 result dicts (one per successful seed).
    output_dir:
        Directory to save PNG files.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_seeds = len(seed_results)
    if n_seeds == 0:
        _log("PLOT", "No results to plot")
        return

    # --- Box plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    data = []
    labels = []
    for col in ERR_COLS:
        vals = [r[col] for r in seed_results]
        data.append(vals)
        labels.append(col.replace("_err_pct", ""))

    ax.boxplot(data, labels=labels)
    ax.set_ylabel("Relative Error (%)")
    ax.set_title(f"Parameter Recovery Errors ({n_seeds} seeds)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "boxplot_errors.png"), dpi=150)
    plt.close(fig)
    _log("PLOT", "Saved boxplot_errors.png")

    # --- Per-seed scatter ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes_flat = axes.flatten()
    seeds = [r.get("seed", i) for i, r in enumerate(seed_results)]

    for idx, col in enumerate(ERR_COLS):
        ax = axes_flat[idx]
        vals = np.array([r[col] for r in seed_results])
        med = np.median(vals)
        outlier_mask = vals > 2.0 * med

        ax.scatter(
            np.array(seeds)[~outlier_mask],
            vals[~outlier_mask],
            c="steelblue", s=30, label="normal",
        )
        if outlier_mask.any():
            ax.scatter(
                np.array(seeds)[outlier_mask],
                vals[outlier_mask],
                c="red", s=50, marker="x", label="outlier (>2x median)",
            )
        ax.axhline(med, color="gray", ls="--", lw=0.8, label=f"median={med:.2f}")
        ax.set_ylabel("Error (%)")
        ax.set_title(col.replace("_err_pct", ""))
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    for ax in axes_flat[2:]:
        ax.set_xlabel("Seed")
    fig.suptitle(f"Per-Seed Recovery Errors ({n_seeds} seeds)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "scatter_per_seed.png"), dpi=150)
    plt.close(fig)
    _log("PLOT", "Saved scatter_per_seed.png")


# ---------------------------------------------------------------------------
# Summary CSV output
# ---------------------------------------------------------------------------

def write_summary_csv(
    seed_results: list[dict],
    summary_stats: dict,
    output_dir: str,
) -> None:
    """Write per-seed results CSV and summary statistics CSV.

    Parameters
    ----------
    seed_results:
        List of P2 result dicts (one per seed).
    summary_stats:
        Output of :func:`aggregate_seed_results`.
    output_dir:
        Directory to save CSVs.
    """
    # Per-seed CSV
    seed_csv = os.path.join(output_dir, "seed_results.csv")
    header = ["seed"] + PARAM_COLS + ERR_COLS
    with open(seed_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for r in seed_results:
            writer.writerow(r)
    _log("CSV", f"Saved {seed_csv}")

    # Summary statistics CSV
    summary_csv = os.path.join(output_dir, "summary_statistics.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter", "median_err_pct", "p25_err_pct", "p75_err_pct", "max_err_pct"])
        for col in ERR_COLS:
            s = summary_stats[col]
            writer.writerow([
                col.replace("_err_pct", ""),
                f"{s['median']:.4f}",
                f"{s['p25']:.4f}",
                f"{s['p75']:.4f}",
                f"{s['max']:.4f}",
            ])
    _log("CSV", f"Saved {summary_csv}")


# ---------------------------------------------------------------------------
# AUDT-04 metadata sidecar
# ---------------------------------------------------------------------------

def write_metadata(output_dir: str, **kwargs: Any) -> None:
    """Write AUDT-04 diagnostic metadata JSON sidecar.

    Parameters
    ----------
    output_dir:
        Directory to write ``metadata.json``.
    **kwargs:
        Override default metadata fields (e.g., n_seeds, noise_percent).
    """
    n_seeds = kwargs.get("n_seeds", 20)
    noise_percent = kwargs.get("noise_percent", 2.0)

    metadata = {
        "tool_name": "Multi-Seed Pipeline Robustness Assessment",
        "phase": "07-baseline-diagnostics",
        "requirement": "DIAG-01",
        "justification_type": "empirical",
        "reference": (
            "Standard practice in inverse problems: test parameter recovery "
            "across noise realizations to assess estimator variance"
        ),
        "rationale": (
            "Running across 20 noise seeds at 2% noise quantifies pipeline "
            "sensitivity to noise realization and identifies parameters with "
            "high variance"
        ),
        "parameters": {
            "n_seeds": n_seeds,
            "noise_percent": noise_percent,
            "pipeline_version": "v13",
        },
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    os.makedirs(output_dir, exist_ok=True)
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    _log("META", f"Saved {meta_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run multi-seed v13 pipeline assessment."""
    parser = argparse.ArgumentParser(
        description="Multi-seed v13 pipeline robustness assessment (DIAG-01)"
    )
    parser.add_argument(
        "--num-seeds", type=int, default=20,
        help="Number of noise seeds to run (default: 20)",
    )
    parser.add_argument(
        "--noise-percent", type=float, default=2.0,
        help="Noise level in percent (default: 2.0)",
    )
    parser.add_argument(
        "--seed-start", type=int, default=0,
        help="Starting seed index (default: 0)",
    )
    parser.add_argument(
        "--timeout", type=int, default=900,
        help="Timeout per seed in seconds (default: 900)",
    )
    args = parser.parse_args()

    config = MultiSeedConfig(
        num_seeds=args.num_seeds,
        noise_percent=args.noise_percent,
        seed_start=args.seed_start,
        timeout_per_seed=args.timeout,
    )

    _log("MAIN", f"Multi-Seed v13 Assessment: {config.num_seeds} seeds, "
         f"{config.noise_percent}% noise, starting at seed {config.seed_start}")

    os.makedirs(config.output_dir, exist_ok=True)

    # Run each seed sequentially
    seed_results: list[dict] = []
    failed_seeds: list[int] = []

    for i in range(config.num_seeds):
        seed = config.seed_start + i
        _log("MAIN", f"--- Seed {seed} ({i + 1}/{config.num_seeds}) ---")
        result = run_single_seed(seed, config)
        if result is not None:
            seed_results.append(result)
        else:
            failed_seeds.append(seed)
            _log("MAIN", f"Seed {seed} failed or had no P2 row, skipping")

    _log("MAIN", f"Completed: {len(seed_results)}/{config.num_seeds} seeds successful")

    if not seed_results:
        _log("MAIN", "No successful seeds -- cannot generate summary")
        sys.exit(1)

    # Aggregate statistics
    summary_stats = aggregate_seed_results(seed_results)

    # Print summary
    _log("SUMMARY", "Per-parameter recovery error statistics:")
    for col in ERR_COLS:
        s = summary_stats[col]
        param = col.replace("_err_pct", "")
        _log("SUMMARY", f"  {param:>8s}: median={s['median']:6.2f}%  "
             f"IQR=[{s['p25']:6.2f}, {s['p75']:6.2f}]%  max={s['max']:6.2f}%")

    if failed_seeds:
        _log("SUMMARY", f"Failed seeds: {failed_seeds}")

    # Generate outputs
    write_summary_csv(seed_results, summary_stats, config.output_dir)
    generate_plots(seed_results, config.output_dir)
    write_metadata(
        config.output_dir,
        n_seeds=config.num_seeds,
        noise_percent=config.noise_percent,
    )

    _log("MAIN", f"All outputs saved to {config.output_dir}/")
    _log("MAIN", "Multi-Seed v13 Assessment Complete")


if __name__ == "__main__":
    main()
