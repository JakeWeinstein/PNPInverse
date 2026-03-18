"""End-to-end inverse benchmark: parameter recovery across all surrogate models.

Runs multistart + cascade inference for each surrogate model against synthetic
targets with known ground truth, varying noise levels. Measures actual parameter
recovery performance under realistic conditions.

Extends the pattern from ``parameter_recovery_all_models.py`` with:
- Multiple synthetic targets (LHS in log-k0 space)
- Multiplicative Gaussian noise at 0%, 1%, 2%
- Per-parameter per-model per-noise-level error reporting
- Wall-clock timing
- Comparison against v12/v13 baselines

Usage::

    source ../venv-firedrake/bin/activate
    python scripts/studies/inverse_benchmark_all_models.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from scipy.stats.qmc import LatinHypercube

from scripts._bv_common import (
    K0_HAT_R1,
    K0_HAT_R2,
    ALPHA_R1,
    ALPHA_R2,
)
from Surrogate.sampling import ParameterBounds
from Surrogate.multistart import MultiStartConfig, run_multistart_inference
from Surrogate.cascade import CascadeConfig, run_cascade_inference

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRUE_K0_1 = K0_HAT_R1
TRUE_K0_2 = K0_HAT_R2
TRUE_ALPHA_1 = ALPHA_R1
TRUE_ALPHA_2 = ALPHA_R2

BOUNDS = ParameterBounds()

_SURROGATE_DIR = os.path.join(_ROOT, "data", "surrogate_models")
_TARGET_CACHE = os.path.join(
    _ROOT, "StudyResults", "target_cache", "clean_targets_1b38f0b9a75d58d3.npz"
)
_OUTPUT_DIR = os.path.join(_ROOT, "StudyResults", "inverse_benchmark")

# Noise configuration
NOISE_LEVELS = [0.0, 1.0, 2.0]   # percent
NOISE_SEEDS = [42, 43, 44]        # 3 realizations per noise level

# LHS target generation
N_LHS_TARGETS = 7                  # + 1 standard target = 8 total
LHS_SEED = 123
LHS_LOG_K0_1_BOUNDS = (-4.0, -1.0)  # log10 space
LHS_LOG_K0_2_BOUNDS = (-5.0, -2.0)  # log10 space
LHS_ALPHA_BOUNDS = (0.2, 0.8)


# ---------------------------------------------------------------------------
# Result container (frozen/immutable)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BenchmarkRow:
    """Single benchmark result row."""
    target_id: int
    noise_pct: float
    seed: int
    model_name: str
    method: str
    k0_1_true: float
    k0_1_recovered: float
    k0_1_error_pct: float
    k0_2_true: float
    k0_2_recovered: float
    k0_2_error_pct: float
    alpha_1_true: float
    alpha_1_recovered: float
    alpha_1_error_pct: float
    alpha_2_true: float
    alpha_2_recovered: float
    alpha_2_error_pct: float
    max_error_pct: float
    time_s: float


@dataclass(frozen=True)
class SyntheticTarget:
    """Immutable synthetic target with known ground truth."""
    target_id: int
    k0_1: float
    k0_2: float
    alpha_1: float
    alpha_2: float
    cd_clean: np.ndarray
    pc_clean: np.ndarray


def _pct_error(true: float, recovered: float) -> float:
    return abs(recovered - true) / max(abs(true), 1e-30) * 100.0


def _make_benchmark_row(
    target: SyntheticTarget,
    noise_pct: float,
    seed: int,
    model_name: str,
    method: str,
    k0_1: float,
    k0_2: float,
    alpha_1: float,
    alpha_2: float,
    elapsed: float,
) -> BenchmarkRow:
    e1 = _pct_error(target.k0_1, k0_1)
    e2 = _pct_error(target.k0_2, k0_2)
    e3 = _pct_error(target.alpha_1, alpha_1)
    e4 = _pct_error(target.alpha_2, alpha_2)
    return BenchmarkRow(
        target_id=target.target_id,
        noise_pct=noise_pct,
        seed=seed,
        model_name=model_name,
        method=method,
        k0_1_true=target.k0_1,
        k0_1_recovered=k0_1,
        k0_1_error_pct=e1,
        k0_2_true=target.k0_2,
        k0_2_recovered=k0_2,
        k0_2_error_pct=e2,
        alpha_1_true=target.alpha_1,
        alpha_1_recovered=alpha_1,
        alpha_1_error_pct=e3,
        alpha_2_true=target.alpha_2,
        alpha_2_recovered=alpha_2,
        alpha_2_error_pct=e4,
        max_error_pct=max(e1, e2, e3, e4),
        time_s=elapsed,
    )


# ---------------------------------------------------------------------------
# Synthetic target generation
# ---------------------------------------------------------------------------
def generate_synthetic_targets(reference_surrogate) -> List[SyntheticTarget]:
    """Generate 8 synthetic targets with known ground truth.

    Target 0: standard true parameters.
    Targets 1-7: LHS samples in log-k0 space.

    Uses the reference_surrogate (RBF baseline) to generate "ground truth"
    I-V curves, avoiding inverse crime.
    """
    targets = []

    # Target 0: standard true parameters
    pred = reference_surrogate.predict(TRUE_K0_1, TRUE_K0_2, TRUE_ALPHA_1, TRUE_ALPHA_2)
    targets.append(SyntheticTarget(
        target_id=0,
        k0_1=TRUE_K0_1,
        k0_2=TRUE_K0_2,
        alpha_1=TRUE_ALPHA_1,
        alpha_2=TRUE_ALPHA_2,
        cd_clean=pred["current_density"].copy(),
        pc_clean=pred["peroxide_current"].copy(),
    ))

    # Targets 1-7: LHS in log-k0 space
    sampler = LatinHypercube(d=4, seed=LHS_SEED)
    unit_samples = sampler.random(n=N_LHS_TARGETS)

    for i in range(N_LHS_TARGETS):
        log_k0_1 = LHS_LOG_K0_1_BOUNDS[0] + unit_samples[i, 0] * (
            LHS_LOG_K0_1_BOUNDS[1] - LHS_LOG_K0_1_BOUNDS[0]
        )
        log_k0_2 = LHS_LOG_K0_2_BOUNDS[0] + unit_samples[i, 1] * (
            LHS_LOG_K0_2_BOUNDS[1] - LHS_LOG_K0_2_BOUNDS[0]
        )
        alpha_1 = LHS_ALPHA_BOUNDS[0] + unit_samples[i, 2] * (
            LHS_ALPHA_BOUNDS[1] - LHS_ALPHA_BOUNDS[0]
        )
        alpha_2 = LHS_ALPHA_BOUNDS[0] + unit_samples[i, 3] * (
            LHS_ALPHA_BOUNDS[1] - LHS_ALPHA_BOUNDS[0]
        )

        k0_1 = 10.0 ** log_k0_1
        k0_2 = 10.0 ** log_k0_2

        try:
            pred = reference_surrogate.predict(k0_1, k0_2, alpha_1, alpha_2)
            targets.append(SyntheticTarget(
                target_id=i + 1,
                k0_1=k0_1,
                k0_2=k0_2,
                alpha_1=alpha_1,
                alpha_2=alpha_2,
                cd_clean=pred["current_density"].copy(),
                pc_clean=pred["peroxide_current"].copy(),
            ))
        except Exception as e:
            print(f"  WARNING: Failed to generate target {i+1}: {e}")
            continue

    return targets


def add_noise(
    cd_clean: np.ndarray,
    pc_clean: np.ndarray,
    noise_pct: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Add multiplicative Gaussian noise to I-V curves.

    Returns new arrays (immutable pattern -- does not modify inputs).
    """
    if noise_pct == 0.0:
        return cd_clean.copy(), pc_clean.copy()

    rng = np.random.RandomState(seed)
    cd_noisy = cd_clean * (1.0 + noise_pct / 100.0 * rng.randn(len(cd_clean)))
    pc_noisy = pc_clean * (1.0 + noise_pct / 100.0 * rng.randn(len(pc_clean)))
    return cd_noisy, pc_noisy


# ---------------------------------------------------------------------------
# Model loaders (reuse existing patterns)
# ---------------------------------------------------------------------------
def _load_nn_ensemble(design: str):
    """Load an NN ensemble model by design name."""
    from Surrogate.ensemble import load_nn_ensemble
    path = os.path.join(_SURROGATE_DIR, "nn_ensemble", design)
    return load_nn_ensemble(path, n_members=5)


class _GPFDWrapper:
    """Wrapper that hides predict_torch from GP to force FD gradient path."""

    def __init__(self, gp_model):
        self._gp = gp_model

    def predict(self, k0_1, k0_2, alpha_1, alpha_2):
        return self._gp.predict(k0_1, k0_2, alpha_1, alpha_2)

    def predict_batch(self, parameters):
        return self._gp.predict_batch(parameters)

    @property
    def phi_applied(self):
        return self._gp.phi_applied

    @property
    def n_eta(self):
        return self._gp.n_eta

    @property
    def is_fitted(self):
        return self._gp.is_fitted

    @property
    def training_bounds(self):
        return self._gp.training_bounds


def _load_gp():
    """Load the GP surrogate model (wrapped to force FD gradient path)."""
    from Surrogate.gp_model import load_gp_surrogate
    gp = load_gp_surrogate(os.path.join(_SURROGATE_DIR, "gp_fixed"))
    return _GPFDWrapper(gp)


def _load_pkl(filename: str):
    """Load a pickle-based surrogate model."""
    import pickle
    path = os.path.join(_SURROGATE_DIR, filename)
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"  Surrogate loaded: {path} ({type(model).__name__})")
    return model


def _load_pce():
    """Load the PCE surrogate model."""
    import pickle
    path = os.path.join(_SURROGATE_DIR, "pce", "pce_model.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Inference runners
# ---------------------------------------------------------------------------
def run_multistart_for_target(
    model,
    model_name: str,
    target: SyntheticTarget,
    target_cd: np.ndarray,
    target_pc: np.ndarray,
    noise_pct: float,
    seed: int,
    n_grid: int = 20_000,
) -> BenchmarkRow:
    """Run multistart inference and return a BenchmarkRow."""
    config = MultiStartConfig(
        n_grid=n_grid,
        n_top_candidates=20,
        polish_maxiter=60,
        secondary_weight=1.0,
        seed=42,
        verbose=False,
    )
    t0 = time.time()
    result = run_multistart_inference(
        surrogate=model,
        target_cd=target_cd,
        target_pc=target_pc,
        bounds_k0_1=BOUNDS.k0_1_range,
        bounds_k0_2=BOUNDS.k0_2_range,
        bounds_alpha=BOUNDS.alpha_1_range,
        config=config,
        subset_idx=None,
    )
    elapsed = time.time() - t0
    return _make_benchmark_row(
        target, noise_pct, seed, model_name, "multistart",
        result.best_k0_1, result.best_k0_2,
        result.best_alpha_1, result.best_alpha_2,
        elapsed,
    )


def run_cascade_for_target(
    model,
    model_name: str,
    target: SyntheticTarget,
    target_cd: np.ndarray,
    target_pc: np.ndarray,
    noise_pct: float,
    seed: int,
) -> BenchmarkRow:
    """Run cascade inference and return a BenchmarkRow."""
    init_k0 = [
        np.sqrt(BOUNDS.k0_1_range[0] * BOUNDS.k0_1_range[1]),
        np.sqrt(BOUNDS.k0_2_range[0] * BOUNDS.k0_2_range[1]),
    ]
    init_alpha = [
        0.5 * (BOUNDS.alpha_1_range[0] + BOUNDS.alpha_1_range[1]),
        0.5 * (BOUNDS.alpha_2_range[0] + BOUNDS.alpha_2_range[1]),
    ]

    config = CascadeConfig(
        pass1_weight=0.5,
        pass2_weight=2.0,
        pass1_maxiter=60,
        pass2_maxiter=60,
        polish_maxiter=30,
        polish_weight=1.0,
        skip_polish=False,
        verbose=False,
    )
    t0 = time.time()
    result = run_cascade_inference(
        surrogate=model,
        target_cd=target_cd,
        target_pc=target_pc,
        initial_k0=init_k0,
        initial_alpha=init_alpha,
        bounds_k0_1=BOUNDS.k0_1_range,
        bounds_k0_2=BOUNDS.k0_2_range,
        bounds_alpha=BOUNDS.alpha_1_range,
        config=config,
        subset_idx=None,
    )
    elapsed = time.time() - t0
    return _make_benchmark_row(
        target, noise_pct, seed, model_name, "cascade",
        result.best_k0_1, result.best_k0_2,
        result.best_alpha_1, result.best_alpha_2,
        elapsed,
    )


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# (name, loader_fn, is_slow)
MODEL_REGISTRY = [
    ("NN D1-default", lambda: _load_nn_ensemble("D1-default"), False),
    ("NN D2-wider", lambda: _load_nn_ensemble("D2-wider"), False),
    ("NN D3-deeper", lambda: _load_nn_ensemble("D3-deeper"), False),
    ("NN D4-no-physics", lambda: _load_nn_ensemble("D4-no-physics"), False),
    ("NN D5-strong-physics", lambda: _load_nn_ensemble("D5-strong-physics"), False),
    ("GP", _load_gp, True),
    ("POD-RBF log", lambda: _load_pkl("model_pod_rbf_log.pkl"), False),
    ("POD-RBF nolog", lambda: _load_pkl("model_pod_rbf_nolog.pkl"), False),
    ("RBF baseline", lambda: _load_pkl("model_rbf_baseline.pkl"), False),
    ("PCE", _load_pce, False),
]


# ---------------------------------------------------------------------------
# CSV / JSON output
# ---------------------------------------------------------------------------
def save_recovery_csv(rows: List[BenchmarkRow], path: str) -> None:
    """Save recovery table to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "target_id", "noise_pct", "seed", "model_name", "method",
        "k0_1_true", "k0_1_recovered", "k0_1_error_pct",
        "k0_2_true", "k0_2_recovered", "k0_2_error_pct",
        "alpha_1_true", "alpha_1_recovered", "alpha_1_error_pct",
        "alpha_2_true", "alpha_2_recovered", "alpha_2_error_pct",
        "max_error_pct", "time_s",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "target_id": row.target_id,
                "noise_pct": f"{row.noise_pct:.1f}",
                "seed": row.seed,
                "model_name": row.model_name,
                "method": row.method,
                "k0_1_true": f"{row.k0_1_true:.10e}",
                "k0_1_recovered": f"{row.k0_1_recovered:.10e}",
                "k0_1_error_pct": f"{row.k0_1_error_pct:.4f}",
                "k0_2_true": f"{row.k0_2_true:.10e}",
                "k0_2_recovered": f"{row.k0_2_recovered:.10e}",
                "k0_2_error_pct": f"{row.k0_2_error_pct:.4f}",
                "alpha_1_true": f"{row.alpha_1_true:.6f}",
                "alpha_1_recovered": f"{row.alpha_1_recovered:.6f}",
                "alpha_1_error_pct": f"{row.alpha_1_error_pct:.4f}",
                "alpha_2_true": f"{row.alpha_2_true:.6f}",
                "alpha_2_recovered": f"{row.alpha_2_recovered:.6f}",
                "alpha_2_error_pct": f"{row.alpha_2_error_pct:.4f}",
                "max_error_pct": f"{row.max_error_pct:.4f}",
                "time_s": f"{row.time_s:.2f}",
            })
    print(f"\nRecovery table saved: {path} ({len(rows)} rows)")


def save_timing_csv(rows: List[BenchmarkRow], path: str) -> None:
    """Aggregate and save timing table."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Group by (model_name, method)
    timing: Dict[Tuple[str, str], List[float]] = {}
    for row in rows:
        key = (row.model_name, row.method)
        timing.setdefault(key, []).append(row.time_s)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model_name", "method", "mean_time_s", "std_time_s", "n_runs",
        ])
        writer.writeheader()
        for (model_name, method), times in sorted(timing.items()):
            writer.writerow({
                "model_name": model_name,
                "method": method,
                "mean_time_s": f"{np.mean(times):.2f}",
                "std_time_s": f"{np.std(times):.2f}",
                "n_runs": len(times),
            })
    print(f"Timing table saved: {path}")


def save_summary_json(rows: List[BenchmarkRow], path: str) -> None:
    """Save machine-readable summary with per-model statistics."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Group by model_name
    model_stats: Dict[str, Dict] = {}
    for row in rows:
        if row.model_name not in model_stats:
            model_stats[row.model_name] = {
                "max_errors": [],
                "k0_2_errors": [],
                "k0_1_errors": [],
                "times": [],
                "methods": set(),
            }
        stats = model_stats[row.model_name]
        stats["max_errors"].append(row.max_error_pct)
        stats["k0_2_errors"].append(row.k0_2_error_pct)
        stats["k0_1_errors"].append(row.k0_1_error_pct)
        stats["times"].append(row.time_s)
        stats["methods"].add(row.method)

    # Build summary
    models_summary = {}
    best_k0_2_model = None
    best_k0_2_median = float("inf")

    for name, stats in model_stats.items():
        max_errs = np.array(stats["max_errors"])
        k0_2_errs = np.array(stats["k0_2_errors"])
        k0_1_errs = np.array(stats["k0_1_errors"])
        times = np.array(stats["times"])

        entry = {
            "n_runs": len(max_errs),
            "methods": sorted(stats["methods"]),
            "max_error_pct": {
                "median": float(np.median(max_errs)),
                "mean": float(np.mean(max_errs)),
                "max": float(np.max(max_errs)),
                "min": float(np.min(max_errs)),
            },
            "k0_2_error_pct": {
                "median": float(np.median(k0_2_errs)),
                "mean": float(np.mean(k0_2_errs)),
                "max": float(np.max(k0_2_errs)),
                "min": float(np.min(k0_2_errs)),
            },
            "k0_1_error_pct": {
                "median": float(np.median(k0_1_errs)),
                "mean": float(np.mean(k0_1_errs)),
                "max": float(np.max(k0_1_errs)),
            },
            "mean_time_s": float(np.mean(times)),
        }
        models_summary[name] = entry

        if float(np.median(k0_2_errs)) < best_k0_2_median:
            best_k0_2_median = float(np.median(k0_2_errs))
            best_k0_2_model = name

    # Noise-level breakdown for 0% noise only
    zero_noise_rows = [r for r in rows if r.noise_pct == 0.0]
    zero_noise_by_model = {}
    for row in zero_noise_rows:
        zero_noise_by_model.setdefault(row.model_name, []).append(row.max_error_pct)

    zero_noise_summary = {}
    for name, errs in zero_noise_by_model.items():
        zero_noise_summary[name] = {
            "median_max_error": float(np.median(errs)),
            "mean_max_error": float(np.mean(errs)),
            "max_max_error": float(np.max(errs)),
        }

    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "n_total_rows": len(rows),
        "n_models": len(models_summary),
        "best_model_for_k0_2": best_k0_2_model,
        "best_k0_2_median_error_pct": best_k0_2_median,
        "baselines": {
            "v12_rbf_baseline_max_error": 4.23,
            "v13_cascade_max_error": 4.63,
            "v13_pde_refined_max_error": 4.33,
        },
        "models": models_summary,
        "zero_noise_summary": zero_noise_summary,
    }

    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON saved: {path}")


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------
def print_baseline_comparison(rows: List[BenchmarkRow]) -> None:
    """Print comparison against v12/v13 baselines."""
    print("\n" + "=" * 100)
    print("BASELINE COMPARISON")
    print("=" * 100)
    print(f"  v12 RBF baseline max error: 4.23%")
    print(f"  v13 cascade max error: 4.63%")
    print(f"  v13 PDE-refined max error: 4.33%")
    print("-" * 100)

    # Show best result per model at 0% noise on target 0
    target0_clean = [
        r for r in rows
        if r.target_id == 0 and r.noise_pct == 0.0
    ]

    if target0_clean:
        print(f"\n  Target 0 (standard params), 0% noise:")
        print(f"  {'Model':<25s} {'Method':<12s} {'k0_1 err%':>10s} {'k0_2 err%':>10s} "
              f"{'a1 err%':>10s} {'a2 err%':>10s} {'max err%':>10s} {'time(s)':>8s} {'Beats v12?'}")
        print(f"  {'-'*105}")
        for r in sorted(target0_clean, key=lambda x: x.max_error_pct):
            beats = "YES" if r.max_error_pct < 4.23 else "no"
            print(f"  {r.model_name:<25s} {r.method:<12s} "
                  f"{r.k0_1_error_pct:>10.2f} {r.k0_2_error_pct:>10.2f} "
                  f"{r.alpha_1_error_pct:>10.2f} {r.alpha_2_error_pct:>10.2f} "
                  f"{r.max_error_pct:>10.2f} {r.time_s:>8.1f} {beats}")

    # Best k0_2 recovery by noise level
    print(f"\n  Best k0_2 recovery by noise level:")
    for noise in NOISE_LEVELS:
        noise_rows = [r for r in rows if r.noise_pct == noise]
        if noise_rows:
            best = min(noise_rows, key=lambda r: r.k0_2_error_pct)
            print(f"    noise={noise:.0f}%: {best.model_name} ({best.method}) "
                  f"k0_2_err={best.k0_2_error_pct:.2f}%")

    print("=" * 100)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def print_summary(rows: List[BenchmarkRow]) -> None:
    """Print human-readable summary table."""
    print("\n" + "=" * 130)
    print("INVERSE BENCHMARK -- ALL MODELS")
    print("=" * 130)
    header = (
        f"{'Model':<25s} {'Method':<12s} {'Tgt':>3s} {'Noise%':>6s} {'Seed':>4s} "
        f"{'k0_1 err%':>10s} {'k0_2 err%':>10s} "
        f"{'a1 err%':>10s} {'a2 err%':>10s} "
        f"{'max err%':>10s} {'time(s)':>8s}"
    )
    print(header)
    print("-" * 130)
    for row in rows:
        line = (
            f"{row.model_name:<25s} {row.method:<12s} {row.target_id:>3d} {row.noise_pct:>6.1f} {row.seed:>4d} "
            f"{row.k0_1_error_pct:>10.2f} {row.k0_2_error_pct:>10.2f} "
            f"{row.alpha_1_error_pct:>10.2f} {row.alpha_2_error_pct:>10.2f} "
            f"{row.max_error_pct:>10.2f} {row.time_s:>8.1f}"
        )
        print(line)
    print("=" * 130)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    t_global_start = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 80)
    print(f"INVERSE BENCHMARK -- ALL SURROGATE MODELS")
    print(f"Started: {timestamp}")
    print("=" * 80)

    # -----------------------------------------------------------------------
    # Step 1: Load reference surrogate (RBF baseline) for target generation
    # -----------------------------------------------------------------------
    print("\n[Step 1] Loading reference surrogate (RBF baseline) for target generation...")
    ref_model = _load_pkl("model_rbf_baseline.pkl")

    # -----------------------------------------------------------------------
    # Step 2: Generate synthetic targets
    # -----------------------------------------------------------------------
    print("\n[Step 2] Generating synthetic targets...")
    targets = generate_synthetic_targets(ref_model)
    print(f"  Generated {len(targets)} synthetic targets")
    for t in targets:
        print(f"    Target {t.target_id}: k0=[{t.k0_1:.4e},{t.k0_2:.4e}] "
              f"alpha=[{t.alpha_1:.4f},{t.alpha_2:.4f}]")

    # -----------------------------------------------------------------------
    # Step 3: Run benchmark for each model
    # -----------------------------------------------------------------------
    results: List[BenchmarkRow] = []
    n_models_tested = 0

    for model_name, loader_fn, is_slow in MODEL_REGISTRY:
        print(f"\n{'='*80}")
        print(f"MODEL: {model_name}")
        print(f"{'='*80}")

        # Load model
        try:
            print(f"  Loading {model_name}...")
            model = loader_fn()
        except Exception as e:
            print(f"  FAILED to load {model_name}: {e}")
            traceback.print_exc()
            continue

        n_models_tested += 1

        # Verify phi_applied compatibility with reference
        model_phi = model.phi_applied
        ref_phi = ref_model.phi_applied if hasattr(ref_model, "phi_applied") else None
        phi_mismatch = False
        if model_phi is not None and ref_phi is not None:
            if not np.allclose(model_phi, ref_phi, atol=1e-10):
                print(f"  WARNING: phi_applied mismatch ({len(model_phi)} vs {len(ref_phi)} points)")
                phi_mismatch = True

        # Determine test matrix for this model
        if is_slow:
            # GP: reduced workload
            test_targets = [t for t in targets if t.target_id == 0]
            test_noise_levels = [0.0, 1.0]
            test_seeds = [42]
            n_grid = 5_000
            print(f"  [GP mode] Reduced matrix: 1 target, 2 noise levels, 1 seed, n_grid={n_grid}")
        else:
            test_targets = targets
            test_noise_levels = NOISE_LEVELS
            test_seeds = NOISE_SEEDS
            n_grid = 20_000

        for target in test_targets:
            for noise_pct in test_noise_levels:
                seeds_to_use = [0] if noise_pct == 0.0 else test_seeds
                for seed in seeds_to_use:
                    # Generate noisy target
                    if phi_mismatch:
                        # Use model's own prediction at true params
                        try:
                            surr_pred = model.predict(
                                target.k0_1, target.k0_2,
                                target.alpha_1, target.alpha_2,
                            )
                            cd_clean = surr_pred["current_density"]
                            pc_clean = surr_pred["peroxide_current"]
                        except Exception as e:
                            print(f"  WARNING: Can't generate target for {model_name}: {e}")
                            continue
                    else:
                        cd_clean = target.cd_clean
                        pc_clean = target.pc_clean

                    cd_noisy, pc_noisy = add_noise(cd_clean, pc_clean, noise_pct, seed)

                    run_label = f"tgt={target.target_id} noise={noise_pct:.0f}% seed={seed}"

                    # Multistart
                    try:
                        t0_run = time.time()
                        row = run_multistart_for_target(
                            model, model_name, target,
                            cd_noisy, pc_noisy,
                            noise_pct, seed,
                            n_grid=n_grid,
                        )
                        results.append(row)
                        dt = time.time() - t0_run
                        print(f"  multistart [{run_label}]: "
                              f"max_err={row.max_error_pct:.2f}% "
                              f"k0_2_err={row.k0_2_error_pct:.2f}% "
                              f"t={dt:.1f}s")
                    except Exception as e:
                        print(f"  MULTISTART FAILED [{run_label}]: {e}")
                        traceback.print_exc()

                    # Cascade
                    try:
                        t0_run = time.time()
                        row = run_cascade_for_target(
                            model, model_name, target,
                            cd_noisy, pc_noisy,
                            noise_pct, seed,
                        )
                        results.append(row)
                        dt = time.time() - t0_run
                        print(f"  cascade    [{run_label}]: "
                              f"max_err={row.max_error_pct:.2f}% "
                              f"k0_2_err={row.k0_2_error_pct:.2f}% "
                              f"t={dt:.1f}s")
                    except Exception as e:
                        print(f"  CASCADE FAILED [{run_label}]: {e}")
                        traceback.print_exc()

    # -----------------------------------------------------------------------
    # Step 4: Save results
    # -----------------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"SAVING RESULTS ({len(results)} rows from {n_models_tested} models)")
    print(f"{'='*80}")

    if results:
        save_recovery_csv(results, os.path.join(_OUTPUT_DIR, "recovery_table.csv"))
        save_timing_csv(results, os.path.join(_OUTPUT_DIR, "timing_table.csv"))
        save_summary_json(results, os.path.join(_OUTPUT_DIR, "recovery_summary.json"))
        print_summary(results)
        print_baseline_comparison(results)
    else:
        print("\nNo results collected. All models failed.")

    total_time = time.time() - t_global_start
    print(f"\nTotal benchmark time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Models tested: {n_models_tested}")
    print(f"Total result rows: {len(results)}")


if __name__ == "__main__":
    main()
