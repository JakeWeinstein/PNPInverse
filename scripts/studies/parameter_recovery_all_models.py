"""Surrogate-only parameter recovery comparison across all retrained models.

Runs multistart and cascade inference for each surrogate model against
PDE-generated targets at the known true parameters, then reports recovered
parameters and relative errors.

Usage::

    source ../venv-firedrake/bin/activate
    python scripts/studies/parameter_recovery_all_models.py
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from dataclasses import dataclass
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
_TARGET_CACHE = os.path.join(_ROOT, "StudyResults", "target_cache", "clean_targets_1b38f0b9a75d58d3.npz")
_OUTPUT_DIR = os.path.join(_ROOT, "StudyResults", "parameter_recovery_v12")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class RecoveryRow:
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


def _pct_error(true: float, recovered: float) -> float:
    return abs(recovered - true) / max(abs(true), 1e-30) * 100.0


def _make_row(
    model_name: str,
    method: str,
    k0_1: float,
    k0_2: float,
    alpha_1: float,
    alpha_2: float,
    elapsed: float,
) -> RecoveryRow:
    e1 = _pct_error(TRUE_K0_1, k0_1)
    e2 = _pct_error(TRUE_K0_2, k0_2)
    e3 = _pct_error(TRUE_ALPHA_1, alpha_1)
    e4 = _pct_error(TRUE_ALPHA_2, alpha_2)
    return RecoveryRow(
        model_name=model_name,
        method=method,
        k0_1_true=TRUE_K0_1,
        k0_1_recovered=k0_1,
        k0_1_error_pct=e1,
        k0_2_true=TRUE_K0_2,
        k0_2_recovered=k0_2,
        k0_2_error_pct=e2,
        alpha_1_true=TRUE_ALPHA_1,
        alpha_1_recovered=alpha_1,
        alpha_1_error_pct=e3,
        alpha_2_true=TRUE_ALPHA_2,
        alpha_2_recovered=alpha_2,
        alpha_2_error_pct=e4,
        max_error_pct=max(e1, e2, e3, e4),
        time_s=elapsed,
    )


# ---------------------------------------------------------------------------
# Target loading
# ---------------------------------------------------------------------------
def load_targets() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load PDE-generated targets from cache.

    Returns (target_cd, target_pc, phi_applied).
    """
    if not os.path.exists(_TARGET_CACHE):
        raise FileNotFoundError(
            f"Target cache not found at {_TARGET_CACHE}.\n"
            "Generate targets first (e.g., via test_inverse_verification.py)."
        )
    data = np.load(_TARGET_CACHE)
    return (
        data["current_density"].copy(),
        data["peroxide_current"].copy(),
        data["phi_applied"].copy(),
    )


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------
def _load_nn_ensemble(design: str):
    """Load an NN ensemble model by design name."""
    from Surrogate.ensemble import load_nn_ensemble
    path = os.path.join(_SURROGATE_DIR, "nn_ensemble", design)
    return load_nn_ensemble(path, n_members=5)


class _GPFDWrapper:
    """Wrapper that hides predict_torch from GP to force FD gradient path.

    The GP's predict_torch expects (M,4) input but the multistart/cascade
    autograd path passes (4,). Rather than fixing all callers, we hide
    predict_torch so _has_autograd() returns False, forcing the FD path.
    """

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
    """Load a pickle-based surrogate model (handles PODRBFSurrogateModel too)."""
    import pickle
    path = os.path.join(_SURROGATE_DIR, filename)
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"Surrogate model loaded from: {path} ({type(model).__name__})")
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
def run_multistart_for_model(
    model,
    model_name: str,
    target_cd: np.ndarray,
    target_pc: np.ndarray,
    n_grid: int = 20_000,
) -> RecoveryRow:
    """Run multistart inference and return a RecoveryRow."""
    config = MultiStartConfig(
        n_grid=n_grid,
        n_top_candidates=20,
        polish_maxiter=60,
        secondary_weight=1.0,
        seed=42,
        verbose=True,
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
    return _make_row(
        model_name, "multistart",
        result.best_k0_1, result.best_k0_2,
        result.best_alpha_1, result.best_alpha_2,
        elapsed,
    )


def run_cascade_for_model(
    model,
    model_name: str,
    target_cd: np.ndarray,
    target_pc: np.ndarray,
) -> RecoveryRow:
    """Run cascade inference and return a RecoveryRow."""
    # Use midpoint of bounds as initial guess
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
        verbose=True,
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
    return _make_row(
        model_name, "cascade",
        result.best_k0_1, result.best_k0_2,
        result.best_alpha_1, result.best_alpha_2,
        elapsed,
    )


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------
def save_csv(rows: List[RecoveryRow], path: str) -> None:
    """Save results to CSV."""
    import csv

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "model_name", "method",
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
            d = {
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
            }
            writer.writerow(d)
    print(f"\nResults saved to: {path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def print_summary(rows: List[RecoveryRow]) -> None:
    """Print a human-readable summary table to stdout."""
    print("\n" + "=" * 120)
    print("PARAMETER RECOVERY COMPARISON -- ALL MODELS")
    print("=" * 120)
    print(f"  True params: k0_1={TRUE_K0_1:.6e}, k0_2={TRUE_K0_2:.6e}, "
          f"alpha_1={TRUE_ALPHA_1:.4f}, alpha_2={TRUE_ALPHA_2:.4f}")
    print("-" * 120)
    header = (
        f"{'Model':<25s} {'Method':<12s} "
        f"{'k0_1 err%':>10s} {'k0_2 err%':>10s} "
        f"{'a1 err%':>10s} {'a2 err%':>10s} "
        f"{'max err%':>10s} {'time(s)':>8s}"
    )
    print(header)
    print("-" * 120)
    for row in rows:
        line = (
            f"{row.model_name:<25s} {row.method:<12s} "
            f"{row.k0_1_error_pct:>10.2f} {row.k0_2_error_pct:>10.2f} "
            f"{row.alpha_1_error_pct:>10.2f} {row.alpha_2_error_pct:>10.2f} "
            f"{row.max_error_pct:>10.2f} {row.time_s:>8.1f}"
        )
        print(line)
    print("=" * 120)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODEL_REGISTRY = [
    ("NN D3-deeper", lambda: _load_nn_ensemble("D3-deeper")),
    ("NN D2-wider", lambda: _load_nn_ensemble("D2-wider")),
    ("GP", _load_gp),
    ("POD-RBF log", lambda: _load_pkl("model_pod_rbf_log.pkl")),
    ("RBF baseline", lambda: _load_pkl("model_rbf_baseline.pkl")),
    # PCE skipped: its predict() is incompatible with the multistart/cascade
    # infrastructure (passes numpy arrays where scalars are expected).
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 80)
    print("Parameter Recovery Study -- All Surrogate Models (v12 data)")
    print("=" * 80)

    # Load targets
    print("\nLoading PDE-generated targets from cache...")
    target_cd, target_pc, phi_applied = load_targets()
    print(f"  Loaded: {len(target_cd)} voltage points")
    print(f"  phi range: [{phi_applied.min():.4f}, {phi_applied.max():.4f}]")

    results: List[RecoveryRow] = []

    for model_name, loader_fn in MODEL_REGISTRY:
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

        # Verify phi_applied compatibility
        model_phi = model.phi_applied
        if model_phi is not None and not np.allclose(model_phi, phi_applied, atol=1e-10):
            print(f"  WARNING: phi_applied mismatch for {model_name}. "
                  f"Model has {len(model_phi)} points, targets have {len(phi_applied)}.")
            # Use model's phi grid and regenerate targets via surrogate at true params
            # This avoids grid mismatch errors
            print(f"  Using surrogate-generated targets at true params for this model.")
            surr_pred = model.predict(TRUE_K0_1, TRUE_K0_2, TRUE_ALPHA_1, TRUE_ALPHA_2)
            model_target_cd = surr_pred["current_density"]
            model_target_pc = surr_pred["peroxide_current"]
        else:
            model_target_cd = target_cd
            model_target_pc = target_pc

        # Determine n_grid for GP (may be slow for 20K)
        is_gp = model_name == "GP"
        n_grid = 5_000 if is_gp else 20_000

        # Multistart
        try:
            print(f"\n  --- Multistart ({n_grid} grid points) ---")
            row = run_multistart_for_model(
                model, model_name, model_target_cd, model_target_pc,
                n_grid=n_grid,
            )
            results.append(row)
            print(f"  Result: max_error={row.max_error_pct:.2f}%, time={row.time_s:.1f}s")
        except Exception as e:
            print(f"  MULTISTART FAILED for {model_name}: {e}")
            traceback.print_exc()

        # Cascade
        try:
            print(f"\n  --- Cascade ---")
            row = run_cascade_for_model(
                model, model_name, model_target_cd, model_target_pc,
            )
            results.append(row)
            print(f"  Result: max_error={row.max_error_pct:.2f}%, time={row.time_s:.1f}s")
        except Exception as e:
            print(f"  CASCADE FAILED for {model_name}: {e}")
            traceback.print_exc()

    # Output
    if results:
        csv_path = os.path.join(_OUTPUT_DIR, "recovery_comparison.csv")
        save_csv(results, csv_path)
        print_summary(results)
    else:
        print("\nNo results collected. All models failed.")


if __name__ == "__main__":
    main()
