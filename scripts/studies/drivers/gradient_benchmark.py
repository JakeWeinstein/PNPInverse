#!/usr/bin/env python3
"""Gradient accuracy and speed benchmark across all surrogate models.

Benchmarks gradient computation accuracy (vs fine-FD reference) and
wall-clock speed for:
  - NN ensemble (autograd + FD)
  - GP (autograd + FD)
  - PCE (analytic + FD)
  - POD-RBF (FD only)
  - RBF baseline (FD only)

Usage
-----
    python scripts/studies/gradient_benchmark.py

Outputs
-------
    StudyResults/gradient_benchmark/gradient_accuracy.json
    StudyResults/gradient_benchmark/gradient_speed.json
    StudyResults/gradient_benchmark/gradient_benchmark_report.md
"""

from __future__ import annotations

import os
import sys

# Ensure repo root is on sys.path when run as a script
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import json
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Gradient computation helpers
# ---------------------------------------------------------------------------


def _fd_gradient(
    obj_fn, x: np.ndarray, h: float = 1e-5,
) -> np.ndarray:
    """Central finite-difference gradient of a scalar function."""
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    for i in range(len(x)):
        xp = x.copy()
        xm = x.copy()
        xp[i] += h
        xm[i] -= h
        grad[i] = (obj_fn(xp) - obj_fn(xm)) / (2 * h)
    return grad


def _make_objective_fn(model, target_cd, target_pc, secondary_weight=1.0):
    """Build a scalar objective function J(x) using model.predict().

    x = [log10(k0_1), log10(k0_2), alpha_1, alpha_2].
    """

    def obj(x: np.ndarray) -> float:
        k0_1 = 10.0 ** x[0]
        k0_2 = 10.0 ** x[1]
        alpha_1 = float(x[2])
        alpha_2 = float(x[3])
        pred = model.predict(k0_1, k0_2, alpha_1, alpha_2)
        cd_diff = pred["current_density"] - target_cd
        pc_diff = pred["peroxide_current"] - target_pc
        return float(
            0.5 * np.sum(cd_diff ** 2)
            + secondary_weight * 0.5 * np.sum(pc_diff ** 2)
        )

    return obj


def _autograd_gradient_nn(model, x, target_cd, target_pc, secondary_weight=1.0):
    """Compute gradient via SurrogateObjective autograd path (NN/ensemble)."""
    from Surrogate.objectives import SurrogateObjective

    obj = SurrogateObjective(
        model, target_cd, target_pc, secondary_weight=secondary_weight,
    )
    obj._use_autograd = True
    _, g = obj._autograd_objective_and_gradient(x)
    return g


def _autograd_gradient_gp(model, x, target_cd, target_pc, secondary_weight=1.0):
    """Compute gradient via GPSurrogateModel.gradient_at()."""
    k0_1 = 10.0 ** x[0]
    k0_2 = 10.0 ** x[1]
    return model.gradient_at(
        k0_1, k0_2, float(x[2]), float(x[3]),
        target_cd, target_pc, secondary_weight=secondary_weight,
    )


def _analytic_gradient_pce(model, x, target_cd, target_pc, secondary_weight=1.0):
    """Compute gradient via PCE analytic polynomial differentiation.

    PCE.predict_gradient() returns d(output)/d(physical_k0) with chain rule
    already applied.  To get dJ/d(log10_k0), we need:
        dJ/d(log10_k0) = dJ/d(k0) * k0 * ln(10)

    For alpha dims, no conversion needed since FD and analytic both
    operate in the same space.
    """
    k0_1 = 10.0 ** x[0]
    k0_2 = 10.0 ** x[1]
    alpha_1, alpha_2 = float(x[2]), float(x[3])

    # Get predictions at this point
    pred = model.predict(k0_1, k0_2, alpha_1, alpha_2)
    cd_pred = pred["current_density"]
    pc_pred = pred["peroxide_current"]

    # Get analytic gradients: d(output)/d(physical params)
    grads = model.predict_gradient(k0_1, k0_2, alpha_1, alpha_2)
    grad_cd = grads["grad_cd"]  # (4, n_eta)
    grad_pc = grads["grad_pc"]  # (4, n_eta)

    # Residuals
    cd_resid = cd_pred - target_cd  # (n_eta,)
    pc_resid = pc_pred - target_pc  # (n_eta,)

    # Chain rule: dJ/d(param_i) = sum_j resid_j * d(output_j)/d(param_i)
    dJ_dparams = np.zeros(4)
    for i in range(4):
        dJ_dparams[i] = (
            np.dot(cd_resid, grad_cd[i, :])
            + secondary_weight * np.dot(pc_resid, grad_pc[i, :])
        )

    # Convert from physical k0 space to log10 k0 space for dims 0, 1
    ln10 = np.log(10.0)
    dJ_dparams[0] *= k0_1 * ln10
    dJ_dparams[1] *= k0_2 * ln10

    return dJ_dparams


# ---------------------------------------------------------------------------
# Accuracy benchmark
# ---------------------------------------------------------------------------


def run_accuracy_benchmark(
    models: Dict[str, Dict[str, Any]],
    test_points: List[np.ndarray],
    target_cd: np.ndarray,
    target_pc: np.ndarray,
    fd_steps: Optional[List[float]] = None,
    ref_fd_step: float = 1e-7,
    secondary_weight: float = 1.0,
) -> Dict[str, List[Dict[str, Any]]]:
    """Benchmark gradient accuracy for all models.

    Parameters
    ----------
    models : dict
        Keys are model names, values are dicts with:
          'model': the surrogate model object
          'has_autograd': bool (True if autograd/analytic gradient available)
          'gradient_type': str ('autograd', 'analytic', or None for FD-only)
    test_points : list of np.ndarray
        Each element is x = [log10(k0_1), log10(k0_2), alpha_1, alpha_2].
    target_cd, target_pc : np.ndarray
        Target curves for objective.
    fd_steps : list of float or None
        FD step sizes to test (default: [1e-3, 1e-4, 1e-5, 1e-6]).
    ref_fd_step : float
        Reference FD step size for computing ground truth.
    secondary_weight : float
        Weight on peroxide current term.

    Returns
    -------
    dict mapping model_name -> list of result dicts, each with:
        'method', 'relative_error', 'max_component_error', 'gradient', 'point_errors'
    """
    if fd_steps is None:
        fd_steps = [1e-3, 1e-4, 1e-5, 1e-6]

    results: Dict[str, List[Dict[str, Any]]] = {}

    for name, spec in models.items():
        model = spec["model"]
        has_autograd = spec.get("has_autograd", False)
        gradient_type = spec.get("gradient_type", "autograd")
        model_results: List[Dict[str, Any]] = []

        obj_fn = _make_objective_fn(model, target_cd, target_pc, secondary_weight)

        # Compute reference gradient (fine FD) for each test point
        ref_grads = []
        for x in test_points:
            ref_grads.append(_fd_gradient(obj_fn, x, h=ref_fd_step))

        # Autograd/analytic method
        if has_autograd:
            point_errors = []
            all_rel_errors = []
            for idx, x in enumerate(test_points):
                g_ref = ref_grads[idx]
                if gradient_type == "analytic":
                    g_method = _analytic_gradient_pce(
                        model, x, target_cd, target_pc, secondary_weight,
                    )
                elif gradient_type == "autograd_gp":
                    g_method = _autograd_gradient_gp(
                        model, x, target_cd, target_pc, secondary_weight,
                    )
                else:
                    g_method = _autograd_gradient_nn(
                        model, x, target_cd, target_pc, secondary_weight,
                    )

                denom = np.maximum(np.abs(g_ref), 1e-15)
                rel_err = np.max(np.abs(g_method - g_ref) / denom)
                point_errors.append(float(rel_err))
                all_rel_errors.append(rel_err)

            method_label = gradient_type if gradient_type != "autograd" else "autograd"
            model_results.append({
                "method": method_label,
                "relative_error": float(np.mean(all_rel_errors)),
                "max_component_error": float(np.max(all_rel_errors)),
                "point_errors": point_errors,
            })

        # FD methods at various step sizes
        for h in fd_steps:
            point_errors = []
            all_rel_errors = []
            for idx, x in enumerate(test_points):
                g_ref = ref_grads[idx]
                g_fd = _fd_gradient(obj_fn, x, h=h)
                denom = np.maximum(np.abs(g_ref), 1e-15)
                rel_err = np.max(np.abs(g_fd - g_ref) / denom)
                point_errors.append(float(rel_err))
                all_rel_errors.append(rel_err)

            model_results.append({
                "method": f"fd_h={h:.0e}",
                "relative_error": float(np.mean(all_rel_errors)),
                "max_component_error": float(np.max(all_rel_errors)),
                "point_errors": point_errors,
            })

        results[name] = model_results

    return results


# ---------------------------------------------------------------------------
# Speed benchmark
# ---------------------------------------------------------------------------


def run_speed_benchmark(
    models: Dict[str, Dict[str, Any]],
    x: np.ndarray,
    target_cd: np.ndarray,
    target_pc: np.ndarray,
    n_iters: int = 50,
    n_warmup: int = 5,
    secondary_weight: float = 1.0,
) -> Dict[str, List[Dict[str, Any]]]:
    """Benchmark gradient computation speed for all models.

    Parameters
    ----------
    models : dict
        Same format as run_accuracy_benchmark.
    x : np.ndarray
        Test point for timing.
    target_cd, target_pc : np.ndarray
        Target curves.
    n_iters : int
        Number of timed iterations.
    n_warmup : int
        Warmup iterations (not timed).
    secondary_weight : float
        Weight on peroxide current term.

    Returns
    -------
    dict mapping model_name -> list of dicts with 'method', 'ms_per_eval',
    'batch_size'.
    """
    results: Dict[str, List[Dict[str, Any]]] = {}

    for name, spec in models.items():
        model = spec["model"]
        has_autograd = spec.get("has_autograd", False)
        gradient_type = spec.get("gradient_type", "autograd")
        model_results: List[Dict[str, Any]] = []

        obj_fn = _make_objective_fn(model, target_cd, target_pc, secondary_weight)

        # Time autograd/analytic single-point
        if has_autograd:
            if gradient_type == "analytic":
                grad_fn = lambda: _analytic_gradient_pce(
                    model, x, target_cd, target_pc, secondary_weight,
                )
            elif gradient_type == "autograd_gp":
                grad_fn = lambda: _autograd_gradient_gp(
                    model, x, target_cd, target_pc, secondary_weight,
                )
            else:
                grad_fn = lambda: _autograd_gradient_nn(
                    model, x, target_cd, target_pc, secondary_weight,
                )

            # Warmup
            for _ in range(n_warmup):
                grad_fn()

            t0 = time.perf_counter()
            for _ in range(n_iters):
                grad_fn()
            elapsed = time.perf_counter() - t0

            method_label = gradient_type if gradient_type != "autograd" else "autograd"
            model_results.append({
                "method": method_label,
                "ms_per_eval": elapsed / n_iters * 1000.0,
                "batch_size": 1,
            })

        # Time FD single-point (using default step)
        fd_fn = lambda: _fd_gradient(obj_fn, x, h=1e-5)

        for _ in range(n_warmup):
            fd_fn()

        t0 = time.perf_counter()
        for _ in range(n_iters):
            fd_fn()
        elapsed = time.perf_counter() - t0

        model_results.append({
            "method": "fd",
            "ms_per_eval": elapsed / n_iters * 1000.0,
            "batch_size": 1,
        })

        results[name] = model_results

    return results


# ---------------------------------------------------------------------------
# Output functions
# ---------------------------------------------------------------------------


def save_accuracy_results(results: Dict, path: str) -> None:
    """Save accuracy benchmark results to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Accuracy results saved to: {path}")


def save_speed_results(results: Dict, path: str) -> None:
    """Save speed benchmark results to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Speed results saved to: {path}")


def generate_report(
    accuracy: Dict, speed: Dict, path: str,
) -> None:
    """Generate a Markdown summary report from benchmark results."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    lines = [
        "# Gradient Benchmark Report",
        "",
        "## Accuracy: Relative Error vs Fine-FD Reference (h=1e-7)",
        "",
        "| Model | Method | Mean Rel Error | Max Rel Error |",
        "|-------|--------|----------------|---------------|",
    ]

    for model_name, entries in accuracy.items():
        for entry in entries:
            method = entry["method"]
            mean_err = entry["relative_error"]
            max_err = entry["max_component_error"]
            lines.append(
                f"| {model_name} | {method} | {mean_err:.2e} | {max_err:.2e} |"
            )

    lines.append("")
    lines.append("## Speed: Wall-Clock Gradient Evaluation Time")
    lines.append("")
    lines.append("| Model | Method | ms/eval | Batch Size |")
    lines.append("|-------|--------|---------|------------|")

    for model_name, entries in speed.items():
        for entry in entries:
            method = entry["method"]
            ms = entry["ms_per_eval"]
            batch = entry["batch_size"]
            lines.append(
                f"| {model_name} | {method} | {ms:.2f} | {batch} |"
            )

    # Speed ranking
    lines.append("")
    lines.append("## Ranking by Autograd/Analytic Speed (single-point)")
    lines.append("")

    autograd_speeds = []
    for model_name, entries in speed.items():
        for entry in entries:
            if entry["method"] != "fd" and entry["batch_size"] == 1:
                autograd_speeds.append((model_name, entry["method"], entry["ms_per_eval"]))

    autograd_speeds.sort(key=lambda t: t[2])
    for rank, (name, method, ms) in enumerate(autograd_speeds, 1):
        lines.append(f"{rank}. **{name}** ({method}): {ms:.2f} ms/eval")

    # Accuracy ranking
    lines.append("")
    lines.append("## Ranking by Autograd/Analytic Accuracy")
    lines.append("")

    autograd_acc = []
    for model_name, entries in accuracy.items():
        for entry in entries:
            if not entry["method"].startswith("fd_"):
                autograd_acc.append((model_name, entry["method"], entry["relative_error"]))

    autograd_acc.sort(key=lambda t: t[2])
    for rank, (name, method, err) in enumerate(autograd_acc, 1):
        lines.append(f"{rank}. **{name}** ({method}): mean rel error = {err:.2e}")

    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))

    print(f"Report saved to: {path}")


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------


def _load_all_models() -> Dict[str, Dict[str, Any]]:
    """Load all available surrogate models with graceful fallback."""
    models: Dict[str, Dict[str, Any]] = {}

    # NN ensemble
    try:
        from Surrogate.ensemble import load_nn_ensemble

        ensemble = load_nn_ensemble("data/surrogate_models/nn_ensemble/D3-deeper")
        models["nn_ensemble"] = {
            "model": ensemble,
            "has_autograd": True,
            "gradient_type": "autograd",
        }
        print("  Loaded: NN ensemble (D3-deeper)")
    except Exception as e:
        print(f"  SKIP NN ensemble: {e}")

    # GP
    try:
        from Surrogate.gp_model import GPSurrogateModel

        gp = GPSurrogateModel.load("data/surrogate_models/gp_fixed/")
        models["gp"] = {
            "model": gp,
            "has_autograd": True,
            "gradient_type": "autograd_gp",
        }
        print("  Loaded: GP (gp_fixed)")
    except Exception as e:
        print(f"  SKIP GP: {e}")

    # PCE
    try:
        from Surrogate.pce_model import PCESurrogateModel

        pce = PCESurrogateModel.load("data/surrogate_models/pce/pce_model.pkl")
        models["pce"] = {
            "model": pce,
            "has_autograd": True,
            "gradient_type": "analytic",
        }
        print("  Loaded: PCE")
    except Exception as e:
        print(f"  SKIP PCE: {e}")

    # POD-RBF (FD only)
    try:
        with open("data/surrogate_models/model_pod_rbf_log.pkl", "rb") as f:
            pod_rbf = pickle.load(f)
        models["pod_rbf"] = {
            "model": pod_rbf,
            "has_autograd": False,
            "gradient_type": None,
        }
        print("  Loaded: POD-RBF")
    except Exception as e:
        print(f"  SKIP POD-RBF: {e}")

    # RBF baseline (FD only)
    try:
        with open("data/surrogate_models/model_rbf_baseline.pkl", "rb") as f:
            rbf = pickle.load(f)
        models["rbf_baseline"] = {
            "model": rbf,
            "has_autograd": False,
            "gradient_type": None,
        }
        print("  Loaded: RBF baseline")
    except Exception as e:
        print(f"  SKIP RBF baseline: {e}")

    return models


def _get_test_points(n_held_out: int = 20, n_corners: int = 5) -> List[np.ndarray]:
    """Select test points from held-out data + parameter space corners."""
    points: List[np.ndarray] = []

    # Try to load from split indices
    try:
        data = np.load("data/surrogate_models/training_data_merged.npz")
        split = np.load("data/surrogate_models/split_indices.npz")
        params = data["parameters"]  # physical space
        test_idx = split["test_idx"]

        # Convert to log-space
        rng = np.random.default_rng(42)
        chosen = rng.choice(len(test_idx), size=min(n_held_out, len(test_idx)), replace=False)
        for i in chosen:
            p = params[test_idx[i]]
            x = np.array([
                np.log10(max(p[0], 1e-30)),
                np.log10(max(p[1], 1e-30)),
                p[2],
                p[3],
            ])
            points.append(x)
        print(f"  Selected {len(points)} test points from held-out set")
    except Exception as e:
        print(f"  Could not load test set: {e}")
        # Fallback: use fixed points
        points.append(np.array([-3.0, -5.0, 0.3, 0.5]))
        points.append(np.array([-4.0, -6.0, 0.4, 0.6]))
        points.append(np.array([-2.5, -4.5, 0.25, 0.45]))
        print(f"  Using {len(points)} fallback test points")

    # Add corner points
    corners = [
        np.array([-1.0, -3.0, 0.1, 0.1]),
        np.array([-6.0, -8.0, 0.9, 0.9]),
        np.array([-1.0, -8.0, 0.1, 0.9]),
        np.array([-6.0, -3.0, 0.9, 0.1]),
        np.array([-3.5, -5.5, 0.5, 0.5]),
    ]
    points.extend(corners[:n_corners])
    print(f"  Total test points: {len(points)}")

    return points


def _get_target_curves(model) -> Tuple[np.ndarray, np.ndarray]:
    """Generate target curves from a reference parameter set."""
    # Use a mid-range parameter set as reference
    k0_1 = 1e-3
    k0_2 = 1e-5
    alpha_1 = 0.3
    alpha_2 = 0.5
    pred = model.predict(k0_1, k0_2, alpha_1, alpha_2)
    return pred["current_density"], pred["peroxide_current"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Run full gradient benchmark and save results."""
    print("=" * 60)
    print("  GRADIENT ACCURACY AND SPEED BENCHMARK")
    print("=" * 60)

    print("\n--- Loading models ---")
    models = _load_all_models()

    if not models:
        print("ERROR: No models loaded. Exiting.")
        return

    print(f"\n--- Loaded {len(models)} models: {list(models.keys())} ---")

    # Get test points
    print("\n--- Selecting test points ---")
    test_points = _get_test_points()

    # Generate target curves from first available model
    first_model = next(iter(models.values()))["model"]
    target_cd, target_pc = _get_target_curves(first_model)
    print(f"  Target curves: n_eta={len(target_cd)}")

    # Run accuracy benchmark
    print("\n--- Running accuracy benchmark ---")
    accuracy = run_accuracy_benchmark(
        models, test_points, target_cd, target_pc,
    )

    # Print summary
    for model_name, entries in accuracy.items():
        print(f"\n  {model_name}:")
        for entry in entries:
            print(
                f"    {entry['method']:20s}  mean_rel_err={entry['relative_error']:.2e}  "
                f"max_rel_err={entry['max_component_error']:.2e}"
            )

    # Run speed benchmark
    print("\n--- Running speed benchmark ---")
    x_timing = np.array([-3.0, -5.0, 0.3, 0.5])
    speed = run_speed_benchmark(
        models, x_timing, target_cd, target_pc,
        n_iters=50, n_warmup=5,
    )

    # Print summary
    for model_name, entries in speed.items():
        print(f"\n  {model_name}:")
        for entry in entries:
            print(
                f"    {entry['method']:20s}  {entry['ms_per_eval']:.2f} ms/eval  "
                f"(batch={entry['batch_size']})"
            )

    # Save results
    print("\n--- Saving results ---")
    out_dir = "StudyResults/gradient_benchmark"
    save_accuracy_results(accuracy, os.path.join(out_dir, "gradient_accuracy.json"))
    save_speed_results(speed, os.path.join(out_dir, "gradient_speed.json"))
    generate_report(accuracy, speed, os.path.join(out_dir, "gradient_benchmark_report.md"))

    print("\n" + "=" * 60)
    print("  BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
