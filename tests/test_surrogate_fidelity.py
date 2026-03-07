"""Surrogate fidelity validation tests.

Validates all 4 v13-era surrogate models on hold-out data not used during
training.  Computes per-sample NRMSE, saves aggregate error statistics to
JSON and per-sample errors to CSV, and asserts a soft gate (median NRMSE < 20%)
to catch catastrophically broken models.

Note: The soft gate uses *median* NRMSE rather than mean because peroxide
current (PC) has many near-zero-range samples where NRMSE denominators are
tiny, inflating mean NRMSE to 50-200% even for well-fitted models.  Median
is robust to these outliers and correctly reflects model quality across the
bulk of the parameter domain.

Artifacts produced (under StudyResults/surrogate_fidelity/):
    - fidelity_summary.json: aggregate error stats per model per output
    - per_sample_errors.csv: per-sample parameters and NRMSE for all models

Requirements covered: SUR-01, SUR-02, SUR-03.
"""

import matplotlib
matplotlib.use("Agg")

import csv
import json
import os
import pickle
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pytest

from Surrogate.ensemble import load_nn_ensemble
from Surrogate.io import load_surrogate
from Surrogate.validation import validate_surrogate

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_V11_DIR = os.path.join(_ROOT, "StudyResults", "surrogate_v11")
_OUTPUT_DIR = os.path.join(_ROOT, "StudyResults", "surrogate_fidelity")

_ENSEMBLE_DIR = os.path.join(_V11_DIR, "nn_ensemble", "D3-deeper")
_RBF_BASELINE_PATH = os.path.join(_V11_DIR, "model_rbf_baseline.pkl")
_POD_RBF_LOG_PATH = os.path.join(_V11_DIR, "model_pod_rbf_log.pkl")
_POD_RBF_NOLOG_PATH = os.path.join(_V11_DIR, "model_pod_rbf_nolog.pkl")

MODEL_NAMES = ["nn_ensemble", "rbf_baseline", "pod_rbf_log", "pod_rbf_nolog"]

# Soft gate threshold: median NRMSE must be below this value.
# Median is used instead of mean because PC outputs have near-zero-range
# samples that inflate mean NRMSE via division by tiny denominators.
_NRMSE_THRESHOLD = 0.20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_pickle_model(path: str):
    """Load a pickle model without the BVSurrogateModel isinstance check.

    PODRBFSurrogateModel is not a subclass of BVSurrogateModel but shares the
    same predict_batch() API.  This loader avoids the TypeError from
    ``load_surrogate()`` for POD-RBF models.
    """
    with open(path, "rb") as f:
        model = pickle.load(f)
    if not hasattr(model, "predict_batch"):
        raise TypeError(
            f"Loaded object from {path} lacks predict_batch() method "
            f"(type={type(model).__name__})"
        )
    return model


def _defensive_test_idx(split_data) -> np.ndarray:
    """Extract test indices from split_indices.npz with defensive key lookup."""
    for key in ("test_idx", "test", "test_indices"):
        if key in split_data:
            return split_data[key]
    available = list(split_data.keys())
    raise KeyError(
        f"Could not find test indices in split_indices.npz. "
        f"Available keys: {available}"
    )


def _defensive_train_idx(split_data) -> np.ndarray:
    """Extract train indices from split_indices.npz with defensive key lookup."""
    for key in ("train_idx", "train", "train_indices"):
        if key in split_data:
            return split_data[key]
    available = list(split_data.keys())
    raise KeyError(
        f"Could not find train indices in split_indices.npz. "
        f"Available keys: {available}"
    )


# ---------------------------------------------------------------------------
# Fixtures (module-scoped -- expensive operations run once)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def holdout_data():
    """Load hold-out test data from training_data_merged.npz and split_indices.npz."""
    merged_path = os.path.join(_V11_DIR, "training_data_merged.npz")
    split_path = os.path.join(_V11_DIR, "split_indices.npz")

    data = np.load(merged_path)
    split = np.load(split_path)

    all_params = data["parameters"]
    all_cd = data["current_density"]
    all_pc = data["peroxide_current"]
    phi_applied = data["phi_applied"]

    test_idx = _defensive_test_idx(split)
    train_idx = _defensive_train_idx(split)
    n_total = all_params.shape[0]

    # Sanity checks
    assert len(test_idx) >= 50, (
        f"Hold-out set too small: {len(test_idx)} samples (expected >= 50)"
    )
    assert np.max(test_idx) < n_total, (
        f"Test index {np.max(test_idx)} out of range for data with {n_total} samples"
    )

    test_params = all_params[test_idx]
    test_cd = all_cd[test_idx]
    test_pc = all_pc[test_idx]

    print(f"\n  Hold-out size: {len(test_idx)} samples "
          f"(from {n_total} total, {len(train_idx)} train)")

    return {
        "test_params": test_params,
        "test_cd": test_cd,
        "test_pc": test_pc,
        "phi_applied": phi_applied,
        "test_idx": test_idx,
        "train_idx": train_idx,
        "n_total": n_total,
    }


@pytest.fixture(scope="module")
def all_models():
    """Load all 4 surrogate models once for the module."""
    models = {}
    models["nn_ensemble"] = load_nn_ensemble(
        _ENSEMBLE_DIR, n_members=5, device="cpu"
    )
    # RBF baseline is a BVSurrogateModel -- load_surrogate works
    models["rbf_baseline"] = load_surrogate(_RBF_BASELINE_PATH)
    # POD-RBF models are PODRBFSurrogateModel (not a BVSurrogateModel subclass)
    # Use direct pickle loading to avoid isinstance check
    models["pod_rbf_log"] = _load_pickle_model(_POD_RBF_LOG_PATH)
    models["pod_rbf_nolog"] = _load_pickle_model(_POD_RBF_NOLOG_PATH)
    return models


@pytest.fixture(scope="module")
def all_metrics(all_models, holdout_data):
    """Run validate_surrogate() for each model against hold-out data.

    Returns dict keyed by model name with validation results.
    """
    results = {}
    test_params = holdout_data["test_params"]
    test_cd = holdout_data["test_cd"]
    test_pc = holdout_data["test_pc"]

    for name in MODEL_NAMES:
        model = all_models[name]
        metrics = validate_surrogate(model, test_params, test_cd, test_pc)
        results[name] = metrics

        # Compute median NRMSE (robust to near-zero-range outliers)
        metrics["cd_median_nrmse"] = float(
            np.median(metrics["cd_nrmse_per_sample"])
        )
        metrics["pc_median_nrmse"] = float(
            np.median(metrics["pc_nrmse_per_sample"])
        )

        # Print per-model summary for visibility
        print(
            f"  {name}: CD median NRMSE = "
            f"{metrics['cd_median_nrmse'] * 100:.2f}% "
            f"(mean={metrics['cd_mean_relative_error'] * 100:.2f}%), "
            f"PC median NRMSE = "
            f"{metrics['pc_median_nrmse'] * 100:.2f}% "
            f"(mean={metrics['pc_mean_relative_error'] * 100:.2f}%)"
        )

    return results


@pytest.fixture(scope="module")
def fidelity_artifacts(all_metrics, holdout_data):
    """Build and save JSON summary and per-sample CSV artifacts.

    Returns the summary dict for use in assertions.
    """
    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    test_params = holdout_data["test_params"]
    n_test = test_params.shape[0]

    # ---- Build JSON summary ----
    summary = {
        "metadata": {
            "n_test": n_test,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_names": MODEL_NAMES,
            "threshold_mean_nrmse": _NRMSE_THRESHOLD,
        },
        "models": {},
    }

    for name in MODEL_NAMES:
        m = all_metrics[name]
        cd_nrmse = m["cd_nrmse_per_sample"]
        pc_nrmse = m["pc_nrmse_per_sample"]
        summary["models"][name] = {
            "cd_max_nrmse": float(np.max(cd_nrmse)),
            "cd_mean_nrmse": float(np.mean(cd_nrmse)),
            "cd_median_nrmse": float(np.median(cd_nrmse)),
            "cd_95th_nrmse": float(np.percentile(cd_nrmse, 95)),
            "pc_max_nrmse": float(np.max(pc_nrmse)),
            "pc_mean_nrmse": float(np.mean(pc_nrmse)),
            "pc_median_nrmse": float(np.median(pc_nrmse)),
            "pc_95th_nrmse": float(np.percentile(pc_nrmse, 95)),
        }

    json_path = os.path.join(_OUTPUT_DIR, "fidelity_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Build per-sample CSV ----
    csv_path = os.path.join(_OUTPUT_DIR, "per_sample_errors.csv")
    header = ["sample_idx", "k0_1", "k0_2", "alpha_1", "alpha_2"]
    for name in MODEL_NAMES:
        header.extend([f"{name}_cd_nrmse", f"{name}_pc_nrmse"])

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(n_test):
            row = [
                int(holdout_data["test_idx"][i]),
                float(test_params[i, 0]),
                float(test_params[i, 1]),
                float(test_params[i, 2]),
                float(test_params[i, 3]),
            ]
            for name in MODEL_NAMES:
                m = all_metrics[name]
                row.append(float(m["cd_nrmse_per_sample"][i]))
                row.append(float(m["pc_nrmse_per_sample"][i]))
            writer.writerow(row)

    return summary


@pytest.fixture(scope="module")
def generate_plots(all_models, all_metrics, holdout_data):
    """Generate worst-case I-V overlay and error-vs-parameter scatter plots.

    Produces 12 PNGs total (per model: 1 worst-case overlay + 2 scatter).
    Returns list of generated file paths for test verification.
    """
    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    test_params = holdout_data["test_params"]
    test_cd = holdout_data["test_cd"]
    phi_applied = holdout_data["phi_applied"]
    generated_paths = []

    param_names = ["k0_1", "k0_2", "alpha_1", "alpha_2"]

    for model_name in MODEL_NAMES:
        model = all_models[model_name]
        metrics = all_metrics[model_name]
        cd_nrmse = metrics["cd_nrmse_per_sample"]
        pc_nrmse = metrics["pc_nrmse_per_sample"]

        # --- (a) Worst-case I-V overlay (top 3 worst CD NRMSE) ---
        worst_3 = np.argsort(cd_nrmse)[-3:][::-1]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, idx in zip(axes, worst_3):
            pred = model.predict_batch(test_params[idx : idx + 1])
            ax.plot(phi_applied, test_cd[idx], "k-", label="PDE truth")
            ax.plot(
                phi_applied, pred["current_density"][0], "r--", label="Surrogate"
            )
            ax.set_title(f"Sample {idx}, NRMSE={cd_nrmse[idx] * 100:.1f}%")
            ax.set_xlabel("phi_applied")
            ax.set_ylabel("CD")
            ax.legend(fontsize=7)
        fig.suptitle(f"{model_name} -- Worst-Case I-V Overlay (CD)")
        fig.tight_layout()
        overlay_path = os.path.join(
            _OUTPUT_DIR, f"worst_iv_overlay_{model_name}.png"
        )
        fig.savefig(overlay_path, dpi=150)
        plt.close(fig)
        generated_paths.append(overlay_path)

        # --- (b) Error vs parameter scatter -- CD ---
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for ax, j, name in zip(axes, range(4), param_names):
            ax.scatter(test_params[:, j], cd_nrmse * 100, s=8, alpha=0.5)
            ax.set_xlabel(name)
            ax.set_ylabel("CD NRMSE (%)")
            if j < 2:
                ax.set_xscale("log")
        fig.suptitle(f"{model_name} -- CD Error vs Parameters")
        fig.tight_layout()
        cd_scatter_path = os.path.join(
            _OUTPUT_DIR, f"error_vs_params_cd_{model_name}.png"
        )
        fig.savefig(cd_scatter_path, dpi=150)
        plt.close(fig)
        generated_paths.append(cd_scatter_path)

        # --- (c) Error vs parameter scatter -- PC ---
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for ax, j, name in zip(axes, range(4), param_names):
            ax.scatter(test_params[:, j], pc_nrmse * 100, s=8, alpha=0.5)
            ax.set_xlabel(name)
            ax.set_ylabel("PC NRMSE (%)")
            if j < 2:
                ax.set_xscale("log")
        fig.suptitle(f"{model_name} -- PC Error vs Parameters")
        fig.tight_layout()
        pc_scatter_path = os.path.join(
            _OUTPUT_DIR, f"error_vs_params_pc_{model_name}.png"
        )
        fig.savefig(pc_scatter_path, dpi=150)
        plt.close(fig)
        generated_paths.append(pc_scatter_path)

    return generated_paths


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestSurrogateFidelity:
    """Hold-out validation tests for all 4 v13-era surrogate models."""

    @pytest.mark.parametrize("model_name", MODEL_NAMES)
    def test_holdout_median_nrmse_below_threshold(self, all_metrics, model_name):
        """Soft gate: median NRMSE < 20% for both CD and PC outputs.

        Median is used instead of mean because PC outputs have near-zero-range
        samples whose NRMSE denominators are tiny, inflating the mean to
        50-200% even for well-fitted models.  Median reflects bulk accuracy.
        """
        metrics = all_metrics[model_name]
        cd_median = metrics["cd_median_nrmse"]
        pc_median = metrics["pc_median_nrmse"]

        assert cd_median < _NRMSE_THRESHOLD, (
            f"{model_name} CD median NRMSE = {cd_median * 100:.2f}% "
            f"(threshold: {_NRMSE_THRESHOLD * 100:.0f}%)"
        )
        assert pc_median < _NRMSE_THRESHOLD, (
            f"{model_name} PC median NRMSE = {pc_median * 100:.2f}% "
            f"(threshold: {_NRMSE_THRESHOLD * 100:.0f}%)"
        )

    def test_error_stats_saved_to_json(self, fidelity_artifacts):
        """JSON summary contains entries for all 4 models with all 6 stats."""
        summary = fidelity_artifacts
        assert "models" in summary

        expected_stats = [
            "cd_max_nrmse", "cd_mean_nrmse", "cd_median_nrmse",
            "cd_95th_nrmse",
            "pc_max_nrmse", "pc_mean_nrmse", "pc_median_nrmse",
            "pc_95th_nrmse",
        ]

        for name in MODEL_NAMES:
            assert name in summary["models"], (
                f"Model {name} missing from fidelity_summary.json"
            )
            model_stats = summary["models"][name]
            for stat in expected_stats:
                assert stat in model_stats, (
                    f"Stat '{stat}' missing for model {name}"
                )
                assert isinstance(model_stats[stat], float), (
                    f"Stat '{stat}' for {name} is not a float"
                )

        # Also verify the file exists on disk
        json_path = os.path.join(_OUTPUT_DIR, "fidelity_summary.json")
        assert os.path.isfile(json_path), (
            f"fidelity_summary.json not found at {json_path}"
        )

    def test_fidelity_csv_has_all_samples(self, fidelity_artifacts, holdout_data):
        """CSV has correct number of rows and all expected columns."""
        csv_path = os.path.join(_OUTPUT_DIR, "per_sample_errors.csv")
        assert os.path.isfile(csv_path), (
            f"per_sample_errors.csv not found at {csv_path}"
        )

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)

        n_test = holdout_data["test_params"].shape[0]
        assert len(rows) == n_test, (
            f"CSV has {len(rows)} data rows, expected {n_test}"
        )

        # Check parameter columns present
        for col in ("sample_idx", "k0_1", "k0_2", "alpha_1", "alpha_2"):
            assert col in header, f"Column '{col}' missing from CSV header"

        # Check error columns for all models
        for name in MODEL_NAMES:
            assert f"{name}_cd_nrmse" in header, (
                f"Column '{name}_cd_nrmse' missing from CSV header"
            )
            assert f"{name}_pc_nrmse" in header, (
                f"Column '{name}_pc_nrmse' missing from CSV header"
            )

    def test_holdout_uses_unseen_data(self, holdout_data):
        """Train and test indices do not overlap (SUR-02: unseen data)."""
        train_idx = holdout_data["train_idx"]
        test_idx = holdout_data["test_idx"]

        train_set = set(train_idx.tolist())
        test_set = set(test_idx.tolist())
        overlap = train_set & test_set

        assert len(overlap) == 0, (
            f"Train/test overlap: {len(overlap)} indices in common "
            f"(first 10: {sorted(overlap)[:10]})"
        )

        # Verify they cover the full dataset
        n_total = holdout_data["n_total"]
        combined = train_set | test_set
        assert len(combined) == n_total, (
            f"Train ({len(train_set)}) + test ({len(test_set)}) = "
            f"{len(combined)}, expected {n_total}"
        )

    def test_worst_iv_overlay_plots_generated(self, generate_plots):
        """All 4 worst-case I-V overlay PNGs exist and have nonzero size."""
        for model_name in MODEL_NAMES:
            path = os.path.join(
                _OUTPUT_DIR, f"worst_iv_overlay_{model_name}.png"
            )
            assert os.path.isfile(path), (
                f"Worst-case overlay plot missing: {path}"
            )
            assert os.path.getsize(path) > 0, (
                f"Worst-case overlay plot is empty: {path}"
            )

    def test_error_scatter_plots_generated(self, generate_plots):
        """All 8 error-vs-parameter scatter PNGs exist (4 models x CD/PC)."""
        for model_name in MODEL_NAMES:
            for output_type in ("cd", "pc"):
                path = os.path.join(
                    _OUTPUT_DIR,
                    f"error_vs_params_{output_type}_{model_name}.png",
                )
                assert os.path.isfile(path), (
                    f"Error scatter plot missing: {path}"
                )
                assert os.path.getsize(path) > 0, (
                    f"Error scatter plot is empty: {path}"
                )
