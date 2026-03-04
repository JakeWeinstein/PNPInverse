"""Unified script to train and compare ALL surrogate model types.

Trains:
1. RBF+POD surrogate (Surrogate.pod_rbf_model.PODRBFSurrogateModel)
2. NN ensemble (Surrogate.nn_training.train_nn_ensemble)

Then compares RMSE/NRMSE metrics and optionally runs a parameter recovery
test using cascade inference.

Usage (from PNPInverse/ directory)::

    # Train all models and compare
    python scripts/surrogate/train_improved_surrogate.py \\
        --training-data StudyResults/surrogate_v9/training_data_500.npz \\
        --output-dir StudyResults/improved_surrogate/ \\
        --mode all

    # Train only RBF+POD
    python scripts/surrogate/train_improved_surrogate.py \\
        --training-data StudyResults/surrogate_v9/training_data_500.npz \\
        --output-dir StudyResults/improved_surrogate/ \\
        --mode rbf

    # Train only NN ensemble
    python scripts/surrogate/train_improved_surrogate.py \\
        --training-data StudyResults/surrogate_v9/training_data_500.npz \\
        --output-dir StudyResults/improved_surrogate/ \\
        --mode nn
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PNPINVERSE_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _PNPINVERSE_ROOT not in sys.path:
    sys.path.insert(0, _PNPINVERSE_ROOT)

import numpy as np

from Surrogate.surrogate_model import SurrogateConfig, BVSurrogateModel
from Surrogate.pod_rbf_model import PODRBFConfig, PODRBFSurrogateModel
from Surrogate.validation import validate_surrogate, print_validation_report
from Surrogate.io import save_surrogate
from Surrogate.cascade import CascadeConfig, run_cascade_inference


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_training_data(path: str) -> dict:
    """Load training data from .npz file.

    Returns
    -------
    dict with parameters, current_density, peroxide_current, phi_applied
    """
    data = np.load(path, allow_pickle=True)
    return {
        "parameters": data["parameters"],
        "current_density": data["current_density"],
        "peroxide_current": data["peroxide_current"],
        "phi_applied": data["phi_applied"],
    }


def _split_data(
    data: dict,
    val_fraction: float,
    seed: int,
) -> tuple[dict, dict]:
    """Split data into train and validation sets.

    Returns
    -------
    (train_data, val_data)
    """
    N = data["parameters"].shape[0]
    rng = np.random.default_rng(seed)
    n_val = max(1, int(N * val_fraction))
    perm = rng.permutation(N)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    train_data = {
        "parameters": data["parameters"][train_idx],
        "current_density": data["current_density"][train_idx],
        "peroxide_current": data["peroxide_current"][train_idx],
        "phi_applied": data["phi_applied"],
    }
    val_data = {
        "parameters": data["parameters"][val_idx],
        "current_density": data["current_density"][val_idx],
        "peroxide_current": data["peroxide_current"][val_idx],
        "phi_applied": data["phi_applied"],
    }
    return train_data, val_data


def _compute_metrics(
    model: object,
    val_data: dict,
    label: str,
) -> dict:
    """Compute validation metrics for a surrogate model.

    Parameters
    ----------
    model : BVSurrogateModel, PODRBFSurrogateModel, or NNSurrogateModel
        Any model implementing the standard surrogate API.
    val_data : dict
        Validation data.
    label : str
        Label for printing.

    Returns
    -------
    dict with cd_rmse, pc_rmse, cd_nrmse, pc_nrmse
    """
    metrics = validate_surrogate(
        model,
        test_parameters=val_data["parameters"],
        test_cd=val_data["current_density"],
        test_pc=val_data["peroxide_current"],
    )
    return {
        "label": label,
        "cd_rmse": metrics["cd_rmse"],
        "pc_rmse": metrics["pc_rmse"],
        "cd_nrmse": metrics["cd_mean_relative_error"],
        "pc_nrmse": metrics["pc_mean_relative_error"],
        "n_test": metrics["n_test"],
    }


def _print_comparison_table(results: list[dict]) -> None:
    """Print a formatted comparison table of surrogate metrics."""
    print(f"\n{'='*80}")
    print(f"  SURROGATE MODEL COMPARISON")
    print(f"{'='*80}")
    header = (
        f"  {'Model':<25s}  {'CD RMSE':>12s}  {'PC RMSE':>12s}  "
        f"{'CD NRMSE':>10s}  {'PC NRMSE':>10s}"
    )
    print(header)
    print(f"  {'-'*75}")
    for r in results:
        print(
            f"  {r['label']:<25s}  {r['cd_rmse']:>12.4e}  {r['pc_rmse']:>12.4e}  "
            f"{r['cd_nrmse']*100:>9.3f}%  {r['pc_nrmse']*100:>9.3f}%"
        )
    print(f"{'='*80}\n")


# ---------------------------------------------------------------------------
# Train RBF+POD
# ---------------------------------------------------------------------------

def train_pod_rbf(
    train_data: dict,
    val_data: dict,
    output_dir: str,
    verbose: bool = True,
) -> tuple[PODRBFSurrogateModel, dict]:
    """Train a POD-RBF surrogate model.

    Returns
    -------
    (model, metrics_dict)
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"  TRAINING POD-RBF SURROGATE")
        print(f"{'='*70}")

    t_start = time.time()

    config = PODRBFConfig(
        variance_threshold=0.999,
        kernel="thin_plate_spline",
        degree=1,
        optimize_smoothing=True,
        n_smoothing_candidates=30,
    )

    model = PODRBFSurrogateModel(config)
    model.fit(
        parameters=train_data["parameters"],
        current_density=train_data["current_density"],
        peroxide_current=train_data["peroxide_current"],
        phi_applied=train_data["phi_applied"],
        verbose=verbose,
    )

    t_fit = time.time() - t_start
    if verbose:
        print(f"  POD-RBF fit time: {t_fit:.2f}s")

    # Validate
    metrics = _compute_metrics(model, val_data, "POD-RBF")
    if verbose:
        print(f"  Validation: CD RMSE={metrics['cd_rmse']:.4e}, "
              f"PC RMSE={metrics['pc_rmse']:.4e}")
        print(f"  CD NRMSE={metrics['cd_nrmse']*100:.3f}%, "
              f"PC NRMSE={metrics['pc_nrmse']*100:.3f}%")

    # Save (use pickle via io module, wrapping in BVSurrogateModel-compatible format)
    pod_dir = os.path.join(output_dir, "pod_rbf")
    os.makedirs(pod_dir, exist_ok=True)

    import pickle
    pkl_path = os.path.join(pod_dir, "pod_rbf_model.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    if verbose:
        print(f"  Saved to: {pkl_path}")

    return model, metrics


# ---------------------------------------------------------------------------
# Train NN ensemble
# ---------------------------------------------------------------------------

def train_nn(
    train_data: dict,
    val_data: dict,
    output_dir: str,
    n_ensemble: int = 5,
    epochs: int = 5000,
    patience: int = 500,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[list, dict, dict]:
    """Train an NN surrogate ensemble.

    Returns
    -------
    (models, ensemble_metrics, per_member_metrics)
    """
    try:
        import torch
    except ImportError:
        print(
            "  SKIP: PyTorch not available. Install with:\n"
            "    pip install torch --index-url https://download.pytorch.org/whl/cpu"
        )
        return [], {}, {}

    from Surrogate.nn_training import (
        NNTrainingConfig,
        train_nn_ensemble,
        predict_ensemble,
    )

    if verbose:
        print(f"\n{'='*70}")
        print(f"  TRAINING NN ENSEMBLE ({n_ensemble} members)")
        print(f"{'='*70}")

    config = NNTrainingConfig(
        epochs=epochs,
        lr=1e-3,
        weight_decay=1e-4,
        patience=patience,
        checkpoint_interval=500,
    )

    t_start = time.time()
    all_params = np.concatenate([
        train_data["parameters"], val_data["parameters"]
    ], axis=0)
    all_cd = np.concatenate([
        train_data["current_density"], val_data["current_density"]
    ], axis=0)
    all_pc = np.concatenate([
        train_data["peroxide_current"], val_data["peroxide_current"]
    ], axis=0)

    models, info = train_nn_ensemble(
        parameters=all_params,
        current_density=all_cd,
        peroxide_current=all_pc,
        phi_applied=train_data["phi_applied"],
        n_ensemble=n_ensemble,
        config=config,
        output_dir=os.path.join(output_dir, "nn_ensemble"),
        base_seed=seed,
        val_fraction=0.15,
        verbose=verbose,
    )
    t_total = time.time() - t_start

    # Per-member metrics
    member_metrics = []
    for i, m in enumerate(models):
        met = _compute_metrics(m, val_data, f"NN member {i}")
        member_metrics.append(met)

    # Ensemble mean metrics
    ens_pred = predict_ensemble(models, val_data["parameters"])
    cd_ens_rmse = float(np.sqrt(np.mean(
        (ens_pred["current_density_mean"] - val_data["current_density"]) ** 2
    )))
    pc_ens_rmse = float(np.sqrt(np.mean(
        (ens_pred["peroxide_current_mean"] - val_data["peroxide_current"]) ** 2
    )))

    # Compute NRMSE for ensemble mean
    N_test = val_data["parameters"].shape[0]
    cd_diff = ens_pred["current_density_mean"] - val_data["current_density"]
    pc_diff = ens_pred["peroxide_current_mean"] - val_data["peroxide_current"]
    cd_per_sample = np.sqrt(np.mean(cd_diff ** 2, axis=1))
    pc_per_sample = np.sqrt(np.mean(pc_diff ** 2, axis=1))
    cd_nrmse_arr = np.zeros(N_test)
    pc_nrmse_arr = np.zeros(N_test)
    for j in range(N_test):
        cd_range = np.ptp(val_data["current_density"][j])
        pc_range = np.ptp(val_data["peroxide_current"][j])
        if cd_range > 1e-12:
            cd_nrmse_arr[j] = cd_per_sample[j] / cd_range
        if pc_range > 1e-12:
            pc_nrmse_arr[j] = pc_per_sample[j] / pc_range

    ens_metrics = {
        "label": "NN Ensemble (mean)",
        "cd_rmse": cd_ens_rmse,
        "pc_rmse": pc_ens_rmse,
        "cd_nrmse": float(cd_nrmse_arr.mean()),
        "pc_nrmse": float(pc_nrmse_arr.mean()),
        "n_test": N_test,
    }

    if verbose:
        print(f"\n  NN Ensemble: CD RMSE={cd_ens_rmse:.4e}, "
              f"PC RMSE={pc_ens_rmse:.4e}")
        print(f"  Total NN training time: {t_total:.1f}s")

    return models, ens_metrics, member_metrics


# ---------------------------------------------------------------------------
# Parameter recovery test
# ---------------------------------------------------------------------------

def run_recovery_test(
    models: dict,
    val_data: dict,
    n_test: int = 5,
    seed: int = 123,
    verbose: bool = True,
) -> None:
    """Run parameter recovery test using cascade inference on each model type.

    Picks n_test random validation samples, uses the model's own prediction
    as the "target", and tries to recover the true parameters via cascade.

    Parameters
    ----------
    models : dict
        {'rbf': model, 'pod_rbf': model, 'nn': [models], ...}
    val_data : dict
    n_test : int
    seed : int
    verbose : bool
    """
    rng = np.random.default_rng(seed)
    N = val_data["parameters"].shape[0]
    test_idx = rng.choice(N, size=min(n_test, N), replace=False)

    if verbose:
        print(f"\n{'='*80}")
        print(f"  PARAMETER RECOVERY TEST ({len(test_idx)} samples)")
        print(f"{'='*80}")

    cascade_cfg = CascadeConfig(
        pass1_weight=0.5,
        pass2_weight=2.0,
        pass1_maxiter=60,
        pass2_maxiter=60,
        polish_maxiter=30,
        skip_polish=False,
        verbose=False,
    )

    for model_name, model in models.items():
        if model is None:
            continue

        # For NN ensemble, use the first member
        if isinstance(model, list):
            if len(model) == 0:
                continue
            surrogate = model[0]
        else:
            surrogate = model

        if not surrogate.is_fitted:
            continue

        if verbose:
            print(f"\n  --- {model_name} ---")

        bounds = surrogate.training_bounds or {
            "k0_1": (1e-6, 1.0),
            "k0_2": (1e-7, 0.1),
            "alpha_1": (0.1, 0.9),
            "alpha_2": (0.1, 0.9),
        }

        errors_k0_1 = []
        errors_k0_2 = []
        errors_a1 = []
        errors_a2 = []

        for ti in test_idx:
            true_params = val_data["parameters"][ti]
            target_cd = val_data["current_density"][ti]
            target_pc = val_data["peroxide_current"][ti]

            # Initial guess: geometric mean of bounds for k0, midpoint for alpha
            k0_1_init = np.sqrt(bounds["k0_1"][0] * bounds["k0_1"][1])
            k0_2_init = np.sqrt(bounds["k0_2"][0] * bounds["k0_2"][1])
            a1_init = 0.5 * (bounds["alpha_1"][0] + bounds["alpha_1"][1])
            a2_init = 0.5 * (bounds["alpha_2"][0] + bounds["alpha_2"][1])

            try:
                result = run_cascade_inference(
                    surrogate=surrogate,
                    target_cd=target_cd,
                    target_pc=target_pc,
                    initial_k0=[k0_1_init, k0_2_init],
                    initial_alpha=[a1_init, a2_init],
                    bounds_k0_1=bounds["k0_1"],
                    bounds_k0_2=bounds["k0_2"],
                    bounds_alpha=bounds["alpha_1"],
                    config=cascade_cfg,
                )

                # Compute relative errors
                err_k0_1 = abs(result.best_k0_1 - true_params[0]) / max(abs(true_params[0]), 1e-30)
                err_k0_2 = abs(result.best_k0_2 - true_params[1]) / max(abs(true_params[1]), 1e-30)
                err_a1 = abs(result.best_alpha_1 - true_params[2])
                err_a2 = abs(result.best_alpha_2 - true_params[3])

                errors_k0_1.append(err_k0_1)
                errors_k0_2.append(err_k0_2)
                errors_a1.append(err_a1)
                errors_a2.append(err_a2)

            except Exception as e:
                if verbose:
                    print(f"    Recovery failed for sample {ti}: {e}")

        if errors_k0_1 and verbose:
            print(f"    k0_1 rel error: mean={np.mean(errors_k0_1)*100:.2f}%, "
                  f"max={np.max(errors_k0_1)*100:.2f}%")
            print(f"    k0_2 rel error: mean={np.mean(errors_k0_2)*100:.2f}%, "
                  f"max={np.max(errors_k0_2)*100:.2f}%")
            print(f"    alpha_1 abs error: mean={np.mean(errors_a1):.4f}, "
                  f"max={np.max(errors_a1):.4f}")
            print(f"    alpha_2 abs error: mean={np.mean(errors_a2):.4f}, "
                  f"max={np.max(errors_a2):.4f}")

    if verbose:
        print(f"\n{'='*80}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and compare improved surrogate models",
    )
    parser.add_argument(
        "--training-data", type=str, required=True,
        help="Path to training data .npz file",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="StudyResults/improved_surrogate",
        help="Output directory",
    )
    parser.add_argument(
        "--mode", type=str, default="all",
        choices=["all", "rbf", "nn"],
        help="Which models to train (default: all)",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.15,
        help="Validation fraction (default: 0.15)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--n-ensemble", type=int, default=5,
        help="Number of NN ensemble members (default: 5)",
    )
    parser.add_argument(
        "--epochs", type=int, default=5000,
        help="NN training epochs (default: 5000)",
    )
    parser.add_argument(
        "--patience", type=int, default=500,
        help="NN early stopping patience (default: 500)",
    )
    parser.add_argument(
        "--skip-recovery", action="store_true",
        help="Skip parameter recovery test",
    )
    args = parser.parse_args()

    print(f"\n{'#'*70}")
    print(f"  IMPROVED SURROGATE TRAINING PIPELINE")
    print(f"  Mode          : {args.mode}")
    print(f"  Training data : {args.training_data}")
    print(f"  Output dir    : {args.output_dir}")
    print(f"{'#'*70}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load and split data
    data = _load_training_data(args.training_data)
    N = data["parameters"].shape[0]
    n_eta = data["phi_applied"].shape[0]
    print(f"  Loaded {N} samples, {n_eta} voltage points")

    train_data, val_data = _split_data(data, args.val_fraction, args.seed)
    print(f"  Train/val split: {train_data['parameters'].shape[0]}/{val_data['parameters'].shape[0]}")

    comparison_results: list[dict] = []
    trained_models: dict = {}

    # --- Standard RBF (baseline) ---
    if args.mode in ("all", "rbf"):
        print(f"\n{'='*70}")
        print(f"  BASELINE: Standard RBF Surrogate")
        print(f"{'='*70}")
        t0 = time.time()
        rbf_config = SurrogateConfig(
            kernel="thin_plate_spline", degree=1,
            smoothing=0.0, log_space_k0=True, normalize_inputs=True,
        )
        rbf_model = BVSurrogateModel(rbf_config)
        rbf_model.fit(
            parameters=train_data["parameters"],
            current_density=train_data["current_density"],
            peroxide_current=train_data["peroxide_current"],
            phi_applied=train_data["phi_applied"],
        )
        rbf_metrics = _compute_metrics(rbf_model, val_data, "RBF (baseline)")
        comparison_results.append(rbf_metrics)
        trained_models["RBF (baseline)"] = rbf_model
        print(f"  RBF fit time: {time.time()-t0:.2f}s")
        print(f"  CD RMSE={rbf_metrics['cd_rmse']:.4e}, "
              f"PC RMSE={rbf_metrics['pc_rmse']:.4e}")

    # --- POD-RBF ---
    if args.mode in ("all", "rbf"):
        pod_model, pod_metrics = train_pod_rbf(
            train_data, val_data, args.output_dir,
        )
        comparison_results.append(pod_metrics)
        trained_models["POD-RBF"] = pod_model

    # --- NN ensemble ---
    nn_models = []
    if args.mode in ("all", "nn"):
        nn_models_list, ens_metrics, member_metrics = train_nn(
            train_data, val_data, args.output_dir,
            n_ensemble=args.n_ensemble,
            epochs=args.epochs,
            patience=args.patience,
            seed=args.seed,
        )
        nn_models = nn_models_list
        if ens_metrics:
            comparison_results.append(ens_metrics)
            # Also add best individual member
            if member_metrics:
                best_member = min(member_metrics, key=lambda m: m["cd_rmse"])
                best_member_entry = dict(best_member)
                best_member_entry["label"] = "NN (best member)"
                comparison_results.append(best_member_entry)
            trained_models["NN Ensemble"] = nn_models_list

    # --- Comparison table ---
    if comparison_results:
        _print_comparison_table(comparison_results)

    # --- Parameter recovery test ---
    if not args.skip_recovery and trained_models:
        run_recovery_test(
            trained_models, val_data,
            n_test=5, seed=args.seed + 1000,
        )

    print(f"\n{'#'*70}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Output: {args.output_dir}")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
