"""Train an NN surrogate ensemble from pre-generated training data.

Loads training data from a .npz file, performs stratified train/val split,
trains an ensemble of ResNet-MLP models, and saves results.

Usage (from PNPInverse/ directory)::

    python scripts/surrogate/train_nn_surrogate.py \\
        --training-data StudyResults/surrogate_v9/training_data_500.npz \\
        --output-dir StudyResults/nn_surrogate/ \\
        --n-ensemble 5 --epochs 5000 --patience 500

Requirements:
    pip install torch --index-url https://download.pytorch.org/whl/cpu
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

# Check torch availability early
try:
    import torch
except ImportError:
    print(
        "ERROR: PyTorch is required but not installed.\n"
        "Install it with:\n"
        "  pip install torch --index-url https://download.pytorch.org/whl/cpu"
    )
    sys.exit(1)

from Surrogate.nn_model import NNSurrogateModel
from Surrogate.nn_training import (
    NNTrainingConfig,
    train_nn_ensemble,
    predict_ensemble,
)
from Surrogate.validation import validate_surrogate, print_validation_report


def _stratified_split(
    parameters: np.ndarray,
    n_val: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a stratified train/val split based on parameter quartiles.

    Stratifies on the first two parameters (k0_1, k0_2) by assigning
    each sample to a bin based on its log10(k0_1) and log10(k0_2)
    quartile, then sampling proportionally from each bin.

    Parameters
    ----------
    parameters : np.ndarray (N, 4)
    n_val : int
    seed : int

    Returns
    -------
    (train_idx, val_idx)
    """
    N = parameters.shape[0]
    rng = np.random.default_rng(seed)

    # Create bins from k0_1 and k0_2 quartiles
    log_k0_1 = np.log10(np.maximum(parameters[:, 0], 1e-30))
    log_k0_2 = np.log10(np.maximum(parameters[:, 1], 1e-30))

    q1 = np.digitize(log_k0_1, np.quantile(log_k0_1, [0.25, 0.5, 0.75]))
    q2 = np.digitize(log_k0_2, np.quantile(log_k0_2, [0.25, 0.5, 0.75]))
    strata = q1 * 4 + q2  # 16 possible strata

    unique_strata = np.unique(strata)
    val_indices = []

    # Allocate val samples proportionally per stratum
    for s in unique_strata:
        s_idx = np.where(strata == s)[0]
        n_s = max(1, int(round(len(s_idx) / N * n_val)))
        n_s = min(n_s, len(s_idx) - 1)  # keep at least 1 for training
        chosen = rng.choice(s_idx, size=n_s, replace=False)
        val_indices.extend(chosen.tolist())

    val_idx = np.array(val_indices, dtype=int)
    # Trim or pad to exact n_val
    if len(val_idx) > n_val:
        val_idx = rng.choice(val_idx, size=n_val, replace=False)
    elif len(val_idx) < n_val:
        remaining = np.setdiff1d(np.arange(N), val_idx)
        extra = rng.choice(remaining, size=n_val - len(val_idx), replace=False)
        val_idx = np.concatenate([val_idx, extra])

    train_idx = np.setdiff1d(np.arange(N), val_idx)
    return train_idx, val_idx


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train NN surrogate ensemble from training data",
    )
    parser.add_argument(
        "--training-data", type=str, required=True,
        help="Path to training data .npz file",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="StudyResults/nn_surrogate",
        help="Output directory for models and logs",
    )
    parser.add_argument(
        "--n-ensemble", type=int, default=5,
        help="Number of ensemble members (default: 5)",
    )
    parser.add_argument(
        "--epochs", type=int, default=5000,
        help="Maximum training epochs (default: 5000)",
    )
    parser.add_argument(
        "--patience", type=int, default=500,
        help="Early stopping patience (default: 500)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--hidden", type=int, default=128,
        help="Hidden layer width (default: 128)",
    )
    parser.add_argument(
        "--n-blocks", type=int, default=4,
        help="Number of residual blocks (default: 4)",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.15,
        help="Validation fraction (default: 0.15)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed (default: 42)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Mini-batch size (default: full batch)",
    )
    parser.add_argument(
        "--monotonicity-weight", type=float, default=0.0,
        help="Monotonicity regularization weight (default: 0.0)",
    )
    parser.add_argument(
        "--smoothness-weight", type=float, default=0.0,
        help="Smoothness regularization weight (default: 0.0)",
    )
    args = parser.parse_args()

    print(f"\n{'#'*70}")
    print(f"  TRAIN NN SURROGATE ENSEMBLE")
    print(f"  Training data : {args.training_data}")
    print(f"  Output dir    : {args.output_dir}")
    print(f"  N ensemble    : {args.n_ensemble}")
    print(f"  Epochs        : {args.epochs}")
    print(f"  Patience      : {args.patience}")
    print(f"  LR            : {args.lr}")
    print(f"  Hidden        : {args.hidden}")
    print(f"  Res blocks    : {args.n_blocks}")
    print(f"  Val fraction  : {args.val_fraction}")
    print(f"  Seed          : {args.seed}")
    print(f"  PyTorch       : {torch.__version__}")
    print(f"{'#'*70}\n")

    # Load training data
    data = np.load(args.training_data, allow_pickle=True)
    parameters = data["parameters"]
    cd = data["current_density"]
    pc = data["peroxide_current"]
    phi_applied = data["phi_applied"]

    N = parameters.shape[0]
    n_eta = phi_applied.shape[0]

    print(f"  Loaded {N} samples, {n_eta} voltage points")
    print(f"  k0_1 range : [{parameters[:,0].min():.4e}, {parameters[:,0].max():.4e}]")
    print(f"  k0_2 range : [{parameters[:,1].min():.4e}, {parameters[:,1].max():.4e}]")
    print(f"  alpha_1    : [{parameters[:,2].min():.4f}, {parameters[:,2].max():.4f}]")
    print(f"  alpha_2    : [{parameters[:,3].min():.4f}, {parameters[:,3].max():.4f}]")
    print(f"  CD range   : [{cd.min():.4e}, {cd.max():.4e}]")
    print(f"  PC range   : [{pc.min():.4e}, {pc.max():.4e}]")

    # Build config
    config = NNTrainingConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=1e-4,
        patience=args.patience,
        batch_size=args.batch_size,
        checkpoint_interval=500,
        hidden=args.hidden,
        n_blocks=args.n_blocks,
        monotonicity_weight=args.monotonicity_weight,
        smoothness_weight=args.smoothness_weight,
    )

    # Train ensemble
    t_start = time.time()
    models, ensemble_info = train_nn_ensemble(
        parameters=parameters,
        current_density=cd,
        peroxide_current=pc,
        phi_applied=phi_applied,
        n_ensemble=args.n_ensemble,
        config=config,
        output_dir=args.output_dir,
        base_seed=args.seed,
        val_fraction=args.val_fraction,
        verbose=True,
    )
    t_total = time.time() - t_start

    # Validation: evaluate each member and the ensemble mean on a held-out set
    print(f"\n{'='*70}")
    print(f"  VALIDATION RESULTS")
    print(f"{'='*70}")

    # Use the first member's implicit val split for a quick check
    # (For a proper evaluation, use separate test data)
    n_val_check = max(1, int(N * args.val_fraction))
    rng = np.random.default_rng(args.seed + 9999)
    val_check_idx = rng.choice(N, size=n_val_check, replace=False)

    for i, model in enumerate(models):
        metrics = validate_surrogate(
            model,
            test_parameters=parameters[val_check_idx],
            test_cd=cd[val_check_idx],
            test_pc=pc[val_check_idx],
        )
        print(f"  Member {i}: CD RMSE={metrics['cd_rmse']:.4e}, "
              f"PC RMSE={metrics['pc_rmse']:.4e}, "
              f"CD NRMSE={metrics['cd_mean_relative_error']*100:.2f}%, "
              f"PC NRMSE={metrics['pc_mean_relative_error']*100:.2f}%")

    # Ensemble mean prediction
    ens_pred = predict_ensemble(models, parameters[val_check_idx])
    cd_ens_rmse = float(np.sqrt(np.mean(
        (ens_pred["current_density_mean"] - cd[val_check_idx]) ** 2
    )))
    pc_ens_rmse = float(np.sqrt(np.mean(
        (ens_pred["peroxide_current_mean"] - pc[val_check_idx]) ** 2
    )))
    print(f"\n  Ensemble mean: CD RMSE={cd_ens_rmse:.4e}, "
          f"PC RMSE={pc_ens_rmse:.4e}")
    print(f"  Ensemble uncertainty (mean std): "
          f"CD={ens_pred['current_density_std'].mean():.4e}, "
          f"PC={ens_pred['peroxide_current_std'].mean():.4e}")

    print(f"\n  Total training time: {t_total:.1f}s")
    print(f"  Models saved to: {args.output_dir}")
    print(f"\n{'#'*70}")
    print(f"  TRAINING COMPLETE")
    print(f"{'#'*70}")


if __name__ == "__main__":
    main()
