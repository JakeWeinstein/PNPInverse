"""Build a BV surrogate model from training data.

Loads .npz training data, splits off validation set, fits RBF surrogate,
validates, and saves to .pkl.

v9 additions (Round 2B):
    - Cross-validation sweep for peroxide_current smoothing parameter
    - Separate smoothing_cd / smoothing_pc in SurrogateConfig
    - Training bounds stored in the surrogate model

Usage (from PNPInverse/ directory)::

    python scripts/surrogate/build_surrogate.py \\
        --training-data StudyResults/surrogate/training_data.npz \\
        --output StudyResults/surrogate/surrogate_model.pkl

    # With cross-validation for PC smoothing:
    python scripts/surrogate/build_surrogate.py \\
        --training-data StudyResults/surrogate_v9/training_data_500.npz \\
        --output StudyResults/surrogate_v9/surrogate_model.pkl \\
        --cross-validate-smoothing --test-fraction 0.1
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
from Surrogate.validation import validate_surrogate, print_validation_report
from Surrogate.io import save_surrogate


def _cross_validate_pc_smoothing(
    parameters_train: np.ndarray,
    cd_train: np.ndarray,
    pc_train: np.ndarray,
    parameters_val: np.ndarray,
    cd_val: np.ndarray,
    pc_val: np.ndarray,
    phi_applied: np.ndarray,
    base_config: SurrogateConfig,
    smoothing_values: list[float],
) -> tuple[float, dict[float, dict]]:
    """Sweep PC smoothing values and return the best one based on val NRMSE.

    CD smoothing is fixed at ``base_config.smoothing`` (or
    ``base_config.smoothing_cd`` if set).

    Parameters
    ----------
    parameters_train, cd_train, pc_train : np.ndarray
        Training data.
    parameters_val, cd_val, pc_val : np.ndarray
        Validation data.
    phi_applied : np.ndarray
        Voltage grid.
    base_config : SurrogateConfig
        Base config; only ``smoothing_pc`` is varied.
    smoothing_values : list of float
        Candidate smoothing values to try.

    Returns
    -------
    best_smoothing_pc : float
        The smoothing value with lowest PC validation NRMSE.
    results : dict
        ``{smoothing: {"pc_nrmse": ..., "cd_nrmse": ..., "pc_rmse": ...}}``
    """
    results: dict[float, dict] = {}
    best_sm = smoothing_values[0]
    best_pc_nrmse = float("inf")

    print(f"\n  {'='*65}")
    print(f"  CROSS-VALIDATION: PC smoothing sweep")
    print(f"  Candidates: {smoothing_values}")
    print(f"  {'='*65}")

    for sm in smoothing_values:
        cfg = SurrogateConfig(
            kernel=base_config.kernel,
            degree=base_config.degree,
            smoothing=base_config.smoothing,
            smoothing_cd=base_config.smoothing_cd,
            smoothing_pc=sm,
            log_space_k0=base_config.log_space_k0,
            normalize_inputs=base_config.normalize_inputs,
        )
        model = BVSurrogateModel(cfg)
        model.fit(
            parameters=parameters_train,
            current_density=cd_train,
            peroxide_current=pc_train,
            phi_applied=phi_applied,
        )
        metrics = validate_surrogate(
            model,
            test_parameters=parameters_val,
            test_cd=cd_val,
            test_pc=pc_val,
        )
        pc_nrmse = metrics["pc_mean_relative_error"]
        cd_nrmse = metrics["cd_mean_relative_error"]
        pc_rmse = metrics["pc_rmse"]
        cd_rmse = metrics["cd_rmse"]

        results[sm] = {
            "pc_nrmse": pc_nrmse,
            "cd_nrmse": cd_nrmse,
            "pc_rmse": pc_rmse,
            "cd_rmse": cd_rmse,
        }

        marker = ""
        if pc_nrmse < best_pc_nrmse:
            best_pc_nrmse = pc_nrmse
            best_sm = sm
            marker = " <-- best"

        print(f"    smoothing_pc={sm:<10.1e}  "
              f"CD NRMSE={cd_nrmse*100:>7.3f}%  "
              f"PC NRMSE={pc_nrmse*100:>7.3f}%  "
              f"PC RMSE={pc_rmse:.4e}{marker}")

    print(f"\n  Best PC smoothing: {best_sm:.1e} (PC NRMSE={best_pc_nrmse*100:.3f}%)")
    print(f"  {'='*65}")

    return best_sm, results


def main():
    parser = argparse.ArgumentParser(description="Build BV surrogate model")
    parser.add_argument("--training-data", type=str, required=True,
                        help="Path to training data .npz file")
    parser.add_argument("--output", type=str,
                        default="StudyResults/surrogate/surrogate_model.pkl",
                        help="Output .pkl file path")
    parser.add_argument("--validation-split", type=float, default=0.1,
                        help="Fraction of data for validation (default 0.1)")
    parser.add_argument("--kernel", type=str, default="thin_plate_spline",
                        help="RBF kernel (default: thin_plate_spline)")
    parser.add_argument("--degree", type=int, default=1,
                        help="Polynomial degree (default: 1)")
    parser.add_argument("--smoothing", type=float, default=0.0,
                        help="RBF smoothing parameter (default: 0.0)")
    parser.add_argument("--smoothing-cd", type=float, default=None,
                        help="CD-specific smoothing (overrides --smoothing for CD)")
    parser.add_argument("--smoothing-pc", type=float, default=None,
                        help="PC-specific smoothing (overrides --smoothing for PC)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/val split")
    parser.add_argument("--cross-validate-smoothing", action="store_true",
                        help="Run cross-validation sweep for PC smoothing")
    parser.add_argument("--test-fraction", type=float, default=0.1,
                        help="Holdout fraction for CV smoothing sweep (default 0.1)")
    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print(f"  BUILD BV SURROGATE MODEL")
    print(f"  Training data: {args.training_data}")
    print(f"  Output: {args.output}")
    if args.cross_validate_smoothing:
        print(f"  Cross-validation: ON (test fraction={args.test_fraction})")
    print(f"{'#'*60}\n")

    # Load training data
    data = np.load(args.training_data, allow_pickle=True)
    parameters = data["parameters"]
    cd = data["current_density"]
    pc = data["peroxide_current"]
    phi_applied = data["phi_applied"]

    N = parameters.shape[0]
    n_eta = phi_applied.shape[0]
    print(f"  Loaded {N} valid samples, {n_eta} voltage points")
    print(f"  Voltage range: [{phi_applied.min():.1f}, {phi_applied.max():.1f}]")
    print(f"  k0_1 range: [{parameters[:,0].min():.4e}, {parameters[:,0].max():.4e}]")
    print(f"  k0_2 range: [{parameters[:,1].min():.4e}, {parameters[:,1].max():.4e}]")
    print(f"  alpha_1 range: [{parameters[:,2].min():.4f}, {parameters[:,2].max():.4f}]")
    print(f"  alpha_2 range: [{parameters[:,3].min():.4f}, {parameters[:,3].max():.4f}]")

    # Train/validation split
    rng = np.random.default_rng(args.seed)
    n_val = max(1, int(N * args.validation_split))
    n_train = N - n_val

    perm = rng.permutation(N)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    print(f"\n  Train/val split: {n_train}/{n_val}")

    # Resolve smoothing values
    smoothing_cd = args.smoothing_cd
    smoothing_pc = args.smoothing_pc

    # Cross-validate PC smoothing if requested
    if args.cross_validate_smoothing:
        cv_config = SurrogateConfig(
            kernel=args.kernel,
            degree=args.degree,
            smoothing=args.smoothing,
            smoothing_cd=smoothing_cd,
            smoothing_pc=None,  # will be varied
            log_space_k0=True,
            normalize_inputs=True,
        )
        smoothing_candidates = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1]
        best_sm_pc, cv_results = _cross_validate_pc_smoothing(
            parameters_train=parameters[train_idx],
            cd_train=cd[train_idx],
            pc_train=pc[train_idx],
            parameters_val=parameters[val_idx],
            cd_val=cd[val_idx],
            pc_val=pc[val_idx],
            phi_applied=phi_applied,
            base_config=cv_config,
            smoothing_values=smoothing_candidates,
        )
        smoothing_pc = best_sm_pc

    # Build surrogate config
    config = SurrogateConfig(
        kernel=args.kernel,
        degree=args.degree,
        smoothing=args.smoothing,
        smoothing_cd=smoothing_cd,
        smoothing_pc=smoothing_pc,
        log_space_k0=True,
        normalize_inputs=True,
    )
    sm_cd_eff = smoothing_cd if smoothing_cd is not None else args.smoothing
    sm_pc_eff = smoothing_pc if smoothing_pc is not None else args.smoothing
    print(f"\n  Config: kernel={config.kernel}, degree={config.degree}")
    print(f"    smoothing_cd={sm_cd_eff}, smoothing_pc={sm_pc_eff}")

    # Fit model
    t_fit_start = time.time()
    model = BVSurrogateModel(config)
    model.fit(
        parameters=parameters[train_idx],
        current_density=cd[train_idx],
        peroxide_current=pc[train_idx],
        phi_applied=phi_applied,
    )
    t_fit = time.time() - t_fit_start
    print(f"  Fit time: {t_fit:.2f}s")

    # Print training bounds
    if model.training_bounds is not None:
        print(f"\n  Training bounds (stored in model):")
        for name, (lo, hi) in model.training_bounds.items():
            if name.startswith("k0"):
                print(f"    {name}: [{lo:.4e}, {hi:.4e}] (log10: [{np.log10(max(lo,1e-30)):.2f}, {np.log10(hi):.2f}])")
            else:
                print(f"    {name}: [{lo:.4f}, {hi:.4f}]")

    # Validate on held-out data
    if n_val > 0:
        metrics = validate_surrogate(
            model,
            test_parameters=parameters[val_idx],
            test_cd=cd[val_idx],
            test_pc=pc[val_idx],
        )
        print_validation_report(metrics)
    else:
        print("  No validation data (all samples used for training)")

    # Also validate on training data (check for overfitting)
    if n_train > 5:
        train_metrics = validate_surrogate(
            model,
            test_parameters=parameters[train_idx],
            test_cd=cd[train_idx],
            test_pc=pc[train_idx],
        )
        print(f"\n  Training set metrics (sanity check):")
        print(f"    CD RMSE: {train_metrics['cd_rmse']:.6e}")
        print(f"    PC RMSE: {train_metrics['pc_rmse']:.6e}")
        print(f"    CD mean NRMSE: {train_metrics['cd_mean_relative_error']*100:.2f}%")
        print(f"    PC mean NRMSE: {train_metrics['pc_mean_relative_error']*100:.2f}%")

    # Test prediction speed
    t_pred_start = time.time()
    n_pred_test = 1000
    test_params = parameters[train_idx[:min(n_train, n_pred_test)]]
    _ = model.predict_batch(test_params)
    t_pred = time.time() - t_pred_start
    print(f"\n  Prediction speed: {len(test_params)} samples in {t_pred*1000:.1f}ms "
          f"({t_pred/len(test_params)*1e6:.1f} us/sample)")

    # Save
    save_surrogate(model, args.output)

    print(f"\n{'#'*60}")
    print(f"  BUILD COMPLETE")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
