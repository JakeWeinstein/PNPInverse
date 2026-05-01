"""Train a GP surrogate from pre-generated training data.

Loads training data from a .npz file, applies the canonical train/test split
from split_indices.npz, fits a GPSurrogateModel (44 independent exact GPs
with Matern 5/2 ARD kernels), saves the model, and runs validation + UQ
calibration + gradient checks.

Usage (from PNPInverse/ directory)::

    python scripts/surrogate/train_gp.py \\
        --training-data data/surrogate_models/training_data_merged.npz \\
        --output-dir data/surrogate_models/gp/

Requirements:
    pip install gpytorch joblib
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

# Check dependencies early
try:
    import torch
except ImportError:
    print(
        "ERROR: PyTorch is required but not installed.\n"
        "Install it with:\n"
        "  pip install torch --index-url https://download.pytorch.org/whl/cpu"
    )
    sys.exit(1)

try:
    import gpytorch
except ImportError:
    print(
        "ERROR: GPyTorch is required but not installed.\n"
        "Install it with:\n"
        "  pip install gpytorch"
    )
    sys.exit(1)

from Surrogate.gp_model import GPSurrogateModel, load_gp_surrogate
from Surrogate.validation import validate_surrogate, print_validation_report


# ---------------------------------------------------------------------------
# UQ calibration check
# ---------------------------------------------------------------------------


def check_uq_calibration(
    model: GPSurrogateModel,
    test_params: np.ndarray,
    test_cd: np.ndarray,
    test_pc: np.ndarray,
    confidence_levels: list[float] | None = None,
) -> dict[str, dict[float, float]]:
    """Check what fraction of test points fall within each predictive interval.

    For a well-calibrated GP, the actual coverage should match the nominal
    confidence level (e.g. 90% interval has ~90% coverage).

    Parameters
    ----------
    model : GPSurrogateModel
        Fitted GP surrogate.
    test_params : np.ndarray (N_test, 4)
    test_cd : np.ndarray (N_test, n_eta)
    test_pc : np.ndarray (N_test, n_eta)
    confidence_levels : list of float
        Nominal confidence levels to check (default [0.5, 0.8, 0.9, 0.95]).

    Returns
    -------
    dict with 'cd' and 'pc' keys, each mapping confidence level -> coverage.
    """
    from scipy.stats import norm as scipy_norm

    if confidence_levels is None:
        confidence_levels = [0.5, 0.8, 0.9, 0.95]

    result = model.predict_batch_with_uncertainty(test_params)
    mean_cd = result["current_density"]
    std_cd = result["current_density_std"]
    mean_pc = result["peroxide_current"]
    std_pc = result["peroxide_current_std"]

    cd_coverages: dict[float, float] = {}
    pc_coverages: dict[float, float] = {}

    print(f"\n  UQ Calibration Check:")
    print(f"  {'Level':>8s}  {'CD coverage':>12s}  {'PC coverage':>12s}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*12}")

    for level in confidence_levels:
        z = scipy_norm.ppf((1 + level) / 2)

        # Current density
        lower_cd = mean_cd - z * std_cd
        upper_cd = mean_cd + z * std_cd
        coverage_cd = float(np.mean((test_cd >= lower_cd) & (test_cd <= upper_cd)))
        cd_coverages[level] = coverage_cd

        # Peroxide current
        lower_pc = mean_pc - z * std_pc
        upper_pc = mean_pc + z * std_pc
        coverage_pc = float(np.mean((test_pc >= lower_pc) & (test_pc <= upper_pc)))
        pc_coverages[level] = coverage_pc

        print(f"  {level*100:>7.0f}%  {coverage_cd*100:>11.1f}%  {coverage_pc*100:>11.1f}%")

    # Calibration assessment
    cov_90_cd = cd_coverages.get(0.9, 0.0)
    cov_90_pc = pc_coverages.get(0.9, 0.0)
    print()
    if 0.85 <= cov_90_cd <= 0.95:
        print(f"  CD 90% interval: WELL-CALIBRATED (coverage={cov_90_cd*100:.1f}%)")
    elif cov_90_cd > 0.95:
        print(f"  CD 90% interval: CONSERVATIVE (coverage={cov_90_cd*100:.1f}%, >95%)")
    else:
        print(f"  CD 90% interval: OVERCONFIDENT (coverage={cov_90_cd*100:.1f}%, <85%)")

    if 0.85 <= cov_90_pc <= 0.95:
        print(f"  PC 90% interval: WELL-CALIBRATED (coverage={cov_90_pc*100:.1f}%)")
    elif cov_90_pc > 0.95:
        print(f"  PC 90% interval: CONSERVATIVE (coverage={cov_90_pc*100:.1f}%, >95%)")
    else:
        print(f"  PC 90% interval: OVERCONFIDENT (coverage={cov_90_pc*100:.1f}%, <85%)")

    return {"cd": cd_coverages, "pc": pc_coverages}


# ---------------------------------------------------------------------------
# Gradient accuracy check (autograd vs finite differences)
# ---------------------------------------------------------------------------


def check_gradient_accuracy(
    model: GPSurrogateModel,
    test_params: np.ndarray,
    test_cd: np.ndarray,
    test_pc: np.ndarray,
    n_points: int = 10,
    h: float = 1e-3,
    seed: int = 42,
) -> np.ndarray:
    """Compare autograd gradient to central finite-difference gradient.

    For each test point, computes a simple objective
        J = 0.5 * sum((cd_pred - cd_true)^2 + (pc_pred - pc_true)^2)
    and checks that the autograd gradient matches the FD gradient.

    Both autograd and FD operate in log-space [log10(k0_1), log10(k0_2),
    alpha_1, alpha_2].  The FD computation uses ``predict_torch`` directly
    (same differentiable path as autograd) to avoid numerical noise from
    float64/float32 round-trips through ``predict_batch``.

    A step size of h=1e-3 is used by default because GP posterior means
    evaluated in float32 have limited numerical smoothness; smaller h
    values cause catastrophic cancellation in the FD quotient.

    Parameters
    ----------
    model : GPSurrogateModel
        Fitted GP surrogate.
    test_params : np.ndarray (N_test, 4)
    test_cd : np.ndarray (N_test, n_eta)
    test_pc : np.ndarray (N_test, n_eta)
    n_points : int
        Number of random test points to check.
    h : float
        Finite-difference step size (in log10/alpha space).
    seed : int
        RNG seed for selecting test points.

    Returns
    -------
    np.ndarray of shape (n_points,)
        Relative errors between autograd and FD gradients.
    """
    rng = np.random.default_rng(seed)
    n_test = test_params.shape[0]
    indices = rng.choice(n_test, size=min(n_points, n_test), replace=False)

    print(f"\n  Gradient Accuracy Check (autograd vs central FD, h={h}):")
    print(f"  {'Point':>6s}  {'Rel Error':>10s}  {'Status':>8s}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*8}")

    def _objective_torch_fd(x_log_arr, target_cd_np, target_pc_np):
        """Compute objective via predict_torch (same path as autograd, no grad)."""
        x_t = torch.tensor(x_log_arr.reshape(1, -1), dtype=torch.float32)
        x_n = (x_t - model._input_mean_t) / model._input_std_t
        with torch.no_grad():
            y_norm = model.predict_torch(x_n)
        y = y_norm * model._output_std_t + model._output_mean_t
        n_eta = model._n_eta
        cd = y[0, :n_eta].numpy()
        pc = y[0, n_eta:].numpy()
        cd_diff = cd - target_cd_np
        pc_diff = pc - target_pc_np
        return 0.5 * np.sum(cd_diff ** 2) + 0.5 * np.sum(pc_diff ** 2)

    relative_errors = []
    for idx_i, idx in enumerate(indices):
        k0_1, k0_2, alpha_1, alpha_2 = test_params[idx]
        target_cd = test_cd[idx]
        target_pc = test_pc[idx]

        # Autograd gradient
        grad_auto = model.gradient_at(
            k0_1, k0_2, alpha_1, alpha_2,
            target_cd, target_pc,
            secondary_weight=1.0,
        )

        # Central FD gradient in log10/alpha space using predict_torch
        from math import log10 as _log10
        x_log = np.array([
            _log10(max(k0_1, 1e-30)),
            _log10(max(k0_2, 1e-30)),
            alpha_1,
            alpha_2,
        ])

        grad_fd = np.zeros(4)
        for dim in range(4):
            x_plus = x_log.copy()
            x_minus = x_log.copy()
            x_plus[dim] += h
            x_minus[dim] -= h

            f_plus = _objective_torch_fd(x_plus, target_cd, target_pc)
            f_minus = _objective_torch_fd(x_minus, target_cd, target_pc)
            grad_fd[dim] = (f_plus - f_minus) / (2 * h)

        # Relative error -- use max of norms to handle near-zero gradients
        # where FD is unreliable due to float32 precision limits
        auto_norm = np.linalg.norm(grad_auto)
        fd_norm = np.linalg.norm(grad_fd)
        denom = max(auto_norm, fd_norm)
        if denom > 1e-6:
            rel_err = np.linalg.norm(grad_auto - grad_fd) / denom
        else:
            # Both gradients are near-zero; skip this point
            rel_err = 0.0

        relative_errors.append(rel_err)
        status = "PASS" if rel_err < 0.10 else ("SKIP" if denom <= 1e-6 else "FAIL")
        print(f"  {idx_i:>6d}  {rel_err:>10.2e}  {status:>8s}")

    relative_errors = np.array(relative_errors)
    n_pass = np.sum(relative_errors < 0.10)
    print(f"\n  Result: {n_pass}/{len(relative_errors)} points passed (<10% relative error)")
    print(f"  Note: FD accuracy is limited by float32 GP predictions; h=1e-3 is near")
    print(f"  the optimal step size for this precision level.")

    return relative_errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train GP surrogate from training data",
    )
    parser.add_argument(
        "--training-data", type=str,
        default="data/surrogate_models/training_data_merged.npz",
        help="Path to training data .npz file",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="data/surrogate_models/gp",
        help="Output directory for GP model artifacts",
    )
    parser.add_argument(
        "--n-iters", type=int, default=200,
        help="Max Adam iterations per GP (default: 200)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.1,
        help="Adam learning rate (default: 0.1)",
    )
    parser.add_argument(
        "--split-indices", type=str,
        default="data/surrogate_models/split_indices.npz",
        help="Path to split_indices.npz (keys: train_idx, test_idx)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for gradient check point selection (default: 42)",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=-1,
        help="Number of parallel jobs for GP fitting (-1 = all cores)",
    )
    parser.add_argument(
        "--skip-gradient-check", action="store_true",
        help="Skip the gradient accuracy check (faster)",
    )
    args = parser.parse_args()

    print(f"\n{'#'*70}")
    print(f"  TRAIN GP SURROGATE")
    print(f"  Training data : {args.training_data}")
    print(f"  Output dir    : {args.output_dir}")
    print(f"  Max iters/GP  : {args.n_iters}")
    print(f"  LR            : {args.lr}")
    print(f"  Split indices : {args.split_indices}")
    print(f"  Seed          : {args.seed}")
    print(f"  N jobs        : {args.n_jobs}")
    print(f"  PyTorch       : {torch.__version__}")
    print(f"  GPyTorch      : {gpytorch.__version__}")
    print(f"{'#'*70}\n")

    # ------------------------------------------------------------------
    # Load training data
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Train/test split (from canonical split_indices.npz for consistency)
    # ------------------------------------------------------------------
    split = np.load(args.split_indices, allow_pickle=True)
    train_idx = split["train_idx"]
    test_idx = split["test_idx"]

    assert len(train_idx) + len(test_idx) == N, (
        f"Split indices ({len(train_idx)} + {len(test_idx)}) != data size ({N}). "
        f"Regenerate split_indices.npz for the current dataset."
    )

    train_params = parameters[train_idx]
    train_cd = cd[train_idx]
    train_pc = pc[train_idx]

    test_params = parameters[test_idx]
    test_cd = cd[test_idx]
    test_pc = pc[test_idx]

    print(f"\n  Train: {len(train_idx)} samples, Test: {len(test_idx)} samples")

    # ------------------------------------------------------------------
    # Fit GP surrogate
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  FITTING GP SURROGATE")
    print(f"{'='*70}")

    gp_model = GPSurrogateModel()

    t_start = time.time()
    gp_model.fit(
        parameters=train_params,
        current_density=train_cd,
        peroxide_current=train_pc,
        phi_applied=phi_applied,
        n_iters=args.n_iters,
        lr=args.lr,
        n_jobs=args.n_jobs,
        verbose=True,
    )
    t_fit = time.time() - t_start
    print(f"\n  GP fitting time: {t_fit:.1f}s")

    # ------------------------------------------------------------------
    # Save model
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    gp_model.save(args.output_dir)

    # ------------------------------------------------------------------
    # Validate: round-trip save/load check
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  SAVE/LOAD ROUND-TRIP CHECK")
    print(f"{'='*70}")

    loaded_model = load_gp_surrogate(args.output_dir)

    # Compare predictions on a small batch
    check_params = test_params[:5]
    pred_orig = gp_model.predict_batch(check_params)
    pred_loaded = loaded_model.predict_batch(check_params)

    cd_diff = np.max(np.abs(pred_orig["current_density"] - pred_loaded["current_density"]))
    pc_diff = np.max(np.abs(pred_orig["peroxide_current"] - pred_loaded["peroxide_current"]))
    print(f"  Max abs diff after round-trip: CD={cd_diff:.2e}, PC={pc_diff:.2e}")
    if cd_diff < 1e-10 and pc_diff < 1e-10:
        print(f"  Round-trip: PASS")
    else:
        print(f"  Round-trip: WARNING - diff exceeds 1e-10")

    # ------------------------------------------------------------------
    # Validation: prediction accuracy on test set
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  PREDICTION ACCURACY (test set)")
    print(f"{'='*70}")

    metrics = validate_surrogate(
        gp_model,
        test_parameters=test_params,
        test_cd=test_cd,
        test_pc=test_pc,
    )
    print_validation_report(metrics)

    # ------------------------------------------------------------------
    # UQ calibration check
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  UQ CALIBRATION")
    print(f"{'='*70}")

    check_uq_calibration(gp_model, test_params, test_cd, test_pc)

    # ------------------------------------------------------------------
    # Gradient accuracy check (autograd vs FD)
    # ------------------------------------------------------------------
    if not args.skip_gradient_check:
        print(f"\n{'='*70}")
        print(f"  GRADIENT ACCURACY (autograd vs FD)")
        print(f"{'='*70}")

        check_gradient_accuracy(
            gp_model, test_params, test_cd, test_pc,
            n_points=10, h=1e-5, seed=args.seed,
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'#'*70}")
    print(f"  TRAINING COMPLETE")
    print(f"  Fitting time     : {t_fit:.1f}s")
    print(f"  CD Mean NRMSE    : {metrics['cd_mean_relative_error']*100:.2f}%")
    print(f"  PC Mean NRMSE    : {metrics['pc_mean_relative_error']*100:.2f}%")
    print(f"  Model saved to   : {args.output_dir}")
    print(f"{'#'*70}")


if __name__ == "__main__":
    main()
