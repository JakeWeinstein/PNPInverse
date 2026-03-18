"""Validation utilities for the BV surrogate model.

Computes error metrics comparing surrogate predictions to PDE solutions
on held-out test data.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from Surrogate.surrogate_model import BVSurrogateModel


def validate_surrogate(
    surrogate: BVSurrogateModel,
    test_parameters: np.ndarray,
    test_cd: np.ndarray,
    test_pc: np.ndarray,
) -> Dict[str, object]:
    """Validate surrogate accuracy on held-out test data.

    Parameters
    ----------
    surrogate : BVSurrogateModel
        Fitted surrogate model.
    test_parameters : np.ndarray of shape (N_test, 4)
        Test parameter samples [k0_1, k0_2, alpha_1, alpha_2].
    test_cd : np.ndarray of shape (N_test, n_eta)
        True current density I-V curves.
    test_pc : np.ndarray of shape (N_test, n_eta)
        True peroxide current I-V curves.

    Returns
    -------
    dict
        Error statistics including:
        - 'cd_rmse' : float, root mean squared error for current density
        - 'pc_rmse' : float, root mean squared error for peroxide current
        - 'cd_max_abs_error' : float
        - 'pc_max_abs_error' : float
        - 'cd_mean_relative_error' : float (mean over samples of per-sample NRMSE)
        - 'pc_mean_relative_error' : float
        - 'cd_per_sample_rmse' : np.ndarray of shape (N_test,)
        - 'pc_per_sample_rmse' : np.ndarray of shape (N_test,)
        - 'n_test' : int
    """
    N_test = test_parameters.shape[0]

    # Predict batch
    pred = surrogate.predict_batch(test_parameters)
    pred_cd = pred["current_density"]
    pred_pc = pred["peroxide_current"]

    # Per-sample RMSE
    cd_diff = pred_cd - test_cd
    pc_diff = pred_pc - test_pc

    cd_per_sample_rmse = np.sqrt(np.mean(cd_diff ** 2, axis=1))
    pc_per_sample_rmse = np.sqrt(np.mean(pc_diff ** 2, axis=1))

    # Global RMSE
    cd_rmse = float(np.sqrt(np.mean(cd_diff ** 2)))
    pc_rmse = float(np.sqrt(np.mean(pc_diff ** 2)))

    # Max absolute error
    cd_max_abs = float(np.max(np.abs(cd_diff)))
    pc_max_abs = float(np.max(np.abs(pc_diff)))

    # Mean relative error (NRMSE: per-sample RMSE / range of true values)
    # Use a robust denominator: max(ptp, global_ptp * 0.01) to avoid
    # division by near-zero for flat curves (especially PC at certain
    # parameter combinations where peroxide current is negligible).
    cd_global_range = float(np.ptp(test_cd))
    pc_global_range = float(np.ptp(test_pc))
    cd_range_floor = max(cd_global_range * 0.01, 1e-12)
    pc_range_floor = max(pc_global_range * 0.01, 1e-12)

    cd_nrmse_per_sample = np.zeros(N_test, dtype=float)
    pc_nrmse_per_sample = np.zeros(N_test, dtype=float)
    for i in range(N_test):
        cd_range = max(np.ptp(test_cd[i]), cd_range_floor)
        pc_range = max(np.ptp(test_pc[i]), pc_range_floor)
        cd_nrmse_per_sample[i] = cd_per_sample_rmse[i] / cd_range
        pc_nrmse_per_sample[i] = pc_per_sample_rmse[i] / pc_range

    cd_mean_rel = float(np.mean(cd_nrmse_per_sample))
    pc_mean_rel = float(np.mean(pc_nrmse_per_sample))

    return {
        "cd_rmse": cd_rmse,
        "pc_rmse": pc_rmse,
        "cd_max_abs_error": cd_max_abs,
        "pc_max_abs_error": pc_max_abs,
        "cd_mean_relative_error": cd_mean_rel,
        "pc_mean_relative_error": pc_mean_rel,
        "cd_per_sample_rmse": cd_per_sample_rmse,
        "pc_per_sample_rmse": pc_per_sample_rmse,
        "cd_nrmse_per_sample": cd_nrmse_per_sample,
        "pc_nrmse_per_sample": pc_nrmse_per_sample,
        "n_test": N_test,
    }


def print_validation_report(metrics: Dict[str, object]) -> None:
    """Print a formatted validation report."""
    print(f"\n{'='*60}")
    print(f"  SURROGATE VALIDATION REPORT")
    print(f"  Test samples: {metrics['n_test']}")
    print(f"{'='*60}")
    print(f"  Current Density:")
    print(f"    RMSE:               {metrics['cd_rmse']:.6e}")
    print(f"    Max absolute error: {metrics['cd_max_abs_error']:.6e}")
    print(f"    Mean NRMSE:         {metrics['cd_mean_relative_error']*100:.2f}%")
    print(f"  Peroxide Current:")
    print(f"    RMSE:               {metrics['pc_rmse']:.6e}")
    print(f"    Max absolute error: {metrics['pc_max_abs_error']:.6e}")
    print(f"    Mean NRMSE:         {metrics['pc_mean_relative_error']*100:.2f}%")
    print(f"{'='*60}")
