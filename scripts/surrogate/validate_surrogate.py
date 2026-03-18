"""Detailed validation of a BV surrogate model.

Loads a .pkl surrogate model and .npz test data, computes error metrics,
and optionally generates comparison plots.

Usage (from PNPInverse/ directory)::

    python scripts/surrogate/validate_surrogate.py \\
        --model StudyResults/surrogate/surrogate_model.pkl \\
        --test-data StudyResults/surrogate/training_data.npz \\
        --output-dir StudyResults/surrogate/validation
"""

from __future__ import annotations

import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PNPINVERSE_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _PNPINVERSE_ROOT not in sys.path:
    sys.path.insert(0, _PNPINVERSE_ROOT)

import numpy as np

from Surrogate.io import load_surrogate
from Surrogate.validation import validate_surrogate, print_validation_report


def main():
    parser = argparse.ArgumentParser(description="Validate BV surrogate model")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to surrogate .pkl model")
    parser.add_argument("--test-data", type=str, required=True,
                        help="Path to test data .npz file")
    parser.add_argument("--output-dir", type=str,
                        default="StudyResults/surrogate/validation",
                        help="Output directory for validation results")
    parser.add_argument("--n-samples", type=int, default=0,
                        help="Number of test samples to use (0=all)")
    parser.add_argument("--split-indices", type=str, default=None,
                        help="Path to split_indices.npz. When provided, only "
                        "the test split is used for validation.")
    args = parser.parse_args()

    print(f"\n{'#'*60}")
    print(f"  SURROGATE VALIDATION")
    print(f"{'#'*60}\n")

    model = load_surrogate(args.model)

    data = np.load(args.test_data, allow_pickle=True)
    parameters = data["parameters"]
    cd = data["current_density"]
    pc = data["peroxide_current"]
    phi_applied = data["phi_applied"]

    # Restrict to test split when split indices are provided
    if args.split_indices is not None:
        splits = np.load(args.split_indices, allow_pickle=True)
        test_idx = splits["test_idx"]
        print(f"  Using test split ({len(test_idx)} samples) from {args.split_indices}")
        parameters = parameters[test_idx]
        cd = cd[test_idx]
        pc = pc[test_idx]

    N = parameters.shape[0]
    if args.n_samples > 0 and args.n_samples < N:
        idx = np.random.default_rng(42).choice(N, size=args.n_samples, replace=False)
        parameters = parameters[idx]
        cd = cd[idx]
        pc = pc[idx]
        N = args.n_samples

    print(f"  Test samples: {N}")
    print(f"  Voltage points: {phi_applied.shape[0]}")

    metrics = validate_surrogate(model, parameters, cd, pc)
    print_validation_report(metrics)

    # Save detailed results
    os.makedirs(args.output_dir, exist_ok=True)

    # Per-sample error table
    csv_path = os.path.join(args.output_dir, "per_sample_errors.csv")
    with open(csv_path, "w") as f:
        f.write("sample,k0_1,k0_2,alpha_1,alpha_2,cd_rmse,pc_rmse,cd_nrmse,pc_nrmse\n")
        for i in range(N):
            f.write(f"{i},{parameters[i,0]:.6e},{parameters[i,1]:.6e},"
                    f"{parameters[i,2]:.6f},{parameters[i,3]:.6f},"
                    f"{metrics['cd_per_sample_rmse'][i]:.6e},"
                    f"{metrics['pc_per_sample_rmse'][i]:.6e},"
                    f"{metrics['cd_nrmse_per_sample'][i]*100:.4f},"
                    f"{metrics['pc_nrmse_per_sample'][i]*100:.4f}\n")
    print(f"\n  Per-sample errors saved to: {csv_path}")

    # Worst-case samples
    worst_cd_idx = np.argmax(metrics["cd_nrmse_per_sample"])
    worst_pc_idx = np.argmax(metrics["pc_nrmse_per_sample"])
    print(f"\n  Worst CD sample: #{worst_cd_idx} "
          f"(NRMSE={metrics['cd_nrmse_per_sample'][worst_cd_idx]*100:.2f}%)")
    print(f"    params: k0=[{parameters[worst_cd_idx,0]:.4e}, {parameters[worst_cd_idx,1]:.4e}], "
          f"alpha=[{parameters[worst_cd_idx,2]:.4f}, {parameters[worst_cd_idx,3]:.4f}]")
    print(f"  Worst PC sample: #{worst_pc_idx} "
          f"(NRMSE={metrics['pc_nrmse_per_sample'][worst_pc_idx]*100:.2f}%)")
    print(f"    params: k0=[{parameters[worst_pc_idx,0]:.4e}, {parameters[worst_pc_idx,1]:.4e}], "
          f"alpha=[{parameters[worst_pc_idx,2]:.4f}, {parameters[worst_pc_idx,3]:.4f}]")

    print(f"\n{'#'*60}")
    print(f"  VALIDATION COMPLETE")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
