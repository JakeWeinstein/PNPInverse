"""Train a PCE surrogate model and run validation + sensitivity analysis.

Usage (from PNPInverse/ directory)::

    python scripts/surrogate/train_pce.py

Loads training_data_merged.npz, splits using the same split_indices.npz
as POD-RBF/NN (for fair comparison), fits PCESurrogateModel, validates,
prints sensitivity report, and saves artefacts.
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

from Surrogate.pce_model import PCEConfig, PCESurrogateModel
from Surrogate.validation import validate_surrogate, print_validation_report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_data_and_split(
    data_path: str,
    split_path: str,
) -> tuple[dict, dict]:
    """Load training data and apply the canonical train/test split.

    Parameters
    ----------
    data_path : str
        Path to training_data_merged.npz.
    split_path : str
        Path to split_indices.npz (contains 'train_idx', 'test_idx').

    Returns
    -------
    (train_data, test_data) dicts with keys:
        parameters, current_density, peroxide_current, phi_applied
    """
    data = np.load(data_path, allow_pickle=True)
    parameters = data["parameters"]
    current_density = data["current_density"]
    peroxide_current = data["peroxide_current"]
    phi_applied = data["phi_applied"]

    split = np.load(split_path, allow_pickle=True)
    train_idx = split["train_idx"]
    test_idx = split["test_idx"]

    train_data = {
        "parameters": parameters[train_idx],
        "current_density": current_density[train_idx],
        "peroxide_current": peroxide_current[train_idx],
        "phi_applied": phi_applied,
    }
    test_data = {
        "parameters": parameters[test_idx],
        "current_density": current_density[test_idx],
        "peroxide_current": peroxide_current[test_idx],
        "phi_applied": phi_applied,
    }
    return train_data, test_data


def _run_degree_convergence_study(
    train_data: dict,
    test_data: dict,
    degrees: tuple = (3, 4, 5, 6, 7, 8),
) -> list[dict]:
    """Fit PCE at multiple degrees and record test-set metrics.

    Parameters
    ----------
    train_data, test_data : dict
        Training and test data.
    degrees : tuple of int
        Degrees to evaluate.

    Returns
    -------
    list of dict with keys: degree, n_terms, cd_rmse, pc_rmse, cd_nrmse, pc_nrmse
    """
    results = []

    print(f"\n{'='*70}")
    print(f"  DEGREE CONVERGENCE STUDY")
    print(f"{'='*70}")

    for deg in degrees:
        config = PCEConfig(
            max_degree=deg,
            cross_validation=False,  # Force this specific degree
        )
        model = PCESurrogateModel(config)
        model.fit(
            parameters=train_data["parameters"],
            current_density=train_data["current_density"],
            peroxide_current=train_data["peroxide_current"],
            phi_applied=train_data["phi_applied"],
            verbose=False,
        )

        metrics = validate_surrogate(
            model,
            test_parameters=test_data["parameters"],
            test_cd=test_data["current_density"],
            test_pc=test_data["peroxide_current"],
        )

        entry = {
            "degree": deg,
            "n_terms": model.n_terms,
            "cd_rmse": metrics["cd_rmse"],
            "pc_rmse": metrics["pc_rmse"],
            "cd_nrmse": metrics["cd_mean_relative_error"],
            "pc_nrmse": metrics["pc_mean_relative_error"],
        }
        results.append(entry)

        print(
            f"  degree={deg:2d}  terms={model.n_terms:4d}  "
            f"CD RMSE={metrics['cd_rmse']:.4e}  PC RMSE={metrics['pc_rmse']:.4e}  "
            f"CD NRMSE={metrics['cd_mean_relative_error']*100:.3f}%  "
            f"PC NRMSE={metrics['pc_mean_relative_error']*100:.3f}%"
        )

    print(f"{'='*70}\n")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PCE surrogate model with Sobol sensitivity analysis",
    )
    parser.add_argument(
        "--training-data",
        type=str,
        default="data/surrogate_models/training_data_merged.npz",
        help="Path to training data .npz file",
    )
    parser.add_argument(
        "--split-indices",
        type=str,
        default="data/surrogate_models/split_indices.npz",
        help="Path to split indices .npz file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/surrogate_models/pce",
        help="Directory to save fitted PCE model",
    )
    parser.add_argument(
        "--sobol-output",
        type=str,
        default="StudyResults/surrogate_fidelity/pce_sobol_indices.json",
        help="Path to save Sobol indices JSON",
    )
    parser.add_argument(
        "--max-degree",
        type=int,
        default=6,
        help="Maximum PCE degree (default: 6)",
    )
    parser.add_argument(
        "--skip-convergence",
        action="store_true",
        help="Skip degree convergence study",
    )
    args = parser.parse_args()

    print(f"\n{'#'*70}")
    print(f"  PCE SURROGATE TRAINING PIPELINE")
    print(f"  Training data : {args.training_data}")
    print(f"  Split indices : {args.split_indices}")
    print(f"  Output dir    : {args.output_dir}")
    print(f"  Max degree    : {args.max_degree}")
    print(f"{'#'*70}\n")

    # Load data
    train_data, test_data = _load_data_and_split(
        args.training_data, args.split_indices
    )
    n_train = train_data["parameters"].shape[0]
    n_test = test_data["parameters"].shape[0]
    n_eta = train_data["phi_applied"].shape[0]
    print(f"  Loaded: {n_train} train, {n_test} test, {n_eta} voltage points\n")

    # Fit PCE model with CV degree selection
    config = PCEConfig(
        max_degree=args.max_degree,
        cross_validation=True,
    )

    model = PCESurrogateModel(config)
    t_start = time.time()
    model.fit(
        parameters=train_data["parameters"],
        current_density=train_data["current_density"],
        peroxide_current=train_data["peroxide_current"],
        phi_applied=train_data["phi_applied"],
        verbose=True,
    )
    t_fit = time.time() - t_start
    print(f"  PCE fit time: {t_fit:.2f}s\n")

    # Validate on test set
    metrics = validate_surrogate(
        model,
        test_parameters=test_data["parameters"],
        test_cd=test_data["current_density"],
        test_pc=test_data["peroxide_current"],
    )
    print_validation_report(metrics)

    # Print sensitivity report
    model.print_sensitivity_report()

    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "pce_model.pkl")
    model.save(model_path)

    # Save Sobol indices
    model.save_sobol_indices(args.sobol_output)

    # Degree convergence study
    if not args.skip_convergence:
        _run_degree_convergence_study(train_data, test_data)

    print(f"\n{'#'*70}")
    print(f"  PCE PIPELINE COMPLETE")
    print(f"  Model: {model_path}")
    print(f"  Sobol: {args.sobol_output}")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
