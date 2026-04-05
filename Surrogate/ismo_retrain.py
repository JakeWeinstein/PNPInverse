"""ISMO surrogate retraining pipeline.

Provides the retraining module for ISMO (Iterative Surrogate Model
Optimization) iterations.  When new PDE training data arrives, each
surrogate model type is retrained on the merged dataset.

Key functions:
    merge_training_data()       -- merge old + new training data
    retrain_surrogate()         -- dispatch to type-specific retraining
    retrain_surrogate_full()    -- batteries-included pipeline

Supports warm-start for NN ensembles (analytical weight correction +
fine-tuning) and from-scratch retraining for fast models (RBF, POD-RBF,
PCE).  All functions return new model objects; originals are never mutated.
"""

from __future__ import annotations

import copy
import logging
import os
import pickle
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ISMORetrainConfig:
    """Configuration for ISMO surrogate retraining."""

    # NN ensemble warm-start settings
    nn_retrain_epochs: int = 100
    nn_retrain_lr: float = 1e-4
    nn_retrain_patience: int = 50
    nn_retrain_weight_decay: float = 1e-4
    nn_from_scratch_fallback: bool = True
    nn_from_scratch_epochs: int = 3000
    nn_from_scratch_patience: int = 500

    # GP warm-start settings
    gp_retrain_iters: int = 100
    gp_retrain_lr: float = 0.05

    # Quality check thresholds
    max_degradation_ratio: float = 1.10
    quality_metric: str = "cd_mean_relative_error"

    # Data merging (log-space normalized duplicate detection)
    duplicate_param_tol: float = 1e-6

    # Output paths
    output_base_dir: str = "data/surrogate_models"


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MergedData:
    """Immutable result of merging old and new training data."""

    parameters: np.ndarray
    current_density: np.ndarray
    peroxide_current: np.ndarray
    phi_applied: np.ndarray
    n_old: int
    n_new_valid: int
    n_duplicates_dropped: int
    n_nan_dropped: int
    new_indices: np.ndarray


@dataclass(frozen=True)
class ISMORetrainResult:
    """Result of ISMO surrogate retraining."""

    surrogate: Any
    merged_data: MergedData
    updated_train_idx: np.ndarray
    updated_test_idx: np.ndarray
    quality_passed: bool
    old_error: float
    new_error: float
    error_ratio: float
    quality_metric: str
    retrain_method: str
    iteration: int
    save_path: str


# ---------------------------------------------------------------------------
# Data merging
# ---------------------------------------------------------------------------


def merge_training_data(
    existing_data: Dict[str, np.ndarray],
    new_data: Dict[str, np.ndarray],
    config: ISMORetrainConfig | None = None,
) -> MergedData:
    """Merge existing and new training data, dropping duplicates and NaN rows.

    Duplicate detection operates in normalized log-space: k0 columns are
    transformed to log10 before per-column z-score normalization using the
    existing data's statistics.  L2 distance is then computed between
    each new row and all existing rows.

    Parameters
    ----------
    existing_data : dict
        Keys: parameters (N_old, 4), current_density (N_old, n_eta),
        peroxide_current (N_old, n_eta), phi_applied (n_eta,).
    new_data : dict
        Same structure with N_new samples.
    config : ISMORetrainConfig or None
        Configuration (uses defaults if None).

    Returns
    -------
    MergedData
        Immutable result with merged arrays and diagnostics.

    Raises
    ------
    ValueError
        If phi_applied grids do not match.
    """
    if config is None:
        config = ISMORetrainConfig()

    old_params = existing_data["parameters"]
    old_cd = existing_data["current_density"]
    old_pc = existing_data["peroxide_current"]
    old_phi = existing_data["phi_applied"]

    new_params = new_data["parameters"].copy()
    new_cd = new_data["current_density"].copy()
    new_pc = new_data["peroxide_current"].copy()
    new_phi = new_data["phi_applied"]

    # Validate phi_applied match
    if not np.allclose(old_phi, new_phi):
        raise ValueError(
            f"phi_applied grids do not match: "
            f"old shape {old_phi.shape}, new shape {new_phi.shape}"
        )

    n_old = old_params.shape[0]
    n_new_orig = new_params.shape[0]

    # --- Drop NaN rows ---
    nan_mask = (
        np.any(np.isnan(new_cd), axis=1)
        | np.any(np.isnan(new_pc), axis=1)
        | np.any(np.isnan(new_params), axis=1)
    )
    n_nan_dropped = int(nan_mask.sum())
    if n_nan_dropped > 0:
        logger.info("Dropped %d new rows with NaN values", n_nan_dropped)
        keep = ~nan_mask
        new_params = new_params[keep]
        new_cd = new_cd[keep]
        new_pc = new_pc[keep]

    # --- Drop duplicates (normalized log-space) ---
    n_duplicates_dropped = 0
    if new_params.shape[0] > 0 and old_params.shape[0] > 0:
        # Transform k0 columns to log10
        old_log = old_params.copy().astype(np.float64)
        old_log[:, 0] = np.log10(np.maximum(old_log[:, 0], 1e-30))
        old_log[:, 1] = np.log10(np.maximum(old_log[:, 1], 1e-30))

        new_log = new_params.copy().astype(np.float64)
        new_log[:, 0] = np.log10(np.maximum(new_log[:, 0], 1e-30))
        new_log[:, 1] = np.log10(np.maximum(new_log[:, 1], 1e-30))

        # Z-score normalize using existing data statistics
        mean = old_log.mean(axis=0)
        std = np.maximum(old_log.std(axis=0), 1e-15)
        old_norm = (old_log - mean) / std
        new_norm = (new_log - mean) / std

        # Pairwise L2 distance: new vs existing
        # Shape: (n_new, n_old)
        diffs = new_norm[:, np.newaxis, :] - old_norm[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        min_dists = dists.min(axis=1)

        dup_mask = min_dists < config.duplicate_param_tol
        n_duplicates_dropped = int(dup_mask.sum())
        if n_duplicates_dropped > 0:
            logger.info(
                "Dropped %d duplicate new rows (tol=%.2e in normalized log-space)",
                n_duplicates_dropped,
                config.duplicate_param_tol,
            )
            keep = ~dup_mask
            new_params = new_params[keep]
            new_cd = new_cd[keep]
            new_pc = new_pc[keep]

    n_new_valid = new_params.shape[0]

    # --- Concatenate ---
    merged_params = np.concatenate([old_params, new_params], axis=0)
    merged_cd = np.concatenate([old_cd, new_cd], axis=0)
    merged_pc = np.concatenate([old_pc, new_pc], axis=0)
    new_indices = np.arange(n_old, n_old + n_new_valid)

    logger.info(
        "Merged data: %d old + %d new valid = %d total "
        "(%d duplicates dropped, %d NaN dropped)",
        n_old,
        n_new_valid,
        n_old + n_new_valid,
        n_duplicates_dropped,
        n_nan_dropped,
    )

    return MergedData(
        parameters=merged_params,
        current_density=merged_cd,
        peroxide_current=merged_pc,
        phi_applied=old_phi.copy(),
        n_old=n_old,
        n_new_valid=n_new_valid,
        n_duplicates_dropped=n_duplicates_dropped,
        n_nan_dropped=n_nan_dropped,
        new_indices=new_indices,
    )


def update_split_indices(
    existing_train_idx: np.ndarray,
    existing_test_idx: np.ndarray,
    n_old: int,
    n_new_valid: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Update train/test split indices after merging new data.

    New data indices are added to the training set (ISMO acquired them
    specifically to improve the surrogate).

    Parameters
    ----------
    existing_train_idx : np.ndarray
        Current training indices.
    existing_test_idx : np.ndarray
        Current test indices.
    n_old : int
        Number of samples in the existing dataset.
    n_new_valid : int
        Number of valid new samples added.

    Returns
    -------
    (new_train_idx, new_test_idx)
        Updated index arrays.

    Raises
    ------
    ValueError
        If train and test indices overlap.
    """
    new_data_indices = np.arange(n_old, n_old + n_new_valid)
    new_train_idx = np.concatenate([existing_train_idx, new_data_indices])
    new_test_idx = existing_test_idx.copy()

    # Validate no overlap
    overlap = np.intersect1d(new_train_idx, new_test_idx)
    if len(overlap) > 0:
        raise ValueError(
            f"Train/test overlap detected: {len(overlap)} indices in common"
        )

    return new_train_idx, new_test_idx


def save_merged_data(
    merged_data: MergedData,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    output_base_dir: str,
    iteration: int,
) -> str:
    """Save merged training data and split indices to disk.

    Parameters
    ----------
    merged_data : MergedData
        Merged training dataset.
    train_idx, test_idx : np.ndarray
        Updated split indices.
    output_base_dir : str
        Base output directory.
    iteration : int
        ISMO iteration number.

    Returns
    -------
    str
        Path to the saved directory.
    """
    out_dir = os.path.join(output_base_dir, f"ismo_iter_{iteration}")
    os.makedirs(out_dir, exist_ok=True)

    np.savez(
        os.path.join(out_dir, "training_data_merged.npz"),
        parameters=merged_data.parameters,
        current_density=merged_data.current_density,
        peroxide_current=merged_data.peroxide_current,
        phi_applied=merged_data.phi_applied,
    )

    np.savez(
        os.path.join(out_dir, "split_indices.npz"),
        train_idx=train_idx,
        test_idx=test_idx,
    )

    logger.info("Saved merged data to %s", out_dir)
    return out_dir


# ---------------------------------------------------------------------------
# Quality check helper
# ---------------------------------------------------------------------------


def _check_retrain_quality(
    old_model: Any,
    new_model: Any,
    test_params: np.ndarray,
    test_cd: np.ndarray,
    test_pc: np.ndarray,
    config: ISMORetrainConfig,
    model_name: str,
) -> Tuple[bool, Dict[str, Any]]:
    """Compare before/after retraining error.

    Parameters
    ----------
    old_model, new_model : surrogate
        Original and retrained models.
    test_params : np.ndarray (N_test, 4)
        Test parameters.
    test_cd, test_pc : np.ndarray (N_test, n_eta)
        Test output data.
    config : ISMORetrainConfig
        Retraining configuration.
    model_name : str
        Name for logging.

    Returns
    -------
    (passed, metrics)
        Whether the quality check passed, and a dict of diagnostics.
    """
    from Surrogate.validation import validate_surrogate

    old_metrics = validate_surrogate(old_model, test_params, test_cd, test_pc)
    new_metrics = validate_surrogate(new_model, test_params, test_cd, test_pc)

    old_error = float(old_metrics[config.quality_metric])
    new_error = float(new_metrics[config.quality_metric])

    ratio = new_error / max(old_error, 1e-30)
    passed = ratio <= config.max_degradation_ratio

    logger.info(
        "%s quality check: old=%.6e, new=%.6e, ratio=%.4f %s",
        model_name,
        old_error,
        new_error,
        ratio,
        "PASS" if passed else "FAIL",
    )

    return passed, {
        "old_error": old_error,
        "new_error": new_error,
        "ratio": ratio,
        "quality_metric": config.quality_metric,
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# NN ensemble weight correction
# ---------------------------------------------------------------------------


def _correct_weights_for_normalizer_shift(
    state_dict: Dict[str, Any],
    old_input_norm: Any,
    new_input_norm: Any,
    old_output_norm: Any,
    new_output_norm: Any,
) -> Dict[str, Any]:
    """Analytically correct first/last layer weights for normalizer change.

    When normalizers change from old to new, the first and last linear
    layers must be corrected so the network's physical input-output
    mapping is preserved.

    Parameters
    ----------
    state_dict : dict
        PyTorch state dict from the old model.
    old_input_norm, new_input_norm : ZScoreNormalizer
        Old and new input normalizers.
    old_output_norm, new_output_norm : ZScoreNormalizer
        Old and new output normalizers.

    Returns
    -------
    dict
        Corrected state dict (new dict, no mutation of input).
    """
    import torch

    corrected = {k: v.clone() for k, v in state_dict.items()}

    # --- Input layer correction ---
    # x_norm_old = (x - mu_old) / sigma_old
    # x_norm_new = (x - mu_new) / sigma_new
    # x = x_norm_new * sigma_new + mu_new
    # x_norm_old = x_norm_new * (sigma_new / sigma_old) + (mu_new - mu_old) / sigma_old
    # So: W_new = W_old * diag(sigma_new / sigma_old)
    #     b_new = b_old + W_old @ ((mu_new - mu_old) / sigma_old)
    sigma_in_old = torch.tensor(old_input_norm.std, dtype=torch.float32)
    sigma_in_new = torch.tensor(new_input_norm.std, dtype=torch.float32)
    mu_in_old = torch.tensor(old_input_norm.mean, dtype=torch.float32)
    mu_in_new = torch.tensor(new_input_norm.mean, dtype=torch.float32)

    ratio_in = sigma_in_new / sigma_in_old
    offset_in = (mu_in_new - mu_in_old) / sigma_in_old

    W_in = corrected["input_layer.0.weight"]   # (hidden, 4)
    b_in = corrected["input_layer.0.bias"]     # (hidden,)
    corrected["input_layer.0.weight"] = W_in * ratio_in.unsqueeze(0)
    corrected["input_layer.0.bias"] = b_in + (W_in @ offset_in)

    # --- Output layer correction ---
    # y = y_norm * sigma_out + mu_out
    # y_norm_new = y_norm_old * (sigma_out_old / sigma_out_new)
    #            + (mu_out_old - mu_out_new) / sigma_out_new
    # So: W_new = W_old * (sigma_out_old / sigma_out_new).unsqueeze(1)
    #     b_new = b_old * (sigma_out_old / sigma_out_new)
    #           + (mu_out_old - mu_out_new) / sigma_out_new
    sigma_out_old = torch.tensor(old_output_norm.std, dtype=torch.float32)
    sigma_out_new = torch.tensor(new_output_norm.std, dtype=torch.float32)
    mu_out_old = torch.tensor(old_output_norm.mean, dtype=torch.float32)
    mu_out_new = torch.tensor(new_output_norm.mean, dtype=torch.float32)

    ratio_out = sigma_out_old / sigma_out_new
    offset_out = (mu_out_old - mu_out_new) / sigma_out_new

    W_out = corrected["output_layer.3.weight"]  # (n_out, hidden//2)
    b_out = corrected["output_layer.3.bias"]    # (n_out,)
    corrected["output_layer.3.weight"] = W_out * ratio_out.unsqueeze(1)
    corrected["output_layer.3.bias"] = b_out * ratio_out + offset_out

    # Diagnostic: log normalizer drift magnitude
    input_drift = float(torch.max(torch.abs(offset_in)).item())
    output_drift = float(
        torch.max(torch.abs((mu_out_old - mu_out_new) / sigma_out_old)).item()
    )
    logger.info(
        "Normalizer drift: input max=%.4e, output max=%.4e",
        input_drift,
        output_drift,
    )

    return corrected


# ---------------------------------------------------------------------------
# NN ensemble retraining
# ---------------------------------------------------------------------------


def retrain_nn_ensemble(
    ensemble: Any,
    merged_data: MergedData,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    config: ISMORetrainConfig,
    iteration: int,
    device: str = "cpu",
) -> Any:
    """Retrain an NN ensemble with warm-start and quality checks.

    For each member: analytically correct weights for normalizer shift,
    fine-tune with reduced LR, validate quality, and fall back to
    from-scratch if degraded.

    Parameters
    ----------
    ensemble : EnsembleMeanWrapper
        Existing ensemble with .models list.
    merged_data : MergedData
        Merged training dataset.
    train_idx, test_idx : np.ndarray
        Train/test split indices.
    config : ISMORetrainConfig
        Retraining configuration.
    iteration : int
        ISMO iteration number.
    device : str
        PyTorch device.

    Returns
    -------
    EnsembleMeanWrapper
        New ensemble with retrained members.

    Raises
    ------
    RuntimeError
        If all ensemble members fail to retrain.
    """
    from Surrogate.ensemble import EnsembleMeanWrapper
    from Surrogate.nn_model import NNSurrogateModel, ZScoreNormalizer
    from Surrogate.validation import validate_surrogate

    phi_applied = merged_data.phi_applied

    # Build train/val arrays.
    # Split train_idx into train (85%) and val (15%) subsets so that
    # test_idx remains truly held out for post-retrain quality checks.
    rng = np.random.default_rng(seed=42 + iteration)
    shuffled_train = rng.permutation(train_idx)
    n_val = max(1, int(0.15 * len(shuffled_train)))
    val_split_idx = shuffled_train[:n_val]
    actual_train_idx = shuffled_train[n_val:]

    train_params = merged_data.parameters[actual_train_idx]
    train_cd = merged_data.current_density[actual_train_idx]
    train_pc = merged_data.peroxide_current[actual_train_idx]
    val_params = merged_data.parameters[val_split_idx]
    val_cd = merged_data.current_density[val_split_idx]
    val_pc = merged_data.peroxide_current[val_split_idx]

    # Held-out test arrays for post-retrain quality evaluation
    test_params = merged_data.parameters[test_idx]
    test_cd = merged_data.current_density[test_idx]
    test_pc = merged_data.peroxide_current[test_idx]

    # Compute new normalizers on merged training data
    X_log = train_params.copy().astype(np.float64)
    X_log[:, 0] = np.log10(np.maximum(X_log[:, 0], 1e-30))
    X_log[:, 1] = np.log10(np.maximum(X_log[:, 1], 1e-30))
    new_input_norm = ZScoreNormalizer.from_data(X_log)

    Y = np.concatenate([train_cd, train_pc], axis=1)
    new_output_norm = ZScoreNormalizer.from_data(Y)

    retrained_members: List[Any] = []
    n_warm_started = 0
    n_from_scratch = 0
    n_failed = 0

    for i, member in enumerate(ensemble.models):
        logger.info("Retraining ensemble member %d/%d", i + 1, len(ensemble.models))
        try:
            # Pre-retrain quality on held-out test set
            old_metrics = validate_surrogate(member, test_params, test_cd, test_pc)
            old_error = float(old_metrics[config.quality_metric])

            # Correct weights for normalizer shift
            import torch

            old_state = {
                k: v.cpu().clone()
                for k, v in member._model.state_dict().items()
            }
            corrected_state = _correct_weights_for_normalizer_shift(
                old_state,
                member._input_normalizer,
                new_input_norm,
                member._output_normalizer,
                new_output_norm,
            )

            # Fine-tune via fit() with warm_start_state_dict
            new_member = NNSurrogateModel(
                hidden=member._hidden,
                n_blocks=member._n_blocks,
                seed=member._seed,
                device=device,
            )
            new_member.fit(
                train_params,
                train_cd,
                train_pc,
                phi_applied,
                epochs=config.nn_retrain_epochs,
                lr=config.nn_retrain_lr,
                weight_decay=config.nn_retrain_weight_decay,
                patience=config.nn_retrain_patience,
                val_parameters=val_params,
                val_cd=val_cd,
                val_pc=val_pc,
                warm_start_state_dict=corrected_state,
                verbose=True,
            )

            # Post-retrain quality on held-out test set
            new_metrics = validate_surrogate(new_member, test_params, test_cd, test_pc)
            new_error = float(new_metrics[config.quality_metric])
            ratio = new_error / max(old_error, 1e-30)

            method = "warm_start"

            # Quality check
            if ratio > config.max_degradation_ratio:
                logger.warning(
                    "Warm-start degraded member %d: %.6e -> %.6e (ratio=%.4f)",
                    i, old_error, new_error, ratio,
                )
                if config.nn_from_scratch_fallback:
                    logger.info(
                        "Falling back to from-scratch training for member %d", i,
                    )
                    new_member = NNSurrogateModel(
                        hidden=member._hidden,
                        n_blocks=member._n_blocks,
                        seed=member._seed,
                        device=device,
                    )
                    new_member.fit(
                        train_params,
                        train_cd,
                        train_pc,
                        phi_applied,
                        epochs=config.nn_from_scratch_epochs,
                        lr=1e-3,
                        patience=config.nn_from_scratch_patience,
                        val_parameters=val_params,
                        val_cd=val_cd,
                        val_pc=val_pc,
                        verbose=True,
                    )
                    method = "from_scratch_fallback"
                    n_from_scratch += 1
                else:
                    n_warm_started += 1
            else:
                n_warm_started += 1

            # Save retrained member
            save_dir = os.path.join(
                config.output_base_dir,
                "nn_ensemble",
                f"ismo_iter_{iteration}",
                f"member_{i}",
                "saved_model",
            )
            new_member.save(save_dir)
            logger.info("Member %d retrained (%s), saved to %s", i, method, save_dir)

            retrained_members.append(new_member)

        except Exception:
            logger.error(
                "Failed to retrain member %d, keeping original:\n%s",
                i, traceback.format_exc(),
            )
            retrained_members.append(member)
            n_failed += 1

    if n_failed == len(ensemble.models):
        raise RuntimeError(
            "All ensemble members failed to retrain. "
            "Check logs for details."
        )

    if n_failed > 0:
        logger.warning(
            "Ensemble retrained with %d failures (kept originals)", n_failed,
        )

    logger.info(
        "Ensemble retrained: %d warm-start, %d from-scratch, %d failed (kept original)",
        n_warm_started, n_from_scratch, n_failed,
    )

    return EnsembleMeanWrapper(retrained_members)


def _retrain_single_nn(
    model: Any,
    merged_data: MergedData,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    config: ISMORetrainConfig,
    iteration: int,
    device: str = "cpu",
) -> Any:
    """Retrain a single NNSurrogateModel with warm-start.

    Uses the same logic as ensemble member retraining.

    Parameters
    ----------
    model : NNSurrogateModel
        Existing model.
    merged_data : MergedData
        Merged training dataset.
    train_idx, test_idx : np.ndarray
        Train/test split indices.
    config : ISMORetrainConfig
        Retraining configuration.
    iteration : int
        ISMO iteration number.
    device : str
        PyTorch device.

    Returns
    -------
    NNSurrogateModel
        Retrained model.
    """
    from Surrogate.ensemble import EnsembleMeanWrapper

    # Wrap in a 1-member ensemble, retrain, extract the single member
    tmp_ensemble = EnsembleMeanWrapper([model])
    retrained_ensemble = retrain_nn_ensemble(
        tmp_ensemble, merged_data, train_idx, test_idx, config, iteration, device,
    )
    return retrained_ensemble.models[0]


# ---------------------------------------------------------------------------
# POD-RBF retraining
# ---------------------------------------------------------------------------


def retrain_pod_rbf(
    model: Any,
    merged_data: MergedData,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    config: ISMORetrainConfig,
    iteration: int,
) -> Any:
    """Retrain a PODRBFSurrogateModel from scratch on merged data.

    Parameters
    ----------
    model : PODRBFSurrogateModel
        Existing model (used to extract config).
    merged_data : MergedData
        Merged training dataset.
    train_idx, test_idx : np.ndarray
        Train/test split indices.
    config : ISMORetrainConfig
        Retraining configuration.
    iteration : int
        ISMO iteration number.

    Returns
    -------
    PODRBFSurrogateModel
        New retrained model.
    """
    from Surrogate.pod_rbf_model import PODRBFSurrogateModel

    train_params = merged_data.parameters[train_idx]
    train_cd = merged_data.current_density[train_idx]
    train_pc = merged_data.peroxide_current[train_idx]

    new_model = PODRBFSurrogateModel(config=model.config)
    new_model.fit(train_params, train_cd, train_pc, merged_data.phi_applied)

    # Quality check
    test_params = merged_data.parameters[test_idx]
    test_cd = merged_data.current_density[test_idx]
    test_pc = merged_data.peroxide_current[test_idx]
    passed, qmetrics = _check_retrain_quality(
        model, new_model, test_params, test_cd, test_pc, config, "POD-RBF",
    )

    # Save
    save_path = os.path.join(
        config.output_base_dir, f"ismo_iter_{iteration}", "model_pod_rbf.pkl",
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(new_model, f)
    logger.info("POD-RBF saved to %s", save_path)

    return new_model


# ---------------------------------------------------------------------------
# RBF baseline retraining
# ---------------------------------------------------------------------------


def retrain_rbf_baseline(
    model: Any,
    merged_data: MergedData,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    config: ISMORetrainConfig,
    iteration: int,
) -> Any:
    """Retrain a BVSurrogateModel (RBF baseline) from scratch.

    Parameters
    ----------
    model : BVSurrogateModel
        Existing model (used to extract config).
    merged_data : MergedData
        Merged training dataset.
    train_idx, test_idx : np.ndarray
        Train/test split indices.
    config : ISMORetrainConfig
        Retraining configuration.
    iteration : int
        ISMO iteration number.

    Returns
    -------
    BVSurrogateModel
        New retrained model.
    """
    from Surrogate.surrogate_model import BVSurrogateModel

    train_params = merged_data.parameters[train_idx]
    train_cd = merged_data.current_density[train_idx]
    train_pc = merged_data.peroxide_current[train_idx]

    new_model = BVSurrogateModel(config=model.config)
    new_model.fit(train_params, train_cd, train_pc, merged_data.phi_applied)

    # Quality check
    test_params = merged_data.parameters[test_idx]
    test_cd = merged_data.current_density[test_idx]
    test_pc = merged_data.peroxide_current[test_idx]
    passed, qmetrics = _check_retrain_quality(
        model, new_model, test_params, test_cd, test_pc, config, "RBF-baseline",
    )

    # Save
    save_path = os.path.join(
        config.output_base_dir, f"ismo_iter_{iteration}", "model_rbf_baseline.pkl",
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(new_model, f)
    logger.info("RBF baseline saved to %s", save_path)

    return new_model


# ---------------------------------------------------------------------------
# PCE retraining
# ---------------------------------------------------------------------------


def retrain_pce(
    model: Any,
    merged_data: MergedData,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    config: ISMORetrainConfig,
    iteration: int,
) -> Any:
    """Retrain a PCESurrogateModel from scratch.

    Parameters
    ----------
    model : PCESurrogateModel
        Existing model (used to extract config).
    merged_data : MergedData
        Merged training dataset.
    train_idx, test_idx : np.ndarray
        Train/test split indices.
    config : ISMORetrainConfig
        Retraining configuration.
    iteration : int
        ISMO iteration number.

    Returns
    -------
    PCESurrogateModel
        New retrained model.
    """
    from Surrogate.pce_model import PCESurrogateModel

    train_params = merged_data.parameters[train_idx]
    train_cd = merged_data.current_density[train_idx]
    train_pc = merged_data.peroxide_current[train_idx]

    new_model = PCESurrogateModel(config=model.config)
    new_model.fit(train_params, train_cd, train_pc, merged_data.phi_applied)

    # Quality check
    test_params = merged_data.parameters[test_idx]
    test_cd = merged_data.current_density[test_idx]
    test_pc = merged_data.peroxide_current[test_idx]
    passed, qmetrics = _check_retrain_quality(
        model, new_model, test_params, test_cd, test_pc, config, "PCE",
    )

    # Save
    save_path = os.path.join(
        config.output_base_dir, f"ismo_iter_{iteration}", "pce_model.pkl",
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(new_model, f)
    logger.info("PCE saved to %s", save_path)

    return new_model


# ---------------------------------------------------------------------------
# GP retraining
# ---------------------------------------------------------------------------


def retrain_gp(
    model: Any,
    merged_data: MergedData,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    config: ISMORetrainConfig,
    iteration: int,
    device: str = "cpu",
) -> Any:
    """Retrain a GPSurrogateModel with reduced iterations.

    Parameters
    ----------
    model : GPSurrogateModel
        Existing model.
    merged_data : MergedData
        Merged training dataset.
    train_idx, test_idx : np.ndarray
        Train/test split indices.
    config : ISMORetrainConfig
        Retraining configuration.
    iteration : int
        ISMO iteration number.
    device : str
        PyTorch device.

    Returns
    -------
    GPSurrogateModel
        New retrained model.
    """
    from Surrogate.gp_model import GPSurrogateModel

    train_params = merged_data.parameters[train_idx]
    train_cd = merged_data.current_density[train_idx]
    train_pc = merged_data.peroxide_current[train_idx]

    n_train = train_params.shape[0]
    if n_train > 1500:
        logger.warning(
            "GP training data size %d may cause slow fitting. "
            "Consider switching to SVGP.",
            n_train,
        )

    new_model = GPSurrogateModel(device=device)
    new_model.fit(
        train_params,
        train_cd,
        train_pc,
        merged_data.phi_applied,
        n_iters=config.gp_retrain_iters,
        lr=config.gp_retrain_lr,
    )

    # Quality check
    test_params = merged_data.parameters[test_idx]
    test_cd = merged_data.current_density[test_idx]
    test_pc = merged_data.peroxide_current[test_idx]
    passed, qmetrics = _check_retrain_quality(
        model, new_model, test_params, test_cd, test_pc, config, "GP",
    )

    # Save
    save_dir = os.path.join(
        config.output_base_dir, f"ismo_iter_{iteration}", "gp",
    )
    new_model.save(save_dir)
    logger.info("GP saved to %s", save_dir)

    return new_model


# ---------------------------------------------------------------------------
# Unified dispatch
# ---------------------------------------------------------------------------


def retrain_surrogate(
    surrogate: Any,
    new_data: Dict[str, np.ndarray],
    existing_data: Dict[str, np.ndarray],
    config: ISMORetrainConfig,
    iteration: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    device: str = "cpu",
) -> Any:
    """Retrain a surrogate model on merged data, dispatching by type.

    Parameters
    ----------
    surrogate : surrogate model
        The existing fitted surrogate.
    new_data : dict
        New training data with keys: parameters, current_density,
        peroxide_current, phi_applied.
    existing_data : dict
        Existing training data (same keys).
    config : ISMORetrainConfig
        Retraining configuration.
    iteration : int
        ISMO iteration number.
    train_idx, test_idx : np.ndarray
        Current train/test split indices.
    device : str
        PyTorch device (for NN/GP models).

    Returns
    -------
    retrained surrogate model

    Raises
    ------
    TypeError
        If the surrogate type is not recognized.
    """
    # Merge data
    merged = merge_training_data(existing_data, new_data, config)
    updated_train_idx, updated_test_idx = update_split_indices(
        train_idx, test_idx, merged.n_old, merged.n_new_valid,
    )

    logger.info(
        "ISMO iter %d: merging %d new samples with %d existing -> %d total "
        "(%d duplicates dropped, %d NaN dropped)",
        iteration,
        new_data["parameters"].shape[0],
        existing_data["parameters"].shape[0],
        merged.parameters.shape[0],
        merged.n_duplicates_dropped,
        merged.n_nan_dropped,
    )

    # Type dispatch
    from Surrogate.ensemble import EnsembleMeanWrapper
    from Surrogate.nn_model import NNSurrogateModel
    from Surrogate.surrogate_model import BVSurrogateModel
    from Surrogate.pod_rbf_model import PODRBFSurrogateModel

    if isinstance(surrogate, EnsembleMeanWrapper):
        return retrain_nn_ensemble(
            surrogate, merged, updated_train_idx, updated_test_idx,
            config, iteration, device,
        )
    elif isinstance(surrogate, NNSurrogateModel):
        return _retrain_single_nn(
            surrogate, merged, updated_train_idx, updated_test_idx,
            config, iteration, device,
        )
    elif isinstance(surrogate, PODRBFSurrogateModel):
        return retrain_pod_rbf(
            surrogate, merged, updated_train_idx, updated_test_idx,
            config, iteration,
        )
    elif isinstance(surrogate, BVSurrogateModel):
        return retrain_rbf_baseline(
            surrogate, merged, updated_train_idx, updated_test_idx,
            config, iteration,
        )

    # Conditional imports for optional model types
    try:
        from Surrogate.gp_model import GPSurrogateModel

        if isinstance(surrogate, GPSurrogateModel):
            return retrain_gp(
                surrogate, merged, updated_train_idx, updated_test_idx,
                config, iteration, device,
            )
    except ImportError:
        pass

    try:
        from Surrogate.pce_model import PCESurrogateModel

        if isinstance(surrogate, PCESurrogateModel):
            return retrain_pce(
                surrogate, merged, updated_train_idx, updated_test_idx,
                config, iteration,
            )
    except ImportError:
        pass

    raise TypeError(f"Unknown surrogate type: {type(surrogate).__name__}")


def retrain_surrogate_full(
    surrogate: Any,
    new_data: Dict[str, np.ndarray],
    existing_data: Dict[str, np.ndarray],
    config: ISMORetrainConfig,
    iteration: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    device: str = "cpu",
) -> ISMORetrainResult:
    """Full ISMO retraining pipeline returning a complete result.

    This is the batteries-included API that merges data, retrains the
    surrogate, saves results, and returns full metadata.

    Parameters
    ----------
    surrogate : surrogate model
        The existing fitted surrogate.
    new_data : dict
        New training data.
    existing_data : dict
        Existing training data.
    config : ISMORetrainConfig
        Retraining configuration.
    iteration : int
        ISMO iteration number.
    train_idx, test_idx : np.ndarray
        Current train/test split indices.
    device : str
        PyTorch device.

    Returns
    -------
    ISMORetrainResult
        Complete retraining result with metadata.
    """
    from Surrogate.validation import validate_surrogate

    # Merge data
    merged = merge_training_data(existing_data, new_data, config)
    updated_train_idx, updated_test_idx = update_split_indices(
        train_idx, test_idx, merged.n_old, merged.n_new_valid,
    )

    # Pre-retrain error
    test_params = merged.parameters[updated_test_idx]
    test_cd = merged.current_density[updated_test_idx]
    test_pc = merged.peroxide_current[updated_test_idx]
    old_metrics = validate_surrogate(surrogate, test_params, test_cd, test_pc)
    old_error = float(old_metrics[config.quality_metric])

    # Dispatch retraining (use the type-specific functions directly)
    from Surrogate.ensemble import EnsembleMeanWrapper
    from Surrogate.nn_model import NNSurrogateModel
    from Surrogate.surrogate_model import BVSurrogateModel
    from Surrogate.pod_rbf_model import PODRBFSurrogateModel

    retrain_method = "from_scratch"
    if isinstance(surrogate, (EnsembleMeanWrapper, NNSurrogateModel)):
        retrain_method = "warm_start"

    if isinstance(surrogate, EnsembleMeanWrapper):
        new_surrogate = retrain_nn_ensemble(
            surrogate, merged, updated_train_idx, updated_test_idx,
            config, iteration, device,
        )
    elif isinstance(surrogate, NNSurrogateModel):
        new_surrogate = _retrain_single_nn(
            surrogate, merged, updated_train_idx, updated_test_idx,
            config, iteration, device,
        )
    elif isinstance(surrogate, PODRBFSurrogateModel):
        new_surrogate = retrain_pod_rbf(
            surrogate, merged, updated_train_idx, updated_test_idx,
            config, iteration,
        )
    elif isinstance(surrogate, BVSurrogateModel):
        new_surrogate = retrain_rbf_baseline(
            surrogate, merged, updated_train_idx, updated_test_idx,
            config, iteration,
        )
    else:
        try:
            from Surrogate.gp_model import GPSurrogateModel

            if isinstance(surrogate, GPSurrogateModel):
                new_surrogate = retrain_gp(
                    surrogate, merged, updated_train_idx, updated_test_idx,
                    config, iteration, device,
                )
            else:
                raise TypeError(f"Unknown surrogate type: {type(surrogate).__name__}")
        except ImportError:
            try:
                from Surrogate.pce_model import PCESurrogateModel

                if isinstance(surrogate, PCESurrogateModel):
                    new_surrogate = retrain_pce(
                        surrogate, merged, updated_train_idx, updated_test_idx,
                        config, iteration,
                    )
                else:
                    raise TypeError(
                        f"Unknown surrogate type: {type(surrogate).__name__}"
                    )
            except ImportError:
                raise TypeError(
                    f"Unknown surrogate type: {type(surrogate).__name__}"
                )

    # Post-retrain error
    new_metrics = validate_surrogate(new_surrogate, test_params, test_cd, test_pc)
    new_error = float(new_metrics[config.quality_metric])
    error_ratio = new_error / max(old_error, 1e-30)
    quality_passed = error_ratio <= config.max_degradation_ratio

    # Save merged data
    save_path = save_merged_data(
        merged, updated_train_idx, updated_test_idx,
        config.output_base_dir, iteration,
    )

    logger.info(
        "ISMO iter %d complete: %s error %.6e -> %.6e (ratio=%.4f, %s)",
        iteration,
        config.quality_metric,
        old_error,
        new_error,
        error_ratio,
        "PASS" if quality_passed else "FAIL",
    )

    return ISMORetrainResult(
        surrogate=new_surrogate,
        merged_data=merged,
        updated_train_idx=updated_train_idx,
        updated_test_idx=updated_test_idx,
        quality_passed=quality_passed,
        old_error=old_error,
        new_error=new_error,
        error_ratio=error_ratio,
        quality_metric=config.quality_metric,
        retrain_method=retrain_method,
        iteration=iteration,
        save_path=save_path,
    )
