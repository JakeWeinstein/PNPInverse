# Bug Report: Surrogate Data Pipeline

**Focus:** Train/test leakage, normalization, feature ordering, data consistency
**Agent:** Surrogate Data Pipeline

---

## BUG 1: train_nn_surrogate.py validates on data that includes training points
**File:** `scripts/surrogate/train_nn_surrogate.py:236-250`
**Severity:** HIGH
**Description:** After training, creates random validation set from full dataset: `val_check_idx = rng.choice(N, size=n_val_check, replace=False)`. Since ensemble uses ~85% of data for training, ~85% of validation indices overlap with training data. Reported per-member metrics are a mix of train and test performance.
**Suggested fix:** Use `split_indices.npz` test indices for final validation.

## BUG 2: validate_surrogate.py can validate on training data
**File:** `scripts/surrogate/validate_surrogate.py:50-67`
**Severity:** MEDIUM
**Description:** Uses ALL data from `--test-data` argument. If user passes `training_data_merged.npz` (default usage pattern), validation includes training points, producing artificially low error.
**Suggested fix:** Add `--split-indices` argument, default to evaluating on test split only.

## BUG 3: Smoothness penalty crosses CD/PC output boundary
**File:** `Surrogate/nn_training.py:158`
**Severity:** MEDIUM
**Description:** Second-order finite differences span the concatenated `[CD | PC]` output. At the boundary, differences mix current density with peroxide current -- physically meaningless penalty.
**Suggested fix:** Split into separate CD and PC halves before computing differences.

## BUG 4: load_surrogate() only works for BVSurrogateModel
**File:** `Surrogate/io.py:33-64`
**Severity:** MEDIUM
**Description:** `isinstance(model, BVSurrogateModel)` check rejects POD-RBF models. POD-RBF `.pkl` files saved by overnight training scripts must be loaded via raw `pickle.load()` with no backward-compatibility patches.
**Suggested fix:** Make `load_surrogate()` polymorphic.

## BUG 5: K-fold CV in POD-RBF does not shuffle data
**File:** `Surrogate/pod_rbf_model.py:282-308`
**Severity:** LOW
**Description:** Contiguous fold slicing without shuffling. If data has spatial ordering, folds are biased.

## BUG 6: Stale backup data files in production path
**File:** `data/surrogate_models/split_indices_v11_backup.npz`, `training_data_merged_v11_backup.npz`
**Severity:** LOW
**Description:** Old v11 backup files alongside current canonical files. Potential for accidental use.
**Suggested fix:** Move to archive/ subdirectory.

## BUG 7: NNSurrogateModel.fit() training_bounds computed on full data before split
**File:** `Surrogate/nn_model.py:306-338`
**Severity:** LOW
**Description:** `training_bounds` reflects full dataset (including validation) while normalizer reflects only training subset. Inconsistent with `nn_training.py` which computes bounds post-split. Only affects informational/warning purposes.

## BUG 8: overnight_train_v11 uses local EnsembleMeanWrapper with ddof=0
**File:** `scripts/surrogate/overnight_train_v11.py:489-512`
**Severity:** LOW
**Description:** Duplicates canonical `EnsembleMeanWrapper` but uses `ddof=0` (population std) vs canonical `ddof=1` (sample std). Also lacks `predict_torch()` method.
**Suggested fix:** Import canonical `Surrogate.ensemble.EnsembleMeanWrapper`.

## BUG 9: No feature ordering validation at inference time
**Severity:** LOW
**Description:** All surrogates expect `[k0_1, k0_2, alpha_1, alpha_2]` positionally. No schema validation. Design limitation, not active bug.

---

## Confirmed NOT Bugs
- **split_indices.npz system**: Correctly used across overnight training, GP, PCE, and ISMO scripts
- **Normalization stats**: Correctly computed from training data only
- **Denormalization**: `predict_batch()` correctly calls `inverse_transform()` before returning
- **Model checkpoint loading**: Architecture metadata correctly read and matched

---

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 0     |
| HIGH     | 1     |
| MEDIUM   | 3     |
| LOW      | 5     |

**Top priority:** Bug 1 -- validation metrics from `train_nn_surrogate.py` are unreliable due to train/test overlap.
