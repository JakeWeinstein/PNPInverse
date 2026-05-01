# Bug Report: Data/IO File Paths

**Focus:** File I/O consistency, path construction, data format assumptions
**Agent:** Data/IO File Paths

---

## BUG 1: File handle leak in nn_training.py CSV writer
**File:** `Surrogate/nn_training.py:283`
**Severity:** HIGH
**Description:** CSV log file opened with `open(csv_path, "w", newline="")` but not wrapped in `with` block or try/finally. If training throws an exception, file handle leaked.
**Suggested fix:** Use context manager or try/finally.

## BUG 2: Broken validation in ismo_pde_eval.py `integrate_new_data()`
**File:** `Surrogate/ismo_pde_eval.py:532`
**Severity:** HIGH
**Description:** Voltage grid validation is syntactically broken -- always evaluates to True. Voltage grid mismatches go undetected during ISMO data integration, potentially corrupting training datasets with misaligned arrays.
**Suggested fix:** Replace with proper length/value comparison or rely on the length check at line 536.

## BUG 3: `np.load()` without `allow_pickle` on potentially mixed data
**File:** `Surrogate/ismo_pde_eval.py:170`
**Severity:** MEDIUM
**Description:** `np.load(training_data_path)` without `allow_pickle`. Augmented .npz files from ISMO iterations include `ismo_metadata` (a pickled string array) which requires `allow_pickle=True`.
**Suggested fix:** Add `allow_pickle=True` or ensure only numeric arrays.

## BUG 4: Multiple `np.load()` calls without `allow_pickle` in scripts
**Files:** `scripts/surrogate/run_ismo.py:281`, `scripts/surrogate/run_ismo_live.py:74,81`, `scripts/studies/gradient_benchmark.py:545-546`, `scripts/studies/parameter_recovery_all_models.py:132`
**Severity:** MEDIUM
**Description:** Missing `allow_pickle=True` for potentially augmented data files.

## BUG 5: Relative path in library function `make_standard_pde_bundle`
**File:** `Surrogate/ismo_pde_eval.py:113`
**Severity:** MEDIUM
**Description:** Default `training_data_path="data/surrogate_models/training_data_merged.npz"` is a relative path in a library function. Callers from different directories get `FileNotFoundError`.
**Suggested fix:** Compute default path relative to package root.

## BUG 6: Relative paths in `run_ismo_live.py`
**File:** `scripts/surrogate/run_ismo_live.py:74, 81`
**Severity:** MEDIUM
**Description:** Hardcoded relative paths assume script runs from PNPInverse/ root.

## BUG 7: `Infer_BVMaster_charged_v13_ultimate.py` uses relative model paths
**File:** `scripts/Inference/Infer_BVMaster_charged_v13_ultimate.py:91-96`
**Severity:** MEDIUM
**Description:** `_MODEL_PATHS` uses relative paths but never prepends `_ROOT`. Only works from PNPInverse/ directory.
**Suggested fix:** Prepend `_ROOT` to model paths.

## BUG 8: `resolve_missing.py` inconsistent `allow_pickle` usage
**File:** `scripts/surrogate/resolve_missing.py:248, 277`
**Severity:** MEDIUM
**Description:** Line 166 uses `allow_pickle=True` correctly, but lines 248 and 277 don't specify it.

## BUG 9: Pickle-based serialization unguarded
**Files:** `Surrogate/io.py:47`, `Surrogate/pce_model.py:827`, multiple scripts
**Severity:** LOW
**Description:** All `pickle.load()` calls unguarded. Acceptable for research code with local files.

## BUG 10: ISMODiagnostics CSV header edge case
**File:** `Surrogate/ismo_diagnostics.py:83-88`
**Severity:** LOW
**Description:** If CSV deleted between calls within same process instance, header won't be re-written.

---

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 0     |
| HIGH     | 2     |
| MEDIUM   | 6     |
| LOW      | 2     |

**Top priority:** Bug 1 (file handle leak), Bug 2 (broken validation enabling silent data corruption).
