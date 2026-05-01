# Bug Report: Scripts Logic Errors

**Focus:** Code quality and logic bugs in scripts/
**Agent:** Scripts Logic Errors

---

## BUG 1: Wrong sign on `observable_scale` in sensitivity_visualization.py
**File:** `scripts/studies/sensitivity_visualization.py:318`
**Severity:** HIGH
**Description:** Passes `observable_scale=float(deps["I_SCALE"])` (positive) while every other script uses `observable_scale=-I_SCALE` (negative). The negative sign accounts for cathodic (negative current) convention. Using wrong sign produces I-V curves with flipped polarity, making sensitivity analysis results incorrect.
**Suggested fix:** Change to `observable_scale=float(-deps["I_SCALE"])`.

## BUG 2: Multi-seed runner reads fixed CSV path (race/stale data risk)
**File:** `scripts/studies/run_multi_seed_v13.py:200-204`
**Severity:** HIGH
**Description:** `run_single_seed()` reads results from fixed path `StudyResults/master_inference_v13/master_comparison_v13.csv`. If subprocess fails silently but stale CSV from previous seed exists, it reads wrong results.
**Suggested fix:** Pass per-seed output directory or verify CSV was modified after subprocess started.

## BUG 3: Hardcoded `N_WORKERS = 8`
**Files:** `scripts/surrogate/overnight_train_v11.py:57`, `overnight_train_v12_gapfill.py:93`, `resolve_missing.py:175`
**Severity:** MEDIUM
**Description:** Over-subscribes CPUs on smaller machines, under-utilizes larger ones.
**Suggested fix:** Use `min(8, os.cpu_count() or 4)` or add `--workers` CLI argument.

## BUG 4: Silent data overwrite without warning
**Files:** `overnight_train_v11.py:328-330`, `validate_surrogate.py:75-83`, `multistart_inference.py:573-616`
**Severity:** MEDIUM
**Description:** Multiple scripts write output files without checking existence. Previous results silently overwritten.
**Suggested fix:** Add `--force` flag or existence check.

## BUG 5: Same noise seed for all parameter recovery test cases
**File:** `scripts/surrogate/overnight_train_v11.py:643-644`
**Severity:** MEDIUM
**Description:** `rng_noise = np.random.default_rng(42)` re-created inside loop. All 10 test cases get same noise pattern, reducing diversity.
**Suggested fix:** Create RNG once before loop, or use `np.random.default_rng(42 + i)`.

## BUG 6: `D_ref` inconsistency between `_bv_common.py` and `test_bv_forward.py`
**File:** `scripts/verification/test_bv_forward.py:66`
**Severity:** MEDIUM
**Description:** Uses `D_ref = np.sqrt(D_O2 * D_H2O2)` with `D_O2 = 1.5e-9`, while `_bv_common.py` uses `D_REF = D_O2` with `D_O2 = 1.9e-9`. Different physical assumptions could cause inconsistencies if outputs used as inputs elsewhere.
**Suggested fix:** Centralize reference scale choices or document independence.

## BUG 7: `training_data_audit.py` assumes specific CSV column names
**File:** `scripts/studies/training_data_audit.py:381`
**Severity:** MEDIUM
**Description:** Reads `nn_ensemble_cd_nrmse` column without checking existence. Will throw unhelpful `KeyError` if columns differ.

## BUG 8: Relative paths in gradient/autograd benchmark scripts
**Files:** `scripts/studies/gradient_benchmark.py:472-531`, `scripts/benchmark_autograd_vs_fd.py:22`
**Severity:** MEDIUM
**Description:** Relative paths only work from PNPInverse/ root.
**Suggested fix:** Use `_ROOT`-based absolute paths.

## BUG 9: `resolve_missing.py` hardcodes checkpoint paths
**File:** `scripts/surrogate/resolve_missing.py:166-170`
**Severity:** MEDIUM
**Description:** Hardcoded relative paths with no CLI argument fallback. Unhelpful error on missing files.

## BUG 10: Progress ETA calculation wrong variable
**File:** `scripts/surrogate/overnight_train_v12_gapfill.py:807-809`
**Severity:** LOW
**Description:** `new_done` is typically 0 because `completed_indices` was already updated, making ETA meaningless.

## BUG 11: Dead `_make_sp_mms` function with wrong stoichiometry
**File:** `scripts/verification/mms_bv_4species.py:190`
**Severity:** LOW
**Description:** Broken function still in file (fixed version `_make_sp_mms_fixed` is used instead). Could confuse developers.

## BUG 12: `np.random.RandomState` instead of `default_rng`
**File:** `scripts/studies/inverse_benchmark_all_models.py:244`
**Severity:** LOW
**Description:** Legacy RNG API produces different noise than `default_rng` used elsewhere. Subtle reproducibility difference.

## BUG 13: Stale worker logs not cleaned
**File:** `scripts/surrogate/overnight_train_v12_gapfill.py`
**Severity:** LOW
**Description:** Previous run's `worker_*_results.npz` files could be mixed with new results.

## BUG 14: Hardcoded paths in docstrings
**File:** `scripts/surrogate/overnight_train_v11.py:5-7`
**Severity:** LOW
**Description:** Absolute paths (`/Users/jakeweinstein/...`) in docstring usage examples.

---

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 0     |
| HIGH     | 2     |
| MEDIUM   | 7     |
| LOW      | 5     |

**Top priority:** Bug 1 (wrong sign flips I-V polarity), Bug 2 (stale CSV from wrong seed).
