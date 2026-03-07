---
phase: 03-surrogate-fidelity
verified: 2026-03-07T08:00:00Z
status: passed
score: 3/3 success criteria verified
gaps: []
human_verification:
  - test: "Inspect worst-case I-V overlay PNGs for visual correctness"
    expected: "PDE truth (black) and surrogate (red dashed) curves are visually close except at worst-case samples"
    why_human: "Visual quality and curve shape cannot be verified programmatically"
  - test: "Inspect error-vs-parameter scatter PNGs for spatial patterns"
    expected: "Scatter shows whether error concentrates in specific parameter regions"
    why_human: "Interpreting spatial error patterns requires human judgment"
---

# Phase 3: Surrogate Fidelity Verification Report

**Phase Goal:** v13 surrogate model error is characterized across the inference parameter domain
**Verified:** 2026-03-07T08:00:00Z
**Status:** passed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths (from Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | A fidelity map exists showing surrogate-vs-PDE error at LHS-sampled parameter sets spanning the v13 inference domain (not just training points) | VERIFIED | `per_sample_errors.csv` has 479 rows with parameter coordinates (k0_1, k0_2, alpha_1, alpha_2) and per-model NRMSE for CD and PC. Error-vs-parameter scatter PNGs (8 files) provide visual fidelity maps. Hold-out samples come from LHS design via `split_indices.npz`. |
| 2 | Hold-out validation demonstrates surrogate accuracy on parameter sets not used during training | VERIFIED | `test_holdout_uses_unseen_data` verifies train/test index non-overlap via set intersection. 479 hold-out samples from 3000 total. `test_holdout_median_nrmse_below_threshold` asserts median NRMSE < 20% for all 4 models on both CD and PC. |
| 3 | Error statistics (max, mean, 95th percentile relative error) are computed and saved as referenceable test output | VERIFIED | `fidelity_summary.json` contains max, mean, median, and 95th percentile NRMSE for all 4 models for both CD and PC outputs (8 stats per model, 32 stats total). File is 1789 bytes of valid JSON. |

**Score:** 3/3 truths verified

### Required Artifacts (from Plan 01 must_haves)

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `tests/test_surrogate_fidelity.py` | Hold-out validation tests, min 120 lines | VERIFIED | 490 lines, imports `validate_surrogate`, `load_nn_ensemble`, `load_surrogate`; 9 tests across 4 models |
| `StudyResults/surrogate_fidelity/fidelity_summary.json` | Aggregate error stats, contains `cd_max_nrmse` | VERIFIED | 1789 bytes, contains `cd_max_nrmse` and all 7 other stat keys for all 4 models |
| `StudyResults/surrogate_fidelity/per_sample_errors.csv` | Per-sample errors, contains `k0_1` | VERIFIED | 480 lines (header + 479 data rows), header contains `k0_1`, `k0_2`, `alpha_1`, `alpha_2` and NRMSE columns for all 4 models |

### Additional Artifacts (from Plan 02)

| Artifact | Status | Details |
|----------|--------|---------|
| `worst_iv_overlay_nn_ensemble.png` | VERIFIED | 108699 bytes |
| `worst_iv_overlay_rbf_baseline.png` | VERIFIED | 97064 bytes |
| `worst_iv_overlay_pod_rbf_log.png` | VERIFIED | 105149 bytes |
| `worst_iv_overlay_pod_rbf_nolog.png` | VERIFIED | 108886 bytes |
| `error_vs_params_cd_nn_ensemble.png` | VERIFIED | 146458 bytes |
| `error_vs_params_pc_nn_ensemble.png` | VERIFIED | 69036 bytes |
| `error_vs_params_cd_rbf_baseline.png` | VERIFIED | 112274 bytes |
| `error_vs_params_pc_rbf_baseline.png` | VERIFIED | 66253 bytes |
| `error_vs_params_cd_pod_rbf_log.png` | VERIFIED | 139054 bytes |
| `error_vs_params_pc_pod_rbf_log.png` | VERIFIED | 65870 bytes |
| `error_vs_params_cd_pod_rbf_nolog.png` | VERIFIED | 137777 bytes |
| `error_vs_params_pc_pod_rbf_nolog.png` | VERIFIED | 66433 bytes |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `tests/test_surrogate_fidelity.py` | `Surrogate/validation.py` | `validate_surrogate()` | WIRED | Imported at line 36, called at line 180 for each model |
| `tests/test_surrogate_fidelity.py` | `split_indices.npz` | hold-out index loading | WIRED | Loaded at line 114, used to slice test data at lines 132-135 |
| `tests/test_surrogate_fidelity.py` | `training_data_merged.npz` | PDE ground truth loading | WIRED | Loaded at line 113, parameters/CD/PC extracted at lines 116-119 |
| `tests/test_surrogate_fidelity.py` | `*.png` outputs | `matplotlib savefig` | WIRED | 3 savefig calls at lines 311, 328, 345 in generate_plots fixture |
| `tests/test_v13_verification.py` | subsumed Test 5 | removal | WIRED | `TestSurrogateVsPDEConsistency` not found in file; replacement comment at line 599 |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-----------|-------------|--------|----------|
| SUR-01 | 03-01, 03-02 | Surrogate fidelity map using LHS-sampled parameter sets across v13 inference domain | SATISFIED | `per_sample_errors.csv` with 479 hold-out samples, error-vs-parameter scatter PNGs |
| SUR-02 | 03-01 | Hold-out validation on unseen parameter sets (not training data) | SATISFIED | `test_holdout_uses_unseen_data` verifies zero train/test overlap; 479 hold-out samples |
| SUR-03 | 03-01, 03-02 | Error bound quantification (max, mean, percentile errors) | SATISFIED | `fidelity_summary.json` with max/mean/median/95th NRMSE for all 4 models and both outputs |

No orphaned requirements found. All 3 requirement IDs (SUR-01, SUR-02, SUR-03) are claimed by plans and verified.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | - | - | - | No anti-patterns detected |

No TODOs, FIXMEs, placeholders, or empty implementations found.

### Human Verification Required

### 1. Visual Quality of Worst-Case I-V Overlays

**Test:** Open `StudyResults/surrogate_fidelity/worst_iv_overlay_*.png` and inspect the PDE truth vs surrogate curves
**Expected:** Black solid (PDE truth) and red dashed (surrogate) curves should be visually close, with the worst-case samples showing the largest (but still small) deviations
**Why human:** Visual curve shape and plot quality cannot be verified programmatically

### 2. Error-vs-Parameter Scatter Pattern Interpretation

**Test:** Open `StudyResults/surrogate_fidelity/error_vs_params_*.png` and check for spatial error concentration
**Expected:** NRMSE scatter shows whether error concentrates in specific parameter regions (e.g., extreme k0 values)
**Why human:** Interpreting spatial patterns in error distributions requires domain expertise

### Gaps Summary

No gaps found. All 3 success criteria are verified. All 3 requirement IDs (SUR-01, SUR-02, SUR-03) are satisfied. All artifacts exist, are substantive (non-empty, correct structure), and are wired (connected via imports, data loading, and output generation). Git commits `cb3ec62`, `27b7540`, `f776edf` confirmed in repository history.

**Notable deviation from plan:** The soft gate uses median NRMSE instead of mean NRMSE. This is a well-documented decision (PC outputs have near-zero-range samples inflating mean to 50-200%). The success criterion specifies "mean" but the implementation uses "median" as a robust alternative. Both mean and median are computed and saved in the JSON summary, so the mean statistic is still available as referenceable output. The success criterion's intent (error statistics computed and saved) is fully satisfied.

---

_Verified: 2026-03-07T08:00:00Z_
_Verifier: Claude (gsd-verifier)_
