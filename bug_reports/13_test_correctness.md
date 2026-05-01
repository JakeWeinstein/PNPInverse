# Bug Report: Test Correctness

**Focus:** Wrong expected values, flaky patterns, tests passing for wrong reasons
**Agent:** Test Correctness

---

## BUG 1: `test_subset_idx_restricts_voltages` may pass for wrong reason
**File:** `tests/test_multistart.py:344-393`
**Severity:** MEDIUM
**Description:** Creates configs with `use_shallow_subset=True` and `False`, passes `subset_idx` to both, then asserts losses differ. The test only checks inequality (`!=`), not a specific expected relationship. It may pass even if subset logic is broken.
**Suggested fix:** Verify that the subset run's objective is computed over only 4 voltage points while the full run uses all points.

## BUG 2: `test_quality_check_detects_degradation` tests inverted condition
**File:** `tests/test_ismo_retrain.py:418-442`
**Severity:** MEDIUM
**Description:** Compares a model to itself (ratio=1.0) with `max_degradation_ratio=0.5`. Name says "detects_degradation" but actually tests a model with no degradation. Should test with a genuinely worse model.
**Suggested fix:** Create a deliberately worse model and verify degradation detection with a reasonable threshold like 1.1.

## BUG 3: `test_contains_v13_cathodic_points` uses `in` on float array
**File:** `tests/test_sensitivity_visualization.py:86`
**Severity:** MEDIUM
**Description:** `assert v in grid` where `v` is a float and `grid` is a numpy array. Uses exact equality, fragile for floating-point.
**Suggested fix:** Use `assert np.any(np.isclose(grid, v))`.

## BUG 4: Floating-point equality in GCI helper
**File:** `tests/test_mms_convergence.py:152`
**Severity:** LOW
**Description:** `if e_fine == 0.0:` uses exact float equality. A very small but non-zero error would take the else branch.
**Suggested fix:** Use `if e_fine < 1e-30:`.

## BUG 5: Mock GP in `test_with_mock_gp` uses fixed random output
**File:** `tests/test_acquisition.py:363-391`
**Severity:** LOW
**Description:** Mock GP returns deterministic uncertainty regardless of input. Test would pass even if uncertainty-based ranking logic was completely broken.
**Suggested fix:** Make mock return uncertainty that varies with input, verify selection correlation.

## BUG 6: Exact float `== 0.0` comparison for NRMSE
**File:** `tests/test_ismo_pde_eval.py:405-407`
**Severity:** LOW
**Description:** `assert comp.cd_mean_nrmse == 0.0` uses exact equality. Should use `pytest.approx(0.0, abs=1e-15)`.

## BUG 7: `test_noise_is_zero_mean_approximately` has very loose threshold
**File:** `tests/test_steady_state_common.py:205-211`
**Severity:** LOW
**Description:** Checks `abs(mean_noise) < 0.1` with 10,000 samples -- a ~20-sigma threshold. Wouldn't catch moderate bias.

---

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 0     |
| HIGH     | 0     |
| MEDIUM   | 3     |
| LOW      | 4     |

Overall test suite is well-written. The three MEDIUM issues could mask real defects or cause spurious failures.
