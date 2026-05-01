# Bug Report: Test Coverage Gaps

**Focus:** Test coverage gaps, missing assertions, unreliable tests
**Agent:** Test Coverage Gaps

---

## CRITICAL Issues

### BUG 1: Overly loose tolerance hides parameter recovery failures
**File:** `tests/test_multistart.py:332-342`
**Severity:** CRITICAL
**Description:** `test_perfect_data_recovery` uses `max_err < 1.0` (100% relative error!) as threshold. With noiseless data, a parameter could be twice its true value and still pass.
**Suggested fix:** Tighten to `< 0.15` (15%).

### BUG 2: `KMP_DUPLICATE_LIB_OK` environment variable mutation at import time
**Files:** `tests/test_pipeline_reproducibility.py:42`, `tests/test_v13_verification.py:34`, `tests/test_inverse_verification.py:45`
**Severity:** CRITICAL
**Description:** `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` at module level leaks state to all tests in the process, masking real OpenMP/MKL library issues.
**Suggested fix:** Move to session-scoped fixture or use `monkeypatch.setenv`.

---

## HIGH Issues

### BUG 3: Tests depend on model files without skip guards
**File:** `tests/test_surrogate_fidelity.py:59-74`
**Severity:** HIGH
**Description:** All `TestSurrogateFidelity` tests depend on trained models on disk. No `pytest.skip()` if missing -- crashes with `FileNotFoundError`.
**Suggested fix:** Add `pytest.mark.skipif(not os.path.isdir(...))`.

### BUG 4: Soft gate NRMSE threshold of 20% is very loose
**File:** `tests/test_surrogate_fidelity.py:79`
**Severity:** HIGH
**Description:** `_NRMSE_THRESHOLD = 0.20` passes models with 19% median error. Too inaccurate for reliable parameter inference.

### BUG 5: Flaky subprocess-dependent pipeline test
**File:** `tests/test_pipeline_reproducibility.py:384-485`
**Severity:** HIGH
**Description:** Fixture runs subprocess with 900s timeout, reads shared mutable CSV artifact. Fragile.

### BUG 6: Class-level state mutation between tests
**File:** `tests/test_mms_convergence.py:229, 245, 285`
**Severity:** HIGH
**Description:** Tests store results on class via `_l2_results`, `_h1_results`, `_gci_results`. Creates ordering dependencies. Breaks parallel execution.
**Suggested fix:** Use class-scoped or module-scoped fixture.

### BUG 7: Test writes to project directory, not tmp_path
**File:** `tests/test_mms_convergence.py:291-362`
**Severity:** HIGH
**Description:** `test_save_convergence_artifacts` writes to `StudyResults/mms_convergence/` instead of `tmp_path`. Persistent side effects.

### BUG 8: Missing assertions on convergence quality
**File:** `tests/test_bv_forward.py:63-95`
**Severity:** HIGH
**Description:** Only asserts strategy returns `True`. No checks on residuals, convergence rate, or flux values. A solver converging to a wrong solution would pass.

---

## MEDIUM Issues

### BUG 9: 70% monotonicity threshold too lenient
**File:** `tests/test_v13_verification.py:346`
**Severity:** MEDIUM
**Description:** 30% of Tafel regime points can be non-monotone. Should be 85-90% for a fundamental physical property.

### BUG 10: `test_subset_idx_restricts_voltages` asserts inequality, not correctness
**File:** `tests/test_multistart.py:344-394`
**Severity:** MEDIUM

### BUG 11: Mock-heavy tests with no `spec=` enforcement
**File:** `tests/test_acquisition.py:56-89`
**Severity:** MEDIUM
**Description:** `MagicMock` without `spec=` won't detect interface changes.

### BUG 12: Non-deterministic hash-based RNG in mock
**File:** `tests/test_ismo_pde_eval.py:103`
**Severity:** MEDIUM
**Description:** `hash(params.tobytes())` not deterministic across Python sessions.

---

## Module Coverage Gaps (MOST SEVERE)

### FluxCurve/ -- 18 modules, ZERO direct tests
The entire flux curve computation pipeline including caching, parallelism, warm-starting, and optimization has no test coverage. This is the highest-risk gap.

### Inverse/ -- 7 of 8 modules untested
Only `inference_runner/config.py` has tests. Core inverse objective, solver interface, and recovery logic have no unit tests.

### Forward/ -- Core PDE solvers untested
`dirichlet_solver.py`, `robin_solver.py`, `bv_solver/forms.py` have no unit tests. Only integration-level coverage via `test_bv_forward.py`.

### Surrogate/ -- Key modules untested
`cascade.py`, `ismo.py`, `gp_model.py`, `pce_model.py`, `nn_training.py`, `training.py`, `io.py`, `bcd.py`, `sampling.py` all lack direct tests.

---

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 2 (4 instances) |
| HIGH     | 6     |
| MEDIUM   | 4     |
| LOW      | 4 (not listed above) |

**Top priorities:**
1. Fix the 100% error tolerance in parameter recovery test
2. Centralize `KMP_DUPLICATE_LIB_OK` setting
3. Add skip guards for file-dependent tests
4. Add any tests at all for FluxCurve/ and Inverse/ core modules
