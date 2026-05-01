# Round 2 Bug Audit: Test Suite

Audited all 21 test files in `tests/` for correctly applied fixes, newly introduced bugs, and remaining issues missed in round 1.

---

## Verified Fixes

### VF-01: conftest.py KMP_DUPLICATE_LIB_OK session fixture
- **File:** `tests/conftest.py`, lines 20-23
- **Status:** CORRECTLY APPLIED
- KMP env var is set once via `@pytest.fixture(autouse=True, scope="session")`, preventing scattered `os.environ` calls across test files. Fixture correctly fires before any test module.

### VF-02: test_pipeline_reproducibility.py KMP line removed
- **File:** `tests/test_pipeline_reproducibility.py`
- **Status:** CORRECTLY APPLIED
- No `KMP_DUPLICATE_LIB_OK` assignment at module level. The conftest session fixture handles it.

### VF-03: test_v13_verification.py KMP line removed
- **File:** `tests/test_v13_verification.py`
- **Status:** CORRECTLY APPLIED
- No `KMP_DUPLICATE_LIB_OK` assignment at module level.

### VF-04: test_inverse_verification.py KMP line removed
- **File:** `tests/test_inverse_verification.py`
- **Status:** CORRECTLY APPLIED
- No `KMP_DUPLICATE_LIB_OK` assignment at module level. Note: the embedded subprocess script at line 234 still correctly sets `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` because that subprocess runs in its own process where conftest does not apply -- this is correct.

### VF-05: test_multistart.py tolerance tightened to 0.15 / 1e-4
- **File:** `tests/test_multistart.py`, lines 332-343
- **Status:** CORRECTLY APPLIED
- `test_perfect_data_recovery` now asserts `max_err < 0.15` (was 1.0) and `best_loss < 1e-4` (meaningful). The old 100% threshold was indeed meaningless for a noiseless surrogate-on-surrogate test.

### VF-06: test_surrogate_fidelity.py skip guards for missing models
- **File:** `tests/test_surrogate_fidelity.py`, lines 69-85
- **Status:** CORRECTLY APPLIED
- `_REQUIRED_FILES_PRESENT` checks all core model files and data paths. GP/PCE are conditionally added to `MODEL_NAMES` only if their dependencies and files exist (lines 91-95). The `_skip_missing_models` decorator on the test class correctly skips when files are absent.

---

## NEW Bugs Introduced by Fixes

### NB-01: conftest.py KMP fixture lacks yield/teardown
- **Severity:** LOW
- **File:** `tests/conftest.py`, line 20-23
- **Description:** The `_set_kmp_env` session fixture sets `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` but does not restore the original value on teardown. For a session-scoped fixture this is effectively harmless (the process exits after tests), but it violates the immutability principle and could leak state if conftest is imported in other contexts.
- **Fix suggestion:** Use a yield fixture pattern:
  ```python
  @pytest.fixture(autouse=True, scope="session")
  def _set_kmp_env():
      old = os.environ.get("KMP_DUPLICATE_LIB_OK")
      os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
      yield
      if old is None:
          os.environ.pop("KMP_DUPLICATE_LIB_OK", None)
      else:
          os.environ["KMP_DUPLICATE_LIB_OK"] = old
  ```

---

## REMAINING Bugs Missed in Round 1

### RB-01: test_nondim.py bare import from conftest without package prefix
- **Severity:** MEDIUM
- **File:** `tests/test_nondim.py`, line 14
- **Line:** `from conftest import skip_without_firedrake`
- **Description:** Uses a bare `from conftest import` instead of `from tests.conftest import`. This works only when pytest is run from the `tests/` directory or when pytest's rootdir detection adds `tests/` to `sys.path`. Running `python -m pytest tests/test_nondim.py` from the project root may fail with `ModuleNotFoundError` depending on pytest configuration. Other files like `test_pipeline_reproducibility.py` (line 50) and `test_mms_convergence.py` (line 37) correctly use `from tests.conftest import` or `from conftest import` with the sys.path insert above.
- **Fix suggestion:** Change to `from tests.conftest import skip_without_firedrake` for consistency with `test_pipeline_reproducibility.py` and `test_inverse_verification.py`.

### RB-02: test_mms_convergence.py bare import from conftest
- **Severity:** MEDIUM
- **File:** `tests/test_mms_convergence.py`, line 37
- **Line:** `from conftest import skip_without_firedrake`
- **Description:** Same issue as RB-01. Uses bare `from conftest import` which is fragile depending on how pytest resolves the path.
- **Fix suggestion:** Change to `from tests.conftest import skip_without_firedrake`.

### RB-03: test_surrogate_fidelity.py does not import or use skip_without_firedrake
- **Severity:** LOW
- **File:** `tests/test_surrogate_fidelity.py`
- **Description:** The module loads `load_nn_ensemble`, `load_surrogate`, and `validate_surrogate` at import time (lines 36-38). If PyTorch is not installed, importing `load_nn_ensemble` will fail at collection time, crashing all tests in the module instead of gracefully skipping. The `_skip_missing_models` guard only checks file paths, not import availability.
- **Fix suggestion:** Wrap the `from Surrogate.ensemble import load_nn_ensemble` in a try/except and add a skip condition for missing PyTorch, similar to how `test_autograd_gradient.py` handles it.

### RB-04: test_surrogate_fidelity.py all_models fixture does not guard GP/PCE loading
- **Severity:** LOW
- **File:** `tests/test_surrogate_fidelity.py`, lines 196-218
- **Description:** The `all_models` fixture loads core models unconditionally (lines 200-208) without the `_skip_missing_models` guard. If any of the 4 core model files are missing but the class-level skip is somehow bypassed (e.g., running a specific test method), the fixture will raise a `FileNotFoundError` instead of a clean skip. The fixture also calls `load_nn_ensemble` without a try/except around PyTorch import failures.
- **Fix suggestion:** Add explicit file existence checks inside the fixture, or rely on the class-level `_skip_missing_models` decorator (which is already present, making this a defense-in-depth issue).

### RB-05: test_inverse_verification.py pde_targets fixture sets KMP in subprocess string
- **Severity:** INFO (not a bug, but worth documenting)
- **File:** `tests/test_inverse_verification.py`, line 234
- **Description:** The subprocess generation script embeds `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"`. This is correct because the subprocess does not inherit the conftest session fixture. No action needed, but it should remain if the conftest approach changes.

### RB-06: test_multistart.py redundant sys.path manipulation
- **Severity:** LOW
- **File:** `tests/test_multistart.py`, lines 26-29
- **Description:** The file manually inserts the project root into `sys.path`, but `conftest.py` already does this at lines 49-52. The redundant manipulation is harmless but adds maintenance burden.
- **Fix suggestion:** Remove the `sys.path` manipulation from `test_multistart.py` (and similarly from `test_v13_verification.py` lines 34-37, `test_inverse_verification.py` lines 45-48, `test_pipeline_reproducibility.py` lines 42-45, `test_multi_seed_aggregation.py` lines 18-21, `test_sensitivity_visualization.py` lines 17-20). The conftest handles this.

### RB-07: test_pipeline_reproducibility.py subprocess does not set KMP env
- **Severity:** LOW
- **File:** `tests/test_pipeline_reproducibility.py`, lines 382-395
- **Description:** The `full_pipeline_results` fixture runs the v13 inference script via `subprocess.run()` but does not pass `KMP_DUPLICATE_LIB_OK=TRUE` in the subprocess environment. The subprocess inherits the parent process env (which has it set via conftest), so this works in practice. However, if the subprocess is spawned in a clean environment (e.g., CI with env isolation), the KMP setting would be lost.
- **Fix suggestion:** Explicitly pass `env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"}` to `subprocess.run()` for robustness, or rely on the script itself setting it.

### RB-08: test_autograd_gradient.py no assertion on J_auto value
- **Severity:** LOW
- **File:** `tests/test_autograd_gradient.py`, lines 159-171
- **Description:** In `test_surrogate_objective_gradient_match`, the function calls `obj._autograd_objective_and_gradient(x)` which returns `(J_auto, g_auto)`, but only `g_auto` is validated against FD. The `J_auto` value is never checked for consistency with `obj.objective(x)` (this is done in a separate test at line 311, but the gradient test itself does not verify J). This is a weak gap -- if the autograd path returns a correct gradient but wrong objective value for some inputs, the per-point gradient tests would miss it.
- **Fix suggestion:** Add `assert abs(J_auto - obj.objective(x)) < 1e-10 * max(abs(J_auto), 1.0)` inside the gradient loop.

### RB-09: test_ismo_pde_eval.py MockSurrogate uses hash-based seed (non-deterministic on Python 3.12+)
- **Severity:** LOW
- **File:** `tests/test_ismo_pde_eval.py`, line 103
- **Line:** `rng = np.random.default_rng(hash(params.tobytes()) % (2**31))`
- **Description:** `hash()` is randomized by default in Python 3.12+ (PYTHONHASHSEED). This means the mock surrogate predictions vary across test runs, which could cause intermittent test failures if any assertion depends on the exact mock output values. Currently the tests using this mock only check structure and convergence detection (not exact values), so this is low risk.
- **Fix suggestion:** Use a deterministic hash function (e.g., `int.from_bytes(hashlib.md5(params.tobytes()).digest()[:4], 'little') % (2**31)`) or a fixed seed.

### RB-10: test_ismo_retrain.py uses tempfile.mkdtemp without cleanup
- **Severity:** LOW
- **File:** `tests/test_ismo_retrain.py`, lines 234, 274, 312, 368, 481, 514, 635, etc.
- **Description:** Multiple tests call `tempfile.mkdtemp()` directly but never clean up the created directories. These accumulate in the system temp directory. Using pytest's `tmp_path` fixture would automatically clean up.
- **Fix suggestion:** Replace `tempfile.mkdtemp()` with `str(tmp_path / "subdir")` using the `tmp_path` fixture.

### RB-11: test_acquisition.py mock GP predict uses non-deterministic random
- **Severity:** LOW
- **File:** `tests/test_acquisition.py`, lines 373-374
- **Description:** The mock GP `predict_batch` uses `np.random.default_rng(0)` and `np.random.default_rng(1)` which creates new RNG instances on every call. Since these are seeded deterministically, this is actually fine for reproducibility. No bug here -- noted for completeness.

### RB-12: test_multistart.py test_subset_idx_restricts_voltages logic may be fragile
- **Severity:** MEDIUM
- **File:** `tests/test_multistart.py`, lines 345-395
- **Description:** The test asserts `result_subset.best_loss != result_full.best_loss` to verify that `subset_idx` has an effect. However, `config` uses `use_shallow_subset=True` while `config_full` uses `use_shallow_subset=False`, AND both pass the same `subset_idx`. The test comment says "full ignores subset_idx when use_shallow_subset=False" but this behavior depends on the implementation of `run_multistart_inference`. If the implementation always uses `subset_idx` regardless of `use_shallow_subset`, this test would fail spuriously. The test conflates two flags that may or may not interact.
- **Fix suggestion:** Either (a) test `subset_idx` vs `None` with the same `use_shallow_subset` setting, or (b) add a clarifying comment about the expected interaction between these two flags.

---

## Summary

| Category | Count |
|----------|-------|
| Verified Fixes | 6 |
| New Bugs from Fixes | 1 (LOW) |
| Remaining Bugs Missed in Round 1 | 12 (2 MEDIUM, 9 LOW, 1 INFO) |

**Overall assessment:** The 6 targeted fixes were all applied correctly. No HIGH or CRITICAL bugs were found. The two MEDIUM issues (RB-01, RB-02) involve fragile bare `from conftest import` statements that may break under certain pytest invocation patterns. The remaining issues are LOW severity cleanup/robustness items.
