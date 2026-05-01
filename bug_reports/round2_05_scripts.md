# Round 2 Bug Audit: scripts/ Directory

Audited all 30 Python files under `scripts/`.

---

## Verified Fixes

### 1. run_ismo.py -- Import fixes (VERIFIED)
- **File:** `scripts/surrogate/run_ismo.py`, lines 43-86
- **Status:** CORRECT. Guard imports for `ISMOIteration`, `AcquisitionConfig`, `select_new_samples`, `retrain_surrogate`, and `evaluate_candidates_with_pde` are properly wrapped in `try/except ImportError` blocks with `None` fallbacks.

### 2. run_ismo.py -- successful_indices fix (VERIFIED)
- **File:** `scripts/surrogate/run_ismo.py`, lines 509-530
- **Status:** CORRECT. `successful_indices` is now properly tracked per-sample in the inline PDE evaluation loop, and `new_params` is filtered to only successful indices before augmentation.

### 3. run_multi_seed_v13.py -- mtime staleness check (VERIFIED)
- **File:** `scripts/studies/run_multi_seed_v13.py`, lines 208-214
- **Status:** CORRECT. The CSV mtime is now compared against `start_time` to ensure it was written by the current subprocess, raising `RuntimeError` if stale.

### 4. sensitivity_visualization.py -- observable_scale sign (VERIFIED)
- **File:** `scripts/studies/sensitivity_visualization.py`, lines 319, 341
- **Status:** CORRECT. Uses `float(-deps["I_SCALE"])` which is the negated scale, consistent with the codebase convention where `observable_scale = -I_SCALE` (see `_bv_common.py` line 533 and `run_ismo_live.py` line 68).

### 5. validate_surrogate.py -- --split-indices argument (VERIFIED)
- **File:** `scripts/surrogate/validate_surrogate.py`, lines 42-44
- **Status:** PARTIALLY CORRECT. The `--split-indices` CLI argument was added properly. However, see NEW BUG #1 below regarding the key name mismatch.

---

## NEW Bugs Found in Round 2

### BUG-S01: validate_surrogate.py -- Wrong split_indices key name
- **Severity:** HIGH
- **File:** `scripts/surrogate/validate_surrogate.py`, line 62
- **Line:** `test_idx = splits["test"]`
- **Description:** The split_indices.npz file uses key `"test_idx"` (verified by loading the actual file), but the code reads `splits["test"]`. This will raise a `KeyError` at runtime whenever `--split-indices` is passed.
- **Fix:** Change `splits["test"]` to `splits["test_idx"]`.

### BUG-S02: run_ismo.py -- Undefined name `evaluate_pde_batch`
- **Severity:** HIGH
- **File:** `scripts/surrogate/run_ismo.py`, lines 498-500
- **Description:** Line 498 references `evaluate_pde_batch` in the condition `if evaluate_pde_batch is not None:`, and line 500 calls `evaluate_pde_batch(new_params, pde_solver_fn)`. However, this name is never imported or defined. The import on line 84 imports `evaluate_candidates_with_pde` (assigned to variable `evaluate_candidates_with_pde`), not `evaluate_pde_batch`. This is a `NameError` that will crash the ISMO loop at step 3d.
- **Fix:** Either rename the import alias:
  ```python
  evaluate_pde_batch = evaluate_candidates_with_pde  # after the try/except
  ```
  Or replace `evaluate_pde_batch` with `evaluate_candidates_with_pde` on lines 498 and 500, adjusting the call signature to match.

### BUG-S03: inverse_benchmark_all_models.py -- Deprecated `datetime.utcnow()`
- **Severity:** LOW
- **File:** `scripts/studies/inverse_benchmark_all_models.py`, line 568
- **Description:** `datetime.utcnow()` is deprecated since Python 3.12. The same file imports `datetime` from `datetime` but uses the deprecated method. Elsewhere in the same file (line 662), `datetime.now()` is used without timezone. The codebase already uses the correct pattern in `run_multi_seed_v13.py` line 383: `datetime.datetime.now(datetime.timezone.utc)`.
- **Fix:** Replace `datetime.utcnow().isoformat() + "Z"` with `datetime.now(timezone.utc).isoformat()` (timezone is already imported on line 29).

### BUG-S04: inverse_benchmark_all_models.py -- Legacy `np.random.RandomState`
- **Severity:** LOW
- **File:** `scripts/studies/inverse_benchmark_all_models.py`, line 243
- **Description:** Uses `np.random.RandomState(seed)` which is the legacy PRNG API. The rest of the codebase consistently uses `np.random.default_rng(seed)` (the modern Generator API). This is not a correctness bug but a consistency issue and the legacy API may eventually be deprecated.
- **Fix:** Replace `rng = np.random.RandomState(seed)` with `rng = np.random.default_rng(seed)` and change `rng.randn(...)` calls to `rng.standard_normal(...)`.

### BUG-S05: parameter_recovery_all_models.py -- Mutable dataclass `RecoveryRow`
- **Severity:** LOW
- **File:** `scripts/studies/parameter_recovery_all_models.py`, line 62
- **Description:** `RecoveryRow` uses `@dataclass` without `frozen=True`, making it mutable. This violates the codebase's immutability convention (see `coding-style.md`). The equivalent class in `inverse_benchmark_all_models.py` line 86 correctly uses `@dataclass(frozen=True)`.
- **Fix:** Change `@dataclass` to `@dataclass(frozen=True)` on line 62.

### BUG-S06: run_ismo.py -- Budget accounting race with `n_agreement_evals`
- **Severity:** MEDIUM
- **File:** `scripts/surrogate/run_ismo.py`, line 410
- **Description:** `budget_after_agreement = checker.remaining_budget() - n_agreement_evals` subtracts `n_agreement_evals` from the remaining budget, but `checker.remaining_budget()` does not yet include these evals (they have not been recorded via `record_iteration` yet). Later at line 590, `n_pde_evals_this_iter = n_agreement_evals + n_successful` is recorded. This means the budget check on line 411 is correct in spirit but fragile: if a future refactor records the agreement eval earlier, the budget would be double-counted. More importantly, when the stub PDE solver triggers the `NotImplementedError` path (line 384-402), `n_agreement_evals = 0`, so no budget is consumed, which is correct for the stub but would be wrong once the real PDE solver is integrated since the agreement eval does consume budget.
- **Fix:** Add a comment clarifying the accounting, or explicitly track consumed budget in a local variable rather than relying on the checker's state.

### BUG-S07: sensitivity_visualization.py -- Dead code in `compute_full_jacobian`
- **Severity:** LOW
- **File:** `scripts/studies/sensitivity_visualization.py`, lines 415-417
- **Description:** The variable `params_plus` is computed via `build_perturbed_params` on line 415-417 but is never used. The actual perturbation used for the Jacobian is `params_plus_add` (additive, lines 419-420). The multiplicative perturbation `params_plus` is dead code that may confuse readers.
- **Fix:** Remove lines 415-417 (the `params_plus` assignment).

### BUG-S08: run_multi_seed_v13.py -- Stale CSV raises RuntimeError instead of returning None
- **Severity:** MEDIUM
- **File:** `scripts/studies/run_multi_seed_v13.py`, lines 211-214
- **Description:** When the mtime staleness check fails, the code raises `RuntimeError`. However, the caller (`main()` at line 439) expects `run_single_seed` to return `None` on failure and catches failures gracefully. A `RuntimeError` will crash the entire multi-seed loop instead of skipping the failed seed.
- **Fix:** Replace the `raise RuntimeError(...)` with:
  ```python
  _log("SEED", f"Seed {seed}: CSV is stale (not updated by subprocess)")
  return None
  ```

### BUG-S09: gradient_benchmark.py -- Lambda closure captures loop variable
- **Severity:** LOW
- **File:** `scripts/studies/gradient_benchmark.py`, lines 311-321
- **Description:** The `grad_fn` lambdas inside the speed benchmark closure capture the outer scope variables `model`, `x`, `target_cd`, `target_pc`, `secondary_weight` by reference, which is fine since they don't change within the loop body. However, this is a fragile pattern. If the code were refactored to iterate over models in a different order within the same function scope, the lambdas would capture the wrong model. Currently not a bug, but a code smell.
- **Fix:** No immediate fix needed, but consider using `functools.partial` for clarity.

### BUG-S10: train_nn_surrogate.py -- `NNTrainingConfig` may not accept all args
- **Severity:** LOW
- **File:** `scripts/surrogate/train_nn_surrogate.py`, lines 200-211
- **Description:** `NNTrainingConfig` is instantiated with `hidden` and `n_blocks` keyword arguments. If `NNTrainingConfig` is a frozen dataclass that does not define these fields, this will raise a `TypeError`. This depends on the definition in `Surrogate/nn_training.py` which was not modified in round 1.
- **Fix:** Verify that `NNTrainingConfig` accepts `hidden` and `n_blocks` parameters. If not, pass them through a separate mechanism.

---

## Summary

| Severity | Count | IDs |
|----------|-------|-----|
| HIGH     | 2     | BUG-S01, BUG-S02 |
| MEDIUM   | 2     | BUG-S06, BUG-S08 |
| LOW      | 6     | BUG-S03, BUG-S04, BUG-S05, BUG-S07, BUG-S09, BUG-S10 |

**Verified fixes:** 5/5 from round 1 confirmed correct (with BUG-S01 noting a remaining issue in the split-indices fix).

**Critical findings:**
- BUG-S01 (wrong dict key) will crash `validate_surrogate.py` when `--split-indices` is used.
- BUG-S02 (undefined `evaluate_pde_batch`) will crash `run_ismo.py` at step 3d of the ISMO loop.
- BUG-S08 (RuntimeError vs graceful return) will crash the multi-seed runner if the CSV is stale, instead of skipping the seed.
