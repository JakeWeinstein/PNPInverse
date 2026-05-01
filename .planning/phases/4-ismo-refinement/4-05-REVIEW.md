# 4-05 Plan Review: Convergence Monitoring, Diagnostics & ISMO Runner Script

**Verdict: FLAG**

---

## Strengths

1. **Well-structured convergence criteria.** The tiered approach (primary: surrogate-PDE agreement, secondary: parameter stability, tertiary: test error degradation) with explicit tolerances is sound. The `min_iterations` guard prevents premature convergence.

2. **Excellent design rationale sections.** The justifications for point-wise agreement over global error, for separating the convergence checker into its own class, and for not using ensemble uncertainty as a convergence signal are all well-argued and correct.

3. **Frozen dataclasses throughout.** Consistent with the codebase's immutability pattern (`MultiStartResult`, `CascadeResult`, etc.).

4. **Per-iteration NPZ state enables resumability.** This is the right approach for a loop that may take hours and involve expensive PDE solves.

5. **Budget accounting is explicit.** The `remaining_budget()` / `is_budget_exhausted()` API makes it hard to accidentally exceed the 200-eval cap.

6. **Post-ISMO validation is well-specified.** Comparing against Phase 3 baseline with the same seeds/noise levels makes the comparison rigorous.

7. **Fallback stubs for missing 4-01 through 4-04 modules.** Good for independent development.

---

## Issues

### Issue 1 -- Off-by-one in budget accounting (Severity: HIGH)

In the runner pseudocode (Step 3c-3d), the agreement eval PDE cost is subtracted *after* acquisition returns candidates:

```python
budget_left = checker.remaining_budget() - 1  # subtract agreement eval
new_params = new_params[:budget_left]
```

But `checker.remaining_budget()` has not yet been updated for the current iteration's agreement eval (the `record_iteration` call happens in Step 3h, after the PDE batch). This means:

- Iteration 0: `remaining_budget()` returns 200, `budget_left = 199`, correct.
- Iteration 1: `remaining_budget()` returns `200 - (n_iter0_new + 1)`. This is correct *if* the record was appended in 3h. But the budget check in `check_convergence()` (Step 3i) uses `n_total_pde_evals` from the *just-recorded* iteration, so it does account for the agreement eval correctly.

The real bug: `new_params[:budget_left]` can still overshoot because `budget_left` is computed *before* the PDE batch runs, but if `acquire_new_samples` returns exactly `budget_left` points, the total for this iteration is `budget_left + 1` (the agreement eval), potentially exceeding the cap by 1 on the final iteration.

**Fix:** Compute `budget_for_acquisition = checker.remaining_budget() - 1` (accounting for the agreement eval that already happened at step 3b), then `new_params = new_params[:max(0, budget_for_acquisition)]`. The current code does this, but the comment is misleading. More importantly, if the agreement eval already consumed the last budget unit, `budget_for_acquisition` could be 0 or negative, and the loop should break before step 3c. Add an explicit check: `if budget_for_acquisition <= 0: break`.

### Issue 2 -- Stagnation detection window is too short (Severity: MEDIUM)

The plan says "last 3 iterations show less than 5% relative improvement in `surrogate_pde_agreement`." With `min_iterations = 2`, stagnation could trigger at iteration 4 (iterations 2, 3, 4 form the window). For a method that may need 5-8 iterations to converge (especially with trust-region acquisition that needs several rounds to explore), a 3-iteration window is aggressive.

**Fix:** Either increase the stagnation window to 4 iterations, or make it configurable via `ISMOConvergenceCriteria.stagnation_window: int = 3`.

### Issue 3 -- `objective_improvement` ratio undefined for iteration 0 (Severity: MEDIUM)

`ISMOIterationRecord.objective_improvement` is defined as `prev_pde_obj / current_pde_obj`. On iteration 0, there is no previous PDE objective. The plan does not specify a sentinel value.

**Fix:** Document that `objective_improvement = float('nan')` for iteration 0, or use `1.0` as a neutral value.

### Issue 4 -- `validate_surrogate` metric name mismatch (Severity: MEDIUM)

The plan maps `surrogate_test_nrmse_cd` to `test_metrics["cd_mean_relative_error"]`. While this works, it is misleading: `cd_mean_relative_error` from `validation.py` is a mean of per-sample NRMSEs across the test set, not a single NRMSE value. The field name `surrogate_test_nrmse_cd` implies a single aggregate NRMSE.

**Fix:** Either rename the field to `surrogate_test_mean_nrmse_cd` for clarity, or add a comment explaining the metric definition.

### Issue 5 -- Test set not loaded in runner pseudocode (Severity: LOW)

Step 3g calls `validate_surrogate(surrogate, test_params, test_cd, test_pc)` but the runner pseudocode never loads these from the split indices. The training data file has `training_data_merged.npz` and separate `split_indices.npz` for train/test splits.

**Fix:** Add explicit loading of test data from `split_indices.npz` in Step 0.

### Issue 6 -- `ISMOResult.final_params` is `np.ndarray` in a frozen dataclass (Severity: LOW)

Frozen dataclasses do not deep-freeze mutable fields. An `np.ndarray` can still be mutated in place after construction (`result.final_params[0] = 999` will succeed). This breaks the immutability contract.

**Fix:** Either store as a tuple of floats (consistent with `MultiStartResult` which stores individual `best_k0_1`, etc.), or document that the array should be treated as read-only. Using `final_params: tuple` is more consistent with the codebase pattern.

### Issue 7 -- `acquisition_details: dict` in frozen dataclass (Severity: LOW)

Same issue as Issue 6. A mutable `dict` inside a frozen dataclass breaks immutability. `field(default_factory=dict)` is mentioned in the comment but doesn't freeze the dict.

**Fix:** Use `MappingProxyType` or `tuple` of key-value pairs instead, or accept the limitation and document it.

### Issue 8 -- No signal/interrupt handling for graceful shutdown (Severity: LOW)

The plan mentions resumability via per-iteration NPZ files, but the runner has no `SIGINT`/`SIGTERM` handler. If killed mid-PDE-batch, the current iteration's data is lost and the augmented training data is not saved.

**Fix:** Add a `try/finally` block around the ISMO loop that saves partial state on interruption, or register a signal handler that sets a "stop after current iteration" flag.

---

## Missing Items

1. **No `target_cd` / `target_pc` loading in runner.** The pseudocode references `target_cd` and `target_pc` at step 3a but never loads them. These must come from synthetic PDE data at the true parameters (for the recovery test) or from experimental data. Specify the source.

2. **No `--resume` CLI flag.** The plan discusses resumability as a design benefit of per-iteration NPZ files but does not expose a `--resume` flag or implement the logic to reload from the last completed iteration.

3. **No handling of PDE solver failures.** `generate_training_data_single` can fail (non-convergence, NaN). The runner pseudocode does not handle partial batch failures -- if 3 of 20 PDE solves fail, what happens? The acquisition budget accounting must handle this.

4. **No train/test split update logic.** When the training set grows via ISMO, the held-out test set stays the same. This is fine but should be stated explicitly. If the test set overlaps with newly acquired points (unlikely but possible with LHS), the post-retrain validation could be biased.

5. **`compute_parameter_stability` normalization formula is wrong.** The normalization divides `log10(k0_1)` by the log-range, but this does not center on [0, 1]. It should be `(log10(k0_1) - log10(k0_1_lo)) / (log10(k0_1_hi) - log10(k0_1_lo))`. The current formula gives values outside [0, 1] for most inputs.

---

## Integration Risks

1. **4-01 interface ambiguity.** The plan says "if 4-01 defines `ISMOLoop.run()`, 4-05's runner calls it; if 4-01 only provides primitives, the runner orchestrates them directly." This dual-path creates ambiguity. Looking at 4-01, it defines `ISMOConfig` and `run_ismo()` as the main entry point with an `ISMOIteration` dataclass, while 4-05 defines its own `ISMOIterationRecord` and `ISMOResult` -- these overlap with 4-01's `ISMOIteration` and `ISMOResult`. There is a high risk of naming collisions and duplicated logic.

    **Recommendation:** Decide now: either 4-05's runner wraps 4-01's `run_ismo()` (and convergence checking lives inside 4-01), or 4-01 provides only primitives and 4-05 does all orchestration. The current plan tries to do both.

2. **4-02 function signature mismatch.** The plan calls `acquire_new_samples(surrogate, best_params, checker.remaining_budget(), strategy=..., n_samples=...)`. But 4-02 defines `select_new_samples(config, surrogate, optimizer_result, existing_params, ...)` with an `AcquisitionConfig` object and an `optimizer_result` (not bare `best_params`). The adapter layer is not specified.

3. **4-03 return type.** The plan calls `retrain_surrogate(surrogate, params_train, cd_train, pc_train, phi_applied)` and expects a new surrogate back. 4-03 defines a unified `retrain_surrogate()` that returns a new model, which matches. But 4-03 also does quality checking internally -- if it falls back to from-scratch training, this could take a long time and affect the timing metrics in `ISMOIterationRecord`.

4. **Surrogate type heterogeneity.** The runner supports `nn_ensemble`, `pod_rbf_log`, `rbf_baseline`, and `gp`, but the save logic uses `save_surrogate()` which only handles RBF/POD-RBF models. NN ensembles and GPs have different save paths. The post-ISMO save step needs type-aware dispatch.
