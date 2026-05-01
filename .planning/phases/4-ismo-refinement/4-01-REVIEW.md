# 4-01 Plan Review: ISMO Core Loop & Configuration

**Reviewer:** Claude Opus 4.6 (automated plan quality check)
**Date:** 2026-03-17
**Verdict:** FLAG (minor issues -- addressable during execution, no blockers)

---

## Strengths

1. **Excellent goal clarity.** The ISMO loop is well-motivated, the 200-PDE budget is concrete, and the convergence criterion (surrogate-PDE gap at optimizer's best) is the natural diagnostic for this algorithm.

2. **Strong API consistency.** Frozen dataclasses (`ISMOConfig`, `ISMOIteration`, `ISMOResult`) match the project convention established by `MultiStartConfig`, `CascadeConfig`, etc. Return types use tuples for immutability. Log-space k0 is handled correctly throughout.

3. **Thorough design rationale.** Each design decision (wrap vs. replace, convergence before retraining, sequential PDE evaluation, fixed batch size) is explicitly justified with quantitative reasoning. This is significantly above average for plan quality.

4. **Clean dependency mapping.** The 4-01 -> 4-02/03/04/05 dependency chain is well-defined with clear interface boundaries (enum, function signatures, dataclass schemas).

5. **Immutable data flow.** The plan explicitly calls out that original training data is never modified and all arrays are copied/concatenated. This matches the codebase's immutability conventions.

6. **Testability.** All private helpers are pure functions with explicit inputs/outputs, making unit testing straightforward.

---

## Issues

### 1. MEDIUM: `run_cascade_inference` call signature mismatch

The plan's pseudocode (line ~396-401) calls `run_cascade_inference` with keyword arguments `initial_k0` and `initial_alpha`, plus `bounds_k0_1`, `bounds_k0_2`, `bounds_alpha`. This matches the actual signature in `cascade.py` (line 563-573). However, the plan does **not** pass `subset_idx` through to the cascade call, despite it being a parameter of `run_ismo()`.

Looking more carefully at line 401, it does pass `subset_idx=subset_idx`. This is correct.

**Status:** No fix needed on closer inspection.

### 2. MEDIUM: `_retrain_surrogate` references `nn_training.py` but uses inconsistent function name

The plan says: "Uses `train_nn_surrogate()` from `nn_training.py`". The actual function is indeed `train_nn_surrogate` in `Surrogate/nn_training.py` (confirmed). However, the plan describes "Clone the existing model's state dict" and "Create a new `NNTrainingConfig`" -- the warm-start mechanism for NN retraining is not natively supported by `train_nn_surrogate()`. The function creates a fresh `NNSurrogateModel` internally; it does not accept pre-loaded weights.

**Suggested fix:** The plan (or 4-04) needs to either:
- (a) Modify `train_nn_surrogate()` to accept an optional `initial_state_dict` parameter for warm-starting, or
- (b) Implement warm-start retraining directly in `_retrain_surrogate()` by calling `NNSurrogateModel.fit()` after loading weights, bypassing `train_nn_surrogate()`.

This is not a blocker because the plan explicitly notes "4-04 may refine the NN warm-start logic," but the 4-01 implementation will need a concrete approach. Option (b) is simpler for the initial implementation.

### 3. MEDIUM: `stall_count` variable is not initialized before the loop

In the pseudocode (line ~462-468), `stall_count` is incremented in the `else` branch but is only initialized to 0 inside the `if` branch. If the very first iteration does not improve `best_pde_loss_global` (which starts at `inf`, so this cannot actually happen), `stall_count` would be undefined. In practice, `inf` guarantees the first iteration always enters the `if` branch, so this is safe -- but for defensive coding, `stall_count` should be initialized to 0 before the loop.

**Suggested fix:** Add `stall_count = 0` alongside `best_pde_loss_global = inf` before the loop.

### 4. LOW: `secondary_weight` parameter missing from `_evaluate_candidates_pde` dispatch

The plan's signature for `_evaluate_candidates_pde` (Task 6) includes `secondary_weight`, but the pseudocode in Task 10 calls it with `...` placeholder. The PDE loss computation inside this function needs to use the same objective formula as the surrogate (0.5 * sum(cd_diff^2) + w * 0.5 * sum(pc_diff^2)). The plan states this correctly in the description but the pseudocode doesn't show how `secondary_weight` is determined.

**Suggested fix:** Use `config.multistart_config.secondary_weight` (defaulting to 1.0) for consistency with the surrogate objective.

### 5. LOW: `bounds` parameter in `_acquire_optimizer_trajectory` is a `dict` but used inconsistently

Task 5a accepts `bounds: dict`, but the existing codebase uses `bounds_k0_1: Tuple[float, float]` etc. as separate arguments (see `run_multistart_inference`, `run_cascade_inference`). The plan should clarify the dict structure (e.g., `{"k0_1": (lo, hi), "k0_2": (lo, hi), "alpha": (lo, hi)}`) or use the same separate-tuple convention.

**Suggested fix:** Use separate `bounds_k0_1`, `bounds_k0_2`, `bounds_alpha` tuples to match the existing API pattern.

### 6. LOW: `ISMOConfig.surrogate_type` string enum should reference existing model loaders

The plan lists `surrogate_type` values including `"gp"` and `"pce"`, but `_retrain_surrogate` does not describe how to refit PCE or GP models in sufficient detail. For PCE, the plan says "Full refit -- PCE is fast" without specifying the API call. Similarly for GP, "Full refit -- GPyTorch hyperparameter optimization from current hyperparams" is vague.

**Suggested fix:** Add brief notes referencing the actual methods: `PCESurrogateModel.fit()`, `GPSurrogateModel.fit()`. Not blocking because these can be looked up during implementation.

### 7. LOW: Convergence criterion may be too strict for noisy PDE targets

The plan uses `convergence_rtol=0.05` meaning the surrogate must agree with PDE to within 5% at the optimizer's best point. When targets are generated from noisy experimental data (not synthetic), the PDE loss at the true parameters is already nonzero, and 5% relative tolerance on the gap may be overly strict -- the surrogate-PDE gap could oscillate rather than converge monotonically.

**Suggested fix:** Consider adding a note that `convergence_rtol` should be tuned based on the noise level in the target data. The default of 0.05 is appropriate for synthetic parameter recovery but may need adjustment for experimental data.

### 8. LOW: Success criterion 4 ("convergence metric decreases monotonically") is unrealistic

Success criterion 4 states: "the convergence metric decreases monotonically across iterations." ISMO does not guarantee monotonic decrease of the surrogate-PDE gap -- retraining can temporarily worsen agreement at the current optimum if the new data shifts the surrogate's loss landscape. A more realistic criterion would be "the convergence metric trends downward and reaches below `convergence_rtol` within the budget."

**Suggested fix:** Weaken to "convergence metric reaches below 0.1 within 3 iterations on the synthetic test case" or "decreases overall (not necessarily monotonically)."

---

## Missing Items

1. **Logging to disk.** The plan saves augmented training data and retrained models but does not describe saving the `ISMOResult` or iteration history to a JSON/NPZ file for post-hoc analysis. The 4-05 diagnostics plan will need this. Consider adding a `_save_ismo_result()` helper that serializes the result to `{output_dir}/ismo_result.json`.

2. **Error handling for PDE solver failures.** Task 6 returns `converged_flags` but the plan doesn't describe what happens if ALL candidates fail to converge. This is unlikely but possible with extreme parameters. The `_augment_training_data` function filters to converged-only, which could result in zero new samples -- the loop should handle this gracefully (skip retraining, log a warning).

3. **Random seed management.** The acquisition functions accept a `seed` parameter, but the plan doesn't describe how seeds are managed across iterations. Using a fixed seed would produce identical LHS fill-in samples every iteration; using `seed + iteration` would be better.

4. **`observable_scale` parameter.** The PDE evaluation helper needs `observable_scale` (from `generate_training_data_single`), but this is buried inside `pde_solver_kwargs`. The plan should explicitly list what `pde_solver_kwargs` must contain.

---

## Integration Risks

1. **`_retrain_surrogate` for NN ensemble is the highest-risk integration point.** The plan needs to: (a) retrain each of 5 members, (b) rebuild `EnsembleMeanWrapper`, and (c) ensure the new wrapper works with `run_multistart_inference` / `run_cascade_inference`. Since `EnsembleMeanWrapper.__init__` validates that all members share the same `phi_applied` grid, and retraining preserves this, the risk is low -- but the warm-start path through `NNSurrogateModel.fit()` is not trivially compatible with loading existing weights (see Issue 2).

2. **`run_ismo()` parameter explosion.** The function takes 12+ arguments. Consider grouping PDE-related args into a frozen `PDESolverConfig` dataclass to reduce the signature. This is a style issue, not a blocker.

3. **Training data shape consistency.** When augmenting training data, the new PDE curves must have the same `n_eta` as the existing training data. The plan assumes `phi_applied` is shared, which is correct as long as `_evaluate_candidates_pde` uses the same `phi_applied` grid. This should be validated at runtime.

---

## Summary

The plan is well-structured, thoroughly documented, and correctly interfaces with the existing codebase. The main risks are (1) the NN warm-start retraining path, which is deferred to 4-04 but needs at least a stub implementation in 4-01, and (2) some minor pseudocode gaps that should be filled during implementation. No blocking issues found.
