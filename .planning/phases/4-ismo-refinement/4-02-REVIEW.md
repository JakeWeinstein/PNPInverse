# 4-02 Plan Review: Acquisition Strategy & Sample Selection

**Reviewer**: Plan Quality Checker
**Date**: 2026-03-17
**Verdict**: FLAG

---

## Strengths

1. **Clear goal and well-scoped responsibility**. The module has a single entry point (`select_new_samples`) with a well-defined contract: takes existing data + optimizer results + optional GP, returns new sample points. Clean separation from 4-01 (orchestrator) and 4-03 (data augmentation).

2. **Consistent with codebase patterns**. Frozen dataclasses for config and result match `MultiStartConfig`/`MultiStartResult` and `CascadeConfig`/`CascadeResult` exactly. Physical-space output matches PDE solver and training data conventions.

3. **Normalized log-space distance is correct**. The plan correctly identifies that raw Euclidean distance would be dominated by k0 (spanning 6+ orders of magnitude vs alpha in [0.1, 0.9]). Normalizing to [0,1]^4 with log10 on k0 dims is the right approach.

4. **GP fallback is well-specified**. The redistribution logic (50% to optimizer, 50% to space-filling) is concrete and easy to implement. This ensures ISMO can start without a GP.

5. **Thorough test plan**. 12 test cases covering roundtrips, de-duplication, fallback, budget allocation, and batch diversity.

6. **Good design rationale sections**. The "Key Design Decisions" section explains *why* greedy over DPP, variance over EI, etc.

---

## Issues

### Issue 1: `_acquire_optimizer_trajectory` extracts params from wrong API (MEDIUM)

The plan says to extract `(k0_1, k0_2, alpha_1, alpha_2)` from `MultiStartResult.candidates` and `CascadeResult.pass_results`. However:

- `MultiStartCandidate` stores individual attributes (`c.k0_1`, `c.k0_2`, `c.alpha_1`, `c.alpha_2`), not a tuple. The plan needs to specify: `np.array([c.k0_1, c.k0_2, c.alpha_1, c.alpha_2])` for each candidate.
- `CascadePassResult` has the same attribute-based API (`pr.k0_1`, etc.), not a single "endpoint" field.
- `CascadeResult` does not directly expose endpoints per pass in array form. You need to iterate `cascade_result.pass_results` and extract the 4 attributes.

**Fix**: Add explicit code showing how to extract the (N, 4) array from each result type. Example:
```python
pts = np.array([[c.k0_1, c.k0_2, c.alpha_1, c.alpha_2] for c in multistart_result.candidates])
```

### Issue 2: `_acquire_uncertainty` calls non-existent method signature (MEDIUM)

The plan calls `gp_model.predict_batch_with_uncertainty(candidates)` where `candidates` is an `(N, 4)` physical-space array. This is correct -- `GPSurrogateModel.predict_batch_with_uncertainty` accepts `(M, 4)` in physical space and returns dict with `current_density_std` and `peroxide_current_std` of shape `(M, n_eta)`.

However, the plan's step 4 references `sigma_cd = mean of current_density_std across voltage points`. The return dict key is `'current_density_std'`, not `'current_density_sigma'`. This is fine as stated, but worth noting that `current_density_std` is already in physical output units (the GP model handles the inverse Z-score transform internally). No fix needed, just confirming compatibility.

### Issue 3: `ParameterBounds` has separate alpha ranges, plan treats them as shared (LOW)

`ParameterBounds` has `alpha_1_range` and `alpha_2_range` as independent fields. The plan's `_to_normalized_log` function says "Dimensions 2,3 (alpha_1, alpha_2): min-max scale to [0,1]" but doesn't specify whether it uses `alpha_1_range` for dim 2 and `alpha_2_range` for dim 3, or uses a single shared range. The existing `ParameterBounds` defaults have them equal (`(0.1, 0.9)` for both), but they are structurally independent.

**Fix**: Clarify that normalization uses `bounds.alpha_1_range` for dimension 2 and `bounds.alpha_2_range` for dimension 3.

### Issue 4: `_acquire_spacefill` uses `generate_lhs_samples` which takes `seed: int`, not `rng: Generator` (LOW)

The plan uses `seed: int` for space-filling (matching the existing API), which is correct. However, the optimizer trajectory function takes `rng: np.random.Generator`. This means two seeding mechanisms coexist. For reproducibility across ISMO iterations, the plan should note that `spacefill_seed` must be varied per iteration (otherwise you get the same LHS points every time).

**Fix**: Add a note that the ISMO orchestrator (4-01) should pass `spacefill_seed = config.spacefill_seed + iteration` or similar to ensure different LHS designs across iterations.

### Issue 5: Neighborhood ball sampling in log-space -- back-conversion not specified (LOW)

Task 5 says "sample uniformly in a ball of radius `neighborhood_radius` in normalized log-space, then converting back to physical space. Clamp to bounds." The back-conversion from normalized [0,1]^4 to physical space requires the inverse of `_to_normalized_log`. This inverse function is not listed in the task breakdown.

**Fix**: Either add an explicit `_from_normalized_log(params_norm, bounds) -> params_physical` utility, or note that the inverse is trivial (denormalize dims 0,1 to log-space bounds, exponentiate; denormalize dims 2,3 to alpha bounds).

### Issue 6: Fraction validation missing (LOW)

`AcquisitionConfig` states "Fractional allocation (must sum to 1.0)" but no validation is shown. If `frac_optimizer + frac_uncertainty + frac_spacefill != 1.0`, the budget arithmetic silently misbehaves.

**Fix**: Add a `__post_init__` validation (or note in the plan that one is needed) that asserts `abs(frac_optimizer + frac_uncertainty + frac_spacefill - 1.0) < 1e-6`.

### Issue 7: k0_2 sensitivity weighting is a no-op in v1 (INFORMATIONAL)

The plan acknowledges `k0_2_sensitivity = 1.0 (uniform weight) as first pass`. This makes the `k0_2_sensitivity_weight` config parameter and the weighting code dead logic in the initial implementation. This is fine for a first pass but should be flagged for follow-up.

---

## Missing Items

1. **No iteration-awareness in seed management**. The plan does not address how seeds evolve across ISMO iterations. If `select_new_samples` is called 5 times with the same config, the space-filling LHS will produce identical points each time (before de-duplication removes them). The `seed` field should incorporate the iteration number.

2. **No handling of the case where both `multistart_result` and `cascade_result` are None**. If neither optimizer has been run (e.g., first ISMO iteration before any optimization), the optimizer-trajectory strategy produces zero points. The plan should explicitly state that the optimizer fraction is redistributed to space-filling in this case, similar to the GP fallback.

3. **No logging of per-strategy yield after de-duplication**. The `AcquisitionResult` tracks total `n_rejected_dedup` but not per-strategy rejection counts. For diagnosing whether one strategy is producing mostly redundant points, per-strategy counts would be valuable.

4. **No mention of how `existing_data` is constructed by the caller**. Is it just the parameter columns from the training NPZ? The plan should confirm the expected shape and format (physical space, 4 columns, no header).

---

## Integration Risks

1. **`MultiStartResult.candidates` may have been polished into the same basin**. If all 20 polished candidates converge to the same point, the optimizer-trajectory strategy generates 20 near-identical endpoints plus their neighborhoods. The de-duplication will reject most, but the optimizer fraction of the budget may yield very few unique points. This is handled gracefully (de-dup removes them, space-fill backfills), but the user should expect the optimizer fraction to underperform when the surrogate landscape has a single dominant basin.

2. **GP `predict_batch_with_uncertainty` can be slow for 5000 candidates**. Each call evaluates 44 independent GPs. For 5000 candidates this involves 44 Cholesky solves. With N_train ~ 200, each solve is O(200^2 * 5000) -- feasible but not instant. If N_train grows to 500+, this could become a bottleneck. Consider mentioning a fallback to reduce `n_uncertainty_candidates` if the GP is slow.

3. **`ParameterBounds` is not frozen**. It is a regular `@dataclass`, not `@dataclass(frozen=True)`. This means it can be mutated after construction. The plan's `AcquisitionConfig` is frozen, but `ParameterBounds` passed to `select_new_samples` is mutable. This is a pre-existing design choice in the codebase, not introduced by this plan, but worth noting.

---

## Summary

The plan is well-structured, algorithmically sound, and compatible with the existing codebase API. The issues are all LOW to MEDIUM severity -- no blockers. The two MEDIUM issues (extracting params from optimizer results, and seed management across iterations) should be addressed before implementation to avoid bugs. The missing item about handling the "no optimizer results" case is the most important gap to fill.

**Verdict: FLAG** -- address the MEDIUM issues and the "no optimizer results" fallback before proceeding to implementation.
