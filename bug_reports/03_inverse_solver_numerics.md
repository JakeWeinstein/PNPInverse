# Bug Report: Inverse Solver Numerics

**Focus:** Numerical and scientific correctness in Inverse/ and Surrogate/ optimization code
**Agent:** Inverse Solver Numerics

---

## BUG 1: NaN predictions silently zeroed out in batch grid evaluation
**File:** `Surrogate/multistart.py:253-254`
**Severity:** HIGH
**Description:** The code zeroes out NaN residuals *before* summing:
```python
cd_diff = np.where(np.isnan(cd_diff), 0.0, cd_diff)
pc_diff = np.where(np.isnan(pc_diff), 0.0, pc_diff)
```
NaN-producing parameter combinations contribute zero to the objective instead of a penalty. Lines 262-264 then attempt to mark rows with NaN predictions as infinite, but the check uses `cd_pred`/`pc_pred` *before* the valid-mask subsetting, creating an inconsistency between the diff computation (which uses the valid subset) and the NaN check (which uses all columns).
**Suggested fix:** Remove the `np.where(np.isnan(...), 0.0, ...)` lines. Instead, compute the objective normally and use `np.where(np.isnan(objectives), np.inf, objectives)` as the final step.

## BUG 2: Diffusion objective passes concentration vector as phi placeholder
**File:** `Inverse/objectives.py:45-53`
**Severity:** MEDIUM
**Description:** For diffusion inference (`objective_fields=("concentration",)`), `phi_placeholder = c_targets[0]` is passed as `phi_target`. `build_reduced_functional` always creates `phi_target_f = _vector_to_function(ctx, phi_target, space_key="V_scalar")`, projecting the concentration vector onto the scalar FE space. If the concentration vector and phi DOF space differ in size, this will silently produce a wrong-sized vector or crash.
**Suggested fix:** Only create `phi_target_f` when `"phi"` is in `target.objective_fields`, or pass `None` and guard the creation.

## BUG 3: Robin kappa bounds format inconsistent with Dirichlet bounds
**File:** `Inverse/parameter_targets.py:208-209`
**Severity:** MEDIUM
**Description:** Robin kappa bounds are returned as `[[lows], [highs]]` (two lists), while Dirichlet phi0 bounds are `(1e-8, None)` (a single tuple). If the minimize function expects scipy-style `[(lo1, hi1), (lo2, hi2), ...]` for multi-control problems, the `[[lows], [highs]]` format would be misinterpreted.
**Suggested fix:** Verify the expected format and normalize to `[(1e-8, None) for _ in range(n_species)]` if scipy-style bounds are needed.

## BUG 4: `_polish_candidate` autograd path computes objective and gradient twice per L-BFGS-B step
**File:** `Surrogate/multistart.py:362-368`
**Severity:** MEDIUM (performance)
**Description:** The `_objective` and `_gradient` closures both call `_autograd_obj_and_grad(x)` independently. When `scipy.optimize.minimize` calls `fun` and then `jac` at the same point, the forward+backward pass runs twice, doubling the compute cost of the polish phase.
**Suggested fix:** Cache the last `(x, J, grad)` result and return the cached value when called at the same `x`.

## BUG 5: Dirichlet phi0 bounds prevent zero or negative potentials
**File:** `Inverse/parameter_targets.py:153-154`
**Severity:** LOW
**Description:** Default lower bound is `1e-8`, preventing the optimizer from reaching zero or negative potentials. For electrochemical systems where ground potential matters, this could prevent correct parameter recovery.
**Suggested fix:** Use `(None, None)` or a physically motivated range.

## BUG 6: `_reduce_guess_anisotropy` sign handling for mixed-sign parameters
**File:** `Inverse/inference_runner/recovery.py:337-354`
**Severity:** LOW
**Description:** For parameters with mixed signs, the geometric mean of absolute values assigns a positive magnitude, then multiplies by sign. The blend `(1-beta)*arr + beta*isotropic` could push positive values negative or vice versa. Fine for diffusion coefficients (always positive) but problematic for general parameter recovery.

## BUG 7: Silent recovery skip when solver_options is not a dict
**File:** `Inverse/inference_runner/recovery.py:56, 87, 110, 136`
**Severity:** LOW
**Description:** When `solver_params[10]` is not a dict, `baseline_solver_options` is set to `None`. This means `_relax_solver_options_for_attempt` returns early without any relaxation, and all retry attempts skip solver relaxation silently.
**Suggested fix:** Log a warning when solver_options is not a dict and recovery relaxation is skipped.

## BUG 8: `objective_and_gradient` double-counts eval counter on FD path
**File:** `Surrogate/objectives.py:247-251, 409-412, 636-638, 828-831`
**Severity:** LOW
**Description:** On the FD path, `objective_and_gradient()` calls `self.objective(x)` (increments `_n_evals`) then `self.gradient(x)` (calls `self.objective()` 2N more times). The initial `objective()` call is redundant; `n_evals` over-reports by 1 per call.

---

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 0     |
| HIGH     | 1     |
| MEDIUM   | 3     |
| LOW      | 4     |

**Most impactful:** Bug 1 -- NaN residuals silently zeroed in grid evaluation can cause the optimizer to select NaN-producing parameter combinations as optimal.
