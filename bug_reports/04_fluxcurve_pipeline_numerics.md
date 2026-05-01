# Bug Report: FluxCurve Pipeline Numerics

**Focus:** Numerical and scientific correctness in FluxCurve/ package
**Agent:** FluxCurve Pipeline Numerics

---

## BUG 1: R-space target assembled over volume -- optimizer converges to wrong parameters
**File:** `FluxCurve/observables.py:101-113`, `FluxCurve/point_solve.py:294-299`, `FluxCurve/bv_point_solve/forward.py:276-282`, `FluxCurve/bv_point_solve/__init__.py:653-661`, `FluxCurve/bv_parallel.py:241-247`
**Severity:** CRITICAL
**Description:** The target flux is placed in an R-space (Real) Function, then assembled as `fd.assemble(target_ctrl * fd.dx(domain=mesh))`. In Firedrake, R-space has an unnormalized basis function phi_0 = 1, so this integral evaluates to `target_flux * Volume(mesh)`. Meanwhile, `simulated_flux_scalar = fd.assemble(observable_form)` is a boundary integral (a scalar). The point objective becomes `0.5 * (boundary_flux - target * Volume)^2` instead of `0.5 * (boundary_flux - target)^2`. **If the mesh volume != 1, the optimizer converges to wrong parameters.** This affects BOTH the Robin and BV pipelines.
**Suggested fix:** Replace `fd.assemble(target_ctrl * dx)` with a direct `fd.Constant(target_flux)` comparison, or use `fd.assemble(target_ctrl * ds(electrode_marker))` (boundary measure).

## BUG 2: Failed points contribute zero gradient but huge penalty objective
**File:** `FluxCurve/curve_eval.py:61-67`, `FluxCurve/point_solve.py:328-340`
**Severity:** HIGH
**Description:** When a point fails to converge, `point_solve.py` returns `objective=fail_penalty` (1e9) with `gradient=np.zeros(n_species)`. The penalty is added to `total_objective`, but the zero gradient is NOT added to `total_gradient`. L-BFGS-B sees high loss with small gradient and may take very large steps or stall.
**Suggested fix:** Return a finite-difference penalty gradient, or use a smooth barrier with analytic gradient.

## BUG 3: `evaluate_curve_loss_forward` ignores observable_mode
**File:** `FluxCurve/curve_eval.py:117-144`
**Severity:** HIGH
**Description:** Does not accept or pass `observable_mode` or `observable_species_index`. Uses default total_species flux. Used in `run.py:659` as final verification, meaning the reported loss may be computed from a different observable than what was optimized.
**Suggested fix:** Pass observable_mode through to the forward solver.

## BUG 4: BV target generation negates current_density_scale -- sign inconsistency
**File:** `FluxCurve/bv_run/io.py:76`
**Severity:** HIGH
**Description:** Target generation uses `i_scale=-current_density_scale` while inference uses positive `current_density_scale`. Relies on undocumented sign convention.
**Suggested fix:** Use the same observable form builder for both target generation and inference.

## BUG 5: Module-level cache has no mesh/config key
**File:** `FluxCurve/bv_point_solve/cache.py:17-18`
**Severity:** HIGH
**Description:** `_all_points_cache` persists across function calls. Not cleared when mesh, observable mode, or solver params change. Can serve stale cached solutions with wrong DOF count.
**Suggested fix:** Include mesh identity (DOF count) and observable mode in cache key, or clear on config change.

## BUG 6: Gradient array silently truncated/zero-padded on component mismatch
**File:** `FluxCurve/observables.py:116-130`
**Severity:** MEDIUM
**Description:** If adjoint returns more or fewer gradient components than `n_species`, the function silently truncates or zero-pads, masking configuration errors.
**Suggested fix:** Add warning or error when `len(grads) != n_species`.

## BUG 7: Replay mode tape replay does not re-converge
**File:** `FluxCurve/replay.py:269-273`
**Severity:** MEDIUM (already disabled)
**Description:** Replayed states are not at steady state for new kappa values. Already disabled with comment.

---

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 1     |
| HIGH     | 4     |
| MEDIUM   | 2     |
| LOW      | 0     |

**Most critical:** Bug 1 -- R-space volume integration means the optimizer minimizes `(flux - target*V)^2` instead of `(flux - target)^2`. If mesh volume != 1, all inferred parameters are systematically biased. Verify whether your mesh has unit volume.
