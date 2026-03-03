# Inference Pipeline Optimization Plan

**Date:** 2026-02-28
**Author:** Agent Opt (Optimization Specialist)
**Target:** Speed up BV k0/alpha/joint inference pipeline by 5-20x

---

## 0. Executive Summary

The inference pipeline is dominated by **repeated forward+adjoint PDE solves** inside the optimizer loop. The call chain is:

```
scipy.minimize (L-BFGS-B)
  -> _evaluate() [bv_run.py:371]
    -> evaluate_bv_curve_objective_and_gradient() [bv_curve_eval.py:15]
      -> solve_bv_curve_points_with_warmstart() [bv_point_solve.py:29]
        -> FOR EACH voltage point (sequential, sorted by |eta|):
          1. tape.clear_tape() + continue_annotation()
          2. bv_build_context(params, mesh=mesh)         [REBUILD mesh/spaces]
          3. bv_build_forms(ctx, params)                   [REBUILD UFL forms]
          4. bv_set_initial_conditions(ctx, params)        [RESET IC]
          5. Warm-start from carry_U_data (numpy copy)
          6. fd.NonlinearVariationalSolver(problem, ...)   [REBUILD solver]
          7. FOR step in range(1, max_steps+1):
               solver.solve()                             [SNES Newton]
               convergence check
          8. Build ReducedFunctional + compute derivative  [ADJOINT]
```

**The single biggest bottleneck**: Steps 2-3 and 6 are done *for every voltage point, for every optimizer evaluation*. With 10 voltage points and 30 optimizer iterations, that is ~300 full context+form+solver rebuilds per inference run. Each rebuild involves Firedrake form compilation, JIT kernel compilation (cached but still overhead), and PETSc solver setup.

---

## 1. Profiling Strategy (Do First)

### What to measure

Add timing instrumentation to identify where wall time is spent:

| Measurement | Where | How |
|---|---|---|
| Total per-evaluation time | `_evaluate()` in `bv_run.py:371` | `time.perf_counter()` around `evaluate_bv_curve_objective_and_gradient()` |
| Per-point forward solve time | `bv_point_solve.py:233` loop | Timer around the `for step in range(1, max_steps+1)` block |
| Per-point adjoint time | `bv_point_solve.py:272-278` | Timer around `rf(control_state)` + `rf.derivative()` |
| Context+forms+solver setup time | `bv_point_solve.py:165-216` | Timer around `bv_build_context` through solver construction |
| Per-SNES-step time | Individual `solver.solve()` calls | Timer inside the step loop |
| Number of SNES iterations | PETSc SNES monitor | Add `"snes_converged_reason": None` to solver_parameters |

### Implementation

```python
# In bv_point_solve.py, at top of the per-point loop:
import time
t_setup = time.perf_counter()
# ... build_context, build_forms, set_initial_conditions, solver construction ...
t_setup = time.perf_counter() - t_setup

t_forward = time.perf_counter()
# ... time-stepping loop ...
t_forward = time.perf_counter() - t_forward

t_adjoint = time.perf_counter()
# ... ReducedFunctional + derivative ...
t_adjoint = time.perf_counter() - t_adjoint

print(f"  [timing] phi={phi_applied_i:.4f} setup={t_setup:.2f}s fwd={t_forward:.2f}s adj={t_adjoint:.2f}s steps={steps_taken}")
```

**Estimated effort:** 15 minutes.
**Files:** `FluxCurve/bv_point_solve.py`

---

## 2. Quick Wins (< 1 hour each, expect > 2x combined speedup)

### 2A. Reduce `max_steps` for Warm-Started Points

**Problem:** `SteadyStateConfig.max_steps` defaults to 200. When warm-starting from the previous point's converged solution, the new point is already very close to its steady state. Typically only 2-5 pseudo-time steps are needed to re-converge.

**Current code** (`bv_point_solve.py:233`):
```python
for step in range(1, max_steps + 1):
    solver.solve()
    ...
```

**Fix:** Add a separate `max_steps_warmstart` parameter (or use a heuristic: if `carry_U_data is not None`, use `min(max_steps, 10)`). The first point (no warm-start) still uses full `max_steps`.

**Expected speedup:** If average steps drop from ~50 to ~5 for warm-started points, the forward solve time for 9/10 points drops by ~10x. Total forward solve speedup ~5x.

**Effort:** 20 minutes. Change `bv_point_solve.py:233` and add parameter to `SteadyStateConfig`.

**Risk:** Very low. The steady-state convergence criterion still guards correctness.

### 2B. Eliminate Per-Point Context/Form/Solver Rebuild

**Problem:** The most wasteful pattern in the codebase. Lines `bv_point_solve.py:165-216` rebuild the *entire* Firedrake context (mesh, function spaces, forms, solver) **for every voltage point**. This is necessary for adjoint taping (each point needs a clean tape), but the mesh and function spaces are identical across points.

**Current architecture:**
```
For each point:
  tape.clear_tape()
  ctx = bv_build_context(params, mesh=mesh)     # REBUILD mesh/spaces
  ctx = bv_build_forms(ctx, params)              # REBUILD UFL forms
  solver = fd.NonlinearVariationalSolver(...)     # REBUILD solver
```

**Fix:** Separate the "one-time setup" from the "per-point adjoint-tracked operations":

```python
# ONCE (before the sweep loop):
base_ctx = bv_build_context(params, mesh=mesh)  # mesh, V_scalar, W
# The mesh and function spaces never change.

# PER POINT:
tape.clear_tape()
adj.continue_annotation()
# Create fresh U, U_prev in the SAME function space (no mesh rebuild)
U = fd.Function(base_ctx["W"])
U_prev = fd.Function(base_ctx["W"])
# Rebuild forms with these new functions (forms must be on the tape)
# But reuse mesh, V_scalar, W, R_space from base_ctx
```

The key insight: `build_context` creates the mesh and function spaces (expensive); `build_forms` creates the UFL forms (must be on tape). We can split these. The mesh construction (especially graded meshes) and `MixedFunctionSpace` creation are the expensive parts and are tape-independent.

**Expected speedup:** 30-50% reduction in per-point overhead (the form compilation is cached by Firedrake/TSFC, but solver setup is not).

**Effort:** 45 minutes. Refactor `bv_point_solve.py` to extract mesh/space creation.

**Risk:** Low-medium. Must ensure pyadjoint sees the new Functions. The mesh itself is not adjoint-tracked.

**Files:** `FluxCurve/bv_point_solve.py`, `Forward/bv_solver.py` (add `build_context_reuse` variant)

### 2C. Cache Evaluation Results More Aggressively

**Problem:** `bv_run.py:309-314` caches by `_key_from_x()` (parameter values rounded to 12 digits). L-BFGS-B calls `_fun(x)` and `_jac(x)` separately, but the cache handles this correctly (same x -> same key -> cache hit on second call).

However, if L-BFGS-B's line search evaluates nearby points, those are all cache misses. No optimization possible here without changing the optimizer.

**Observation:** The callback `_callback` at `bv_run.py:515-516` calls `_evaluate(xk)` again, which should be a cache hit. Verify this is actually hitting the cache (add a print on cache hit/miss).

**Effort:** 10 minutes. Diagnostic only.

### 2D. Use `fd.Constant` for `dt_model` Instead of Float

**Problem:** `bv_solver.py:536`:
```python
dt_m = float(scaling["dt_model"])
```

This embeds `dt_model` as a Python float in the UFL form. If we ever want to change the timestep (for adaptive stepping), the entire form must be recompiled. Using `fd.Constant` instead means the value can be changed without recompilation.

**Fix:**
```python
dt_const = fd.Constant(float(scaling["dt_model"]))
# Replace all uses of dt_m with dt_const in the form
```

**Expected speedup:** No immediate speedup, but this is a prerequisite for Quick Win 2A and Medium-effort items 3A/3B.

**Effort:** 15 minutes. Change `bv_solver.py:536` and the form assembly lines.

**Risk:** Very low. `fd.Constant` is the standard Firedrake pattern.

**Files:** `Forward/bv_solver.py`

### 2E. Early Termination for Converged Optimizer

**Problem:** L-BFGS-B's `gtol` is set to 1e-4 (`bv_config.py:75`), which may be unnecessarily tight for noisy synthetic data with 2% noise. The optimizer may waste iterations reducing the gradient below the noise floor.

**Fix:** Increase `gtol` to `1e-3` or add a custom convergence check in `_callback` that stops when `|objective_change| / objective < tol` for 3 consecutive iterations:

```python
def _callback(xk):
    ...
    # Check for stagnation
    if len(history_rows) >= 3:
        objs = [float(r["objective"]) for r in history_rows[-3:]]
        if max(objs) - min(objs) < 1e-6 * abs(objs[-1]):
            raise StopIteration("Objective stagnated")
```

Note: `scipy.optimize.minimize` does not natively support early stopping from callbacks, but raising `StopIteration` or returning `True` from the callback is not supported. Instead, use `options={"maxiter": N}` and reduce `N`, or implement a wrapper.

**Expected speedup:** Depends on problem. Could save 5-15 evaluations (each costing ~10 forward solves).

**Effort:** 15 minutes.

**Risk:** Low. The optimizer result will indicate non-convergence, but the best parameters are tracked separately.

---

## 3. Medium Effort (1-3 hours, potentially large speedup)

### 3A. Adaptive Pseudo-Timestep for Warm-Started Points

**Problem:** Currently each SNES solve uses the same fixed `dt_model` regardless of how close the initial guess is to the solution. For warm-started points, the initial guess is already very close; a large pseudo-timestep would converge faster.

**Strategy (SER — Switched Evolution/Relaxation):**

After each successful SNES solve, compute:
```python
# Residual-based timestep adaptation
r_new = norm(F(U))
r_old = norm(F(U_prev))
if r_new < r_old:
    dt_new = min(dt * growth * (r_old / r_new), dt_max)
else:
    dt_new = max(dt / 2, dt_min)
dt_const.assign(dt_new)
```

For warm-started points, `r_old` is already small, so `dt` immediately jumps to `dt_max` and the "time-stepping" converges in 1-2 steps.

**Prerequisite:** Quick Win 2D (mutable `dt_const`).

**Expected speedup:** 3-5x for warm-started points (1-2 steps instead of 5-10).

**Effort:** 1-2 hours. Modify `bv_point_solve.py` time-stepping loop.

**Risk:** Medium. Must handle the case where the residual ratio is noisy or the SNES diverges at a large dt.

**Files:** `FluxCurve/bv_point_solve.py`, `Forward/bv_solver.py`

### 3B. Predictor Step (Linear Extrapolation Between Voltage Points)

**Problem:** Currently the warm-start just copies the previous solution. A linear extrapolation from the last two solutions gives a much better initial guess:

```python
# After solving points k and k-1:
if carry_U_data_prev is not None and carry_U_data is not None:
    # Linear extrapolation in phi_applied direction
    dphi = phi_applied_values[sorted_indices[sweep_idx]] - phi_applied_values[sorted_indices[sweep_idx - 1]]
    dphi_prev = phi_applied_values[sorted_indices[sweep_idx - 1]] - phi_applied_values[sorted_indices[sweep_idx - 2]]
    if abs(dphi_prev) > 1e-12:
        ratio = dphi / dphi_prev
        for src, src_prev, dst in zip(carry_U_data, carry_U_data_prev, ctx["U"].dat):
            dst.data[:] = src + ratio * (src - src_prev)
        # Clip concentrations to prevent negative values
        for i in range(n_species):
            ci_data = ctx["U"].dat[i].data
            ci_data[:] = np.maximum(ci_data, 1e-10)
```

**Expected speedup:** 30-50% fewer SNES iterations per point (from literature on continuation methods). Combined with 3A, could reduce warm-started points to 1 step.

**Effort:** 1 hour. Modify `bv_point_solve.py`.

**Risk:** Medium. Extrapolation can overshoot, especially for concentrations near zero. The clipping mitigates this but could degrade the initial guess quality.

**Files:** `FluxCurve/bv_point_solve.py`

### 3C. Tape Optimization: Minimize Annotated Operations

**Problem:** Every operation inside the annotation block (`bv_point_solve.py:162-307`) is recorded on the pyadjoint tape. This includes:

1. `bv_build_context` (mesh/space creation) -- NOT needed on tape
2. `bv_build_forms` (UFL form creation) -- NEEDED on tape
3. `bv_set_initial_conditions` (interpolation) -- NEEDED on tape (for IC sensitivity)
4. `k0_func.assign()`, `alpha_func.assign()` -- NEEDED (control variables)
5. `solver.solve()` (Newton iterations) -- NEEDED (state equation)
6. `U_prev.assign(U)` -- NEEDED (time-stepping dependency)
7. Convergence check `fd.assemble(observable_form)` -- Currently done with `stop_annotating()` (good!)

**Optimization opportunities:**

a. **Move mesh/space creation outside the tape block.** The mesh and function spaces are not adjoint-controlled. Only the Functions and their assignments need to be on the tape.

b. **Reduce the number of time steps recorded.** Currently all `max_steps` Newton solves are on the tape. If steady state is reached at step 5, the tape has 5 solve records. But the adjoint only needs the *final* state and the path to it from the controls. With checkpointing, we can avoid storing all intermediate states.

c. **Use `adj.pause_annotation()` for non-essential operations.** E.g., the warm-start copy (`carry_U_data` assignment at line 170-173) is currently annotated but the carry data is a numpy array copy -- the `U.dat[:] = src` assignment should still be annotated since it sets the state from which the solve proceeds.

**Expected speedup:** 10-30% reduction in tape overhead (memory and time for tape replay during adjoint).

**Effort:** 1-2 hours.

**Risk:** Medium. Must be careful not to break the adjoint computation chain.

**Files:** `FluxCurve/bv_point_solve.py`

### 3D. Jacobian and Preconditioner Lagging (PETSc)

**Problem:** Currently the Jacobian and preconditioner are recomputed at every Newton iteration. For the BV problem, the Jacobian changes slowly between iterations (especially near convergence).

**Fix:** Add PETSc solver parameters:

```python
solver_parameters = {
    ...
    "snes_lag_jacobian": 2,          # Recompute Jacobian every 2nd iteration
    "snes_lag_preconditioner": 2,    # Recompute preconditioner every 2nd iteration
    "snes_lag_jacobian_persists": True,  # Persist lagging across solves
    "snes_lag_preconditioner_persists": True,
}
```

For warm-started points where the Jacobian barely changes between voltage steps, `snes_lag_jacobian_persists: True` means the Jacobian from the previous point's final iteration is reused for the first iteration of the new point.

**Expected speedup:** 20-40% reduction in linear solve time (MUMPS factorization is the expensive part, and lagging avoids refactoring when unnecessary).

**Effort:** 30 minutes (parameter change only). But requires testing to ensure it doesn't hurt convergence.

**Risk:** Medium. Stale Jacobian can cause SNES divergence at large voltage steps. Safe to use with `snes_max_it: 15` as a guard.

**Files:** Caller scripts (solver_parameters dict), or `Forward/bv_solver.py` defaults.

### 3E. Reduce Voltage Points During Optimization

**Problem:** With 10 voltage points, each evaluation costs 10 forward+adjoint solves. Early in the optimization (far from the optimum), the gradient direction can be determined with fewer points.

**Strategy: Coarse-to-fine voltage grid.**

```python
# Phase 1: 5 voltage points, 10 optimizer iterations
# Phase 2: 10 voltage points, 20 optimizer iterations (warm-start from Phase 1)
```

Or: randomly subsample 5/10 points per evaluation (stochastic gradient), which is noisier but 2x cheaper per evaluation.

**Expected speedup:** 2x for the coarse phase (5 points instead of 10).

**Effort:** 1-2 hours. Modify `bv_run.py` to support multi-phase optimization.

**Risk:** Medium. The coarse grid may miss features in the I-V curve.

**Files:** `FluxCurve/bv_run.py`

---

## 4. Architectural Changes (Larger Effort)

### 4A. Persistent Solver with `phi_applied` as `fd.Constant`

**The ultimate refactor.** Currently, changing `phi_applied` requires a full rebuild because the adjoint tape must record the dependency of the solution on the control variables. But `phi_applied` is NOT a control variable -- it is a *parameter* of the experiment.

**Idea:** Build the solver ONCE with `phi_applied` as an `fd.Constant`. For each voltage point:
1. Clear tape
2. `phi_applied_func.assign(new_value)` (annotated)
3. `k0_func.assign(new_value)` (annotated -- this is the control)
4. Warm-start U from previous
5. `solver.solve()` (reuses the SAME solver instance)
6. Compute objective + adjoint

This avoids ALL per-point setup cost. The solver instance, preconditioner, assembled Jacobian pattern, etc. are all reused.

**Challenge:** pyadjoint must see the `phi_applied_func.assign()` on the tape to correctly propagate adjoint information. Since `phi_applied` is not a control (we don't differentiate w.r.t. it), this should work -- the tape just records it as a fixed parameter assignment.

The real challenge is that `build_forms` creates new `fd.Function` objects for controls (k0, alpha) each time. If we build forms once, the control Functions are fixed objects. We can still `assign` new values to them on each tape.

**Expected speedup:** 5-10x (eliminates form compilation, solver setup, preconditioner allocation for all but the first point).

**Effort:** 3-5 hours. Major refactor of `bv_point_solve.py`.

**Risk:** High. Must validate that pyadjoint correctly computes gradients when the solver is reused across tape clears. The tape replay must re-solve with the correct phi_applied and k0 values.

**Files:** `FluxCurve/bv_point_solve.py`, `Forward/bv_solver.py`

### 4B. Checkpointed Adjoint (Revolve Algorithm)

**Problem:** The current approach records ALL time steps on the tape, then replays them in reverse for the adjoint. With 50 time steps and a 4-species system on a 4x200 mesh, this stores ~50 full solution snapshots.

**Fix:** Use the `checkpoint_schedules` package (already in Firedrake) to store only sqrt(N) checkpoints and recompute intermediate states during the adjoint sweep:

```python
from checkpoint_schedules import Revolve
schedule = Revolve(max_n=max_steps, snaps_in_ram=5)
# Integrate schedule into the time-stepping loop
```

**Expected speedup:** Reduces memory by 10x (sqrt(50) ~ 7 checkpoints instead of 50). Time overhead is ~2x more forward solves for recomputation, but memory savings enable larger problems.

**Caveat:** For the current problem (small mesh, few time steps), memory is not the bottleneck. This is more relevant if the mesh is refined or the number of species increases.

**Effort:** 3-5 hours.

**Risk:** Medium. The checkpoint_schedules integration with Firedrake's adjoint is documented but not trivially plug-and-play for custom time-stepping loops.

**Files:** `FluxCurve/bv_point_solve.py`

### 4C. Surrogate-Accelerated Optimization

**Problem:** Each forward+adjoint evaluation is expensive. A surrogate model can provide cheap gradient approximations.

**Strategy:**

1. Run 5-10 full evaluations to build a training set: `{(k0, alpha) -> (J, dJ/dk0, dJ/dalpha)}`.
2. Fit a Gaussian Process (GP) or radial basis function (RBF) surrogate.
3. Optimize the surrogate (cheap) to find a candidate `x_new`.
4. Evaluate `x_new` with the full model.
5. Add to training set, refit, repeat.

This is "Bayesian Optimization" and typically converges in 10-20 full evaluations rather than 30+.

**Expected speedup:** 2-5x fewer full evaluations.

**Effort:** 4-8 hours (using `scikit-optimize` or `GPyOpt`).

**Risk:** High. The surrogate may not capture the non-convexity of the BV I-V curve objective well.

**Files:** New module `FluxCurve/surrogate.py`, modify `FluxCurve/bv_run.py`.

### 4D. Parallel Voltage Point Solves

**Problem:** The voltage points are solved sequentially for warm-starting. But within the optimizer, each evaluation is completely independent from the previous evaluation.

**Potential parallelism:**

a. **Inter-point parallelism (limited):** Points must be sequential for warm-starting. However, once a point is converged, its adjoint computation is independent and could overlap with the next point's forward solve. This is a pipeline parallelism.

b. **MPI-based ensemble parallelism:** Firedrake supports `fd.Ensemble` for running multiple independent solves in parallel. Each voltage point could run on a separate MPI rank. However, this breaks warm-starting.

c. **Parallel evaluations (most practical):** L-BFGS-B is inherently sequential (each evaluation depends on the previous), but the line search evaluates multiple trial points. A parallel line search could evaluate them simultaneously.

**Expected speedup:** 2-4x with ensemble parallelism (if warm-starting is abandoned in favor of more steps per point).

**Effort:** 8-16 hours.

**Risk:** High. Firedrake's MPI ensemble mode has constraints. Warm-starting is critical for convergence at large |eta|.

---

## 5. Prioritized Implementation Order

| # | Item | Speedup | Effort | Risk | Dependencies |
|---|------|---------|--------|------|-------------|
| 1 | **Profiling** (Section 1) | Diagnostic | 15 min | None | None |
| 2 | **2A: Reduce max_steps for warm-start** | ~5x forward | 20 min | Very low | None |
| 3 | **2D: fd.Constant for dt_model** | Prerequisite | 15 min | Very low | None |
| 4 | **2E: Relax gtol** | Save 5-15 evals | 15 min | Low | None |
| 5 | **2B: Eliminate per-point rebuild** | ~30-50% setup | 45 min | Low-med | None |
| 6 | **3D: Jacobian lagging** | ~20-40% SNES | 30 min | Medium | None |
| 7 | **3B: Predictor step** | ~30-50% Newton | 1 hr | Medium | None |
| 8 | **3A: Adaptive dt** | ~3-5x forward | 1-2 hr | Medium | #3 |
| 9 | **3C: Tape optimization** | ~10-30% adjoint | 1-2 hr | Medium | None |
| 10 | **4A: Persistent solver** | ~5-10x total | 3-5 hr | High | #5 |
| 11 | **3E: Coarse-to-fine grid** | ~2x early iters | 1-2 hr | Medium | None |
| 12 | **4C: Surrogate optimization** | ~2-5x evals | 4-8 hr | High | None |

**Recommended first session (< 2 hours):** Items 1-5 (profiling + all quick wins). Expected combined speedup: **3-10x**.

**Recommended second session (2-4 hours):** Items 6-9 (medium effort). Expected additional speedup on top of quick wins: **2-3x**.

**Total expected speedup after both sessions: 6-30x** (depending on problem size and which bottleneck dominates).

---

## 6. Specific Code Changes Reference

### File: `FluxCurve/bv_point_solve.py`

| Line(s) | Current | Proposed | Item |
|---------|---------|----------|------|
| 110 | `max_steps = int(max(1, steady.max_steps))` | Add `max_steps_warm = min(max_steps, 10)` | 2A |
| 165-167 | `ctx = bv_build_context(params, mesh=mesh)` etc. (per-point) | Move to before loop; per-point only creates U, U_prev, forms | 2B |
| 216 | `solver = fd.NonlinearVariationalSolver(...)` (per-point) | Reuse solver when possible | 2B, 4A |
| 233 | `for step in range(1, max_steps + 1):` | Use `max_steps_warm` when `carry_U_data is not None` | 2A |
| 170-173 | Warm-start: copy numpy data | Add linear extrapolation predictor | 3B |

### File: `Forward/bv_solver.py`

| Line(s) | Current | Proposed | Item |
|---------|---------|----------|------|
| 536 | `dt_m = float(scaling["dt_model"])` | `dt_const = fd.Constant(float(scaling["dt_model"]))` | 2D |
| 618 | `F_res += ((c_i - c_old) / dt_m) * v * dx` | `F_res += ((c_i - c_old) / dt_const) * v * dx` | 2D |

### File: `FluxCurve/bv_run.py`

| Line(s) | Current | Proposed | Item |
|---------|---------|----------|------|
| 232 | `options.setdefault("maxiter", ...)` | Also set higher `gtol` | 2E |

### Solver parameters (various scripts)

| Parameter | Current | Proposed | Item |
|-----------|---------|----------|------|
| `snes_lag_jacobian` | (not set, default 1) | 2 | 3D |
| `snes_lag_preconditioner` | (not set, default 1) | 2 | 3D |
| `snes_converged_reason` | (not set) | `None` (enables output) | 1 |

---

## 7. Risk Mitigation

1. **Always validate gradients after changes.** Use `rf.taylor_test(...)` on a single point to verify the adjoint gradient is correct after any tape-related changes.

2. **Profile before AND after.** Compare wall-clock times for a fixed test case (e.g., 3 voltage points, 5 optimizer iterations) before and after each change.

3. **Keep a baseline.** Before any changes, run the full inference and save results. After optimization, verify the final k0/alpha values and objective are consistent (within noise).

4. **Incremental changes.** Implement one item at a time, test, commit, then move to the next.
