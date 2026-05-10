# V15 Pipeline Codex Audit Log

## Round 1: Initial 6 Error-Prone Areas

### Area 1 (`_extend_voltage_cache_for_p2`, Infer_BVMaster_charged_v15.py:386-475) — RISK / LOW
The scalar-vs-list guard at line 445 works today because `make_bv_solver_params` always supplies a list for `sp[4]`. The fragility is that `isinstance(sp[4], (list, tuple))` will miss `np.ndarray`, so passing numpy z-values would silently take the scalar branch. Suggested fix: use `np.asarray` normalization or read from the solver context dict directly.

### Area 2 (`_target_cache_path`, Infer_BVMaster_charged_v15.py:156-184) — RISK / MEDIUM
The mesh parameters are hardcoded as the string `"Nx=8,Ny=200,beta=3.0"` at line 178, while the actual mesh is built at line 207 with the same values. If someone changes the mesh construction without updating the hash string, stale cached targets will be silently reused. The eta/shape validation at line 275 does not cover mesh changes. Suggested fix: parameterize the mesh values into `_target_cache_path` and hash the actual numbers.

### Area 3 (`_subset_targets`, Infer_BVMaster_charged_v15.py:315-323) — RISK / LOW
Unmatched voltages are silently dropped (lines 319-320), so the returned arrays can be shorter than `subset_eta`. A downstream shape check in `pipelines.py:1348` will eventually catch this, but the error message will be unhelpful. Suggested fix: assert `len(idx) == len(subset_eta)` inside `_subset_targets` and raise with the list of missing voltages.

### Area 4 (`_solve_single_sample_adaptive`, overnight_train_v15.py:201-344) — RISK / MEDIUM
The `ctx.get("achieved_z_factor", 0.0)` call at line 278 is correct -- `solve_bv_with_charge_continuation` does set `ctx["achieved_z_factor"]` at `solvers.py:565`. The real issue is the bare `except Exception: pass` at lines 289 and 331-332, which swallows all errors including genuine coding bugs and runtime failures, silently marking them as non-converged samples. Suggested fix: catch only expected solver exceptions and log or re-raise unexpected ones.

### Area 5 (Checkpoint resume, overnight_train_v15.py:560-576) — BENIGN / INFO
The condition `ckpt_converged[i] or (not np.isnan(ckpt_cd[i]).all())` at line 567 has correct operator precedence (the parentheses disambiguate). The logic correctly restores samples that either converged or have any finite data. No bug found.

### Area 6 (Noise seeding, Infer_BVMaster_charged_v15.py:296-297) — BENIGN / INFO
`add_percent_noise` creates a fresh local `np.random.default_rng(seed)` at `common.py:197`, so no global state is mutated. Using `seed` and `seed+1` produces deterministic, independent streams for CD and PC noise. No correlation bug.

---

## Round 1 Verification (Independent Claude Agent)

| Finding | Codex Rating | Verification | Notes |
|---------|-------------|-------------|-------|
| 1: sp[4] isinstance miss | LOW | **FALSE ALARM** | `make_bv_solver_params` always returns a Python `list` for z_vals. ndarray never occurs. |
| 2: Hardcoded mesh in cache key | MEDIUM | **PARTIAL** | All callers use 8x200 beta=3.0 today. Maintenance smell, not active bug. |
| 3: Silent voltage drop in _subset_targets | LOW | **PARTIAL** | Grids are subsets by construction. Would only fail if grids diverged in future. |
| 4: Broad except + achieved_z_factor | MEDIUM | **PARTIAL** | `achieved_z_factor` IS always set (false alarm on that). Broad `except` is real but intentional for batch training fallback. |
| 5: Checkpoint operator precedence | BENIGN | **FALSE ALARM** | Logic is correct as written. |
| 6: Noise seeding correlation | BENIGN | **FALSE ALARM** | Local RNG, independent streams. |

**Round 1 Score: 0 confirmed bugs, 3 partial (maintenance risks), 3 false alarms**

---

## Round 2: Downstream Infrastructure (multistart, Jacobian, parallel)

### Area 7 (`_evaluate_grid_objectives`, Surrogate/multistart.py:205-269) — OK
Separate masks are intentionally independent: `valid_cd`/`valid_pc` built separately; residuals computed on each mask's own subset; NaN penalization is per-observable then unioned (`has_nan_cd | has_nan_pc`). No shape mismatch or asymmetric over-penalization found.

### Area 8 (`build_residual_jacobian`, FluxCurve/bv_curve_eval.py:136-165) — SUSPICIOUS
Jacobian row reconstruction uses `jacobian[i, :] = point.gradient / r_i` with absolute cutoff `abs(r_i) > 1e-10`. Residual scale depends on configurable observable scaling; consumed directly by `least_squares`. Potential numerical instability for poorly scaled observables: rows near threshold can be zeroed too aggressively or noise-amplified just above threshold. Suggested fix: scale-aware threshold `eps_r = sqrt(eps_float) * max(1.0, |sim|, |target|)`.

### Area 9 (Worker task parsing + result collection, FluxCurve/bv_parallel.py:308-320, 738-747) — BUG
Collector allocates `results` by `n_tasks` then indexes by `point_index`. Upstream task builders use original indices and can skip NaN points, creating sparse `point_index` values. Sparse indices can trigger `IndexError` at `results[expected_idx]`, causing exception path and serial fallback. Practical effect is false parallel failure/performance loss, not silent wrong answers. Suggested fix: store by submission position, not `point_index`.

---

## Round 2 Verification (Independent Claude Agent)

| Finding | Codex Rating | Verification | Notes |
|---------|-------------|-------------|-------|
| 7: multistart NaN masking | OK | **FALSE ALARM** (agrees) | Masks are intentionally independent, union penalization is correct. |
| 8: Jacobian 1e-10 threshold | SUSPICIOUS | **FALSE ALARM** | Typical residuals are O(1e-3 to 1e-1) given I_SCALE~0.18 mA/cm². Threshold is 7+ orders below realistic values. Zeroing is mathematically correct. |
| 9: Parallel result IndexError | BUG | **CONFIRMED** | `point_index` is sparse when NaN targets skipped. `results[5]` on length-3 list causes IndexError → serial fallback. Frequent in practice with experimental data gaps. |

**Round 2 Score: 1 confirmed bug (Area 9), 2 false alarms**

---

## Round 3: Optimizer + Solver Deep Dive (Codex stalled — resubmitted as Round 5)

*Codex Round 3 timed out after 40+ tool calls without producing final verdicts. The 4 areas (log-space bounds, charge continuation exceptions, steady-state metrics, warm-start recovery) will be resubmitted.*

---

## Round 4: Cache, Regularization, Gradient Truncation, Recovery

### Area 10 (Cache key precision, optimization.py:300) — LIKELY-BUG (opposite direction)
Not-a-bug for the stated concern (cache misses). But **likely-bug for cache collisions**: `.12g` format rounds to 12 significant digits; IEEE-754 float64 needs 17 digits for unique round-trip. Distinct nearby points can alias to the same key, causing stale objective/gradient reuse. Severity: Medium. Fix: use `tuple(np.float64(v).hex() for v in parts)`.

### Area 11 (Regularization gradient, optimization.py:397-408) — NOT A BUG
Chain rule is correct. `J_reg = lambda * sum(diff_log^2)`, derivative `2*lambda * diff_log / (k0 * ln(10))` matches the code exactly when optimizing in linear k0-space. No sign/factor error.

### Area 12 (Silent gradient truncation, bv_observables.py:108-123) — CONFIRMED BUG (HIGH)
If gradient count from BV solver doesn't match `n_controls`, code warns but continues with zero-padded/truncated gradients. These are directly accumulated into the curve gradient at `bv_curve_eval.py:78`. Missing control gradients become zero → optimizer receives false zero-gradient → parameters can freeze at initial values or drive incorrect convergence. Fix: raise ValueError on size mismatch.

### Area 13 (Line-search schedule empty-array, recovery.py:104-105) — NOT A BUG
Guarded by truthiness check at `recovery.py:103` (`if recovery.line_search_schedule:`). `local_step` clamped to >=1. Default schedule is non-empty. No negative index path exists.

---

## Round 4 Verification (Independent Claude Agent)

| Finding | Codex Rating | Verification | Notes |
|---------|-------------|-------------|-------|
| 10: Cache key .12g collision | LIKELY-BUG | **FALSE ALARM** | L-BFGS-B step sizes are many orders of magnitude larger than 1e-12 relative precision. Aliasing cannot occur between genuinely distinct optimizer iterates in practice. |
| 11: Regularization gradient | NOT A BUG | **FALSE ALARM** (agrees) | Math verified: chain rule is correct for both log and linear k0 space. |
| 12: Silent gradient truncation | CONFIRMED BUG | **CONFIRMED** | Zero-padded gradients feed false zero-gradient signal to optimizer. Called from 4 locations. Mismatch can occur in joint/full control modes. Should raise ValueError or return NaN. |
| 13: Line-search schedule | NOT A BUG | **FALSE ALARM** (agrees) | Truthiness guard on line 103, default schedule non-empty. |

**Round 4 Score: 1 confirmed bug (Area 12), 3 false alarms**

---

## Round 5: Retry of Round 3 (log-space bounds, charge cont, steady-state, warm-start)

### Area 14 (Log-space bounds inversion, optimization.py:108-109) — BUG
`log10(max(lower, 1e-30))` vs `log10(upper)` without validating `upper > 1e-30`. If `upper <= 1e-30`, produces invalid/reversed log bounds. Reachable because `pipelines.py:86,577` passes `request.k0_upper` with no positivity check. Default `k0_upper=100.0` is safe, so only triggered by custom configs.

### Area 15 (Broad exception catch, solvers.py:526-530) — NOT A BUG
Although catches `Exception`, it re-raises anything that is neither `fd.ConvergenceError` nor `PETSc.Error`. Programming errors like KeyError/AttributeError propagate correctly.

### Area 16 (Steady-state zero-flux metric, bv.py:261-267) — NOT A BUG
Convergence uses `(rel <= rtol) OR (abs <= atol)`. With default `absolute_tolerance=1e-8`, the `1e-16` floor is inactive. Only affects extreme custom atol values.

### Area 17 (Warm-start exception recovery, bv.py:374-385) — NOT A BUG
`U_prev` is updated after every successful step. On exception at first step of new voltage, it IS the prior voltage's final state. On later steps, it's the latest successful state. No warm-start corruption found.

---

## Round 6: Solver Limits + State Management

### Area 18 (Charge ramp missing iteration limit, solvers.py:533-562) — SUSPICIOUS
Not unbounded in normal use: failed steps halve the gap, and `min_delta_z` floor enforces termination. But if `min_delta_z <= 0` or non-finite is passed, no guaranteed break and queue growth becomes unbounded. Fix: validate `min_delta_z > 0 and finite` at entry; optional `max_z_attempts` cap.

### Area 19 (Steady-state carry-over between voltages, bv.py:365-403) — NOT A BUG
`prev_current` and `steady_count` are reinitialized per voltage at lines 365-366. No carry-over from voltage N-1. False early convergence is not possible.

### Area 20 (Exception path None metrics, bv.py:243-253) — NOT A BUG
`SteadyStateResult` explicitly types metrics as optional (`common.py:83-84`). Writers/consumers guard None before float conversion. No downstream crash.

---

## Round 5 Verification (Independent Claude Agent)

| Finding | Codex Rating | Verification | Notes |
|---------|-------------|-------------|-------|
| 14: Log-space bounds inversion | BUG | **CONFIRMED** | No validators on BVFluxCurveInferenceRequest. Asymmetric guard (lower clamped, upper not). Requires pathological user input but is a real defensive gap. |
| 15: Broad exception catch | NOT A BUG | **CONFIRMED** (agrees) | Re-raise logic verified: only ConvergenceError and PETSc.Error swallowed, all others propagate. |
| 16: Steady-state zero-flux | NOT A BUG | **CONFIRMED** (agrees) | OR condition (`rel <= rtol or abs <= atol`). Absolute check alone can declare convergence. 1e-16 floor only affects denominator. |
| 17: Warm-start recovery | NOT A BUG | **CONFIRMED** (agrees) | `U_prev.assign(U)` after every successful step. Rollback restores last good state correctly. |

**Round 5 Score: 1 confirmed bug (Area 14, config-dependent), 3 confirmed not-bugs**

---

## Round 6: Solver Limits + State Management

### Area 21 (Config deep-copy aliasing, _bv_common.py:315-334) — BUG
`reaction_2["cathodic_conc_factors"] = list(h_factor)` only copies the list, not the inner dicts. Both reactions share the same factor dict object. Mutating one reaction's factor mutates the other. Current code often updates both together (masking the bug), but reaction-specific edits will corrupt the other. Fix: `[dict(f) for f in h_factor]` or `copy.deepcopy`.

---

## Round 6 Verification (Independent Claude Agent)

| Finding | Codex Rating | Verification | Notes |
|---------|-------------|-------------|-------|
| 18: z-ramp iteration limit | SUSPICIOUS | **CONFIRMED (minor)** | `gap < min_delta_z` (strict <, not <=). If min_delta_z=0, loop never terminates. No caller passes 0 today; default is 0.005. Defensive gap only. |
| 19: Steady-state carry-over | NOT A BUG | **CONFIRMED** (agrees) | All state vars reset inside voltage loop at lines 365-370. |
| 20: Exception None metrics | NOT A BUG | **CONFIRMED** (agrees) | SteadyStateResult declares Optional fields. Consumers guard None. |
| 21: Config aliasing | BUG | **CONFIRMED** | Shallow copy shares inner dict. Downstream code at `bv_curve_eval.py:473` and `pipelines.py:1700` mutates `ccf["c_ref_nondim"]` in-place. Currently latent because both reactions get same value, but will corrupt if extended to per-reaction values. |

**Round 6 Score: 1 confirmed bug (Area 21, latent), 1 confirmed minor defensive gap (Area 18), 2 confirmed not-bugs**

---

## Round 8: Training Infrastructure + I/O

### Area 22 (Checkpoint resume dimension mismatch, Surrogate/training.py:290-311) — BUG
`all_cd/all_pc` allocated as `(N, n_eta)`, then checkpoint arrays copied directly using `n_done` from file with no checks for `n_done <= N`, no shape validation, and no `phi_applied` match check (even though it's saved). Fix: validate n_done, array shapes, and phi_applied equality before restore.

### Area 23 (Silent worker group loss, Surrogate/training.py:853-858) — PARTIAL BUG
On worker-group exception, code logs and `continue`s. Group indices stay NaN. But final file hardcodes `n_completed=N` at line 941, misreporting actual completion. `n_failed = N - n_valid_final` catches missing samples, but `n_completed` metadata is wrong. Fix: track actual n_completed; on group failure, mark indices explicitly.

### Area 24 (Adjoint failure index misalignment, FluxCurve/bv_point_solve/forward.py:291-298) — NOT A BUG
Adjoint failure returns None, but caller checks and falls back to full sequential sweep (`__init__.py:305-309`). Sequential path fills `results[orig_idx]` densely. No index gap.

---

## Round 7: Predictor, Ensemble, Training, Adjoint

### Area 26 (Predictor clamp applied to potential field, predictor.py:76-78, 91-93) — BUG
Clamp `np.maximum(d.data, 1e-10)` is applied to ALL mixed-state components via `for d in ctx_U.dat`, including the electrical potential (not just concentrations). Potential field can be incorrectly clipped to >= 1e-10 when it should be negative. Fix: only clamp concentration DOFs, not potential DOFs.

### Area 27 (Ensemble NaN propagation, ensemble.py:93-103) — WARN
`mean/std` will propagate NaN from any member since no masking/`nanmean`/`nanstd` is used. If one ensemble member produces NaN, the whole prediction becomes NaN.

### Area 28 (Stale cross-eval cache in training, training.py:124-126, 184) — BUG
`return_solutions` can return stale seeded cache entries. Function seeds `_cross_eval_cache` but solver only overwrites first-point cache on convergence. Returned dict is raw cache without convergence validation.

### Area 29 (Adjoint tape annotation gap, bv_point_solve/forward.py:239-240) — BUG
`U_prev.assign(U)` is excluded from tape, but `U_prev` is in the residual form. Multi-step gradients can miss dependence through step-to-step state updates. This means adjoint gradients may be incorrect for multi-step solves.

### Area 25 (Cache cleanup on exception, FluxCurve/bv_run/io.py:152-176) — BUG
`_clear_caches()` called before and after solve without `try/finally`. Exception during generation skips second clear. Solver mutates caches during execution, so stale state leaks. Fix: wrap in `try: ... finally: _clear_caches()`.

---

## Round 8 Verification (Independent Claude Agent)

| Finding | Codex Rating | Verification | Notes |
|---------|-------------|-------------|-------|
| 22: Checkpoint resume no validation | BUG | **CONFIRMED** | Zero guards: no n_done <= N check, no shape check, no phi_applied match. Blindly slices. |
| 23: Silent worker group loss + n_completed | PARTIAL | **CONFIRMED** | Two issues: (1) exception skips samples without counting them, (2) final save hardcodes `n_completed=N` instead of actual count. |
| 24: Adjoint failure index gap | NOT A BUG | **CONFIRMED** (agrees) | Fast path returns None → falls back to sequential sweep → dense results. |
| 25: Cache cleanup on exception | BUG | **FALSE ALARM** | `_clear_caches()` is called at START of next invocation too (line 152), so "clear before use" pattern handles stale cache. Minor robustness opportunity, not a real bug. |

**Round 8 Score: 2 confirmed bugs (Areas 22, 23), 1 confirmed not-bug, 1 false alarm (Codex overcalled)**

---

## Round 7 Verification (Independent Claude Agent)

| Finding | Codex Rating | Verification | Notes |
|---------|-------------|-------------|-------|
| 26: Predictor clamp on potential | BUG | **CONFIRMED** | `for d in ctx_U.dat` clamps ALL components including electrical potential (last component). Potential can legitimately be negative. Comment says "clamp concentrations" but loop doesn't distinguish. |
| 27: Ensemble NaN propagation | WARN | **CONFIRMED** | Uses `mean/std` not `nanmean/nanstd`. Single failing member poisons entire prediction. |
| 28: Stale cross-eval cache | BUG | **PARTIAL** | Cache IS updated during solve (not stale seeds as Codex claimed). But `converged_solutions` at line 184 is unfiltered by convergence status — unconverged solutions passed downstream. Name is misleading. |
| 29: Adjoint tape gap | BUG | **CONFIRMED (CRITICAL)** | `U_prev.assign(U)` at line 240 is inside `stop_annotating()`, but `U_prev` feeds the variational form via `ci_prev` (forms.py:200-203). Multi-step adjoint gradients are incorrect. **This is the most critical bug found — adjoint-based PDE optimization computes wrong gradients.** |

**Round 7 Score: 2 confirmed bugs (Areas 26, 29 — 29 is CRITICAL), 1 confirmed warn (Area 27), 1 partial (Area 28)**

---

## FINAL SUMMARY — All Verified Findings

### CRITICAL
- **Area 29**: Adjoint tape gap — `U_prev.assign(U)` excluded from pyadjoint tape, breaking multi-step gradient propagation (forward.py:239-240)

### HIGH (Confirmed Bugs)
- **Area 9**: Parallel result IndexError on sparse point indices (bv_parallel.py:738-747) — causes unnecessary serial fallback
- **Area 12**: Silent gradient zero-padding on control count mismatch (bv_observables.py:108-123) — false zero-gradient to optimizer
- **Area 22**: Checkpoint resume with no dimension/shape validation (training.py:290-311)
- **Area 23**: Worker group loss misreports `n_completed=N` hardcoded (training.py:941)
- **Area 26**: Predictor clamps electrical potential to >= 1e-10 (predictor.py:76-78) — corrupts extrapolation

### MEDIUM (Config-dependent / Latent)
- **Area 14**: Log-space bounds inversion if k0_upper <= 1e-30 (optimization.py:108-109) — needs pathological input
- **Area 21**: Config shallow-copy aliasing between reactions (_bv_common.py:315-334) — latent, masked by current usage
- **Area 28**: `converged_solutions` dict unfiltered by convergence (training.py:184) — misleading name, unconverged warm-starts

### LOW (Defensive Gaps / Warnings)
- **Area 18**: z-ramp loop has no max iteration cap if min_delta_z=0 (solvers.py:533-562)
- **Area 27**: Ensemble NaN propagation — no nanmean/nanstd (ensemble.py:93-103)

### Confirmed Not-Bugs (15 areas)
Areas 1, 5, 6, 7, 8, 10, 11, 13, 15, 16, 17, 19, 20, 24, 25

### Partial / Maintenance Risks (3 areas)
Areas 2, 3, 4

---

## FIX STATUS — All Bugs Patched and Dual-Verified

| Fix | Bug | File | Claude | Codex | Post-Codex Fixes |
|-----|-----|------|--------|-------|-----------------|
| 1 | CRITICAL: Adjoint tape gap | `forward.py:239` | PASS | VERIFIED | — |
| 2 | Parallel sparse IndexError | `bv_parallel.py:733-747` | PASS | VERIFIED | — |
| 3 | Gradient zero-padding → ValueError | `bv_observables.py:108-111` | PASS | VERIFIED | — |
| 4 | Predictor clamps potential field | `predictor.py:79,96` + `__init__.py:520` | PASS | VERIFIED | — |
| 5 | Log-space bounds no upper guard | `optimization.py:103-107,115,135,160,664,681,704` | PASS | VERIFIED | Fixed least-squares path (3 more locations) |
| 6 | Config shallow-copy aliasing | `_bv_common.py:334` | PASS | VERIFIED | — |
| 7A | Checkpoint resume no validation | `training.py:312-327` | PASS | PARTIAL→PASS | Added parameter-sample identity check |
| 7B | n_completed hardcoded as N | `training.py:822,964` | PASS | Found gap→PASS | Initialized before branch; clarified semantics |
| 7C | converged_solutions unfiltered | `training.py:184-187` | PASS | VERIFIED | — |

**9 bugs fixed across 7 files. All dual-verified (Claude + Codex).**
