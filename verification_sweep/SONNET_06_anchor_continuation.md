# SONNET_06: Anchor Continuation Logic Verification
**Agent:** 06/13  
**Scope:** `Forward/bv_solver/anchor_continuation.py` (2023 lines) + `Forward/bv_solver/solvers.py` (21 lines)  
**Date:** 2026-05-22

---

## Bottom Line

**PASS with two annotated concerns.** The anchor continuation implementation is sound and substantially complete. All four ladders (k0, kw_eff, c_s, lambda_hydrolysis) are implemented individually; combinations are gated by NotImplementedError with clear messages. Adjoint hygiene, failure handling, and rollback are correctly implemented. Two concerns below are noteworthy but not blocking correctness.

---

## Findings by Verification Item

### 1. `solve_anchor_with_continuation` — Ladder Config Support

**PASS.** The function signature (line 902–921) accepts all four ladder parameters:
- `k0_targets` (always required, non-empty dict with positive values)
- `kw_eff_ladder` (Optional[Tuple[float, ...]])
- `c_s_ladder` (Optional[Tuple[float, ...]])
- `lambda_hydrolysis_ladder` (Optional[Tuple[float, ...]])

**NotImplementedError scope (lines 1048–1063):** Only *combinations* raise NotImplementedError — specifically `c_s_ladder + kw_eff_ladder` together, and `lambda_hydrolysis_ladder + (kw_eff_ladder OR c_s_ladder)` together. Each ladder is individually supported. This is a change from v10a' where `c_s_ladder` was NotImplementedError entirely; it is now implemented as an outer loop over `cs_seq` at lines 1246–1316.

**Status from memory vs current code:** Memory entry `project_v10a_prime_two_stage_anchor.md` notes "c_s_ladder and kw_eff_ladder were NotImplementedError." Both are now individually implemented. The only remaining NotImplementedError is the pairwise combination guard.

### 2. `extract_preconverged_anchor` and `PreconvergedAnchor`

**PASS.** 

`PreconvergedAnchor` is a frozen dataclass (line 110) with `__post_init__` validation covering:
- `phi_applied_eta` finite
- `mesh_dof_count` positive int
- `U_snapshot` non-empty tuple of numpy arrays
- `k0_targets` non-empty tuple of (int, float) pairs with k0 > 0

`extract_preconverged_anchor` (line 191) correctly gates on `result.converged` and `result.U_data is not None` before constructing the anchor. U_snapshot entries are deep-copied via `np.asarray(arr).copy()`. The `k0_targets` are sorted by key for determinism. The ladder_history is serialized to `(float, str)` tuples.

`solve_grid_with_anchor` in `grid_per_voltage.py` asserts `isinstance(anchor, PreconvergedAnchor)` (line 974) — the type check is enforced on the receiver side.

### 3. Ladder Ascending/Descending Logic — Bisection, Monotone Progress, Bounded Retries

**PASS.**

`AdaptiveLadder` (line 719–896):
- Validates `initial_scales` strictly increasing to 1.0 at construction.
- `record_failure_and_insert` inserts `sqrt(prev * scale)` (geometric midpoint) between `previous_scale` and `current_scale`.
- `_inserts_at_current_step` counter is reset to 0 only on `record_success` (line 836). Incremented on each insert. Cap: `max_inserts_per_step` (default 4).
- When cap is exhausted, returns `False` (line 862). Caller raises `LadderExhausted`.
- **No infinite loop risk:** Each failure either inserts a strictly smaller midpoint (geometrically: `sqrt(prev*scale) < scale`) or returns False. The insert counter prevents unbounded subdivision. The float-precision guard at line 879 (`not (floor < midpoint < scale)`) provides an additional termination guarantee.
- `is_done()` (line 825) returns True when `self._idx >= len(self._planned)` — correctly handles inserts because they are prepended at `self._idx` and `self._idx` only advances on success.

**One subtle point:** `record_success` resets `_inserts_at_current_step` to 0, not the `_planned` list index. This means the per-step insert budget is correct: each distinct "step" (pair of successive scale values in the sequence) gets its own fresh budget of `max_inserts_per_step`. Verified correct.

### 4. Adaptive Ladder `warm_start_floor` — λ-from-warm-start

**PASS with one concern.**

The arithmetic-bisection floor is implemented at lines 739–883. When `warm_start_floor` is set, first-rung failures insert `0.5 * (warm_start_floor + scale)` instead of the geometric path (which is undefined at `prev=None`).

**Byte-equivalence for default (None) case:** When `warm_start_floor=None`, the `if self._warm_start_floor is None: return False` branch (line 868) preserves the original first-rung-fail-fast behavior. No change to existing k0 ladder behavior.

**Where `warm_start_floor` is used:**
- `solve_lambda_ramp_from_warm_start` (line 1909–1912): uses `warm_start_floor=0.0`. Correct — the warm-start IS the λ=0 state; the floor is 0.
- `solve_anchor_with_continuation`'s internal `lam_ladder` (line 1386): does NOT use `warm_start_floor`. This means first-rung failures in the lambda_hydrolysis_ladder path inside `solve_anchor_with_continuation` will NOT benefit from arithmetic bisection — they will raise `LadderExhausted` on first-rung failure.

**CONCERN C1 (minor):** There is an asymmetry: `solve_lambda_ramp_from_warm_start` uses `warm_start_floor=0.0` (the feature described in memory), but `solve_anchor_with_continuation`'s lambda ladder at line 1386 does NOT. Both code paths serve similar use cases. Callers expecting the step-9.5 adaptive floor behavior must use `solve_lambda_ramp_from_warm_start`, not the `lambda_hydrolysis_ladder=` arg to `solve_anchor_with_continuation`. This is not a bug if the intent is that the anchor path always starts from λ=0 floor (with a guaranteed k0 ladder there), but the asymmetry is undocumented and could surprise callers.

### 5. Stern Bump Ladder

**PASS.**

`set_stern_capacitance_model` (line 412):
- Updates `ctx['nondim']['bv_stern_capacitance_model']` (metadata, new dict copy — no mutation)
- Updates `ctx['stern_coeff_const'].assign(nondim_value)` (live UFL Constant)
- Physical → nondim conversion uses `ctx['nondim']['bv_stern_phys_to_nondim_factor']` stashed at form-build time.
- Raises `ValueError` if `c_s_f_m2 < 0` or `stern_coeff_const` absent.

The `c_s_ladder` path in `solve_anchor_with_continuation` (lines 1246–1316):
- Validates strictly *decreasing* sequence (correct: ramping from relaxed/large down to production C_S).
- Calls `set_stern_capacitance_model(ctx, cs_val)` at each rung.
- Then calls `_run_k0_ladder(...)` — a full k0 ladder is run at each C_S rung. Newton is re-run (not bypassed).
- On C_S rung failure: rolls back U to last successful C_S rung, raises `LadderExhausted`. No auto-bisection of C_S midpoints (documented: "Caller can densify `c_s_ladder` and retry").
- The verified rung schedule `{0.20, 0.50, 1.0, 2.0, 5.0, 10.0, 100.0}` from memory would need to be inverted (decreasing) for the ladder: `[100.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.20]`. The code validates decreasing monotonicity. Caller convention matches.

**One note:** The initial Stern value at form-build is whatever `stern_capacitance_f_m2` was set to in `sp`. The c_s_ladder starts by calling `set_stern_capacitance_model(ctx, cs_seq[0])`, overriding the build-time value. If the build-time Stern is NOT the starting value of the ladder, there is a one-step mismatch. Not a bug — the form is built with any valid positive Stern, and the ladder immediately overrides on rung 0 — but callers should be aware.

### 6. Newton Solver Options

**PASS.**

Solver options are inherited from `sp[10]` (the params block), filtered by `NON_PETSC_KEYS`, then only `snes_error_if_not_converged=True` is unconditionally forced (lines 1095–1106 and 1829–1840).

Production defaults from `_bv_common.py:SNES_OPTS_CHARGED`:
- `snes_type = "newtonls"` (Newton with line search)
- `snes_max_it = 300`
- `snes_atol = 1e-7`, `snes_rtol = 1e-10`, `snes_stol = 1e-12`
- `snes_linesearch_type = "l2"`, `snes_linesearch_maxlambda = 0.5`
- `snes_divergence_tolerance = 1e12`
- `ksp_type = "preonly"`, `pc_type = "lu"`, `pc_factor_mat_solver_type = "mumps"`

**No `--no-verify` escape hatches.** `snes_error_if_not_converged=True` is the only forced override; it tightens (not relaxes) convergence checking. The continuation orchestrator checks the `ok` boolean returned by `run_ss` and rolls back on failure — non-convergence is surfaced, not masked.

`solvers.py` post-cleanup contains only `_clone_params_with_phi`; all legacy continuation helpers (`forsolve_bv`, `solve_bv_with_continuation`, etc.) were removed in May 2026. The file is 21 lines and does not inject any solver options.

### 7. Failure Handling

**PASS.**

On Newton failure at any rung:
- **k0 ladder (line 1225):** Calls `k0_lad.record_failure_and_insert()`. If it returns True (midpoint inserted, budget not exhausted), rolls back U to `last_success_snap` and retries at the new midpoint scale. If False (budget exhausted or first-rung-no-floor), `_run_k0_ladder` returns `(False, k0_lad, snap)` to the caller, which raises `LadderExhausted`.
- **C_S ladder (line 1299):** Rolls back and raises `LadderExhausted` immediately (no auto-bisection of C_S steps).
- **λ ladder in solve_anchor_with_continuation (line 1477):** Calls `lam_ladder.record_failure_and_insert()`. If exhausted, raises `LadderExhausted`.
- **kw_eff ladder (line 1593):** Same pattern — bisects, rolls back, raises on exhaustion.
- **λ floor at λ=0 (lines 1373–1377 and 1887–1890):** Raises `LadderExhausted` immediately (no bisection at the floor — correct, since there's no prior state to interpolate from in the anchor path).

**CONCERN C2 (documentation gap):** `LadderExhausted` is raised mid-`with adj.stop_annotating():` context. The docstring says "The caller should catch this and inspect the partial result, which is *not* constructed." However, `AnchorContinuationResult` is not constructed on `LadderExhausted` — the exception propagates out. Callers catching `LadderExhausted` can inspect `ladder.history()` only if they hold a reference (they don't, in the main paths). This is acceptable for the MVP but means post-failure diagnostics are limited to the exception message string. Not a correctness issue.

### 8. `stop_annotating()` for Adjoint Hygiene

**PASS.**

Both `solve_anchor_with_continuation` (line 1245) and `solve_lambda_ramp_from_warm_start` (line 1856) wrap their entire continuation loop bodies in `with adj.stop_annotating():`. This matches the CLAUDE.md directive to wrap "unannotated cold-ramp / continuation work."

`firedrake.adjoint` is imported at the top of each function's body (lines 1029, 1724). The `with adj.stop_annotating():` context covers all rung solves, ladder walks, and rollback operations. No solve occurs outside this context manager within the continuation functions.

Strategy B (`solve_grid_with_charge_continuation`) is NOT present — it was removed in May 2026 per `bv_solver_unified_api.md §Architecture`. No invocation of Strategy B exists anywhere in `anchor_continuation.py`.

---

## Summary of Concerns

| ID | Severity | Description |
|----|----------|-------------|
| C1 | LOW | `warm_start_floor=0.0` only used in `solve_lambda_ramp_from_warm_start`, NOT in `solve_anchor_with_continuation`'s lambda ladder. Asymmetry is undocumented; callers expecting step-9.5 adaptive bisection must use the dedicated function. |
| C2 | LOW | `LadderExhausted` raised mid-context without constructing `AnchorContinuationResult`. Partial history only available in the exception message string, not a structured object. Acceptable for MVP; limits post-failure diagnostics. |

No blocking issues found.

---

## File Paths Referenced

- `/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/Forward/bv_solver/anchor_continuation.py`
- `/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/Forward/bv_solver/solvers.py`
- `/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/scripts/_bv_common.py` (SNES_OPTS_CHARGED, lines 281–284)
- `/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/Forward/bv_solver/grid_per_voltage.py` (solve_grid_with_anchor, line 875)
