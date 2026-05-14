# Correctness Audit — Pass 3 (Sonnet C-L3)
Date: 2026-05-13  
Files: `anchor_continuation.py`, `grid_per_voltage.py`, `observables.py`

---

## Q1. AdaptiveLadder midpoint math edge cases

**SEVERITY: LOW / BENIGN (prev=None + warm_start_floor=None path; geometric path)**

`record_failure_and_insert` with geometric midpoint:
```python
midpoint = math.sqrt(prev * scale)
```
If `prev > 0` and `scale > 0` this is safe. The constructor enforces all `initial_scales > 0` and `warm_start_floor < initial_scales[0]`, so `prev` is always a formerly-succeeded positive scale. No zero-product collapse is possible on the geometric path.

Arithmetic path (`warm_start_floor` set, first rung): `midpoint = 0.5 * (warm_start_floor + scale)`. Guard `if not (warm_start_floor < midpoint < scale): return False` correctly detects FP collapse when gap is below epsilon and returns False rather than inserting a degenerate rung. No infinite-loop risk.

FP collapse scenario worth noting: if `warm_start_floor=0.0` and `scale = 5e-324` (denormal), `0.5 * (0 + 5e-324) == 0.0` in IEEE-754 — the guard catches it (`0 < 0 < 5e-324` is False) and returns False correctly.

**No bug.**

---

## Q2. AdaptiveLadder.is_done() off-by-one

```python
def is_done(self) -> bool:
    return self._idx >= len(self._planned)
```
`record_success` increments `_idx` after appending. After the last rung's `record_success`, `_idx == len(_planned)`. `is_done()` returns True. Loop guard `while not lad.is_done()` exits correctly. No off-by-one.

**No bug.**

---

## Q3. max_inserts_per_step bookkeeping: per-rung or per-ladder

Docstring says per-step ("The insert counter is per-step: `record_success` resets it"). Implementation:

```python
def record_success(self):
    ...
    self._inserts_at_current_step = 0   # reset on success

def record_failure_and_insert(self):
    if self._inserts_at_current_step >= self._max_inserts_per_step:
        return False
    ...
    self._inserts_at_current_step += 1
```

Insertions accumulate for the current `(prev, scale)` gap. Once `record_success` fires (the inserted midpoint converged), the counter resets for the next gap. This is per-gap (per-step), not per-whole-ladder. Matches the documented intent.

**No bug.**

---

## Q4. NonlinearVariationalSolver creation — solver_parameters source/validation

In `solve_anchor_with_continuation` (line 1095-1106):
```python
params_block = sp[10] if hasattr(sp, "__getitem__") else {}
items = (params_block.items() if isinstance(params_block, dict) else [])
solve_opts = {k: v for k, v in items if k not in NON_PETSC_KEYS}
solve_opts.setdefault("snes_error_if_not_converged", True)
```

`NON_PETSC_KEYS = frozenset({"bv_bc", "bv_convergence", "nondim", "robin_bc"})` — only strips four known non-PETSc keys. Any non-PETSc key NOT in that set would be passed through to Firedrake and silently ignored or raise a PETSc error at solve-time, not at construction.

**SEVERITY: LOW / INFORMATIONAL.** No validation that `solve_opts` keys are actually valid PETSc options. An invalid key in `sp[10]` (not in `NON_PETSC_KEYS`) would surface as a cryptic PETSc error on first `solver.solve()` call, not at construction. This is a known limitation of PETSc's option handling — not a new bug introduced here. No actionable code defect.

In `_build_for_voltage` in `solve_grid_with_anchor` (line 1023-1030): identical pattern using `params` (the 11th element of the unpacked tuple). Same analysis applies.

---

## Q5. ctx['_last_solver'] cross-stage isolation (Stage 3 builds fresh ctx or reuses?)

Stage 1 (`solve_anchor_with_continuation`): builds one ctx, stores `ctx["_last_solver"] = solver`.

Stage 3 (`solve_grid_with_anchor` → `_build_for_voltage`): calls `build_context` + `build_forms` fresh per voltage, then `ctx["_last_solver"] = solver`. Each voltage gets its own ctx dict. The Stage 1 ctx is a different dict object. No aliasing.

The `_last_solver` key is only ever read by external diagnostic code. It is not read by `run_ss`, `warm_walk_phi`, or any ladder logic — those receive `solver` as an explicit parameter. Cross-contamination is impossible.

**No bug.**

---

## Q6. set_stern_capacitance_model unit conversion

```python
factor = float(scaling.get("bv_stern_phys_to_nondim_factor", 1.0))
nondim_value = float(c_s_f_m2) * factor
```

**SEVERITY: MEDIUM / LATENT.** If `bv_stern_phys_to_nondim_factor` is absent from `ctx["nondim"]` — which can happen if the context was built with an older `build_forms` path that did not populate this key — then `factor` defaults to `1.0` silently. The Constant would be set to the raw physical value (e.g. `0.20`) interpreted as nondimensional. This is incorrect, and the error is silent. There is no assertion or warning that the factor was actually populated.

**Trigger condition:** User calls `set_stern_capacitance_model` on a ctx built by a code path that does not write `bv_stern_phys_to_nondim_factor` to `nondim`. In the current production stack this factor is populated at form-build time, but the fallback to 1.0 is a correctness hazard for ctx objects produced by non-standard builders or unit tests that partially initialize `nondim`.

**Recommendation:** Assert `"bv_stern_phys_to_nondim_factor" in ctx.get("nondim", {})` and raise rather than defaulting silently, or at minimum emit a warning.

---

## Q7. snapshot_U / restore_U atomicity (both U and U_prev?)

```python
def _snapshot_U(U) -> tuple:
    return tuple(d.data_ro.copy() for d in U.dat)

def _restore_U(snap: tuple, U, U_prev) -> None:
    for src, dst in zip(snap, U.dat):
        dst.data[:] = src
    U_prev.assign(U)
```

`_snapshot_U` only snapshots `U`, not `U_prev`. `_restore_U` writes `U` then does `U_prev.assign(U)` to keep them consistent. This means: after restore, `U` and `U_prev` both hold the snapshot state. The SER time-stepper reads `U_prev` to compute `U - U_prev` for the implicit term; after restore both are identical, so the first Newton step at the new rung starts from a zero time-derivative initial condition. This is correct behavior — it avoids the time-stepper "seeing" a spurious large jump.

The atomicity concern: there is no lock between writing `U.dat` and assigning `U_prev`. In a single-threaded Python + Firedrake execution this is fine. In a parallel MPI context, `dst.data[:] = src` is a local write followed by `U_prev.assign(U)` which is a collective Firedrake call. The local writes happen before the collective — correct ordering.

**No bug.**

---

## Q8. warm_walk_phi._march recursion state restoration

```python
def _march(v0: float, v1: float, depth: int) -> bool:
    substeps = np.linspace(v0, v1, n_substeps + 1)[1:]
    ckpt_outer = _snapshot_U(U)
    v_prev_substep = float(v0)
    for v_sub in substeps:
        ckpt_inner = _snapshot_U(U)
        paf.assign(float(v_sub))
        if run_ss(max_ss_steps_per_substep):
            v_prev_substep = float(v_sub)
            continue
        _restore_U(ckpt_inner, U, U_prev)
        paf.assign(v_prev_substep)
        if depth >= bisect_depth:
            _restore_U(ckpt_outer, U, U_prev)
            paf.assign(float(v0))
            return False
        v_mid = 0.5 * (v_prev_substep + float(v_sub))
        if not _march(v_prev_substep, v_mid, depth + 1):
            _restore_U(ckpt_outer, U, U_prev)
            paf.assign(float(v0))
            return False
        if not _march(v_mid, float(v_sub), depth + 1):
            _restore_U(ckpt_outer, U, U_prev)
            paf.assign(float(v0))
            return False
        v_prev_substep = float(v_sub)
    return True
```

**SEVERITY: LOW / SUBTLE.** After a failed `_march(v_prev_substep, v_mid, ...)` the code restores `ckpt_outer` and assigns `paf` to `v0` before returning False. But `ckpt_outer` was taken at entry to **this** `_march` call (at the start of `v0→v1`). If the recursive call to `_march(v_prev_substep, v_mid, ...)` internally advanced `paf` and modified `U` before ultimately failing and restoring its own `ckpt_outer`, then this outer `ckpt_outer` correctly reverts to where we were at the start of the outer substep loop — this is the correct checkpoint.

One subtlety: `ckpt_inner` is snapshotted BEFORE `paf.assign(float(v_sub))`, i.e., at the state after the previous substep's `run_ss`. On `run_ss` failure, `_restore_U(ckpt_inner, U, U_prev)` puts U back to pre-substep-attempt state, then `paf` is re-assigned to `v_prev_substep`. This is correct — it rolls back the failed Newton state.

After the recursive bisection `_march(v_prev_substep, v_mid)` returns True AND `_march(v_mid, v_sub)` returns True, `v_prev_substep` is updated to `v_sub`. **But `ckpt_inner` at this point still holds the snapshot taken at the START of this substep (before either recursive call).** If the next substep loop iteration runs and a subsequent substep fails, `_restore_U(ckpt_inner, ...)` restores to a state from BEFORE the successful bisection, discarding the bisection's progress. However, `ckpt_inner` is re-assigned at the top of each iteration, so by the time we're at substep `v_sub_next`, `ckpt_inner` is refreshed. **No stale-checkpoint bug here** — `ckpt_inner` is always the state entering the current substep, which is the correct rollback point.

**SEVERITY: LOW / POTENTIAL BUG.** There is one case: when BOTH recursive marches succeed (`_march(v_prev, v_mid)` and `_march(v_mid, v_sub)` both True), `v_prev_substep = float(v_sub)` is set. The next substep iteration takes a fresh `ckpt_inner`. This is correct. No bug.

**Overall: No correctness bug in _march state restoration.**

---

## Q9. make_run_ss plateau detection (sign-flip false positives?)

```python
delta = abs(fv - prev_flux)
sv = max(abs(fv), abs(prev_flux), ss_abs_tol)
is_steady = (delta / sv <= ss_rel_tol) or (delta <= ss_abs_tol)
```

Sign-flip scenario: if the observable crosses zero (e.g., from +ε to −ε), `fv = -1e-9`, `prev_flux = +1e-9`. Then `delta = 2e-9`, `sv = max(1e-9, 1e-9, 1e-8) = 1e-8`. `delta/sv = 0.2 > ss_rel_tol = 1e-4`. `delta = 2e-9 < ss_abs_tol = 1e-8`. **`is_steady = True` fires on the abs floor condition** even though the observable has flipped sign and is still changing.

**SEVERITY: MEDIUM / LATENT BUG.** A sign-flip event where both `|fv|` and `|prev_flux|` are below `ss_abs_tol` satisfies the `delta <= ss_abs_tol` branch and gets counted as a plateau step. For the standard production stack (current density, which is negative cathodically and ~O(1e-2 to 1e-1) in nondim units at non-trivial voltages), the current density is rarely smaller than `ss_abs_tol = 1e-8`. Near V=0 (OCV) the current density can be ~0 by symmetry — at those points a sign-flip false positive is harmless (near-zero current genuinely is near-equilibrium). In practice this is unlikely to cause a false-convergence acceptance at a voltage where the physics is non-trivial. But the edge case exists.

---

## Q10. make_run_ss dt update (could dt grow unboundedly?)

```python
dt_max = float(dt_init) * float(dt_max_ratio)   # fixed ceiling

if ratio > 1.0:
    grow = min(ratio, dt_growth_cap)
    dt_val = min(dt_val * grow, dt_max)          # capped by dt_max
else:
    dt_val = max(dt_val * 0.5, float(dt_init))   # floored by dt_init
```

`dt_val` is capped at `dt_max = dt_init * dt_max_ratio` (default: `0.25 * 20 = 5.0`). Each call to `run_ss` resets `dt_val = float(dt_init)` at the top of `run_ss`, so there is no carry-over between rung calls. Within one call, `dt_val` is bounded above by `dt_max`.

**No unbounded growth bug.**

---

## Q11. U_prev.assign(U) placement (after solve, before next)

In `make_run_ss`:
```python
solver.solve()
U_prev.assign(U)
fv = float(fd.assemble(of_cd))
```

`U_prev` is updated immediately after each successful Newton solve, before the observable is assembled and before the next step. The implicit time-stepping scheme reads `U_prev` in the FE residual to compute `(U - U_prev)/dt`. Setting `U_prev = U` at end-of-step means the next step's residual starts from the just-converged state. Correct.

**No bug.**

---

## Q12. solve_grid_with_anchor source pool selection

```python
sources: list[tuple[float, tuple]] = [
    (float(anchor.phi_applied_eta), anchor.U_snapshot),
]
...
src_phi, src_snap = min(
    sources,
    key=lambda s: abs(float(s[0]) - target_phi),
)
```

Visit order is sorted by distance from anchor. Each successful solve appends its snapshot to `sources`. The `min` over `sources` picks the closest phi-neighbor from all previously-converged voltages. This is correct greedy nearest-neighbor — it will always find a better warm-start as more voltages succeed.

**Potential issue:** If two sources have the same distance from `target_phi` (e.g., symmetric grid), `min` returns the first one found (leftmost in list, which is the anchor or first-appended). Tie-breaking is deterministic but arbitrary. No correctness problem — either neighbor is equally valid.

**No bug.**

---

## Q13. per_point_callback gated on success

In `solve_grid_with_anchor` (line 1107-1113):
```python
if ok:
    snap = _snapshot_U(U)
    sources.append((target_phi, snap))
    method = f"warm<-{src_phi:+.3f}"
    converged = True
    if per_point_callback is not None:
        per_point_callback(orig_idx, target_phi, ctx)
```

Callback fires only on success. On failure, `converged = False` and callback is not invoked. This matches the documented contract ("On success: ... optional per_point_callback").

In `solve_grid_per_voltage_cold_with_warm_fallback` (Phase 1 at line 613, Phase 2 at lines 686, 734): same pattern — callback only on success. Consistent.

**No bug.**

---

## Q14. observables.py mode dispatch correctness

All four modes (`current_density`, `gross_h2o2_current`, `peroxide_current`, `reaction`) are dispatched via `mode_norm = str(mode).strip().lower()`. No ambiguity in the comparison chain — the four branches are mutually exclusive and exhaustive (else raises ValueError).

`gross_h2o2_current` defaults `idx = 0` when `reaction_index is None`. In the parallel preset, index 0 is R_2e (peroxide formation, E°=0.695 V) and index 1 is R_4e (water formation, E°=1.23 V). The docstring confirms "R_2e is at index 0 in the parallel preset." So `gross_h2o2_current` with default `reaction_index=None` correctly selects the 2e peroxide reaction, not the 4e water reaction.

**No bug.**

---

## NEW Q: record_failure_and_insert — immediate raise or deferred?

`record_failure_and_insert` returns `False` when exhausted. It does NOT raise `LadderExhausted`. The caller is responsible for raising:

```python
# In _run_k0_ladder:
if not k0_lad.record_failure_and_insert():
    return False, k0_lad, k0_last_ok_snap

# In the kw/lambda outer loops:
if not lam_ladder.record_failure_and_insert():
    raise LadderExhausted(...)
```

The `_run_k0_ladder` inner function returns False (does not raise). The outer callers of `_run_k0_ladder` inspect the bool and raise `LadderExhausted` themselves. The kw and lambda outer loops call `record_failure_and_insert` directly and raise immediately when it returns False.

**SEVERITY: LOW / INCONSISTENCY.** The k0 ladder exhaustion path in `_run_k0_ladder` returns False (silent), and the outer caller (`elif kw_eff_ladder is None` branch, line 1491) raises `LadderExhausted`. This is a two-step indirection vs. the direct raise in outer loops. It works correctly but makes the k0 exhaustion trace one stack frame deeper. The docstring for `solve_anchor_with_continuation` says `LadderExhausted` is raised mid-ladder — this is accurate regardless of which branch, since the outer code at line 1492 raises it. No correctness defect.

---

## NEW Q: Alternative ladder paths (c_s_ladder, kw_eff_ladder, lambda_hydrolysis_ladder) — cross-contamination

Each alternative path is gated by `if c_s_ladder is not None: ... elif lambda_hydrolysis_ladder is not None: ... elif kw_eff_ladder is None: ... else: ...` — mutually exclusive branches (enforced by the `NotImplementedError` checks at the top). Only one path runs per call.

**ctx state written by each path:**
- `c_s_ladder` path: `ctx["c_s_ladder_history"]`, calls `set_stern_capacitance_model` (mutates `stern_coeff_const`).
- `lambda_hydrolysis` path: `ctx["lambda_hydrolysis_ladder_history"]`, calls `set_reaction_lambda_hydrolysis_model`.
- `kw_eff` path: `ctx["kw_eff_ladder_history"]`, calls `set_reaction_kw_eff_model`.
- bare k0 path: writes nothing extra.

Since only one branch executes, there is no cross-contamination of ctx state between paths. A ctx produced by the `c_s_ladder` path will have `stern_coeff_const` mutated to the final ladder value (the production C_S), which is correct.

**No cross-contamination bug.**

---

## NEW Q: _grab / orig_idx mapping through visit_order sort

There is no `_grab` callback in the codebase. The question referred to `per_point_callback(orig_idx, target_phi, ctx)` in `solve_grid_with_anchor`. In the loop:

```python
for visit_n, orig_idx in enumerate(visit_order):
    target_phi = float(phi_applied_values[orig_idx])
    ...
    if ok:
        ...
        per_point_callback(orig_idx, target_phi, ctx)
    ...
    points[orig_idx] = PerVoltagePointResult(orig_idx=orig_idx, ...)
```

`orig_idx` comes directly from `visit_order[visit_n]`, which is an index into `phi_applied_values`. `phi_applied_values[orig_idx]` gives the correct voltage for that original index. `points[orig_idx]` stores the result at the correct key. The callback receives the true `orig_idx`, not `visit_n`. If `visit_order = [3, 1, 0, 2]`, then at `visit_n=0`, `orig_idx=3`, `target_phi = phi_applied_values[3]`, `points[3]` is set — all correct.

**No index mapping bug.**

---

## NEW Q: snapshot_U — copy or shared reference?

```python
def _snapshot_U(U) -> tuple:
    return tuple(d.data_ro.copy() for d in U.dat)
```

`d.data_ro` is a read-only view of the underlying PETSc/numpy array. `.copy()` creates an independent numpy array. The returned tuple contains independent numpy arrays — not views into Firedrake's internal buffers. Subsequent `U.dat` mutations (via `solver.solve()`) do not affect the snapshot.

**No shared-reference mutation risk.**

---

## NEW Q: Firedrake form kernel caching vs. Constant mutation (Stern bump)

Firedrake caches compiled form kernels keyed on the UFL structure of the form, not on Constant values. At solve time, `stern_coeff_const` (a `Constant`) is read by the kernel from its current value. Reassigning via `stern_const.assign(nondim_value)` updates the value in-place in the Constant's internal storage. The next `solver.solve()` reads the new value.

This is the standard Firedrake pattern for parametric solving and is correct. No recompilation is triggered by `Constant.assign`. The form kernel is reused with the updated coefficient.

**No bug.**

---

## NEW Q: extract_preconverged_anchor — snapshot before or after Stern mutation?

Stage 2 (Stern bump loop via `set_stern_capacitance_model`) mutates `ctx["stern_coeff_const"]` and `ctx["nondim"]["bv_stern_capacitance_model"]`. The bump loop also calls `run_ss` (the k0 ladder) at each C_S rung, and `snapshot_U` is taken after the k0 ladder succeeds at each C_S rung. By the time all C_S rungs complete and `converged_to_target = True`, `last_success_snap` holds the snapshot from the LAST successful k0 ladder run (at the final, lowest C_S target).

`extract_preconverged_anchor` is called by the user AFTER `solve_anchor_with_continuation` returns. It takes `result.U_data`, which is `last_success_snap` (set at line 1611 when `converged_to_target`). This snapshot was taken AFTER the full bump ladder completed — i.e., at the final production C_S. Correct.

**No ordering bug.**

---

## NEW Q: observables.py gross_h2o2_current — reaction index verification

```python
elif mode_norm == "gross_h2o2_current":
    idx = 0 if reaction_index is None else int(reaction_index)
    ...
    return scale_const * bv_rate_exprs[idx] * ds(electrode_marker)
```

`bv_rate_exprs` is `list(ctx["bv_rate_exprs"])`. In the parallel 2e/4e preset (`PARALLEL_2E_4E_REACTIONS`), reaction ordering is: index 0 = R_2e (peroxide, E°=0.695 V), index 1 = R_4e (water, E°=1.23 V). This is confirmed by the docstring ("R_2e is at index 0 in the parallel preset and was R_0 in the legacy sequential preset"). Default `idx=0` correctly selects R_2e.

**No bug.**

---

## VERDICT

| # | Finding | Severity |
|---|---------|----------|
| Q6 | `set_stern_capacitance_model` silently defaults `bv_stern_phys_to_nondim_factor=1.0` if the key is absent from `nondim` — wrong nondim value applied with no warning | MEDIUM |
| Q9 | `make_run_ss` plateau detection: `delta <= ss_abs_tol` branch can accept a sign-flip oscillation as "steady" when both `|fv|` and `|prev_flux|` are sub-threshold | MEDIUM / LOW-RISK in practice |
| Q4 | `solve_opts` passes non-PETSc keys through silently if not in `NON_PETSC_KEYS` set | LOW / INFORMATIONAL |
| NEW-Q (k0 exhaustion) | Two-step indirection for k0 ladder exhaustion (returns False, outer raises) vs. direct raise in other paths — inconsistent but functionally correct | LOW / INFORMATIONAL |
| All others | No correctness defect found | — |

**VERDICT: TWO MEDIUM, TWO INFORMATIONAL. No HIGH/CRITICAL bugs. Functional correctness is solid for the production demo (k0 ladder only, no c_s/kw/lambda ladders). The MEDIUM items are latent hazards relevant to corner cases or non-standard ctx builders, not to the main happy path.**
