# Correctness Audit: Adaptive Ladder, Warm Walk, SER, Snapshots

Scope constraint followed: only the requested source ranges were read. Where a behavior depends on code outside those ranges, it is marked UNVERIFIED rather than inferred.

## Findings

SEVERITY: HIGH
LOCATION: Forward/bv_solver/anchor_continuation.py:829-887
TRIGGER: A failed original rung gets an inserted midpoint; the midpoint succeeds; the original rung fails again; this repeats.
EVIDENCE: `record_success()` advances `_idx` and resets the insert counter: lines 833-836, `"self._idx += 1"` and `"self._inserts_at_current_step = 0"`. `record_failure_and_insert()` inserts at the current index without advancing it: lines 860-887, especially `"self._planned.insert(self._idx, float(midpoint))"` and `"self._inserts_at_current_step += 1"`. Therefore consecutive insert-of-insert failures share the counter only until a success. Once an inserted midpoint succeeds, the counter resets and the original failing rung is retried with a fresh insert budget. The claimed bound `5 * (4 + 1) = 25` is false; finite `initial_scales` plus finite `max_inserts_per_step` does not by itself prove finite termination.

SEVERITY: HIGH
LOCATION: Forward/bv_solver/anchor_continuation.py:864-887
TRIGGER: `prev` and `scale` are close enough that `math.sqrt(prev * scale)` rounds to `prev` or `scale`, or `prev * scale` underflows to `0.0`.
EVIDENCE: The warm-start arithmetic branch checks strict progress at lines 874-880: `"if not (self._warm_start_floor < midpoint < scale): return False"`. The geometric branch has no equivalent guard: lines 884-887 are only `"midpoint = math.sqrt(prev * scale)"`, insert, increment, return. If the midpoint collapses to `prev`, a duplicate successful rung can reset the counter and feed the nontermination pattern above. If it collapses to `scale`, a duplicate failing rung is inserted. If the product underflows to zero, a nonpositive rung can be inserted even though constructor validation only covered the original `initial_scales` at lines 771-784.

SEVERITY: MEDIUM
LOCATION: Forward/bv_solver/anchor_continuation.py:794-800,874-881
TRIGGER: `AdaptiveLadder(..., warm_start_floor=-1.0)` with a small positive first scale.
EVIDENCE: The constructor only requires `"warm_start_floor must be strictly less than initial_scales[0]"` at lines 794-800. The arithmetic branch then computes `"midpoint = 0.5 * (self._warm_start_floor + scale)"` at line 874 and inserts it if it lies between floor and scale at lines 879-881. A negative floor can therefore insert a negative scale. This conflicts with the class contract that scales are positive and with downstream `set_reaction_k0_model()` rejecting `k0_model_value <= 0.0` at lines 281-285.

SEVERITY: MEDIUM
LOCATION: Forward/bv_solver/anchor_continuation.py:299-310,459-467
TRIGGER: The metadata update succeeds but the live Firedrake coefficient assignment raises.
EVIDENCE: `set_reaction_k0_model()` updates `ctx["nondim"]` before assigning the live function: lines 304-310 set `new_rxns`, assign `ctx["nondim"] = nondim`, then call `"funcs[j].assign(float(k0_model_value))"`. Its own docstring warns at lines 256-257: `"Either failing leaves an inconsistent state"`. `set_stern_capacitance_model()` has the same pattern: lines 462-464 update `ctx["nondim"]`, then line 467 calls `"stern_const.assign(nondim_value)"`. There is no rollback path.

SEVERITY: MEDIUM
LOCATION: Forward/bv_solver/grid_per_voltage.py:181-213
TRIGGER: The observable current has four consecutive small deltas while other state components are still changing, for example near a cancellation or zero crossing.
EVIDENCE: SER steady state is detected from one scalar observable only: line 193 `"fv = float(fd.assemble(of_cd))"`, lines 195-198 compute `delta`, `sv`, and `"is_steady = (delta / sv <= ss_rel_tol) or (delta <= ss_abs_tol)"`, and lines 211-212 return true once `"steady_count >= ss_consec"`. The default `ss_consec=4` at line 145 mitigates one-step zero-crossing noise, but the visible code has no check on `||U - U_prev||`, residual norm, or multi-observable consistency.

SEVERITY: LOW
LOCATION: Forward/bv_solver/grid_per_voltage.py:187-192
TRIGGER: `solver.solve()` raises after mutating `U` internally.
EVIDENCE: The exception path is lines 188-191: `"solver.solve()"` inside `try`, then `except Exception: return False`. There is no restore inside `run_ss()`. Callers must snapshot and roll back. `warm_walk_phi()` does this around substeps at lines 282 and 287, but the k0-ladder failure handling after line 1200 was outside the allowed read range and is UNVERIFIED.

SEVERITY: MEDIUM
LOCATION: Forward/bv_solver/observables.py:123-135
TRIGGER: `mode="gross_h2o2_current"` is used without `reaction_index` on a reaction list whose 2e peroxide-forming reaction is not index 0.
EVIDENCE: The code defaults to index 0 at line 129: `"idx = 0 if reaction_index is None else int(reaction_index)"`, bounds-checks only length at lines 130-134, and returns `"bv_rate_exprs[idx]"` at line 135. The only evidence that index 0 is R_2e is the comment at lines 124-128. The code does not validate reaction identity or `n_electrons` metadata.

SEVERITY: LOW
LOCATION: Forward/bv_solver/observables.py:110-121
TRIGGER: Heterogeneous 2e/4e reactions are present but `ctx["nondim"]["bv_reactions"][j]["n_electrons"]` is missing, malformed, or length-mismatched.
EVIDENCE: Lines 110-116 correctly weight reactions when `n_e_list` is present and length-matched. Otherwise lines 117-120 silently fall back to `"rate_sum = rate_sum + R_j"`. The file header explicitly says the fallback emits no warning at lines 20-21. For mixed electron counts, this can underweight 4e reactions.

SEVERITY: LOW
LOCATION: Forward/bv_solver/grid_per_voltage.py:225-226,271-310
TRIGGER: `n_substeps <= 0`, negative `bisect_depth`, or a floating-point-degenerate voltage interval is supplied.
EVIDENCE: The visible `warm_walk_phi()` signature accepts `n_substeps` and `bisect_depth` at lines 225-226, but no validation appears in lines 218-310. `_march()` uses `np.linspace(v0, v1, n_substeps + 1)[1:]` at line 278 and compares `depth >= bisect_depth` at line 289. With `n_substeps=0`, `_march()` performs no substeps and the code jumps directly to the final solve at lines 307-310. With a negative depth cap, the first failure immediately exhausts bisection. With midpoint collapse, recursion remains finite because of the depth cap, but the code may spend solves on degenerate intervals.

SEVERITY: INFO
LOCATION: Forward/bv_solver/grid_per_voltage.py:279-300,1083-1087; Forward/bv_solver/anchor_continuation.py:117-120,1033,1125
TRIGGER: Need to prove `snapshot_U` / `restore_U` atomicity and whether `U_prev` is restored.
EVIDENCE: The function bodies for `snapshot_U`, `restore_U`, `_snapshot_U`, and `_restore_U` were outside the allowed read ranges. Visible evidence: `PreconvergedAnchor` documents `U_snapshot` as `"tuple(d.data_ro.copy() for d in U.dat)"` at lines 117-120; `solve_anchor_with_continuation()` imports `snapshot_U, restore_U` at line 1033 and takes `"last_success_snap = snapshot_U(ctx["U"])"` at line 1125; warm-walk restore call sites pass both targets, for example `"_restore_U(ckpt_inner, U, U_prev)"` at line 287 and source restore at line 1085. The call signature strongly suggests intent to restore both `U` and `U_prev`, but the implementation and all-or-nothing atomicity are UNVERIFIED.

SEVERITY: INFO
LOCATION: Forward/bv_solver/grid_per_voltage.py:1038-1088
TRIGGER: Need to prove dynamic source-pool growth during `solve_grid_with_anchor()`.
EVIDENCE: The code sorts `visit_order` once by anchor distance at lines 1038-1044, initializes `sources` with the anchor at lines 1046-1050, and selects source by nearest absolute voltage distance at lines 1077-1081. The actual success path that should append a new `(phi, snapshot)` source is after line 1100 and was outside the allowed read range. The docstring says successful snapshots become candidate neighbours at lines 919-921, but the implementation of that append is UNVERIFIED.

## Q1-Q9 Answers

### Q1. AdaptiveLadder termination proof

The finite-termination claim is not proven and the `5 * (4 + 1) = 25` worst-case bound is false.

Consecutive failures before any success do share `_inserts_at_current_step`: `record_failure_and_insert()` does not advance `_idx`, appends a failure at lines 860-861, checks the cap at lines 862-863, inserts a midpoint at lines 881 or 885, and increments at lines 882 or 886. A nested insert-of-an-insert therefore shares the counter only as long as no inserted rung succeeds.

However, every success resets the counter at lines 829-836. If a midpoint succeeds, `_idx` advances and the original failed rung becomes current again with `_inserts_at_current_step = 0`. That allows an unbounded sequence in exact arithmetic: fail original scale `s`, insert midpoint, succeed midpoint, fail original `s`, insert a closer midpoint, succeed, repeat. The class has no per-original-rung or global insert budget.

### Q2. AdaptiveLadder midpoint correctness

For the default numeric example, `prev=1e-12` and `scale=1e-9` gives `sqrt(1e-21) = 3.1622776601683794e-11`, which is correctly between the endpoints in ordinary fp64 arithmetic.

The literal `prev=0` geometric case is avoided for valid initial scales because constructor validation rejects `initial_scales` entries `<= 0.0` at lines 771-777, and the first rung has `previous_scale is None` at lines 819-823. The warm-start first-rung branch uses an arithmetic midpoint at line 874 and guards strict progress at lines 879-880.

The geometric branch lacks the same guard. Lines 884-887 insert `math.sqrt(prev * scale)` unconditionally. Therefore fp64 collapse to `prev`, collapse to `scale`, or underflow to zero is not checked. The code should check `prev < midpoint < scale` before insert and fail the ladder if strict progress is impossible.

### Q3. warm_walk_phi recursion correctness

For `_march(v0=v_anchor, v1=v_target, depth=0)` with `n_substeps=8`, line 278 builds `np.linspace(v0, v1, n_substeps + 1)[1:]`, so the loop sees eight target-side substeps. Line 280 initializes `v_prev_substep = float(v0)`.

If substep 5 fails, line 282 has already taken `ckpt_inner`, line 283 assigned `paf` to `v_5`, and line 284 called `run_ss()`. On failure, line 287 restores `U`/`U_prev` through `_restore_U(ckpt_inner, U, U_prev)`, and line 288 reassigns `paf` to `v_prev_substep`. If depth is still allowed, line 293 uses the arithmetic midpoint `0.5 * (v_prev_substep + float(v_sub))`, line 294 marches first half, and line 298 marches second half. After both recursive calls succeed, line 302 updates `v_prev_substep = float(v_sub)`. Therefore the parent loop proceeds to substep 6 with `v_prev_substep` set to the now-successful substep 5.

State restore: immediate substep failure restores `ckpt_inner` at line 287 and reasserts `paf` at line 288. Exhausted depth restores `ckpt_outer` and `paf=v0` at lines 290-292. Failed first or second recursive half restores `ckpt_outer` and `paf=v0` at lines 295-300. `ckpt_outer` is the snapshot at the start of the current `_march()` call, line 279; `ckpt_inner` is the snapshot at the start of one substep, line 282.

The outer caller does not visibly reassign `paf` after `_march()` returns false; lines 305-306 immediately return false. The visible `_march()` failure exits reassign `paf` before returning. Final-SS failure handling after line 310 is outside the allowed range and is UNVERIFIED.

The body of `_restore_U` is outside the allowed range, so the actual restoration of both `U` and `U_prev` is UNVERIFIED even though all visible restore call sites pass both objects.

### Q4. solve_grid_with_anchor source-pool dynamics

The source selection is nearest-neighbor among the current `sources` list. Lines 1038-1044 create a one-time `visit_order` by distance from the anchor. Lines 1046-1050 initialize `sources` with the anchor. Lines 1077-1081 select:

```python
src_phi, src_snap = min(
    sources,
    key=lambda s: abs(float(s[0]) - target_phi),
)
```

Therefore, if `+0.11` has already been appended to `sources`, target `+0.07` selects `+0.11` over the anchor. If `+0.03` has already been appended, target `-0.01` selects `+0.03`. It is not first-source or anchor-only selection.

The actual append of newly converged grid points is not visible in the allowed 875-1100 range. The docstring says this happens at lines 919-921, and the comment at lines 1046-1047 says successful grid solves are appended, but the implementation is UNVERIFIED.

### Q5. `ctx["_last_solver"]` across stages

Stage 1 solver storage is verified. Lines 1101-1103 build `fd.NonlinearVariationalProblem(ctx["F_res"], ctx["U"], bcs=ctx["bcs"], J=ctx["J_form"])`; lines 1104-1106 build the solver; line 1107 stores it as `ctx["_last_solver"]`.

The live Stern mutation path is verified. `set_stern_capacitance_model()` updates metadata at lines 459-464 and assigns the live Constant at line 467. The comment at lines 466-467 says this makes the residual reflect the new value. The Stage 2 demo loop itself was not in the allowed ranges, so the exact loop is UNVERIFIED, but if it mutates `stern_coeff_const` and calls the stored solver, the visible code is consistent with Firedrake's live-coefficient reference semantics.

Stage 3 fresh-build behavior is verified. `_build_for_voltage()` creates a new `sp_v`, `ctx = build_context(...)`, and `ctx = build_forms(...)` at lines 1011-1013. It builds a fresh `NonlinearVariationalProblem` and `NonlinearVariationalSolver` at lines 1026-1031, then stores that fresh solver in the fresh ctx at line 1032. No alias back to the Stage 1/2 solver is visible.

### Q6. `snapshot_U` / `restore_U` atomicity

The implementation is UNVERIFIED because the function bodies are outside the allowed ranges. The strongest visible evidence is the `PreconvergedAnchor` docstring at lines 117-120, which describes `U_snapshot` as `tuple(d.data_ro.copy() for d in U.dat)`, and the call sites.

Visible call sites pass both `U` and `U_prev` to restore: `warm_walk_phi()` calls `_restore_U(ckpt_inner, U, U_prev)` at line 287 and repeats that shape at lines 290, 295, and 299; `solve_grid_with_anchor()` calls `_restore_U(src_snap, U, U_prev)` at line 1085. This strongly suggests intended restoration of both `U` and `U_prev`.

But whether `restore_U` actually restores both, whether it validates per-subspace shapes, and whether it is atomic on partial failure cannot be proven from the allowed spans. If `U_prev` is not restored, the next SER solve can see stale transient history because `make_run_ss()` only assigns `U_prev.assign(U)` after a successful `solver.solve()` at line 192.

### Q7. make_run_ss SER plateau detection

The plateau formula is exactly visible at lines 193-198:

```python
fv = float(fd.assemble(of_cd))
delta = abs(fv - prev_flux)
sv = max(abs(fv), abs(prev_flux), ss_abs_tol)
is_steady = (delta / sv <= ss_rel_tol) or (delta <= ss_abs_tol)
```

`ss_consec=4` by default at line 145, `steady_count` is initialized at line 186, incremented/reset at line 198, and accepted at lines 211-212. This protects against a single accidental small delta at a zero crossing. It does not prove physical steady state if the scalar observable stalls for four steps while hidden state variables continue to change.

There is a dt cap. Line 179 sets `dt_max = dt_init * dt_max_ratio`, and line 203 applies `dt_val = min(dt_val * grow, dt_max)`. If Newton converges but the plateau never hits, the loop stops at `max_steps` from line 187 and returns false at line 213.

`U_prev.assign(U)` is placed after `solver.solve()` and before the next iteration: line 189 solves, line 192 assigns `U_prev`, and the loop continues. That placement is correct for using the converged state as the next transient reference.

### Q8. observables.py BV observable forms and signs

For `mode="current_density"`, lines 113-116 compute `weight = fd.Constant(float(n_e_j) / ref)` with `ref = float(N_ELECTRONS_REF)`. `N_ELECTRONS_REF = 2` at line 43. Thus `n_e=2` gives weight `1.0`, and `n_e=4` gives weight `2.0`. The returned form is `scale_const * rate_sum * ds(electrode_marker)` at line 121.

For `mode="gross_h2o2_current"`, the default index is 0 at line 129, and line 135 returns `scale_const * bv_rate_exprs[idx] * ds(electrode_marker)`. Comments at lines 124-128 state that index 0 is R_2e in the parallel preset and legacy sequential preset. The code does not validate that index against reaction metadata, so the identity of R_2e is comment-verified only from the allowed file.

The visible sign chain is: with positive `scale`, positive `R_j` gives a positive assembled observable because weights and `scale_const` are positive in lines 104, 115, and 121. If an external caller passes `scale=-I_SCALE`, the form becomes negative for positive forward/cathodic rates. That external demo wrapping and `_to_json_list` conversion to `mA/cm^2` are not in the allowed ranges, so whether positive `cd_mA_cm2` corresponds to cathodic current by standard convention is UNVERIFIED. `assemble_observable_validated()` preserves the assembled sign at line 198 and only uses `abs(value)` for diffusion-limit validation at lines 207-209.

### Q9. Other algorithmic or state-management issues

Additional issues found in the targeted ranges:

- `set_reaction_k0_model()` and `set_stern_capacitance_model()` are not atomic across metadata and live Firedrake coefficient layers. Evidence and trigger are in the MEDIUM finding above.
- `run_ss()` catches solver exceptions and returns false without restoring potentially mutated `U`; callers must handle rollback. `warm_walk_phi()` does; k0-ladder failure rollback is beyond line 1200 and therefore UNVERIFIED.
- `AdaptiveLadder` accepts negative `warm_start_floor` values that can insert negative scales.
- `warm_walk_phi()` does not visibly validate `n_substeps` or `bisect_depth`.
- `gross_h2o2_current` silently assumes R_2e is index 0 unless the caller overrides `reaction_index`.
- `current_density` silently falls back to unweighted reaction sums when electron-count metadata is unavailable.
- `solve_grid_with_anchor()` selects nearest source correctly among current sources, but the actual append of successful sources is outside the allowed range and UNVERIFIED.

VERDICT: FAIL - The ladder termination/bounded-iteration claim is false, the geometric midpoint branch lacks a strict-progress guard, and key snapshot/source-pool behaviors needed for a complete proof are outside the allowed evidence windows.
