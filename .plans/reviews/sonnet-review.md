# Plan Review

**Verdict:** NEEDS REVISION
**Reviewer:** Sonnet

## Summary

The plan is architecturally sound and well-reasoned. It correctly identifies the key differences from v15, properly specifies the two-script design, and has a sensible verification plan. However, it contains one critical MAJOR issue — the "fast path" design directly contradicts the API contract of `solve_grid_with_charge_continuation`, which operates across the entire voltage grid in a single call and cannot be used as a per-point solver. Beyond this, there are three additional MAJOR issues: the adjoint methodology is technically wrong (loading saved U_data into a fresh context and running one SNES step does NOT put pyadjoint on the tape correctly), the IC cache update path from workers back to the main process has an unresolved IPC gap in spawn-based multiprocessing, and the plan's scope (2 new scripts, ~800-1200 lines, 5 subsystems) is too large for reliable single-agent execution without explicit phasing.

---

## Major Issues

### MAJOR 1: Fast Path Violates `solve_grid_with_charge_continuation` API Contract

**What:** The plan proposes a "fast path" where workers check the IC cache, load ICs into a context, and run a warm-started PTC solve at each voltage point individually (with `_WARMSTART_MAX_STEPS = 20`). This is a per-point loop over voltage points with cached initial conditions.

**Why it matters:** `solve_grid_with_charge_continuation` is explicitly NOT a per-point function. It runs Phase 1 (neutral sweep) once across ALL voltage points, then Phase 2 (z-ramp) per point — all internally. The function accepts the full `phi_applied_values` array and returns a `GridChargeContinuationResult` with all points. There is no supported way to call it for a single voltage point. The plan's fast path reimplements an ad-hoc per-point solver that bypasses the unified interface entirely — reproducing exactly the pattern the user wants to eliminate from v15. The worker-level fast path as described is not compatible with the unified function and would require directly driving the internal `_run_to_steady_state` loop, which is not part of the public API.

**Suggested fix:** Remove or redesign the fast path to operate at the sample level, not the voltage-point level. The correct warm-start strategy is: when the IC cache has a nearby entry, pre-seed the initial conditions before calling `solve_grid_with_charge_continuation` on the full voltage grid. This can be done by passing `U_data` arrays as pre-loaded state, or by patching `set_initial_conditions` behavior via `solver_params`. The "fast path" benefit is then realized through reduced Phase 1 iteration counts due to a better starting point, not by bypassing Phase 2. Alternatively, acknowledge that the fast path is a deliberate lower-level optimization that bypasses the unified API, implement it as a clearly-named separate function, and document why this is acceptable.

---

### MAJOR 2: Adjoint Gradient Methodology is Technically Incorrect

**What:** The deferred adjoint script (Step 3b) proposes: "1. Build context, load converged U_data from .npz. 2. Assign k0/alpha as tape-tracked fd.Function controls. 3. Tape pass 1: J_CD = fd.assemble(cd_observable_form); rf = ReducedFunctional(J_CD, controls); grad_cd = rf.derivative()"

**Why it matters:** `pyadjoint` / `firedrake-adjoint` computes gradients by replaying the recorded tape of operations that produced the functional from the controls. Simply assembling an observable form from a pre-loaded state does NOT put the correct sequence of PDE solves on the tape — pyadjoint has no record of how U_data depends on (k0, alpha). Calling `ReducedFunctional(J_CD, controls).derivative()` in this context will either raise an error, or return zeros because the solve (the computational graph connecting controls to outputs) was never taped. A correct adjoint computation requires: (1) creating the controls and registering them with pyadjoint, (2) building forms, (3) calling `solver.solve()` while pyadjoint is recording, and (4) then assembling the functional and computing derivatives. The deferred adjoint pass MUST re-solve — it cannot reconstruct the gradient from snapshots alone.

**Suggested fix:** Revise the adjoint script to: enable `fd.annotate_tape()` (via `from firedrake_adjoint import *`), build the context with k0/alpha as `fd.Control`-wrapped functions, load U_data as the initial guess (warm-start), run `solver.solve()` with annotation enabled (from a converged warm-start, this converges in ~1 Newton step), then assemble the functional and call `rf.derivative()`. This is the standard Firedrake adjoint pattern. Add to the verification plan: validate this warm-start-then-solve approach produces correct tape annotation before implementing the batch loop.

---

### MAJOR 3: IC Cache Update Path from Workers is Unspecified

**What:** The plan says the IC cache "lives in the main process" and workers "receive relevant entries via task arguments." Workers also need to update the cache with newly converged solutions. Step 4 says `ICCacheManager.store()` is called from workers, but in spawn-based multiprocessing, workers are separate processes — they cannot directly mutate the main process's cache object.

**Why it matters:** This is a concurrency architecture gap. If workers call `cache.store()` locally, they update their own in-process copy (which is discarded when the group completes). The main process cache never grows. The fast-path hit rate will be 0% after the first group because no new entries are ever added to the main cache. This defeats the entire purpose of the global IC cache.

**Suggested fix:** Specify explicitly that workers return their converged U_data tuples alongside observables in the result dict (already done for the adjoint pass). The main process updates the IC cache from these return values after each group completes. This is the correct spawn-safe pattern — synchronous, no shared memory, no race conditions. The plan should add a concrete code sketch showing: `for r in group_results: if r["converged_u_data"] is not None: cache.store(r["k0_1"], ..., r["converged_u_data"])`.

---

### MAJOR 4: Plan Scope is Too Large for Single-Agent Execution

**What:** Two new scripts totaling ~800-1200 implementation lines across 5 interconnected subsystems (sampling, parallel execution, IC cache, checkpoint I/O, Firedrake adjoint). The plan flagged this risk itself.

**Why it matters:** The adjoint methodology issue (MAJOR 2) is an example of where implementation-level mistakes produce silently wrong results. The forward script would run fine, the adjoint script would produce zero gradients, and this might not be caught until surrogate training shows poor gradient predictions. At this scope, the probability of a subtle bug that is hard to debug post-hoc is high.

**Suggested fix:** Execute in two sequential phases:
- **Phase A**: Forward script only (`overnight_train_v16.py`) — IC cache, parallel workers, checkpointing, validated by smoke test and resume test before running overnight.
- **Phase B**: Adjoint script (`compute_adjoint_gradients_v16.py`) — implemented and validated separately, starting with the finite-difference comparison on a single point before the full batch loop.

This split is already implicit in the two-script design; make it explicit in the execution instructions.

---

## Minor Issues

- **Relaxed retry SNES params**: The retry uses `snes_max_it=500, atol=1e-6, rtol=1e-6`. The SNES options live inside `solver_params` (the 11-element list, element index 10). The plan should note that the retry requires rebuilding `solver_params` with the relaxed options, not mutating the existing object — the frozen dataclass pattern used in `_bv_common.py` makes this non-obvious.

- **U_data storage size is potentially prohibitive**: For a 42-point grid, 4-species system on an 8×200 mesh (DOF count ~9,600), each sample's U_data is approximately 42 × 4 × 9,600 × 8 bytes ≈ 12 MB uncompressed. At 350 valid samples per batch, that is ~4 GB of solution data per batch. The plan acknowledges size as a concern but does not quantify or mitigate it. Consider computing gradients inline via `per_point_callback` during the forward solve (eliminating the need to store U_data entirely) or storing only a subset of voltage points.

- **KDTree rebuild cost**: The plan says "cache entries are never evicted (memory bounded by total samples)." Rebuilding a KDTree over N entries is O(N log N). At 5,000 entries this is negligible; at 50,000+ (many overnight batches) it becomes noticeable. Mention a rebuild-every-N-entries strategy or use a data structure that supports incremental updates.

- **`per_point_callback` usage should be committed**: The critical context note identifies `per_point_callback` as the supported observable extraction mechanism. The plan should explicitly state that the slow path uses `per_point_callback` to extract observables inline rather than re-assembling from stored U_data. This is mentioned obliquely but not committed as the implementation approach.

- **`generate_multi_region_lhs_samples` returns physical-space k0**: The context note correctly flags that k0 must be divided by `K_SCALE` before passing to `make_bv_solver_params`. This is critical — the plan should include the nondimensionalization explicitly in the Step 1 constants block (e.g., `k0_hat_r1 = k0_1_phys / K_SCALE`) to prevent implementers from passing physical-space values directly.

- **Voltage grid**: The plan says "same expanded grid as v15 (42 unique eta points)" but does not reproduce the `_build_voltage_grid()` code. Copy it verbatim from v15 to prevent transcription errors.

- **Group size / checkpoint I/O interaction**: GROUP_SIZE=5 means checkpointing every 40 total samples (8 workers × 5). If U_data is stored per checkpoint (4 GB per batch), each checkpoint write is ~320 MB. At ~14 seconds per sample and 8 parallel workers, a group completes every ~9 seconds on the fast path. This could produce a checkpoint write every ~9 seconds — potentially slower than the solve itself. Ensure checkpointing is non-blocking or deferred.

---

## Strengths

- The two-script design correctly separates concerns: forward throughput vs. gradient computation. This is architecturally elegant and lets overnight runs maximize PDE solve throughput.
- Strict 100% convergence validation is clearly specified and correctly justified — the plan explicitly rejects partial convergence without interpolation, which is the right scientific decision.
- The checkpoint system with atomic `os.replace()` writes matches v15's proven pattern and is correct.
- Graceful signal handling (SIGINT/SIGTERM → clean checkpoint) is a meaningful operational improvement over v15.
- The IC cache distance metric (log-space k0, linear alpha) is appropriate for the parameter space geometry.
- The verification plan is comprehensive and includes finite-difference validation of adjoint gradients — this is the most important test and must not be skipped.
- Solver configuration matches v13 paper parameters and is specified concretely rather than left as placeholders.
- The plan correctly identifies that `GridChargeContinuationResult` uses `.all_converged()` and `.get_U_data(idx)` as the validation/extraction API.
