# Plan Review

**Verdict:** NEEDS REVISION
**Reviewer:** Opus

## Summary

The plan is well-structured and demonstrates strong understanding of the codebase. The two-script design (forward + deferred adjoint) is sound and matches the existing adjoint pattern in `bv_parallel.py`. However, there are several major issues: the adjoint script's description omits the critical "1 SNES step on tape" requirement, the fast-path IC cache is a large novel subsystem that adds risk, and the plan does not address the k0 physical-vs-nondimensional scaling question that will silently produce wrong results if mishandled. The plan is also too ambitious for single-agent execution without phasing.

## Major Issues

### 1. Deferred adjoint script omits the tape-recording SNES step

**What**: Step 3b describes the adjoint workflow as "load converged U_data from .npz" then immediately "tape pass: J = fd.assemble(obs_form)" followed by `rf.derivative()`. This skips the critical step of running 1 SNES solve on the tape.

**Why it matters**: Firedrake's pyadjoint requires that the forward operations (solver + assemble) are recorded on the tape for the adjoint to differentiate through. Simply loading U_data and assembling an observable form does NOT record the PDE solve on the tape. The gradient would be wrong or zero. The existing working pattern in `FluxCurve/bv_parallel.py:162-262` shows exactly what is needed: (1) clear tape, (2) enable annotation, (3) rebuild context+forms from scratch, (4) load converged U_data into context, (5) run 1 SNES solve step (converged state -> converged state, ~1 Newton iteration), (6) assemble observable, (7) ReducedFunctional + derivative.

**Fix**: Rewrite Step 3b to explicitly include: `tape.clear_tape()`, `adj.continue_annotation()`, full context+forms rebuild, U_data loading, 1 SNES solve step, THEN observable assembly and adjoint differentiation. Reference `bv_parallel.py:162-262` as the template.

### 2. k0 physical vs. nondimensional scaling not addressed

**What**: The plan says `generate_multi_region_lhs_samples` returns "PHYSICAL-space values" (per the critical context), and that "k0 must be nondimensionalized by K_SCALE before passing to `make_bv_solver_params`." However, the plan's implementation steps never specify WHERE this nondimensionalization happens. The ParameterBounds docstring says "k0 ranges are in physical (linear) space" but v15 passes raw sample values directly to the solver without dividing by K_SCALE.

**Why it matters**: If the plan assumes physical k0 but the solver expects dimensionless k0_hat (or vice versa), every training sample will have the wrong kinetics. The surrogate would train on garbage data. This is a silent failure -- no convergence error, just wrong physics.

**Fix**: The plan must explicitly state: (a) what units ParameterBounds ranges are in (physical m/s or dimensionless k0_hat), (b) whether v15 had a bug here or the docstring is misleading, and (c) where in the v16 code the conversion `k0_hat = k0_phys / K_SCALE` should occur (or confirm it is unnecessary). Investigate how v15 actually works and replicate the correct behavior.

### 3. Fast-path IC cache is a large novel subsystem with insufficient specification

**What**: Steps 2 and 4 propose a global IC cache with KD-tree lookup, persistence, and a warm-started PTC solve. This is entirely new infrastructure not present in v15 or anywhere in the codebase.

**Why it matters**: (a) The warm-started fast path calls PTC solve directly -- but the plan does not specify which solver function to call. `solve_grid_with_charge_continuation` is the unified interface and runs the FULL Phase 1+Phase 2 pipeline, which is not what you want for a warm-start. The fast path would need to call lower-level functions (build context, load ICs, run PTC), which means reaching into internals not designed as a public API. (b) KD-tree in log10(k0) + alpha space requires careful distance metric tuning. (c) Sending IC arrays through ProcessPoolExecutor task arguments means pickling large numpy arrays per task, which can be slow. (d) The "fall through to slow path for the ENTIRE sample if ANY voltage point fails" means a single failed fast-path point wastes all fast-path work.

**Fix**: Either (a) defer the IC cache to a v16.1 follow-up and use v15's within-group warm-start chain for v16 (simpler, proven), or (b) specify the exact solver function calls for the fast path, estimate pickle overhead for IC data transfer, and add a fallback that preserves fast-path successes before retrying failures via slow path.

### 4. U_data storage size not estimated; no disk budget

**What**: The plan stores converged U_data for ALL voltage points for ALL samples in `batch_solutions.npz`. With 4 species + potential on an 8x200 graded mesh (likely ~6400 nodes at P2), each voltage point stores ~5 arrays of ~6400 floats = ~256 KB. For 42 voltage points x 500 samples = ~5.3 GB per batch, uncompressed.

**Why it matters**: Running multiple batches overnight (the stated goal) could fill disk. Compressed npz will help but the plan gives no estimate. The IC cache adds further storage.

**Fix**: Add a storage estimate to the plan. Consider storing U_data only for the adjoint script (maybe only the last N batches), or compress aggressively, or store U_data in a separate directory that can be cleaned after the adjoint pass.

## Minor Issues

- **Step 1 voltage grid**: "Same expanded grid as v15 (42 unique eta points)" -- the plan should copy the exact grid-building code or reference it explicitly. v15's `_build_voltage_grid()` is the canonical source.

- **Step 5 parallel architecture**: The plan says workers "receive relevant entries via task arguments" for IC cache. This means serializing numpy arrays through pickle for every task dispatch, which adds latency. Consider memory-mapped files or a shared cache directory instead.

- **Step 7 metrics**: The output format shows "adj=OK" but the forward script does NOT compute adjoints (they are deferred). This is a copy-paste error from an earlier design.

- **Checkpoint frequency**: GROUP_SIZE=5 is 4x more frequent than v15's GROUP_SIZE=20. More frequent checkpointing is good for resilience but adds I/O overhead, especially with the large U_data arrays. The plan should note this tradeoff.

- **Plan scope**: At 800-1200 estimated lines across 2 scripts + IC cache manager + checkpoint system + signal handling, this is ambitious for single-agent execution. Consider splitting into two phases: (a) forward script only (with within-group warm-starts, no global IC cache), then (b) adjoint script + IC cache as a follow-up.

- **`per_point_callback` usage**: The plan mentions `per_point_callback` for observable extraction during the solve, but Step 2 uses it only implicitly. The forward script should use the callback to extract observables during `solve_grid_with_charge_continuation` rather than re-assembling them afterward, since the ctx is live at callback time.

- **Strict convergence at 100%**: The plan correctly enforces 100% convergence with no interpolation. However, it should note the expected yield impact -- v15 used 80% threshold, and some parameter regions legitimately fail at extreme voltages. A 100% strict policy may reduce valid sample yield significantly. Consider logging rejection reasons to diagnose yield problems.

- **Signal handling**: The `_SHUTDOWN_REQUESTED` global with `signal.signal` is not safe with `ProcessPoolExecutor` (signal handlers run in the main process only, which is correct, but the plan should note that in-flight worker futures cannot be cancelled -- `executor.shutdown(wait=True)` must be called).

## Strengths

- The two-script design cleanly separates forward throughput from adjoint computation, which is the right architectural choice for overnight runs.
- The plan correctly identifies the unified `solve_grid_with_charge_continuation` as the replacement for v15's 3-tier approach.
- Strict 100% convergence with no interpolation is the right design for gradient-aware training data -- interpolated values would produce incorrect adjoint gradients.
- The checkpoint and resume design is thorough, including IC cache persistence across restarts.
- The verification plan (Step 9) is well-structured with smoke tests, resume tests, and finite-difference gradient validation.
- The plan correctly identifies that the adjoint computes dO/dtheta (raw observable gradient) rather than squared-residual gradients, and notes that the surrogate can reconstruct the latter.
