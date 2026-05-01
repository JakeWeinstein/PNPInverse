# Plan Review Report (Round 3)

**Plan:** `~/.claude/plans/zippy-tickling-sutton.md` (Overnight Training Script v16)
**Date:** 2026-04-03
**Level:** 2 (Sonnet + Opus)
**Reviewers:** Sonnet, Opus
**Verdict:** NEEDS REVISION

## Summary

Both reviewers found the plan architecturally sound — the two-script design, strict 100% convergence, and checkpoint/resume system are well-designed. However, both independently flagged the same critical flaw in the adjoint methodology, and the fast-path IC cache introduces more risk than value as currently specified. Four major issues must be resolved before implementation.

## Major Issues

| # | Issue | Impact | Found By |
|---|-------|--------|----------|
| 1 | Adjoint script omits SNES-on-tape step | Gradients will be zero/wrong — pyadjoint has no tape of U_data's dependence on controls | **Both** |
| 2 | k0 physical vs. nondimensional scaling unresolved | Silent wrong-physics failure — every sample trains on garbage kinetics | **Opus** |
| 3 | Fast-path IC cache is under-specified & violates unified API | Bypasses `solve_grid_with_charge_continuation`, reimplements v15's ad-hoc pattern; IPC gap means cache never grows in spawn mode | **Both** |
| 4 | U_data storage size unbudgeted (~5 GB/batch) | Multiple overnight batches could exhaust disk | **Both** |

### 1. Adjoint script omits the tape-recording SNES step (CRITICAL)

**Agreement: Both reviewers flagged this identically.**

Step 3b describes: load U_data -> assemble observable -> `ReducedFunctional.derivative()`. This skips the critical SNES solve on the tape. pyadjoint needs to record the forward PDE solve to know how U depends on (k0, alpha). Without it, gradients are zero.

**Fix (from Opus):** Follow the working pattern in `FluxCurve/bv_parallel.py:162-262`:
1. `tape.clear_tape()` + `continue_annotation()`
2. Rebuild context+forms with k0/alpha as `fd.Control`-wrapped functions
3. Load converged U_data as initial guess
4. Run 1 SNES solve step (converged -> converged, ~1 Newton iteration)
5. THEN assemble observable + `ReducedFunctional` + `.derivative()`

### 2. k0 physical vs. nondimensional scaling (SILENT FAILURE)

**Found by Opus only.** Sonnet did not flag this.

`generate_multi_region_lhs_samples` returns physical-space k0 values. `make_bv_solver_params` expects dimensionless k0_hat. The plan never specifies where `k0_hat = k0_phys / K_SCALE` happens. v15 may have a bug here or the docstring may be misleading.

**Fix:** Investigate v15's actual behavior. Explicitly state in Step 1: (a) what units the ParameterBounds ranges are in, (b) where the conversion occurs, (c) add an assertion that k0_hat values are in a sane range (e.g., 1e-4 to 1e4).

### 3. Fast-path IC cache: under-specified, violates API, IPC gap

**Both reviewers flagged different aspects of the same problem:**
- **Sonnet:** The fast path does per-voltage-point warm-start solves, bypassing the unified `solve_grid_with_charge_continuation` API — exactly the ad-hoc pattern v16 is supposed to eliminate. Also, workers in spawn mode can't mutate the main process's cache.
- **Opus:** No existing code for KD-tree IC lookup. The fast path needs lower-level solver functions not in the public API. Pickling large numpy arrays per task adds latency.

**Fix (both agree):** Defer the global IC cache to v16.1. For v16, use v15's proven within-group warm-start chain (previous sample's ICs seed next sample). This is simpler, proven, and doesn't require novel infrastructure.

### 4. U_data storage size (~5 GB/batch uncompressed)

**Both flagged, Opus provided the estimate:** 42 eta points x 500 samples x ~256 KB/point = ~5.3 GB per batch uncompressed.

**Fix:** Add a storage estimate. Consider: (a) computing gradients inline via `per_point_callback` during forward solve (eliminating U_data storage entirely), (b) storing U_data only temporarily and cleaning after adjoint pass, (c) adding `--max-stored-batches` CLI flag.

## Minor Issues (consolidated)

- **Metrics output shows "adj=OK"** but forward script doesn't compute adjoints (copy-paste from earlier design) — *Both*
- **`per_point_callback` usage should be explicit** — the slow path should use it for observable extraction, not re-assemble afterward — *Both*
- **Voltage grid code**: copy `_build_voltage_grid()` verbatim from v15 rather than describing it — *Both*
- **Relaxed retry SNES params**: requires rebuilding `solver_params` (frozen dataclass), not mutating — *Sonnet*
- **k0 nondimensionalization**: add explicit conversion in Step 1 constants block — *Sonnet*
- **GROUP_SIZE=5 + U_data I/O**: checkpoint writes could be ~320 MB every ~9 seconds, potentially slower than solves — *Sonnet*
- **100% strict yield impact**: should log rejection reasons; may significantly reduce yield vs v15's 80% threshold — *Opus*
- **Signal handling**: note that in-flight worker futures can't be cancelled; need `executor.shutdown(wait=True)` — *Opus*
- **Plan scope**: both recommend splitting into Phase A (forward script) and Phase B (adjoint script) for execution — *Both*

## Agreement Analysis

- **Agreed on:** Adjoint tape issue (critical), IC cache risk, U_data storage, plan scope, `per_point_callback` usage, voltage grid specification
- **Disagreements:** None — reviewers found complementary issues, not conflicting conclusions
- **Unique to Sonnet:** IC cache IPC gap detail, checkpoint I/O bottleneck analysis
- **Unique to Opus:** k0 scaling issue, `bv_parallel.py` template reference, pickle overhead concern

## Strengths (consolidated from both)

- Two-script design cleanly separates forward throughput from adjoint computation
- Strict 100% convergence is the correct choice for gradient-aware training data
- Checkpoint + resume + signal handling is a meaningful operational improvement over v15
- Verification plan is comprehensive, especially finite-difference gradient validation
- Raw observable gradients (dO/dtheta) rather than squared-residual gradients is the right abstraction
- Solver configuration matches v13 paper parameters concretely
