# SONNET_07 — Grid-Per-Voltage Warm-Walk Solver Verification

**Agent:** 07/13  
**Scope:** `Forward/bv_solver/grid_per_voltage.py` (1154 lines), `anchor_continuation.py` (solve_grid_with_anchor entry point), `sweep_order.py` (163 lines)  
**Date:** 2026-05-22  
**Verdict:** PASS with one minor observation

---

## Check-by-check findings

### 1. `solve_grid_per_voltage_cold_with_warm_fallback` algorithm (C+D)

PASS. Implementation exactly matches the documented C+D strategy:

- **Phase 1 (C):** Each voltage is solved independently via `_solve_cold`. Fresh context built per-V. For `debye_boltzmann` IC path, tries direct z=1 SS first, falls back to linear-phi z-ramp on failure. For linear-phi path, zeroes all charges at z=0, runs SS, then linearly ramps z from 0 to 1 in `max_z_steps` steps with `_snapshot_U` + `_restore_U` checkpoint/rollback on each step failure.

- **Phase 2 (D):** Three sub-passes:
  - **Cathodic walk** (indices below `anchor_lo`): walks from `anchor_lo` downward, always warm-walking from the nearest converged right-neighbor. On failure, `break`s the chain (does not try to skip over a failed point).
  - **Anodic walk** (indices above `anchor_hi`): symmetric, walks from `anchor_hi` upward, breaks on first failure.
  - **Interior fill** (gaps between non-contiguous cold successes): iterative while-loop with `made_progress` flag; tries closest converged neighbor first (both sides), skips retries on unchanged anchor pairs.

**Neighbor choice:** Correctly finds the nearest converged index by scanning adjacent indices, not by sorting all distances. For cathodic walk: `j = orig_idx + 1`, scans right until converged. For anodic: `j = orig_idx - 1`, scans left. For interior: tries both sides sorted by distance.

### 2. `solve_grid_with_anchor` (anchor + grid pair, Phase 5γ)

PASS. Lives in `grid_per_voltage.py` (lines 875–1154), not in `anchor_continuation.py`.

- Consumes a frozen `PreconvergedAnchor` (defined in `anchor_continuation.py`).
- Visit order: `sorted(range(n_points), key=lambda i: abs(phi_applied_values[i] - anchor.phi_applied_eta))` — explicitly closest-to-anchor first. This is a true bidirectional outward walk from the anchor voltage, implemented via distance-sorted traversal rather than two explicit directional loops.
- Maintains a `sources` list seeded with the anchor; each successful grid solve appends its `(target_phi, snap)` to `sources`. Each new target picks the nearest source by `min(sources, key=lambda s: abs(s[0] - target_phi))`.
- Entire loop wrapped in `with adj.stop_annotating()` (adjoint tape hygiene correct).
- Asserts `dof == anchor.mesh_dof_count` before restoring state (mesh DOF guard present).
- Pins `k0` from `anchor.k0_targets` both BEFORE and AFTER `set_initial_conditions` (defensive double-pin to guard against IC reseed).

Matches `project_pass_a_outcome.md` description: "Pass A grid 8/8 converged via PreconvergedAnchor + solve_grid_with_anchor".

### 3. Sweep ordering (`sweep_order.py`)

PASS. `_build_sweep_order` is a utility used by callers that want explicit ordering; `solve_grid_with_anchor` implements its own distance-sort inline (not calling `_build_sweep_order`). `solve_grid_per_voltage_cold_with_warm_fallback` sweeps Phase 1 sequentially by index (0 to n-1), then Phase 2 walks cathodic/anodic/interior outward.

`_build_sweep_order` logic:
- Single-sign (all positive or all negative): ascending |eta| from equilibrium (no degenerate cases).
- Mixed-sign: two-branch sweep. Negative branch sorted ascending |eta|; positive branch sorted ascending eta. Branch with smallest |eta| point goes first. No degenerate ordering: guard `abs(d_eta) > 1e-14` prevents zero-denominator in predictor.

**No degenerate orderings found.**

### 4. Warm-start state copy

PASS. `_restore_U` (line 106–109):
```python
def _restore_U(snap: tuple, U, U_prev) -> None:
    for src, dst in zip(snap, U.dat):
        dst.data[:] = src   # in-place copy, not alias
    U_prev.assign(U)
```
Snapshot is `tuple(d.data_ro.copy() for d in U.dat)` — `.copy()` makes a new numpy array. Restore writes `dst.data[:] = src` (copy by value). `U_prev.assign(U)` keeps time-stepping state consistent. No aliasing.

In `solve_grid_with_anchor`, `_restore_U(src_snap, U, U_prev)` uses the same pattern. The `sources` list stores `_snapshot_U(U)` (deep copy) at each success; no aliasing between source snapshots and live state.

### 5. Per-voltage convergence reporting

PASS. Every voltage — cold, warm, and failed — produces a `PerVoltagePointResult` stored in `points[orig_idx]`. Fields: `converged` (bool), `method` (string: "cold", "cold-failed", "warm<-{V:+.3f}", "warm<-{V:+.3f}-FAILED"), `U_data` (snapshot or None), `achieved_z_factor`, `diagnostics`. `PerVoltageContinuationResult` provides `all_converged()`, `converged_indices()`, `failed_indices()`.

### 6. Fallback semantics on double failure

PASS — with noted behavior. In C+D:

- **Phase 1 failures** are recorded and do not abort: sweep continues over all voltages independently.
- **Phase 2 cathodic walk**: if warm-walk fails at index `orig_idx`, prints "warm-walk FAILED (broke chain)" and **breaks** the cathodic loop. Points beyond the failed one remain at their Phase-1 state (converged=False). Rationale: if V_i failed walking from V_{i+1}, V_{i-1} is even farther; continuing would warm-walk from a non-converged anchor.
- **Phase 2 anodic walk**: same `break` semantics.
- **Phase 2 interior**: does NOT break; uses `made_progress` iteration and `failed_with_anchors` cache to skip repeated futile retries.

In `solve_grid_with_anchor`:
- Each failure records `converged=False` with `method=f"warm<-{src_phi:+.3f}-FAILED"` and continues to the next target. The failed point's snapshot is NOT added to `sources`. This means later targets skip the failed V as an anchor, which is correct.

**No abort on failure; sweep always completes the full grid.**

### 7. `l_eff_m` / `domain_height_hat` consistency

PASS. `l_eff_m` is consumed by `make_bv_solver_params` (in `scripts/_bv_common.py`) to compute `domain_height_hat = l_eff_m / L_REF`, which is stored in `bv_convergence['domain_height_hat']` inside the `params` dict (the 11th element of `solver_params`).

In both `solve_grid_per_voltage_cold_with_warm_fallback` and `solve_grid_with_anchor`, `_build_for_voltage` calls `build_context(sp, mesh=mesh)` with the **shared mesh** passed by the caller. The `domain_height_hat` in `params['bv_convergence']` is read at form-build time inside `forms_logc.py` / `forms_logc_muh.py` (lines 908–918, 1248–1254). The shared `mesh` parameter is passed through from the outer caller, which is responsible for ensuring mesh y-extent matches `domain_height_hat`.

**No re-build with mismatched L_eff**: the `params` dict is not modified between voltage calls; `l_eff_m` / `domain_height_hat` is frozen in `solver_params` at construction time.

Minor note: `grid_per_voltage.py` does not validate that the shared mesh's y-extent matches `domain_height_hat` — this responsibility sits entirely with the caller. If a caller passes an inconsistent mesh, the mismatch is silent (no assertion). `solve_grid_with_anchor` does assert DOF count equality (anchor vs. live ctx), which provides a partial guard.

### 8. Newton solver options propagation

PASS. In both `_build_for_voltage` implementations:

- `solve_opts` is extracted from `params` (the dict portion of `solver_params`) by filtering out non-PETSc keys (`NON_PETSC_KEYS = {"bv_bc", "bv_convergence", "nondim", "robin_bc"}`).
- `solve_opts.setdefault("snes_error_if_not_converged", True)` ensures Newton divergence raises rather than silently accepting.
- Solver is rebuilt fresh per-voltage (new `NonlinearVariationalSolver` from shared `solve_opts`). No solver state is carried over between voltages — the warm-start is entirely in the function-space data `U`, not in Newton iteration history. This is correct for Firedrake's Newton solver.
- The `make_run_ss` factory receives the per-voltage `solver` and `of_cd` but reuses the same SER dt parameters from the outer `solve_grid_*` call. Consistent across anchor and grid.

### 9. Strategy B existence and hard rule #1

PASS. `solve_grid_with_charge_continuation` does NOT exist in the codebase. `solvers.py` explicitly documents: "The continuation-style solvers (`forsolve_bv`, `solve_bv_with_continuation`, `solve_bv_with_charge_continuation`) were removed alongside the legacy concentration backend." References in `boltzmann.py` (line 294) and `observables.py` (line 4) are documentation-only cross-references, not implementations. Hard rule #1 is structurally enforced by the removal of Strategy B.

---

## Issues found

### Minor: No mesh y-extent vs. `domain_height_hat` validation in grid solvers

**Severity:** Minor / latent.

Neither `solve_grid_per_voltage_cold_with_warm_fallback` nor `solve_grid_with_anchor` validates that the shared `mesh` y-extent matches `domain_height_hat` in `params['bv_convergence']`. The DOF count assertion in `solve_grid_with_anchor` (line 1069) provides a partial guard (wrong mesh would likely have different DOF count), but a mesh with the same DOF count and wrong y-extent would pass silently. The burden falls entirely on the caller. This is documented convention (CLAUDE.md: "mesh y-extent must match") but not enforced at the API boundary.

No fix required for current production use (callers uniformly use `make_graded_rectangle_mesh` with the same `domain_height_hat`), but a future hardening point.

---

## Summary table

| Check | Status | Notes |
|---|---|---|
| C+D cold + z-ramp algorithm | PASS | Correct per-V isolation; debye_boltzmann fast-path with z=0 fallback |
| C+D Phase 2 warm-walk logic | PASS | Cathodic/anodic break-on-fail; interior iterative fill |
| Neighbor selection | PASS | Scan-to-nearest, not global sort |
| solve_grid_with_anchor bidirectional walk | PASS | Distance-sorted traversal; sources list grows with successes |
| Sweep ordering (sweep_order.py) | PASS | Two-branch mixed-sign; no degenerate orderings |
| Warm-start state copy (no aliasing) | PASS | `.copy()` in snapshot; `[:] = src` in restore |
| Per-voltage convergence reporting | PASS | All outcomes recorded; method string identifies anchor |
| Fallback on double-failure | PASS | Break chain in cathodic/anodic; continue in anchor-grid |
| l_eff_m / domain_height_hat consistency | PASS (minor) | Frozen in params; no runtime re-build; mesh guard caller-side only |
| Newton options propagation | PASS | Consistent solve_opts; SNES error-if-not-converged set |
| Strategy B absent / hard rule #1 | PASS | Removed from codebase; only documentation references remain |

---

## Files reviewed

- `/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/Forward/bv_solver/grid_per_voltage.py` (all 1154 lines)
- `/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/Forward/bv_solver/anchor_continuation.py` (PreconvergedAnchor, extract_preconverged_anchor, set_reaction_k0_model, companion helpers)
- `/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/Forward/bv_solver/sweep_order.py` (all 163 lines)
- `/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/Forward/bv_solver/solvers.py` (Strategy B removal confirmation)
- `/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/scripts/_bv_common.py` (l_eff_m → domain_height_hat path, lines 1289–1295)
