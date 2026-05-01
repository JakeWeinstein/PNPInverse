# Plan Review (Round 2)

**Verdict:** NEEDS REVISION
**Reviewer:** Opus

## Previous Fix Verification

**M1 (callback signature):** CORRECT. Verified at `grid_charge_continuation.py:564` -- the callback is invoked as `per_point_callback(orig_idx, eta_i, ctx)`. The plan's Step 2 code uses `callback(orig_idx, phi_applied, ctx)` which matches.

**M2 (solver_params distinction):** CORRECT. Step 3 now explicitly documents that `warm_k0`/`warm_alpha` come from P1's estimated parameters, not the true kinetic parameters. The `NOTE` comment in the docstring makes this clear.

**M3 (G4 unresolved):** CORRECT. v15 line 518 confirms `--pde-secondary-weight` defaults to 1.0 (a single fixed weight), matching the v13 behavior. The cascade weight sweep was surrogate-only (S3), which v15/v16 intentionally do not restore. G4 is properly closed.

**G1 (tolerances):** CORRECT. v15 lines 857-858 confirm `ftol=1e-8, gtol=5e-6`, consistent with the plan's claim. No change needed.

## Summary

The previous round's fixes are all correct and complete. However, this round identifies one major issue (missing required keyword argument that will cause a TypeError at runtime), one medium issue (unused function parameter), and two minor issues. The plan is close but needs one more pass.

## Major Issues

### M1: `_build_bv_observable_form` calls in Step 2 are missing required `reaction_index` argument

The plan's Step 2 code calls:
```python
form_cd = _build_bv_observable_form(ctx, mode="current_density", scale=observable_scale)
form_pc = _build_bv_observable_form(ctx, mode="peroxide_current", scale=observable_scale)
```

But `_build_bv_observable_form` in `Forward/bv_solver/observables.py:13-18` has the signature:
```python
def _build_bv_observable_form(ctx, *, mode, reaction_index, scale):
```

**All three keyword arguments are required** -- `reaction_index` has no default value. Omitting it will raise `TypeError: _build_bv_observable_form() missing 1 required keyword-only argument: 'reaction_index'`.

v15 gets this right (lines 237-243):
```python
form_cd = _build_bv_observable_form(
    ctx, mode="current_density", reaction_index=None, scale=observable_scale,
)
```

**Fix:** Add `reaction_index=None` to both calls in the Step 2 callback.

## Minor Issues

### m1: `base_sp` parameter in `_extend_voltage_cache_for_p2_unified` is accepted but never used

Step 3's function signature includes `base_sp` to match v15's interface, but the function body builds its own `sp` from scratch using `make_bv_solver_params(...)` with `warm_k0`/`warm_alpha`. The `base_sp` parameter is dead code. This is harmless but confusing -- either remove it or use it (e.g., extract `dt`, `t_end`, mesh params from `base_sp` instead of hardcoding them).

v15's `_extend_voltage_cache_for_p2` (line 386-395) also accepts `base_sp` and also ignores it, so this is inherited tech debt rather than a new bug. Still worth noting for v16 cleanup.

### m2: `_target_cache_path` needs `max_eta_gap` instead of `eta_steps` in ALL call sites

The plan correctly says Step 5 replaces `--eta-steps` with `--max-eta-gap` and Step 2 updates the hash. But `_generate_targets_with_charge_cont` (v15 lines 256-289) passes `eta_steps` and `charge_steps` to `_target_cache_path`. In v16, this call site must also be updated to pass `max_eta_gap` instead of `eta_steps`. The plan should explicitly list this function as needing modification (it currently only mentions `_target_cache_path` hash changes and `_solve_clean_targets_unified`).

### m3: Import path for `_build_bv_observable_form` in v16 target gen callback

The plan's Step 2 code uses `_build_bv_observable_form` inside the callback but does not specify the import. v15 imports from `FluxCurve.bv_observables` (line 200). Since the function now canonically lives in `Forward.bv_solver.observables` (with `FluxCurve.bv_observables` as a re-export shim), v16 should import from `Forward.bv_solver.observables` directly. This avoids a Forward-script -> FluxCurve dependency for what is fundamentally a Forward-layer operation. The plan should state the import explicitly.

## Strengths

- The callback-based architecture in Step 2 is well-designed: extracting observables inside `per_point_callback` avoids the need to re-solve or maintain a separate ctx, and naturally fits the unified framework's API.
- The distinction between target generation (true params) and voltage extension (estimated params) is now clearly documented in both code comments and the plan narrative.
- The gap analysis (G1-G4) is thorough and correctly closes items that don't need changes.
- Cache invalidation via hash key update is the right approach for avoiding v15/v16 cache collisions.
- Keeping PDE tolerances unchanged is the correct call -- verified against actual v15 code.
