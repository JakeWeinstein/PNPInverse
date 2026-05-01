# Plan Review (Round 2)

**Verdict:** NEEDS REVISION
**Reviewer:** Sonnet

## Previous Fix Verification

- **M1 (callback signature):** CONFIRMED correct. The plan now uses `(orig_idx, phi_applied, ctx)` which matches `grid_charge_continuation.py:564` exactly. Fix is complete.
- **M2 (solver_params distinction):** CONFIRMED correct. Step 3 now contains an explicit docstring note and comment clarifying that `warm_k0`/`warm_alpha` come from P1's result, not the true parameters. The intent is unambiguous. Fix is complete.
- **M3 (G4 unresolved):** CONFIRMED correct. G4 is closed as "not restoring" with clear rationale. Fix is complete.
- **G1 (tolerances):** CONFIRMED correct. The plan states v13 and v15 use the same `ftol=1e-8, gtol=5e-6` and requires no change. Fix is complete.

## Summary

All four Round 1 issues are correctly resolved. However, one pre-existing bug was missed in Round 1 and remains in the revised plan: the Step 2 callback calls `_build_bv_observable_form` without the required `reaction_index` keyword argument. The actual function signature (confirmed in `Forward/bv_solver/observables.py`) requires `reaction_index` as a keyword-only parameter with no default. The plan omits it from both the `form_cd` and `form_pc` calls, which will raise a `TypeError` at runtime. The correct calls (as used in v15's existing code) pass `reaction_index=None`. This is a straightforward one-word fix per call site, but it is a hard blocker for the callback.

## Major Issues

### M1: Missing `reaction_index` in Step 2 callback (hard runtime error)

The plan's `_extract_observables` callback in Step 2 calls `_build_bv_observable_form` as:

```python
form_cd = _build_bv_observable_form(ctx, mode="current_density", scale=observable_scale)
form_pc = _build_bv_observable_form(ctx, mode="peroxide_current", scale=observable_scale)
```

The actual function signature in `Forward/bv_solver/observables.py:13` is:

```python
def _build_bv_observable_form(ctx, *, mode, reaction_index, scale):
```

`reaction_index` is a required keyword argument with no default. These calls will raise `TypeError: _build_bv_observable_form() missing 1 required keyword-only argument: 'reaction_index'`.

The fix is to add `reaction_index=None` to both calls, exactly as v15 does at lines 238 and 243.

Note: This same function is used correctly in Step 3 of v15 (no callback needed there), so the issue is localized to Step 2's new callback code.

## Minor Issues

### m1: Import path for `_build_bv_observable_form` not specified in Step 2

The plan imports `_build_bv_observable_form` implicitly inside the callback but does not specify which path to import from. The plan notes the two paths (`FluxCurve.bv_observables` vs `Forward.bv_solver.observables`) differ, then defers to "whichever import v15 uses." V15 uses `from FluxCurve.bv_observables import _build_bv_observable_form` (line 200). Since `FluxCurve.bv_observables` is a confirmed re-export of the same function, either path is correct, but the plan should explicitly state which import to use (or copy v15's import) to avoid ambiguity during implementation.

### m2: Step 2 places `import firedrake as fd` inside the callback at module-doc level but not in the function body

The callback `_extract_observables` calls `fd.assemble(...)` but `fd` is not imported inside the callback or shown to be in scope. In v15, `import firedrake as fd` appears at the top of the containing function (`_solve_clean_targets_charge_cont`). The plan should note that `fd` must be imported (lazily or at top of the enclosing function) before it is referenced in the callback.

### m3: No verification step for cache key correctness

The plan changes the cache key from `method=charge_continuation,eta_steps=...,charge_steps=...` to `method=unified_grid,max_eta_gap=...`. This is correct in intent, but there is no proposed test that verifies the new key actually differs from a v15-generated cache (so that a v15 cache present on disk does not silently pass stale data to v16). A simple assert or a unit test comparing the two hash outputs would close this gap.

## Strengths

- The three-phase S→P1→P2 architecture is preserved exactly, minimizing diff surface.
- The `adj.stop_annotating()` guard is correctly applied in both Steps 2 and 3, preventing adjoint tape pollution.
- Step 3 correctly iterates over `result.points.items()` post-hoc rather than overcomplicating the callback path — this matches `GridChargeContinuationResult`'s actual API.
- The warning on partial convergence in Step 2 is appropriate and matches v15's behavior for the same condition.
- The `--eta-steps` deprecation strategy (keep as no-op with warning) is safe and backward-compatible.
- The single-file modification scope (`_v16.py` only) is appropriately minimal.
- The dependency table correctly identifies all required imports and their verified locations.
