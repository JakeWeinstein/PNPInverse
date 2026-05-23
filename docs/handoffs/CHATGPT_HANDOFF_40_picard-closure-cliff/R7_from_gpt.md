No blocking issues remain. The deltas close the remaining state-management and factory-contract holes.

1. WHAT: The proposed `test_make_run_ss_factory_None_normalizes_to_bare` needs to capture the post-normalization factory, not just the raw kwarg passed by the caller. WHY: If the spy records before `warm_walk_phi` normalizes `None`, it will see `None` and fail incorrectly. WHAT TO DO: Put the assertion close to the normalization helper or expose a tiny `_normalize_make_run_ss_factory(...)` helper and unit-test that directly.

2. WHAT: `D_per_species_hat` is treated as deck-invariant in `PicardConfig`. WHY: That is fine for this study, but if future code mutates `logD_funcs`, Picard would use stale D. WHAT TO DO: Non-blocking: either document D is fixed for this feature, or assert at factory call that config D matches ctx D for the cathodic species.

3. WHAT: `test_non_picard_mode_does_not_touch_picard_ctx_keys` makes the absence of `packing_expr`/`theta_inner_expr` part of the contract. WHY: That is fine for this plan, but it would block future non-Picard diagnostics from reusing those ctx keys. WHAT TO DO: Keep the test if you want strict inertness; otherwise narrow it to Picard-specific keys only.

VERDICT: APPROVED