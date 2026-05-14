1. WHAT: The independence check still has an ordering gap.
WHY: `_assert_runtime_stack_invariants(ctx, sp, manuf, closure, rxn_rates)` is described as running before source injection, but the independence check needs the actual source terms `S_c_i`, `g_i`, `S_phi`, `g_S`. Those terms are only created inside `_build_manufactured_source`.
WHAT TO DO: Make source construction two-stage: build source terms into a local `SourceTerms` object, run independence/runtime checks on that object, then mutate `ctx["F_res"]`. Do not try to check source independence before source terms exist.

2. WHAT: `_OWNED_COEFFS` only covers constants created inside `_build_manufactured_source`.
WHY: Source terms include expressions from manufactured fields, closure, and BV-rate builders. Their `fd.Constant(...)` objects can also appear in `extract_coefficients(S)` and fail the unknown-coefficient check.
WHAT TO DO: Route all MMS-only constants through `_owned_constant`: manufactured fields, closure helper, BV helper, and source injection. Or replace the global set with an explicit per-mesh coefficient tracker passed through all builders.

3. WHAT: The fallback `isinstance(c, (fd.Function, fd.Constant))` is not a safe plan.
WHY: Depending on Firedrake version, `fd.Constant` may not behave like a normal class for `isinstance`, and Firedrake constants are exactly the coefficient type causing the ambiguity.
WHAT TO DO: Avoid type-based filtering for constants. Use object identity: forbidden `{ctx["U"], ctx["U_prev"]}`, allowed live objects, and owned MMS objects. Anything else from `extract_coefficients` fails.

4. WHAT: `_OWNED_COEFFS` as a module global is fragile.
WHY: A failed mesh setup, nested smoke test, or future parallel execution can leave stale objects or race the set. You clear it at source-build start, but manufactured/closure/rate builders happen before that.
WHAT TO DO: Use a local `OwnedCoeffTracker` per mesh. Pass it into every MMS expression builder and into the independence check.

5. WHAT: The closure smoke and source builder now intentionally differ on clamp/floor algebra.
WHY: Smoke mirrors production exactly; source uses unclamped/unfloored expressions guarded by margin asserts. That is acceptable, but only if the margin asserts run before source mutation and are tied to the same envelope.
WHAT TO DO: Explicitly state the ordering: static asserts → manufactured fields → analytic closure/rates → margin asserts → build source terms → source independence check → mutate residual.

VERDICT: ISSUES_REMAIN