1. WHAT: `_get_nondim_cfg(sp)` is wrong in the prebuild assert.
WHY: The helper expects the params dict, not the SolverParams object. Passing `sp` returns `{}` in the current helper pattern, so `poisson_suppressed` will not be caught.
WHAT TO DO: Use `params = sp[10]`; then `nondim_cfg = params.get("nondim", {})`. Add a `poisson_suppressed` broken-config test that proves this assertion fires before `build_forms`.

2. WHAT: `exp_clip = ctx["nondim"].get("exponent_clip", 100.0)` is the wrong source.
WHY: `exponent_clip` lives in `ctx["bv_convergence"]` / `sp[10]["bv_convergence"]`, not in `ctx["nondim"]`. This silently falls back to `100.0` and would miss a bad clip config.
WHAT TO DO: Read `conv_cfg = ctx["bv_convergence"]`; assert `clip_exponent is True` and `exponent_clip == 100.0` in prebuild/static invariants; use that same value in eta-margin checks.

3. WHAT: `ALLOWED_GEOMETRY` cannot be module-level state populated by pilot 10.0.
WHY: Pytest order is not guaranteed, so the source-independence check may run before the pilot. Also geometry coefficient identities are mesh-specific; a set populated on an `N=8` mesh may not match `N=16/32/64`.
WHAT TO DO: Compute allowed geometry coefficients per mesh inside `_assert_source_independence(ctx, ...)`, or attach them to `ctx` during that mesh setup. Keep pilot 10.0 as documentation, not as required state.

4. WHAT: The geometry preflight expression is under-specified.
WHY: `fd.dot(...)*fd.ds` creates a form, and `fd.ds` lacks an explicit domain/degree. That may not reflect source-term extraction behavior.
WHAT TO DO: Probe both expressions and forms explicitly using `fd.ds(domain=mesh, degree=quad_degree)`, and use the exact same extraction path as `_assert_source_independence`.

5. WHAT: `wrong_counterion` is listed as prebuild, but the shown prebuild checks only count bikerman entries.
WHY: A wrong two-counterion config can pass count-only checks.
WHAT TO DO: In prebuild, assert exact counterion identities: two bikerman entries with expected `(z, c_bulk_nondim, a_nondim, steric_mode)` for Cs+ and SO4--. Then the `wrong_counterion` regex is meaningful.

6. WHAT: `R_ratio` uses default `ds` quadrature.
WHY: The rates contain exponentials and cosines; default quadrature can pollute the diagnostic that pins K0/R4e relevance.
WHAT TO DO: Use `ds_e = fd.ds(marker, domain=mesh, degree=quad_degree)` in `_assert_post_rates_invariants`.

VERDICT: ISSUES_REMAIN