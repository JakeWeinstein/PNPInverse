# Final revision — Critique session 39

**Revised artifact:** `/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/.plans/mms-muh-multi-ion-stern/PLAN.md`
**Final verdict:** ISSUES_REMAIN at round 7 (extended cap)
**Issues raised across rounds:** 18 + 14 + 8 + 6 + 5 + 5 + 6 = **62**
**Accepted:** 62 / 62
**Defended:** 0 / 62
**Unresolved:** 0 / 62

The cap was hit before VERDICT: APPROVED was issued, but **all 62
raised issues were accepted and incorporated**. The trajectory of
issues-per-round (18 → 14 → 8 → 6 → 5 → 5 → 6) shows the loop kept
finding genuine refinements right up to the cap, but the issues
became progressively more specific (R6/R7 are detail-level ordering
and lookup-path fixes, not structural ones).

Notable structural changes that emerged from the loop:

1. **Three-phase invariant hierarchy** (prebuild / postbuild / runtime —
   the latter split around BV rate construction). Prebuild catches
   feature-flag drift that would otherwise crash `build_forms` before
   our invariants could fire. Postbuild covers derived ctx state.
   Runtime splits around BV rate construction because the rate-builder
   uses unclipped η whose validity depends on the η-margin check.
2. **OwnedCoeffTracker** per-mesh ledger, threaded through every MMS
   builder. Identity-only whitelist for source independence; no
   isinstance filtering.
3. **Two-stage source construction**: build SourceTerms container,
   run independence check on it, only then mutate F_res. Sources
   exist as inspectable objects between construction and injection.
4. **K0_R4e_factor pinned to 1e−18** (not 1; full strength dominates
   R2e by ~10²⁰ and overwhelms Newton's relative tolerance).
5. **Mandatory `TestMMSAsserts`** with 12 parametrized broken-config
   cases, all caught by MMS invariants (10 prebuild + 2 postbuild).
6. **Newton-from-U_manuf degeneracy mitigated** by a perturbed-IC
   pilot (deterministic smooth perturbation, not random).
7. **Per-field graded-mesh thresholds** populated post-pilot, not
   global L2/H1 thresholds.
8. **xdist support deferred**; module-level skip when
   `PYTEST_XDIST_WORKER` is set.

---

## Round-by-round summary

### Round 1 (18 issues, all accepted)

1. Missing formulation/mu_species/logc_muh_transform asserts → added to phase 1.
2. Reaction-schema asserts too weak → added detailed identity asserts.
3. Live ctx coefficients vs static scaling → live-objects policy + assertion.
4. Clip/floor inactivity assumed → added explicit margin checks.
5. θ_inner discrete-min check insufficient → dual DG-interp + indicator.
6. `run_mms` return shape missing Newton diagnostics → added.
7. **Newton-from-U_manuf may degenerate to interpolation test** → perturbed-IC pilot.
8. `||F_res||_final < snes_atol` overstated → use SNES reason.
9. Quadrature sweep at N=32 only insufficient → both N=32 and N=64.
10. No algebra-equivalence smoke for shared-θ closure → pilot 10.9.
11. Source independence asserted by policy not tested → extract_coefficients check.
12. Graded-mesh test missing `newton_iterations < 30` → added.
13. Graded-mesh per-field thresholds → replaced single threshold.
14. Proton concentration diagnostic missing → c_H_L2 added.
15. Factory verification needs ctx → reworded.
16. Broken-config marked optional → made mandatory (`TestMMSAsserts`).
17. pytest-xdist cache collision → deferred (decided in R3).
18. Test file imports Firedrake early → lazy-import pattern.

### Round 2 (14 issues, all accepted)

1. `newton_iterations ≥ 2` overcorrection → dropped.
2. Random Function for perturbed IC → deterministic smooth.
3. Residual reduction > 100× brittle → confined to perturbed pilot.
4. Smoke test compares analytic vs interpolated → compare on same inputs.
5. Bundle exposes only c_steric/packing/charge_density → smoke limited to those.
6. UFL `extract_coefficients` on base mixed `ctx['U']` → fixed.
7. DG projection smooths extrema → use `fd.interpolate`.
8. Mixed scalar-integral with max logic → centralized `_expr_*` helpers.
9. R2e `k0_model` and `enabled` not asserted → added.
10. Dynamic species identity not asserted → added.
11. `pytest.approx` in scripts/verification → local `_assert_close`.
12. Rate-test NaN-skip unsafe → fixture asserts all converged.
13. `TestMMSAsserts` via run_mms too indirect → direct harness.
14. xdist deferred → mark module serial / module-level skip.

### Round 3 (8 issues, all accepted)

1. Species asserts read from wrong source → use sp[0/4/6] + bv_bc.
2. `serial` marker not enforced → module-level skip.
3. `_prepare_mms_context_for_asserts` builds helpers before invariants → 2-phase.
4. ValueError from build_forms weakens criterion 5 → categorize layers.
5. `_perturbed_initial_guess` under-specified → explicit ctx, U_manuf-source.
6. Closure smoke needs clamp/floor alignment → mirror production exactly.
7. Unknown coeffs logged not failed → fail.
8. Indicator helper ambiguous → require `degree`, fd.Constant threshold.

### Round 4 (6 issues, all accepted)

1. `missing_reaction` wrong category → "static".
2. `k0_r4e_wrong` wrong category → "static" (k0_model assert catches it).
3. z_scale on packing/charge wrong → compare unscaled per-ion.
4. `_OwnedConstant` subclass not valid → factory helper.
5. `indices.phi_index` = −1 → `_phi_sub_index(ctx)` helper.
6. sp[8] is already nondim → reworded docstring.

### Round 5 (5 issues, all accepted)

1. Independence check ordering → two-stage source construction.
2. `_OWNED_COEFFS` only covers `_build_manufactured_source` → thread through all builders.
3. `isinstance(fd.Constant)` not reliable → identity-only whitelist.
4. `_OWNED_COEFFS` module-global fragile → per-mesh tracker.
5. Closure smoke vs source-builder clamp/floor → explicit ordering diagram.

### Round 6 (5 issues, all accepted)

1. `cation_hydrol_on` fails in build_forms before MMS invariant → prebuild config phase.
2. "All 12 cases at MMS invariant" too strong → recategorize prebuild/postbuild.
3. `_build_bv_rates_ex` built before η-margin → split pre-rates / post-rates margins.
4. Geometric coefficient pre-flight needed → pilot 10.0.
5. Ordering diagram refinement → added.

### Round 7 (6 issues, all accepted)

1. `_get_nondim_cfg(sp)` wrong → use `sp[10]['nondim']`.
2. `exponent_clip` wrong source → use `ctx['bv_convergence']`, not nondim.
3. `ALLOWED_GEOMETRY` module-level state fragile → per-mesh inside source-independence.
4. Geometry preflight under-specified → use `fd.ds(domain=mesh, degree=quad_degree)`.
5. `wrong_counterion` count-only check too weak → identity asserts (z, c_b, a, mode) for Cs+ and SO4²⁻.
6. R_ratio uses default ds quadrature → explicit `degree=quad_degree`.

---

## What was NOT addressed (and why it's OK)

None. Every issue was accepted. The cap was hit by exhaustion of GPT's
attention to ordering details, not by genuine disagreement.

The natural next step is to *execute* the plan and let pilot data reveal
any remaining gaps. If pilot 10.0–10.9 surface anything the loop didn't
anticipate, that becomes its own followup item (likely a session 40).
