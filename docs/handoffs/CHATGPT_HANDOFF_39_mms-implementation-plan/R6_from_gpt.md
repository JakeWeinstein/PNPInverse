1. WHAT: `cation_hydrol_on` still may not reach the MMS static invariant layer.
WHY: In this 3-species stack the roles are `["neutral", "neutral", "proton"]`; enabling cation hydrolysis makes `build_forms` call `resolve_counterion_index`, which raises because there is no dynamic species with role `"counterion"`. That happens before `_assert_static_stack_invariants`.
WHAT TO DO: Add a pre-build config invariant phase that inspects `sp[10]["bv_convergence"]` before `build_forms`, at least for feature gates like water/cation hydrolysis. Pipeline should be: pre-build static config asserts → `build_forms` → post-build static ctx asserts.

2. WHAT: The “all 12 cases fire at MMS invariant layer” claim is still too strong.
WHY: Any broken config that makes production form construction fail cannot be counted as post-build MMS invariant coverage. `cation_hydrol_on` is one such case unless you add the pre-build phase above.
WHAT TO DO: Either move those cases to a pre-build MMS invariant harness and count them there, or categorize them honestly as production-validation/build-forms failures.

3. WHAT: `_build_bv_rates_ex` is still built before the eta-clip margin assert.
WHY: The helper uses the unclipped manufactured BV algebra. If an envelope change violates the eta margin, you can still construct invalid/extreme `rxn_rates` before the check that proves they are valid.
WHAT TO DO: Split runtime checks: evaluate eta/u/phi/floor margins before building `rxn_rates`, then build rates and run `R_ratio`. Or ensure `_assert_runtime_stack_invariants` checks eta margin before it ever assembles/evaluates `rxn_rates`.

4. WHAT: The source-independence whitelist may falsely reject geometric coefficients.
WHY: Source expressions depend on `SpatialCoordinate(mesh)` and `FacetNormal(mesh)`. Depending on Firedrake/UFL extraction behavior, mesh coordinate/geometric terminals may appear in coefficient extraction.
WHAT TO DO: Add a tiny pre-flight assertion documenting actual `extract_coefficients` behavior on `x`, `FacetNormal`, and a simple manufactured source. If geometry appears, explicitly whitelist mesh geometry coefficients; do not discover this during the pilot.

5. WHAT: The ordering diagram still says runtime margins happen after closure and rates are built.
WHY: Closure is fine because margin checks inspect closure quantities, but BV rates are different: they can overflow or encode the wrong clipped branch if eta margins fail.
WHAT TO DO: Refine the diagram to: build manufactured fields/closure → margin checks that do not need rates → build BV rates → R_ratio → source terms.

VERDICT: ISSUES_REMAIN