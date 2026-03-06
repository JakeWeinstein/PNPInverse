# Phase 1: Nondimensionalization & Weak Form Audit - Context

**Gathered:** 2026-03-06
**Status:** Ready for planning

<domain>
## Phase Boundary

Prove dimensional transforms are correct via roundtrip tests, and confirm MMS convergence script tests the actual production BV solver weak form (not a hand-built replica). Requirements: FWD-02, FWD-04.

</domain>

<decisions>
## Implementation Decisions

### Nondim roundtrip test scope
- Cover 1-species, 2-species, and 4-species (O2, H2O2, H+, ClO4-) configurations
- Test ALL transform inputs: D, c0, c_inf, phi, dt/t_end, kappa, plus derived quantities (Debye length, flux scale, current density scale)
- Only test `enabled=True` (nondim) mode; `enabled=False` is trivial identity and already covered by existing tests
- Use both v13 production parameter values AND parametrized synthetic values covering different orders of magnitude
- Roundtrip tolerance: `rel=1e-12`

### Existing test validation
- Hand-check existing `test_nondim.py` assertions against textbook formulas (thermal voltage, Debye length, scale relationships) before building new tests on top
- Document which tests were audited and confirmed correct
- User has no confidence in existing tests — validation is a prerequisite, not optional

### MMS weak form audit
- Refactor MMS script to import and use production `bv_solver` weak form code, not its own inline assembly
- Align with `bv_solver` (Butler-Volmer BC solver) specifically — this is what v13 uses
- If refactor reveals a bug in production weak form: fix the production code (per PROJECT.md: "don't change solver code unless a bug is found during verification")
- Produce a written audit document (term-by-term correspondence) PLUS passing tests — both feed into Phase 6 V&V report
- Include a light MMS smoke test (2-3 mesh sizes) to confirm convergence still works after refactor; full rate assertions with GCI belong in Phase 2

### Convergence standards (for Phase 1 smoke test only)
- R-squared > 0.99 on log-log fit (matches Phase 2 success criteria)
- Full rate assertions (L2 ~ O(h^2), H1 ~ O(h)) and GCI are Phase 2 scope

### Claude's Discretion
- Exact structure of the roundtrip test parametrization
- How to extract/share weak form building code between MMS and production solver
- Synthetic parameter value ranges for edge case coverage
- Organization of the textbook verification notes

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `Nondim/transform.py:build_model_scaling()`: Central nondim function — roundtrip tests will exercise this
- `Nondim/scales.py:build_physical_scales()`: Computes reference scales from physical params — feeds into transform
- `Nondim/constants.py`: Single source of truth for physical constants (F, R, T, epsilon)
- `tests/test_nondim.py`: 40+ existing tests covering scales, transform helpers, enabled/disabled modes — needs validation before extension
- `tests/conftest.py`: Has `default_nondim_kwargs` fixture used by existing tests

### Established Patterns
- Tests use `pytest.approx(expected, rel=...)` for floating-point comparison
- Tests organized by class per module (`TestNondimScales`, `TestBuildModelScalingEnabled`, etc.)
- Nondim package uses pure Python/NumPy — no Firedrake dependency
- MMS script (`scripts/verification/mms_bv_convergence.py`) has 3 test cases: single neutral, two neutral, two charged

### Integration Points
- `Forward/bv_solver/forms.py:build_forms()`: Production weak form assembly — MMS must be refactored to use this
- `Forward/bv_solver/config.py`: BV configuration parsing (k0, alpha, stoichiometry, multi-reaction mode)
- `Forward/bv_solver/nondim.py`: BV-specific nondim scaling additions
- MMS script uses `sys.path` hacks and standalone Firedrake setup — will need to integrate with production import paths

</code_context>

<specifics>
## Specific Ideas

- User explicitly stated no confidence in existing tests — validation of existing test correctness is a hard prerequisite before adding roundtrip tests
- The written audit document should go in the codebase (e.g., `docs/weak_form_audit.md` or `scripts/verification/WEAK_FORM_AUDIT.md`), not in `.planning/`
- MMS tests should use `@pytest.mark.firedrake` with skip-if-unavailable, keeping nondim roundtrip tests runnable without Firedrake

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-nondimensionalization-weak-form-audit*
*Context gathered: 2026-03-06*
