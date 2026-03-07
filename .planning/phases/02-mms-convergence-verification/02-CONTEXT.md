# Phase 2: MMS Convergence Verification - Context

**Gathered:** 2026-03-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Automated, publication-grade MMS convergence proof for the forward PDE solver with GCI uncertainty quantification. Wraps the existing 4-species MMS script into pytest with rate assertions and structured output. Requirements: FWD-01, FWD-03, FWD-05.

</domain>

<decisions>
## Implementation Decisions

### Mesh refinement levels
- 4 levels: N = 8, 16, 32, 64 for all test cases
- Consistent across all fields (concentrations + potential)
- N=128 excluded from pytest to keep runtime manageable

### Test case scope
- Only the 4-species case (O2, H2O2, H+, ClO4-) is wrapped as pytest — matches v13 production config (FWD-03)
- Simpler MMS scripts (1-species, 2-species neutral, 2-species charged in `mms_bv_convergence.py`) should be deprecated/removed
- The 4-species case subsumes the physics of the simpler cases (neutral + charged + multi-reaction + cathodic_conc_factors)

### GCI methodology
- Roache 3-grid GCI formula: GCI = Fs * |f2 - f1| / (r^p - 1)
- Safety factor Fs = 1.25 (standard for 3+ grids per Roache/ASME V&V 20)
- GCI is output only — no pytest assertions on GCI values
- Rate assertions (L2 ~ O(h^2), H1 ~ O(h), R-squared > 0.99 on log-log fit) are the pass/fail gate

### Test organization
- Test file: `tests/test_mms_convergence.py`
- Import `run_mms_4species()` from existing `scripts/verification/mms_bv_4species.py` — add rate assertions + GCI computation in the test
- Markers: `@pytest.mark.slow` + `@skip_without_firedrake` (consistent with existing pattern)
- No new custom markers

### Output artifacts
- Save convergence data to `StudyResults/mms_convergence/`
- JSON file with mesh sizes, errors, rates, and GCI values (machine-readable for Phase 6 report generation)
- Convergence plot (PNG) generated during pytest run alongside JSON data
- Both artifacts saved so Phase 6 can read them directly without re-running MMS

### Claude's Discretion
- Exact structure of the GCI computation function
- How to handle the log-log regression (numpy polyfit vs scipy)
- Convergence plot styling and layout
- Whether to add a GCI table to the text output or only include in JSON
- How to structure the JSON schema for convergence data

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `scripts/verification/mms_bv_4species.py:run_mms_4species()`: Core MMS driver — returns dict with N, h, and errors per field. Import directly into pytest.
- `scripts/verification/mms_bv_4species.py:compute_rates()`: Pairwise convergence rate computation — reuse for GCI observed order.
- `scripts/verification/mms_bv_4species.py:plot_convergence()`: Existing plotting function — may need minor adaptation for pytest output path.
- `scripts/verification/mms_bv_4species.py:_make_sp_mms_fixed()`: Builds SolverParams for 4-species MMS — already tested and working.
- `tests/conftest.py:skip_without_firedrake`: Existing skip decorator for Firedrake-dependent tests.

### Established Patterns
- `@pytest.mark.slow` + `@skip_without_firedrake` for Firedrake tests (see `test_v13_verification.py`, `test_bv_forward.py`)
- Test classes with class-level docstrings explaining what they cover
- `pytest.approx(expected, rel=...)` for floating-point assertions
- Test output saved to `StudyResults/` directory (existing convention from scripts)

### Integration Points
- `Forward.bv_solver.forms.build_context` / `build_forms`: Production weak form assembly (already used by mms_bv_4species.py)
- `Forward.params.SolverParams.from_list()`: SolverParams construction (already used by _make_sp_mms_fixed)
- Phase 6 will read `StudyResults/mms_convergence/*.json` for automated report generation

</code_context>

<specifics>
## Specific Ideas

- User wants only the 4-species case in pytest — the simpler cases are redundant since the 4-species case covers all the same physics plus cathodic_conc_factors and |s|=2 stoichiometry
- Simpler MMS scripts should be actively removed, not just left unused
- GCI is purely informational for the V&V report — the convergence rate R-squared is the actual test gate

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-mms-convergence-verification*
*Context gathered: 2026-03-07*
