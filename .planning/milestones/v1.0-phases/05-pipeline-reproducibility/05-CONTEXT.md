# Phase 5: Pipeline Reproducibility - Context

**Gathered:** 2026-03-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Prove the v13 inference pipeline produces deterministic, regression-tested results. Run the full 7-phase pipeline (S1-S5 surrogate + P1-P2 PDE refinement) against saved baselines and detect numerical drift. Requirements: PIP-01, PIP-02.

</domain>

<decisions>
## Implementation Decisions

### Reproducibility scope
- Run the full 7-phase v13 pipeline end-to-end (S1-S5 + P1-P2), not just individual layer outputs
- Use PDE-generated synthetic targets (same approach as Phase 4, reuse pde_targets_cache.npz pattern)
- Compare one fresh run against saved baseline values (run-vs-baseline, not run-vs-run)
- Compare: final inferred parameters (k0_1, k0_2, alpha_1, alpha_2) + final loss + key intermediates (S1 output and P2 output)

### Regression baseline strategy
- Baselines stored as JSON in `StudyResults/pipeline_reproducibility/regression_baselines.json`
- Custom `--update-baselines` pytest flag to regenerate baselines (test fails if baselines don't exist and flag not passed)
- Baseline JSON includes metadata: generation timestamp, git commit hash, Python/NumPy versions
- Baselines cover full-pipeline outputs only (Phases 2-4 tests already assert on their own artifacts)

### Drift tolerance thresholds
- Solver-tolerance-aware: surrogate-only outputs rel=1e-10 (float64 deterministic), PDE-refined outputs rel=1e-4 (solver tolerance)
- Final parameters: rel=1e-4 (dominated by PDE refinement precision)
- Objective function values: relative tolerance with absolute floor (pytest.approx with rel=1e-4, abs=1e-8)
- On failure: detailed diff table showing parameter name, baseline value, current value, absolute diff, relative diff, tolerance, pass/fail
- Pass/fail only — no warning band

### Test organization and runtime
- New file: `tests/test_pipeline_reproducibility.py`
- Two test classes:
  - Fast surrogate-only reproducibility (S1+S2, no Firedrake) — runs in every test session
  - Full 7-phase pipeline reproducibility — `@pytest.mark.slow` + `@skip_without_firedrake`
- Single `--update-baselines` flag updates both surrogate-only and full-pipeline baselines
- Follows existing marker conventions from Phases 2-4

### Claude's Discretion
- How to invoke the 7-phase pipeline programmatically (import from v13 script or wrap subprocess)
- How to implement the --update-baselines conftest plugin
- Fixture organization for PDE target generation and caching
- Exact JSON schema for baseline storage
- How to extract S1 and P2 intermediate outputs from the pipeline

</decisions>

<specifics>
## Specific Ideas

- The fast surrogate-only test complements the slow full-pipeline test, giving reproducibility coverage in quick test runs without Firedrake
- PDE-generated targets match Phase 4's approach (no inverse crime, reuse caching pattern)
- The diff table on failure should make it immediately clear what changed and by how much, without needing to dig through logs

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `scripts/surrogate/Infer_BVMaster_charged_v13_ultimate.py`: 7-phase pipeline entry point
- `scripts/_bv_common.py`: True parameter values, `_generate_targets_with_pde()` caching, `make_bv_solver_params()`
- `Surrogate/objectives.py:SurrogateObjective`: Surrogate-based inference (S1+S2 fast path)
- `Surrogate/multistart.py:run_multistart_inference()`: Production multistart with seed control
- `tests/test_inverse_verification.py:_pde_cd_at_params()`: Subprocess-based PDE target generation with disk caching
- `tests/conftest.py:skip_without_firedrake`: Skip decorator for Firedrake-dependent tests

### Established Patterns
- `StudyResults/` for test output artifacts (JSON + CSV + PNG) — all phases
- `@pytest.mark.slow` + `@skip_without_firedrake` for expensive Firedrake tests
- Module-scoped fixtures for shared model/data loading
- Seeds used consistently: 42, 43, 44 in Phase 4; seed=42 in multistart
- JSON artifacts with structured data for downstream consumption (Phase 6 report)

### Integration Points
- Phase 4's `pde_targets_cache.npz` pattern for target generation
- v13 pipeline script for full 7-phase execution
- `StudyResults/pipeline_reproducibility/` for baseline storage
- Phase 6 will read reproducibility results for the V&V report

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 05-pipeline-reproducibility*
*Context gathered: 2026-03-09*
