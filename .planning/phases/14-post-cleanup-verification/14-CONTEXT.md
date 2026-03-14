# Phase 14: Post-Cleanup Verification - Context

**Gathered:** 2026-03-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Confirm that the v13 pipeline, all remaining scripts, src/ library modules, and the kept pytest test suite work correctly after phases 12-13 deleted ~84 files and archived ~108 directories. Fix broken imports; report deeper test failures.

</domain>

<decisions>
## Implementation Decisions

### Failure handling
- Fix broken imports in both scripts and tests (update paths, remove stale references)
- If a test file primarily tests deleted code, delete the entire test file
- Report deeper test failures (non-import issues) without fixing — document them for follow-up
- Import fixes apply to production scripts, src/ modules, AND test files

### Verification scope
- Verify ALL 15 remaining scripts in scripts/ (verification, surrogate, studies, _bv_common.py)
- Verify all src/ library modules (pnp_solver, surrogates, etc.) import cleanly
- Verify all 16 test files in tests/
- Two-pass approach: static analysis first (AST parse to find obvious broken imports), then runtime import check

### Test environment
- Run the full pytest suite to completion, no timeout
- Activate `venv-firedrake` from parent directory before running tests
- MMS convergence and PDE tests expected to take several minutes — let them finish
- Any Firedrake test failures are real issues to report (env will be properly activated)

### Claude's Discretion
- Order of verification steps (static scan vs runtime vs pytest)
- How to structure the import-check script/commands
- Level of detail in the failure report

</decisions>

<specifics>
## Specific Ideas

- Firedrake virtualenv location: `../venv-firedrake` (parent directory of project)
- The v13 master script is `scripts/surrogate/Infer_BVMaster_charged_v13_ultimate.py`
- Phase 13 deleted 82 tracked files + 2 untracked files in a single atomic commit

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `tests/conftest.py`: Shared test fixtures and configuration
- `scripts/_bv_common.py`: Shared utilities across scripts

### Established Patterns
- Subprocess isolation for Firedrake/PyTorch PETSc conflicts (PDE targets run in subprocess)
- `sys.path` hacks in scripts for module imports (known tech debt, not to be fixed here)

### Integration Points
- Scripts import from src/ packages (pnp_solver, surrogates, etc.)
- Tests import from both src/ and scripts/
- Some tests use subprocess to invoke scripts (test_pipeline_reproducibility, test_v13_verification)

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 14-post-cleanup-verification*
*Context gathered: 2026-03-14*
