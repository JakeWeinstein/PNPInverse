---
phase: 02-mms-convergence-verification
plan: 01
subsystem: testing
tags: [mms, convergence, gci, pytest, scipy, firedrake, pnp-bv]

# Dependency graph
requires:
  - phase: 01-nondim-audit
    provides: "Production build_forms() pipeline and nondim passthrough verified by MMS smoke tests"
provides:
  - "4-species MMS convergence pytest test with L2/H1 rate assertions and R-squared gating"
  - "GCI uncertainty quantification via Roache 3-grid formula (Fs=1.25)"
  - "Machine-readable convergence data JSON for Phase 6 report"
  - "Log-log convergence PNG plot for Phase 6 report"
affects: [06-vv-report]

# Tech tracking
tech-stack:
  added: []
  patterns: [class-scoped-fixture, lazy-firedrake-import, numpy-json-encoder, gci-roache]

key-files:
  created:
    - tests/test_mms_convergence.py
  modified:
    - pyproject.toml

key-decisions:
  - "Removed deprecated 1/2-species MMS scripts (mms_bv_convergence.py) since 4-species case strictly subsumes them"
  - "Removed orphaned 'firedrake' pytest marker; all Firedrake tests now use @pytest.mark.slow"
  - "Class-scoped fixture runs MMS study once for all 4 test methods (test_l2, test_h1, test_gci, test_artifacts)"

patterns-established:
  - "assert_convergence_rate(): reusable log-log regression assertion helper with rate tolerance and R-squared gating"
  - "compute_gci(): reusable Roache GCI computation for any error sequence"
  - "_NumpyEncoder: JSON encoder for numpy types in test artifact output"

requirements-completed: [FWD-01, FWD-03, FWD-05]

# Metrics
duration: 2min
completed: 2026-03-07
---

# Phase 2 Plan 01: MMS Convergence Test Summary

**4-species MMS convergence pytest with L2/H1 rate assertions (R^2 > 0.99), GCI uncertainty (Roache Fs=1.25), and JSON+PNG artifacts for V&V report**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-07T06:41:10Z
- **Completed:** 2026-03-07T06:42:57Z
- **Tasks:** 2
- **Files modified:** 4 (1 created, 3 deleted/modified)

## Accomplishments
- Created `tests/test_mms_convergence.py` with TestMMSConvergence class containing 4 test methods covering L2 rates, H1 rates, GCI computation, and artifact generation
- Formal convergence assertions: L2 rate in [1.8, 2.2], H1 rate in [0.8, 1.2], R-squared > 0.99 for all 5 fields (c0-c3, phi)
- GCI uncertainty via Roache 3-grid formula with Fs=1.25, sanity-checked for finiteness and non-negativity
- Removed deprecated 1/2-species MMS scripts and smoke tests, cleaned up orphaned pytest marker

## Task Commits

Each task was committed atomically:

1. **Task 1: Create 4-species MMS convergence test** - `f190cf8` (feat)
2. **Task 2: Remove deprecated MMS scripts and smoke tests** - `7c94656` (refactor)

## Files Created/Modified
- `tests/test_mms_convergence.py` - 4-species MMS convergence test with rate assertions, GCI, and artifact output
- `scripts/verification/mms_bv_convergence.py` - DELETED (1/2-species MMS subsumed by 4-species)
- `tests/test_mms_smoke.py` - DELETED (replaced by test_mms_convergence.py)
- `pyproject.toml` - Removed orphaned 'firedrake' marker registration

## Decisions Made
- Removed deprecated mms_bv_convergence.py and test_mms_smoke.py since 4-species case provides strictly stronger coverage
- Removed 'firedrake' pytest marker from pyproject.toml since only the deleted smoke test used it; new test uses @pytest.mark.slow
- Used class-scoped fixture so the expensive MMS study runs once across all 4 test methods

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Forward solver convergence verification complete (FWD-01, FWD-03, FWD-05)
- JSON and PNG artifacts will be generated at test runtime in StudyResults/mms_convergence/
- Phase 3 (Surrogate Fidelity) and Phase 6 (V&V Report) can proceed

---
*Phase: 02-mms-convergence-verification*
*Completed: 2026-03-07*
