---
phase: 07-baseline-diagnostics
plan: 01
subsystem: testing
tags: [multi-seed, subprocess, numpy, matplotlib, aggregation, diagnostics, AUDT-04]

# Dependency graph
requires: []
provides:
  - Multi-seed v13 wrapper with subprocess isolation
  - AUDT-04 metadata schema validation infrastructure
  - parse_v13_csv and aggregate_seed_results reusable functions
affects: [07-02, 07-03, 08-ablation-audit]

# Tech tracking
tech-stack:
  added: [subprocess isolation, csv.DictReader/DictWriter]
  patterns: [frozen dataclass config, tagged print logging, TDD red-green]

key-files:
  created:
    - scripts/studies/run_multi_seed_v13.py
    - tests/test_diagnostic_metadata.py
    - tests/test_multi_seed_aggregation.py
  modified: []

key-decisions:
  - "Sequential seed execution (not parallel) to avoid Firedrake/PETSc process conflicts"
  - "numpy.percentile for IQR computation (consistent with scipy conventions)"
  - "validate_metadata helper defined in test file for reuse by other diagnostic test modules"

patterns-established:
  - "AUDT-04 metadata sidecar: every diagnostic tool writes metadata.json with required keys"
  - "parse_v13_csv + aggregate_seed_results: reusable extraction/aggregation for v13 output"

requirements-completed: [DIAG-01, AUDT-04]

# Metrics
duration: 3min
completed: 2026-03-10
---

# Phase 7 Plan 01: Multi-Seed v13 Baseline Summary

**Multi-seed v13 wrapper running 20 noise seeds via subprocess with per-parameter median/IQR/max aggregation, box plots, and AUDT-04 metadata validation**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-10T16:16:31Z
- **Completed:** 2026-03-10T16:19:41Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Created multi-seed v13 wrapper script with subprocess isolation per seed, full CLI, and 900s timeout
- Implemented parse_v13_csv (P2 row extraction) and aggregate_seed_results (median/IQR/max per parameter)
- Built box plot and per-seed scatter plot generation with outlier highlighting (>2x median)
- Created AUDT-04 metadata schema validation test infrastructure reusable by all Phase 7 plans
- All 9 unit tests pass covering metadata validation, CSV parsing, and aggregation logic

## Task Commits

Each task was committed atomically:

1. **Task 1: Create AUDT-04 metadata tests and multi-seed aggregation tests** - `1279125` (test) - RED phase
2. **Task 2: Implement multi-seed wrapper script** - `28d6a59` (feat) - GREEN phase

## Files Created/Modified
- `scripts/studies/run_multi_seed_v13.py` - Multi-seed wrapper with subprocess isolation, aggregation, plots, metadata
- `tests/test_diagnostic_metadata.py` - AUDT-04 metadata schema validation tests and validate_metadata helper
- `tests/test_multi_seed_aggregation.py` - parse_v13_csv and aggregate_seed_results tests with mock data

## Decisions Made
- Sequential seed execution chosen over parallel to avoid Firedrake/PETSc process conflicts in subprocess
- numpy.percentile used for IQR (p25/p75) computation for consistency with scipy conventions
- validate_metadata helper placed in test file (not a separate module) since it is only needed by tests

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- v13 baseline infrastructure ready; running `python scripts/studies/run_multi_seed_v13.py` will produce full 20-seed assessment
- AUDT-04 metadata validation tests available for import by Plans 07-02 and 07-03
- parse_v13_csv and aggregate_seed_results importable for downstream diagnostic scripts

---
*Phase: 07-baseline-diagnostics*
*Completed: 2026-03-10*

## Self-Check: PASSED

All files exist. All commits verified.
