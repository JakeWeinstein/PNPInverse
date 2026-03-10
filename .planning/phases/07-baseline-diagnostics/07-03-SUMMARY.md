---
phase: 07-baseline-diagnostics
plan: 03
subsystem: diagnostics
tags: [sensitivity, jacobian, finite-differences, parameter-sweep, matplotlib]

# Dependency graph
requires:
  - phase: 07-baseline-diagnostics
    provides: "Forward solver infrastructure (_bv_common, bv_point_solve)"
provides:
  - "1D parameter sweep visualization for k0_1, k0_2, alpha_1, alpha_2"
  - "Jacobian heatmap d(observable)/d(parameter) at each voltage"
  - "Extended voltage grid beyond v13 with warm-starting"
  - "AUDT-04 metadata sidecar for DIAG-03"
affects: [09-objective-component, phase-9-voltage-selection]

# Tech tracking
tech-stack:
  added: []
  patterns: [central-finite-differences, warm-start-continuation, frozen-dataclass-config]

key-files:
  created:
    - scripts/studies/sensitivity_visualization.py
    - tests/test_sensitivity_visualization.py
  modified: []

key-decisions:
  - "Extended voltage grid to -60 by default with optional extension to -75"
  - "Central FD step h=1e-5 consistent with codebase convention"
  - "Separate forward solves for total vs peroxide current (observable_reaction_index)"
  - "Column-normalized Jacobian heatmap for cross-parameter comparison"

patterns-established:
  - "SensitivityConfig frozen dataclass for analysis configuration"
  - "Lazy Firedrake import pattern for testability of pure functions"

requirements-completed: [DIAG-03, AUDT-04]

# Metrics
duration: 4min
completed: 2026-03-10
---

# Phase 7 Plan 3: Sensitivity Visualization Summary

**1D parameter sweeps + Jacobian heatmap via central FD (h=1e-5) with extended voltage range and warm-start continuation**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-10T16:16:41Z
- **Completed:** 2026-03-10T16:20:30Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Sensitivity visualization script with 13 components (config, helpers, evaluation, plotting, CSV, metadata, CLI)
- Extended voltage grid beyond v13 default (-46.5) to -60 with descending order for warm-start continuation
- Jacobian heatmap showing d(observable)/d(parameter) at each voltage for both total and peroxide current
- AUDT-04 compliant metadata sidecar documenting DIAG-03 justification
- 14 unit tests covering all pure helper functions

## Task Commits

Each task was committed atomically:

1. **Task 1: Create sensitivity visualization tests** - `1279125` (test) - TDD RED phase
2. **Task 2: Implement sensitivity visualization script** - `3e728c9` (feat) - TDD GREEN phase

_Note: TDD tasks have RED (test) then GREEN (feat) commits._

## Files Created/Modified
- `scripts/studies/sensitivity_visualization.py` - 1D parameter sweeps, Jacobian heatmap, CSV output, metadata, CLI (786 lines)
- `tests/test_sensitivity_visualization.py` - Unit tests for sweep factors, voltage grid, Jacobian FD, parameter perturbation, metadata (169 lines)

## Decisions Made
- Extended voltage grid to -60 by default (matches RESEARCH.md guidance); CLI allows extension to -75
- Central FD step h=1e-5 matches codebase convention used in gradient verification
- Separate forward solves for total vs peroxide current using observable_reaction_index parameter
- Column-normalized Jacobian heatmap enables cross-parameter visual comparison despite different scales
- Lazy Firedrake import pattern keeps pure helper functions testable without Firedrake environment

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Sensitivity analysis tool ready for use in Phase 9 voltage selection strategy
- Jacobian heatmap will identify which voltage regions carry most information about each parameter
- Extended voltage range validated for solver convergence configuration

## Self-Check: PASSED

All files and commits verified.

---
*Phase: 07-baseline-diagnostics*
*Completed: 2026-03-10*
