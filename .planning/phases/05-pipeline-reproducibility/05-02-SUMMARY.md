---
phase: 05-pipeline-reproducibility
plan: 02
subsystem: testing
tags: [pytest, regression-baselines, reproducibility, full-pipeline, subprocess, firedrake, pde-solver]

# Dependency graph
requires:
  - phase: 05-pipeline-reproducibility/01
    provides: "Baseline infrastructure (_assert_baselines, --update-baselines, regression_baselines.json)"
  - phase: 04-inverse-problem-verification
    provides: "v13 pipeline script (Infer_BVMaster_charged_v13_ultimate.py)"
provides:
  - "TestFullPipelineReproducibility: slow integration test running full 7-phase pipeline via subprocess"
  - "Full pipeline baselines in regression_baselines.json (full_pipeline, full_pipeline_loss, full_pipeline_s1 sections)"
  - "End-to-end determinism proof for v13 pipeline at rel=1e-4 solver tolerance"
affects: [06-vv-report]

# Tech tracking
tech-stack:
  added: []
  patterns: ["subprocess pipeline execution to avoid Firedrake/PyTorch segfault", "CSV parsing of pipeline output for regression comparison"]

key-files:
  created: []
  modified:
    - "tests/test_pipeline_reproducibility.py"
    - "StudyResults/pipeline_reproducibility/regression_baselines.json"

key-decisions:
  - "Subprocess execution for full pipeline (avoids PETSc/PyTorch segfault in test process)"
  - "rel=1e-4 tolerance for PDE-refined parameters (matches PDE solver tolerance)"
  - "Three separate test methods: parameters, loss (with abs=1e-8 floor), and S1 intermediates"

patterns-established:
  - "Subprocess pipeline test pattern: run script via subprocess, parse output CSV, compare against baselines"

requirements-completed: [PIP-01, PIP-02]

# Metrics
duration: 7min
completed: 2026-03-09
---

# Phase 5 Plan 02: Full Pipeline Reproducibility Summary

**Full 7-phase v13 pipeline (S1-S5 + P1-P2) determinism test via subprocess with PDE-solver-tolerance baselines at rel=1e-4**

## Performance

- **Duration:** 7 min (including ~5:43 user verification runtime)
- **Started:** 2026-03-09T19:15:00Z
- **Completed:** 2026-03-09T19:36:00Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- TestFullPipelineReproducibility runs the full v13 7-phase pipeline via subprocess, parses master_comparison_v13.csv, and compares against saved baselines
- Three test methods cover parameter regression (S1 + P2 + final), objective function regression (with abs=1e-8 floor), and S1 intermediate regression
- All 5 tests pass (2 surrogate-only from Plan 01 + 3 full pipeline) in venv-firedrake environment
- regression_baselines.json now contains full_pipeline, full_pipeline_loss, and full_pipeline_s1 sections alongside surrogate-only baselines

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement TestFullPipelineReproducibility with subprocess execution** - `a0178bb` (feat)
2. **Task 2: User verifies pipeline reproducibility tests** - `ae0adc1` (feat, baselines update after verification)

**Plan metadata:** TBD (docs: complete plan)

## Files Created/Modified
- `tests/test_pipeline_reproducibility.py` - Added TestFullPipelineReproducibility class with subprocess execution, CSV parsing, and 3 test methods
- `StudyResults/pipeline_reproducibility/regression_baselines.json` - Added full_pipeline, full_pipeline_loss, full_pipeline_s1 baseline sections

## Decisions Made
- Subprocess execution avoids Firedrake/PyTorch PETSc segfault (same pattern as 04-03 PDE target generation)
- rel=1e-4 matches PDE solver tolerance for P1/P2 phases; abs=1e-8 floor for near-zero loss values
- S1 intermediates tested separately to isolate surrogate-only vs PDE-refined regression

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 5 (Pipeline Reproducibility) is now complete: both surrogate-only and full-pipeline tests in place
- All v1 requirements except RPT-01 are complete
- Ready for Phase 6 (V&V Report)

## Self-Check: PASSED

All files exist on disk, all commits verified in git log.

---
*Phase: 05-pipeline-reproducibility*
*Completed: 2026-03-09*
