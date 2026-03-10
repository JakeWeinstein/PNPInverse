---
phase: 03-surrogate-fidelity
plan: 02
subsystem: testing
tags: [surrogate, diagnostic-plots, matplotlib, nrmse, worst-case-overlay, scatter-plot]

# Dependency graph
requires:
  - phase: 03-surrogate-fidelity
    plan: 01
    provides: Hold-out validation fixtures, per-sample NRMSE arrays, model loading infrastructure
provides:
  - 12 diagnostic PNG plots (worst-case I-V overlay + error-vs-parameter scatter for all 4 models)
  - Cleaned test_v13_verification.py with subsumed Test 5 removed
affects: [06-report]

# Tech tracking
tech-stack:
  added: []
  patterns: [worst-case-overlay-diagnostics, error-vs-parameter-scatter]

key-files:
  created:
    - StudyResults/surrogate_fidelity/worst_iv_overlay_nn_ensemble.png
    - StudyResults/surrogate_fidelity/worst_iv_overlay_rbf_baseline.png
    - StudyResults/surrogate_fidelity/worst_iv_overlay_pod_rbf_log.png
    - StudyResults/surrogate_fidelity/worst_iv_overlay_pod_rbf_nolog.png
    - StudyResults/surrogate_fidelity/error_vs_params_cd_nn_ensemble.png
    - StudyResults/surrogate_fidelity/error_vs_params_pc_nn_ensemble.png
    - StudyResults/surrogate_fidelity/error_vs_params_cd_rbf_baseline.png
    - StudyResults/surrogate_fidelity/error_vs_params_pc_rbf_baseline.png
    - StudyResults/surrogate_fidelity/error_vs_params_cd_pod_rbf_log.png
    - StudyResults/surrogate_fidelity/error_vs_params_pc_pod_rbf_log.png
    - StudyResults/surrogate_fidelity/error_vs_params_cd_pod_rbf_nolog.png
    - StudyResults/surrogate_fidelity/error_vs_params_pc_pod_rbf_nolog.png
  modified:
    - tests/test_surrogate_fidelity.py
    - tests/test_v13_verification.py

key-decisions:
  - "Worst-case overlay uses top 3 highest CD NRMSE samples (not PC) since CD is the primary inference target"

patterns-established:
  - "Worst-case overlay: top N worst-NRMSE samples with PDE truth vs surrogate prediction overlaid"
  - "Error-vs-parameter scatter: NRMSE vs each parameter with log x-axis for rate constants"

requirements-completed: [SUR-01, SUR-03]

# Metrics
duration: 2min
completed: 2026-03-07
---

# Phase 3 Plan 02: Diagnostic Plots and Test Cleanup Summary

**12 diagnostic PNGs (worst-case I-V overlays and error-vs-parameter scatter) for all 4 surrogate models, plus removal of subsumed Test 5 from test_v13_verification.py**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-07T07:29:35Z
- **Completed:** 2026-03-07T07:32:18Z
- **Tasks:** 2
- **Files modified:** 14

## Accomplishments
- Generated worst-case I-V overlay plots showing top 3 highest-NRMSE samples per model with PDE truth vs surrogate curves
- Generated error-vs-parameter scatter plots (NRMSE vs k0_1, k0_2, alpha_1, alpha_2) for both CD and PC outputs across all 4 models
- Removed subsumed TestSurrogateVsPDEConsistency (Test 5) from test_v13_verification.py without breaking remaining tests
- All 9 tests pass in test_surrogate_fidelity.py; all 10 remaining tests collect in test_v13_verification.py

## Task Commits

Each task was committed atomically:

1. **Task 1: Add diagnostic plots to surrogate fidelity test** - `27b7540` (feat)
2. **Task 2: Remove subsumed TestSurrogateVsPDEConsistency from test_v13_verification.py** - `f776edf` (refactor)

**Plan metadata:** [pending final commit] (docs: complete plan)

## Files Created/Modified
- `tests/test_surrogate_fidelity.py` - Added generate_plots fixture and 2 plot verification tests
- `tests/test_v13_verification.py` - Removed TestSurrogateVsPDEConsistency class (122 lines), added subsumption comment
- `StudyResults/surrogate_fidelity/worst_iv_overlay_*.png` (4 files) - Top 3 worst-NRMSE I-V overlays per model
- `StudyResults/surrogate_fidelity/error_vs_params_cd_*.png` (4 files) - CD NRMSE vs 4 parameters per model
- `StudyResults/surrogate_fidelity/error_vs_params_pc_*.png` (4 files) - PC NRMSE vs 4 parameters per model

## Decisions Made
- Worst-case overlay ranks by CD NRMSE (not PC) since current density is the primary inference target and PC has near-zero-range outlier issues documented in Plan 01.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 3 (surrogate fidelity) fully complete with quantified error statistics and diagnostic visualizations
- All artifacts (JSON, CSV, 12 PNGs) ready for Phase 6 V&V report generation
- test_v13_verification.py cleaned of redundant surrogate-vs-PDE test

## Self-Check: PASSED

All 15 files verified present on disk. Both task commits (27b7540, f776edf) confirmed in git log.

---
*Phase: 03-surrogate-fidelity*
*Completed: 2026-03-07*
