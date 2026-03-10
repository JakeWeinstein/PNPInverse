---
phase: 05-pipeline-reproducibility
plan: 01
subsystem: testing
tags: [pytest, regression-baselines, reproducibility, surrogate-inference, json]

# Dependency graph
requires:
  - phase: 04-inverse-problem-verification
    provides: "NN ensemble, _run_surrogate_phases(), SurrogateObjective infrastructure"
provides:
  - "--update-baselines conftest plugin and update_baselines fixture"
  - "Baseline load/save/diff helpers (_load_baselines, _save_baselines, _format_diff_table, _assert_baselines)"
  - "TestSurrogateReproducibility: fast S1+S2 reproducibility test at rel=1e-10"
  - "regression_baselines.json with surrogate-only baselines and metadata"
affects: [05-pipeline-reproducibility]

# Tech tracking
tech-stack:
  added: []
  patterns: ["--update-baselines pytest plugin", "JSON baselines with metadata", "diff table on regression failure"]

key-files:
  created:
    - "tests/test_pipeline_reproducibility.py"
    - "StudyResults/pipeline_reproducibility/regression_baselines.json"
  modified:
    - "tests/conftest.py"

key-decisions:
  - "surr_strategy='joint' runs only S1+S2 (no cascade/multistart) for fast test"
  - "Surrogate targets generated at true params via model.predict (no PDE needed for surrogate-only test)"
  - "Voltage grids replicated from v13 main() to ensure identical _run_surrogate_phases inputs"

patterns-established:
  - "Baseline infrastructure: _load_baselines / _save_baselines / _assert_baselines pattern reusable for full-pipeline test"
  - "Diff table format: Parameter | Baseline | Current | Abs Diff | Rel Diff | Tol | Pass/Fail"

requirements-completed: [PIP-01, PIP-02]

# Metrics
duration: 2min
completed: 2026-03-09
---

# Phase 5 Plan 01: Pipeline Reproducibility Summary

**Surrogate-only S1+S2 reproducibility test at rel=1e-10 with --update-baselines conftest plugin and JSON baseline infrastructure**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-09T19:10:10Z
- **Completed:** 2026-03-09T19:12:39Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- conftest.py now registers --update-baselines pytest option with update_baselines fixture
- Baseline helpers: _load_baselines, _save_baselines (with git/python/numpy metadata), _format_diff_table (tabular diff), _assert_baselines (compare-or-update)
- TestSurrogateReproducibility runs S1+S2 in <2s, verifies s1_alpha, surr_best_k0, surr_best_alpha, surr_best_loss at rel=1e-10
- regression_baselines.json generated with surrogate_only and surrogate_only_loss sections

## Task Commits

Each task was committed atomically:

1. **Task 1: Add --update-baselines conftest plugin and baseline helpers** - `742c8ab` (feat)
2. **Task 2: Implement TestSurrogateReproducibility and generate baselines** - `a0178bb` (feat)

**Plan metadata:** TBD (docs: complete plan)

## Files Created/Modified
- `tests/conftest.py` - Added pytest_addoption for --update-baselines and update_baselines fixture
- `tests/test_pipeline_reproducibility.py` - Baseline helpers + TestSurrogateReproducibility class
- `StudyResults/pipeline_reproducibility/regression_baselines.json` - Saved surrogate-only baselines with metadata

## Decisions Made
- Used surr_strategy="joint" to run only S1+S2 (fast path, no cascade or multistart)
- Surrogate targets at true params via model.predict() (no PDE needed for surrogate-only determinism test)
- Replicated v13 voltage grids (eta_symmetric + eta_shallow) directly in fixture for input fidelity

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Baseline infrastructure ready for Plan 02 (full 7-phase pipeline test)
- --update-baselines flag, _assert_baselines, and _format_diff_table are reusable for full-pipeline baselines
- The surrogate_only_results fixture demonstrates the pattern for calling _run_surrogate_phases programmatically

## Self-Check: PASSED

All files exist on disk, all commits verified in git log.

---
*Phase: 05-pipeline-reproducibility*
*Completed: 2026-03-09*
