---
phase: 13-delete-dead-code
plan: "01"
subsystem: infra
tags: [cleanup, dead-code, repo-hygiene]

requires:
  - phase: 12-archive-old-results
    provides: "Old StudyResults archived so scripts referencing them can be safely deleted"
provides:
  - "Clean repo with only v13 pipeline scripts and relevant tests"
  - "scripts/inference/ directory removed entirely"
  - "scripts/bv/ directory removed entirely"
affects: [14-docs-and-final-cleanup]

tech-stack:
  added: []
  patterns: [v13-only-pipeline]

key-files:
  created: []
  modified:
    - scripts/surrogate/ (13 files deleted, 8 kept)
    - scripts/studies/ (15 files deleted, 4 kept)
    - scripts/inference/ (42 files deleted, directory removed)
    - scripts/bv/ (3 files deleted, directory removed)
    - tests/ (10 test files deleted, 16 kept)

key-decisions:
  - "Single atomic commit for all deletions per user preference"
  - "Infer_PDE_only_v14.py deleted (untracked, removed with rm instead of git rm)"
  - "generate_presentation_plots.py deleted (untracked, removed with rm instead of git rm)"

patterns-established:
  - "v13-only: repo contains only v13 pipeline and supporting utilities"

requirements-completed: [SCRP-01, SCRP-02, SCRP-03, SCRP-04, SCRP-05, TEST-01]

duration: 1min
completed: 2026-03-13
---

# Phase 13 Plan 01: Delete Dead Code Summary

**Removed 82 tracked files (35,536 lines) plus 2 untracked files -- repo stripped to v13 pipeline essentials**

## Performance

- **Duration:** 1 min
- **Started:** 2026-03-13T05:03:39Z
- **Completed:** 2026-03-13T05:05:00Z
- **Tasks:** 2
- **Files modified:** 84 deleted (82 tracked + 2 untracked)

## Accomplishments
- Deleted all 42 v1-v7 inference scripts and removed scripts/inference/ directory
- Deleted 13 dead surrogate scripts (v8-v12, bcd, cascade, v14) while preserving 8 kept utilities
- Deleted 15 dead study/benchmark scripts while preserving 4 kept studies
- Deleted 3 BV scripts and removed scripts/bv/ directory
- Deleted 10 dead test files while preserving 15 test files + conftest.py
- Single atomic commit: e6a026d

## Task Commits

Both tasks committed as a single atomic commit per plan specification:

1. **Task 1: Delete all dead scripts** - `e6a026d` (chore) -- combined with Task 2
2. **Task 2: Delete dead test files and commit** - `e6a026d` (chore)

## Files Created/Modified
- `scripts/inference/` - 42 files deleted, directory removed
- `scripts/surrogate/` - 13 dead scripts deleted (v8, v8.1, v9, v10, v11, v12, v14, bcd, cascade, cascade_pde_hybrid, run_nopde_batch, run_v9_pipeline.sh, sweep_secondary_weight)
- `scripts/studies/` - 15 dead scripts deleted (benchmarks, BV studies, Robin studies, voltage range studies)
- `scripts/bv/` - 3 scripts deleted (bv_iv_curve.py, bv_iv_curve_symmetric.py, bv_iv_curve_charged.py), directory removed
- `scripts/generate_presentation_plots.py` - deleted
- `tests/` - 10 dead test files deleted

### Kept Files (verified present)
- `scripts/surrogate/Infer_BVMaster_charged_v13_ultimate.py` - v13 pipeline
- `scripts/_bv_common.py` - shared BV utilities
- `scripts/surrogate/build_surrogate.py`, `generate_training_data.py`, `train_nn_surrogate.py`, `validate_surrogate.py`, `multistart_inference.py`, `overnight_train_v11.py`, `train_improved_surrogate.py` - utilities
- `scripts/studies/run_multi_seed_v13.py`, `profile_likelihood_study.py`, `profile_likelihood_pde.py`, `sensitivity_visualization.py` - kept studies
- `tests/test_v13_verification.py` and 14 other test files + `conftest.py`

## Decisions Made
- Single atomic commit for all deletions (Tasks 1 and 2 combined) per plan specification
- Two files (Infer_PDE_only_v14.py, generate_presentation_plots.py) were untracked -- deleted with `rm` instead of `git rm`

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Infer_PDE_only_v14.py was untracked, not tracked**
- **Found during:** Task 1
- **Issue:** `git rm` failed because the file was untracked (shown as `??` in git status)
- **Fix:** Used `rm` instead of `git rm` to delete the file
- **Files modified:** scripts/surrogate/Infer_PDE_only_v14.py
- **Verification:** File no longer exists on disk
- **Committed in:** N/A (untracked file, not in any commit)

**2. [Rule 3 - Blocking] generate_presentation_plots.py was untracked**
- **Found during:** Task 1
- **Issue:** `git rm` returned "did not match" because file was untracked
- **Fix:** Used `rm` instead of `git rm`
- **Files modified:** scripts/generate_presentation_plots.py
- **Verification:** File no longer exists on disk
- **Committed in:** N/A (untracked file)

---

**Total deviations:** 2 auto-fixed (2 blocking)
**Impact on plan:** Minor -- untracked files required `rm` instead of `git rm`. Same end result.

## Issues Encountered
None beyond the untracked file handling noted in deviations.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Repo is stripped to v13 pipeline essentials
- Ready for Phase 14 (docs and final cleanup)
- All kept scripts verified present and untouched

---
*Phase: 13-delete-dead-code*
*Completed: 2026-03-13*
