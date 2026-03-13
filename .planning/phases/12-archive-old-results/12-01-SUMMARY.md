---
phase: 12-archive-old-results
plan: 01
subsystem: infra
tags: [archive, cleanup, study-results]

# Dependency graph
requires: []
provides:
  - Clean StudyResults/ with only 7 current directories
  - archive/StudyResults/ with 100 historical result directories and summary files
affects: [13-prune-dead-code, 14-docs-gitignore]

# Tech tracking
tech-stack:
  added: []
  patterns: [archive-directory-convention]

key-files:
  created:
    - archive/StudyResults/
  modified:
    - StudyResults/

key-decisions:
  - "Delete-then-archive two-commit strategy per user preference"
  - "Flat dump into archive/StudyResults/ with no subdirectory grouping"

patterns-established:
  - "Archive convention: old results go to archive/StudyResults/ preserving original directory names"

requirements-completed: [ARCH-01, ARCH-02]

# Metrics
duration: 5min
completed: 2026-03-13
---

# Phase 12 Plan 01: Archive Old StudyResults Summary

**Archived 96 old result directories and 4 summary files to archive/StudyResults/, deleted v14_pde_only and temporary artifacts, leaving 7 current directories in StudyResults/**

## Performance

- **Duration:** 5 min
- **Started:** 2026-03-13T04:40:12Z
- **Completed:** 2026-03-13T04:45:00Z
- **Tasks:** 2
- **Files modified:** 1201 (29 deleted + 1172 renamed)

## Accomplishments
- Deleted bad outputs (v14_pde_only/) and temporary artifacts (tmp dirs, loose logs) -- 29 files removed
- Archived 100 items (96 directories + 4 .md summary files) to archive/StudyResults/ with contents intact
- StudyResults/ now contains exactly 7 directories: master_inference_v13, v14, inverse_verification, surrogate_fidelity, mms_convergence, pipeline_reproducibility, target_cache

## Task Commits

Each task was committed atomically:

1. **Task 1: Delete bad outputs and temporary artifacts** - `44cc0a0` (chore)
2. **Task 2: Move old directories and loose .md files to archive** - `26fd04f` (chore)

## Files Created/Modified
- `archive/StudyResults/` - New archive location containing 100 items (96 dirs + 4 .md files)
- `StudyResults/` - Cleaned to 7 keep-set directories only

## Decisions Made
- Delete-then-archive two-commit strategy per user preference
- Flat dump into archive/StudyResults/ with no subdirectory grouping, no manifest
- v14_pde_only was untracked so removed with rm -rf rather than git rm

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- StudyResults/v14_pde_only/ was untracked (not in git index), so `git rm -r` would have failed. Used `rm -rf` instead. Similarly, mms_convergence/, target_cache/, and v14/ had untracked content that was kept in place (they are in the keep set).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- StudyResults/ is clean and ready for phase 13 (prune dead code) and phase 14 (docs/gitignore)
- archive/StudyResults/ provides historical reference if needed

## Self-Check: PASSED

- archive/StudyResults/: FOUND
- 12-01-SUMMARY.md: FOUND
- Commit 44cc0a0: FOUND
- Commit 26fd04f: FOUND
- StudyResults count: 7 (expected 7)
- Archive count: 100 (expected ~100)

---
*Phase: 12-archive-old-results*
*Completed: 2026-03-13*
