---
phase: 07-baseline-diagnostics
plan: 02
subsystem: diagnostics
tags: [profile-likelihood, identifiability, chi-squared, l-bfgs-b, pde-inverse]

# Dependency graph
requires:
  - phase: v13 inference pipeline
    provides: BVFluxCurveInferenceRequest, run_bv_multi_observable_flux_curve_inference, _bv_common constants
provides:
  - PDE-only profile likelihood script (30-point profiles for 4 kinetic parameters)
  - Chi-squared identifiability assessment with 95% CI threshold
  - Identifiability summary CSV and AUDT-04 metadata JSON sidecar
affects: [07-03, 08-ablation-audit, 09-objective-component]

# Tech tracking
tech-stack:
  added: []
  patterns: [frozen-dataclass-config, build-assess-write separation, chi2-identifiability]

key-files:
  created:
    - scripts/studies/profile_likelihood_pde.py
    - tests/test_profile_likelihood.py
  modified: []

key-decisions:
  - "Joint control_mode with per-component bounds for pinning profiled parameter while re-optimizing remaining 3"
  - "k0_2 gets 6-decade grid (0.001x-1000x) vs 4-decade for k0_1 due to identifiability risk"
  - "Multi-observable (current_density + peroxide_current) objective for profile evaluation, matching v13 pipeline"

patterns-established:
  - "Profile likelihood pattern: build_profile_grid -> run_profile -> assess_identifiability -> write outputs"
  - "Immutable config via frozen dataclass (ProfileLikelihoodConfig)"

requirements-completed: [DIAG-02, AUDT-04]

# Metrics
duration: 3min
completed: 2026-03-10
---

# Phase 7 Plan 2: Profile Likelihood Summary

**PDE-only profile likelihood with chi2 identifiability assessment for k0_1, k0_2, alpha_1, alpha_2 using 30-point grids and joint re-optimization**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-10T16:16:35Z
- **Completed:** 2026-03-10T16:20:16Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- TDD-driven implementation: 10 unit tests covering grid construction, identifiability assessment, and bound construction
- Profile likelihood script with all 10 components from plan (config, grid, assess, bounds, runner, plots, CSVs, summary, metadata, CLI)
- k0_2 gets intentionally wider grid range (0.001x-1000x vs standard 0.01x-100x) to probe identifiability boundary
- AUDT-04 metadata sidecar documents Raue et al. 2009 justification for profile likelihood approach

## Task Commits

Each task was committed atomically:

1. **Task 1: Create profile likelihood tests (RED)** - `549d19c` (test)
2. **Task 2: Implement PDE-only profile likelihood script (GREEN)** - `826eedf` (feat)

_TDD: Task 1 wrote failing tests, Task 2 made them pass._

## Files Created/Modified
- `scripts/studies/profile_likelihood_pde.py` - PDE-only profile likelihood analysis (740 lines): grid construction, identifiability assessment, bound pinning, PDE re-optimization runner, plot/CSV/JSON writers, CLI
- `tests/test_profile_likelihood.py` - 10 test cases (176 lines): grid ranges, chi2 assessment, bound construction, immutability

## Decisions Made
- Used joint control_mode with k0_lower_per_component/k0_upper_per_component to pin one parameter while freeing others (plan specified joint re-optimization of remaining 3)
- Multi-observable objective (current_density + peroxide_current) for profile evaluation, consistent with v13 master inference
- assess_identifiability handles degenerate case (sigma2_hat near zero) gracefully

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Profile likelihood script ready for execution (requires Firedrake environment with PDE solver)
- k0_2 identifiability question (DIAG-02) will be answered when script runs on full 30-point grid
- Results feed into phase 8 ablation audit and phase 9 objective/component decisions

---
*Phase: 07-baseline-diagnostics*
*Completed: 2026-03-10*
