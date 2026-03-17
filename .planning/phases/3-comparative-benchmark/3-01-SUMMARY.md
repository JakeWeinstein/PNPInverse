---
phase: 3-comparative-benchmark
plan: 01
subsystem: testing
tags: [surrogate, fidelity, benchmark, gp, pce, nrmse]

requires: []
provides:
  - "6-model head-to-head fidelity benchmark on 524 hold-out samples"
  - "fidelity_summary.json with aggregate NRMSE stats for all 6 surrogates"
  - "per_sample_errors.csv for downstream k0_2-stratified analysis"
  - "Worst-case I-V overlay and error-vs-parameter scatter plots for GP and PCE"
affects: [3-02, 3-03, 3-04, 3-05, phase-4-ismo, phase-5-pde-refinement]

tech-stack:
  added: []
  patterns: ["Dynamic MODEL_NAMES based on available dependencies and model files"]

key-files:
  created:
    - "StudyResults/surrogate_fidelity/worst_iv_overlay_gp_fixed.png"
    - "StudyResults/surrogate_fidelity/worst_iv_overlay_pce.png"
    - "StudyResults/surrogate_fidelity/error_vs_params_cd_gp_fixed.png"
    - "StudyResults/surrogate_fidelity/error_vs_params_pc_gp_fixed.png"
    - "StudyResults/surrogate_fidelity/error_vs_params_cd_pce.png"
    - "StudyResults/surrogate_fidelity/error_vs_params_pc_pce.png"
  modified:
    - "tests/test_surrogate_fidelity.py"
    - "StudyResults/surrogate_fidelity/fidelity_summary.json"
    - "StudyResults/surrogate_fidelity/per_sample_errors.csv"

key-decisions:
  - "Dynamic MODEL_NAMES: GP and PCE appended only when dependencies (gpytorch, chaospy) and model files exist on disk"
  - "GP model ranks #1 by CD median NRMSE (1.03%) -- best current density predictor"

patterns-established:
  - "Conditional model inclusion: try/except import + os.path check gates model into benchmark"

requirements-completed: []

duration: 3min
completed: 2026-03-17
---

# Phase 3 Plan 01: Prediction Accuracy Benchmark Summary

**6-model surrogate fidelity benchmark on 524 hold-out samples: GP ranks #1 for CD (1.03% median NRMSE), PCE ranks #6 (3.44%); all models pass 20% soft gate**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-17T23:02:35Z
- **Completed:** 2026-03-17T23:05:15Z
- **Tasks:** 2
- **Files modified:** 9

## Accomplishments
- Extended surrogate fidelity benchmark from 4 to 6 models (added gp_fixed, pce)
- All 11 parametrized tests pass including 6 per-model NRMSE threshold checks
- Generated 18 total plot files (6 overlays + 12 scatter) for visual diagnosis
- Produced ranking: gp_fixed > pod_rbf_log > pod_rbf_nolog > nn_ensemble > rbf_baseline > pce

## Model Ranking (CD median NRMSE)

| Rank | Model        | CD median | PC median |
|------|-------------|-----------|-----------|
| 1    | gp_fixed     | 1.03%     | 2.24%     |
| 2    | pod_rbf_log  | 1.15%     | 3.39%     |
| 3    | pod_rbf_nolog| 1.16%     | 3.58%     |
| 4    | nn_ensemble  | 1.21%     | 2.80%     |
| 5    | rbf_baseline | 1.25%     | 2.21%     |
| 6    | pce          | 3.44%     | 15.84%    |

## Task Commits

Each task was committed atomically:

1. **Task 1: Extend test_surrogate_fidelity.py** - `192eca9` (feat)
2. **Task 2: Run benchmark and verify artifacts** - `2f8a743` (feat)

## Files Created/Modified
- `tests/test_surrogate_fidelity.py` - Extended from 4 to 6 models with conditional GP/PCE loading
- `StudyResults/surrogate_fidelity/fidelity_summary.json` - Updated with all 6 model entries (8 stats each)
- `StudyResults/surrogate_fidelity/per_sample_errors.csv` - Extended to 17 columns (5 param + 12 error)
- `StudyResults/surrogate_fidelity/worst_iv_overlay_gp_fixed.png` - GP worst-case I-V overlay
- `StudyResults/surrogate_fidelity/worst_iv_overlay_pce.png` - PCE worst-case I-V overlay
- `StudyResults/surrogate_fidelity/error_vs_params_{cd,pc}_gp_fixed.png` - GP error scatter
- `StudyResults/surrogate_fidelity/error_vs_params_{cd,pc}_pce.png` - PCE error scatter

## Decisions Made
- Dynamic MODEL_NAMES: models only included when their Python dependencies and saved model files exist, preventing test failures in environments without gpytorch or chaospy
- Hold-out set expanded from 479 (plan estimate) to 524 samples due to updated training data split

## Deviations from Plan

None - plan executed exactly as written. The hold-out set size is 524 (not 479 as estimated in plan) because the training data was augmented since the plan was written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- 6-model fidelity data available for Plans 3-02 through 3-05
- GP model identified as strongest CD predictor; PCE weakest but passes soft gate
- per_sample_errors.csv ready for k0_2-stratified analysis in Plan 3-02

---
*Phase: 3-comparative-benchmark*
*Completed: 2026-03-17*
