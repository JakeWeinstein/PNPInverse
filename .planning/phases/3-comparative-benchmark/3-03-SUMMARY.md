---
phase: 3-comparative-benchmark
plan: 03
subsystem: analysis
tags: [surrogate, k0_2, stratified-error, heatmap, gp-uq, pce-sobol]

requires:
  - phase: 3-comparative-benchmark (plan 01)
    provides: per_sample_errors.csv with NRMSE for 4 models
provides:
  - k0_2-stratified error metrics (JSON, CSV) across 6 log-decade bins for 4 models
  - Error-vs-k0_2 scatter plot with binned summary
  - Model-x-bin worst-case heatmap
  - GP uncertainty correlation with actual error (Spearman rho)
  - PCE Sobol cross-reference for k0_2 sensitivity context
affects: [3-comparative-benchmark plan-04, 3-comparative-benchmark plan-05, phase-4 ISMO augmentation]

tech-stack:
  added: []
  patterns: [log-decade binning for k0_2, GP UQ correlation analysis]

key-files:
  created:
    - scripts/studies/k02_stratified_analysis.py
    - StudyResults/surrogate_fidelity/k02_stratified_errors.json
    - StudyResults/surrogate_fidelity/k02_error_vs_value.png
    - StudyResults/surrogate_fidelity/k02_error_heatmap.png
    - StudyResults/surrogate_fidelity/k02_bin_table.csv
  modified: []

key-decisions:
  - "GP UQ correlation uses nn_ensemble NRMSE as reference error (GP model is separate, not in per_sample_errors.csv)"
  - "Heatmap shows max NRMSE (worst-case) per model-bin combination per plan specification"
  - "524 test samples used (not 479 as plan estimated), all bins populated"

patterns-established:
  - "k0_2 log-decade binning: 6 bins [1e-7..1e-1) matching training_data_audit structure"
  - "GP UQ validation: Spearman rank correlation between predicted std and actual NRMSE"

requirements-completed: [BENCH-03]

duration: 3min
completed: 2026-03-17
---

# Phase 3 Plan 03: k0_2 Stratified Error Analysis Summary

**k0_2-stratified error analysis across 4 surrogates with GP UQ correlation (CD rho=0.69) and PCE Sobol cross-reference showing worst PC errors in [1e-5,1e-4) and [1e-2,1e-1) bins**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-17T23:02:38Z
- **Completed:** 2026-03-17T23:06:00Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Per-bin NRMSE metrics (mean, median, p95, max) computed for 4 models x 2 outputs x 6 k0_2 bins
- GP uncertainty correlation successfully computed: CD rho=0.690 (good), PC rho=0.478 (moderate)
- PCE Sobol cross-reference confirms k0_2 has only 3.2% CD variance but 16.8% PC variance
- Worst-case PC errors concentrated in [1e-5,1e-4) bin (max NRMSE 2.15 for rbf_baseline) and [1e-2,1e-1) bin (p95 > 0.5 for all models)
- nn_ensemble has lowest worst-case PC errors overall

## Task Commits

Each task was committed atomically:

1. **Task 1: Build k0_2 stratified analysis script** - `9490253` (feat)
2. **Task 2: Run analysis and validate outputs** - `9c38fbe` (feat)

## Files Created/Modified
- `scripts/studies/k02_stratified_analysis.py` - Complete 567-line analysis script with argparse CLI
- `StudyResults/surrogate_fidelity/k02_stratified_errors.json` - Full per-bin metrics + GP UQ + PCE Sobol
- `StudyResults/surrogate_fidelity/k02_error_vs_value.png` - Scatter + binned median plot (488 KB)
- `StudyResults/surrogate_fidelity/k02_error_heatmap.png` - Model x bin worst-case heatmap (107 KB)
- `StudyResults/surrogate_fidelity/k02_bin_table.csv` - Flat table with 48 data rows

## Decisions Made
- GP UQ correlation uses nn_ensemble NRMSE as the reference error metric since the GP model itself is not included in per_sample_errors.csv
- Used max NRMSE for heatmap cells (worst-case focus) as specified in the plan
- 524 test samples available (slightly more than the 479 estimated in the plan)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Key Findings

| Metric | Value |
|--------|-------|
| GP UQ CD Spearman rho | 0.690 (strong positive correlation) |
| GP UQ PC Spearman rho | 0.478 (moderate correlation) |
| PCE Sobol k0_2 CD mean | 0.032 (3.2% of variance) |
| PCE Sobol k0_2 PC mean | 0.168 (16.8% of variance) |
| Worst PC bin (all models) | [1e-5,1e-4) with max NRMSE up to 2.15 |
| Most problematic tail | [1e-2,1e-1) with p95 > 0.5 for all models |
| Best overall model | nn_ensemble (lowest max NRMSE in most bins) |

## Next Phase Readiness
- Stratified error data ready for comparative benchmark report (plan 04/05)
- k0_2 bin analysis identifies [1e-5,1e-4) and [1e-2,1e-1) as priority regions for ISMO augmentation in Phase 4
- GP UQ correlation data available for uncertainty-aware inference decisions

---
*Phase: 3-comparative-benchmark*
*Completed: 2026-03-17*
