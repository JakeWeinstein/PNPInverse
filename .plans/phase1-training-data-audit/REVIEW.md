# Plan Review: Phase 1 -- Training Data Audit

**Reviewer:** plan-checker agent
**Date:** 2026-03-16
**Verdict:** APPROVE with minor notes

---

## Alignment with User Request

**PASS.** The user asked for a plan covering Phase 1 only (Training Data Audit) of the surrogate pipeline roadmap. The plan stays tightly scoped to auditing the existing 3,194-sample training set for coverage gaps, biases, and insufficient density -- particularly in k0_2. It does not bleed into Phase 2 (data augmentation execution) or Phase 3 (model architecture). The go/no-go decision at the end is the correct deliverable for an audit phase.

## Completeness

**PASS.** The plan covers all essential audit dimensions:

1. Basic statistics (Step 1)
2. Marginal distributions (Step 2)
3. Pairwise joint distributions (Step 3)
4. Quantitative coverage heatmaps focused on k0_2 (Step 4)
5. Error-density correlation (Step 5) -- the causal link between gaps and surrogate quality
6. Sensitivity analysis to identify where coverage matters most (Step 6)
7. Convergence failure distribution (Step 7)
8. Max-empty-ball gap detection (Step 8)
9. Synthesis report with go/no-go (Step 9)
10. End-to-end execution (Step 10)

Prerequisites are listed and verified against the actual codebase. Dependencies between steps are explicit and correct.

## Feasibility

**PASS.** All verified against the actual codebase:

- `data/surrogate_models/training_data_merged.npz` exists with confirmed structure: keys `parameters` (3194, 4), `current_density` (3194, 22), `peroxide_current` (3194, 22), `phi_applied` (22,), `n_v9` (scalar), `n_new` (scalar).
- `StudyResults/surrogate_fidelity/per_sample_errors.csv` exists with 479 data rows (480 lines including header), columns include `k0_1`, `k0_2`, `alpha_1`, `alpha_2`, and per-model NRMSE values. Confirmed.
- `Surrogate/sampling.py` contains `ParameterBounds`, `generate_lhs_samples()`, and `generate_multi_region_lhs_samples()` exactly as described.
- `Surrogate/ensemble.py` contains `load_nn_ensemble()` with the API described.
- Libraries needed (numpy, matplotlib, scipy) are standard and available in venv-firedrake.
- No PDE solves required -- the audit uses only existing data and existing surrogates.
- The `scripts/studies/` directory exists and is the appropriate location for the audit script.

## Research Alignment

**PASS.** The plan correctly:

- Focuses on k0_2 as the most noise-sensitive parameter (287% error at 1% noise from research context).
- Operates in log10 space for k0 parameters (matching the LHS sampling strategy in `sampling.py`).
- Investigates whether the 10.7% irreducible bias is data-driven or architectural -- with a clear decision framework for each outcome.
- Accounts for the 8.4% convergence failure rate and its potential impact on coverage.
- References the correct parameter ranges: k0_1 [1e-6, 1.0], k0_2 [1e-7, 0.1], alpha [0.1, 0.9] -- confirmed against `ParameterBounds` defaults in `sampling.py`.

## Specificity

**PASS.** Steps are concrete and executable:

- Bin counts, grid sizes, and thresholds are all specified numerically (20 bins for marginals, 12x12 for heatmaps, 5 samples/bin minimum, 0.15 max-empty-ball threshold, 50k candidate points).
- Output file paths are specified.
- Success criteria are measurable (Spearman rho < -0.2 with p < 0.05, max-empty-ball radius thresholds, samples-per-log-decade targets).
- The go/no-go decision criteria are quantitative and unambiguous.

## Risk Awareness

**PASS.** Key risks are identified with reasonable mitigations. The most important risk -- "coverage looks fine but bias persists" -- is correctly framed as a finding that redirects effort rather than a failure of the audit.

---

## Issues

### MINOR: k0_2 log-decade count is 4, not 6

The plan states k0_2 range [-7, -1] spans 6 log-decades (Step 4: 12 bins = "2 bins per log-decade"), but log10(0.1) = -1, so the range is [-7, -1] which is indeed 6 log-decades. However, the research context says k0_2 spans "4 orders of magnitude" -- this is incorrect in the research context (it actually spans 6). The plan is correct. No action needed, but worth noting for the executor: the plan's 12-bin / 6-decade gridding is correct.

### MINOR: Step 5 density ball radius units need care

The plan proposes a ball radius of r=0.5 "in log10 units for k0, linear-scaled to match for alpha." The alpha parameters span [0.1, 0.9] = 0.8 units, while log10(k0_2) spans 6 units. A radius of 0.5 in log10 space corresponds to 0.5/6 = 0.083 of the k0_2 range, but for alpha it would be 0.5/0.8 = 0.625 of the alpha range -- which is enormous. The executor should normalize all dimensions to [0, 1] before computing distances (as Step 8 does for max-empty-ball), or use a scaled radius. Otherwise the density metric will be dominated by alpha dimensions.

**Recommendation:** In Step 5, normalize all 4 parameters to [0, 1] (log10 for k0, linear for alpha) before computing the ball-based density, matching the normalization in Step 8. This is a straightforward fix during implementation.

### MINOR: Peroxide current NRMSE anomalies in test data

Inspecting `per_sample_errors.csv`, some samples have extremely large peroxide NRMSE values (e.g., sample 612 has `nn_ensemble_pc_nrmse = 151.2`, sample 1836 has `nn_ensemble_pc_nrmse = 1.38`). The Step 5 analysis should either use robust statistics (median instead of mean) or handle these outliers explicitly, as a few extreme values could dominate the correlation. The plan does use Spearman (rank-based), which is the right choice, but the top-5%-worst analysis should also report whether these extreme samples are outliers or part of a pattern.

### MINOR: No explicit check for v9/v11 overlap

The plan discusses v9/v11 overlap conceptually in the Approach section but does not have a dedicated analysis step. Steps 2-3 show batch coloring, which partially addresses this. The executor could add a quick metric: what fraction of v11 samples fall within the convex hull of v9 samples (in normalized space)? This would quantify overlap directly. Not critical -- visual inspection from Steps 2-3 may suffice.

### MINOR: Output directory creation

The plan assumes `StudyResults/training_data_audit/` exists. Step 1 should include `os.makedirs("StudyResults/training_data_audit", exist_ok=True)`.

---

## Summary

This is a well-structured, thorough plan that correctly targets the key question: is the surrogate's 10.7% bias explained by training data coverage gaps, particularly in k0_2? The steps are concrete, the success criteria are measurable, the risks are identified, and the plan stays within scope. The minor issues above are implementation details that the executor can handle without plan revision.

**Recommendation: Proceed with execution.** The only item the executor should watch for is normalizing dimensions properly in Step 5's density computation (see MINOR issue #2 above).
