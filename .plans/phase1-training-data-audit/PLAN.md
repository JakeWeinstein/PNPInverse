# Plan: Phase 1 -- Training Data Audit

**Goal:** Determine whether the 3,194-sample training set has coverage gaps, biases, or insufficient density -- especially in the k0_2 dimension -- that could explain surrogate bias and k0_2 sensitivity, and produce a go/no-go decision on data augmentation.

**Date:** 2026-03-16

**Research used:**
- `.research/pde-constrained-inverse-surrogates/SUMMARY.md` (k0_2 catastrophic at 1% noise, 10.7% irreducible bias)
- `Surrogate/sampling.py` (ParameterBounds, LHS with log-space k0, multi-region sampling)
- `data/surrogate_models/training_data_merged.npz` (3,194 samples: 445 v9 + 2,749 v11)
- `StudyResults/surrogate_fidelity/per_sample_errors.csv` (per-sample NRMSE with parameter values)
- `StudyResults/surrogate_fidelity/fidelity_summary.json` (aggregate fidelity metrics)

## Approach

The audit proceeds in four stages: (1) load and characterize the raw training data, (2) visualize marginal and pairwise distributions, (3) compute quantitative coverage metrics in log-space, and (4) cross-reference coverage gaps with surrogate error to determine whether sparse regions explain the bias.

For k0_1 and k0_2, all analysis operates in log10 space because these parameters span 6 and 4 orders of magnitude respectively (k0_1: [1e-6, 1.0], k0_2: [1e-7, 0.1]). Alpha parameters are analyzed in linear space since their range [0.1, 0.9] is narrow and linear.

The key insight driving this plan: if the 8.4% convergence failure rate in v11 is concentrated in specific parameter regions, those regions will appear as coverage gaps. Similarly, if the two batches (v9: seed=42, v11: multi-region) overlap heavily in some regions and leave others sparse, that will show up in the density analysis. We also leverage the existing per-sample error data to directly correlate sample density with surrogate accuracy.

The coverage threshold is set at 5 samples per log-decade bin (for a 4x4 grid of k0_2 vs each other parameter). Below this threshold, the surrogate is essentially extrapolating. The sensitivity analysis uses the existing surrogates (no new PDE solves needed) to identify which k0_2 sub-ranges have weakest I-V signal and thus need the densest coverage.

## Prerequisites

- Python environment: `venv-firedrake` (in parent directory)
- Files exist:
  - `data/surrogate_models/training_data_merged.npz` (confirmed: 3,194 x 4 parameters + 22-point I-V curves)
  - `StudyResults/surrogate_fidelity/per_sample_errors.csv` (confirmed: 479 test samples with NRMSE per model)
- Libraries needed: numpy, matplotlib, scipy (all available in venv-firedrake)

## Steps

### Step 1: Create the audit script scaffold

**Action:** Create `scripts/studies/training_data_audit.py` with a main function that loads `training_data_merged.npz`, extracts the parameter array (shape 3194x4), splits into v9 (first 445) and v11 (remaining 2749) batches using the stored `n_v9` field, and prints basic summary statistics.

**Expected output:** A script that, when run, prints:
- Total samples, v9 count, v11 count
- Per-parameter min, max, mean, std in both linear and log10 space (for k0 params)
- Confirmation that parameters fall within declared bounds

**Success criterion:** Script runs without error and printed stats match: 3194 total, 445 v9, 2749 v11, k0_1 in [~1e-6, ~1.0], k0_2 in [~1e-7, ~0.1], alphas in [~0.1, ~0.9].

**Depends on:** None

### Step 2: Generate 1D marginal histograms

**Action:** In the audit script, add a function `plot_marginals()` that creates a 2x2 figure with one histogram per parameter. For k0_1 and k0_2, plot in log10 space. For alpha_1 and alpha_2, plot in linear space. Each histogram should:
- Use 20 bins spanning the full declared range
- Overlay v9 (blue, alpha=0.5) and v11 (orange, alpha=0.5) as stacked bars so batch composition is visible
- Draw a horizontal line at the "ideal uniform" count (total_samples / 20 bins = ~160)
- Annotate any bin with fewer than 10 samples in red

**Expected output:** Saved figure at `StudyResults/training_data_audit/marginal_histograms.png`

**Success criterion:** All four histograms render. Visual check: log-space k0 histograms should look approximately uniform (since LHS was log-space). Any non-uniformity or empty bins are flagged.

**Depends on:** Step 1

### Step 3: Generate 2D pairwise scatter plots with density coloring

**Action:** Add a function `plot_pairwise()` that creates a 4x4 grid (or 6 unique pairs) of 2D scatter plots. Each dot is colored by local density (using `scipy.stats.gaussian_kde` in the appropriate space -- log10 for k0, linear for alpha). The critical plots are k0_2 vs each of the other 3 parameters. Use different markers for v9 (circle) and v11 (triangle) samples.

For each of the 6 pairwise plots, also compute and overlay a 10x10 bin grid showing sample count per bin. Bins with 0 samples get a red X marker. Bins with 1-4 samples get a yellow border.

**Expected output:** Saved figure at `StudyResults/training_data_audit/pairwise_scatter.png`

**Success criterion:** All 6 pairwise combinations rendered. k0_2 vs k0_1 plot should show whether the two log-space LHS designs tile well together or leave systematic gaps.

**Depends on:** Step 1

### Step 4: Compute log-space coverage heatmaps for k0_2

**Action:** Add a function `compute_coverage_heatmaps()` that bins the 4D parameter space into a coarse grid and counts samples per cell. Specifically:

1. **k0_2 marginal density:** Divide log10(k0_2) range [-7, -1] into 12 equal bins (2 bins per log-decade). Count samples per bin. Compute samples-per-log-decade.
2. **k0_2 x k0_1 heatmap:** 12 bins for log10(k0_2) x 12 bins for log10(k0_1). Plot as heatmap with counts annotated.
3. **k0_2 x alpha_1 heatmap:** 12 bins for log10(k0_2) x 8 bins for alpha_1.
4. **k0_2 x alpha_2 heatmap:** 12 bins for log10(k0_2) x 8 bins for alpha_2.

Report: number of empty bins, number of bins with < 5 samples, minimum and maximum bin counts.

**Expected output:** Saved figures at `StudyResults/training_data_audit/coverage_k02_marginal.png`, `coverage_k02_k01.png`, `coverage_k02_alpha1.png`, `coverage_k02_alpha2.png`. A JSON summary at `StudyResults/training_data_audit/coverage_metrics.json` containing bin counts and gap locations.

**Success criterion:** For 3,194 samples in a 12x12 grid (144 cells), expected uniform density is ~22 samples/cell. Coverage is "adequate" if no bin has fewer than 5 samples and the min/max ratio is above 0.2. Record exact numbers.

**Depends on:** Step 1

### Step 5: Correlate surrogate error with sample density

**Action:** Add a function `error_vs_density()` that:

1. Loads `StudyResults/surrogate_fidelity/per_sample_errors.csv` (479 test samples with per-model NRMSE).
2. For each test sample, computes the local training sample density: count training samples within a ball of radius r in log-normalized space. Use r = 0.5 (in log10 units for k0, linear-scaled to match for alpha). Also try r = 0.25 and r = 1.0 for robustness.
3. Creates scatter plots: local density vs NRMSE for the nn_ensemble model (best performer), separately for cd and pc observables.
4. Computes Spearman rank correlation between density and error.
5. Specifically flags: are the worst-error test samples (top 5% NRMSE) in low-density regions?

**Expected output:** Figure at `StudyResults/training_data_audit/error_vs_density.png`. Correlation values and top-5%-worst-sample analysis printed and saved to `coverage_metrics.json`.

**Success criterion:** A statistically significant negative correlation (Spearman rho < -0.2, p < 0.05) between density and error would confirm that coverage gaps cause surrogate bias. If no correlation, the bias is structural (model architecture) not data-driven.

**Depends on:** Steps 1, 4

### Step 6: Quick sensitivity analysis via existing surrogates

**Action:** Add a function `k02_sensitivity_scan()` that uses the existing nn_ensemble surrogate to map I-V signal strength as a function of k0_2. Specifically:

1. Fix k0_1, alpha_1, alpha_2 at their midpoints (geometric mean for k0_1: ~1e-3, arithmetic mean for alphas: 0.5).
2. Sweep k0_2 across 50 log-spaced points from 1e-7 to 0.1.
3. For each k0_2 value, evaluate the surrogate to get the I-V curve (22 points for cd and pc).
4. Compute the "signal range" for pc (peroxide current): max(pc) - min(pc) across voltage. This measures how much the peroxide observable responds to k0_2 at that value.
5. Plot signal range vs log10(k0_2). Identify the k0_2 sub-range where signal is weakest (flattest I-V for peroxide) -- this is where the surrogate struggles to distinguish k0_2 values and needs the densest training data.

Also do a finite-difference sensitivity: for each of the 50 k0_2 points, perturb k0_2 by +/- 1% and measure the I-V change. This gives the local Jacobian |dI/d(log k0_2)|.

**Expected output:** Figure at `StudyResults/training_data_audit/k02_signal_strength.png`. Identified "weak signal" k0_2 sub-range saved to `coverage_metrics.json`.

**Success criterion:** Successfully identifies which log-decade(s) of k0_2 have the weakest peroxide signal. Cross-reference with Step 4 coverage: if the weak-signal region is also under-sampled, that is the critical gap.

**Depends on:** Step 1 (also needs the nn_ensemble model loaded; check `Surrogate/ensemble.py` for loading API)

### Step 7: Analyze convergence failure distribution

**Action:** Add a function `convergence_failure_analysis()` that investigates where the ~251 failed samples from v11 would have been. Since the merged data only contains converged samples, we reconstruct the intended v11 sample design:

1. Regenerate the v11 LHS design using the same seed and parameters (from `Surrogate/sampling.py`'s `generate_multi_region_lhs_samples()` or `generate_lhs_samples()` with the v11 seed). We need to determine the exact call that was used -- check scripts in `scripts/surrogate/` for the v11 generation script.
2. If the original design is recoverable, subtract converged samples to identify failed parameter locations.
3. Plot failed samples in the k0_2 vs k0_1 scatter to see if failures cluster in specific regions.

If the original seed/call cannot be recovered, skip this step and note it as a limitation.

**Expected output:** Figure at `StudyResults/training_data_audit/convergence_failures.png` (if recoverable). Analysis notes in the report.

**Success criterion:** Either identifies clustering of failures or documents that the analysis is infeasible.

**Depends on:** Step 1

### Step 8: Compute max-empty-ball radius

**Action:** Add a function `max_empty_ball()` that computes the largest empty ball in the normalized parameter space:

1. Normalize all 4 parameters to [0, 1]: k0_1 and k0_2 in log10 space mapped to [0, 1], alphas linearly mapped to [0, 1].
2. Build a KD-tree from the 3,194 normalized training points.
3. Generate 50,000 random candidate points uniformly in [0, 1]^4.
4. For each candidate, find the distance to its nearest training point.
5. The maximum of these distances is the max-empty-ball radius (an upper bound).
6. Report the location (in physical parameter space) and radius of the largest gap.

**Expected output:** Max-empty-ball radius and location printed and saved to `coverage_metrics.json`.

**Success criterion:** For 3,194 points in [0,1]^4, a well-distributed LHS should have max-empty-ball radius around 0.05-0.08. If the radius exceeds 0.15, there is a significant coverage gap.

**Depends on:** Step 1

### Step 9: Generate diagnostic report

**Action:** Add a function `generate_report()` that compiles all findings into a structured text report at `StudyResults/training_data_audit/REPORT.md`:

1. **Summary statistics:** Total samples, batch breakdown, parameter ranges.
2. **Marginal distributions:** Reference to histograms. Flag any non-uniform bins.
3. **Coverage analysis:** Reference to heatmaps. List empty and under-sampled bins with coordinates.
4. **Error-density correlation:** Spearman rho and interpretation.
5. **Sensitivity analysis:** Weak-signal k0_2 sub-range. Cross-reference with coverage.
6. **Max-empty-ball:** Location and radius of largest gap.
7. **Convergence failures:** Distribution (if recoverable).
8. **Go/No-Go Decision:** Based on:
   - If max-empty-ball radius > 0.15 OR any k0_2 log-decade has < 100 samples OR the weak-signal k0_2 region has < 200 samples: **augmentation needed**.
   - Otherwise: **proceed with existing data**.
9. **If augmentation needed:** Specify a sampling plan:
   - Number of new samples (target 1,000-3,000, constrained by overnight budget)
   - Sampling strategy: use `generate_multi_region_lhs_samples()` with focused bounds on the identified gap regions
   - Exact focused bounds for each gap region
   - Estimated runtime based on ~3s/sample

**Expected output:** `StudyResults/training_data_audit/REPORT.md` with all sections filled.

**Success criterion:** Report contains quantitative data for every section, a clear go/no-go decision, and (if needed) an actionable augmentation plan with specific parameter bounds and sample counts.

**Depends on:** Steps 2-8

### Step 10: Run the audit script end-to-end

**Action:** Activate `venv-firedrake` and run the complete audit script:
```bash
source ../venv-firedrake/bin/activate
python scripts/studies/training_data_audit.py
```

Review all outputs in `StudyResults/training_data_audit/`. Verify figures are readable and the report is complete.

**Expected output:** All figures saved, `coverage_metrics.json` populated, `REPORT.md` generated with go/no-go decision.

**Success criterion:** Script completes without error. All 7+ figures generated. Report contains quantitative assessment of sample density per log-decade of k0_2, identified gaps with coordinates, and a clear go/no-go decision.

**Depends on:** Steps 1-9

## Success Criteria

1. **Quantitative density assessment:** Sample count per log-decade of k0_2 is computed and reported (target: at least 400 samples per log-decade for the 4 decades from 1e-7 to 0.1).
2. **Gap identification:** Any empty or under-sampled bins in the 2D k0_2 heatmaps are listed with their physical-space coordinates.
3. **Error-density link:** Spearman correlation between local density and surrogate NRMSE is computed with p-value.
4. **Sensitivity-aware coverage:** The k0_2 sub-range where peroxide signal is weakest is identified, and its sample density is reported.
5. **Go/no-go decision:** A clear, justified recommendation on whether to augment training data or proceed to Phase 2.
6. **Augmentation plan (if needed):** Specific focused bounds, sample count, and estimated runtime.

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| nn_ensemble model fails to load for sensitivity scan | Low | Medium | Fall back to RBF baseline model which is a simple pickle load |
| v11 LHS seed not recoverable for failure analysis | Medium | Low | Skip Step 7, note as limitation -- the coverage analysis from Steps 2-4 still identifies gaps regardless |
| KD-tree max-empty-ball underestimates true gap (50k candidates insufficient) | Low | Low | 50k candidates in 4D is adequate; increase to 200k if result seems unreasonably small |
| Coverage looks fine but surrogate still has 10.7% bias | Medium | Medium | This would mean bias is architectural, not data-driven -- important finding that redirects effort to Phase 3 (model architecture) rather than Phase 2 (augmentation) |
| venv-firedrake missing matplotlib | Low | Low | Install via pip if needed |

## Open Questions

1. **v11 generation seed:** What seed and exact function call was used to generate the v11 batch? This is needed for Step 7 (convergence failure analysis). Check `scripts/surrogate/` for the generation script. If not recoverable, Step 7 is skipped.
2. **Focused region bounds for v11:** The v11 batch used `generate_multi_region_lhs_samples()` -- what were the focused bounds? This affects interpretation of the density analysis (non-uniform density may be intentional, not a gap).
3. **User preference on augmentation threshold:** The plan uses 5 samples/bin as the minimum and 0.15 max-empty-ball as the gap threshold. Should these be more or less aggressive?
