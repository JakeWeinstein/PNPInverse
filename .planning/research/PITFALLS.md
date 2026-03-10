# Pitfalls Research

**Domain:** Surrogate-assisted inverse parameter recovery pipeline redesign (PNP-BV electrochemical kinetics)
**Researched:** 2026-03-09
**Confidence:** HIGH (codebase-derived evidence + domain literature)

## Critical Pitfalls

### Pitfall 1: Overfitting to a Single Noise Seed

**What goes wrong:**
The pipeline is tuned (stage count, voltage grids, optimizer tolerances, observable weights) to produce excellent results on one noise seed, but fails catastrophically on others. The v13 parameter recovery results already show this: at 2% noise, seed 42 gives 27% max error, seed 43 gives 15%, but seed 44 gives 400% error. The pipeline appears to "work" on 2 of 3 seeds but is fundamentally non-robust. Any pipeline redesign that evaluates against a single seed will replicate this failure mode.

**Why it happens:**
Each noise realization shifts the objective landscape differently. At 2% noise, the perturbation can move the apparent optimum far from truth, especially for weakly identifiable parameters (k0_2, alpha_2). A pipeline tuned on one seed may exploit a lucky noise pattern that happens to align the surrogate bias with the noise realization, producing a cancellation that does not generalize. The existing `target_seed=20260226` is hardcoded in every inference script -- the entire v7-v13 evolution was tuned against this one seed.

**How to avoid:**
- Evaluate every pipeline variant against a minimum of 10 noise seeds at 2% noise.
- Report the **worst-case** max relative error across seeds (not median or mean).
- A pipeline variant is only "better" if its worst-case error across seeds is lower.
- Never tune hyperparameters (weights, tolerances, voltage grids) against a single seed.

**Warning signs:**
- Mean error looks good but std is large (the existing 2% noise results show mean=1.47 but std=1.79 -- the std exceeds the mean, indicating extreme seed sensitivity).
- Prototype "works perfectly" on the default seed but hasn't been tested on others.
- When you add a pipeline improvement, check: does it help ALL seeds or just the default?

**Phase to address:**
Baseline phase (Phase 1). The very first task must be running v13 on 10+ seeds to establish the ground truth about which seeds fail and why.

---

### Pitfall 2: Surrogate Bias Masquerading as Noise Sensitivity

**What goes wrong:**
The pipeline's ~11% irreducible surrogate bias (documented in `parameter_recovery_summary.json`: surrogate optimum differs from PDE truth by 10.67% even at 0% noise) is conflated with noise sensitivity. Efforts are spent on regularization and noise-robust optimization when the dominant error source is surrogate approximation error. The PDE refinement stages (P1, P2) exist specifically to correct this bias, but the correction is incomplete -- P2 only reduces errors to ~2-5% from the surrogate's ~3%.

**Why it happens:**
The surrogate I-V curves differ systematically from PDE-computed curves. The peroxide current observable has particularly poor surrogate fidelity: the NN ensemble's pc_max_nrmse is 151% (from fidelity_summary.json), meaning some parameter regions have 150x prediction errors. Optimizing a joint objective with secondary_weight=1.0 on the peroxide channel feeds this massive surrogate error directly into the parameter estimates for k0_2 and alpha_2. The PDE refinement stages cannot fully recover because they start from a surrogate-biased initial point, and L-BFGS-B with limited iterations may not escape the biased basin.

**How to avoid:**
- Quantify surrogate bias *per parameter* at the surrogate-optimal point before designing PDE refinement.
- Consider asymmetric observable weighting: downweight the peroxide current in surrogate stages (where its error is 150x worst-case) and upweight it in PDE stages (where it is computed exactly).
- Evaluate whether P1 (shallow PDE) actually moves parameters meaningfully from surrogate optimum, or is redundant.
- Consider a "bias correction" approach: characterize systematic surrogate error as a function of parameters and apply a correction before PDE refinement.

**Warning signs:**
- Zero-noise surrogate-only recovery gives >5% error on any parameter.
- Adding noise *improves* some parameter estimates (suggesting lucky noise-bias cancellation).
- PDE refinement stages barely move parameters from surrogate optimum (loss drops but parameters do not change much).

**Phase to address:**
Audit phase (Phase 2). Quantify the per-parameter bias contribution of each stage. This determines whether effort should go into better surrogates, better PDE refinement, or better observable weighting.

---

### Pitfall 3: Comparing Pipeline Variants Without Controlling for Confounds

**What goes wrong:**
When comparing pipeline variant A (e.g., cascade + PDE) vs. variant B (e.g., multistart + PDE), the comparison is contaminated by uncontrolled variables: different noise seeds, different initial guesses, different optimizer convergence states, different random seeds in LHS sampling. The result is "variant A beats B" on one run, "B beats A" on another, and no scientifically defensible conclusion.

**Why it happens:**
The pipeline has many moving parts (surrogate model, initial guess, voltage grid, optimizer, PDE solver tolerances, noise seed). A "comparison" that changes two things simultaneously (e.g., switching from cascade to multistart AND changing the voltage grid) cannot attribute the improvement to either change. The v8-v13 evolution in the codebase appears to have added features incrementally without controlled ablation -- each version adds something new but does not isolate the contribution.

**How to avoid:**
- **Controlled ablation**: Change exactly one component at a time. Measure the effect across all noise seeds.
- **Fixed random seeds**: Use the same LHS seeds for multistart across variants. Use the same noise seeds for all comparisons.
- **Paired comparison**: For each noise seed, run both variants and compare paired results. Report paired differences, not marginal means.
- **Statistical test**: With 10+ seeds, use a paired Wilcoxon signed-rank test to determine if variant A is significantly better than B (not just better on average).
- **Ablation table**: Each row = one component (e.g., "S1 alpha-only init"), columns = "included" vs "excluded", metric = worst-case error across seeds.

**Warning signs:**
- Improvement results are reported as "average across N runs" without confidence intervals.
- The winning variant changes depending on which seeds are included.
- Two pipeline changes were made simultaneously and attributed to one.

**Phase to address:**
Comparison phase (Phase 3). Design the comparison protocol *before* running experiments. Pre-register which variants will be compared and which metric determines the winner.

---

### Pitfall 4: Redundant Surrogate Stages Wasting Budget Without Improving Robustness

**What goes wrong:**
The v13 pipeline has 5 surrogate stages (S1-S5) that take ~15 seconds total. The v13 CSV shows that S2 (joint L-BFGS-B) already achieves 2.89%/4.78%/1.85%/2.86% errors on k0_1/k0_2/alpha_1/alpha_2. S3 (cascade) and S4 (multistart) barely improve: S3 achieves 2.80%/4.63%/1.80%/2.79%, and S4 achieves 2.84%/4.70%/1.82%/2.82%. These three stages converge to essentially the same surrogate minimum. Meanwhile, the PDE stages (P1, P2) do the actual heavy lifting of correcting surrogate bias, but get limited iteration budget.

**Why it happens:**
The surrogate objective has a single basin for these parameter values (confirmed by the 20K multistart CV<0.10 result in the V&V report). Once any gradient-based optimizer finds the basin, additional gradient-based optimizations from nearby initial points converge to the same minimum. The redundant stages provide false confidence ("three methods agree!") rather than actual improvement.

**How to avoid:**
- Run the ablation: S2-only vs. S2+S3 vs. S2+S3+S4 as PDE refinement starting points. Measure across 10+ seeds whether the extra surrogate stages actually change PDE refinement outcomes.
- If they do not: remove S3 and S4, saving complexity and 10 seconds per run.
- If they do: understand *why* (perhaps different seeds lead to different surrogate optima, and having multiple starting points helps PDE refinement).
- The only scenario where S4 (multistart) adds value is if the surrogate landscape has multiple basins for some noise realizations. Test this explicitly.

**Warning signs:**
- All surrogate stages produce parameter estimates within 0.1% of each other.
- S5 (best selection) always picks S2, making S3/S4 dead code.
- Total surrogate stage time exceeds PDE solve time (inverted priority).

**Phase to address:**
Audit phase (Phase 2). Include as a specific ablation experiment.

---

### Pitfall 5: Parameter Identifiability Collapse Under Noise

**What goes wrong:**
At 2% noise, the Fisher Information Matrix for (k0_1, k0_2, alpha_1, alpha_2) can become near-singular, meaning some parameter combinations become practically unidentifiable. The I-V curve's sensitivity to k0_2 is much weaker than to k0_1 (k0_2 governs H2O2 decomposition, a secondary effect on total current). With noise, the k0_2 signal is drowned out, and the optimizer wanders in a flat cost landscape for k0_2 while appearing to converge for the other parameters.

**Why it happens:**
The Butler-Volmer rate expressions for the two reactions contribute additively to the total current. Reaction 1 (O2 reduction) dominates the current density, so k0_1 and alpha_1 are strongly identifiable from the primary observable. Reaction 2 (H2O2 decomposition) contributes mainly through the peroxide current difference (R0 - R1). At 2% noise on the primary observable, the k0_2 signal in total current may be below the noise floor. The peroxide observable was added precisely to address this (as noted in `Infer_BVMultiObs_charged.py`), but the peroxide surrogate has 150x max NRMSE, undermining this strategy.

**How to avoid:**
- Compute the Fisher Information Matrix at the true parameter values using PDE derivatives (not surrogate). Check the condition number and per-parameter sensitivity.
- If k0_2 is weakly identifiable: consider adding prior/regularization specifically for k0_2, or using a constrained search space.
- Profile likelihood analysis: fix k0_2 at a grid of values, optimize the remaining 3 parameters at each, and plot the profile. If the profile is flat, k0_2 is not identifiable from these data.
- Consider whether the voltage grid can be designed to maximize k0_2 sensitivity (D-optimal experimental design).

**Warning signs:**
- k0_2 recovery error is consistently the largest across all seeds.
- Different noise seeds produce wildly different k0_2 estimates while k0_1 stays stable.
- The optimizer reports convergence but k0_2 has not moved from its initial guess.

**Phase to address:**
Baseline phase (Phase 1) for diagnosis, Redesign phase (Phase 4) for mitigation.

---

### Pitfall 6: PDE Refinement Starting from Wrong Basin

**What goes wrong:**
The PDE refinement stages (P1, P2) start from the surrogate optimum and run L-BFGS-B with limited iterations (40 for P1, 25 for P3 in v7). If the surrogate optimum is in a different basin than the PDE optimum (which is plausible given the ~11% surrogate bias), the PDE refinement may converge to a local PDE minimum near the surrogate optimum rather than the global PDE optimum. With limited iteration budget and no multistart on the PDE side, there is no escape mechanism.

**Why it happens:**
The surrogate approximation systematically shifts the objective landscape. The surrogate optimum may be in a valley on the PDE landscape that is not the deepest valley. L-BFGS-B is a local optimizer and cannot jump between basins. The v13 pipeline addresses this by trying multiple surrogate strategies (S2-S4) and selecting the best as PDE starting point, but if all surrogate strategies converge to the same biased minimum (which they do -- see Pitfall 4), the PDE refinement starts from the same wrong place regardless.

**How to avoid:**
- Run PDE refinement from multiple starting points: surrogate optimum, perturbed versions of the surrogate optimum, and at least one point from a coarse PDE grid search.
- Increase the PDE refinement iteration budget -- 40 iterations of L-BFGS-B may not be enough to escape the surrogate-biased basin.
- Consider a trust-region method (e.g., dogleg, Steihaug-CG) instead of L-BFGS-B for PDE refinement -- trust-region methods are more robust to poor initial points.
- Use the PDE gradient (adjoint-based or FD with the PDE solver) rather than surrogate gradient for the PDE refinement -- mixing surrogate gradients with PDE objectives causes inconsistency.

**Warning signs:**
- PDE refinement reduces the loss substantially but parameters barely change from the surrogate estimate.
- PDE refinement loss plateau is much higher than expected from PDE solver noise alone.
- Running PDE refinement from a randomly perturbed starting point gives lower PDE loss than from the surrogate optimum.

**Phase to address:**
Redesign phase (Phase 4). Test whether PDE multistart with 3-5 perturbed starting points significantly improves worst-case error.

---

### Pitfall 7: Objective Function Design Mixing Incompatible Scales

**What goes wrong:**
The joint objective `J = 0.5 * sum((cd_sim - cd_target)^2) + w * 0.5 * sum((pc_sim - pc_target)^2)` sums two terms with potentially very different magnitudes. The current density scale (`-I_SCALE`) and peroxide current scale use the same `observable_scale`, but the actual magnitudes of these curves can differ by orders of magnitude depending on the voltage range. A secondary_weight of 1.0 may effectively ignore the peroxide channel if its residuals are much smaller than the primary, or swamp the primary if the peroxide residuals happen to be larger. The optimal weight depends on the noise seed.

**Why it happens:**
The two observables (total current density and peroxide current) have different physical magnitudes and different noise sensitivities. The cascade module (`cascade.py`) tries to address this by using CD-dominant (w=0.5) and PC-dominant (w=2.0) weights in sequential passes, but these are hardcoded and not adapted to the actual data magnitudes. The weight that recovers k0_2 well may worsen k0_1 recovery, creating an inherent tradeoff.

**How to avoid:**
- Normalize each observable channel by its target norm: `J = sum((cd - cd_target)^2 / ||cd_target||^2) + w * sum((pc - pc_target)^2 / ||pc_target||^2)`.
- Or normalize by per-point noise variance if known (Mahalanobis distance / weighted least squares).
- Sweep the secondary_weight across [0.1, 0.5, 1.0, 2.0, 5.0] and measure the Pareto front of (k0_1 error, k0_2 error) across seeds to find a robust weight.
- Consider a max-norm objective instead of sum-of-squares: `J = max(max_residual_cd, w * max_residual_pc)` -- this prevents either channel from being ignored.

**Warning signs:**
- One observable's residual is 100x smaller than the other, meaning it contributes <1% to the total objective.
- Changing secondary_weight from 0.5 to 2.0 dramatically changes which parameters are well-recovered.
- The cascade's Pass 1 (CD-dominant) and Pass 2 (PC-dominant) produce very different parameter estimates.

**Phase to address:**
Comparison phase (Phase 3). Include weight normalization as one of the component variants to test.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Hardcoded noise seed (`target_seed=20260226`) | Reproducible results for debugging | False confidence in pipeline robustness; entire design tuned to one realization | During initial development only; must switch to multi-seed evaluation before any "pipeline X is better" claims |
| FD gradients with fixed step `1e-5` for all parameters | Simple implementation, no adjoint needed | Wrong step for log-space k0 vs linear alpha dimensions; gradient inaccuracy causes early L-BFGS-B termination | Acceptable for surrogate objectives (cheap evals); must verify FD convergence for PDE objectives |
| Sequential script-per-version pattern (v7, v8, ..., v13) | Easy to compare by running old scripts | 42 inference scripts with copy-pasted logic; changing the solver or objective requires editing every script | Never; refactor into a configurable pipeline with a single entry point and YAML/dict configuration |
| `sys.path.insert(0, _ROOT)` in every script | Bypasses proper packaging | Fragile imports, breaks if directory structure changes, confuses IDE tools | Tolerable for research code, but a proper `setup.py`/`pyproject.toml` would cost 30 minutes and eliminate the problem |
| Best-of selection via max-error comparison (S5 stage) | Simple logic | Does not account for how the starting point affects PDE refinement; the surrogate-best may not be the PDE-best | Only if validated that surrogate-best starting point consistently leads to PDE-best endpoint |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| Firedrake + PyTorch in same process | PETSc and PyTorch both use MKL/OpenBLAS, causing `KMP_DUPLICATE_LIB_OK` issues and SNES segfaults | Existing subprocess isolation for PDE targets is correct; never load both in the same process for batch runs |
| Surrogate predict_batch for 20K LHS grid | Assuming predict_batch returns identical results to 20K individual predict() calls | Verify batch consistency; NN batch normalization layers (if any) behave differently in batch vs. single-point mode |
| PDE solver steady-state detection | Using `relative_tolerance=1e-4` may declare convergence before the solution has truly equilibrated at extreme voltages (eta < -40) | Monitor both flux and concentration convergence; tighten tolerance for extreme voltages or increase max_steps |
| Nondimensional vs. physical parameter bounds | Specifying k0 bounds in physical units but passing to optimizer in nondimensional space (or vice versa) | The existing code correctly uses nondimensional k0_hat throughout; verify any new code maintains this convention |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Running PDE refinement from only 1 starting point | Appears fast (~300s total for P1+P2) but misses global PDE optimum | Run from 3-5 perturbed starting points in parallel; ~3x wall time but enables worker parallelism | When surrogate bias places starting point in wrong PDE basin (any difficult noise seed) |
| 20K LHS multistart with predict_batch | Fast (~3-8s) but produces 20K identical loss values if surrogate landscape is unimodal | Check whether top-20 candidates' polished results have CV < 0.01 on all parameters; if so, 1K samples suffice | Always for this problem (the surrogate landscape IS unimodal based on existing evidence) |
| Per-point FD gradient in PDE refinement | 8 PDE solves per gradient evaluation (2 per parameter x 4 parameters), each taking seconds | Use adjoint-based gradients if available; otherwise, reduce from 4-parameter joint to 2-parameter sequential optimization to halve gradient cost | At 15+ voltage points x 8 PDE solves/gradient x 40 iterations = 4800 PDE solves per PDE phase |
| Parallel worker pool creation per phase | ~10s overhead to spawn Firedrake workers per phase (documented in v7 comments) | Reuse worker pool across phases with shared mesh/control config (already done in v7 for P2/P3) | When phases use different mesh sizes or control modes |

## "Looks Done But Isn't" Checklist

- [ ] **Multi-seed robustness:** Pipeline variant tested on 10+ seeds, not just the default. Verify worst-case error, not mean.
- [ ] **Surrogate bias accounting:** Reported pipeline errors separate surrogate bias (~11%) from noise-induced error. Verify by comparing 0% noise surrogate-only recovery to noisy recovery.
- [ ] **Ablation completeness:** Each component justified by removing it and measuring degradation, not just by adding it and measuring improvement.
- [ ] **PDE refinement convergence:** PDE stages have converged (gradient norm below threshold), not just hit iteration limit. Check optimizer convergence flag.
- [ ] **Observable normalization:** Both channels contribute meaningfully to objective. Verify by checking per-channel residual magnitudes at optimum.
- [ ] **Parameter bounds not active:** Optimal parameters are in the interior of bounds, not pushed against a bound constraint. Hitting a bound means the search space is too narrow or the optimizer is stuck.
- [ ] **Noise model match:** Noise is applied per-point as multiplicative Gaussian (`y * (1 + sigma * randn)`), matching what the pipeline expects. Additive noise with wrong sigma would change identifiability.
- [ ] **Comparison fairness:** All variants use the same target data (same PDE solve, same noise realizations). Re-generating targets between comparisons introduces confounds.

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Overfitting to single noise seed (Pitfall 1) | LOW | Re-run evaluation on 10+ seeds; results are still valid, just need broader validation |
| Surrogate bias conflated with noise (Pitfall 2) | MEDIUM | Compute per-parameter bias at 0% noise; reweight observable channels; may need to retrain surrogate with more data near the optimum |
| Uncontrolled comparison (Pitfall 3) | MEDIUM | Re-run comparisons with controlled ablation; existing code is reusable, just need structured experiment design |
| Redundant surrogate stages (Pitfall 4) | LOW | Remove stages and verify PDE refinement outcomes are unchanged; simple ablation |
| Identifiability collapse for k0_2 (Pitfall 5) | HIGH | May require redesigning the observation strategy (add data at different conditions), adding regularization, or accepting that k0_2 has irreducible uncertainty at 2% noise |
| PDE refinement in wrong basin (Pitfall 6) | MEDIUM | Add PDE multistart with 3-5 starting points; increases runtime ~3x but is parallelizable |
| Objective scale mismatch (Pitfall 7) | LOW | Add normalization to objective function; a few lines of code change in objectives.py |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Overfitting to single noise seed | Phase 1 (Baseline) | v13 tested on 10+ seeds; per-seed error distribution documented |
| Surrogate bias vs. noise | Phase 2 (Audit) | Per-parameter bias quantified at 0% noise; bias contribution vs. noise contribution separated |
| Uncontrolled comparison | Phase 3 (Comparison) | Pre-registered comparison protocol with paired tests across seeds |
| Redundant surrogate stages | Phase 2 (Audit) | Ablation showing S3/S4 do/do not improve PDE refinement outcomes |
| Identifiability collapse | Phase 1 (Baseline) + Phase 4 (Redesign) | Fisher Information Matrix computed; profile likelihood for k0_2 plotted |
| PDE wrong basin | Phase 4 (Redesign) | PDE multistart from 3+ starting points tested; compare single-start vs. multi-start worst-case errors |
| Objective scale mismatch | Phase 3 (Comparison) | Normalized vs. unnormalized objective compared across seeds |

## Sources

- Existing codebase analysis: `Infer_BVMaster_charged_v7.py` (3-phase PDE pipeline), `Infer_BVMaster_charged_v13_ultimate.py` (7-phase surrogate+PDE pipeline), `Surrogate/objectives.py`, `Surrogate/cascade.py`, `Surrogate/multistart.py`
- `StudyResults/inverse_verification/parameter_recovery_summary.json`: surrogate bias = 10.67%, 2% noise per-seed errors = [27%, 15%, 400%]
- `StudyResults/surrogate_fidelity/fidelity_summary.json`: NN ensemble pc_max_nrmse = 151%, demonstrating extreme surrogate error in peroxide channel
- `StudyResults/master_inference_v13/master_comparison_v13.csv`: S2/S3/S4 converge to same surrogate minimum within 0.1%
- [Open Issues in Surrogate-Assisted Optimization](https://www.researchgate.net/publication/333560864_Open_Issues_in_Surrogate-Assisted_Optimization) -- surrogate accuracy/bias trade-offs (MEDIUM confidence)
- [Enhancing Inverse Problem Solutions with Accurate Surrogate Simulators](https://arxiv.org/abs/2304.13860) -- surrogate accuracy impact on inverse solutions (MEDIUM confidence)
- [Sensitivity, robustness, and identifiability in stochastic chemical kinetics models](https://pmc.ncbi.nlm.nih.gov/articles/PMC3102369/) -- parameter identifiability under noise (HIGH confidence)
- [Parameter identifiability analysis: Mitigating non-uniqueness](https://www.sciencedirect.com/science/article/pii/S0020768322000920) -- identifiability in inverse problems (MEDIUM confidence)
- [Comparison of automatic techniques for estimating the regularization parameter](https://academic.oup.com/gji/article/156/3/411/1999599) -- regularization and overfitting to noise realizations (HIGH confidence)
- [Reduced order and surrogate modeling for inverse problems](https://kiwi.oden.utexas.edu/papers/Surrogate-reduced-model-comparison-Frangos-Marzouk-Willcox.pdf) -- surrogate bias correction approaches (HIGH confidence)
- [A hybrid two-level MCMC framework with deep learning surrogates](https://www.sciencedirect.com/science/article/abs/pii/S0021999125007843) -- bias correction via correction chains (MEDIUM confidence)

---
*Pitfalls research for: PNP-BV surrogate-assisted inverse pipeline redesign (v14)*
*Researched: 2026-03-09*
