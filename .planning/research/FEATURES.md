# Feature Landscape

**Domain:** Robust surrogate-assisted inverse parameter recovery pipeline (PNP-BV kinetics)
**Researched:** 2026-03-09

## Table Stakes

Features that a robust inverse pipeline must have to achieve <10% parameter recovery at 2% noise across seeds. Missing any of these likely means the pipeline fails on adversarial noise realizations.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Weighted least squares (WLS) objective | The current objective sums raw squared residuals across CD and PC observables. CD magnitudes are ~100x larger than PC, so the optimizer effectively ignores PC -- the observable that constrains k0_2. WLS with inverse-variance or inverse-magnitude weighting is standard practice in electrochemical parameter estimation (Revisiting parameter identification in EIS, Electrochimica Acta 2013). Without it, k0_2 recovery is noise-seed-dependent. | Low | Replace `0.5 * sum((cd_sim - cd_target)^2) + w * 0.5 * sum((pc_sim - pc_target)^2)` with per-point weighting: `sum(((sim_i - target_i) / sigma_i)^2)`. The `secondary_weight` parameter is a crude version of this but applies a single scalar to all PC points rather than per-point normalization. |
| Tikhonov regularization on PDE objective | PDE refinement phases (P1, P2) currently optimize an unregularized least-squares objective. With noisy targets, the optimizer overfits to noise -- explaining why P2 degrades results on 3/5 seeds. A penalty term `lambda * ||theta - theta_prior||^2` centered at the surrogate optimum prevents the PDE optimizer from wandering too far from the surrogate's warm-start. | Low | Add `+ lambda * sum((x - x_surrogate)^2)` to the PDE objective. The surrogate optimum serves as a natural prior. Lambda selection via discrepancy principle or L-curve. |
| Per-seed surrogate warm-start quality gate | v13 runs PDE refinement regardless of surrogate quality. Seed 42 starts PDE from 55% surrogate error -- no PDE budget can recover from this. A quality gate skips or modifies PDE refinement when the surrogate warm-start is clearly poor (e.g., surrogate loss > threshold). | Low | Compare surrogate loss against noise-free floor. If surrogate max parameter error exceeds a threshold (e.g., 30%), flag as unreliable and either skip PDE or use a different strategy. |
| Relative/normalized residuals | The current objective uses absolute residuals. This means voltage points with large current magnitudes dominate the objective, while points near zero current (where parameter sensitivity may be highest) contribute nothing. Relative residuals `((sim - target) / target)^2` or hybrid formulations are standard. | Low | Use `((sim_i - target_i) / max(|target_i|, epsilon))^2` to normalize. Must handle near-zero targets with a floor. |
| Multi-seed validation protocol | v13 was tuned and declared successful on a single seed (20260226). The 5-seed study revealed this was an outlier (median max error 23%). A robust pipeline must report median/worst-case across 10+ seeds, not best-case on one. | Low | Infrastructure to run N seeds, collect results, report median/IQR. Already partially exists in the v13 study. Formalize as a standard evaluation protocol. |
| Sensitivity-weighted voltage selection | Not all voltage points contribute equally to parameter identifiability. The Fisher Information Matrix (FIM) at the current parameter estimate identifies which voltages most constrain each parameter. The existing "shallow cathodic" and "full cathodic" subsets were chosen heuristically. Sensitivity-based selection is standard in optimal experimental design for electrochemistry. | Medium | Compute Jacobian J(theta) at the surrogate optimum, form FIM = J^T W J, analyze eigenvalues. Select voltage subsets that maximize the smallest eigenvalue (D-optimal or A-optimal criterion). Requires surrogate Jacobian, which is cheap. |
| Surrogate bias correction | The surrogate has ~11% structural bias on k0_2. All three surrogate strategies converge to the same biased optimum. Bias correction (additive or multiplicative) using a small number of PDE evaluations at or near the surrogate optimum is standard in multi-fidelity optimization. | Medium | Evaluate PDE at surrogate optimum and a few perturbations. Compute bias = PDE_residual - surrogate_residual. Apply correction to surrogate predictions in subsequent optimizations. This is the "space mapping" or "manifold mapping" approach from microwave engineering optimization. |

## Differentiators

Features that elevate the pipeline from "works on most seeds" to "robustly <10% on all seeds." Not strictly required but provide significant robustness improvements.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Parameter identifiability analysis (profile likelihood) | Quantifies whether each parameter is practically identifiable from the available data at each noise level. v13 data shows k0_2 is weakly identifiable from shallow cathodic data -- this can be diagnosed a priori via profile likelihood. A flat profile means the parameter cannot be recovered regardless of optimization quality. Enables principled decisions about which parameters to fix vs optimize. | Medium | For each parameter: fix it at a grid of values, re-optimize the others, plot the resulting objective vs parameter value. A U-shaped profile = identifiable; flat profile = non-identifiable. Use surrogate for cheap profiling, verify key points with PDE. Requires ~100 surrogate optimizations per parameter. |
| Bayesian inference with MCMC | Replaces point estimation with posterior distributions over parameters. Provides confidence intervals and reveals correlations (e.g., k0_1-k0_2 anti-correlation). A Bayesian approach with proper likelihood (accounting for noise variance) naturally regularizes the inverse problem. The surrogate is fast enough for MCMC (10K surrogate evaluations take ~4s). | Medium-High | Use emcee or PyMC with the surrogate as the forward model. Likelihood: Gaussian with known noise variance sigma=0.02*target. Prior: uniform on log(k0) and alpha bounds. Run 10K samples with 32 walkers. Post-process for MAP estimate and credible intervals. Risk: surrogate bias propagates into posterior. |
| Ensemble surrogate uncertainty weighting | The NN ensemble provides prediction variance at each voltage point. Points with high ensemble disagreement should be down-weighted in the objective because the surrogate is unreliable there. This is a cheap, automatic form of data quality weighting. | Low-Medium | Weight each residual by `1 / (sigma_noise^2 + sigma_surrogate^2)` where sigma_surrogate comes from ensemble variance. Requires NN ensemble (already exists). Does not help for PDE phases but improves surrogate phase robustness. |
| Hybrid surrogate warm-start | POD-RBF-log has 1.64% k0_2 error (vs NN's 12%), but 11.77% k0_1 error (vs NN's 6.88%). Use POD-RBF-log's k0_2 estimate with NN's k0_1/alpha estimates as a hybrid warm-start for PDE refinement. Already identified as a promising strategy in v13 writeup but not implemented. | Low | Take best k0_2 from POD-RBF-log, best k0_1/alpha from NN ensemble, combine as PDE initial guess. Trivial to implement; the question is whether it helps empirically. |
| Continuation in noise level | Instead of optimizing directly against 2%-noisy targets, start with heavily smoothed targets (low effective noise) and gradually reduce smoothing. Analogous to graduated non-convexity in image processing. The P1-to-P2 voltage continuation in v13 is already a form of this (coarse-to-fine in data space). | Medium | Apply Gaussian smoothing kernel to target I-V curves. Optimize against smoothed targets first, then reduce kernel width. Requires choosing smoothing schedule. May conflict with the discrete voltage grid (only 15-22 points). |
| Adaptive PDE budget allocation | Currently P1 gets 25 L-BFGS-B iterations and P2 gets 20, regardless of convergence state. An adaptive strategy would allocate more iterations to seeds where progress is being made and terminate early on seeds where the optimizer is diverging. | Low | Monitor per-iteration objective decrease. If objective increases for 3 consecutive iterations, terminate early and revert to best iterate. Already partially implemented (L-BFGS-B stops at gradient tolerance) but explicit divergence detection would help. |
| Surrogate-corrected PDE objective | Use the surrogate as a "physics prior" in the PDE objective: `J_PDE(theta) + mu * J_surrogate(theta)`. The surrogate term prevents the PDE optimizer from exploring regions where the surrogate disagrees with the PDE. As PDE iterations progress, reduce mu (simulated annealing on the surrogate weight). | Medium | Requires simultaneous surrogate and PDE evaluation. mu schedule must be tuned. Risk: if surrogate is biased, this prevents the PDE from correcting the bias. Use only as initial damping, annealing to mu=0. |

## Anti-Features

Features to explicitly NOT build. These seem appealing but are traps for this specific problem.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| More surrogate training data | The surrogate bias is ~11% on k0_2 even at 0% noise. This is a structural limitation of the NN architecture/training, not a data quantity problem. The NN ensemble already trains on adequate data (implied by 0.06-0.41% median NRMSE on hold-out). More data will not fix the fundamental k0_1-k0_2 trade-off in the surrogate landscape. | Fix the objective function weighting and use surrogate bias correction instead. |
| More surrogate phases (S5, S6, ...) | v13 proved all three surrogate strategies converge to the identical optimum. The surrogate landscape has a single basin. Adding more surrogate optimization strategies is futile -- they will all find the same point. | Invest compute budget in PDE refinement or bias correction instead. |
| Global optimization on PDE objective (e.g., genetic algorithm) | PDE evaluations cost ~10s each. A population-based optimizer needs 1000+ evaluations = 3+ hours per run per seed. The surrogate multistart already provides global basin verification. | Use surrogate for global search, PDE for local refinement (current architecture is correct). |
| Complex multi-objective optimization (Pareto front) | The problem has a single goal (recover 4 parameters). Framing it as multi-objective (CD fit vs PC fit) adds complexity without benefit. The real issue is objective function weighting, not Pareto optimality. | Use properly weighted single-objective least squares. |
| Deep learning surrogate replacement (transformers, etc.) | The NN ensemble already achieves 0.06-0.41% median NRMSE. The bottleneck is not surrogate accuracy but surrogate-to-PDE bias and noise amplification. A fancier surrogate architecture will not fix the ill-conditioning of the inverse problem. | Focus on regularization and objective function design. |
| Full Bayesian PDE inference | MCMC with PDE forward model is computationally infeasible (~10s per evaluation, need ~100K samples = ~12 days). | Use Bayesian inference with surrogate only, then refine MAP estimate with PDE. Or use surrogate-accelerated MCMC where PDE is called only for proposal validation. |

## Feature Dependencies

```
Weighted least squares objective -> All optimization features (foundational change)
Relative/normalized residuals -> Weighted least squares (combined in same objective redesign)
Sensitivity-weighted voltage selection -> Surrogate Jacobian computation
Parameter identifiability analysis -> Sensitivity-weighted voltage selection (shares FIM infrastructure)
Tikhonov regularization -> Surrogate bias correction (surrogate optimum serves as regularization center)
Multi-seed validation protocol -> All other features (needed to evaluate any change)
Ensemble surrogate uncertainty weighting -> NN ensemble infrastructure (already exists)
Hybrid surrogate warm-start -> Both POD-RBF-log and NN ensemble models (already exist)
Bayesian inference with MCMC -> Weighted least squares objective (likelihood formulation)
Adaptive PDE budget allocation -> PDE refinement infrastructure (already exists)
```

## MVP Recommendation

Prioritize (immediate impact, low complexity):
1. **Multi-seed validation protocol** -- Formalize the 5-seed (or 10-seed) evaluation as the standard test. Every subsequent change is evaluated on this. Without this, you cannot tell if any change actually helps.
2. **Weighted least squares / normalized residuals** -- Replace the raw squared residual objective with properly weighted residuals. This is the single most likely fix for the k0_2 bottleneck: the current objective under-weights PC residuals where k0_2 sensitivity lives. The `secondary_weight` parameter is a crude proxy; proper per-point normalization is better.
3. **Tikhonov regularization on PDE phases** -- Add `lambda * ||theta - theta_surr||^2` to the PDE objective. Prevents P2 from overfitting to noise (the cause of degradation on seeds 7777, 99999). Lambda tuned via L-curve or discrepancy principle.

Prioritize (medium-term, medium complexity):
4. **Surrogate bias correction** -- Use a few PDE evaluations to estimate and correct the surrogate's structural k0_2 bias. This directly addresses the ~11% bias floor.
5. **Parameter identifiability analysis** -- Profile likelihood plots reveal which parameters are fundamentally recoverable from the data. If k0_2 is non-identifiable at certain voltage ranges, no optimization trick will fix it -- the data itself must change.
6. **Hybrid surrogate warm-start** -- Combine POD-RBF-log k0_2 with NN k0_1/alpha. Trivial to implement, directly addresses each surrogate's strength.

Defer:
- **Bayesian inference with MCMC**: High complexity, and the surrogate bias issue must be resolved first or it propagates into the posterior. Pursue after the point-estimation pipeline is robust.
- **Continuation in noise level**: Medium complexity, unclear benefit with only 15-22 voltage points. The voltage continuation (P1 -> P2) already provides a form of this.
- **Surrogate-corrected PDE objective**: Risk of preventing PDE from correcting surrogate bias. Only consider if Tikhonov regularization proves insufficient.

## Sources

- Existing codebase: `Surrogate/objectives.py`, `v13_ultimate_inference.md`, `pipelines.py`
- [Revisiting parameter identification in EIS: Weighted least squares and optimal experimental design](https://www.sciencedirect.com/science/article/abs/pii/S0013468612015514) -- WLS and OED for electrochemical systems
- [PINN surrogate of Li-ion battery models for parameter inference. Part II: Regularization](https://arxiv.org/html/2312.17336) -- Regularization strategies for surrogate-based parameter inference in electrochemistry
- [Structural and practical identifiability analysis by exploiting the profile likelihood](https://academic.oup.com/bioinformatics/article/25/15/1923/213246) -- Profile likelihood for practical identifiability
- [Model inversion via multi-fidelity Bayesian optimization](https://royalsocietypublishing.org/doi/10.1098/rsif.2015.1107) -- Multi-fidelity surrogate frameworks for inverse problems
- [Uncertainty quantification and propagation in surrogate-based Bayesian inference](https://link.springer.com/article/10.1007/s11222-025-10597-8) -- Surrogate uncertainty propagation in parameter estimation
- [Practical identifiability of electrochemical P2D models](https://link.springer.com/article/10.1007/s10800-021-01579-5) -- Identifiability challenges in electrochemical models with Butler-Volmer kinetics
- [Optimal Input Design for Parameter Identification (Berkeley)](https://ecal.studentorg.berkeley.edu/pubs/ACC19_OED_CVX.pdf) -- OED via convex optimization for electrochemical systems
- [Sensitivity, robustness, and identifiability in stochastic chemical kinetics models](https://www.pnas.org/doi/10.1073/pnas.1015814108) -- Fisher Information Matrix for kinetics parameter identifiability
- [Profile-Wise Analysis workflow for identifiability](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011515) -- Unified PWA workflow
- [Tikhonov regularization overview](https://www.sciencedirect.com/topics/computer-science/tikhonov-regularization) -- General regularization theory
- [Surrogate-assisted evolutionary algorithms for expensive optimization](https://link.springer.com/article/10.1007/s40747-024-01465-5) -- Noise handling in surrogate-assisted optimization
