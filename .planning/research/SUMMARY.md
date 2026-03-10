# Project Research Summary

**Project:** PNP-BV v14 Pipeline Redesign -- Robust Surrogate-Assisted Inverse Parameter Recovery
**Domain:** Scientific computing / electrochemical inverse problem (Poisson-Nernst-Planck + Butler-Volmer kinetics)
**Researched:** 2026-03-09
**Confidence:** HIGH

## Executive Summary

The PNP-BV inverse pipeline recovers 4 kinetic parameters (k0_1, k0_2, alpha_1, alpha_2) from noisy I-V curves using a surrogate-first, PDE-refinement architecture. The v13 pipeline achieves good results on a single noise seed but is fundamentally non-robust: testing across multiple seeds reveals worst-case errors of 400%, with the pipeline tuned to exploit a lucky noise realization on the default seed. The v14 redesign must shift the success metric from single-seed accuracy to worst-case robustness across 10+ seeds at 2% noise. This is a measurement-before-optimization problem -- the first priority is building the multi-seed evaluation infrastructure, then using ablation experiments to determine which of v13's 7 phases actually contribute to robustness.

The recommended approach is ablation-first, redesign-second. The existing stack (Firedrake, SciPy, PyTorch) is validated and sufficient; only SALib (sensitivity analysis) and pandas (experiment results tabulation) need to be added. The core technical issues are: (1) the objective function improperly weights the two observables, causing k0_2 to be effectively ignored; (2) the surrogate has an ~11% structural bias on k0_2 that PDE refinement incompletely corrects; and (3) three of five surrogate stages (S1, S3, S4) converge to the same minimum, wasting complexity without improving robustness. The likely minimal pipeline is a single surrogate warm-start plus one PDE refinement phase, but this must be validated empirically.

The key risks are parameter identifiability collapse (k0_2 may be fundamentally unrecoverable from noisy I-V data at 2% noise) and overfitting to noise realizations. Mitigation requires Fisher Information / profile likelihood analysis to determine identifiability bounds, properly weighted least-squares objectives, and Tikhonov regularization on the PDE phases to prevent noise overfitting. If k0_2 proves non-identifiable, no amount of optimization engineering will fix it -- the experimental design (voltage grid, observables) must change.

## Key Findings

### Recommended Stack

The existing stack is validated and should not be changed. Two small additions are needed: SALib for Sobol sensitivity analysis (quantify which parameters the objective is actually sensitive to) and pandas for structured experiment comparison tables. All recommended optimization alternatives (differential evolution, dual annealing, least_squares) already exist in scipy and require no new installs. lmfit is an optional addition for standardized profile likelihood analysis but the existing manual implementation works.

**Core technologies (additions only):**
- SALib >=1.5.1: Sobol/Morris sensitivity analysis on surrogate objectives -- quantifies parameter identifiability
- pandas >=2.1: tabular aggregation of multi-seed, multi-variant experiment results -- replaces ad-hoc CSV wrangling
- scipy.optimize.differential_evolution (already installed): gradient-free global optimizer benchmark against multistart L-BFGS-B

**Explicitly avoid:** pymoo, Optuna, MLflow, emcee/PyMC, Hydra, nlopt. All are overkill or wrong abstraction for this problem.

### Expected Features

**Must have (table stakes):**
- Multi-seed validation protocol -- every variant tested on 10+ seeds, worst-case reported
- Weighted least squares / normalized residuals -- fix the objective so both observables contribute meaningfully
- Tikhonov regularization on PDE phases -- prevent noise overfitting (P2 currently degrades 3/5 seeds)
- Per-seed surrogate quality gate -- skip PDE refinement when surrogate warm-start is clearly wrong
- Relative/normalized residuals -- prevent high-magnitude voltage points from dominating

**Should have (differentiators for robustness):**
- Surrogate bias correction (space mapping) -- address the ~11% structural k0_2 bias
- Parameter identifiability analysis (profile likelihood) -- determine if k0_2 is recoverable
- Hybrid surrogate warm-start (POD-RBF-log k0_2 + NN k0_1/alpha) -- exploit each surrogate's strength
- Sensitivity-weighted voltage selection -- use FIM to choose informative voltage points

**Defer:**
- Bayesian inference with MCMC -- requires resolved surrogate bias first, high complexity
- Continuation in noise level -- unclear benefit with only 15-22 voltage points
- Surrogate-corrected PDE objective -- risk of preventing PDE from correcting surrogate bias

### Architecture Approach

The v14 architecture replaces v13's monolithic 1100-line script with a declarative PipelineConfig system. Pipeline variants are defined as data (dataclass configs), not new scripts. A single ExperimentRunner orchestrates seed sweeps, and a ResultsCollector aggregates metrics. The existing Surrogate and FluxCurve modules are reused unchanged -- only the orchestration layer is new.

**Major components:**
1. ExperimentRunner -- orchestrates pipeline variants across noise seeds, manages target caching
2. PipelineConfig / PDEPhaseConfig -- declarative pipeline definition replacing hardcoded phase sequences
3. ResultsCollector / ComparisonTable -- standardized metric computation and aggregation (median, worst-case, IQR)
4. Surrogate.objectives (existing, modify) -- add WLS normalization and Tikhonov regularization
5. FluxCurve (existing, unchanged) -- PDE-based refinement phases

### Critical Pitfalls

1. **Overfitting to a single noise seed** -- The entire v7-v13 evolution was tuned against seed 20260226. Mitigate by making multi-seed evaluation the first task, before any redesign.
2. **Surrogate bias masquerading as noise sensitivity** -- The ~11% surrogate bias on k0_2 is the dominant error source, not noise. Mitigate by quantifying per-parameter bias at 0% noise and designing asymmetric observable weighting (downweight peroxide in surrogate stages where its error is 150x worst-case).
3. **Uncontrolled pipeline comparisons** -- Changing two things simultaneously (e.g., optimizer + voltage grid) prevents attribution. Mitigate with controlled single-variable ablation and paired Wilcoxon tests across seeds.
4. **Redundant surrogate stages** -- S2, S3, S4 converge to the same minimum (losses identical to 4 significant figures). The 20K multistart costs 10s for zero new information. Mitigate by ablating and removing stages that do not improve PDE refinement outcomes.
5. **Parameter identifiability collapse for k0_2** -- k0_2 governs a secondary electrochemical reaction and may be below the noise floor at 2% noise. Mitigate with FIM analysis and profile likelihood; if non-identifiable, accept uncertainty bounds rather than chasing optimization improvements.

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Baseline and Multi-Seed Evaluation
**Rationale:** Cannot improve what is not measured. The single-seed evaluation is the root cause of false confidence in v13. This must come first because every subsequent phase depends on having a reliable multi-seed benchmark.
**Delivers:** v13 baseline results across 10+ noise seeds; standardized metrics module; seed-sweep harness.
**Addresses:** Multi-seed validation protocol (table stakes); FIM/identifiability diagnosis for k0_2.
**Avoids:** Pitfall 1 (single-seed overfitting), Pitfall 5 (identifiability collapse -- diagnosed here).

### Phase 2: Controlled Ablation of v13 Components
**Rationale:** Before adding anything new, determine which existing components matter. Intuition about S1/S3/S4/P1 may be wrong. Data-driven ablation settles this cheaply.
**Delivers:** Ablation table showing per-component contribution to robustness; identification of the minimal effective pipeline; quantified surrogate bias per parameter at 0% noise.
**Addresses:** Surrogate quality gate; redundant stage removal.
**Avoids:** Pitfall 3 (uncontrolled comparison), Pitfall 4 (redundant stages).
**Uses:** pandas for comparison tables, scipy.stats.wilcoxon for paired tests.

### Phase 3: Objective Function Redesign and Component Experiments
**Rationale:** The objective function is the highest-leverage single fix. WLS normalization, Tikhonov regularization, and observable weighting are all low-complexity changes that address the root causes (k0_2 under-weighting, PDE noise overfitting). Test one change at a time against the ablated baseline.
**Delivers:** Redesigned objective with proper normalization; Tikhonov-regularized PDE phases; weight sweep results; alternative optimizer benchmarks (differential evolution vs. multistart L-BFGS-B).
**Addresses:** WLS objective, normalized residuals, Tikhonov regularization, sensitivity-weighted voltage selection.
**Avoids:** Pitfall 2 (surrogate bias), Pitfall 7 (objective scale mismatch).
**Uses:** SALib for sensitivity analysis of the redesigned objective.

### Phase 4: v14 Pipeline Implementation
**Rationale:** Only build the final pipeline after empirical evidence from Phases 2-3 determines the optimal component set. Implement as a declarative PipelineConfig system to prevent script sprawl.
**Delivers:** v14 pipeline as a PipelineConfig; ExperimentRunner; surrogate bias correction; hybrid warm-start.
**Addresses:** Surrogate bias correction, hybrid warm-start, adaptive PDE budget.
**Avoids:** Pitfall 6 (PDE wrong basin -- add PDE multistart if ablation shows value).
**Implements:** ExperimentRunner, PipelineConfig, ResultsCollector architecture.

### Phase 5: Robustness Validation and Documentation
**Rationale:** The pipeline is only "done" when it passes <10% error on all 4 parameters across 20+ seeds at 2% noise. This phase also produces the comparison artifacts (v14 vs v13) for publication.
**Delivers:** 20+ seed validation sweep; v14 vs v13 comparison with statistical tests; pytest robustness test; documentation.
**Addresses:** Final validation of all table stakes features.

### Phase Ordering Rationale

- Phases 1-2 produce no new pipeline code -- they measure and ablate the existing system. This is essential because the v13 pipeline may already be close to optimal with minor fixes, and building new components without measurement would waste effort.
- Phase 3 addresses the highest-leverage changes (objective function, regularization) independently of architectural changes. These are small code changes with large potential impact.
- Phase 4 comes after empirical evidence because the architecture of v14 depends on knowing which components survive ablation. Building PipelineConfig before knowing the pipeline structure is premature abstraction.
- Phase 5 is pure validation and cannot run until v14 exists.
- Dependencies: Phase 2 depends on Phase 1 metrics. Phase 3 depends on Phase 2 baseline. Phase 4 depends on Phase 2+3 results. Phase 5 depends on Phase 4.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 3:** Objective function redesign involves choosing between WLS formulations (inverse-variance vs inverse-magnitude vs per-point normalization). The optimal formulation depends on the noise model and observable magnitudes, which vary across the voltage range. Sensitivity analysis (SALib) should guide this, but the design space is nontrivial.
- **Phase 4:** Surrogate bias correction (space mapping / manifold mapping) is well-documented in microwave engineering but less common in electrochemistry. The specific implementation for NN surrogates with PDE truth needs careful design.

Phases with standard patterns (skip research-phase):
- **Phase 1:** Multi-seed evaluation is straightforward infrastructure (loops + CSV output). No domain-specific complexity.
- **Phase 2:** Ablation experiments follow a standard protocol (remove component, measure, compare). Well-documented in ML literature.
- **Phase 5:** Validation sweeps are standard; the pytest integration follows existing patterns in the codebase.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All recommendations are either already installed (scipy) or de facto standards (SALib, pandas). No speculative technology choices. |
| Features | HIGH | Feature priorities derived directly from v13 failure analysis with quantitative evidence (per-seed errors, surrogate bias numbers, convergence data). |
| Architecture | HIGH | Architecture recommendations based on direct codebase analysis of the 1100-line v13 script and 42+ existing inference scripts. Component boundaries follow existing module structure. |
| Pitfalls | HIGH | All pitfalls documented with specific quantitative evidence from existing results (e.g., S2/S3/S4 loss identity, 151% peroxide NRMSE, per-seed error distributions). |

**Overall confidence:** HIGH

### Gaps to Address

- **k0_2 identifiability at 2% noise:** Profile likelihood analysis will determine whether k0_2 is fundamentally recoverable. If it is not, the pipeline target must be revised (e.g., accept wider bounds on k0_2, or require additional experimental conditions). This is the single highest-risk unknown.
- **Optimal observable weighting:** The best secondary_weight (or per-point WLS scheme) is unknown. The weight sweep in Phase 3 will resolve this, but the search space is large if per-point weights are considered.
- **PDE refinement basin structure:** Whether the PDE objective has multiple basins (and whether the surrogate optimum lands in the right one) is unknown. PDE multistart experiments in Phase 4 will resolve this.
- **Surrogate bias correction effectiveness:** Space mapping has not been applied to NN surrogates in this specific domain. Phase 4 implementation may reveal that the bias is parameter-dependent in ways that simple additive/multiplicative correction cannot capture.

## Sources

### Primary (HIGH confidence)
- Existing codebase: `Infer_BVMaster_charged_v13_ultimate.py`, `Surrogate/objectives.py`, `Surrogate/cascade.py`, `Surrogate/multistart.py`, `FluxCurve/`
- `StudyResults/master_inference_v13/master_comparison_v13.csv` -- v13 per-stage recovery results
- `StudyResults/inverse_verification/parameter_recovery_summary.json` -- surrogate bias quantification
- `StudyResults/surrogate_fidelity/fidelity_summary.json` -- NN ensemble fidelity metrics
- SALib documentation (v1.5.2) -- sensitivity analysis methods
- SciPy documentation (v1.17) -- optimize, stats modules

### Secondary (MEDIUM confidence)
- [Revisiting parameter identification in EIS](https://www.sciencedirect.com/science/article/abs/pii/S0013468612015514) -- WLS and OED for electrochemical systems
- [PINN surrogate of Li-ion battery models: Regularization](https://arxiv.org/html/2312.17336) -- regularization strategies for surrogate-based inference
- [Structural and practical identifiability via profile likelihood](https://academic.oup.com/bioinformatics/article/25/15/1923/213246) -- profile likelihood methodology
- [Practical identifiability of electrochemical P2D models](https://link.springer.com/article/10.1007/s10800-021-01579-5) -- identifiability in Butler-Volmer kinetics
- [Reduced order and surrogate modeling for inverse problems](https://kiwi.oden.utexas.edu/papers/Surrogate-reduced-model-comparison-Frangos-Marzouk-Willcox.pdf) -- surrogate bias correction approaches

### Tertiary (needs validation)
- Space mapping / manifold mapping for NN surrogates -- well-established for RBF/kriging surrogates in engineering optimization, but application to NN ensembles in electrochemistry is novel. Validate in Phase 4.

---
*Research completed: 2026-03-09*
*Ready for roadmap: yes*
