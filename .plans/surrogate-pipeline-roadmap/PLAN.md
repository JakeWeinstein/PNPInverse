# Roadmap: Surrogate Model Expansion and Inverse Pipeline Refinement

**Goal:** Systematically improve parameter recovery (especially k0_2) by diagnosing training data gaps, building new surrogate architectures with analytic gradients, implementing iterative surrogate refinement, and hardening the PDE optimization stage.

**Date:** 2026-03-16

**Research used:** ISMO iterative refinement, DeepONet/FNO/PEDS architecture studies, GP surrogate (GPyTorch) literature, multi-fidelity trust-region methods, weight annealing for PDE-constrained optimization, Ramirez & Latz 2024 multi-fidelity PINN for BV kinetics.

---

## Approach

The current pipeline achieves ~10.7% irreducible surrogate error and catastrophic k0_2 failure under even 1% noise. The roadmap attacks this from four angles in sequence:

1. **Diagnose before building** -- understand whether training data coverage (especially in the k0_2 dimension) is a root cause of surrogate bias before investing in new architectures.
2. **Expand the surrogate zoo** -- train GP, DeepONet, FNO, and PEDS surrogates on the existing (and potentially augmented) data, all with analytic gradient support via PyTorch autograd or GPyTorch.
3. **Close the loop** -- implement ISMO-style iterative refinement so surrogates improve where they matter most (near the optimizer's trajectory).
4. **Harden PDE refinement** -- improve the final PDE stage with trust-region surrogate-to-PDE transitions and/or objective weight annealing to stabilize k0_2 recovery.

Each phase produces artifacts that feed the next, but Phases 2a-2d (individual surrogate builds) can run in parallel.

---

## Phase Overview

| Phase | Title | Depends On | Parallelizable |
|-------|-------|-----------|----------------|
| 1 | Training Data Audit | -- | -- |
| 2a | GP Surrogate (GPyTorch) | 1 | Yes (with 2b-2d) |
| 2b | DeepONet Surrogate | 1 | Yes (with 2a,2c,2d) |
| 2c | FNO Surrogate | 1 | Yes (with 2a,2b,2d) |
| 2d | PEDS Surrogate | 1 | Yes (with 2a-2c) |
| 2e | Autograd Retrofit for NN Ensemble | 1 | Yes (with 2a-2d) |
| 3 | Comparative Surrogate Benchmark | 2a-2e | No |
| 4 | Iterative Surrogate Refinement (ISMO) | 3 | No |
| 2f | PCE Surrogate (ChaosPy) | 1 | Yes (with 2a-2e) |
| 5 | PDE Refinement Stage Improvements | 3 | Partially (with 4) |
| 6 | End-to-End Integration and Validation | 4, 5 | No |
| 7 | Bayesian Posterior Sampling | 6 | No |

---

## Phase 1: Training Data Audit

**Goal:** Determine whether the 3,194-sample training set has gaps, biases, or insufficient coverage -- particularly in the k0_2 dimension -- that could explain surrogate bias and k0_2 sensitivity.

**Inputs:**
- `data/surrogate_models/training_data_merged.npz` (3,194 samples, 4D parameter space)
- Parameter ranges: k0_1 in [1e-6, 1.0], k0_2 in [1e-7, 0.1], alpha_1/alpha_2 in [0.1, 0.9]

**Outputs:**
- Diagnostic report with visualizations: 1D marginal histograms, 2D pairwise scatter plots (especially k0_2 vs every other param), coverage heatmaps in log-space
- Quantified gap analysis: identify under-sampled regions (e.g., bins with <N samples)
- Sensitivity-aware coverage metric: does the k0_2 range have sufficient density in the region where k0_2 has the weakest I-V signal?
- Recommendation: augment data (run more PDE solves) or proceed with existing data
- If augmentation needed: a sampling plan (LHS or targeted) specifying how many new samples and where

**Key decisions for detailed planning:**
- What coverage metric to use (e.g., discrepancy, max-empty-ball radius in log-space)
- Whether to run a quick sensitivity analysis (Morris or Sobol via existing surrogates) to identify which k0_2 sub-ranges matter most
- Threshold for "sufficient coverage" -- when is augmentation needed?

**Success criteria:**
- Clear quantitative assessment: sample density per log-decade of k0_2
- Identified specific gaps (if any) with coordinates
- Go/no-go decision on data augmentation before Phase 2

**Depends on:** Nothing

**Research basis:** k0_2 is the most noise-sensitive parameter (research summary noise robustness results). Its range spans 4 orders of magnitude in log-space vs 3 for k0_1, making uniform coverage harder.

---

## Phase 2a: GP Surrogate (GPyTorch)

**Goal:** Build a Gaussian Process surrogate using GPyTorch that provides both predictions and calibrated uncertainty estimates, with analytic gradients via autograd.

**Inputs:**
- Training data from Phase 1 (original or augmented)
- Existing surrogate API contract: `predict(k0_1, k0_2, alpha_1, alpha_2)` and `predict_batch(parameters)` returning `{current_density, peroxide_current, phi_applied}`

**Outputs:**
- `Surrogate/gp_model.py` implementing the shared API plus `predict_with_uncertainty()`
- Trained GP model artifacts saved to `data/surrogate_models/gp/`
- Gradient method using GPyTorch autograd (no finite differences)
- Per-sample error and UQ calibration metrics on held-out test set

**Key decisions for detailed planning:**
- Kernel choice (Matern 5/2 vs RBF, ARD vs isotropic)
- Multi-output strategy: independent GPs per output dimension, or multi-task GP (ICM/LMC) to share information across the 44 output dimensions (22 CD + 22 PC)
- Scalability: with 3,194 samples and 44 outputs, exact GP is likely too expensive -- default to SVGP (Sparse Variational GP) or KISS-GP; only try exact GP if SVGP is insufficient
- Whether to fit in log-space or natural space for outputs

**Success criteria:**
- Prediction error competitive with or better than current NN ensemble (~10.7% worst-case)
- Calibrated uncertainty: 90% of test points within 90% predictive interval
- Gradient computation via autograd, no FD fallback

**Depends on:** Phase 1

**Research basis:** GP surrogates are excellent with moderate training sets (research summary: sweet spot for 4D/200 samples, GPyTorch recommended). Built-in UQ is unique among the candidate architectures.

---

## Phase 2b: DeepONet Surrogate [DEPRIORITIZED — optional, implement only if time permits]

**Status:** Skipped for now. The operator-learning inductive bias (function→function) doesn't strongly benefit a 4-scalar→44-scalar problem. Revisit if moving to finer voltage grids where trunk basis functions generalize across resolutions.

**Goal:** Train a DeepONet that learns the parameter-to-IV-curve operator, with PyTorch autograd gradients.

**Inputs:**
- Training data from Phase 1
- Existing surrogate API contract

**Outputs:**
- `Surrogate/deeponet_model.py` implementing the shared API
- Trained model artifacts in `data/surrogate_models/deeponet/`
- Autograd-based gradient method

**Key decisions for detailed planning:**
- Branch net architecture (MLP taking 4 parameters) and trunk net architecture (MLP taking voltage eta)
- Whether to use the unstacked (separate CD/PC DeepONets) or stacked (single DeepONet with 2-channel output) approach
- Training strategy: learning rate schedule, batch size, early stopping
- Whether to incorporate physics loss (DeepBayONet-style) or keep it purely data-driven initially

**Success criteria:**
- Prediction error on held-out set <= current NN ensemble error
- Autograd gradients match FD gradients to within 1% relative error
- Training completes in < 2 hours on available hardware

**Depends on:** Phase 1

**Research basis:** DeepBayONet achieved 0.999 parameter recovery on reaction-diffusion (research summary). Needs 1000s+ samples -- current 3,194 is borderline sufficient.

---

## Phase 2c: FNO Surrogate [DEPRIORITIZED — 22-point output too coarse for spectral methods]

**Status:** Skipped. 22 voltage points gives only 11 Fourier modes — not enough for FNO's spectral learning to outperform a plain MLP. DIFNO (the main justification) requires overnight Jacobian data generation for marginal benefit when Phase 2e autograd retrofit gives exact NN gradients for free. Revisit only if moving to finer voltage grids (100+ points).

**Goal:** Train a Fourier Neural Operator variant for the parameter-to-IV-curve mapping, with autograd gradients.

**Inputs:**
- Training data from Phase 1
- Existing surrogate API contract

**Outputs:**
- `Surrogate/fno_model.py` implementing the shared API
- Trained model artifacts in `data/surrogate_models/fno/`
- Autograd-based gradient method

**Key decisions for detailed planning:**
- Whether to use standard FNO (1D in voltage dimension) or the DIFNO variant that trains on Frechet derivatives (critical for optimization)
- Number of Fourier modes to retain (the I-V curves have 22 points -- fairly low resolution)
- Whether the 22-point voltage discretization is too coarse for FNO to shine (FNO benefits most from resolution invariance on fine grids)
- Lifting/projection layer design for the 4D parameter input

**Success criteria:**
- Prediction error on held-out set competitive with NN ensemble
- If DIFNO: gradient accuracy verified against PDE adjoint gradients
- Clear assessment of whether FNO adds value at this output resolution

**Depends on:** Phase 1

**Research basis:** FNO is resolution-invariant with spectral learning; DIFNO trains on outputs AND Frechet derivatives (research summary). However, with only 22 voltage points, the spectral advantage may be limited -- this phase should evaluate whether FNO is a good fit.

---

## Phase 2d: PEDS Surrogate [DEPRIORITIZED — coarse mesh convergence risk + inference latency]

**Status:** Skipped. The BV solver with Butler-Volmer exponentials is unlikely to converge reliably on a 10x coarser mesh (N=30), especially at high k0 and large overpotentials where the fine solver already has 8.4% failure rate. Additionally, every predict() call requires a ~0.5s PDE solve, making PEDS unusable for the 20,000-candidate multi-start stage. Revisit only if a stable coarse solver configuration is identified.

**Goal:** Build a Physics-Enhanced Deep Surrogate that combines a coarse PDE solve with a neural correction, potentially requiring far less training data.

**Inputs:**
- Training data from Phase 1
- Existing Firedrake BV solver (`Forward/bv_solver/` package)
- Coarser mesh configuration for the "cheap physics" baseline

**Outputs:**
- `Surrogate/peds_model.py` implementing the shared API
- Coarse solver wrapper that produces baseline I-V curves on a reduced mesh
- Neural corrector model trained on (coarse_prediction, fine_prediction) residuals
- Trained artifacts in `data/surrogate_models/peds/`

**Key decisions for detailed planning:**
- Coarse mesh resolution (what mesh parameters give ~10x speedup over full solve?)
- Neural corrector architecture: MLP on residuals, or conditional on parameters?
- Whether to propagate gradients through the coarse solver (complex, requires Firedrake adjoint on coarse mesh) or treat it as a fixed feature extractor
- Training data generation: do we need paired (coarse, fine) solves for all 3,194 samples, or can we train on a subset?

**Success criteria:**
- Prediction error significantly better than pure NN (research suggests 3x improvement)
- Coarse solver adds < 0.5s per evaluation (vs ~seconds for full PDE)
- Can train effectively on fewer samples than pure NN approaches

**Depends on:** Phase 1

**Research basis:** PEDS is 3x more accurate than NN-only with 100x less training data (research summary). Requires a coarse solver -- the existing Firedrake BV solver on a coarser mesh is a natural fit.

---

## Phase 2e: Autograd Retrofit for Existing NN Ensemble

**Goal:** Replace finite-difference gradient computation in the existing NN ensemble with PyTorch autograd, as a quick win that improves gradient accuracy and reduces evaluation count.

**Inputs:**
- Existing `Surrogate/nn_model.py` (PyTorch ResNet-MLP, already uses torch)
- Existing `Surrogate/objectives.py` (currently uses 8-eval central FD for 4D gradient)

**Outputs:**
- Updated `objectives.py` with autograd gradient path for NN-based surrogates
- Verified gradient correctness: autograd vs FD comparison
- Benchmark: wall-clock speedup from eliminating 8 FD evaluations per gradient

**Key decisions for detailed planning:**
- Whether to modify `SurrogateObjective` to detect torch-based surrogates and switch to autograd, or create a parallel `AutogradSurrogateObjective` class
- How to handle the log-space transform in the autograd graph (need torch operations, not numpy)
- Whether to also support autograd for the ensemble mean (average of 5 member gradients)

**Success criteria:**
- Autograd gradients match FD gradients to < 0.1% relative error
- Gradient evaluation is >= 4x faster (eliminating 8 forward passes)
- No change to optimization results (same converged parameters to within tolerance)

**Depends on:** None -- can start immediately (no real Phase 1 dependency)

**Research basis:** Research summary identified this as an "easy fix" gap: "No analytic NN gradients (using FD instead of PyTorch autograd)."

---

## Phase 2f: PCE Surrogate (ChaosPy)

**Goal:** Build a Polynomial Chaos Expansion surrogate that provides Sobol sensitivity indices alongside predictions, quantifying how much each parameter contributes to I-V curve variance.

**Inputs:**
- Training data from Phase 1 (original or augmented)
- Existing surrogate API contract

**Outputs:**
- `Surrogate/pce_model.py` implementing the shared API
- Per-parameter and interaction Sobol sensitivity indices (directly from PCE coefficients)
- Sensitivity report: how much information do the I-V curves carry about k0_2 vs other parameters?
- Trained PCE artifacts in `data/surrogate_models/pce/`

**Key decisions for detailed planning:**
- Polynomial basis type (Legendre for bounded uniform inputs in log-space)
- Maximum polynomial degree and sparse truncation strategy
- Fitting method: least-squares regression or sparse regression (LARS/OMP) on the 3,194 samples
- Whether to build one PCE per output dimension or use a joint formulation

**Success criteria:**
- Prediction error competitive with POD-RBF baseline
- Sobol indices computed for all 4 parameters and key 2-way interactions
- Clear quantitative answer: what fraction of I-V variance is attributable to k0_2?

**Depends on:** Phase 1

**Research basis:** PCE provides Sobol sensitivity indices directly from coefficients (research summary). Spectral convergence for smooth maps. ChaosPy (Python) is the recommended library. 4D is well within PCE's comfort zone.

---

## Phase 3: Comparative Surrogate Benchmark

**Goal:** Run a standardized head-to-head comparison of all surrogate models (existing RBF, POD-RBF, NN ensemble, plus new GP, DeepONet, FNO, PEDS, PCE) on identical test data, focusing on k0_2 recovery.

**Inputs:**
- All trained surrogate models from Phases 2a-2e
- Held-out test set (from Phase 1 split or new PDE solves)
- Existing benchmark infrastructure in `StudyResults/surrogate_fidelity/`

**Outputs:**
- Updated `StudyResults/surrogate_fidelity/` with per-model error metrics
- k0_2-stratified error analysis: error as a function of k0_2 value
- Gradient accuracy comparison (autograd models vs FD vs PDE adjoint)
- Inference speed benchmark (prediction + gradient wall-clock time)
- Ranking and selection: which 1-2 surrogates to carry forward into Phases 4-5

**Key decisions for detailed planning:**
- Test set design: random hold-out vs structured grid vs adversarial (k0_2-focused)
- Whether to run the full inverse pipeline (multistart + cascade + PDE) with each surrogate, or just measure prediction/gradient accuracy
- Metrics: L2 error, max relative error, k0_2-specific error, UQ calibration (for GP)

**Success criteria:**
- At least one new surrogate shows measurable improvement over current 10.7% worst-case error; target < 5% if data augmentation was performed in Phase 1, otherwise < 8%
- Clear winner(s) identified for the ISMO and PDE refinement phases
- k0_2 error specifically reduced compared to current surrogates
- PCE Sobol indices provide quantitative sensitivity ranking for all 4 parameters

**Depends on:** Phases 2a-2f

**Research basis:** The existing fidelity study (`StudyResults/surrogate_fidelity/fidelity_summary.json`) provides the baseline to beat.

---

## Phase 4: Iterative Surrogate Refinement (ISMO)

**Goal:** Implement an ISMO-style loop that alternates between surrogate optimization and PDE evaluation to iteratively improve the surrogate in regions that matter for inference.

**Inputs:**
- Best surrogate(s) from Phase 3
- PDE solver (Firedrake BV solver with pyadjoint)
- Existing multistart + cascade pipeline

**Outputs:**
- `Surrogate/ismo.py` implementing the train-optimize-evaluate-augment loop
- Updated training data with adaptively sampled points near optimizer trajectories
- Convergence curves showing surrogate error reduction per ISMO iteration
- Refined surrogate model artifacts

**Key decisions for detailed planning:**
- How many PDE evaluations per ISMO iteration (budget per round)
- Acquisition strategy: sample at surrogate optima? Along optimizer trajectory? At points of maximum surrogate uncertainty (for GP)?
- Convergence criterion: when to stop the ISMO loop (surrogate-vs-PDE agreement threshold)
- Whether to retrain from scratch or incrementally update the surrogate each round. **Note:** For PyTorch-based surrogates (NN, DeepONet, FNO), warm-start from previous iteration's weights (fine-tune ~50-100 epochs vs ~5000 from scratch). This cuts retrain time by 10-50x and makes ISMO feasible on CPU. Only retrain from scratch if warm-start shows degradation.
- Integration with existing pipeline: does ISMO replace multistart, or wrap around it?

**Success criteria:**
- Demonstrable convergence: surrogate error at optimizer solution decreases monotonically across ISMO iterations
- k0_2 recovery improves by at least 50% relative to Phase 3 best
- Total PDE evaluation budget is bounded (e.g., < 200 new PDE solves)

**Depends on:** Phase 3

**Research basis:** ISMO achieves exponential convergence by focusing samples where the optimizer needs them (research summary). For GP surrogates, uncertainty-guided acquisition is a natural fit.

---

## Phase 5: PDE Refinement Stage Improvements

**Goal:** Improve the final PDE-based optimization stage that polishes surrogate solutions, addressing the surrogate-to-PDE transition and objective function design.

**Inputs:**
- Current PDE pipeline (`FluxCurve/bv_run/pipelines.py`, `optimization.py`)
- Best surrogate from Phase 3 (for warm-starting)
- Research on trust-region methods and weight annealing

**Outputs:**
- Improved PDE optimization stage with one or both of:
  - **Trust-region surrogate-to-PDE transition:** surrogate proposes step, PDE verifies, trust-region adjusts step size based on agreement
  - **Weight annealing:** gradual shift of objective weights during PDE optimization (e.g., start CD-heavy to lock k0_1/alpha_1, then anneal toward PC-heavy to recover k0_2)
- Updated `FluxCurve/bv_run/optimization.py` with new refinement strategies
- Comparison: current PDE stage vs improved PDE stage on the same surrogate warm-starts

**Key decisions for detailed planning:**
- Trust-region implementation: use scipy's trust-region solvers, or implement a custom surrogate-model-management framework?
- Weight annealing schedule: linear, exponential, or adaptive (based on convergence)?
- Whether to combine trust-region and weight annealing or test separately
- How to handle the surrogate-PDE fidelity gap (the ~10.7% bias) during transition
- Note: Tikhonov regularization was already tested extensively (WeekOfMar4 writeup). It prevents PDE regression but does not improve accuracy beyond the surrogate baseline. The winning v11/v13 pipeline uses no regularization. Focus effort on trust-region and weight annealing instead.

**Success criteria:**
- k0_2 recovery error reduced by >= 30% compared to current PDE stage
- No regression on k0_1, alpha_1, alpha_2 recovery
- Robust under 1-2% noise (no more catastrophic k0_2 failures at seed 44)

**Depends on:** Phase 3 (can begin in parallel with Phase 4 since it improves the PDE stage independently)

**Research basis:** Multi-fidelity trust-region (research summary) manages surrogate-to-PDE transitions. The existing cascade (`Surrogate/cascade.py`) already does sequential CD-then-PC optimization -- weight annealing generalizes this to the PDE stage.

---

## Phase 6: End-to-End Integration and Validation

**Goal:** Combine the best surrogate(s), ISMO refinement, and improved PDE stage into a single pipeline and validate on the full test suite including noise robustness.

**Inputs:**
- ISMO-refined surrogate from Phase 4
- Improved PDE stage from Phase 5
- Existing validation infrastructure (`StudyResults/inverse_verification/`)

**Outputs:**
- Integrated pipeline configuration (which surrogate, ISMO settings, PDE refinement strategy)
- Updated `StudyResults/inverse_verification/parameter_recovery_summary.json`
- Noise robustness study: 0%, 1%, 2%, 5% noise at multiple seeds
- Updated V&V report figures and tables (`writeups/vv_report/`)

**Key decisions for detailed planning:**
- Final pipeline architecture: single surrogate or multi-fidelity cascade?
- Whether to include UQ (GP posterior or ensemble spread) in the final output
- Validation protocol: same test cases as current study, or expanded?

**Success criteria:**
- k0_2 max relative error < 20% at 1% noise (vs current 287% catastrophic failure)
- All 4 parameters recovered to < 10% at 0% noise (vs current 10.7% surrogate bias)
- No catastrophic failures at 2% noise across all tested seeds
- Pipeline wall-clock time comparable to or better than current

**Depends on:** Phases 4 and 5

**Research basis:** Full validation against the noise robustness results in the research summary.

---

## Phase 7: Bayesian Posterior Sampling

**Goal:** Add full Bayesian uncertainty quantification to the pipeline, providing posterior distributions over parameters rather than just point estimates.

**Inputs:**
- Validated pipeline from Phase 6
- GP surrogate from Phase 2a (for surrogate-accelerated proposals)
- PDE solver for accept/reject verification

**Outputs:**
- Posterior distributions for all 4 parameters under various noise levels
- Credible intervals and correlation structure
- Comparison: surrogate-only posterior vs delayed-acceptance (surrogate + PDE) posterior
- Assessment: are posteriors well-calibrated? Do credible intervals capture true values?

**Key decisions for detailed planning:**
- MCMC engine: PyMC, emcee, or custom HMC
- Sampling strategy: Calibrate-Emulate-Sample (GP-based MCMC) or Multi-Fidelity HMC (surrogate proposes, PDE verifies via delayed acceptance)
- Number of posterior samples needed for convergence
- Whether GP posterior variance is sufficiently calibrated to drive proposals

**Success criteria:**
- Posterior credible intervals contain true parameters at stated coverage level
- Posterior width quantifies actual recovery uncertainty (not overconfident or underconfident)
- Computational cost manageable (< 1 day wall-clock for a single inference run)

**Depends on:** Phase 6

**Research basis:** Calibrate-Emulate-Sample (Cleary et al. 2021) and Multi-Fidelity HMC (arXiv 2405.05033) from research summary. GP surrogate from Phase 2a provides the natural proposal model.

---

## Phase Dependencies (DAG)

```
Phase 2e (Autograd) -- starts immediately, no dependency
  |
Phase 1 (Data Audit)
  |
  +---> Phase 2a (GP)  --------+
  +---> Phase 2b (DeepONet) ----+
  +---> Phase 2c (FNO) --------+---> Phase 3 (Benchmark)
  +---> Phase 2d (PEDS) -------+         |
  +---> Phase 2f (PCE)  -------+         +---> Phase 4 (ISMO) --------+
                                          |                            |
                                          +---> Phase 5 (PDE Refine) -+---> Phase 6 (Integration)
                                                                              |
                                                                        Phase 7 (Bayesian)
```

- Phase 2e (autograd retrofit) has no Phase 1 dependency and can start immediately
- Phases 2a-2d, 2f are fully parallelizable after Phase 1
- Phases 4 and 5 can run in parallel after Phase 3
- Phase 6 requires both Phase 4 and Phase 5
- Phase 7 (Bayesian posterior sampling) follows Phase 6

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Training data has fundamental k0_2 coverage gaps | Medium | High | Phase 1 diagnoses this early; if gaps found, run targeted PDE solves before Phase 2 |
| GP doesn't scale to 44 output dimensions x 3,194 samples | Medium | Low | Use SVGP or independent-per-mode GPs; GP is one of several candidates |
| DeepONet/FNO underperform with only 3,194 samples | Medium | Medium | These architectures typically want 10k+ samples; PEDS and GP are fallbacks that work with less data |
| PEDS requires too much engineering for the coarse solver wrapper | Low | Medium | The existing Firedrake solver can be re-run on a coarser mesh with minimal code changes |
| ISMO loop is expensive in PDE evaluations | Medium | Medium | Budget cap at 200 new PDE solves; use GP uncertainty to prioritize samples |
| Weight annealing destabilizes convergence | Low | Low | Test on synthetic problems first; can always fall back to current cascade approach |
| Firedrake environment issues block PDE solves | Low | High | Use existing venv-firedrake environment; test PDE solver before starting data generation |

---

## Resolved Decisions

1. **Data augmentation budget:** ~3,000 PDE solves overnight is feasible (previously demonstrated). If Phase 1 finds k0_2 gaps, we will run targeted augmentation at that scale.
2. **Hardware:** NVIDIA RTX 4070 available for GPU training. Use only for DeepONet (Phase 2b) and FNO (Phase 2c) if needed; all other models train on CPU.
3. **PCE surrogate:** Added as Phase 2f. Provides Sobol sensitivity indices to quantify k0_2's contribution to I-V curve variance.
4. **Bayesian inference:** Added as Phase 7 (after end-to-end validation). Calibrate-Emulate-Sample or Multi-Fidelity HMC.
5. **Tikhonov regularization:** Already tested extensively (WeekOfMar4 writeup). Prevents PDE regression but does not improve accuracy. The winning v11/v13 pipeline uses no regularization. Dropped from Phase 5 scope; focus on trust-region and weight annealing.
6. **Identifiability:** Assumed all 4 parameters are identifiable under reasonable conditions (user's assessment). No formal identifiability analysis phase included.

## Remaining Open Questions

1. **ISMO PDE budget per iteration:** How many new PDE solves per ISMO round? Depends on Phase 4 detailed planning.
2. **Weight annealing schedule:** Linear vs exponential vs adaptive — to be determined experimentally in Phase 5.
3. **Final pipeline architecture:** Single best surrogate or multi-fidelity cascade? Decided after Phase 3 benchmark.

## Environment Note

All phases should use the `venv-firedrake` virtual environment in the parent directory. New package installations (GPyTorch, ChaosPy, DeepXDE/NeuralOperator) should be installed into this environment.
