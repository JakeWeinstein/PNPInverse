# Research Summary: PDE-Constrained Inverse Problems with Surrogate Models

**Query:** Methods, architectures, and pipelines for surrogate-assisted PDE-constrained inverse problems, with focus on electrochemical PNP systems with Butler-Volmer kinetics.
**Date:** 2026-03-16
**Sources:** 4 research agents (codebase analysis, literature review, web research on PDE optimization, web research on surrogates)

## Executive Summary

The codebase already implements a complete multi-phase PDE-constrained inverse pipeline for electrochemical kinetics: LHS sampling, three surrogate types (RBF, POD-RBF, NN ensemble), multi-start/cascade/BCD optimization, and Firedrake adjoint-based PDE refinement. The literature and web research confirm this architecture aligns with the state of the art, but identify several high-value gaps: (1) no Bayesian uncertainty quantification beyond ensemble standard deviations, (2) surrogate gradients use finite differences rather than analytic backpropagation, (3) no active learning or adaptive sampling, (4) no identifiability analysis, and (5) no regularization in the objective. The most impactful near-term improvements are adding analytic NN gradients via PyTorch autograd, conducting structural identifiability analysis for the 4-parameter BV system, and implementing a Bayesian inference layer (GP emulator + MCMC or delayed-acceptance sampling).

---

## 1. PDE-Constrained Optimization Foundations

### 1.1 The Reduced-Space Paradigm

For problems with O(1)--O(10) parameters (like the 4D `[k0_1, k0_2, alpha_1, alpha_2]` space in this project), the reduced-space approach dominates: eliminate the PDE state by solving the forward problem at each optimization step, yielding an unconstrained problem over parameters alone. The gradient of the reduced functional is computed via the adjoint equation, requiring exactly one additional (adjoint) PDE solve regardless of parameter dimension.

The codebase implements this exactly via Firedrake's `ReducedFunctional` + `pyadjoint` tape-based automatic differentiation. L-BFGS-B through scipy is the universal optimizer at all stages.

**Full-space alternatives** (treating state, parameters, and adjoints as simultaneous unknowns in a KKT system) can be 5--10x faster with good preconditioners but require intrusive solver modifications. For the current 4-parameter problem, the reduced-space approach is more than sufficient. [Confidence: high; corroborated by Hinze et al. 2009, Heinkenschloss 2008 tutorial, and web-pde-optimization findings]

### 1.2 Continuous vs. Discrete Adjoint

The discrete adjoint (discretize-then-differentiate) provides the exact gradient of the discrete objective, which is critical for optimizer convergence. The continuous adjoint (differentiate-then-discretize) may produce gradients inconsistent with the discrete objective. Firedrake's pyadjoint implements the discrete adjoint by operator-overloading (taping the forward solve and replaying in reverse), making it the preferred choice.

| Aspect | Continuous Adjoint | Discrete Adjoint (pyadjoint) |
|--------|-------------------|------------------------------|
| Gradient accuracy | Approximates continuous gradient | Exact for discrete objective |
| Verification | Hard (continuous != discrete gradient) | Easy via Taylor test |
| Robustness | Can fail with discontinuities | Handles stiff/discontinuous problems |
| Implementation | Manual derivation possible | Automatic via AD tape |

**Codebase status:** The Taylor test is available via `firedrake.adjoint.taylor_test()` and gradient verification results are stored in `StudyResults/inverse_verification/gradient_fd_convergence.json` and `gradient_pde_consistency.json`. [Confidence: high]

### 1.3 Adjoint-Free Alternatives

When the adjoint is unavailable or impractical:

- **Ensemble Kalman Inversion (EKI):** Particle-based, derivative-free optimizer using ensemble covariance. Mathematically equivalent to Tikhonov regularization in regularized variants. Relevant for future Bayesian extensions.
- **Finite-difference gradients:** O(n) forward solves for n parameters. The codebase currently uses central differences (8 evaluations for 4D) for surrogate-based optimization. Viable for 4 parameters but does not scale.
- **Bayesian optimization:** GP-based global optimization. Effective for <20 parameters. Not currently implemented.

[Confidence: high; web-pde-optimization agent]

---

## 2. Surrogate Model Architectures

### 2.1 What the Codebase Has

The project implements three surrogate families, all sharing a uniform `predict(k0_1, k0_2, alpha_1, alpha_2)` / `predict_batch(parameters)` API:

| Model | Architecture | Training Data | Key Features | Status |
|-------|-------------|---------------|--------------|--------|
| `BVSurrogateModel` | Direct RBF (thin_plate_spline) via `scipy.interpolate.RBFInterpolator` | ~200 LHS samples | Baseline, fast to fit, exact/smoothed interpolation | Production |
| `PODRBFSurrogateModel` | SVD (99.9% variance) + per-mode RBF with LOO/k-fold CV smoothing optimization | Same | Better generalization, optional log1p PC transform | Production |
| `NNSurrogateModel` / `EnsembleMeanWrapper` | ResNet-MLP (4->128->4xResBlock->64->44), 5-member ensemble | Same, AdamW + cosine annealing, early stopping | Smooth, differentiable, uncertainty via ensemble std | Production |

Log-space k0 transformation is used everywhere (k0 spans 1e-7 to 1.0). Alpha values stay linear in [0.1, 0.9]. Z-score normalization applied to NN inputs and outputs.

Surrogate fidelity results are in `StudyResults/surrogate_fidelity/` with per-sample NRMSE comparisons across all model types.

### 2.2 Surrogate Architectures from the Literature

The research identified several additional surrogate architectures not currently in the codebase, organized by relevance:

**Highest relevance for this project:**

- **Gaussian Process (GP) surrogates:** Built-in uncertainty quantification via posterior variance. Excellent with small training sets (10s--100s of points). Ideal for the Calibrate-Emulate-Sample pipeline (Cleary et al. 2021). Scales to ~10,000 training points before O(N^3) cost becomes prohibitive. For the 4D parameter space with ~200 training points, GPs are a natural fit and would provide principled UQ that the NN ensemble currently approximates. Software: GPyTorch, GPflow. [Confidence: high; both literature and web-surrogates agents]

- **Polynomial Chaos Expansion (PCE):** Represents the parameter-to-observable map as orthogonal polynomial sum. Provides Sobol sensitivity indices directly from coefficients (valuable for identifiability). Spectral convergence for smooth maps. Sparse PCE handles moderate dimensions. Has been applied to vanadium redox flow batteries for parameter estimation. Software: ChaosPy, OpenTURNS. [Confidence: high; web-surrogates agent]

- **POD-DEIM (Discrete Empirical Interpolation Method):** Extension of the existing POD-RBF that handles nonlinear terms (like Butler-Volmer kinetics) efficiently by reducing the assembly cost from full-order to proportional to the number of reduced variables. Critical for reaction-diffusion systems. The codebase uses POD-RBF but not DEIM. Available via pyMOR. [Confidence: high; web-surrogates agent]

**Moderate relevance (higher data requirements, more complex):**

- **DeepONet (Deep Operator Networks):** Learns the solution operator via branch (input encoder) + trunk (output location) networks. Can evaluate at arbitrary spatial points. DeepBayONet variant achieved parameter recovery of 0.999 (true=1.0) with std 5.4e-4 on a 2D reaction-diffusion problem. More flexible than FNO for irregular geometries. [Confidence: high]

- **Fourier Neural Operator (FNO):** Resolution-invariant spectral learning. Excels for smooth, periodic solutions on regular grids. The Derivative-Informed FNO (DIFNO) variant trains on both outputs AND Frechet derivatives, which is critical for optimization -- standard surrogates can have accurate predictions but inaccurate gradients. [Confidence: high]

- **Physics-Enhanced Deep Surrogates (PEDS):** Embeds a coarse PDE solver as a differentiable layer within a neural network. Up to 3x more accurate than feedforward NN ensembles with limited data. Reduces training data need by 100x vs. black-box NNs. Requires a coarse solver, which could be the existing BV solver on a coarser mesh. [Confidence: high]

### 2.3 Surrogate Selection Decision Guide

For the specific characteristics of this project (4 parameters, ~200 training samples, BV nonlinearity, dual I-V curve outputs):

```
Current state: 4 parameters, ~200 training samples, need optimization + UQ

For optimization (point estimates):
  -> Current POD-RBF and NN ensemble are well-suited
  -> Add analytic NN gradients (PyTorch autograd) to replace FD gradients
  -> Consider DIFNO if scaling to higher parameter dimensions

For uncertainty quantification:
  -> GP surrogate is the natural next step (4D, 200 samples = sweet spot)
  -> PCE for sensitivity analysis / Sobol indices
  -> Delayed-acceptance MCMC with surrogate proposals for rigorous posteriors

For data efficiency (if training budget is limited):
  -> GP (best with <500 samples)
  -> PEDS (if coarse mesh solver available)
  -> POD-DEIM (extends existing POD-RBF infrastructure)
```

| Criterion | POD/RBF | GP | PCE | NN Ensemble | DeepONet | FNO | PEDS |
|---|---|---|---|---|---|---|---|
| Param dims suited | <10 | <15 | <15 | Any | Any | Any | Any |
| Samples needed | 10s-100s | 10s-100s | 10s-1000s | 1000s+ | 1000s+ | 1000s+ | 100s-1000s |
| Built-in UQ | No | Yes | Yes (Sobol) | No (ensemble) | No | No | Yes (variance net) |
| Derivative accuracy | FD only | Analytical | Analytical | Backprop | Backprop | DIFNO | Backprop |
| Typical speedup | 100-1000x | 10-100x | 100-1000x | 1000-10000x | 1000x+ | 1000x+ | 30-100x |
| Maturity | Very high | High | High | High | Medium | Medium | Low-Medium |

[Confidence: high; synthesized across all four agents]

---

## 3. Multi-Phase Inference Pipelines

### 3.1 Current Pipeline Architecture

The codebase implements a well-structured multi-phase cascade:

**Phase 1: Surrogate-based coarse search**
- LHS sampling in 4D parameter space -> PDE forward solves -> training pairs
- Train surrogates (RBF, POD-RBF, NN ensemble)
- Multi-start: 20,000 LHS points -> batch surrogate eval -> top-20 L-BFGS-B polish

**Phase 2: Surrogate refinement with observable weighting**
- Cascade inference exploits key finding: low `secondary_weight` (CD-dominant) recovers k0_1/alpha_1; high weight (PC-dominant) recovers k0_2
- Three strategies: cascade (3-pass sequential), BCD (alternating 2D blocks), multi-start

**Phase 3: PDE-based refinement**
- Takes surrogate-optimized parameters as initial guess
- Firedrake adjoint (`ReducedFunctional`) with L-BFGS-B
- Resilient minimization with up to 15 recovery attempts (max_it escalation -> anisotropy reduction -> tolerance relaxation -> line search cycling)

### 3.2 State-of-the-Art Multi-Phase Approaches from the Literature

The literature strongly validates the multi-phase approach and suggests several enhancements:

**Calibrate-Emulate-Sample (Cleary et al. 2021)** -- The most directly applicable pipeline template:
1. **Calibrate:** Use ensemble Kalman sampling (derivative-free, parallelizable) for approximate parameter estimates
2. **Emulate:** Build GP emulator of the parameter-to-data map using calibration samples
3. **Sample:** Run MCMC on the emulated posterior for full Bayesian inference

This maps naturally onto the existing pipeline: multi-start surrogate optimization (calibrate) -> GP construction from optimization samples (emulate) -> MCMC for posterior (sample). [Confidence: high; literature agent]

**ISMO (Iterative Surrogate Model Optimization, Lye et al. 2020):** Alternates between training surrogate, optimizing surrogate, evaluating candidates with true PDE, and augmenting training data. Converges exponentially with respect to training samples. Significantly outperforms one-shot surrogate optimization (the current approach). This is the active learning strategy the codebase is missing. [Confidence: high; web-pde-optimization agent]

**Multi-Fidelity Trust-Region (arXiv 2503.21252, 2025):** Three fidelity levels (ML surrogate -> reduced basis -> full FEM) with active learning enrichment during optimization. Trust-region methods ensure local accuracy, avoiding expensive offline training. Provides convergence guarantees to the full-model optimum. [Confidence: high; both literature and web-pde-optimization agents]

**Adaptive Reduced Basis Trust Region (arXiv 2309.07627, 2023):** Certified a posteriori error bounds for the reduced model drive trust-region acceptance/rejection. Achieves order-of-magnitude speedup over full-model optimization while maintaining convergence guarantees. Could be applied to the existing POD-RBF surrogate. [Confidence: high; literature agent]

**Physics-Based Deep Kernel Learning + HMC (arXiv 2509.14054, 2025):** Two-stage Bayesian framework: Stage 1 trains a DKL surrogate with physics regularization. Stage 2 fixes NN weights and runs HMC for full posterior sampling. Directly implements "surrogate warm-start -> Bayesian refinement." [Confidence: high; literature agent]

**Multi-Fidelity HMC (arXiv 2405.05033, 2024):** Uses surrogate for HMC proposal generation (cheap gradients), PDE solver for accept/reject (exact posterior). Achieves exact posterior sampling with significant computational savings via delayed acceptance. [Confidence: high; literature agent]

**PINN Surrogate for Li-Ion Battery Models (Ramirez & Latz 2024):** Most directly relevant existing work -- applies surrogate-based inference to electrochemical PDE models with Butler-Volmer kinetics. Multi-fidelity PINN hierarchy (coarse -> medium -> fine physics loss) improves surrogate accuracy. Maps directly to the POD-RBF -> NN -> PDE pipeline. [Confidence: high; literature agent]

### 3.3 Handling Surrogate Bias in the Inference Loop

Using an approximate surrogate introduces bias in parameter estimates and posterior distributions. The research identified several proven strategies:

- **Delayed-acceptance MCMC:** Surrogate proposes, PDE verifies. Maintains exact posterior despite surrogate error. ~5x speedup over standard MCMC. [Confidence: high]
- **Learned noise variance (DeepBayONet):** Model surrogate-data mismatch through learnable residual noise, inflating posterior uncertainty appropriately. [Confidence: high]
- **Rigorous convergence bounds:** If surrogate approximation converges at rate r in prior-weighted L2 norm, KL divergence between approximate and true posteriors converges at rate 2r. [Confidence: high]
- **Coarse-solver + neural residual correction (PEDS):** Learns systematic bias pattern of the coarse solver. [Confidence: high]

---

## 4. Identifiability: A Critical Gap

Multiple sources emphasize that structural and practical identifiability analysis is a prerequisite for parameter estimation in complex PDE models with Butler-Volmer kinetics. This is identified as a gap by the codebase agent and strongly reinforced by the literature.

**Key findings:**

- Bizeray et al. (2020) showed that for the P2D electrochemical model with BV kinetics, many individual parameters are structurally unidentifiable from standard charge/discharge data. Only certain parameter combinations can be recovered.
- A 2025 identifiability analysis of the P2D model (arXiv 2507.13931) demonstrates an SPM-aided cascade: use simplified models to constrain the identifiable parameter space before full estimation. This is directly analogous to the surrogate -> PDE cascade.
- The codebase's multi-observable approach (fitting both current density and peroxide current simultaneously) likely improves identifiability compared to single-observable fitting, but this has not been formally analyzed.

**Implication:** Before investing in more sophisticated inference methods, a formal identifiability analysis for the 4-parameter `[k0_1, k0_2, alpha_1, alpha_2]` system given dual I-V curve observations would determine whether all four parameters are individually recoverable or only certain combinations. The existing `scripts/studies/profile_likelihood_study.py` and `scripts/studies/sensitivity_visualization.py` provide partial practical identifiability information, but structural identifiability analysis is missing. [Confidence: high; literature agent]

---

## 5. Software Frameworks

### 5.1 Current Stack

The project uses Firedrake + pyadjoint (adjoint-based PDE optimization), SciPy (L-BFGS-B optimization, RBF interpolation, LHS sampling), PyTorch (NN surrogate), NumPy/h5py/Matplotlib (data handling/visualization).

### 5.2 Framework Comparison

| Feature | Firedrake+pyadjoint | PETSc TAO | ROL (Trilinos) | ADCME | JAX-FEM |
|---------|---------------------|-----------|-----------------|-------|---------|
| Language | Python (UFL) | C/Fortran/Python | C++ | Julia/Python | Python |
| Adjoint method | Automatic (tape) | User-provided | User-provided | Automatic (TF) | Automatic (JAX) |
| PDE discretization | Built-in FEM | External | PDE-OPT kit | External | Built-in |
| Parallel scale | MPI via PETSc | MPI native | MPI via Tpetra | GPU via TF | GPU via JAX |
| Stochastic opt | No | No | Yes (CVaR, SAA) | No | No |
| Ease of use | High | Medium | Low | Medium | Medium |

**Recommendation:** The current Firedrake + pyadjoint stack is the right choice for this project. PETSc TAO is accessible through Firedrake's backend for additional optimizer options (`tao_ntr` for trust-region Newton, `tao_lcl` for constrained optimization). No framework change is warranted. [Confidence: high; corroborated by web-pde-optimization agent]

### 5.3 Additional Software for Extensions

- **Bayesian inference:** PyMC, emcee, or TensorFlow Probability for MCMC sampling
- **GP surrogates:** GPyTorch (PyTorch-native, integrates with existing NN infrastructure) or GPflow
- **PCE / sensitivity analysis:** ChaosPy or OpenTURNS (Python-native)
- **POD-DEIM:** pyMOR (Python model order reduction library)
- **Neural operators:** DeepXDE, NeuralOperator (PyTorch), NVIDIA PhysicsNeMo

---

## Contradictions and Open Questions

### Contradictions

1. **Training data requirements for NNs.** The web-surrogates agent reports NNs need "thousands to tens of thousands" of forward solves, but the codebase trains a 5-member NN ensemble on ~200 samples with competitive fidelity. This discrepancy likely reflects the low dimensionality (4D input, 44D output) of the current problem, where NNs are data-efficient. Scaling to higher parameter dimensions would likely require substantially more training data.

2. **GP vs. NN ensemble for UQ.** The literature recommends GPs for principled UQ, while the codebase uses NN ensembles. Both are valid: GPs provide calibrated posterior variance, while NN ensembles provide empirical spread. For 4 parameters and ~200 samples, GPs would likely provide better-calibrated uncertainty, but the NN ensemble's UQ is pragmatically useful for the current optimization pipeline.

### Open Questions

1. **PNP-specific identifiability:** No existing work performs structural identifiability analysis specifically for the Poisson-Nernst-Planck system with Butler-Volmer boundary conditions. The battery P2D literature provides analogies but not direct results. Can all four parameters `[k0_1, k0_2, alpha_1, alpha_2]` be individually identified from dual I-V curves?

2. **Surrogate fidelity for multi-observable inference:** When fitting current density and peroxide current simultaneously, how should surrogate training be weighted across observables? The cascade approach addresses this empirically, but no theoretical framework exists.

3. **Active learning sample allocation:** How many surrogate evaluations vs. PDE evaluations should be allocated at each stage for optimal computational efficiency? ISMO provides exponential convergence but no practical guidance on allocation.

4. **Neural operator scalability for stiff PNP systems:** The PNP system with thin double layers creates boundary layer structures that challenge neural operators. No existing benchmark evaluates neural operators specifically for PNP-type stiff systems.

5. **Transfer learning across experimental conditions:** Can a surrogate trained for one voltage range or electrolyte composition transfer to adjacent conditions?

6. **Model discrepancy:** When the PDE model itself is imperfect, how should the inference pipeline account for structural model error?

---

## Codebase Relevance

The codebase implements a mature and well-structured pipeline that aligns with current best practices. Key mapping between literature recommendations and existing code:

| Literature Recommendation | Codebase Status | Location |
|--------------------------|-----------------|----------|
| Reduced-space adjoint optimization | Implemented | `Inverse/inference_runner/`, `FluxCurve/` |
| POD-RBF surrogate | Implemented | `Surrogate/pod_rbf_model.py` |
| NN ensemble surrogate | Implemented | `Surrogate/nn_model.py`, `Surrogate/ensemble.py` |
| Multi-start global search | Implemented | `Surrogate/multistart.py` |
| Observable-weighted cascade | Implemented | `Surrogate/cascade.py`, `Surrogate/bcd.py` |
| Surrogate-to-PDE refinement handoff | Implemented | `FluxCurve/bv_run/pipelines.py` |
| Resilient minimization with recovery | Implemented | `Inverse/inference_runner/recovery.py` |
| Log-space parameter transforms | Implemented | Throughout `Surrogate/`, `Forward/` |
| LHS sampling with focused regions | Implemented | `Surrogate/sampling.py` |
| Gradient verification (Taylor test) | Available | Via `firedrake.adjoint.taylor_test()` |
| Bayesian UQ (MCMC, HMC) | **Not implemented** | -- |
| Analytic surrogate gradients (autograd) | **Not implemented** | NN uses FD instead of PyTorch autograd |
| Active learning / ISMO | **Not implemented** | Fixed LHS design only |
| Identifiability analysis | **Partial** | Profile likelihood scripts exist |
| GP surrogate | **Not implemented** | -- |
| Regularization (Tikhonov, TV) | **Not implemented** | Pure data misfit objective |
| Trust-region methods | **Not implemented** | L-BFGS-B only |
| Multi-fidelity hierarchy | **Not implemented** | Single mesh resolution |
| POD-DEIM for nonlinear terms | **Not implemented** | POD-RBF without DEIM |

---

## Recommendations

Ordered by estimated impact-to-effort ratio:

### High Priority (Low-Medium Effort, High Impact)

1. **Add analytic NN surrogate gradients via PyTorch autograd.** The NN ensemble already uses PyTorch. Replacing the 8-evaluation central-difference gradient with `torch.autograd` is straightforward and eliminates the dominant cost in surrogate-based L-BFGS-B. This also enables second-order information (Hessian-vector products) for Newton-CG or Laplace approximation UQ.

2. **Conduct identifiability analysis.** Use profile likelihood (scripts already exist in `scripts/studies/`) systematically for all four parameters, and consider Fisher information matrix analysis at the optimal point. Before investing in Bayesian methods, confirm that the 4-parameter system is identifiable from dual I-V data.

3. **Add Tikhonov regularization to the objective.** The current pure data-misfit objective (`J = 0.5*||sim - target||^2`) is susceptible to overfitting noise and ill-conditioning. Adding a regularization term (`+ lambda/2 * ||m - m_prior||^2`) is a one-line change that can significantly improve robustness.

### Medium Priority (Medium Effort, High Impact)

4. **Implement a GP surrogate using GPyTorch.** For 4D input and ~200 training samples, a GP is in its sweet spot. Provides calibrated uncertainty estimates, enables Bayesian optimization for active learning, and is the natural input to the Calibrate-Emulate-Sample pipeline.

5. **Implement ISMO-style active learning.** Instead of fixed LHS training data, iterate: train surrogate -> optimize -> evaluate candidates with PDE -> augment training data -> repeat. Literature shows exponential convergence improvement over one-shot surrogate training.

6. **Add delayed-acceptance MCMC for Bayesian inference.** Use the NN ensemble or GP as the proposal model, PDE solver for accept/reject. Provides rigorous posterior distributions with manageable computational cost. PyMC or emcee can serve as the MCMC engine.

### Lower Priority (Higher Effort or More Speculative)

7. **Implement trust-region methods for surrogate-to-PDE transitions.** Replace the current direct handoff with a trust-region framework where the surrogate model validity controls the step size, following the adaptive reduced basis trust-region approach (arXiv 2309.07627).

8. **Explore PEDS architecture.** Use the existing BV solver on a coarser mesh as the embedded physics layer, with a neural network learning the coarse-to-fine correction. Literature suggests 100x data efficiency improvement over black-box NNs.

9. **Scale to higher parameter dimensions.** The current 4-parameter pipeline is specialized. Extending to diffusivities, steric coefficients, etc. would benefit from the neural operator approaches (DeepONet, FNO/DIFNO) that handle higher-dimensional parameter spaces.

10. **Implement sensitivity-based experimental design.** Use PCE Sobol indices or Fisher information to identify voltage ranges and measurement configurations that maximize parameter identifiability, before running physical experiments.

---

## Sources

### Foundational Texts
- Hinze, Pinnau, Ulbrich, Ulbrich (2009). *Optimization with PDE Constraints*. Springer. [Textbook]
- Heinkenschloss (2008). *PDE Constrained Optimization*. SIAM Tutorial. [Tutorial]

### Adjoint Methods and PDE-Constrained Optimization
- Bradley, A. *Adjoint tutorial*. Stanford CS. https://cs.stanford.edu/~ambrad/adjoint_tutorial.pdf
- dolfin-adjoint documentation. https://www.dolfin-adjoint.org/
- Firedrake adjoint documentation. https://www.firedrakeproject.org/adjoint.html
- Biros & Ghattas. *Parallel Lagrange-Newton-Krylov-Schur*. http://www.aladdin.cs.cmu.edu/papers/pdfs/y2000/new_lagr_kryl.pdf
- arXiv 2601.10920 (2025). *Variational State-Dependent Inverse Problems Survey*.
- arXiv 2209.03270 (2022). *Comprehensive Study of Adjoint-Based Optimization of Non-Linear Systems*.

### Surrogate-Assisted Inverse Problems
- Cleary et al. (2021). *Calibrate, Emulate, Sample*. J. Comput. Phys. arXiv 2001.03689
- Molinaro et al. (2023). *Neural Inverse Operators*. ICML. arXiv 2301.11167
- Li et al. (2024). *DINO for PDE-Constrained Optimization*. SIAM J. Sci. Comput.
- Boyce & Yeh (2012). *POD for Inverse Problems in Groundwater Flow*. Adv. Water Res.
- arXiv 2501.10684 (2025). *Deep Operator Networks for Bayesian Parameter Estimation in PDEs*.
- arXiv 2512.14086 (2024). *Derivative-Informed FNO for PDE-Constrained Optimization*.

### Multi-Phase Pipelines
- Lye, Mishra, Ray (2020). *ISMO: Iterative Surrogate Model Optimization*. arXiv 2008.05730
- arXiv 2309.07627 (2023). *Adaptive Reduced Basis Trust Region Methods*.
- arXiv 2503.21252 (2025). *Multi-fidelity Learning of ROMs for Parabolic PDE Optimization*.
- arXiv 2405.05033 (2024). *Multi-Fidelity Hamiltonian Monte Carlo*.
- arXiv 2509.14054 (2025). *Physics-Based Deep Kernel Learning for Parameter Inference*.

### Electrochemistry-Specific
- Ramirez & Latz (2024). *PINN Surrogate of Li-Ion Battery Models*, Parts I & II. arXiv 2312.17329, 2312.17336
- Bizeray et al. (2020). *Structural Identifiability of P2D Model*. IFAC. arXiv 2012.01853
- arXiv 2507.13931 (2025). *Identifiability Analysis of P2D Model*.
- arXiv 2412.13200 (2025). *Forward and Inverse Simulation of P2D Model Using Neural Networks*.
- Jasielec et al. (2012). *EIS of Ion Sensors: NPP Model and HGS*. Electrochimica Acta.

### Surrogate Architectures
- Chaturantabut & Sorensen (2010). *Nonlinear Model Reduction via DEIM*. SIAM J. Sci. Comput.
- Meng & Karniadakis (2021). *Multi-Fidelity Bayesian Neural Networks*. J. Comput. Phys.
- *Physics-Enhanced Deep Surrogates* (2023). Nature Machine Intelligence.
- arXiv 2511.04576 (2025). *PINNs and Neural Operators Survey*.

### Codebase Files
- `Forward/bv_solver/` -- PNP-BV forward solver (forms, solvers, config, mesh)
- `Surrogate/` -- All surrogate models, objectives, training, sampling, cascade, BCD, multistart
- `Inverse/` -- Adjoint-based inference pipeline (objectives, solver interface, recovery)
- `FluxCurve/` -- End-to-end BV inference with adjoint gradients
- `StudyResults/` -- Surrogate fidelity, inverse verification, MMS convergence, inference results
- `scripts/studies/profile_likelihood_study.py` -- Existing identifiability analysis
