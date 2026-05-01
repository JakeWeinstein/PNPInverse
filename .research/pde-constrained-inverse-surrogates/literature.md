# Literature Review: PDE-Constrained Optimization, Surrogate-Assisted Inverse Problems, and Multi-Phase Inference Pipelines

---

## 1. PDE-Constrained Optimization Theory and Algorithms

### Key Papers

#### Optimization with PDE Constraints (2009)
- **Authors:** Hinze, M., Pinnau, R., Ulbrich, M., Ulbrich, S.
- **Venue:** Springer, Mathematical Modelling: Theory and Applications, Vol. 23
- **URL:** https://link.springer.com/book/10.1007/978-1-4020-8839-1
- **Key Contribution:** Foundational textbook covering the analytical background and optimality theory for PDE-constrained optimization. Introduces functional analytic techniques, function space theory, existence/uniqueness results, and derives first-order optimality conditions (KKT systems) for PDE-constrained problems. Covers both reduced-space (eliminate state via solve) and full-space (simultaneous state+control) formulations.
- **Methodology:** Lagrangian-based derivation of adjoint equations; discuss-then-optimize vs. optimize-then-discretize paradigms; SQP and trust-region methods for the resulting nonlinear programs.
- **Relevance:** The canonical reference for the mathematical foundations underlying any PDE-constrained inverse problem. Directly applicable to formulating the PNP parameter identification as a constrained optimization.
- **Confidence:** high

#### Variational State-Dependent Inverse Problems in PDE-Constrained Optimization: A Survey (2025)
- **Authors:** (Multiple, see arXiv)
- **Venue:** arXiv preprint 2601.10920
- **URL:** https://arxiv.org/html/2601.10920
- **Key Contribution:** Comprehensive survey of PDE-constrained optimization approaches for inverse problems since 2011. Covers variational formulations, adjoint-based gradient methods, regularization strategies (Tikhonov, total variation, sparsity-promoting), and modern computational frameworks. Emphasizes identifiability, ill-posedness, and structural limits of state-dependent inverse problems.
- **Methodology:** Reviews reduced-space vs. full-space approaches, adjoint derivations, regularization selection criteria, and computational scaling strategies.
- **Relevance:** Directly addresses the theory needed for PNP inverse problems -- particularly the discussion of identifiability limitations when parameters appear nonlinearly in the PDE.
- **Confidence:** high

#### PDE Constrained Optimization (2008, SIAM Tutorial)
- **Authors:** Heinkenschloss, M.
- **Venue:** SIAM Conference on Optimization
- **URL:** https://archive.siam.org/meetings/op08/Heinkenschloss.pdf
- **Key Contribution:** Tutorial overview of reduced-space vs. full-space approaches. In the reduced-space approach, the PDE constraint is eliminated by solving the state equation for each parameter evaluation, yielding an unconstrained problem in the parameter space alone -- computationally simpler but requires repeated PDE solves. The full-space approach treats states, parameters, and adjoints as independent variables in a large coupled system, enabling exploitation of sparsity structure. Trust-region globalization strategies are discussed for both.
- **Methodology:** Compares Newton-based methods (SQP, reduced Hessian) with gradient-only methods (steepest descent, L-BFGS, nonlinear CG) in the context of PDE constraints.
- **Relevance:** Provides practical guidance on algorithm selection -- for moderate parameter dimensions (as in Butler-Volmer kinetics), reduced-space with L-BFGS is often preferred; for distributed parameter fields, full-space methods may be necessary.
- **Confidence:** high

#### Adjoint-Based Enforcement of State Constraints in PDE Optimization Problems (2024)
- **Authors:** (See ScienceDirect)
- **Venue:** Journal of Computational Physics
- **URL:** https://www.sciencedirect.com/science/article/pii/S0021999124005461
- **Key Contribution:** Extends adjoint-based gradient computation to handle additional constraints on the PDE state variables (e.g., positivity of concentrations, bounds on potential). This is critical for electrochemical problems where physical constraints must be enforced.
- **Methodology:** Augmented Lagrangian and penalty methods combined with adjoint-based gradients.
- **Relevance:** Directly applicable to PNP systems where ion concentrations must remain positive and potentials bounded.
- **Confidence:** medium

#### A Comprehensive Study of Adjoint-Based Optimization of Non-Linear Systems (2022)
- **Authors:** (See arXiv)
- **Venue:** arXiv preprint 2209.03270
- **URL:** https://arxiv.org/pdf/2209.03270
- **Key Contribution:** Detailed treatment of adjoint methods for nonlinear PDE systems, including time-dependent problems. Addresses practical implementation challenges: checkpointing for adjoint solves, handling nonlinear iterations in the forward/adjoint coupling, and convergence of the overall optimization.
- **Methodology:** Continuous vs. discrete adjoint comparison; automatic differentiation vs. hand-derived adjoints.
- **Relevance:** Implementation guidance for building adjoint solvers for nonlinear reaction-diffusion systems like PNP with Butler-Volmer boundary conditions.
- **Confidence:** medium

#### Adaptive Reduced Basis Trust Region Methods for Parameter Identification Problems (2023)
- **Authors:** (See arXiv)
- **Venue:** arXiv preprint 2309.07627
- **URL:** https://arxiv.org/html/2309.07627
- **Key Contribution:** Combines certified reduced basis models with trust-region optimization for PDE parameter identification. The reduced model serves as a surrogate within the trust region, with error bounds controlling the trust region radius. Achieves order-of-magnitude speedup over full-model optimization while maintaining convergence guarantees.
- **Methodology:** Certified a posteriori error bounds for the reduced model drive trust-region acceptance/rejection; adaptive enrichment of the reduced basis during optimization.
- **Relevance:** Provides a principled framework for using surrogate models within optimization that could be applied to PNP inverse problems -- the trust-region mechanism automatically manages surrogate fidelity.
- **Confidence:** high

#### Efficient PDE-Constrained Optimization Under High-Dimensional Uncertainty Using Derivative-Informed Neural Operators (2024)
- **Authors:** Li, Z., O'Leary-Roseberry, T., Chen, P., Ghattas, O.
- **Venue:** SIAM Journal on Scientific Computing
- **URL:** https://epubs.siam.org/doi/abs/10.1137/23M157956X
- **Key Contribution:** Introduces the multi-input reduced basis derivative-informed neural operator (MR-DINO) for optimization under uncertainty governed by large-scale PDEs. The key innovation is training the neural operator to approximate both the parameter-to-state map AND its derivative with respect to optimization variables, enabling accurate gradient-based optimization without repeated PDE solves.
- **Methodology:** Reduced basis architecture combined with derivative-informed training loss; achieves 3-8 orders of magnitude speedup in state and gradient evaluations.
- **Relevance:** State-of-the-art approach for combining neural operator surrogates with PDE-constrained optimization. Directly applicable to replacing expensive PNP forward solves with a trained surrogate during optimization.
- **Confidence:** high

#### Multi-fidelity Learning of Reduced Order Models for Parabolic PDE Constrained Optimization (2025)
- **Authors:** (See arXiv)
- **Venue:** arXiv preprint 2503.21252
- **URL:** https://arxiv.org/abs/2503.21252
- **Key Contribution:** Hierarchical trust-region algorithm using an active learning enrichment strategy to construct multi-fidelity reduced order models on-the-fly during optimization. Overcomes the traditional offline/online splitting limitation of model reduction for PDE-constrained optimization.
- **Methodology:** Combines certified model reduction with machine learning in an adaptive trust-region framework; error certification guarantees convergence to the full-model optimum.
- **Relevance:** Provides a principled multi-fidelity framework that could incorporate both POD-RBF and neural operator surrogates within a convergent optimization algorithm for PNP parameter identification.
- **Confidence:** high

---

## 2. Surrogate-Assisted PDE Inverse Problems

### Key Papers

#### Neural Inverse Operators for Solving PDE Inverse Problems (2023)
- **Authors:** Molinaro, R., Yang, Y., Engquist, B., Mishra, S.
- **Venue:** ICML 2023 / arXiv 2301.11167
- **URL:** https://arxiv.org/abs/2301.11167
- **Key Contribution:** Introduces Neural Inverse Operators (NIOs), a novel architecture based on composition of DeepONets and FNOs that directly maps from observations to PDE parameters. Unlike forward-surrogate + optimization approaches, NIOs learn the inverse map end-to-end, amortizing the cost of inversion across many problem instances.
- **Methodology:** Composition architecture: DeepONet encodes observation data, FNO processes in spectral domain; trained on pairs of (observations, ground-truth parameters).
- **Relevance:** Represents the direct-inversion paradigm -- an alternative to the optimize-with-surrogate approach. Could provide fast initial estimates for PNP parameters that are then refined.
- **Confidence:** high

#### Surrogate Modeling for Bayesian Inverse Problems Based on Physics-Informed Neural Networks (2022)
- **Authors:** Li, Z., Peng, Z., et al.
- **Venue:** Journal of Computational Physics (2022)
- **URL:** https://www.sciencedirect.com/science/article/abs/pii/S0021999122009044
- **Key Contribution:** Uses offline-trained PINNs as low-fidelity emulators to accelerate MCMC sampling in Bayesian inversion. The PINN surrogate is fine-tuned locally based on the current MCMC sample position, ensuring accuracy in the neighborhood being explored. Demonstrates significant speedup (10-100x) over full-model MCMC.
- **Methodology:** Two-stage: (1) offline PINN training on parameter space, (2) online MCMC with local PINN refinement. Delayed acceptance MCMC filters out unlikely proposals cheaply.
- **Relevance:** Directly applicable pipeline: train a PINN/surrogate for the PNP forward map offline, then use it to accelerate Bayesian inference for kinetic parameters.
- **Confidence:** high

#### Deep Operator Networks for Bayesian Parameter Estimation in PDEs (2025)
- **Authors:** (See arXiv)
- **Venue:** arXiv preprint 2501.10684
- **URL:** https://arxiv.org/html/2501.10684
- **Key Contribution:** Demonstrates use of DeepONet as a surrogate forward model within Bayesian parameter estimation, replacing expensive PDE solves in the likelihood evaluation. Compares vanilla DeepONet vs. physics-informed variants for accuracy of posterior distributions.
- **Methodology:** DeepONet trained on forward PDE solutions; embedded in MCMC/ensemble Kalman methods for posterior sampling.
- **Relevance:** Closest existing methodology to the POD-RBF and neural network surrogates already built for PNP forward problems. Provides comparison framework.
- **Confidence:** medium

#### Inverse Parameter Estimation Using Compressed Sensing and POD-RBF Reduced Order Models (2024)
- **Authors:** (See ScienceDirect)
- **Venue:** Computer Methods in Applied Mechanics and Engineering
- **URL:** https://www.sciencedirect.com/science/article/abs/pii/S0045782524000768
- **Key Contribution:** Combines POD-RBF surrogate models with compressed sensing for parameter identification from sparse observations. The POD-RBF model provides fast forward evaluations while compressed sensing handles the sparse measurement regime.
- **Methodology:** POD for dimensionality reduction of the solution manifold; RBF interpolation in parameter space; L1-regularized optimization for sparse recovery.
- **Relevance:** Directly relevant to the existing POD-RBF surrogate infrastructure in this codebase. Validates POD-RBF as a competitive surrogate for inverse problems.
- **Confidence:** high

#### Application of Proper Orthogonal Decomposition (POD) to Inverse Problems in Saturated Groundwater Flow (2012)
- **Authors:** Boyce, S.E., Yeh, W.W.-G.
- **Venue:** Advances in Water Resources
- **URL:** https://www.sciencedirect.com/science/article/abs/pii/S0309170811001746
- **Key Contribution:** Demonstrates that POD-based surrogates reduce inverse problem solve time by at least an order of magnitude while maintaining accuracy. Shows how POD basis selection affects inverse problem solution quality.
- **Methodology:** POD snapshots from forward model; Galerkin projection for surrogate; gradient-based optimization with surrogate-in-the-loop.
- **Relevance:** Classic demonstration of POD surrogates for PDE inverse problems. The groundwater flow setting is analogous to ion transport (diffusion-dominated with heterogeneous coefficients).
- **Confidence:** high

#### Parameter Inference Based on Gaussian Processes Informed by Nonlinear PDEs (2023)
- **Authors:** (See SIAM/ASA JUQ)
- **Venue:** SIAM/ASA Journal on Uncertainty Quantification
- **URL:** https://doi.org/10.1137/22m1514131
- **Key Contribution:** PDE-informed Gaussian process (PIGP) method that models the PDE solution as a GP and derives manifold constraints from the PDE structure. For nonlinear PDEs, an augmentation method transforms the nonlinear PDE into an equivalent system linear in all derivatives. Provides joint uncertainty quantification for parameters and PDE solution.
- **Methodology:** GP prior on solution; PDE constraints as likelihood terms; augmentation for nonlinear PDEs; MCMC for posterior sampling.
- **Relevance:** Could be applied to PNP systems by encoding the PDE structure into the GP prior, potentially offering better uncertainty quantification than black-box surrogates.
- **Confidence:** medium

#### Calibrate, Emulate, Sample (2021)
- **Authors:** Cleary, E., Garbuno-Inigo, A., Lan, S., Schneider, T., Stuart, A.M.
- **Venue:** Journal of Computational Physics
- **URL:** https://arxiv.org/abs/2001.03689
- **Key Contribution:** Three-stage pipeline: (1) Calibrate -- use ensemble Kalman sampling to find approximate parameter estimates; (2) Emulate -- build GP emulator of the parameter-to-data map using calibration samples; (3) Sample -- run MCMC on the emulated posterior. Designed for expensive forward models where derivatives are unavailable.
- **Methodology:** EKS for initial calibration (derivative-free, parallelizable); GP emulation trained on EKS output; emulation-based MCMC for full Bayesian posterior.
- **Relevance:** Highly relevant as a blueprint for the multi-phase inference pipeline: surrogate warm-start (calibrate) -> emulator construction -> full Bayesian sampling. Directly applicable to PNP parameter inference.
- **Confidence:** high

#### Physics-Informed Neural Networks and Neural Operators for Parametric PDEs: A Collaborative Survey (2025)
- **Authors:** (ICAIS 2025 submission)
- **Venue:** arXiv preprint 2511.04576
- **URL:** https://arxiv.org/html/2511.04576v1
- **Key Contribution:** Comprehensive survey comparing PINNs, DeepONet, FNO, and their variants for parametric PDE solving. Reports: Separable Physics-Informed DeepONet achieves 100x training speedup (289 hours -> 2.5 hours for 4D heat equation); PAR-DeepONet with physical adaptive refinement achieves up to 71.3% accuracy improvement; FNO excels at dissipative systems due to spectral-domain learning.
- **Methodology:** Systematic comparison of architectures, training strategies, and application domains.
- **Relevance:** Provides guidance on architecture selection for building PNP surrogates. FNO may be preferred for the diffusion-dominated PNP system; DeepONet variants offer more flexibility for multi-physics coupling.
- **Confidence:** high

#### Latent Neural Operator for Solving Forward and Inverse PDE Problems (2024)
- **Authors:** (See arXiv)
- **Venue:** arXiv preprint 2406.03923
- **URL:** https://arxiv.org/html/2406.03923v2
- **Key Contribution:** Proposes latent neural operators that work in a compressed latent space, reducing both training data requirements and computational cost. Demonstrates simultaneous forward and inverse problem solving capability.
- **Methodology:** Autoencoder for latent space compression; neural operator in latent space; inverse problems solved via optimization in latent space.
- **Relevance:** Addresses the data efficiency challenge -- PNP forward solves are expensive, so a latent-space approach requiring fewer training samples is attractive.
- **Confidence:** medium

#### Multi-fidelity Gaussian Process Surrogate Modeling for Regression Problems in Physics (2024)
- **Authors:** (See arXiv)
- **Venue:** arXiv preprint 2404.11965
- **URL:** https://arxiv.org/html/2404.11965v1
- **Key Contribution:** Comprehensive framework for multi-fidelity GP surrogate modeling, employing hierarchical Kriging and Bayesian inference. Handles heterogeneous multi-fidelity datasets where different fidelity levels have different output dimensions or noise characteristics.
- **Methodology:** Hierarchical Kriging with autoregressive correlation structure; Bayesian hyperparameter estimation; active learning for sample allocation across fidelities.
- **Relevance:** Provides the theoretical framework for combining cheap surrogate evaluations (POD-RBF, neural network) with expensive PDE solves in a principled multi-fidelity emulator.
- **Confidence:** high

---

## 3. Multi-Phase/Cascade Inference Pipelines for PDE Parameter Estimation

### Key Papers

#### Physics-Based Deep Kernel Learning for Parameter Inference in High-Dimensional PDEs (2025)
- **Authors:** (See arXiv)
- **Venue:** arXiv preprint 2509.14054
- **URL:** https://arxiv.org/html/2509.14054
- **Key Contribution:** Two-stage Bayesian framework: Stage 1 uses physics-based deep kernel learning (DKL) to train a surrogate and obtain initial parameter estimates. Stage 2 fixes the neural network weights and uses Hamiltonian Monte Carlo for full Bayesian posterior sampling of PDE parameters and kernel hyperparameters.
- **Methodology:** DKL combines neural network feature extraction with GP regression; physics loss regularizes the neural network; HMC provides rigorous posterior sampling in the second stage.
- **Relevance:** Directly implements the "surrogate warm-start -> Bayesian refinement" pipeline. The two-stage separation is exactly the pattern needed for PNP inference.
- **Confidence:** high

#### A Two-Step Surrogate Method for Sequential Uncertainty Quantification in High-Dimensional Inverse Problems (2024)
- **Authors:** (See arXiv)
- **Venue:** arXiv preprint 2407.11600
- **URL:** https://arxiv.org/html/2407.11600
- **Key Contribution:** Two-step approach: first recovers coarse-scale features of the parameter field using a computationally cheap surrogate, then refines via neural network correction. Handles high-dimensional parameter fields where traditional MCMC is intractable.
- **Methodology:** Coarse-to-fine estimation: surrogate-based MAP estimate -> neural network residual correction -> uncertainty quantification via linearization or sampling.
- **Relevance:** Demonstrates the cascade approach for going from surrogate-quality estimates to PDE-quality estimates, exactly the multi-phase pipeline concept.
- **Confidence:** high

#### PINN Surrogate of Li-Ion Battery Models for Parameter Inference, Parts I & II (2024)
- **Authors:** Ramirez, G.E., Latz, A., et al.
- **Venue:** arXiv preprints 2312.17329, 2312.17336
- **URL:** https://arxiv.org/html/2312.17329
- **Key Contribution:** Two-part work on using PINN surrogates for electrochemical model parameter inference. Part I establishes multi-fidelity hierarchies for the single-particle model (SPM), showing that training PINNs at multiple physics-loss fidelities significantly improves surrogate accuracy. Part II addresses regularization and application to the pseudo-2D (P2D) model, incorporating Butler-Volmer kinetics.
- **Methodology:** Multi-fidelity PINN hierarchy: coarse physics loss -> medium -> fine; Bayesian calibration via MCMC using the PINN surrogate; demonstrates on SPM and P2D electrochemical models with Butler-Volmer kinetics.
- **Relevance:** Most directly relevant work to this project -- applies surrogate-based inference to electrochemical PDE models with Butler-Volmer kinetics. The multi-fidelity hierarchy concept maps directly to the POD-RBF -> neural network -> PDE pipeline.
- **Confidence:** high

#### Multi-Fidelity Hamiltonian Monte Carlo (2024)
- **Authors:** (See arXiv)
- **Venue:** arXiv preprint 2405.05033
- **URL:** https://arxiv.org/html/2405.05033
- **Key Contribution:** MFHMC algorithm that uses a surrogate forward model for proposal generation (cheap gradient computation) while using the high-fidelity PDE solver for accept/reject decisions. Achieves exact posterior sampling with significant computational savings.
- **Methodology:** Two-level delayed acceptance: surrogate-based HMC proposals filtered by high-fidelity likelihood evaluation; convergence guarantees preserved via Metropolis-Hastings correction.
- **Relevance:** Provides a rigorous way to combine surrogate speed with PDE accuracy in Bayesian inference. The surrogate proposes, the PDE verifies -- maintaining posterior exactness.
- **Confidence:** high

#### Multi-Fidelity Bayesian Neural Networks: Algorithms and Applications (2021)
- **Authors:** Meng, X., Karniadakis, G.E.
- **Venue:** Journal of Computational Physics
- **URL:** https://www.sciencedirect.com/science/article/abs/pii/S0021999121002564
- **Key Contribution:** Three-network architecture: (1) low-fidelity network (MAP training), (2) cross-correlation Bayesian network (UQ between fidelity levels), (3) physics-informed network encoding PDEs. Enables principled fusion of cheap surrogate data with expensive PDE solves.
- **Methodology:** Mean-field variational inference and HMC for training; automatic fidelity management; physics-informed regularization.
- **Relevance:** Framework for systematically combining multiple surrogate fidelities (POD-RBF as low-fidelity, neural network as medium, PDE as high) with uncertainty quantification.
- **Confidence:** high

#### FUSE: Fast Unified Simulation and Estimation for PDEs (2024)
- **Authors:** (See arXiv)
- **Venue:** arXiv preprint 2405.14558
- **URL:** https://arxiv.org/html/2405.14558
- **Key Contribution:** Unified framework that performs simultaneous forward simulation and parameter estimation, avoiding the traditional two-step (solve forward, then optimize) approach. Uses amortized inference to enable real-time parameter estimation.
- **Methodology:** Encoder-decoder architecture: encoder maps observations to parameter estimates, decoder maps parameters to PDE solutions; trained end-to-end.
- **Relevance:** Alternative paradigm to the cascade approach -- could serve as a fast initializer for the multi-phase pipeline.
- **Confidence:** medium

#### Estimating Parameter Fields in Multi-Physics PDEs from Scarce Measurements (2025)
- **Authors:** (See arXiv)
- **Venue:** arXiv preprint 2509.00203
- **URL:** https://arxiv.org/html/2509.00203v1
- **Key Contribution:** Two-stage estimation strategy: first recovers coarse scalar parameters, then refines local variations via neural networks. Specifically designed for nonlinear multiphysics systems with multiple coupled parameter fields.
- **Methodology:** Stage 1: global sensitivity analysis + coarse optimization; Stage 2: neural network-based refinement of spatially varying fields. Demonstrated on coupled PDE systems.
- **Relevance:** Directly addresses the multi-physics PDE setting (like PNP) with a cascade estimation approach.
- **Confidence:** high

#### Electrochemical Impedance Spectroscopy of Ion Sensors: NPP Model and HGS(FP) Optimization (2012)
- **Authors:** Jasielec, J.J., et al.
- **Venue:** Electrochimica Acta
- **URL:** https://www.sciencedirect.com/science/article/abs/pii/S1572665711002293
- **Key Contribution:** Applies Nernst-Planck-Poisson model to numerically simulate EIS of ion-selective electrodes, then solves the inverse problem using Hierarchical Genetic Strategy (HGS) optimization. Demonstrates estimation of diffusion coefficients from impedance data.
- **Methodology:** NPP forward model for EIS simulation; evolutionary optimization (HGS) for parameter fitting; validated on ion-selective electrode membranes.
- **Relevance:** One of very few papers applying PNP inverse methods to electrochemical systems. Demonstrates feasibility but uses evolutionary optimization rather than gradient-based methods.
- **Confidence:** high

#### Identifiability Analysis of a Pseudo-Two-Dimensional Model (2025)
- **Authors:** (See arXiv)
- **Venue:** arXiv preprint 2507.13931
- **URL:** https://arxiv.org/html/2507.13931
- **Key Contribution:** Rigorous structural identifiability analysis of the P2D electrochemical model. Shows that many parameter combinations (not individual parameters) are identifiable from voltage data alone. The SPM-aided approach uses simplified models to constrain the identifiable parameter space before full P2D estimation.
- **Methodology:** Structural identifiability via differential algebra; practical identifiability via Fisher information matrix; multi-model cascade (SPM -> P2D) for staged parameter estimation.
- **Relevance:** Critical for PNP inverse problems: before attempting parameter estimation, one must verify which parameters (or parameter combinations) are identifiable from available observations. The cascade SPM -> P2D strategy is directly analogous to surrogate -> PDE refinement.
- **Confidence:** high

#### Structural Identifiability of a Pseudo-2D Li-ion Battery Electrochemical Model (2020)
- **Authors:** Bizeray, A.M., et al.
- **Venue:** IFAC-PapersOnLine / arXiv 2012.01853
- **URL:** https://arxiv.org/abs/2012.01853
- **Key Contribution:** First structural identifiability analysis of the full P2D model with Butler-Volmer kinetics. Shows the model is uniquely parametrized by conductivities and diffusion coefficients in certain experimental configurations, but many parameters are structurally unidentifiable from standard charge/discharge data alone.
- **Methodology:** Linearization and decoupling of the P2D model; differential algebra-based identifiability analysis.
- **Relevance:** Foundational for understanding what can and cannot be estimated from electrochemical data. Must be considered before designing the inference pipeline.
- **Confidence:** high

#### Forward and Inverse Simulation of P2D Model Using Neural Networks (2025)
- **Authors:** (See arXiv)
- **Venue:** Computer Methods in Applied Mechanics and Engineering / arXiv 2412.13200
- **URL:** https://arxiv.org/html/2412.13200
- **Key Contribution:** Addresses the challenge of the highly nonlinear Butler-Volmer equation in PINN-based forward and inverse simulation. Introduces a bypassing term that reduces the Hessian condition number of the PINN loss, improving numerical stability for inverse problems involving BV kinetics.
- **Methodology:** Modified PINN loss with BV bypassing term; demonstrated on P2D Li-ion battery model; inverse problem for recovering kinetic parameters.
- **Relevance:** Directly addresses the numerical challenges of inverting models with Butler-Volmer kinetics -- the same challenge faced in PNP inverse problems.
- **Confidence:** high

#### Physics-Informed Inference Time Scaling via Defect Correction (2025)
- **Authors:** (See arXiv)
- **Venue:** arXiv preprint 2504.16172
- **URL:** https://arxiv.org/html/2504.16172
- **Key Contribution:** Introduces inference-time refinement for PDE solutions: a base surrogate model is trained once, and computationally intensive defect correction is applied at inference time only when high precision is needed. Enables "elastic compute" -- trading inference time for accuracy on demand.
- **Methodology:** Base neural operator + iterative defect correction at inference time; convergence analysis shows progressive accuracy improvement.
- **Relevance:** Provides the theoretical justification for the surrogate warm-start -> PDE refinement cascade. The surrogate gives a fast initial guess; defect correction (or PDE re-solve) provides final accuracy.
- **Confidence:** medium

---

## Themes and Findings

### 1. The Reduced-Space Paradigm Dominates for Moderate Parameter Dimensions
For problems with O(1)-O(10) parameters (like kinetic rate constants, transfer coefficients, diffusion coefficients in Butler-Volmer/PNP systems), the reduced-space approach -- eliminating the PDE state via forward solve and optimizing over parameters alone -- is consistently preferred. L-BFGS with adjoint-computed gradients is the workhorse algorithm.

### 2. Surrogates are Essential for Bayesian Inference
Every paper attempting full Bayesian posterior characterization (not just point estimates) uses some form of surrogate to replace the expensive PDE forward solve in the likelihood evaluation. The three main surrogate families are:
- **POD-RBF / Reduced Basis:** Most mature, with certified error bounds; 10-100x speedup; best when the solution manifold is low-dimensional.
- **Neural Operators (DeepONet, FNO):** Highest potential speedup (1000x+); no error certificates; requires substantial training data; excels when the parameter space is high-dimensional.
- **Gaussian Process Emulators:** Natural uncertainty quantification; scales poorly beyond ~20 parameters; ideal for the "Calibrate, Emulate, Sample" pipeline.

### 3. Multi-Fidelity and Multi-Stage Pipelines are the State of the Art
The most successful recent approaches use hierarchical or cascade strategies:
- **Stage 1 (Cheap):** Surrogate-based optimization or ensemble Kalman methods for rough parameter estimates.
- **Stage 2 (Medium):** Refined surrogate or multi-fidelity sampling for better estimates with uncertainty.
- **Stage 3 (Expensive):** PDE-based verification or delayed-acceptance MCMC for rigorous posterior.

### 4. Identifiability Must Precede Estimation
Multiple papers emphasize that structural and practical identifiability analysis is a prerequisite for parameter estimation in complex PDE models. For electrochemical models with Butler-Volmer kinetics, many individual parameters are not identifiable -- only certain parameter combinations can be recovered from data.

### 5. Butler-Volmer Kinetics Present Special Numerical Challenges
The hyperbolic sine nonlinearity in the Butler-Volmer equation creates ill-conditioning in both forward solves and inverse problems. Recent work on PINN bypassing terms and specialized loss formulations addresses this, but the challenge remains for gradient-based optimization.

---

## Methodological Landscape

### When to Use Each Approach

| Method | Best For | Parameter Dim | Data Requirement | UQ Capability |
|--------|----------|---------------|------------------|---------------|
| Adjoint + L-BFGS (reduced-space) | Point estimates, moderate params | 1-50 | Forward model only | Via Hessian approximation |
| Full-space SQP/Newton | Distributed parameter fields | 1000+ | Forward model only | Limited |
| POD-RBF surrogate + optimization | Fast point estimates | 1-20 | 50-500 snapshots | Via ensemble methods |
| GP emulator + MCMC | Full Bayesian posterior | 1-20 | 20-200 evaluations | Native |
| Neural operator + optimization | High-dim params, fast inference | 1-1000 | 1000+ samples | Via ensembles |
| Calibrate-Emulate-Sample | Full posterior, expensive models | 1-30 | Adaptive (50-500) | Native |
| Multi-fidelity HMC | Exact posterior, expensive models | 1-50 | Surrogate + HF model | Exact |
| Two-stage DKL + HMC | Warm-started Bayesian inference | 1-100 | Moderate | Full posterior |

### Recommended Pipeline for PNP Inverse Problems

Based on the literature, the optimal pipeline for PNP parameter estimation with Butler-Volmer kinetics:

1. **Identifiability Analysis:** Determine which parameters/combinations are identifiable from available observations (voltage, concentration profiles, impedance).
2. **Surrogate Construction:** Build POD-RBF and/or neural operator surrogate for the PNP forward map.
3. **Warm-Start Phase:** Use surrogate with L-BFGS or ensemble Kalman to find MAP estimate and rough uncertainty bounds.
4. **Refinement Phase:** Use multi-fidelity HMC or delayed-acceptance MCMC with PDE solver for rigorous posterior.
5. **Validation:** Compare surrogate-based and PDE-based posteriors; check for surrogate-induced bias.

---

## Open Questions

1. **PNP-specific identifiability:** No existing work performs structural identifiability analysis specifically for the Poisson-Nernst-Planck system with Butler-Volmer boundary conditions in the electrochemical sensing/corrosion context. The battery P2D literature provides analogies but not direct results.

2. **Surrogate fidelity for multi-observable inference:** When fitting to multiple observables simultaneously (e.g., both current-voltage curves and concentration profiles), how should surrogate training be prioritized across observables?

3. **Transfer learning across experimental conditions:** Can a surrogate trained for one voltage range or electrolyte composition transfer to adjacent conditions, reducing the training burden for multi-condition inference?

4. **Scalability of neural operator surrogates for stiff PNP systems:** The PNP system with thin double layers creates boundary layer structures that challenge neural operators. No existing work benchmarks neural operators specifically for PNP-type stiff systems.

5. **Optimal fidelity allocation in multi-stage pipelines:** How many surrogate evaluations vs. PDE evaluations should be allocated in each stage for optimal computational efficiency?

6. **Handling model discrepancy:** When the PDE model itself is imperfect (e.g., simplified geometry, missing physics), how should the inference pipeline account for structural model error?

---

## Key Takeaways

- The **adjoint method** remains the gold standard for computing gradients in PDE-constrained optimization. For PNP systems, implementing a discrete adjoint solver enables efficient gradient-based parameter estimation.

- **POD-RBF surrogates** are the most established and reliable choice for moderate-dimensional parameter spaces (1-20 parameters), offering 10-100x speedup with well-understood error behavior. They are a natural fit for PNP systems where the solution manifold is relatively low-dimensional.

- **Neural operators** (DeepONet, FNO) offer the highest speedup potential but require more training data and lack error certificates. The derivative-informed variants (DINO, DIFNO) are particularly promising because they enable both fast forward evaluation AND fast gradient computation for optimization.

- The **"Calibrate, Emulate, Sample" paradigm** (Cleary et al., 2021) provides an elegant and practical template for multi-phase PNP inference: use ensemble methods to calibrate initial estimates, build an emulator from the calibration output, then sample the full posterior.

- **Identifiability analysis is non-negotiable** before designing any inference pipeline. For electrochemical PDE models with Butler-Volmer kinetics, many parameters are structurally unidentifiable from standard data, and attempting to estimate them leads to ill-conditioned optimization and meaningless posteriors.

- **Multi-fidelity methods** that combine cheap surrogates with expensive PDE solves (via trust regions, delayed acceptance MCMC, or hierarchical Bayesian models) represent the current state of the art for balancing computational cost with inference accuracy.

- The **Li-ion battery literature** (P2D model, SPM) provides the closest methodological parallels to PNP inverse problems and should be mined for practical implementation strategies, particularly regarding Butler-Volmer kinetics handling and multi-stage estimation cascades.

---

## References (Compact)

1. Hinze et al. (2009). Optimization with PDE Constraints. Springer.
2. Heinkenschloss (2008). PDE Constrained Optimization. SIAM Tutorial.
3. Cleary et al. (2021). Calibrate, Emulate, Sample. J. Comput. Phys.
4. Li et al. (2024). Efficient PDE-Constrained Optimization Using DINO. SIAM J. Sci. Comput.
5. Molinaro et al. (2023). Neural Inverse Operators. ICML.
6. Meng & Karniadakis (2021). Multi-Fidelity Bayesian Neural Networks. J. Comput. Phys.
7. Ramirez & Latz (2024). PINN Surrogate of Li-Ion Battery Models, Parts I & II.
8. Bizeray et al. (2020). Structural Identifiability of P2D Model. IFAC.
9. Jasielec et al. (2012). EIS of Ion Sensors: NPP Model and HGS. Electrochimica Acta.
10. Boyce & Yeh (2012). POD for Inverse Problems in Groundwater Flow. Adv. Water Res.
