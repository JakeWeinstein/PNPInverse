## Surrogate Models for PDE Forward Solvers in Inverse Problems

### 1. POD/RBF (Proper Orthogonal Decomposition / Radial Basis Functions)

POD compresses high-fidelity PDE solution snapshots into a low-dimensional reduced basis, then RBF interpolation maps new parameter values to coefficients in that basis. This is the classical reduced-order modeling approach.

**How it works:** Collect solution snapshots at sampled parameter values, compute SVD to extract dominant modes (POD basis), then train an RBF interpolant that maps parameter vectors to POD coefficients. Online evaluation requires only the RBF lookup plus basis reconstruction.

**Strengths:**
- Mature, well-understood theory with rigorous error bounds
- Very fast online evaluation (microseconds for small bases)
- Excellent for mildly nonlinear, smooth parameter-to-solution maps
- Low training data requirements (tens to hundreds of snapshots)
- Naturally produces interpretable reduced coordinates

**Weaknesses:**
- POD basis is global: struggles with sharp fronts, traveling waves, or bifurcations
- RBF interpolation degrades in high parameter dimensions (>10-15)
- Nonlinear PDEs require hyper-reduction (POD-DEIM) to avoid full-order assembly costs

**POD-DEIM for nonlinear terms:** The Discrete Empirical Interpolation Method (DEIM) reduces the complexity of evaluating nonlinear terms in the ROM from full-order to proportional to the number of reduced variables. This is critical for reaction-diffusion systems with nonlinear source terms like Butler-Volmer kinetics.
- **Source:** [POD/DEIM Reduced-Order Modeling with Applications in Parameter Identification](https://link.springer.com/article/10.1007/s10915-017-0433-8)
- **Source:** [Nonlinear Model Reduction via Discrete Empirical Interpolation (Chaturantabut & Sorensen, SIAM)](https://epubs.siam.org/doi/10.1137/090766498)

**Stability concern for reaction-diffusion:** Adaptive POD-DEIM corrections have been proposed specifically for Turing pattern approximation in reaction-diffusion PDEs, addressing unstable error behavior in classical POD surrogates when the dynamics are sensitive to perturbations.
- **Source:** [Adaptive POD-DEIM correction for Turing pattern approximation in reaction-diffusion PDE systems](https://www.degruyterbrill.com/document/doi/10.1515/jnma-2022-0025/html)

**POD+RBF for magnetostatics (representative benchmark):** IEEE study demonstrated POD combined with RBF interpolation for nonlinear FE models, achieving good accuracy with modest snapshot counts.
- **Source:** [Surrogate Model Based on the POD Combined With the RBF Interpolation of Nonlinear Magnetostatic FE Model](https://ieeexplore.ieee.org/document/8936617/)

**Best suited for:** Low-to-moderate parameter dimensions (<10), smooth parameter dependence, problems where the solution manifold has rapidly decaying singular values. Strong candidate for electrochemistry problems with a few kinetic/transport parameters.
- **Confidence:** high

---

### 2. Gaussian Process (GP) Surrogates

GPs provide a probabilistic surrogate that returns both a mean prediction and uncertainty estimate, making them naturally suited for Bayesian inference loops.

**How it works:** A GP prior is placed on the parameter-to-observable map. Training on (parameter, observable) pairs yields a posterior GP. The posterior mean serves as the surrogate prediction, and the posterior variance quantifies surrogate uncertainty.

**Strengths:**
- Built-in uncertainty quantification (posterior variance)
- Excellent with small training sets (10s-100s of points)
- Kernel choice encodes prior knowledge (smoothness, periodicity, etc.)
- Well-suited for Bayesian optimization and sequential experimental design

**Weaknesses:**
- Cubic scaling O(N^3) in the number of training points -- impractical beyond ~10,000 points
- Scales poorly with input dimension (>15-20 parameters)
- Struggles with discontinuities or highly nonlinear maps without careful kernel engineering

**Physical law-corrected GP surrogates:** A novel LC-prior GP framework integrates physical constraints by using POD to parameterize high-dimensional PDE solutions via dominant modes, then building GP surrogates in the reduced coefficient space. This combines the data efficiency of GPs with the dimensionality reduction of POD.
- **Source:** [Gaussian process surrogate with physical law-corrected prior for multi-coupled PDEs](https://arxiv.org/html/2509.02617)
- **Confidence:** high

**GP for MCMC acceleration:** GPs have been used as surrogates inside MCMC loops for Bayesian PDE inversion. A key finding: GP parameterization combined with deep learning surrogates enables efficient posterior inference for infinite-dimensional coefficient functions.
- **Source:** [A MCMC method based on surrogate model and Gaussian process parameterization for infinite Bayesian PDE inversion](https://www.sciencedirect.com/science/article/abs/pii/S0021999124002195)
- **Confidence:** high

**PDE-constrained GP surrogates:** Recent work places GPs directly on the PDE solution operator with uncertain data locations, incorporating PDE structure into the kernel. This produces more physically consistent surrogates.
- **Source:** [PDE-constrained Gaussian process surrogate modeling with uncertain data locations](https://link.springer.com/article/10.1186/s40323-025-00308-3)
- **Confidence:** medium

**Best suited for:** Small parameter spaces (<10-15 dims), when UQ on the surrogate itself is needed, Bayesian optimization for experimental design, and problems with expensive forward solves where only 100s of training evaluations are feasible.

---

### 3. Polynomial Chaos Expansion (PCE)

PCE represents the parameter-to-observable map as a weighted sum of orthogonal polynomials, chosen to be orthogonal with respect to the input parameter distribution.

**How it works:** The QoI is expanded as a sum of multivariate orthogonal polynomials (Hermite for Gaussian inputs, Legendre for uniform, etc.). Coefficients are computed via least-squares regression or sparse regression (compressive sensing) on a set of forward model evaluations.

**Strengths:**
- Analytically tractable: Sobol sensitivity indices come directly from coefficients
- Fast convergence for smooth parameter dependence (spectral convergence)
- Sparse PCE (e.g., LARS, OMP) handles moderate dimensions efficiently
- Well-established theory and software (UQLab, OpenTURNS, ChaosPy)

**Weaknesses:**
- Curse of dimensionality: basis size grows combinatorially with dimension and polynomial degree
- Poor for non-smooth or discontinuous parameter-to-observable maps
- No built-in adaptivity (unlike GPs)

**Physics-constrained PCE:** Recent work embeds PDE constraints directly into the PCE framework, enabling simultaneous surrogate modeling and UQ. The constrained formulation can incorporate initial/boundary conditions and governing equations.
- **Source:** [Physics-constrained polynomial chaos expansion for scientific machine learning and UQ](https://arxiv.org/html/2402.15115v1)
- **Confidence:** high

**Application to batteries/electrochemistry:** PCE-based surrogates have been constructed for vanadium redox flow batteries, building statistical relationships between parameter sets and cell voltages from PDE-based models. Sobol indices from PCE quantify parameter sensitivity.
- **Source:** [Surrogate model-based parameter estimation framework for vanadium redox flow batteries](https://www.sciencedirect.com/science/article/abs/pii/S0306261925000510)
- **Source:** [Stochastic model of Li-ion batteries based on PCE](https://www.researchgate.net/publication/337047083_Stochastic_model_of_Lithium-ion_Batteries_based_on_Polynomial_Chaos_Expansion)
- **Confidence:** high

**Best suited for:** Low-to-moderate parameter dimensions (<15), smooth parameter dependence, when global sensitivity analysis (Sobol indices) is a primary goal, and when the parameter distribution is known a priori.

---

### 4. Neural Network Emulators (Feedforward / Convolutional)

Standard neural networks trained on (parameter, observable) pairs to approximate the forward map.

**How it works:** A feedforward or convolutional network is trained on a dataset of (parameter vector, PDE solution or observable) pairs generated by running the high-fidelity solver. The trained network replaces the solver in the inference loop.

**Strengths:**
- Flexible function approximators; can handle complex nonlinear maps
- Scale to high input/output dimensions
- Fast inference (milliseconds on GPU)
- Can be combined with physics-informed losses (PINNs)

**Weaknesses:**
- Require large training datasets (thousands to tens of thousands of forward solves)
- No built-in uncertainty quantification (requires ensembles, MC dropout, or Bayesian NN)
- Prone to overfitting and poor extrapolation outside training distribution
- Training can be unstable for stiff PDE systems

**Bayesian Physics-Informed Neural Networks (B-PINNs):** Combine PINNs with Bayesian inference (HMC or variational inference) to quantify uncertainty. B-PINNs avoid overfitting in high-noise scenarios and provide posterior distributions over PDE parameters.
- **Source:** [B-PINNs: Bayesian Physics-Informed Neural Networks for Forward and Inverse PDE Problems](https://www.researchgate.net/publication/347391626_B-PINNs_Bayesian_physics-informed_neural_networks_for_forward_and_inverse_PDE_problems_with_noisy_data)
- **Confidence:** high

**Multi-Fidelity Bayesian PINNs (MF-BPINN):** Leverage abundant low-fidelity simulations alongside sparse high-fidelity data through a hierarchical architecture. Achieves comparable accuracy to high-fidelity PINNs while reducing computational cost by 73-86%.
- **Source:** [Multi-Fidelity Physics-Informed Neural Networks with Bayesian UQ](https://arxiv.org/html/2602.01176)
- **Confidence:** medium

**Best suited for:** Problems with large training budgets, high-dimensional parameter spaces, when GPU acceleration is available, and when combined with physics-informed or Bayesian extensions to address UQ.

---

### 5. DeepONet (Deep Operator Networks)

DeepONet learns operators -- mappings from function spaces to function spaces -- making it naturally suited for parametric PDE surrogates.

**Architecture:** Two sub-networks: a **branch net** that encodes the input function (e.g., parameter field, boundary condition, forcing) at sensor locations, and a **trunk net** that encodes the output location (spatial/temporal coordinates). The output is a dot product of branch and trunk outputs.

**Strengths:**
- Learns the solution operator, not just point evaluations
- Can evaluate at arbitrary output points (not tied to a fixed mesh)
- Handles irregular domains and complex geometries well
- Generalizes across different input functions

**Weaknesses:**
- Requires careful sensor placement for the branch net
- Can struggle with highly oscillatory or multi-scale solutions
- Training data generation still requires many forward solves

**DeepBayONet for Bayesian parameter estimation:** Combines DeepONet with variational inference for posterior estimation of PDE parameters. On a 2D reaction-diffusion problem, achieved parameter recovery of 0.999 (true value 1.0) with std 5.4e-4. Handles surrogate error through learned residual noise variance.
- **Source:** [Deep Operator Networks for Bayesian Parameter Estimation in PDEs](https://arxiv.org/html/2501.10684v1)
- **Confidence:** high

**Physics-informed DeepONet:** Can be trained without labeled data by embedding the PDE residual in the loss function, learning the solution operator from physics alone.
- **Source:** [Learning the solution operator of parametric PDEs with physics-informed DeepONets](https://www.science.org/doi/10.1126/sciadv.abi8605)
- **Confidence:** high

**NVIDIA PhysicsNeMo implementation:** NVIDIA provides a production-ready DeepONet implementation in their PhysicsNeMo framework for scientific computing.
- **Source:** [Deep Operator Network -- NVIDIA PhysicsNeMo](https://docs.nvidia.com/physicsnemo/25.08/physicsnemo-sym/user_guide/neural_operators/deeponet.html)
- **Confidence:** high

**Best suited for:** Problems requiring evaluation at arbitrary spatial points, irregular domains, when the input is a function (not just a parameter vector), and when generalization across input function families is needed.

---

### 6. Fourier Neural Operator (FNO)

FNO learns in spectral space by parameterizing convolutional kernels in the Fourier domain, enabling resolution-invariant learning.

**Architecture:** Lifting layer -> sequence of Fourier layers (each applies FFT, multiplies by learnable weights in frequency space, applies inverse FFT, adds a local linear transform, and passes through nonlinearity) -> projection layer.

**Strengths:**
- Resolution invariant: train on coarse, evaluate on fine grids
- Efficient for problems with smooth, periodic solutions
- Strong generalization for dissipative systems (captures dominant spectral modes)
- Fast inference

**Weaknesses:**
- Struggles with complex geometries (requires regular grids or extensions like geo-FNO)
- Performance degrades for non-smooth solutions or sharp interfaces
- Fixed input grid structure (unlike DeepONet)

**DeepONet vs FNO comparison:** A comprehensive FAIR comparison found that for simple settings, both have comparable performance, but for complex geometries FNO's performance deteriorates significantly. FNO generally achieves higher accuracy on regular domains; DeepONet offers greater flexibility.
- **Source:** [A comprehensive and fair comparison of two neural operators](https://www.sciencedirect.com/science/article/abs/pii/S0045782522001207)
- **Source:** [GitHub: deeponet-fno comparison](https://github.com/lu-group/deeponet-fno)
- **Confidence:** high

**Derivative-Informed FNO (DIFNO) for PDE-constrained optimization:** DIFNOs train on both operator outputs AND Frechet derivatives, which is critical for optimization. Key finding: optimization solution errors depend on BOTH pointwise operator errors and derivative errors. DIFNOs achieve high accuracy at low training sample sizes and eliminate quadratic dependence of training cost on grid size through dimension reduction.
- **Source:** [Derivative-Informed Fourier Neural Operator for PDE-Constrained Optimization](https://arxiv.org/abs/2512.14086)
- **Confidence:** high

**Sensitivity-Constrained FNO:** Recent work constrains FNO training to preserve sensitivity structure, improving performance in both forward and inverse parametric PDE problems.
- **Source:** [Sensitivity-Constrained FNOs for Forward and Inverse Problems](https://arxiv.org/html/2505.08740v1)
- **Confidence:** medium

**Best suited for:** Regular domains, periodic or smooth solutions, problems where resolution invariance matters, and PDE-constrained optimization (with DIFNO extension).

---

### 7. Physics-Enhanced Deep Surrogates (PEDS)

PEDS combines a low-fidelity physics solver embedded as a differentiable layer within a neural network, trained end-to-end against high-fidelity data.

**Architecture:** A neural network "generator" produces modified inputs that feed into a coarse PDE solver. The coarse solver output is the surrogate prediction. A separate variance network estimates model uncertainty. The generator learns to pre-distort inputs such that the coarse solver produces outputs matching the fine solver.

**Key performance numbers:**
- Up to **3x more accurate** than feedforward NN ensembles with limited data (~1000 training points)
- Reduces training data need by **at least 100x** to achieve 5% target error
- **36x faster** evaluation than mid-fidelity solvers (5ms vs 180ms per evaluation)
- Outperforms NN-only by 66%, polynomial chaos by 74%, GP by 49% on Maxwell's equations
- **Source:** [Physics-enhanced deep surrogates for PDEs (Nature Machine Intelligence)](https://www.nature.com/articles/s42256-023-00761-y)
- **Source:** [PEDS supplementary information](https://arxiv.org/html/2111.05841v4)
- **Confidence:** high

**When PEDS excels vs. alternatives:** For complex, nonlinear problems (Maxwell's equations, nonlinear diffusion), PEDS dramatically outperforms alternatives. However, for simpler cases like linear diffusion, "traditional surrogates such as Poly Chaos, RadialBasis and GP are competitive."

**Best suited for:** Problems where a fast coarse solver exists, highly nonlinear PDE systems, limited training data regimes, and cases where the coarse solver captures essential physics that a pure neural network would need many samples to learn.

---

### 8. Handling Surrogate Bias/Error in the Inference Loop

A critical challenge: using an approximate surrogate in place of the true forward model introduces bias in the posterior distribution. Several strategies address this.

**8a. Two-Level / Delayed Acceptance MCMC**

Run a base MCMC chain with the surrogate for fast proposal screening, then use the high-fidelity model in a correction step. The correction chain ensures the stationary distribution matches the true posterior despite surrogate approximation error.

- Hybrid two-level MCMC: base chain with DL surrogate, correction chain with numerical model of known accuracy.
- **Source:** [Hybrid Two-level MCMC with Deep Learning Surrogates for Bayesian Inverse Problems](https://arxiv.org/html/2307.01463)
- **Confidence:** high

**8b. Multi-Fidelity Delayed Acceptance (MFDA)**

Hierarchical MCMC that trains neural networks to fuse information from multiple fidelity levels. The high-fidelity model is NOT called during sampling -- only offline for training. Achieves ~5x speedup over standard MH, ~3x over multi-level DA, with comparable posterior accuracy.

Key practical recommendations:
- Use uniform mesh refinement to create fidelity hierarchy
- ~16,000 training samples for groundwater-type problems (problem-specific)
- Ensure the finest surrogate level achieves high accuracy since the chain's stationary distribution is determined by this level
- Assign longer sub-chains to coarser, cheaper levels
- **Source:** [Multi-Fidelity Delayed Acceptance MCMC](https://arxiv.org/html/2512.16430)
- **Confidence:** high

**8c. Adaptive Multi-Fidelity Posterior Convergence**

Rigorous error bounds show: if the surrogate approximation converges at rate r in the prior-weighted L2 norm, then the KL divergence between the approximate and true posteriors converges at rate 2r. This means surrogate accuracy translates to posterior accuracy at double the rate.
- **Source:** [Adaptive multi-fidelity polynomial chaos approach to Bayesian inference in inverse problems](https://www.researchgate.net/publication/330234543_Adaptive_multi-fidelity_polynomial_chaos_approach_to_Bayesian_inference_in_inverse_problems)
- **Confidence:** high

**8d. Learned Noise Variance for Surrogate Error**

DeepBayONet and similar frameworks model surrogate-data mismatch through a learnable residual noise variance parameter. This inflates posterior uncertainty appropriately to account for surrogate inaccuracy, rather than producing overconfident posteriors.
- **Source:** [Deep Operator Networks for Bayesian Parameter Estimation in PDEs](https://arxiv.org/html/2501.10684v1)
- **Confidence:** high

**8e. Coarse-Solver + Neural Residual Correction**

Combine a coarse PDE solver with a residual network that corrects the bias. The residual network learns the systematic error pattern of the coarse solver. This is the core idea behind PEDS (Section 7) but applies more broadly.
- **Source:** [Physics-enhanced deep surrogates for PDEs](https://www.nature.com/articles/s42256-023-00761-y)
- **Confidence:** high

---

### 9. Multi-Fidelity Approaches

Multi-fidelity methods exploit a hierarchy of models (e.g., coarse mesh / fine mesh, linearized / full nonlinear, 1D / 2D / 3D) to reduce overall computational cost while maintaining accuracy.

**Multi-fidelity Bayesian Neural Networks:** Three-network architecture: (1) feedforward NN trained on low-fidelity data via MAP, (2) Bayesian NN for cross-correlation between fidelities with UQ, (3) PINN encoding physical laws. Demonstrated on Burgers, Navier-Stokes, and heat transfer.
- **Source:** [Multi-fidelity Bayesian neural networks: Algorithms and applications](https://www.sciencedirect.com/science/article/abs/pii/S0021999121002564)
- **Confidence:** high

**Multi-fidelity for physiological boundary conditions:** Neural emulators at reduced fidelity and dimensionality for hemodynamics inference, demonstrating that multi-fidelity emulators can match high-fidelity inference at a fraction of the cost.
- **Source:** [On the performance of multi-fidelity and reduced-dimensional neural emulators for inference](https://arxiv.org/abs/2506.11683)
- **Confidence:** medium

**Bayesian Optimization + Bayesian Inversion (BO-BI):** BO adaptively constructs an accurate GP surrogate using limited high-fidelity evaluations, then BI uses this surrogate for probabilistic parameter inference. The adaptive sampling focuses training points where they matter most for the inverse problem.
- **Source:** [Efficient Bayesian Framework for Inverse Problems via Optimization and Inversion](https://arxiv.org/html/2602.04537v1)
- **Confidence:** medium

---

### 10. Practical Guidance for Electrochemistry / Reaction-Diffusion PDE Systems

Electrochemical systems governed by Poisson-Nernst-Planck or reaction-diffusion equations with Butler-Volmer kinetics have specific characteristics that inform surrogate selection:

**Problem characteristics:**
- Moderate parameter dimension (typically 3-10 kinetic/transport parameters)
- Nonlinear source terms (Butler-Volmer, Tafel kinetics)
- Coupled multi-species transport (concentration, potential fields)
- Stiff dynamics (thin boundary/double layers, fast reactions)
- Outputs are often integral quantities (current-voltage curves, impedance spectra)

**Recommended surrogate strategy:**

1. **Start with POD-DEIM** if you have a working FEM solver (e.g., in Firedrake/FEniCS):
   - Natural fit for moderate parameter dimensions
   - DEIM handles nonlinear reaction terms efficiently
   - Well-established theory; easier to validate
   - Typical speedup: 100-1000x over full FEM

2. **If POD-DEIM is insufficient** (too many parameters, insufficient accuracy, bifurcations):
   - Move to **neural network emulator** trained on FEM snapshots
   - Use ensemble or MC-dropout for UQ
   - Consider **DeepONet** if you need to evaluate at arbitrary spatial points or generalize across different geometries

3. **For Bayesian inference specifically:**
   - Use GP surrogate if parameter dimension < 10 and training budget < 500 forward solves
   - Use neural emulator + delayed acceptance MCMC for larger problems
   - Always validate the surrogate posterior against a small number of full-model MCMC runs

4. **For PDE-constrained optimization:**
   - DIFNO if the problem has regular geometry and smooth solutions
   - PEDS if a coarse solver is available and the problem is highly nonlinear
   - Always train with derivative information when optimization gradients are needed

**Software ecosystem:**
- **POD/ROM:** [pyMOR](https://pymor.org/), FEniCS/Firedrake built-in ROM tools
- **Neural operators:** NVIDIA PhysicsNeMo, DeepXDE, NeuralOperator (PyTorch)
- **GPs:** GPyTorch, GPflow, scikit-learn (small scale)
- **PCE:** UQLab (MATLAB), OpenTURNS (Python), ChaosPy (Python)
- **Bayesian inference:** PyMC, emcee, TensorFlow Probability

---

### Quick Reference: Surrogate Selection Decision Guide

| Criterion | POD/RBF | GP | PCE | NN Emulator | DeepONet | FNO | PEDS |
|---|---|---|---|---|---|---|---|
| Parameter dims | <10 | <15 | <15 | Any | Any | Any | Any |
| Training samples needed | 10s-100s | 10s-100s | 10s-1000s | 1000s-10000s | 1000s+ | 1000s+ | 100s-1000s |
| Built-in UQ | No | Yes | Yes (Sobol) | No (needs ensemble) | No (needs Bayesian ext.) | No | Yes (variance net) |
| Nonlinear PDEs | DEIM needed | OK with kernel | Poor if non-smooth | Good | Good | Good | Excellent |
| Irregular geometry | Yes | Yes | Limited | Yes | Yes | Poor | Yes |
| Resolution invariant | No | No | No | No | Partial | Yes | Partial |
| Derivative accuracy | Analytical | Analytical | Analytical | Backprop | Backprop | DIFNO variant | Backprop |
| Typical speedup | 100-1000x | 10-100x | 100-1000x | 1000-10000x | 1000x+ | 1000x+ | 30-100x |
| Maturity | Very high | High | High | High | Medium | Medium | Low-Medium |

**Decision flowchart for electrochemistry inverse problems:**

1. Do you have < 5 parameters and < 500 training budget? -> **GP** or **PCE**
2. Do you have a fast coarse solver available? -> **PEDS**
3. Do you need UQ on the surrogate? -> **GP** (small scale) or **B-PINN / DeepBayONet** (large scale)
4. Is the geometry irregular or changing? -> **DeepONet**
5. Is the geometry regular with smooth solutions? -> **FNO** (or **DIFNO** for optimization)
6. Is the parameter space high-dimensional (>15)? -> **NN emulator** with ensemble UQ
7. Default for moderate complexity: **POD-DEIM** + delayed acceptance MCMC

---

### Key Takeaways

- **No single surrogate dominates all settings.** POD/RBF and GPs remain competitive for low-dimensional, smooth problems with small training budgets. Neural operators (DeepONet, FNO) excel in higher dimensions and complex geometries but need more training data.
- **Surrogate bias in the inference loop is a solved problem in principle.** Delayed acceptance MCMC, multi-fidelity hierarchies, and learned noise variance all provide rigorous mechanisms to prevent surrogate error from corrupting posterior estimates. The practical challenge is implementation complexity.
- **For electrochemistry/reaction-diffusion systems with moderate parameter dimensions (3-10), POD-DEIM is the pragmatic starting point** given existing FEM infrastructure. It provides 100-1000x speedup, handles nonlinear reaction terms via DEIM, and has well-established error theory.
- **PEDS is the most data-efficient neural approach** (100x less data than black-box NNs for 5% error), making it attractive when forward solves are very expensive. It requires a coarse solver to embed.
- **Always validate surrogate-based posteriors** against a small number of full-model evaluations. The KL convergence rate result (surrogate L2 error rate r gives posterior KL rate 2r) provides theoretical backing but practical validation remains essential.
- **Derivative-informed training (DIFNO) is critical for PDE-constrained optimization.** Standard operator surrogates can have accurate predictions but inaccurate gradients, leading to poor optimization performance.
- **Multi-fidelity strategies provide the best cost-accuracy tradeoff** for Bayesian inference, combining cheap coarse evaluations with targeted fine evaluations. The MFDA approach achieves ~5x speedup over standard MCMC with neural network fusion of fidelity levels.
