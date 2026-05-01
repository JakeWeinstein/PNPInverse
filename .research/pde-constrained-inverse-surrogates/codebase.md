## Codebase Analysis: PDE-Constrained Inverse Problem Pipeline

### Relevant Existing Code

**Forward PDE Solver (Firedrake-based PNP-BV)**
- `Forward/bv_solver/solvers.py` -- Three BV solver strategies: basic continuation (`solve_bv_with_continuation`), pseudo-transient continuation PTC (`solve_bv_with_ptc`), and charge continuation z-ramping (`solve_bv_with_charge_continuation`). All use Firedrake's `NonlinearVariationalSolver` with Newton iteration and time-stepping to steady state.
- `Forward/bv_solver/forms.py` -- Assembles weak forms for the coupled Nernst-Planck + Poisson system with Butler-Volmer electrode boundary conditions. Supports multi-reaction stoichiometry, Bikerman steric effects, softplus concentration regularization, exponent clipping, and log-diffusivity controls for adjoint compatibility.
- `Forward/bv_solver/config.py` -- Parses BV configuration: per-species `k0`, `alpha`, stoichiometry, `c_ref`, `E_eq_v`, multi-reaction config with cathodic concentration factors, convergence strategy options.
- `Forward/bv_solver/mesh.py` -- Graded meshes with boundary-layer refinement (beta-stretching).
- `Forward/bv_solver/nondim.py` -- BV-specific nondimensionalization extensions.
- `Forward/params.py` -- Frozen `SolverParams` dataclass (11-element: n_species, order, dt, t_end, z_vals, D_vals, a_vals, phi_applied, c0_vals, phi0, solver_options). Immutable with `.with_*()` mutation helpers.
- `Forward/steady_state/bv.py` -- Steady-state voltage sweep and `configure_bv_solver_params()` for injecting k0/alpha into the solver config.
- `Forward/robin_solver.py` -- Robin-BC PNP solver (alternative to BV).
- `Forward/dirichlet_solver.py` -- Dirichlet-BC PNP solver.
- `Forward/noise.py` -- Gaussian noise injection: `sigma = (noise_percent/100) * RMS(field)`. Functions for both Dirichlet and Robin solvers.

**Nondimensionalization**
- `Nondim/transform.py` -- Full nondimensionalization transform: L_ref, D_ref, c_ref, V_T = RT/F. Produces Poisson coefficient `(lambda_D/L_ref)^2`, electromigration prefactor, scaled BV rate constants. Handles dimensional/nondimensional input flags.
- `Nondim/constants.py` -- Physical constants (F, R, T, epsilon_water, etc.).
- `Nondim/scales.py` -- Scale factor calculations.

**Surrogate Models**
- `Surrogate/surrogate_model.py` -- `BVSurrogateModel`: baseline RBF interpolation surrogate using `scipy.interpolate.RBFInterpolator`. Maps 4D input `[k0_1, k0_2, alpha_1, alpha_2]` to vector-valued I-V curves (current_density + peroxide_current). Supports log-space k0 transform and z-score input normalization.
- `Surrogate/pod_rbf_model.py` -- `PODRBFSurrogateModel`: POD (SVD) dimensionality reduction + per-mode RBF interpolation. Retains modes for 99.9% variance. Per-mode smoothing optimized via LOO or k-fold CV on a 30-point log grid. Optional log1p transform for peroxide current.
- `Surrogate/nn_model.py` -- `NNSurrogateModel`: PyTorch ResNet-MLP (4->128->4xResBlock->64->44). Uses LayerNorm + SiLU activation, AdamW optimizer with cosine annealing warm restarts, early stopping. Z-score normalization on both inputs and outputs. Save/load with model.pt + normalizers.npz.
- `Surrogate/ensemble.py` -- `EnsembleMeanWrapper`: wraps N `NNSurrogateModel` instances, prediction = mean across members. `predict_with_uncertainty()` returns per-point std. `load_nn_ensemble()` loads from `member_0/saved_model/`, ..., `member_{n-1}/saved_model/`.
- `Surrogate/nn_training.py` -- Training utilities for NN surrogate.
- `Surrogate/sampling.py` -- Latin Hypercube Sampling with log-space k0. `ParameterBounds` dataclass with default ranges: k0_1 in [1e-6, 1.0], k0_2 in [1e-7, 0.1], alpha in [0.1, 0.9]. `generate_multi_region_lhs_samples()` for wide + focused region sampling.
- `Surrogate/training.py` -- `generate_training_dataset()` and `generate_training_dataset_parallel()`: runs BV solver for each parameter sample to build (params, I-V curve) training pairs. Supports checkpointing, resume, cross-sample warm-starting, nearest-neighbor ordering for warm-start efficiency, parallel workers.
- `Surrogate/validation.py` -- `validate_surrogate()`: computes RMSE, max absolute error, per-sample NRMSE for CD and PC.
- `Surrogate/io.py` -- Save/load surrogate models and training data.

**Inverse/Inference Pipeline**
- `Inverse/objectives.py` -- Pre-built objective factories: `make_diffusion_objective_and_grad`, `make_dirichlet_phi0_objective_and_grad`, `make_robin_kappa_objective_and_grad`. All build Firedrake adjoint `ReducedFunctional` objects.
- `Inverse/solver_interface.py` -- `ForwardSolverAdapter`: runtime adapter for plugging any forward solver module into the inverse pipeline. `run_forward()` method executes build_context -> build_forms -> set_initial_conditions -> solve.
- `Inverse/inference_runner/objective.py` -- `build_reduced_functional()`: constructs the Firedrake adjoint reduced functional. Tapes the forward solve, builds L2 objective `0.5 * integral((U - U_target)^2 dx)`, creates `adj.Control` objects, and returns `adj.ReducedFunctional`.
- `Inverse/inference_runner/config.py` -- `InferenceRequest` dataclass: adapter, target, base_solver_params, true_value, initial_guess, noise_percent, optimizer_method (default "L-BFGS-B"), tolerance (1e-8), bounds, recovery config. `RecoveryConfig`: max_attempts=15, staged recovery (max_it -> anisotropy -> tolerance_relax), line search schedule (bt, l2, cp, basic).
- `Inverse/inference_runner/recovery.py` -- `resilient_minimize()`: always-on recovery loop. Calls `firedrake.adjoint.minimize()` with L-BFGS-B. On failure: cycles through phases (increase snes_max_it, reduce anisotropy, relax tolerances). Tracks best-seen estimate via `_AttemptMonitor`.
- `Inverse/parameter_targets.py` -- Registry of parameter targets (diffusion, dirichlet_phi0, robin_kappa, etc.).

**Surrogate-Based Optimization Strategies**
- `Surrogate/objectives.py` -- `SurrogateObjective` (4D joint), `AlphaOnlySurrogateObjective` (2D alpha-only), `ReactionBlockSurrogateObjective` (2D per-reaction), `SubsetSurrogateObjective` (voltage subset). All use central finite-difference gradients (8 evals per gradient for 4D). Objective: `J = 0.5*||cd_sim - cd_target||^2 + w * 0.5*||pc_sim - pc_target||^2`.
- `Surrogate/cascade.py` -- `run_cascade_inference()`: 3-pass per-observable cascade. Pass 1: CD-dominant (low weight, all 4 params). Pass 2: PC-dominant (high weight, k0_2+alpha_2 only, k0_1+alpha_1 fixed from Pass 1). Pass 3: joint polish. All use L-BFGS-B via `scipy.optimize.minimize`.
- `Surrogate/bcd.py` -- `run_block_coordinate_descent()`: alternating 2D sub-problems. Block 1: optimize (k0_1, alpha_1) with low weight. Block 2: optimize (k0_2, alpha_2) with high weight. Convergence check on parameter change. Uses L-BFGS-B.
- `Surrogate/multistart.py` -- `run_multistart_inference()`: 20,000 LHS grid points evaluated via `predict_batch`, top-20 polished with L-BFGS-B. Uses `scipy.stats.qmc.LatinHypercube`.

**FluxCurve (Adjoint-Based BV Inference)**
- `FluxCurve/bv_run/pipelines.py` -- End-to-end BV k0/alpha inference using PDE adjoint gradients. `run_bv_k0_flux_curve_inference()` entry point.
- `FluxCurve/bv_run/optimization.py` -- `run_scipy_bv_adjoint_optimization()`: wraps scipy L-BFGS-B with Firedrake adjoint gradients.
- `FluxCurve/bv_curve_eval.py` -- Multi-observable objective evaluation with PDE adjoint gradients.
- `FluxCurve/bv_point_solve/` -- Point-by-point BV solve with warm-starting and caching.
- `FluxCurve/bv_config.py` -- `BVFluxCurveInferenceRequest` config.

**Scripts**
- `scripts/surrogate/generate_training_data.py` -- Generates PDE training data.
- `scripts/surrogate/build_surrogate.py` -- Builds RBF surrogate.
- `scripts/surrogate/train_nn_surrogate.py` -- Trains NN ensemble.
- `scripts/surrogate/train_improved_surrogate.py` -- Trains POD-RBF and improved models.
- `scripts/surrogate/multistart_inference.py` -- Runs multi-start inference.
- `scripts/surrogate/overnight_train_v11.py` -- Large overnight training pipeline.
- `scripts/surrogate/validate_surrogate.py` -- Validates surrogate vs PDE.
- `scripts/verification/mms_bv_4species.py` -- Method of Manufactured Solutions for BV solver verification.
- `scripts/studies/profile_likelihood_study.py` -- Profile likelihood analysis.
- `scripts/studies/profile_likelihood_pde.py` -- PDE-based profile likelihood.
- `scripts/studies/sensitivity_visualization.py` -- Sensitivity analysis plots.

**Tests**
- `tests/test_inference_config.py`, `tests/test_multistart.py`, `tests/test_surrogate_fidelity.py`, `tests/test_inverse_verification.py`, `tests/test_mms_convergence.py`, `tests/test_v13_verification.py`, `tests/test_pipeline_reproducibility.py`, `tests/test_nondim.py`, `tests/test_params.py`, etc.

**Study Results**
- `StudyResults/surrogate_fidelity/` -- Fidelity comparison across surrogate types (RBF baseline, POD-RBF log/nolog, NN ensemble).
- `StudyResults/inverse_verification/` -- Gradient FD convergence, PDE gradient consistency, multistart basin analysis, parameter recovery summary.
- `StudyResults/mms_convergence/` -- MMS convergence data.
- `StudyResults/master_inference_v13/` -- Multi-observable inference results for P1 (shallow) and P2 (full cathodic) regimes.
- `StudyResults/v14/` -- Sensitivity and multi-seed studies.

### Implementation Patterns

1. **Immutable parameter passing**: `SolverParams` is a frozen dataclass with `.with_*()` mutation helpers returning new instances. Deep copies used throughout the inverse pipeline.

2. **Adapter pattern for solvers**: `ForwardSolverAdapter` dynamically imports forward solver modules and exposes a uniform `run_forward()` API. This decouples the inverse pipeline from specific solver implementations.

3. **Uniform surrogate API**: All surrogate types (`BVSurrogateModel`, `PODRBFSurrogateModel`, `NNSurrogateModel`, `EnsembleMeanWrapper`) implement the same `predict(k0_1, k0_2, alpha_1, alpha_2)` and `predict_batch(parameters)` interface, enabling drop-in substitution.

4. **Log-space k0 everywhere**: Rate constants k0 are always transformed to `log10(k0)` in the optimizer space because they span orders of magnitude (1e-7 to 1.0). Alpha values stay in linear space [0.1, 0.9].

5. **Resilient minimization**: The PDE-based inverse path uses a multi-phase recovery loop (max_it escalation -> anisotropy reduction -> tolerance relaxation) to handle Newton solver divergence.

6. **Checkpointing and warm-starting**: Training data generation supports resume from checkpoint and cross-sample warm-starting (reusing converged solutions as initial conditions for nearby parameter samples).

7. **Multi-observable weighting**: The surrogate objective uses a `secondary_weight` parameter to balance current_density vs peroxide_current. Different weights favor different parameter recoveries (CD-dominant recovers k0_1/alpha_1, PC-dominant recovers k0_2).

### Dependencies

- **Firedrake** (PDE assembly, Newton solvers, adjoint): `firedrake`, `firedrake.adjoint`, PETSc/SNES nonlinear solvers
- **NumPy** -- Array operations throughout
- **SciPy** -- `scipy.optimize.minimize` (L-BFGS-B), `scipy.interpolate.RBFInterpolator`, `scipy.stats.qmc.LatinHypercube`, `scipy.linalg` (SVD via numpy)
- **PyTorch** (optional) -- NN surrogate model training and inference
- **Matplotlib** -- Plotting and visualization
- **h5py** -- Data I/O
- **pyadjoint** (via Firedrake) -- Automatic differentiation for adjoint-based gradients

### Multi-Phase Pipeline Architecture

The inference pipeline uses a **multi-phase surrogate-to-PDE cascade**:

**Phase 1: Surrogate-based coarse search**
- Generate training data: LHS samples in 4D parameter space -> PDE forward solves -> (params, I-V curve) pairs
- Train surrogate(s): RBF baseline, POD-RBF (SVD + per-mode RBF), NN ensemble (5x ResNet-MLP)
- Coarse optimization: Multi-start LHS (20,000 points) -> batch surrogate evaluation -> top-20 L-BFGS-B polish
- Alternative: Cascade inference (CD-dominant -> PC-dominant -> joint polish) or BCD

**Phase 2: Surrogate refinement with observable weighting**
- Cascade per-observable inference exploits finding that:
  - Low `secondary_weight` (CD-dominant) recovers k0_1, alpha_1 well
  - High `secondary_weight` (PC-dominant) recovers k0_2 well
- BCD alternates: Block 1 (reaction 1, low weight) <-> Block 2 (reaction 2, high weight)

**Phase 3: PDE-based refinement (optional)**
- Takes surrogate-optimized parameters as initial guess
- Uses Firedrake adjoint (`ReducedFunctional`) with true PDE forward solves
- Optimization via `firedrake.adjoint.minimize` with L-BFGS-B
- Resilient minimization with recovery retries

**Observables**: Two I-V curves measured simultaneously:
- `current_density`: total electrode current density vs applied voltage
- `peroxide_current`: peroxide species current vs applied voltage

### Surrogate Models

| Model | Architecture | Input | Output | Training | Key Features |
|-------|-------------|-------|--------|----------|-------------|
| `BVSurrogateModel` | Direct RBF (thin_plate_spline) | 4D: [k0_1, k0_2, alpha_1, alpha_2] | 2 x n_eta (CD + PC curves) | Exact/smoothed interpolation | Baseline, fast to fit |
| `PODRBFSurrogateModel` | SVD + per-mode RBF | Same 4D | Same output | SVD truncation (99.9% variance) + per-mode optimized smoothing via LOO/k-fold CV | Better generalization, optional log1p PC transform |
| `NNSurrogateModel` | ResNet-MLP (4->128->4xResBlock->64->44) | Same 4D (log10 k0) | Same output (Z-score normalized) | AdamW + cosine annealing, early stopping, 5000 epochs | Smooth, differentiable, good extrapolation |
| `EnsembleMeanWrapper` | Mean of 5 NN members | Same | Mean prediction + uncertainty (std) | 5 independent seeds | Uncertainty quantification |

**Training data generation**: LHS sampling -> PDE forward solves (with continuation strategies for convergence) -> ~200 training samples with 22 voltage points each. Supports parallel generation with warm-starting chains.

### Optimization Methods

**Surrogate-based (adjoint-free)**:
- **L-BFGS-B** via `scipy.optimize.minimize` -- Primary optimizer at all stages
- **Finite-difference gradients**: Central differences with step size `fd_step=1e-5` (8 surrogate evals per 4D gradient)
- **Multi-start**: 20,000 LHS points -> batch surrogate eval -> top-20 L-BFGS-B polish
- **Cascade**: Sequential 3-pass with observable-weighted objectives
- **BCD**: Alternating 2D sub-problems with convergence monitoring

**PDE-based (adjoint)**:
- **L-BFGS-B** via `firedrake.adjoint.minimize` -- Uses pyadjoint for exact adjoint gradients through the PDE solve tape
- **Resilient minimization**: Up to 15 recovery attempts cycling through max_it escalation, anisotropy reduction, tolerance relaxation, line search schedule changes
- **Log-diffusivity controls**: `m[i] = log(D_i)`, `D_i = exp(m[i])` -- ensures positivity and better conditioning

### Integration Points

1. **Surrogate-to-PDE handoff** (`FluxCurve/bv_run/pipelines.py`): Surrogate-optimized parameters feed into PDE-based refinement as initial guesses.

2. **Forward solver interface** (`Inverse/solver_interface.py`): Any forward solver (Dirichlet, Robin, BV) can be plugged into the inverse pipeline via `ForwardSolverAdapter`.

3. **Surrogate API boundary** (`Surrogate/*.py`): All surrogates share `predict()`/`predict_batch()` API -- new surrogate types can be swapped in without changing the optimization code.

4. **Objective function layer** (`Surrogate/objectives.py`): Clean separation between surrogate evaluation and optimization logic. `secondary_weight` parameter controls observable weighting.

5. **Nondimensionalization** (`Nondim/transform.py`): Centralized scaling transform ensures consistent units between physical inputs and model-space PDE assembly.

6. **Training data pipeline** (`Surrogate/training.py`): Decoupled from surrogate fitting -- training data can be generated once and reused across model types.

### Gaps

1. **No Bayesian inference**: No MCMC, Hamiltonian Monte Carlo, or Bayesian optimization. Only point estimates via L-BFGS-B. No posterior distributions or credible intervals for recovered parameters.

2. **No adjoint through the surrogate**: Surrogate gradients use finite differences (8 evals for 4D). An analytically differentiable surrogate (e.g., NN with autograd) could provide exact gradients for the surrogate-based optimization.

3. **No active learning / adaptive sampling**: Training data uses fixed LHS designs. No iterative refinement of the surrogate in regions of high objective sensitivity.

4. **No multi-fidelity methods**: No coarse-mesh / fine-mesh hierarchical surrogates. Each PDE solve uses the same mesh resolution.

5. **No reduced-order model with error bounds**: POD-RBF is used but without rigorous a-posteriori error estimators for the reduced model.

6. **No Gaussian Process surrogate**: GP would provide built-in uncertainty quantification and could be used for Bayesian optimization. Current NN ensemble provides uncertainty but not in a principled Bayesian framework.

7. **No trust-region methods**: All optimization uses L-BFGS-B. Trust-region methods could better handle the transition from surrogate to PDE objectives.

8. **No regularization in the inverse problem**: The objective is pure data misfit (L2 norm). No Tikhonov regularization, TV regularization, or prior terms.

9. **Limited to 4 parameters**: The current pipeline is specialized for `[k0_1, k0_2, alpha_1, alpha_2]`. Extending to additional parameters (diffusivities, steric coefficients, etc.) would require generalizing the surrogate input/output structure.

10. **No sensitivity-based experimental design**: No optimal experiment design for choosing voltage ranges or measurement configurations that maximize parameter identifiability.

### Key Takeaways

- The codebase implements a complete PDE-constrained inverse problem pipeline for electrochemical kinetics parameter inference, using a Firedrake-based PNP-BV forward solver with three continuation strategies for Newton convergence.

- Three surrogate model types (RBF, POD-RBF, NN ensemble) share a uniform API and map a 4D parameter space `[k0_1, k0_2, alpha_1, alpha_2]` to dual I-V curve observables (current density + peroxide current).

- The multi-phase inference strategy (multi-start LHS -> cascade/BCD -> PDE refinement) is well-structured, with the key insight that different observable weightings favor different parameter recoveries. L-BFGS-B is the universal optimizer.

- The adjoint-based PDE path uses Firedrake's pyadjoint for exact gradients, with a resilient minimization wrapper that handles Newton solver failures through staged recovery.

- Primary opportunities for improvement: Bayesian uncertainty quantification, analytic surrogate gradients (NN autograd), active learning for training data, multi-fidelity hierarchies, and regularization terms in the objective.

- The 4-species nondimensionalized PNP-BV system with Butler-Volmer electrode kinetics, steric effects, and multi-reaction stoichiometry represents a sophisticated and physically realistic electrochemistry model.
