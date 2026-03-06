# Architecture

**Analysis Date:** 2026-03-06

## Pattern Overview

**Overall:** Scientific computing pipeline with layered forward-inverse-surrogate architecture

**Key Characteristics:**
- Poisson-Nernst-Planck (PNP) forward PDE solver built on Firedrake FEM library
- Inverse parameter estimation via PDE-constrained optimization (adjoint gradients) and surrogate-accelerated inference
- Multi-phase inference pipeline: surrogate warm-start followed by PDE refinement
- Frozen dataclass configuration pattern throughout (immutable config objects)
- Nondimensionalization layer decouples physical units from solver numerics

## Layers

**Nondimensionalization (`Nondim/`):**
- Purpose: Convert physical parameters (m/s, mol/m3, V) to dimensionless solver inputs and back
- Location: `Nondim/`
- Contains: Physical constants (`constants.py`), scale computation (`scales.py`), parameter transforms (`transform.py`), backward-compat shim (`compat.py`)
- Depends on: numpy only
- Used by: `Forward/`, `scripts/_bv_common.py`, all inference scripts

**Forward Solvers (`Forward/`):**
- Purpose: Solve the coupled PNP system of PDEs for ion transport + electrostatics
- Location: `Forward/`
- Contains: Three solver families (Dirichlet, Robin, Butler-Volmer), steady-state sweep utilities, noise injection, plotting
- Depends on: Firedrake (FEM), PETSc (nonlinear/linear solvers), `Nondim/`
- Used by: `Inverse/`, `FluxCurve/`, inference scripts
- Key abstraction: Every solver exposes `build_context()` -> `build_forms()` -> `set_initial_conditions()` -> `forsolve_*()` pipeline

**Inverse Framework (`Inverse/`):**
- Purpose: PDE-constrained parameter identification via adjoint-based optimization
- Location: `Inverse/`
- Contains: `ForwardSolverAdapter` (pluggable solver interface), `ParameterTarget` (what to infer), `InferenceRequest`/`InferenceResult` (config + output), resilient minimization with recovery strategies
- Depends on: `Forward/`, Firedrake-adjoint, scipy.optimize
- Used by: Older inference scripts in `scripts/inference/`

**FluxCurve (BV Curve Inference) (`FluxCurve/`):**
- Purpose: Infer Butler-Volmer kinetics parameters (k0, alpha) from I-V curve data using adjoint gradients
- Location: `FluxCurve/`
- Contains: Point-solve with caching/parallelism (`bv_point_solve/`), curve evaluation (`bv_curve_eval.py`), least-squares optimization pipelines (`bv_run/`), recovery strategies, live plotting
- Depends on: `Forward/`, Firedrake, scipy, multiprocessing
- Used by: PDE refinement phases in `scripts/surrogate/` scripts

**Surrogate Models (`Surrogate/`):**
- Purpose: Fast approximate I-V curve prediction to replace expensive PDE solves during optimization
- Location: `Surrogate/`
- Contains: RBF interpolation model (`surrogate_model.py`), PyTorch NN ResNet-MLP model (`nn_model.py`), ensemble wrapper (`ensemble.py`), multiple optimization strategies (cascade, BCD, multistart), objective functions, training data generation, validation
- Depends on: numpy, scipy, PyTorch (optional), `Forward/` (for training data generation only)
- Used by: `scripts/surrogate/` inference scripts

**Shared Script Utilities (`scripts/_bv_common.py`):**
- Purpose: Centralize physical constants, species configurations, solver parameter factories, and SNES options shared across 35+ inference scripts
- Location: `scripts/_bv_common.py`
- Contains: Physical constants (F, R, V_T), species presets (TWO_SPECIES_NEUTRAL, FOUR_SPECIES_CHARGED), `make_bv_solver_params()` factory, `make_recovery_config()` factory, SNES option dicts
- Depends on: `Forward/`, `FluxCurve/`, `Nondim/`
- Used by: All scripts in `scripts/bv/`, `scripts/inference/`, `scripts/surrogate/`

## Data Flow

**Full v13 Inference Pipeline (primary workflow):**

1. Script loads physical constants and species config from `scripts/_bv_common.py`
2. `make_bv_solver_params()` constructs `SolverParams` frozen dataclass with nondimensionalized values
3. Load pre-trained surrogate model (RBF or NN ensemble) from `StudyResults/`
4. **Surrogate phases (S1-S5):** Run cascade, multistart, and joint L-BFGS-B on surrogate to find warm-start parameters (~15s total)
5. **PDE phases (P1-P2):** Use warm-start as initial guess for `FluxCurve.run_bv_multi_observable_flux_curve_inference()` which runs full Firedrake PDE solves with adjoint gradients
6. Each PDE point-solve: `build_context_bv()` -> `build_forms_bv()` -> `set_initial_conditions_bv()` -> `solve_bv_with_continuation()` -> extract current density
7. Results saved to `StudyResults/` as CSV files and plots

**Forward Solve (single point):**

1. `build_context(solver_params)` creates Firedrake mesh, function spaces, mixed function space, boundary conditions
2. `build_forms(ctx, solver_params)` assembles weak form (Nernst-Planck transport + Poisson + Butler-Volmer electrode BC)
3. `set_initial_conditions(ctx, solver_params)` populates initial state (uniform or blob)
4. `forsolve_bv(ctx, solver_params)` time-steps to steady state using PETSc SNES Newton solver

**Surrogate Training:**

1. `generate_training_dataset()` in `Surrogate/training.py` samples parameter space via Latin Hypercube
2. For each sample, run full PDE forward solve to get I-V curves
3. `BVSurrogateModel.fit()` (RBF) or `NNSurrogateModel.fit()` (PyTorch) learns parameter-to-curve mapping
4. `save_surrogate()` / `load_surrogate()` persist to disk via pickle/torch

**State Management:**
- `SolverParams` frozen dataclass is the central state object; mutation via `.with_*()` methods returns new instances
- Forward solver `ctx` dict carries Firedrake mesh, function spaces, forms, and solution state through the solve pipeline
- Surrogate models are stateful (fitted) objects with `predict()` API
- All config objects are frozen dataclasses (immutable)

## Key Abstractions

**SolverParams:**
- Purpose: Immutable container for all PDE solver inputs (species count, diffusivities, charges, voltages, solver options, BV config, nondim config)
- Examples: `Forward/params.py`
- Pattern: Frozen dataclass with `.with_*()` mutation helpers, backward-compatible list indexing

**ForwardSolverAdapter:**
- Purpose: Pluggable interface to swap between Dirichlet, Robin, and BV solvers
- Examples: `Inverse/solver_interface.py`
- Pattern: Adapter pattern with `from_module_path()` factory; resolves `build_context`, `build_forms`, `set_initial_conditions`, `solve` from any compatible module

**SpeciesConfig:**
- Purpose: Immutable preset for species properties (charges, diffusivities, concentrations, stoichiometry)
- Examples: `scripts/_bv_common.py` -- `TWO_SPECIES_NEUTRAL`, `FOUR_SPECIES_CHARGED`
- Pattern: Frozen dataclass used as factory input for `make_bv_solver_params()`

**Surrogate API contract:**
- Purpose: Unified interface for any surrogate model (RBF, NN, ensemble)
- Examples: `Surrogate/surrogate_model.py` (BVSurrogateModel), `Surrogate/nn_model.py` (NNSurrogateModel), `Surrogate/ensemble.py` (EnsembleMeanWrapper)
- Pattern: Duck-typed API -- any object with `predict(k0_1, k0_2, alpha_1, alpha_2) -> dict` and `predict_batch(params) -> dict` works with objectives and optimization strategies

**Objective functions:**
- Purpose: Compute loss and gradient for parameter optimization
- Examples: `Surrogate/objectives.py` (SurrogateObjective, AlphaOnlySurrogateObjective, ReactionBlockSurrogateObjective, SubsetSurrogateObjective), `FluxCurve/bv_curve_eval.py`
- Pattern: Class with `objective(x)`, `gradient(x)`, `objective_and_gradient(x)` methods; central FD gradients for surrogate, adjoint gradients for PDE

**Optimization strategies:**
- Purpose: Different approaches to navigate the 4D parameter space
- Examples: `Surrogate/cascade.py` (CascadeConfig/CascadeResult), `Surrogate/multistart.py` (MultiStartConfig/MultiStartResult), `Surrogate/bcd.py` (BCDConfig/BCDResult)
- Pattern: Frozen config dataclass + result dataclass + `run_*()` entry function

## Entry Points

**Inference Scripts:**
- Location: `scripts/surrogate/Infer_BVMaster_charged_v13_ultimate.py` (latest, primary)
- Triggers: `python scripts/surrogate/Infer_BVMaster_charged_v13_ultimate.py [--no-pde] [--model-type rbf|nn] [--compare]`
- Responsibilities: Orchestrate the 7-phase surrogate+PDE pipeline, save results to `StudyResults/`

**BV I-V Curve Generation:**
- Location: `scripts/bv/bv_iv_curve_charged.py`, `scripts/bv/bv_iv_curve.py`
- Triggers: Direct execution
- Responsibilities: Generate synthetic I-V curves from PDE forward solves

**Surrogate Training:**
- Location: `scripts/surrogate/generate_training_data.py`, `scripts/surrogate/build_surrogate.py`, `scripts/surrogate/train_nn_surrogate.py`
- Triggers: Direct execution
- Responsibilities: Generate training data via PDE sweeps, fit surrogate models

**Study/Benchmark Scripts:**
- Location: `scripts/studies/`
- Triggers: Direct execution
- Responsibilities: Parameter sensitivity studies, optimization method comparisons, convergence benchmarks

**Tests:**
- Location: `tests/`
- Triggers: `pytest` (some tests marked `slow` require Firedrake)
- Responsibilities: Unit and integration tests for all packages

## Error Handling

**Strategy:** Multi-level recovery with progressive solver relaxation

**Patterns:**
- Forward solve failures trigger `ForwardRecoveryConfig` retry logic: increase SNES iterations, relax tolerances, cycle line search methods, reduce parameter anisotropy (`FluxCurve/recovery.py`, `Inverse/inference_runner/recovery.py`)
- `resilient_minimize()` in `Inverse/inference_runner/recovery.py` wraps scipy.optimize with multiple recovery attempts, catching solver divergence and retrying with perturbed initial guesses
- BV solver uses voltage continuation (`solve_bv_with_continuation` in `Forward/bv_solver/solvers.py`) to handle stiff nonlinear problems by ramping applied potential
- Surrogate objectives mask NaN targets via `~np.isnan()` boolean arrays, allowing partial I-V curve fitting
- Concentration floor regularization (`conc_floor: 1e-12`) and exponent clipping (`exponent_clip: 50.0`) prevent numerical overflow in Butler-Volmer exponentials

## Cross-Cutting Concerns

**Logging:** Print-based logging throughout (`print()` with `[tag]` prefixes). No structured logging framework.

**Validation:** Assertion-based validation in surrogate model `fit()`. Schema validation via frozen dataclass construction (type normalization in `SolverParams.__post_init__`). No formal input validation framework.

**Configuration:** Nested dict inside `solver_options` field of `SolverParams` carries PETSc options, BV boundary conditions, convergence settings, and nondimensionalization config. Factory functions in `scripts/_bv_common.py` centralize construction.

**Parallelism:** `FluxCurve/bv_point_solve/parallel.py` provides multiprocessing pool for parallel point solves across voltage values. `set_parallel_pool()` / `close_parallel_pool()` manage lifecycle.

---

*Architecture analysis: 2026-03-06*
