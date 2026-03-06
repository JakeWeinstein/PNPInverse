# Codebase Structure

**Analysis Date:** 2026-03-06

## Directory Layout

```
PNPInverse/
├── Forward/                # PDE forward solvers (Dirichlet, Robin, Butler-Volmer)
│   ├── __init__.py         # Re-exports all solver APIs
│   ├── params.py           # SolverParams frozen dataclass
│   ├── dirichlet_solver.py # Dirichlet BC solver
│   ├── robin_solver.py     # Robin BC solver
│   ├── bv_solver/          # Butler-Volmer BC solver (multi-module)
│   │   ├── __init__.py     # Re-exports BV solver API
│   │   ├── config.py       # BV config parsing (_get_bv_cfg, _get_bv_reactions_cfg)
│   │   ├── forms.py        # Weak form assembly (build_context, build_forms, set_initial_conditions)
│   │   ├── mesh.py         # Graded mesh construction (make_graded_interval_mesh, make_graded_rectangle_mesh)
│   │   ├── nondim.py       # BV-specific nondimensionalization transforms
│   │   └── solvers.py      # Time-stepping + continuation solvers (forsolve_bv, solve_bv_with_continuation)
│   ├── steady_state/       # Steady-state sweep utilities
│   │   ├── __init__.py     # Re-exports steady-state API
│   │   ├── common.py       # SteadyStateConfig, SteadyStateResult, shared helpers
│   │   ├── robin.py        # Robin-BC steady-state sweep
│   │   └── bv.py           # BV-BC steady-state sweep (configure_bv_solver_params, sweep)
│   ├── noise.py            # Noise injection for synthetic data
│   └── plotter.py          # Visualization (plot_solutions, create_animations)
├── Inverse/                # PDE-constrained inverse inference framework
│   ├── __init__.py         # Re-exports inverse API
│   ├── solver_interface.py # ForwardSolverAdapter, deep_copy_solver_params
│   ├── parameter_targets.py # ParameterTarget registry (what to infer)
│   ├── objectives.py       # Objective functions for diffusion/dirichlet/robin inference
│   └── inference_runner/   # End-to-end inference orchestration
│       ├── __init__.py     # run_inverse_inference() orchestrator
│       ├── config.py       # InferenceRequest, InferenceResult, RecoveryConfig, SyntheticData
│       ├── data.py         # build_default_solver_params, generate_synthetic_data
│       ├── objective.py    # build_reduced_functional (adjoint-based)
│       ├── recovery.py     # resilient_minimize with retry strategies
│       └── formatting.py   # Log formatting helpers
├── FluxCurve/              # BV I-V curve inference with adjoint gradients
│   ├── __init__.py         # Re-exports FluxCurve API
│   ├── config.py           # ForwardRecoveryConfig, RobinFluxCurveInferenceRequest
│   ├── bv_config.py        # BVFluxCurveInferenceRequest
│   ├── results.py          # Result containers (PointAdjointResult, CurveAdjointResult)
│   ├── recovery.py         # Recovery strategies (clip_kappa, solver relaxation)
│   ├── observables.py      # Observable form construction (flux, current density)
│   ├── point_solve.py      # Single-point objective+gradient (Robin)
│   ├── bv_point_solve/     # BV-specific point solve with caching + parallelism
│   │   ├── __init__.py
│   │   ├── cache.py        # Solution caching
│   │   ├── forward.py      # BV forward solve wrapper
│   │   ├── parallel.py     # Multiprocessing pool management
│   │   └── predictor.py    # Initial-guess predictor from previous solves
│   ├── bv_curve_eval.py    # BV curve residual+Jacobian evaluation
│   ├── bv_observables.py   # BV observable forms (current density, peroxide current)
│   ├── curve_eval.py       # Robin curve evaluation
│   ├── replay.py           # Replay-based (reuse-solution) curve evaluator
│   ├── plot.py             # Live fit plotting (_LiveFitPlot, export_live_fit_gif)
│   ├── run.py              # Robin pipeline (run_robin_kappa_flux_curve_inference)
│   └── bv_run/             # BV pipeline modules
│       ├── __init__.py     # Re-exports BV pipeline functions
│       ├── io.py           # Target curve I/O, normalization
│       ├── optimization.py # Scipy optimizer dispatch
│       └── pipelines.py    # Pipeline functions (run_bv_k0/alpha/joint/steric/full/multi_obs/multi_ph)
├── Nondim/                 # Nondimensionalization layer
│   ├── __init__.py         # Re-exports nondim API
│   ├── constants.py        # Physical constants (F, R, epsilon_0, etc.)
│   ├── scales.py           # NondimScales dataclass, build_physical_scales()
│   ├── transform.py        # build_model_scaling(), verify_model_params()
│   └── compat.py           # Backward-compat wrappers (build_physical_scales_dict)
├── Surrogate/              # Surrogate models for fast inference
│   ├── __init__.py         # Re-exports surrogate API
│   ├── sampling.py         # ParameterBounds, generate_lhs_samples
│   ├── surrogate_model.py  # BVSurrogateModel (RBF interpolation)
│   ├── nn_model.py         # NNSurrogateModel (PyTorch ResNet-MLP)
│   ├── nn_training.py      # NN training loop (EarlyStopping, train_nn_surrogate)
│   ├── ensemble.py         # EnsembleMeanWrapper, load_nn_ensemble
│   ├── pod_rbf_model.py    # POD+RBF model variant
│   ├── training.py         # Training data generation (PDE-based)
│   ├── validation.py       # validate_surrogate, print_validation_report
│   ├── io.py               # save_surrogate, load_surrogate (pickle/torch)
│   ├── objectives.py       # Surrogate objective classes (full, alpha-only, block, subset)
│   ├── cascade.py          # CascadeConfig/Result, run_cascade_inference (3-pass)
│   ├── multistart.py       # MultiStartConfig/Result, run_multistart_inference (LHS grid)
│   └── bcd.py              # BCDConfig/Result, run_block_coordinate_descent
├── scripts/                # Executable experiment scripts
│   ├── _bv_common.py       # Shared constants, species presets, solver param factories
│   ├── bv/                 # BV I-V curve generation scripts
│   │   ├── bv_iv_curve.py
│   │   ├── bv_iv_curve_charged.py
│   │   └── bv_iv_curve_symmetric.py
│   ├── inference/          # Legacy PDE-only inference scripts (v1-v7, various strategies)
│   │   ├── Infer_BVMaster_charged.py
│   │   ├── Infer_BVMaster_charged_v2.py ... v7.py
│   │   ├── Infer_BVJoint_*.py
│   │   ├── Infer_BVk0_*.py
│   │   ├── Infer_BVAlpha_*.py
│   │   ├── Infer_BVStaged_*.py
│   │   ├── Infer_BVSteric_*.py
│   │   ├── Infer_BVFull_*.py
│   │   ├── Infer_BVHybrid_*.py
│   │   ├── Infer_BVMultiObs_*.py
│   │   ├── Infer_BVMultiPH_*.py
│   │   ├── Infer_BVProfileLikelihood_*.py
│   │   └── Infer_D_from_data*.py, Infer_DirichletBC_*.py, Infer_Robin*.py
│   ├── surrogate/          # Surrogate-accelerated inference (v8-v13, latest)
│   │   ├── Infer_BVMaster_charged_v13_ultimate.py  # PRIMARY: 7-phase surr+PDE pipeline
│   │   ├── Infer_BVMaster_charged_v12_nn_surrogate_pde.py
│   │   ├── Infer_BVMaster_charged_v11_surrogate_pde.py
│   │   ├── Infer_BVMaster_charged_v10_fixed_pde.py
│   │   ├── Infer_BVMaster_charged_v8_surrogate.py ... v9
│   │   ├── generate_training_data.py
│   │   ├── build_surrogate.py
│   │   ├── train_nn_surrogate.py
│   │   ├── train_improved_surrogate.py
│   │   ├── validate_surrogate.py
│   │   ├── cascade_inference.py
│   │   ├── multistart_inference.py
│   │   ├── bcd_inference.py
│   │   ├── cascade_pde_hybrid.py
│   │   ├── sweep_secondary_weight.py
│   │   └── overnight_train_v11.py
│   ├── studies/            # Benchmarks, sensitivity analysis, feasibility studies
│   │   ├── bv_k0_noise_sensitivity.py
│   │   ├── charged_voltage_range_study.py
│   │   ├── forward_solver_D_stability_study.py
│   │   ├── optimization_method_study.py
│   │   ├── profile_likelihood_study.py
│   │   └── benchmark_*.py
│   └── verification/       # Method of Manufactured Solutions (MMS) convergence tests
│       ├── mms_bv_4species.py
│       ├── mms_bv_convergence.py
│       └── test_bv_forward.py
├── tests/                  # pytest test suite
│   ├── conftest.py         # Shared fixtures
│   ├── test_bv_forward.py
│   ├── test_cascade.py
│   ├── test_cascade_pde_hybrid.py
│   ├── test_ensemble_and_v12.py
│   ├── test_fixed_pde.py
│   ├── test_inference_config.py
│   ├── test_inference_robustness.py
│   ├── test_multistart.py
│   ├── test_nondim.py
│   ├── test_params.py
│   ├── test_steady_state_common.py
│   ├── test_v11_e2e_pde.py
│   ├── test_v11_surrogate_pde.py
│   ├── test_v13_verification.py
│   ├── test_weight_sweep.py
│   └── test_bcd.py
├── StudyResults/           # Experiment outputs (CSVs, PNGs, model artifacts)
├── Renders/                # Animation outputs
├── docs/                   # Research documentation and session logs
├── writeups/               # LaTeX papers and markdown writeups
├── pyproject.toml          # Package config (setuptools, pytest markers)
├── README.md               # Project overview
└── .gitignore
```

## Directory Purposes

**`Forward/`:**
- Purpose: All PDE forward solver code
- Contains: Three solver families (Dirichlet, Robin, BV), each following `build_context` -> `build_forms` -> `set_initial_conditions` -> `forsolve` pattern
- Key files: `params.py` (SolverParams), `bv_solver/solvers.py` (BV time-stepping), `bv_solver/forms.py` (weak form assembly)

**`Inverse/`:**
- Purpose: Adjoint-based PDE-constrained parameter inference framework
- Contains: Solver adapter, parameter target registry, inference request/result config, resilient optimization
- Key files: `solver_interface.py` (ForwardSolverAdapter), `inference_runner/__init__.py` (run_inverse_inference)

**`FluxCurve/`:**
- Purpose: I-V curve fitting via adjoint gradients on BV forward solves
- Contains: Point-level and curve-level objective/gradient evaluation, parallel point solving, recovery strategies, live plotting
- Key files: `bv_run/pipelines.py` (pipeline entry points), `bv_point_solve/parallel.py` (multiprocessing), `bv_curve_eval.py` (residual+Jacobian)

**`Nondim/`:**
- Purpose: Single source of truth for physical constants and nondimensionalization
- Contains: Constants, scale computation, physical-to-model transforms
- Key files: `constants.py`, `scales.py` (NondimScales), `transform.py` (build_model_scaling)

**`Surrogate/`:**
- Purpose: Fast surrogate models replacing PDE solves for rapid parameter optimization
- Contains: RBF and NN models, training/validation, multiple optimization strategies, serialization
- Key files: `surrogate_model.py` (BVSurrogateModel), `nn_model.py` (NNSurrogateModel), `ensemble.py` (EnsembleMeanWrapper), `cascade.py`, `multistart.py`

**`scripts/`:**
- Purpose: Executable experiment entry points organized by type
- Contains: BV curve generation, legacy inference (v1-v7), surrogate inference (v8-v13), studies/benchmarks, MMS verification
- Key files: `_bv_common.py` (shared constants/factories), `surrogate/Infer_BVMaster_charged_v13_ultimate.py` (primary pipeline)

**`tests/`:**
- Purpose: pytest test suite for all packages
- Contains: Unit tests for params, nondim, steady-state; integration tests for inference pipelines, surrogate strategies, forward solver
- Key files: `conftest.py` (fixtures), `test_v13_verification.py` (latest pipeline tests)

**`StudyResults/`:**
- Purpose: Output directory for experiment results
- Contains: Per-study subdirectories with CSVs, PNGs, trained model artifacts, LaTeX report generators
- Generated: Yes (by scripts)
- Committed: Partially (key results committed, large model files likely gitignored)

## Key File Locations

**Entry Points:**
- `scripts/surrogate/Infer_BVMaster_charged_v13_ultimate.py`: Primary 7-phase inference pipeline
- `scripts/surrogate/generate_training_data.py`: Training data generation
- `scripts/surrogate/train_nn_surrogate.py`: NN surrogate training

**Configuration:**
- `pyproject.toml`: Package metadata, pytest config, package discovery
- `scripts/_bv_common.py`: Physical constants, species presets, solver param factories, SNES options
- `Forward/params.py`: SolverParams frozen dataclass definition

**Core Logic:**
- `Forward/bv_solver/forms.py`: BV weak form assembly (PDE definition)
- `Forward/bv_solver/solvers.py`: BV time-stepping and continuation solvers
- `FluxCurve/bv_run/pipelines.py`: BV inference pipeline functions
- `Surrogate/cascade.py`: 3-pass cascade inference strategy
- `Surrogate/multistart.py`: LHS grid search + L-BFGS-B polish
- `Surrogate/objectives.py`: All surrogate objective function classes

**Testing:**
- `tests/conftest.py`: Shared pytest fixtures
- `tests/test_v13_verification.py`: Latest pipeline verification tests
- `tests/test_cascade.py`: Cascade strategy tests
- `tests/test_ensemble_and_v12.py`: NN ensemble + v12 pipeline tests

## Naming Conventions

**Files:**
- PascalCase for major packages: `Forward/`, `Inverse/`, `FluxCurve/`, `Nondim/`, `Surrogate/`
- snake_case for all `.py` files within packages: `solver_interface.py`, `surrogate_model.py`
- Inference scripts use `Infer_BV{Strategy}_{variant}_v{N}.py` pattern
- Private/internal functions and modules prefixed with `_`: `_bv_common.py`, `_make_bv_convergence_cfg()`

**Directories:**
- PascalCase for library packages (importable): `Forward/`, `Inverse/`, `Surrogate/`
- snake_case for subpackages: `bv_solver/`, `bv_point_solve/`, `bv_run/`, `inference_runner/`, `steady_state/`
- lowercase for non-library directories: `scripts/`, `tests/`, `docs/`, `writeups/`

**Classes:**
- PascalCase: `SolverParams`, `BVSurrogateModel`, `ForwardSolverAdapter`, `CascadeConfig`
- Frozen dataclasses for all config and result containers

**Functions:**
- snake_case: `build_context()`, `make_bv_solver_params()`, `run_cascade_inference()`
- Private helpers prefixed with `_`: `_make_subset_objective_fn()`, `_run_pass1()`

## Where to Add New Code

**New inference strategy (e.g., Bayesian optimization):**
- Implementation: `Surrogate/bayesian.py` (config dataclass + result dataclass + `run_bayesian_inference()` function)
- Re-export in: `Surrogate/__init__.py`
- Script: `scripts/surrogate/bayesian_inference.py`
- Tests: `tests/test_bayesian.py`

**New forward solver variant:**
- Implementation: `Forward/new_solver.py` following `build_context()` / `build_forms()` / `set_initial_conditions()` / `forsolve_new()` pattern
- Re-export in: `Forward/__init__.py`
- Must be compatible with `ForwardSolverAdapter.from_module_path()`

**New surrogate model type:**
- Implementation: `Surrogate/new_model.py` implementing `fit()`, `predict(k0_1, k0_2, alpha_1, alpha_2) -> dict`, `predict_batch(params) -> dict`
- Must expose `n_eta`, `phi_applied`, `is_fitted`, `training_bounds` properties
- Re-export in: `Surrogate/__init__.py`

**New objective function:**
- Implementation: `Surrogate/objectives.py` (add new class following `SurrogateObjective` pattern with `objective()`, `gradient()`, `objective_and_gradient()` methods)
- Or `FluxCurve/bv_curve_eval.py` for PDE-based objectives

**New species configuration:**
- Add preset to `scripts/_bv_common.py` as frozen `SpeciesConfig` dataclass instance
- Follow pattern of `TWO_SPECIES_NEUTRAL` and `FOUR_SPECIES_CHARGED`

**New test:**
- Location: `tests/test_{feature}.py`
- Use `@pytest.mark.slow` for tests requiring Firedrake FEM environment
- Import fixtures from `tests/conftest.py`

**Utilities:**
- Shared script helpers: `scripts/_bv_common.py` (if script-specific)
- Nondimensionalization helpers: `Nondim/` package
- Forward solver helpers: appropriate submodule in `Forward/`

## Special Directories

**`StudyResults/`:**
- Purpose: All experiment output artifacts (CSVs, plots, trained models)
- Generated: Yes, by inference and study scripts
- Committed: Partially (key results committed for reproducibility)

**`Renders/`:**
- Purpose: Animation output directory
- Generated: Yes, by `Forward/plotter.py`
- Committed: No (empty)

**`__pycache__/`:**
- Purpose: Python bytecode cache
- Generated: Yes
- Committed: No (gitignored)

**`.pytest_cache/`:**
- Purpose: pytest cache for test rerun optimization
- Generated: Yes
- Committed: No (gitignored)

---

*Structure analysis: 2026-03-06*
