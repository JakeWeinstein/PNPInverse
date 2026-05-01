# 4-05: Convergence Monitoring, Diagnostics & ISMO Runner Script

## Goal

Build the convergence monitoring infrastructure, per-iteration diagnostics, and a top-level runner script (`scripts/surrogate/run_ismo.py`) that ties together the full ISMO loop: initial surrogate training, iterative solve-acquire-retrain cycle, convergence checking, and post-ISMO validation. The system must produce clear, auditable evidence of whether ISMO improved k0_2 recovery by at least 50% relative to Phase 3 best, stayed within the 200 PDE-solve budget, and converged monotonically.

## Files to Create

| File | Purpose |
|------|---------|
| `Surrogate/ismo_convergence.py` | `ISMOConvergenceChecker` class + convergence metrics |
| `Surrogate/ismo_diagnostics.py` | Per-iteration logging, scatter plots, convergence curves |
| `scripts/surrogate/run_ismo.py` | Top-level CLI runner that orchestrates the full loop |

## Files to Modify

| File | Change |
|------|--------|
| `Surrogate/__init__.py` | Export `ISMOConvergenceChecker`, `ISMODiagnosticRecord` |

## Dependencies on Other 4-0x Plans

- **4-01 (ISMO Loop Core)**: Provides `ISMOConfig`, `ISMOIteration`, `ISMOResult`, and `run_ismo()`. This plan reuses 4-01's canonical dataclasses rather than redefining them. The convergence checker consumes `ISMOIteration` records. The runner script calls `run_ismo()` if available, or orchestrates primitives directly.
- **4-02 (Acquisition Strategy)**: Provides `AcquisitionConfig`, `AcquisitionResult`, and `select_new_samples(existing_data, bounds, config, multistart_result, cascade_result, gp_model)`. Diagnostics in 4-05 log which acquisition strategy was used and how many samples it produced.
- **4-03 (Surrogate Retraining)**: Provides `retrain_surrogate()` (or equivalent). The runner calls this after each batch of new PDE solves. Diagnostics track pre- vs post-retrain surrogate error.
- **4-04 (PDE Solver Integration)**: Provides the PDE evaluation function used to solve new candidate points. The convergence checker tracks the cumulative PDE solve budget via the count returned by 4-04.

If the other plans are not yet implemented, 4-05 defines clean interfaces (function signatures and dataclasses) that the runner expects, allowing stub implementations during development.

## Detailed Task Breakdown

### [REVISED] Task 1: `ISMODiagnosticRecord` Dataclass (was `ISMOIterationRecord`)

**File**: `Surrogate/ismo_convergence.py`

A frozen dataclass capturing **diagnostic-only** fields that extend 4-01's canonical `ISMOIteration` via composition. This does NOT redefine `ISMOIteration` -- it references one.

**Rationale for composition over redefinition**: 4-01 defines the canonical `ISMOIteration` with core loop state (iteration count, sample counts, losses, convergence metric, best params, wall time). This plan needs additional diagnostic fields (timing breakdowns, test set metrics, acquisition metadata) that are specific to the monitoring/diagnostics layer. Rather than duplicating or conflicting with 4-01's definition, we wrap it.

```python
from Surrogate.ismo import ISMOIteration  # canonical from 4-01

@dataclass(frozen=True)
class ISMODiagnosticRecord:
    """Extends ISMOIteration with diagnostics-layer metadata.

    References the canonical ISMOIteration from 4-01 via composition.
    """
    core: ISMOIteration                  # the canonical record from 4-01

    # Additional convergence diagnostics (not in 4-01)
    surrogate_pde_agreement: float       # NRMSE of surrogate vs PDE at optimizer solution
    parameter_stability: float           # L2 distance in normalized log-space from prev iter
    surrogate_test_nrmse_cd: float       # NRMSE on held-out test set (CD)
    surrogate_test_nrmse_pc: float       # NRMSE on held-out test set (PC)
    objective_improvement: float         # ratio: prev_pde_obj / current_pde_obj (>1 means improved)
                                         # NOTE: float('nan') for iteration 0 (no previous)

    # Timing breakdowns
    pde_solve_time_s: float
    retrain_time_s: float
    inference_time_s: float

    # Acquisition info
    acquisition_strategy: str            # e.g. "optimizer_trajectory", "hybrid"
    acquisition_details: tuple           # tuple of (key, value) pairs for immutability
```

**Changes from original plan**:
- Removed standalone `ISMOIterationRecord` -- replaced with `ISMODiagnosticRecord` that holds a reference to 4-01's `ISMOIteration`.
- `acquisition_details` changed from `dict` to `tuple` of key-value pairs for true immutability in a frozen dataclass.
- `objective_improvement` documented as `float('nan')` for iteration 0.

### Task 2: `ISMOConvergenceChecker` Class

**File**: `Surrogate/ismo_convergence.py`

```python
@dataclass(frozen=True)
class ISMOConvergenceCriteria:
    # Primary: surrogate-PDE agreement at optimizer solution
    agreement_tol: float = 0.01           # NRMSE < 1%
    # Secondary: parameter estimate stability
    stability_tol: float = 0.01           # L2 distance in normalized log-space
    # Tertiary: surrogate test error not degrading
    test_error_degradation_tol: float = 0.05  # allow up to 5% relative increase
    # Budget cap
    max_pde_evals: int = 200
    # Max iterations
    max_iterations: int = 10
    # Minimum iterations before convergence can be declared
    min_iterations: int = 2
    # [REVISED] Stagnation window -- configurable with minimum of 2
    stagnation_window: int = 3            # default 3, minimum 2
    # [REVISED] Minimum useful batch size for budget check
    min_useful_batch_size: int = 5        # stop if remaining budget < this
```

The `ISMOConvergenceChecker` class:

```python
class ISMOConvergenceChecker:
    def __init__(self, criteria: ISMOConvergenceCriteria | None = None):
        self.criteria = criteria or ISMOConvergenceCriteria()
        # [REVISED] Validate stagnation_window >= 2
        if self.criteria.stagnation_window < 2:
            raise ValueError(
                f"stagnation_window must be >= 2, got {self.criteria.stagnation_window}"
            )
        self.history: list[ISMODiagnosticRecord] = []

    def record_iteration(self, record: ISMODiagnosticRecord) -> None:
        """Append a new iteration record to history."""

    def check_convergence(self) -> tuple[bool, str]:
        """Check all convergence criteria. Returns (converged, reason)."""
        # Returns (True, reason) if any stopping criterion is met:
        # 1. Budget exceeded: n_total_pde_evals >= max_pde_evals
        # 2. [REVISED] Budget too low for useful work:
        #    remaining_budget < min_useful_batch_size
        # 3. Max iterations reached
        # 4. Primary convergence: agreement < tol AND stability < tol
        #    (requires min_iterations met)
        # 5. [REVISED] Stagnation: last `stagnation_window` iterations show
        #    no improvement in agreement (uses configurable window, default 3)

    def is_budget_exhausted(self) -> bool:
        """Check if PDE evaluation budget is exhausted."""

    def remaining_budget(self) -> int:
        """Return number of PDE evaluations remaining."""

    def get_convergence_summary(self) -> dict:
        """Return a JSON-serializable summary of convergence state."""
```

**Key design decisions**:

1. **[REVISED] Normalized log-space for parameter stability**: k0 values span orders of magnitude, so raw L2 distance is meaningless. We normalize: `x_norm = [(log10(k0_1) - log10(k0_1_lo)) / (log10(k0_1_hi) - log10(k0_1_lo)), (log10(k0_2) - log10(k0_2_lo)) / (log10(k0_2_hi) - log10(k0_2_lo)), (alpha_1 - alpha_1_lo) / (alpha_1_hi - alpha_1_lo), (alpha_2 - alpha_2_lo) / (alpha_2_hi - alpha_2_lo)]`. This correctly maps each dimension to [0, 1].

2. **[REVISED] Stagnation detection**: If the last `stagnation_window` iterations (configurable, default 3, minimum 2) show less than 5% relative improvement in `surrogate_pde_agreement`, declare stagnation and stop. This prevents wasting PDE budget when ISMO has plateaued. Making it configurable allows tuning for different acquisition strategies (trust-region may need a larger window).

3. **Minimum iterations**: Require at least 2 iterations before declaring convergence. A single iteration cannot establish stability.

4. **Budget check happens first**: Even if convergence criteria are met, if the budget is exceeded we stop and report "budget_exhausted" as the reason.

### Task 3: Surrogate-PDE Agreement Metric

**File**: `Surrogate/ismo_convergence.py`

A standalone function that computes the primary convergence metric:

```python
def compute_surrogate_pde_agreement(
    surrogate,
    params_physical: np.ndarray,       # shape (4,) -- optimizer's best solution
    pde_solver_fn,                      # callable: params -> {"current_density": ..., "peroxide_current": ...}
    phi_applied: np.ndarray,
) -> dict:
    """Evaluate surrogate and PDE at the same point, return NRMSE metrics.

    Returns dict with:
        'cd_nrmse': float
        'pc_nrmse': float
        'combined_nrmse': float  (max of cd and pc)
        'cd_surrogate': np.ndarray
        'cd_pde': np.ndarray
        'pc_surrogate': np.ndarray
        'pc_pde': np.ndarray
    """
```

This requires exactly 1 PDE evaluation per call. The runner uses this after each iteration's optimization to measure how trustworthy the surrogate is at the optimizer's solution.

**Rationale**: The surrogate could have low global error but be locally wrong at the optimizer's solution (especially in the k0_2-sensitive region). This point-wise check is the critical convergence signal.

### Task 4: Per-Iteration Diagnostics

**File**: `Surrogate/ismo_diagnostics.py`

```python
class ISMODiagnostics:
    def __init__(self, output_dir: str = "StudyResults/ismo"):
        self.output_dir = output_dir

    def log_iteration(self, record: ISMODiagnosticRecord) -> None:
        """Append iteration data to a CSV log file."""
        # Writes to: {output_dir}/ismo_iteration_log.csv
        # Columns: all ISMODiagnosticRecord fields (flattened from core + diagnostic)

    def plot_surrogate_vs_pde_scatter(
        self,
        iteration: int,
        candidate_params: np.ndarray,     # (N_candidates, 4)
        surrogate_objectives: np.ndarray,  # (N_candidates,)
        pde_objectives: np.ndarray,        # (N_candidates,)
    ) -> None:
        """Scatter plot: surrogate obj vs PDE obj at candidate points.

        Saved to: {output_dir}/iter_{iteration:02d}_surrogate_vs_pde.png
        Includes 1:1 line, R^2 annotation, and NRMSE annotation.
        """

    def plot_iv_comparison_at_best(
        self,
        iteration: int,
        phi_applied: np.ndarray,
        surrogate_cd: np.ndarray,
        pde_cd: np.ndarray,
        surrogate_pc: np.ndarray,
        pde_pc: np.ndarray,
    ) -> None:
        """2-panel I-V curve overlay (surrogate vs PDE) at optimizer's best point.

        Saved to: {output_dir}/iter_{iteration:02d}_iv_comparison.png
        """

    def save_iteration_state(
        self,
        iteration: int,
        record: ISMODiagnosticRecord,
        new_params: np.ndarray,
        new_cd: np.ndarray,
        new_pc: np.ndarray,
    ) -> None:
        """Save per-iteration NPZ with new training points and metadata.

        Saved to: {output_dir}/iter_{iteration:02d}_state.npz
        """
```

### Task 5: Convergence Curve Visualization

**File**: `Surrogate/ismo_diagnostics.py`

```python
def plot_convergence_curves(
    history: list[ISMODiagnosticRecord],
    output_path: str = "StudyResults/ismo/convergence_curves.png",
) -> None:
    """Generate 2x2 summary figure after ISMO completes.

    Subplots:
    1. Top-left: Surrogate-PDE agreement vs iteration (log scale y-axis)
       - Horizontal dashed line at agreement_tol
       - Points colored by convergence status (green=met, red=not)
    2. Top-right: Parameter estimates vs iteration (4 lines, one per param)
       - k0_1, k0_2 on log10 scale; alpha_1, alpha_2 on linear scale
       - Horizontal dashed lines at true values (if known)
    3. Bottom-left: Training set size vs iteration (bar chart)
       - Stacked bars: existing samples + new samples per iteration
    4. Bottom-right: Parameter stability metric vs iteration
       - Horizontal dashed line at stability_tol

    Saved to output_path.
    """
```

```python
def plot_k0_2_recovery_comparison(
    phase3_errors: dict,    # {"0pct_noise": float, "1pct_noise": float}
    ismo_errors: dict,      # same structure
    output_path: str = "StudyResults/ismo/k0_2_improvement.png",
) -> None:
    """Bar chart comparing k0_2 recovery error: Phase 3 vs ISMO.

    Shows the target 50% improvement line.
    """
```

### Task 6: ISMO Runner Script

**File**: `scripts/surrogate/run_ismo.py`

The top-level entry point. This is the most complex piece -- it ties together all ISMO components.

**CLI Arguments**:

```
--surrogate-type     {nn_ensemble, pod_rbf_log, pod_rbf_nolog, rbf_baseline, gp}
                     Default: nn_ensemble
--design             Design name for NN ensemble (e.g., D1-default, D3-deeper)
                     Only used when surrogate-type=nn_ensemble
--training-data      Path to initial training data .npz
                     Default: data/surrogate_models/training_data_merged.npz
--max-iterations     Maximum ISMO iterations (default: 10)
--budget             Maximum new PDE evaluations (default: 200)
--agreement-tol      Surrogate-PDE agreement tolerance (default: 0.01)
--stability-tol      Parameter stability tolerance (default: 0.01)
--acquisition        {trust_region, exploit_explore, error_based}
                     Default: trust_region
--samples-per-iter   New PDE samples per ISMO iteration (default: 20)
--stagnation-window  Iterations without improvement before stopping (default: 3, min: 2)
--output-dir         Output directory (default: StudyResults/ismo)
--seed               Random seed (default: 42)
--skip-post-validation  Skip post-ISMO parameter recovery study
--verbose            Print detailed progress
```

**Script Structure**:

```python
def main():
    args = parse_args()

    # ── Step 0: Load initial training data, surrogate, and test data ──
    training_data = np.load(args.training_data)
    params_train = training_data["parameters"]
    cd_train = training_data["current_density"]
    pc_train = training_data["peroxide_current"]
    phi_applied = training_data["phi_applied"]

    # [REVISED] Load test data from split indices for validation
    split_indices = np.load("data/surrogate_models/split_indices.npz")
    test_idx = split_indices["test_idx"]
    test_params = params_train[test_idx]
    test_cd = cd_train[test_idx]
    test_pc = pc_train[test_idx]
    # Use only training subset for ISMO augmentation
    train_idx = split_indices["train_idx"]
    params_train = params_train[train_idx]
    cd_train = cd_train[train_idx]
    pc_train = pc_train[train_idx]

    # [REVISED] Load target data explicitly
    # Target comes from PDE solve at true parameters (for recovery test)
    # or from experimental data. Must be specified or generated.
    target_cd, target_pc = load_target_data(args)

    surrogate = load_or_build_surrogate(args)
    # load_or_build_surrogate dispatches to:
    #   - load_nn_ensemble() for nn_ensemble
    #   - load_surrogate() for RBF/POD-RBF models
    #   - load_gp_surrogate() for GP

    # ── Step 1: Setup solver config (same as generate_training_data.py) ──
    # Uses scripts._bv_common for solver params, steady config, mesh, etc.
    # Defines pde_solver_fn: params_physical -> {"current_density": ..., "peroxide_current": ...}

    # ── Step 2: Initialize convergence checker and diagnostics ────────
    criteria = ISMOConvergenceCriteria(
        agreement_tol=args.agreement_tol,
        stability_tol=args.stability_tol,
        max_pde_evals=args.budget,
        max_iterations=args.max_iterations,
        stagnation_window=args.stagnation_window,
    )
    checker = ISMOConvergenceChecker(criteria)
    diagnostics = ISMODiagnostics(output_dir=args.output_dir)

    # ── Step 3: ISMO Loop ─────────────────────────────────────────────
    for iteration in range(args.max_iterations):
        t_iter_start = time.time()

        # 3a. Run inference on current surrogate
        ms_result = run_multistart_inference(surrogate, target_cd, target_pc, ...)

        # 3b. Compute surrogate-PDE agreement at optimizer's solution
        #     This costs 1 PDE eval
        agreement = compute_surrogate_pde_agreement(
            surrogate, best_params, pde_solver_fn, phi_applied
        )

        # [REVISED] 3b'. Check if remaining budget allows useful work
        # The agreement eval above consumed 1 PDE eval from the budget.
        # Check BEFORE acquiring new samples to prevent off-by-one.
        budget_after_agreement = checker.remaining_budget() - 1  # subtract agreement eval
        if budget_after_agreement < criteria.min_useful_batch_size:
            # Not enough budget for a useful acquisition batch.
            # Record this iteration with agreement data, then stop.
            # (record creation below will capture the agreement eval)
            print(f"Remaining budget ({budget_after_agreement}) < "
                  f"min_useful_batch_size ({criteria.min_useful_batch_size}). Stopping.")
            # ... record final iteration with 0 new acquisition samples ...
            break

        # [REVISED] 3c. Acquire new sample points (delegates to 4-02)
        # Call signature aligned with 4-02's select_new_samples() API
        acq_config = AcquisitionConfig(
            budget=min(args.samples_per_iter, budget_after_agreement),
            # ... other config from args ...
        )
        acq_result = select_new_samples(
            existing_data=params_train,
            bounds=parameter_bounds,
            config=acq_config,
            multistart_result=ms_result,
            cascade_result=cascade_result,    # from cascade run if available
            gp_model=gp_model,               # None if not using GP
        )
        new_params = acq_result.samples
        # new_params is already capped by acq_config.budget which respects
        # budget_after_agreement

        # 3d. Evaluate PDE at new points (delegates to 4-04)
        new_cd, new_pc = evaluate_pde_batch(new_params, pde_solver_fn)
        # Costs len(new_params) PDE evals

        # 3e. Augment training data
        params_train = np.vstack([params_train, new_params])
        cd_train = np.vstack([cd_train, new_cd])
        pc_train = np.vstack([pc_train, new_pc])

        # 3f. Retrain surrogate (delegates to 4-03)
        surrogate = retrain_surrogate(surrogate, params_train, cd_train, pc_train, phi_applied)

        # 3g. Validate on held-out test set
        test_metrics = validate_surrogate(surrogate, test_params, test_cd, test_pc)

        # 3h. Record iteration using 4-01's ISMOIteration + diagnostic wrapper
        core_record = ISMOIteration(
            iteration=iteration,
            n_new_samples=len(new_params) + 1,  # +1 for agreement eval
            n_total_training=len(params_train),
            surrogate_loss_at_best=ms_result.best_loss,
            pde_loss_at_best=...,
            surrogate_pde_gap=agreement["combined_nrmse"],
            convergence_metric=agreement["combined_nrmse"],
            best_params=tuple(best_params),
            best_loss=...,
            candidate_pde_losses=...,
            candidate_surrogate_losses=...,
            retrain_val_rmse_cd=test_metrics.get("cd_mean_relative_error"),
            retrain_val_rmse_pc=test_metrics.get("pc_mean_relative_error"),
            wall_time_s=time.time() - t_iter_start,
        )
        diag_record = ISMODiagnosticRecord(
            core=core_record,
            surrogate_pde_agreement=agreement["combined_nrmse"],
            parameter_stability=compute_parameter_stability(checker.history, best_params, training_bounds),
            surrogate_test_nrmse_cd=test_metrics["cd_mean_relative_error"],
            surrogate_test_nrmse_pc=test_metrics["pc_mean_relative_error"],
            objective_improvement=compute_objective_improvement(checker.history, pde_loss),
            pde_solve_time_s=...,
            retrain_time_s=...,
            inference_time_s=...,
            acquisition_strategy=args.acquisition,
            acquisition_details=tuple(sorted(acq_result.__dict__.items())),
        )
        checker.record_iteration(diag_record)
        diagnostics.log_iteration(diag_record)
        diagnostics.plot_iv_comparison_at_best(...)
        diagnostics.save_iteration_state(...)

        # 3i. Check convergence
        converged, reason = checker.check_convergence()
        if converged:
            print(f"ISMO converged: {reason}")
            break

    # ── Step 4: Post-ISMO outputs ─────────────────────────────────────
    # Save augmented training data
    np.savez_compressed(f"{args.output_dir}/augmented_training_data.npz", ...)

    # [REVISED] Save retrained surrogate (type-aware dispatch)
    save_surrogate_typed(surrogate, args.surrogate_type, f"{args.output_dir}/ismo_surrogate")

    # Save convergence report
    report = checker.get_convergence_summary()
    with open(f"{args.output_dir}/convergence_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Plot convergence curves
    plot_convergence_curves(checker.history, f"{args.output_dir}/convergence_curves.png")

    # ── Step 5: Post-ISMO Validation ──────────────────────────────────
    if not args.skip_post_validation:
        run_post_ismo_validation(surrogate, args.output_dir)


def run_post_ismo_validation(surrogate, output_dir):
    """Run parameter recovery at 0% and 1% noise, compare to Phase 3 baseline."""
    # Phase 3 baseline from StudyResults/inverse_verification/parameter_recovery_summary.json:
    #   0% noise: surrogate_bias = 10.67% (k0_2 is the bottleneck)
    #   1% noise: median_max_relative_error = 17.69%

    # Run parameter recovery with ISMO-refined surrogate at 0% and 1% noise
    # using 3 realizations per noise level (seeds 42, 43, 44 -- matching Phase 3)
    # Record per-parameter relative errors

    # Compare k0_2 specifically:
    #   Phase 3 k0_2 error at 0% noise: ~10.67% (surrogate bias)
    #   Success criterion: ISMO k0_2 error < 10.67% * 0.5 = 5.34%

    # Save to: {output_dir}/parameter_recovery_comparison.json
    # Structure:
    # {
    #   "phase3_baseline": {
    #     "0pct": {"k0_1": ..., "k0_2": ..., "alpha_1": ..., "alpha_2": ..., "max_error": ...},
    #     "1pct": {...}
    #   },
    #   "ismo_refined": {
    #     "0pct": {...},
    #     "1pct": {...}
    #   },
    #   "k0_2_improvement_pct": ...,
    #   "success_criterion_met": bool
    # }

    plot_k0_2_recovery_comparison(phase3_errors, ismo_errors, f"{output_dir}/k0_2_improvement.png")
```

### [REVISED] Task 7: No separate `ISMOResult` (use 4-01's canonical version)

**Removed.** 4-01 defines the canonical `ISMOResult` dataclass in `Surrogate/ismo.py` with fields: `converged`, `termination_reason`, `n_iterations`, `total_pde_evals`, `iteration_history`, `final_params` (as tuple), `final_loss`, `final_surrogate_path`, `augmented_data_path`, `total_wall_time_s`. The runner script and convergence checker use this directly.

If 4-05 needs to attach diagnostic metadata to the result, it does so via the `ISMODiagnosticRecord` history stored in the `ISMOConvergenceChecker`, not by redefining `ISMOResult`.

### [REVISED] Task 8: Helper -- Parameter Stability Computation

**File**: `Surrogate/ismo_convergence.py`

```python
def compute_parameter_stability(
    history: list[ISMODiagnosticRecord],
    current_params: np.ndarray,
    training_bounds: dict,               # from surrogate.training_bounds
) -> float:
    """L2 distance in normalized log-space between current and previous iteration's best params.

    [REVISED] Normalization (correctly maps to [0, 1]):
        x_norm[0] = (log10(k0_1) - log10(k0_1_lo)) / (log10(k0_1_hi) - log10(k0_1_lo))
        x_norm[1] = (log10(k0_2) - log10(k0_2_lo)) / (log10(k0_2_hi) - log10(k0_2_lo))
        x_norm[2] = (alpha_1 - alpha_1_lo) / (alpha_1_hi - alpha_1_lo)
        x_norm[3] = (alpha_2 - alpha_2_lo) / (alpha_2_hi - alpha_2_lo)

    Returns 0.0 if no previous iteration exists.
    """
```

**Rationale**: Without normalization, a tiny change in k0_2 (which is ~1e-5 in physical space) would dominate or be invisible relative to alpha changes (~0.5). The min-max normalization `(val - lo) / (hi - lo)` correctly puts all dimensions on [0, 1] scale. The previous formula that divided `log10(k0_1)` by the log-range (without subtracting the lower bound) produced values outside [0, 1] for most inputs.

## Success Criteria

1. **Functional**: `python scripts/surrogate/run_ismo.py --max-iterations 2 --budget 50 --skip-post-validation` completes without error, produces:
   - `StudyResults/ismo/ismo_iteration_log.csv` with 2 rows
   - `StudyResults/ismo/convergence_report.json`
   - `StudyResults/ismo/convergence_curves.png`
   - `StudyResults/ismo/iter_00_iv_comparison.png`, `iter_01_iv_comparison.png`
   - `StudyResults/ismo/augmented_training_data.npz`
   - `StudyResults/ismo/ismo_surrogate.pkl` (or equivalent for NN)

2. **Convergence correctness**: Unit tests verify:
   - Budget-exceeded stops the loop at the right iteration
   - **[REVISED]** Budget too low for useful work (`remaining < min_useful_batch_size`) stops before acquisition
   - Agreement + stability convergence triggers only after `min_iterations`
   - **[REVISED]** Stagnation detection works with configurable window (default 3, minimum 2)
   - `remaining_budget()` is always non-negative

3. **k0_2 recovery**: Full ISMO run (with post-validation) shows k0_2 recovery error at 0% noise improved by at least 50% vs Phase 3 baseline (10.67% -> target < 5.34%).

4. **Budget compliance**: Total PDE evaluations across all iterations is <= 200.

5. **Monotonic convergence**: `surrogate_pde_agreement` in the convergence report is non-increasing across iterations (with tolerance for noise: no iteration increases by more than 10% relative).

## Key Design Decisions

### [REVISED] Why composition with 4-01's dataclasses instead of redefining?

4-01 defines the canonical `ISMOIteration` and `ISMOResult` dataclasses that are used across the ISMO subsystem. Redefining these in 4-05 would create naming collisions, import ambiguity, and duplicated logic. Instead, 4-05 defines `ISMODiagnosticRecord` which holds a reference to 4-01's `ISMOIteration` and adds diagnostic-only fields. This follows the composition-over-inheritance principle and keeps the data model clean.

### Why a separate `ISMOConvergenceChecker` instead of inline logic?

The checker is independently testable, reusable, and keeps the runner script focused on orchestration rather than convergence math. It also makes it trivial to serialize the full history for post-hoc analysis.

### Why compute surrogate-PDE agreement at only the optimizer solution (not globally)?

Global surrogate error (from `validate_surrogate()`) measures average accuracy across the parameter space, but ISMO cares about **local** accuracy at the point the optimizer found. A surrogate with 5% global NRMSE could have 0.1% error at the optimum (which is fine) or 20% error at the optimum (which means the optimizer was misled). The point-wise check is the actionable signal.

### Why 1 PDE eval per iteration for agreement + N for acquisition?

The agreement eval is mandatory -- it is the primary convergence signal. The acquisition evals are the main budget cost. With a 200-eval budget and ~20 evals/iteration, we get ~9-10 iterations max, which is sufficient for convergence in typical adaptive surrogate workflows.

### [REVISED] Why check budget before acquisition?

The agreement eval consumes 1 PDE eval. If the remaining budget after the agreement eval is less than `min_useful_batch_size` (default 5), there is not enough budget to acquire a meaningful batch of new samples. The runner checks this immediately after the agreement eval and stops the loop if the budget is insufficient. This prevents the off-by-one scenario where the agreement eval + acquisition batch exceeds the budget cap by 1 on the final iteration.

### Why not use the ensemble uncertainty for convergence?

Ensemble disagreement (std across members) is useful for *acquisition* (4-02) but not reliable for *convergence*. Low ensemble disagreement means the members agree, but they could all be wrong in the same way (e.g., all trained on the same biased region). The PDE truth check is the only reliable convergence signal.

### Why save per-iteration state as separate NPZ files?

This enables resumability (if ISMO crashes at iteration 5, we can restart from iteration 4's state) and post-hoc debugging (inspect exactly which points were added at each iteration). The cumulative augmented training data is saved at the end for the final surrogate.

### Runner script approach: orchestration vs delegation

The runner script orchestrates at a high level but delegates each substep to the modules defined in 4-01 through 4-04. If those modules are not yet available, the runner can use inline stubs that:
- For acquisition (4-02): fall back to LHS sampling around the current best
- For retraining (4-03): rebuild the RBF surrogate from scratch with augmented data
- For PDE evaluation (4-04): call `generate_training_data_single()` directly

This ensures 4-05 is independently testable even if other plans are not yet implemented.

---

## Revision Log

| Date | Issue | Severity | Change |
|------|-------|----------|--------|
| 2026-03-17 | Overlapping dataclass definitions with 4-01 | HIGH | Removed standalone `ISMOIterationRecord` and `ISMOResult`. Replaced with `ISMODiagnosticRecord` that holds a reference to 4-01's canonical `ISMOIteration` via composition. Runner and convergence checker now import `ISMOIteration` and `ISMOResult` from `Surrogate.ismo` (4-01). |
| 2026-03-17 | Budget off-by-one on final iteration | HIGH | Added explicit check after agreement eval: if `remaining_budget - 1 < min_useful_batch_size`, stop before acquisition. Added `min_useful_batch_size` field to `ISMOConvergenceCriteria` (default 5). Moved budget check to step 3b' in the runner loop. |
| 2026-03-17 | Parameter stability normalization bug | MEDIUM | Fixed formula from `log10(val) / log10_range` to `(log10(val) - log10(lo)) / (log10(hi) - log10(lo))` for k0 dims, and `(val - lo) / (hi - lo)` for alpha dims. This correctly maps to [0, 1]. |
| 2026-03-17 | Stagnation window too aggressive | MEDIUM | Made stagnation window configurable via `ISMOConvergenceCriteria.stagnation_window` (default 3, minimum 2). Added validation in `ISMOConvergenceChecker.__init__()`. Added `--stagnation-window` CLI flag. |
| 2026-03-17 | Acquisition function signature mismatch with 4-02 | MEDIUM | Replaced `acquire_new_samples(surrogate, best_params, budget, strategy, n_samples)` with `select_new_samples(existing_data, bounds, config, multistart_result, cascade_result, gp_model)` matching 4-02's `AcquisitionConfig`/`AcquisitionResult` API. Runner now constructs an `AcquisitionConfig` and passes optimizer results directly. |
| 2026-03-17 | `acquisition_details: dict` in frozen dataclass | LOW | Changed to `tuple` of key-value pairs for true immutability. |
| 2026-03-17 | `objective_improvement` undefined for iteration 0 | MEDIUM | Documented as `float('nan')` for iteration 0. |
| 2026-03-17 | Test set not loaded in runner | LOW | Added explicit loading from `split_indices.npz` in Step 0. |
| 2026-03-17 | Target data not loaded in runner | LOW | Added `load_target_data(args)` call in Step 0. |
| 2026-03-17 | `__init__.py` exports updated | LOW | Changed exports from `ISMOIterationRecord, ISMOResult` to `ISMOConvergenceChecker, ISMODiagnosticRecord` (canonical `ISMOIteration`/`ISMOResult` are exported by 4-01). |
