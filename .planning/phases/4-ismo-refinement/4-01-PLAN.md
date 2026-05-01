# 4-01: ISMO Core Loop & Configuration

## Goal

Implement `Surrogate/ismo.py` -- the core Iterative Surrogate Model Optimization
(ISMO) orchestration module. This module wraps the existing multistart + cascade
surrogate optimization pipeline in an outer loop that iteratively:

1. Optimizes the current surrogate to find candidate parameters.
2. Evaluates those candidates with the true PDE solver.
3. Measures surrogate-vs-PDE discrepancy at the candidates.
4. Augments the training data with new (parameter, PDE output) pairs.
5. Retrains the surrogate on the expanded dataset.
6. Checks convergence and repeats or stops.

The goal is to concentrate surrogate accuracy in the regions the optimizer
actually visits, eliminating the "surrogate-optimal != PDE-optimal" gap
without requiring a dense global training set. Budget: at most 200 new
PDE solves total.

Reference: Lye, Mishra, Ray (2020) -- "Iterative surrogate model optimization."

---

## Files to Create

| File | Purpose |
|------|---------|
| `Surrogate/ismo.py` | Core ISMO loop, dataclasses, orchestration logic |

## Files to Modify

| File | Change |
|------|--------|
| `Surrogate/__init__.py` | Export `ISMOConfig`, `ISMOResult`, `ISMOIteration`, `run_ismo` |

---

## Detailed Task Breakdown

### Task 1: Define `AcquisitionStrategy` Enum

```python
class AcquisitionStrategy(str, Enum):
    OPTIMIZER_TRAJECTORY = "optimizer_trajectory"
    UNCERTAINTY = "uncertainty"
    HYBRID = "hybrid"
```

**Semantics:**
- `OPTIMIZER_TRAJECTORY`: New sample points come from the top-K optimizer
  solutions (multistart candidates + cascade result). These are the points
  where the optimizer "thinks" the answer is -- exactly where surrogate
  accuracy matters most.
- `UNCERTAINTY`: Sample where the surrogate is most uncertain. For NN
  ensembles, use inter-member std from `predict_with_uncertainty()`. For
  GP, use posterior variance from `predict_batch_with_uncertainty()`. For
  RBF/POD-RBF/PCE (no native uncertainty), fall back to `OPTIMIZER_TRAJECTORY`.
- `HYBRID`: Take half from optimizer trajectory and half from uncertainty-
  weighted LHS sampling in a neighborhood of the current best.

**Rationale:** `OPTIMIZER_TRAJECTORY` is the default and most directly follows
the ISMO paper. `UNCERTAINTY` and `HYBRID` are provided for experimentation
but are secondary priorities.

---

### Task 2: Define `ISMOConfig` Dataclass (frozen)

```python
@dataclass(frozen=True)
class ISMOConfig:
    max_iterations: int = 5
    samples_per_iteration: int = 30
    total_pde_budget: int = 200
    convergence_rtol: float = 0.05
    convergence_atol: float = 1e-4
    surrogate_type: str = "nn_ensemble"
    acquisition_strategy: AcquisitionStrategy = AcquisitionStrategy.OPTIMIZER_TRAJECTORY
    warm_start_retrain: bool = True
    retrain_epochs: int = 200
    multistart_config: MultiStartConfig | None = None
    cascade_config: CascadeConfig | None = None
    n_top_candidates_to_evaluate: int = 10
    neighborhood_fraction: float = 0.1
    output_dir: str = "data/ismo_runs"
    verbose: bool = True
```

**Field details:**

| Field | Description |
|-------|-------------|
| `max_iterations` | Hard cap on ISMO outer iterations. 5 is sufficient given the 200-PDE budget. |
| `samples_per_iteration` | Number of new PDE solves per iteration. 30 default gives ~6 iterations at budget 200. |
| `total_pde_budget` | Absolute maximum PDE evaluations across all iterations. Early stop if exhausted. |
| `convergence_rtol` | Relative tolerance for convergence check: `|J_surr - J_pde| / J_pde < rtol` at best candidate. |
| `convergence_atol` | Absolute tolerance floor: `|J_surr - J_pde| < atol` (handles near-zero loss). |
| `surrogate_type` | One of `"nn_ensemble"`, `"pod_rbf_log"`, `"pod_rbf_nolog"`, `"rbf_baseline"`, `"gp"`, `"pce"`. Used by retraining dispatch. |
| `acquisition_strategy` | How to select new sample points (see Task 1). |
| `warm_start_retrain` | If True, retrain from current weights (NN) or just refit (RBF/POD). Warm start is critical for NN to avoid full retraining cost. |
| `retrain_epochs` | Number of epochs for warm-start NN retraining. Ignored for non-NN surrogates. |
| `multistart_config` | Override for the inner multistart optimizer. If None, uses defaults from `MultiStartConfig()`. |
| `cascade_config` | Override for the inner cascade optimizer. If None, uses defaults from `CascadeConfig()`. |
| `n_top_candidates_to_evaluate` | From the multistart's top-20 polished candidates, evaluate this many with the PDE solver. Remaining samples come from acquisition_strategy. |
| `neighborhood_fraction` | For HYBRID/UNCERTAINTY, fraction of the parameter range to define the sampling neighborhood around the current best. |
| `output_dir` | Directory for saving augmented training data and retrained models per iteration. |
| `verbose` | Print progress. |

**Design decisions:**
- Frozen dataclass matches the project convention (see `MultiStartConfig`, `CascadeConfig`).
- Default `samples_per_iteration=30` balances iteration count vs. per-iteration information gain. At 5-30s per PDE solve, each iteration takes 2.5-15 minutes.
- `convergence_rtol=0.05` means we accept the surrogate when it agrees with the PDE to within 5% at the optimizer's best point.

---

### Task 3: Define `ISMOIteration` Dataclass (frozen)

```python
@dataclass(frozen=True)
class ISMOIteration:
    iteration: int
    n_new_samples: int
    n_total_training: int
    surrogate_loss_at_best: float
    pde_loss_at_best: float
    surrogate_pde_gap: float
    convergence_metric: float
    best_params: tuple  # (k0_1, k0_2, alpha_1, alpha_2)
    best_loss: float
    candidate_pde_losses: tuple  # PDE loss at each evaluated candidate
    candidate_surrogate_losses: tuple
    retrain_val_rmse_cd: float | None
    retrain_val_rmse_pc: float | None
    wall_time_s: float
```

**Field details:**

| Field | Description |
|-------|-------------|
| `iteration` | 0-indexed iteration number. |
| `n_new_samples` | PDE evaluations added this iteration. |
| `n_total_training` | Total training set size after augmentation. |
| `surrogate_loss_at_best` | Surrogate objective at the best candidate (before retraining). |
| `pde_loss_at_best` | PDE objective at the same point. |
| `surrogate_pde_gap` | `|surrogate_loss - pde_loss| / max(pde_loss, atol)`. The key convergence diagnostic. |
| `convergence_metric` | Same as `surrogate_pde_gap` (alias for clarity in result). |
| `best_params` | The best parameter set found by the optimizer this iteration. |
| `best_loss` | The PDE loss at the best candidate (ground truth). |
| `candidate_pde_losses` | PDE losses at all evaluated candidates (for diagnostics). |
| `candidate_surrogate_losses` | Corresponding surrogate losses. |
| `retrain_val_rmse_cd / _pc` | Validation RMSE after retraining (None if retraining skipped on last iteration). |
| `wall_time_s` | Wall clock time for this iteration. |

---

### Task 4: Define `ISMOResult` Dataclass (frozen)

```python
@dataclass(frozen=True)
class ISMOResult:
    converged: bool
    termination_reason: str
    n_iterations: int
    total_pde_evals: int
    iteration_history: tuple  # tuple[ISMOIteration, ...]
    final_params: tuple  # (k0_1, k0_2, alpha_1, alpha_2)
    final_loss: float
    final_surrogate_path: str | None
    augmented_data_path: str | None
    total_wall_time_s: float
```

**Termination reasons** (string enum):
- `"converged"`: surrogate-PDE gap below tolerance.
- `"budget_exhausted"`: total PDE evaluations reached `total_pde_budget`.
- `"max_iterations"`: reached `max_iterations` without convergence.
- `"no_improvement"`: PDE loss at best candidate did not improve for 2 consecutive iterations.

---

### Task 5: Implement Acquisition Functions (private)

#### 5a: `_acquire_optimizer_trajectory()`

```
_acquire_optimizer_trajectory(
    multistart_result: MultiStartResult,
    cascade_result: CascadeResult | None,
    n_candidates: int,
    n_total: int,
    bounds: dict,
    seed: int,
) -> np.ndarray  # shape (n_total, 4)
```

**Logic:**
1. Collect the top `n_candidates` from `multistart_result.candidates` (by polished_loss).
2. If cascade_result is provided, prepend its best point.
3. Deduplicate (within 1e-6 relative tolerance in log-k0 space).
4. If fewer than `n_total` unique candidates, fill remaining slots with
   LHS samples from a shrunk neighborhood (10% of full range, centered on best).
5. Clip all samples to bounds.
6. Return `(n_total, 4)` array in physical space.

#### 5b: `_acquire_uncertainty()`

```
_acquire_uncertainty(
    surrogate: Any,
    n_total: int,
    bounds: dict,
    best_params: np.ndarray,
    neighborhood_fraction: float,
    seed: int,
) -> np.ndarray
```

**Logic:**
1. Generate 5000 LHS candidate points within the neighborhood.
2. If surrogate has `predict_with_uncertainty()` or `predict_batch_with_uncertainty()`:
   compute total uncertainty = mean(cd_std) + mean(pc_std) at each point.
3. Select the top `n_total` by uncertainty.
4. If surrogate lacks uncertainty method, fall back to `_acquire_optimizer_trajectory`.

#### 5c: `_acquire_hybrid()`

Split `n_total` 50/50 between optimizer trajectory and uncertainty sampling.

---

### Task 6: Implement PDE Evaluation Helper (private)

```
_evaluate_candidates_pde(
    candidates: np.ndarray,        # (M, 4) physical space
    phi_applied: np.ndarray,
    target_cd: np.ndarray,
    target_pc: np.ndarray,
    pde_solver_kwargs: dict,       # base_solver_params, steady, observable_scale, mesh
    secondary_weight: float,
    subset_idx: np.ndarray | None,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[bool]]
    # Returns: (cd_curves, pc_curves, pde_losses, converged_flags)
```

**Logic:**
1. Loop over candidates sequentially (PDE solver is not thread-safe with Firedrake).
2. Call `generate_training_data_single()` for each candidate.
3. Compute the PDE objective (same formula as surrogate objective) using the
   true PDE I-V curves against target_cd/target_pc.
4. Return raw curves (for training data augmentation) and losses (for diagnostics).
5. Print per-candidate timing if verbose.

**Design decision:** Sequential, not parallel. The ISMO iteration calls this
with 10-50 points, and Firedrake process pool setup overhead would dominate at
this scale. The parallel path in `training.py` is designed for 100+ point batches.
For future optimization (4-02 or beyond), candidates could be batched in the
parallel generator.

---

### Task 7: Implement Surrogate Retraining Dispatch (private)

```
_retrain_surrogate(
    surrogate: Any,
    surrogate_type: str,
    parameters: np.ndarray,      # full augmented training set
    current_density: np.ndarray,
    peroxide_current: np.ndarray,
    phi_applied: np.ndarray,
    config: ISMOConfig,
    iteration: int,
    output_dir: str,
) -> tuple[Any, dict]  # (new_surrogate, retrain_metrics)
```

**Logic by surrogate type:**

| Type | Retraining approach |
|------|-------------------|
| `nn_ensemble` | Warm-start each member from current weights. Train for `retrain_epochs` with reduced LR (1e-4). Save to `{output_dir}/iter_{i}/nn_ensemble/`. Uses `train_nn_surrogate()` from `nn_training.py` with existing model weights loaded. |
| `pod_rbf_log` / `pod_rbf_nolog` | Full refit (`model.fit(...)`) -- POD-RBF is fast to refit from scratch. Save pickle to `{output_dir}/iter_{i}/`. |
| `rbf_baseline` | Full refit (`model.fit(...)`). Save pickle. |
| `gp` | Full refit -- GPyTorch hyperparameter optimization from current hyperparams (warm start kernel params). |
| `pce` | Full refit -- PCE is fast. |

**Warm-start NN detail:**
- Clone the existing model's state dict.
- Create a new `NNTrainingConfig` with `epochs=retrain_epochs`, `lr=1e-4` (reduced),
  `patience=50`.
- Call `train_nn_surrogate()` passing the full augmented dataset plus val split.
- For ensemble: retrain each member independently, rebuild `EnsembleMeanWrapper`.

**Returns:** `(new_surrogate, {"val_cd_rmse": float, "val_pc_rmse": float})`.

---

### Task 8: Implement Training Data Augmentation (private)

```
_augment_training_data(
    existing_params: np.ndarray,    # (N, 4)
    existing_cd: np.ndarray,        # (N, n_eta)
    existing_pc: np.ndarray,        # (N, n_eta)
    new_params: np.ndarray,         # (M, 4)
    new_cd: np.ndarray,             # (M, n_eta)
    new_pc: np.ndarray,             # (M, n_eta)
    converged_mask: list[bool],
    output_path: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]
```

**Logic:**
1. Filter new data to only converged samples.
2. Deduplicate against existing training data (match within 1e-8 in all 4 params).
3. Concatenate: `np.vstack([existing, new_valid])`.
4. If `output_path` provided, save augmented data as `.npz`.
5. Return `(augmented_params, augmented_cd, augmented_pc)`.

**Immutability:** Creates new arrays, never modifies inputs.

---

### Task 9: Implement Convergence Check (private)

```
_check_convergence(
    surrogate_loss: float,
    pde_loss: float,
    config: ISMOConfig,
) -> tuple[bool, float]
    # Returns: (converged, convergence_metric)
```

**Logic:**
```python
gap = abs(surrogate_loss - pde_loss)
denominator = max(abs(pde_loss), config.convergence_atol)
metric = gap / denominator
converged = metric < config.convergence_rtol
return converged, metric
```

**Rationale:** The surrogate-PDE gap at the optimizer's best point is the
natural convergence diagnostic for ISMO. When the surrogate is accurate where
it matters (at the optimum), further iteration provides diminishing returns.
The `atol` floor prevents division-by-near-zero when PDE loss is very small.

---

### Task 10: Implement `run_ismo()` Main Entry Point

```python
def run_ismo(
    surrogate: Any,
    target_cd: np.ndarray,
    target_pc: np.ndarray,
    training_params: np.ndarray,
    training_cd: np.ndarray,
    training_pc: np.ndarray,
    phi_applied: np.ndarray,
    bounds_k0_1: tuple[float, float],
    bounds_k0_2: tuple[float, float],
    bounds_alpha: tuple[float, float],
    pde_solver_kwargs: dict,
    config: ISMOConfig | None = None,
    subset_idx: np.ndarray | None = None,
) -> ISMOResult:
```

**Orchestration pseudocode:**

```
config = config or ISMOConfig()
total_pde_evals = 0
current_surrogate = surrogate
current_params, current_cd, current_pc = training_params, training_cd, training_pc
iterations = []
best_pde_loss_global = inf

for i in range(config.max_iterations):
    t0 = time.time()

    # --- Step 1: Optimize current surrogate ---
    multistart_result = run_multistart_inference(
        current_surrogate, target_cd, target_pc,
        bounds_k0_1, bounds_k0_2, bounds_alpha,
        config=config.multistart_config, subset_idx=subset_idx,
    )
    cascade_result = run_cascade_inference(
        current_surrogate, target_cd, target_pc,
        initial_k0=[multistart_result.best_k0_1, multistart_result.best_k0_2],
        initial_alpha=[multistart_result.best_alpha_1, multistart_result.best_alpha_2],
        bounds_k0_1=bounds_k0_1, bounds_k0_2=bounds_k0_2,
        bounds_alpha=bounds_alpha,
        config=config.cascade_config, subset_idx=subset_idx,
    )

    # --- Step 2: Select acquisition points ---
    n_remaining_budget = config.total_pde_budget - total_pde_evals
    n_samples = min(config.samples_per_iteration, n_remaining_budget)
    if n_samples <= 0:
        break  # budget exhausted

    candidates = _acquire_*(...)  # dispatch on config.acquisition_strategy

    # --- Step 3: Evaluate candidates with PDE solver ---
    cd_new, pc_new, pde_losses, converged = _evaluate_candidates_pde(
        candidates, phi_applied, target_cd, target_pc,
        pde_solver_kwargs, ...,
    )
    total_pde_evals += n_samples

    # --- Step 4: Measure surrogate-PDE gap at best candidate ---
    surr_losses = [surrogate_objective(c) for c in candidates]
    best_idx = np.argmin(pde_losses)
    converged_flag, gap = _check_convergence(
        surr_losses[best_idx], pde_losses[best_idx], config,
    )

    # --- Step 5: Augment training data ---
    current_params, current_cd, current_pc = _augment_training_data(
        current_params, current_cd, current_pc,
        candidates, cd_new, pc_new, converged,
        output_path=f"{config.output_dir}/iter_{i}/augmented_data.npz",
    )

    # --- Step 6: Record iteration ---
    iteration_result = ISMOIteration(
        iteration=i, n_new_samples=n_samples,
        n_total_training=len(current_params),
        surrogate_loss_at_best=surr_losses[best_idx],
        pde_loss_at_best=pde_losses[best_idx],
        surrogate_pde_gap=abs(surr_losses[best_idx] - pde_losses[best_idx]),
        convergence_metric=gap,
        best_params=tuple(candidates[best_idx]),
        best_loss=pde_losses[best_idx],
        candidate_pde_losses=tuple(pde_losses),
        candidate_surrogate_losses=tuple(surr_losses),
        retrain_val_rmse_cd=..., retrain_val_rmse_pc=...,
        wall_time_s=time.time() - t0,
    )
    iterations.append(iteration_result)

    # --- Step 7: Check convergence ---
    if converged_flag:
        return ISMOResult(converged=True, termination_reason="converged", ...)

    # --- Step 8: Retrain surrogate ---
    if total_pde_evals < config.total_pde_budget:
        current_surrogate, retrain_metrics = _retrain_surrogate(
            current_surrogate, config.surrogate_type,
            current_params, current_cd, current_pc, phi_applied,
            config, i, config.output_dir,
        )

    # --- Step 9: No-improvement check ---
    if pde_losses[best_idx] < best_pde_loss_global:
        best_pde_loss_global = pde_losses[best_idx]
        stall_count = 0
    else:
        stall_count += 1
    if stall_count >= 2:
        return ISMOResult(termination_reason="no_improvement", ...)

return ISMOResult(termination_reason="max_iterations" or "budget_exhausted", ...)
```

**Key design decisions:**

1. **Multistart + cascade per iteration:** Each ISMO iteration runs the full
   optimization pipeline (not just a single L-BFGS-B). This costs ~1 second
   of surrogate evaluations per iteration but ensures robust candidate
   selection.

2. **Convergence checked before retraining:** We measure the surrogate-PDE
   gap at the optimizer's best point BEFORE retraining. If it's already small
   enough, we skip retraining and return. This saves wasted compute.

3. **No-improvement early stop:** If the PDE loss at the best candidate does
   not improve for 2 consecutive iterations, we stop. This handles the case
   where the surrogate is being refined but the optimum isn't moving.

4. **Sequential PDE evaluation:** Firedrake is single-threaded per process
   and has heavy setup costs. For 10-50 evaluations, sequential is simpler
   and competitive with parallel dispatch.

5. **Immutable data flow:** All arrays are copied/concatenated, never mutated.
   The original training data passed to `run_ismo()` is never modified.

---

### Task 11: Update `Surrogate/__init__.py`

Add exports:
```python
from Surrogate.ismo import (
    AcquisitionStrategy,
    ISMOConfig,
    ISMOIteration,
    ISMOResult,
    run_ismo,
)
```

Add to `__all__`.

---

## Dependencies on Other 4-0x Plans

| Plan | Dependency |
|------|-----------|
| **4-02** (Acquisition Strategies) | 4-01 defines the `AcquisitionStrategy` enum and provides the `_acquire_*` function signatures. 4-02 may add more sophisticated acquisition (e.g., expected improvement, Thompson sampling). |
| **4-03** (ISMO Driver Script) | 4-03 creates the runnable script that calls `run_ismo()` with production settings. Depends on 4-01 being complete. |
| **4-04** (Warm-Start NN Retraining) | 4-01 sketches warm-start retraining in `_retrain_surrogate()`. 4-04 may refine the NN warm-start logic (learning rate schedule, layer freezing, etc.) if needed. |
| **4-05** (ISMO Diagnostics & Plotting) | 4-05 consumes `ISMOResult` and `ISMOIteration` dataclasses to produce convergence plots. |

4-01 is the foundational plan -- all other 4-0x plans depend on it.

---

## Success Criteria

1. `Surrogate/ismo.py` passes `python -c "from Surrogate.ismo import run_ismo, ISMOConfig"` without error.
2. All dataclasses (`ISMOConfig`, `ISMOIteration`, `ISMOResult`) are frozen and have correct type annotations.
3. `run_ismo()` accepts a fitted surrogate + training data + PDE solver kwargs and returns an `ISMOResult`.
4. On a synthetic test (known parameter recovery with RBF surrogate, 3 iterations, budget=30), the convergence metric decreases monotonically across iterations.
5. Augmented training data is saved to disk as `.npz` after each iteration.
6. Budget enforcement: total PDE evaluations never exceed `total_pde_budget`.
7. All private helper functions are unit-testable in isolation (pure functions with explicit inputs/outputs).
8. Code follows project conventions: frozen dataclasses, docstrings on all public symbols, `flush=True` on progress prints.

---

## Key Design Decisions & Rationale

### Why wrap multistart+cascade rather than replace them?

The existing multistart (20k LHS + top-20 polish) and cascade (CD-dominant then
PC-dominant) pipeline is well-tested and known to work. ISMO is an outer loop
that improves the surrogate between optimization calls. Replacing the inner
optimizer would conflate two concerns and risk regressions.

### Why convergence_rtol=0.05?

From Phase 3 benchmarks, the best surrogates (NN ensemble, POD-RBF-log) achieve
~1-5% relative error on I-V curve predictions in the region the optimizer
explores. A 5% tolerance on the surrogate-PDE objective gap is achievable within
2-3 ISMO iterations and represents a meaningful improvement over no ISMO.

### Why samples_per_iteration=30 and not adaptive?

Fixed batch size keeps the logic simple and predictable. Adaptive strategies
(e.g., doubling batch size when improvement stalls) add complexity without
clear benefit at our scale (200 total budget). The no-improvement early stop
handles the case where more samples aren't helping.

### Why sequential PDE evaluation?

Each PDE solve takes 5-30 seconds. At 30 solves per iteration, that's 2.5-15
minutes. Parallel dispatch (via `generate_training_dataset_parallel`) requires
spawning Firedrake worker processes, which has ~30s startup cost per worker.
For 30 evaluations, sequential is simpler and only marginally slower. If
profiling shows this is a bottleneck, 4-02 can add parallel dispatch.

### Why check convergence before retraining?

If the surrogate already agrees with the PDE at the optimizer's best point,
there's no need to retrain. Checking first avoids a wasted training cycle
(100-200 epochs of NN training for no benefit).
