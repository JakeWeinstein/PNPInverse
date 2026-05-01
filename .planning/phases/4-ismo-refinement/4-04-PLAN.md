# 4-04: PDE Evaluation & Data Integration

## Goal

Build the module that bridges ISMO's acquisition strategy (4-03) and surrogate retraining (4-05). Given a batch of candidate parameter points selected by the acquisition strategy, evaluate them with the true PDE solver, compare surrogate predictions against PDE truth, merge results into the training dataset with full provenance tracking, and report quality diagnostics.

## Files to Create

| File | Purpose |
|------|---------|
| `Surrogate/ismo_pde_eval.py` | Core module: PDE evaluation, surrogate-PDE comparison, data integration, quality checks, provenance |

## Files to Modify

| File | Change |
|------|--------|
| `Surrogate/__init__.py` | Export new public API names from `ismo_pde_eval` |

## Dependencies on Other 4-0x Plans

- **4-01 (ISMOConfig)**: Provides `ISMOConfig` dataclass with budget, convergence thresholds, data paths. This plan reads `ismo_config.pde_budget`, `ismo_config.data_dir`, `ismo_config.convergence_nrmse_threshold`.
- **4-02 (ISMO Loop Orchestrator)**: Calls `evaluate_candidates_with_pde()` and `integrate_new_data()` from this module at each iteration.
- **4-03 (Acquisition Strategy)**: Produces the `candidate_params: np.ndarray` of shape `(B, 4)` that this module evaluates.
- **4-05 (Surrogate Retraining)**: Consumes the augmented `.npz` dataset produced by `integrate_new_data()`.

## Detailed Task Breakdown

### Task 1: Solver Configuration Factory

**Rationale**: The PDE solver requires `base_solver_params`, `steady`, `observable_scale`, `mesh`, and `phi_applied_values`. These must exactly match the configuration used for original training data. Rather than threading 5+ objects through ISMOConfig, provide a single factory function that returns all of them in a frozen bundle.

**Implementation**:

```python
@dataclass(frozen=True)
class PDESolverBundle:
    """Immutable bundle of everything needed to run PDE evaluations."""
    base_solver_params: Any          # SolverParams (11-element)
    steady: Any                      # SteadyStateConfig
    observable_scale: float          # -I_SCALE
    mesh_params: Tuple[int, int, float]  # (Nx, Ny, beta) for parallel workers
    phi_applied_values: np.ndarray   # 22-point voltage grid
    min_converged_fraction: float    # default 0.8
```

```python
def make_standard_pde_bundle(
    *,
    Nx: int = 8,
    Ny: int = 200,
    beta: float = 3.0,
    min_converged_fraction: float = 0.8,
) -> PDESolverBundle:
```

- Calls `scripts/_bv_common.make_bv_solver_params(eta_hat=0.0, dt=0.5, t_end=50.0, species=FOUR_SPECIES_CHARGED, snes_opts=SNES_OPTS_CHARGED)` for `base_solver_params`.
- Calls `Forward.steady_state.SteadyStateConfig(relative_tolerance=1e-4, ...)` for `steady`.
- Sets `observable_scale = -I_SCALE` from `scripts/_bv_common`.
- Builds the 22-point voltage grid via `_build_voltage_grid()` (the same function used by `overnight_train_v11.py`).
- Does NOT build the mesh object here (mesh is process-local in Firedrake); instead stores `mesh_params` tuple for `generate_training_dataset_parallel`.

**Design decision**: Use a factory function (Option 3 from the spec) rather than ISMOConfig fields or pickle files. This is the safest path: the factory function references the same constants as the original training scripts, so configuration drift is impossible. The bundle is frozen/immutable to prevent accidental mutation.

### Task 2: ISMO PDE Evaluation Function

**Implementation**:

```python
@dataclass(frozen=True)
class PDEEvalResult:
    """Results from evaluating candidate parameters with the PDE solver."""
    candidate_params: np.ndarray       # (B, 4) input parameters
    current_density: np.ndarray        # (B_valid, 22) CD curves for valid solves
    peroxide_current: np.ndarray       # (B_valid, 22) PC curves for valid solves
    valid_mask: np.ndarray             # (B,) bool: which candidates converged
    timings: np.ndarray                # (B,) seconds per solve
    n_valid: int
    n_failed: int
    valid_params: np.ndarray           # (B_valid, 4) parameters for valid solves
```

```python
def evaluate_candidates_with_pde(
    candidate_params: np.ndarray,
    pde_bundle: PDESolverBundle,
    *,
    n_workers: int = 4,
    output_path: Optional[str] = None,
) -> PDEEvalResult:
```

- Validates `candidate_params` shape is `(B, 4)` with `B <= 100` (safety bound).
- For small batches (B <= 8), uses sequential `generate_training_dataset()` to avoid process spawn overhead.
- For larger batches, uses `generate_training_dataset_parallel()` with warm-start chains.
- Wraps the result into `PDEEvalResult` with separate valid/failed tracking.
- Optionally saves raw PDE results to `output_path` for debugging.
- Logs timing summary: total wall time, mean/min/max per solve, failure count.

**Key detail**: The function delegates entirely to existing `training.py` infrastructure. No new PDE solving code is written. The value is in the wrapper that provides ISMO-specific result structuring and error handling.

### Task 3: Surrogate-PDE Comparison at Candidates

**Implementation**:

```python
@dataclass(frozen=True)
class SurrogatePDEComparison:
    """Per-candidate comparison of surrogate predictions vs PDE truth."""
    candidate_params: np.ndarray       # (B_valid, 4)
    cd_nrmse_per_candidate: np.ndarray # (B_valid,)
    pc_nrmse_per_candidate: np.ndarray # (B_valid,)
    cd_rmse_per_candidate: np.ndarray  # (B_valid,)
    pc_rmse_per_candidate: np.ndarray  # (B_valid,)
    cd_max_error: float                # worst-case CD NRMSE
    pc_max_error: float                # worst-case PC NRMSE
    cd_mean_nrmse: float               # mean CD NRMSE across candidates
    pc_mean_nrmse: float               # mean PC NRMSE across candidates
    is_converged: bool                 # True if max(cd_mean, pc_mean) < threshold
```

```python
def compare_surrogate_vs_pde(
    surrogate,                         # any model with predict_batch()
    pde_result: PDEEvalResult,
    convergence_threshold: float = 0.02,
) -> SurrogatePDEComparison:
```

- Calls `surrogate.predict_batch(pde_result.valid_params)` to get surrogate predictions.
- Computes per-candidate NRMSE for CD and PC using the same normalization logic as `Surrogate/validation.py` (ptp-based with floor).
- Sets `is_converged = True` if both `cd_mean_nrmse` and `pc_mean_nrmse` are below `convergence_threshold`.
- This is the primary convergence diagnostic for the ISMO loop (4-02 checks `comparison.is_converged`).

**Design decision**: Accept any object with `predict_batch()` rather than a specific type. All 6 surrogate model types (RBF, POD-RBF, NN ensemble, GP, PCE, POD-RBF-log) implement this interface, so the comparison function works with any of them.

### Task 4: Data Integration

**Implementation**:

```python
@dataclass(frozen=True)
class AugmentedDataset:
    """Result of merging new PDE data into existing training set."""
    output_path: str                   # where the augmented .npz was saved
    n_original: int                    # samples from original dataset
    n_new: int                         # valid new samples added
    n_total: int                       # n_original + n_new
    provenance: np.ndarray             # (n_total,) string array: source tags
```

```python
def integrate_new_data(
    pde_result: PDEEvalResult,
    existing_data_path: str,
    output_path: str,
    *,
    iteration_tag: str = "ismo_iter_1",
    comparison: Optional[SurrogatePDEComparison] = None,
) -> AugmentedDataset:
```

- Loads existing `.npz` from `existing_data_path` (keys: `parameters`, `current_density`, `peroxide_current`, `phi_applied`).
- Validates that `phi_applied` grids match between existing and new data.
- Appends valid new samples: `np.concatenate` on axis 0 for `parameters`, `current_density`, `peroxide_current`.
- Builds provenance array: `"original"` for existing samples, `iteration_tag` for new ones.
- Saves to `output_path` (never overwrites `existing_data_path`).
- Saves with additional metadata keys:
  - `provenance`: string array of source tags
  - `ismo_metadata`: JSON string with iteration number, acquisition strategy, comparison stats
- Returns `AugmentedDataset` summary.

**File naming convention**: `data/surrogate_models/training_data_ismo_iter{N}.npz`. The original `training_data_merged.npz` is never modified.

**Design decision**: Versioned output paths instead of in-place mutation. This enables rollback and post-hoc analysis of what each ISMO iteration contributed. The provenance array uses simple string tags rather than a separate metadata database, keeping everything self-contained in the `.npz`.

### Task 5: Quality Checks

**Implementation**:

```python
@dataclass(frozen=True)
class QualityReport:
    """Quality check results for a batch of PDE evaluations."""
    n_candidates: int
    n_converged: int
    n_nan_detected: int
    n_extreme_values: int
    n_bounds_violations: int
    n_passed_all_checks: int
    flagged_indices: np.ndarray        # indices of problematic samples
    flags: List[str]                   # human-readable flag descriptions
    passed: bool                       # True if n_passed == n_converged
```

```python
def check_pde_quality(
    pde_result: PDEEvalResult,
    bounds: ParameterBounds,
    *,
    extreme_cd_threshold: float = 50.0,
    extreme_pc_threshold: float = 50.0,
) -> QualityReport:
```

Checks applied to each valid PDE result:

1. **NaN detection**: Any NaN in CD or PC curves after interpolation.
2. **Extreme value detection**: `|CD| > extreme_cd_threshold` or `|PC| > extreme_pc_threshold` at any voltage point (dimensionless units).
3. **Parameter bounds compliance**: All 4 parameters within `ParameterBounds` ranges.
4. **Convergence fraction**: Already handled by `generate_training_dataset`'s `min_converged_fraction`, but re-verified here.

Samples that fail any check are flagged but NOT silently removed. The `QualityReport` is returned to the ISMO orchestrator (4-02) which decides whether to include flagged samples or discard them.

### Task 6: Provenance Tracking

**Implementation** (integrated into `integrate_new_data`, not a separate function):

Each training sample is tagged with:
- `source`: `"original_lhs"` | `"ismo_iter_{N}"`
- `acquisition_strategy`: `"uncertainty"` | `"error_based"` | `"hybrid"` (from 4-03)
- `surrogate_prediction`: the surrogate's CD/PC prediction at this point before the PDE solve (optional, for post-hoc analysis)
- `pde_surrogate_nrmse`: the per-point NRMSE from `SurrogatePDEComparison`

Storage format in the `.npz`:
- `provenance_source`: `np.array` of strings, shape `(N_total,)`
- `provenance_strategy`: `np.array` of strings, shape `(N_total,)` (empty string for original samples)
- `provenance_nrmse_cd`: `np.array` of floats, shape `(N_total,)` (NaN for original samples)
- `provenance_nrmse_pc`: `np.array` of floats, shape `(N_total,)` (NaN for original samples)

This enables post-hoc analysis questions like:
- "Did ISMO-acquired points concentrate in high-error regions?"
- "What was the surrogate error at ISMO points before vs after retraining?"
- "Which acquisition strategy contributed the most useful points?"

## Integration with ISMO Loop (4-02)

The ISMO orchestrator calls this module's functions in sequence:

```python
# Inside the ISMO loop iteration:

# 1. Evaluate candidates with PDE
pde_result = evaluate_candidates_with_pde(
    candidate_params,
    pde_bundle,
    n_workers=ismo_config.n_pde_workers,
)

# 2. Quality check
quality = check_pde_quality(pde_result, bounds)
if not quality.passed:
    log.warning(f"Quality issues: {quality.flags}")

# 3. Compare surrogate vs PDE at these points
comparison = compare_surrogate_vs_pde(
    current_surrogate,
    pde_result,
    convergence_threshold=ismo_config.convergence_nrmse_threshold,
)

# 4. Check convergence
if comparison.is_converged:
    log.info("ISMO converged: surrogate agrees with PDE at new points")
    break

# 5. Integrate new data
augmented = integrate_new_data(
    pde_result,
    existing_data_path=current_data_path,
    output_path=next_data_path,
    iteration_tag=f"ismo_iter_{iteration}",
    comparison=comparison,
)

# 6. Pass augmented.output_path to retraining (4-05)
```

## Success Criteria

1. `evaluate_candidates_with_pde()` correctly delegates to existing `training.py` functions and returns structured results with valid/failed separation.
2. `compare_surrogate_vs_pde()` computes NRMSE metrics consistent with `Surrogate/validation.py` and reports convergence status.
3. `integrate_new_data()` produces a valid `.npz` that is loadable by all existing surrogate training pipelines (`BVSurrogateModel.fit()`, NN training, etc.) without modification.
4. Original `training_data_merged.npz` is never modified; all augmented datasets use versioned paths.
5. Provenance arrays enable filtering samples by source after the fact.
6. Quality checks catch NaN, extreme values, and out-of-bounds parameters.
7. The module introduces no Firedrake import at module level (all Firedrake imports are deferred inside functions, matching the pattern in `training.py`).
8. Unit tests verify data integration round-trip: create small dataset, integrate new data, reload and verify shapes/provenance.

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Solver config | Factory function (`make_standard_pde_bundle`) | Guarantees config consistency with original training; no config drift risk from serialization |
| Mesh handling | Store `mesh_params` tuple, not mesh object | Firedrake meshes are process-local; workers build their own via existing `_training_worker_init` |
| Small batch threshold | Sequential for B <= 8, parallel for B > 8 | Process spawn overhead (~5s per worker) exceeds benefit for tiny batches |
| Data versioning | New `.npz` per iteration, never overwrite | Enables rollback, post-hoc analysis, and debugging without data loss |
| Surrogate interface | Duck-typed `predict_batch()` | Works with all 6 model types without coupling to a specific class |
| Quality check policy | Flag but don't auto-remove | Let the orchestrator decide; different strategies may want different tolerance |
| Provenance storage | Arrays in `.npz` alongside training data | Self-contained; no external database; compatible with existing `np.load` workflows |
