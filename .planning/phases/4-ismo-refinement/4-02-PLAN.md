# 4-02: Acquisition Strategy & Sample Selection

## Goal

Design and implement an acquisition module (`Surrogate/acquisition.py`) that decides **where to place new PDE evaluations** each ISMO iteration. The module must support three complementary strategies -- optimizer-trajectory sampling, GP-uncertainty-guided sampling, and space-filling LHS -- and combine them via a configurable hybrid allocation. All strategies share a common de-duplication and batch-diversity pipeline to avoid wasting PDE budget on redundant points.

## Files to Create

| File | Purpose |
|------|---------|
| `Surrogate/acquisition.py` | Core acquisition module (all strategies + hybrid orchestrator) |
| `tests/test_acquisition.py` | Unit tests for every public function |

## Files to Modify

| File | Change |
|------|--------|
| `Surrogate/__init__.py` | Export `AcquisitionConfig`, `AcquisitionResult`, `select_new_samples` |

## Dependencies on Other 4-0x Plans

- **4-01 (ISMO Loop Orchestration)**: The orchestrator calls `select_new_samples()` each iteration and passes the returned parameter array to the PDE solver. This plan defines the interface that 4-01 consumes.
- **4-03 (Data Augmentation & Retraining)**: Receives the newly-evaluated PDE data and appends it to the training set. No direct code dependency from this module, but the `AcquisitionResult` metadata (strategy labels per point) is useful for 4-03 logging.

## Detailed Task Breakdown

### Task 1: Configuration Dataclass

```python
@dataclass(frozen=True)
class AcquisitionConfig:
    """Controls how the acquisition budget is allocated."""

    # Total number of new PDE samples to request
    budget: int = 30

    # Fractional allocation (must sum to 1.0)
    frac_optimizer: float = 0.5     # from optimizer trajectories
    frac_uncertainty: float = 0.3   # from GP posterior variance
    frac_spacefill: float = 0.2     # from space-filling LHS

    # Optimizer trajectory settings
    neighborhood_radius_log: float = 0.3   # ball radius in normalized log-space
    n_neighbors_per_candidate: int = 2     # random neighbors per optimizer point

    # Uncertainty settings (GP only)
    n_uncertainty_candidates: int = 5000   # LHS candidates to evaluate GP variance on
    k0_2_sensitivity_weight: float = 2.0   # upweight k0_2 dimension in variance

    # De-duplication
    min_distance_log: float = 0.05         # min distance in normalized log-space to existing data
    min_distance_batch: float = 0.08       # min distance between points in the same batch

    # Space-filling
    spacefill_seed: int = 777

    # General
    seed: int = 42
    verbose: bool = True
```

**Rationale for defaults**: 50/30/20 split gives priority to the optimizer trajectory (cheapest, most targeted) while ensuring GP-guided exploration and global coverage. The 0.05 log-space distance threshold means two points must differ by at least ~12% in each k0 dimension to be considered distinct (10^0.05 ~ 1.12x).

### Task 2: Result Container

```python
@dataclass(frozen=True)
class AcquisitionResult:
    """Output of the acquisition pipeline."""

    samples: np.ndarray              # (N_acquired, 4) in physical space [k0_1, k0_2, alpha_1, alpha_2]
    strategy_labels: tuple           # length N_acquired, each element in {"optimizer", "uncertainty", "spacefill"}
    n_requested: int                 # = config.budget
    n_acquired: int                  # may be < budget if de-dup removes too many
    n_rejected_dedup: int            # points removed by distance checks
    gp_variance_used: bool           # whether GP uncertainty was actually available
```

### Task 3: Normalize-to-Log Utility

```python
def _to_normalized_log(
    params: np.ndarray,
    bounds: ParameterBounds,
) -> np.ndarray:
    """Convert physical-space parameters (N,4) to normalized log-space [0,1]^4.

    Dimensions 0,1 (k0_1, k0_2): log10 then min-max scale to [0,1].
    Dimensions 2,3 (alpha_1, alpha_2): min-max scale to [0,1].
    """
```

This is the common coordinate system for all distance computations. Using normalized [0,1]^4 makes the distance threshold interpretable across dimensions with very different scales.

### Task 4: Distance and De-duplication Functions

```python
def _min_distance_to_set(
    candidate: np.ndarray,          # (4,) normalized log-space
    existing: np.ndarray,           # (M, 4) normalized log-space
) -> float:
    """Euclidean distance from candidate to nearest point in existing set."""

def _deduplicate_candidates(
    candidates: np.ndarray,         # (K, 4) physical space
    existing_data: np.ndarray,      # (N_train, 4) physical space
    bounds: ParameterBounds,
    min_dist_to_existing: float,
    min_dist_within_batch: float,
) -> Tuple[np.ndarray, int]:
    """Remove candidates too close to existing data or to each other.

    Algorithm:
    1. Convert all points to normalized log-space.
    2. For each candidate (in order), reject if:
       a. min distance to existing_data < min_dist_to_existing, OR
       b. min distance to already-accepted candidates < min_dist_within_batch.
    3. Return (accepted_candidates_physical, n_rejected).

    Returns candidates in physical space, preserving input order.
    """
```

**Design decision**: Greedy sequential filtering (not batch-optimal) because the candidate pool is small (~100 points) and greedy is simple and deterministic. The ordering matters: optimizer trajectory points are checked first (highest priority), then uncertainty points, then space-filling.

### Task 5: Optimizer Trajectory Acquisition

```python
def _acquire_optimizer_trajectory(
    multistart_result: Optional[MultiStartResult],
    cascade_result: Optional[CascadeResult],
    bounds: ParameterBounds,
    n_target: int,
    neighborhood_radius: float,
    n_neighbors_per_candidate: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Extract candidate points from optimizer outputs + neighborhood ball.

    Algorithm:
    1. Collect all unique optimizer endpoints:
       - From MultiStartResult.candidates: extract (k0_1, k0_2, alpha_1, alpha_2) for each.
       - From CascadeResult.pass_results: extract endpoint of each pass.
    2. De-duplicate these endpoints (within tolerance in normalized log-space).
    3. For each unique endpoint, generate n_neighbors_per_candidate points
       by sampling uniformly in a ball of radius neighborhood_radius
       in normalized log-space, then converting back to physical space.
       - Clamp to bounds.
    4. Return up to n_target points (the endpoints themselves + neighbors),
       prioritizing the lowest-loss endpoints first.

    Returns np.ndarray of shape (<=n_target, 4) in physical space.
    """
```

**Key detail**: The neighborhood ball is sampled in normalized log-space so that "radius 0.3" means the same proportional perturbation for k0_1 (spanning 6 orders) and alpha_1 (spanning 0.1-0.9). This ensures the neighborhood doesn't collapse to a point in one dimension while spanning the entire range in another.

### Task 6: Uncertainty-Based Acquisition (GP)

```python
def _acquire_uncertainty(
    gp_model: Optional[GPSurrogateModel],
    bounds: ParameterBounds,
    n_target: int,
    n_candidates: int,
    k0_2_weight: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select points with highest GP posterior variance.

    Algorithm:
    1. If gp_model is None or not fitted, return empty array (graceful fallback).
    2. Generate n_candidates LHS points in the parameter space.
    3. Call gp_model.predict_batch_with_uncertainty(candidates).
    4. Compute aggregate uncertainty score for each candidate:
       - sigma_cd = mean of current_density_std across voltage points
       - sigma_pc = mean of peroxide_current_std across voltage points
       - score = sigma_cd + sigma_pc
    5. Apply k0_2-sensitivity weighting:
       - Compute the normalized k0_2 coordinate for each candidate (in [0,1]).
       - Boost score by factor (1 + k0_2_weight * k0_2_sensitivity(candidate))
         where k0_2_sensitivity peaks at mid-range k0_2 values (where
         the I-V curve is most sensitive to k0_2 variations).
       - Implementation: k0_2_sensitivity = 1.0 (uniform weight) as first pass,
         upgradeable to local gradient norm w.r.t. k0_2 later.
    6. Rank by score, return top n_target candidates.

    Returns np.ndarray of shape (<=n_target, 4) in physical space.
    """
```

**Design decision**: Using maximum posterior variance (exploration) rather than Expected Improvement (exploitation) because the goal is to improve the *surrogate* globally, not to find the optimum directly. The k0_2 weighting addresses the known sensitivity gap: k0_2 recovery is hardest, so we bias sampling toward regions where the surrogate is uncertain about k0_2-sensitive outputs.

**Fallback**: If no GP model is available (e.g., using only RBF/NN), the uncertainty budget is redistributed: 50% to optimizer trajectory, 50% to space-filling.

### Task 7: Space-Filling LHS Acquisition

```python
def _acquire_spacefill(
    bounds: ParameterBounds,
    n_target: int,
    seed: int,
) -> np.ndarray:
    """Generate space-filling LHS samples across the full parameter space.

    Uses the existing generate_lhs_samples() from Surrogate.sampling with
    log_space_k0=True.

    Returns np.ndarray of shape (n_target, 4) in physical space.
    """
```

This is intentionally simple -- it reuses the existing LHS infrastructure. The purpose is to prevent the ISMO loop from collapsing into a single basin by maintaining global coverage.

### Task 8: Hybrid Orchestrator (Main Entry Point)

```python
def select_new_samples(
    existing_data: np.ndarray,
    bounds: ParameterBounds,
    config: AcquisitionConfig,
    multistart_result: Optional[MultiStartResult] = None,
    cascade_result: Optional[CascadeResult] = None,
    gp_model: Optional[GPSurrogateModel] = None,
) -> AcquisitionResult:
    """Select new PDE evaluation points using the hybrid acquisition strategy.

    This is the main entry point called by the ISMO loop orchestrator (4-01).

    Algorithm:
    1. Compute per-strategy budgets from config fractions:
       n_opt = round(budget * frac_optimizer)
       n_unc = round(budget * frac_uncertainty)
       n_lhs = budget - n_opt - n_unc   (remainder to space-filling)

    2. If gp_model is None, redistribute uncertainty budget:
       n_opt += n_unc // 2
       n_lhs += n_unc - n_unc // 2
       n_unc = 0

    3. Generate candidates from each strategy (Tasks 5-7).

    4. Concatenate candidates with strategy labels:
       [optimizer_candidates, uncertainty_candidates, spacefill_candidates]
       Labels track provenance for logging.

    5. Run _deduplicate_candidates() on the full pool against existing_data.
       Order matters: optimizer points first (highest priority), then
       uncertainty, then space-filling.

    6. If n_acquired < budget * 0.5, log a warning (the parameter space
       may be saturated).

    7. Return AcquisitionResult.

    Parameters
    ----------
    existing_data : np.ndarray (N_train, 4)
        Current training parameters in physical space.
    bounds : ParameterBounds
        Parameter space bounds.
    config : AcquisitionConfig
        Acquisition configuration.
    multistart_result : MultiStartResult, optional
        Most recent multistart optimizer output.
    cascade_result : CascadeResult, optional
        Most recent cascade optimizer output.
    gp_model : GPSurrogateModel, optional
        Fitted GP model for uncertainty quantification.

    Returns
    -------
    AcquisitionResult
        New sample points and metadata.
    """
```

### Task 9: Unit Tests (`tests/test_acquisition.py`)

Test cases:

1. **test_normalized_log_roundtrip**: Convert to normalized log-space and back, check accuracy.
2. **test_min_distance_to_set**: Known geometry, verify correct distances.
3. **test_deduplicate_removes_close_points**: Place candidates at known distances, verify rejection.
4. **test_deduplicate_preserves_order**: Optimizer points survive before spacefill points.
5. **test_optimizer_trajectory_extracts_candidates**: Mock `MultiStartResult` with 5 candidates, verify extraction.
6. **test_optimizer_trajectory_with_neighbors**: Verify neighborhood points are within radius and within bounds.
7. **test_uncertainty_fallback_no_gp**: When gp_model=None, returns empty array.
8. **test_spacefill_returns_correct_count**: Verify LHS returns exactly n_target points.
9. **test_select_new_samples_budget_allocation**: Verify strategy counts match config fractions.
10. **test_select_new_samples_no_gp_redistribution**: With gp_model=None, uncertainty budget goes to optimizer+spacefill.
11. **test_select_new_samples_all_strategies**: Integration test with mock GP, MultiStart, Cascade results.
12. **test_batch_diversity**: Verify no two accepted points are closer than min_distance_batch.

### Task 10: Export from `__init__.py`

Add to `Surrogate/__init__.py`:
```python
from Surrogate.acquisition import AcquisitionConfig, AcquisitionResult, select_new_samples
```

And add to `__all__`.

## Algorithm Summary

```
ISMO Iteration k:
  1. Train/retrain surrogate on D_k
  2. Optimize surrogate -> MultiStartResult, CascadeResult
  3. select_new_samples(                          <-- THIS MODULE
       existing_data=D_k,
       bounds=bounds,
       config=AcquisitionConfig(budget=30),
       multistart_result=ms_result,
       cascade_result=cas_result,
       gp_model=gp,                              <-- optional
     ) -> AcquisitionResult
  4. Evaluate PDE at AcquisitionResult.samples    <-- 4-01 handles this
  5. D_{k+1} = D_k + new_data                    <-- 4-03 handles this
```

## Key Design Decisions

1. **Normalized log-space for all distance computations**: k0 spans 3-7 orders of magnitude while alpha spans [0.1, 0.9]. Raw Euclidean distance would be dominated by k0. Normalizing to [0,1]^4 with log-transform on k0 dims makes distance meaningful.

2. **Greedy sequential de-duplication, not optimal**: An optimal batch design (e.g., DPP or k-DPP) would maximize batch diversity but adds complexity. Greedy filtering with priority ordering (optimizer > uncertainty > spacefill) is simple, deterministic, and sufficient for batches of 20-50 points.

3. **GP variance as exploration, not EI**: Expected Improvement targets the optimum of the *objective*. We want to improve the *surrogate* globally, so maximum variance (pure exploration) is more appropriate. The k0_2 weighting biases exploration toward the dimension that matters most for parameter recovery.

4. **Graceful GP fallback**: The module works without a GP model by redistributing the uncertainty budget. This means the ISMO loop can start with RBF/NN-only surrogates and add GP-guided sampling later.

5. **Immutable dataclasses**: Both config and result are frozen dataclasses, consistent with the existing `MultiStartConfig`/`MultiStartResult` and `CascadeConfig`/`CascadeResult` patterns.

6. **Physical-space output**: The returned `samples` array is in physical space [k0_1, k0_2, alpha_1, alpha_2], matching the PDE solver input format and the training data format. All log/normalization transforms are internal.

## Success Criteria

- [ ] `select_new_samples()` returns 20-50 physically-valid parameter points per call
- [ ] No returned point is within `min_distance_log` of any existing training point (in normalized log-space)
- [ ] No two returned points are within `min_distance_batch` of each other
- [ ] Strategy labels correctly track which strategy generated each point
- [ ] When gp_model=None, the module still works (uncertainty budget redistributed)
- [ ] All returned k0 values are within bounds; all alpha values are within bounds
- [ ] Unit tests pass with >90% line coverage on `acquisition.py`
- [ ] Integration: the ISMO orchestrator (4-01) can call `select_new_samples()` and feed the result directly to the PDE solver
