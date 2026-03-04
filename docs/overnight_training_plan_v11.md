# Overnight Surrogate Training Plan (v11)

## Overview

**Total estimated wall-clock time:** ~10 hours (down from ~12h in v10 thanks to warm-start optimizations)
**Machine:** 10-core CPU (macOS Darwin)
**Python:** `/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/venv-firedrake/bin/python`
**Working directory:** `/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse`
**Output root:** `StudyResults/surrogate_v11/`

The plan is structured as a single script (`scripts/surrogate/overnight_train_v11.py`) that executes three phases sequentially. Each phase is idempotent and checkpoint-resumable. If the script crashes at hour 7, re-running it picks up from the last checkpoint.

```
Phase 1: Data Generation  (~7 hours)   -- parallel, warm-started, checkpoint-resumable
Phase 2: Model Training    (~1 hour)   -- RBF, POD-RBF, NN ensemble
Phase 3: Evaluation        (~20 min)   -- accuracy metrics, parameter recovery, visualization
```

---

## Phase 1: Training Data Generation (~7 hours)

### 1.1 Goal

Expand the training set from 445 to ~3,145 valid samples by generating ~3,000 new samples. This ~7x data increase is critical because:
- POD-RBF per-mode smoothing optimization needs dense coverage to avoid overfitting
- NN ensemble (5 models with different 85/15 splits) needs enough data that each member trains on ~2,500 samples
- The focused-region samples near the Mangan2025 operating point (where k0_2 recovery matters most) will improve PC accuracy in the critical region

### 1.2 Sampling Strategy

Use `generate_multi_region_lhs_samples()` from `Surrogate/sampling.py` to generate samples in two tiers:

**Tier 1 -- Wide coverage (2,000 new samples):**
```python
from Surrogate.sampling import ParameterBounds, generate_multi_region_lhs_samples

wide_bounds = ParameterBounds(
    k0_1_range=(1e-6, 1.0),
    k0_2_range=(1e-7, 0.1),
    alpha_1_range=(0.1, 0.9),
    alpha_2_range=(0.1, 0.9),
)
```
Same bounds as v9. Seed=200 (different from v9's seed=42 to avoid duplicate samples).

**Tier 2 -- Focused refinement (1,000 new samples):**
```python
focused_bounds = ParameterBounds(
    k0_1_range=(1e-4, 1e-1),      # 2 decades around typical true k0_1
    k0_2_range=(1e-5, 1e-2),      # 2 decades around typical true k0_2
    alpha_1_range=(0.2, 0.7),     # narrower alpha range
    alpha_2_range=(0.2, 0.7),
)
```
Seed=300.

Combined: 3,000 new samples. At ~90% convergence rate, expect ~2,700 valid. Plus the existing 445 = ~3,145 total.

Generate via:
```python
new_samples = generate_multi_region_lhs_samples(
    wide_bounds=wide_bounds,
    focused_bounds=focused_bounds,
    n_base=2000,
    n_focused=1000,
    seed_base=200,
    seed_focused=300,
    log_space_k0=True,
)
# new_samples.shape == (3000, 4)
```

### 1.3 Parameter-Space Sample Ordering

Before assigning samples to workers, reorder them by nearest-neighbor chain in (log10(k0_1), log10(k0_2), alpha_1, alpha_2) space. This ensures that consecutive samples have similar parameters, enabling cross-sample warm-starting (Section 1.5) that reduces solve time by 15-25%.

```python
def _order_samples_nearest_neighbor(
    samples: np.ndarray,
    log_space_k0: bool = True,
) -> np.ndarray:
    """Reorder samples by nearest-neighbor greedy chain in parameter space.

    Parameters
    ----------
    samples : np.ndarray of shape (N, 4)
        Columns: [k0_1, k0_2, alpha_1, alpha_2].
    log_space_k0 : bool
        Transform k0 columns to log10 before computing distances.

    Returns
    -------
    np.ndarray
        Permutation indices giving the nearest-neighbor order.
    """
    N = samples.shape[0]
    coords = samples.copy()
    if log_space_k0:
        coords[:, 0] = np.log10(np.maximum(coords[:, 0], 1e-20))
        coords[:, 1] = np.log10(np.maximum(coords[:, 1], 1e-20))

    # Normalize each column to [0, 1] for uniform distance weighting
    for col in range(4):
        cmin, cmax = coords[:, col].min(), coords[:, col].max()
        if cmax > cmin:
            coords[:, col] = (coords[:, col] - cmin) / (cmax - cmin)

    # Greedy nearest-neighbor starting from the sample closest to center
    center = coords.mean(axis=0)
    dists_to_center = np.linalg.norm(coords - center, axis=1)
    start = int(np.argmin(dists_to_center))

    visited = np.zeros(N, dtype=bool)
    order = np.empty(N, dtype=int)
    order[0] = start
    visited[start] = True

    for i in range(1, N):
        current = order[i - 1]
        dists = np.linalg.norm(coords - coords[current], axis=1)
        dists[visited] = np.inf
        order[i] = int(np.argmin(dists))
        visited[order[i]] = True

    return order
```

**Complexity**: O(N^2) for N=3000 samples -- takes ~2 seconds (negligible vs the multi-hour data generation).

### 1.4 Warm-Start Strategies

Three levels of warm-starting combine to dramatically reduce per-sample solve time. They are listed here from innermost (within a single I-V curve) to outermost (across parameter samples).

#### 1.4.1 Voltage Sweep Ordering (Eta Continuation) -- Already Implemented

Every call to `generate_training_data_single()` (in `Surrogate/training.py`) invokes `solve_bv_curve_points_with_warmstart()` (in `FluxCurve/bv_point_solve/__init__.py`). Inside that function, the 22 voltage points are NOT solved in their input order. Instead, they are reordered by `_build_sweep_order()` (in `FluxCurve/bv_point_solve/predictor.py`):

```python
def _build_sweep_order(phi_applied_values: np.ndarray) -> np.ndarray:
    """Build sweep order for warm-start continuation with mixed-sign eta.

    When all phi_applied values share the same sign (or are zero), this
    reduces to np.argsort(np.abs(phi_applied_values)) (ascending |eta|).

    When both positive and negative values are present, a two-branch sweep
    is used:
    1. Negative branch (eta <= 0): sorted ascending in |eta|.
    2. Positive branch (eta > 0):  sorted ascending in eta.
    """
```

For our training voltage grid (union of `eta_symmetric`, `eta_shallow`, `eta_cathodic`, containing both positive values like +5.0, +3.0, +1.0 and negative values down to -46.5), the **two-branch sweep** activates:

1. **Negative branch first** (because the smallest-|eta| point, eta = -0.5, is negative): solve -0.5, -1.0, -2.0, -3.0, -4.0, -5.0, -6.5, -8.0, -10.0, -11.5, -13.0, -15.0, -17.0, -20.0, -22.0, -28.0, -35.0, -41.0, -46.5 (ascending |eta|)
2. **Positive branch second**: solve +1.0, +3.0, +5.0 (ascending eta)

Between branches, the "hub state" mechanism (lines 316-323 and 382-401 of `FluxCurve/bv_point_solve/__init__.py`) saves the converged solution at eta = -0.5 (the near-equilibrium hub). When the positive branch starts at eta = +1.0, the solver restores the hub state rather than warm-starting from the eta = -46.5 solution (which would be far from equilibrium and cause SNES divergence).

**Within-sample warm-start chain details:**

1. **Carry state**: `carry_U_data = tuple(d.data_ro.copy() for d in U.dat)` -- a tuple of numpy arrays, one per sub-function in the mixed function space (line 702 of `__init__.py`).

2. **Predictor step**: Before the forward solve at each point, `_apply_predictor()` (in `FluxCurve/bv_point_solve/predictor.py`) extrapolates from up to 3 prior converged solutions using quadratic Lagrange interpolation. If the extrapolation produces DOFs that deviate by more than 10x from the carry state, it reverts to simple copy. Falls back to linear extrapolation (2 points) or simple copy (1 point).

3. **Bridge points**: When the gap between consecutive solved etas exceeds `max_eta_gap` (default 3.0 in training), forward-only intermediate points are auto-inserted by `_solve_bridge_points()` (in `FluxCurve/bv_point_solve/predictor.py`) with generous step caps (`_BRIDGE_MAX_STEPS = 40`, `_BRIDGE_SER_DT_MAX_RATIO = 50.0`).

4. **SER adaptive pseudo-timestep**: Each point starts with `dt_initial` (typically 0.5) and adaptively grows/shrinks. Constants from `FluxCurve/bv_point_solve/cache.py`: `_SER_GROWTH_CAP = 4.0`, `_SER_SHRINK = 0.5`, `_SER_DT_MAX_RATIO = 20.0`. Warm-started points typically converge in 3-8 steps (vs 20-100 for cold starts).

5. **Reduced step cap**: `_WARMSTART_MAX_STEPS = 20` (vs the full `max_steps = 100`). Since the IC is close to the converged solution, 20 steps is more than sufficient.

**Net effect**: Without eta continuation, each of the 22 voltage points would start from the blob IC and need 50-100 pseudo-time steps. With continuation, the first point converges in ~10-15 steps and subsequent points in 3-8 steps each. **No code changes are needed** -- this is already active inside `generate_training_data_single()`.

**Important detail -- two separate sweeps per sample**: `generate_training_data_single()` calls `solve_bv_curve_points_with_warmstart()` twice per sample -- once for `observable_mode="current_density"` and once for `observable_mode="peroxide_current"`. Between these calls, it calls `_clear_caches()` (lines 102 and 125 of `Surrogate/training.py`). The CD sweep benefits from full eta continuation, but the PC sweep starts cold.

**Optimization -- reuse CD solutions for PC sweep**: Modify `generate_training_data_single()` to skip `_clear_caches()` between CD and PC sweeps, instead seeding the PC sweep's `_cross_eval_cache` with the CD sweep's converged solutions. The PC sweep then needs only 1-3 steps per point (same physics, different observable assembly). This could cut per-sample time by ~40% (from ~75s to ~45s). See Section 1.10 for the code change.

#### 1.4.2 Cross-Sample Warm-Starting (Parameter-Space Continuity) -- New

After the nearest-neighbor ordering in Section 1.3, consecutive samples have similar (k0, alpha) parameters. The converged PDE solution at eta = -0.5 for sample i is an excellent IC for sample i+1 at eta = -0.5.

**Mechanism**: The BV point solve module (`FluxCurve/bv_point_solve/cache.py`) maintains a `_cross_eval_cache` dict that persists across calls within the same process. At the start of each sweep, `solve_bv_curve_points_with_warmstart()` checks this cache for a stored IC for the first point (line 306 of `__init__.py`):

```python
# Line 306 of FluxCurve/bv_point_solve/__init__.py:
carry_U_data = _cache_mod._cross_eval_cache.get(first_orig_idx, None)
```

To enable cross-sample warm-starting, the training loop seeds `_cross_eval_cache` with the previous sample's converged solution before calling the BV solver. Instead of a full `_clear_caches()` between samples, we clear only `_all_points_cache` (which assumes same parameters) and keep `_cross_eval_cache` populated.

**Expected benefit**: Cold start for the first voltage point takes ~8-15 pseudo-time steps. Parameter-warm-started first point: ~2-4 steps. Combined with eta continuation for subsequent points, this yields 15-25% reduction in total per-sample time.

#### 1.4.3 Cache Architecture and Per-Worker Isolation

The BV point solve module (`FluxCurve/bv_point_solve/cache.py`) defines three global caches:

```python
# Module-level globals in FluxCurve/bv_point_solve/cache.py:
_cross_eval_cache: Dict[int, tuple] = {}     # P2: first-point IC from previous eval
_all_points_cache: Dict[int, tuple] = {}     # P7: all-points IC for fast path
_cache_populated: bool = False               # P7: flag for fast-path eligibility
```

- **`_cross_eval_cache`**: Suitable for cross-sample warm-starting. Stores one point's IC (~64 KB for our 4-species, 8x200 mesh: 5 sub-functions x 1600 DOFs x 8 bytes).
- **`_all_points_cache`**: Stores all 22 points' ICs (~1.4 MB). Designed for optimizer iterations at the SAME parameters. NOT suitable for training (parameters change each sample). Must be cleared between samples.
- **`_clear_caches()`**: Resets everything. For cross-sample warm-starting, we replace full `_clear_caches()` with selective clearing.

Each parallel worker process (spawn context) has its own independent copy of these module-level caches. There is no shared-memory cache between workers. This is **desirable** for the grouped-sequential approach: each worker maintains its own warm-start chain independently, with negligible memory overhead.

**Why NOT BVPointSolvePool**: The existing `BVPointSolvePool` (in `FluxCurve/bv_parallel.py`) is designed for parallel point solves at the SAME parameters during inference (multiple voltage points solved concurrently). Its `BVParallelPointConfig` is frozen at initialization with fixed (k0, alpha) values, and its workers compute adjoint gradients we do not need for training data. For training data generation, dedicated worker functions are the correct approach.

### 1.5 Parallelization Strategy

The current `generate_training_dataset()` in `Surrogate/training.py` processes samples sequentially -- each sample calls `generate_training_data_single()` which itself calls `solve_bv_curve_points_with_warmstart()` twice (once for CD, once for PC). Each sample takes ~60-90 seconds.

**Sequential estimate:** 3,000 samples x 75s = 62.5 hours. Far too slow.

**Parallel approach:** Create a new function `generate_training_dataset_parallel()` that combines `ProcessPoolExecutor` parallelism with parameter-space warm-starting. The design uses **batched sequential within ordered groups**: divide the nearest-neighbor-ordered 3,000 samples into 8 groups (one per worker), each group containing ~375 parameter-space-adjacent samples. Workers process their groups sequentially, carrying warm-start state between consecutive samples, while groups run in parallel across workers.

```
Worker 0: samples [0, 1, 2, ..., 374]       (parameter-space neighbors)
Worker 1: samples [375, 376, ..., 749]       (parameter-space neighbors)
...
Worker 7: samples [2625, 2626, ..., 2999]    (parameter-space neighbors)
```

Each worker solves ~375 samples sequentially, benefiting from parameter-space warm-starts within its group. Only the first sample of each group is a cold start; subsequent samples reuse the previous sample's converged solutions.

#### 1.5.1 Spawn Context (required for Firedrake/PETSc)

```python
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

ctx = mp.get_context("spawn")
executor = ProcessPoolExecutor(
    max_workers=n_workers,
    mp_context=ctx,
    initializer=_training_worker_init,
    initargs=(base_solver_params, steady_config, observable_scale, mesh_Nx, mesh_Ny, mesh_beta),
)
```

#### 1.5.2 Worker Initializer (mirrors `_bv_worker_init` pattern from `FluxCurve/bv_parallel.py`)

```python
def _training_worker_init(base_solver_params, steady_config, obs_scale, Nx, Ny, beta):
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ.setdefault("FIREDRAKE_TSFC_KERNEL_CACHE_DIR", "/tmp/firedrake-tsfc")
    os.environ.setdefault("PYOP2_CACHE_DIR", "/tmp/pyop2")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp")
    global _TRAIN_WORKER_STATE
    from Forward.bv_solver import make_graded_rectangle_mesh
    _TRAIN_WORKER_STATE = {
        "base_solver_params": base_solver_params,
        "steady": steady_config,
        "observable_scale": obs_scale,
        "mesh": make_graded_rectangle_mesh(Nx=Nx, Ny=Ny, beta=beta),
    }
```

#### 1.5.3 Worker Function -- Processes a Group with Cross-Sample Warm-Starts

```python
def _training_worker_solve_group(group_tasks):
    """Solve a group of nearby samples sequentially, carrying warm-start state.

    group_tasks: list of (sample_index, k0_1, k0_2, alpha_1, alpha_2, phi_applied)
    """
    from FluxCurve.bv_point_solve import cache as _cache_mod

    results = []
    prev_solutions = None

    for task in group_tasks:
        idx, k0_1, k0_2, alpha_1, alpha_2, phi_applied = task
        st = _TRAIN_WORKER_STATE

        # Seed cache from previous sample (if available)
        _cache_mod._clear_caches()  # clear stale _all_points_cache
        if prev_solutions is not None:
            for cache_idx, u_data in prev_solutions.items():
                _cache_mod._cross_eval_cache[cache_idx] = u_data

        result = generate_training_data_single(
            k0_values=[k0_1, k0_2],
            alpha_values=[alpha_1, alpha_2],
            phi_applied_values=phi_applied,
            base_solver_params=st["base_solver_params"],
            steady=st["steady"],
            observable_scale=st["observable_scale"],
            mesh=st["mesh"],
            return_solutions=True,  # NEW: return converged U_data for warm-start chain
        )

        # Extract converged solutions for next sample's warm-start
        if result.get("converged_solutions") is not None:
            prev_solutions = result["converged_solutions"]

        results.append({
            "index": idx,
            "current_density": result["current_density"].tolist(),
            "peroxide_current": result["peroxide_current"].tolist(),
            "converged_mask": result["converged_mask"].tolist(),
            "n_converged": result["n_converged"],
        })

    return results
```

#### 1.5.4 Worker Count

Use `n_workers = 8` on this 10-core machine (leaving 2 cores for OS + main process). Each worker sets `OMP_NUM_THREADS=1`, so total CPU usage = 8 cores. This matches the `BVPointSolvePool` default of `cpu_count() - 1` but uses 8 instead of 9 to leave more headroom for the long overnight run.

#### 1.5.5 Orchestrator -- `generate_training_dataset_parallel()`

```python
import time
import datetime
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait

def generate_training_dataset_parallel(
    parameter_samples, *, phi_applied_values, base_solver_params,
    steady, observable_scale, mesh_params, output_path,
    n_workers=8, checkpoint_interval=40,
    min_converged_fraction=0.8, verbose=True,
):
    N = parameter_samples.shape[0]
    n_eta = len(phi_applied_values)

    # Step 1: Order samples by parameter-space proximity
    nn_order = _order_samples_nearest_neighbor(parameter_samples)

    # Step 2: Divide into worker groups (contiguous segments of the NN chain)
    group_size = (N + n_workers - 1) // n_workers  # ceil division
    groups = []
    for w in range(n_workers):
        start = w * group_size
        end = min(start + group_size, N)
        if start < N:
            group_indices = nn_order[start:end]
            group_tasks = [
                (int(i), *parameter_samples[i].tolist(), phi_applied_values)
                for i in group_indices
            ]
            groups.append(group_tasks)

    # Pre-allocate storage
    all_cd = np.full((N, n_eta), np.nan, dtype=float)
    all_pc = np.full((N, n_eta), np.nan, dtype=float)
    all_converged = np.zeros(N, dtype=bool)
    all_timings = np.zeros(N, dtype=float)

    t_total_start = time.time()
    n_valid = 0
    n_failed = 0
    n_completed = 0

    # Step 3: Submit groups to worker pool
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=n_workers, mp_context=ctx,
        initializer=_training_worker_init,
        initargs=(base_solver_params, steady, observable_scale, *mesh_params),
    ) as executor:
        future_to_group = {}
        for g_idx, group in enumerate(groups):
            future = executor.submit(_training_worker_solve_group, group)
            future_to_group[future] = g_idx

        # Step 4: Collect results with heartbeat monitoring
        pending = set(future_to_group.keys())
        while pending:
            done, pending = wait(pending, timeout=300, return_when=FIRST_COMPLETED)

            # Heartbeat if nothing completed in the 5-minute timeout window
            if not done:
                wall = time.time() - t_total_start
                print(
                    f"[HEARTBEAT] {datetime.datetime.now().strftime('%H:%M:%S')}  "
                    f"waiting for workers...  "
                    f"{n_completed}/{N} complete  "
                    f"wall={_fmt_duration(wall)}",
                    flush=True,
                )
                continue

            for future in done:
                g_idx = future_to_group[future]
                try:
                    group_results = future.result()
                except Exception as e:
                    print(f"[ERROR] Worker group {g_idx} failed: {e}", flush=True)
                    continue

                for r in group_results:
                    idx = r["index"]
                    n_completed += 1
                    cd = np.array(r["current_density"])
                    pc = np.array(r["peroxide_current"])
                    conv_mask = np.array(r["converged_mask"])
                    frac = sum(conv_mask) / n_eta

                    if frac >= min_converged_fraction:
                        all_cd[idx] = _interpolate_failed_points(cd, conv_mask, phi_applied_values)
                        all_pc[idx] = _interpolate_failed_points(pc, conv_mask, phi_applied_values)
                        all_converged[idx] = True
                        n_valid += 1
                    else:
                        n_failed += 1

                # Progress output (per-group completion)
                if verbose:
                    wall = time.time() - t_total_start
                    remaining = N - n_completed
                    avg = wall / max(n_completed, 1)
                    eta = avg * remaining
                    pct = n_completed / N * 100
                    print(
                        f"[GROUP {g_idx+1}/{len(groups)}] completed  "
                        f"{n_completed}/{N} ({pct:.1f}%)  "
                        f"valid={n_valid} fail={n_failed}  "
                        f"wall={_fmt_duration(wall)}  "
                        f"ETA={_fmt_duration(eta)}  "
                        f"avg={wall/max(n_completed,1):.1f}s/sample",
                        flush=True,
                    )

                # Checkpoint after each group completes
                _save_checkpoint(
                    output_path, parameter_samples, all_cd, all_pc,
                    all_converged, all_timings, phi_applied_values, n_completed,
                )

    # Save final output
    valid_mask = all_converged
    np.savez_compressed(
        output_path,
        parameters=parameter_samples[valid_mask],
        current_density=all_cd[valid_mask],
        peroxide_current=all_pc[valid_mask],
        phi_applied=phi_applied_values,
        all_parameters=parameter_samples,
        all_current_density=all_cd,
        all_peroxide_current=all_pc,
        all_converged=all_converged,
        all_timings=all_timings,
        n_completed=N,
    )

    return {
        "parameters": parameter_samples[valid_mask],
        "current_density": all_cd[valid_mask],
        "peroxide_current": all_pc[valid_mask],
        "phi_applied": phi_applied_values,
        "n_valid": int(valid_mask.sum()),
        "n_total": N,
        "n_failed": n_failed,
        "timings": all_timings,
    }
```

#### 1.5.6 Time Estimate

With 8 workers and parameter-space warm-starts:
- Base solve time per sample: ~75s (no warm-start) -> ~60s (with cross-sample warm-start)
- 3,000 samples / 8 workers x 60s = ~6.25 hours
- Add ~30 min for worker initialization, serialization, Firedrake TSFC cache warmup
- **Phase 1 parallel estimate: ~7.0 hours** (vs ~8.3h without warm-starts)

### 1.6 Data Merging

After generating the new 3,000 samples, merge with the existing v9 data:

```python
v9 = np.load("StudyResults/surrogate_v9/training_data_500.npz")
# v9 has: parameters (445,4), current_density (445,22), peroxide_current (445,22), phi_applied (22,)

# Verify phi_applied grids match
assert np.allclose(v9["phi_applied"], new_phi_applied)

# Concatenate
merged_params = np.concatenate([v9["parameters"], new_valid_params], axis=0)
merged_cd = np.concatenate([v9["current_density"], new_valid_cd], axis=0)
merged_pc = np.concatenate([v9["peroxide_current"], new_valid_pc], axis=0)

# Deduplicate: remove any samples with identical k0 values (unlikely with different seeds but defensive)
# Save
np.savez_compressed(
    "StudyResults/surrogate_v11/training_data_merged.npz",
    parameters=merged_params,
    current_density=merged_cd,
    peroxide_current=merged_pc,
    phi_applied=phi_applied,
    n_v9=len(v9["parameters"]),
    n_new=len(new_valid_params),
)
```

### 1.7 Train/Test Split

Hold out 15% of the merged data as a fixed test set for all model comparisons. Use stratified splitting (stratify by k0_1 quartile, matching the pattern in `scripts/surrogate/train_nn_surrogate.py` `_stratified_split()`):

```python
from scripts.surrogate.train_nn_surrogate import _stratified_split
# If _stratified_split is not importable, implement inline:
# Sort by log10(k0_1) quartile, sample proportionally from each

rng = np.random.default_rng(seed=777)
n_test = max(50, int(len(merged_params) * 0.15))
perm = rng.permutation(len(merged_params))
test_idx = perm[:n_test]
train_idx = perm[n_test:]
```

Save split indices alongside merged data for reproducibility.

### 1.8 Checkpoint File Format

```
StudyResults/surrogate_v11/
    training_data_new_3000.npz                  # raw new samples (final)
    training_data_new_3000.npz.checkpoint.npz   # intermediate checkpoint
    training_data_merged.npz                    # merged v9 + v11
    split_indices.npz                           # train_idx, test_idx
```

### 1.9 Failure Recovery

- If a worker process crashes (Firedrake segfault, PETSc error), the `ProcessPoolExecutor` raises `BrokenProcessPool`. Catch this, log the failed group, restart the executor, and continue from the checkpoint.
- If the script is killed entirely, re-running it loads the `.checkpoint.npz` file and continues from the last completed group (the existing `generate_training_dataset()` already supports `resume_from`; the parallel version mirrors this).
- Individual sample failures (SNES divergence) are recorded as `converged=False` and skipped, matching the existing pattern in `generate_training_dataset()`.

### 1.10 Required Code Changes for Phase 1

| Change | File | Complexity | Impact |
|--------|------|-----------|--------|
| Add `_order_samples_nearest_neighbor()` | `Surrogate/training.py` | Low | 15-25% speedup via parameter-space ordering |
| Add `initial_solutions` param to `generate_training_data_single()` | `Surrogate/training.py` | Low | Enables cross-sample warm-start seeding |
| Add `return_solutions` param to `generate_training_data_single()` | `Surrogate/training.py` | Low | Exposes converged U_data for warm-start chain |
| Add `generate_training_dataset_parallel()` with grouped workers | `Surrogate/training.py` | Medium | 8x parallelism + parameter-space warm-starts |
| Add `_training_worker_init()` / `_training_worker_solve_group()` | `Surrogate/training.py` | Medium | Worker process setup for parallel generation |
| Reuse CD sweep solutions for PC sweep IC | `Surrogate/training.py` | Low | ~40% faster per sample (avoid double cold-start) |
| Add `forward_only` mode to `solve_bv_curve_points_with_warmstart()` | `FluxCurve/bv_point_solve/__init__.py` | Medium | ~40% faster per point (skip adjoint tape) |

**Priority order for implementation:**
1. `generate_training_dataset_parallel()` with batched grouped workers (required for overnight timing)
2. `_order_samples_nearest_neighbor()` (easy win, used by the parallel function)
3. Reuse CD solutions for PC sweep IC (moderate win, changes to `generate_training_data_single()`)
4. `forward_only` mode (optional, requires changes to BV solver -- skip adjoint annotation entirely)

**Details on `return_solutions`**: This parameter instructs `generate_training_data_single()` to extract the converged `U_data` from `_all_points_cache` before calling `_clear_caches()`, and include it in the return dict as `"converged_solutions"`. The worker function uses this to seed the next sample's `_cross_eval_cache`.

**Details on CD-to-PC IC reuse**: Currently `generate_training_data_single()` calls `_clear_caches()` between the CD and PC sweeps (lines 102 and 125). To reuse CD solutions for the PC sweep, save `_all_points_cache` contents after the CD sweep, call `_clear_caches()`, then populate `_cross_eval_cache` with the saved CD solutions before the PC sweep. Since the physics is identical (same PDE, different observable integral), the CD solution is an excellent IC for the PC sweep, reducing steps from ~8-15 to ~1-3 per point.

**Details on `forward_only` mode**: Training data generation does not need adjoint gradients. The current code passes `target_flux=np.zeros(n_eta)` (a dummy target), which means the adjoint gradient is computed against zero (wasting time). A `forward_only=True` flag would wrap the entire solve in `adj.stop_annotating()` and skip `adj.Control`, `ReducedFunctional`, and `rf.derivative()` calls. This is an optional optimization that requires changes to `solve_bv_curve_points_with_warmstart()` in `FluxCurve/bv_point_solve/__init__.py`.

### 1.11 Estimated Phase 1 Timeline

| Sub-step | Wall-clock |
|----------|-----------|
| Sample ordering (nearest-neighbor chain) | 2 sec |
| Worker pool init + TSFC cache warm | 5 min |
| 3,000 samples (8 grouped workers, warm-started) | ~6.5 h |
| Data merging + dedup + split | 1 min |
| **Phase 1 Total** | **~7.0 h** |

---

## Phase 2: Model Training (~1 hour)

### 2.1 Overview

Train four model variants on the merged dataset, all using the same train/test split:

| Model | Class | File | Est. Time |
|-------|-------|------|-----------|
| A. Baseline RBF | `BVSurrogateModel` | `Surrogate/surrogate_model.py` | 2 min |
| B. POD-RBF (with log PC) | `PODRBFSurrogateModel` | `Surrogate/pod_rbf_model.py` | 15-30 min |
| C. POD-RBF (no log PC) | `PODRBFSurrogateModel` | `Surrogate/pod_rbf_model.py` | 15-30 min |
| D. NN Ensemble (5 members) | `NNSurrogateModel` via `train_nn_ensemble()` | `Surrogate/nn_training.py` | 60-90 min |

### 2.2 Model A: Baseline RBF

```python
from Surrogate.surrogate_model import BVSurrogateModel, SurrogateConfig

config_rbf = SurrogateConfig(
    smoothing_cd=0.0,
    smoothing_pc=1e-3,  # from v9 cross-validation
    log_space_k0=True,
    normalize_inputs=True,
)
model_rbf = BVSurrogateModel(config=config_rbf)
model_rbf.fit(
    parameters=train_params,
    current_density=train_cd,
    peroxide_current=train_pc,
    phi_applied=phi_applied,
)
```

Also run PC smoothing cross-validation on the new larger dataset:
```python
# Sweep smoothing_pc over [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1]
# Use 5-fold CV to find best PC smoothing
# (matches pattern in scripts/surrogate/build_surrogate.py _cross_validate_pc_smoothing)
```

**Estimated time:** 2 minutes (RBF fitting is fast, CV adds ~1 min).

### 2.3 Model B: POD-RBF with log1p PC transform

This is the main Track A model. The log1p transform on peroxide current addresses the sign-magnitude issue where PC values span orders of magnitude and have different signs across the voltage range.

```python
from Surrogate.pod_rbf_model import PODRBFSurrogateModel, PODRBFConfig

config_pod_log = PODRBFConfig(
    variance_threshold=0.999,          # retain 99.9% variance
    kernel="thin_plate_spline",
    degree=1,
    log_space_k0=True,
    normalize_inputs=True,
    optimize_smoothing=True,           # per-mode LOO/k-fold CV
    n_smoothing_candidates=30,         # 30 log-spaced from 1e-8 to 1e0
    smoothing_range=(1e-8, 1e0),
    log_transform_pc=True,             # KEY: sign-preserving log1p for PC
    max_modes=None,                    # auto from variance threshold
)

model_pod_log = PODRBFSurrogateModel(config=config_pod_log)
model_pod_log.fit(
    parameters=train_params,
    current_density=train_cd,
    peroxide_current=train_pc,
    phi_applied=phi_applied,
    verbose=True,
)
```

**Estimated time:** 15-30 minutes. The bottleneck is `_optimize_mode_smoothing()` which runs LOO-CV (for N<=100) or 5-fold CV (for N>100) over 30 smoothing candidates for each retained POD mode. With ~2,700 training samples, N>100, so 5-fold CV is used. Each mode: 30 candidates x 5 folds = 150 RBF fits. Expected ~8-12 modes. Total: ~1,500 RBF fits. Each fit on 2,700 samples takes ~0.1s = ~150s = ~2.5 min. With overhead for evaluation: ~15 min.

### 2.4 Model C: POD-RBF without log1p PC transform

Identical to Model B but with `log_transform_pc=False`. This serves as a control to measure the isolated effect of the log1p transform.

```python
config_pod_nolog = PODRBFConfig(
    variance_threshold=0.999,
    kernel="thin_plate_spline",
    degree=1,
    log_space_k0=True,
    normalize_inputs=True,
    optimize_smoothing=True,
    n_smoothing_candidates=30,
    smoothing_range=(1e-8, 1e0),
    log_transform_pc=False,
    max_modes=None,
)

model_pod_nolog = PODRBFSurrogateModel(config=config_pod_nolog)
model_pod_nolog.fit(
    parameters=train_params,
    current_density=train_cd,
    peroxide_current=train_pc,
    phi_applied=phi_applied,
    verbose=True,
)
```

**Estimated time:** 15-30 minutes (same as Model B).

### 2.5 Model D: NN Ensemble (5 members)

```python
from Surrogate.nn_training import NNTrainingConfig, train_nn_ensemble

nn_config = NNTrainingConfig(
    epochs=5000,
    lr=1e-3,
    weight_decay=1e-4,
    patience=500,                # early stopping
    batch_size=None,             # full-batch (N~2700 fits in CPU memory)
    checkpoint_interval=500,
    T_0=500,                     # CosineAnnealingWarmRestarts period
    T_mult=2,
    eta_min=1e-6,
    hidden=128,                  # ResNetMLP width
    n_blocks=4,                  # 4 residual blocks
    monotonicity_weight=0.01,    # physics regularization: CD should decrease with eta
    smoothness_weight=0.001,     # physics regularization: smooth I-V curves
)

nn_models, nn_meta = train_nn_ensemble(
    parameters=train_params,
    current_density=train_cd,
    peroxide_current=train_pc,
    phi_applied=phi_applied,
    n_ensemble=5,
    config=nn_config,
    output_dir="StudyResults/surrogate_v11/nn_ensemble",
    base_seed=42,
    val_fraction=0.15,
    verbose=True,
)
```

Architecture per `ResNetMLP` in `Surrogate/nn_model.py`:
- Input: 4 (log10(k0_1), log10(k0_2), alpha_1, alpha_2)
- Hidden: 128, 4 ResBlocks with LayerNorm + SiLU
- Output: 44 (22 CD + 22 PC)
- Parameters: ~100K weights

Each ensemble member: up to 5,000 epochs, early stopping at patience=500. With ~2,700 training samples, full-batch training. Each epoch ~5ms on CPU. Expected convergence at ~2000-3000 epochs = ~15s per member. 5 members = ~75s.

But with the CosineAnnealingWarmRestarts schedule (T_0=500, T_mult=2, periods at 500, 1000, 2000), the scheduler tends to push training to the full 5000 epochs before early stopping kicks in, so estimate ~25s per member = ~2 min total for NN.

### 2.6 NN Hyperparameter Sweep

Train additional ensemble configurations to find the best:

| Config | hidden | n_blocks | monotonicity_weight | smoothness_weight |
|--------|--------|----------|---------------------|-------------------|
| D1 (default) | 128 | 4 | 0.01 | 0.001 |
| D2 (wider) | 256 | 4 | 0.01 | 0.001 |
| D3 (deeper) | 128 | 6 | 0.01 | 0.001 |
| D4 (no physics) | 128 | 4 | 0.0 | 0.0 |
| D5 (strong physics) | 128 | 4 | 0.1 | 0.01 |

Each ensemble = 5 members x ~25s = ~2 min. 5 configs = ~10 min. Very affordable.

### 2.7 Model Serialization

```python
import pickle

# RBF
with open("StudyResults/surrogate_v11/model_rbf_baseline.pkl", "wb") as f:
    pickle.dump(model_rbf, f)

# POD-RBF (save both variants)
with open("StudyResults/surrogate_v11/model_pod_rbf_log.pkl", "wb") as f:
    pickle.dump(model_pod_log, f)
with open("StudyResults/surrogate_v11/model_pod_rbf_nolog.pkl", "wb") as f:
    pickle.dump(model_pod_nolog, f)

# NN ensemble -- use NNSurrogateModel.save() which saves model.pt + normalizers.npz + metadata.npz
for i, model in enumerate(nn_models):
    model.save(f"StudyResults/surrogate_v11/nn_ensemble/member_{i}/saved_model")
```

### 2.8 Estimated Phase 2 Timeline

| Sub-step | Wall-clock |
|----------|-----------|
| Model A: Baseline RBF + CV | 2 min |
| Model B: POD-RBF (log PC) | 20 min |
| Model C: POD-RBF (no log PC) | 20 min |
| Model D: NN ensemble sweep (5 configs x 5 members) | 15 min |
| Serialization | 1 min |
| **Phase 2 Total** | **~1 hour** |

---

## Phase 3: Evaluation and Comparison (~20 minutes)

### 3.1 Surrogate Accuracy Metrics

Run `validate_surrogate()` from `Surrogate/validation.py` on every model using the held-out test set:

```python
from Surrogate.validation import validate_surrogate, print_validation_report

models_to_evaluate = {
    "RBF-baseline": model_rbf,
    "POD-RBF-log": model_pod_log,
    "POD-RBF-nolog": model_pod_nolog,
    "NN-D1-ensemble-mean": nn_ensemble_wrapper_d1,
    "NN-D2-ensemble-mean": nn_ensemble_wrapper_d2,
    # ... etc for each NN config
}

results = {}
for name, model in models_to_evaluate.items():
    metrics = validate_surrogate(model, test_params, test_cd, test_pc)
    results[name] = metrics
    print(f"\n--- {name} ---")
    print_validation_report(metrics)
```

For NN ensembles, create a wrapper that returns the ensemble mean prediction:
```python
from Surrogate.nn_training import predict_ensemble

class EnsembleMeanWrapper:
    """Wraps predict_ensemble to match the validate_surrogate API."""
    def __init__(self, models):
        self.models = models
        self.phi_applied = models[0].phi_applied
    def predict_batch(self, parameters):
        ens = predict_ensemble(self.models, parameters)
        return {
            "current_density": ens["current_density_mean"],
            "peroxide_current": ens["peroxide_current_mean"],
            "phi_applied": ens["phi_applied"],
        }
```

Key metrics to compare:
- `cd_rmse`, `pc_rmse` (lower is better)
- `cd_mean_relative_error`, `pc_mean_relative_error` (NRMSE, lower is better)
- `cd_max_abs_error`, `pc_max_abs_error`

**Critical comparison:** PC RMSE and PC mean relative error. This is where the baseline RBF fails and where k0_2 recovery degrades.

### 3.2 Parameter Recovery Test

Run the cascade inference (`run_cascade_inference()` from `Surrogate/cascade.py`) on 10 synthetic test cases with known true parameters. This directly measures the quantity we care about: can the improved surrogate recover k0_2 accurately?

```python
from Surrogate.cascade import run_cascade_inference, CascadeConfig

cascade_cfg = CascadeConfig(
    pass1_weight=0.5,
    pass2_weight=2.0,
    pass1_maxiter=60,
    pass2_maxiter=60,
    polish_maxiter=30,
    polish_weight=1.0,
    skip_polish=False,
    verbose=True,
)

# 10 test parameter sets (spanning the range, including edge cases)
test_true_params = [
    (1e-3, 1e-4, 0.5, 0.5),   # baseline case
    (1e-2, 1e-3, 0.4, 0.6),   # high k0
    (1e-5, 1e-6, 0.3, 0.3),   # low k0
    (5e-4, 5e-5, 0.5, 0.5),   # intermediate
    (1e-3, 1e-5, 0.6, 0.4),   # large k0 ratio
    (1e-4, 1e-3, 0.4, 0.5),   # inverted k0 ratio
    (1e-3, 1e-4, 0.2, 0.8),   # extreme alpha
    (1e-3, 1e-4, 0.7, 0.3),   # inverted alpha
    (5e-2, 5e-3, 0.45, 0.55), # near upper bound
    (5e-5, 5e-6, 0.55, 0.45), # near lower bound
]

for name, model in best_models.items():
    errors_k0_1, errors_k0_2, errors_a1, errors_a2 = [], [], [], []
    for true_k0_1, true_k0_2, true_a1, true_a2 in test_true_params:
        # Generate synthetic target from the model itself
        target_pred = model.predict(true_k0_1, true_k0_2, true_a1, true_a2)
        target_cd = target_pred["current_density"]
        target_pc = target_pred["peroxide_current"]

        # Add small noise to simulate real data
        rng = np.random.default_rng(42)
        target_cd += rng.normal(0, 0.01 * np.abs(target_cd).max(), target_cd.shape)
        target_pc += rng.normal(0, 0.01 * np.abs(target_pc).max(), target_pc.shape)

        # Initial guess (deliberately off by ~1 decade in k0, ~0.2 in alpha)
        initial_k0 = [true_k0_1 * 3.0, true_k0_2 * 0.3]
        initial_alpha = [0.5, 0.5]

        result = run_cascade_inference(
            surrogate=model,
            target_cd=target_cd,
            target_pc=target_pc,
            initial_k0=initial_k0,
            initial_alpha=initial_alpha,
            bounds_k0_1=(1e-6, 1.0),
            bounds_k0_2=(1e-7, 0.1),
            bounds_alpha=(0.1, 0.9),
            config=cascade_cfg,
        )

        # Compute relative errors
        errors_k0_1.append(abs(result.best_k0_1 - true_k0_1) / true_k0_1)
        errors_k0_2.append(abs(result.best_k0_2 - true_k0_2) / true_k0_2)
        errors_a1.append(abs(result.best_alpha_1 - true_a1))
        errors_a2.append(abs(result.best_alpha_2 - true_a2))

    print(f"\n{name} -- Parameter Recovery (10 test cases):")
    print(f"  k0_1 error: mean={np.mean(errors_k0_1)*100:.1f}%, max={np.max(errors_k0_1)*100:.1f}%")
    print(f"  k0_2 error: mean={np.mean(errors_k0_2)*100:.1f}%, max={np.max(errors_k0_2)*100:.1f}%")
    print(f"  alpha_1 abs error: mean={np.mean(errors_a1):.4f}, max={np.max(errors_a1):.4f}")
    print(f"  alpha_2 abs error: mean={np.mean(errors_a2):.4f}, max={np.max(errors_a2):.4f}")
```

**Success criterion:** k0_2 recovery error drops from the current 7.5-31% (baseline RBF) to <5% with the best new model.

### 3.3 Multi-Start Recovery Test

Also run `run_multistart_inference()` from `Surrogate/multistart.py` on the same 10 test cases with the best model, to verify the cascade + multistart pipeline works end-to-end.

### 3.4 Comparison Table Output

Generate a CSV comparison table:

```python
import csv

with open("StudyResults/surrogate_v11/model_comparison.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "model", "n_train", "n_test",
        "cd_rmse", "pc_rmse",
        "cd_nrmse_%", "pc_nrmse_%",
        "cd_max_err", "pc_max_err",
        "k0_2_recovery_mean_%", "k0_2_recovery_max_%",
        "fit_time_s",
    ])
    for name, metrics in results.items():
        writer.writerow([
            name, len(train_params), metrics["n_test"],
            f"{metrics['cd_rmse']:.6e}", f"{metrics['pc_rmse']:.6e}",
            f"{metrics['cd_mean_relative_error']*100:.2f}",
            f"{metrics['pc_mean_relative_error']*100:.2f}",
            f"{metrics['cd_max_abs_error']:.6e}",
            f"{metrics['pc_max_abs_error']:.6e}",
            f"{recovery_results[name]['k0_2_mean']*100:.1f}",
            f"{recovery_results[name]['k0_2_max']*100:.1f}",
            f"{fit_times[name]:.1f}",
        ])
```

### 3.5 Visualization

Generate diagnostic plots:

1. **Per-voltage-point error curves:** For each model, plot RMSE as a function of eta. This reveals whether the improved models fix the intermediate-voltage PC accuracy problem.

2. **POD mode spectrum:** Plot singular values and cumulative variance for the POD-RBF models. Show how many modes are needed and whether log1p changes the spectrum.

3. **NN loss curves:** Already generated by `_plot_loss_curves()` during training. Compile into a single figure.

4. **Parameter recovery scatter:** For the 10 test cases, scatter plot of (true_k0_2 vs recovered_k0_2) for each model. The closer to the diagonal, the better.

Save all plots to `StudyResults/surrogate_v11/plots/`.

### 3.6 Select Best Model

After all evaluations, automatically select the best model based on a composite score:
```python
# Weighted composite: 40% PC RMSE + 30% k0_2 recovery + 20% CD RMSE + 10% PC max error
composite = (
    0.4 * normalize(pc_rmse) +
    0.3 * normalize(k0_2_recovery_mean) +
    0.2 * normalize(cd_rmse) +
    0.1 * normalize(pc_max_error)
)
```

Copy the best model to `StudyResults/surrogate_v11/best_model/` for use in the inference pipeline.

### 3.7 Estimated Phase 3 Timeline

| Sub-step | Wall-clock |
|----------|-----------|
| Surrogate accuracy (all models) | 2 min |
| Parameter recovery (10 cases x ~7 models) | 5 min |
| Multi-start recovery (10 cases, best model) | 5 min |
| Comparison table + CSV | 1 min |
| Visualization plots | 5 min |
| Best model selection + copy | 1 min |
| **Phase 3 Total** | **~20 min** |

---

## Script Structure and Progress Monitoring

### Script Skeleton

The implementer should create a single file at:

**`scripts/surrogate/overnight_train_v11.py`**

```python
#!/usr/bin/env python
"""Overnight surrogate training pipeline v11.

Usage:
    cd /Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse
    /Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/venv-firedrake/bin/python \
        scripts/surrogate/overnight_train_v11.py 2>&1 | tee StudyResults/surrogate_v11/run.log

Phases:
    1. Data generation (parallel, warm-started, ~7h)
    2. Model training (~1h)
    3. Evaluation (~20min)

Resume: Re-run the same command. Checkpoints are loaded automatically.
"""

import sys, os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

OUTPUT_DIR = os.path.join(_ROOT, "StudyResults", "surrogate_v11")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Phase flags (set to False to skip) ----
RUN_PHASE_1 = True
RUN_PHASE_2 = True
RUN_PHASE_3 = True

def phase_1_data_generation():
    """Generate ~3000 new training samples using 8 parallel grouped workers."""
    ...

def phase_2_model_training():
    """Train RBF, POD-RBF, and NN ensemble models."""
    ...

def phase_3_evaluation():
    """Evaluate all models, generate comparison table and plots."""
    ...

if __name__ == "__main__":
    import time
    t0 = time.time()

    if RUN_PHASE_1:
        print("=" * 78)
        print("  PHASE 1: DATA GENERATION")
        print("=" * 78)
        phase_1_data_generation()

    if RUN_PHASE_2:
        print("=" * 78)
        print("  PHASE 2: MODEL TRAINING")
        print("=" * 78)
        phase_2_model_training()

    if RUN_PHASE_3:
        print("=" * 78)
        print("  PHASE 3: EVALUATION")
        print("=" * 78)
        phase_3_evaluation()

    total = time.time() - t0
    print(f"\nTotal elapsed: {total/3600:.1f}h")
```

Additionally, the implementer will need to add the following to `Surrogate/training.py`:
- `_order_samples_nearest_neighbor()` -- nearest-neighbor greedy chain (Section 1.3)
- `_training_worker_init()` -- worker process initializer (Section 1.5.2)
- `_training_worker_solve_group()` -- group-sequential worker function with cross-sample warm-starts (Section 1.5.3)
- `generate_training_dataset_parallel()` -- parallel orchestrator with heartbeat monitoring (Section 1.5.5)
- Modifications to `generate_training_data_single()` for `return_solutions` and `initial_solutions` params (Section 1.10)

### Progress Output Format

All print statements use `flush=True` so output streams immediately to the terminal and log file, even when piped through `tee` or redirected. The script should be run as:

```bash
cd /Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse
/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/venv-firedrake/bin/python \
    scripts/surrogate/overnight_train_v11.py 2>&1 | tee StudyResults/surrogate_v11/run.log
```

#### Per-Group Completion

When a worker group finishes, the main process prints:

```
[GROUP 1/8] completed  375/3000 (12.5%)  valid=338 fail=37  wall=58m 12s  ETA=6h 29m  avg=9.3s/sample
[GROUP 2/8] completed  750/3000 (25.0%)  valid=679 fail=71  wall=1h 02m  ETA=3h 06m  avg=5.0s/sample
```

#### Per-Sample Detail (within each worker, printed to stdout by the worker process)

```
  [W3]  #42  k0=[1.234e-03,5.678e-04]  a=[0.45,0.52]  OK  conv=20/22  dt=62.3s
  [W1]  #38  k0=[8.901e-02,3.456e-03]  a=[0.38,0.61]  OK  conv=22/22  dt=55.1s
  [W5]  #44  k0=[1.111e-05,2.222e-06]  a=[0.30,0.30]  SKIP(68%<80%)  conv=15/22  dt=89.2s
```

#### Periodic Progress Summaries (after each group completion)

```
======================================================================
  PROGRESS SUMMARY at 750/3000 (25.0%)
  Wall elapsed  : 1h 02m
  ETA remaining : 3h 06m
  Valid samples : 679 (90.5%)
  Failed/skipped: 71 (9.5%)
  Avg per sample: 5.0s (parallel, 8 workers)
  Avg per sample: 59.3s (sequential equivalent)
  Convergence by region:
    Wide (k0 > 0.01) : 95.2% converge rate
    Wide (k0 < 0.01) : 82.1% converge rate
    Focused           : 91.3% converge rate
  Min/Max/Med sample time: 32.1s / 142.7s / 61.4s
======================================================================
```

#### Heartbeat During Long Waits

When no worker group has completed for 5 minutes, the `wait(timeout=300)` heartbeat fires:

```
[HEARTBEAT] 02:34:12  waiting for workers...  375/3000 complete  wall=1h 02m
```

#### Checkpoint Saves

```
=== CHECKPOINT at 750/3000, elapsed: 1h 02m, valid: 679, failed: 71 ===
```

#### Phase Transitions

```
##########################################################################
  PHASE 1 COMPLETE: DATA GENERATION
  Total time     : 6h 52m
  Total samples  : 3000
  Valid          : 2714 (90.5%)
  Failed/skipped : 286 (9.5%)
  Saved to       : StudyResults/surrogate_v11/training_data_new_3000.npz
##########################################################################

==========================================================================
  PHASE 2: MODEL TRAINING
==========================================================================
```

#### Post-Hoc Log Analysis

The structured format (bracketed prefixes) makes it easy to grep for specific event types:

```bash
grep '^\[GROUP'     StudyResults/surrogate_v11/run.log | tail -5   # overall progress
grep 'SKIP\|FAIL\|ERROR' StudyResults/surrogate_v11/run.log        # failures
grep 'PHASE'        StudyResults/surrogate_v11/run.log              # phase transitions
grep 'HEARTBEAT'    StudyResults/surrogate_v11/run.log              # heartbeats
grep 'CHECKPOINT'   StudyResults/surrogate_v11/run.log              # checkpoints
```

---

## Summary of Estimated Timeline

| Phase | Sub-step | Wall-clock |
|-------|----------|-----------|
| 1 | Sample ordering | 2 sec |
| 1 | Worker pool init + TSFC cache warm | 5 min |
| 1 | 3,000 samples (8 grouped workers, warm-started, ~60s/sample) | ~6.5h |
| 1 | Data merging + dedup + split | 1 min |
| **1 Total** | | **~7.0h** |
| 2 | Baseline RBF + CV | 2 min |
| 2 | POD-RBF x 2 variants | 40 min |
| 2 | NN ensemble x 5 configs | 15 min |
| 2 | Serialization | 1 min |
| **2 Total** | | **~1h** |
| 3 | Accuracy metrics | 2 min |
| 3 | Parameter recovery | 10 min |
| 3 | Plots + comparison | 6 min |
| 3 | Best model selection | 1 min |
| **3 Total** | | **~20 min** |
| **Grand Total** | | **~8.3h** |

This fits within the 12-hour overnight budget with ~3.7 hours of margin for unexpected delays (slow convergence, worker restarts, etc.).

---

## Key Files Referenced

| File | Role in Plan |
|------|-------------|
| `Surrogate/training.py` | `generate_training_data_single()`, `generate_training_dataset()`, `_save_checkpoint()` -- extend with `generate_training_dataset_parallel()`, `_order_samples_nearest_neighbor()`, `_training_worker_init()`, `_training_worker_solve_group()` |
| `Surrogate/sampling.py` | `generate_multi_region_lhs_samples()`, `ParameterBounds` |
| `Surrogate/surrogate_model.py` | `BVSurrogateModel`, `SurrogateConfig` (Model A) |
| `Surrogate/pod_rbf_model.py` | `PODRBFSurrogateModel`, `PODRBFConfig` (Models B, C) |
| `Surrogate/nn_model.py` | `NNSurrogateModel`, `ResNetMLP`, `ZScoreNormalizer` (Model D) |
| `Surrogate/nn_training.py` | `NNTrainingConfig`, `train_nn_ensemble()`, `predict_ensemble()` (Model D training) |
| `Surrogate/validation.py` | `validate_surrogate()`, `print_validation_report()` (Phase 3) |
| `Surrogate/cascade.py` | `run_cascade_inference()`, `CascadeConfig` (Phase 3 recovery test) |
| `Surrogate/multistart.py` | `run_multistart_inference()` (Phase 3 recovery test) |
| `FluxCurve/bv_parallel.py` | `BVParallelPointConfig`, `BVPointSolvePool`, `_bv_worker_init` -- pattern for parallel worker design (NOT used directly for training) |
| `FluxCurve/bv_point_solve/__init__.py` | `solve_bv_curve_points_with_warmstart()` -- the core BV solver with eta continuation, predictor steps, bridge points, hub state |
| `FluxCurve/bv_point_solve/cache.py` | `_clear_caches()`, `_cross_eval_cache`, `_all_points_cache` -- cache globals for warm-start management |
| `FluxCurve/bv_point_solve/predictor.py` | `_build_sweep_order()`, `_apply_predictor()`, `_solve_bridge_points()` -- eta continuation internals |
| `FluxCurve/bv_point_solve/forward.py` | `_solve_cached_fast_path()` -- sequential fast path with cached ICs |
| `scripts/_bv_common.py` | `make_bv_solver_params()`, `compute_i_scale()`, `FOUR_SPECIES_CHARGED`, `SNES_OPTS_CHARGED` |
| `StudyResults/surrogate_v9/training_data_500.npz` | Existing 445-sample training data to merge |
