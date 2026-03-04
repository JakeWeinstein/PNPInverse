# Overnight Surrogate Training Plan

## Overview

**Total estimated wall-clock time:** ~12 hours
**Machine:** 10-core CPU (macOS Darwin)
**Python:** `/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/venv-firedrake/bin/python`
**Working directory:** `/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse`
**Output root:** `StudyResults/surrogate_v10/`

The plan is structured as a single script (`scripts/surrogate/overnight_train_v10.py`) that executes three phases sequentially. Each phase is idempotent and checkpoint-resumable. If the script crashes at hour 7, re-running it picks up from the last checkpoint.

---

## Phase 1: Training Data Generation (~8.5 hours)

### 1.1 Goal

Expand the training set from 445 to ~3,145 valid samples by generating ~3,000 new samples. This ~7x data increase is critical because:
- POD-RBF per-mode smoothing optimization needs dense coverage to avoid overfitting
- NN ensemble (5 models with different 85/15 splits) needs enough data that each member trains on ~2,500 samples
- The focused-region samples near the Mangan2025 operating point (where k0_2 recovery matters most) will improve PC accuracy in the critical region

### 1.2 Sampling Strategy

Use `generate_multi_region_lhs_samples()` from `/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/Surrogate/sampling.py` to generate samples in two tiers:

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

### 1.3 Parallelization Strategy

The current `generate_training_dataset()` in `/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/Surrogate/training.py` processes samples sequentially -- each sample calls `generate_training_data_single()` which itself calls `solve_bv_curve_points_with_warmstart()` twice (once for CD, once for PC). Each sample takes ~60-90 seconds.

**Sequential estimate:** 3,000 samples x 75s = 62.5 hours. Far too slow.

**Parallel approach:** Create a new function `generate_training_dataset_parallel()` that uses Python's `concurrent.futures.ProcessPoolExecutor` with the spawn context (matching the pattern in `/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/FluxCurve/bv_parallel.py`).

Key design decisions, modeled after `BVPointSolvePool`:

1. **Spawn context** (required for Firedrake/PETSc):
   ```python
   ctx = multiprocessing.get_context("spawn")
   executor = ProcessPoolExecutor(
       max_workers=n_workers,
       mp_context=ctx,
       initializer=_training_worker_init,
       initargs=(base_solver_params, steady_config, observable_scale, mesh_Nx, mesh_Ny, mesh_beta),
   )
   ```

2. **Worker initializer** (mirrors `_bv_worker_init` pattern from `bv_parallel.py`):
   ```python
   def _training_worker_init(base_solver_params, steady_config, obs_scale, Nx, Ny, beta):
       import os
       os.environ["OMP_NUM_THREADS"] = "1"
       os.environ.setdefault("FIREDRAKE_TSFC_KERNEL_CACHE_DIR", "/tmp/firedrake-tsfc")
       os.environ.setdefault("PYOP2_CACHE_DIR", "/tmp/pyop2")
       os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
       global _TRAIN_WORKER_STATE
       from Forward.bv_solver import make_graded_rectangle_mesh
       _TRAIN_WORKER_STATE = {
           "base_solver_params": base_solver_params,
           "steady": steady_config,
           "observable_scale": obs_scale,
           "mesh": make_graded_rectangle_mesh(Nx=Nx, Ny=Ny, beta=beta),
       }
   ```

3. **Worker function** -- wraps `generate_training_data_single()`:
   ```python
   def _training_worker_solve(task):
       """task = (sample_index, k0_1, k0_2, alpha_1, alpha_2, phi_applied_values)"""
       idx, k0_1, k0_2, alpha_1, alpha_2, phi_applied = task
       st = _TRAIN_WORKER_STATE
       result = generate_training_data_single(
           k0_values=[k0_1, k0_2],
           alpha_values=[alpha_1, alpha_2],
           phi_applied_values=phi_applied,
           base_solver_params=st["base_solver_params"],
           steady=st["steady"],
           observable_scale=st["observable_scale"],
           mesh=st["mesh"],
       )
       return {
           "index": idx,
           "current_density": result["current_density"].tolist(),
           "peroxide_current": result["peroxide_current"].tolist(),
           "converged_mask": result["converged_mask"].tolist(),
           "n_converged": result["n_converged"],
       }
   ```

4. **Worker count:** Use `n_workers = 8` on this 10-core machine (leaving 2 cores for OS + main process). Each worker sets `OMP_NUM_THREADS=1`, so total CPU usage = 8 cores. This matches the `BVPointSolvePool` default of `cpu_count() - 1` but we use 8 instead of 9 to leave more headroom for the long overnight run.

5. **Batched submission with checkpointing:** Submit tasks in batches of 40 (5 per worker). After each batch completes, checkpoint to disk. This bounds memory usage and ensures progress is saved every ~5 minutes.

   ```python
   BATCH_SIZE = 40  # 5 samples per worker
   for batch_start in range(0, n_remaining, BATCH_SIZE):
       batch_tasks = tasks[batch_start : batch_start + BATCH_SIZE]
       futures = [executor.submit(_training_worker_solve, t) for t in batch_tasks]
       for future in as_completed(futures):
           result = future.result()
           # store in arrays, update counters
       # checkpoint after each batch
       _save_checkpoint(...)
   ```

**Parallel time estimate:** 3,000 samples / 8 workers * 75s per sample = ~7.8 hours. Add ~30 min overhead for worker initialization, serialization, and Firedrake TSFC cache warmup = **~8.3 hours**.

### 1.4 Data Merging

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
    "StudyResults/surrogate_v10/training_data_merged.npz",
    parameters=merged_params,
    current_density=merged_cd,
    peroxide_current=merged_pc,
    phi_applied=phi_applied,
    n_v9=len(v9["parameters"]),
    n_new=len(new_valid_params),
)
```

### 1.5 Train/Test Split

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

### 1.6 Checkpoint File Format

```
StudyResults/surrogate_v10/
    training_data_new_3000.npz           # raw new samples (checkpointed)
    training_data_new_3000.npz.checkpoint.npz  # intermediate checkpoint
    training_data_merged.npz             # merged v9 + v10
    split_indices.npz                    # train_idx, test_idx
```

### 1.7 Failure Recovery

- If a worker process crashes (Firedrake segfault, PETSc error), the `ProcessPoolExecutor` raises `BrokenProcessPool`. Catch this, log the failed batch, restart the executor, and continue from the checkpoint.
- If the script is killed entirely, re-running it loads the `.checkpoint.npz` file and continues from the last completed batch (the existing `generate_training_dataset()` already supports `resume_from`; the parallel version should mirror this).
- Individual sample failures (SNES divergence) are recorded as `converged=False` and skipped, matching the existing pattern in `generate_training_dataset()`.

### 1.8 Estimated Phase 1 Timeline

| Sub-step | Wall-clock |
|----------|-----------|
| Worker pool init + TSFC cache warm | 5 min |
| 3,000 new samples (8 workers) | ~7.8 h |
| Data merging + dedup + split | 1 min |
| **Phase 1 Total** | **~8.3 h** |

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
    output_dir="StudyResults/surrogate_v10/nn_ensemble",
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
with open("StudyResults/surrogate_v10/model_rbf_baseline.pkl", "wb") as f:
    pickle.dump(model_rbf, f)

# POD-RBF (save both variants)
with open("StudyResults/surrogate_v10/model_pod_rbf_log.pkl", "wb") as f:
    pickle.dump(model_pod_log, f)
with open("StudyResults/surrogate_v10/model_pod_rbf_nolog.pkl", "wb") as f:
    pickle.dump(model_pod_nolog, f)

# NN ensemble -- use NNSurrogateModel.save() which saves model.pt + normalizers.npz + metadata.npz
for i, model in enumerate(nn_models):
    model.save(f"StudyResults/surrogate_v10/nn_ensemble/member_{i}/saved_model")
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

## Phase 3: Evaluation and Comparison (~30 minutes)

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

with open("StudyResults/surrogate_v10/model_comparison.csv", "w", newline="") as f:
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

Save all plots to `StudyResults/surrogate_v10/plots/`.

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

Copy the best model to `StudyResults/surrogate_v10/best_model/` for use in the inference pipeline.

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

## Script Structure

The implementer should create a single file at:

**`scripts/surrogate/overnight_train_v10.py`**

```python
#!/usr/bin/env python
"""Overnight surrogate training pipeline v10.

Usage:
    cd /Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse
    /Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/venv-firedrake/bin/python \
        scripts/surrogate/overnight_train_v10.py 2>&1 | tee StudyResults/surrogate_v10/run.log

Phases:
    1. Data generation (parallel, ~8.5h)
    2. Model training (~1h)
    3. Evaluation (~20min)

Resume: Re-run the same command. Checkpoints are loaded automatically.
"""

import sys, os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
os.chdir(_ROOT)
sys.path.insert(0, _ROOT)

OUTPUT_DIR = os.path.join(_ROOT, "StudyResults", "surrogate_v10")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Phase flags (set to False to skip) ----
RUN_PHASE_1 = True
RUN_PHASE_2 = True
RUN_PHASE_3 = True

def phase_1_data_generation():
    """Generate ~3000 new training samples using 8 parallel workers."""
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

Additionally, the implementer will need to add a helper function to `Surrogate/training.py`:

**`generate_training_dataset_parallel()`** -- a new function that mirrors the existing `generate_training_dataset()` but uses `ProcessPoolExecutor` with spawn context, OMP_NUM_THREADS=1 per worker, and batched submission with checkpointing. The worker pattern should follow `_bv_worker_init` / `_bv_worker_solve_point` from `FluxCurve/bv_parallel.py`.

---

## Summary of Estimated Timeline

| Phase | Sub-step | Wall-clock |
|-------|----------|-----------|
| 1 | Worker init | 5 min |
| 1 | 3,000 samples (8 workers, 75s/sample) | ~7.8h |
| 1 | Merge + split | 1 min |
| **1 Total** | | **~8.3h** |
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
| **Grand Total** | | **~9.7h** |

This fits within the 12-hour overnight budget with ~2.3 hours of margin for unexpected delays (slow convergence, worker restarts, etc.).

---

## Key Files Referenced

| File | Role in Plan |
|------|-------------|
| `Surrogate/training.py` | `generate_training_data_single()`, `generate_training_dataset()`, `_save_checkpoint()` -- extend with `generate_training_dataset_parallel()` |
| `Surrogate/sampling.py` | `generate_multi_region_lhs_samples()`, `ParameterBounds` |
| `Surrogate/surrogate_model.py` | `BVSurrogateModel`, `SurrogateConfig` (Model A) |
| `Surrogate/pod_rbf_model.py` | `PODRBFSurrogateModel`, `PODRBFConfig` (Models B, C) |
| `Surrogate/nn_model.py` | `NNSurrogateModel`, `ResNetMLP`, `ZScoreNormalizer` (Model D) |
| `Surrogate/nn_training.py` | `NNTrainingConfig`, `train_nn_ensemble()`, `predict_ensemble()` (Model D training) |
| `Surrogate/validation.py` | `validate_surrogate()`, `print_validation_report()` (Phase 3) |
| `Surrogate/cascade.py` | `run_cascade_inference()`, `CascadeConfig` (Phase 3 recovery test) |
| `Surrogate/multistart.py` | `run_multistart_inference()` (Phase 3 recovery test) |
| `FluxCurve/bv_parallel.py` | `BVParallelPointConfig`, `BVPointSolvePool`, `_bv_worker_init`, `_bv_worker_solve_point` -- pattern for parallel data generation |
| `FluxCurve/bv_point_solve/cache.py` | `_clear_caches()` -- must be called per worker for training data generation |
| `scripts/_bv_common.py` | `make_bv_solver_params()`, `compute_i_scale()`, `FOUR_SPECIES_CHARGED`, `SNES_OPTS_CHARGED` |
| `StudyResults/surrogate_v9/training_data_500.npz` | Existing 445-sample training data to merge |

---

## Addendum: Warm-Start Strategies and Progress Monitoring

This addendum details the warm-start mechanisms that dramatically reduce per-sample solve time during training data generation, and specifies the streaming progress feedback format for overnight monitoring.

---

### A. Voltage Sweep Ordering (Eta Continuation)

#### A.1 What Already Exists

Every call to `generate_training_data_single()` (in `Surrogate/training.py`) invokes `solve_bv_curve_points_with_warmstart()` (in `FluxCurve/bv_point_solve/__init__.py`). Inside that function, the 22 voltage points are NOT solved in their input order. Instead, they are reordered by `_build_sweep_order()` (in `FluxCurve/bv_point_solve/predictor.py`, lines 246-310):

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

Between branches, the "hub state" mechanism saves the converged solution at eta = -0.5 (the near-equilibrium hub). When the positive branch starts at eta = +1.0, the solver restores the hub state rather than warm-starting from the eta = -46.5 solution.

#### A.2 Within-Sample Warm-Start Chain

For each consecutive pair of voltage points in the sweep order, the converged solution from point N is used as the initial condition for point N+1:

1. **Carry state**: `carry_U_data = tuple(d.data_ro.copy() for d in U.dat)` -- a tuple of numpy arrays, one per sub-function in the mixed function space.

2. **Predictor step**: Before the forward solve at each point, `_apply_predictor()` extrapolates from up to 3 prior converged solutions using quadratic Lagrange interpolation. If the extrapolation produces DOFs that deviate by more than 10x from the carry state, it reverts to simple copy. Falls back to linear extrapolation (2 points) or simple copy (1 point).

3. **Bridge points**: When the gap between consecutive solved etas exceeds `max_eta_gap` (default 3.0 in training), forward-only intermediate points are auto-inserted and solved to carry the warm-start across the gap.

4. **SER adaptive pseudo-timestep**: Each point starts with `dt_initial` (typically 0.5) and adaptively grows/shrinks the pseudo-timestep. Growth cap: 4.0, shrink factor: 0.5, max dt ratio: 20.0. Warm-started points typically converge in 3-8 steps (vs 20-100 for cold starts).

5. **Reduced step cap for warm-started points**: 20 steps (vs the full 100). Since the IC is close to the converged solution, 20 steps is more than sufficient.

#### A.3 Why This Matters for Training Speed

Without eta continuation, each of the 22 voltage points would start from the initial blob condition and need 50-100 pseudo-time steps to reach steady state. With continuation:
- The first point (eta = -0.5, near equilibrium) converges in ~10-15 steps from the blob IC.
- Subsequent points converge in 3-8 steps each thanks to the warm-started IC.
- Bridge points fill gaps seamlessly.
- Total per-sample time: ~60-90s (both CD and PC observables), dominated by the MUMPS LU factorization cost per SNES step, not by the number of steps.

**No changes are needed to leverage this**: it is already active inside `generate_training_data_single()`. The overnight script benefits automatically.

#### A.4 Important Detail: Two Separate Sweeps Per Sample

`generate_training_data_single()` calls `solve_bv_curve_points_with_warmstart()` **twice** per sample -- once for `observable_mode="current_density"` and once for `observable_mode="peroxide_current"`. Between these calls, it calls `_clear_caches()`. This means:

- The CD sweep benefits from full eta continuation (22 points, warm-start chain).
- The PC sweep starts cold -- it does NOT reuse the CD sweep's converged solutions.
- Each sweep takes ~30-45s.

**Potential optimization**: Modify `generate_training_data_single()` to skip the `_clear_caches()` between CD and PC sweeps. Instead, pass the CD sweep's carry states to the PC sweep as ICs. The PC sweep then needs only 1-3 steps per point (re-converge with different observable assembly, same physics). This could cut per-sample time by ~40% (from ~75s to ~45s).

---

### B. Parameter-Space Warm Starts (Cross-Sample Ordering)

#### B.1 Current State: No Cross-Sample Warm-Starting

In the existing `generate_training_dataset()`, samples are processed in their **input order** (the order returned by LHS sampling). Each sample calls `generate_training_data_single()`, which calls `_clear_caches()` at the start. This means:

- **Sample i's eta = -0.5 starts from the blob IC** (not from sample i-1's converged solution at eta = -0.5).
- The first voltage point of every sample is a cold start.
- There is no parameter-space locality exploitation.

This is the single largest efficiency opportunity in the training pipeline.

#### B.2 Why Parameter-Space Ordering Helps

For two LHS samples with similar parameters (k0_1, k0_2, alpha_1, alpha_2), the converged PDE solutions at the same eta are also similar. If sample i has parameters (1e-3, 1e-4, 0.5, 0.5) and sample i+1 has parameters (1.2e-3, 1.1e-4, 0.52, 0.48), then sample i's converged solution at eta = -0.5 is an excellent IC for sample i+1 at eta = -0.5.

When the IC is close to the true solution, the SNES solver converges in 1-3 Newton iterations (vs 5-15 from a cold start). With the adaptive SER timestep, this translates to:
- Cold start: ~8-15 pseudo-time steps for the first voltage point.
- Parameter-warm start: ~2-4 pseudo-time steps for the first voltage point.
- Subsequent voltage points already benefit from eta continuation.

**Expected speedup**: 15-25% reduction in total per-sample time for ordered samples vs random-order samples.

#### B.3 Implementation: Nearest-Neighbor Chain in Log Parameter Space

Sort the LHS samples so that consecutive samples are close in (log10(k0_1), log10(k0_2), alpha_1, alpha_2) space:

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

**Complexity**: O(N^2) for N=3000 samples — takes ~2 seconds (negligible vs 8-hour data generation).

#### B.4 Passing Warm-Start State Between Samples

The key code change is in the training loop. Instead of calling `_clear_caches()` at the start of each sample, carry the last sample's converged solutions forward via the existing `_cross_eval_cache` mechanism:

```python
from FluxCurve.bv_point_solve import cache as _cache_mod

# Pre-populate cache with previous sample's converged solutions
_cache_mod._clear_caches()
if prev_solutions is not None:
    for cache_idx, u_data in prev_solutions.items():
        _cache_mod._cross_eval_cache[cache_idx] = u_data
```

The `_cross_eval_cache` is already read at the start of `solve_bv_curve_points_with_warmstart()` — the first solved point checks for a cached IC before falling back to the blob condition.

#### B.5 Parallel Worker Strategy: Batched Sequential Within Ordered Groups

Divide the 3000 nearest-neighbor-ordered samples into 8 groups (one per worker). Within each group, samples are adjacent in parameter space. Each worker processes its group sequentially, carrying warm-start state between samples. Different groups run in parallel across workers.

```
Worker 0: samples [0, 1, 2, ..., 374]       (parameter-space neighbors)
Worker 1: samples [375, 376, ..., 749]       (parameter-space neighbors)
...
Worker 7: samples [2625, 2626, ..., 2999]    (parameter-space neighbors)
```

Each worker solves ~375 samples sequentially (3000/8), benefiting from parameter-space warm-starts within its group. The first sample of each group is a cold start, but subsequent samples reuse the previous sample's solutions.

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
        _cache_mod._clear_caches()
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

**Expected benefit**: ~1.5 hours saved (3000 samples * 15s savings / 8 workers).

---

### C. Cross-Sample Warm-Start Cache Architecture

#### C.1 Existing Cache Infrastructure

The BV point solve module (`FluxCurve/bv_point_solve/cache.py`) defines three global caches:

- **`_cross_eval_cache`**: Persists across calls within the same process. Keyed by original index (position in phi_applied_values). After a successful sweep, the first solved point's IC is cached. Designed for optimizer iterations where parameters change slightly between evaluations. **This IS suitable for cross-sample warm-starting.**

- **`_all_points_cache`**: Stores converged solutions for ALL voltage points. Enables a "fast path" where subsequent evaluations skip the sequential sweep. **This is NOT suitable for training** (assumes same parameters between evaluations).

- **`_clear_caches()`**: Resets all caches. Currently called between CD and PC sweeps.

#### C.2 Per-Worker Cache Isolation

Each parallel worker process (spawn context) has its own independent copy of the module-level caches. There is no shared-memory cache between workers. This is **desirable** for the batched-sequential approach: each worker maintains its own warm-start chain independently.

Memory per worker: ~64 KB for `_cross_eval_cache` (one point's state). Negligible.

#### C.3 Why NOT BVPointSolvePool

The existing `BVPointSolvePool` is designed for parallel point solves at the SAME parameters (multiple voltage points solved concurrently during inference). It is NOT suitable for training data generation because its `BVParallelPointConfig` is frozen at initialization and its multi-observable support computes adjoint gradients we don't need.

---

### D. Streaming Progress Feedback

#### D.1 Per-Group Completion

```
[GROUP 1/8] completed  375/3000 (12.5%)  valid=338 fail=37  wall=58m 12s  ETA=6h 29m  avg=9.3s/sample
[GROUP 2/8] completed  750/3000 (25.0%)  valid=679 fail=71  wall=1h 02m  ETA=3h 06m  avg=5.0s/sample
```

#### D.2 Per-Sample Detail (within each group)

```
  [W3]  #42  k0=[1.234e-03,5.678e-04]  a=[0.45,0.52]  OK  conv=20/22  dt=62.3s
  [W1]  #38  k0=[8.901e-02,3.456e-03]  a=[0.38,0.61]  OK  conv=22/22  dt=55.1s
  [W5]  #44  k0=[1.111e-05,2.222e-06]  a=[0.30,0.30]  SKIP(68%<80%)  conv=15/22  dt=89.2s
```

#### D.3 Periodic Progress Summaries (every ~400 samples)

```
======================================================================
  PROGRESS SUMMARY at 400/3000 (13.3%)
  Wall elapsed  : 49m 30s
  ETA remaining : 5h 22m
  Valid samples : 358 (89.5%)
  Failed/skipped: 42 (10.5%)
  Avg per sample: 7.4s (parallel, 8 workers)
  Avg per sample: 59.3s (sequential equivalent)
  Convergence by region:
    Wide (k0 > 0.01) : 95.2% converge rate
    Wide (k0 < 0.01) : 82.1% converge rate
    Focused           : 91.3% converge rate
  Min/Max/Med sample time: 32.1s / 142.7s / 61.4s
======================================================================
```

#### D.4 Checkpoint Saves

```
=== CHECKPOINT at 400/3000, elapsed: 49m 30s, valid: 358, failed: 42 ===
```

#### D.5 Phase Transitions

```
##########################################################################
  PHASE 1 COMPLETE: DATA GENERATION
  Total time     : 7h 52m
  Total samples  : 3000
  Valid          : 2714 (90.5%)
  Failed/skipped : 286 (9.5%)
  Saved to       : StudyResults/surrogate_v10/training_data_new_3000.npz
##########################################################################

==========================================================================
  PHASE 2: MODEL TRAINING
==========================================================================
```

#### D.6 Heartbeat During Long Waits

Use `concurrent.futures.wait()` with a 5-minute timeout to emit heartbeats even when no workers have completed:

```python
from concurrent.futures import FIRST_COMPLETED, wait

pending = set(future_to_group.keys())
while pending:
    done, pending = wait(pending, timeout=300, return_when=FIRST_COMPLETED)

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
        # process results...
```

All print statements use `flush=True`. Run with:
```bash
python scripts/surrogate/overnight_train_v10.py 2>&1 | tee StudyResults/surrogate_v10/run.log
```

Post-hoc log analysis:
```bash
grep '^\[GROUP' run.log | tail -5     # overall progress
grep 'SKIP\|FAIL\|ERROR' run.log      # failures
grep 'PHASE' run.log                   # phase transitions
grep 'HEARTBEAT' run.log               # heartbeats
```

---

### Summary of Required Code Changes

| Change | File | Complexity | Impact |
|--------|------|-----------|--------|
| Add `_order_samples_nearest_neighbor()` | `Surrogate/training.py` | Low | 15-25% speedup |
| Add `initial_solutions` param to `generate_training_data_single()` | `Surrogate/training.py` | Low | Enables cross-sample warm-start |
| Add `return_solutions` param to `generate_training_data_single()` | `Surrogate/training.py` | Low | Exposes converged U_data for warm-start chain |
| Add `generate_training_dataset_parallel()` with grouped workers | `Surrogate/training.py` | Medium | 8x parallelism + parameter-space warm-starts |
| Add `_training_worker_init()` / `_training_worker_solve_group()` | `Surrogate/training.py` | Medium | Worker process setup |
| Add `forward_only` mode to `solve_bv_curve_points_with_warmstart()` | `FluxCurve/bv_point_solve/__init__.py` | Medium | ~40% faster (skip adjoint) |
| Reuse CD sweep solutions for PC sweep IC | `Surrogate/training.py` | Low | ~40% faster per sample |

**Priority order:**
1. `generate_training_dataset_parallel()` with batched workers (required for overnight timing)
2. `_order_samples_nearest_neighbor()` (easy win)
3. Reuse CD solutions for PC sweep IC (moderate win)
4. `forward_only` mode (optional, requires BV solver changes)
