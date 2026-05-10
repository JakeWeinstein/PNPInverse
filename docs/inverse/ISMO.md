# ISMO: Iterative Surrogate Model Optimization

## Table of Contents

1. [Overview](#overview)
2. [The Loop](#the-loop)
3. [Noise Handling](#noise-handling)
4. [Outputs](#outputs)
5. [Typical Run Profile](#typical-run-profile)
6. [Module Map](#module-map)
7. [CLI Usage](#cli-usage)
8. [Key Design Decisions](#key-design-decisions)

---

## Overview

`run_ismo()` takes a fitted surrogate and iteratively improves it by running PDE solves at the specific parameter points where the optimizer thinks the answer is. Instead of training the surrogate to be accurate everywhere (expensive), ISMO makes it accurate where it matters -- at the optimizer's solution.

The approach wraps the existing multistart + cascade surrogate optimization pipeline in an outer loop. Each iteration: optimize the surrogate, run true PDE solves at the optimizer's best guesses, measure the surrogate-PDE discrepancy, augment the training data, retrain, repeat.

Based on Lye, Mishra, Ray (2020) -- "Iterative surrogate model optimization."

---

## The Loop

### Inputs

- **Fitted surrogate**: any supported model (NN ensemble, POD-RBF, RBF baseline, GP, PCE)
- **Target I-V curves**: `target_cd`, `target_pc` (the experimental or synthetic data to fit)
- **Existing training data**: ~3,194 samples in `training_data_merged.npz`
- **PDE solver config**: mesh, steady-state tolerances, voltage grid
- **Parameter bounds**: `[k0_1, k0_2, alpha_1, alpha_2]` via `ParameterBounds`

### Step 1 -- Optimize the surrogate

Runs the full multistart pipeline (20k LHS screening, top-20 L-BFGS-B polish) then cascade refinement (CD-dominant pass, PC-dominant pass, joint polish). Produces the optimizer's best guess for the parameters.

Runtime: ~1 second. All evaluations are surrogate calls.

### Step 2 -- Pick where to run new PDE solves

Three acquisition strategies, controlled by `AcquisitionStrategy`:

| Strategy | Description |
|---|---|
| `OPTIMIZER_TRAJECTORY` (default) | Top candidates from multistart + cascade endpoints, deduplicated, filled with LHS in a neighborhood of the best point |
| `UNCERTAINTY` | 5k LHS candidates ranked by surrogate prediction uncertainty (`predict_with_uncertainty`). Falls back to trajectory if uncertainty is unavailable |
| `HYBRID` | 50/50 split between optimizer trajectory and uncertainty |

The acquisition module (`acquisition.py`) supports a finer three-way split: optimizer trajectory, GP posterior variance, and space-filling LHS. Budget fractions are configurable via `AcquisitionConfig` (default 50/30/20). If a GP model is not available, the uncertainty budget is redistributed to optimizer and space-filling.

All candidates are deduplicated in normalized log-space (k0 dimensions log-transformed, alphas linearly scaled to [0,1]) against existing training data (`min_distance_log=0.05`) and within the batch (`min_distance_batch=0.08`).

Seed is varied per iteration (`seed + i`) to avoid duplicate LHS draws.

### Step 3 -- Run true PDE solves

Calls `generate_training_data_single()` sequentially for each candidate (~30 points per iteration). Each solve takes 5-30 seconds. This is the only place PDE budget is consumed.

For batches of 8 or fewer, evaluation is sequential (avoids process spawn overhead). For larger batches, `generate_training_dataset_parallel` is used with configurable worker count.

### Step 4 -- Measure surrogate-PDE gap

At the best candidate point:

```
gap = |surr_loss - pde_loss| / max(|pde_loss|, atol)
```

If `gap < convergence_rtol` (default 5%), the loop declares convergence and stops. The convergence checker (`ISMOConvergenceChecker`) also evaluates parameter stability (normalized log-space L2 distance between successive best estimates) and stagnation.

### Step 5 -- Augment training data

New PDE results are filtered (drop NaN rows, drop duplicates in normalized log-space), then appended to the training set. Saved to a versioned `.npz` file (e.g., `ismo_iter_1/training_data_merged.npz`). The original data file is never modified. If 0 new samples pass filtering, retraining is skipped.

Provenance tracking tags each sample with its source (`"original"` or `"ismo_iter_N"`) and per-candidate NRMSE values from the surrogate-PDE comparison.

### Step 6-7 -- Record and check convergence

The `ISMOConvergenceChecker` evaluates five stopping criteria in priority order:

1. **Budget exhausted**: total PDE evals >= `max_pde_evals`
2. **Budget too low**: remaining budget < `min_useful_batch_size` (default 5)
3. **Max iterations**: iteration count >= `max_iterations`
4. **Converged**: both surrogate-PDE agreement < `agreement_tol` AND parameter stability < `stability_tol` (requires `min_iterations` completed)
5. **Stagnation**: last `stagnation_window` iterations (default 3) show < 5% relative improvement in agreement

### Step 8 -- Retrain surrogate

Retraining is dispatched by model type via `retrain_surrogate()`:

- **NN ensemble**: warm-start. Analytically correct first/last layer weights for normalizer shift (see below), then fine-tune all 5 members at reduced LR (1e-4) for 100 epochs with patience 50. If warm-start degrades quality by > 10%, falls back to from-scratch training (3000 epochs).
- **POD-RBF, RBF baseline, PCE**: refit from scratch on merged data.
- **GP**: refit with reduced iterations (100 iters, LR 0.05).

Quality check: post-retrain validation error must not exceed pre-retrain error by more than 10% (`max_degradation_ratio=1.10`).

### Step 9 -- Stall detection

If the `stagnation_window` (default 3) most recent iterations show less than 5% relative improvement in surrogate-PDE agreement, the loop stops with reason `"stagnation"`.

Budget check: if total PDE evaluations >= `total_pde_budget`, the loop stops with reason `"budget_exhausted"`.

---

## Noise Handling

`run_ismo()` adds NO noise. It takes `target_cd` / `target_pc` as-is. Noise is added upstream (in `Forward/noise.py` or by the caller). PDE solves within the ISMO loop are also noiseless -- they produce exact I-V curves for training data augmentation.

---

## Outputs

`ISMOResult` dataclass with:

| Field | Type | Description |
|---|---|---|
| `converged` | `bool` | Whether the surrogate-PDE gap reached tolerance |
| `termination_reason` | `str` | `"converged"`, `"budget_exhausted"`, `"max_iterations"`, `"no_improvement"`, `"stagnation"`, `"budget_too_low"` |
| `n_iterations` | `int` | Completed ISMO iterations |
| `total_pde_evals` | `int` | Total PDE evaluations performed |
| `iteration_history` | `tuple[ISMOIteration]` | Per-iteration snapshots |
| `final_params` | `tuple` | Best `(k0_1, k0_2, alpha_1, alpha_2)` |
| `final_loss` | `float` | PDE loss at best parameters |
| `final_surrogate_path` | `str` or `None` | Path to retrained surrogate |
| `augmented_data_path` | `str` or `None` | Path to final augmented training data |
| `total_wall_time_s` | `float` | Total wall clock time |

Additional outputs saved to disk:

- `convergence_report.json` -- full convergence summary from `ISMOConvergenceChecker`
- `ismo_iteration_log.csv` -- per-iteration CSV with agreement, stability, timing, best params
- `augmented_training_data.npz` -- final merged training dataset
- `iter_NN_iv_comparison.png` -- surrogate vs PDE I-V overlay at best point per iteration
- `convergence_curves.png` -- 2x2 summary figure (agreement, parameter estimates, training set size, stability)

---

## Typical Run Profile

Default configuration: 5 iterations max, 30 samples/iteration, 200 total PDE budget.

~6 iterations before budget exhaustion. Per iteration:

| Phase | Time |
|---|---|
| Surrogate optimization (multistart + cascade) | ~1 second |
| PDE solves (30 samples) | 2--15 minutes |
| NN ensemble retraining (warm-start) | 1--5 minutes |

**Total runtime**: 15--90 minutes depending on PDE convergence behavior.

**Expected outcome**: surrogate-PDE gap drops from ~10% to < 5%, improving k0_2 recovery accuracy.

---

## Module Map

| File | Purpose |
|---|---|
| `Surrogate/ismo.py` | Core loop, dataclasses (`ISMOConfig`, `ISMOIteration`, `ISMOResult`, `AcquisitionStrategy`), `run_ismo()`, LHS-in-neighborhood sampling, optimizer trajectory acquisition |
| `Surrogate/acquisition.py` | Acquisition strategies (`AcquisitionConfig`, `AcquisitionResult`, `select_new_samples`), three-way budget allocation (optimizer/uncertainty/spacefill), normalized log-space deduplication, GP variance ranking |
| `Surrogate/ismo_retrain.py` | Retraining dispatch (`merge_training_data`, `retrain_surrogate`, `retrain_surrogate_full`), per-model-type retraining (NN warm-start, POD-RBF/RBF/PCE/GP from-scratch), analytical weight correction for normalizer shifts, quality checks |
| `Surrogate/ismo_pde_eval.py` | PDE evaluation (`PDESolverBundle`, `evaluate_candidates_with_pde`), surrogate-PDE NRMSE comparison (`compare_surrogate_vs_pde`), data integration with provenance tracking (`integrate_new_data`), quality checks (`check_pde_quality`) |
| `Surrogate/ismo_convergence.py` | Convergence checker (`ISMOConvergenceChecker`, `ISMOConvergenceCriteria`), diagnostic records (`ISMODiagnosticRecord`), parameter stability in normalized log-space, objective improvement ratio |
| `Surrogate/ismo_diagnostics.py` | CSV logging (`ISMODiagnostics`), scatter plots (surrogate vs PDE objectives), I-V overlays at best point, convergence curve visualization (`plot_convergence_curves`), k0_2 recovery comparison bar chart |
| `scripts/surrogate/run_ismo.py` | CLI runner with argparse, surrogate loading, fallback stubs for missing modules, main orchestration loop, post-ISMO validation |

---

## CLI Usage

```
python scripts/surrogate/run_ismo.py --help
```

Key flags:

```
--surrogate-type {nn_ensemble,pod_rbf_log,pod_rbf_nolog,rbf_baseline,gp}
                        Type of surrogate model to use. (default: nn_ensemble)
--design DESIGN         Design name for NN ensemble (e.g. D1-default, D3-deeper).
                        Only used when surrogate-type=nn_ensemble. (default: D1-default)
--budget BUDGET         Maximum new PDE evaluations. (default: 200)
--max-iterations N      Hard cap on ISMO iterations. (default: 10)
--agreement-tol TOL     Surrogate-PDE agreement NRMSE tolerance. (default: 0.01)
--stability-tol TOL     Parameter stability tolerance. (default: 0.01)
--stagnation-window N   Iterations without improvement before stopping. (default: 3)
--samples-per-iter N    New PDE samples per ISMO iteration. (default: 20)
--acquisition {trust_region,exploit_explore,error_based}
                        Acquisition strategy for new sample selection. (default: trust_region)
--output-dir DIR        Output directory. (default: StudyResults/ismo)
--seed SEED             Random seed. (default: 42)
--skip-post-validation  Skip post-ISMO parameter recovery study.
--verbose               Print detailed progress.
```

Example:

```bash
python scripts/surrogate/run_ismo.py \
    --surrogate-type nn_ensemble \
    --design D1-default \
    --budget 100 \
    --max-iterations 5 \
    --samples-per-iter 20 \
    --agreement-tol 0.02 \
    --output-dir StudyResults/ismo_test \
    --skip-post-validation \
    --verbose
```

---

## Key Design Decisions

1. **Wraps multistart + cascade, does not replace them.** ISMO is an outer loop. The inner optimization pipeline is unchanged. This means any improvements to multistart or cascade automatically benefit ISMO.

2. **Convergence checked BEFORE retraining.** If the surrogate-PDE gap is already below tolerance, the loop exits without spending compute on an unnecessary retrain cycle.

3. **Sequential PDE evaluation.** Firedrake is not thread-safe. For typical batch sizes (~30 samples), the overhead of spawning parallel processes is not worth it. The parallel path (`generate_training_dataset_parallel`) is available for batches > 8 but is not the default ISMO path.

4. **Immutable data flow throughout.** All configuration dataclasses are `frozen=True`. Functions return new objects; inputs are never mutated. Training data arrays are copied before modification.

5. **Versioned training data.** Each ISMO iteration saves its augmented dataset to `ismo_iter_N/training_data_merged.npz`. The original `training_data_merged.npz` is never overwritten. This allows rollback and debugging of any iteration.

6. **NN warm-start uses analytical weight correction.** When the training data changes, the input/output normalizers change. Rather than just loading old weights with mismatched normalizers (which would cause a discontinuity), the first and last linear layers are analytically corrected:
   - Input layer: `W_new = W_old * diag(sigma_new / sigma_old)`, `b_new = b_old + W_old @ ((mu_new - mu_old) / sigma_old)`
   - Output layer: `W_new = W_old * (sigma_old / sigma_new)`, `b_new = b_old * (sigma_old / sigma_new) + (mu_old - mu_new) / sigma_new`

   This preserves the network's physical input-output mapping exactly, so fine-tuning starts from the correct baseline rather than recovering from normalizer mismatch. Avoids catastrophic forgetting.

7. **Quality gate on retraining.** If a retrained model's validation error exceeds the pre-retrain error by more than 10%, the warm-start is rejected and the model falls back to from-scratch training. If from-scratch also fails, the original model is kept.

8. **Provenance tracking.** Every sample in the augmented dataset is tagged with its source (`"original"` or `"ismo_iter_N"`) and the surrogate-PDE NRMSE at the time it was acquired. This supports post-hoc analysis of which ISMO iterations were most useful.
