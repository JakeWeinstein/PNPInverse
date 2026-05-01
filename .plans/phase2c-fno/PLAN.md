# Phase 2c: FNO Surrogate -- Detailed Implementation Plan

**Date:** 2026-03-17
**Goal:** Evaluate and implement a Fourier Neural Operator variant for the parameter-to-IV-curve mapping, with autograd gradients, implementing the existing surrogate API.
**Output:** `Surrogate/fno_model.py`, trained artifacts in `data/surrogate_models/fno/`

---

## Honest Assessment: Is FNO a Good Fit?

**Short answer: Probably not for this problem, but the evaluation itself is scientifically valuable.**

FNO's core advantages are:
1. **Resolution invariance** -- train on coarse, evaluate on fine grids. With only 22 voltage points and no plan to evaluate on finer grids, this advantage is irrelevant.
2. **Spectral learning** -- efficient for smooth, periodic solutions on regular grids. I-V curves are smooth but NOT periodic, and 22 points means at most 11 Fourier modes, which is barely enough to justify spectral processing over direct MLP output.
3. **Spatial-field-to-spatial-field mapping** -- FNO's canonical use case maps an input function on a grid to an output function on a grid. Our input is a 4D parameter vector, not a spatial field. This requires an unconventional architecture: a lifting network to map parameters onto the voltage grid before Fourier layers.

**Why proceed anyway:**
- The roadmap committed to evaluating FNO as part of the surrogate zoo. A rigorous negative result is valuable -- it tells us FNO is not worth pursuing for this class of problems and justifies focusing resources on GP/DeepONet/PEDS.
- The adapted architecture (parameter lifting + 1D Fourier layers along voltage) could still capture spectral structure in I-V curves that MLP misses, even at 22 points.
- DIFNO (Derivative-Informed FNO) training on both outputs and Frechet derivatives could yield better gradient accuracy than standard NN + FD, which matters for the inverse problem. This is the strongest justification.

**The plan includes an explicit go/no-go gate after Step 3.** If the standard FNO does not match MLP baseline performance, we abandon DIFNO and document findings.

---

## Baseline Performance (Target to Beat)

From `StudyResults/surrogate_fidelity/fidelity_summary.json` (479 test samples):

| Model | CD mean NRMSE | CD 95th NRMSE | PC mean NRMSE | PC 95th NRMSE |
|-------|--------------|--------------|--------------|--------------|
| NN ensemble | 0.31% | 0.62% | 0.56% (skewed by outliers) | 44.8% |
| RBF baseline | 0.26% | 1.45% | 1.96% | 71.0% |

The NN ensemble is the primary baseline. FNO must achieve CD mean NRMSE < 0.5% and PC mean NRMSE comparable to or better than NN ensemble to be considered competitive.

---

## Architecture Design

### Problem Formulation

- **Input:** 4 parameters `[k0_1, k0_2, alpha_1, alpha_2]` (after log-transform and normalization)
- **Output:** 2 curves on 22 voltage points: `current_density` (22,) and `peroxide_current` (22,)
- **Voltage grid:** `phi_applied` (22,), non-uniform spacing, stored in training data

### Adapted FNO Architecture

Since the input is NOT a spatial field, we need a modified FNO:

```
Parameters (4,)
    |
    v
[Lifting MLP]  -- maps (4,) -> (22, d_model) by:
    1. MLP: (4,) -> (d_lift,)        # standard MLP
    2. Expand: (d_lift,) -> (22, d_lift)  # broadcast to voltage grid
    3. Concatenate voltage coordinates: (22, d_lift+1)
    4. Pointwise linear: (22, d_lift+1) -> (22, d_model)
    |
    v
[Fourier Layer 1]  -- spectral conv (1D FFT along voltage dim)
    |
[Fourier Layer 2]
    |
[Fourier Layer 3]
    |
[Fourier Layer 4]
    |
    v
[Projection MLP]  -- maps (22, d_model) -> (22, 2)  [CD, PC per voltage point]
```

Each **Fourier Layer** contains:
- 1D FFT along the 22-point voltage dimension
- Learnable complex weights on the retained Fourier modes
- Inverse FFT
- Plus a local linear bypass (W*x)
- GELU activation

### Key Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `d_model` (channel width) | 64 | Small enough for 22 points; larger wastes compute |
| `n_fourier_layers` | 4 | Standard FNO depth |
| `n_modes` | 8 | Max meaningful modes for 22-point signal (Nyquist = 11); start conservative |
| `d_lift` | 128 | MLP width for parameter lifting |
| `activation` | GELU | Standard for FNO |

### Two-Channel Output vs Separate Models

Use a **single model with 2-channel output** (CD + PC per voltage point), analogous to how the NN model concatenates 22 CD + 22 PC into 44 outputs. The Fourier layers operate on all channels jointly, allowing cross-channel spectral learning.

---

## Implementation Steps

### Step 1: FNO Architecture and Model Class (1-2 hours)

**File:** `Surrogate/fno_model.py`

**Tasks:**
- [ ] 1.1 Implement `FourierLayer1d` module: 1D spectral convolution with mode truncation, local linear bypass, and activation
- [ ] 1.2 Implement `ParameterLiftingNet` module: MLP to lift (4,) to (22, d_model), concatenating voltage grid coordinates
- [ ] 1.3 Implement `FNO1d` (the full torch.nn.Module): lifting -> N Fourier layers -> projection
- [ ] 1.4 Implement `FNOSurrogateModel` class with the same public API as `NNSurrogateModel`:
  - `fit()` -- trains the FNO
  - `predict(k0_1, k0_2, alpha_1, alpha_2)` -> dict with `current_density`, `peroxide_current`, `phi_applied`
  - `predict_batch(parameters)` -> batch version
  - `save(path)` / `load(path)` class method
  - Properties: `n_eta`, `phi_applied`, `is_fitted`, `training_bounds`
- [ ] 1.5 Implement `predict_with_gradients(parameters)` method using `torch.autograd`:
  - Takes (N, 4) parameter array
  - Returns predictions AND Jacobian d(output)/d(params) via `torch.autograd.functional.jacobian` or `torch.func.jacrev`
  - This is the key advantage over FD-based gradients

**API contract (must match exactly):**
```python
model = FNOSurrogateModel(...)
result = model.predict(k0_1, k0_2, alpha_1, alpha_2)
# result = {'current_density': (22,), 'peroxide_current': (22,), 'phi_applied': (22,)}

result = model.predict_batch(params)  # params shape (N, 4)
# result = {'current_density': (N, 22), 'peroxide_current': (N, 22), 'phi_applied': (22,)}
```

**Design decisions:**
- Input normalization: Use `ZScoreNormalizer` from `nn_model.py` (import, do not duplicate)
- Output normalization: Z-score on concatenated [CD, PC] (same as NN model)
- Log-space k0: Always apply log10 to k0_1, k0_2 before normalization (matching NN convention)
- Voltage grid encoding: Normalize `phi_applied` to [0, 1] range as positional input to lifting layer
- Device handling: Support 'cpu' and 'mps' (M4 MacBook)

### Step 2: Training Infrastructure (1-2 hours)

**File:** `Surrogate/fno_training.py`

**Tasks:**
- [ ] 2.1 Implement `FNOTrainingConfig` dataclass:
  ```
  epochs: 5000
  lr: 1e-3
  weight_decay: 1e-4
  patience: 500
  batch_size: None (full batch, 3194 samples fits in memory)
  n_modes: 8
  d_model: 64
  n_layers: 4
  d_lift: 128
  checkpoint_interval: 500
  ```
- [ ] 2.2 Implement `train_fno_surrogate()` function following the pattern in `nn_training.py`:
  - Data loading, log-transform, normalization
  - Auto train/val split (85/15)
  - AdamW + CosineAnnealingWarmRestarts scheduler
  - Early stopping with patience
  - CSV logging (epoch, train_loss, val_loss, cd_rmse, pc_rmse, lr, elapsed)
  - Checkpoint saving
  - Loss curve plotting (reuse `_plot_loss_curves` from nn_training or factor it out)
  - Returns `(FNOSurrogateModel, history_dict)`
- [ ] 2.3 Implement a training script `scripts/train_fno.py`:
  - Loads `data/surrogate_models/training_data_merged.npz`
  - Runs training with configurable hyperparameters via argparse
  - Saves model to `data/surrogate_models/fno/`
  - Runs validation on held-out data and prints report

### Step 3: Initial Training and Go/No-Go Evaluation (2-3 hours)

**This is the critical gate. Do NOT proceed to DIFNO unless this step passes.**

**Tasks:**
- [ ] 3.1 Train standard FNO on the 3,194-sample dataset with default hyperparameters
- [ ] 3.2 Evaluate on the same 479-sample test set used for the fidelity study
- [ ] 3.3 Compute error metrics: CD/PC RMSE, mean NRMSE, 95th percentile NRMSE, max NRMSE
- [ ] 3.4 Compare against NN ensemble baseline:

**Go/No-Go Criteria:**

| Metric | GO threshold | NO-GO threshold |
|--------|-------------|----------------|
| CD mean NRMSE | < 0.5% (relaxed vs NN's 0.31%) | > 1.0% |
| PC mean NRMSE | < 2.0% (relaxed vs NN's 0.56%) | > 5.0% |
| CD 95th NRMSE | < 1.5% | > 3.0% |

- **If GO:** Proceed to Step 4 (hyperparameter tuning) and Step 5 (DIFNO)
- **If NO-GO but close:** Try hyperparameter sweep (Step 4 only), then re-evaluate
- **If NO-GO and not close:** Document findings, write assessment report, archive code. FNO is not suited for this problem at 22-point resolution. This is a valid and useful result.

- [ ] 3.5 **Ablation: Fourier vs MLP.** To isolate whether spectral processing helps, train a "FNO with 0 Fourier modes" (i.e., only the local linear bypass, no FFT). If this matches full FNO, the Fourier component adds no value.

- [ ] 3.6 **Ablation: Number of modes.** Train with n_modes in {4, 6, 8, 10} to see if the 22-point resolution limits are hit.

- [ ] 3.7 Write go/no-go assessment to `data/surrogate_models/fno/go_no_go_assessment.md`

### Step 4: Hyperparameter Tuning (if GO) (2-3 hours)

**Only execute if Step 3 passes go/no-go.**

**Tasks:**
- [ ] 4.1 Grid search over key hyperparameters:
  - `n_modes`: {4, 6, 8, 10}
  - `d_model`: {32, 64, 128}
  - `n_layers`: {3, 4, 6}
  - `d_lift`: {64, 128, 256}
- [ ] 4.2 This is 4 x 3 x 3 x 3 = 108 configurations. At ~2 min each on CPU, this is ~3.5 hours. Use RTX 4070 to cut to ~30 min.
  - Alternative: Use a coarser search (e.g., 20 random configs) if time-constrained
- [ ] 4.3 Select best configuration based on validation CD+PC combined RMSE
- [ ] 4.4 Retrain best config 3 times with different seeds to check stability
- [ ] 4.5 Save best model to `data/surrogate_models/fno/best_model/`

### Step 5: DIFNO -- Derivative-Informed Training (if GO) (3-4 hours)

**Only execute if Step 3 passes go/no-go.**

DIFNO trains on BOTH function values (I-V curves) AND Frechet derivatives (Jacobian d(IV)/d(params)). This is the strongest justification for FNO on this problem -- even if prediction accuracy matches MLP, gradient accuracy may be superior.

**Tasks:**
- [ ] 5.1 **Generate derivative training data:**
  - Option A (preferred): Use PDE adjoint from Firedrake pyadjoint to compute d(IV)/d(params) at a subset of training points (e.g., 200-500 samples). Each adjoint solve gives the exact Jacobian (4 adjoint solves for 4 parameters, per sample).
  - Option B (fallback): Use finite differences on the PDE solver. More expensive (8 PDE solves per sample for central FD on 4 params) but no pyadjoint dependency.
  - Option C (cheapest): Use the existing NN ensemble's autograd Jacobian as a noisy proxy. Only viable if Phase 2e (autograd retrofit) is complete.
  - **Decision: Start with Option C if available, then validate with Option A on a small subset.**

- [ ] 5.2 Implement DIFNO training loss:
  ```
  L_total = L_data + lambda_deriv * L_derivative

  L_data     = MSE(FNO(params), IV_true)
  L_derivative = MSE(d_FNO/d_params, d_IV/d_params_true)
  ```
  where `d_FNO/d_params` is computed via `torch.autograd` through the FNO, and `d_IV/d_params_true` comes from the PDE adjoint or FD reference.

- [ ] 5.3 Implement `DiFNOTrainingConfig` extending `FNOTrainingConfig`:
  - `lambda_deriv`: derivative loss weight (start at 0.1, schedule up)
  - `deriv_data_path`: path to Jacobian training data
  - `deriv_schedule`: 'constant' | 'linear_ramp' | 'cosine_ramp'

- [ ] 5.4 Train DIFNO and evaluate:
  - Prediction accuracy (same metrics as Step 3)
  - **Gradient accuracy:** Compare autograd Jacobian from DIFNO vs:
    - PDE adjoint reference (gold standard)
    - NN ensemble autograd Jacobian (if available)
    - Central FD on PDE solver (expensive but definitive)
  - Metric: relative Jacobian error per parameter, per test sample

- [ ] 5.5 Assess whether DIFNO gradient accuracy justifies the extra training complexity

### Step 6: Final Evaluation and Integration (1-2 hours)

**Tasks:**
- [ ] 6.1 Run the full validation suite (`Surrogate/validation.py`) on the best FNO/DIFNO model
- [ ] 6.2 Run the same 479-sample fidelity study and produce metrics comparable to `fidelity_summary.json`
- [ ] 6.3 Test autograd gradient computation:
  - Compute Jacobian via `torch.autograd` for 100 test points
  - Compare against central FD on the FNO itself (should match to ~1e-5 relative)
  - Compare against central FD on the PDE solver for 10 test points (the real gradient accuracy test)
- [ ] 6.4 Measure inference speed:
  - Single prediction (latency)
  - Batch prediction (throughput, N=100, N=1000)
  - Single gradient evaluation (Jacobian for 1 sample)
- [ ] 6.5 Register `FNOSurrogateModel` in `Surrogate/__init__.py`
- [ ] 6.6 Save final trained model artifacts to `data/surrogate_models/fno/`
- [ ] 6.7 Write summary report to `data/surrogate_models/fno/evaluation_report.md`

---

## File Inventory

| File | Status | Purpose |
|------|--------|---------|
| `Surrogate/fno_model.py` | **NEW** | FNO architecture + FNOSurrogateModel class |
| `Surrogate/fno_training.py` | **NEW** | Training loop, config, hyperparameter sweep |
| `scripts/train_fno.py` | **NEW** | Training entry point script |
| `Surrogate/__init__.py` | MODIFY | Register FNOSurrogateModel export |
| `data/surrogate_models/fno/` | **NEW DIR** | Trained model artifacts |
| `data/surrogate_models/fno/go_no_go_assessment.md` | **NEW** | Go/no-go decision after Step 3 |
| `data/surrogate_models/fno/evaluation_report.md` | **NEW** | Final evaluation results |

---

## Dependencies

**Python packages (install into venv-firedrake):**
- `torch` -- already available (used by NN surrogate)
- No additional packages required. We implement the FNO from scratch in PyTorch rather than using the `neuraloperator` library, because:
  1. Our architecture is non-standard (parameter input, not field input)
  2. The 1D case with 22 points is trivial to implement
  3. Avoids dependency management issues
  4. Full control over the autograd graph for DIFNO

**Codebase dependencies:**
- `Surrogate/nn_model.py` -- import `ZScoreNormalizer` (reuse, do not duplicate)
- `Surrogate/validation.py` -- for standardized error evaluation
- `data/surrogate_models/training_data_merged.npz` -- training data (3194 samples)
- Phase 1 (Data Audit) results for any data augmentation decisions

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| FNO fails go/no-go at Step 3 | **HIGH** (60-70%) | Low -- expected given 22 points | This is a valid result. Document it, move on to GP/DeepONet/PEDS. Budget 4 hours max before the gate. |
| Fourier modes aliasing on 22-point grid | Medium | Medium | Cap n_modes at 10 (Nyquist limit 11). Ablation in Step 3.6 will reveal this. |
| Non-periodic I-V curves cause Gibbs ringing | Medium | Low | FNO handles this via the local linear bypass. Zero-padding the FFT may help. |
| DIFNO derivative data is expensive to generate | Medium | Medium | Start with NN ensemble Jacobians (cheap proxy), validate on small PDE adjoint subset. |
| Autograd through FNO is slow for full Jacobian | Low | Low | Use `torch.func.vmap(jacrev(...))` for batched Jacobian. 22-point output with 4 inputs = small Jacobian (22x4 per curve, 44x4 total). |

---

## Time Budget

| Step | Estimated Time | Cumulative |
|------|---------------|-----------|
| Step 1: Architecture | 1.5 hours | 1.5 hours |
| Step 2: Training infra | 1.5 hours | 3 hours |
| Step 3: Go/no-go eval | 2 hours | 5 hours |
| **GO/NO-GO GATE** | -- | -- |
| Step 4: Hyperparameter tuning | 2.5 hours | 7.5 hours |
| Step 5: DIFNO | 3.5 hours | 11 hours |
| Step 6: Final eval | 1.5 hours | 12.5 hours |

**If NO-GO at Step 3:** Total time = ~5 hours. This is acceptable.
**If GO through DIFNO:** Total time = ~12.5 hours across 2-3 sessions.

---

## Appendix: FNO 1D Spectral Convolution -- Implementation Reference

For a 1D signal `v` of length `N=22` with `d_model` channels:

```python
# Forward pass of one Fourier layer
v_ft = torch.fft.rfft(v, dim=-2)           # (batch, N//2+1, d_model)
v_ft_filtered = v_ft[:, :n_modes, :] @ R   # R is (n_modes, d_model, d_model) complex
v_out = torch.fft.irfft(v_ft_filtered, n=N, dim=-2)  # back to (batch, N, d_model)
v_out = v_out + W @ v                       # local linear bypass
v_out = activation(v_out)
```

For `N=22`, `rfft` produces 12 frequency components. With `n_modes=8`, we retain 8 of 12 frequencies. This is a reasonable compression ratio that discards only the highest-frequency components.
