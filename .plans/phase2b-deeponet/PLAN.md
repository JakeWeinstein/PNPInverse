# Phase 2b: DeepONet Surrogate with Autograd Gradients

## Overview

Train a DeepONet that learns the (parameters, voltage) -> (current_density, peroxide_current) operator, producing differentiable predictions via PyTorch autograd. The model must implement the shared surrogate API (`predict`, `predict_batch`, `n_eta`, `phi_applied`, `is_fitted`, `training_bounds`) and drop into the existing inverse-problem pipeline.

## Architecture Decision: Custom PyTorch (not DeepXDE)

**Rationale:** DeepXDE adds a heavy dependency and its DeepONet API does not natively produce the `predict(k0_1, k0_2, alpha_1, alpha_2) -> dict` signature we need. A custom implementation in ~300 lines keeps the codebase self-contained, matches the style of `nn_model.py`, and gives full control over autograd plumbing.

## Architecture Design

### Classical DeepONet formulation adapted to this problem

- **Branch net** (encodes the 4 kinetics parameters):
  - Input: `[log10(k0_1), log10(k0_2), alpha_1, alpha_2]` (z-score normalized, same transform as `nn_model.py`)
  - Architecture: MLP `4 -> 128 -> 128 -> 128 -> p` (p = latent dimension, default 64)
  - Activation: SiLU throughout, LayerNorm after each hidden layer
  - Output: p-dimensional coefficient vector (one per output channel, so 2p total for unstacked variant)

- **Trunk net** (encodes the query voltage):
  - Input: single scalar `eta` (z-score normalized)
  - Architecture: MLP `1 -> 64 -> 64 -> p`
  - Activation: SiLU, LayerNorm
  - Output: p-dimensional basis vector

- **Output combination:**
  - `CD(params, eta) = sum_k branch_cd_k(params) * trunk_k(eta) + bias_cd`
  - `PC(params, eta) = sum_k branch_pc_k(params) * trunk_k(eta) + bias_pc`
  - This is the "unstacked" variant: one shared trunk, two branch heads. The trunk learns a voltage basis shared by both outputs; the branch heads learn channel-specific coefficients.

- **Why unstacked with shared trunk:** CD and PC are measured on the same voltage grid and share electrochemical physics. A shared trunk captures the common voltage dependence. Separate branch heads allow each output channel to specialize. This is more parameter-efficient than two fully independent DeepONets.

- **Latent dimension p:** Start with p=64. This gives 64 basis functions for representing I-V curves, which is generous for 22-point outputs but allows the trunk to learn smooth basis functions.

### Parameter count estimate

| Component | Parameters |
|-----------|-----------|
| Branch input: 4 -> 128 | 640 |
| Branch hidden: 128 -> 128 (x2) | ~33K |
| Branch output: 128 -> 128 (2 heads x 64) | ~16.5K |
| Trunk input: 1 -> 64 | 128 |
| Trunk hidden: 64 -> 64 | ~4.2K |
| Trunk output: 64 -> 64 | ~4.2K |
| Bias terms | 2 |
| **Total** | **~59K** |

This is 3-4x smaller than the ResNet-MLP (~200K params), which is appropriate -- DeepONet has a strong inductive bias (operator structure) that reduces the need for raw capacity.

## Implementation Steps

### Step 1: Create `Surrogate/deeponet_model.py` -- Network + API wrapper

**File:** `/Surrogate/deeponet_model.py`

**Classes to implement:**

#### 1a. `BranchNet(nn.Module)`
```
__init__(n_params=4, hidden=128, n_hidden=2, latent_dim=64, n_channels=2)
forward(params_normalized) -> (B, n_channels * latent_dim)
```
- MLP with LayerNorm + SiLU
- Output is reshaped to `(B, n_channels, latent_dim)` -- one set of coefficients per output channel

#### 1b. `TrunkNet(nn.Module)`
```
__init__(n_input=1, hidden=64, n_hidden=2, latent_dim=64)
forward(eta_normalized) -> (n_eta, latent_dim)
```
- MLP with LayerNorm + SiLU
- Input: normalized voltage values
- Output: basis functions evaluated at each voltage point

#### 1c. `DeepONet(nn.Module)`
```
__init__(branch_net, trunk_net, n_channels=2)
forward(params_norm, eta_norm) -> (B, n_channels, n_eta)
```
- `branch_out = branch_net(params_norm)` -> `(B, n_channels, p)`
- `trunk_out = trunk_net(eta_norm)` -> `(n_eta, p)`
- `output[b, c, i] = sum_k branch_out[b, c, k] * trunk_out[i, k] + bias[c]`
- Implemented as: `torch.einsum('bcp,ep->bce', branch_out, trunk_out) + bias`

#### 1d. `DeepONetSurrogateModel` -- API wrapper class

Implements the same public interface as `NNSurrogateModel`:

| Method/Property | Behavior |
|----------------|----------|
| `__init__(latent_dim, branch_hidden, trunk_hidden, ...)` | Config, no model created yet |
| `fit(parameters, current_density, peroxide_current, phi_applied, **kwargs)` | Train from data, return self |
| `predict(k0_1, k0_2, alpha_1, alpha_2)` | Single prediction, returns dict |
| `predict_batch(parameters)` | Batch prediction, returns dict |
| `predict_torch(params_tensor)` | **NEW**: returns torch tensors with grad graph intact |
| `jacobian(k0_1, k0_2, alpha_1, alpha_2)` | **NEW**: autograd Jacobian d(output)/d(params) |
| `save(path)` / `load(path)` | Serialization |
| `n_eta`, `phi_applied`, `is_fitted`, `training_bounds` | Properties |

**Key design detail -- `predict_torch`:**
```python
def predict_torch(self, params: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Differentiable prediction -- keeps autograd graph alive.

    Parameters
    ----------
    params : torch.Tensor of shape (B, 4)
        [k0_1, k0_2, alpha_1, alpha_2] in PHYSICAL space.
        If params.requires_grad is True, gradients flow back.

    Returns
    -------
    dict with 'current_density' (B, n_eta) and 'peroxide_current' (B, n_eta),
    both as torch.Tensor on the same device as params.
    """
```

This method must:
1. Apply log10 transform to k0 columns using `torch.log10` (not numpy)
2. Apply z-score normalization using stored mean/std as torch tensors
3. Forward through DeepONet
4. Inverse-transform output using stored output normalizer as torch tensors
5. Return CD and PC tensors with autograd graph intact

**Key design detail -- `jacobian`:**
```python
def jacobian(self, k0_1, k0_2, alpha_1, alpha_2) -> Dict[str, np.ndarray]:
    """Compute d(CD)/d(params) and d(PC)/d(params) via autograd.

    Returns
    -------
    dict with:
        'dcd_dparams' : np.ndarray (n_eta, 4)  -- d(CD_i)/d(param_j)
        'dpc_dparams' : np.ndarray (n_eta, 4)  -- d(PC_i)/d(param_j)
    """
```

Uses `torch.autograd.functional.jacobian` or manual backward passes.

### Step 2: Training infrastructure

**Add to `Surrogate/deeponet_model.py` or create `Surrogate/deeponet_training.py`:**

Decision: put training in the `fit()` method (following `nn_model.py` pattern) with a separate `train_deeponet_surrogate()` function in a new `Surrogate/deeponet_training.py` for advanced logging, mirroring the `nn_model.py` / `nn_training.py` split.

#### Training strategy

1. **Data reshaping:** The training data is (3194, 4) parameters and (3194, 22) outputs. For DeepONet, we reshape to pointwise: each sample becomes 22 training pairs `(params, eta_j) -> (CD_j, PC_j)`. Total: 3194 x 22 = 70,268 pointwise samples. This is sufficient for DeepONet.

2. **Loss function:** MSE on (CD, PC) jointly, with optional per-channel weighting if scales differ significantly.

3. **Optimizer:** AdamW, lr=1e-3, weight_decay=1e-4 (same as nn_model.py).

4. **Scheduler:** CosineAnnealingWarmRestarts, T_0=500, T_mult=2, eta_min=1e-6.

5. **Early stopping:** Patience 500 on validation loss (same as nn_model.py).

6. **Batch construction:** Sample B parameter indices, use ALL 22 voltages for each. Effective batch = B x 22. This preserves the full curve structure per sample while shuffling across parameters. B=256 -> effective 5,632 pointwise pairs per batch.

7. **Normalization:**
   - Input params: log10(k0) + z-score (identical to nn_model.py)
   - Input voltage: z-score on phi_applied
   - Output CD/PC: z-score per channel (mean/std of flattened training CD, flattened training PC)

8. **Epochs:** Max 5000. With 3194 samples and full-batch over voltages, each epoch is fast (~2714 train samples / 256 batch = ~11 steps). Expect convergence in 1000-3000 epochs. Total time estimate: < 30 minutes on MPS.

#### Training data flow

```
Load training_data_merged.npz
  parameters: (3194, 4)
  current_density: (3194, 22)
  peroxide_current: (3194, 22)
  phi_applied: (22,)

Split 85/15 train/val (same logic as nn_model.py)

Compute normalizers:
  input_normalizer: ZScoreNormalizer on log-transformed params
  eta_normalizer: ZScoreNormalizer on phi_applied
  cd_normalizer: ZScoreNormalizer on flattened CD
  pc_normalizer: ZScoreNormalizer on flattened PC

Precompute:
  eta_norm = eta_normalizer.transform(phi_applied)  # (22,) -- fixed tensor

Per batch:
  Sample B parameter indices
  params_norm = input_normalizer.transform(log_transform(params[idx]))  # (B, 4)
  targets_cd = cd_normalizer.transform(current_density[idx])  # (B, 22)
  targets_pc = pc_normalizer.transform(peroxide_current[idx])  # (B, 22)

  pred = deeponet(params_norm, eta_norm)  # (B, 2, 22)
  loss = MSE(pred[:, 0, :], targets_cd) + MSE(pred[:, 1, :], targets_pc)
```

### Step 3: Autograd gradient verification

**Add a test/verification script:** `scripts/verify_deeponet_gradients.py`

1. Load trained DeepONet model
2. For 10 random parameter points:
   a. Compute autograd Jacobian via `model.jacobian()`
   b. Compute finite-difference Jacobian with h=1e-5 (central differences)
   c. Compare element-wise: `max |autograd - FD| / (|FD| + eps)`
   d. Assert relative error < 1% for all entries
3. Also verify that `predict_torch` with `requires_grad=True` produces correct gradients through a scalar loss (e.g., `loss = pred_cd.sum()`, then `loss.backward()`)

### Step 4: Validation and comparison to NN ensemble

**Script:** `scripts/train_deeponet.py`

1. Load `training_data_merged.npz` and `split_indices.npz`
2. Train DeepONet with default config
3. Run `validate_surrogate()` on held-out test set
4. Compare metrics (cd_rmse, pc_rmse, cd_mean_relative_error) to NN ensemble baseline
5. Save trained model to `data/surrogate_models/deeponet/`
6. Run gradient verification
7. Print comparison table

### Step 5: Register in `Surrogate/__init__.py`

Add exports:
```python
from Surrogate.deeponet_model import DeepONetSurrogateModel
```

Add to `__all__`.

### Step 6: (Optional) Physics-informed loss term

If data-only training doesn't meet the error target, add a physics residual loss. Since the PDE is complex (Nernst-Planck-Poisson), a simpler physics constraint is more practical:

- **Monotonicity constraint** on CD (should be monotonically decreasing with eta) -- already implemented in `nn_training.py`, reuse the pattern
- **Smoothness penalty** via second-order finite differences -- also already in `nn_training.py`
- **Boundary consistency** at extreme voltages

This is a fallback, not a primary strategy. The 3194 x 22 = 70K pointwise samples should be sufficient for a ~59K parameter model.

## File Inventory

| File | Action | Purpose |
|------|--------|---------|
| `Surrogate/deeponet_model.py` | CREATE | BranchNet, TrunkNet, DeepONet, DeepONetSurrogateModel |
| `Surrogate/deeponet_training.py` | CREATE | train_deeponet_surrogate() with logging, checkpointing |
| `Surrogate/__init__.py` | EDIT | Add DeepONetSurrogateModel export |
| `scripts/train_deeponet.py` | CREATE | Training + validation script |
| `scripts/verify_deeponet_gradients.py` | CREATE | Autograd vs FD gradient check |
| `data/surrogate_models/deeponet/` | CREATE (dir) | Trained model artifacts |

## Success Criteria Checklist

- [ ] `DeepONetSurrogateModel` implements full shared API: `predict()`, `predict_batch()`, `n_eta`, `phi_applied`, `is_fitted`, `training_bounds`, `save()`, `load()`
- [ ] `predict_torch()` returns differentiable tensors (autograd graph intact)
- [ ] `jacobian()` method returns d(output)/d(params) via autograd
- [ ] Autograd gradients match central FD (h=1e-5) to within 1% relative error across 10+ test points
- [ ] Prediction error (cd_rmse, pc_rmse, mean_nrmse) is <= NN ensemble baseline
- [ ] Training completes in < 2 hours on MacBook Air M4 (MPS) -- target < 30 minutes
- [ ] Model artifacts saved to `data/surrogate_models/deeponet/`
- [ ] `validate_surrogate()` from `Surrogate/validation.py` runs cleanly on the DeepONet model
- [ ] Model is importable via `from Surrogate.deeponet_model import DeepONetSurrogateModel`

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| DeepONet underfits with 3194 samples | Medium | 70K pointwise pairs is actually generous for ~59K params. If needed: increase latent_dim, add more branch/trunk layers, or use physics regularization. |
| Autograd through log10 transform is numerically unstable near k0 ~ 0 | Low | k0 bounds are [1e-6, 1.0], so log10(k0) is in [-6, 0]. Clamp input to >= 1e-30 (already done in nn_model.py). |
| MPS backend issues with custom ops | Low | Fall back to CPU. Training is lightweight enough to run on CPU in < 2 hours. |
| Output scale mismatch between CD and PC channels | Medium | Use per-channel z-score normalization and optionally per-channel loss weights. Monitor both channels independently during training. |

## Execution Order

1. **Implement `Surrogate/deeponet_model.py`** -- network classes + API wrapper with predict_torch and jacobian
2. **Implement `Surrogate/deeponet_training.py`** -- training function with logging
3. **Write `scripts/train_deeponet.py`** -- end-to-end training script
4. **Train and validate** -- run training, check metrics
5. **Write `scripts/verify_deeponet_gradients.py`** -- gradient verification
6. **Run gradient verification** -- confirm < 1% error
7. **Update `Surrogate/__init__.py`** -- register exports
8. **Compare to NN ensemble** -- confirm error parity

## Hyperparameter Summary

| Parameter | Default | Notes |
|-----------|---------|-------|
| latent_dim (p) | 64 | Number of basis functions |
| branch_hidden | 128 | Branch MLP width |
| branch_n_hidden | 2 | Branch hidden layers |
| trunk_hidden | 64 | Trunk MLP width |
| trunk_n_hidden | 2 | Trunk hidden layers |
| n_channels | 2 | CD + PC (unstacked) |
| epochs | 5000 | Max training epochs |
| lr | 1e-3 | AdamW learning rate |
| weight_decay | 1e-4 | AdamW regularization |
| patience | 500 | Early stopping patience |
| batch_size | 256 | Parameter-level batch size |
| T_0 | 500 | Cosine annealing period |
| activation | SiLU | Throughout both nets |
| normalization | LayerNorm | After each hidden layer |
