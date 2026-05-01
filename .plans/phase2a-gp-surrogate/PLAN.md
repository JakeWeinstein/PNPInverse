# Phase 2a: GP Surrogate (GPyTorch) -- Implementation Plan

**Date:** 2026-03-17
**Goal:** Build a Gaussian Process surrogate using GPyTorch with predictions, calibrated UQ, and analytic autograd gradients, implementing the existing surrogate API.

**Output file:** `Surrogate/gp_model.py`
**Artifacts directory:** `data/surrogate_models/gp/`

---

## Architecture Decisions (Resolved)

### 1. Multi-output strategy: Independent GPs per output dimension
- 44 independent GPs (22 CD + 22 PC), one per voltage point
- Rationale: Multi-task GP (ICM/LMC) with 44 tasks is intractable. Independent GPs are the standard approach at this output count. Correlation across voltage points is nice-to-have but not needed for the API.
- Each GP: 4D input (log10(k0_1), log10(k0_2), alpha_1, alpha_2) -> 1D output

### 2. GP type: Exact GP (not SVGP)
- With N=3,194 and 4D input, exact GP is feasible. The O(N^3) cost is for the Cholesky of a 3194x3194 matrix, which is ~seconds on CPU with GPyTorch's CG-based inference.
- GPyTorch's default CG solver + preconditioning handles this scale well.
- Fall back to SVGP only if exact GP training exceeds 30 minutes or prediction latency exceeds 1 second per batch.

### 3. Kernel: Matern 5/2 with ARD
- Matern 5/2: twice-differentiable (smooth enough for autograd gradients), but less smooth than RBF (avoids over-smoothing). Standard choice for physical surrogates.
- ARD (Automatic Relevance Determination): one length scale per input dimension. Reveals parameter sensitivity (short length scale = high sensitivity). With only 4 inputs, ARD adds negligible cost.
- Full kernel: `ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=4))`

### 4. Output normalization: Z-score (same as NN surrogate)
- Z-score normalize each output dimension independently before fitting.
- Store normalizer stats; inverse-transform predictions before returning.
- Consistent with existing `ZScoreNormalizer` in `nn_model.py`.

### 5. Input normalization: Z-score in log-space (same as NN surrogate)
- Transform k0_1, k0_2 to log10 space, then Z-score all 4 inputs.
- Reuse the `ZScoreNormalizer` class from `nn_model.py`.

---

## Step-by-Step Plan

### Step 0: Environment Setup
**Task:** Install GPyTorch into the venv-firedrake environment.
**Commands:**
```bash
source ../venv-firedrake/bin/activate
pip install gpytorch
```
**Verification:** `python -c "import gpytorch; print(gpytorch.__version__)"` succeeds.
**Expected output:** GPyTorch version string (e.g., "1.12").
**Risk:** GPyTorch may have PyTorch version conflicts with the existing torch install. If so, install a compatible version: `pip install gpytorch==<version>`.

---

### Step 1: Define the GPyTorch Model Classes
**Task:** Create `Surrogate/gp_model.py` with the GPyTorch model definition and the `GPSurrogateModel` wrapper class.

**GPyTorch model (internal):**
```python
class ExactGPModel(gpytorch.models.ExactGP):
    """Single-output exact GP with Matern 5/2 ARD kernel."""
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=4)
        )
    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)
```

**Wrapper class API (must match existing surrogates):**
```
class GPSurrogateModel:
    fit(parameters, current_density, peroxide_current, phi_applied, ...) -> self
    predict(k0_1, k0_2, alpha_1, alpha_2) -> dict
    predict_batch(parameters) -> dict
    predict_with_uncertainty(k0_1, k0_2, alpha_1, alpha_2) -> dict
    predict_batch_with_uncertainty(parameters) -> dict
    save(path) -> None
    load(path, device) -> GPSurrogateModel  (classmethod)
    n_eta -> int  (property)
    phi_applied -> np.ndarray  (property)
    is_fitted -> bool  (property)
    training_bounds -> dict  (property)
```

**Key implementation details:**
- Store 44 independent `(ExactGPModel, GaussianLikelihood)` pairs after fitting
- `_transform_inputs()` method: log10(k0) + Z-score, identical to `NNSurrogateModel._transform_inputs()`
- Import and reuse `ZScoreNormalizer` from `Surrogate.nn_model`
- Lazy import of gpytorch (same pattern as torch in nn_model.py: try/except with `_GPYTORCH_AVAILABLE` flag)

**Success criteria:** File parses, class instantiates, methods exist with correct signatures.

---

### Step 2: Implement `fit()`
**Task:** Train 44 independent exact GPs on Z-score-normalized data.

**Algorithm:**
1. Validate input shapes: `parameters` (N, 4), `current_density` (N, 22), `peroxide_current` (N, 22)
2. Transform inputs: log10(k0) + compute Z-score normalizer from training data
3. Concatenate outputs: Y = [CD | PC] shape (N, 44)
4. Compute per-output Z-score normalizers (one per output dim)
5. For each output dimension d in 0..43:
   a. Create `GaussianLikelihood()` with learned noise
   b. Create `ExactGPModel(train_x, train_y_d, likelihood)`
   c. Set to training mode
   d. Optimize marginal log-likelihood using Adam (GPyTorch standard):
      - `mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)`
      - Adam optimizer, lr=0.1, ~200 iterations (GPyTorch default recommendation)
      - Monitor loss convergence; stop early if change < 1e-6 for 20 iterations
   e. Set to eval mode
   f. Store `(model, likelihood)` pair
6. Store phi_applied, training_bounds, normalizers, fitted flag

**Hyperparameter choices:**
- Training iterations: 200 per GP (conservative; exact GP hyperparameter optimization is fast)
- Learning rate: 0.1 (GPyTorch recommended default)
- Noise: learned (GaussianLikelihood default); no noise constraint needed since we trust the PDE training data is nearly noise-free, but a small learned noise acts as jitter for numerical stability

**Output normalization decision:**
- Use a single `ZScoreNormalizer` for the concatenated 44D output (same as NN model), OR
- Use 44 individual normalizers (one per GP)
- Decision: Use a single output normalizer of shape (44,) for consistency with the NN model. Each GP d trains on `Y_norm[:, d]`.

**Parallelization:** The 44 GP fits are independent. Use `joblib.Parallel` with `n_jobs=-1` (all cores) to fit all 44 GPs in parallel. Each GP fit is CPU-bound (Cholesky + Adam), so this should scale near-linearly with core count. On an 10-core M4, expect ~4-5x speedup (from ~15 min to ~3 min). Wrap with a `tqdm` progress bar. Implementation:
```python
from joblib import Parallel, delayed

def _fit_single_gp(train_x, train_y_d, n_iters=200, lr=0.1):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y_d, likelihood)
    model.train(); likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for i in range(n_iters):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y_d)
        loss.backward()
        optimizer.step()
    model.eval(); likelihood.eval()
    return model, likelihood

results = Parallel(n_jobs=-1)(
    delayed(_fit_single_gp)(train_x, train_y_norm[:, d])
    for d in range(44)
)
self._gp_models = results
```

**Success criteria:**
- `fit()` completes in < 5 minutes on CPU with joblib parallelism (10-core M4)
- No NaN/Inf in learned hyperparameters
- Marginal log-likelihood increases during training for all 44 GPs

---

### Step 3: Implement `predict()` and `predict_batch()`
**Task:** Implement prediction methods matching the existing API contract.

**Algorithm for `predict_batch(parameters)`:**
1. Transform inputs: log10(k0) + Z-score normalize using stored normalizer
2. Convert to torch tensor (float32)
3. For each GP d in 0..43:
   a. `model.eval()`, `likelihood.eval()`
   b. `with torch.no_grad(), gpytorch.settings.fast_pred_var():`
   c. `posterior = likelihood(model(X_tensor))`
   d. `mean_d = posterior.mean.numpy()`
4. Stack means -> shape (M, 44)
5. Inverse Z-score transform
6. Split into CD (first 22) and PC (last 22)
7. Return `{'current_density': cd, 'peroxide_current': pc, 'phi_applied': phi}`

**`predict()` delegates to `predict_batch()` with a single-row input (same pattern as NN model).**

**Performance note:** `fast_pred_var()` enables LOVE (Lanczos-based fast predictive variance) which also speeds up mean predictions. Essential for batch prediction.

**Success criteria:**
- `predict_batch()` returns correct shapes
- Predictions are finite (no NaN/Inf)
- Single prediction takes < 100ms

---

### Step 4: Implement `predict_with_uncertainty()` and `predict_batch_with_uncertainty()`
**Task:** Return mean + standard deviation from the GP posterior.

**Algorithm for `predict_batch_with_uncertainty(parameters)`:**
1. Same as predict_batch steps 1-3, but also extract variance:
   a. `variance_d = posterior.variance.numpy()`
2. Stack means and variances -> (M, 44) each
3. Inverse-transform mean (same as predict)
4. Transform variance back to original scale: `std_original = sqrt(var_norm) * output_std[d]`
   (Since Z-score is linear: if y_norm = (y - mu)/sigma, then var(y) = sigma^2 * var(y_norm))
5. Split into CD/PC components
6. Return dict with keys: `current_density`, `peroxide_current`, `phi_applied`, `current_density_std`, `peroxide_current_std`

**API matches `EnsembleMeanWrapper.predict_with_uncertainty()` exactly.**

**Success criteria:**
- Uncertainty values are positive everywhere
- Uncertainty is larger far from training data, smaller near training data (basic sanity check)

---

### Step 5: Implement Autograd Gradient Support
**Task:** Enable PyTorch autograd to compute gradients of the GP mean prediction w.r.t. input parameters. This is the key differentiator over FD-based gradients.

**Approach:**
Add a `predict_with_grad(x_tensor)` method that:
1. Takes a torch tensor with `requires_grad=True`
2. Passes through the GP forward model (mean prediction only, no sampling)
3. Returns the mean prediction as a differentiable tensor

The gradient is then computed by the caller via `torch.autograd.grad()` or `.backward()`.

**Implementation:**
```python
def predict_torch(self, x: torch.Tensor) -> torch.Tensor:
    """Differentiable prediction: input (M,4) tensor -> output (M,44) tensor.

    Input must be in TRANSFORMED space (log10 k0 + Z-score normalized).
    Output is in Z-score normalized space.
    Caller handles normalization and inverse transforms.
    """
    means = []
    for model, likelihood in self._gp_models:
        model.eval()
        likelihood.eval()
        posterior = model(x)  # NOT likelihood(model(x)) -- we want the latent GP mean, not the noisy prediction
        means.append(posterior.mean)  # shape (M,)
    return torch.stack(means, dim=1)  # shape (M, 44)
```

**Key detail:** Use `model(x)` not `likelihood(model(x))` for the differentiable path. The likelihood adds observation noise variance which we don't want in the gradient of the mean.

**Higher-level gradient helper:**
```python
def gradient_at(self, k0_1, k0_2, alpha_1, alpha_2, target_cd, target_pc, secondary_weight=1.0):
    """Compute dJ/dx via autograd where J is the surrogate objective.

    Returns gradient w.r.t. [log10(k0_1), log10(k0_2), alpha_1, alpha_2].
    """
    # Build differentiable input tensor
    x = torch.tensor([[log10(k0_1), log10(k0_2), alpha_1, alpha_2]],
                      dtype=torch.float32, requires_grad=True)
    x_norm = (x - self._input_mean_t) / self._input_std_t  # torch ops
    y_norm = self.predict_torch(x_norm)  # (1, 44)
    y = y_norm * self._output_std_t + self._output_mean_t  # inverse Z-score
    cd = y[0, :22]
    pc = y[0, 22:]
    J = 0.5 * torch.sum((cd - target_cd_t)**2) + secondary_weight * 0.5 * torch.sum((pc - target_pc_t)**2)
    J.backward()
    return x.grad.numpy()[0]  # shape (4,)
```

**Normalization in torch:** Store `_input_mean_t`, `_input_std_t`, `_output_mean_t`, `_output_std_t` as `torch.tensor` buffers during `fit()` so the entire forward path stays in the autograd graph.

**Success criteria:**
- Autograd gradient matches central FD gradient to < 1% relative error on 10 random test points
- Gradient computation takes < 200ms per point (vs ~800ms for 8 FD evals on NN ensemble)

---

### Step 6: Implement `save()` and `load()`
**Task:** Serialize/deserialize 44 GP models + metadata.

**Save format (directory structure):**
```
data/surrogate_models/gp/
    metadata.npz          # phi_applied, n_eta, training_bounds
    normalizers.npz       # input_mean, input_std, output_mean, output_std
    gp_model_00.pt        # torch state dict for GP 0
    gp_model_01.pt        # ...
    ...
    gp_model_43.pt
    likelihood_00.pt      # torch state dict for likelihood 0
    ...
    likelihood_43.pt
    train_x.pt            # training inputs (needed for exact GP prediction)
    train_y.npz           # training targets (44 dims)
```

**Critical detail for exact GP:** Unlike NNs, exact GPs need the training data at prediction time (the posterior conditions on it). Must save `train_x` and `train_y` alongside model hyperparameters.

**Load procedure:**
1. Load metadata and normalizers
2. Load train_x and train_y
3. For each d in 0..43:
   a. Create `GaussianLikelihood()`, load state dict
   b. Create `ExactGPModel(train_x, train_y[:, d], likelihood)`, load state dict
   c. Set to eval mode
4. Set `_is_fitted = True`

**Success criteria:**
- Round-trip: `save()` then `load()` produces identical predictions (max abs diff < 1e-10)

---

### Step 7: Integrate with `Surrogate/__init__.py`
**Task:** Export `GPSurrogateModel` and a convenience `load_gp_surrogate()` function.

**Changes to `Surrogate/__init__.py`:**
```python
from Surrogate.gp_model import GPSurrogateModel, load_gp_surrogate
```

Add to `__all__`.

**Success criteria:** `from Surrogate import GPSurrogateModel` works.

---

### Step 8: Training Script
**Task:** Create `scripts/Surrogate/train_gp.py` to train the GP surrogate from the merged training data and save artifacts.

**Script flow:**
1. Load `data/surrogate_models/training_data_merged.npz`
2. Split: 85% train, 15% test (same split ratio and seed as NN for fair comparison)
3. Create `GPSurrogateModel()` and call `fit()`
4. Save to `data/surrogate_models/gp/`
5. Run validation on test set using `Surrogate.validation.validate_surrogate()`
6. Print report using `print_validation_report()`
7. Run UQ calibration check (see Step 9)

**Success criteria:** Script runs end-to-end, produces saved model artifacts, prints validation metrics.

---

### Step 9: Validation and UQ Calibration
**Task:** Validate prediction accuracy and uncertainty calibration on held-out test data.

**Prediction accuracy (reuse existing infrastructure):**
- Call `validate_surrogate(gp_model, test_params, test_cd, test_pc)`
- Compare RMSE, max abs error, mean NRMSE against NN ensemble baselines

**UQ calibration check (new):**
For each test sample, check whether the true value falls within the GP predictive interval:
```python
def check_uq_calibration(model, test_params, test_cd, test_pc, confidence_levels=[0.5, 0.8, 0.9, 0.95]):
    """Check what fraction of test points fall within each predictive interval."""
    result = model.predict_batch_with_uncertainty(test_params)
    mean_cd, std_cd = result['current_density'], result['current_density_std']

    for level in confidence_levels:
        z = scipy.stats.norm.ppf((1 + level) / 2)
        lower = mean_cd - z * std_cd
        upper = mean_cd + z * std_cd
        coverage = np.mean((test_cd >= lower) & (test_cd <= upper))
        print(f"  {level*100:.0f}% interval: actual coverage = {coverage*100:.1f}%")
    # Ideal: 90% interval has ~90% coverage. If >95%, uncertainty is too wide (conservative).
    # If <85%, uncertainty is too narrow (overconfident).
```

**Gradient accuracy check:**
For 10 random test points, compare autograd gradient vs central FD gradient (h=1e-5):
```python
for each test point x:
    grad_autograd = model.gradient_at(x, target_cd, target_pc)
    grad_fd = central_fd(objective_fn, x, h=1e-5)
    relative_error = np.linalg.norm(grad_autograd - grad_fd) / np.linalg.norm(grad_fd)
    assert relative_error < 0.01  # < 1% relative error
```

**Success criteria (the three acceptance gates):**
1. **Accuracy:** Mean NRMSE within 2x of NN ensemble (~10.7% worst case target). Ideal: competitive or better.
2. **UQ calibration:** 90% predictive interval achieves 85-95% actual coverage (calibrated, not wildly over/under-confident)
3. **Gradient:** Autograd matches FD to < 1% relative error on all 10 test points

---

### Step 10: Fallback -- SVGP if Exact GP is Too Slow
**Task:** If Step 2 reveals that exact GP training exceeds 30 minutes or prediction latency exceeds 1 second per batch, implement an SVGP variant.

**SVGP changes:**
- Replace `ExactGP` with `gpytorch.models.ApproximateGP` using `VariationalStrategy`
- Use `CholeskyVariationalDistribution` with M=500 inducing points
- Train via ELBO: `gpytorch.mlls.VariationalELBO`
- Inducing points initialized via k-means on training inputs
- Training: Adam, lr=0.01, 500 iterations with mini-batches of 256

**This step is contingent -- only execute if exact GP fails the performance gate.**

---

## File Inventory

| File | Action | Description |
|------|--------|-------------|
| `Surrogate/gp_model.py` | CREATE | GP surrogate model with full API |
| `Surrogate/__init__.py` | EDIT | Add GPSurrogateModel export |
| `scripts/Surrogate/train_gp.py` | CREATE | Training + validation script |
| `data/surrogate_models/gp/` | CREATE (dir) | Trained model artifacts |

---

## Estimated Timeline

| Step | Time | Cumulative |
|------|------|-----------|
| 0. Environment setup | 5 min | 5 min |
| 1. Model class skeleton | 30 min | 35 min |
| 2. fit() implementation | 45 min | 1h 20m |
| 3. predict()/predict_batch() | 20 min | 1h 40m |
| 4. predict_with_uncertainty() | 15 min | 1h 55m |
| 5. Autograd gradient support | 45 min | 2h 40m |
| 6. save()/load() | 30 min | 3h 10m |
| 7. __init__.py integration | 5 min | 3h 15m |
| 8. Training script | 20 min | 3h 35m |
| 9. Validation + UQ calibration | 30 min | 4h 05m |
| 10. SVGP fallback (if needed) | 1h | 5h 05m |

**Total: ~3.5-4.5 hours** (including actual GP training time of ~3-5 min with parallel fitting)

---

## Success Criteria Summary

| Criterion | Target | How Measured |
|-----------|--------|-------------|
| Prediction accuracy | Mean NRMSE < 15% (competitive with NN ensemble ~10.7%) | `validate_surrogate()` on held-out test set |
| Worst-case error | < 20% worst-case NRMSE | Max per-sample NRMSE on test set |
| UQ calibration | 90% interval has 85-95% actual coverage | Custom calibration check on test set |
| Gradient correctness | Autograd vs FD relative error < 1% | 10-point gradient comparison |
| No FD fallback | Gradients computed purely via autograd | Code inspection |
| API compatibility | All standard API methods work | predict(), predict_batch(), properties match NN model |
| Training time | < 5 min on CPU (parallel via joblib) | Wall-clock measurement |
| Prediction latency | < 1 sec for batch of 100 | Wall-clock measurement |
| Save/load round-trip | Predictions match to < 1e-10 | Numerical comparison |

---

## Dependencies

- **Phase 1 (Training Data Audit):** Uses same training data. If Phase 1 recommends data augmentation, re-train the GP on the augmented dataset. The GP code itself does not depend on Phase 1 results.
- **Phase 2e (Autograd Retrofit):** The GP model's autograd gradient approach can inform the NN ensemble retrofit. No blocking dependency.
- **Phase 3 (Benchmark):** GP model is one input to the comparative benchmark. Must complete before Phase 3.
