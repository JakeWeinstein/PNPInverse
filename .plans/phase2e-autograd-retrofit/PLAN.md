# Phase 2e: PyTorch Autograd Gradient Retrofit

## Goal

Replace the 8-eval (4D) or 4-eval (2D) central finite-difference gradient computation in `Surrogate/objectives.py` with PyTorch autograd. The NN ensemble already uses PyTorch internally; this phase threads `requires_grad` through the forward pass so `torch.autograd.grad` computes exact gradients in a single forward + backward pass.

## Success Criteria

| Criterion | Threshold |
|---|---|
| Autograd gradient vs FD gradient relative error | < 0.1% at 10 random test points |
| Gradient eval speedup (wall-clock) | >= 4x faster than 8-eval central FD |
| Optimization results unchanged | Same recovered parameters to 3 significant figures on the v13 test cases |
| All existing tests pass | Zero regressions |

## Key Design Decision

**Modify existing classes in-place** (not a parallel class hierarchy). Each objective class gets an internal `_use_autograd` flag that defaults to `True` when the surrogate is a torch-backed model (`NNSurrogateModel` or `EnsembleMeanWrapper`), and falls back to FD for `BVSurrogateModel` (RBF). This keeps the call sites in `cascade.py`, `multistart.py`, and pipeline scripts unchanged.

## Architecture Overview

The autograd path requires the forward pass to produce a **differentiable torch tensor** from the optimizer's numpy x-vector. The chain is:

```
x (numpy, from scipy L-BFGS-B)
  -> torch.tensor(x, requires_grad=True)
  -> z-score normalize (torch ops, using stored mean/std tensors)
  -> model.forward(z)  (torch, gradient tracked)
  -> inverse z-score (torch ops)
  -> split into cd_sim, pc_sim tensors
  -> compute J = 0.5*||cd_sim - target||^2 + w*0.5*||pc_sim - target||^2  (torch)
  -> J.backward()
  -> x_tensor.grad.numpy()  -> return to scipy
```

Critical: the `10**(x[0:2])` log-space transform and z-score normalization must use torch ops so gradients flow through them.

---

## Implementation Steps

### Step 1: Add `predict_torch()` to `NNSurrogateModel`

**File:** `Surrogate/nn_model.py`

Add a new method that returns a differentiable torch tensor instead of numpy:

```python
def predict_torch(self, x_logspace: torch.Tensor) -> torch.Tensor:
    """Forward pass returning raw torch tensor (44,) with grad support.

    Parameters
    ----------
    x_logspace : torch.Tensor of shape (4,), requires_grad=True
        [log10(k0_1), log10(k0_2), alpha_1, alpha_2] in log-space.
        Note: k0 values are ALREADY in log10 space (matching optimizer space).

    Returns
    -------
    torch.Tensor of shape (44,)
        Concatenated [current_density(22), peroxide_current(22)] in physical units.
    """
```

Implementation details:
- Convert stored `_input_normalizer.mean` and `_input_normalizer.std` to `torch.tensor` (cache as `_input_mean_t`, `_input_std_t` on first call)
- Z-score normalize: `z = (x_logspace - mean_t) / std_t`
- Forward: `y_norm = self._model(z.unsqueeze(0)).squeeze(0)`
- Inverse z-score: `y = y_norm * out_std_t + out_mean_t`
- Return `y` (shape 44, gradient-tracked)

No `torch.no_grad()` context. No `.detach()`. No `.numpy()`.

**Estimated LOC:** ~30 lines

### Step 2: Add `predict_torch()` to `EnsembleMeanWrapper`

**File:** `Surrogate/ensemble.py`

```python
def predict_torch(self, x_logspace: torch.Tensor) -> torch.Tensor:
    """Ensemble mean prediction as a differentiable torch tensor.

    Averages the output of all members. Gradient is the mean of per-member gradients
    (equivalent to gradient of the mean, by linearity).
    """
```

Implementation:
- Stack predictions from each member's `predict_torch()`: `preds = torch.stack([m.predict_torch(x_logspace) for m in self.models])`
- Return `preds.mean(dim=0)` (shape 44)
- Guard: `if not hasattr(self.models[0], 'predict_torch')`, raise informative error

**Estimated LOC:** ~15 lines

### Step 3: Add `_has_autograd()` detection helper

**File:** `Surrogate/objectives.py`

Add a module-level helper:

```python
def _has_autograd(surrogate) -> bool:
    """Return True if the surrogate supports predict_torch()."""
    return hasattr(surrogate, 'predict_torch') and callable(surrogate.predict_torch)
```

**Estimated LOC:** ~3 lines

### Step 4: Add `_autograd_objective_and_gradient()` to `SurrogateObjective`

**File:** `Surrogate/objectives.py`

This is the core method. Add to `SurrogateObjective`:

```python
def _autograd_objective_and_gradient(
    self, x: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """Compute J and dJ/dx via PyTorch autograd (single forward + backward)."""
    import torch

    x_t = torch.tensor(x, dtype=torch.float64, requires_grad=True)

    # Forward through surrogate (expects log-space k0, which x already has)
    y = self.surrogate.predict_torch(x_t)  # shape (44,)
    n_eta = len(y) // 2
    cd_sim = y[:n_eta]
    pc_sim = y[n_eta:]

    # Targets as torch tensors (cache on first call)
    if not hasattr(self, '_target_cd_t'):
        self._target_cd_t = torch.tensor(
            self.target_cd[self._valid_cd], dtype=torch.float64)
        self._target_pc_t = torch.tensor(
            self.target_pc[self._valid_pc], dtype=torch.float64)
        self._valid_cd_idx = torch.tensor(
            np.where(self._valid_cd)[0], dtype=torch.long)
        self._valid_pc_idx = torch.tensor(
            np.where(self._valid_pc)[0], dtype=torch.long)

    cd_diff = cd_sim[self._valid_cd_idx] - self._target_cd_t
    pc_diff = pc_sim[self._valid_pc_idx] - self._target_pc_t

    J = 0.5 * torch.sum(cd_diff ** 2) + self.secondary_weight * 0.5 * torch.sum(pc_diff ** 2)

    J.backward()

    J_val = float(J.detach())
    grad_val = x_t.grad.numpy().copy()

    self._n_evals += 1
    return J_val, grad_val
```

Key details:
- Use `float64` throughout for consistency with scipy L-BFGS-B (which uses float64)
- The `predict_torch` input is `x_t` directly (already in log-space), NOT physical k0
- Cache torch target tensors to avoid re-creating each call

**Estimated LOC:** ~35 lines

### Step 5: Rewire `objective()`, `gradient()`, `objective_and_gradient()`

**File:** `Surrogate/objectives.py`, class `SurrogateObjective`

Modify `__init__` to detect autograd support:

```python
self._use_autograd = _has_autograd(surrogate)
```

Modify `objective_and_gradient()`:

```python
def objective_and_gradient(self, x):
    if self._use_autograd:
        return self._autograd_objective_and_gradient(x)
    J = self.objective(x)
    g = self.gradient(x)
    return J, g
```

Modify `objective()` when called standalone (e.g., for grid eval):
- Keep the existing numpy path unchanged (grid eval does not need gradients)
- When `_use_autograd` is True and `objective()` is called alone, use the existing numpy path (no reason to use autograd for objective-only)

Modify `gradient()` when called standalone:
- When `_use_autograd` is True: compute via `_autograd_objective_and_gradient` and return only the gradient
- This is slightly wasteful (computes objective too) but `objective_and_gradient()` is the primary call site from L-BFGS-B

**Estimated LOC:** ~15 lines of changes

### Step 6: Repeat for other objective classes

Apply the same pattern to:

1. **`ReactionBlockSurrogateObjective`** (2D: `[log10(k0_r), alpha_r]`)
   - `_autograd_objective_and_gradient()` constructs full 4D input from 2D x-vector using torch ops
   - The fixed parameters are detached constants (no gradient)

2. **`AlphaOnlySurrogateObjective`** (2D: `[alpha_1, alpha_2]`)
   - Constructs 4D input: `[log10(fixed_k0_1), log10(fixed_k0_2), x[0], x[1]]`
   - k0 values are constants (detached)

3. **`SubsetSurrogateObjective`** (4D with voltage subset)
   - Same as `SurrogateObjective` but indexes into the 44-element output at `subset_idx`
   - Use `y[subset_idx]` and `y[n_eta + subset_idx]` for torch indexing (supports grad)

**Estimated LOC:** ~80 lines total across 3 classes

### Step 7: Retrofit `cascade.py` and `multistart.py` inline FD

**File:** `Surrogate/cascade.py`

The `_make_subset_objective_fn` and `_make_subset_block_objective_fn` functions build inline closures with FD gradients. These bypass the objective classes entirely.

**Approach:** Replace the inline FD with autograd when the surrogate has `predict_torch()`:

```python
def _make_subset_objective_fn(surrogate, target_cd, target_pc, secondary_weight, subset_idx):
    # ... existing setup ...

    if _has_autograd(surrogate):
        def _objective_and_grad(x):
            # torch path (similar to Step 4 but with subset_idx)
            ...
            return float(J), grad.numpy()

        def _objective(x):
            val, _ = _objective_and_grad(x)
            return val

        def _gradient(x, fd_step=None):
            _, g = _objective_and_grad(x)
            return g

        return _objective, _gradient, counter

    # ... existing FD path (unchanged) ...
```

Same pattern for `_make_subset_block_objective_fn`.

**File:** `Surrogate/multistart.py`

The `_polish_candidate` function has its own inline FD. Same retrofit:
- Detect `_has_autograd(surrogate)` and branch to torch path

**Estimated LOC:** ~60 lines across both files

### Step 8: float64 precision for `predict_torch()`

**File:** `Surrogate/nn_model.py`

The existing `ResNetMLP` uses float32 weights. For autograd gradient accuracy matching FD (which uses float64 numpy), we need float64 computation in the gradient path.

**Approach:** In `predict_torch()`, cast model to float64 on first autograd call and cache:

```python
if not hasattr(self, '_model_f64'):
    self._model_f64 = copy.deepcopy(self._model).double()
```

Then use `self._model_f64(z.unsqueeze(0))` in the autograd path. This doubles memory for the model (~1MB total, negligible) but ensures gradient accuracy.

Alternative: keep float32 model and cast input to float32, accept ~1e-7 precision. Since FD with `h=1e-5` has truncation error ~1e-10, float32 autograd (~1e-7 precision) is actually fine. **Decision: use float32 model, cast x to float32 for forward pass, cast gradient back to float64 for scipy.** This avoids the deepcopy overhead.

Revised `predict_torch()`:
```python
# x_logspace arrives as float64 (from requires_grad tensor)
# Cast to float32 for model, cast output back to float64
x_f32 = x_logspace.float()  # gradient still flows through .float()
z = (x_f32 - self._input_mean_t) / self._input_std_t  # float32 tensors
y_norm = self._model(z.unsqueeze(0)).squeeze(0)
y = y_norm * self._out_std_t + self._out_mean_t
return y.double()  # back to float64 for loss computation
```

**Estimated LOC:** ~5 lines adjustment to Step 1

### Step 9: Write verification test

**File:** `tests/test_autograd_gradient.py`

```python
"""Verify autograd gradients match finite-difference gradients."""

import numpy as np
import pytest
import torch

from Surrogate.nn_model import NNSurrogateModel
from Surrogate.ensemble import EnsembleMeanWrapper
from Surrogate.objectives import SurrogateObjective, SubsetSurrogateObjective


@pytest.fixture
def dummy_nn_model():
    """Create a small fitted NN model for testing (no real training data needed)."""
    # Build a tiny model, fit on synthetic data
    ...

class TestAutogradVsFD:
    """Compare autograd gradient to central FD at multiple random points."""

    def test_surrogate_objective_gradient_match(self, dummy_nn_model):
        """Autograd and FD gradients agree to < 0.1% relative error."""
        ...

    def test_ensemble_gradient_match(self, dummy_nn_model):
        """Ensemble mean autograd gradient matches FD."""
        ...

    def test_subset_objective_gradient_match(self, dummy_nn_model):
        """SubsetSurrogateObjective autograd gradient matches FD."""
        ...

    def test_block_objective_gradient_match(self, dummy_nn_model):
        """ReactionBlockSurrogateObjective autograd gradient matches FD."""
        ...

    def test_fallback_to_fd_for_rbf(self, mock_rbf_surrogate):
        """Non-torch surrogates still use FD without error."""
        ...
```

Test strategy:
- Create a small NNSurrogateModel (4->44, hidden=16, 1 block), fit on 50 random samples
- For each objective class: compute gradient via autograd AND via FD at 10 random x-points
- Assert `max(|g_auto - g_fd| / max(|g_fd|, 1e-12)) < 1e-3` (0.1%)
- Also test that RBF surrogate falls back to FD gracefully

**Estimated LOC:** ~120 lines

### Step 10: Write speed benchmark script

**File:** `scripts/benchmark_autograd_vs_fd.py`

```python
"""Benchmark: autograd vs finite-difference gradient computation time."""

import time
import numpy as np
from Surrogate.ensemble import load_nn_ensemble
from Surrogate.objectives import SurrogateObjective

def benchmark():
    model = load_nn_ensemble("data/surrogate_models/nn_ensemble/D3-deeper")
    # ... create objective with known targets ...

    x = np.array([-3.0, -5.0, 0.3, 0.5])

    # Warmup
    for _ in range(5):
        obj.objective_and_gradient(x)

    # Time autograd
    t0 = time.perf_counter()
    for _ in range(100):
        obj.objective_and_gradient(x)
    t_autograd = (time.perf_counter() - t0) / 100

    # Force FD path
    obj._use_autograd = False
    for _ in range(5):
        obj.objective_and_gradient(x)

    t0 = time.perf_counter()
    for _ in range(100):
        obj.objective_and_gradient(x)
    t_fd = (time.perf_counter() - t0) / 100

    print(f"FD:       {t_fd*1000:.2f} ms/eval")
    print(f"Autograd: {t_autograd*1000:.2f} ms/eval")
    print(f"Speedup:  {t_fd/t_autograd:.1f}x")
```

**Estimated LOC:** ~60 lines

---

## File Change Summary

| File | Change Type | LOC (est.) |
|---|---|---|
| `Surrogate/nn_model.py` | Add `predict_torch()` method | +35 |
| `Surrogate/ensemble.py` | Add `predict_torch()` method | +15 |
| `Surrogate/objectives.py` | Add autograd paths to all 4 classes | +130 |
| `Surrogate/cascade.py` | Autograd branch in 2 factory functions | +40 |
| `Surrogate/multistart.py` | Autograd branch in `_polish_candidate` | +25 |
| `tests/test_autograd_gradient.py` | New test file | +120 |
| `scripts/benchmark_autograd_vs_fd.py` | New benchmark script | +60 |
| **Total** | | **~425** |

## Execution Order

1. Step 1 (nn_model.py `predict_torch`) -- foundation, test in isolation
2. Step 2 (ensemble.py `predict_torch`) -- depends on Step 1
3. Steps 3-6 (objectives.py) -- depends on Steps 1-2
4. Step 7 (cascade.py, multistart.py) -- depends on Steps 3-6
5. Step 8 (precision tuning) -- done inline with Step 1
6. Step 9 (tests) -- verify everything
7. Step 10 (benchmark) -- final measurement

Steps 1-6 can be implemented in a single focused session (~2 hours). Steps 7-10 in a second session (~1 hour).

## Risk Assessment

| Risk | Likelihood | Mitigation |
|---|---|---|
| float32 precision causes >0.1% gradient mismatch | Low | Use float64 model copy if needed (Step 8 fallback) |
| `torch.autograd` slower than expected due to overhead | Low | Single backward pass for 4 params is extremely fast; overhead is tensor creation |
| Breaking change to objective API | None | All changes are internal; public API (`objective`, `gradient`, `objective_and_gradient`) unchanged |
| Ensemble `predict_torch` memory | None | 5 forward passes through small MLP, trivial |

## What This Does NOT Change

- No changes to RBF/POD surrogate paths (they have no torch model)
- No changes to the grid evaluation in `multistart.py` (uses `predict_batch`, no gradients needed)
- No changes to the optimizer interface (still scipy L-BFGS-B)
- No changes to training code
- No new dependencies (PyTorch is already required)
