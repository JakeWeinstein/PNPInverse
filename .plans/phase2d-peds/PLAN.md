# Phase 2d: Physics-Enhanced Deep Surrogate (PEDS)

## Goal

Build a PEDS surrogate that combines a coarse Firedrake BV solver with a neural corrector network to predict I-V curves (current_density, peroxide_current) from BV kinetics parameters (k0_1, k0_2, alpha_1, alpha_2). The PEDS must implement the shared surrogate API (`predict`, `predict_batch`) and live in `Surrogate/peds_model.py`.

## Architecture Overview

```
                    +------------------+
  (k0_1, k0_2,     |  Coarse PDE      |   I_coarse(22)
  alpha_1, alpha_2) |  Solver Wrapper  |---+
       |            +------------------+   |
       |                                   |   +------------------+
       +-----------------------------------+-->|  Neural Corrector |---> I_fine_hat(22)
                                               |  MLP             |
                                               +------------------+
```

**Design choice: fixed feature extractor, not end-to-end differentiable.**

Rationale: Firedrake is not natively differentiable through PyTorch. Making it differentiable (via pyadjoint + custom autograd) would be a major engineering effort for uncertain gain. Instead, the coarse solver acts as a fixed feature extractor: it produces approximate I-V curves that the neural corrector learns to correct. This is the simpler variant of PEDS that still captures most of the benefit: the coarse solver provides a physics-informed "warm start" that the NN only needs to learn a small residual correction for.

## Deliverables

| File | Purpose |
|------|---------|
| `Surrogate/peds_model.py` | `PEDSSurrogateModel` class implementing shared API |
| `Surrogate/coarse_solver.py` | `CoarseSolverWrapper` — thin wrapper around BV solver with a coarse mesh |
| `scripts/peds_generate_paired_data.py` | Script to generate paired (coarse, fine) training data |
| `scripts/peds_train.py` | Training script for the neural corrector |
| `data/surrogate_models/peds/` | Trained artifacts (corrector weights, normalizers, coarse solver config) |

## Detailed Implementation Plan

### Step 1: Coarse Solver Wrapper (`Surrogate/coarse_solver.py`)

**Goal:** Wrap the existing BV solver to run with a coarse mesh, producing approximate I-V curves fast enough for surrogate use (< 0.5s per eval).

**Key design decisions:**
- Coarse mesh: `make_graded_interval_mesh(N=30, beta=2.0)` instead of the fine mesh (`N=300, beta=2.0`). This is a 10x reduction in elements while preserving boundary-layer grading.
- Use `sweep_phi_applied_steady_bv_flux` from `Forward/steady_state/bv.py` which already implements warm-start voltage continuation — essential for BV convergence at large overpotentials.
- The wrapper caches the Firedrake context (mesh, function spaces, forms) on first call to avoid re-initialization overhead.

```python
class CoarseSolverWrapper:
    """Wraps the BV forward solver with a coarse mesh for PEDS.

    Caches the Firedrake mesh and base solver params on initialization.
    Produces I-V curves by running voltage continuation on the coarse mesh.
    """

    def __init__(
        self,
        base_solver_params: SolverParams,
        phi_applied_values: np.ndarray,  # (22,) voltage grid
        coarse_N: int = 30,
        coarse_beta: float = 2.0,
        steady_config: SteadyStateConfig = ...,
        i_scale: float = 1.0,
    ): ...

    def solve(self, k0_1, k0_2, alpha_1, alpha_2) -> dict:
        """Returns {'current_density': (22,), 'peroxide_current': (22,)}"""
        ...

    def solve_batch(self, parameters: np.ndarray) -> dict:
        """Batch version. parameters shape (M, 4)."""
        ...
```

**Implementation details:**
1. On `__init__`, create the coarse mesh once via `make_graded_interval_mesh(N=coarse_N, beta=coarse_beta)`.
2. Store the base `SolverParams` (from training data generation config).
3. `solve()` calls `configure_bv_solver_params(base_params, phi_applied=..., k0_values=[k0_1, k0_2], alpha_values=[alpha_1, alpha_2])` then `sweep_phi_applied_steady_bv_flux(...)` with the coarse mesh.
4. Extract `observed_flux` from each `SteadyStateResult` to build the I-V curve arrays.
5. `solve_batch()` loops over parameter rows (Firedrake is sequential; no GPU parallelism available).

**Performance target:** Each coarse sweep (22 voltage points) should take < 0.5s total. The fine mesh takes ~seconds per voltage point; with 10x fewer elements the coarse mesh should be much faster. If 30 elements is still too slow, we can try N=15.

**Validation checkpoint:** Before proceeding, benchmark the coarse solver:
- [ ] Time a single 22-point voltage sweep on coarse mesh
- [ ] Compare coarse vs fine I-V curves for 10 random parameter sets
- [ ] Quantify the coarse-to-fine error (this is what the corrector must learn)

### Step 2: Generate Paired Training Data (`scripts/peds_generate_paired_data.py`)

**Goal:** For each of the 3,194 training samples, generate coarse I-V curves to pair with the existing fine I-V curves.

**Input:** `data/surrogate_models/training_data_merged.npz`
- `parameters`: shape (3194, 4) — [k0_1, k0_2, alpha_1, alpha_2]
- `current_density`: shape (3194, 22) — fine CD curves
- `peroxide_current`: shape (3194, 22) — fine PC curves
- `phi_applied`: shape (22,) — voltage grid

**Output:** `data/surrogate_models/peds/paired_training_data.npz`
- `parameters`: (3194, 4)
- `fine_cd`: (3194, 22) — from existing data
- `fine_pc`: (3194, 22) — from existing data
- `coarse_cd`: (3194, 22) — newly generated
- `coarse_pc`: (3194, 22) — newly generated
- `phi_applied`: (22,)

**Implementation:**
1. Load existing training data.
2. Initialize `CoarseSolverWrapper` with the same base solver params used for fine data generation.
3. Loop over all 3194 samples, calling `coarse_solver.solve(k0_1, k0_2, alpha_1, alpha_2)`.
4. Save paired data to NPZ.
5. Handle failures: if a coarse solve fails for a sample, mark it with NaN and skip during training.
6. **Parallelism:** This is the bottleneck (3194 samples x ~0.5s = ~27 minutes). Consider running with `multiprocessing` but note Firedrake may not be fork-safe — test first. If not parallelizable, the serial runtime is still manageable.

**Important:** The coarse solver must use the same `base_solver_params` and `phi_applied_values` grid as the fine solver, differing only in mesh resolution. Verify this by checking that a fine solve on the coarse mesh parameters matches the coarse output.

### Step 3: Neural Corrector Architecture

**Goal:** Learn the mapping from (parameters, coarse I-V) to fine I-V residual.

**Architecture: Residual MLP**

```
Input: [log10(k0_1), log10(k0_2), alpha_1, alpha_2, coarse_cd(22), coarse_pc(22)]
       = 48-dimensional input

Output: [delta_cd(22), delta_pc(22)]
        = 44-dimensional residual correction

Final prediction: fine_hat = coarse + delta
```

The corrector learns the residual `delta = fine - coarse`, not the absolute fine output. This is the key PEDS insight: the residual is much smaller and smoother than the full output, so a smaller network with less data can learn it accurately.

**Network design (in `Surrogate/peds_model.py`):**

```python
class ResidualCorrectorMLP(nn.Module):
    """MLP that predicts the coarse-to-fine residual.

    Input: normalized [params(4) | coarse_iv(44)] = 48
    Output: residual [delta_cd(22) | delta_pc(22)] = 44
    """
    def __init__(self, n_in=48, n_out=44, hidden=128, n_blocks=3):
        # Same ResBlock architecture as NNSurrogateModel but:
        # - Smaller (3 blocks vs 4) since residual is simpler
        # - Input includes coarse features (48-dim vs 4-dim)
```

**Why include parameters as corrector input (not just coarse output)?**
The coarse solver output is a lossy representation of the physics. Including the raw parameters gives the corrector direct access to the kinetics information, which helps it learn parameter-dependent correction patterns (e.g., the correction at high k0 differs systematically from low k0).

### Step 4: Training Pipeline (`scripts/peds_train.py`)

**Goal:** Train the neural corrector on paired (coarse, fine) data.

**Training procedure:**
1. Load paired data from `data/surrogate_models/peds/paired_training_data.npz`.
2. Compute residuals: `delta_cd = fine_cd - coarse_cd`, `delta_pc = fine_pc - coarse_pc`.
3. Build input features: `X = [log10(k0), alpha, coarse_cd, coarse_pc]` (48-dim).
4. Build output targets: `Y = [delta_cd, delta_pc]` (44-dim).
5. Z-score normalize inputs and outputs (using `ZScoreNormalizer` from `nn_model.py`).
6. 85/15 train/val split (same as `NNSurrogateModel`).
7. Train with AdamW, CosineAnnealingWarmRestarts, early stopping (patience=500).
8. Save best model checkpoint + normalizers + metadata.

**Hyperparameters (starting point):**
- Hidden dim: 128
- ResBlocks: 3
- Learning rate: 1e-3
- Weight decay: 1e-4
- Batch size: 64
- Max epochs: 5000
- Early stopping patience: 500

**Data efficiency experiment:** Train on subsets (100, 300, 1000, 3194 samples) to verify the PEDS data efficiency claim (should achieve good accuracy with far fewer samples than pure NN).

**Output artifacts saved to `data/surrogate_models/peds/`:**
- `corrector_model.pt` — PyTorch state dict
- `corrector_normalizers.npz` — input/output normalizer stats
- `corrector_metadata.npz` — architecture config, phi_applied, training bounds
- `coarse_solver_config.json` — coarse mesh parameters (N, beta, steady config)
- `training_log.csv` — epoch, train_loss, val_loss

### Step 5: `PEDSSurrogateModel` Class (`Surrogate/peds_model.py`)

**Goal:** Implement the shared surrogate API, combining coarse solver + trained corrector.

```python
class PEDSSurrogateModel:
    """Physics-Enhanced Deep Surrogate combining coarse PDE + neural corrector.

    Implements the same API as BVSurrogateModel, NNSurrogateModel, etc.:
      predict(k0_1, k0_2, alpha_1, alpha_2) -> dict
      predict_batch(parameters) -> dict
      n_eta, phi_applied, is_fitted, training_bounds
    """

    def __init__(
        self,
        coarse_solver: CoarseSolverWrapper,
        corrector: ResidualCorrectorMLP,
        input_normalizer: ZScoreNormalizer,
        output_normalizer: ZScoreNormalizer,
        phi_applied: np.ndarray,
        training_bounds: dict | None = None,
    ): ...

    def predict(self, k0_1, k0_2, alpha_1, alpha_2) -> dict:
        """
        1. Run coarse solver -> coarse_cd, coarse_pc
        2. Build corrector input: [log10(k0s), alphas, coarse_cd, coarse_pc]
        3. Normalize, run corrector, denormalize -> delta_cd, delta_pc
        4. Return coarse + delta
        """
        ...

    def predict_batch(self, parameters: np.ndarray) -> dict:
        """Batch version: coarse solver runs sequentially, corrector runs batched."""
        ...

    @classmethod
    def load(cls, path: str, base_solver_params: SolverParams, device: str = "cpu"):
        """Load from saved artifacts directory.

        Requires base_solver_params to reconstruct the CoarseSolverWrapper
        (Firedrake objects cannot be serialized).
        """
        ...

    def save(self, path: str) -> None:
        """Save corrector weights + normalizers + config (not the Firedrake solver)."""
        ...
```

**Critical implementation note:** The `load()` method requires `base_solver_params` because Firedrake meshes/solvers cannot be pickled. The coarse solver must be reconstructed at load time. This is a departure from the pure-NN surrogates which are fully self-contained. Document this clearly.

**Predict flow:**
1. `predict(k0_1, k0_2, alpha_1, alpha_2)`:
   - Call `coarse_solver.solve(k0_1, k0_2, alpha_1, alpha_2)` -> `coarse_cd`, `coarse_pc`
   - Build feature vector: `[log10(k0_1), log10(k0_2), alpha_1, alpha_2, coarse_cd, coarse_pc]`
   - Normalize with `input_normalizer`
   - Forward pass through corrector MLP -> normalized delta
   - Denormalize with `output_normalizer` -> `delta_cd`, `delta_pc`
   - Return `{'current_density': coarse_cd + delta_cd, 'peroxide_current': coarse_pc + delta_pc, 'phi_applied': phi_applied}`

### Step 6: Integration and Validation

**6a. Register in `Surrogate/__init__.py`:**
```python
from Surrogate.peds_model import PEDSSurrogateModel
```

**6b. Fidelity benchmark (compare against existing surrogates):**
Run the same validation suite used for RBF/NN/POD-RBF surrogates on the PEDS model:
- Per-sample relative L2 error on held-out test set
- Worst-case I-V overlay plots
- Error vs parameter space visualization

**6c. Success criteria:**
- [ ] Prediction error < 1/3 of NN ensemble error (the 3x improvement claim)
- [ ] Coarse solver < 0.5s per 22-point sweep
- [ ] Total predict() time < 1s (coarse solve + corrector forward pass)
- [ ] Data-efficient: achieves NN-level accuracy with 100-300 training samples
- [ ] Works with existing `SurrogateObjective` / `run_multistart_inference` pipeline

**6d. Inference integration test:**
Run `run_multistart_inference` with PEDS surrogate on a known parameter recovery problem to verify end-to-end compatibility.

## Execution Order and Dependencies

```
Step 1: CoarseSolverWrapper
    |
    v
Step 1.5: Benchmark coarse solver (GATE: must be < 0.5s, reasonable accuracy)
    |
    v
Step 2: Generate paired training data (requires Step 1)
    |
    v
Step 3+4: Neural corrector + training (requires Step 2)
    |
    v
Step 5: PEDSSurrogateModel integration (requires Steps 1, 3, 4)
    |
    v
Step 6: Validation and registration (requires Step 5)
```

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Coarse solver too slow (> 0.5s) | Medium | High | Try N=15 mesh; reduce steady-state max_steps; cache coarse solutions |
| Coarse solver fails to converge for some parameter combos | Medium | Medium | Use continuation strategy; mark failures as NaN; ensure corrector handles missing coarse features gracefully |
| Corrector cannot learn residual well enough | Low | High | Increase network capacity; try conditional architecture; add more training data |
| Firedrake initialization overhead dominates | Low | Medium | Cache mesh + function spaces; reuse solver objects |
| Serialization: cannot pickle Firedrake objects | Certain | Medium | Document that `load()` requires `base_solver_params`; save coarse config as JSON |

## Key Engineering Decisions (Resolved)

1. **Fixed feature extractor (not end-to-end differentiable):** Chosen for simplicity. The coarse solver is treated as a black box. If end-to-end training is needed later, pyadjoint provides a path forward.

2. **Residual learning (delta = fine - coarse):** The corrector predicts the residual, not the absolute output. This makes the learning problem much simpler since the residual is small and smooth.

3. **Parameters + coarse output as corrector input:** Including raw parameters alongside coarse output gives the corrector maximum information to work with.

4. **All 3,194 samples for paired data:** Since generating coarse solves is relatively cheap (~27 min total), we generate paired data for all samples. This avoids the complexity of a data-efficiency-driven subset selection.

5. **Coarse mesh N=30 with beta=2.0 grading:** Starting point. 10x fewer elements than fine mesh (N=300) while preserving boundary-layer resolution through power-law grading. Will be tuned based on the Step 1.5 benchmark.

## File Inventory (Existing Files to Understand)

| File | Relevance |
|------|-----------|
| `Forward/bv_solver/mesh.py` | `make_graded_interval_mesh()` — used to create coarse mesh |
| `Forward/bv_solver/solvers.py` | `solve_bv_with_continuation()` — solver strategies |
| `Forward/bv_solver/forms.py` | `build_context()`, `build_forms()`, `set_initial_conditions()` |
| `Forward/bv_solver/config.py` | BV config parsing |
| `Forward/steady_state/bv.py` | `sweep_phi_applied_steady_bv_flux()` — voltage sweep with warm start |
| `Forward/steady_state/common.py` | `SteadyStateConfig`, `SteadyStateResult` |
| `Forward/params.py` | `SolverParams` frozen dataclass |
| `Surrogate/nn_model.py` | `NNSurrogateModel`, `ResNetMLP`, `ZScoreNormalizer` — reuse components |
| `Surrogate/surrogate_model.py` | `BVSurrogateModel` — API contract reference |
| `Surrogate/ensemble.py` | `EnsembleMeanWrapper` — API contract reference |
| `Surrogate/__init__.py` | Public API registration |
| `data/surrogate_models/training_data_merged.npz` | Existing fine training data |
