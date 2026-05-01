---
phase: 4-ismo-refinement
plan: 03
type: execute
wave: 2
depends_on: [4-01, 4-02]
files_modified:
  - Surrogate/ismo_retrain.py
  - Surrogate/nn_model.py
  - Surrogate/__init__.py
  - tests/test_ismo_retrain.py
autonomous: true
requirements: [ISMO-03]
must_haves:
  truths:
    - "Unified retrain_surrogate() dispatches to type-specific retraining for all 5 surrogate families"
    - "NN ensemble warm-start analytically corrects first/last layer weights for normalizer shift, then fine-tunes via fit(warm_start_state_dict=...) with reduced LR"
    - "Data merging validates no duplicates (in normalized log-space), no NaN, and appends new points to training set only"
    - "Retraining quality check compares before/after error and falls back to from-scratch if degraded"
    - "All retraining functions return new model objects (immutable pattern, no mutation of inputs)"
  artifacts:
    - path: "Surrogate/ismo_retrain.py"
      provides: "Unified ISMO retraining module with per-type dispatch and quality checks"
      min_lines: 400
    - path: "tests/test_ismo_retrain.py"
      provides: "Unit tests for data merging, dispatch, quality check logic"
      min_lines: 150
  key_links:
    - from: "Surrogate/ismo_retrain.py"
      to: "Surrogate/nn_model.py"
      via: "NNSurrogateModel.load(), NNSurrogateModel.fit(), NNSurrogateModel.save()"
      pattern: "NNSurrogateModel"
    - from: "Surrogate/ismo_retrain.py"
      to: "Surrogate/ensemble.py"
      via: "EnsembleMeanWrapper construction from retrained members"
      pattern: "EnsembleMeanWrapper"
    - from: "Surrogate/ismo_retrain.py"
      to: "Surrogate/pod_rbf_model.py"
      via: "PODRBFSurrogateModel.fit() from-scratch retraining"
      pattern: "PODRBFSurrogateModel"
    - from: "Surrogate/ismo_retrain.py"
      to: "Surrogate/gp_model.py"
      via: "GPSurrogateModel.fit() with warm-start hyperparameters"
      pattern: "GPSurrogateModel"
    - from: "Surrogate/ismo_retrain.py"
      to: "Surrogate/validation.py"
      via: "validate_surrogate() for quality checks"
      pattern: "validate_surrogate"
---

<objective>
Build the surrogate retraining pipeline for ISMO iterations. When new PDE training data arrives (from 4-02's acquisition step), each surrogate model type must be retrained on the merged dataset. The module must handle warm-start for NN ensembles (critical for feasibility), from-scratch retraining for fast models (RBF, POD-RBF, PCE), and warm-start for GPs. A unified interface dispatches to the correct retraining strategy, validates that retraining did not degrade quality, and returns new model objects without mutating the originals.

Purpose: ISMO requires iterating: optimize -> acquire new PDE data -> retrain surrogate -> re-optimize. Without a reliable, automated retraining pipeline, the loop cannot close. Warm-start for NN ensembles is the critical-path item because from-scratch training takes hours, while fine-tuning takes minutes.

Output: `Surrogate/ismo_retrain.py` with all retraining logic, updated `Surrogate/__init__.py` exports, and unit tests.
</objective>

<execution_context>
@/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/.claude/get-shit-done/workflows/execute-plan.md
@/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@Surrogate/nn_model.py
@Surrogate/ensemble.py
@Surrogate/surrogate_model.py
@Surrogate/pod_rbf_model.py
@Surrogate/gp_model.py
@Surrogate/pce_model.py
@Surrogate/validation.py
@Surrogate/io.py
@Surrogate/__init__.py

<interfaces>
<!-- Key types and contracts needed by the retraining module. -->

Training data format (from .npz files):
```python
# data/surrogate_models/training_data_merged.npz
data = np.load("training_data_merged.npz")
parameters = data["parameters"]       # (N, 4) -- [k0_1, k0_2, alpha_1, alpha_2]
current_density = data["current_density"]  # (N, 22)
peroxide_current = data["peroxide_current"]  # (N, 22)
phi_applied = data["phi_applied"]      # (22,)

# data/surrogate_models/split_indices.npz
splits = np.load("split_indices.npz")
train_idx = splits["train_idx"]  # indices into the merged data
test_idx = splits["test_idx"]
```

Common surrogate API (all model types implement this):
```python
class AnySurrogate:
    def fit(self, parameters, current_density, peroxide_current, phi_applied, **kwargs) -> self
    def predict_batch(self, parameters) -> Dict[str, np.ndarray]
    @property
    def is_fitted(self) -> bool
    @property
    def phi_applied(self) -> np.ndarray
    @property
    def training_bounds(self) -> Optional[Dict[str, Tuple[float, float]]]
```

NNSurrogateModel specifics (from nn_model.py):
```python
class NNSurrogateModel:
    def __init__(self, hidden=128, n_blocks=4, seed=0, device="cpu")
    def fit(self, parameters, current_density, peroxide_current, phi_applied,
            *, epochs=5000, lr=1e-3, weight_decay=1e-4, patience=500,
            batch_size=None, val_parameters=None, val_cd=None, val_pc=None,
            verbose=True) -> self
    def save(self, path: str) -> None
    @classmethod
    def load(cls, path: str, device: str = "cpu") -> NNSurrogateModel
    # Internal: _model (ResNetMLP), _input_normalizer, _output_normalizer (ZScoreNormalizer)
```

EnsembleMeanWrapper (from ensemble.py):
```python
class EnsembleMeanWrapper:
    def __init__(self, models: Sequence[NNSurrogateModel])
    # models attribute: list of NNSurrogateModel

def load_nn_ensemble(ensemble_dir, n_members=5, device="cpu") -> EnsembleMeanWrapper
```

GPSurrogateModel specifics (from gp_model.py):
```python
class GPSurrogateModel:
    def fit(self, parameters, current_density, peroxide_current, phi_applied,
            *, n_iters=200, lr=0.1, early_stop_tol=1e-6,
            early_stop_patience=20, n_jobs=-1, verbose=True) -> self
    def save(self, path: str) -> None
    @classmethod
    def load(cls, path: str, device="cpu") -> GPSurrogateModel
    # Internal: _gp_models list of (ExactGPModel, GaussianLikelihood)
```

PODRBFSurrogateModel (from pod_rbf_model.py):
```python
class PODRBFSurrogateModel:
    def __init__(self, config: PODRBFConfig | None = None)
    def fit(self, parameters, current_density, peroxide_current, phi_applied,
            verbose=True) -> self
```

BVSurrogateModel (from surrogate_model.py):
```python
class BVSurrogateModel:
    def __init__(self, config: SurrogateConfig | None = None)
    def fit(self, parameters, current_density, peroxide_current, phi_applied) -> self
```

PCESurrogateModel (from pce_model.py):
```python
class PCESurrogateModel:
    def __init__(self, config: PCEConfig | None = None)
    def fit(self, parameters, current_density, peroxide_current, phi_applied,
            verbose=True) -> self
    def save(self, path: str) -> None
    @staticmethod
    def load(path: str) -> PCESurrogateModel
```

Validation (from validation.py):
```python
def validate_surrogate(surrogate, test_parameters, test_cd, test_pc) -> Dict:
    # Returns: cd_rmse, pc_rmse, cd_mean_relative_error, pc_mean_relative_error, ...
```
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Implement data merging and validation utilities</name>
  <files>Surrogate/ismo_retrain.py</files>
  <action>
Create `Surrogate/ismo_retrain.py` with the following data-layer functions:

1. **`ISMORetrainConfig` dataclass:**
   ```python
   @dataclass
   class ISMORetrainConfig:
       """Configuration for ISMO surrogate retraining."""
       # NN ensemble warm-start settings
       nn_retrain_epochs: int = 100          # Fine-tuning epochs per member
       nn_retrain_lr: float = 1e-4           # Reduced LR for warm-start (10x lower than initial 1e-3)
       nn_retrain_patience: int = 50         # Early stopping patience during fine-tuning
       nn_retrain_weight_decay: float = 1e-4 # Same as initial training
       nn_from_scratch_fallback: bool = True  # Fall back to from-scratch if warm-start degrades
       nn_from_scratch_epochs: int = 3000    # Epochs for from-scratch fallback (less than initial 5000)
       nn_from_scratch_patience: int = 500

       # GP warm-start settings
       gp_retrain_iters: int = 100           # Fewer iterations for warm-start
       gp_retrain_lr: float = 0.05           # Reduced LR for warm-start (vs 0.1 initial)

       # Quality check thresholds
       max_degradation_ratio: float = 1.10   # Max allowed ratio of new_error/old_error (10% worse)
       quality_metric: str = "cd_mean_relative_error"  # Metric to compare

       # Data merging [REVISED: Issue 2 — operate in normalized log-space]
       duplicate_param_tol: float = 1e-6     # Tolerance for detecting duplicates in normalized space (see merge_training_data)

       # Output paths
       output_base_dir: str = "data/surrogate_models"
   ```

2. **`merge_training_data(existing_data, new_data, config) -> MergedData`:**
   - `existing_data`: dict with keys `parameters` (N_old, 4), `current_density` (N_old, 22), `peroxide_current` (N_old, 22), `phi_applied` (22,)
   - `new_data`: same structure with N_new samples
   - Validate `phi_applied` grids match between old and new (raise ValueError if not)
   - [REVISED: Issue 2] Check for duplicate parameters in **normalized space**: transform k0 columns (0, 1) to log10, leave alpha columns (2, 3) as-is, then compute per-column z-score normalization using the existing data's mean/std. Compute pairwise L2 distance between normalized new rows and normalized existing rows; drop any new row within `duplicate_param_tol` of any existing row. This ensures the tolerance is scale-invariant across parameters with vastly different magnitudes (k0 ~ 1e-4 vs alpha ~ 0.5). Log the number of duplicates dropped.
   - Check for NaN in new data outputs. Drop rows with any NaN. Log the number dropped.
   - Concatenate valid new data with existing data along axis 0
   - Return a new `MergedData` dataclass (immutable pattern) containing:
     - `parameters`: (N_old + N_new_valid, 4)
     - `current_density`: (N_old + N_new_valid, 22)
     - `peroxide_current`: (N_old + N_new_valid, 22)
     - `phi_applied`: (22,)
     - `n_old`: N_old
     - `n_new_valid`: count of new samples kept
     - `n_duplicates_dropped`: count
     - `n_nan_dropped`: count
     - `new_indices`: np.ndarray of indices in the merged array corresponding to new data (for train/test split update)

3. **`update_split_indices(existing_train_idx, existing_test_idx, n_old, n_new_valid) -> Tuple[np.ndarray, np.ndarray]`:**
   - New data indices go into the training set (ISMO acquired them specifically to improve the surrogate; putting them in the test set would waste them)
   - Returns (new_train_idx, new_test_idx) where new_train_idx = np.concatenate([existing_train_idx, np.arange(n_old, n_old + n_new_valid)])
   - Test indices remain unchanged
   - Validate no overlap between train and test

4. **`save_merged_data(merged_data, output_path, iteration) -> str`:**
   - Save to `{output_base_dir}/ismo_iter_{iteration}/training_data_merged.npz`
   - Also save updated split indices to `{output_base_dir}/ismo_iter_{iteration}/split_indices.npz`
   - Return the path to the saved file

Use `from __future__ import annotations` at the top. Use `logging` module (not print) for all diagnostic messages. Import types from the existing Surrogate modules only when needed (inside functions) to avoid circular imports.
  </action>
  <verify>
    <automated>cd /Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse && source ../venv-firedrake/bin/activate && python -c "from Surrogate.ismo_retrain import ISMORetrainConfig, merge_training_data, update_split_indices; print('OK')"</automated>
  </verify>
  <done>
    - ISMORetrainConfig dataclass importable with all fields
    - merge_training_data validates phi_applied match, drops duplicates, drops NaN rows
    - update_split_indices appends new indices to training set
    - No mutation of input data (immutable pattern)
  </done>
</task>

<task type="auto">
  <name>Task 2: Implement NN ensemble warm-start retraining</name>
  <files>Surrogate/ismo_retrain.py, Surrogate/nn_model.py</files>
  <action>
Add the NN ensemble warm-start retraining function to `Surrogate/ismo_retrain.py`:

**[REVISED: Issues 1 and 3] This section has been substantially revised to address:**
- **Issue 1 (HIGH):** Normalizer re-computation breaking warm-start weights. Fixed via analytical first/last layer weight correction.
- **Issue 3 (MEDIUM):** Training loop duplication. Fixed by extending `NNSurrogateModel.fit()` with an optional `warm_start_state_dict` parameter instead of writing a separate `_finetune_nn_member()`.
- **Missing item:** Rollback for partial ensemble member failures.
- **Missing item:** Save/load round-trip test for warm-start path.

**Step 0: Extend `NNSurrogateModel.fit()` with warm-start support** (in `Surrogate/nn_model.py`):

Add an optional `warm_start_state_dict: dict | None = None` parameter to `fit()`. Implementation:
- After the `self._model = ResNetMLP(...)` construction (line ~359), insert:
  ```python
  if warm_start_state_dict is not None:
      self._model.load_state_dict(warm_start_state_dict)
  ```
- This is a 3-line change. The existing `torch.manual_seed(self._seed)` still runs (ensures reproducible architecture construction), but the random weights are immediately overwritten by the warm-start state_dict.
- The caller controls epochs, lr, patience, and scheduler via the existing `fit()` parameters. No separate training loop needed.
- Add `warm_start_state_dict` to the method's docstring.
- Update `files_modified` in YAML header to include `Surrogate/nn_model.py`.

**Step 1: Implement `_correct_weights_for_normalizer_shift()` helper** (in `Surrogate/ismo_retrain.py`):

When normalizers change from old to new, the first and last linear layers must be analytically corrected so the network's input-output mapping is preserved under the new normalization.

The network operates as: `y_norm = f(x_norm)` where `x_norm = (x - mu_in) / sigma_in` and `y = y_norm * sigma_out + mu_out`.

If normalizers change from `(mu_in_old, sigma_in_old, mu_out_old, sigma_out_old)` to `(mu_in_new, sigma_in_new, mu_out_new, sigma_out_new)`, we need the network to produce the same physical output for the same physical input under the new normalization.

**Input layer correction** (first `nn.Linear` in `input_layer`, which is `input_layer[0]`):
```python
# Old: z_old = W_old * x_norm_old + b_old
#      where x_norm_old = (x - mu_old) / sigma_old
# New: z_new = W_new * x_norm_new + b_new  must equal z_old for same x
#      where x_norm_new = (x - mu_new) / sigma_new
# => x_norm_old = x_norm_new * (sigma_new / sigma_old) + (mu_new - mu_old) / sigma_old  ... (incorrect)
# Actually: x = x_norm_old * sigma_old + mu_old = x_norm_new * sigma_new + mu_new
# => x_norm_old = x_norm_new * (sigma_new / sigma_old) + (mu_new - mu_old) / sigma_old
# Wait, solving: x_norm_old = (x - mu_old) / sigma_old
#                           = ((x_norm_new * sigma_new + mu_new) - mu_old) / sigma_old
#                           = x_norm_new * (sigma_new / sigma_old) + (mu_new - mu_old) / sigma_old
# So: W_new = W_old * diag(sigma_new / sigma_old)
#     b_new = b_old + W_old @ ((mu_new - mu_old) / sigma_old)

ratio_in = sigma_in_new / sigma_in_old  # shape (4,)
offset_in = (mu_in_new - mu_in_old) / sigma_in_old  # shape (4,)
W_in = state_dict["input_layer.0.weight"]  # (hidden, 4)
b_in = state_dict["input_layer.0.bias"]    # (hidden,)
state_dict["input_layer.0.weight"] = W_in * ratio_in.unsqueeze(0)  # broadcast over hidden dim
state_dict["input_layer.0.bias"] = b_in + (W_in @ offset_in)
```

**Output layer correction** (last `nn.Linear` in `output_layer`, which is `output_layer[3]`):
```python
# Old: y_old = y_norm_old * sigma_out_old + mu_out_old
# New: y_new = y_norm_new * sigma_out_new + mu_out_new  must equal y_old
# => y_norm_new = (y_norm_old * sigma_out_old + mu_out_old - mu_out_new) / sigma_out_new
#               = y_norm_old * (sigma_out_old / sigma_out_new) + (mu_out_old - mu_out_new) / sigma_out_new
# So: W_new = W_old * (sigma_out_old / sigma_out_new).unsqueeze(1)  -- scale rows
#     b_new = b_old * (sigma_out_old / sigma_out_new) + (mu_out_old - mu_out_new) / sigma_out_new

ratio_out = sigma_out_old / sigma_out_new  # shape (44,)
offset_out = (mu_out_old - mu_out_new) / sigma_out_new  # shape (44,)
W_out = state_dict["output_layer.3.weight"]  # (44, hidden//2)
b_out = state_dict["output_layer.3.bias"]    # (44,)
state_dict["output_layer.3.weight"] = W_out * ratio_out.unsqueeze(1)
state_dict["output_layer.3.bias"] = b_out * ratio_out + offset_out
```

This function takes `(state_dict, old_input_norm, new_input_norm, old_output_norm, new_output_norm)` and returns a corrected `state_dict` (new dict, no mutation). Log the normalizer drift magnitude: `max(|mu_new - mu_old| / sigma_old)` for both input and output as a diagnostic.

1. **`retrain_nn_ensemble(ensemble, merged_data, train_idx, test_idx, config, iteration, device="cpu") -> EnsembleMeanWrapper`:**
   - `ensemble`: existing `EnsembleMeanWrapper` with `.models` list of `NNSurrogateModel` instances
   - Extract the number of members from `len(ensemble.models)`
   - Extract architecture config from the first member: `hidden = ensemble.models[0]._hidden`, `n_blocks = ensemble.models[0]._n_blocks`

   **[REVISED: Missing item — rollback for partial failures]** Wrap per-member retraining in try/except. Track `retrained_members` and `failed_members`. If a member fails:
   - Log the exception with traceback
   - Keep the original (un-retrained) member as fallback
   - Continue to next member
   - After the loop, if ALL members failed, raise an error. If some succeeded, return the mixed ensemble (retrained + original fallbacks) with a warning log.

   For each member `m` (index `i`) in `ensemble.models`:

   a. **Save pre-retrain validation error.** Run `validate_surrogate(m, test_params, test_cd, test_pc)` using the held-out test set. Store `old_error = metrics[config.quality_metric]`.

   b. **Build merged training arrays** from `merged_data` using `train_idx`:
      - `train_params = merged_data.parameters[train_idx]`
      - `train_cd = merged_data.current_density[train_idx]`
      - `train_pc = merged_data.peroxide_current[train_idx]`
      - `val_params = merged_data.parameters[test_idx]` (use test set for validation during fine-tuning)
      - Similarly for val_cd, val_pc

   c. **Re-compute normalizers** on the full merged training data.
      - Build log-space X from `train_params`
      - Compute new `ZScoreNormalizer.from_data(X_log)` for inputs
      - Compute new `ZScoreNormalizer.from_data(Y)` for outputs (where Y = concat(train_cd, train_pc))

   d. **[REVISED: Issue 1] Analytically correct weights for normalizer shift.** Call `_correct_weights_for_normalizer_shift()` with the old member's state_dict, old normalizers (`m._input_normalizer`, `m._output_normalizer`), and new normalizers. This produces a corrected state_dict that preserves the network's physical input-output mapping under the new normalization. The network's output on existing data should be nearly identical before and after correction (up to floating-point precision).

   e. **[REVISED: Issue 3] Fine-tune using `NNSurrogateModel.fit()` with `warm_start_state_dict`.** Create a new `NNSurrogateModel(hidden=m._hidden, n_blocks=m._n_blocks, seed=m._seed, device=device)`. Call:
      ```python
      new_member.fit(
          train_params, train_cd, train_pc, phi_applied,
          epochs=config.nn_retrain_epochs,       # default 100
          lr=config.nn_retrain_lr,               # default 1e-4
          weight_decay=config.nn_retrain_weight_decay,
          patience=config.nn_retrain_patience,    # default 50
          val_parameters=val_params,
          val_cd=val_cd,
          val_pc=val_pc,
          warm_start_state_dict=corrected_state_dict,
          verbose=True,
      )
      ```
      The `fit()` method constructs the `ResNetMLP`, then immediately overwrites its random weights with the corrected state_dict, then runs the normal training loop with reduced epochs/LR. No training loop duplication.

   f. **Post-retrain validation.** Run `validate_surrogate(new_member, test_params, test_cd, test_pc)`. Compute `new_error = metrics[config.quality_metric]`.

   g. **Quality check.** If `new_error / old_error > config.max_degradation_ratio`:
      - Log a warning: "Warm-start degraded member {i}: {old_error:.6e} -> {new_error:.6e}"
      - If `config.nn_from_scratch_fallback`:
        - Retrain from scratch using `NNSurrogateModel.fit()` with `epochs=config.nn_from_scratch_epochs`, `lr=1e-3` (original), `patience=config.nn_from_scratch_patience` (no `warm_start_state_dict`)
        - Log: "Falling back to from-scratch training for member {i}"
        - Re-run validation on the from-scratch model
      - If still degraded after from-scratch, log error but keep the from-scratch model (it should be at least as good as the original given more data)

   h. **Save the retrained member** to `{config.output_base_dir}/nn_ensemble/ismo_iter_{iteration}/member_{i}/saved_model/`

   - After all members are retrained, construct and return a new `EnsembleMeanWrapper(retrained_members)`.
   - Log summary: "Ensemble retrained: {n_warm_started} warm-start, {n_from_scratch} from-scratch, {n_failed} failed (kept original)"
  </action>
  <verify>
    <automated>cd /Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse && source ../venv-firedrake/bin/activate && python -c "from Surrogate.ismo_retrain import retrain_nn_ensemble, _correct_weights_for_normalizer_shift; print('OK')"</automated>
  </verify>
  <done>
    - retrain_nn_ensemble is importable and has correct signature
    - [REVISED] _correct_weights_for_normalizer_shift analytically corrects first/last layer weights when normalizers change
    - [REVISED] Warm-start uses NNSurrogateModel.fit(warm_start_state_dict=...) — no training loop duplication
    - [REVISED] NNSurrogateModel.fit() in nn_model.py extended with warm_start_state_dict parameter
    - Quality check compares before/after error
    - Falls back to from-scratch if warm-start degrades beyond threshold
    - [REVISED] Per-member try/except with fallback to original member on failure
    - Returns new EnsembleMeanWrapper (does not mutate input ensemble)
    - Saves retrained members to versioned directory
  </done>
</task>

<task type="auto">
  <name>Task 3: Implement retraining for POD-RBF, RBF baseline, PCE, and GP</name>
  <files>Surrogate/ismo_retrain.py</files>
  <action>
Add retraining functions for the remaining surrogate types to `Surrogate/ismo_retrain.py`:

1. **`retrain_pod_rbf(model, merged_data, train_idx, test_idx, config, iteration) -> PODRBFSurrogateModel`:**
   - POD-RBF is fast enough to always retrain from scratch (SVD + RBF fitting takes seconds)
   - Create a new `PODRBFSurrogateModel` with the same `config` as the original: `new_model = PODRBFSurrogateModel(config=model.config)`
   - Call `new_model.fit(train_params, train_cd, train_pc, phi_applied)`
   - Validate: run `validate_surrogate(new_model, test_params, test_cd, test_pc)`
   - Compare to old model error. Log if degraded (should not happen with more data, but worth checking).
   - Save via pickle to `{config.output_base_dir}/ismo_iter_{iteration}/model_pod_rbf.pkl`
   - Return `new_model`

2. **`retrain_rbf_baseline(model, merged_data, train_idx, test_idx, config, iteration) -> BVSurrogateModel`:**
   - Same pattern as POD-RBF: always from scratch
   - `new_model = BVSurrogateModel(config=model.config)`
   - `new_model.fit(train_params, train_cd, train_pc, phi_applied)`
   - Validate, save to `{config.output_base_dir}/ismo_iter_{iteration}/model_rbf_baseline.pkl`
   - Return `new_model`

3. **`retrain_pce(model, merged_data, train_idx, test_idx, config, iteration) -> PCESurrogateModel`:**
   - PCE is fast (least-squares fit): always from scratch
   - `new_model = PCESurrogateModel(config=model.config)`
   - `new_model.fit(train_params, train_cd, train_pc, phi_applied)`
   - Validate, save to `{config.output_base_dir}/ismo_iter_{iteration}/pce_model.pkl`
   - Return `new_model`

4. **`retrain_gp(model, merged_data, train_idx, test_idx, config, iteration, device="cpu") -> GPSurrogateModel`:**
   - GP retraining can warm-start hyperparameters. The strategy:
     a. Create new `GPSurrogateModel(device=device)`
     b. Call `new_model.fit()` with `n_iters=config.gp_retrain_iters`, `lr=config.gp_retrain_lr`
        - Note: GPSurrogateModel.fit() uses joblib parallel fitting of 44 independent GPs
        - The current fit() always starts from random hyperparameters
     c. **Warm-start enhancement (if feasible):** Before fit(), extract old model's kernel hyperparameters (lengthscales, outputscale, noise) from `model._gp_models`. After creating the new GP model objects inside fit(), load these old hyperparameters as initial values. However, since fit() is self-contained and creates models internally, the cleanest approach is:
        - Call `new_model.fit()` normally (it optimizes hyperparameters from scratch, but with fewer iterations since we use `config.gp_retrain_iters=100` vs the default 200)
        - The GP kernel optimization is fast enough that warm-starting hyperparameters is a nice-to-have, not critical
        - **Alternative:** Extract old hyperparameters, pass them as initial values to `_fit_single_gp()`. This requires modifying the function signature. For now, skip this complexity and rely on the fact that GP fitting is already fast (~10s for 44 GPs with 500 samples).
     d. Validate, compare to old error
     e. Save to `{config.output_base_dir}/ismo_iter_{iteration}/gp/`
     f. Return `new_model`
   - **Scaling concern:** Exact GP scales as O(N^3). For N > ~2000, this becomes slow. Log a warning if N_train > 1500: "GP training data size {N} may cause slow fitting. Consider switching to SVGP."

5. **Quality check helper `_check_retrain_quality(old_model, new_model, test_params, test_cd, test_pc, config, model_name) -> Tuple[bool, Dict]`:**
   - Runs `validate_surrogate()` on both old and new models
   - Computes ratio = new_error / old_error
   - Returns (passed: bool, metrics: dict with old_error, new_error, ratio, quality_metric)
   - Logs the comparison
  </action>
  <verify>
    <automated>cd /Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse && source ../venv-firedrake/bin/activate && python -c "from Surrogate.ismo_retrain import retrain_pod_rbf, retrain_rbf_baseline, retrain_pce, retrain_gp; print('OK')"</automated>
  </verify>
  <done>
    - All 4 retraining functions importable
    - POD-RBF, RBF baseline, PCE retrain from scratch (fast, no warm-start needed)
    - GP retrains with reduced iterations
    - Quality check helper compares before/after for all types
    - Each function returns a new model object
    - Each function saves to versioned ismo_iter directory
  </done>
</task>

<task type="auto">
  <name>Task 4: Implement unified dispatch and ISMO retrain orchestrator</name>
  <files>Surrogate/ismo_retrain.py, Surrogate/__init__.py</files>
  <action>
Add the unified dispatch interface to `Surrogate/ismo_retrain.py`:

1. **`retrain_surrogate(surrogate, new_data, existing_data, config, iteration, train_idx, test_idx, device="cpu") -> Any`:**
   - This is the primary public API. It dispatches to the correct retraining function based on the surrogate type.
   - Type dispatch logic:
     ```python
     from Surrogate.ensemble import EnsembleMeanWrapper
     from Surrogate.nn_model import NNSurrogateModel
     from Surrogate.surrogate_model import BVSurrogateModel
     from Surrogate.pod_rbf_model import PODRBFSurrogateModel
     from Surrogate.gp_model import GPSurrogateModel
     from Surrogate.pce_model import PCESurrogateModel

     if isinstance(surrogate, EnsembleMeanWrapper):
         return retrain_nn_ensemble(surrogate, merged, train_idx, test_idx, config, iteration, device)
     elif isinstance(surrogate, NNSurrogateModel):
         # Single NN member -- wrap in a 1-member ensemble for consistency, or retrain directly
         # For now, retrain as a single member using the same warm-start logic
         return _retrain_single_nn(surrogate, merged, train_idx, test_idx, config, iteration, device)
     elif isinstance(surrogate, PODRBFSurrogateModel):
         return retrain_pod_rbf(surrogate, merged, train_idx, test_idx, config, iteration)
     elif isinstance(surrogate, BVSurrogateModel):
         return retrain_rbf_baseline(surrogate, merged, train_idx, test_idx, config, iteration)
     elif isinstance(surrogate, GPSurrogateModel):
         return retrain_gp(surrogate, merged, train_idx, test_idx, config, iteration, device)
     elif isinstance(surrogate, PCESurrogateModel):
         return retrain_pce(surrogate, merged, train_idx, test_idx, config, iteration)
     else:
         raise TypeError(f"Unknown surrogate type: {type(surrogate).__name__}")
     ```
   - Before dispatching:
     a. Call `merge_training_data(existing_data, new_data, config)` to get `merged`
     b. Call `update_split_indices(train_idx, test_idx, merged.n_old, merged.n_new_valid)` to get updated indices
     c. Log: "ISMO iter {iteration}: merging {new_data N} new samples with {existing_data N} existing -> {merged total} total ({merged.n_duplicates_dropped} duplicates dropped, {merged.n_nan_dropped} NaN dropped)"
   - After dispatching:
     a. Log the quality check results
     b. Return the new surrogate model

2. **`ISMORetrainResult` dataclass:**
   ```python
   @dataclass
   class ISMORetrainResult:
       """Result of ISMO surrogate retraining."""
       surrogate: Any                       # The retrained surrogate model
       merged_data: MergedData              # The merged training dataset
       updated_train_idx: np.ndarray        # Updated training indices
       updated_test_idx: np.ndarray         # Updated test indices (unchanged)
       quality_passed: bool                 # Whether quality check passed
       old_error: float                     # Pre-retrain error
       new_error: float                     # Post-retrain error
       error_ratio: float                   # new_error / old_error
       quality_metric: str                  # Which metric was used
       retrain_method: str                  # "warm_start", "from_scratch", or "from_scratch_fallback"
       iteration: int                       # ISMO iteration number
       save_path: str                       # Where the retrained model was saved
   ```

3. **`retrain_surrogate_full(surrogate, new_data, existing_data, config, iteration, train_idx, test_idx, device="cpu") -> ISMORetrainResult`:**
   - Higher-level wrapper that returns the full result dataclass
   - Calls `retrain_surrogate()` internally
   - Populates all fields of `ISMORetrainResult`
   - Saves merged data via `save_merged_data()`
   - Logs a complete summary

4. **Update `Surrogate/__init__.py`:**
   - Add imports: `from Surrogate.ismo_retrain import ISMORetrainConfig, ISMORetrainResult, retrain_surrogate, retrain_surrogate_full, merge_training_data`
   - Add to `__all__`: `"ISMORetrainConfig"`, `"ISMORetrainResult"`, `"retrain_surrogate"`, `"retrain_surrogate_full"`, `"merge_training_data"`
  </action>
  <verify>
    <automated>cd /Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse && source ../venv-firedrake/bin/activate && python -c "from Surrogate import retrain_surrogate, ISMORetrainConfig, ISMORetrainResult, merge_training_data; print('OK')"</automated>
  </verify>
  <done>
    - retrain_surrogate dispatches correctly based on isinstance checks
    - ISMORetrainResult captures all retraining metadata
    - retrain_surrogate_full provides the complete pipeline in one call
    - Surrogate/__init__.py updated with new exports
    - Type dispatch covers all 5 surrogate families + raises TypeError for unknown
  </done>
</task>

<task type="auto">
  <name>Task 5: Write unit tests for ISMO retraining pipeline</name>
  <files>tests/test_ismo_retrain.py</files>
  <action>
Create `tests/test_ismo_retrain.py` with the following test categories. Use synthetic data (no PDE solves) to keep tests fast.

**Test fixtures (module-level or conftest):**
```python
@pytest.fixture
def synthetic_training_data():
    """Generate synthetic I-V curve data for testing."""
    rng = np.random.default_rng(42)
    N = 50
    n_eta = 22
    params = rng.uniform(size=(N, 4))
    params[:, 0] *= 1e-3  # k0_1 scale
    params[:, 1] *= 1e-4  # k0_2 scale
    params[:, 2] = 0.3 + 0.4 * params[:, 2]  # alpha_1 in [0.3, 0.7]
    params[:, 3] = 0.3 + 0.4 * params[:, 3]  # alpha_2 in [0.3, 0.7]
    phi = np.linspace(-1.5, 0.5, n_eta)
    cd = rng.standard_normal((N, n_eta)) * 0.1
    pc = rng.standard_normal((N, n_eta)) * 0.01
    return {"parameters": params, "current_density": cd, "peroxide_current": pc, "phi_applied": phi}
```

**Test categories:**

1. **`test_merge_training_data_basic`**: Merge 50 old + 10 new samples. Verify output shape is (60, 4), (60, 22), (60, 22). Verify n_old=50, n_new_valid=10.

2. **`test_merge_training_data_duplicates`**: [REVISED: Issue 2] Create new_data where 3 rows have identical parameters to existing_data. Verify n_duplicates_dropped=3, total is 50+7=57. Verify that duplicate detection operates in normalized log-space (k0 columns transformed to log10 before distance computation).

3. **`test_merge_training_data_nan_dropped`**: Create new_data with 2 rows containing NaN in current_density. Verify n_nan_dropped=2, total is 50+8=58.

4. **`test_merge_training_data_phi_mismatch`**: Pass new_data with a different phi_applied grid. Verify ValueError is raised.

5. **`test_merge_training_data_immutability`**: Verify that the input arrays are not modified after merge_training_data returns. Make copies before calling, compare after.

6. **`test_update_split_indices`**: With train_idx=[0,1,2,...,39], test_idx=[40,...,49], n_old=50, n_new_valid=10. Verify new_train_idx includes [0,...,39, 50,...,59]. Verify test_idx unchanged. Verify no overlap.

7. **`test_retrain_rbf_baseline`**: Fit a BVSurrogateModel on synthetic data, then retrain with 10 new samples. Verify the returned model is a different object, is_fitted=True, and predict_batch works.

8. **`test_retrain_pod_rbf`**: Same pattern for PODRBFSurrogateModel. Verify n_modes > 0, is_fitted=True.

9. **`test_retrain_dispatch_rbf`**: Call retrain_surrogate with a BVSurrogateModel. Verify it returns a BVSurrogateModel. Verify it does NOT mutate the original.

10. **`test_retrain_dispatch_unknown_type`**: Call retrain_surrogate with an unsupported type (e.g., a plain object). Verify TypeError is raised.

11. **`test_quality_check_passes`**: Mock or use a retrained model that has lower error than the original. Verify quality_passed=True.

12. **`test_quality_check_detects_degradation`**: Use a config with `max_degradation_ratio=0.5` (very strict). Verify the quality check correctly detects that retraining "degraded" the model.

Skip NN ensemble and GP tests if torch/gpytorch are not available (use `pytest.importorskip("torch")`). If torch is available, include:

13. **`test_retrain_nn_single_member`**: Create a small NNSurrogateModel (hidden=16, n_blocks=1), fit on synthetic data, retrain with warm-start. Verify weights changed, model is fitted, predict works.

14. **`test_retrain_dispatch_nn_ensemble`**: Create a 2-member ensemble of small NNSurrogateModels, retrain via dispatch. Verify returns EnsembleMeanWrapper with 2 members.

15. **[REVISED: Missing item] `test_weight_correction_preserves_output`**: Create a small NNSurrogateModel, fit it, then compute new normalizers from slightly different data. Apply `_correct_weights_for_normalizer_shift()` to get a corrected state_dict. Build a new model with the corrected weights and new normalizers. Verify that `predict_batch()` output on a set of test points is nearly identical (atol=1e-5) between the original model and the corrected model. This confirms the analytical correction is mathematically correct.

16. **[REVISED: Missing item] `test_nn_warm_start_save_load_roundtrip`**: Create a small NNSurrogateModel (hidden=16, n_blocks=1), fit on synthetic data, warm-start retrain with new data, save to a temp directory, load from that directory, and verify that `predict_batch()` output matches between the in-memory retrained model and the loaded model. This catches normalizer serialization bugs.

17. **[REVISED: Missing item] `test_nn_ensemble_partial_failure_rollback`**: Create a 3-member ensemble. Mock one member to raise an exception during retraining. Verify that `retrain_nn_ensemble()` returns a 3-member ensemble where the failed member is the original (un-retrained) model and the other 2 are retrained.

Use `source ../venv-firedrake/bin/activate` for the test runner. Run with `pytest tests/test_ismo_retrain.py -v`.
  </action>
  <verify>
    <automated>cd /Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse && source ../venv-firedrake/bin/activate && python -m pytest tests/test_ismo_retrain.py -v --tb=short 2>&1 | tail -30</automated>
  </verify>
  <done>
    - All non-torch tests pass (merge, split, RBF, POD-RBF, dispatch, quality check)
    - NN tests pass if torch is available, skipped otherwise
    - No test mutates input data
    - Tests use synthetic data only (no PDE solves, fast execution)
    - [REVISED] At least 15 test functions total (added weight correction, save/load roundtrip, partial failure rollback)
  </done>
</task>

</tasks>

<verification>
1. `python -c "from Surrogate.ismo_retrain import retrain_surrogate, ISMORetrainConfig, merge_training_data; print('imports OK')"` -- all public API importable
2. `python -c "from Surrogate import retrain_surrogate, ISMORetrainConfig; print('re-export OK')"` -- re-exported from Surrogate.__init__
3. `python -m pytest tests/test_ismo_retrain.py -v --tb=short` -- all tests pass
4. `python -c "from Surrogate.ismo_retrain import ISMORetrainConfig; c = ISMORetrainConfig(); print(f'nn_retrain_epochs={c.nn_retrain_epochs}, nn_retrain_lr={c.nn_retrain_lr}')"` -- config defaults correct
5. `grep -c 'def retrain_' Surrogate/ismo_retrain.py` -- at least 6 (retrain_surrogate, retrain_surrogate_full, retrain_nn_ensemble, retrain_pod_rbf, retrain_rbf_baseline, retrain_pce, retrain_gp)
6. `grep -c 'def test_' tests/test_ismo_retrain.py` -- at least 15
</verification>

<success_criteria>
- Unified retrain_surrogate() dispatches to all 5 surrogate families via isinstance
- [REVISED] NN ensemble warm-start: re-computes normalizers, analytically corrects first/last layer weights, fine-tunes via fit(warm_start_state_dict=...) with 10x lower LR, verifies no degradation
- NN ensemble fallback: if warm-start degrades, falls back to from-scratch training
- [REVISED] Per-member try/except with rollback to original member on failure; mixed ensemble returned with warning
- POD-RBF, RBF baseline, PCE: retrain from scratch (fast, no warm-start complexity)
- GP: retrain with reduced iterations (hyperparameter warm-start deferred to future enhancement)
- Data merging: validates phi_applied match, drops duplicates within tolerance, drops NaN rows, appends new data to training split
- Quality check: compares before/after error using configurable metric, flags degradation if ratio exceeds threshold
- All functions return new model objects (immutable pattern -- no mutation of inputs)
- ISMORetrainResult captures full metadata: errors, ratio, method, iteration, save path
- Unit tests cover data merging, split updates, dispatch, quality checks, and at least RBF/POD-RBF retraining
- All code uses logging (not print) for diagnostic output
</success_criteria>

<design_decisions>

1. **Why warm-start only for NN, not for GP?**
   GP hyperparameter warm-start requires modifying the internal `_fit_single_gp()` function to accept initial hyperparameter values. This adds complexity for marginal benefit since GP fitting is already fast (~10s for 44 GPs). The NN ensemble warm-start is critical because from-scratch training takes 1-2 hours for 5 members at 5000 epochs each, while fine-tuning at 100 epochs takes ~2 minutes. GP warm-start is logged as a future enhancement.

2. **[REVISED] Why re-compute normalizers AND analytically correct weights?**
   Z-score normalizers (mean, std) are computed from the training data. When ISMO adds new samples, the data distribution shifts. Re-computing normalizers is necessary so all data is consistently normalized. However, simply re-computing normalizers while keeping old weights creates a catastrophic mismatch: the old weights were trained under the old normalization, so the first forward pass with new normalizers produces nonsensical outputs. The fix is to analytically correct the first linear layer (input normalization change) and last linear layer (output normalization change) so the network's physical input-output mapping is preserved. The math is straightforward because normalization is a linear transform, and the first/last layers are linear. Interior layers (ResBlocks) operate in a normalized space that is invariant to the affine correction, so they need no changes. After correction, fine-tuning adapts the network to the new data points with minimal disruption to existing accuracy.

3. **Why from-scratch for POD-RBF/RBF/PCE?**
   SVD + RBF fitting takes <5 seconds for ~500 samples. PCE least-squares takes <1 second. The overhead of implementing warm-start logic far exceeds the time saved. From-scratch also avoids subtle bugs where warm-start hyperparameters (e.g., smoothing values) are suboptimal for the new data distribution.

4. **Why put new ISMO data into training set, not test set?**
   ISMO explicitly acquires data at points where the surrogate is uncertain or where the optimizer solution lies. Putting these points in the test set would waste them -- they were acquired specifically to improve surrogate accuracy in regions that matter for inference. The test set remains the original held-out set, providing a consistent benchmark across ISMO iterations.

5. **Why max_degradation_ratio = 1.10 (10% tolerance)?**
   Adding data should never make a model worse (in expectation). However, stochastic training (random weight initialization, mini-batch ordering) can cause small fluctuations. A 10% tolerance absorbs this noise. If retraining degrades beyond 10%, it signals a real problem (e.g., normalizer bug, data corruption) and triggers the from-scratch fallback.

6. **Why return new objects instead of mutating?**
   The immutable pattern (required by project coding style) ensures the caller always has the original model available for comparison. It also prevents subtle bugs where an ISMO iteration partially modifies a model and then fails, leaving the model in an inconsistent state. The old model is never touched; the new model is a completely independent object.

7. **Why save to versioned `ismo_iter_{n}/` directories?**
   Each ISMO iteration produces a distinct surrogate model. Versioned directories enable: (a) rollback if a later iteration degrades, (b) comparison of surrogate accuracy across iterations, (c) debugging by replaying specific iterations, (d) the ISMO convergence check (4-04) can compare errors across iterations.

</design_decisions>

<output>
After completion, create `.planning/phases/4-ismo-refinement/4-03-SUMMARY.md`
</output>

---

## Revision Log

**Revision 1** (addressing 4-03-REVIEW.md findings):

### HIGH — Issue 1: Normalizer re-computation breaks warm-start weights
- **Resolution:** Option A selected — analytical first/last layer weight correction.
- **Changes:** Task 2 rewritten to include `_correct_weights_for_normalizer_shift()` helper that adjusts input_layer[0] (Linear) and output_layer[3] (Linear) weights/biases to preserve the network's physical I/O mapping when normalizers change. Full derivation included in Task 2 action. Design decision 2 updated.
- **Rationale:** Option A is the cleanest: the math is exact for linear layers, zero training is needed to compensate for the normalizer shift, and fine-tuning then only needs to learn the new data points rather than recover from a distribution mismatch. Options B (keep old normalizers) and C (increase epochs) were rejected because B prevents proper normalization of new data and C wastes compute without guarantees.

### MEDIUM — Issue 2: Duplicate detection in wrong space
- **Resolution:** Duplicate detection now operates in normalized log-space.
- **Changes:** `duplicate_param_tol` default changed from 1e-8 to 1e-6. `merge_training_data()` description updated to transform k0 columns to log10 and z-score normalize all columns before computing L2 distance. Test 2 updated to verify log-space operation.

### MEDIUM — Issue 3: Training loop duplication
- **Resolution:** Extend `NNSurrogateModel.fit()` with `warm_start_state_dict` parameter instead of writing a separate `_finetune_nn_member()`.
- **Changes:** Task 2 Step 0 added: 3-line change to `fit()` in `nn_model.py` to load state_dict after model construction if provided. `files_modified` updated to include `nn_model.py`. Task 2 step (e) rewritten to use `fit(warm_start_state_dict=corrected_state_dict, ...)`. CosineAnnealingWarmRestarts T_0 issue (Issue 7) resolved implicitly — the existing fit() scheduler (T_0=500) is used as-is; with 100 fine-tuning epochs the LR decays smoothly from the configured value without any restart, which is appropriate for fine-tuning.

### Missing Item — No rollback for partial ensemble member failures
- **Resolution:** Per-member try/except added to `retrain_nn_ensemble()`.
- **Changes:** Task 2 now specifies that failed members fall back to the original (un-retrained) model. If all members fail, an error is raised. Summary log updated to report failed count. Test 17 added to verify rollback behavior.

### Missing Item — No save/load round-trip test for warm-start path
- **Resolution:** Test 16 (`test_nn_warm_start_save_load_roundtrip`) added to Task 5.
- **Changes:** Fits a small NN, warm-start retrains, saves, loads, and verifies predict_batch output matches. Catches normalizer serialization bugs.

### LOW issues (acknowledged, no plan changes needed):
- **Issue 4 (dual API):** Accepted as-is. `retrain_surrogate()` is for advanced callers who manage merging themselves; `retrain_surrogate_full()` is the batteries-included API. Both are documented.
- **Issue 5 (dependency metadata):** Correct as-is. 4-03 depends on 4-01 and 4-02, not 4-04.
- **Issue 6 (GP no .config):** Noted. GP retraining function already handles this correctly by constructing `GPSurrogateModel(device=device)` directly.
- **Issue 7 (CosineAnnealingWarmRestarts T_0):** Resolved by using existing `fit()` scheduler. See Issue 3 resolution above.
