# Phase 3: Surrogate Fidelity - Research

**Researched:** 2026-03-07
**Domain:** Surrogate model validation / error characterization (numpy, pytest, matplotlib)
**Confidence:** HIGH

## Summary

Phase 3 characterizes the approximation error of all 4 v13-era surrogate models (NN ensemble, RBF baseline, POD-RBF log, POD-RBF nolog) against PDE ground truth on held-out test data. The existing codebase provides nearly all the infrastructure needed: `validate_surrogate()` already computes per-sample NRMSE, the training data and split indices are on disk, and all 4 model types share the same `predict_batch()` API returning `{"current_density": (N,n_eta), "peroxide_current": (N,n_eta), "phi_applied": (n_eta,)}`.

The implementation is a single pytest file that loads the 4 models and the hold-out split, runs `validate_surrogate()` (or equivalent vectorized NRMSE logic) per model, saves JSON + CSV artifacts to `StudyResults/surrogate_fidelity/`, generates diagnostic plots, and asserts the soft gate (mean NRMSE < 20%). No PDE solves are needed -- the ground truth is precomputed in `training_data_merged.npz`.

**Primary recommendation:** Build one test file `tests/test_surrogate_fidelity.py` with a module-scoped fixture loading all 4 models and the hold-out data once, parametrize tests over model names, and emit all artifacts in the test body (following the Phase 2 MMS convergence pattern).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Full treatment (SUR-01/02/03) for ALL 4 surrogate models: NN ensemble (D3-deeper), RBF baseline, POD-RBF log, POD-RBF nolog
- All 4 models get: LHS fidelity map, hold-out validation, error stats (max/mean/95th NRMSE)
- Same hold-out test points shared across all models (PDE ground truth computed once)
- Subsumes existing `TestSurrogateVsPDEConsistency` (Test 5) from `test_v13_verification.py` -- remove that test to avoid redundancy
- Use existing hold-out split from `StudyResults/surrogate_v11/split_indices.npz` -- no fresh PDE solves needed
- Use whatever hold-out size the existing split provides (expected ~10% of 3000 = ~300 samples)
- Test on the full union voltage grid (all voltage points the training data was generated on)
- Save per-sample parameters + errors so the fidelity map shows error as a function of parameter space location
- Diagnostic with soft gates: compute all error stats, save to JSON, assert only on catastrophic failure
- Soft gate: mean NRMSE < 20% per model -- same threshold for all 4 model types
- Use NRMSE only (normalized by per-sample range), not pointwise relative error
- Required statistics per model per output (CD and PC): max NRMSE, mean NRMSE, 95th percentile NRMSE
- Results directory: `StudyResults/surrogate_fidelity/`
- JSON summary file with aggregate stats (max/mean/95th NRMSE per model per output)
- CSV per-sample file with parameters (k0_1, k0_2, alpha_1, alpha_2) and errors for each model
- Plots: worst-case I-V overlay (top 3 worst NRMSE samples per model), error vs parameter scatter (NRMSE vs each of 4 parameters, separate for CD and PC)
- CD and PC errors plotted separately

### Claude's Discretion
- Exact structure of the test file and fixture organization
- How to load and interface with all 4 model types uniformly (they have different APIs)
- Plot styling, layout, and figure sizing
- Whether to use a single test class or separate classes per model
- How to handle the split_indices.npz loading and validation

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| SUR-01 | Surrogate fidelity map using LHS-sampled parameter sets across the v13 inference domain | Hold-out set from `split_indices.npz` already provides LHS-sampled test points across the domain; per-sample CSV with parameter coordinates enables parameter-space error mapping |
| SUR-02 | Hold-out validation testing the v13 surrogate on unseen parameter sets (not training data) | `split_indices.npz` contains pre-split train/test indices; test indices select ~300 unseen samples from `training_data_merged.npz` |
| SUR-03 | Error bound quantification (max, mean, percentile errors) for the v13 surrogate | `validate_surrogate()` already computes per-sample NRMSE; aggregate stats (max, mean, 95th percentile) computed from these arrays |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | (project version) | Array operations, NRMSE computation | Already used throughout; per-sample NRMSE is pure numpy |
| pytest | (project version) | Test framework | Project convention; `@pytest.mark.slow` for expensive tests |
| matplotlib | (project version) | Diagnostic plots (I-V overlays, scatter) | Already available; used in Phase 2 for convergence plots |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| json (stdlib) | n/a | Save aggregate stats JSON | Matches Phase 2 `convergence_data.json` pattern |
| csv (stdlib) | n/a | Save per-sample error CSV | Matches `validate_surrogate.py` CSV output pattern |
| datetime (stdlib) | n/a | Timestamp in JSON metadata | Convention from Phase 2 |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Manual CSV writing | pandas.to_csv | Adds unnecessary dependency; manual CSV is 5 lines for this simple format |
| validate_surrogate() | Custom NRMSE loop | validate_surrogate() returns per-sample NRMSE arrays; reuse it directly for RBF/POD-RBF models via their predict_batch() API |

## Architecture Patterns

### Recommended Project Structure
```
tests/
  test_surrogate_fidelity.py    # Single test file for all SUR-01/02/03
StudyResults/
  surrogate_fidelity/
    fidelity_summary.json       # Aggregate stats per model per output
    per_sample_errors.csv       # All models, all samples, all params + errors
    worst_iv_overlay_{model}.png   # Top-3 worst NRMSE I-V overlay per model
    error_vs_params_cd_{model}.png # NRMSE vs 4 params for CD per model
    error_vs_params_pc_{model}.png # NRMSE vs 4 params for PC per model
```

### Pattern 1: Uniform Model Interface via predict_batch()
**What:** All 4 model types (BVSurrogateModel, PODRBFSurrogateModel, EnsembleMeanWrapper) share the same `predict_batch(params) -> {"current_density": (N,n_eta), "peroxide_current": (N,n_eta)}` API.
**When to use:** Always -- this is how to treat all 4 models uniformly.
**Example:**
```python
# All model types support this identical call signature
pred = model.predict_batch(test_parameters)  # (N, 4) -> dict
cd_pred = pred["current_density"]  # (N, n_eta)
pc_pred = pred["peroxide_current"]  # (N, n_eta)
```

### Pattern 2: Module-Scoped Fixture for Expensive Loading
**What:** Load all 4 models + training data + split indices once per module.
**When to use:** Models are expensive to load (especially NN ensemble with 5 PyTorch members). Module scope avoids redundant loading.
**Example:**
```python
# Source: tests/test_v13_verification.py existing pattern
@pytest.fixture(scope="module")
def all_models():
    """Load all 4 surrogate models once for the module."""
    models = {}
    models["nn_ensemble"] = load_nn_ensemble(ENSEMBLE_DIR, n_members=5, device="cpu")
    models["rbf_baseline"] = load_surrogate(RBF_PATH)
    models["pod_rbf_log"] = load_surrogate(POD_RBF_LOG_PATH)
    models["pod_rbf_nolog"] = load_surrogate(POD_RBF_NOLOG_PATH)
    return models
```

### Pattern 3: Parametrized Tests Over Model Names
**What:** Use `@pytest.mark.parametrize` to run the same error-stat/assertion logic for each model.
**When to use:** When all 4 models should pass the same soft gate (mean NRMSE < 20%).
**Example:**
```python
MODEL_NAMES = ["nn_ensemble", "rbf_baseline", "pod_rbf_log", "pod_rbf_nolog"]

@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_mean_nrmse_below_threshold(self, all_models, holdout_data, model_name):
    model = all_models[model_name]
    # ... compute NRMSE, assert mean < 0.20
```

### Pattern 4: Artifact Emission During Test (Phase 2 Convention)
**What:** Tests emit JSON/CSV/PNG artifacts to `StudyResults/` as side effects.
**When to use:** For all V&V phases. Phase 6 reads these artifacts for report generation.
**Example:**
```python
# Source: tests/test_mms_convergence.py existing pattern
output_dir = os.path.join(ROOT, "StudyResults", "surrogate_fidelity")
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, "fidelity_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
```

### Anti-Patterns to Avoid
- **Loading models per-test:** Each NN ensemble load pulls 5 PyTorch models from disk. Module-scoped fixture is mandatory.
- **Recomputing PDE solutions:** All PDE ground truth is in `training_data_merged.npz`. Never import Firedrake or run the forward solver in this phase.
- **Using pointwise relative error:** Division by near-zero at small currents. Use NRMSE (normalized by per-sample range) per the locked decision.
- **Separate test files per model:** Redundant structure. One file with parametrized tests is cleaner.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Per-sample NRMSE computation | Custom NRMSE loop | `Surrogate.validation.validate_surrogate()` | Already returns `cd_nrmse_per_sample` and `pc_nrmse_per_sample` arrays; handles edge cases (zero range protection) |
| Model loading | Custom pickle/torch loading | `load_surrogate()` for RBF/POD-RBF, `load_nn_ensemble()` for NN | Handles backward compatibility (missing attributes), path validation |
| LHS sampling | Custom sampling code | `Surrogate.sampling.generate_lhs_samples()` | Not needed -- using existing hold-out split, not generating new samples |

**Key insight:** `validate_surrogate()` is the core metric engine. It accepts any object with `predict_batch()` returning the standard dict. All 4 model types have this API. The function returns all per-sample arrays needed for CSV output, scatter plots, and aggregate stats.

## Common Pitfalls

### Pitfall 1: NN Ensemble Returns Different Voltage Grid Than RBF Models
**What goes wrong:** The NN ensemble's `phi_applied` grid may differ from the voltage grid in `training_data_merged.npz` used by RBF models.
**Why it happens:** The NN ensemble stores its own `phi_applied` from training. The RBF models store theirs from their separate training pipeline.
**How to avoid:** After loading, verify all models share the same `n_eta` and that their `phi_applied` grids match the grid in the training data. If grids differ, use the training data grid as canonical and only compare at matching voltage points.
**Warning signs:** Shape mismatch errors in `predict_batch()` output vs ground truth arrays.

### Pitfall 2: split_indices.npz Index Interpretation
**What goes wrong:** Indices in `split_indices.npz` may reference `training_data_merged.npz` or `training_data_new_3000.npz` differently.
**Why it happens:** Two training data files exist on disk. The split was likely computed for the merged data.
**How to avoid:** Load `training_data_merged.npz`, extract test indices from `split_indices.npz`, verify the indices are in range for the merged data shape. Assert `max(test_idx) < N_total`.
**Warning signs:** IndexError or unexpectedly small hold-out set.

### Pitfall 3: validate_surrogate() Requires BVSurrogateModel Type
**What goes wrong:** `validate_surrogate()` has a type hint for `BVSurrogateModel` but the NN ensemble is `EnsembleMeanWrapper`.
**Why it happens:** The type hint is structural, not enforced at runtime. Both types have `predict_batch()`.
**How to avoid:** Pass the ensemble directly -- Python duck typing works. If it fails, extract the NRMSE logic into a standalone function that takes `pred_cd, pred_pc, true_cd, true_pc` arrays.
**Warning signs:** TypeError from explicit isinstance check (unlikely based on code review -- no isinstance check in validate_surrogate).

### Pitfall 4: NRMSE Division by Zero for Flat I-V Curves
**What goes wrong:** `np.ptp(test_cd[i])` returns 0 for a flat I-V curve, causing 0/0 in NRMSE.
**Why it happens:** Some parameter combinations produce nearly constant current over the voltage range.
**How to avoid:** `validate_surrogate()` already guards this with `if cd_range > 1e-12`. Samples with flat curves will have NRMSE = 0 (numerator is also small). This is correct behavior -- do not treat as an error.
**Warning signs:** Unexpectedly many samples with NRMSE = 0.0.

### Pitfall 5: Matplotlib Backend in Headless Pytest
**What goes wrong:** `plt.show()` or missing backend causes errors when running pytest in CI/headless mode.
**Why it happens:** No display server available.
**How to avoid:** Use `matplotlib.use("Agg")` at the top of the test file (before any pyplot import). Only use `savefig()`, never `show()`. Close figures with `plt.close()` to avoid memory leaks.
**Warning signs:** RuntimeError about display or Tcl/Tk.

### Pitfall 6: Removing Test 5 from test_v13_verification.py
**What goes wrong:** Forgetting to remove the subsumed `TestSurrogateVsPDEConsistency` class, or removing it and breaking other tests that depend on its fixtures.
**Why it happens:** The class is standalone (no other tests depend on it), but must verify no imports reference it.
**How to avoid:** Remove the entire `TestSurrogateVsPDEConsistency` class and its associated comment block. Verify no other test file imports from it. The nn_ensemble fixture it uses is shared but still needed by Tests 1, 3, 6, 7.

## Code Examples

### Loading All 4 Models Uniformly
```python
# Source: Surrogate/io.py, Surrogate/ensemble.py (verified from codebase)
import os
from Surrogate.io import load_surrogate
from Surrogate.ensemble import load_nn_ensemble

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_V11_DIR = os.path.join(_ROOT, "StudyResults", "surrogate_v11")

models = {
    "nn_ensemble": load_nn_ensemble(
        os.path.join(_V11_DIR, "nn_ensemble", "D3-deeper"),
        n_members=5, device="cpu",
    ),
    "rbf_baseline": load_surrogate(os.path.join(_V11_DIR, "model_rbf_baseline.pkl")),
    "pod_rbf_log": load_surrogate(os.path.join(_V11_DIR, "model_pod_rbf_log.pkl")),
    "pod_rbf_nolog": load_surrogate(os.path.join(_V11_DIR, "model_pod_rbf_nolog.pkl")),
}
```

### Loading Hold-Out Data
```python
# Source: StudyResults/surrogate_v11/ file listing (verified on disk)
import numpy as np

data = np.load(os.path.join(_V11_DIR, "training_data_merged.npz"))
split = np.load(os.path.join(_V11_DIR, "split_indices.npz"))

# Expected keys in training_data_merged.npz: parameters, current_density,
# peroxide_current, phi_applied (based on validate_surrogate.py usage pattern)
all_params = data["parameters"]       # (3000, 4)
all_cd = data["current_density"]      # (3000, n_eta)
all_pc = data["peroxide_current"]     # (3000, n_eta)
phi_applied = data["phi_applied"]     # (n_eta,)

# Expected keys in split_indices.npz: train_idx, test_idx (or similar)
test_idx = split["test_idx"]  # verify key name at runtime
test_params = all_params[test_idx]
test_cd = all_cd[test_idx]
test_pc = all_pc[test_idx]
```

### Computing NRMSE via validate_surrogate()
```python
# Source: Surrogate/validation.py (verified from codebase)
from Surrogate.validation import validate_surrogate

# Works for any model with predict_batch() API (duck typing)
metrics = validate_surrogate(model, test_params, test_cd, test_pc)

cd_nrmse = metrics["cd_nrmse_per_sample"]  # (N_test,)
pc_nrmse = metrics["pc_nrmse_per_sample"]  # (N_test,)

# Aggregate stats
stats = {
    "cd_max_nrmse": float(np.max(cd_nrmse)),
    "cd_mean_nrmse": float(np.mean(cd_nrmse)),
    "cd_95th_nrmse": float(np.percentile(cd_nrmse, 95)),
    "pc_max_nrmse": float(np.max(pc_nrmse)),
    "pc_mean_nrmse": float(np.mean(pc_nrmse)),
    "pc_95th_nrmse": float(np.percentile(pc_nrmse, 95)),
}
```

### Worst-Case I-V Overlay Plot
```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Top 3 worst NRMSE samples for this model
worst_3 = np.argsort(cd_nrmse)[-3:][::-1]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, idx in zip(axes, worst_3):
    pred = model.predict_batch(test_params[idx:idx+1])
    ax.plot(phi_applied, test_cd[idx], "k-", label="PDE truth")
    ax.plot(phi_applied, pred["current_density"][0], "r--", label="Surrogate")
    ax.set_title(f"Sample {idx}, NRMSE={cd_nrmse[idx]*100:.1f}%")
    ax.set_xlabel("phi_applied")
    ax.set_ylabel("CD")
    ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(output_dir, f"worst_iv_overlay_{model_name}.png"), dpi=150)
plt.close(fig)
```

### Error vs Parameter Scatter Plot
```python
param_names = ["k0_1", "k0_2", "alpha_1", "alpha_2"]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, j, name in zip(axes, range(4), param_names):
    ax.scatter(test_params[:, j], cd_nrmse * 100, s=8, alpha=0.5)
    ax.set_xlabel(name)
    ax.set_ylabel("CD NRMSE (%)")
    if j < 2:
        ax.set_xscale("log")  # k0 spans orders of magnitude
fig.suptitle(f"{model_name} -- CD Error vs Parameters")
fig.tight_layout()
fig.savefig(os.path.join(output_dir, f"error_vs_params_cd_{model_name}.png"), dpi=150)
plt.close(fig)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Test 5 in test_v13_verification.py (3 param sets, PDE comparison, NN only) | Full fidelity map across all 4 models on hold-out data (~300 samples) | Phase 3 | Much more comprehensive; no PDE solves needed; covers all model types |
| Per-sample RMSE only | Per-sample NRMSE (range-normalized) | Phase 3 decision | Avoids scale dependence; comparable across CD and PC outputs |

**Deprecated/outdated:**
- `TestSurrogateVsPDEConsistency` in `test_v13_verification.py`: Will be removed (subsumed by this phase). Only tested NN ensemble at 3 parameter sets on 5 voltage points with live PDE solves.

## Open Questions

1. **Exact keys in split_indices.npz**
   - What we know: File exists at `StudyResults/surrogate_v11/split_indices.npz`
   - What's unclear: Exact key names (likely `train_idx`/`test_idx` or `train`/`test`, could be `train_indices`/`test_indices`)
   - Recommendation: Load and inspect at fixture time; print keys if unexpected. Use defensive key lookup with fallback.

2. **Whether RBF/POD-RBF models share the same phi_applied grid as NN ensemble**
   - What we know: All were trained on `training_data_merged.npz` which has one phi_applied grid
   - What's unclear: Whether the pickle files stored the same grid or a subset
   - Recommendation: Assert `model.n_eta == phi_applied.shape[0]` for each model. If mismatch, fail with informative error.

3. **Exact hold-out size**
   - What we know: Expected ~300 samples (10% of 3000)
   - What's unclear: Actual split ratio used during training
   - Recommendation: Print hold-out size in test output; assert `len(test_idx) >= 50` as sanity check.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (project standard) |
| Config file | existing project pytest config |
| Quick run command | `pytest tests/test_surrogate_fidelity.py -x -v` |
| Full suite command | `pytest tests/test_surrogate_fidelity.py -m slow -v` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| SUR-01 | Fidelity map: per-sample NRMSE at LHS-sampled hold-out points, CSV with param coords | integration | `pytest tests/test_surrogate_fidelity.py::TestSurrogateFidelity::test_fidelity_artifacts_generated -x` | No -- Wave 0 |
| SUR-02 | Hold-out validation: test on unseen param sets from split_indices.npz | integration | `pytest tests/test_surrogate_fidelity.py::TestSurrogateFidelity::test_holdout_mean_nrmse_below_threshold -x` | No -- Wave 0 |
| SUR-03 | Error stats: max/mean/95th NRMSE computed and saved as JSON | integration | `pytest tests/test_surrogate_fidelity.py::TestSurrogateFidelity::test_error_stats_saved_to_json -x` | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_surrogate_fidelity.py -x -v`
- **Per wave merge:** `pytest tests/ -x -v -m slow`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_surrogate_fidelity.py` -- covers SUR-01, SUR-02, SUR-03
- [ ] `StudyResults/surrogate_fidelity/` directory -- created at test runtime

*(No framework install needed -- pytest and numpy already in the environment.)*

## Sources

### Primary (HIGH confidence)
- `Surrogate/validation.py` -- verified source code: `validate_surrogate()` returns per-sample NRMSE arrays with zero-range protection
- `Surrogate/ensemble.py` -- verified source code: `EnsembleMeanWrapper.predict_batch()` API identical to `BVSurrogateModel`
- `Surrogate/surrogate_model.py` -- verified source code: `BVSurrogateModel.predict_batch()` returns `{"current_density", "peroxide_current", "phi_applied"}`
- `Surrogate/pod_rbf_model.py` -- verified source code: same `predict_batch()` API
- `Surrogate/io.py` -- verified source code: `load_surrogate()` loads any `.pkl` BVSurrogateModel (including POD-RBF via pickle)
- `StudyResults/surrogate_v11/` -- verified directory listing: all 4 model files present
- `scripts/surrogate/validate_surrogate.py` -- verified CSV output pattern for per-sample errors
- `tests/test_mms_convergence.py` + `StudyResults/mms_convergence/convergence_data.json` -- verified Phase 2 artifact pattern

### Secondary (MEDIUM confidence)
- `split_indices.npz` key names -- file exists but exact keys not inspected due to environment constraints (no numpy in shell). LOW risk; defensive loading will handle.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already in use in the project
- Architecture: HIGH -- all model APIs verified from source; predict_batch() is uniform across types
- Pitfalls: HIGH -- code review confirms duck typing works, zero-range guard exists, matplotlib Agg pattern established

**Research date:** 2026-03-07
**Valid until:** 2026-04-07 (stable domain; no external library dependencies changing)
