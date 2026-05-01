---
phase: 3-comparative-benchmark
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - tests/test_surrogate_fidelity.py
  - StudyResults/surrogate_fidelity/fidelity_summary.json
  - StudyResults/surrogate_fidelity/per_sample_errors.csv
  - StudyResults/surrogate_fidelity/worst_iv_overlay_gp_fixed.png
  - StudyResults/surrogate_fidelity/worst_iv_overlay_pce.png
  - StudyResults/surrogate_fidelity/error_vs_params_cd_gp_fixed.png
  - StudyResults/surrogate_fidelity/error_vs_params_pc_gp_fixed.png
  - StudyResults/surrogate_fidelity/error_vs_params_cd_pce.png
  - StudyResults/surrogate_fidelity/error_vs_params_pc_pce.png
autonomous: true
requirements: []

must_haves:
  truths:
    - "All 6 surrogate models (nn_ensemble, rbf_baseline, pod_rbf_log, pod_rbf_nolog, gp_fixed, pce) are evaluated on the same 479 hold-out test samples"
    - "Error metrics (max/mean/median/95th NRMSE for CD and PC) are computed identically for all models"
    - "Per-sample errors CSV contains columns for all 6 models"
    - "fidelity_summary.json contains entries for all 6 models"
    - "Error-vs-parameter scatter plots and worst-case I-V overlays are generated for GP and PCE models"
  artifacts:
    - path: "tests/test_surrogate_fidelity.py"
      provides: "Extended benchmark test covering all 6 models"
      contains: "gp_fixed"
    - path: "StudyResults/surrogate_fidelity/fidelity_summary.json"
      provides: "Aggregate error statistics for all 6 models"
      contains: "gp_fixed"
    - path: "StudyResults/surrogate_fidelity/per_sample_errors.csv"
      provides: "Per-sample NRMSE for all 6 models"
  key_links:
    - from: "tests/test_surrogate_fidelity.py"
      to: "Surrogate/gp_model.py"
      via: "load_gp_surrogate()"
      pattern: "load_gp_surrogate"
    - from: "tests/test_surrogate_fidelity.py"
      to: "Surrogate/pce_model.py"
      via: "PCESurrogateModel.load()"
      pattern: "PCESurrogateModel\\.load"
    - from: "tests/test_surrogate_fidelity.py"
      to: "Surrogate/validation.py"
      via: "validate_surrogate()"
      pattern: "validate_surrogate"
---

<objective>
Extend the existing surrogate fidelity benchmark (tests/test_surrogate_fidelity.py) to include GP and PCE models alongside the 4 existing models (nn_ensemble, rbf_baseline, pod_rbf_log, pod_rbf_nolog), producing a standardized head-to-head prediction accuracy comparison on identical held-out test data.

Purpose: Phase 3 of the surrogate pipeline roadmap requires a comparative benchmark of ALL available surrogates. The existing test infrastructure already handles 4 models with proper hold-out data loading, metric computation, plot generation, and artifact saving. Extending it to 6 models is the minimal, clean approach.

Output: Updated fidelity_summary.json (6 models), extended per_sample_errors.csv, new overlay/scatter plots for GP and PCE, and passing benchmark tests.
</objective>

<execution_context>
@/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/.claude/get-shit-done/workflows/execute-plan.md
@/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@tests/test_surrogate_fidelity.py
@Surrogate/validation.py
@Surrogate/__init__.py
@Surrogate/gp_model.py (lines 1-15 for API, load_gp_surrogate at line 933)
@Surrogate/pce_model.py (lines 1-18 for API, PCESurrogateModel.load at line 814)
@Surrogate/ensemble.py (lines 1-60 for EnsembleMeanWrapper API)
@StudyResults/surrogate_fidelity/fidelity_summary.json

<interfaces>
<!-- Key loading APIs the executor needs -->

From Surrogate/gp_model.py:
```python
def load_gp_surrogate(path: str, device: str = "cpu") -> GPSurrogateModel:
    # path is directory like "data/surrogate_models/gp_fixed/"
    # Returns GPSurrogateModel with predict_batch(parameters) -> dict
```

From Surrogate/pce_model.py:
```python
class PCESurrogateModel:
    @staticmethod
    def load(path: str) -> "PCESurrogateModel":
        # path is pickle file like "data/surrogate_models/pce/pce_model.pkl"
        # Returns PCESurrogateModel with predict_batch(parameters) -> dict

    def predict_batch(self, parameters: np.ndarray) -> Dict[str, np.ndarray]:
        # Returns {"current_density": ..., "peroxide_current": ..., "phi_applied": ...}
```

From Surrogate/validation.py:
```python
def validate_surrogate(
    surrogate,           # any object with predict_batch()
    test_parameters,     # (N, 4) array
    test_cd,             # (N, n_eta) array
    test_pc,             # (N, n_eta) array
) -> dict:
    # Returns dict with cd_rmse, pc_rmse, cd_per_sample_rmse, pc_per_sample_rmse,
    # cd_nrmse_per_sample, pc_nrmse_per_sample, cd_mean_relative_error, etc.
```

Existing model paths in test_surrogate_fidelity.py:
```python
_ENSEMBLE_DIR = os.path.join(_SURROGATE_DIR, "nn_ensemble", "D3-deeper")
_RBF_BASELINE_PATH = os.path.join(_SURROGATE_DIR, "model_rbf_baseline.pkl")
_POD_RBF_LOG_PATH = os.path.join(_SURROGATE_DIR, "model_pod_rbf_log.pkl")
_POD_RBF_NOLOG_PATH = os.path.join(_SURROGATE_DIR, "model_pod_rbf_nolog.pkl")
MODEL_NAMES = ["nn_ensemble", "rbf_baseline", "pod_rbf_log", "pod_rbf_nolog"]
```

New model paths to add:
```python
_GP_FIXED_DIR = os.path.join(_SURROGATE_DIR, "gp_fixed")    # 44 .pt files + metadata
_PCE_PATH = os.path.join(_SURROGATE_DIR, "pce", "pce_model.pkl")
```
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Extend test_surrogate_fidelity.py to include GP and PCE models</name>
  <files>tests/test_surrogate_fidelity.py</files>
  <action>
Modify `tests/test_surrogate_fidelity.py` to add GP (gp_fixed) and PCE models to the benchmark. The changes are surgical -- the existing infrastructure (holdout_data fixture, validate_surrogate calls, JSON/CSV artifact generation, plot generation) already works. We just need to extend MODEL_NAMES and the model-loading logic.

Specific changes:

1. **Add imports** at top of file:
   ```python
   from Surrogate.gp_model import load_gp_surrogate
   from Surrogate.pce_model import PCESurrogateModel
   ```

2. **Add path constants** after the existing ones:
   ```python
   _GP_FIXED_DIR = os.path.join(_SURROGATE_DIR, "gp_fixed")
   _PCE_PATH = os.path.join(_SURROGATE_DIR, "pce", "pce_model.pkl")
   ```

3. **Extend MODEL_NAMES** to include the new models:
   ```python
   MODEL_NAMES = ["nn_ensemble", "rbf_baseline", "pod_rbf_log", "pod_rbf_nolog", "gp_fixed", "pce"]
   ```

4. **Update `all_models` fixture** to load GP and PCE:
   ```python
   models["gp_fixed"] = load_gp_surrogate(_GP_FIXED_DIR, device="cpu")
   models["pce"] = PCESurrogateModel.load(_PCE_PATH)
   ```
   Add these lines after the existing POD-RBF loading. Use try/except ImportError for each in case gpytorch or chaospy is not installed in the test environment, and pytest.skip if missing.

5. **No changes needed** to `all_metrics`, `fidelity_artifacts`, `generate_plots`, or test methods -- they all iterate over MODEL_NAMES and will automatically pick up the new models.

6. **Guard against missing model files**: Wrap GP and PCE loading in checks for file existence. If `_GP_FIXED_DIR` or `_PCE_PATH` doesn't exist, log a warning and skip that model (remove from MODEL_NAMES for the test session). Use a module-level helper that filters MODEL_NAMES to only models whose artifacts exist on disk.

Important considerations:
- The `validate_surrogate()` function accepts any object with a `predict_batch()` method. Both GPSurrogateModel and PCESurrogateModel implement this API, so no validation code changes are needed.
- The GP model loading requires torch and gpytorch. If not available, the import will fail. Handle with try/except and pytest.skip.
- The PCE model loading requires chaospy. Same handling.
- Keep the original 4 models listed first in MODEL_NAMES so existing CSV column ordering is preserved for backward compatibility (new columns appended at end).
  </action>
  <verify>
    <automated>cd /Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse && source ../venv-firedrake/bin/activate && python -c "from Surrogate.gp_model import load_gp_surrogate; from Surrogate.pce_model import PCESurrogateModel; print('Imports OK')" && python -m pytest tests/test_surrogate_fidelity.py -x -v --timeout=300 2>&1 | tail -40</automated>
  </verify>
  <done>
    - MODEL_NAMES contains all 6 models: nn_ensemble, rbf_baseline, pod_rbf_log, pod_rbf_nolog, gp_fixed, pce
    - all_models fixture successfully loads GP from gp_fixed/ and PCE from pce/pce_model.pkl
    - All test_holdout_median_nrmse_below_threshold parametrized tests pass for all 6 models
    - test_error_stats_saved_to_json passes with 6-model summary
    - test_fidelity_csv_has_all_samples passes with 12 error columns (6 models x CD/PC)
    - test_worst_iv_overlay_plots_generated passes (6 PNG files)
    - test_error_scatter_plots_generated passes (12 PNG files)
  </done>
</task>

<task type="auto">
  <name>Task 2: Run benchmark and verify artifacts</name>
  <files>
    StudyResults/surrogate_fidelity/fidelity_summary.json,
    StudyResults/surrogate_fidelity/per_sample_errors.csv,
    StudyResults/surrogate_fidelity/worst_iv_overlay_gp_fixed.png,
    StudyResults/surrogate_fidelity/worst_iv_overlay_pce.png,
    StudyResults/surrogate_fidelity/error_vs_params_cd_gp_fixed.png,
    StudyResults/surrogate_fidelity/error_vs_params_pc_gp_fixed.png,
    StudyResults/surrogate_fidelity/error_vs_params_cd_pce.png,
    StudyResults/surrogate_fidelity/error_vs_params_pc_pce.png
  </files>
  <action>
Run the full benchmark test suite to generate all artifacts, then validate the outputs:

1. **Execute the benchmark:**
   ```bash
   cd /Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse
   source ../venv-firedrake/bin/activate
   python -m pytest tests/test_surrogate_fidelity.py -v --timeout=600 -s
   ```
   The `-s` flag shows print output so we can see per-model metric summaries during execution.

2. **Verify fidelity_summary.json** contains all 6 models with the 8 expected statistics each (cd_max_nrmse, cd_mean_nrmse, cd_median_nrmse, cd_95th_nrmse, pc_max_nrmse, pc_mean_nrmse, pc_median_nrmse, pc_95th_nrmse).

3. **Verify per_sample_errors.csv** has columns for all 6 models (sample_idx, k0_1, k0_2, alpha_1, alpha_2, plus 12 error columns: {model}_cd_nrmse, {model}_pc_nrmse for each model).

4. **Verify all plot files exist** and are non-empty:
   - 6 worst-case overlay PNGs (one per model)
   - 12 error-vs-parameter scatter PNGs (2 per model: CD and PC)

5. **Print a ranking summary** by reading the JSON and sorting models by cd_median_nrmse and pc_median_nrmse to identify which surrogates perform best. This informs the Phase 3 selection decision.

6. **Spot-check GP and PCE results** for reasonableness:
   - GP cd_median_nrmse should be in the ballpark of the NN ensemble (0.25% median)
   - PCE may be worse but should not be catastrophically bad (median < 20%)
   - If any model shows median NRMSE > 20%, the test gate will flag it but the artifacts should still be generated
  </action>
  <verify>
    <automated>cd /Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse && python -c "
import json, os
with open('StudyResults/surrogate_fidelity/fidelity_summary.json') as f:
    d = json.load(f)
expected = ['nn_ensemble','rbf_baseline','pod_rbf_log','pod_rbf_nolog','gp_fixed','pce']
for m in expected:
    assert m in d['models'], f'Missing model: {m}'
    stats = d['models'][m]
    assert len(stats) == 8, f'{m} has {len(stats)} stats, expected 8'
    print(f'{m}: CD median={stats[\"cd_median_nrmse\"]*100:.2f}%, PC median={stats[\"pc_median_nrmse\"]*100:.2f}%')
print('All 6 models present with 8 stats each -- PASS')

# Check plots exist
plots_dir = 'StudyResults/surrogate_fidelity'
for m in expected:
    assert os.path.isfile(f'{plots_dir}/worst_iv_overlay_{m}.png'), f'Missing overlay: {m}'
    for t in ('cd','pc'):
        assert os.path.isfile(f'{plots_dir}/error_vs_params_{t}_{m}.png'), f'Missing scatter: {t}_{m}'
print('All 18 plot files exist -- PASS')
"</automated>
  </verify>
  <done>
    - fidelity_summary.json contains entries for all 6 models with 8 statistics each
    - per_sample_errors.csv has 479 rows and 17 columns (5 param + 12 error)
    - 6 worst-case I-V overlay PNGs exist and are non-empty
    - 12 error-vs-parameter scatter PNGs exist and are non-empty
    - Model ranking by cd_median_nrmse is printed and readable
    - GP and PCE results are reasonable (no catastrophic failures unless model itself is poor)
  </done>
</task>

</tasks>

<verification>
1. `python -m pytest tests/test_surrogate_fidelity.py -v` -- all tests pass with 6 models
2. `fidelity_summary.json` contains 6 model entries with 8 stats each
3. `per_sample_errors.csv` has 479 data rows and columns for all 6 models
4. 18 total plot files exist in `StudyResults/surrogate_fidelity/`
5. No regressions: the 4 original models produce identical metrics to before (values match within float tolerance since same data + same code)
</verification>

<success_criteria>
- Head-to-head comparison of all 6 available surrogates on identical 479-sample hold-out set
- Standardized metrics (max/mean/median/95th NRMSE) computed with same normalization for all models
- Per-sample error CSV enables downstream k0_2-stratified analysis
- Plots enable visual diagnosis of where each model fails
- Ranking identifies which surrogates to carry forward to Phase 4 (ISMO) and Phase 5 (PDE refinement)
</success_criteria>

<output>
After completion, create `.planning/phases/3-comparative-benchmark/3-01-SUMMARY.md`
</output>
