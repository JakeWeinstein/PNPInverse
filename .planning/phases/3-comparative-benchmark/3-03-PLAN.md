---
phase: 3-comparative-benchmark
plan: 03
type: execute
wave: 1
depends_on: []
files_modified:
  - scripts/studies/k02_stratified_analysis.py
  - StudyResults/surrogate_fidelity/k02_stratified_errors.json
  - StudyResults/surrogate_fidelity/k02_error_vs_value.png
  - StudyResults/surrogate_fidelity/k02_error_heatmap.png
  - StudyResults/surrogate_fidelity/k02_bin_table.csv
autonomous: true
requirements: [BENCH-03]
must_haves:
  truths:
    - "Error metrics are computed per k0_2 log-decade bin for every surrogate model"
    - "Plots show clear error-vs-k0_2 trends on log scale for each model"
    - "Heatmap reveals which model-bin combinations are worst"
    - "PCE Sobol sensitivity context is cross-referenced with error patterns"
    - "GP uncertainty correlation with actual error is quantified per k0_2 bin"
  artifacts:
    - path: "scripts/studies/k02_stratified_analysis.py"
      provides: "Complete k0_2 stratified analysis script"
      min_lines: 200
    - path: "StudyResults/surrogate_fidelity/k02_stratified_errors.json"
      provides: "Per-bin error metrics for all models"
      contains: "k02_bins"
    - path: "StudyResults/surrogate_fidelity/k02_error_vs_value.png"
      provides: "Error vs k0_2 scatter/line plots"
    - path: "StudyResults/surrogate_fidelity/k02_error_heatmap.png"
      provides: "Model x bin heatmap"
    - path: "StudyResults/surrogate_fidelity/k02_bin_table.csv"
      provides: "Tabular per-bin error summary"
  key_links:
    - from: "scripts/studies/k02_stratified_analysis.py"
      to: "StudyResults/surrogate_fidelity/per_sample_errors.csv"
      via: "pandas read_csv"
      pattern: "per_sample_errors\\.csv"
    - from: "scripts/studies/k02_stratified_analysis.py"
      to: "StudyResults/surrogate_fidelity/pce_sobol_indices.json"
      via: "json load"
      pattern: "pce_sobol_indices\\.json"
    - from: "scripts/studies/k02_stratified_analysis.py"
      to: "Surrogate/gp_model.py"
      via: "predict_batch_with_uncertainty for GP UQ correlation"
      pattern: "predict_batch_with_uncertainty"
---

<objective>
Perform a k0_2-stratified error analysis across all surrogate models to understand which models handle the challenging k0_2 parameter best, and where in k0_2-space each model breaks down.

Purpose: k0_2 is the hardest parameter to recover (287% error at 1% noise, near-zero PCE Sobol sensitivity on CD at mid-voltages). Understanding per-model error as a function of k0_2 value identifies which surrogates to carry forward and which k0_2 regions need ISMO augmentation in Phase 4.

Output: Analysis script, stratified error JSON, error-vs-k0_2 plots, model-x-bin heatmap, GP UQ correlation analysis, and cross-reference with PCE Sobol indices.
</objective>

<execution_context>
@/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/.claude/get-shit-done/workflows/execute-plan.md
@/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@StudyResults/surrogate_fidelity/per_sample_errors.csv
@StudyResults/surrogate_fidelity/fidelity_summary.json
@StudyResults/surrogate_fidelity/pce_sobol_indices.json
@StudyResults/training_data_audit/coverage_metrics.json
@Surrogate/validation.py
@Surrogate/gp_model.py
@Surrogate/pce_model.py
@scripts/surrogate/validate_surrogate.py

<interfaces>
<!-- Key data contracts the executor needs -->

per_sample_errors.csv columns:
sample_idx, k0_1, k0_2, alpha_1, alpha_2,
nn_ensemble_cd_nrmse, nn_ensemble_pc_nrmse,
rbf_baseline_cd_nrmse, rbf_baseline_pc_nrmse,
pod_rbf_log_cd_nrmse, pod_rbf_log_pc_nrmse,
pod_rbf_nolog_cd_nrmse, pod_rbf_nolog_pc_nrmse

479 test samples total. k0_2 ranges from ~1e-7 to ~0.1.

coverage_metrics.json k02_per_decade counts:
[1e-7, 1e-6): 268 training, [1e-6, 1e-5): 357, [1e-5, 1e-4): 764,
[1e-4, 1e-3): 772, [1e-3, 1e-2): 714, [1e-2, 1e-1): 319

pce_sobol_indices.json structure:
- parameter_names: [log10_k0_1, log10_k0_2, alpha_1, alpha_2]
- current_density.mean_first_order: [0.634, 0.032, 0.260, 0.019] (k0_2 has only 3.2% CD variance)
- peroxide_current.mean_first_order: [0.459, 0.168, 0.177, 0.170] (k0_2 has 16.8% PC variance)
- first_order arrays are per-voltage-point (22 values each)

GP model API (from Surrogate/gp_model.py):
- predict_batch_with_uncertainty(params) -> dict with 'current_density', 'peroxide_current', 'cd_std', 'pc_std'
- GP model path: data/surrogate_models/gp/ or data/surrogate_models/gp_fixed/

Training/test data: data/surrogate_models/training_data_merged.npz
Split indices: data/surrogate_models/split_indices.npz
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Build k0_2 stratified analysis script with per-bin error computation and GP UQ correlation</name>
  <files>scripts/studies/k02_stratified_analysis.py</files>
  <action>
Create a self-contained analysis script that:

1. **Load data**: Read `per_sample_errors.csv` with pandas. Extract k0_2 column and all model error columns (cd_nrmse and pc_nrmse for each of the 4 models: nn_ensemble, rbf_baseline, pod_rbf_log, pod_rbf_nolog).

2. **Define k0_2 bins**: Use 6 log-decade bins matching the training data audit structure:
   - Bin edges: [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
   - Use `np.digitize` on `np.log10(k0_2)` with edges at [-7, -6, -5, -4, -3, -2, -1]
   - Label bins as "[1e-7,1e-6)", "[1e-6,1e-5)", etc.

3. **Compute per-bin error metrics** for each model and each output (CD, PC):
   - Mean NRMSE per bin
   - Median NRMSE per bin
   - 95th percentile NRMSE per bin
   - Max NRMSE per bin
   - Sample count per bin
   Store in a nested dict: model -> output -> bin -> {mean, median, p95, max, count}

4. **GP uncertainty correlation** (if GP model exists at `data/surrogate_models/gp/` or `data/surrogate_models/gp_fixed/`):
   - Load GP model using `Surrogate.io.load_surrogate` or directly via `GPSurrogateModel.load()`
   - Load test data from `data/surrogate_models/training_data_merged.npz` + `split_indices.npz`
   - Call `predict_batch_with_uncertainty()` on test parameters
   - Compute Spearman rank correlation between GP predicted std and actual NRMSE, per k0_2 bin
   - If GP model not available, skip gracefully with a warning and output null for GP fields

5. **PCE Sobol cross-reference**: Load `pce_sobol_indices.json`, extract the k0_2 first-order Sobol index per voltage point for both CD and PC. Compute mean Sobol index and note that k0_2 contributes only ~3.2% of CD variance but ~16.8% of PC variance. Include these in the output JSON.

6. **Generate outputs**:

   a. **k02_stratified_errors.json**: Complete results dict with:
      - `k02_bins`: list of bin labels
      - `models`: dict of model_name -> {cd: {bin_label: metrics}, pc: {bin_label: metrics}}
      - `gp_uq_correlation`: {per_bin: {bin_label: {cd_spearman_rho, cd_p, pc_spearman_rho, pc_p}}, overall: {cd_rho, pc_rho}}
      - `pce_sobol_k02`: {cd_mean_first_order: 0.032, pc_mean_first_order: 0.168, cd_per_voltage: [...], pc_per_voltage: [...]}
      - `training_coverage`: copy of k02_per_decade from coverage_metrics.json for reference

   b. **k02_error_vs_value.png**: Two subplots (CD top, PC bottom). For each subplot:
      - X-axis: log10(k0_2), range [-7, -1]
      - Y-axis: NRMSE (log scale)
      - Scatter points: individual test samples, colored by model (4 colors, alpha=0.3)
      - Overlay: per-bin median line with error bars (25th-75th percentile) for each model
      - Use distinct colors: nn_ensemble=blue, rbf_baseline=orange, pod_rbf_log=green, pod_rbf_nolog=red
      - Add vertical dashed lines at bin edges
      - Title: "Surrogate Error vs k0_2 Value"
      - Legend outside plot area

   c. **k02_error_heatmap.png**: Two heatmaps side by side (CD, PC):
      - Rows: model names (4 models)
      - Columns: k0_2 bins (6 bins)
      - Cell values: max NRMSE per bin (use max to catch worst cases, per user decision)
      - Color scale: log scale (LogNorm), sequential colormap (viridis or YlOrRd)
      - Annotate cells with the actual max NRMSE value (scientific notation, 1 decimal)
      - Title: "Worst-Case Error by Model and k0_2 Range"

   d. **k02_bin_table.csv**: Flat table with columns:
      model, output, bin, mean_nrmse, median_nrmse, p95_nrmse, max_nrmse, count
      One row per model-output-bin combination (4 models x 2 outputs x 6 bins = 48 rows)

7. **Script interface**: Use argparse with defaults:
   - `--errors-csv` default `StudyResults/surrogate_fidelity/per_sample_errors.csv`
   - `--sobol-json` default `StudyResults/surrogate_fidelity/pce_sobol_indices.json`
   - `--coverage-json` default `StudyResults/training_data_audit/coverage_metrics.json`
   - `--output-dir` default `StudyResults/surrogate_fidelity/`
   - `--gp-model-dir` default `data/surrogate_models/gp_fixed/` (fall back to `data/surrogate_models/gp/`)
   - `--test-data` default `data/surrogate_models/training_data_merged.npz`
   - `--split-indices` default `data/surrogate_models/split_indices.npz`

Use the project's existing pattern: add PNPInverse root to sys.path via `_THIS_DIR`/`_PNPINVERSE_ROOT` (see `scripts/surrogate/validate_surrogate.py` for the pattern). Use matplotlib with `Agg` backend for headless rendering. Save all figures at 150 dpi.
  </action>
  <verify>
    <automated>cd /Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse && python -c "import ast; ast.parse(open('scripts/studies/k02_stratified_analysis.py').read()); print('Syntax OK')"</automated>
  </verify>
  <done>Script parses without errors, contains all 7 components listed above, follows project conventions for path setup and argument parsing.</done>
</task>

<task type="auto">
  <name>Task 2: Run the k0_2 stratified analysis and validate outputs</name>
  <files>
    StudyResults/surrogate_fidelity/k02_stratified_errors.json,
    StudyResults/surrogate_fidelity/k02_error_vs_value.png,
    StudyResults/surrogate_fidelity/k02_error_heatmap.png,
    StudyResults/surrogate_fidelity/k02_bin_table.csv
  </files>
  <action>
1. Activate the venv-firedrake environment (source the activate script in the parent directory's venv-firedrake).

2. Run the analysis script:
   ```
   cd /Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse
   python scripts/studies/k02_stratified_analysis.py
   ```

3. If the script fails due to GP model not being loadable (import errors, missing model files, etc.), that is acceptable -- the script should handle this gracefully and produce all non-GP outputs. Fix any other errors by editing the script.

4. Validate the outputs exist and are non-trivial:
   - `k02_stratified_errors.json` exists and contains `k02_bins`, `models` keys with 4 model entries
   - `k02_bin_table.csv` exists and has 48 data rows (4 models x 2 outputs x 6 bins)
   - `k02_error_vs_value.png` exists and is > 10KB
   - `k02_error_heatmap.png` exists and is > 10KB

5. Print a brief summary of key findings from the JSON:
   - Which model has lowest max NRMSE in the hardest bin (typically [1e-7,1e-6) or [1e-2,1e-1) for PC)
   - Whether the catastrophic PC errors (>100 NRMSE) are concentrated in specific k0_2 bins
   - Whether GP UQ correlation was computed (or skipped)
  </action>
  <verify>
    <automated>cd /Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse && python -c "
import json, os
j = json.load(open('StudyResults/surrogate_fidelity/k02_stratified_errors.json'))
assert 'k02_bins' in j, 'Missing k02_bins'
assert len(j['models']) == 4, f'Expected 4 models, got {len(j[\"models\"])}'
assert os.path.getsize('StudyResults/surrogate_fidelity/k02_error_vs_value.png') > 10000, 'Plot too small'
assert os.path.getsize('StudyResults/surrogate_fidelity/k02_error_heatmap.png') > 10000, 'Heatmap too small'
import csv
with open('StudyResults/surrogate_fidelity/k02_bin_table.csv') as f:
    rows = list(csv.reader(f))
assert len(rows) >= 49, f'Expected 49+ rows (header+48 data), got {len(rows)}'
print('All outputs validated successfully')
print(f'Bins: {j[\"k02_bins\"]}')
print(f'Models: {list(j[\"models\"].keys())}')
print(f'GP UQ: {\"computed\" if j.get(\"gp_uq_correlation\") and j[\"gp_uq_correlation\"].get(\"overall\") else \"skipped\"}')
"
    </automated>
  </verify>
  <done>All 4 output files exist with correct structure. JSON contains per-bin error metrics for 4 models across 6 k0_2 bins. Plots are rendered and non-trivial. CSV has 48 data rows. Script runs end-to-end without crashing.</done>
</task>

</tasks>

<verification>
1. `python scripts/studies/k02_stratified_analysis.py` runs without errors
2. `StudyResults/surrogate_fidelity/k02_stratified_errors.json` contains complete per-bin metrics
3. Both PNG files render properly (non-empty, >10KB)
4. CSV table has correct dimensions (48 data rows)
5. PCE Sobol cross-reference data is included in the JSON output
</verification>

<success_criteria>
- k0_2 stratified error metrics computed for all 4 existing surrogate models across 6 log-decade bins
- Error-vs-k0_2 plot shows per-model trends with scatter + binned summary
- Heatmap shows worst-case (max) NRMSE per model-bin combination
- PCE Sobol indices for k0_2 are cross-referenced in the output JSON
- GP uncertainty correlation is computed if GP model is available, gracefully skipped if not
- All outputs saved to StudyResults/surrogate_fidelity/
</success_criteria>

<output>
After completion, create `.planning/phases/3-comparative-benchmark/3-03-SUMMARY.md`
</output>
