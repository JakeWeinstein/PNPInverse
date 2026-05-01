---
phase: 3-comparative-benchmark
plan: 05
type: execute
wave: 1
depends_on: []
files_modified:
  - scripts/studies/surrogate_ranking_report.py
  - StudyResults/surrogate_fidelity/fidelity_summary.json
  - StudyResults/surrogate_fidelity/ranking_report.json
  - writeups/vv_report/tables/surrogate_fidelity.tex
  - writeups/vv_report/figures/surrogate_comparison.pdf
  - writeups/vv_report/generate_figures.py
autonomous: true
requirements: [BENCH-05]
must_haves:
  truths:
    - "All 6 surrogate models ranked across 5 dimensions with weighted composite score"
    - "1-2 surrogates selected for Phase 4 (ISMO) and Phase 5 (PDE refinement) with documented rationale"
    - "Go/no-go decision: at least one model beats current 10.7% worst-case error"
    - "V&V report tables and figures updated with all model results"
  artifacts:
    - path: "scripts/studies/surrogate_ranking_report.py"
      provides: "Automated ranking and report generation script"
      min_lines: 200
    - path: "StudyResults/surrogate_fidelity/ranking_report.json"
      provides: "Comprehensive multi-criteria ranking with composite scores"
      contains: "composite_score"
    - path: "StudyResults/surrogate_fidelity/fidelity_summary.json"
      provides: "Updated fidelity summary with all 6 models"
      contains: "gp"
    - path: "writeups/vv_report/tables/surrogate_fidelity.tex"
      provides: "Updated LaTeX table with all model fidelity stats"
      contains: "GP"
    - path: "writeups/vv_report/figures/surrogate_comparison.pdf"
      provides: "Multi-panel comparison figure across all models"
  key_links:
    - from: "scripts/studies/surrogate_ranking_report.py"
      to: "StudyResults/surrogate_fidelity/ranking_report.json"
      via: "JSON output write"
      pattern: "json\\.dump.*ranking_report"
    - from: "scripts/studies/surrogate_ranking_report.py"
      to: "StudyResults/surrogate_fidelity/fidelity_summary.json"
      via: "JSON read + update + write"
      pattern: "fidelity_summary"
    - from: "writeups/vv_report/generate_figures.py"
      to: "StudyResults/surrogate_fidelity/fidelity_summary.json"
      via: "load_surrogate_fidelity()"
      pattern: "load_surrogate_fidelity"
---

<objective>
Synthesize all Phase 3 benchmark results (Stages 3.1-3.4) into a comprehensive ranking, select 1-2 surrogates for Phases 4-5, and update the V&V report.

Purpose: The benchmark stages produced individual metrics (prediction accuracy, gradient quality, k0_2 stratified error, inverse recovery, speed) for 6 surrogate models (nn_ensemble, rbf_baseline, pod_rbf_log, pod_rbf_nolog, gp, pce). This plan aggregates those into a single weighted ranking, makes the selection decision, and updates all reporting artifacts so the project can proceed to ISMO and PDE refinement with a clear, justified choice.

Output: ranking_report.json with composite scores, updated fidelity_summary.json with all 6 models, updated V&V report tables/figures, selection recommendation document embedded in ranking_report.json.
</objective>

<execution_context>
@/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/.claude/get-shit-done/workflows/execute-plan.md
@/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.plans/surrogate-pipeline-roadmap/PLAN.md
@.research/pde-constrained-inverse-surrogates/SUMMARY.md
@StudyResults/surrogate_fidelity/fidelity_summary.json
@StudyResults/surrogate_fidelity/pce_sobol_indices.json
@writeups/vv_report/generate_figures.py
@Surrogate/validation.py

<interfaces>
<!-- Key types and contracts the executor needs. -->

From Surrogate/validation.py:
```python
def validate_surrogate(surrogate, test_parameters, test_cd, test_pc) -> Dict:
    # Returns: cd_rmse, pc_rmse, cd_max_abs_error, pc_max_abs_error,
    #          cd_mean_relative_error, pc_mean_relative_error,
    #          cd_nrmse_per_sample, pc_nrmse_per_sample, n_test
```

From StudyResults/surrogate_fidelity/fidelity_summary.json:
```json
{
  "metadata": {"n_test": 479, "model_names": ["nn_ensemble", "rbf_baseline", "pod_rbf_log", "pod_rbf_nolog"]},
  "models": {
    "<model_name>": {
      "cd_max_nrmse": float, "cd_mean_nrmse": float, "cd_median_nrmse": float, "cd_95th_nrmse": float,
      "pc_max_nrmse": float, "pc_mean_nrmse": float, "pc_median_nrmse": float, "pc_95th_nrmse": float
    }
  }
}
```

Trained model artifacts available:
- data/surrogate_models/nn_ensemble/ (5 members, D1-D5 designs)
- data/surrogate_models/model_rbf_baseline.pkl
- data/surrogate_models/model_pod_rbf_log.pkl, model_pod_rbf_nolog.pkl
- data/surrogate_models/gp/ (44 GP models + likelihoods + metadata)
- data/surrogate_models/pce/pce_model.pkl
- data/surrogate_models/gp_fixed/ (alternate GP variant)

From writeups/vv_report/generate_figures.py:
```python
def load_surrogate_fidelity() -> dict  # reads fidelity_summary.json
def make_surrogate_fidelity_table(data) -> Path  # writes surrogate_fidelity.tex
# name_map currently has 4 models; needs extension to 6
```

From scripts/studies/parameter_recovery_all_models.py:
```python
# Runs multistart + cascade inference per model against PDE targets
# Uses: Surrogate.multistart.run_multistart_inference, Surrogate.cascade.run_cascade_inference
```
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create surrogate_ranking_report.py that computes multi-criteria ranking and generates all outputs</name>
  <files>scripts/studies/surrogate_ranking_report.py, StudyResults/surrogate_fidelity/ranking_report.json, StudyResults/surrogate_fidelity/fidelity_summary.json</files>
  <action>
Create `scripts/studies/surrogate_ranking_report.py` that:

1. **Loads all benchmark data from Stages 3.1-3.4.** Read:
   - `StudyResults/surrogate_fidelity/fidelity_summary.json` (prediction accuracy: cd/pc NRMSE stats)
   - `StudyResults/surrogate_fidelity/per_sample_errors.csv` (per-sample errors for k0_2 stratification)
   - `StudyResults/surrogate_fidelity/pce_sobol_indices.json` (sensitivity context)
   - Any Stage 3.2 gradient quality results (look in `StudyResults/` for gradient comparison files between autograd and FD models)
   - Any Stage 3.3 k0_2 stratified error results
   - Any Stage 3.4 inverse recovery results (e.g., from `StudyResults/parameter_recovery_v12/` or similar)
   - Training data file `data/surrogate_models/training_data_merged.npz` for the test split indices

   If Stage 3.2-3.4 result files do not yet exist, the script must compute the missing metrics directly:
   - **Prediction accuracy** (Stage 3.1): Load each surrogate via the standard API, run `validate_surrogate()` on the held-out test set, compute cd/pc NRMSE statistics.
   - **Gradient quality** (Stage 3.2): For models with autograd (nn_ensemble, gp), compute gradient at 5 random test points and compare to central FD (h=1e-5). Report mean relative difference. For models without autograd (rbf_baseline, pod_rbf_log, pod_rbf_nolog, pce), report "FD only" with a penalty score of 1.0 (worst).
   - **k0_2 performance** (Stage 3.3): From per_sample_errors.csv or by re-running validation, stratify test samples into 4 bins by log10(k0_2) value. Report mean NRMSE per bin. The "k0_2 score" is the mean NRMSE in the worst bin (lowest k0_2 values, which are hardest).
   - **Inverse recovery** (Stage 3.4): If recovery results exist, load them. If not, run a lightweight recovery test: use `run_multistart_inference()` with `n_candidates=2000` (not 20000 for speed) against the known true parameters from `scripts/_bv_common.py`. Report max relative error across 4 parameters.
   - **Speed**: Time `predict_batch()` on 1000 samples for each model. Time gradient computation (1 sample, 4D) for each model.

2. **Compute weighted composite score.** For each model, normalize each dimension to [0, 1] where 0 = best, 1 = worst across all models. Then compute:
   ```
   composite = 0.40 * inverse_recovery_norm
             + 0.25 * prediction_accuracy_norm
             + 0.20 * k02_performance_norm
             + 0.10 * speed_norm
             + 0.05 * gradient_quality_norm
   ```
   Where:
   - `prediction_accuracy_norm` = normalized cd_median_nrmse + pc_median_nrmse (averaged)
   - `inverse_recovery_norm` = normalized max_relative_error from recovery test
   - `k02_performance_norm` = normalized worst-bin k0_2 NRMSE
   - `speed_norm` = normalized total_time (predict + gradient)
   - `gradient_quality_norm` = normalized mean_relative_gradient_diff (1.0 for FD-only models)

3. **Apply go/no-go criterion.** Check if at least one model has worst-case NRMSE (max of cd_max_nrmse and pc_max_nrmse, ignoring the inflated near-zero-range PC samples by using the 99th percentile instead of max) below 10.7%. If none does, flag "NO-GO" in the report.

4. **Make selection recommendation.** Based on composite score:
   - Rank all 6 models.
   - Select the top-1 model as the primary surrogate for Phase 4 (ISMO) and Phase 5 (PDE refinement).
   - If the #2 model has a meaningfully different strength (e.g., GP has UQ but NN has better speed), recommend a cascade of two.
   - Document rationale: why this model, what its strengths/weaknesses are, how it compares to the current baseline.

5. **Write outputs:**
   - `StudyResults/surrogate_fidelity/ranking_report.json` with structure:
     ```json
     {
       "metadata": {"timestamp": "...", "weights": {...}, "go_no_go_threshold": 0.107},
       "models": {
         "<name>": {
           "prediction_accuracy": {"cd_median_nrmse": ..., "pc_median_nrmse": ..., "score_norm": ...},
           "gradient_quality": {"method": "autograd|fd", "mean_rel_diff": ..., "score_norm": ...},
           "k02_performance": {"worst_bin_nrmse": ..., "per_bin": {...}, "score_norm": ...},
           "inverse_recovery": {"max_rel_error": ..., "per_param": {...}, "score_norm": ...},
           "speed": {"predict_1000_ms": ..., "gradient_ms": ..., "score_norm": ...},
           "composite_score": ...,
           "rank": ...
         }
       },
       "go_no_go": {"passed": true/false, "best_worst_case": ..., "threshold": 0.107},
       "recommendation": {
         "primary_surrogate": "...",
         "secondary_surrogate": null or "...",
         "rationale": "...",
         "phase4_config": "...",
         "phase5_config": "..."
       }
     }
     ```
   - Update `StudyResults/surrogate_fidelity/fidelity_summary.json`: add entries for "gp" and "pce" models with the same NRMSE statistics format as existing models. Update `metadata.model_names` to include all 6.

   The script should load each surrogate model using the existing API:
   - NN ensemble: `Surrogate.nn_model.NNSurrogateModel` or `Surrogate.ensemble.EnsembleMeanWrapper`
   - RBF/POD-RBF: `Surrogate.surrogate_model.BVSurrogateModel` / `Surrogate.pod_rbf_model.PODRBFSurrogateModel`
   - GP: `Surrogate.gp_model.GPSurrogateModel`
   - PCE: `Surrogate.pce_model.PCESurrogateModel`

   Load test data from `data/surrogate_models/training_data_merged.npz` and `data/surrogate_models/split_indices.npz`.

   Use `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` at the top (standard for this codebase).

   Print a formatted ranking table to stdout during execution.
  </action>
  <verify>
    <automated>cd /Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse && source ../venv-firedrake/bin/activate && python scripts/studies/surrogate_ranking_report.py 2>&1 | tail -40</automated>
  </verify>
  <done>
    - ranking_report.json exists with composite scores for all 6 models
    - fidelity_summary.json updated with gp and pce entries (6 models total)
    - Go/no-go decision documented
    - Selection recommendation with rationale present
    - All scores are finite, no NaN values
  </done>
</task>

<task type="auto">
  <name>Task 2: Update V&V report tables and figures for all 6 models</name>
  <files>writeups/vv_report/generate_figures.py, writeups/vv_report/tables/surrogate_fidelity.tex, writeups/vv_report/figures/surrogate_comparison.pdf</files>
  <action>
Update `writeups/vv_report/generate_figures.py` to handle all 6 surrogate models:

1. **Update `make_surrogate_fidelity_table()`:**
   - Extend `name_map` to include: `"gp": "GP (GPyTorch)"`, `"pce": "PCE (ChaosPy)"`.
   - Update the iteration loop to include all 6 models from the loaded fidelity_summary.json (iterate over `data["models"].keys()` instead of hardcoded list, or extend the hardcoded list to 6 entries).
   - Keep the same column format (CD Median, CD 95th, CD Max, PC Median, PC 95th).

2. **Add `make_surrogate_comparison_figure()`:**
   Create a new figure function that produces a multi-panel comparison PDF:
   - **Panel A (top-left):** Grouped bar chart of CD median NRMSE and PC median NRMSE for all 6 models. Use log scale on y-axis. Label bars with model names on x-axis.
   - **Panel B (top-right):** Radar/spider chart (or grouped bar) showing the 5 normalized dimension scores from ranking_report.json for the top-3 models. This shows the tradeoff profile at a glance.
   - **Panel C (bottom):** Horizontal bar chart of composite scores for all 6 models, sorted best-to-worst. Annotate with the rank number. Highlight the selected model(s) with a distinct color.

   Load ranking data from `StudyResults/surrogate_fidelity/ranking_report.json`.

   Follow the existing publication styling from `matplotlib.rcParams` at the top of the file. Use `fig.savefig(FIGDIR / "surrogate_comparison.pdf", bbox_inches="tight")`.

3. **Update `make_summary_table()`:**
   - Update the "Surrogate" row to mention the number of models compared: e.g., `r"CD median NRMSE $< 0.005$ (best of 6 models)"`.

4. **Update `main()`:**
   - Add calls to generate the new comparison figure.
   - Handle loading ranking_report.json (with try/except like other loaders).

5. **Run the updated script** to regenerate all figures and tables.

Do NOT use a radar/spider chart if matplotlib does not support it cleanly -- use a grouped bar chart with hatching patterns instead. Keep it simple and publication-quality.
  </action>
  <verify>
    <automated>cd /Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse && source ../venv-firedrake/bin/activate && python writeups/vv_report/generate_figures.py 2>&1</automated>
  </verify>
  <done>
    - surrogate_fidelity.tex lists all 6 models (grep for "GP" and "PCE" in the file)
    - surrogate_comparison.pdf exists in writeups/vv_report/figures/
    - All existing figures and tables still generate successfully (no regressions)
    - generate_figures.py exits with code 0
  </done>
</task>

</tasks>

<verification>
1. `cat StudyResults/surrogate_fidelity/ranking_report.json | python -m json.tool | head -5` -- valid JSON
2. `python -c "import json; d=json.load(open('StudyResults/surrogate_fidelity/fidelity_summary.json')); print(sorted(d['models'].keys()))"` -- prints all 6 model names
3. `grep -c 'GP\|PCE' writeups/vv_report/tables/surrogate_fidelity.tex` -- returns >= 2
4. `ls -la writeups/vv_report/figures/surrogate_comparison.pdf` -- file exists with non-zero size
5. `python -c "import json; d=json.load(open('StudyResults/surrogate_fidelity/ranking_report.json')); print(d['recommendation']['primary_surrogate'])"` -- prints the selected model name
</verification>

<success_criteria>
- All 6 surrogate models have complete metrics across 5 dimensions
- Composite weighted ranking produced with clear #1 and #2
- Go/no-go decision documented (pass if any model < 10.7% worst-case)
- Selection recommendation with rationale for Phase 4 and Phase 5
- fidelity_summary.json extended from 4 to 6 models
- V&V report table includes GP and PCE rows
- Multi-panel comparison figure generated as PDF
- All existing V&V report artifacts still generate without regression
</success_criteria>

<output>
After completion, create `.planning/phases/3-comparative-benchmark/3-05-SUMMARY.md`
</output>
