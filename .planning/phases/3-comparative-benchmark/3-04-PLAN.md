---
phase: 3-comparative-benchmark
plan: 04
type: execute
wave: 1
depends_on: []
files_modified:
  - scripts/studies/inverse_benchmark_all_models.py
  - StudyResults/inverse_benchmark/recovery_table.csv
  - StudyResults/inverse_benchmark/recovery_summary.json
  - StudyResults/inverse_benchmark/timing_table.csv
autonomous: true
requirements: [BENCH-04]

must_haves:
  truths:
    - "Each surrogate model runs the full multistart+cascade inverse pipeline against synthetic targets with known ground truth"
    - "Parameter recovery errors are computed per-parameter per-model per-noise-level"
    - "k0_2 recovery accuracy is explicitly compared across all models"
    - "Wall-clock timing is recorded for each model's full inference run"
    - "Results are compared against the existing v12 baseline (recovery_comparison.csv) and v13 master inference"
  artifacts:
    - path: "scripts/studies/inverse_benchmark_all_models.py"
      provides: "End-to-end inverse benchmark script for all surrogate models"
      min_lines: 250
    - path: "StudyResults/inverse_benchmark/recovery_table.csv"
      provides: "model x parameter x noise_level recovery error table"
    - path: "StudyResults/inverse_benchmark/recovery_summary.json"
      provides: "Machine-readable summary with best model identification"
    - path: "StudyResults/inverse_benchmark/timing_table.csv"
      provides: "Wall-clock time per model per method"
  key_links:
    - from: "scripts/studies/inverse_benchmark_all_models.py"
      to: "Surrogate/multistart.py"
      via: "run_multistart_inference()"
      pattern: "run_multistart_inference"
    - from: "scripts/studies/inverse_benchmark_all_models.py"
      to: "Surrogate/cascade.py"
      via: "run_cascade_inference()"
      pattern: "run_cascade_inference"
    - from: "scripts/studies/inverse_benchmark_all_models.py"
      to: "Surrogate/ensemble.py"
      via: "load_nn_ensemble()"
      pattern: "load_nn_ensemble"
---

<objective>
Run the full inverse pipeline (multistart -> cascade) with each surrogate model to measure actual parameter recovery performance under varying noise conditions. This is Stage 4 of the Phase 3 Comparative Surrogate Benchmark.

Purpose: Prediction accuracy alone does not guarantee good inverse performance. A surrogate with 5% prediction error but biased gradients may recover parameters worse than one with 8% error but accurate gradients. This test measures the metric that actually matters: can the surrogate-based pipeline recover the true parameters?

Output: Parameter recovery table (model x parameter x noise level -> relative error), wall-clock timing, identification of best surrogate for k0_2 recovery, comparison against existing v12/v13 baselines.
</objective>

<execution_context>
@/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/.claude/get-shit-done/workflows/execute-plan.md
@/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@scripts/studies/parameter_recovery_all_models.py
@Surrogate/multistart.py
@Surrogate/cascade.py
@Surrogate/objectives.py
@Surrogate/ensemble.py
@Surrogate/gp_model.py
@Surrogate/pce_model.py
@scripts/_bv_common.py
@StudyResults/inverse_verification/parameter_recovery_summary.json
@StudyResults/parameter_recovery_v12/recovery_comparison.csv
@StudyResults/master_inference_v13/master_comparison_v13.csv

<interfaces>
<!-- Existing infrastructure the executor will use directly -->

From Surrogate/multistart.py:
```python
@dataclass(frozen=True)
class MultiStartConfig:
    n_grid: int = 20_000
    n_top_candidates: int = 20
    polish_maxiter: int = 60
    secondary_weight: float = 1.0
    fd_step: float = 1e-5
    use_shallow_subset: bool = True
    seed: int = 42
    verbose: bool = True

def run_multistart_inference(
    surrogate, target_cd, target_pc,
    bounds_k0_1, bounds_k0_2, bounds_alpha,
    config=None, subset_idx=None,
) -> MultiStartResult
```

From Surrogate/cascade.py:
```python
@dataclass(frozen=True)
class CascadeConfig:
    pass1_weight: float = 0.5
    pass2_weight: float = 2.0
    pass1_maxiter: int = 60
    pass2_maxiter: int = 60
    polish_maxiter: int = 30
    polish_weight: float = 1.0
    skip_polish: bool = False
    fd_step: float = 1e-5
    verbose: bool = True

def run_cascade_inference(
    surrogate, target_cd, target_pc,
    initial_k0, initial_alpha,
    bounds_k0_1, bounds_k0_2, bounds_alpha,
    config=None, subset_idx=None,
) -> CascadeResult
```

From Surrogate/ensemble.py:
```python
def load_nn_ensemble(ensemble_dir, n_members=5, device="cpu") -> EnsembleMeanWrapper
```

From scripts/_bv_common.py:
```python
K0_HAT_R1  # true k0_1 (~1.263e-3)
K0_HAT_R2  # true k0_2 (~5.263e-5)
ALPHA_R1 = 0.627
ALPHA_R2 = 0.5
```

From Surrogate/sampling.py:
```python
@dataclass(frozen=True)
class ParameterBounds:
    k0_1_range: tuple = (1e-6, 1.0)
    k0_2_range: tuple = (1e-7, 0.1)
    alpha_1_range: tuple = (0.1, 0.9)
    alpha_2_range: tuple = (0.1, 0.9)
```

Surrogate models available on disk:
- NN ensemble designs: D1-default, D2-wider, D3-deeper, D4-no-physics, D5-strong-physics
- GP: data/surrogate_models/gp_fixed/
- POD-RBF: model_pod_rbf_log.pkl, model_pod_rbf_nolog.pkl
- RBF: model_rbf_baseline.pkl
- PCE: data/surrogate_models/pce/

Existing baselines to compare against:
- v12 recovery: RBF baseline multistart gets 4.23% max error (best), NN D3 gets 27.3% (worst)
- v13 master: NN ensemble cascade gets 4.63% max error on surrogate-only, 4.33% after PDE refinement
- parameter_recovery_summary.json: 10.7% surrogate bias at 0% noise, catastrophic 287% at 1% noise (seed 44)
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Build comprehensive inverse benchmark script</name>
  <files>scripts/studies/inverse_benchmark_all_models.py</files>
  <action>
Create a comprehensive parameter recovery benchmark script that extends the existing `scripts/studies/parameter_recovery_all_models.py` pattern with these key additions:

1. **Synthetic target generation with noise:**
   - Generate 8 synthetic target parameter sets spanning the parameter space using LHS in log-k0 space (seed=123 for reproducibility). Include the standard true parameters (K0_HAT_R1, K0_HAT_R2, ALPHA_R1, ALPHA_R2) as target 0.
   - For each target, generate clean I-V curves using the surrogate at those true params (surrogate-vs-surrogate test, no inverse crime).
   - Add multiplicative Gaussian noise at 0%, 1%, 2% levels: `noisy = clean * (1 + noise_pct/100 * randn)`, using seeds [42, 43, 44] for 3 realizations per noise level.

2. **Model registry** (reuse the pattern from `parameter_recovery_all_models.py`):
   - NN D1-default, D2-wider, D3-deeper, D4-no-physics, D5-strong-physics (all via `load_nn_ensemble`)
   - GP (via `load_gp_surrogate` wrapped in `_GPFDWrapper` to force FD gradients)
   - POD-RBF log, POD-RBF nolog, RBF baseline (via pickle)
   - PCE (via pickle, if compatible with multistart/cascade API -- skip gracefully if not)

3. **Inference methods per model:**
   - `multistart` with n_grid=20,000 (5,000 for GP due to speed), n_top=20
   - `cascade` with default CascadeConfig
   - For each run, record: recovered params, per-param relative error, wall-clock time

4. **Target generation strategy:**
   - Use LHS to generate 7 additional targets beyond the standard one. Sample log10(k0_1) in [-4, -1], log10(k0_2) in [-5, -2], alpha in [0.2, 0.8].
   - For each target, use the RBF baseline surrogate to generate "ground truth" I-V curves (this avoids inverse crime -- all models are tested against a different model's predictions).
   - Important: verify each surrogate's phi_applied grid matches and handle mismatches gracefully (as done in the existing script).

5. **Output structure:**
   - CSV: `StudyResults/inverse_benchmark/recovery_table.csv` with columns: target_id, noise_pct, seed, model_name, method, k0_1_true, k0_1_recovered, k0_1_error_pct, k0_2_true, k0_2_recovered, k0_2_error_pct, alpha_1_true, alpha_1_recovered, alpha_1_error_pct, alpha_2_true, alpha_2_recovered, alpha_2_error_pct, max_error_pct, time_s
   - JSON summary: `StudyResults/inverse_benchmark/recovery_summary.json` with aggregated statistics per model: median/mean/max recovery error across all targets and noise levels, best model for k0_2
   - Timing CSV: `StudyResults/inverse_benchmark/timing_table.csv` with model_name, method, mean_time_s, std_time_s

6. **Comparison with baselines:**
   - After all runs, print a comparison table showing: current best (v12/v13 baselines) vs each model
   - Highlight which model gives best k0_2 recovery at each noise level
   - Flag any model that beats the v12 RBF baseline (4.23% max error) or the v13 cascade (4.63%)

7. **Implementation details:**
   - Use `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` at top
   - Add path setup matching the existing script pattern (`sys.path.insert`)
   - Use try/except around each model+method combination so one failure does not block others
   - Print progress to stdout with timestamps
   - Skip the full 8-target x 3-noise x 3-seed matrix for GP (too slow) -- run GP only on target 0 at noise=[0%, 1%] with seed=42

Follow the immutable data patterns from the codebase: create new result objects rather than mutating. Use the frozen dataclass pattern for result containers.
  </action>
  <verify>
    <automated>cd /Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse && python -c "import ast; ast.parse(open('scripts/studies/inverse_benchmark_all_models.py').read()); print('Syntax OK')"</automated>
  </verify>
  <done>Script exists, parses without errors, contains model registry with all available surrogates, generates synthetic targets with noise, runs multistart+cascade for each model, outputs CSV+JSON results with baseline comparison.</done>
</task>

<task type="auto">
  <name>Task 2: Execute benchmark and validate outputs</name>
  <files>StudyResults/inverse_benchmark/recovery_table.csv, StudyResults/inverse_benchmark/recovery_summary.json, StudyResults/inverse_benchmark/timing_table.csv</files>
  <action>
Run the benchmark script created in Task 1 using the venv-firedrake environment:

```bash
cd /Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse
source ../venv-firedrake/bin/activate
python scripts/studies/inverse_benchmark_all_models.py
```

**Important runtime considerations:**
- This will take significant time (estimate 30-60 minutes depending on GP speed). Run with patience.
- If the script fails on a specific model, fix the error and re-run. Common issues:
  - GP model path mismatch (try both `gp/` and `gp_fixed/` directories)
  - PCE API incompatibility (the PCE `predict()` may not match the scalar API -- wrap or skip)
  - Memory issues with large n_grid for GP (reduce to 2,000 if needed)
- If runtime exceeds 60 minutes, reduce the test matrix: drop targets 5-7, keep only noise=[0%, 1%]

After execution completes, validate:
1. `recovery_table.csv` has rows for each model x method x target x noise x seed combination
2. `recovery_summary.json` contains per-model aggregated statistics
3. `timing_table.csv` shows wall-clock times per model
4. Print the summary comparison table to verify results make physical sense:
   - 0% noise errors should be bounded by surrogate bias (~5-15%)
   - Errors should increase with noise level
   - k0_2 should generally have higher error than k0_1 (known from prior work)
   - NN ensemble should show benefit of autograd gradients vs FD-only models

If any result looks anomalous (e.g., 0% noise giving >50% error), investigate and fix before declaring done. Common causes: wrong target curves fed to wrong model, phi_applied grid mismatch, log-space vs linear-space confusion.
  </action>
  <verify>
    <automated>cd /Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse && python -c "
import json, csv, os
base = 'StudyResults/inverse_benchmark'
# Check all output files exist
for f in ['recovery_table.csv', 'recovery_summary.json', 'timing_table.csv']:
    path = os.path.join(base, f)
    assert os.path.exists(path), f'Missing: {path}'
    assert os.path.getsize(path) > 100, f'Too small: {path}'
# Check CSV has multiple rows
with open(os.path.join(base, 'recovery_table.csv')) as f:
    rows = list(csv.DictReader(f))
    assert len(rows) >= 10, f'Too few rows: {len(rows)}'
    assert 'k0_2_error_pct' in rows[0], 'Missing k0_2_error_pct column'
# Check JSON has model entries
with open(os.path.join(base, 'recovery_summary.json')) as f:
    summary = json.load(f)
    assert 'models' in summary or 'results' in summary, 'Missing models/results key'
print(f'Validation passed: {len(rows)} recovery rows, summary has required keys')
"</automated>
  </verify>
  <done>All three output files exist with valid data. Recovery table has rows for multiple models x noise levels. Summary JSON identifies best model for k0_2 recovery. Timing table shows wall-clock per model. Results are physically plausible (errors increase with noise, k0_2 harder than k0_1).</done>
</task>

</tasks>

<verification>
1. The benchmark script runs end-to-end without crashing (tolerant of individual model failures)
2. At least 5 different surrogate models are benchmarked (NN D3, D2, GP, POD-RBF, RBF)
3. Results at 0% noise show errors consistent with surrogate bias (~5-15% range)
4. Results at 1-2% noise show graceful degradation (not catastrophic)
5. k0_2 recovery is explicitly tracked and the best model identified
6. Wall-clock timing shows meaningful differences between models
7. Comparison with v12/v13 baselines is present in the output
</verification>

<success_criteria>
- recovery_table.csv contains per-run results for >= 5 models x 2 methods x >= 2 noise levels
- recovery_summary.json identifies the best model for k0_2 recovery with supporting statistics
- timing_table.csv shows inference time per model (enables cost-benefit analysis)
- At least one model achieves < 10% max parameter error at 0% noise on the standard target
- k0_2 error statistics are explicitly reported and compared across models
</success_criteria>

<output>
After completion, create `.planning/phases/3-comparative-benchmark/3-04-SUMMARY.md`
</output>
