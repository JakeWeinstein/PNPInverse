---
phase: 3-comparative-benchmark
plan: 02
type: execute
wave: 1
depends_on: []
files_modified:
  - scripts/studies/gradient_benchmark.py
  - tests/test_gradient_benchmark.py
  - StudyResults/gradient_benchmark/gradient_accuracy.json
  - StudyResults/gradient_benchmark/gradient_speed.json
  - StudyResults/gradient_benchmark/gradient_benchmark_report.md
autonomous: true
requirements: [BENCH-GRAD-01, BENCH-GRAD-02, BENCH-GRAD-03]
must_haves:
  truths:
    - "Gradient accuracy is quantified for every surrogate model (NN ensemble, GP, PCE) using both autograd/analytic and FD methods"
    - "Wall-clock gradient timing is measured for single-point and batch evaluation across all models"
    - "Fine FD gradient serves as reference and autograd/analytic gradients match it within tolerance"
    - "Results are saved as reproducible JSON artifacts extending the existing gradient verification data"
  artifacts:
    - path: "scripts/studies/gradient_benchmark.py"
      provides: "Main benchmark script covering accuracy + speed for all models"
      min_lines: 200
    - path: "tests/test_gradient_benchmark.py"
      provides: "Smoke test that benchmark script runs and produces valid output"
      min_lines: 40
    - path: "StudyResults/gradient_benchmark/gradient_accuracy.json"
      provides: "Model x method -> relative error vs fine-FD reference"
      contains: "relative_error"
    - path: "StudyResults/gradient_benchmark/gradient_speed.json"
      provides: "Wall-clock timing per gradient evaluation"
      contains: "ms_per_eval"
  key_links:
    - from: "scripts/studies/gradient_benchmark.py"
      to: "Surrogate/objectives.py"
      via: "SurrogateObjective with autograd and FD paths"
      pattern: "SurrogateObjective"
    - from: "scripts/studies/gradient_benchmark.py"
      to: "Surrogate/gp_model.py"
      via: "GPSurrogateModel.gradient_at() autograd path"
      pattern: "gradient_at"
    - from: "scripts/studies/gradient_benchmark.py"
      to: "Surrogate/pce_model.py"
      via: "PCESurrogateModel.predict_gradient() analytic path"
      pattern: "predict_gradient"
---

<objective>
Benchmark gradient computation accuracy and speed across all surrogate models (NN ensemble, GP, PCE) and gradient methods (autograd, analytic polynomial differentiation, finite differences).

Purpose: Determine which surrogate + gradient method combination provides the best accuracy-speed tradeoff for the inverse pipeline. This directly informs surrogate selection for Phase 4 (ISMO) and Phase 5 (PDE refinement).

Output: Gradient accuracy table, speed benchmark table, and a reproducible script that can be re-run after any model retraining.
</objective>

<execution_context>
@/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/.claude/get-shit-done/workflows/execute-plan.md
@/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.plans/surrogate-pipeline-roadmap/PLAN.md
@Surrogate/objectives.py
@Surrogate/nn_model.py
@Surrogate/gp_model.py
@Surrogate/pce_model.py
@Surrogate/ensemble.py
@scripts/benchmark_autograd_vs_fd.py
@tests/test_autograd_gradient.py
@StudyResults/inverse_verification/gradient_fd_convergence.json
@StudyResults/inverse_verification/gradient_pde_consistency.json

<interfaces>
<!-- Key APIs the executor needs for gradient computation -->

From Surrogate/objectives.py:
```python
class SurrogateObjective:
    def __init__(self, surrogate, target_cd, target_pc, secondary_weight=1.0, fd_step=1e-5, log_space_k0=True, bounds=None)
    def objective(self, x: np.ndarray) -> float  # x = [log10(k0_1), log10(k0_2), alpha_1, alpha_2]
    def gradient(self, x: np.ndarray) -> np.ndarray  # auto-selects autograd or FD
    def objective_and_gradient(self, x: np.ndarray) -> Tuple[float, np.ndarray]
    # Internal: _use_autograd bool, _autograd_objective_and_gradient()
```

From Surrogate/nn_model.py:
```python
class NNSurrogateModel:
    def predict_torch(self, x_logspace: torch.Tensor) -> torch.Tensor  # (4,) -> (2*n_eta,)
    @classmethod
    def load(cls, path: str, device: str = "cpu") -> NNSurrogateModel
```

From Surrogate/ensemble.py:
```python
class EnsembleMeanWrapper:
    def predict_torch(self, x_logspace: torch.Tensor) -> torch.Tensor  # averages member outputs
def load_nn_ensemble(ensemble_dir: str) -> EnsembleMeanWrapper
```

From Surrogate/gp_model.py:
```python
class GPSurrogateModel:
    def predict_torch(self, x: torch.Tensor) -> torch.Tensor  # (M,4) -> (M,44) in normalized space
    def gradient_at(self, k0_1, k0_2, alpha_1, alpha_2, target_cd, target_pc, secondary_weight=1.0) -> np.ndarray
    @classmethod
    def load(cls, path: str, device: str = "cpu") -> GPSurrogateModel
```

From Surrogate/pce_model.py:
```python
class PCESurrogateModel:
    def predict_gradient(self, k0_1, k0_2, alpha_1, alpha_2) -> Dict[str, np.ndarray]
        # Returns {'grad_cd': (4, n_eta), 'grad_pc': (4, n_eta)} -- analytic via polynomial differentiation
        # NOTE: chain rule for log-space k0 IS applied -- returns d/d(physical k0), NOT d/d(log10_k0)
    @staticmethod
    def load(path: str) -> PCESurrogateModel
```

Trained model paths:
- NN ensemble: data/surrogate_models/nn_ensemble/D3-deeper (or D1-default, D2-wider, etc.)
- GP: data/surrogate_models/gp_fixed/
- PCE: data/surrogate_models/pce/ (check for .pkl files)
- POD-RBF: data/surrogate_models/model_pod_rbf_log.pkl (FD only, no autograd)
- RBF baseline: data/surrogate_models/model_rbf_baseline.pkl (FD only, no autograd)
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Create gradient benchmark script and test</name>
  <files>scripts/studies/gradient_benchmark.py, tests/test_gradient_benchmark.py</files>
  <behavior>
    - Test 1: benchmark script imports and `run_accuracy_benchmark` returns a dict with keys per model
    - Test 2: `run_speed_benchmark` returns timing dict with ms_per_eval > 0 for each model/method
    - Test 3: accuracy results contain relative_error fields for each model x method combination
  </behavior>
  <action>
Create `scripts/studies/gradient_benchmark.py` that benchmarks gradient computation across all surrogate models. The script must:

1. **Load all available surrogate models** with graceful fallback if a model is missing:
   - NN ensemble via `load_nn_ensemble("data/surrogate_models/nn_ensemble/D3-deeper")`
   - GP via `GPSurrogateModel.load("data/surrogate_models/gp_fixed/")`
   - PCE via `PCESurrogateModel.load(...)` -- check `data/surrogate_models/pce/` for the actual .pkl filename
   - POD-RBF via pickle load from `data/surrogate_models/model_pod_rbf_log.pkl`
   - RBF baseline via pickle load from `data/surrogate_models/model_rbf_baseline.pkl`

2. **Select test points** -- 20 points from the held-out test set (load from `data/surrogate_models/split_indices.npz` and `training_data_merged.npz`), plus 5 manually chosen points covering corners of the parameter space. Each point is `x = [log10(k0_1), log10(k0_2), alpha_1, alpha_2]`.

3. **Gradient accuracy benchmark** (`run_accuracy_benchmark`):
   - For each model and each test point, compute the objective gradient dJ/dx where J = 0.5*||cd_pred - cd_target||^2 + w*0.5*||pc_pred - pc_target||^2 (use secondary_weight=1.0).
   - Use target curves from a reference parameter set (e.g., the first test point as target).
   - Gradient methods per model:
     - NN ensemble: autograd (via SurrogateObjective with _use_autograd=True)
     - GP: autograd (via GPSurrogateModel.gradient_at())
     - PCE: analytic (via PCESurrogateModel.predict_gradient(), then compose with chain rule to get dJ/dx)
     - ALL models: FD with step sizes [1e-3, 1e-4, 1e-5, 1e-6]
   - Reference gradient: fine FD at step=1e-7 (or smallest stable step per model).
   - Compute relative error = max(|g_method - g_ref|) / max(|g_ref|, 1e-15) for each (model, method, point).
   - Record per-component and max relative error.

4. **Gradient speed benchmark** (`run_speed_benchmark`):
   - For each model and gradient method, time N_ITERS=50 gradient evaluations (after 5 warmup calls).
   - Measure single-point gradient time (one x vector).
   - Measure batch gradient time: 10 points simultaneously (where supported -- NN and GP can batch, PCE loops).
   - Report ms_per_eval for each (model, method, batch_size) combination.

5. **Output functions**:
   - `save_accuracy_results(results, path)` -> JSON to `StudyResults/gradient_benchmark/gradient_accuracy.json`
   - `save_speed_results(results, path)` -> JSON to `StudyResults/gradient_benchmark/gradient_speed.json`
   - `generate_report(accuracy, speed, path)` -> Markdown summary table to `StudyResults/gradient_benchmark/gradient_benchmark_report.md`

6. **CLI entry point**: `if __name__ == "__main__"` that runs both benchmarks and saves all outputs.

Key implementation details:
- For the PCE analytic gradient, you need to compose: the PCE gives d(output)/d(params) per voltage point, but you need dJ/dx. Compute: dJ/dx_i = sum_j (cd_pred_j - cd_target_j) * d(cd_j)/dx_i + w * sum_j (pc_pred_j - pc_target_j) * d(pc_j)/dx_i. The PCE predict_gradient returns gradients in physical param space (with chain rule for log k0 already applied); but the optimizer works in log-space x = [log10(k0_1), log10(k0_2), alpha_1, alpha_2], so for the FD reference in log-space you get dJ/d(log10_k0). The PCE analytic gradient gives dJ/d(k0) via chain rule. To compare: convert PCE gradient to log-space: dJ/d(log10_k0) = dJ/d(k0) * k0 * ln(10). For alpha dimensions, no conversion needed.
- For FD-only models (RBF, POD-RBF), only measure FD gradient speed/accuracy -- no autograd.
- Use `time.perf_counter()` for timing.
- All paths should be relative to repo root (script run from repo root).

Create `tests/test_gradient_benchmark.py` with a lightweight smoke test:
- Mock or use tiny synthetic surrogates (like in test_autograd_gradient.py) to verify the benchmark functions return correctly structured output.
- Do NOT require loading full trained models -- test the structure, not the values.
  </action>
  <verify>
    <automated>cd /Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse && source ../venv-firedrake/bin/activate && python -m pytest tests/test_gradient_benchmark.py -x -v 2>&1 | tail -30</automated>
  </verify>
  <done>
    - gradient_benchmark.py exists with run_accuracy_benchmark(), run_speed_benchmark(), save functions, and CLI entry point
    - test_gradient_benchmark.py passes with structural validation of benchmark outputs
    - Script is importable without errors
  </done>
</task>

<task type="auto">
  <name>Task 2: Run gradient benchmark and generate results</name>
  <files>StudyResults/gradient_benchmark/gradient_accuracy.json, StudyResults/gradient_benchmark/gradient_speed.json, StudyResults/gradient_benchmark/gradient_benchmark_report.md</files>
  <action>
Run the gradient benchmark script created in Task 1 against all trained surrogate models:

```bash
cd /Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse
source ../venv-firedrake/bin/activate
python scripts/studies/gradient_benchmark.py
```

If any model fails to load (missing artifacts), the script should skip it gracefully and benchmark the available models.

After the script completes, verify the output files exist and contain meaningful results:
1. `StudyResults/gradient_benchmark/gradient_accuracy.json` -- must have entries for at least NN ensemble and one other model
2. `StudyResults/gradient_benchmark/gradient_speed.json` -- must have ms_per_eval > 0 entries
3. `StudyResults/gradient_benchmark/gradient_benchmark_report.md` -- human-readable summary

If the benchmark reveals any gradient accuracy issues (relative error > 1% for autograd methods), investigate and document the cause in the report.

Expected results based on existing data:
- NN autograd vs FD: less than 0.1% relative error (confirmed by existing test_autograd_gradient.py)
- GP autograd vs FD: less than 1% relative error (GP float32 precision may add noise)
- PCE analytic vs FD: less than 0.1% relative error (polynomial differentiation is exact)
- NN autograd speed: approximately 4-10x faster than FD (confirmed by existing benchmark_autograd_vs_fd.py)
- GP autograd speed: likely slower than NN due to 44 independent GP posterior evaluations
- PCE analytic speed: likely fast (polynomial evaluation is cheap)
  </action>
  <verify>
    <automated>cd /Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse && source ../venv-firedrake/bin/activate && python -c "import json; d=json.load(open('StudyResults/gradient_benchmark/gradient_accuracy.json')); print(f'Models tested: {list(d.keys())}'); print(f'Accuracy file valid: True')" && python -c "import json; d=json.load(open('StudyResults/gradient_benchmark/gradient_speed.json')); print(f'Speed entries: {len(d)}'); print(f'Speed file valid: True')" && test -f StudyResults/gradient_benchmark/gradient_benchmark_report.md && echo "Report exists: True"</automated>
  </verify>
  <done>
    - gradient_accuracy.json contains relative error data for at least 2 surrogate models
    - gradient_speed.json contains timing data with ms_per_eval for each model x method combination
    - gradient_benchmark_report.md contains a readable summary table ranking models by accuracy and speed
    - All autograd/analytic gradients match FD reference within 1% for NN and PCE, within 5% for GP
  </done>
</task>

</tasks>

<verification>
1. `python -m pytest tests/test_gradient_benchmark.py -x` passes
2. `StudyResults/gradient_benchmark/gradient_accuracy.json` exists and is valid JSON with model entries
3. `StudyResults/gradient_benchmark/gradient_speed.json` exists with timing data
4. `StudyResults/gradient_benchmark/gradient_benchmark_report.md` exists with summary tables
5. The benchmark script can be re-run reproducibly: `python scripts/studies/gradient_benchmark.py`
</verification>

<success_criteria>
- Gradient accuracy quantified for all available surrogate models (at minimum NN ensemble + GP or PCE)
- Autograd/analytic gradients verified against FD reference with documented relative errors
- Wall-clock timing measured for single-point and batch gradient evaluation
- Results saved as reproducible JSON artifacts
- Clear ranking of models by gradient accuracy and speed in the report
</success_criteria>

<output>
After completion, create `.planning/phases/3-comparative-benchmark/3-02-SUMMARY.md`
</output>
