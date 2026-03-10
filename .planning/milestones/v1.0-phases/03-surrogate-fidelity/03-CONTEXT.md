# Phase 3: Surrogate Fidelity - Context

**Gathered:** 2026-03-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Characterize error of all v13-era surrogate models across the inference parameter domain. Produce hold-out validation, per-sample error statistics, and comparative analysis across model types. Requirements: SUR-01, SUR-02, SUR-03.

</domain>

<decisions>
## Implementation Decisions

### Model scope
- Full treatment (SUR-01/02/03) for ALL 4 surrogate models in `StudyResults/surrogate_v11/`:
  - NN ensemble (D3-deeper) — the v13 production model
  - `model_rbf_baseline.pkl` — RBF with thin-plate spline kernel
  - `model_pod_rbf_log.pkl` — POD-RBF with log-space k0
  - `model_pod_rbf_nolog.pkl` — POD-RBF without log-space k0
- All 4 models get: LHS fidelity map, hold-out validation, error stats (max/mean/95th NRMSE)
- Same hold-out test points shared across all models (PDE ground truth computed once)
- Subsumes existing `TestSurrogateVsPDEConsistency` (Test 5) from `test_v13_verification.py` — remove that test to avoid redundancy

### Hold-out strategy
- Use existing hold-out split from `StudyResults/surrogate_v11/split_indices.npz` — no fresh PDE solves needed
- Use whatever hold-out size the existing split provides (expected ~10% of 3000 = ~300 samples)
- Test on the full union voltage grid (all voltage points the training data was generated on)
- Sub-grid analysis (symmetric, shallow, cathodic separately) can be extracted in post-processing if needed
- Save per-sample parameters + errors so the fidelity map shows error as a function of parameter space location

### Error thresholds
- Diagnostic with soft gates: compute all error stats, save to JSON, assert only on catastrophic failure
- Soft gate: mean NRMSE < 20% per model — same threshold for all 4 model types
- Use NRMSE only (normalized by per-sample range), not pointwise relative error — avoids division-by-near-zero at small currents
- Required statistics per model per output (CD and PC): max NRMSE, mean NRMSE, 95th percentile NRMSE
- Actual error values are for the Phase 6 report; the pytest gate only catches fundamentally broken models

### Output artifacts
- Results directory: `StudyResults/surrogate_fidelity/`
- JSON summary file with aggregate stats (max/mean/95th NRMSE per model per output)
- CSV per-sample file with parameters (k0_1, k0_2, alpha_1, alpha_2) and errors for each model
- Plots generated during test run:
  - Worst-case I-V overlay: top 3 worst NRMSE samples per model, surrogate vs PDE curves overlaid
  - Error vs parameter scatter: NRMSE vs each of the 4 parameters, separate plots for CD and PC errors
- CD and PC errors plotted separately (peroxide current has different error patterns than total current density)

### Claude's Discretion
- Exact structure of the test file and fixture organization
- How to load and interface with all 4 model types uniformly (they have different APIs)
- Plot styling, layout, and figure sizing
- Whether to use a single test class or separate classes per model
- How to handle the split_indices.npz loading and validation

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `Surrogate/validation.py:validate_surrogate()`: Computes RMSE, max abs error, NRMSE per sample — core metric engine, reuse directly
- `Surrogate/validation.py:print_validation_report()`: Formatted console output of metrics
- `Surrogate/io.py:load_surrogate()`: Loads .pkl RBF/POD-RBF models
- `Surrogate/ensemble.py:load_nn_ensemble()`: Loads NN ensemble from directory
- `Surrogate/sampling.py:ParameterBounds, generate_lhs_samples()`: LHS sampling with log-space k0
- `scripts/surrogate/validate_surrogate.py`: CLI script for validation — pattern to follow for per-sample CSV output

### Established Patterns
- Test output saved to `StudyResults/` directory (convention from Phase 2 MMS convergence)
- JSON + PNG artifacts pattern from `StudyResults/mms_convergence/`
- `@pytest.mark.slow` for expensive tests
- Module-scoped fixtures for shared model loading (see `test_v13_verification.py:nn_ensemble`)
- Existing `split_indices.npz` in `StudyResults/surrogate_v11/` for train/test split

### Integration Points
- `StudyResults/surrogate_v11/training_data_merged.npz`: 3000-sample PDE ground truth data
- `StudyResults/surrogate_v11/split_indices.npz`: Existing train/test split indices
- `StudyResults/surrogate_v11/model_*.pkl`: 3 RBF model files to load
- `StudyResults/surrogate_v11/nn_ensemble/D3-deeper/`: NN ensemble directory (5 members)
- Phase 6 will read `StudyResults/surrogate_fidelity/*.json` and `*.csv` for automated report generation
- `test_v13_verification.py:TestSurrogateVsPDEConsistency` to be removed (subsumed)

</code_context>

<specifics>
## Specific Ideas

- All 4 models share the same hold-out test points — PDE ground truth computed once, surrogate predictions compared per model
- Per-sample data must include parameter coordinates so Phase 6 can generate parameter-space error maps
- Worst-case I-V overlay plots are the top 3 highest-NRMSE samples per model (not just the single worst)
- Error-vs-parameter scatter plots separate CD and PC because peroxide current is a smaller signal with different error patterns
- The 20% mean NRMSE soft gate is generous — it catches catastrophically broken models but doesn't fail on models that are merely mediocre

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-surrogate-fidelity*
*Context gathered: 2026-03-07*
