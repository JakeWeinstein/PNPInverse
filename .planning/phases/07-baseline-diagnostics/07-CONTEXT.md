# Phase 7: Baseline Diagnostics - Context

**Gathered:** 2026-03-10
**Status:** Ready for planning

<domain>
## Phase Boundary

Quantify v13 pipeline performance across noise seeds and determine which parameters are practically identifiable at 2% noise. This phase produces diagnostic data consumed by Phase 8 (ablation) and Phase 9 (redesign). No pipeline modifications — measurement only.

</domain>

<decisions>
## Implementation Decisions

### Multi-seed run design
- 20 noise seeds, sequential integers 0-19
- Full 7-phase v13 pipeline (S1-S5 + P1-P2) per seed — this is what we're diagnosing
- New wrapper script that calls v13 per seed, collects results, and generates summary CSV — keeps v13 script unchanged
- 2% noise level throughout

### Profile likelihood method
- New implementation (do not reuse existing Infer_BVProfileLikelihood_charged.py or profile_likelihood_study.py)
- PDE-only profile likelihood — no surrogate profiles
- 30 profile points per parameter (k0_1, k0_2, alpha_1, alpha_2)
- Chi-squared 95% CI threshold (delta-chi2 = 3.84 for 1 DOF) for identifiability determination
- ~120 PDE re-optimizations total across 4 parameters

### Sensitivity visualization
- Both 1D parameter sweeps AND Jacobian heatmap
- 1D sweeps: 5 values per parameter (e.g., 0.5x, 0.75x, 1x, 1.5x, 2x of true value), plotting total current and peroxide current vs voltage
- Extended voltage range beyond v13 default — explore wider cathodic window to find informative regions
- **Critical: use voltage continuation/warm-starting and bridge points already implemented in the I-V sweep code for extended voltages. Allow more SNES iterations or smaller dt for convergence at extreme voltages.**
- Jacobian heatmap via central finite differences (h=1e-5), consistent with codebase convention
- Heatmap shows d(observable)/d(parameter) at each voltage point for all 4 parameters

### Results reporting
- Summary statistics: median relative error, IQR (25th/75th percentile), worst-case (max) per parameter
- Auto-generated plots: box plots of relative error per parameter across seeds, per-seed scatter for outlier identification
- JSON metadata sidecar for each diagnostic tool (AUDT-04 compliance): tool name, justification type (literature/empirical/simplest), reference, rationale
- Output directory: `StudyResults/v14/` with sub-folders for each diagnostic (multi_seed/, profile_likelihood/, sensitivity/)

### Claude's Discretion
- Exact parameter sweep ranges for 1D sensitivity (multiplicative factors around true values)
- Extended voltage range bounds (how far beyond v13 default to go)
- Profile likelihood parameter range selection
- Plot styling (colors, layout, figure sizes)
- Multi-seed wrapper script architecture details
- JSON metadata schema details beyond the required fields

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `scripts/surrogate/Infer_BVMaster_charged_v13_ultimate.py`: Primary pipeline with `--noise-seed` and `--noise-percent` CLI args — wrapper script can invoke this directly
- `Forward/noise.py`: `add_percent_noise()` utility for synthetic data generation
- `Forward/steady_state/bv.py`: Voltage continuation and bridge point logic for extended sweeps
- `FluxCurve/bv_point_solve/`: Parallel point solve with caching — reusable for sensitivity sweeps
- `Surrogate/objectives.py`: Objective function classes with `objective()`, `gradient()`, `objective_and_gradient()` — profile likelihood re-optimization needs these
- `scripts/_bv_common.py`: Physical constants, species presets, solver param factories — shared across all new scripts

### Established Patterns
- Frozen dataclass config + result pattern for all strategies (CascadeConfig/Result, etc.)
- `StudyResults/` subdirectories for experiment outputs (CSV + PNG)
- Print-based logging with `[tag]` prefixes and `verbose` flag
- Central FD gradients with h=1e-5 for surrogate objectives

### Integration Points
- New wrapper script calls v13 pipeline via subprocess or direct import
- Profile likelihood script uses FluxCurve PDE objective directly
- Sensitivity scripts use Forward solver directly (build_context → build_forms → solve)
- All outputs land in `StudyResults/v14/` for downstream phase consumption

</code_context>

<specifics>
## Specific Ideas

- Extended voltage sweep must use voltage warm-starting and bridge points that are already implemented — the PDE solver will struggle at extreme voltages without continuation
- Give the solver "more rope" at extended voltages: more SNES iterations, smaller dt if needed for convergence
- Profile likelihood is PDE-only (gold standard) — no surrogate shortcuts for identifiability assessment

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 07-baseline-diagnostics*
*Context gathered: 2026-03-10*
