# Phase 6: V&V Report - Context

**Gathered:** 2026-03-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Publication-grade standalone V&V report documenting pipeline correctness evidence across all layers (forward solver, surrogate, inverse, pipeline). All figures and tables generated programmatically from existing JSON/CSV test outputs in `StudyResults/`. No new tests or verification work — this phase is purely report generation.

</domain>

<decisions>
## Implementation Decisions

### Document format
- LaTeX standalone V&V report (not appendix, not Jupyter)
- Self-contained document with introduction, pipeline architecture, per-layer verification sections, and conclusions
- Located at `writeups/vv_report/vv_report.tex`

### Report structure
1. Introduction (with governing PNP-BV equations)
2. Pipeline Architecture (with TikZ flowchart: PDE solver -> surrogate -> optimizer)
3. Forward Solver Verification (MMS method, convergence results, GCI analysis)
4. Surrogate Fidelity (hold-out validation, error statistics)
5. Inverse Problem Verification (parameter recovery, gradient consistency, multistart basin)
6. Pipeline Reproducibility (determinism, regression baselines)
7. Summary (synthesis table + prose stating overall verification confidence)

### Figure generation
- Fresh `writeups/vv_report/generate_figures.py` script (not extending existing presentation plots script)
- Reads all data from `StudyResults/` JSON/CSV files
- Outputs PDF vector figures to `writeups/vv_report/figures/`
- Essential figures:
  - MMS convergence log-log plot (L2/H1 error vs mesh size, 5 fields, reference slope lines)
  - Parameter recovery vs noise (median max error at 0%, 1%, 2%, 5% with gate thresholds)
  - Worst-case I-V overlay (surrogate vs PDE at worst-case parameter samples)
- Surrogate fidelity presented as table only (no figure needed)

### Table generation
- Auto-generated LaTeX table snippets via the same Python script
- Output to `writeups/vv_report/tables/*.tex`
- Report uses `\input{tables/mms_rates.tex}` etc.
- Tables:
  - MMS convergence: rates + R² + GCI at finest mesh (compact, 5 rows)
  - Surrogate fidelity: median/95th/max NRMSE for each model, CD and PC columns
  - Parameter recovery: noise level, median max error, gate threshold, pass/fail
  - Gradient consistency: FD convergence rates, PDE agreement
  - Summary: layer × test × key result × status

### Narrative depth
- Brief methodology per section (1-2 paragraphs explaining the verification approach + references)
- Not minimal captions, not full textbook — enough for a reviewer to understand without prior V&V knowledge
- Governing PNP-BV equations included in introduction/architecture section
- Summary section with synthesis table + concluding prose

### Claude's Discretion
- Publication figure styling (serif fonts, color palette, axis labels)
- TikZ pipeline architecture diagram design
- Exact LaTeX document class and package choices
- Table formatting details (significant figures, column alignment)
- Bibliography entries and citation style
- Page length and layout decisions

</decisions>

<specifics>
## Specific Ideas

- Summary table format: Layer | Test | Key Result | Status (pass/fail) — strong closer for reviewers
- MMS convergence table preview shown and approved: Species | L2 Rate | L2 R² | H1 Rate | H1 R² | GCI(finest)
- Parameter recovery table: must note the ~11% surrogate bias floor at 0% noise
- 5% noise level marked as "informational" (not a pass/fail gate) since it exceeds surrogate approximation limits

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `scripts/generate_presentation_plots.py`: Existing matplotlib plotting for presentations — different styling goals but data loading patterns are reusable
- `StudyResults/mms_convergence/convergence_data.json`: MMS convergence data (4 species + phi, L2/H1 errors, rates, R², GCI)
- `StudyResults/surrogate_fidelity/fidelity_summary.json`: Surrogate error stats (4 models × CD/PC × max/mean/median/95th NRMSE)
- `StudyResults/inverse_verification/parameter_recovery_summary.json`: Parameter recovery at 4 noise levels
- `StudyResults/inverse_verification/gradient_fd_convergence.json`: FD gradient convergence data
- `StudyResults/inverse_verification/gradient_pde_consistency.json`: PDE gradient consistency data
- `StudyResults/inverse_verification/multistart_basin.json`: Multistart basin analysis
- `StudyResults/pipeline_reproducibility/regression_baselines.json`: Pipeline reproducibility baselines

### Established Patterns
- All V&V data stored as JSON in `StudyResults/` subdirectories
- Tests produce JSON artifacts consumed by other phases — report consumes all of them
- Matplotlib used throughout for plotting

### Integration Points
- Report reads only from `StudyResults/` (no test execution needed)
- `writeups/vv_report/` is the self-contained report directory
- Pipeline: `generate_figures.py` -> `figures/` + `tables/` -> `\input{}` in `vv_report.tex` -> PDF

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 06-v-v-report*
*Context gathered: 2026-03-09*
