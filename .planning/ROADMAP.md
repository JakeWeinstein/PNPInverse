# Roadmap: PNP-BV Inverse Solver

## Milestones

- v1.0 V&V Framework (Phases 1-6) -- shipped 2026-03-10
- v14 Pipeline Redesign (Phases 7-11) -- in progress

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)
- v1.0 completed phases 1-6; v14 starts at phase 7

<details>
<summary>v1.0 V&V Framework (Phases 1-6) -- SHIPPED 2026-03-10</summary>

See .planning/MILESTONES.md for v1.0 details.
6 phases, 14 plans, 69 commits.

</details>

### v14 Pipeline Redesign (In Progress)

**Milestone Goal:** Systematically audit and redesign the v13 inference pipeline for robust parameter recovery (<10% relative error on k0_1, k0_2, alpha_1, alpha_2 at 2% noise across 20+ seeds), with every component justified.

- [x] **Phase 7: Baseline Diagnostics** - Establish v13 multi-seed performance baseline and parameter identifiability bounds (completed 2026-03-10)
- [ ] **Phase 8: Ablation Audit** - Determine which v13 stages actually contribute to robustness via controlled ablation
- [ ] **Phase 9: Objective and Component Experiments** - Redesign objective function and test surrogate bias correction
- [ ] **Phase 10: Multi-pH Exploration** - Explore whether multi-pH experimental design breaks parameter correlations
- [ ] **Phase 11: Pipeline Build and Validation** - Assemble and validate the redesigned v14 pipeline

## Phase Details

### Phase 7: Baseline Diagnostics
**Goal**: Quantify v13 pipeline performance across noise seeds and determine which parameters are practically identifiable at 2% noise
**Depends on**: v1.0 complete (phases 1-6)
**Requirements**: DIAG-01, DIAG-02, DIAG-03, AUDT-04
**Success Criteria** (what must be TRUE):
  1. v13 pipeline results exist for 10+ noise seeds at 2% noise, with per-parameter median and worst-case relative error reported in a CSV
  2. Profile likelihood plots exist for each of k0_1, k0_2, alpha_1, alpha_2, showing whether each parameter has a well-defined minimum or a flat valley
  3. Extended voltage sweep plots show total and peroxide current sensitivity to each parameter, revealing which voltage regions carry information about which parameters
  4. Every diagnostic tool introduced in this phase has a justification entry (literature, empirical, or simplest) documented in its output
**Plans**: 3 plans

Plans:
- [ ] 07-01-PLAN.md -- Multi-seed v13 wrapper with summary statistics and visualizations (DIAG-01, AUDT-04)
- [ ] 07-02-PLAN.md -- PDE-only profile likelihood identifiability analysis (DIAG-02, AUDT-04)
- [ ] 07-03-PLAN.md -- Sensitivity visualization with 1D sweeps and Jacobian heatmap (DIAG-03, AUDT-04)

### Phase 8: Ablation Audit
**Goal**: Identify which v13 pipeline stages are necessary vs redundant, producing an empirically justified minimal pipeline specification
**Depends on**: Phase 7
**Requirements**: AUDT-01, AUDT-02, AUDT-03, AUDT-04
**Success Criteria** (what must be TRUE):
  1. Ablation table shows per-stage contribution of S3/S4 across 10+ seeds, with statistical test (Wilcoxon) determining whether their removal changes median or worst-case error
  2. Ablation table shows P1 (shallow PDE) contribution, quantifying whether direct S2-to-P2 transition matches or improves on the S2-to-P1-to-P2 path
  3. Every v13 stage has a justification status entry (justified / unjustified / redundant) with supporting evidence
  4. A minimal pipeline specification exists listing only the stages that survived ablation, ready for Phase 9 experiments
**Plans**: TBD

Plans:
- [ ] 08-01: TBD
- [ ] 08-02: TBD

### Phase 9: Objective and Component Experiments
**Goal**: Improve parameter recovery by redesigning the objective function, adding regularization, and correcting surrogate bias -- each change tested individually against the ablated baseline
**Depends on**: Phase 8
**Requirements**: OBJF-01, OBJF-02, OBJF-03, SURR-01, AUDT-04
**Success Criteria** (what must be TRUE):
  1. Weighted least squares objective with per-point normalization exists and is tested across 10+ seeds, with comparison table showing improvement (or not) over raw squared residuals
  2. Tikhonov regularization on PDE objective exists with lambda selected by L-curve or cross-validation, tested across 10+ seeds showing reduced worst-case error relative to unregularized PDE
  3. Fisher Information Matrix analysis at surrogate optimum identifies which voltage points are most informative, and sensitivity-weighted voltage selection is tested against uniform voltage grid
  4. Surrogate bias correction (space mapping) using PDE evaluations near surrogate optimum is implemented and tested, showing reduced systematic bias on k0_2
  5. Every new component introduced has a justification entry documenting why it was included (literature, empirical superiority, or simplest approach)
**Plans**: TBD

Plans:
- [ ] 09-01: TBD
- [ ] 09-02: TBD
- [ ] 09-03: TBD

### Phase 10: Multi-pH Exploration
**Goal**: Determine whether varying bulk pH and anion concentrations provides additional constraints that break parameter correlations limiting single-condition identifiability
**Depends on**: Phase 7 (identifiability results), Phase 8 (minimal pipeline)
**Requirements**: PHEX-01, PHEX-02, AUDT-04
**Success Criteria** (what must be TRUE):
  1. Multi-pH I-V curves exist for at least 3 distinct bulk H+ / anion concentration conditions, generated from the forward solver
  2. Profile likelihood or FIM analysis at multiple pH conditions shows whether previously correlated parameters become independently identifiable
  3. If multi-pH breaks correlations: a joint-fitting objective combining multiple conditions is prototyped and tested on synthetic data; if it does not: the result is documented with quantitative evidence
**Plans**: TBD

Plans:
- [ ] 10-01: TBD
- [ ] 10-02: TBD

### Phase 11: Pipeline Build and Validation
**Goal**: Assemble the validated components from phases 7-10 into a final v14 pipeline and demonstrate robust parameter recovery across 20+ seeds at 2% noise
**Depends on**: Phase 9, Phase 10
**Requirements**: PIPE-01, PIPE-02
**Success Criteria** (what must be TRUE):
  1. v14 pipeline exists as a runnable script or config-driven system incorporating only the components that survived audit and showed empirical improvement
  2. v14 pipeline achieves <10% median relative error on all 4 parameters (k0_1, k0_2, alpha_1, alpha_2) across 20+ seeds at 2% noise
  3. v14 vs v13 comparison table exists showing per-parameter improvement with statistical significance (paired test across seeds)
  4. If k0_2 was found non-identifiable in Phase 7, the pipeline reports uncertainty bounds rather than point estimates for that parameter, and the <10% target applies to the identifiable parameters only
**Plans**: TBD

Plans:
- [ ] 11-01: TBD
- [ ] 11-02: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 7 -> 8 -> 9 -> 10 -> 11
(Note: Phase 10 depends on Phases 7+8 but not 9; could run in parallel with Phase 9 if desired)

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 7. Baseline Diagnostics | 3/3 | Complete   | 2026-03-10 |
| 8. Ablation Audit | 0/2 | Not started | - |
| 9. Objective and Component Experiments | 0/3 | Not started | - |
| 10. Multi-pH Exploration | 0/2 | Not started | - |
| 11. Pipeline Build and Validation | 0/2 | Not started | - |
