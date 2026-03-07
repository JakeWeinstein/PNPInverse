# Roadmap: PNP-BV V&V Framework

## Overview

Bottom-up verification of the PNP-BV electrochemical inference pipeline, starting from dimensional analysis correctness, through forward solver MMS convergence, surrogate fidelity, inverse problem recovery, and pipeline reproducibility, culminating in a publication-grade V&V report. Each layer is verified independently before trusting the next layer up.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Nondimensionalization & Weak Form Audit** - Verify dimensional transforms and confirm MMS tests the production weak form (completed 2026-03-06)
- [ ] **Phase 2: MMS Convergence Verification** - Automated convergence rate tests with GCI for the forward PDE solver
- [ ] **Phase 3: Surrogate Fidelity** - Error characterization of v13 surrogate across parameter space
- [ ] **Phase 4: Inverse Problem Verification** - Parameter recovery, gradient consistency, and optimizer convergence for v13
- [ ] **Phase 5: Pipeline Reproducibility** - End-to-end determinism and numerical regression baselines
- [ ] **Phase 6: V&V Report** - Publication-grade written report with convergence plots and error tables

## Phase Details

### Phase 1: Nondimensionalization & Weak Form Audit
**Goal**: Dimensional analysis is proven correct and MMS is confirmed to test the actual production solver
**Depends on**: Nothing (first phase)
**Requirements**: FWD-02, FWD-04
**Success Criteria** (what must be TRUE):
  1. Running `pytest` executes nondimensionalization roundtrip tests that convert physical parameters to nondim and back, recovering original values within machine epsilon for all parameter types (diffusivities, concentrations, potentials, rate constants)
  2. A documented audit confirms MMS convergence script uses the same weak form as production `bv_solver.py`, or differences are identified and resolved
  3. All nondim and audit tests pass without manual intervention
**Plans:** 3/3 plans complete

Plans:
- [x] 01-01-PLAN.md — Textbook audit of existing nondim tests + roundtrip tests for all parameter types
- [x] 01-02-PLAN.md — MMS weak form refactor to use production build_forms() + audit document + smoke test
- [ ] 01-03-PLAN.md — Gap closure: add skip_without_firedrake guard to TestBVScalingRoundtrip

### Phase 2: MMS Convergence Verification
**Goal**: Forward PDE solver has automated, publication-grade convergence proof with uncertainty quantification
**Depends on**: Phase 1
**Requirements**: FWD-01, FWD-03, FWD-05
**Success Criteria** (what must be TRUE):
  1. Running `pytest` executes MMS convergence tests that assert L2 convergence rate ~ O(h^2) and H1 ~ O(h) with R-squared > 0.99 on log-log fit
  2. A 4-species MMS case (O2, H2O2, H+, ClO4-) matching the v13 production configuration passes convergence rate assertions
  3. Grid Convergence Index (GCI) uncertainty bounds are computed and available as test output for each mesh refinement level
  4. All convergence tests run as standard pytest tests without manual script execution
**Plans:** 1 plan

Plans:
- [ ] 02-01-PLAN.md — 4-species MMS convergence test with rate assertions, GCI output, and deprecated script cleanup

### Phase 3: Surrogate Fidelity
**Goal**: v13 surrogate model error is characterized across the inference parameter domain
**Depends on**: Phase 2
**Requirements**: SUR-01, SUR-02, SUR-03
**Success Criteria** (what must be TRUE):
  1. A fidelity map exists showing surrogate-vs-PDE error at LHS-sampled parameter sets spanning the v13 inference domain (not just training points)
  2. Hold-out validation demonstrates surrogate accuracy on parameter sets not used during training
  3. Error statistics (max, mean, 95th percentile relative error) are computed and saved as referenceable test output
**Plans**: TBD

Plans:
- [ ] 03-01: TBD
- [ ] 03-02: TBD

### Phase 4: Inverse Problem Verification
**Goal**: v13 parameter inference is proven to recover known parameters from synthetic data
**Depends on**: Phase 2
**Requirements**: INV-01, INV-02, INV-03
**Success Criteria** (what must be TRUE):
  1. Parameter recovery tests infer known parameters from synthetic v13 data at noise levels 0%, 1%, 2%, 5% and report relative error at each level
  2. Gradient consistency tests show finite-difference and adjoint gradients agree within a defined tolerance for the v13 objective function
  3. Multistart analysis demonstrates the v13 optimizer converges to the correct minimum from multiple initial guesses, with convergence basin statistics
**Plans**: TBD

Plans:
- [ ] 04-01: TBD
- [ ] 04-02: TBD

### Phase 5: Pipeline Reproducibility
**Goal**: v13 pipeline produces deterministic, regression-tested results
**Depends on**: Phase 3, Phase 4
**Requirements**: PIP-01, PIP-02
**Success Criteria** (what must be TRUE):
  1. Running the same v13 inference inputs twice produces bitwise-identical (or within solver tolerance) outputs
  2. Saved numerical regression baselines exist and pytest automatically compares current outputs against them, failing if values drift beyond tolerance
**Plans**: TBD

Plans:
- [ ] 05-01: TBD

### Phase 6: V&V Report
**Goal**: Publication-grade written evidence of pipeline correctness suitable for journal supplementary material
**Depends on**: Phase 2, Phase 3, Phase 4, Phase 5
**Requirements**: RPT-01
**Success Criteria** (what must be TRUE):
  1. A written report exists containing convergence tables, convergence plots, GCI uncertainty bounds, surrogate error statistics, and parameter recovery results
  2. All figures and tables in the report are generated from actual test outputs (not manually constructed)
  3. The report is formatted for inclusion as a journal appendix or supplementary material
**Plans**: TBD

Plans:
- [ ] 06-01: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5 -> 6

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Nondimensionalization & Weak Form Audit | 3/3 | Complete   | 2026-03-06 |
| 2. MMS Convergence Verification | 0/1 | Not started | - |
| 3. Surrogate Fidelity | 0/2 | Not started | - |
| 4. Inverse Problem Verification | 0/2 | Not started | - |
| 5. Pipeline Reproducibility | 0/1 | Not started | - |
| 6. V&V Report | 0/1 | Not started | - |
