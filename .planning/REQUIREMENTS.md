# Requirements: PNP-BV Inverse Solver

**Defined:** 2026-03-10
**Core Value:** Robust parameter recovery (<10% relative error) at 2% noise across all seeds, with every pipeline component justified.

## v14 Requirements

Requirements for pipeline redesign milestone. Each maps to roadmap phases.

### Diagnostics

- [x] **DIAG-01**: Run v13 pipeline across 10+ noise seeds at 2% noise, report per-parameter median/worst-case relative error
- [ ] **DIAG-02**: Profile likelihood analysis for each of k0_1, k0_2, alpha_1, alpha_2 to determine practical identifiability
- [ ] **DIAG-03**: Extended voltage sweep visualization of total and peroxide current across parameter values for visual sensitivity analysis

### Audit

- [ ] **AUDT-01**: Ablation study removing S3/S4 surrogate stages, quantifying per-stage contribution across 10+ seeds
- [ ] **AUDT-02**: Ablation study of P1 (shallow PDE), quantifying contribution vs direct S2→P2
- [ ] **AUDT-03**: Document justification status of each v13 stage against the 3 criteria (literature, empirical best, simplest)
- [x] **AUDT-04**: Continuous justification audit — every new component added in later phases must pass the 3-criteria test before inclusion

### Objective Function

- [ ] **OBJF-01**: Implement weighted least squares objective with per-point normalization replacing raw squared residuals
- [ ] **OBJF-02**: Implement Tikhonov regularization on PDE objective phases with lambda selection
- [ ] **OBJF-03**: Sensitivity-weighted voltage selection using Fisher Information Matrix at surrogate optimum

### Surrogate

- [ ] **SURR-01**: Surrogate bias correction using PDE evaluations near surrogate optimum (space mapping approach)

### Multi-pH

- [ ] **PHEX-01**: Explore multi-pH experimental design by varying bulk H+ and anion concentrations as additional constraints
- [ ] **PHEX-02**: Assess whether multi-pH data breaks parameter correlations that limit single-condition identifiability

### Pipeline Build

- [ ] **PIPE-01**: Build redesigned pipeline incorporating validated components from audit and comparison phases
- [ ] **PIPE-02**: Robustness validation of v14 pipeline: <10% relative error on all 4 parameters across 20+ seeds at 2% noise

## Future Requirements

Deferred beyond v14. Tracked but not in current roadmap.

### Surrogate Retraining

- **SURR-F01**: Retrain surrogate models with additional training data after PDE pipeline and observables are finalized

### Extended Noise Testing

- **NOIS-F01**: Extend robustness validation to 0%, 1%, 5% noise levels
- **NOIS-F02**: Characterize pipeline degradation curve as noise increases

### Bayesian Inference

- **BAYES-F01**: Bayesian inference with MCMC using surrogate as forward model for posterior distributions
- **BAYES-F02**: Credible intervals and parameter correlation analysis

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| More surrogate phases (S5+) | S2/S3/S4 converge to same minimum -- proven redundant |
| Global optimization on PDE objective | Too expensive (~3h per seed); surrogate multistart handles global search |
| Deep learning surrogate replacement | NN ensemble already 0.06-0.41% NRMSE; bottleneck is bias, not accuracy |
| Noise levels beyond 2% | Target 2% first, extend in future milestone |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| DIAG-01 | Phase 7 | Complete |
| DIAG-02 | Phase 7 | Pending |
| DIAG-03 | Phase 7 | Pending |
| AUDT-01 | Phase 8 | Pending |
| AUDT-02 | Phase 8 | Pending |
| AUDT-03 | Phase 8 | Pending |
| AUDT-04 | Phase 7, 8, 9, 10 (cross-cutting) | Complete |
| OBJF-01 | Phase 9 | Pending |
| OBJF-02 | Phase 9 | Pending |
| OBJF-03 | Phase 9 | Pending |
| SURR-01 | Phase 9 | Pending |
| PHEX-01 | Phase 10 | Pending |
| PHEX-02 | Phase 10 | Pending |
| PIPE-01 | Phase 11 | Pending |
| PIPE-02 | Phase 11 | Pending |

**Coverage:**
- v14 requirements: 15 total
- Mapped to phases: 15
- Unmapped: 0

---
*Requirements defined: 2026-03-10*
*Last updated: 2026-03-10 after roadmap creation*
