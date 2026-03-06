# Requirements: PNP-BV V&V Framework

**Defined:** 2026-03-06
**Core Value:** Every layer of the pipeline has independently verifiable proof of correctness that can withstand peer review.

## v1 Requirements

### Forward Solver Verification

- [ ] **FWD-01**: MMS convergence tests wrapped in pytest with automated rate assertions (L2 ~ O(h^2), H1 ~ O(h))
- [x] **FWD-02**: MMS weak form audit confirming MMS tests the production `bv_solver.py` weak form, not a hand-built replica
- [ ] **FWD-03**: 4-species MMS case matching the v13 production configuration (O2, H2O2, H+, ClO4-)
- [x] **FWD-04**: Nondimensionalization roundtrip tests verifying physical -> nondim -> physical identity for all parameter types
- [ ] **FWD-05**: Mesh convergence study with Grid Convergence Index (GCI) uncertainty quantification

### Surrogate Verification

- [ ] **SUR-01**: Surrogate fidelity map using LHS-sampled parameter sets across the v13 inference domain
- [ ] **SUR-02**: Hold-out validation testing the v13 surrogate on unseen parameter sets (not training data)
- [ ] **SUR-03**: Error bound quantification (max, mean, percentile errors) for the v13 surrogate

### Inverse Problem Verification

- [ ] **INV-01**: Parameter recovery from v13 synthetic data at multiple noise levels (0%, 1%, 2%, 5%)
- [ ] **INV-02**: Gradient consistency verification (finite-difference vs adjoint) for the v13 objective function
- [ ] **INV-03**: Multistart convergence basin analysis showing the v13 optimizer finds the correct minimum

### Pipeline & Reproducibility

- [ ] **PIP-01**: End-to-end v13 reproducibility test: same inputs produce same outputs across runs
- [ ] **PIP-02**: Numerical regression baselines with saved reference values to catch future breakage

### Report

- [ ] **RPT-01**: Written V&V report with convergence tables, convergence plots, and GCI uncertainty bounds suitable for journal supplementary material

## v2 Requirements

### Forward Solver (Deferred)

- **FWD-06**: Boundary layer resolution study at physical-scale Debye lengths (eps_hat ~ 1e-8)
- **FWD-07**: Mesh convergence with Richardson extrapolation for solution error estimation

### Surrogate (Deferred)

- **SUR-04**: Error heatmaps showing spatial degradation patterns across parameter space
- **SUR-05**: Comparative accuracy analysis between RBF and NN ensemble surrogates

### Inverse Problem (Deferred)

- **INV-04**: Fisher information matrix / sensitivity analysis for parameter identifiability
- **INV-05**: Error-vs-noise publication plots with confidence intervals

### Infrastructure (Deferred)

- **INF-01**: CI/CD pipeline for automated V&V test execution
- **INF-02**: Performance benchmarking with pytest-benchmark

## Out of Scope

| Feature | Reason |
|---------|--------|
| Verification of old inference scripts (v1-v12) | Only v13 is the canonical pipeline; old scripts are frozen experiments |
| Codebase refactoring (script sprawl, sys.path hacks) | This project is about verification, not cleanup |
| New physics or solver capabilities | Verify what exists first |
| Experimental data validation | Code verification (solving equations right), not model validation (right equations) |
| Performance optimization | Correctness before speed |
| Structured logging overhaul | Out of scope for V&V |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| FWD-01 | Phase 2 | Pending |
| FWD-02 | Phase 1 | Complete |
| FWD-03 | Phase 2 | Pending |
| FWD-04 | Phase 1 | Complete |
| FWD-05 | Phase 2 | Pending |
| SUR-01 | Phase 3 | Pending |
| SUR-02 | Phase 3 | Pending |
| SUR-03 | Phase 3 | Pending |
| INV-01 | Phase 4 | Pending |
| INV-02 | Phase 4 | Pending |
| INV-03 | Phase 4 | Pending |
| PIP-01 | Phase 5 | Pending |
| PIP-02 | Phase 5 | Pending |
| RPT-01 | Phase 6 | Pending |

**Coverage:**
- v1 requirements: 14 total
- Mapped to phases: 14
- Unmapped: 0

---
*Requirements defined: 2026-03-06*
*Last updated: 2026-03-06 after roadmap creation*
