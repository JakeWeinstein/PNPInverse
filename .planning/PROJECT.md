# PNP-BV Verification & Validation Framework

## What This Is

A publication-grade verification and validation (V&V) framework for an existing Poisson-Nernst-Planck + Butler-Volmer electrochemical inference pipeline. The codebase has a multi-layer architecture — PDE forward solver, surrogate models, and parameter inference — all built rapidly and needing systematic proof of correctness at every layer.

## Core Value

Every layer of the pipeline (forward solver, surrogate, inference) has independently verifiable proof of correctness that can withstand peer review.

## Requirements

### Validated

- Forward PDE solver (PNP + Butler-Volmer) — existing
- Nondimensionalization layer with unit transforms — existing
- RBF and NN surrogate models for I-V curve approximation — existing
- 7-phase surrogate-to-PDE inference pipeline (v13) — existing
- Basic test suite with pytest (nondim, ensemble, v13 end-to-end) — existing
- MMS convergence verification script (exists but unvalidated) — existing

### Active

- [ ] Verified Method of Manufactured Solutions (MMS) implementation for the PNP-BV forward solver
- [ ] Mesh convergence study with expected convergence rates for finite element order
- [ ] Surrogate fidelity analysis: error bounds between surrogate predictions and PDE ground truth
- [ ] Parameter recovery tests: infer known parameters from synthetic data, measure accuracy
- [ ] End-to-end pipeline reproducibility: same inputs produce same outputs across runs
- [ ] Automated V&V test suite that runs without manual intervention
- [ ] Written verification report with convergence plots, error tables, and benchmark results suitable for a journal appendix or supplementary material

### Out of Scope

- Refactoring the existing codebase (script sprawl, sys.path hacks, etc.) — this project is about verification, not cleanup
- Adding new physics or solver capabilities — verify what exists first
- Performance optimization — correctness before speed
- CI/CD pipeline setup — focus on the V&V framework itself
- Experimental data validation — this is code verification (are we solving the equations right?), not model validation against lab data

## Context

- The entire codebase was built quickly with AI assistance, including an MMS reference that hasn't been independently verified
- The pipeline has multiple layers where errors can compound: nondimensionalization -> PDE assembly -> nonlinear solve -> surrogate fitting -> optimization -> parameter estimates
- Firedrake provides automatic adjoint differentiation, which is powerful but adds another layer that could introduce subtle bugs
- The nondimensionalization layer had a past bug (hardcoded value that should have been physical) — dimensional analysis errors produce wrong results silently
- No structured logging exists; diagnostic output is all print statements
- The v13 pipeline is the canonical inference script; older versions are frozen experiments

## Constraints

- **Runtime**: Firedrake is required and installed via its own installer (not pip). Tests involving PDE solves are slow (seconds per point).
- **Environment**: macOS/Linux with Firedrake virtual environment. PyTorch optional for NN surrogate.
- **Scope**: Verify the existing pipeline as-is. Don't change solver code unless a bug is found during verification.
- **Standard**: Publication-grade rigor — convergence rates, error norms, reproducibility. Results must be defensible to journal reviewers.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Verify MMS implementation first | MMS is the foundation — if the forward solver isn't right, nothing downstream matters | -- Pending |
| Layer-by-layer verification (bottom-up) | Errors compound through layers; verify each independently before testing the pipeline | -- Pending |
| Both automated tests and written report | Tests catch regressions; report provides citable evidence for publications | -- Pending |

---
*Last updated: 2026-03-06 after initialization*
