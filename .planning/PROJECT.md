# PNP-BV Verification & Validation Framework

## What This Is

A publication-grade verification and validation (V&V) framework for a Poisson-Nernst-Planck + Butler-Volmer electrochemical inference pipeline. The framework provides bottom-up proof of correctness at every layer — nondimensionalization, forward PDE solver, surrogate models, and parameter inference — with automated pytest tests, numerical regression baselines, and a LaTeX V&V report suitable for journal supplementary material.

## Core Value

Every layer of the pipeline (forward solver, surrogate, inference) has independently verifiable proof of correctness that can withstand peer review.

## Requirements

### Validated

- Forward PDE solver (PNP + Butler-Volmer) — existing
- Nondimensionalization layer with unit transforms — existing
- RBF and NN surrogate models for I-V curve approximation — existing
- 7-phase surrogate-to-PDE inference pipeline (v13) — existing
- Basic test suite with pytest (nondim, ensemble, v13 end-to-end) — existing
- ✓ Verified MMS implementation with production weak form audit — v1.0
- ✓ 4-species MMS convergence test with L2~O(h²), H1~O(h) rate assertions and GCI — v1.0
- ✓ Surrogate fidelity validation across 479 hold-out samples for all 4 models — v1.0
- ✓ Parameter recovery from synthetic PDE targets at 4 noise levels (0-5%) — v1.0
- ✓ Gradient consistency verification (FD convergence) for surrogate and PDE objectives — v1.0
- ✓ Multistart convergence basin analysis with 20K LHS grid — v1.0
- ✓ End-to-end pipeline reproducibility with numerical regression baselines — v1.0
- ✓ Automated V&V test suite running via pytest without manual intervention — v1.0
- ✓ Publication-grade LaTeX V&V report with programmatic figures/tables — v1.0

### Active

(None yet — define in next milestone)

### Out of Scope

- Refactoring the existing codebase (script sprawl, sys.path hacks, etc.) — this project is about verification, not cleanup
- Adding new physics or solver capabilities — verify what exists first
- Performance optimization — correctness before speed
- CI/CD pipeline setup — focus on the V&V framework itself
- Experimental data validation — this is code verification (are we solving the equations right?), not model validation against lab data

## Context

- V1.0 shipped 2026-03-10: 6 phases, 14 plans, 69 commits across 4 days
- Codebase: 72,234 lines Python; V&V adds ~12,300 lines (tests, report pipeline, LaTeX)
- MMS weak form audit confirmed production `bv_solver.py` is correct (no bugs found)
- Nondimensionalization roundtrip tests confirmed all parameter transforms at 1e-12 tolerance
- Surrogate models validated on 479 hold-out samples; CD median NRMSE 0.06-0.41%
- Parameter recovery works at up to 5% noise with ~11% surrogate bias from PDE truth (expected)
- Full pipeline reproducibility confirmed via subprocess-based regression tests
- Key tech debt: sys.path manipulation in test imports (info-level), test_inverse_verification.py at 1136 lines

## Constraints

- **Runtime**: Firedrake is required and installed via its own installer (not pip). Tests involving PDE solves are slow (seconds per point).
- **Environment**: macOS/Linux with Firedrake virtual environment. PyTorch optional for NN surrogate.
- **Scope**: Verify the existing pipeline as-is. Don't change solver code unless a bug is found during verification.
- **Standard**: Publication-grade rigor — convergence rates, error norms, reproducibility. Results must be defensible to journal reviewers.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Verify MMS implementation first | MMS is the foundation — if the forward solver isn't right, nothing downstream matters | ✓ Good — confirmed production weak form correct |
| Layer-by-layer verification (bottom-up) | Errors compound through layers; verify each independently before testing the pipeline | ✓ Good — each layer verified before trusting the next |
| Both automated tests and written report | Tests catch regressions; report provides citable evidence for publications | ✓ Good — pytest suite + LaTeX report both complete |
| MMS thin wrapper (call production build_forms) | Test the actual production code, not a hand-built replica | ✓ Good — audit confirmed MMS = production weak form |
| Surrogate-only inference for parameter recovery | Full 7-phase pipeline too slow for pytest; surrogate S1+S2 sufficient | ✓ Good — fast tests with meaningful recovery results |
| PDE targets via subprocess | Firedrake/PyTorch PETSc segfault when both loaded in same process | ✓ Good — subprocess isolation resolved segfault |
| Median NRMSE for surrogate gates | PC near-zero-range samples inflate mean to 50-200% | ✓ Good — median gives stable, meaningful thresholds |
| Multistart as basin uniqueness (CV<0.10) | "Within 10% of true" not achievable with ~11% surrogate bias | ✓ Good — tests optimizer convergence, not surrogate accuracy |

---
*Last updated: 2026-03-10 after v1.0 milestone*
