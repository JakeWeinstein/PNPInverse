# PNP-BV Inverse Solver

## What This Is

A surrogate-accelerated inverse solver for Poisson-Nernst-Planck + Butler-Volmer electrochemical systems. The solver recovers kinetic parameters (k0_1, k0_2, alpha_1, alpha_2) from noisy I-V curve observations. Built on a V&V-verified forward solver and surrogate models, with the goal of robust parameter recovery across noise realizations.

## Core Value

Robust parameter recovery (<10% relative error) at 2% noise across all seeds, with every pipeline component justified by literature, empirical comparison, or simplicity.

## Current Milestone: v14 Pipeline Redesign

**Goal:** Systematically audit and redesign the v13 inference pipeline for robust parameter recovery, justifying every component.

**Target features:**
- Seed-by-seed performance baseline of v13 pipeline at 2% noise
- Stage-by-stage audit against 3 justification criteria (literature, empirical best, simplest)
- Identification and removal of redundant stages
- Brainstorm and prototype alternative approaches (objective, optimizer, surrogate, regularization)
- Build redesigned pipeline that beats v13 on robustness

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

- [ ] v13 performance baseline across noise seeds at 2% noise
- [ ] Stage-by-stage audit of v13 pipeline against justification criteria
- [ ] Empirical comparison studies for each pipeline component
- [ ] Redesigned pipeline implementation
- [ ] Robustness validation (<10% relative error on k0_1, k0_2, alpha_1, alpha_2 at 2% noise)

### Out of Scope

- Refactoring the existing codebase (script sprawl, sys.path hacks, etc.) — focus is on pipeline design, not code cleanup
- Adding new physics or solver capabilities — work with existing PDE model
- CI/CD pipeline setup — focus on the pipeline itself
- Experimental data validation — synthetic targets only for now
- Noise levels beyond 2% — target 2% first, extend later
- Performance optimization for speed — correctness and robustness first

## Context

- V1.0 shipped 2026-03-10: V&V framework complete (6 phases, 14 plans, 69 commits)
- Forward solver verified correct via MMS (L2~O(h²), H1~O(h))
- Surrogate models validated: CD median NRMSE 0.06-0.41% on 479 hold-out samples
- v13 pipeline has 7 stages (P1-P7) with surrogate and PDE refinement phases
- Known: ~11% surrogate bias from PDE truth at optimum
- Known: Multiple surrogate phases (P2-P4) likely converge to same minimum — redundant
- Known: 20K LHS multistart reliably finds global minimum — keep
- Suspicion: P1 (shallow surrogate?) may not be necessary — needs investigation
- Parameters of interest: k0_1, k0_2, alpha_1, alpha_2

## Constraints

- **Runtime**: Firedrake PDE solves are slow (seconds per point). Surrogate evaluations are fast.
- **Environment**: macOS/Linux with Firedrake virtual environment. PyTorch for NN surrogate.
- **Justification standard**: Every pipeline component must be justified by (1) literature precedent, (2) empirical superiority over alternatives, or (3) simplest thing that works (rare).
- **Standard**: Results must be defensible to journal reviewers with quantitative evidence.

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
*Last updated: 2026-03-09 after v14 milestone start*
