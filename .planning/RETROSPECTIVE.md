# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

## Milestone: v1.0 — PNP-BV V&V Framework

**Shipped:** 2026-03-10
**Phases:** 6 | **Plans:** 14 | **Timeline:** 4 days

### What Was Built
- Bottom-up V&V test suite: nondim roundtrips, MMS convergence, surrogate fidelity, parameter recovery, pipeline reproducibility
- Automated pytest suite with ~12,300 lines of new test/report code
- Publication-grade LaTeX V&V report with programmatic figure/table generation from StudyResults/ data
- Numerical regression baselines with --update-baselines conftest plugin

### What Worked
- Bottom-up verification order (nondim -> MMS -> surrogate -> inverse -> pipeline) caught issues at the right layer
- MMS thin wrapper approach (calling production build_forms) confirmed actual solver correctness without code duplication
- Subprocess isolation for PDE targets resolved Firedrake/PyTorch PETSc segfault cleanly
- Gap closure plans (04-03, 04-04) efficiently fixed inverse crime and relaxed gates after initial implementation revealed surrogate bias

### What Was Inefficient
- Phase 4 required 4 plans (most of any phase) — initial test gates were too tight for surrogate approximation error, requiring two gap closure rounds
- Milestone audit was created before Phases 5-6 executed, making the audit stale by completion time
- ROADMAP.md Phase 3 checkbox was never updated to [x] despite phase completion (stale tracking)

### Patterns Established
- Surrogate gates should use median (not mean) for metrics with outlier-prone distributions
- PDE-based tests need subprocess isolation when PyTorch is also loaded
- FD convergence rate verification as adjoint fallback (general pattern for gradient testing)
- --update-baselines plugin pattern for reproducibility tests

### Key Lessons
1. Surrogate bias (~11%) is inherent and must be accounted for in test gates — don't gate on absolute accuracy when testing inference
2. PETSc/PyTorch memory conflicts require process isolation — plan for subprocess from the start when both are needed
3. Hold-out validation (not training data) is essential for surrogate fidelity — 479 LHS samples caught model-specific weaknesses

### Cost Observations
- Model mix: 100% sonnet (balanced profile)
- Average plan execution: ~5 min
- Notable: Phase 4 (inverse verification) was most complex — 4 plans, 23 min total — but produced the most defensible results

---

## Milestone: v14 — Pipeline Redesign

**Closed:** 2026-03-13 (early — Phase 7 only)
**Phases:** 1 of 5 | **Plans:** 3 | **Timeline:** 3 days

### What Was Built
- Multi-seed v13 baseline wrapper with subprocess isolation and per-parameter error statistics
- PDE-only profile likelihood identifiability analysis with chi-squared thresholds for all 4 parameters
- Sensitivity visualization with 1D parameter sweeps and Jacobian heatmaps across extended voltage grid
- AUDT-04 metadata schema for justification tracking as cross-cutting standard

### What Worked
- Analysis-first approach (diagnose before redesign) produced clear identifiability and sensitivity data to inform future work
- Reusing subprocess isolation pattern from v1.0 kept multi-seed runs stable
- TDD approach with frozen dataclass configs kept diagnostic scripts clean and testable

### What Was Inefficient
- Milestone scope was too ambitious for the available time — only 1 of 5 phases completed before closing
- Phase 8 was fully researched and planned but never executed — planning work was wasted
- Phases 9-11 had TBD plans, indicating scope uncertainty that should have been resolved earlier

### Patterns Established
- Profile likelihood as standard identifiability diagnostic for inverse problems
- AUDT-04 metadata sidecar pattern for justification tracking on every new component
- Central FD step h=1e-5 for Jacobian computation consistent with codebase convention

### Key Lessons
1. Scope milestones more conservatively — better to ship a complete 2-phase milestone than abandon a 5-phase one
2. Don't plan phases in detail until preceding phases are complete — Phase 8 planning was premature
3. Diagnostic phases (measurement) are independently valuable even when redesign phases (implementation) are deferred

### Cost Observations
- Model mix: balanced profile (sonnet execution)
- Sessions: ~3
- Notable: Phase 7 executed quickly (3 plans, ~10min) — diagnostic scripts are fast to implement when the infrastructure exists

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Timeline | Phases | Key Change |
|-----------|----------|--------|------------|
| v1.0 | 4 days | 6 | Initial V&V framework established |
| v14 | 3 days | 1 of 5 | Diagnostics complete, closed early |

### Cumulative Quality

| Milestone | Plans | Files Changed | Lines Added |
|-----------|-------|---------------|-------------|
| v1.0 | 14 | 90 | +12,300 |
| v14 | 3 | 30 | +6,566 |

### Top Lessons (Verified Across Milestones)

1. Bottom-up verification catches errors at the right layer before they compound
2. Subprocess isolation is mandatory when mixing Firedrake and PyTorch in tests
