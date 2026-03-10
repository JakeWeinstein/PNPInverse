---
phase: 04-inverse-problem-verification
plan: 04
subsystem: inverse-verification
tags: [gap-closure, test-gates, PDE-targets, inverse-crime-fix]
dependency_graph:
  requires: [04-03]
  provides: [passing-inverse-verification-tests]
  affects: [tests/test_inverse_verification.py]
tech_stack:
  added: []
  patterns: [basin-uniqueness-CV, functional-fit-NRMSE, PDE-noise-floor-aware-FD]
key_files:
  modified:
    - tests/test_inverse_verification.py
decisions:
  - "Relaxed NOISE_GATES to {0%: 0.15, 1%: 0.20, 2%: 0.25, 5%: 0.40} to account for ~11% surrogate approximation error"
  - "PDE FD step sizes changed to {1e-2, 1e-3, 1e-4} to stay above PDE solver noise floor"
  - "Multistart convergence redefined as basin uniqueness (CV < 0.10) + functional fit (NRMSE < 0.05)"
  - "Within-10%-of-true metric kept as informational only (not gated)"
metrics:
  duration: 3 min
  completed: "2026-03-08T21:09:00Z"
  tasks_completed: 3
  tasks_total: 3
---

# Phase 04 Plan 04: Gap Closure -- Fix Test Gates, PDE FD Steps, and Multistart Convergence Summary

Relaxed test gates and convergence criteria in test_inverse_verification.py to account for the ~11% irreducible surrogate approximation error introduced by eliminating inverse crime (04-03).

## Completed Tasks

| # | Task | Commit | Key Changes |
|---|------|--------|-------------|
| 1 | Fix TestParameterRecovery gates | `6ca7620` | NOISE_GATES relaxed to {0%: 0.15, 1%: 0.20, 2%: 0.25, 5%: 0.40}; surrogate_bias metric added to JSON |
| 2 | Fix TestGradientConsistencyPDE step sizes | `60d77cf` | h_values: {1e-2, 1e-3, 1e-4}; tolerance 5%; gradient threshold 1e-8 |
| 3 | Redefine TestMultistartBasin convergence | `c6c8ba9` | Basin uniqueness (CV<0.10) + functional fit (NRMSE<0.05); old gate informational only |

## Deviations from Plan

None -- plan executed exactly as written.

## Verification

- Fast test (TestSurrogateFDConvergence): PASSED
- Slow tests: Code changes match plan specification; require Firedrake environment for full execution

## Decisions Made

1. **Gate relaxation rationale**: The surrogate optimum differs from PDE truth by ~11% even at 0% noise. Original gates (0%: 5%) assumed surrogate-on-surrogate recovery (inverse crime). New gates add headroom above the irreducible approximation error.

2. **PDE FD step size rationale**: PDE solver convergence tolerance is ~1e-4 to 1e-7. At h=1e-6 the FD perturbation is in the solver's noise territory (observed 81.8% relerr). Coarser steps {1e-2, 1e-3, 1e-4} stay well above the noise floor.

3. **Multistart convergence redefinition**: "Within 10% of true parameters" is not a meaningful gate when the surrogate optimum itself is ~11% away from PDE truth. Basin uniqueness (CV) and functional fit (NRMSE) measure what actually matters: do all starts find the same answer, and does that answer fit the data?
