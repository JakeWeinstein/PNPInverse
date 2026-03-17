---
phase: 3-comparative-benchmark
plan: 02
subsystem: testing
tags: [gradient, benchmark, autograd, finite-difference, PCE, GP, NN-ensemble]

requires:
  - phase: 3-comparative-benchmark
    provides: trained surrogate models (NN, GP, PCE, POD-RBF, RBF)
provides:
  - gradient accuracy comparison across all 5 surrogate models
  - gradient speed benchmark (autograd vs FD wall-clock timing)
  - reproducible JSON artifacts for gradient accuracy and speed
  - fixed PCE predict() and predict_gradient() bugs
affects: [4-ismo-pipeline, 5-pde-refinement, surrogate-selection]

tech-stack:
  added: []
  patterns: [FD-reference validation, autograd vs analytic comparison]

key-files:
  created:
    - scripts/studies/gradient_benchmark.py
    - tests/test_gradient_benchmark.py
    - StudyResults/gradient_benchmark/gradient_accuracy.json
    - StudyResults/gradient_benchmark/gradient_speed.json
    - StudyResults/gradient_benchmark/gradient_benchmark_report.md
  modified:
    - Surrogate/pce_model.py

key-decisions:
  - "Use h=1e-7 FD as reference gradient (good for PCE, less reliable for NN/GP due to float cancellation)"
  - "PCE analytic gradient is gold standard: <1e-8 relative error, 258ms/eval"
  - "NN autograd is fastest differentiable method: 3.87ms/eval, ~2x faster than FD"
  - "GP autograd too slow for practical use: 2843ms/eval (44 independent GP posteriors)"

patterns-established:
  - "Gradient benchmark pattern: accuracy vs fine-FD + speed timing with JSON output"

requirements-completed: [BENCH-GRAD-01, BENCH-GRAD-02, BENCH-GRAD-03]

duration: 17min
completed: 2026-03-17
---

# Phase 3 Plan 02: Gradient Benchmark Summary

**Gradient accuracy and speed benchmark across 5 surrogates: NN autograd fastest (3.9ms), PCE analytic most accurate (<1e-8 error), GP autograd impractically slow (2.8s)**

## Performance

- **Duration:** 17 min
- **Started:** 2026-03-17T23:02:28Z
- **Completed:** 2026-03-17T23:19:54Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- Benchmarked gradient computation across all 5 trained surrogate models (NN ensemble, GP, PCE, POD-RBF, RBF baseline)
- NN ensemble autograd confirmed ~2x faster than FD (3.87 vs 8.29 ms/eval)
- PCE analytic gradient validated to machine precision (<1e-8 relative error vs FD reference)
- Fixed pre-existing bugs in PCE model: predict() float conversion and predict_gradient() chaospy API

## Key Results

| Model | Method | ms/eval | Accuracy (mean rel err vs h=1e-7 FD) |
|-------|--------|---------|---------------------------------------|
| NN ensemble | autograd | 3.87 | N/A (ref unreliable for NN) |
| NN ensemble | FD | 8.29 | N/A |
| PCE | analytic | 258 | 9.2e-09 |
| PCE | FD | 312 | varies by step |
| GP | autograd | 2843 | unreliable (float32) |
| GP | FD | 441 | unreliable |
| POD-RBF | FD | 3.83 | well-conditioned |
| RBF baseline | FD | 0.66 | well-conditioned |

## Task Commits

1. **Task 1: Create gradient benchmark script and test** - `9c38fbe` (test)
2. **Task 2: Run gradient benchmark and generate results** - `5d84190` (feat)

## Files Created/Modified
- `scripts/studies/gradient_benchmark.py` - Main benchmark with accuracy + speed for all models (380+ lines)
- `tests/test_gradient_benchmark.py` - Smoke tests using synthetic surrogates (100+ lines)
- `StudyResults/gradient_benchmark/gradient_accuracy.json` - Per-model, per-method accuracy data
- `StudyResults/gradient_benchmark/gradient_speed.json` - Wall-clock timing data
- `StudyResults/gradient_benchmark/gradient_benchmark_report.md` - Human-readable summary tables
- `Surrogate/pce_model.py` - Fixed predict() and predict_gradient() bugs

## Decisions Made
- h=1e-7 FD reference works well for PCE (polynomial, exact arithmetic) but is unreliable for NN/GP models due to floating-point cancellation at extreme parameter values
- NN autograd vs FD accuracy is well-validated by existing test_autograd_gradient.py at h=1e-5 (the benchmark confirms both give same results vs h=1e-7 reference)
- GP autograd is functional but too slow for optimization (44 independent GP posterior evaluations)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed PCE predict() float conversion for chaospy array output**
- **Found during:** Task 2
- **Issue:** `float(self._pce_cd[j](*x))` fails when chaospy returns shape (1,) array
- **Fix:** Wrapped in `np.squeeze()` before `float()` conversion
- **Files modified:** Surrogate/pce_model.py
- **Committed in:** 5d84190

**2. [Rule 1 - Bug] Fixed PCE predict_gradient() using wrong chaospy API**
- **Found during:** Task 2
- **Issue:** `cp.differential` does not exist in chaospy 4.3.21; correct API is `cp.derivative`
- **Fix:** Replaced `cp.differential` with `cp.derivative`
- **Files modified:** Surrogate/pce_model.py
- **Committed in:** 5d84190

**3. [Rule 3 - Blocking] Fixed split_indices key name and sys.path for script execution**
- **Found during:** Task 2
- **Issue:** Key was `test_idx` not `test_indices`; script couldn't import Surrogate when run directly
- **Fix:** Fixed key name; added sys.path repo root insertion at script top
- **Files modified:** scripts/studies/gradient_benchmark.py
- **Committed in:** 5d84190

---

**Total deviations:** 3 auto-fixed (2 bugs, 1 blocking)
**Impact on plan:** All auto-fixes necessary for correctness. No scope creep.

## Issues Encountered
- h=1e-7 FD reference gradient is unreliable for NN and GP models due to floating-point cancellation. All FD step sizes and autograd show similar relative errors vs this reference, confirming the reference itself is the issue. For these models, the existing h=1e-5 autograd-vs-FD comparison (from test_autograd_gradient.py) remains the authoritative accuracy validation.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Gradient benchmark artifacts ready for surrogate selection decisions
- NN ensemble recommended for speed-critical optimization (3.9ms autograd)
- PCE recommended when gradient accuracy is paramount (<1e-8 analytic)
- GP autograd not recommended for iterative optimization (too slow)

## Self-Check: PASSED

All 5 artifacts verified present. Both task commits (9c38fbe, 5d84190) confirmed in git log.

---
*Phase: 3-comparative-benchmark*
*Completed: 2026-03-17*
