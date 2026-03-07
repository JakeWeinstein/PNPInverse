# Phase 4: Inverse Problem Verification - Context

**Gathered:** 2026-03-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Prove v13 parameter inference recovers known parameters from synthetic data at multiple noise levels, verify gradient consistency (adjoint vs FD for PDE, FD convergence for surrogate), and demonstrate multistart optimizer convergence basin statistics. Requirements: INV-01, INV-02, INV-03.

</domain>

<decisions>
## Implementation Decisions

### Parameter recovery scope (INV-01)
- Replace and consolidate existing Tests 1, 2, and 7 from `test_v13_verification.py` into a new Phase 4 test file — one authoritative place for all parameter recovery and multistart tests
- Run the full 7-phase v13 pipeline (S1-S5 surrogate + P1-P2 PDE refinement) at each noise level: 0%, 1%, 2%, 5%
- Soft gates scaled by noise level: 0% noise < 5% error, 1% < 10%, 2% < 15%, 5% < 30% (per-parameter max relative error)
- Actual errors saved as artifacts; soft gates only catch catastrophic failures
- Output: JSON summary + CSV per-run details to `StudyResults/inverse_verification/`

### Gradient consistency (INV-02)
- **PDE adjoint vs FD**: Compare Firedrake automatic adjoint gradient against central FD on the PDE objective function. Tolerance: component-wise relative agreement within 1%. Evaluate at 3 points: true parameters + 2 random perturbations.
- **Surrogate FD self-consistency**: Verify FD gradient convergence rate at step sizes h=1e-3, 1e-5, 1e-7. Central FD should show O(h^2) convergence. Fast test (no Firedrake).
- PDE gradient test is `@pytest.mark.slow`; surrogate FD convergence is a fast test

### Multistart convergence basin (INV-03)
- Use full production `run_multistart_inference()` config: 20K LHS grid + top-20 candidate polish
- Soft gate: >50% of polished candidates recover all 4 parameters within 10% of truth (matches existing Test 7 threshold)
- Report all 4 statistics:
  - % of candidates converging to true minimum (within 10%)
  - Loss distribution (min, median, max polished loss)
  - Parameter spread (std dev of recovered params across converged candidates)
  - Best-vs-worst gap (ratio of worst to best polished loss)
- All statistics saved to JSON for Phase 6 report

### Noise handling
- Always use PDE-generated synthetic targets (never surrogate-generated — avoids inverse crime)
- Cache clean PDE solutions: use existing `_generate_targets_with_pde()` caching pattern. For non-true parameter values, create and cache a new PDE target on first run
- **Noise model change**: Switch from percent-of-range (additive) to percent-of-signal (multiplicative): `noise_std = noise_percent/100 * |signal|` at each voltage point. Update existing `add_percent_noise()` implementation
- 3 noise realizations per noise level with fixed seeds (e.g., seeds 42, 43, 44). Report mean/std of recovery error across realizations
- Scaled soft gates apply to the mean error across realizations

### Claude's Discretion
- Exact test file organization (single class vs multiple classes per requirement)
- How to structure the PDE adjoint gradient extraction from Firedrake's ReducedFunctional
- FD step size selection for the PDE adjoint-vs-FD test (optimal sqrt(eps) or similar)
- Plot styling for any diagnostic figures
- How to handle the noise model update across the codebase (backward compatibility)

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `Surrogate/objectives.py:SurrogateObjective`: 4-param joint objective with FD gradient — directly testable for INV-02 surrogate FD convergence
- `Surrogate/multistart.py:run_multistart_inference()`: Full LHS + polish pipeline — directly testable for INV-03
- `scripts/surrogate/Infer_BVMaster_charged_v13_ultimate.py`: 7-phase pipeline entry point — drives INV-01 full pipeline tests
- `scripts/_bv_common.py`: True parameter values (`K0_HAT_R1`, `K0_HAT_R2`, `ALPHA_R1`, `ALPHA_R2`), `_generate_targets_with_pde()` with caching, `_compute_errors()`, `make_bv_solver_params()`
- `Forward/steady_state:add_percent_noise()`: Current noise model (to be updated to percent-of-signal)
- `tests/test_v13_verification.py`: Existing Tests 1, 2, 7 to be replaced/consolidated
- `tests/conftest.py:skip_without_firedrake`: Skip decorator for Firedrake-dependent tests

### Established Patterns
- `StudyResults/` for test output artifacts (JSON + PNG + CSV) — Phase 2 MMS, Phase 3 surrogate fidelity
- `@pytest.mark.slow` + `@skip_without_firedrake` for expensive Firedrake tests
- Module-scoped fixtures for shared model loading (`nn_ensemble` fixture pattern)
- Soft diagnostic gates with generous thresholds (Phase 3 precedent: 20% NRMSE)

### Integration Points
- `Inverse/objectives.py`: PDE-based objective factories (Firedrake adjoint) — needed for PDE gradient consistency test
- `Inverse/inference_runner/`: PDE inference runner with `build_reduced_functional()` — builds the ReducedFunctional for adjoint gradients
- Phase 6 will read `StudyResults/inverse_verification/*.json` and `*.csv` for automated report generation
- `test_v13_verification.py` Tests 1, 2, 7 will be removed after consolidation into Phase 4 test file

</code_context>

<specifics>
## Specific Ideas

- User wants PDE-generated targets always (never surrogate-on-surrogate) to avoid inverse crime
- Noise model must be updated globally: percent-of-signal multiplicative noise replaces percent-of-range additive noise
- 3 noise realizations per level gives mean/std for error reporting — more statistically meaningful for the V&V report
- Existing Tests 1/2/7 in test_v13_verification.py are subsumed — remove them to avoid redundancy (same pattern as Phase 3 removing Test 5)
- Full 20K multistart and full 7-phase pipeline chosen over lighter alternatives — user prefers production-realistic testing

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 04-inverse-problem-verification*
*Context gathered: 2026-03-07*
