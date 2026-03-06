# Research Summary: PNP-BV V&V Framework

**Domain:** PDE Verification & Validation for Electrochemical Inference Pipeline
**Researched:** 2026-03-06
**Overall confidence:** MEDIUM-HIGH

## Executive Summary

The PNP-BV Inverse codebase is in an unusually strong position for V&V: the hardest parts are already implemented correctly. The MMS convergence script (`mms_bv_convergence.py`) produces clean O(h^2) L2 and O(h) H1 convergence rates across three progressively complex test cases (single neutral species, two neutral species with two reactions, and two charged species with Poisson coupling). The manufactured solutions are well-designed -- smooth trigonometric-exponential functions that exercise BV boundary conditions, electromigration coupling, and multi-species stoichiometry without pushing into the pre-asymptotic regime. The existing `test_v13_verification.py` provides thorough parameter recovery tests including zero-noise identity recovery, gradient FD consistency, sensitivity monotonicity, and multistart convergence basin analysis.

The primary gap is organizational, not technical: the MMS verification runs as a standalone script rather than an automated pytest test with rate assertions, there are no saved numerical baselines for regression detection, and there is no surrogate fidelity characterization across the parameter space (only at 3 specific parameter sets). The GCI uncertainty quantification needed for a publication-quality report is trivially implementable but absent.

The recommended stack adds only two lightweight pytest plugins (pytest-regressions for numerical baselines, pytest-benchmark for performance tracking) to the existing pytest + Firedrake + SciPy infrastructure. No new PDE or verification frameworks are needed -- Firedrake's `errornorm` and UFL autodifferentiation already provide everything required for MMS and convergence analysis. The `scipy.stats.linregress` function on log-log data provides convergence order with R-squared confidence, which is more robust than the existing consecutive log-ratio approach.

The most dangerous pitfall is nondimensionalization errors producing silently wrong results -- the codebase has a documented history of this type of bug. MMS tests operate in nondimensional space and therefore cannot catch dimensional analysis errors. Separate nondimensionalization verification tests are essential and partially exist in `test_nondim.py`.

## Key Findings

**Stack:** Keep existing Firedrake + pytest + SciPy. Add only pytest-regressions 2.10.0 and pytest-benchmark 5.2.3. No new PDE verification frameworks needed.

**Architecture:** Layered verification pyramid (foundation -> forward solver -> surrogate -> pipeline), each layer tested independently before cross-layer tests.

**Critical pitfall:** Nondimensionalization errors are silent killers. MMS does not catch them. Dimensional analysis tests are a separate, essential verification layer.

## Implications for Roadmap

Based on research, suggested phase structure:

1. **Phase 1: Foundation & MMS Formalization** - Wrap existing MMS in pytest, expand nondim tests, add GCI utility
   - Addresses: MMS convergence verification, nondim roundtrip, mesh convergence rate assertions
   - Avoids: Pitfall 4 (pre-asymptotic regime) by requiring R-squared > 0.99

2. **Phase 2: Surrogate & Pipeline Verification** - Surrogate fidelity map, noise robustness, reproducibility baselines
   - Addresses: Surrogate error bounds, parameter recovery with noise, regression detection
   - Avoids: Pitfall 5 (testing only at training points) by using LHS parameter sampling

3. **Phase 3: Report & Publication Artifacts** - Written V&V report with convergence plots, error tables, GCI bounds
   - Addresses: Publication-grade evidence for journal supplementary material
   - Avoids: Starting report before all tests pass

**Phase ordering rationale:**
- Phase 1 must come first because surrogate fidelity (Phase 2) is meaningless if the forward solver is not verified.
- Phase 2 before Phase 3 because the report needs all test results as input.
- Within Phase 1, nondim tests should come before MMS tests because MMS uses nondimensional quantities.

**Research flags for phases:**
- Phase 1: Standard patterns, unlikely to need research. The existing MMS code just needs pytest wrappers.
- Phase 2: Surrogate fidelity study design (how many LHS samples, what error metrics to report) may benefit from reviewing comparable published V&V studies in electrochemical modeling.
- Phase 3: Standard document compilation, no research needed.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Recommendations are for existing tools (Firedrake, pytest, SciPy) plus 2 well-established pytest plugins. No novel technology. |
| Features | HIGH | Feature list derived directly from PROJECT.md requirements and existing codebase gaps. What exists is high quality; what's missing is well-defined. |
| Architecture | HIGH | Layered verification pyramid is the standard ASME V&V approach. Matches the existing codebase structure. |
| Pitfalls | MEDIUM-HIGH | Pitfalls derived from codebase analysis and V&V literature. The nondim error history is documented in PROJECT.md. MMS sign convention risks are specific to this solver. |

## Gaps to Address

- **Existing MMS script independence:** The MMS code builds its own weak form. Manual code review comparing MMS form to production `bv_solver.py` form is needed to confirm they test the same equations. This is a one-time review, not an automated test.
- **4-species MMS case:** The convergence script has 3 cases (1 species, 2 neutral, 2 charged). The production solver uses 4 species (O2, H2O2, H+, ClO4-). A 4-species MMS case would close this gap but adds complexity.
- **Surrogate fidelity sample size:** How many LHS samples are needed for statistically meaningful error characterization? 50 is a reasonable starting point, but this should be validated by checking that error statistics stabilize with increasing sample count.
- **Publication norms for V&V:** Different journals have different expectations for V&V evidence. A brief survey of V&V sections in recent electrochemistry publications would calibrate the report scope.
