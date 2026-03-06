# Feature Landscape

**Domain:** PDE Verification & Validation Framework for Electrochemical Inference
**Researched:** 2026-03-06

## Table Stakes

Features required for a publication-grade V&V framework. Missing any of these means the work cannot withstand peer review.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| MMS convergence verification (automated) | Proves the PDE solver is implementing the equations correctly. Reviewers expect convergence plots showing theoretical rates. | Low | `mms_bv_convergence.py` already exists and produces correct results. Wrap in pytest with rate assertions. |
| Mesh convergence study with rate verification | Demonstrates solution converges at expected rate (O(h^2) L2, O(h) H1 for CG1). Must show asymptotic regime is reached. | Low | Existing MMS script already does this. Need to formalize the rate assertion (e.g., `abs(observed_rate - 2.0) < 0.15`). |
| Error norm tables | Tabular L2, H1 errors and observed convergence rates for each variable (concentrations, potential). Standard format for journal appendices. | Low | `format_table_single` and `format_table_multi` already generate these. |
| Convergence plots (log-log) | Visual evidence of convergence. Reviewers look for straight lines with correct slope on log-log plots. | Low | `plot_convergence` already generates publication-quality plots. |
| Nondimensionalization roundtrip tests | Proves the dimensional-to-nondimensional transform is invertible and correct. Critical because a past bug (hardcoded value) was found in this layer. | Low | `test_nondim.py` exists. May need expansion to cover all species and boundary conditions. |
| Parameter recovery from synthetic data | Proves the inference pipeline can recover known parameters from PDE-generated data. Core claim of the paper. | Medium | `test_v13_verification.py` Tests 1-2 cover this for zero-noise case. Need to add noisy-data recovery. |
| Gradient verification (FD consistency) | Proves optimization gradients are correct. Essential for L-BFGS-B convergence guarantees. | Low | `test_v13_verification.py` Test 3 already does this with two-step-size comparison. |
| Automated test suite (pytest) | All V&V checks runnable via `pytest -m slow` without manual intervention. Ensures reproducibility and regression detection. | Medium | Partially done. MMS scripts are standalone; need pytest wrappers. |

## Differentiators

Features that elevate the V&V from "adequate" to "thorough" or "exemplary."

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Surrogate fidelity map | Quantify max/mean error between NN surrogate and PDE across the parameter space, not just at the training parameters. Shows where the surrogate is trustworthy. | Medium | Sample parameter space with LHS, evaluate both surrogate and PDE at each point, compute error statistics. Expensive (many PDE solves) but high value. |
| Noise robustness analysis | Add Gaussian noise to synthetic targets, show parameter recovery degrades gracefully (not catastrophically). Quantifies sensitivity to measurement error. | Medium | Extend existing zero-noise recovery test with noise levels [1%, 5%, 10%]. Report parameter error vs noise level. |
| GCI uncertainty estimation | Grid Convergence Index per ASME V&V 20. Provides quantified numerical uncertainty ("the discretization error is estimated to be X +/- Y"). More rigorous than just showing convergence rates. | Low-Medium | Richardson extrapolation from 3 mesh levels. ~30 lines of code. Publishable uncertainty bound. |
| Sensitivity monotonicity verification | Perturbing any parameter from truth increases the objective. Proves local identifiability. | Low | Already implemented in Test 6 of `test_v13_verification.py`. |
| Multistart convergence basin analysis | Multiple random initializations converge to the same optimum. Demonstrates landscape has a single dominant basin (no spurious local minima). | Low | Already implemented in Test 7 of `test_v13_verification.py`. |
| Reproducibility baselines | Saved numerical baselines (error norms, convergence rates, recovered parameters) with tolerance-bounded comparison across runs. | Low | Use pytest-regressions to snapshot and compare. |
| Written V&V report | LaTeX or Markdown document with all convergence plots, error tables, and benchmark results formatted for journal supplementary material. | Medium | Synthesis of all test outputs into a structured document. |

## Anti-Features

Features to explicitly NOT build.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Adaptive mesh refinement V&V | Out of scope. The solver uses fixed meshes (graded rectangular). AMR verification is a separate research problem. | Verify convergence on the fixed mesh topology actually used in the pipeline. |
| Experimental data validation | This project is code VERIFICATION (are we solving the equations right?), not model VALIDATION (are the equations the right model for the physics?). Conflating the two weakens both claims. | State clearly in the V&V report: "This work verifies the numerical implementation. Model validation against experimental data is a separate study." |
| CI/CD pipeline | Out of scope per PROJECT.md. Setting up GitHub Actions with Firedrake is nontrivial (requires custom Docker image with PETSc). | Run tests locally. Document the command: `pytest tests/ -m slow --tb=short`. |
| Performance benchmarking as primary goal | Correctness before speed. Performance metrics are secondary to proving the solver is correct. | Include pytest-benchmark for regression detection, but don't optimize solve times. |
| Higher-order element convergence | The solver uses CG1. Verifying CG2/CG3 convergence rates adds complexity without verifying the actual production code. | Verify CG1 convergence rates only (L2 rate ~2, H1 rate ~1). |
| Stochastic/UQ framework | Adds substantial complexity. The paper's core contribution is deterministic parameter inference, not uncertainty quantification of the inferred parameters. | Mention as future work. The noise robustness test provides a lightweight version. |

## Feature Dependencies

```
Nondimensionalization tests -> MMS convergence verification (MMS uses nondim quantities)
MMS convergence verification -> Surrogate fidelity analysis (surrogate trained on PDE data; PDE must be verified first)
MMS convergence verification -> Parameter recovery tests (recovery depends on correct forward solver)
Surrogate fidelity analysis -> End-to-end pipeline verification
Parameter recovery (zero-noise) -> Noise robustness analysis
All V&V tests -> Written V&V report
```

## MVP Recommendation

Prioritize (Phase 1):
1. **MMS in pytest** -- Wrap existing `mms_bv_convergence.py` as parameterized pytest tests with convergence rate assertions. This is the foundation.
2. **Nondimensionalization verification** -- Expand `test_nondim.py` if gaps exist.
3. **GCI computation** -- Simple utility for publication-grade uncertainty quantification.

Prioritize (Phase 2):
4. **Surrogate fidelity map** -- Quantify surrogate error across parameter space.
5. **Noise robustness** -- Extend parameter recovery tests with noise.
6. **Reproducibility baselines** -- Snapshot key numerical results with pytest-regressions.

Defer:
- **Written V&V report** -- Compile after all tests pass. This is output synthesis, not new testing.
- **pytest-benchmark integration** -- Nice to have but not critical for publication.

## Sources

- Existing codebase analysis: `mms_bv_convergence.py`, `test_v13_verification.py`, `test_nondim.py`
- [ASME V&V 20-2009 Standard](https://www.asme.org/codes-standards/find-codes-standards/standard-for-verification-and-validation-in-computational-fluid-dynamics-and-heat-transfer) -- verification hierarchy and GCI methodology
- [COMSOL MMS Blog](https://www.comsol.com/blogs/verify-simulations-with-the-method-of-manufactured-solutions) -- MMS best practices
- PROJECT.md requirements and constraints
