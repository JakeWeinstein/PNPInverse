# Stack Research

**Domain:** PDE Verification & Validation for Electrochemical Inference Pipeline
**Researched:** 2026-03-06
**Confidence:** MEDIUM-HIGH

## Recommended Stack

### Core Technologies (Already In Place -- Keep)

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Firedrake | latest (pip-independent) | FEM solver, `errornorm`, MMS source term via UFL autodiff | Already the project's FEM engine. `errornorm` computes L2/H1 norms directly. UFL symbolic differentiation generates MMS source terms automatically -- this is the correct approach, already used in `mms_bv_convergence.py`. No alternative needed. |
| NumPy | >=2.0 | Error norm arrays, convergence rate computation, log-log regression | Universal dependency; already used everywhere. |
| SciPy | >=1.12 | `scipy.stats.linregress` for convergence rate fitting, `scipy.optimize` for parameter recovery | Already a core dependency. `linregress` on log-log data gives observed convergence order with R-squared confidence. |
| pytest | >=8.0 | Test runner for V&V test suite | Already configured in `pyproject.toml`. Markers (`@pytest.mark.slow`) already separate fast/slow tests. |
| matplotlib | >=3.8 | Convergence plots, error tables, publication figures | Already used for I-V curves and MMS convergence plots. |

### New V&V-Specific Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest-regressions | 2.10.0 | Numerical regression testing -- saves baseline data (NumPy arrays, dicts) and compares future runs against them with configurable atol/rtol | **Use for:** reproducibility tests (same inputs produce same outputs), baseline convergence rates, surrogate prediction baselines. Saves `.npz` or `.json` files alongside tests. Install: `pip install pytest-regressions==2.10.0` |
| pytest-benchmark | 5.2.3 | Performance regression tracking -- measures and tracks execution time of PDE solves across code changes | **Use for:** detecting solver performance regressions (e.g., a code change doubles MMS solve time). Stores JSON history. Less critical than correctness testing but useful for catching unintended slowdowns. Install: `pip install pytest-benchmark==5.2.3` |
| pytest-cov | (already installed) | Coverage reporting | Already a dev dependency. Use `--cov=Forward --cov=Surrogate --cov=Nondim` flags to track V&V coverage of production code. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| `scipy.stats.linregress` | Compute observed convergence order from log-log error data | No new dependency needed. `slope, _, r_value, _, _ = linregress(log_h, log_err)`. R-squared > 0.99 indicates clean asymptotic regime. |
| `numpy.polyfit` | Alternative convergence rate fitting | `p = np.polyfit(np.log(h), np.log(err), 1); rate = p[0]`. Equivalent to linregress but less diagnostic output. |
| Firedrake `errornorm` | L2, H1, Linf error norms between exact and computed solutions | Already used correctly in `mms_bv_convergence.py`. Handles function space projection internally. |

## Installation

```bash
# V&V-specific additions (within Firedrake venv)
pip install pytest-regressions==2.10.0
pip install pytest-benchmark==5.2.3

# Already installed (verify versions)
pip install --upgrade pytest>=8.0 pytest-cov
```

## What You Already Have and Should Keep

The existing codebase already implements the hardest parts of V&V correctly:

1. **MMS with UFL autodiff** (`scripts/verification/mms_bv_convergence.py`): Source terms computed symbolically via `fd.div(fd.grad(...))`. Boundary corrections computed correctly. Three test cases covering neutral, multi-species, and charged+Poisson coupling. This is publication-grade MMS implementation.

2. **Convergence rate computation** (`compute_rates` function): Log-ratio convergence rates. Results show clean O(h^2) L2 and O(h) H1 rates for CG1, matching theory.

3. **Parameter recovery tests** (`tests/test_v13_verification.py`): Zero-noise identity recovery, gradient verification via FD step comparison, sensitivity monotonicity, multistart convergence basin, surrogate-vs-PDE consistency. This is a thorough verification suite.

4. **Nondimensionalization tests** (`tests/test_nondim.py`): Roundtrip tests for unit transforms.

## What's Missing (Stack Gaps to Fill)

| Gap | What to Add | Stack Component |
|-----|-------------|-----------------|
| MMS tests not in pytest | Wrap `mms_bv_convergence.py` functions in pytest tests with assertions on convergence rates | pytest + existing Firedrake code |
| No regression baselines | Use pytest-regressions to snapshot convergence rates, error norms, and surrogate predictions | pytest-regressions |
| No GCI uncertainty | Compute Grid Convergence Index for reporting numerical uncertainty in publication | Custom utility using Richardson extrapolation formula (no library needed -- simple enough to implement in 20 lines) |
| No surrogate error bounds | Compute max/mean/percentile error between surrogate and PDE across parameter space | NumPy (already available) |
| No reproducibility tests | Seed-controlled runs checking bitwise or tolerance-bounded reproducibility | pytest-regressions + NumPy |

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| `scipy.stats.linregress` for convergence rates | pyGCS 1.1.1 (Grid Convergence Study) | Only if you need GCI with formatted LaTeX/Markdown output. pyGCS is CFD-focused and computes GCI per Celik et al. (2008). Overkill for this project where convergence rates are the primary metric, not GCI uncertainty bands. |
| pytest-regressions for baselines | Manual `.npz` file comparison | Only if you want full control over comparison logic. pytest-regressions handles file management, diff reporting, and tolerance configuration automatically. |
| Custom MMS source terms via UFL | SymPy for symbolic MMS source generation | Never for this project. Firedrake's UFL already does this correctly and integrates directly with the solver. SymPy would add a translation step with no benefit. |
| pytest-benchmark for timing | Manual `time.time()` calls | pytest-benchmark provides statistical analysis (min, median, stddev, IQR) and historical comparison. Manual timing is unreliable for regression detection. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| SymPy for MMS source terms | Firedrake UFL already computes source terms symbolically via autodiff. Adding SymPy creates a translation layer (SymPy -> UFL) that introduces bugs and adds no value. The existing `mms_bv_convergence.py` does this correctly. | UFL `fd.div(fd.grad(...))` (already in use) |
| FEniCS/DOLFINx convergence utilities | Different FEM backend. Firedrake and FEniCS have diverged significantly. FEniCS utilities won't work with Firedrake function spaces. | Firedrake `errornorm` + custom convergence rate computation (already implemented) |
| pyGCS for this project | Designed for CFD grid convergence with GCI reporting per ASME V&V 20. This project needs convergence ORDER verification (is the rate 2.0 for L2?), not GCI uncertainty bands. pyGCS solves the wrong problem. | `linregress(log_h, log_err)` for order estimation |
| Hypothesis (property-based testing) | Useful for fuzzing API inputs, but MMS and convergence tests have fixed structure. Property-based testing adds complexity without catching the bugs that matter (sign errors, wrong norms, incorrect source terms). | Parameterized pytest tests with explicit MMS cases |
| MOOSE MMS framework | C++-based. Would require reimplementing the solver. | Firedrake UFL (already working) |

## Stack Patterns by V&V Layer

**Forward Solver MMS Verification:**
- Use Firedrake `errornorm` + `RectangleMesh` refinement sequence
- Compute rates via `linregress(np.log(h_vals), np.log(err_vals))`
- Assert `abs(rate - expected_rate) < tolerance` where tolerance = 0.15 for CG1 (expected L2=2.0, H1=1.0)
- Because the existing `mms_bv_convergence.py` already shows clean rates, the main work is wrapping this in pytest

**Surrogate Fidelity Validation:**
- Compute error metrics: max absolute error, mean relative error, RMSE over parameter space
- Use Latin Hypercube samples (already available via `scipy.stats.qmc.LatinHypercube`)
- Because surrogate errors compound into inference errors, this is a critical validation layer

**Parameter Recovery Tests:**
- Generate synthetic data at known parameters via PDE solver
- Recover parameters via inference pipeline
- Assert relative error < threshold
- Because `test_v13_verification.py` already does most of this, extend with noise robustness

**Reproducibility:**
- Use pytest-regressions `ndarrays_regression` fixture for numerical baselines
- Because Firedrake + PETSc have nondeterministic iteration counts in edge cases

## Version Compatibility

| Package | Compatible With | Notes |
|---------|-----------------|-------|
| pytest-regressions 2.10.0 | pytest >=8.0, NumPy >=1.20 | Uses NumPy for array comparison. Python >=3.10 required. |
| pytest-benchmark 5.2.3 | pytest >=9.0 | Latest version adds pytest 9 support. |
| Firedrake (2024/2025) | PETSc 3.20+, Python 3.10-3.12 | Firedrake manages its own PETSc. Version pinned by Firedrake installer. |
| NumPy 2.x | scipy >=1.12, matplotlib >=3.8 | NumPy 2.0 broke some C API users. Firedrake's NumPy version is managed by its installer -- do not upgrade independently. |

## Sources

- [Firedrake errornorm source code](https://www.firedrakeproject.org/_modules/firedrake/norms.html) -- verified `errornorm` API supports L2, H1, Linf norms (HIGH confidence)
- [Firedrake solving interface docs](https://www.firedrakeproject.org/solving-interface.html) -- verified SNES Jacobian verification via `fd_jacobian` option (HIGH confidence)
- [pytest-regressions PyPI](https://pypi.org/project/pytest-regressions/) -- v2.10.0, Feb 2026, supports NumPy array regression (HIGH confidence)
- [pytest-benchmark PyPI](https://pypi.org/project/pytest-benchmark/) -- v5.2.3, Nov 2025 (HIGH confidence)
- [pyGCS PyPI](https://pypi.org/project/pyGCS/) -- v1.1.1, Jul 2022, GCI computation (HIGH confidence)
- [ASME V&V 20-2009 overview](https://www.osti.gov/servlets/purl/1368927) -- Richardson extrapolation and GCI methodology (HIGH confidence)
- [convergence PyPI](https://pypi.org/project/convergence/0.1/) -- NASA grid convergence study port (MEDIUM confidence, older package)
- [ORNL cfd-verify](https://github.com/ORNL/cfd-verify) -- ORNL CFD verification package (MEDIUM confidence)
- Existing codebase analysis: `mms_bv_convergence.py`, `test_v13_verification.py`, `pyproject.toml` (HIGH confidence)

---
*Stack research for: PDE V&V Framework for PNP-BV Electrochemical Inference*
*Researched: 2026-03-06*
