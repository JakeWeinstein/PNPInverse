# Architecture Patterns

**Domain:** PDE Verification & Validation Framework for Electrochemical Inference
**Researched:** 2026-03-06
**Confidence:** HIGH (based on ASME V&V standards, Firedrake documentation, existing codebase analysis)

## Recommended Architecture

The V&V framework is a **layered verification pyramid** that mirrors the existing codebase's multi-layer architecture. Each layer is verified independently before testing cross-layer interactions.

```
Layer 4: End-to-End Pipeline Verification
         (parameter recovery, reproducibility)
              |
Layer 3: Surrogate Fidelity Validation
         (NN ensemble vs PDE error bounds)
              |
Layer 2: Forward Solver Verification (MMS)
         (convergence rates, error norms, GCI)
              |
Layer 1: Foundation Verification
         (nondimensionalization, physical constants, boundary conditions)
```

### Component Boundaries

| Component | Responsibility | Communicates With |
|-----------|---------------|-------------------|
| `tests/test_mms_convergence.py` | Automated MMS convergence tests with rate assertions | Forward solver via `mms_bv_convergence.py` functions |
| `tests/test_nondim.py` (expanded) | Nondim roundtrip and dimensional analysis | `Nondim/` package |
| `tests/test_surrogate_fidelity.py` | Error metrics between surrogate and PDE across parameter space | `Surrogate/ensemble.py`, `FluxCurve/bv_point_solve.py` |
| `tests/test_v13_verification.py` (existing) | Parameter recovery, gradient consistency, sensitivity | Full pipeline |
| `tests/test_reproducibility.py` | Bitwise/tolerance-bounded reproducibility | All layers via pytest-regressions baselines |
| `verification/convergence_utils.py` | Shared utilities: rate computation, GCI, log-log regression | Used by all test modules |

### Data Flow

```
1. MMS Tests:
   Manufactured solution (UFL) -> Firedrake solver -> errornorm(exact, computed)
   -> error array -> convergence rate (linregress) -> assert rate ~ expected

2. Surrogate Fidelity:
   LHS parameter samples -> PDE solver (each sample) -> PDE I-V curves
                          -> NN surrogate (each sample) -> Surrogate I-V curves
   -> per-sample error metrics -> aggregate statistics (max, mean, RMSE)

3. Parameter Recovery:
   True params -> PDE solver -> synthetic target I-V curve
   -> inference pipeline (surrogate + PDE stages) -> recovered params
   -> relative error vs true params -> assert error < threshold

4. Reproducibility:
   Any test -> numerical output -> pytest-regressions baseline file
   Next run -> same test -> new output -> compare against baseline with atol/rtol
```

## Patterns to Follow

### Pattern 1: Parameterized MMS Tests

**What:** Use `@pytest.mark.parametrize` to run the same convergence test logic across multiple MMS cases (single species, multi-species, charged).

**When:** Always for MMS tests. Avoids code duplication across test cases.

**Example:**
```python
@pytest.mark.slow
@pytest.mark.parametrize("case_runner,expected_rates", [
    (run_mms_single_species, {"c_L2": 2.0, "c_H1": 1.0, "phi_L2": 2.0, "phi_H1": 1.0}),
    (run_mms_two_species, {"c0_L2": 2.0, "c0_H1": 1.0, "c1_L2": 2.0, "phi_L2": 2.0}),
    (run_mms_charged, {"c0_L2": 2.0, "c0_H1": 1.0, "c1_L2": 2.0, "phi_L2": 2.0}),
])
def test_mms_convergence_rates(case_runner, expected_rates):
    results = case_runner([16, 32, 64, 128])
    for field, expected in expected_rates.items():
        h = np.array(results["h"])
        err = np.array(results[field])
        slope, _, r_value, _, _ = linregress(np.log(h), np.log(err))
        assert abs(slope - expected) < 0.15, f"{field}: rate={slope:.2f}, expected={expected}"
        assert r_value**2 > 0.99, f"{field}: R^2={r_value**2:.4f}, not in asymptotic regime"
```

### Pattern 2: Convergence Rate Verification via Linear Regression

**What:** Fit `log(error) = slope * log(h) + intercept` using `scipy.stats.linregress`. The slope is the observed convergence order. R-squared confirms the asymptotic regime.

**When:** For all convergence studies (MMS, Richardson extrapolation).

**Why not log-ratio rates?** The existing `compute_rates` function uses consecutive log-ratios `log(e_{k-1}/e_k) / log(h_{k-1}/h_k)`. This is noisy (each rate uses only 2 data points). Linear regression uses all data points simultaneously and provides R-squared as a confidence measure.

**Example:**
```python
from scipy.stats import linregress

def assert_convergence_rate(h_vals, err_vals, expected_rate, field_name, tol=0.15):
    """Assert that errors converge at the expected rate."""
    log_h = np.log(np.array(h_vals))
    log_err = np.log(np.array(err_vals))
    slope, intercept, r_value, p_value, std_err = linregress(log_h, log_err)
    assert abs(slope - expected_rate) < tol, (
        f"{field_name}: observed rate {slope:.3f}, expected {expected_rate:.1f} "
        f"(R^2={r_value**2:.4f})"
    )
    assert r_value**2 > 0.99, (
        f"{field_name}: R^2={r_value**2:.4f} < 0.99, not in asymptotic regime"
    )
    return slope, r_value**2
```

### Pattern 3: GCI Uncertainty Estimation

**What:** Compute Grid Convergence Index from 3 mesh levels using Richardson extrapolation per ASME V&V 20.

**When:** For the written V&V report. Provides a quantified uncertainty band on the finest-mesh solution.

**Example:**
```python
def compute_gci(f1, f2, f3, r=2.0, safety_factor=1.25):
    """Compute GCI from 3 mesh levels (f1=finest, f3=coarsest).

    r: refinement ratio (h_coarse / h_fine)
    safety_factor: 1.25 for 3+ grids (Roache recommendation)
    """
    # Observed order
    p = np.log(abs((f3 - f2) / (f2 - f1))) / np.log(r)
    # Richardson extrapolation
    f_exact = f1 + (f1 - f2) / (r**p - 1)
    # GCI on fine grid
    e_fine = abs((f2 - f1) / f1)
    gci_fine = safety_factor * e_fine / (r**p - 1)
    return p, f_exact, gci_fine
```

### Pattern 4: Surrogate Error Characterization

**What:** Compute error statistics between surrogate and PDE predictions across a Latin Hypercube sample of the parameter space.

**When:** After MMS verification passes. Before trusting surrogate-based inference results.

**Example:**
```python
def surrogate_fidelity_study(ensemble, pde_solver_fn, n_samples=50, seed=42):
    """Compare surrogate vs PDE at LHS-sampled parameter points."""
    from scipy.stats.qmc import LatinHypercube

    sampler = LatinHypercube(d=4, seed=seed)  # k0_1, k0_2, alpha_1, alpha_2
    samples = sampler.random(n_samples)
    # Scale to parameter bounds...

    errors = []
    for params in scaled_samples:
        surr_pred = ensemble.predict(*params)
        pde_pred = pde_solver_fn(*params)
        rel_err = np.max(np.abs(surr_pred - pde_pred) / np.abs(pde_pred + 1e-30))
        errors.append(rel_err)

    return {
        "max_relative_error": np.max(errors),
        "mean_relative_error": np.mean(errors),
        "p95_relative_error": np.percentile(errors, 95),
    }
```

## Anti-Patterns to Avoid

### Anti-Pattern 1: Testing MMS Against the Production Solver Code Path
**What:** Running MMS by calling the production `solve_bv` function with injected source terms.
**Why bad:** If the production solver has a bug that the MMS test exercises the same code path, the bug cancels out and MMS "passes" spuriously. The existing `mms_bv_convergence.py` correctly builds its own weak form independently -- keep this separation.
**Instead:** Keep MMS tests as self-contained weak form constructions that mirror but do not call the production solver. Compare the weak form structure manually to the production code.

### Anti-Pattern 2: Overly Tight Convergence Rate Tolerances
**What:** Asserting `abs(rate - 2.0) < 0.01`.
**Why bad:** Pre-asymptotic effects, boundary layer pollution, and the nonlinearity of BV kinetics can cause rates to deviate from theory by 0.05-0.10 even when the implementation is correct. Overly tight tolerances cause false failures.
**Instead:** Use `abs(rate - expected) < 0.15` and require R-squared > 0.99 on the log-log fit. The R-squared check is more diagnostic than a tight rate tolerance.

### Anti-Pattern 3: Testing Surrogate Fidelity at Training Points Only
**What:** Comparing surrogate and PDE at the exact parameter values used to train the surrogate.
**Why bad:** The surrogate trivially matches at training points (it was optimized to do so). Fidelity at interpolated/extrapolated points is what matters.
**Instead:** Use LHS samples that span the parameter space, including points between and near the edges of the training domain.

### Anti-Pattern 4: Ignoring the H1 Norm
**What:** Only checking L2 convergence rates.
**Why bad:** L2 rates can look correct even when the gradient (flux) is computed incorrectly. For PNP, the fluxes (gradients of concentration and potential) are the physically meaningful quantities. H1 convergence verifies these.
**Instead:** Always check both L2 and H1 convergence rates. For PNP-BV, the H1 rate on concentrations directly tests flux accuracy.

## Scalability Considerations

| Concern | Current (dev) | At publication | Future |
|---------|---------------|----------------|--------|
| MMS solve time | ~2 min (3 cases, N=8..128) | Same (fixed mesh sequence) | Add N=256 only if rates unclear |
| Surrogate fidelity | ~5 PDE points in test | 50-100 LHS samples | Cache PDE results to avoid re-computation |
| Parameter recovery | 5 voltage points | 5-10 points (balance speed vs. coverage) | Full voltage grid only for final publication results |
| Total V&V suite runtime | ~10 min | ~30-60 min | Acceptable for manual runs; CI would need parallelism |

## Sources

- [ASME V&V 20-2009 overview](https://www.osti.gov/servlets/purl/1368927) -- verification hierarchy (HIGH confidence)
- [Firedrake norms module](https://www.firedrakeproject.org/_modules/firedrake/norms.html) -- errornorm API (HIGH confidence)
- [Roache, P.J. "Verification and Validation in Computational Science and Engineering"](https://www.researchgate.net/publication/278408318_Code_Verification_by_the_Method_of_Manufactured_Solutions) -- MMS methodology (HIGH confidence)
- Existing codebase: `mms_bv_convergence.py`, `test_v13_verification.py` (HIGH confidence)
