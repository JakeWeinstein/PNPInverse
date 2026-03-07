# Phase 2: MMS Convergence Verification - Research

**Researched:** 2026-03-07
**Domain:** MMS convergence testing, GCI uncertainty quantification, pytest integration
**Confidence:** HIGH

## Summary

Phase 2 wraps the existing `run_mms_4species()` driver from `scripts/verification/mms_bv_4species.py` into a pytest test with formal convergence rate assertions (L2 ~ O(h^2), H1 ~ O(h)) via log-log linear regression, R-squared gating (> 0.99), and Grid Convergence Index (GCI) computation using the Roache 3-grid formula with safety factor Fs = 1.25. The codebase already has all core infrastructure: the 4-species MMS driver returning error dictionaries, the `skip_without_firedrake` decorator, `@pytest.mark.slow` convention, and `scipy.stats.linregress` for log-log fitting (used identically in `test_mms_smoke.py`).

The work is primarily integration: import the existing driver, run it with N = [8, 16, 32, 64], compute convergence rates via log-log regression, assert rate and R-squared thresholds, compute GCI from consecutive mesh triplets, serialize everything to JSON + PNG in `StudyResults/mms_convergence/`, and remove the deprecated simpler MMS scripts and their smoke tests.

**Primary recommendation:** Build `tests/test_mms_convergence.py` by reusing `run_mms_4species()` and the `_check_convergence()` pattern from `test_mms_smoke.py`, adding GCI computation as a standalone helper function, and saving structured JSON output for Phase 6.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Mesh refinement levels: 4 levels N = 8, 16, 32, 64 (N=128 excluded from pytest)
- Only the 4-species case (O2, H2O2, H+, ClO4-) is wrapped as pytest
- Simpler MMS scripts (1-species, 2-species neutral, 2-species charged in `mms_bv_convergence.py`) should be deprecated/removed
- Roache 3-grid GCI formula: GCI = Fs * |f2 - f1| / (r^p - 1), Fs = 1.25
- GCI is output only -- no pytest assertions on GCI values
- Rate assertions (L2 ~ O(h^2), H1 ~ O(h), R-squared > 0.99 on log-log fit) are the pass/fail gate
- Test file: `tests/test_mms_convergence.py`
- Import `run_mms_4species()` from existing `scripts/verification/mms_bv_4species.py`
- Markers: `@pytest.mark.slow` + `@skip_without_firedrake`
- No new custom markers
- Output to `StudyResults/mms_convergence/` with JSON + PNG artifacts

### Claude's Discretion
- Exact structure of the GCI computation function
- How to handle the log-log regression (numpy polyfit vs scipy)
- Convergence plot styling and layout
- Whether to add a GCI table to the text output or only include in JSON
- How to structure the JSON schema for convergence data

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| FWD-01 | MMS convergence tests wrapped in pytest with automated rate assertions (L2 ~ O(h^2), H1 ~ O(h)) | Log-log regression via `scipy.stats.linregress`, R-squared > 0.99 gate, rate range assertions. Pattern already established in `test_mms_smoke.py:_check_convergence()` |
| FWD-03 | 4-species MMS case matching the v13 production configuration (O2, H2O2, H+, ClO4-) | `run_mms_4species()` in `mms_bv_4species.py` already implements this. Import directly into pytest. |
| FWD-05 | Mesh convergence study with Grid Convergence Index (GCI) uncertainty quantification | Roache 3-grid GCI formula with Fs=1.25. Compute from consecutive mesh triplets. Output-only (no assertions). |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pytest | (project dep) | Test framework | Already used throughout `tests/` |
| scipy.stats.linregress | (project dep) | Log-log regression for convergence rates | Already used in `test_mms_smoke.py`; returns slope, R-squared directly |
| numpy | (project dep) | Numerical arrays, log transforms | Already used everywhere |
| matplotlib | (project dep) | Convergence plots | Already used in `mms_bv_4species.py:plot_convergence()` |
| json (stdlib) | N/A | Serialize convergence data | Standard library, no install needed |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| firedrake | (FEM env) | PDE solver backend | Required at runtime; guarded by `skip_without_firedrake` |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| scipy.stats.linregress | numpy.polyfit(log_h, log_err, 1) | linregress gives R-squared directly; polyfit would require manual R-squared computation. Use linregress. |

**Installation:**
No new packages needed. All dependencies already installed.

## Architecture Patterns

### Recommended Project Structure
```
tests/
  test_mms_convergence.py      # NEW: pytest wrapper with rate + GCI assertions
scripts/verification/
  mms_bv_4species.py           # EXISTING: 4-species MMS driver (keep)
  mms_bv_convergence.py        # REMOVE: deprecated simpler MMS scripts
  test_bv_forward.py           # EXISTING: unrelated BV tests (keep)
StudyResults/mms_convergence/
  convergence_data.json         # NEW: machine-readable convergence results
  mms_convergence.png           # NEW: log-log convergence plot
```

### Pattern 1: Lazy Import for Firedrake-Dependent Tests
**What:** Import Firedrake-dependent modules inside test functions or fixtures, not at module level.
**When to use:** Any test that depends on Firedrake.
**Example:**
```python
# Source: tests/test_mms_smoke.py (existing pattern)
def _import_mms_4species():
    """Import MMS functions lazily to avoid Firedrake import errors at collection."""
    from scripts.verification.mms_bv_4species import run_mms_4species
    return run_mms_4species
```

### Pattern 2: Log-Log Regression with R-squared Gate
**What:** Fit log(h) vs log(error) with `linregress`, assert R-squared > threshold and slope within expected range.
**When to use:** Convergence rate verification.
**Example:**
```python
# Source: tests/test_mms_smoke.py:_check_convergence() (existing pattern)
from scipy.stats import linregress

log_h = np.log(np.array(h_vals))
log_err = np.log(np.array(err_vals))
slope, intercept, r_value, p_value, std_err = linregress(log_h, log_err)
r_squared = r_value ** 2

assert r_squared > 0.99, f"R^2 = {r_squared:.6f} < 0.99"
assert 1.8 < slope < 2.2, f"L2 rate = {slope:.3f}, expected ~2.0"
```

### Pattern 3: Class-Level Skip Decorator
**What:** Apply `@skip_without_firedrake` at the class level so all methods inherit it.
**When to use:** Test classes where every method needs Firedrake.
**Example:**
```python
# Source: tests/test_nondim_audit.py (established in Phase 1)
@skip_without_firedrake
@pytest.mark.slow
class TestMMSConvergence:
    """4-species MMS convergence with rate assertions and GCI output."""
    ...
```

### Anti-Patterns to Avoid
- **Top-level Firedrake imports in test modules:** Causes collection failures in environments without Firedrake. Always use lazy imports.
- **Asserting on GCI values:** User decision: GCI is output-only. The rate R-squared is the gate.
- **Re-implementing MMS solve logic in the test:** Import `run_mms_4species()` directly; do not duplicate solver setup code.
- **Using pairwise rates as the convergence gate:** Pairwise rates are noisy. The log-log regression slope over all 4 mesh levels is the proper convergence rate estimate.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Log-log regression | Manual slope calculation from pairs | `scipy.stats.linregress` | Gives slope, R-squared, standard error in one call |
| MMS solve | Inline weak form assembly | `run_mms_4species()` from `mms_bv_4species.py` | Already tested, uses production `build_forms()` |
| Skip decorators | Custom skip logic | `skip_without_firedrake` from `conftest.py` | Established project pattern |
| Convergence plotting | Custom plotting code | Adapt `plot_convergence()` from `mms_bv_4species.py` | Already handles 5-field log-log plots |

**Key insight:** The existing `mms_bv_4species.py` script already does the hard work (MMS source computation, production solver pipeline integration, error norm computation). The test just needs to call it, compute rates via regression, compute GCI, and serialize.

## Common Pitfalls

### Pitfall 1: Firedrake Import at Collection Time
**What goes wrong:** `import firedrake` at module level causes pytest collection to fail in non-Firedrake environments.
**Why it happens:** Firedrake has heavy C-extension dependencies that are absent outside its conda env.
**How to avoid:** Use lazy imports inside functions/fixtures. Pattern established in `test_mms_smoke.py` and `test_bv_forward.py`.
**Warning signs:** `ImportError` during `pytest --collect-only`.

### Pitfall 2: GCI with Non-Constant Refinement Ratio
**What goes wrong:** GCI formula assumes a constant refinement ratio r between grids. If N = [8, 16, 32, 64], r = 2 everywhere, which is correct.
**Why it happens:** If someone changes mesh sizes to non-uniform spacing.
**How to avoid:** Assert r is constant in the GCI function, or compute r per pair. With N = [8, 16, 32, 64], r = 2 is guaranteed.
**Warning signs:** GCI values that seem unreasonably large or negative.

### Pitfall 3: Observed Order p Computation with Oscillatory Convergence
**What goes wrong:** The Roache formula `p = ln((f3-f2)/(f2-f1)) / ln(r)` requires monotonic convergence. If errors oscillate, (f3-f2)/(f2-f1) can be negative, making log undefined.
**Why it happens:** Pre-asymptotic regime, or mesh-dependent solver behavior.
**How to avoid:** Use the log-log regression slope as the observed order (robust to individual outliers). For GCI, use the regression slope as p rather than the local Roache formula when local computation fails.
**Warning signs:** Negative argument to log in GCI computation.

### Pitfall 4: Rate Tolerance Too Tight
**What goes wrong:** Asserting slope == 2.0 exactly fails because CG1 convergence is asymptotic.
**Why it happens:** Pre-asymptotic effects on coarse meshes, boundary layer resolution.
**How to avoid:** Use a range: L2 rate in [1.8, 2.2], H1 rate in [0.8, 1.5]. The R-squared > 0.99 gate catches non-convergent cases.
**Warning signs:** Tests that pass on one machine but fail on another due to small floating-point differences.

### Pitfall 5: JSON Serialization of numpy Types
**What goes wrong:** `json.dumps()` cannot serialize `numpy.float64` or `numpy.int64`.
**Why it happens:** Results dict from `run_mms_4species()` contains numpy types.
**How to avoid:** Use a custom encoder or convert to Python types: `float(x)`, `int(x)`, or `json.dumps(data, default=lambda o: float(o) if hasattr(o, 'item') else o)`.
**Warning signs:** `TypeError: Object of type float64 is not JSON serializable`.

## Code Examples

### Log-Log Convergence Rate Assertion
```python
# Adapted from test_mms_smoke.py:_check_convergence()
from scipy.stats import linregress

def assert_convergence_rate(
    h_vals, err_vals, *,
    expected_rate: float,
    rate_tol: float = 0.2,
    min_r_squared: float = 0.99,
    label: str = "",
):
    """Assert convergence rate via log-log linear regression.

    Returns (slope, r_squared) for reporting.
    """
    log_h = np.log(np.array(h_vals))
    log_err = np.log(np.array(err_vals))
    slope, intercept, r_value, p_value, std_err = linregress(log_h, log_err)
    r_squared = r_value ** 2

    assert r_squared > min_r_squared, (
        f"{label}: R^2 = {r_squared:.6f} < {min_r_squared} "
        f"(slope={slope:.3f}, std_err={std_err:.3f})"
    )
    assert abs(slope - expected_rate) < rate_tol, (
        f"{label}: rate = {slope:.3f}, expected {expected_rate} +/- {rate_tol}"
    )
    return slope, r_squared
```

### GCI Computation (Roache 3-Grid)
```python
# Source: NASA Glenn tutorial + ASME V&V 20 standard
def compute_gci(errors, h_vals, observed_order, Fs=1.25):
    """Compute Grid Convergence Index for consecutive mesh triplets.

    Parameters
    ----------
    errors : list[float]
        Error norms from coarsest to finest mesh.
    h_vals : list[float]
        Mesh sizes from coarsest to finest.
    observed_order : float
        Global observed convergence order (from log-log regression).
    Fs : float
        Safety factor (1.25 for 3+ grids per Roache/ASME V&V 20).

    Returns
    -------
    list[dict]
        GCI info for each consecutive pair (fine/coarse).
    """
    gci_results = []
    for i in range(1, len(errors)):
        r = h_vals[i - 1] / h_vals[i]  # refinement ratio (>1)
        e_coarse = errors[i - 1]
        e_fine = errors[i]

        # GCI_fine = Fs * |epsilon| / (r^p - 1)
        # where epsilon = (e_coarse - e_fine) / e_fine (relative change)
        if e_fine > 0:
            relative_error = abs(e_coarse - e_fine) / e_fine
            gci = Fs * relative_error / (r ** observed_order - 1.0)
        else:
            gci = float('nan')

        gci_results.append({
            "h_fine": h_vals[i],
            "h_coarse": h_vals[i - 1],
            "refinement_ratio": r,
            "error_fine": e_fine,
            "error_coarse": e_coarse,
            "gci": gci,
        })
    return gci_results
```

### JSON Output Schema
```python
# Recommended schema for StudyResults/mms_convergence/convergence_data.json
{
    "metadata": {
        "date": "2026-03-07T12:00:00",
        "mesh_sizes": [8, 16, 32, 64],
        "species": ["O2", "H2O2", "H+", "ClO4-"],
        "safety_factor_Fs": 1.25,
    },
    "fields": {
        "c0_L2": {
            "errors": [1.2e-3, 3.1e-4, 7.8e-5, 1.9e-5],
            "rate": 1.98,
            "r_squared": 0.9999,
            "gci": [
                {"h_fine": 0.0625, "gci": 0.012},
                {"h_fine": 0.03125, "gci": 0.003},
                {"h_fine": 0.015625, "gci": 0.0008},
            ],
        },
        # ... repeat for c1_L2, c2_L2, c3_L2, phi_L2, c0_H1, ..., phi_H1
    },
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Inline MMS weak form | Production `build_forms()` pipeline | Phase 1 (2026-03-06) | MMS now tests the actual production code (FWD-02) |
| Pairwise convergence rates | Log-log regression over all mesh levels | Phase 1 smoke tests | More robust rate estimation |
| No GCI | Roache 3-grid GCI with Fs=1.25 | This phase | Adds uncertainty quantification for V&V report |

**Deprecated/outdated:**
- `scripts/verification/mms_bv_convergence.py`: Contains 1-species, 2-species neutral, 2-species charged MMS drivers. All subsumed by the 4-species case. To be removed.
- `tests/test_mms_smoke.py`: Smoke tests for the deprecated simpler MMS cases. To be removed since the full convergence test in `test_mms_convergence.py` supersedes them.

## Open Questions

1. **Rate tolerance bounds for H1 norm**
   - What we know: CG1 H1 rate should be ~1.0. The existing smoke tests use `min_slope=1.5` which is L2-specific.
   - What's unclear: Exact acceptable range for H1 rate (could be 0.8-1.5 or tighter).
   - Recommendation: Use [0.8, 1.5] for H1 rate tolerance. The R-squared gate catches pathological cases.

2. **Whether to keep `test_mms_smoke.py` or remove it**
   - What we know: User said simpler MMS scripts should be removed. The smoke tests import from `mms_bv_convergence.py`.
   - What's unclear: Whether to remove smoke tests immediately or after verifying the new full convergence test passes.
   - Recommendation: Remove `test_mms_smoke.py` and `mms_bv_convergence.py` as part of this phase. The 4-species convergence test provides strictly stronger coverage.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (already configured in `pyproject.toml`) |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` |
| Quick run command | `pytest tests/test_mms_convergence.py -x -v` |
| Full suite command | `pytest tests/ -m slow -x --tb=short` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| FWD-01 | L2 rate ~ O(h^2), H1 ~ O(h), R^2 > 0.99 | integration | `pytest tests/test_mms_convergence.py::TestMMSConvergence::test_l2_convergence_rates -x` | Wave 0 |
| FWD-03 | 4-species (O2, H2O2, H+, ClO4-) MMS passes | integration | `pytest tests/test_mms_convergence.py::TestMMSConvergence -x` | Wave 0 |
| FWD-05 | GCI values computed and available in JSON output | integration | `pytest tests/test_mms_convergence.py::TestMMSConvergence::test_gci_output -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_mms_convergence.py -x -v` (requires Firedrake env)
- **Per wave merge:** `pytest tests/ -m slow -x --tb=short`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_mms_convergence.py` -- covers FWD-01, FWD-03, FWD-05 (this IS the deliverable)
- No framework install or config gaps -- pytest already configured

## Sources

### Primary (HIGH confidence)
- `scripts/verification/mms_bv_4species.py` -- existing 4-species MMS driver (read in full)
- `tests/test_mms_smoke.py` -- existing convergence smoke tests with `_check_convergence()` pattern
- `tests/conftest.py` -- `skip_without_firedrake` decorator
- `pyproject.toml` -- pytest markers and configuration

### Secondary (MEDIUM confidence)
- [NASA Glenn GCI Tutorial](https://www.grc.nasa.gov/www/wind/valid/tutorial/spatconv.html) -- GCI formula verification
- ASME V&V 20 / Celik et al. (2008) -- GCI methodology standard (Fs=1.25 for 3+ grids)

### Tertiary (LOW confidence)
None -- all findings verified against codebase and authoritative sources.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already in use in the project
- Architecture: HIGH -- patterns directly from existing test files in this codebase
- Pitfalls: HIGH -- identified from actual code patterns and numpy/JSON interop issues
- GCI methodology: HIGH -- verified against NASA Glenn tutorial and ASME standard

**Research date:** 2026-03-07
**Valid until:** 2026-04-07 (stable domain, no fast-moving dependencies)
