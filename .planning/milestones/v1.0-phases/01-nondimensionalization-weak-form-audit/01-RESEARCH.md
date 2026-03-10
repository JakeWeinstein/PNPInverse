# Phase 1: Nondimensionalization & Weak Form Audit - Research

**Researched:** 2026-03-06
**Domain:** Nondimensionalization correctness verification, MMS weak form correspondence
**Confidence:** HIGH

## Summary

Phase 1 has two distinct work streams: (1) nondimensionalization roundtrip tests that verify the `build_model_scaling()` transform is mathematically correct for all parameter types across multiple species configurations, and (2) a weak form audit confirming the MMS convergence script (`mms_bv_convergence.py`) tests the same PDE weak form as the production `bv_solver/forms.py`.

After reading the codebase thoroughly, the MMS script builds its own weak form inline -- it does NOT import from `Forward.bv_solver.forms`. The MMS script constructs its own diffusion, electromigration, BV boundary flux, and Poisson terms directly in each `run_mms_*` function. This is the exact problem FWD-02 was created to address. The refactor is non-trivial because the production `build_forms()` is tightly coupled to the full solver infrastructure (solver_params tuple, Robin/BV config parsing, log-diffusivity controls, steric terms, concentration regularization, adaptive dt), while MMS needs a stripped-down steady-state version with MMS source injection.

The nondim roundtrip testing is straightforward pure-Python work. The existing `test_nondim.py` has ~40 tests covering basic scaling arithmetic, but lacks explicit roundtrip tests (physical -> nondim -> physical recovery) and does not cover 4-species configurations. The existing tests appear mathematically sound based on code inspection but need formal hand-verification against textbook formulas.

**Primary recommendation:** Structure as two sequential work streams: (A) validate existing tests then add roundtrip tests (pure Python, no Firedrake), then (B) refactor MMS to share production weak form code and run a smoke convergence check.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Cover 1-species, 2-species, and 4-species (O2, H2O2, H+, ClO4-) configurations
- Test ALL transform inputs: D, c0, c_inf, phi, dt/t_end, kappa, plus derived quantities (Debye length, flux scale, current density scale)
- Only test `enabled=True` (nondim) mode; `enabled=False` is trivial identity and already covered by existing tests
- Use both v13 production parameter values AND parametrized synthetic values covering different orders of magnitude
- Roundtrip tolerance: `rel=1e-12`
- Hand-check existing `test_nondim.py` assertions against textbook formulas (thermal voltage, Debye length, scale relationships) before building new tests on top
- Document which tests were audited and confirmed correct
- User has no confidence in existing tests -- validation is a prerequisite, not optional
- Refactor MMS script to import and use production `bv_solver` weak form code, not its own inline assembly
- Align with `bv_solver` (Butler-Volmer BC solver) specifically -- this is what v13 uses
- If refactor reveals a bug in production weak form: fix the production code (per PROJECT.md: "don't change solver code unless a bug is found during verification")
- Produce a written audit document (term-by-term correspondence) PLUS passing tests -- both feed into Phase 6 V&V report
- Include a light MMS smoke test (2-3 mesh sizes) to confirm convergence still works after refactor; full rate assertions with GCI belong in Phase 2
- R-squared > 0.99 on log-log fit (matches Phase 2 success criteria)
- Full rate assertions (L2 ~ O(h^2), H1 ~ O(h)) and GCI are Phase 2 scope

### Claude's Discretion
- Exact structure of the roundtrip test parametrization
- How to extract/share weak form building code between MMS and production solver
- Synthetic parameter value ranges for edge case coverage
- Organization of the textbook verification notes

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope

</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| FWD-02 | MMS weak form audit confirming MMS tests the production `bv_solver.py` weak form, not a hand-built replica | MMS script analysis confirms it builds its own inline weak form; refactor strategy documented in Architecture Patterns section |
| FWD-04 | Nondimensionalization roundtrip tests verifying physical -> nondim -> physical identity for all parameter types | `build_model_scaling()` API fully analyzed; roundtrip test pattern documented in Code Examples section |

</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pytest | >=7.0 | Test framework | Already used in `tests/` |
| numpy | >=1.24 | Numerical computation for nondim | Already used in `Nondim/` |
| firedrake | (installed) | FEM assembly for MMS tests | Production solver dependency |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest.approx | built-in | Floating-point comparison | All roundtrip assertions (`rel=1e-12`) |
| pytest.mark.parametrize | built-in | Multi-config test parametrization | 1/2/4-species roundtrip tests |
| pytest.mark.skipif | built-in | Skip Firedrake-dependent tests | MMS smoke tests |
| scipy.stats.linregress | >=1.10 | Log-log regression for convergence R-squared | MMS smoke test rate fitting |

**No new dependencies needed.** Everything is already installed.

## Architecture Patterns

### Recommended Project Structure
```
tests/
  test_nondim.py            # Extended with roundtrip tests (new classes)
  test_nondim_audit.py      # NEW: textbook verification audit tests
  test_mms_smoke.py         # NEW: MMS smoke test (pytest-wrapped, 2-3 meshes)
  conftest.py               # Existing, add firedrake skip marker
Forward/
  bv_solver/
    forms.py                # Production weak form (may need minor refactor for reuse)
    weak_form_core.py       # NEW or refactored: extractable weak form building blocks
scripts/
  verification/
    mms_bv_convergence.py   # REFACTORED: imports from Forward.bv_solver
    WEAK_FORM_AUDIT.md      # NEW: term-by-term audit document
```

### Pattern 1: Nondim Roundtrip Test
**What:** Physical values -> `build_model_scaling()` -> extract scales -> multiply back -> compare to original.
**When to use:** For every parameter type (D, c0, c_inf, phi, dt/t_end, kappa, derived quantities).
**Example:**
```python
# Roundtrip pattern for diffusivity
def test_roundtrip_diffusivity(D_phys, params):
    scaling = build_model_scaling(
        params=params, n_species=len(D_phys), dt=0.1, t_end=1.0,
        D_vals=D_phys, c0_vals=[100.0]*len(D_phys),
        phi_applied=0.05, phi0=0.0,
    )
    D_model = scaling["D_model_vals"]
    D_scale = scaling["diffusivity_scale_m2_s"]
    D_recovered = [d * D_scale for d in D_model]
    for orig, recov in zip(D_phys, D_recovered):
        assert recov == pytest.approx(orig, rel=1e-12)
```

### Pattern 2: MMS Weak Form Extraction Strategy
**What:** Extract the core weak form assembly from `build_forms()` into a reusable function that both the production solver and MMS script can call.
**When to use:** For FWD-02 audit compliance.

There are two viable approaches:

**Option A (Recommended): Thin wrapper for MMS.** Create an MMS-specific helper that calls `build_forms()` with appropriate solver_params, then adds MMS source terms and boundary corrections to the returned `F_res`. This avoids modifying `build_forms()` at all.

**Option B: Extract weak form core.** Factor out the weak form assembly (NP terms, Poisson, BV boundary) from `build_forms()` into a standalone function that both `build_forms()` and MMS call. This is cleaner but requires more code changes to production code.

**Decision: Use Option A.** It minimizes production code changes (per PROJECT.md constraint) and the audit document simply confirms term-by-term correspondence between the MMS wrapper's additions and the production `build_forms()` output.

Key challenge: `build_forms()` expects the full 11-element `solver_params` tuple and returns a context dict with Firedrake UFL forms. MMS needs to inject source terms (`S_c`, `S_phi`) and boundary corrections (`g_i`) into `F_res`. The MMS refactor will:
1. Build solver_params with MMS-appropriate nondim configuration
2. Call `build_forms()` to get `F_res`
3. Add MMS source terms: `F_res -= S_c * v * dx` for each species, `F_res -= S_phi * w * dx`
4. Add boundary corrections: `F_res -= g_i * v * ds(electrode)` for each species

### Pattern 3: Textbook Formula Verification
**What:** Independent computation of expected nondim values from textbook definitions, compared against code output.
**When to use:** For auditing existing `test_nondim.py` correctness.
```python
# Example: verify thermal voltage independently
V_T_expected = R * T / F  # = 8.314 * 298.15 / 96485 = 0.02569 V
# Verify Debye length independently
lambda_D_expected = sqrt(eps_r * eps_0 * R * T / (F**2 * c_ref))
```

### Anti-Patterns to Avoid
- **Modifying production `build_forms()` for MMS convenience:** The audit should confirm MMS uses production code, not that production code was modified to suit MMS. Minimal changes only.
- **Testing nondim with `enabled=False`:** User explicitly excluded this -- it is trivial identity and already tested.
- **Hand-rolling convergence rate computation in tests:** Use `scipy.stats.linregress` on log-log data for R-squared, not manual pairwise rate calculation.
- **Making MMS tests depend on having Firedrake in CI:** Use `@pytest.mark.skipif(not FIREDRAKE_AVAILABLE)` from conftest.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Floating-point comparison | Custom epsilon checks | `pytest.approx(val, rel=1e-12)` | Handles edge cases, clear error messages |
| Log-log regression | Manual rate = log(e1/e2)/log(h1/h2) | `scipy.stats.linregress(log(h), log(e))` | Gives R-squared directly, handles >2 points properly |
| Parametrized test configs | Manual loops in test functions | `@pytest.mark.parametrize` with fixture IDs | Better reporting, isolated failures |
| Weak form assembly for MMS | Inline UFL terms (current approach) | Import from `Forward.bv_solver.forms.build_forms()` | Entire point of FWD-02 |

## Common Pitfalls

### Pitfall 1: Roundtrip Tests Passing Trivially
**What goes wrong:** If roundtrip tests just test `x / scale * scale == x`, they always pass even if the scale is wrong.
**Why it happens:** The test verifies arithmetic identity, not physical correctness.
**How to avoid:** Combine roundtrips with independent textbook-derived expected values. For example, verify that `D_model = D_phys / D_ref` where `D_ref = geometric_mean(D_phys)` -- check both the division AND that the scale was computed correctly.
**Warning signs:** All roundtrip tests pass immediately without any code changes.

### Pitfall 2: MMS Refactor Breaks Convergence
**What goes wrong:** After refactoring MMS to use production weak form, convergence rates degrade or solve diverges.
**Why it happens:** Production `build_forms()` includes features the old MMS script didn't (log-diffusivity controls, concentration regularization, steric terms, exponent clipping). These can subtly alter the numerical problem.
**How to avoid:** Disable all optional features in the MMS solver_params config: `regularize_concentration=False`, `clip_exponent=False`, `a_vals=[0]*n`, `use_eta_in_bv=True`. Run smoke test BEFORE and AFTER refactor to confirm identical results.
**Warning signs:** Newton iteration count changes significantly after refactor.

### Pitfall 3: Marker Convention Mismatch
**What goes wrong:** MMS uses `RectangleMesh` with markers 1=left, 2=right, 3=bottom, 4=top. Production `build_forms()` uses configurable markers defaulting to electrode=1, ground=3.
**Why it happens:** The mesh marker convention differs between 1D interval meshes (used in production) and 2D rectangle meshes (used in MMS).
**How to avoid:** Explicitly set `electrode_marker=3`, `concentration_marker=4`, `ground_marker=4` in the MMS solver_params to match `RectangleMesh` conventions.
**Warning signs:** BCs applied to wrong boundaries, zero BV flux.

### Pitfall 4: Nondim vs Dimensional Mode in MMS
**What goes wrong:** The current MMS script operates in nondimensional space with hardcoded coefficients (e.g., `em=1.0`, `eps_hat=0.01`). The production solver can run in either mode.
**Why it happens:** MMS was written standalone, not through the production nondim pipeline.
**How to avoid:** Configure MMS to use `nondim.enabled=True` with `*_inputs_are_dimensionless=True` flags so MMS's nondimensional parameters pass through unchanged. Set `potential_scale_v=V_T` so `electromigration_prefactor=1.0`.
**Warning signs:** Electromigration prefactor != 1.0 when MMS expects it to be.

### Pitfall 5: 4-Species Configuration for Roundtrip Tests
**What goes wrong:** The v13 production 4-species config (O2, H2O2, H+, ClO4-) uses the multi-reaction BV path with `bv_bc.reactions`, not the legacy per-species path. Roundtrip tests that only test the base `build_model_scaling()` miss the BV-specific scaling in `_add_bv_reactions_scaling_to_transform()`.
**Why it happens:** `build_model_scaling()` handles base nondim; BV parameters (k0, c_ref, E_eq) are scaled separately.
**How to avoid:** Roundtrip tests for 4-species must also verify BV-specific scaling by calling `_add_bv_reactions_scaling_to_transform()` or by going through `build_forms()`.
**Warning signs:** 4-species roundtrip tests only check D/c0/phi, not k0/c_ref/E_eq.

### Pitfall 6: build_model_scaling() Auto-Computes Scales from Inputs
**What goes wrong:** When `diffusivity_scale_m2_s` is not explicitly provided, `build_model_scaling()` computes it as the geometric mean of input D values. If a roundtrip test doesn't account for this, the "expected" scale won't match.
**Why it happens:** The function has auto-computed defaults for scales: `D_ref = exp(mean(log(D_arr)))`, `c_ref = max(|c_all|)`, `L_ref = 1e-4`.
**How to avoid:** Either (a) explicitly provide all scales in the nondim config, or (b) compute expected auto-scales in the test using the same formula.
**Warning signs:** Roundtrip tests fail with small relative errors because the scale was auto-computed differently than expected.

## Code Examples

### Roundtrip Test for All Parameter Types (1-species)
```python
@pytest.mark.parametrize("n_species,D_phys,c0_phys,c_inf_phys,phi,kappa", [
    (1, [9.311e-9], [100.0], [10.0], 0.05, [1e-4]),
    (2, [9.311e-9, 5.273e-9], [100.0, 50.0], [10.0, 5.0], 0.1, [1e-4, 2e-4]),
    (4, [1.97e-9, 1.4e-9, 9.311e-9, 1.792e-9],
        [0.27, 0.01, 0.01, 0.01],  # mol/m3
        [0.27, 0.01, 0.01, 0.01],
        0.05,
        [1e-4, 1e-4, 1e-4, 1e-4]),
])
def test_nondim_roundtrip(n_species, D_phys, c0_phys, c_inf_phys, phi, kappa):
    from Nondim.transform import build_model_scaling

    params = {
        "nondim": {"enabled": True},
        "robin_bc": {
            "kappa": kappa,
            "c_inf": c_inf_phys,
            "electrode_marker": 1,
            "concentration_marker": 3,
            "ground_marker": 3,
        },
    }
    dt_phys, t_end_phys = 0.001, 1.0

    scaling = build_model_scaling(
        params=params, n_species=n_species,
        dt=dt_phys, t_end=t_end_phys,
        D_vals=D_phys, c0_vals=c0_phys,
        phi_applied=phi, phi0=0.0,
    )

    # Roundtrip: model * scale = physical
    D_scale = scaling["diffusivity_scale_m2_s"]
    for orig, model in zip(D_phys, scaling["D_model_vals"]):
        assert model * D_scale == pytest.approx(orig, rel=1e-12)

    c_scale = scaling["concentration_scale_mol_m3"]
    for orig, model in zip(c0_phys, scaling["c0_model_vals"]):
        assert model * c_scale == pytest.approx(orig, rel=1e-12)

    for orig, model in zip(c_inf_phys, scaling["c_inf_model_vals"]):
        assert model * c_scale == pytest.approx(orig, rel=1e-12)

    phi_scale = scaling["potential_scale_v"]
    assert scaling["phi_applied_model"] * phi_scale == pytest.approx(phi, rel=1e-12)

    t_scale = scaling["time_scale_s"]
    assert scaling["dt_model"] * t_scale == pytest.approx(dt_phys, rel=1e-12)
    assert scaling["t_end_model"] * t_scale == pytest.approx(t_end_phys, rel=1e-12)

    kappa_scale = scaling["kappa_scale_m_s"]
    for orig, model in zip(kappa, scaling["kappa_model_vals"]):
        assert model * kappa_scale == pytest.approx(orig, rel=1e-12)
```

### Textbook Verification Pattern
```python
def test_thermal_voltage_textbook():
    """Verify V_T = RT/F against NIST values."""
    from Nondim.constants import GAS_CONSTANT, FARADAY_CONSTANT
    from Nondim.scales import build_physical_scales

    scales = build_physical_scales(d_species_m2_s=(1e-9,), c_bulk_m=0.1)
    # NIST 2018 CODATA: R = 8.314462618 J/(mol*K), F = 96485.33212 C/mol
    V_T_textbook = 8.314462618 * 298.15 / 96485.3329
    assert scales.thermal_voltage_v == pytest.approx(V_T_textbook, rel=1e-10)
    # Sanity: ~25.69 mV at 25 C
    assert abs(scales.thermal_voltage_v - 0.02569) < 0.0001

def test_debye_length_textbook():
    """Verify lambda_D = sqrt(eps*V_T / (F*c)) against Bard & Faulkner."""
    import math
    from Nondim.constants import (
        FARADAY_CONSTANT, VACUUM_PERMITTIVITY_F_PER_M,
        DEFAULT_RELATIVE_PERMITTIVITY_WATER,
    )
    from Nondim.scales import build_physical_scales

    c_bulk_M = 0.1  # 0.1 M = 100 mol/m3
    c_mol_m3 = c_bulk_M * 1000.0
    eps = DEFAULT_RELATIVE_PERMITTIVITY_WATER * VACUUM_PERMITTIVITY_F_PER_M

    scales = build_physical_scales(d_species_m2_s=(1e-9,), c_bulk_m=c_bulk_M)
    V_T = scales.thermal_voltage_v
    lambda_D_expected = math.sqrt(eps * V_T / (FARADAY_CONSTANT * c_mol_m3))
    assert scales.debye_length_m == pytest.approx(lambda_D_expected, rel=1e-10)
    # For 0.1M 1:1 electrolyte, Debye length ~ 0.97 nm
    assert 0.5e-9 < scales.debye_length_m < 2e-9
```

### MMS Smoke Test Pattern (Post-Refactor)
```python
import pytest
import numpy as np
from scipy.stats import linregress

skip_without_firedrake = pytest.mark.skipif(
    not _firedrake_available(), reason="Firedrake not available"
)

@skip_without_firedrake
def test_mms_single_species_convergence():
    """Smoke test: L2 convergence rate ~ 2 for CG1 on 2-3 mesh sizes."""
    from scripts.verification.mms_bv_convergence import run_mms_single_species

    results = run_mms_single_species([8, 16, 32], verbose=False)
    h = np.array(results["h"])
    err = np.array(results["c_L2"])

    slope, _, r_value, _, _ = linregress(np.log(h), np.log(err))
    assert r_value**2 > 0.99, f"R^2 = {r_value**2:.4f} < 0.99"
    # Slope should be near 2.0 for L2 norm with CG1
    assert slope > 1.5, f"L2 rate = {slope:.2f} < 1.5 (expected ~2.0)"
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| MMS with hand-built inline weak forms | MMS importing production weak form code | This phase | Ensures MMS actually tests production code |
| Nondim with `eps_coeff=1.0` for disabled mode | Nondim using actual permittivity | Already done in codebase | Fixed incorrect Poisson equation for non-symmetric species |
| No roundtrip tests | Roundtrip for all parameter types | This phase | Catches scaling bugs that silently produce wrong results |

## Open Questions

1. **How tightly coupled is `build_forms()` to the 11-element solver_params tuple?**
   - What we know: It destructures the tuple on line 86 and uses all 11 elements. It requires a Firedrake mesh in `ctx`.
   - What's unclear: Whether the MMS refactor can construct valid solver_params without understanding all the internal config parsing.
   - Recommendation: Build a minimal `solver_params` tuple for MMS that sets all optional features to their simplest defaults. Test that `build_forms()` accepts it before proceeding with the full refactor.

2. **Does the production solver's `build_forms()` use a time-stepping residual while MMS tests steady state?**
   - What we know: Yes. `build_forms()` includes `(c_i - c_old) / dt_const * v * dx` time-stepping terms. The MMS script currently solves a steady-state problem (no time derivative).
   - What's unclear: Whether the MMS refactor should use the production time-stepping with a single large timestep, or add a steady-state option.
   - Recommendation: Set `U_prev` equal to the MMS initial guess and use a very large `dt` (effectively infinite) so the time-stepping term contributes negligibly. Alternatively, subtract the time-stepping term from `F_res` after `build_forms()` returns. The former is simpler and more honest about testing the actual production code.

3. **v13 4-species parameters for roundtrip tests?**
   - What we know: The CONTEXT.md specifies testing with "v13 production parameter values" but doesn't list them explicitly.
   - What's unclear: Exact D values, concentrations, BV parameters for the 4-species v13 config.
   - Recommendation: Search the codebase for v13 parameter definitions during implementation (likely in an inference script or config file).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >=7.0 |
| Config file | none -- see Wave 0 |
| Quick run command | `python -m pytest tests/test_nondim.py tests/test_nondim_audit.py -x -q` |
| Full suite command | `python -m pytest tests/test_nondim.py tests/test_nondim_audit.py tests/test_mms_smoke.py -v` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| FWD-04 | Nondim roundtrip: physical -> nondim -> physical identity for D, c0, c_inf, phi, dt, kappa | unit | `python -m pytest tests/test_nondim.py::TestNondimRoundtrip -x` | No -- Wave 0 |
| FWD-04 | Nondim roundtrip: derived quantities (Debye length, flux scale, current density) | unit | `python -m pytest tests/test_nondim.py::TestDerivedQuantityConsistency -x` | No -- Wave 0 |
| FWD-04 | Textbook formula verification (V_T, lambda_D, scale relationships) | unit | `python -m pytest tests/test_nondim_audit.py -x` | No -- Wave 0 |
| FWD-02 | MMS weak form uses production build_forms() | integration | `python -m pytest tests/test_mms_smoke.py -x` | No -- Wave 0 |
| FWD-02 | MMS convergence R^2 > 0.99 on log-log fit (smoke, 2-3 meshes) | integration | `python -m pytest tests/test_mms_smoke.py::test_mms_convergence_smoke -x` | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/test_nondim.py tests/test_nondim_audit.py -x -q`
- **Per wave merge:** `python -m pytest tests/test_nondim.py tests/test_nondim_audit.py tests/test_mms_smoke.py -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_nondim_audit.py` -- covers FWD-04 textbook verification
- [ ] `tests/test_mms_smoke.py` -- covers FWD-02 MMS weak form audit
- [ ] New test classes in `tests/test_nondim.py` -- covers FWD-04 roundtrip tests
- [ ] `scripts/verification/WEAK_FORM_AUDIT.md` -- covers FWD-02 written audit document
- [ ] `pytest.ini` or `pyproject.toml` test config -- optional but recommended for marker registration

## Sources

### Primary (HIGH confidence)
- Direct code inspection of `Nondim/transform.py` (build_model_scaling, 445 lines)
- Direct code inspection of `Nondim/scales.py` (build_physical_scales, 206 lines)
- Direct code inspection of `Nondim/constants.py` (physical constants, 12 lines)
- Direct code inspection of `Forward/bv_solver/forms.py` (build_forms, 372 lines)
- Direct code inspection of `Forward/bv_solver/nondim.py` (BV scaling, 130 lines)
- Direct code inspection of `Forward/bv_solver/config.py` (BV config parsing, 125 lines)
- Direct code inspection of `scripts/verification/mms_bv_convergence.py` (MMS script, 910 lines)
- Direct code inspection of `tests/test_nondim.py` (existing tests, 572 lines)
- Direct code inspection of `tests/conftest.py` (shared fixtures, 129 lines)

### Secondary (MEDIUM confidence)
- PNP nondimensionalization theory (standard textbook: Bard & Faulkner, Newman & Thomas-Alyea)
- MMS verification methodology (Roache, "Verification and Validation in Computational Science and Engineering")

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already in use in the project
- Architecture: HIGH -- all source files read and analyzed in detail
- Pitfalls: HIGH -- identified from direct code analysis of production/MMS differences
- MMS refactor feasibility: MEDIUM -- approach is clear but untested; build_forms() coupling may surface unexpected issues

**Research date:** 2026-03-06
**Valid until:** 2026-04-06 (stable domain, no moving dependencies)
