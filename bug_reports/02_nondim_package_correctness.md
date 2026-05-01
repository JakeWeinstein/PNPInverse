# Bug Report: Nondim Package Correctness

**Focus:** Numerical and scientific correctness in Nondim/ package
**Agent:** Nondim Package Correctness

---

## FINDING: Core nondimensionalization is scientifically correct

The PDE coefficients (electromigration prefactor, Poisson coefficient, charge RHS prefactor) are all derived correctly from first principles in both dimensional and nondimensional modes. Physical constants, thermal voltage, geometric mean reference diffusivity, and unit conversions are all verified correct.

---

## BUG 1: Debye length convention omits factor of 2 for symmetric electrolytes
**File:** `Nondim/scales.py:186`, `Nondim/transform.py:370-374`
**Severity:** MEDIUM
**Description:** The Debye length is computed as `lambda_D = sqrt(eps * V_T / (F * c_ref))`. The standard textbook Debye length for a symmetric 1:1 electrolyte includes a factor of 2 in the denominator: `lambda_D = sqrt(eps * R * T / (2 * F^2 * c_bulk))`. The code's formula differs by `sqrt(2)`. This is NOT a PDE bug -- `debye_length_m` is only used for diagnostics/warnings, and the `poisson_coefficient` is computed independently and correctly. The computed value is a "single-species scaling Debye length" consistent with the PDE formulation.
**Suggested fix:** Add a docstring note clarifying this is the single-ion Debye length (without the factor of `sum(z_i^2)`), or rename to `debye_scale_m`.

## BUG 2: Auto-computed scales incorrect when `*_inputs_are_dimensionless=True` without explicit scales
**File:** `Nondim/transform.py:291-310`
**Severity:** MEDIUM
**Description:** When `diffusivity_inputs_are_dimensionless=True` but no explicit `diffusivity_scale_m2_s` is provided, the auto-computed scale uses the geometric mean of the already-dimensionless input values, which is physically meaningless. For example, if D_vals are `[1.5, 1.6]` (dimensionless), the auto-computed diffusivity_scale would be ~1.55 m^2/s, which is obviously wrong.
**Suggested fix:** Add a validation check: if any `*_inputs_are_dimensionless` flag is True, require the corresponding scale to be explicitly provided, or emit a warning.

## BUG 3: Unused `import copy` statement
**File:** `Nondim/transform.py:72`
**Severity:** LOW
**Description:** `import copy` is imported but never used anywhere in the module.
**Suggested fix:** Remove the unused import.

## BUG 4: `build_solver_options` hardcodes symmetric c_inf for both species
**File:** `Nondim/compat.py:55`
**Severity:** LOW
**Description:** `"c_inf": [float(scales["c_inf_mol_m3"]), float(scales["c_inf_mol_m3"])]` assumes exactly 2 species with identical c_inf values. Will fail silently for multi-species or asymmetric configurations.
**Suggested fix:** Accept n_species as a parameter, or document the 2-species symmetric assumption.

## BUG 5: Faraday constant minor discrepancy
**File:** `Nondim/constants.py:6`
**Severity:** LOW
**Description:** `FARADAY_CONSTANT = 96485.3329` vs. CODATA 2018 exact value of 96485.33212 C/mol. Difference is ~8 ppb relative error -- negligible.

---

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 0     |
| HIGH     | 0     |
| MEDIUM   | 2     |
| LOW      | 3     |

The Nondim package is well-implemented. No bugs affect PDE assembly or results.
