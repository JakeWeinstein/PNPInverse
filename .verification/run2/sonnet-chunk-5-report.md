# Second-Pass Verification Report
## Scope: Nondim/transform.py + Nondim/constants.py

**Verifier:** Claude Sonnet 4.6 (independent second pass)
**Date:** 2026-05-02
**Files verified in full:**
- `Nondim/transform.py` (481 lines)
- `Nondim/constants.py` (11 lines)

**Context files read (not verified):**
- `Forward/bv_solver/forms_logc.py`
- `Forward/bv_solver/nondim.py`
- `scripts/_bv_common.py`

---

## NO CRITICAL ISSUES FOUND

No critical or major bugs were found. The first-pass findings are confirmed. All PDE prefactors are mathematically correct for the production config.

---

## Prefactor Verification (Numerical)

Using production scales: D_REF=1.9e-9, C_SCALE=0.5, L_REF=1e-4, V_T≈0.025693 V, eps=78.5·8.854e-12 F/m.

| Prefactor | Formula | Computed value | Expected |
|---|---|---|---|
| `electromigration_prefactor` | (F/RT)·V_T | 1.000000000000000 | 1.0 exactly |
| `poisson_coefficient` | ε·V_T/(F·c_ref·L²) | 3.702e-8 | (λ_D/L)² |
| `charge_rhs_prefactor` | hardcoded 1.0 | 1.0 | 1.0 (nondim RHS) |
| `current_density_scale_a_m2` | F·D_ref·c_scale/L_ref | 9.166e-1 A/m² | correct |
| `debye_length_m` | √(ε·V_T/(F·c_ref)) | 1.924e-8 m | correct |
| `debye_to_length_ratio` | λ_D/L | 1.924e-4 | correct (< 1, well-separated) |

The `electromigration_prefactor` is exactly 1.0 at machine precision when `potential_scale_v = V_T`, as intended by the PDE nondimensionalization.

---

## Key Completeness: build_model_scaling → forms_logc.py

All keys consumed by `forms_logc.py` are unconditionally present in the nondim-enabled return dict:

| Key consumed | Location in forms_logc | Present in build_model_scaling? |
|---|---|---|
| `D_model_vals` | line 169 | Yes |
| `electromigration_prefactor` | line 200 | Yes |
| `dt_model` | line 201 | Yes |
| `phi_applied_model` | line 205 | Yes |
| `phi0_model` | line 207 | Yes |
| `poisson_coefficient` | line 415 | Yes |
| `charge_rhs_prefactor` | line 416 | Yes |
| `c0_model_vals` | line 431 (via .get) | Yes |
| `potential_scale_v` | line 141 (Stern path) | Yes |
| `length_scale_m` | line 140 (Stern path) | Yes |
| `concentration_scale_mol_m3` | line 142 (Stern path) | Yes |

Note: `bv_E_eq_model` and `bv_exponent_scale` (consumed at lines 210-211) are NOT returned by `build_model_scaling` directly. They are added by `_add_bv_scaling_to_transform` or `_add_bv_reactions_scaling_to_transform` (in `nondim.py`), which `build_forms_logc` always calls before those lines. This is correct by construction.

---

## Helper Function Safety

### `_bool(value)` (transform.py:109-124)
- `bool` literals: fast path, correct.
- `str`: strip/lower dispatch with explicit known-good set, raises `ValueError` for unknown strings.
- `None`: raises `ValueError` with clear message.
- `int` 0/1: falls to `bool(value)` — correct.
- `numpy.bool_`: `isinstance(numpy.bool_(True), bool)` is `False` in Python, so falls to `bool(value)` — returns correct result.
- No silent swallowing. Exception-safe for all production input types.

### `_as_list(values, n, name)` (transform.py:90-98)
- Scalar broadcast: `np.isscalar` handles float, int, numpy scalar — all correct.
- Sequence path: `TypeError` is caught and re-raised as `ValueError` with context.
- Length mismatch: raises `ValueError`.
- Exception-safe for all production input types.

### `_pos(value, name)` (transform.py:102-106)
- `float(value)` is not wrapped in try/except. A non-numeric type (e.g., list) would propagate `TypeError` instead of `ValueError`. This is a minor API inconsistency.
- In production all `_pos` inputs are numeric (from `dict.get` with numeric defaults), so this never triggers.
- **SEVERITY: minor** (see below).

---

## Issues Found

### MINOR-1: `_pos` propagates `TypeError` instead of `ValueError` for non-numeric input

**LOCATION:** `transform.py:103` (`_pos` function)

**DESCRIPTION:** `float(value)` is called without a try/except. If a caller accidentally passes a non-numeric type (e.g., a list), `float()` raises `TypeError`. All other helpers (`_as_list`, `_bool`) raise `ValueError`. The inconsistency could confuse callers catching `ValueError` to validate inputs.

**Impact in production:** None. All `_pos` call sites receive numeric values from `dict.get(..., numeric_default)`.

**Smallest fix:**
```python
def _pos(value: Any, name: str) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive number; got {value!r}") from exc
    if v <= 0.0:
        raise ValueError(f"{name} must be > 0; got {v}.")
    return v
```

---

### MINOR-2: `verify_model_params` uses `print()` while `build_model_scaling` uses `warnings.warn()`

**LOCATION:** `transform.py:459-481` (`verify_model_params`)

**DESCRIPTION:** The heuristic sanity checks in `verify_model_params` write to stdout via `print()`. The warnings in `build_model_scaling` (lines 342-365) correctly use `warnings.warn()`. This means the heuristic check output bypasses any `warnings.filterwarnings` configuration, cannot be suppressed without monkey-patching stdout, and does not include the usual warnings metadata (category, stacklevel).

**Impact in production:** Low — these are heuristic-only checks, not correctness gates. But it means CI/log filtering cannot mute them.

**Smallest fix:** Replace each `print(...)` in `verify_model_params` with `warnings.warn(..., stacklevel=2)`.

---

### MINOR-3: Documentation gap — diffusivity_scale warning does not mention cascade to time_scale

**LOCATION:** `transform.py:341-345` (warning for `d_inputs_dimless` without explicit scale)

**DESCRIPTION:** When `diffusivity_inputs_are_dimensionless=True` and `diffusivity_scale_m2_s` is not provided, the auto-computed scale (geometric mean of the already-dimensionless D values) is physically meaningless. The warning says "auto-computed scale may be meaningless." It does NOT mention that `time_scale_s` (if also not provided) will be derived from this bad scale, compounding the error.

The dangerous config is: `d_inputs_dimless=True`, `time_inputs_dimless=False`, neither `diffusivity_scale_m2_s` nor `time_scale_s` explicitly set. The user gets one warning but the time discretization is silently corrupted.

**Impact in production:** None (production always provides all six explicit scales).

**Smallest fix:** Extend the existing warning message:
```python
warnings.warn(
    "diffusivity_inputs_are_dimensionless=True but no explicit "
    "diffusivity_scale_m2_s provided; auto-computed scale may be meaningless. "
    "If time_scale_s is also not provided, the time discretization will be "
    "derived from this scale and will also be incorrect."
)
```

---

## Items Confirmed Correct from First Pass

All first-pass findings are re-confirmed:

1. **electromigration_prefactor = 1.0 exactly** when `potential_scale_v = V_T`. Verified numerically at machine precision.
2. **poisson_coefficient = (λ_D/L)² ≈ 3.7e-8**. Formula `ε·V_T/(F·c_ref·L²)` is correct.
3. **charge_rhs_prefactor = 1.0** hardcoded for nondim case. Correct (RHS is dimensionless `Σ zᵢ ĉᵢ`).
4. **current_density_scale = F·D_ref·c_scale/L_ref**. Correct for single-electron; callers multiply by `n_e`.
5. **Faraday constant 96485.3329** (CODATA 2014). Discrepancy from CODATA 2018 is 8 ppb — negligible.
6. **Stern capacitance nondimensionalization** is consistent between `nondim.py` (legacy BV path) and `forms_logc.py` (reactions path). Both use `stern_raw * potential_scale / (F * conc_scale * length_scale)`.
7. **`_get_robin_cfg` is bypassed** in the BV logc path (dummy_robin is constructed explicitly at `forms_logc.py:98-103` and passed as `robin=`). The fallback logic in `_get_robin_cfg` is never exercised by the production BV stack.
8. **`phi0_func`** is stored in context as metadata only; it is not used in the weak form. `phi0=0.0` in production — no correctness concern.
9. **Auto-scale for `concentration_scale`** uses `max(1e-16, np.max(np.abs(c_all)))`. With C_H2O2=0, the max picks up C_O2=1.0. Correct. The `1e-16` floor prevents division by zero when all concentrations happen to be zero.
10. **kappa warning** (line 361-365): Production provides `kappa_scale_m_s` explicitly, so this warning is never triggered.

---

## Summary

No critical issues. No major issues. Three minor issues found (two of which were pre-identified in the first pass). The module is mathematically correct for the production config and the PDE prefactors match the nondimensionalization writeup.
