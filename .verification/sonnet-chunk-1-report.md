# Re-Verification Report: Chunk 1 (validation.py, forms.py, observables.py)

**Date:** 2026-04-13
**Scope:** Forward/bv_solver/validation.py, forms.py, observables.py
**Focus:** Fix #6 correctness + previously-identified warnings status

---

## Fix #6 — `assemble_observable_validated()` in observables.py

### VERDICT: FIX IS CORRECT AND COMPLETE

**What the fix does (lines 104–113 of observables.py):**
```python
# Only check F2 (magnitude exceeds diffusion limit) for single-observable
# validation.
failures = []
warnings_list = []

if mode in ("current_density", "peroxide_current"):
    if abs(value) > abs(I_lim) * 1.05:
        failures.append(f"F2: |{mode}|={abs(value):.4g} exceeds I_lim={abs(I_lim):.4g}")

result = ValidationResult(valid=len(failures) == 0, failures=failures, warnings=warnings_list)
return value, result
```

**Correctness assessment:**

1. **Scope limited to F2 only — CORRECT.** The old vacuous `validate_observables(cd=0, pc=value)` call would have also fired F3 (peroxide selectivity, with `cd=0` → div-by-zero guard at `abs(cd) > 1e-10` saves it from exploding but silently disables F3) and F7 (wrong-sign check against a `cd=0` baseline). The fix correctly removes those phantom checks. F3 and F7 legitimately require both `cd` and `pc` simultaneously; that belongs at pipeline level.

2. **F2 threshold — MATHEMATICALLY CORRECT.** `abs(value) > abs(I_lim) * 1.05` is numerically identical to the F2 check in `validate_observables` (`abs(cd) > abs(I_lim) * 1.05`, line 154 of validation.py). The 5% tolerance is consistent.

3. **`abs(I_lim)` guard is correct.** Using `abs(I_lim)` means the check is sign-agnostic for `I_lim` input, which is robust (caller may pass a signed or unsigned diffusion limit).

4. **Mode guard `mode in ("current_density", "peroxide_current")` — CORRECT.** Any future mode added (e.g., "reaction") will skip the F2 check until explicitly added, which is safe-by-default behavior.

5. **No import of `validate_observables` in observables.py** — confirmed. The old vacuous call is completely removed; there is no lingering partial call to the old function.

---

## Previously-Identified Warnings — Status

### W1: `exponent_clip` accepted but never used in `validate_solution_state` — STILL PRESENT

`validate_solution_state` accepts `exponent_clip: float` as a parameter (line 53 of validation.py) but never uses it. The W1 check lives in `check_clip_saturation()` (a separate function). Nothing in `validate_solution_state` calls `check_clip_saturation`. This warning was **not fixed** and **is still present**.

Impact: Low. `check_clip_saturation` is a separate callable; callers wanting W1 must call it directly. The unused parameter is confusing but not a correctness bug.

### F1 and F4 double-fire on negative concentrations — STILL PRESENT

In `validate_solution_state` (lines 66–79 of validation.py):
- F1 fires when `c_min < 0.0`
- F4 fires when `c_min <= eps_c * 2.0`

For any `c_min < 0.0`, both F1 and F4 will trigger (since `c_min < 0 < eps_c * 2.0` always). This was **not fixed** and **is still present**.

Impact: Medium. Downstream consumers see two failures for the same root cause (negative concentration). Not a safety issue — both checks correctly flag the solution as invalid — but the redundant F4 message is misleading when the real cause is negative concentration, not floor domination. Callers may de-duplicate by checking F1 first.

### forms.py:346 — `E_eq_j_val != 0.0` should be `is not None` — STILL PRESENT

Line 346 of forms.py:
```python
if E_eq_j_val is not None and E_eq_j_val != 0.0:
```

The `!= 0.0` guard incorrectly skips the per-reaction E_eq when `E_eq_model` is exactly `0.0` (e.g., a reaction with equilibrium potential at the reference), causing fallback to the global `eta_clipped` instead of a reaction-specific one. A reaction with `E_eq = 0.0 V` is physically valid and should use its own E_eq constant.

This was **not fixed** and **is still present**.

Impact: Medium. For reactions with `E_eq_j_val == 0.0`, the per-reaction overpotential silently falls back to the global E_eq. If the global E_eq is also 0.0, behavior is accidentally correct. If they differ, the wrong E_eq is used — a silent physics error.

---

## New Issues Introduced by Fix #6

**None.** The fix is additive/replacement only, confined to `assemble_observable_validated`. No other functions were modified. The `validate_observables` function in validation.py is untouched and remains available for pipeline-level callers.

One minor style note: the `phi_applied` and `V_T` parameters are now accepted by `assemble_observable_validated` but unused (they were previously forwarded to `validate_observables`). They are dead parameters. This is not a correctness bug but could cause confusion.

---

## Summary Table

| Issue | Status | Severity |
|---|---|---|
| Fix #6: F2-only check in `assemble_observable_validated` | CORRECT | — |
| Fix #6: F2 threshold `abs(value) > abs(I_lim) * 1.05` | CORRECT | — |
| W1: `exponent_clip` unused in `validate_solution_state` | NOT FIXED (still present) | Low |
| F1+F4 double-fire on negative concentrations | NOT FIXED (still present) | Medium |
| forms.py:346 `!= 0.0` guard instead of `is not None` | NOT FIXED (still present) | Medium |
| New issues introduced by Fix #6 | NONE | — |
| Dead params `phi_applied`, `V_T` in `assemble_observable_validated` | NEW (minor) | Low |
