# SONNET_08 — Non-dimensionalization Verification

**Scope:** Static review of scale definitions and consistency.
**Files reviewed:** `Forward/bv_solver/nondim.py`, `Forward/bv_solver/units.py`,
`Nondim/constants.py`, `Nondim/transform.py`, `scripts/_bv_common.py`,
`Forward/bv_solver/forms_logc_muh.py` (selective grep).
**Date:** 2026-05-22

---

## 1. Physical constants

**Source of truth:** `Nondim/constants.py`

| Constant | Stored value | Standard value | Status |
|---|---|---|---|
| `FARADAY_CONSTANT` | 96485.3329 C/mol | 96485.3321 C/mol (CODATA 2018) | OK (within 1e-5 relative) |
| `GAS_CONSTANT` | 8.314462618 J/(mol·K) | 8.314462618 J/(mol·K) | EXACT |
| `DEFAULT_TEMPERATURE_K` | 298.15 K | 298.15 K (25 °C) | EXACT |
| `VACUUM_PERMITTIVITY_F_PER_M` | 8.8541878128e-12 F/m | 8.8541878128e-12 F/m | EXACT |
| `DEFAULT_RELATIVE_PERMITTIVITY_WATER` | 78.5 | ~78.5 at 25 °C | OK |
| `V_T` (derived in `_bv_common.py`) | R·T/F = **0.025693 V** | 0.025693 V at 298.15 K | OK |

All constants are correct. `V_T` is computed at every import site as
`R_GAS * T_REF / F_CONST`, not hardcoded — appropriate.

---

## 2. `electromigration_prefactor` (em)

**Definition in `Nondim/transform.py` line 396:**
```python
electromigration_prefactor = (FARADAY_CONSTANT / (GAS_CONSTANT * temperature_k)) * potential_scale
```

**Production path** (`_make_nondim_cfg` in `scripts/_bv_common.py`):
- `potential_scale_v = V_T = R·T/F`
- Therefore: `em = (F/(R·T)) * (R·T/F) = 1.0` exactly.

This is verified: the formula is algebraically exact (not a numerical
approximation), so `em = 1.0` is guaranteed as long as the same
temperature value is used for both `V_T` and the prefactor. Both use
`DEFAULT_TEMPERATURE_K = 298.15 K`. **PASS.**

**Consumption in `forms_logc_muh.py`:** `em` is read from
`scaling["electromigration_prefactor"]` at lines 313, 922, and 1116.
The `forms_logc_muh` build does correctly pull `em` from the scaling
dict rather than hardcoding 1.0.  **No inconsistency detected.**

---

## 3. Reference scales

**Defined in `scripts/_bv_common.py` lines 131–134:**

| Scale | Value | Notes |
|---|---|---|
| `L_REF` | 1.0e-4 m (100 µm) | Consistent with `l_eff_m = 100e-6` default |
| `D_REF` | `D_O2 = 1.9e-9` m²/s | Single D_ref; OK for PNP scaling |
| `C_SCALE` | `C_O2 = 1.2` mol/m³ | Ruggiero 2022 §2 deck value; migrated 2026-05-07 |
| `K_SCALE` | `D_REF / L_REF = 1.9e-5` m/s | Velocity / k0 scale |
| `T_SCALE` | `L_REF²/D_REF = 5.26` s | Explicitly set in `_make_nondim_cfg` |

The `_make_nondim_cfg` function (lines 544–559) passes all five of these
into the `Nondim.transform.build_model_scaling` path, which then computes
the downstream PDE coefficients.

---

## 4. Debye length and Poisson coefficient

**Derived in `Nondim/transform.py` line 401:**
```
poisson_coefficient = eps * potential_scale / (F * c_scale * L^2)
                    = eps * V_T / (F * C_SCALE * L_REF^2)
                    = (λ_D / L_REF)^2
```

Numerically at production parameters:
- `λ_D = sqrt(ε·V_T / (F·C_SCALE))` = `sqrt(6.944e-10 * 0.02569 / (96485 * 1.2))` ≈ 1.242e-8 m (12.4 nm)
- `poisson_coefficient` = (1.242e-8 / 1e-4)² ≈ 1.54e-8

This is the correct nondim Poisson coefficient `(λ_D/L)²`. The Poisson
residual in the weak form uses `eps_coeff = poisson_coefficient` and
`charge_rhs_prefactor = 1.0` (nondim mode), consistent with:
```
(λ_D/L)² · Δφ̂ = −Σᵢ zᵢ ĉᵢ
```
**PASS.**

---

## 5. Diffusivity values and D̂_i

| Species | D_i (m²/s) | D̂_i = D_i/D_REF | Literature check |
|---|---|---|---|
| O₂ | 1.9e-9 | **1.000** | ~1.9e-9 at 25 °C — OK |
| H₂O₂ | 1.6e-9 | **0.842** | Literature range 1.4–1.6e-9; upper-range OK |
| H⁺ | 9.311e-9 | **4.900** | 9.311e-9 (CRC) — EXACT |
| ClO₄⁻ | 1.792e-9 | **0.943** | ~1.79e-9 literature — OK |
| K⁺ | 1.96e-9 | **1.032** | ~1.96e-9 (CRC) — OK |
| Cs⁺ (counterion, Boltzmann only) | — | — | no NP diffusivity needed |
| OH⁻ | 5.273e-9 | **2.775** | ~5.273e-9 (CRC) — EXACT |

All values are physically plausible. `D_H2O2 = 1.6e-9` is at the
high end of the 1.4–1.6e-9 literature range but not outside it.

---

## 6. Charge values z_i

| Species | z_i in code | Expected | Status |
|---|---|---|---|
| O₂ | 0 | 0 | OK |
| H₂O₂ | 0 | 0 | OK |
| H⁺ | +1 | +1 | OK |
| K⁺ (counterion) | +1 | +1 | OK |
| Cs⁺ (counterion) | +1 | +1 | OK |
| SO₄²⁻ (counterion) | −2 | −2 | OK |
| ClO₄⁻ (counterion) | −1 | −1 | OK |
| OH⁻ | −1 | −1 | OK |

All charge values correct.

---

## 7. Bulk concentrations and ĉᵢ = cᵢ / C_SCALE

| Species | cᵢ (mol/m³) | ĉᵢ = cᵢ/C_SCALE | Notes |
|---|---|---|---|
| O₂ | 1.2 | **1.000** | = C_SCALE by construction |
| H₂O₂ | 0 (seed 1e-4 nondim) | 0 + seed | OK |
| H⁺ (pH 4) | 0.1 | **0.0833** | pH 4 ↔ 1e-4 mol/L = 0.1 mol/m³ ✓ |
| K⁺ | 199.9 | **166.58** | electroneutrality: K⁺ + H⁺ = 2·SO₄²⁻ |
| SO₄²⁻ | 100.0 | **83.33** | 0.1 M ✓ |
| Cs⁺ | 199.9 | **166.58** | mirrors K⁺ |
| ClO₄⁻ | 0.1 | **0.0833** | electroneutral with H⁺ in ClO₄ stack |
| OH⁻ (pH 4 equilibrium) | KW/C_HP = 1e-7 mol/m³ | ~8.33e-8 | derived from KW_HAT |

All consistent. The K⁺/Cs⁺ + SO₄²⁻ ionic strength: ½(199.9 + 0.1 + 4·100) = 300 mol/m³ = 0.3 M ✓

**KW_HAT**: `KW_PHYS / C_SCALE² = 1e-8 / 1.44 = 6.944e-9`. Computed in
`_bv_common.py` line 224 correctly.

---

## 8. `domain_height_hat` and `l_eff_m`

In `make_bv_solver_params` (line 1289):
```python
domain_height_hat = float(l_eff_m) / float(L_REF)
```
Single definition; consumed by `forms_logc_muh.py` at lines 975 and 1311
via `conv_cfg["domain_height_hat"]`. No duplication detected. **PASS.**

---

## 9. Stern C_S non-dimensionalization

**Formula** (both `nondim.py` lines 86–89 and `forms_logc_muh.py` lines 248–254
— both code paths implement independently and must agree):

```
C_S_model = C_S_phys * potential_scale / (F * c_scale * L)
           = C_S_phys * V_T / (F * C_SCALE * L_REF)
```

Derivation check: the Stern weak-form BC is `ε_coeff·∇φ̂·n = C_S_model·(φ̂_m - φ̂)`.
The IBP ε_coeff is `(λ_D/L)²`, and the physical BC is `ε·∇φ·n = C_S·(φ_m - φ)`.
Converting: C_S_model = `C_S · V_T / (F · c_scale · L)` = 0.20 × 2.219e-3 = 4.44e-4 nondim.

Both code paths use the same formula — **consistent**. The derivation is
also consistent with the docstring in `nondim.py` lines 62–89.

---

## 10. Bikerman `a_nondim` — physical vs legacy

**Formula:** `a_nondim = a_phys · C_SCALE` where `a_phys = (4/3)·π·r³·N_A` (m³/mol)

**Computed vs code values:**

| Species | r (Å) | a_hat computed | a_hat in code | Discrepancy |
|---|---|---|---|---|
| O₂ | 1.70 | 1.487e-5 | 1.487e-5 (dynamically computed) | EXACT |
| H₂O₂ | 2.00 | 2.422e-5 | 2.422e-5 (dynamically computed) | EXACT |
| H⁺ | 2.80 | 6.645e-5 | 6.645e-5 (dynamically computed) | EXACT |
| Cs⁺ | 2.20 | 3.223e-5 | **3.23e-5** (hardcoded) | ~0.2% rounding |
| SO₄²⁻ | 2.40 | 4.185e-5 | **4.20e-5** (hardcoded) | **~0.35% MISMATCH** |
| K⁺ | 2.30 | 3.683e-5 | dynamically computed | EXACT |
| OH⁻ | 1.76 | 1.650e-5 | dynamically computed | EXACT |
| ClO₄⁻ | — | A_DEFAULT = 0.01 | 0.01 (legacy, ~14.9 Å) | **KNOWN ISSUE** |

**SO₄²⁻ hardcoded `a_hat = 4.20e-5`:** the dynamically computed value
from r = 2.40 Å is 4.185e-5 (0.35% low). The discrepancy is small but
the comment says r = 2.4 Å and the hardcoded value doesn't match that
radius exactly. The hardcoded value would correspond to r ≈ 2.41 Å.
This is a **minor inconsistency**: either the comment radius is wrong or
the hardcoded value is slightly off. It does not affect physical
correctness materially (< 0.4%), but it should be noted.

The `A_DEFAULT = 0.01` (≈ r = 14.9 Å) for ClO₄⁻ steric mode is an
acknowledged known issue (Hard Rule #7 in `CLAUDE.md`). The production
Cs⁺/K⁺/SO₄²⁻ counterion stacks all use physical radii.

---

## 11. Current density scale

`I_SCALE = n_e · F · D_REF · C_SCALE / L_REF · 0.1`
= 2 × 96485 × 1.9e-9 × 1.2 / 1e-4 × 0.1
= **0.440 mA/cm²**

The factor 0.1 converts A/m² to mA/cm². Formula in `compute_i_scale`
at `_bv_common.py` line 253 is correct.

---

## 12. Potential inputs flag consistency

`_make_nondim_cfg` sets `"potential_inputs_are_dimensionless": True`,
meaning `phi_applied` (already in V_T units) is NOT divided by `potential_scale`
again in `build_model_scaling`. This is consistent: production scripts pass
`phi_applied = V_RHE / V_T` (already nondim). **PASS.**

---

## Issues found

| Severity | Issue | Location |
|---|---|---|
| MINOR | `A_SO4_HAT = 4.20e-5` (hardcoded) does not match r = 2.40 Å formula (gives 4.185e-5, ~0.35% off). Either the comment radius is wrong or the constant was rounded to a slightly wrong value. | `_bv_common.py` line 745 |
| KNOWN | `A_DEFAULT = 0.01` for ClO₄⁻ steric corresponds to r ≈ 14.9 Å (non-physical). Acknowledged in Hard Rule #7; ClO₄⁻ is legacy stack only. | `_bv_common.py` line 716 |
| NOTE | `D_H2O2 = 1.6e-9` m²/s is at the high end of literature range (some sources give 1.4e-9). Not a bug but worth flagging for sensitivity analysis. | `_bv_common.py` line 74 |
| NOTE | Stern C_S nondim is implemented in two separate code paths: `nondim.py` (via `_add_bv_scaling_to_transform`) and `forms_logc_muh.py` (inline at lines 248–254). Both use the same formula, but duplication creates a maintenance risk if one is updated without the other. | `nondim.py:86-89`, `forms_logc_muh.py:248-254` |

No blocking errors found. All primary scales (L, C, D, V_T, em) are
internally consistent and match standard values.

---

## Bottom Line

The non-dimensionalization is **correct and internally consistent** for
the production stack. V_T = 0.02569 V (correct), F/(RT) × V_T = 1.0
(em verified algebraically exact), Poisson coefficient = (λ_D/L)² as
derived. All species charges are correct. One minor inconsistency: the
hardcoded `A_SO4_HAT = 4.20e-5` is ~0.35% higher than the formula value
for r = 2.40 Å; this is inconsequential for physics but the comment and
value are inconsistent.

**5-bullet summary:**
- V_T = R·T/F = 0.025693 V, F = 96485.33 C/mol, R = 8.31446 J/(mol·K): all exact matches to CODATA.
- `em = (F/RT)·potential_scale = 1.0` exactly when `potential_scale = V_T`; production path guarantees this algebraically; `forms_logc_muh` reads em from scaling dict consistently.
- Poisson coefficient `(λ_D/L)² ≈ 1.54e-8` correctly computed; `charge_rhs_prefactor = 1.0` in nondim mode is correct.
- Stern C_S nondim `= C_S · V_T/(F·c_scale·L)` is implemented correctly and consistently in both `nondim.py` and `forms_logc_muh.py`.
- **Minor issue:** `A_SO4_HAT = 4.20e-5` (hardcoded) vs computed 4.185e-5 for r = 2.40 Å (0.35% discrepancy); all dynamic species now use correct physical radii (Step-10 update 2026-05-21).
