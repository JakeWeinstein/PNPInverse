# Nondimensionalization Correctness Verification
## Scope: Nondim/transform.py, Nondim/constants.py (plot_iv_curve_unified.py codepath)
**Verifier:** claude-sonnet-4-6 | **Date:** 2026-05-02

---

## SUMMARY

**No critical bugs found.** The nondimensionalization machinery is mathematically correct for the `plot_iv_curve_unified.py` codepath. The dimensionless PDE prefactors (`electromigration_prefactor`, `poisson_coefficient`, `charge_rhs_prefactor`) are derived correctly. All keys consumed by `forms_logc.py` and `nondim.py` are produced. Two minor issues and one prompt documentation error are noted below.

---

## A. Nondim/constants.py

**FARADAY_CONSTANT = 96485.3329 C/mol**
- Correct value to use. This is the CODATA 2014 recommended value.
- The SI 2019 / CODATA 2018 exact value is 96485.33212 C/mol (derived from exact $e$ and $N_A$).
- Relative difference: 8.05e-9 (8 ppb). Negligible for all electrochemical computations.
- SEVERITY: none (no fix required).

**GAS_CONSTANT = 8.314462618 J/(mol·K)**
- Matches SI 2019 exact ($k_B \cdot N_A = 1.380649\times10^{-23} \times 6.02214076\times10^{23}$) to at least 10 significant figures. Correct.

**DEFAULT_TEMPERATURE_K = 298.15 K**
- Correct (25 °C exactly).

**V_T computed at these values = 0.025692579 V**
- Matches the exact SI 2019 value to 8 ppb. Correct.

**DEFAULT_RELATIVE_PERMITTIVITY_WATER = 78.5**
- Standard rounded literature value at 25 °C. Correct.

**VACUUM_PERMITTIVITY_F_PER_M = 8.8541878128e-12 F/m**
- Matches CODATA 2018 NIST value. Correct.

**MOLAR_TO_MOL_PER_M3 = 1000.0**
- Correct (1 M = 1 mol/L = 1000 mol/m³).

---

## B. transform._bool, _as_list, _pos

**_bool (lines 109-124)**
- Handles: `True`/`False` (direct), strings `true/false/yes/no/on/off/1/0/""`, `None` (raises), integers and arbitrary objects via `bool()` fallthrough.
- One minor concern: `_bool(2)` returns `True` via the `bool()` fallthrough. No explicit guard that integer inputs must be 0 or 1. In practice, all callers pass Python booleans from `nondim_cfg.get(key, default_bool)`, so this path is never reached on the active codepath.
- SEVERITY: minor / question (no fix needed on this path).

**_as_list (lines 90-99)**
- Correctly broadcasts a scalar to length-n list, validates list length, raises with name in error. No issues.

**_pos (lines 102-106)**
- Correctly validates > 0 and includes the parameter name in the error message. No issues.

---

## C. transform._get_nondim_cfg (lines 127-131)

- Returns `{}` if `params` is not a dict or if `params["nondim"]` is absent or not a dict.
- When `{}` is returned, all `nondim_cfg.get(key, default)` calls in `build_model_scaling` use their defaults: `enabled=False`, scales auto-computed, all `*_inputs_are_dimensionless=False` (except `kappa_inputs_are_dimensionless=True`).
- On the `plot_iv_curve_unified.py` path, `params["nondim"]` is set to a full dict by `_make_nondim_cfg()`, so the fallback is not exercised. No issues.

---

## D. transform._get_robin_cfg (lines 134-165)

**Default markers**
- The actual code defaults are: `electrode_marker=1`, `concentration_marker=3`, `ground_marker=3`.
- The prompt description states "(1,2,2)" — this is a documentation error in the prompt, not a code bug.

**Path taken by forms_logc.py**
- `build_model_scaling` is called with `robin=dummy_robin` (explicit argument).
- `transform.py:225-226`: `if robin is None: robin = _get_robin_cfg(params, n_species)` — the `if` branch is skipped because `robin` is not `None`.
- `_get_robin_cfg` is never called on this path. The dummy_robin markers come from `bv_cfg` (electrode=3, concentration=4, ground=4 per `_make_bv_bc_cfg` defaults in `_bv_common.py`). No conflict.

**dummy_robin c_inf_vals**
- `c_inf_vals = bv_cfg["c_ref_vals"]` are used in `c_all = np.asarray(c0_raw + c_inf_raw)` for auto-computing `concentration_scale`. Since `concentration_scale_mol_m3` is explicitly provided (= `C_SCALE = 0.5`), the auto-computation is skipped. The dummy values have no effect on any output.

---

## E. transform.build_model_scaling

### E1. enabled=True, dimensionless input pass-through

With all `*_inputs_are_dimensionless=True` (as set by `_make_nondim_cfg()`):

| Quantity | Code | Verdict |
|---|---|---|
| `dt_model` | `dt_raw` (passthrough, line 377) | Correct |
| `t_end_model` | `t_end_raw` (passthrough, line 378) | Correct |
| `D_model_vals` | `D_raw` (passthrough, line 368) | Correct |
| `c0_model_vals` | `c0_raw` (passthrough, line 372) | Correct |
| `c_inf_model_vals` | `c_inf_raw` (passthrough, line 373) | Correct |
| `kappa_model_vals` | `kappa_raw` (passthrough, line 387) | Correct |
| `phi_applied_model` | `phi_applied_raw` (passthrough, line 384) | Correct |
| `phi0_model` | `phi0_raw` (passthrough, line 385) | Correct |

All eight pass-throughs are correct for the `*_inputs_are_dimensionless=True` flags.

### E2. Dimensionless PDE prefactors

**electromigration_prefactor (line 396)**

```python
electromigration_prefactor = (FARADAY_CONSTANT / (GAS_CONSTANT * temperature_k)) * potential_scale
```

- With `potential_scale = V_T = RT/F`, this equals `(F/RT) * (RT/F) = 1.0` exactly.
- Numerical verification: computed value = 1.0000000000 (no floating-point error at double precision for this product).
- Derivation: In the dimensionless NP equation `∂ĉ/∂t̂ + ∇̂·[-D̂(∇̂ĉ + ẑ·ĉ·∇̂φ̂)] = 0`, the electromigration coefficient is 1 because `φ = V_T · φ̂` implies `∇φ = V_T ∇̂φ̂`, and the drift velocity `(FD/RT)·∇φ = (F/RT)·V_T·D·∇̂φ̂ = D·∇̂φ̂` when `V_T = RT/F`. **Correct.**

**poisson_coefficient (lines 401-403)**

```python
poisson_coefficient = (permittivity_f_m * potential_scale) / (FARADAY_CONSTANT * concentration_scale * (length_scale * length_scale))
```

- This equals `ε·V_T / (F·c_ref·L²) = (λ_D/L)²` where `λ_D = sqrt(ε·V_T/(F·c_ref))`.
- Numerical value with script parameters: `(78.5 × 8.854×10⁻¹² × 0.025693) / (96485 × 0.5 × (10⁻⁴)²) ≈ 3.70×10⁻⁸`.
- Equivalently `(λ_D/L)² ≈ (1.924×10⁻⁸ / 10⁻⁴)² ≈ 3.70×10⁻⁸`. **Values match exactly** (confirmed numerically, 0.00e+00 relative difference).
- Derivation: The dimensionless Poisson equation is `−(λ_D/L)²·∇̂²φ̂ = Σᵢzᵢĉᵢ`. In weak form `eps_coeff·(∇̂φ̂,∇̂w) − rhs_coeff·Σᵢzᵢĉᵢ·w = 0` gives `eps_coeff = (λ_D/L)²`, `rhs_coeff = 1`. **Correct.**
- This is a singularly perturbed problem with `ε ≈ 3.70×10⁻⁸ ≪ 1`, confirming why graded mesh + log-c primary variable is essential.

**charge_rhs_prefactor (line 429)**

```python
"charge_rhs_prefactor": 1.0,
```

- Hardcoded to 1.0 in nondimensional mode. **Correct**: in the dimensionless equation `−(λ_D/L)²·∇̂²φ̂ = Σᵢzᵢĉᵢ`, the RHS coefficient is exactly 1.

**Internal consistency of module docstring (lines 25-30)**

The docstring states `eps_coeff = (λ_D/L)²` and `rhs_coeff = 1` in nondimensional mode. The code produces exactly these values. **Consistent.**

### E3. flux_scale and current_density_scale

```python
flux_scale = diffusivity_scale * concentration_scale / length_scale       # mol/(m²·s)
current_density_scale = FARADAY_CONSTANT * flux_scale                     # A/m²
```

- `flux_scale = D_ref × c_scale / L_ref = 1.9e-9 × 0.5 / 1e-4 = 9.5e-6 mol/(m²·s)`. Correct.
- `current_density_scale = F × flux_scale = 96485 × 9.5e-6 = 0.9166 A/m²`. This is the **per-electron-per-unit-area** scale.
- `_bv_common.py` defines `I_SCALE = n_e × F × D_ref × c_scale / L_ref × 0.1` (where 0.1 converts A/m² → mA/cm²).
- Therefore: `I_SCALE = n_e × current_density_scale × 0.1 = 2 × 0.9166 × 0.1 = 0.1833 mA/cm²`. **Exact match** (confirmed numerically). The comment in `build_model_scaling` (line 412: "callers must multiply by n_electrons") correctly documents this convention.

### E4. Dirichlet marker assumptions

- `_get_robin_cfg` is bypassed on the `forms_logc.py` path (see §D).
- The dummy_robin and the BV BC markers (electrode=3, concentration=4, ground=4) flow entirely from `bv_cfg`, never from `_get_robin_cfg` defaults.
- No code anywhere on this path assumes markers (1,2,2) or (1,3,3). **No cross-contamination.**

### E5. Validation

- `D_vals ≤ 0`: raises at line 235 (before scaling) and line 370 (after scaling). **Covered.**
- `temperature_k ≤ 0`: raises via `_pos` at line 229. **Covered.**
- All scales (diffusivity, concentration, length, potential, permittivity, time, kappa) validated via `_pos`. **Covered.**
- `dt` and `t_end`: validated positive via `_pos` at lines 375-376, then checked again as model-space values at line 379. The second check is redundant when `time_inputs_are_dimensionless=True` (since `dt_model = dt_raw > 0` already proven), but harmless.
- Concentrations `c0` and `c_inf`: no validation against negativity. Not a problem: `c_inf = 0` is legitimate (product species), log-c solver applies a floor `_C_FLOOR = 1e-20` before taking `ln`. No fix needed.
- Error messages include parameter names. **Good.**

---

## F. transform.verify_model_params (lines 448-481)

- Logic is correct: checks `D_model_vals` in `[1e-8, 1e4]`, `c0_model_vals >= 0` and `≤ 1e3`, `debye_to_length_ratio` in `[1e-5, 1.0]`.
- For the script's parameters: D̂ values are O2=1.0, H2O2=0.84, H⁺=4.9, ClO₄⁻=0.94 — all within bounds.
- `debye_to_length_ratio ≈ 1.924e-4`, within `[1e-5, 1.0]` — no warning triggered.
- Not on the codepath until called at line 444 (inside `build_model_scaling`). Early-exit at line 455 (`if not scaling.get("enabled", False): return`) is correct.

**SEVERITY: minor** — `verify_model_params` uses `print()` for warnings (lines 461, 467, 471, 479) while the rest of the module uses `warnings.warn()` (lines 342-365). Inconsistent. Not a correctness bug.

---

## G. Key cross-checks: forms_logc.py ↔ build_model_scaling output

All keys read by `forms_logc.py` from the final scaling dict are accounted for:

| Key | Source | Lines in forms_logc.py |
|---|---|---|
| `electromigration_prefactor` | `build_model_scaling` | 200 |
| `dt_model` | `build_model_scaling` | 201 |
| `phi_applied_model` | `build_model_scaling` | 205 |
| `phi0_model` | `build_model_scaling` | 207 |
| `bv_E_eq_model` | `_add_bv_reactions_scaling_to_transform` | 210 |
| `bv_exponent_scale` | `_add_bv_reactions_scaling_to_transform` | 211 |
| `bv_stern_capacitance_model` | `_add_bv_reactions_scaling_to_transform` | 214 |
| `D_model_vals` | `build_model_scaling` | 169 |
| `poisson_coefficient` | `build_model_scaling` | 415 |
| `charge_rhs_prefactor` | `build_model_scaling` | 416 |
| `c0_model_vals` | `build_model_scaling` | 431 |
| `bv_reactions` | `_add_bv_reactions_scaling_to_transform` | 306 |
| `potential_scale_v` | `build_model_scaling` | 141 (Stern block) |
| `length_scale_m` | `build_model_scaling` | 140 (Stern block) |
| `concentration_scale_mol_m3` | `build_model_scaling` | 142 (Stern block) |

All keys are unconditionally produced in the nondim-enabled path. **No missing keys.**

Keys read by `nondim._add_bv_reactions_scaling_to_transform` from `base_scaling`:
- `kappa_scale_m_s`, `concentration_scale_mol_m3`, `potential_scale_v`, `temperature_k` — all present in `build_model_scaling` output. **No missing keys.**

---

## Prompt Documentation Errors (not code bugs)

1. **Debye length example in prompt**: The prompt states "λ_D ≈ 195 nm". The correct value with ε_r=78.5 (code default) is **λ_D ≈ 19.2 nm** (19.4 nm with ε_r=80). The factor-of-10 discrepancy in the prompt is a unit/arithmetic error in the problem statement. The code's computed value of `debye_length_m ≈ 1.924e-8 m` is correct. The qualitative conclusion (singularly perturbed, small parameter ≈ 3.7e-8) is correct.

2. **_get_robin_cfg default markers**: The prompt says "defaults in _get_robin_cfg are (1,2,2) for the legacy 1D path". The actual code defaults are `electrode_marker=1`, `concentration_marker=3`, `ground_marker=3`, i.e., (1,3,3). This doesn't affect correctness since `_get_robin_cfg` is bypassed on the active path.

---

## Issue Register

| # | Severity | Location | Description | Smallest Fix |
|---|---|---|---|---|
| 1 | Minor | `constants.py:6` | `FARADAY_CONSTANT` is CODATA 2014 (96485.3329) vs CODATA 2018 (96485.33212). Error = 8 ppb, negligible. | Update to `96485.33212` for accuracy completeness. |
| 2 | Minor | `transform.py:461,467,471,479` | `verify_model_params` uses `print()` for diagnostics while `build_model_scaling` uses `warnings.warn()`. Inconsistent; `print()` bypasses warning filters. | Replace with `warnings.warn(..., stacklevel=2)` in `verify_model_params`. |
| 3 | Question | `transform.py:124` | `_bool(2)` returns `True` via `bool()` fallthrough, not an error. On this codepath all flag values are Python booleans or the specific strings/ints in the nondim_cfg dict, so the fallthrough is never reached. | No fix needed on this path. |

**No critical or major issues found.**
