# SONNET_11: Canonical Factory Verification
**Scope**: `calibration/v10b.py`, `calibration/__init__.py`, `scripts/_bv_common.py`
**Agent**: 11/13 | **Date**: 2026-05-22 | **Status**: PASS with advisory

---

## 1. v10b Constants

| Constant | Expected | Actual (v10b.py) | Status |
|---|---|---|---|
| `C_S_F_M2_V10B` | 0.20 | 0.20 | PASS |
| `GAMMA_MAX_HAT_V10B` | 0.047 | 0.047 | PASS |
| `K_DES_NONDIM_V10B` | 1.0 | 1.0 | PASS |

All three are exported in `__all__` and referenced in `V10B_KINETICS` dict. `V10B_KINETICS` re-references both `K_DES_NONDIM_V10B` and `GAMMA_MAX_HAT_V10B` symbolically (no magic numbers). `C_S_F_M2_V10B` is not in `V10B_KINETICS` (correct — it is wired at the `make_bv_solver_params` call site via `stern_capacitance_f_m2`).

Metadata blocks (`_gamma_max_metadata`, `_k_des_metadata`, `_c_s_metadata`) all read the locked constants symbolically. Engineering-choice flag is `True` for `k_des`, `False` for `gamma_max` and `C_S` — correct per calibration outcome.

## 2. Deprecation Alias `SMOKE = V10A_SMOKE`

`_bv_common.py` lines 976-983:
```python
GAMMA_MAX_HAT_SMOKE = GAMMA_MAX_HAT_V10A_SMOKE  # = 0.047 frozen historical
```
Correct. No `SMOKE = V10B` or `GAMMA_MAX_HAT_SMOKE = GAMMA_MAX_HAT_V10B` assignment anywhere in the codebase (confirmed by repo-wide grep). The forbidden alias does not exist.

## 3. `make_bv_solver_params` Signature

**Advisory (not a bug):** The function defaults diverge from the production target on four parameters:

| Parameter | Default | Production Target | Notes |
|---|---|---|---|
| `formulation` | `"logc"` | `"logc_muh"` | Must be passed explicitly |
| `log_rate` | `False` | `True` | Must be passed explicitly |
| `initializer` | `"linear_phi"` | `"debye_boltzmann"` | Must be passed explicitly |
| `stern_capacitance_f_m2` | `None` | `0.20` | Must be passed explicitly |
| `multi_ion_enabled` | `False` | `True` (for K⁺/SO₄²⁻) | Must be passed explicitly |

All other parameters in scope are present: `species`, `bv_reactions`, `boltzmann_counterions`, `l_eff_m`, `enable_water_ionization`. The defaults are intentionally conservative (legacy-safe) rather than production-safe. This is not a regression — no caller relies on default-formulation being muh — but it is a latent footgun: a bare call to `make_bv_solver_params(...)` will silently use `formulation="logc"` instead of `"logc_muh"`, and `log_rate=False`. **Recommendation**: add a docstring WARNING or assert that callers explicitly pass `formulation` when the intent is production.

`enable_water_ionization` default is `False` — correct per CLAUDE.md requirement.

## 4. `THREE_SPECIES_LOGC_BOLTZMANN` Species Definitions

| Idx | Species | z | D_hat | a_hat | c0_hat |
|---|---|---|---|---|---|
| 0 | O₂ | 0 | D_O2_HAT (=1.0) | A_O2_HAT (≈1.487e-5, r=1.70Å) | C_O2_HAT (=1.0) |
| 1 | H₂O₂ | 0 | D_H2O2_HAT (D_H2O2/D_O2) | A_H2O2_HAT (≈2.422e-5, r=2.00Å) | H2O2_SEED_NONDIM (=1e-4) |
| 2 | H⁺ | +1 | D_HP_HAT (D_HP/D_O2) | A_HP_HAT (≈6.645e-5, r=2.80Å) | C_HP_HAT (=0.1/1.2≈0.0833) |

Physical `a_nondim` are now the production values following the step-10 follow-up (2026-05-21). `A_DEFAULT=0.01` (r≈14.9Å) is no longer used in `THREE_SPECIES_LOGC_BOLTZMANN`. CLAUDE.md Hard Rule #7 (H⁺ Bikerman cap discrepancy) is now resolved in `_bv_common.py` for `THREE_SPECIES_LOGC_BOLTZMANN`.

`FOUR_SPECIES_LOGC_DYNAMIC_K2SO4` likewise uses physical radii for O₂/H₂O₂/H⁺/K⁺ (step-10 follow-up comment confirmed). `FOUR_SPECIES_LOGC_DYNAMIC` (ClO₄⁻ equivalence test) retains `A_DEFAULT` for ClO₄⁻ — correct per the comment (not production).

`roles` field: `["neutral", "neutral", "proton"]` — correctly set per Gate 1 requirement.

## 5. `PARALLEL_2E_4E_REACTIONS`

| Reaction | E°_eq (V vs RHE) | n_e | alpha | stoichiometry | reversible |
|---|---|---|---|---|---|
| R_2e | 0.695 (`E_EQ_R2E_V`) | 2 | ALPHA_R2E (=0.627) | [-1, +1, -2] | True |
| R_4e | 1.23 (`E_EQ_R4E_V`) | 4 | ALPHA_R4E (=0.5) | [-1, 0, -4] | False |

E° values match CLAUDE.md Hard Rule #4 and Ruggiero 2022. H⁺ stoichiometric concentration factors present on both reactions with correct `power` (2 for R_2e, 4 for R_4e).

`k0` values: R_2e uses `K0_HAT_R2E = K0_PHYS_R2E / K_SCALE`, R_4e uses `K0_HAT_R4E = K0_PHYS_R4E / K_SCALE`. Note: per comments in the file, `K0_PHYS_R4E = K0_PHYS_R1` (same placeholder as 2e) — intentional prior placeholder pending M4 calibration.

## 6. Counterion Defaults

**K⁺/SO₄²⁻ deck baseline (CLAUDE.md: `[SO₄²⁻]=0.1 M, [K⁺]=0.2 M`)**:

| Entry | z | c_bulk_nondim | a_nondim | Status |
|---|---|---|---|---|
| `DEFAULT_KPLUS_BOLTZMANN_COUNTERION_STERIC` | +1 | C_KPLUS_HAT = 199.9/1.2 ≈ 166.58 | A_KPLUS_HAT (computed, r=2.3Å) | PASS |
| `DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC` | -2 | C_SO4_HAT = 100/1.2 ≈ 83.33 | A_SO4_HAT = 4.20e-5 (r=2.4Å, 0.4% rounding) | PASS |
| `DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC` | +1 | C_CSPLUS_HAT = 199.9/1.2 ≈ 166.58 | A_CSPLUS_HAT = 3.23e-5 (r=2.2Å, 0.2% rounding) | PASS |

K⁺ bulk = 199.9 mol/m³ (not 200): this is intentional. The 0.1 mol/m³ difference accounts for H⁺ = 0.1 mol/m³ maintaining electroneutrality (H⁺ + K⁺ = 2·SO₄²⁻ → 0.1 + 199.9 = 200 = 2×100 ✓). Ionic strength: I = ½(199.9 + 400 + 0.1) = 300 mol/m³ = 0.3 M ✓.

`A_KPLUS_HAT` is computed dynamically from the formula (not hardcoded) — no rounding risk. `A_CSPLUS_HAT` and `A_SO4_HAT` are hardcoded with 0.2–0.4% rounding error (acceptable).

`DEFAULT_OH_BOLTZMANN_COUNTERION_STERIC` does **not exist** in `_bv_common.py`. OH⁻ is handled via `a_oh_hat`/`d_oh_hat`/`kw_eff_hat` parameters passed through `make_bv_solver_params` into the `bv_convergence` sub-dict. This is the intended design (OH⁻ is a derived equilibrium species, not a static counterion entry).

## 7. C_O2 Bulk Concentration

`C_O2 = 1.2` mol/m³ (line 86). Comment confirms M3a.2.1 migration from legacy 0.5. Legacy value retained as `C_O2_PHYS_LEGACY = 0.5` with explicit "do not use for new runs" warning. PASS.

## 8. Plumbing Gotchas

**`add_boltzmann` double-count**: All `add_boltzmann` call sites found are in inverse/legacy scripts (`v18_*`, `v19_*`, `v23_*`, `plot_iv_curves_3sp_true.py`). These are non-operational per CLAUDE.md ("Inverse scripts are non-operational"). No production Phase 6β scripts use `add_boltzmann` alongside `boltzmann_counterions`. PASS.

**`bv_reactions` with legacy k0_hat_r1/r2**: `_make_bv_bc_cfg` (lines 619-635) correctly takes the `bv_reactions` list path when provided and skips legacy construction. The `"k0"` and `"alpha"` legacy keys are still written to the cfg dict (lines 677-678) but those are the `species.k0_legacy`/`alpha_legacy` fallback fields (used for markers, not reactions). This does NOT cause the described gotcha — the warning in CLAUDE.md refers to passing a `k0_hat_r1`-keyed bundle as a reaction, not passing legacy scalar params alongside `bv_reactions`. No production caller does both. PASS.

## 9. `enable_water_ionization` Default

`enable_water_ionization: bool = False` in both `make_bv_solver_params` (line 1131) and `_make_bv_convergence_cfg` (line 459). PASS.

---

## Summary

**PASS with advisory.** All locked v10b constants correct. Deprecation alias clean. `enable_water_ionization` default is `False`. `C_O2 = 1.2`. K⁺/SO₄²⁻ counterion c_bulk values are electroneutrality-correct. Physical `a_nondim` now in `THREE_SPECIES_LOGC_BOLTZMANN` (step-10 follow-up). No forbidden `SMOKE = V10B` alias. No production double-count.

**Advisory**: `make_bv_solver_params` defaults (`formulation="logc"`, `log_rate=False`, `initializer="linear_phi"`, `stern_capacitance_f_m2=None`, `multi_ion_enabled=False`) are all legacy-safe but diverge from the production target on 5 parameters. Every production caller must override all 5 explicitly. No safety net (assert/warning) prevents a silent downgrade to the legacy stack. Low-risk in current codebase (all Phase 6β callers set them correctly), but a new caller could silently fall back to pre-muh formulation.
