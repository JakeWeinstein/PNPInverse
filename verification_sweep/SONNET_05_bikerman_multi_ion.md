# Agent 05/13 — Bikerman / Multi-Ion Verification Report
**Date:** 2026-05-22
**Scope:** `Forward/bv_solver/boltzmann.py`, `Forward/bv_solver/multi_ion.py`, `scripts/_bv_common.py` steric constants

---

## 1. Bikerman Formula Correctness

**PASS.** `build_steric_boltzmann_expressions` (`boltzmann.py` lines 241, 254) implements:

```
denom = theta_b + sum_k(a_k * c_b_k * exp(-z_k * phi_clamped_k))
c_steric_k = c_b_k * exp(-z_k * phi) * free_dyn / denom
```

where `theta_b = 1 - A_dyn_bulk - A_an_bulk` and `free_dyn = max(1 - A_dyn_local, 1e-10)`.

This algebraically matches the task-description spec `c_i = c_b * exp(-z*psi) * (1-A_dyn) / (theta_b + a*c_b*exp(-z*psi))`. The task spec omits the dynamic-species packing factor `(1-A_dyn)` in the numerator (it is the simplified single-counterion-only form); the code extends correctly to include dynamic-species excluded volume. Bulk recovery at φ=0: `denom = 1 - A_dyn_bulk`, `c_k = c_b_k` — verified analytically.

**Dimensionalization:** `a_nondim = a_phys [m³/mol] × C_SCALE [mol/m³]` — dimensionless, consistent throughout. PASS.

---

## 2. Hard Rule #7 — A_DEFAULT Issue

**KEY FINDING: PARTIALLY RESOLVED as of the current working tree (uncommitted changes in `_bv_common.py`).**

The git diff against HEAD confirms that `THREE_SPECIES_LOGC_BOLTZMANN`, `FOUR_SPECIES_LOGC_DYNAMIC_K2SO4`, and `FOUR_SPECIES_LOGC_DYNAMIC` (O₂/H₂O₂/H⁺ slots) have all been updated from `A_DEFAULT = 0.01` to physical per-species radii (`A_O2_HAT`, `A_H2O2_HAT`, `A_HP_HAT`) in the current working tree. These changes are staged/unstaged but not yet committed.

**Numerical verification:**
- `A_DEFAULT = 0.01`, `C_SCALE = 1.2 mol/m³` → `a_phys = 0.01/1.2 = 8.33e-3 m³/mol` → `r = 14.89 Å`. CLAUDE.md claim of ~14.9 Å confirmed.
- `c_max = 1/A_DEFAULT × C_SCALE = 100 × 1.2 = 120 mol/m³`. CLAUDE.md claim of ~120 mol/m³ confirmed.
- Physical H₃O⁺ at r=2.8 Å: `A_HP_HAT = 6.645e-5`, `c_max_phys = (1/6.645e-5) × 1.2 = 18,059 mol/m³`. Ratio `18059/120 = 150.5×`. CLAUDE.md claim of ~150× clamping confirmed.

**Remaining A_DEFAULT uses after the fix:**
1. `DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC["a_nondim"] = A_DEFAULT` (line 719) — ClO₄⁻ as Bikerman counterion for 3sp+ClO₄ stacks; NOT the production K⁺/SO₄²⁻ deck stack.
2. `FOUR_SPECIES_LOGC_DYNAMIC` ClO₄⁻ slot (index 3): `A_DEFAULT` retained; documented as non-production equivalence test stack.

Both remaining usages are non-production ClO₄⁻ paths. The production K⁺/SO₄²⁻ and Cs⁺/SO₄²⁻ stacks are NOT affected by A_DEFAULT.

**ALERT — ClO₄ legacy scripts:** Several scripts still call `DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC` (`plot_iv_curve_unified.py`, `peroxide_window_3sp_parallel_2e_4e.py`, `parallel_2e_4e_warmstart_probe.py`, `mangan_p15_comparison.py`). These carry `A_DEFAULT` for ClO₄⁻ as the Bikerman `a_nondim`. These are not Phase 6β production runs but the ClO₄⁻ radius is still unphysical (r=14.9 Å). This does not affect the H⁺ pile-up issue — but the ClO₄⁻ steric saturation cap will be set ~50×-100× too low when these scripts run.

---

## 3. Counterion Radii Verification

All physically-derived counterion `a_nondim` values verified against documented radii:

| Ion | Radius | `a_nondim` (computed) | `a_nondim` (code) | Status |
|-----|--------|----------------------|-------------------|--------|
| K⁺  | 2.3 Å (Linsey deck) | 3.683e-5 | `A_KPLUS_HAT` (math expr) | PASS |
| Cs⁺ | 2.2 Å (Linsey deck) | 3.223e-5 | 3.23e-5 (hardcoded) | PASS (<0.2% rounding) |
| SO₄²⁻ | 2.4 Å (Marcus) | 4.184e-5 | 4.20e-5 (hardcoded) | PASS (<0.4% rounding) |
| OH⁻ | 1.76 Å (Marcus) | 1.650e-5 | `A_OH_HAT` (math expr) | PASS |

K⁺, OH⁻ are computed from the radius formula at parse time; Cs⁺ and SO₄²⁻ are hardcoded with small rounding. All are physically consistent. The literature ranges quoted in the task (K⁺ crystal 1.3 Å, Stokes 3.3 Å; Cs⁺ crystal 1.7 Å, Stokes 3.3 Å; SO₄²⁻ crystal 2.3 Å, Stokes 3.8 Å) reflect *bare crystal* vs *Stokes* vs *effective hydrated radius* uncertainty; the code uses effective hydrated radii from the Linsey 2025 deck (K⁺ 2.3 Å, Cs⁺ 2.2 Å), which are in the physically reasonable range for hydrated radii. **PASS** with the understanding that the radius choice is a modeling assumption, consistently applied.

---

## 4. Multi-Ion Shared-θ Closure

**PASS.** `multi_ion.py` implements the shared-denominator closure:

```python
denom = theta_b + sum(a_k * c_b_k * exp(-z_k * phi) for k in bikerman_ions)
c_k = c_b_k * exp(-z_k * phi) * max(1-A_dyn_local, 1e-12) / max(denom, 1e-300)
```

- **Single-ion degeneration:** For one bikerman entry, `_ck_at_phi` degenerates exactly to the `boltzmann.py` UFL formula. PASS.
- **Consistency between `multi_ion.py` (scalar, Picard) and `boltzmann.py` (UFL, residual):** The `_phi_safe_exp` helper in `multi_ion.py` mirrors the `phi_clamped_k` clamp in `boltzmann.py` (line 224). Both clamp φ to `[-phi_clamp_val, +phi_clamp_val]` before evaluating `exp(-z*phi)`. PASS.
- **No coupled nonlinear solve needed:** The denominator is analytically shared across all bikerman ions — this is an explicit closed-form solution, not an iterative scheme. Correct.
- **`build_counterion_ctx` is the single producer of `theta_b`:** Both `_electroneutrality_residual` and `per_ion_outer_concs` read `theta_b` from the ctx rather than recomputing it. PASS — guards against sign/inclusion drift.
- **`effective_debye_length_local`:** Uses central FD on the electroneutrality residual, consistent with the shared-theta closure. Accounts for steric depression of screening at high |φ|. Correct.

**z=0 species (O₂, H₂O₂) handling:** In all production stacks, O₂ and H₂O₂ are dynamic NP species, NOT Boltzmann counterions. They only enter `A_dyn = sum(a_i * c_i(x))` in the numerator packing factor. If a z=0 ion were mistakenly added as a Bikerman counterion, `exp(-0*phi) = 1` would make its contribution to the denominator constant in φ — physically wrong (no Boltzmann response), but not a concern for current production configs. PASS.

---

## 5. Sign Convention

**PASS. Sign is CORRECT in current code.** Both `forms_logc.py` (line 402) and `forms_logc_muh.py` (line 453) implement:

```python
mu_steric = -fd.ln(packing)
```

This is `μ_steric = -ln(1 - Σ a_i c_i)`, matching Borukhov-Andelman-Orland (1997) eq (3) and Bazant et al. (2009) eq (20). The sign correction plan (`steric_sign_correction_plan.md`) documents that the old code used `+fd.ln(packing)` (wrong sign); the fix has been applied in both `logc` and `logc_muh` backends. The plan notes `robin_solver.py` and `dirichlet_solver.py` still have the old sign, but those are not in the production dispatch graph.

---

## 6. z=0 Species in Bikerman (Production Context)

**PASS.** O₂ and H₂O₂ (z=0) are dynamic NP species. They enter the UFL `A_dyn = sum(steric_a_funcs[j] * ci[j])` as packing occupants with physically-sized exclusion volumes (`A_O2_HAT`, `A_H2O2_HAT`). They do not appear in the Boltzmann-closure denominator (which sums over *counterion* entries only). This is correct: neutral species have no electrostatic Boltzmann response and their presence only affects the free volume available to charged counterions.

---

## 7. Exposed Output and Poisson Consistency

**PASS.** `StericBoltzmannBundle` exposes three UFL expressions per counterion:
- `c_steric_expr` — counterion concentration
- `packing_contribution = a_k * c_steric_expr` — counterion's excluded volume
- `charge_density = z_k * c_steric_expr` — charge source for Poisson

All three are built from the **same** `c_steric_k` UFL expression, guaranteeing that the charge in Poisson and the packing in `theta_inner` are computed from the same closure. The handoff doc identifies this as the critical correctness requirement (same `c_steric_expr` entering both Poisson and the dynamic-species `theta`). PASS.

The `packing_contribution` is further multiplied by `z_scale_shared` (lines 394–397 of `forms_logc.py`) so Strategy-B/C+D z-ramps zero out both contributions consistently. PASS.

---

## 8. Issues Found

### Issue A — A_DEFAULT still in DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC (LOW severity)
`DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC["a_nondim"] = A_DEFAULT = 0.01` (r=14.9 Å). This is used in several scripts (`plot_iv_curve_unified.py`, `peroxide_window_3sp_parallel_2e_4e.py`, etc.) but NOT in Phase 6β production (K⁺/SO₄²⁻ or Cs⁺/SO₄²⁻). The steric saturation cap for ClO₄⁻ in those scripts is 120 mol/m³ rather than the physical value. Impact on those old scripts only; not a production regression.

### Issue B — Cs⁺ and SO₄²⁻ a_nondim values hardcoded (NOT derived from radius formulas) (INFO)
`A_CSPLUS_HAT = 3.23e-5` and `A_SO4_HAT = 4.20e-5` are hardcoded floats, not computed from the `(4/3)πr³N_A` formula the way K⁺ and OH⁻ are. The values are consistent with the documented radii (<0.4% error), but the hardcoding style differs from the other ions and could lead to silent inconsistency if a radius is later updated. Recommend converting to formula-computed constants for consistency (as was done for K⁺ in the current working tree).

### Issue C — Production presets A_DEFAULT fix is UNCOMMITTED (MEDIUM severity)
The three production presets (`THREE_SPECIES_LOGC_BOLTZMANN`, `FOUR_SPECIES_LOGC_DYNAMIC_K2SO4`, `FOUR_SPECIES_LOGC_DYNAMIC`) have had A_DEFAULT replaced with physical radii in `_bv_common.py`, but this change is only in the working tree — it is NOT committed. The git status shows `_bv_common.py` as modified. Until committed, a stale checkout or `git stash` would silently revert to the unphysical radii. COMMIT URGENTLY.

### Issue D — theta_b validation uses A_dyn_bulk, but A_dyn_bulk uses model concentrations (INFO)
`theta_b` is validated at build time using `c0_dyn` (bulk model concentrations), but the runtime `free_dyn = max(1 - A_dyn_local, 1e-10)` uses the actual FE solution. If the FE solution has higher local concentrations than the bulk (e.g., H⁺ pile-up near the electrode), `theta_inner` can transiently go negative and activates `packing_floor = 1e-8`. This is handled by `fd.max_value` flooring and is expected behavior — flagged for awareness, not a bug.

---

## 9. Summary of All Checks

| Check | Status | Notes |
|-------|--------|-------|
| Bikerman formula (single-ion) | PASS | Matches spec; bulk recovery verified |
| Dimensionalization a_nondim | PASS | a_phys × C_SCALE = dimensionless |
| Hard rule #7: A_DEFAULT magnitude | PASS | 14.89 Å, 120 mol/m³, 150× clamp all confirmed |
| A_DEFAULT fix in production presets | PASS (working tree only) | Committed? NO — Issue C |
| K⁺ a_nondim | PASS | r=2.3 Å, formula-computed |
| Cs⁺ a_nondim | PASS | r=2.2 Å, hardcoded (0.2% rounding) |
| SO₄²⁻ a_nondim | PASS | r=2.4 Å, hardcoded (0.4% rounding) |
| OH⁻ a_nondim | PASS | r=1.76 Å, formula-computed |
| Multi-ion shared-θ closure | PASS | Explicit closed form, single producer |
| Single-ion degeneration | PASS | multi_ion = boltzmann for N=1 |
| multi_ion/boltzmann phi_clamp consistency | PASS | _phi_safe_exp mirrors UFL clamp |
| Sign convention (forms_logc.py) | PASS | -fd.ln(packing), post-fix |
| Sign convention (forms_logc_muh.py) | PASS | -fd.ln(packing), post-fix |
| z=0 neutral species handling | PASS | Not counterions; only in A_dyn |
| Poisson + packing consistency | PASS | Same c_steric_expr in both |
| z_scale ramp consistency | PASS | Shared Function, both Poisson and packing |
| theta_b positivity validator | PASS | Raises ValueError if ≤ 0 at build time |
| Double-counting guard | PASS | Raises ValueError if (z,c_bulk) duplicates |
| ClO₄⁻ steric a_nondim (legacy scripts) | ISSUE A | A_DEFAULT=0.01 unphysical, LOW priority |
| Cs⁺/SO₄²⁻ hardcoded vs computed | ISSUE B | Style inconsistency, INFO only |

---

## Bottom Line

The Bikerman/multi-ion closure is mathematically correct and the sign convention is properly fixed (`-fd.ln(packing)`). The most important finding is **Issue C**: the A_DEFAULT-to-physical-radii fix for the production presets (`THREE_SPECIES_LOGC_BOLTZMANN`, `FOUR_SPECIES_LOGC_DYNAMIC_K2SO4`) exists only in the uncommitted working tree — commit immediately to prevent revert risk. No formula errors were found; the multi-ion shared-θ closure is consistent between the UFL residual (`boltzmann.py`) and scalar Picard helper (`multi_ion.py`).

**5 key bullets:**
1. Formula is correct: `c_k = c_b_k * exp(-z_k*phi) * (1-A_dyn) / (theta_b + sum_k a_k*c_b_k*exp(-z_k*phi))` — bulk recovery and single-ion degeneration verified analytically.
2. A_DEFAULT = 0.01 corresponds to r=14.89 Å and c_max=120 mol/m³, confirming CLAUDE.md; the H⁺ clamp is 150× tighter than physical r=2.8 Å (H₃O⁺).
3. The A_DEFAULT fix for production presets is in the working tree but NOT committed — this is Issue C, the highest-priority finding.
4. All counterion radii (K⁺, Cs⁺, SO₄²⁻, OH⁻) are physically consistent with documented Linsey/Marcus values to <0.5%.
5. Sign convention is correct post-fix: `mu_steric = -fd.ln(packing)` in both `forms_logc.py` and `forms_logc_muh.py`.
