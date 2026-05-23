# SONNET_09 — Cation Hydrolysis Verification

**Scope:** Phase 6β cation hydrolysis layer — Singh 2016 formula, σ-mapping,
source-term plumbing, λ-ramp, override ablation, and Δ_β Phase D path.
**Files reviewed:** `Forward/bv_solver/cation_hydrolysis.py` (1518 lines),
`calibration/singh2016.py`, `Forward/bv_solver/forms_logc.py` (σ extraction
section), `Forward/bv_solver/forms_logc_muh.py` (σ extraction + source wiring),
`Forward/bv_solver/config.py` (flag validation), `Forward/bv_solver/units.py`.
**Reference:** `docs/phase6/singh_2016_pka_formula.md`.

---

## 1. Singh 2016 Eq. (3)/(4) — PASS

`calibration/singh2016.py` hardcodes A = 620.32 pm, B = 17.154, r_O = 63.0 pm,
per-cation Table S1 rows (z_eff, r_M_pm, pKa_bulk, r_H_El_pm_Cu) exactly matching
the formula doc (§3.4 numerics, §4 Table S1, §7 back-fits).

`compute_beta_per_cation` computes:

```python
geometric = 1.0 - (r_M_O**2) / (r_H_El**2)
return 2.0 * A * z * r_H_El * geometric
```

This matches Singh Eq. (4'): β = 2·A·z·r_H_El·(1 − r_M-O²/r_H_El²). At K⁺ Cu
default (r_H_El = 200.98 pm), r_M-O = 201 pm, G ≈ −0.0001, β ≈ −45.61 pm²
(negative, as expected). The docstring cites "−45.608196 pm² to 6 decimal places."
No discrepancy found.

`_build_singh_2016_eq_4_pka_shift` in `cation_hydrolysis.py` (lines 535–617)
implements the identical formula as a live UFL expression, with r_H_El either
baked as a `Constant` or read from `r_H_El_pm_func`. Consistent with module.

**Verdict: Singh formula correctly implemented.**

---

## 2. σ-Mapping — PASS WITH NOTE

σ_S is extracted via the Stern Robin residual term:

```python
sigma_S_expr = (
    stern_coeff
    * (phi_applied_func - phi)
    * fd.Constant(sigma_phys_per_nondim)
)
```

where `sigma_phys_per_nondim = F * C_SCALE * L_SCALE` (forms_logc.py:659–673,
mirrored in forms_logc_muh.py:693–707). This uses the physical definition
`σ_S = C_S · ψ_S = C_S · (φ_metal − φ_OHP)`, consistent with the Stern BC
residual term (`F_res -= stern_coeff * (phi_applied - phi) * w * ds`). The
formula doc §5.2 notes this is the natural solver-side equivalent of Singh's
Eq. (5) (cell-voltage route).

Signed convention: `phi_applied - phi` is positive at cathodic bias (φ_metal <
φ_OHP with cathodic φ_metal → negative applied means negative phi_applied value
in nondim, with phi_OHP even more negative... **This requires closer inspection.**

Actually: the solver convention is φ_applied = V_RHE (nondim). At cathodic
V_RHE < 0 and the OHP φ ≈ 0 (ground), so phi_applied_func − phi is
approximately V_RHE − 0 < 0. This gives `stern_coeff * (negative) * physical_factor`
= **negative** σ_S — correct cathodic convention.

In `_build_singh_2016_eq_4_pka_shift`:
```python
sigma_count_per_pm2 = sigma_S * (N_A/F) * 1e-24     # negative at cathode
sigma_singh = max(0, -sigma_count_per_pm2)            # positive magnitude
```

So `sigma_singh > 0` when σ_S < 0 (cathodic), consistent with Singh's positive-scalar
convention. Anode clamp is enforced via `fd.max_value`.

**Caveat (Non-blocking):** the σ extracted from `C_S·(φ_metal − φ_OHP)` is the
Stern sheet charge, which differs from Singh's cell-level σ (Eq. 5) by ≈10⁵× in
magnitude (noted in formula doc §7, acceptance bundle). At V_kin the local Stern σ
≈ −0.017 C/m² → σ_singh ≈ 1.07e−7 counts/pm² → ΔpKa ≈ −4.88e−6 (near zero).
This is the documented reason Phase D was identifiability-limited (the σ magnitude
is orders of magnitude below Singh's Cu/Ag values), not an implementation bug.

---

## 3. Field-Dependent pKa Effect on Hydrolysis Rate — PASS

The rate formula throughout the file is:

```python
pka_factor = fd.Constant(10.0) ** (-pka_shift_expr)
R_forward = k_hyd_func * c_M * pka_factor * vacancy_factor
```

ΔpKa(σ) = β · σ_singh. At cathodic conditions:
- β < 0, σ_singh > 0 → ΔpKa < 0
- `-pka_shift_expr = -ΔpKa > 0`
- `pka_factor = 10^(−ΔpKa) > 1`
- R_forward > k_hyd · c_M (rate increases at cathode) ✓

Physical interpretation: pKa drops at cathodic OHP → hydrolysis equilibrium
shifts toward proton release → effective forward rate increases. The sign is
correct.

The equivalence to `k_hyd_eff = k_hyd · 10^(−ΔpKa)` (using `pKa_eff =
pKa_bulk + ΔpKa`) is exact: `10^(−ΔpKa) = 10^(pKa_bulk − pKa_field)`.

**Verdict: pKa sign convention correct.**

---

## 4. Source-Term Mass Balance — PASS WITH FLAG

In both `forms_logc.py:747–750` and `forms_logc_muh.py:786–789`:

```python
H_residual_term = lam_func * R_net * v_list[h_idx] * ds(electrode_marker)
K_residual_term = lam_func * (-R_net) * v_list[counterion_idx] * ds(electrode_marker)

if apply_h_source:
    F_res -= H_residual_term      # adds +λ·R_net to H⁺ residual
if apply_k_sink:
    F_res -= K_residual_term      # adds −λ·R_net to K⁺ residual (sink)
```

The sign convention in the FEniCS/Firedrake weak form: `F_res -= term` adds the
term as a source (residual = 0 at steady state → source adds to species). H⁺ gains
`+λ·R_net`, K⁺ loses `+λ·R_net` (equivalent to `-λ·R_net` as a flux), matching
the 1:1 stoichiometry of Singh Eq. (2): one H⁺ produced per hydrolysis event with
one M⁺ consumed.

**FLAG — Ablation inconsistency (MEDIUM):** `apply_h_source=True, apply_k_sink=False`
is guarded in `config.py` to require `manufactured_R_inj is not None`. However,
`apply_h_source=False, apply_k_sink=True` is similarly guarded — but the guard only
checks `half_physical = (not apply_h_source) or (not apply_k_sink)` and raises if
`manufactured_R_inj is None`. This is correct. Production runs default to
`apply_h_source=True, apply_k_sink=True`, so mass balance is intact in all
non-ablation runs.

When both flags are True (production): H⁺ source and K⁺ sink are exactly
balanced: same `R_net` applied to both. Stoichiometry is 1:1. The product KOH is
implicit via Γ_MOH; no KOH species in the PNP system. This is a deliberate
simplification (the KOH is the surface-adsorbed species tracked as Γ, not a
free PNP species).

**Verdict: Mass balance enforced in production runs. Ablation guard is correct.**

---

## 5. λ (lambda) Parameter — PASS

`lambda_hydrolysis_func` is an R-space `Function` initialized from
`conv_cfg['lambda_hydrolysis']` (default 0.0). Both forms wrap every
hydrolysis contribution with `lam_func * ...`:

```python
H_residual_term = lam_func * R_net * v * ds
K_residual_term = lam_func * (-R_net) * v * ds
```

In `update_gamma_from_solution`:
```python
lam_val = float(bundle.lambda_hydrolysis_func)
if lam_val == 0.0:
    bundle.gamma_func.assign(0.0); return 0.0
```

At λ=0: Γ→0 exactly, both residual terms vanish → disabled-feature baseline.
At λ=1: full physical hydrolysis. `gamma_ss_langmuir` formula has `(1−λ)` in
the denominator as a regularizer that vanishes at λ=1 (correct — no artificial
damping at full coupling).

In `_bv_common.py` the `lambda_hydrolysis` parameter threads through to
`solve_anchor_with_continuation` via the outer API. The step 9.5 `warm_start_floor`
opt-in supports arithmetic bisection at the λ-ramp floor.

**Verdict: λ-ramp plumbing is correct and consistent.**

---

## 6. `override_sigma_singh_counts_pm2` — PASS

In both form modules:

```python
if sigma_singh_override is not None:
    inv_factor_C_m2_per_count_pm2 = 1.602176634e-19 / 1.0e-24
    fake_signed_sigma_S = fd.Constant(
        -float(sigma_singh_override) * inv_factor_C_m2_per_count_pm2
    )
    pka_shift_expr = build_pka_shift(..., sigma_S=fake_signed_sigma_S, ...)
```

This back-converts from counts/pm² to a signed C/m² value that, after the
`max(0, -sigma_count_per_pm2)` anode clamp and unit conversion inside
`_build_singh_2016_eq_4_pka_shift`, recovers the override value exactly.

`config.py` validates: override must be finite, non-negative, and mutually
exclusive with `manufactured_R_inj`. The override bypasses the Stern σ extraction
and uses a constant Constant — confirming it's a clean ablation path that doesn't
couple to the live Newton solution.

**Verdict: Override plumbing is correct.**

---

## 7. Δ_β (Phase D) — PASS (identifiability issue, not implementation bug)

`beta_offset_pm2_func` is an R-space `Function` defaulting to 0.0 at bundle
build time. Inside `_build_singh_2016_eq_4_pka_shift`:

```python
if beta_offset_pm2_func is not None:
    beta_carbon_expr = beta_per_cation_live + beta_offset_pm2_func
else:
    beta_carbon_expr = beta_per_cation_live
delta_pKa = beta_carbon_expr * sigma_singh
```

At Δ_β=0: β_carbon = β_per_cation_Cu (Cu prior). Phase D swept Δ_β via
`set_reaction_beta_offset_pm2_model`. The σ-mapping outcome confirms the
identifiability failure was physical, not a code path error: at the local Stern
σ magnitudes, ΔpKa ≈ −5e−6 regardless of Δ_β (the σ_singh multiplier is ≈1e−7
counts/pm², making the product negligibly small at any reasonable Δ_β). The
flat Stern σ-mapping loss (15.629 pp uniform, Phase D outcome) is consistent
with this numerical analysis.

**Verdict: Δ_β code path is correct. Phase D flatness is a physical identifiability
limit of the local Stern σ being ~10⁵× smaller than Singh's cell-level σ.**

---

## 8. Gamma/Picard Single Source of Truth — PASS

Step 6 stashes `_cation_hydrolysis_pka_shift_expr` on ctx. Both
`update_gamma_from_solution` and `collect_v10a_rung_diagnostics` first try to
read this ctx key before falling back to rebuilding, ensuring the Picard Γ update
and the residual use the same UFL expression object (including any live
`r_H_El_pm_func` or `beta_offset_pm2_func` updates).

The fallback path (`sigma_S_expr = fd.Constant(0.0)`) is safe for legacy callers
and for the manufactured-R_inj path where σ is irrelevant.

---

## Findings Summary

| # | Check | Result |
|---|---|---|
| 1 | Singh Eq. (3)/(4) constants in `calibration/singh2016.py` | PASS |
| 2 | σ extraction (Stern Robin, signed, physical C/m²) | PASS WITH NOTE |
| 3 | pKa factor sign: cathodic → ΔpKa < 0 → rate increase | PASS |
| 4 | H⁺ source / K⁺ sink symmetry; mass balance | PASS WITH FLAG |
| 5 | λ-ramp (0→1) correct; Γ→0 at λ=0 pinned | PASS |
| 6 | `override_sigma_singh_counts_pm2` ablation path | PASS |
| 7 | Δ_β Phase D plumbing; flat σ-mapping explained | PASS |
| 8 | Picard/residual single source of truth (step 6) | PASS |

---

## Issues

### MEDIUM — σ_S = 0.0 Picard fallback silently uses zero σ if ctx lacks step-6 artifacts (Non-blocking)

**Location:** `cation_hydrolysis.py:966–968`, also `collect_v10a_rung_diagnostics:1146–1148`.

**Issue:** The fallback path `sigma_S_expr = ctx.get("_cation_hydrolysis_sigma_S_expr")` followed by `if sigma_S_expr is None: sigma_S_expr = fd.Constant(0.0)` means any legacy caller that builds a ctx without step-6 artifacts will silently get ΔpKa = 0 in the Picard update (no σ-driven pKa shift). Since production runs (post-step-6) always set this key, this is non-blocking, but it's an invisible correctness hazard for any future code path that skips the step-6 form build.

**Recommendation:** Add a `warnings.warn` when the fallback fires, rather than silent Constant(0.0).

### LOW — No desorption product (KOH free species) — documented design choice

**Location:** Throughout `cation_hydrolysis.py`.

**Observation:** The product KOH is tracked only as surface coverage Γ_MOH, not as a free PNP species. If KOH desorbs (k_des > 0), the K⁺ sink in the PNP residual removes a K⁺, but the KOH desorption product does not reappear as K⁺ in the bulk. This is intentional (the hypothesis is that KOH stays at the OHP on the time scale of the RRDE experiment), but it means desorption effectively destroys K⁺ mass in the PNP system. Mass is only conserved if k_des ≈ 0 or if the Γ steady-state is quasi-static.

This is documented in the architecture (Γ is an external coefficient, not a PNP unknown), but worth flagging for future multi-scale validation.

---

## Bottom Line

**All 8 checked invariants pass.** The Singh 2016 formula constants, σ-mapping sign convention, pKa factor direction, H⁺/K⁺ source-sink symmetry, λ-ramp, override ablation, and Δ_β plumbing are correctly implemented. Phase D's flat identifiability result is confirmed as a physical consequence of local Stern σ being ~10⁵× smaller than Singh's cell-level σ, not a code bug. One medium issue: the Picard σ fallback is silently zero for legacy ctx callers and should emit a warning.
