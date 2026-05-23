# SONNET_12: Post-Solve Validation + Observable Extraction
**Scope:** `validation.py`, `observables.py`, `rrde_observables.py`, `diagnostics.py`
**Date:** 2026-05-22
**Reviewer:** claude-sonnet-4-6 (Agent 12/13)

---

## Summary

All core observable computations are **correct and internally consistent**. Three issues found: one missing check (Bikerman c_max for dynamic H⁺, Hard Rule #7), one inconsistency in N_COLLECTION across scripts, and one documented Jensen approximation in surface diagnostics.

---

## Verification Results by Item

### 1. `validate_solution_state` — required kwargs

**PASS.** Signature (`validation.py:45–61`) accepts `is_logc`, `mu_species`, `em` with correct defaults (`is_logc=False`, `mu_species=None`, `em=1.0`). Recovery of `c_H` from `μ_H` is implemented correctly in `_conc_array` (lines 103–117):

```python
log(c_i) = U.dat[i] - em * z_i * U.dat[n_species]   # line 115
c_i = exp(...)                                         # line 115
```

This exactly matches the `forms_logc_muh.py` transform `μ_H = u_H + em·z_H·φ`.

### 2. `validate_solution_state` — physics checks

| Check | Status | Detail |
|---|---|---|
| (a) Positivity c_i > 0 | PASS | F1 fires on `c_min < -eps_c`; structurally unreachable in logc mode (exp > 0 always) |
| (b) Bikerman c_max for counterions | PARTIAL — see Issue #1 below | W2 flags `c > c_bulk * 5x` for charged species but has no check vs `1/a_nondim` Bikerman cap for dynamic species |
| (c) ψ sign at electrode | NOT CHECKED — see Issue #2 below | No explicit "phi at electrode should be negative for cathodic" check; W3 only flags gross oscillations using `±2·|phi_applied|+25` bounds |
| (d) O₂/H₂O₂ bulk mass balance | NOT IMPLEMENTED | No FEM domain-integral mass balance (∫Ω ∂c/∂t + ∫∂Ω J·n − ∫Ω S = 0) exists anywhere in `diagnostics.py` |
| (e) Charge balance Σ z_i·c_i | NOT IMPLEMENTED | No Σ z_i·c_i + σ_steric check in `diagnostics.py` or `validation.py` |
| F4 floor domination | PASS | Correctly skipped in logc mode (no eps_c floor) |
| F6 stoichiometric limit | PASS | Fires when c_H2O2_max > 1.05·c_O2_bulk |

### 3. Disc current density extraction — `observables.py`

**PASS.** `mode="current_density"` builds `Σ_j (n_e_j / N_ELECTRONS_REF) * R_j * ds(electrode)` with `N_ELECTRONS_REF = 2` (lines 106–121). For parallel 2e+4e: weights are `(2/2)·R_2e + (4/2)·R_4e = R_2e + 2·R_4e`. Callers universally use `scale=-I_SCALE` (confirmed across all study scripts), yielding cathodic = negative convention. The electrode boundary is selected via `electrode_marker` from `bv_settings`, consistent with form-build time.

### 4. Peroxide selectivity — `rrde_observables.py` vs `phase6b_v10a_phase_A2_v_kin.py`

**PASS — formulas are internally consistent, despite apparent mismatch.**

`rrde_observables.py` uses:
```
S = 200 * (j_ring/N) / (|j_disk| + j_ring/N)
```

`phase6b_v10a_phase_A2_v_kin.py:augment_rung_diagnostics` uses:
```
H2O2_selectivity_pct = 100 * R_2e / (R_2e + R_4e)
```

These are algebraically identical given the conventions:
- `j_disk = I_SCALE · (R_2e + 2·R_4e)` (electron-weighted)
- `j_h2o2_disk = I_SCALE · R_2e` (gross 2e current, no extra n_e weight — `gross_h2o2_current` mode)
- Substituting: `S = 200·R_2e / ((R_2e + 2·R_4e) + R_2e) = 100·R_2e / (R_2e + R_4e)` ✓

### 5. Ring current via collection efficiency — `rrde_observables.py`

**PASS with minor inconsistency.** `j_ring = N · |j_h2o2_disk|` (line 118). Sign flip is explicit: cathodic disk peroxide → positive ring current. `_validate_collection` guards N ∈ (0, 1].

**Issue #3 (LOW):** Most study scripts use `N_COLLECTION = 0.224` but `_phase_D_plot_vs_slide15.py:54` uses `N_collection = 0.2237`. The `assemble_rrde_observables` entry-point correctly accepts N as a parameter (no hardcoded default), so the discrepancy is in the caller, not the library. The 0.15% difference in N produces negligible selectivity error but should be standardized.

### 6. Mass balance — `diagnostics.py`

**PARTIAL IMPLEMENTATION.** The `mass_balance_residual_rel` discussed in MEMORY (step 9 B.2, "≤ 5e-13") is the **Langmuir surface coverage balance** (`R_forward_capped − k_prot·γ − k_des·γ ≈ 0`) computed in `phase6b_v10a_phase_A2_v_kin.py`, not a FEM global PDE mass balance. There is no `∫Ω ∂c_i/∂t dV + ∫∂Ω J_i·n dA − ∫Ω S_i dV = 0` check anywhere in `diagnostics.py`. The 5e-13 values reported are machine precision on the Langmuir ODE identity, not a PDE flux check.

### 7. Charge balance — `diagnostics.py`

**NOT IMPLEMENTED.** No `Σ_i z_i·c_i + σ_steric` bulk electroneutrality check exists. `collect_diagnostics` reports per-counterion surface concentrations and Bikerman saturation for analytic Boltzmann counterions only.

### 8. Sign conventions

**PASS.** Cathodic current is negative: `scale=-I_SCALE` is used in all production study scripts. `rrde_observables.py` docstring (lines 21–31) locks the sign convention explicitly: cathodic disk → negative; ring → positive via `abs(j_h2o2_disk)`.

---

## Issues

### Issue #1 — MEDIUM: Missing Bikerman c_max check for H⁺ in `validate_solution_state`

**File:** `Forward/bv_solver/validation.py`

Hard Rule #7 documents that H⁺ is seeded with `A_DEFAULT = 0.01` (r ≈ 14.9 Å), giving a Bikerman cap of `c_max ≈ 1/a_nondim ≈ 100 nondim ≈ 120 mol/m³`. Physical H₃O⁺ Stokes radius r = 2.8 Å would give `c_max ≈ 1.8×10⁴ mol/m³`. W2 checks `c_max > c_bulk * 5x` (5x tolerance for charged species) but does NOT check `c_max` vs the actual Bikerman hard-sphere cap `1/a_nondim`. If c_H reaches the A_DEFAULT cap and saturates there, validation currently issues only a W2 (if even that — c_bulk[H⁺] at pH 4 is ≈ 10⁻⁴ mol/m³ = very small, so W2's 5x threshold won't trigger on H⁺ pileup).

**Recommendation:** Add a check in `validate_solution_state` for any species with a known `a_nondim` value: if `c_max * a_nondim ≥ 0.9` (approaching steric saturation), emit a new warning, e.g. `W9: H+ near Bikerman cap`. Requires passing `a_nondim` per species as an optional kwarg.

### Issue #2 — LOW: No electrode-side φ sign check

**File:** `Forward/bv_solver/validation.py`

W3 uses a symmetric `±(2·|phi_applied| + 25)` band, which does not detect the pathology where `φ(electrode)` has the wrong sign relative to `phi_applied` (indicating a BC wiring error). This is a diagnostic gap, not a physics bug, since BC enforcement is verified elsewhere.

### Issue #3 — LOW: N_COLLECTION inconsistency

**File:** `scripts/studies/_phase_D_plot_vs_slide15.py:54`

Uses `N_collection = 0.2237` while all other study scripts use `0.224`. The 0.15% difference is numerically negligible but could cause confusion in comparative analysis. Should be unified to the canonical 0.224 per Ruggiero 2022 §1.

### Issue #4 — INFO: Jensen approximation in `surface_field_means`

**File:** `Forward/bv_solver/diagnostics.py:46–73`

`c{i}_surface_mean = exp(mean(u_i))` is `exp(E[u])`, not `E[exp(u)] = E[c_i]` (Jensen inequality: `exp(E[u]) ≤ E[exp(u)]`). This underestimates mean surface concentration when `u_i` has large variance (e.g., near electrode under strong cathodic polarization). The code comments explicitly acknowledge this as a tracked fault ("exp(mean(u)) Jensen-fault for both formulations — a separate correctness fix is tracked"). Downstream: `c_H_surface_nondim` fed to `compute_surface_pH_proxy` inherits this bias, making surface-pH slightly alkaline-shifted. The ring-current and selectivity paths do not use this value, so the observable chain is unaffected.

---

## Bottom Line

The core observables (disc current, ring current, selectivity, n_e) are **correctly implemented and internally consistent**. The most actionable finding for production use is the **missing Bikerman c_max cap check for H⁺** (Issue #1): validation cannot currently detect the H⁺ steric saturation pathology flagged in Hard Rule #7. There is also no FEM domain mass balance or charge balance in the diagnostic layer — both are absent by design (not implemented), not broken.

---

*Files reviewed:* `Forward/bv_solver/validation.py` (324 lines), `Forward/bv_solver/observables.py` (212 lines), `Forward/bv_solver/rrde_observables.py` (211 lines), `Forward/bv_solver/diagnostics.py` (356 lines).
