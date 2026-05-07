# Correctness Verification Report: forms_logc.py + boltzmann.py

**Verifier:** Claude Sonnet 4.6  
**Date:** 2026-05-05  
**Files reviewed:**
- `Forward/bv_solver/forms_logc.py` (971 lines)
- `Forward/bv_solver/boltzmann.py` (365 lines)

**Cross-references consulted:** `dispatch.py`, `config.py`, `nondim.py`, `scripts/_bv_common.py`, `docs/steric_analytic_clo4_reduction_handoff.md`

---

## Summary: MOSTLY CORRECT, one warning, one note

No critical bugs found. One substantive warning (latent wrong fallback in IC Picard code), one note (documentational sign inconsistency in NP flux definition).

---

## Requirement-by-Requirement Findings

### 1. NP Residual in Log-c Form

**Result: CORRECT**

**Evidence:**

- `ci[i] = exp(clamp(ui[i], -_U_CLAMP, +_U_CLAMP))` (lines 212–215). `_U_CLAMP` defaults from `conv_cfg["u_clamp"]` which defaults to 100.0. `exp(u_i)` is used correctly as the concentration.
- `Jflux = D[i] * c_i * (grad(u_i) + grad(drift))` where `drift = em * z[i] * phi` (lines 327–333). This is the integrated-by-parts form of the flux, NOT the physical flux `J_i`. In the physical flux: `J_i = -D_i c_i (∇u_i + em·z_i·∇φ)`. After IBP on `∫ ∂_t c · v dx + ∫ ∇·J · v dx = BC`, using `∫ ∇·J·v dx = -∫ J·∇v dx + boundary`, we get: `∫ (c-c_old)/dt·v dx + ∫ D·c·(∇u + em·z·∇φ)·∇v dx + BV_BC = 0`. The code assigns `Jflux := D·c·(∇u + ∇drift)` which equals `−J_i` physically, then adds `+fd.dot(Jflux, grad(v)) dx`. This is consistent.
- **Minor note:** The inline comment at line 310 says `J_i = -D_i·c_i·(∇u_i + z_i·∇φ)` (physical flux, with minus sign), but the code variable `Jflux` is the negated flux (no minus sign in the assignment). The sign is correct mathematically — `F_res += dot(Jflux, grad(v)) dx` — but the comment labeling `Jflux` as `J` could cause future confusion. This is a documentation issue, not a code bug.
- No spurious migration sign flip. `grad(drift) = em·z[i]·grad(phi)`, positive for cations (z>0) and negative for anions (z<0), which correctly drives cations toward the negative electrode and anions toward the positive electrode.
- Steric path: `Jflux = D[i]*c_i*(grad(u_i) + grad(drift) + grad(mu_steric))` where `mu_steric = -ln(packing)` (lines 329–330, 303). The steric contribution drives species away from high-packing regions, which is physically correct for Bikerman steric potential.

### 2. Log-rate Butler–Volmer (`bv_log_rate=True`)

**Result: CORRECT**

**Evidence:**

- Cathodic (lines 379–390): `log_cathodic = ln(k0) + u_cat - alpha*n_e*eta_j + Σ power*(u_sp - ln(c_ref))`. Then `cathodic = exp(log_cathodic)`. The structure `k0 * c_cat * exp(-alpha*n_e*eta)` in log form, matching the requirement: `exp(ln k0 + α·n_e·η_clipped + Σ stoich_i·u_i,surf)` — note the requirement uses `+α·n_e·η`, but `η` in the requirement is the cathodic overpotential (negative at reductive conditions), so `+α·n_e·η_negative` = `-α·n_e·|η|`, which equals the code's `-alpha*n_e*eta_j` when `eta_j > 0` (anodic convention). This is consistent.
- Anodic (lines 393–408): `log_anodic = ln(k0) + u_anod + (1-alpha)*n_e*eta_j`. Correct for anodic branch.
- R1 is reversible (cathodic O2, anodic H2O2); R2 is irreversible (anodic = 0). `R_j = cathodic - anodic` (line 434).
- **Sign convention for R1 cathodic:** `stoichiometry_r1 = [-1, +1, -2]` (O2 consumed, H2O2 produced, H+ consumed). `F_res -= stoi[i] * R_j * v * ds`. For O2: `stoi=-1`, so `F_res -= (-1)*R_j*v_O2*ds = F_res += R_j*v_O2*ds`. At cathodic conditions `R_j > 0`, this adds positive source to O2 equation, which in the residual context means the NP flux must supply O2 to compensate — i.e., O2 flows from bulk toward the electrode. Correct.
- Legacy path (`bv_log_rate=False`, lines 410–432): `cathodic = k0*c_surf[cat]*exp(-alpha*n_e*eta)`, `anodic = k0*c_surf[anod]*exp((1-alpha)*n_e*eta)`. Correct standard BV form.
- The BV log-rate form is byte-identical between `forms_logc.py` and `forms_logc_muh.py`, confirmed by inspection.

### 3. exponent_clip = 100 Convention

**Result: CORRECT**

**Evidence:**

- `config.py` line 111: `"exponent_clip": 100.0` in `_default_bv_convergence_cfg()`.
- `config.py` line 131: `exponent_clip = float(raw.get("exponent_clip", 100.0))` in `_get_bv_convergence_cfg`.
- `config.py` line 147: returned in the config dict.
- `forms_logc.py` `_build_eta_clipped` (lines 250–252): clip applied to `eta_scaled = bv_exp_scale*(phi_applied - E_eq)` (scaled by `F/RT`), not to `alpha*n_e*eta_scaled`. This is the correct convention — clip is on `η/V_T`, so a reaction's BV exponent unclips when `|V - E_eq| < clip*V_T`.
- `_diag_exponent_clip` stored in ctx at line 534 reads from `conv_cfg["exponent_clip"]`, which is always the parsed value (100.0 default). ✓

### 4. Stern BC

**Result: CORRECT**

**Evidence:**

- Lines 486–488: `F_res -= stern_coeff * (phi_applied_func - phi) * w * ds(electrode_marker)`. This is the Stern Robin condition: the normal flux of φ at the electrode boundary is proportional to `C_S * (V_metal - φ_solution)`. Added only at `electrode_marker`.
- `use_stern` flag gates this condition correctly (line 232, 486).
- When `use_stern=True`, Dirichlet BC `phi = phi_applied` at electrode is replaced by the Robin condition (line 507 does NOT add the electrode BC in that branch). ✓
- Overpotential `eta_raw = phi_applied_func - phi - E_eq_const` in Stern mode (line 244): correctly measures the diffuse-layer potential drop `(V_metal - φ_solution)` minus `E_eq`. ✓

### 5. Bulk Dirichlet BCs

**Result: CORRECT**

**Evidence:**

- Lines 497–503: `bc_ui[i] = DirichletBC(W.sub(i), Constant(ln(c0_i)), concentration_marker)` for each species. Floor `_C_FLOOR = 1e-20` prevents `ln(0)`.
- H2O2 bulk BC uses the seed concentration `H2O2_SEED_NONDIM = 1e-4` passed via `c0_vals_hat` in `_bv_common.py` (lines 198–204). This is passed as `c0` in the solver_params and becomes `c0_model[1]` in the BC, so `bc_u1 = DirichletBC(W.sub(1), Constant(ln(1e-4)), concentration_marker)`. ✓ The seed is the bulk BC value, which is finite and physically reasonable.
- `ground_marker` sets `phi = 0` at the counter-electrode (line 497). ✓
- Electrode φ BC (Dirichlet, when no Stern): `DirichletBC(W.sub(n), phi_applied_func, electrode_marker)` (line 507). ✓

### 6. IC Routines

**Result: CORRECT, with one warning**

**`set_initial_conditions_logc` (linear-phi IC):**
- Sets `u_i = ln(c0_i)` (uniform) for each species (lines 568–570). ✓
- Sets `phi` to linear profile from `phi_applied` at y=0 to 0 at y=1 (lines 573–577). ✓

**`set_initial_conditions_debye_boltzmann_logc` (DB-IC):**
- Falls back to `set_initial_conditions_logc` (linear-phi) when: `n<3`, fewer than 2 reactions, or no Boltzmann counterion (lines 644–689). ✓ Matches CLAUDE.md spec.
- Specifically: if no `synthesised_4sp` counterion AND no `steric_mode='bikerman'` entry, falls back at line 689: `return False, "no_boltzmann_counterion", 0`.
- Bikerman IC path (lines 858–940): builds composite-psi profile + multispecies-gamma correction. The IC formula is correct: `u_i = ln(c_outer_i) + z_i*(-psi) + ln(gamma)` = `ln(c_outer_i * gamma * exp(-z_i*psi))`. For O2 (z=0): `u_O2 = ln(O_outer) + log_gamma` (line 931). For H2O2 (z=0): `u_P = ln(P_outer) + log_gamma` (line 932). For H+ (z=+1): `u_H = ln(H_outer) - psi + log_gamma` (line 933). Correct.
- `phi_init_expr = ln(H_outer/c_clo4_bulk) + psi` (line 913): outer electroneutrality + Debye profile. ✓
- 4sp synthesised ClO4- IC (lines 935–939): `u_ClO4 = ln(c_clo4_bulk) + phi_init_expr + log_gamma`. As derived: `ln(c_clo4_bulk) + phi_init_expr = ln(H_outer) + psi` (since `phi_init_expr = ln(H_outer/c_clo4_bulk) + psi`). So `u_ClO4 = ln(H_outer) + psi + log_gamma`. For the bikerman case, `c_ClO4 = c_cl_anchor * gamma * exp(psi)` (z=-1), and `c_cl_anchor = H_outer` for the 4sp case. So `u_ClO4 = ln(H_outer) + ln(gamma) + psi`. ✓
- All Picard iterations run under `adj.stop_annotating()` (line 609). ✓
- `blob=True` is silently ignored in `dispatch.py` `set_initial_conditions` (lines 108–110). ✓

**`set_initial_conditions(ctx, sp, blob=True)` silencing:**
- `dispatch.py` line 97–110: `blob: bool = False` accepted and `""blob is accepted for backward-compatible kwargs and silently ignored""`. ✓

### 7. boltzmann.py

**Result: CORRECT**

**`build_steric_boltzmann_expressions`:**

- Bikerman formula (line 225): `c_steric = c_b * q * free_dyn / (theta_b + a_b * c_b * q)` where `q = exp(-z_b * phi_clamped)`. For `z_b=-1`: `q = exp(+phi)`, giving `c_steric = c_b * exp(+phi) * (1-A) / (theta_b + a_b*c_b*exp(+phi))`. **Matches the derivation in `docs/steric_analytic_clo4_reduction_handoff.md` exactly.** ✓
- `theta_b = 1 - A_dyn_bulk - a_b*c_b` (lines 181–182). ✓
- `packing_contribution = a_b * c_steric` (line 227). This is the ClO4- contribution to the total packing fraction. ✓
- `charge_density = z_b * c_steric = (-1) * c_steric` (line 228). The Poisson term is `F_res -= z_scale * charge_rhs * charge_density * w dx` = `F_res -= (-1) * charge_rhs * c_steric * w dx = F_res += charge_rhs * c_steric * w dx`. Correct: ClO4- (negative charge) reduces the net source in Poisson (which has `F_res -= charge_rhs * sum(z_i*c_i) dx` for dynamic species with positive z dominating). ✓
- `phi_clamped` applies symmetric clamp `|phi| <= phi_clamp_val` (lines 206–208). Default `phi_clamp = 50.0` from `config.py` line 226. Applied to ALL exp() calls (the single `q = exp(-z_b*phi_clamped)` at line 215). ✓
- Double-count guard (lines 192–203): raises `ValueError` if bikerman entry has same (z, c_bulk) as a dynamic species. ✓
- `theta_b > 0` validation (lines 183–189). ✓
- Multi-counterion bikerman raises `NotImplementedError` (lines 159–165). ✓
- Shared `boltzmann_z_scale` (lines 232–237): reuses existing Function if present, otherwise creates new one. Consistent with `add_boltzmann_counterion_residual` behavior. ✓

**`add_boltzmann_counterion_residual` (ideal path):**

- Ideal counterion term (lines 355–358): `F_res -= z_scale * charge_rhs * z_const * c_bulk * exp(-z*phi_clamped) * w dx`. For `z=-1`: `F_res -= z_scale * charge_rhs * (-1) * c_b * exp(+phi) * w dx = F_res += z_scale * charge_rhs * c_b * exp(+phi) * w dx`. Correct sign: negative charge species increases φ source in the physical Poisson equation. ✓
- `skip_bikerman=True` correctly bypasses bikerman entries (lines 341–342). ✓
- Re-derives `J_form` from updated `F_res` (lines 361–362). ✓

**No double-counting:**
- `forms_logc.py` line 471–483: adds `steric_boltz.charge_density` to Poisson (for bikerman only, gated on `steric_boltz is not None`).
- `forms_logc.py` line 541: calls `add_boltzmann_counterion_residual(ctx, params, skip_bikerman=True)` — bikerman entries skipped.
- The same bikerman counterion is NOT added twice. ✓

**Saturation agreement between NP and Poisson sides:**
- `packing_contribution = a_b * c_steric` enters the NP `mu_steric` via `theta_inner = 1 - A_dyn - z_scale * packing_contribution` (forms_logc.py line 296–298).
- Poisson gets `charge_density = z_b * c_steric = -c_steric` (for ClO4-).
- Both use the SAME `c_steric` UFL expression (shared via `StericBoltzmannBundle`). ✓ No mismatch in saturation.

---

## Issues Found

### WARNING: Picard IC uses stale fallback default for `exponent_clip`

- **SEVERITY:** warning
- **LOCATION:** `forms_logc.py:702`, function `_try_debye_boltzmann_ic`
- **DESCRIPTION:** `exponent_clip = float(conv_cfg.get("exponent_clip", 50.0))` uses fallback 50.0, inconsistent with the config-layer default of 100.0. The production `conv_cfg` (populated by `_get_bv_convergence_cfg`) always contains `exponent_clip`, so this fallback is never reached in normal operation. However, if `_try_debye_boltzmann_ic` is called with a bare `{}` bv_convergence context (e.g., in a test or if `ctx` is built manually without `build_forms_logc`), the Picard loop would clip at `±50` instead of `±100`, computing a systematically wrong IC for the warm-start seed. This is a latent bug triggered by non-standard calling patterns.
- **EVIDENCE:** `config.py` line 131: `float(raw.get("exponent_clip", 100.0))`; `forms_logc.py` line 702: `float(conv_cfg.get("exponent_clip", 50.0))`. The fallback values disagree.
- **SUGGESTED FIX:** Change line 702 to `exponent_clip = float(conv_cfg.get("exponent_clip", 100.0))`.

### NOTE: Comment labels `Jflux` as `J` (the physical flux) but `Jflux` is `−J`

- **SEVERITY:** note
- **LOCATION:** `forms_logc.py:310` (comment), lines 329–333 (code)
- **DESCRIPTION:** The comment at line 310 writes `J_i = -D_i·c_i·(∇u_i + z_i·∇φ)` (correct physical flux, with leading minus). The code variable `Jflux` is assigned `D[i]*c_i*(...)` (no minus sign), i.e., `Jflux = -J_i` physically. The comment block at lines 329 writes `J = D·c·(∇u + ∇drift + ∇μ_steric)` which is also the negated flux. The `F_res += dot(Jflux, grad(v)) dx` is correct because the IBP requires `-J` dotted with the gradient. However, the inconsistency between the `J_i` comment at line 310 (which has the minus sign) and the `J = ...` comment at lines 329/332 (which drops the minus sign) could cause a future maintainer to introduce a sign error when modifying the steric or migration terms.
- **EVIDENCE:** Lines 310 vs 329 and 332 use inconsistent sign notation for the same quantity.
- **IMPACT:** Documentation only; no code bug.

---

## Key Correctness Arguments (Summary)

1. **NP log-c residual:** `Jflux = D·c·(∇u + em·z·∇φ)` is the negated physical flux; `F_res += dot(Jflux, ∇v) dx` implements the IBP form `∫ D·c·(∇u + em·z·∇φ)·∇v dx` of the NP equation correctly.

2. **BV rates:** Log-rate and legacy paths agree mathematically. `R_j = cathodic - anodic` with cathodic using `exp(-α·n_e·η)` and anodic using `exp(+(1-α)·n_e·η)`. Stoichiometry signs correctly consume O2 and H+ (negative stoich) and produce H2O2 (positive stoich) in R1.

3. **Clip convention:** `exponent_clip = 100` is the default at both the config layer (`config.py:131`) and the operational forms (`conv_cfg["exponent_clip"]` in `_build_eta_clipped`). Clip applied to `eta_scaled = (V-E_eq)/V_T` before α·n_e multiplication.

4. **Stern BC:** Robin condition `−C_S·(V_metal − φ_sol)·w ds` replaces the Dirichlet electrode BC and correctly uses the diffuse-layer potential drop in the BV overpotential.

5. **Bikerman closure:** `c_steric = c_b·exp(+φ)·(1-A)/(θ_b + a_b·c_b·exp(+φ))` (z=-1) matches the steady-state inert-counterion algebraic reduction derived in the handoff doc. The `packing_contribution` and `charge_density` are built from the same `c_steric` expression and shared between the NP and Poisson sides via `StericBoltzmannBundle`. No double-counting is possible because `skip_bikerman=True` guards the legacy ideal-path call.

6. **No double-counting:** Bikerman path uses `build_steric_boltzmann_expressions` (called inside `build_forms_logc`) + inline Poisson term; then `add_boltzmann_counterion_residual(skip_bikerman=True)` handles ideal-only entries and cannot touch bikerman entries.

7. **`blob=True` silencing:** `dispatch.py` `set_initial_conditions` accepts `blob` as a kwarg and explicitly ignores it per the docstring. ✓

8. **DB-IC fallback to ideal/linear:** `_try_debye_boltzmann_ic` returns `(False, "no_boltzmann_counterion", 0)` when no counterion config is present, triggering `set_initial_conditions_logc` fallback. For 3sp+ideal-only (no bikerman), `apply_bikerman_ic = False` and the code takes the `else` branch (lines 942–949) with the GC-only IC.
