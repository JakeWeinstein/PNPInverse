# Correctness Audit: Forward/bv_solver/forms_logc_muh.py
**Auditor:** Claude Sonnet 4.6 (subagent, independent read)
**Date:** 2026-05-05
**File:** `Forward/bv_solver/forms_logc_muh.py` (1044 lines)
**Cross-references read:** `boltzmann.py`, `forms_logc.py`, `nondim.py`, `config.py`, `dispatch.py`, `scripts/_bv_common.py`, `docs/clipping_conventions.md`, `docs/bv_solver_unified_api.md`

---

## Summary verdict: CORRECT on all key requirements

No critical correctness bugs found. One low-severity note (inconsistent fallback constant, harmless at runtime) and several observations are documented below. The file correctly implements the `logc_muh` formulation as described in the module docstring and CLAUDE.md.

---

## Requirement-by-requirement analysis

### 1. mu_H bookkeeping

**Status: CORRECT**

- `mu_H = u_H + em*z_H*phi` is consistently used throughout.
- Reconstruction: `_u_expr(i, ui_split, phi_var)` at lines 266-269 returns `ui_split[i] - Constant(em) * z[i] * phi_var` for mu species (i.e., `log(c_H) = mu_H - em*z_H*phi`), and `ui_split[i]` unchanged for non-mu species.
- `c_H = exp(clamp(mu_H - em*z_H*phi, ±u_clamp))` is built at lines 274-277.
- **Critical check:** `c_H_old` is reconstructed using `phi_prev` (line 272: `u_prev_exprs = [_u_expr(i, ui_prev, phi_prev) for i in range(n)]`), not the current `phi`. The module docstring at line 30 calls this out explicitly. Correct.
- The clamp is applied to the reconstructed `log(c_H)`, not raw `mu_H` — consistent with the module note at lines 33-36 and the docstring at lines 127-128. Correct.
- `em` is sourced from `scaling["electromigration_prefactor"]` (line 249), which matches the key published by `Nondim/transform.py` at lines 268, 427. No name mismatch.

### 2. Log-rate BV form

**Status: CORRECT**

The log-rate cathodic branch (lines 418-429):
```
log_cathodic = ln(k0) + u_exprs[cat_idx] - alpha * n_e * eta_clipped
             + sum(power * (u_exprs[sp_idx] - ln(c_ref)))
cathodic = exp(log_cathodic)
```

- Uses `u_exprs[cat_idx]` which is `log(c_cat)` — for the proton this is the muh-recovered `mu_H - em*z_H*phi`, not raw `mu_H`. Correct.
- For the anodic branch of reversible R1 (lines 432-437):
  ```
  log_anodic = ln(k0) + u_exprs[anod_idx] + (1 - alpha) * n_e * eta_clipped
  ```
  Uses `u_exprs[anod_idx]` (H2O2 at index 1), which for non-mu species is simply `ui[1] = log(c_H2O2)`. Correct.
- The non-log-rate path (lines 449-472) correctly uses `c_surf[i] = ci[i]`, which is muh-reconstructed for the proton species. Comment at line 450 notes this explicitly.
- Both branches are identical to `forms_logc.py` lines 376-432 with the key substitution: `ui[sp_idx]` (logc) → `u_exprs[sp_idx]` (muh). This is the correct and complete substitution.

### 3. exponent_clip = 100 convention

**Status: CORRECT**

`_build_eta_clipped` (lines 293-308):
- `eta_scaled = bv_exp_scale * eta_raw` — clip is on `eta_scaled = (V - E_eq) / V_T` before the `alpha * n_e` multiplication. Correct.
- `clip_val = Constant(float(conv_cfg["exponent_clip"]))` — reads from the convergence config, which defaults to 100.0 per `config.py:_default_bv_convergence_cfg()` (line 111).
- Symmetric clip: `min_value(max_value(eta_scaled, -clip_val), clip_val)`. Correct.
- No downstream clipping of `eta` is applied before the BV exponent.
- Implementation is a byte-for-byte copy of `forms_logc.py:_build_eta_clipped` (lines 234-253), including the Stern vs. non-Stern branch. Correct.

### 4. Stern BC

**Status: CORRECT**

Weak-form Stern Robin BC (line 522):
```python
F_res -= stern_coeff * (phi_applied_func - phi) * w * ds(electrode_marker)
```

Derivation: Poisson IBP gives the boundary integral `eps * grad(phi).n * w * ds` at the electrode. The Stern physical condition `eps * grad(phi).n = C_S * (phi_m - phi_s)` replaces this with `C_S * (phi_m - phi) * w * ds`. Since this term is on the RHS of the residual, it enters `F_res` as `-C_S * (phi_m - phi) * w * ds`. Code matches.

Nondimensionalization of `stern_capacitance_model` (lines 189-203) is consistent with `nondim.py:_add_bv_scaling_to_transform` lines 80-88.

The Stern overpotential uses `eta_raw = phi_applied_func - phi - E_eq_const` (line 299), where `phi` is the solution potential at the electrode — the Frumkin-corrected overpotential. Correct.

### 4b. Electrode/ground markers

**Status: CORRECT**

- BV flux at `electrode_marker` only (line 480: `ds(electrode_marker)`).
- Dirichlet BCs for species: at `concentration_marker` (line 549).
- Dirichlet ground BC for phi: at `ground_marker` (line 538).
- Without Stern: additional Dirichlet phi at `electrode_marker` (line 553). With Stern: electrode phi is free, replaced by the Robin BC in the weak form.

### 5. Initial conditions

**Status: CORRECT**

**Linear-phi IC** (`set_initial_conditions_logc_muh`, lines 597-656):
- Non-mu species: `U_prev.sub(i) = ln(c0_i)`. Correct.
- Mu species: `U_prev.sub(mu_h_idx) = ln(c0_H) + em*z_H*phi_init(y)`. Reconstruction `exp(mu_H - em*z_H*phi_init) = exp(ln(c0_H)) = c0_H`. Correct.
- Phi: `phi_init = phi_applied * (1 - y)`. Correct linear profile.

**Debye-Boltzmann IC** (`set_initial_conditions_debye_boltzmann_logc_muh`, lines 659-701 + `_try_debye_boltzmann_ic_muh`, lines 704-1044):
- Picard outer loop is byte-for-byte identical to `forms_logc.py` Picard loop (lines 764-807 vs 861-905). Both use `H_o = max(H_b - (R1 + R2) / D_H, 1e-300)` with denominator D_H (not 2*D_H), consistent with the documented cancellation at forms_logc.py lines 798-800.
- Proton seed for bikerman path (lines 1011-1016):
  ```
  u_h_init_expr  = ln(H_outer) - psi + log_gamma
  mu_h_init_expr = u_h_init_expr + em*z_H * phi_init_expr
  ```
  With em*z_H=1, psi cancels: `mu_H_init = 2*ln(H_outer) - ln(c_clo4_bulk) + log_gamma`. Matches module docstring line 39-40 analytic calculation. Correct.
- Proton seed for ideal counterion path (lines 1035-1040): same cancellation, without log_gamma. Correct.
- 4sp synthesised counterion path (lines 1019-1024): `U_prev.sub(3) = ln(c_clo4_bulk) + phi_init + log_gamma` — ClO4- Boltzmann IC matches logc.py line 937-939 exactly. Correct.
- IC is wrapped in `adj.stop_annotating()` context (line 690), preventing Picard iterations from reaching the adjoint tape. Correct per CLAUDE.md adjoint hygiene note.
- mu_H IC is consistent with the residual: the reconstructed `c_H` at these initial conditions equals the Boltzmann equilibrium profile.

### 6. Bikerman counterion residual (no double-counting)

**Status: CORRECT**

- `build_steric_boltzmann_expressions` is called during form assembly (line 323), before the steric chemical potential and Poisson residual are assembled. Its `charge_density` is added to F_res at lines 513-517.
- `add_boltzmann_counterion_residual(ctx, params, skip_bikerman=True)` at line 589 skips all bikerman entries. Only ideal-mode counterions go through the legacy path.
- `boltzmann.py:add_boltzmann_counterion_residual` at lines 341-342 confirms it skips entries with `steric_mode='bikerman'` when `skip_bikerman=True`.
- No double-counting of the bikerman ClO4- charge density. Correct.

The `boltzmann_z_scale` function is shared between the bikerman path (created in `build_steric_boltzmann_expressions`, line 235-237) and the ideal path (reused at line 328-333). A single `_set_z_factor(ctx, z)` ramps both consistently. Correct.

### 7. NP residual construction

**Status: CORRECT**

For mu species (H+, index `mu_h_idx`):
- `ideal_grad = fd.grad(ui[i])` where `ui[i]` is `mu_H` (lines 364-366).
- `Jflux = D_H * c_H * fd.grad(mu_H)` (or with steric: `D_H * c_H * (grad(mu_H) + grad(mu_steric))`).
- Mathematical identity: `grad(u_H) + em*z_H*grad(phi) = grad(u_H + em*z_H*phi) = grad(mu_H)`. So the single `grad(mu_H)` correctly encodes both diffusion and migration. Correct.

For non-mu species (O2 index 0, H2O2 index 1, both z=0 in production):
- `drift = Constant(em) * z[i] * phi`. With `z[i] = Constant(0.0)`, this is 0.
- `ideal_grad = fd.grad(u_exprs[i]) + fd.grad(drift) = fd.grad(ui[i]) + 0`. Correct for neutral species.
- Even if a charged non-mu species were present (hypothetical), the formula `fd.grad(u_i) + em*z_i*fd.grad(phi)` is the correct log-c NP flux.

Comparison to `forms_logc.py` NP loop (lines 318-337): the muh version correctly substitutes `fd.grad(ui[i])` with `fd.grad(ui[i])` (mu_H gradient) for the mu species, instead of `fd.grad(u_i) + fd.grad(em*z[i]*phi)`. This is mathematically equivalent and the explicit purpose of the muh formulation.

### 8. Nondimensionalization

**Status: CORRECT**

- `em = float(scaling["electromigration_prefactor"])` at line 249. Key matches `transform.py` line 427.
- `bv_exp_scale`, `bv_E_eq_model`, `dt_model`, `D_model_vals`, `c0_model_vals`, `phi_applied_model`, `phi0_model`, `poisson_coefficient`, `charge_rhs_prefactor` all sourced from `scaling` dict produced by `_add_bv_reactions_scaling_to_transform` / `_add_bv_scaling_to_transform`.
- The reactions-path Stern nondim (lines 189-203) duplicates `nondim.py` formula exactly: `stern_model = stern_raw * potential_scale / (F * c_scale * L)`. Correct.

---

## Findings

### NOTE-1 (note): Inconsistent u_clamp fallback in forms builder

**SEVERITY:** note
**LOCATION:** `forms_logc_muh.py:262`
**DESCRIPTION:** `_U_CLAMP = float(conv_cfg.get("u_clamp", 30.0))` uses fallback 30.0, whereas `config.py:_default_bv_convergence_cfg()` returns 100.0 and `_get_bv_convergence_cfg` always includes the `u_clamp` key. The `.get(..., 30.0)` fallback is unreachable in normal usage because `conv_cfg` is always the output of `_get_bv_convergence_cfg(params)` which always populates `u_clamp`. Same pattern exists in `forms_logc.py:211`.
**EVIDENCE:** `config.py:118` returns `u_clamp: 100.0` from the default dict; `_get_bv_convergence_cfg:133-134` always sets `u_clamp`. The 30.0 fallback would only fire if someone passes a raw dict without this key, which bypasses the config parser.
**IMPACT:** None in production. Both files share this inconsistency; it is not a regression in the muh file.

### NOTE-2 (note): exponent_clip Picard IC fallback is 50.0, not 100.0

**SEVERITY:** note
**LOCATION:** `forms_logc_muh.py:800`
**DESCRIPTION:** `exponent_clip = float(conv_cfg.get("exponent_clip", 50.0))` in `_try_debye_boltzmann_ic_muh` uses fallback 50.0. The production default is 100.0. Since `conv_cfg` is populated from `ctx["bv_convergence"]` (set during `build_forms_logc_muh`), and `_get_bv_convergence_cfg` always sets `exponent_clip`, the fallback 50.0 is unreachable in normal usage.
**EVIDENCE:** Same pattern at `forms_logc.py:702`. Not a muh-specific issue.
**IMPACT:** None in production. The fallback would only fire if `_try_debye_boltzmann_ic_muh` is called with a ctx that hasn't been through `build_forms_logc_muh`, which is not a supported usage pattern.

### NOTE-3 (note): J_form re-derivation is redundant when all counterions are bikerman

**SEVERITY:** note
**LOCATION:** `boltzmann.py:360-362` (called from `forms_logc_muh.py:589`)
**DESCRIPTION:** When `boltzmann_counterions=[{steric_mode='bikerman', ...}]` and `skip_bikerman=True`, the `for` loop in `add_boltzmann_counterion_residual` skips all entries but still re-derives `J_form` from an unchanged `F_res`. This is a wasted derivative computation.
**EVIDENCE:** `boltzmann.py:336-362`: the skip-all path still executes lines 360-362.
**IMPACT:** None on correctness. Slight computational overhead during form build, not during solve.

---

## Cross-interface verifications

- **dispatch.py**: Correctly dispatches `formulation='logc_muh'` to `build_context_logc_muh`, `build_forms_logc_muh`, `set_initial_conditions_logc_muh/debye_boltzmann`. No missing IC path.
- **boltzmann.py `build_steric_boltzmann_expressions`**: Call signature at lines 323-333 matches function definition at lines 90-101 exactly (keyword args: `ctx`, `params`, `ci`, `a_dyn_funcs`, `a_dyn_floats`, `c0_dyn`, `z_dyn`, `phi`, `R_space`). No mismatch.
- **boltzmann.py `add_boltzmann_counterion_residual`**: Call at line 589 passes `(ctx, params, skip_bikerman=True)`, matching signature at `boltzmann.py:255-260` (`ctx`, `params`, `*, skip_bikerman=False`). Correct.
- **nondim.py**: `electromigration_prefactor` key is published at line 427 and consumed at forms_logc_muh.py:249. No name mismatch.
- **CLAUDE.md `set_initial_conditions_logc_muh` gotcha**: The CLAUDE.md warns that `set_initial_conditions(ctx, sp, blob=True)` is silently ignored in log-c mode. `dispatch.py:96-120` confirms that `blob` is accepted and ignored. Not a bug in `forms_logc_muh.py` itself.
- **CLAUDE.md `validate_solution_state` gotcha**: The warning applies to external callers, not to `forms_logc_muh.py`. `validation.py` supports the muh case via `mu_species` kwarg.

---

## Key correctness arguments (positive case)

1. **mu_H transform**: The substitution is applied at exactly five c_H-touch sites identified in the module docstring (NP flux, BV surface concentration, BV log-rate, Bikerman packing, Poisson source). All five are correctly handled via the `_u_expr` helper and `ci = exp(clamp(_u_expr(...)))`.

2. **No migration term in the H+ NP equation**: `fd.grad(ui[mu_h_idx])` = `fd.grad(mu_H)` correctly encodes `grad(u_H) + em*z_H*grad(phi)` in one term. No separate drift term is added (and none should be).

3. **phi_prev hygiene**: The reconstructed `c_H_old` uses `phi_prev` from the previous time step, not the current `phi`. This is critical for transient correctness and is implemented correctly at line 272.

4. **Debye-layer cancellation**: The analytic cancellation documented in the module docstring (mu_H_init is smooth across the Debye layer because psi cancels) is verified algebraically from lines 1011-1016 and 1035-1040.

5. **Stern BC sign**: The Robin term `F_res -= C_S*(phi_m - phi)*w*ds` correctly replaces the Neumann integral from the IBP of the Poisson equation, with the sign consistent between muh and logc forms.

6. **Bikerman counterion no-double-count**: The `skip_bikerman=True` flag and the ordering (steric expressions built first, ideal Boltzmann appended after) prevent any double-counting of the ClO4- charge density in Poisson.
