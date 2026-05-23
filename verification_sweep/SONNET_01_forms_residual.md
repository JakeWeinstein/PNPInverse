# SONNET_01: Residual Assembly (non-BV) of forms_logc_muh.py

## Bottom line

FLAG-MINOR — residual assembly is structurally correct and the μ_H change-of-variable is faithfully implemented; two minor issues found: (1) a dead fallback default `u_clamp=30.0` in the forms layer contradicts the config default of 100.0 (never triggered but misleading); (2) the non-mu NP flux constructs `fd.grad(u_exprs[i]) + fd.grad(drift)` rather than the algebraically equivalent but marginally cleaner `fd.grad(ui[i]) + fd.Constant(em)*z[i]*fd.grad(phi)` — no physical error but relies on `u_exprs[i] == ui[i]` holding for all non-mu species, which is true but implicit.

---

## Verified claims

- **μ_H reconstruction is correct** — `u_exprs[mu_h_idx] = ui[mu_h_idx] - fd.Constant(em)*z[mu_h_idx]*phi` (`forms_logc_muh.py:331–333`); `ci[mu_h_idx] = exp(clamp(μ_H − em·z_H·φ))` (`forms_logc_muh.py:338`). c_H_old correctly uses `phi_prev` not `phi` (`forms_logc_muh.py:336`). The documented landmine is properly handled.

- **H⁺ NP residual is linear in ∇μ_H** — For `i in mu_species`, `ideal_grad = fd.grad(ui[i])` i.e. `∇μ_H` (since `ui[mu_h_idx]` is the raw primary variable μ_H). No `em·z·∇φ` term separately appears. `F_res += dot(D_H·c_H·(∇μ_H + ∇μ_steric), ∇v) * dx` (`forms_logc_muh.py:473,499,505`). This is the central point of the μ_H change of variable; it is correct.

- **Neutral-species (O₂, H₂O₂, z=0) NP residual is pure diffusion** — `drift = fd.Constant(em)*z[i]*phi` evaluates to zero for z=0 species so `ideal_grad = fd.grad(u_exprs[i])` only. No electromigration coupling (`forms_logc_muh.py:475–477`).

- **Poisson signs and charges are correct** — `+eps_coeff * dot(∇φ, ∇w) * dx` (stiffness, positive), `−charge_rhs * Σ_i z_i·c_i·w * dx` (source, negative), counterion Bikerman contribution `−z_scale·charge_rhs·charge_density_total·w * dx` (`forms_logc_muh.py:644–660`). `ci[mu_h_idx]` is the muh-reconstructed c_H, so z_H·c_H correctly contributes to ρ. Poisson form is byte-identical to `forms_logc.py:600–616`.

- **em consistent and correctly sourced** — `em = float(scaling["electromigration_prefactor"])` at `forms_logc_muh.py:313`; `electromigration_prefactor = (F/RT)·V_T` (`Nondim/transform.py:396`); equals 1.0 when `potential_scale = V_T` (production). Used consistently to wrap in `fd.Constant(em)` for UFL expressions. IC path uses `scaling.get("electromigration_prefactor", 1.0)` (line 922) with safe default.

- **Steric Bikerman wiring** — `build_steric_boltzmann_expressions` is called before the NP loop (`forms_logc_muh.py:418–428`); returns `StericBoltzmannBundle` list with `charge_density`, `packing_contribution`, and shared `z_scale`. Poisson uses `b.charge_density` (`boltzmann.py:258 = z_const * c_steric_k`). Packing uses `b.packing_contribution` (`boltzmann.py:257 = a_const * c_steric_k`). Both enter the correct residual positions. The shared `z_scale` Constant is the same object in both Poisson and mu_steric (`boltzmann.py:244–250`).

- **Test/trial space partitioning** — `v_list = v_tests[indices.species_slice]`, `w = v_tests[indices.phi_index]` (`forms_logc_muh.py:307–308`). NP loop uses `v = v_list[i]`; Poisson uses `w`. No cross-leakage. `forms_indexing.py` documents both legacy (`phi=-1`) and Γ-augmented (`phi=-2`) layouts and the forms correctly request `has_gamma=False` (`forms_logc_muh.py:300`), matching current production layout.

- **Stern Robin BC** — `F_res -= stern_coeff * (phi_applied_func − phi) * w * ds(electrode_marker)` (`forms_logc_muh.py:668`). Sign: when phi < phi_applied (cathodic), this adds a positive source to the φ-equation on the electrode boundary, consistent with `g_S = C_S·(V_app − φ)`. Byte-identical to `forms_logc.py` Stern term.

- **add_boltzmann_counterion_residual called with skip_bikerman=True** (`forms_logc_muh.py:881`) — prevents double-counting the Bikerman counterion charge that was already wired via `build_steric_boltzmann_expressions`. Correct.

---

## Discrepancies / issues

### MINOR — Dead fallback `u_clamp=30.0` in forms layer — `forms_logc_muh.py:326`, `forms_logc.py:274`

Both forms use `conv_cfg.get("u_clamp", 30.0)` as a local fallback. However `_get_bv_convergence_cfg` (`config.py:166`) always populates `conv_cfg["u_clamp"]` with a minimum of 100.0 (default), so the 30.0 fallback is unreachable in any call path that goes through the normal factory. If someone were to pass a hand-crafted `conv_cfg` dict directly without the key, they would get 30.0 (half the safe value) instead of 100.0. This is a documentation/safety hazard, not a current-execution bug.

**Impact:** Low. Production paths always call `_get_bv_convergence_cfg`; no path skips it. The fallback is never exercised. However it contradicts Hard Rule #2 (`u_clamp = 100`) and is confusing to anyone reading the forms code in isolation.

**Recommendation:** Change `conv_cfg.get("u_clamp", 30.0)` → `conv_cfg.get("u_clamp", 100.0)` in both `forms_logc_muh.py:326` and `forms_logc.py:274`.

---

### MINOR — Non-mu NP flux uses `u_exprs[i]` not `ui[i]` — `forms_logc_muh.py:477`

For non-mu species: `ideal_grad = fd.grad(u_exprs[i]) + fd.grad(drift)`. Since `u_exprs[i] = ui[i]` (the `_u_expr` helper returns `ui_split[i]` unchanged for non-mu species, `forms_logc_muh.py:333`), this is mathematically equivalent to `fd.grad(ui[i]) + ...`. However, the equivalence is implicit and relies on `_u_expr` not doing anything for non-mu. The analogous `forms_logc.py:427` explicitly uses `u_i = ui[i]` making it clearer.

**Impact:** None in practice. UFL will see the same expression tree. Not a bug.

---

### OPEN — `add_boltzmann_counterion_residual` re-derives `J_form` by calling `fd.derivative(F_res, U)` (`boltzmann.py:386`) AFTER the main build returns — `forms_logc_muh.py:881`

This is called after `ctx.update(...)` stores the final `F_res` and `J_form`. The `add_boltzmann_counterion_residual` call modifies `ctx['F_res']` and re-derives `ctx['J_form']`. This is the correct designed pattern (skip_bikerman=True means only ideal counterions get added here; in production K+/SO4 runs all counterions are Bikerman so no ideal entries remain and this call is a no-op). However, if any downstream code caches a local reference to `ctx['J_form']` from before this call, it would hold a stale Jacobian. Static review cannot confirm no such stale reference exists in the dispatcher/orchestrator layer; recommend confirming in agent 02+ scope.

---

## Open / unverified

- **Whether `build_proton_condition_flux` correctly consumes `ideal_grad = ∇μ_H` for the water-ionization path** (`forms_logc_muh.py:483–495`): the ideal_grad passed in is `∇μ_H` (the mu-species branch), but the water bundle may internally expect `∇log(c_H)`. Static review of `water_ionization.py` is out of scope for this agent; flagged for the water-ionization agent.

- **C_SCALE absence**: the residual uses nondimensional concentrations throughout (ci are nondim, phi is nondim). C_SCALE appears in the module docstring (`c_H = C_SCALE·exp(u_H)`) but is not an explicit symbol in the UFL — the production nondim path sets `c0_model` and `concentration_scale_mol_m3` in `scaling` so the dimensional conversion is handled upstream. Confirmed that `ci` in UFL are nondim; the docstring's C_SCALE is informal notation.

- **packing_floor path for zero-a_vals with steric_boltz active**: `steric_active` is True when `steric_boltz` is non-empty even if all `a_vals_list` are zero. In that case `A_dyn = 0` (sum of zero-coefficient terms) but the `theta_inner = 1 - A_dyn - z_scale * packing_total` path is still taken. This is correct if dynamic-species steric sizes are zero (only counterion sizes matter), but relies on the counterion `packing_total` being small enough that `theta_inner > 0`. Out of scope for this agent.
