# Correctness Verification Report: Chunk 2
## Scope: dispatch.py, forms_logc.py (production log-c stack), __init__.py

**Verifier:** Claude Sonnet 4.6  
**Date:** 2026-05-02  
**Files in scope (verified):**
- `Forward/bv_solver/__init__.py` (149 lines — re-exports)
- `Forward/bv_solver/dispatch.py` (125 lines)
- `Forward/bv_solver/forms_logc.py` (531 lines)

**Reference files consulted (not verified):**
- `Forward/bv_solver/forms.py`, `config.py`, `boltzmann.py`, `observables.py`, `grid_per_voltage.py`
- `docs/bv_solver_unified_api.md`, `docs/CONTINUATION_STRATEGY_HANDOFF.md`

---

## Summary

**No critical bugs found.** Three minor issues and two questions are documented below. The core physics (NP weak form, BV rate assembly, Poisson term, BCs) are correct. The dispatcher routing is correct and backward compatible.

---

## A. Dispatcher Behavior (`dispatch.py`)

### A1. Default formulation and routing — PASS

`_get_formulation` (lines 65–71) returns `"concentration"` when `bv_convergence.formulation` is absent or any non-string value, by virtue of `bv_conv.get("formulation", "concentration")` with a `.strip().lower()`. `build_context` (line 84) routes to `build_context_logc` only when the result is exactly `"logc"`. All other values fall through to the concentration backend. Backward compatible.

### A2. `build_forms` mismatch guard — PASS

Lines 96–104: `formulation = _get_formulation(...)`, then `is_logc_ctx = bool(ctx.get("logc_transform", False))`. If `formulation == "logc"`, calls `build_forms_logc`. If `formulation != "logc"` but `is_logc_ctx` is True, raises `ValueError` with a clear message. The only remaining case (both concentration) falls through to `_build_forms_concentration`. This is correct.

### A3. `set_initial_conditions` blob no-op — PASS

Lines 112–115: when `formulation == "logc"`, `blob` is silently dropped and `set_initial_conditions_logc` is called without it. Documented in the API doc's "Common gotchas" section. No silent data corruption.

### A4. `_params_dict` input shapes — PASS with minor question

Three shapes handled:
1. `dict` → returned directly (line 53–54). Correct.
2. `hasattr(solver_params, "solver_options")` (SolverParams) → returns `solver_options` if dict, else `{}` (lines 55–57). **The SolverParams branch returns `solver_options` directly**, which is the `params` dict (i.e., `solver_params[10]` equivalent). This is correct per the API: `SolverParams.solver_options` IS the params dict.
3. `solver_params[10]` (11-tuple list) → with IndexError/TypeError/KeyError guard (lines 58–61). Correct.

**QUESTION (not a bug):** If `solver_params.solver_options` is not a dict (e.g., `None` or a non-dict type), `_params_dict` returns `{}`, and `_get_formulation` returns `"concentration"` silently. This is safe but could mask a misconfigured SolverParams. No action required unless SolverParams can have non-dict `solver_options`.

---

## B. `build_context_logc` Function Space Layout — PASS

Lines 57–58: `W = fd.MixedFunctionSpace([V_scalar for _ in range(n_species)] + [V_scalar])`. This is an `(n_species+1)`-component mixed space. In `build_forms_logc` (lines 178–183):
- `ui = fd.split(U)[:-1]` → indices 0..n-1, one per species (u_i = ln c_i). Correct.
- `phi = fd.split(U)[-1]` → index n, the potential. Correct.
- `v_list = v_tests[:-1]`, `w = v_tests[-1]`. Consistent.
- BCs: `W.sub(i)` for i in [0,n) for species; `W.sub(n)` for phi. Correct.

---

## C. `build_forms_logc` — Residual Correctness

### C1. Imports and config wiring — PASS

`bv_cfg`, `conv_cfg`, `reactions_cfg` parsed at lines 92–95. `use_reactions = reactions_cfg is not None` selects multi-reaction path. `_add_bv_reactions_scaling_to_transform` called at line 126 when `use_reactions`. The scaling dict keys used later (`D_model_vals`, `electromigration_prefactor`, `dt_model`, `phi_applied_model`, `phi0_model`, `bv_E_eq_model`, `bv_exponent_scale`, `poisson_coefficient`, `charge_rhs_prefactor`, `bv_reactions`, `c0_model_vals`) are all populated by `_add_bv_reactions_scaling_to_transform` (cross-checked against `forms.py` which uses the same call pattern identically at lines 149–155 with the same downstream key accesses). Pass.

### C2. Log-D controls — PASS

Lines 167–170: `m[i] = ln(D_model_vals[i])` (numpy scalar log), then `D[i] = fd.exp(m[i])`. At the assigned value, `exp(ln(D)) = D` identically. The `m[i]` are `R`-space Functions, so pyadjoint can differentiate through `fd.exp(m[i])` w.r.t. `m[i]` when annotation is on. Correct.

### C3. U_CLAMP application — PASS with minor note

`_U_CLAMP = float(conv_cfg.get("u_clamp", 30.0))` (line 194). `ci[i] = exp(clamp(ui[i], -30, 30))` (lines 195–198) is used in:
- The NP time derivative (`c_i - c_old`) / dt (line 285)
- The NP flux coefficient `D[i] * c_i` (lines 279, 282)
- The Poisson source `z[i] * ci[i]` (line 419)
- The steric packing sum `steric_a_funcs[j] * ci[j]` (line 249)
- `c_surf = ci` (line 293) — used only in the non-log-rate BV path

The log-rate BV path (lines 325–357) uses `ui[cat_idx]` and `ui[sp_idx]` directly (unclamped). The comment at lines 190–193 correctly states this. The clamped `ci` is used for bulk PDE terms; the unclamped `ui` is used for boundary BV evaluation when `bv_log_rate=True`. This is the intended design per Change 3 (avoids the phantom R2 sink).

**MINOR NOTE:** In the non-log-rate BV path (line 293: `c_surf = ci`), the surface concentration IS the clamped version. This is the known limitation that log-rate was designed to fix. Not a bug — it is the legacy path behavior, documented.

### C4. Nernst-Planck residual — PASS

**Time derivative (line 285):** `((c_i - c_old) / dt_const) * v * dx`  
This is the backward-Euler discretization `(c^n - c^{n-1})/dt` with `c^n = exp(clamp(u^n))` and `c^{n-1} = exp(clamp(u^{n-1}_prev))`. Correct.

**Flux term (lines 276–286):**  
`drift = em * z[i] * phi` (electromigration potential, scalar; `em` is the dimensionless electromigration prefactor, `z[i]` is a Constant charge number).  
`Jflux = D[i] * c_i * (grad(u_i) + grad(drift))`  
`grad(drift) = em * z[i] * grad(phi)` since `em` and `z[i]` are constants.  
So `Jflux = D[i] * c_i * (grad(u_i) + em * z[i] * grad(phi))`.

This matches `J = -D c (∇u + z ∇φ)` (with sign handled by the weak-form integration by parts). In the standard PNP weak form, after IBP of `∂c/∂t + ∇·J = 0` multiplied by `v`:

`∫ (∂c/∂t) v dx - ∫ J · ∇v dx + boundary terms = 0`

Code assembles `+∫ (c - c_old)/dt * v dx + ∫ Jflux · ∇v dx`, so the sign of Jflux relative to J must be negative for this to equal zero (i.e., code puts `Jflux = -J`). With `J = -D c (∇u + z ∇φ)`, we have `-J = D c (∇u + z ∇φ)` which matches the code's `Jflux`. Correct. The BV boundary flux is then subtracted from `F_res` (line 389), which closes the natural boundary term correctly.

**Steric path (lines 246–252, 279):**  
`packing = max(1 - Σ a_i c_i, floor)`, `mu_steric = ln(packing)`. Then `grad(mu_steric)` is added to the drift. This is the Bikerman steric chemical potential gradient. Numerically well-posed for the stated parameter values (`Σ a c ≈ 0.012 << 1`). Packing floor prevents `ln(0)`. Correct.

### C5. `_build_eta_clipped` — PASS

With `use_stern=False` and `conv_cfg["use_eta_in_bv"]=True` (defaults):  
`eta_raw = phi_applied_func - E_eq_const` (line 229).  
`eta_scaled = bv_exp_scale * eta_raw` (line 232).

This gives `η = (V_applied - E_eq) / V_T` when `bv_exponent_scale = 1/V_T` (nondim mode). In nondim mode the potentials are already in units of V_T, so `bv_exponent_scale = 1.0` and `eta_scaled = (V_applied_nondim - E_eq_nondim)`. The docstring comment correctly notes this is the classical form, NOT the Frumkin-corrected form (which would subtract phi_solution). Cross-checked against the writeup's classical BV model (Stern layer is OFF in the production configuration).

**Clip:** `min(max(eta_scaled, -50), 50)` when `clip_exponent=True`. Correct. The R2 unclipping threshold at `V_RHE = +0.495 V` documented in the comment (line 224) matches the memory note `project_unclipping_threshold.md`.

**Per-reaction E_eq** (lines 318–323): when `E_eq_j_val` is set and non-zero, a per-reaction `_build_eta_clipped(E_eq_j)` is called. When not set, falls back to the global `eta_clipped`. Correct for R1 (E_eq=0.68 V) and R2 (E_eq=1.78 V) having independent overpotentials.

### C6. BV rate, log-rate branch — PASS with one minor flag

**Cathodic log-rate (lines 328–339):**  
`log_cathodic = ln(k0_j) + ui[cat_idx] - alpha_j * n_e_j * eta_j`  
Plus optional concentration factors: `+ power * (ui[sp_idx] - c_ref_log)`.  
`cathodic = exp(log_cathodic)`.

Equivalence to non-log path: `cathodic_nonlog = k0_j * c_cat * exp(-alpha*n_e*eta) * prod((c_sp/c_ref)^power)`. Taking log: `= exp(ln(k0) + ln(c_cat) - alpha*n_e*eta + Σ power*(ln(c_sp) - ln(c_ref)))`. Since `ui[i]` is the UFL split of U (not the clamped `ci[i]`), and `c_cat = exp(ui[cat_idx])` at the solution, `ln(c_cat) = ui[cat_idx]`. So `log_cathodic` exactly equals the log of `cathodic_nonlog`. Correct.

**Anodic branches:**  
- `rxn["reversible"] and rxn["anodic_species"] is not None`: uses `ui[anod_idx]` (line 345). Correct.
- `rxn["reversible"] and float(rxn["c_ref_model"]) > 1e-30`: uses `ln(c_ref_model)` as the log-concentration (line 350). Correct — this treats c_ref as a constant concentration for the anodic term.
- `else`: `anodic = fd.Constant(0.0)`. This covers both: irreversible reactions AND reversible reactions where `c_ref_model <= 1e-30`. 

**MINOR FLAG (C6a):** For R2 (irreversible, `reversible=False`), the code reaches `else: anodic = fd.Constant(0.0)` at line 357. This is correct. However, the condition structure is:
```python
if rxn["reversible"] and rxn["anodic_species"] is not None:
    ...
elif rxn["reversible"] and float(rxn["c_ref_model"]) > 1e-30:
    ...
else:
    anodic = fd.Constant(0.0)
```
A reversible reaction with `c_ref_model <= 1e-30` would silently get zero anodic current (same `else` branch as irreversible). This is a potential logic ambiguity: the intent for irreversible R2 is clear (zero anodic), but a misconfigured reversible reaction with `c_ref_model=0` would silently become irreversible. This is acceptable if the config validator enforces `c_ref > 0` for reversible reactions, but config.py's `_get_bv_reactions_cfg` does not check this. 

**MINOR (no immediate impact):** The non-log-rate branch at line 375 handles `elif rxn["reversible"]:` without the `c_ref_model > 1e-30` guard — it uses `c_ref_j = Constant(rxn["c_ref_model"])` directly. A zero `c_ref_model` there would give anodic = 0 via multiplication (silently), but behave differently from the log-rate path's `1e-30` threshold. Minor inconsistency between the two paths for edge-case config.

**`fd.ln(k0_j)` adjoint differentiability (C6b):** `k0_j` is an `R`-space Function assigned the scalar value (line 309). `fd.ln(k0_j)` is UFL-differentiable w.r.t. `k0_j` when pyadjoint annotation is on. This is the correct setup for adjoint k0 inference. No bug — the `stop_annotating()` context in the forward sweep just means it is not currently exercised, but the tape would be built correctly in an annotated context.

### C7. BV flux assembly — PASS

Line 389: `F_res -= fd.Constant(float(stoi[i])) * R_j * v_list[i] * ds(electrode_marker)`

The weak form for NP with boundary flux `J·n = stoi_i * R_j` at the electrode gives, after IBP:  
`- ∫ J·n v ds = - stoi_i * R_j * v * ds`

Since we assembled `F_res += ∫ Jflux · ∇v dx` (the volume term) and the natural boundary contribution from IBP is `- ∫ J·n v ds`, the total NP residual becomes:

`∫ (c-c_old)/dt * v dx + ∫ Jflux·∇v dx - ∫ J·n * v * ds(electrode) = 0`

where `J·n = stoi_i * R_j` at the electrode (positive `R_j` means cathodic flux consuming species with negative `stoi`). The code does `F_res -= stoi[i] * R_j * v * ds`, which subtracts the natural BC term. Setting `F_res = 0` recovers the weak equation correctly. Sign convention matches forms.py identically (line 385 of forms.py). Pass.

Cross-check with the writeup: cathodic current consumes O₂ (stoi R1: `[-1, +1, ...]` for [O₂, H₂O₂, ...]). R1 cathodic > 0 (reduction). `F_res -= (-1) * R_1 * v_{O₂} * ds = +R_1 * v_{O₂} * ds`. This adds a positive source to the O₂ residual at the electrode, meaning O₂ is consumed there — consistent with `J_{O₂}·n < 0` at the electrode (flux into boundary = consumption). Correct.

### C8. Poisson term — PASS

Lines 417–419:
```python
F_res += eps_coeff * fd.dot(fd.grad(phi), fd.grad(w)) * dx
F_res -= charge_rhs * sum(z[i] * ci[i] * w for i in range(n)) * dx
```

Strong form: `-ε∇²φ = F Σ z_i c_i`. After multiplying by `w` and IBP: `ε ∫∇φ·∇w dx - ε ∫∇φ·n w ds - ∫(F Σ z_i c_i) w dx = 0`. The electrode BC (Dirichlet `φ = phi_applied`) removes the boundary integral there; at the bulk (Dirichlet `φ = 0`) likewise. Interior: `ε ∫∇φ·∇w dx = ∫ charge_rhs * Σ z_i c_i * w dx`, giving `F_res = eps_coeff * ∫∇φ·∇w dx - charge_rhs * Σ z_i c_i * w dx = 0`. Correct. The `charge_rhs_prefactor` is the dimensionless `F·c_ref·L²/ε·V_T` prefactor (verified by cross-reference with nondim conventions; checked by another chunk).

### C9. Stern layer (inactive branch) — PASS

`use_stern = False` in production (no `stern_capacitance_f_m2` set). At line 440–444: when `use_stern=False`, adds `bc_phi_electrode = DirichletBC(W.sub(n), phi_applied_func, electrode_marker)`. Robin coupling at line 422–424 is guarded by `if use_stern:` and skipped. The Dirichlet BC for phi at the electrode is correctly applied. Pass.

### C10. Boundary conditions — PASS

- `bc_phi_ground`: `W.sub(n)`, value `0.0`, marker `ground_marker=4` (top boundary). Correct.
- `bc_ui[i]`: `W.sub(i)`, value `ln(max(c0_model[i], 1e-20))`, marker `concentration_marker=4` (top boundary). Same marker as ground for phi. Correct.
- `bc_phi_electrode`: `W.sub(n)`, value `phi_applied_func`, marker `electrode_marker=3` (bottom boundary). Correct.

**`bcs` assembly (line 444):** `bcs = bc_ui + [bc_phi_electrode, bc_phi_ground]`. Two BCs on `W.sub(n)` (potential), one at marker 3 (electrode) and one at marker 4 (ground). Firedrake handles multiple DirichletBCs on the same sub-space on different markers without conflict — each enforces on its marker's DOFs independently. Correct.

Same-marker dual BC: `ground_marker=4` and `concentration_marker=4` both apply to the top boundary. `bc_phi_ground` targets `W.sub(n)` and `bc_ui[i]` targets `W.sub(i)`. These are different sub-spaces of W, so there is no conflict. Firedrake's DirichletBC mechanism applies each BC to its designated sub-space component's DOFs. Correct.

**Log floor check:** c0 values `[1.0, 1e-4, 0.2]` all exceed `_C_FLOOR=1e-20`. `ln(1.0) = 0`, `ln(1e-4) ≈ -9.21`, `ln(0.2) ≈ -1.61`. None triggers the floor. Pass.

### C11. J_form derivation order vs Boltzmann mutation — PASS

Line 446: `J_form = fd.derivative(F_res, U)` — derived BEFORE calling `add_boltzmann_counterion_residual`.

Line 475: `add_boltzmann_counterion_residual(ctx, params)` — mutates `ctx["F_res"]` and overwrites `ctx["J_form"]` with `fd.derivative(updated_F_res, U)` (boltzmann.py lines 125–126).

Between lines 446 and 475, `J_form` is put into `ctx` at line 450. `add_boltzmann_counterion_residual` then overwrites it. No consumer reads `ctx["J_form"]` between lines 450 and 475 — the ctx.update happens as one call, and the Boltzmann call happens immediately after. The final `ctx["J_form"]` (set by boltzmann.py) is the correct Jacobian including the Boltzmann contribution. Pass.

**When Boltzmann counterions are absent** (no-op path, `boltzmann.py:77`): `add_boltzmann_counterion_residual` returns 0 without touching `ctx`. The `J_form` set at line 450 remains the final Jacobian. Correct.

### C12. ctx.update keys — PASS with one note

Keys stored at lines 448–473: `F_res`, `J_form`, `bcs`, `logD_funcs`, `D_consts`, `z_consts`, `dt_const`, `phi_applied_func`, `phi0_func`, `bv_settings`, `bv_convergence`, `bv_rate_exprs`, `bv_k0_funcs`, `bv_alpha_funcs`, `steric_a_funcs`, `nondim`, `use_stern`, `ci_exprs`, `_diag_bv_exp_scale`, `_diag_exponent_clip`, `_diag_eps_c`, `logc_transform=True`.

**`bv_rate_exprs` count:** For the production 2-reaction config (R1, R2), the loop `for j, rxn in enumerate(rxns_scaled)` iterates twice. `bv_rate_exprs.append(R_j)` appends once per iteration. Final length = 2. `observables.py:47–52` requires `len(bv_rate_exprs) >= 2` for `peroxide_current` mode, and indexes `[0]` (R1) and `[1]` (R2). The ordering (R1 first, R2 second) matches the ordering of `reactions` in the config, which matches the `make_bv_solver_params` factory. Pass.

**NOTE on `ci_exprs`:** Line 466 stores `"ci_exprs": ci` where `ci` is the list of clamped expressions `exp(clamp(ui[i]))`. Observable code that reads `ci_exprs` gets the clamped concentrations. For the BV rate expressions in `bv_rate_exprs` (log-rate path), these are assembled using `ui[i]` directly (not `ci_exprs`). No inconsistency — `ci_exprs` is for post-processing observables, not for the rate expressions themselves.

**Missing key `_diag_E_eq_per_reaction`:** forms.py (line 388) stores `ctx["_diag_E_eq_per_reaction"]`, `_diag_alpha_per_reaction`, `_diag_n_e_per_reaction` in the multi-reaction path. `forms_logc.py` does NOT store these diagnostic keys. If any downstream validation code reads these keys from a logc context, it will get a `KeyError`. This is a minor gap — not a correctness issue for the forward solve, but could break diagnostic/validation routines.

**MINOR (C12a):** `forms_logc.py` omits `_diag_E_eq_per_reaction`, `_diag_alpha_per_reaction`, `_diag_n_e_per_reaction` that `forms.py` stores. Severity: minor. Any downstream validator reading these keys from a logc ctx will KeyError.

---

## D. `set_initial_conditions_logc` — PASS

Lines 502–504: `U_prev.sub(i).assign(Constant(ln(max(c0_model[i], 1e-20))))` for each species. For the production preset (`c0 = [1.0, 1e-4, 0.2]`), values are `0.0`, `-9.21`, `-1.61`. All finite. Pass.

Lines 507–511: Linear phi profile.
- `ndim = mesh.geometric_dimension()`. For the graded rectangle mesh, `ndim = 2`.
- `spatial_var = coords[1]` (y-coordinate). At y=0 (electrode, marker=3): `phi = phi_applied_model * (1 - 0) = phi_applied_model`. At y=1 (bulk, marker=4): `phi = phi_applied_model * (1 - 1) = 0`. Matches the BCs (`phi = phi_applied` at electrode, `phi = 0` at bulk). Correct.

Line 512: `ctx["U"].assign(U_prev)`. Sets both `U` and `U_prev` to the IC. At the first SS time step, `(c_n - c_{n-1})/dt = (exp(u) - exp(u_prev))/dt = 0` since `u = u_prev`. The time-derivative term drops out, making the first SS step equivalent to a steady-state solve seeded at the IC. This is the correct behavior for steady-state initialization. Pass.

---

## E. Adjoint Controls — PASS

Lines 302–313: `bv_k0_funcs` and `bv_alpha_funcs` are `R`-space Functions assigned scalar values. They enter the BV residual as:
- `fd.ln(k0_j)` — UFL expression, differentiable w.r.t. `k0_j`
- `alpha_j * n_e_j * eta_j` — linear in `alpha_j`, differentiable

When pyadjoint annotation is on (the inverse pipeline), the tape records operations on these Functions. The `adj.stop_annotating()` context in the forward sweep means the tape is NOT built during grid evaluation, but the structure is correct for annotated solves. No structural bug. Pass.

**`z_consts` mutability:** `z = [fd.Constant(float(z_vals[i])) for i in range(n)]` (line 172). `fd.Constant` objects support `.assign()` calls. `grid_per_voltage.py:_set_z_factor` calls `ctx["z_consts"][i].assign(z_nominal[i] * z_val)`. This mutates the Constant in place, which changes the UFL expression that `F_res` and `J_form` evaluate with — exactly the intended behavior for z-continuation. Pass.

---

## F. `build_bv_observable_form_logc` — PASS

Lines 519–531: Delegates directly to `observables._build_bv_observable_form(ctx, ...)`. No additional scaling, no sign flip. `observables._build_bv_observable_form` reads `ctx["bv_rate_exprs"]` (already assembled in log-c-aware form) and `ctx["bv_settings"]["electrode_marker"]`. No double-scaling. The BV rate expressions in `ctx["bv_rate_exprs"]` already use the correct sign convention (cathodic - anodic), so the observable's `scale` parameter is the only external scaling. Pass.

---

## Cross-Reference Checks

### Boltzmann mutation contract — PASS

`forms_logc.py` does not assume `F_res` is final after the ctx.update (lines 448–473). `build_forms_logc` immediately calls `add_boltzmann_counterion_residual` at line 475, which may mutate `F_res` and `J_form`. The function then returns `ctx` (line 476). The caller (orchestrator or dispatcher) receives the fully-assembled ctx. No consumer reads `J_form` from ctx before `add_boltzmann` runs.

### bv_rate_exprs ordering for observables — PASS

`bv_rate_exprs[0]` = R1 (O₂ → H₂O₂), `bv_rate_exprs[1]` = R2 (H₂O₂ → H₂O). `observables.py:52`: `peroxide_current = scale * (bv_rate_exprs[0] - bv_rate_exprs[1]) * ds`. R1 produces H₂O₂ (positive for peroxide yield); R2 consumes H₂O₂ (negative for peroxide yield). The observable `R0 - R1` gives net peroxide production. Matches the API doc's description. Pass.

### grid_per_voltage.py call order — PASS

`_build_for_voltage` calls `build_context`, then `build_forms`, then `set_initial_conditions` (grid_per_voltage.py lines 221–224). This matches the required sequence. Pass.

### z_consts and boltzmann_z_scale mutability — PASS

`ctx["z_consts"][i]` are `fd.Constant` objects (line 172 of forms_logc.py). `ctx["boltzmann_z_scale"]` is an R-space Function (boltzmann.py line 102). Both support `.assign()`. `_set_z_factor` in grid_per_voltage.py assigns to both. Firedrake Constants/Functions assigned in this way update the coefficient value seen by the assembled form at the next solver call. Correct.

---

## Issues Summary

| ID | Severity | Location | Description | Fix |
|----|----------|----------|-------------|-----|
| C6a | minor | `forms_logc.py:349–357` | Reversible reaction with `c_ref_model <= 1e-30` silently gets zero anodic current (same `else` as irreversible). Non-log-rate path at line 375 doesn't have this guard. Minor inconsistency; safe in production since R2 has `reversible=False`. | Add explicit `not rxn["reversible"]` check to the `else` branch, or add config validation requiring `c_ref > 0` for reversible reactions. |
| C12a | minor | `forms_logc.py:448–473` | Missing diagnostic keys `_diag_E_eq_per_reaction`, `_diag_alpha_per_reaction`, `_diag_n_e_per_reaction` (present in forms.py for the equivalent code path). | Add these three keys to the `ctx.update` dict in `build_forms_logc`. |
| A4-Q | question | `dispatch.py:55–57` | If `SolverParams.solver_options` is `None` or non-dict, `_params_dict` returns `{}` silently → formulation defaults to `"concentration"`. | Consider adding a warning or assertion if `solver_options` is not a dict when `hasattr(solver_params, "solver_options")` is True. |
| C6b | question | `forms_logc.py:365` | Non-log-rate path: `cathodic_conc_factors[factor]["power"]` is used as a Python int (line 365: `power = factor["power"]`) passed directly to `** power` (line 367). In the log-rate path, it's cast to `fd.Constant(float(...))` (line 334). The non-log-rate `(c_surf[sp_idx] / c_ref_f) ** power` with integer `power` works in UFL (integer exponentiation), but adjoint differentiability through integer `**` may differ from through `fd.Constant`. For the production stack with `bv_log_rate=True`, this code path is not taken. | Cast `power` to `fd.Constant(float(factor["power"]))` in the non-log-rate path for consistency with the log-rate path and adjoint correctness if this path is ever used with pyadjoint. |

---

## Conclusion

**No critical bugs.** The dispatcher correctly routes formulation selection; the logc backend correctly implements the NP weak form with backward-Euler time discretization, log-concentration substitution, BV boundary fluxes with log-rate and non-log-rate paths, Poisson source, and Dirichlet BCs. The Boltzmann mutation contract is sound. The IC sets U = U_prev, making the first SS step time-derivative-free. Two minor diagnostic key gaps and two minor cross-path inconsistencies are noted but do not affect the production logc+Boltzmann+log-rate stack.
