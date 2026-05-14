# Correctness Audit L3 — Pass B (Sonnet)
Date: 2026-05-13
Files: forms_logc_muh.py, boltzmann.py, picard_ic.py

---

## FINDINGS

---

### FINDING 1 — MEDIUM
**z_scale_shared multiplies packing_total in theta_inner (Strategy-B Phase-1 coupling)**

Location: `forms_logc_muh.py:447-448`
```python
theta_inner = (
    fd.Constant(1.0) - A_dyn - z_scale_shared * packing_total
)
```
`z_scale_shared` is 1.0 in production (standalone / C+D path), so this is correct for production runs. However if Strategy B is ever re-enabled and `z_scale=0` is set in Phase 1 to zero-out the counterion charge, `packing_total` (which is `sum(a_k * c_steric_k)` — each `c_steric_k` itself contains `free_dyn / denom`) is also zeroed from theta. This means the NP Bikerman steric potential `mu_steric = -ln(theta)` loses the analytic counterion volume fraction during Phase 1, making the dynamic species' steric chemical potential inconsistent with the Boltzmann closure even at zero ionic strength. Since C+D is the only live strategy, the practical impact is zero — but the coupling is a latent bug that would fire if `z_scale` is ever ramped during a Bikerman steric run.

Trigger: Only if Strategy B is re-enabled with Bikerman counterions AND `z_scale` is set to 0 during Phase 1.
Evidence: Line 448 vs the physical closure derivation where `theta = 1 - A_dyn - sum(a_k c_k)` and `c_k` is independent of the Poisson z-ramp.

---

### FINDING 2 — INFORMATIONAL (confirmed correct)
**Shared free_dyn for multi-counterion Bikerman: CORRECT**

Location: `boltzmann.py:204-206`
```python
A_dyn_local = sum(a_dyn_funcs[i] * ci[i] for i in range(len(ci)))
free_dyn_floor = fd.Constant(1e-10)
free_dyn = fd.max_value(fd.Constant(1.0) - A_dyn_local, free_dyn_floor)
```
The single `free_dyn` UFL expression is built once and then embedded in every `c_steric_k = c_const * q_k * free_dyn / denom` at line 254. All bundles in `steric_boltz` reference the same UFL node. The shared-theta closure is correctly implemented: Cs+ and SO4²⁻ both use exactly the same `free_dyn` and the same `denom`. No leakage of TestFunction/TrialFunction — `ci` is a list of `fd.exp(...)` UFL expressions derived from `fd.split(U)` (not `fd.TestFunctions` or `fd.TrialFunctions`). Verdict: CORRECT.

---

### FINDING 3 — INFORMATIONAL (confirmed correct)
**Sign of the Stern term in F_res: CORRECT for cathodic polarization**

Location: `forms_logc_muh.py:668`
```python
F_res -= stern_coeff * (phi_applied_func - phi) * w * ds(electrode_marker)
```
The Poisson residual accumulates as:
- `F_res += eps * grad(phi).grad(w) dx`  (diffuse layer)
- `F_res -= charge_rhs * sum(z_i * c_i) * w dx`  (charge source)
- `F_res -= stern_coeff * (phi_applied - phi) * w * ds`  (Stern Robin)

The Stern Robin BC enforces `eps * dphi/dn = C_S * (phi_applied - phi)` at the electrode. Cathodically, `phi_applied < 0` and `phi(electrode) < 0` with `phi_applied < phi(electrode)`, so `(phi_applied - phi) < 0`, giving the correct sign for an inward flux. The `-=` with the positive `(phi_applied - phi)` sign is consistent with the Robin condition written as a Neumann-type boundary integral subtracted from the bulk Poisson form. Verdict: CORRECT.

---

### FINDING 4 — INFORMATIONAL (confirmed correct)
**eta_raw uses phi_applied_func (V_RHE proxy) not phi (local FE potential): CORRECT**

Location: `forms_logc_muh.py:393-398`
```python
if use_stern:
    eta_raw = phi_applied_func - phi - E_eq_const
elif conv_cfg["use_eta_in_bv"]:
    eta_raw = phi_applied_func - E_eq_const
else:
    eta_raw = phi - E_eq_const
```
In the Stern path (production), `eta_raw = phi_applied_func - phi - E_eq`. Here `phi_applied_func` is the externally applied potential (V_RHE in nondim) and `phi` is the local FE potential at the OHP. The combination `phi_applied_func - phi` is the Stern voltage drop `psi_S`, which is the correct overpotential argument for the BV rate. The `no-Stern + use_eta_in_bv` path uses `phi_applied_func - E_eq` (a constant per voltage step, no local phi dependence) which is correct for the linearized limit. The `else` path (legacy) uses `phi - E_eq` which gives a spatially varying eta — only appropriate for the non-Stern formulation where `phi` at the electrode IS the applied potential. Verdict: CORRECT for all three branches.

---

### FINDING 5 — INFORMATIONAL (confirmed correct)
**Boltzmann counterion injection into Poisson: CORRECT sign**

Location: `boltzmann.py:36-41` (module docstring) and `forms_logc_muh.py:652-660`
```python
# boltzmann.py module docstring:
#   F_poisson = eps*grad(phi).grad(w)*dx
#             - charge_rhs * sum_i z_i*c_i*w*dx
#             - charge_rhs * sum_k z_k*c_steric_k*w*dx

# forms_logc_muh.py:656-659:
charge_density_total = sum(b.charge_density for b in steric_boltz)
F_res -= (
    z_scale_shared * charge_rhs * charge_density_total * w * dx
)
```
`charge_density = z_k * c_steric_k` (line 258 of boltzmann.py). For Cs+ (z=+1), this is positive; for SO4²⁻ (z=-2), negative. The `-=` in the Poisson residual puts the counterion source on the same footing as the dynamic species' `- charge_rhs * sum(z_i * c_i) * w * dx`. Electroneutrality: Cs+ adds +1 * c_Cs+ and SO4²⁻ adds -2 * c_SO4 to the Poisson source, which is the correct sign convention. Verdict: CORRECT.

---

### FINDING 6 — INFORMATIONAL (confirmed correct)
**IC seed for psi_S vs psi_D: Stern voltage drop correctly handled**

Location: `forms_logc_muh.py:935-963` (set_initial_conditions_logc_muh), `forms_logc_muh.py:1219-1237` (_try_debye_boltzmann_ic_muh)

Linear IC: `phi_surface = solve_stern_split(...)` returns `phi_applied - psi_S`. The IC then seeds `phi_init(y) = phi_surface * (1 - y)` so phi at the electrode starts at `phi_applied - psi_S`, correctly omitting the Stern voltage drop from the FE domain (Stern is a boundary condition, not an interior profile). Debye-Boltzmann IC: `psi_D` from Picard is used for the diffuse-layer profile; Stern split is applied consistently via `use_stern_at_ic`. The `mu_H_init = u_H_init + em*z_H*phi_init` assignment with `em*z_H=1` yields the analytic psi-cancellation documented in the module docstring. Verdict: CORRECT.

---

### FINDING 7 — INFORMATIONAL (confirmed correct)
**_build_eta_clipped: clip semantics and fd.min_value vs python min**

Location: `forms_logc_muh.py:388-403`
```python
eta_scaled = bv_exp_scale * eta_raw
if conv_cfg["clip_exponent"]:
    clip_val = fd.Constant(float(conv_cfg["exponent_clip"]))
    return fd.min_value(fd.max_value(eta_scaled, -clip_val), clip_val)
return eta_scaled
```
`fd.min_value` / `fd.max_value` are UFL elementwise ops — correct for FE expressions. Python `min` would collapse to a scalar and break the UFL graph. The clip is applied to `eta_scaled = bv_exp_scale * eta_raw` BEFORE multiplication by `alpha * n_e` (that multiplication happens at the call sites: `-alpha_j * n_e_j * eta_j`). This is the documented convention in `clipping_conventions.md`. In the clipped plateau, `fd.min_value` / `fd.max_value` have zero UFL derivative — so the Jacobian contribution from the BV terms is zero when `|eta_scaled| >= clip_val`. This is the correct behavior for the hard-clipped Newton: the plateau is treated as a nonlinear constant term during the Newton step. No issue. Verdict: CORRECT.

---

### FINDING 8 — INFORMATIONAL (confirmed correct)
**Bikerman exponent overflow for SO4²⁻ at large negative phi: GUARDED**

Location: `boltzmann.py:220-228`
```python
phi_clamp_k = float(entry["phi_clamp"])
phi_clamped_k = fd.min_value(
    fd.max_value(phi, fd.Constant(-phi_clamp_k)),
    fd.Constant(phi_clamp_k),
)
q_k = fd.exp(-z_k_const * phi_clamped_k)
```
For SO4²⁻: z=-2, phi_clamped is bounded. The exponent inside `fd.exp` is `-(-2) * phi_clamped = 2 * phi_clamped`. With `phi_clamp_k` from the entry config (not read here; must verify config value is reasonable), the maximum exponent is `2 * phi_clamp_k`. If `phi_clamp_k = 40` (typical), max exponent = 80 → `exp(80) ~ 5.5e34`, which is representable in float64. UFL pushes this to the mesh quadrature; no overflow. The ideal-path fallback (`add_boltzmann_counterion_residual`, lines 374-381) also applies `phi_clamp_val` from config. The clip is config-driven — correctness depends on the caller providing a finite `phi_clamp` in the counterion entry. If `phi_clamp` is absent or zero, `fd.min_value(phi, 0)` collapses the ion. This is a config invariant, not a code bug. Verdict: GUARDED (config-dependent).

---

### FINDING 9 — INFORMATIONAL (confirmed correct)
**J_form derivative ordering: F_res fully assembled before derivative**

Location: `forms_logc_muh.py:839`
```python
J_form = fd.derivative(F_res, U)
```
This appears after all F_res mutations: NP terms (lines 464-505), BV terms (lines 519-636), Poisson (lines 644-660), Stern Robin (lines 666-668), cation hydrolysis (lines 676-803), BCs (lines 821-837). The `add_boltzmann_counterion_residual` call at line 881 comes AFTER this derivative is stored, but that function re-derives `J_form` internally (`ctx["J_form"] = fd.derivative(F_res, U)` — see boltzmann.py line 386). So the final `J_form` in ctx is consistent with the final `F_res`. Verdict: CORRECT.

---

### FINDING 10 — INFORMATIONAL (confirmed correct)
**Picard fallback path: U_prev correctly initialized**

Location: `forms_logc_muh.py:1028-1039`
```python
ok, reason, picard_iters = _try_debye_boltzmann_ic_muh(...)
if ok:
    ...
    return
ctx["initializer_fallback"] = True
...
set_initial_conditions_logc_muh(ctx, solver_params)
```
On Picard failure, `set_initial_conditions_logc_muh` is called. That function writes to `U_prev.sub(i)` for all i in range(n+1) (lines 981-993) and then does `ctx["U"].assign(U_prev)` (line 994). `U_prev` at the time of the fallback call may have been partially written by `_try_debye_boltzmann_ic_muh` — but `set_initial_conditions_logc_muh` is a full overwrite (every sub-component assigned). No stale state from a failed Picard attempt survives. Verdict: CORRECT.

---

### FINDING 11 — INFORMATIONAL (confirmed correct)
**Picard singular Jacobian guard**

Location: `picard_ic.py:1416-1432`
```python
if N == 2:
    singular = (
        not math.isfinite(det) or abs(det) < 1e-300
    )
else:
    singular = any(not math.isfinite(v) for v in R_solve)
if singular:
    return (False, f"singular_jacobian_iter_{k}_det=...", ...)
```
For N=2, determinant check is explicit. For N>2, the fallback is checking R_solve finiteness (which catches NaN/Inf from a singular solve but not near-singular with small det). This is a gap for the general N-reaction case — a nearly-singular system with very large-but-finite R_solve would not be caught. In practice, the production stack is always N=2 (parallel 2e/4e), so this branch is never exercised. Documented as a gap for future N>2 topology support.

---

### FINDING 12 — LOW
**Picard oscillation detection: absent**

The Picard outer loops (both legacy 2x2 and `picard_outer_loop_general`) use a fixed relaxation factor `omega=0.5` and check convergence via a relative delta on R values. There is no oscillation detection (e.g., alternating sign of `R - R_old` across iterations, or monitoring of delta trend). If the loop oscillates with a fixed-amplitude cycle (delta never decreasing past some threshold), it exits only when `max_iters` is hit with the `picard_max_iters_delta=...` failure reason. The fallback to linear-phi IC is then triggered. This is a silent efficiency loss — the fallback may succeed where a more adaptive relaxation (e.g., line search or adaptive omega) would have converged the Picard. No incorrect results; just potential unnecessary fallbacks at difficult operating points.

---

### FINDING 13 — INFORMATIONAL (confirmed correct)
**mu_h_index resolution: called per build, not cached on global**

Location: `forms_logc_muh.py:290` (build_forms_logc_muh), `forms_logc_muh.py:925` (set_initial_conditions_logc_muh), `forms_logc_muh.py:1120` (_try_debye_boltzmann_ic_muh)
```python
mu_h_idx = _resolve_mu_h_index(list(z_vals), roles=species_roles)
```
Called independently in all three entry points. This is correct: each call is self-contained with its own `z_vals` slice. There is no shared global mutable state — each call creates a fresh local variable. With `species_roles` wired through `_get_species_roles(params, n)` consistently, all three calls resolve to the same index for the same params. No caching hazard. Verdict: CORRECT.

---

### FINDING 14 — INFORMATIONAL (confirmed correct)
**Constants vs Functions for k0, alpha, n_e**

- `k0_j`: `fd.Function(R_space)` — mutable via `.assign(...)`, correct for k0 ladders.
- `alpha_j`: `fd.Function(R_space)` — mutable, correct.
- `n_e_j`: `fd.Constant(float(rxn["n_electrons"]))` — immutable.

`n_e_j` as `fd.Constant` is the right choice: electron stoichiometry is fixed by physics, not a continuation parameter. `fd.Constant` is differentiable and enters the Jacobian cleanly. `k0` and `alpha` as `fd.Function` allow `.assign()` for ladders without rebuilding the form. Verdict: CORRECT.

---

### FINDING 15 — INFORMATIONAL (confirmed correct)
**Stern bisection bracket failure: fallback to linear-Debye analytical solution**

Location: `picard_ic.py:259-270`
```python
if f_lo * f_hi > 0.0:
    # Linear-Debye limit fallback
    denom = eps_nondim + stern_coeff_nondim * lambda_D
    if abs(denom) < 1e-30:
        return 0.0, full_drop, phi_applied_model
    psi_D_lin = stern_coeff_nondim * full_drop * lambda_D / denom
    ...
```
When `[0, full_drop]` fails to bracket (happens near `psi_D ~ 0` / high-symmetry points), the code falls through to the linear-Debye analytical solution. This is physically correct in the regime and documented. The edge case where `denom` is near-zero gives `(0, full_drop, phi_applied_model)` — assigning all voltage drop to the diffuse layer. This is conservative (never crashes) and only fires at degenerate `eps ~ -stern * lambda_D` which cannot happen with physical parameters (both eps and stern_coeff are positive). Verdict: CORRECT.

---

## SUMMARY TABLE

| # | Severity | Component | Issue |
|---|----------|-----------|-------|
| 1 | MEDIUM | forms_logc_muh.py:448 | z_scale_shared multiplies packing_total — latent Strategy-B + Bikerman bug |
| 2 | INFO/PASS | boltzmann.py:204-254 | Shared free_dyn across all bikerman bundles — CORRECT |
| 3 | INFO/PASS | forms_logc_muh.py:668 | Stern term sign — CORRECT for cathodic |
| 4 | INFO/PASS | forms_logc_muh.py:393-398 | eta_raw uses phi_applied_func not phi — CORRECT |
| 5 | INFO/PASS | boltzmann.py + forms_logc_muh.py:652-660 | Counterion Poisson injection sign — CORRECT |
| 6 | INFO/PASS | forms_logc_muh.py:935-963, 1219-1237 | IC psi_S/psi_D Stern split — CORRECT |
| 7 | INFO/PASS | forms_logc_muh.py:388-403 | fd.min_value clip semantics + Jacobian-zero plateau — CORRECT |
| 8 | INFO/CONFIG | boltzmann.py:220-228 | SO4²⁻ exp overflow guarded by phi_clamp — config-dependent |
| 9 | INFO/PASS | forms_logc_muh.py:839 + boltzmann.py:386 | J_form derived after full F_res — CORRECT |
| 10 | INFO/PASS | forms_logc_muh.py:1028-1039 | Picard fallback full-overwrite of U_prev — CORRECT |
| 11 | INFO/PASS | picard_ic.py:1416-1432 | N=2 singular guard — CORRECT; N>2 gap (not in production) |
| 12 | LOW | picard_ic.py both loops | No oscillation detection — silent efficiency loss only |
| 13 | INFO/PASS | forms_logc_muh.py:290,925,1120 | mu_h_idx resolved per-call, no global cache hazard — CORRECT |
| 14 | INFO/PASS | forms_logc_muh.py:522-537 | k0/alpha as Function, n_e as Constant — CORRECT |
| 15 | INFO/PASS | picard_ic.py:259-270 | Stern bisection bracket fallback — CORRECT |

---

## VERDICT

**PASS with one latent defect.**

The production stack (C+D strategy, z_scale=1.0, parallel 2e/4e, N=2) has no active correctness bugs in the audited scope. All five new questions from this pass resolve as CORRECT.

**One latent MEDIUM bug**: `z_scale_shared * packing_total` in `theta_inner` (forms_logc_muh.py:448, mirrored in forms_logc.py:397) means that if Strategy B is ever re-enabled with Bikerman steric counterions and `z_scale=0` is set during Phase 1, the analytic counterion packing fraction is incorrectly excluded from `theta`, corrupting the dynamic species' steric chemical potential. Mitigation: document that z_scale must not be ramped when Bikerman steric is active, or decouple the Poisson z-scale from the steric packing term.

**One LOW gap**: No oscillation detection in the Picard outer loop — only manifests as unnecessary fallbacks to linear-phi IC at difficult operating points.

All other findings are confirmations of correct behavior.
