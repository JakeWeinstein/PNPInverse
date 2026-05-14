# Correctness Audit — Chunk B
**Reviewer:** claude-sonnet-4-6  
**Date:** 2026-05-13  
**Files:** `Forward/bv_solver/forms_logc_muh.py`, `boltzmann.py`, `picard_ic.py`

---

## Q1 — _build_eta_clipped: Jacobian killed by min/max clip
**SEVERITY: LOW / KNOWN DESIGN TRADE-OFF**  
**Location:** `forms_logc_muh.py:388-403`

The clip is `fd.min_value(fd.max_value(eta_scaled, -clip_val), clip_val)`. UFL computes subgradients of `min_value`/`max_value` via piecewise-constant indicators: the derivative is 1 inside the interval and 0 outside (at the plateau). This means dF/dU w.r.t. phi or mu_H is **exactly zero** at any quadrature point where the clipped expression is on the plateau.

**Trigger:** Cathodic extreme voltages where `eta_scaled < -clip_val` (= -100). At production exponent_clip=100, R2e unclips at V_RHE > -0.79 V, so the production grid is fully unclipped — Jacobian is correct everywhere in the production V-sweep. Risk is nonzero only for exploratory runs at V_RHE << -0.8 V or if exponent_clip is accidentally lowered.

**Evidence:** `forms_logc_muh.py:400-402`. No guard prevents evaluation at clipped voltages; SNES would see a zero Jacobian block in the phi/BV coupling rows when clipped. This manifests as near-singular J (SNES iterating with tiny steps), not divergence. Flagged as KNOWN in `docs/solver/clipping_conventions.md` (clip=100 chosen precisely to keep production grid unclipped). No immediate code change needed unless the V grid is extended below -0.8 V.

---

## Q2 — free_dyn floor: Bikerman counterion magnitude at floor
**SEVERITY: MEDIUM / LATENT PHYSICAL ERROR**  
**Location:** `boltzmann.py:204-206, 254`

```python
free_dyn = fd.max_value(fd.Constant(1.0) - A_dyn_local, free_dyn_floor)  # floor=1e-10
c_steric_k = p["c_const"] * p["q"] * free_dyn / denom
```

**Dimensional analysis:** When `A_dyn → 1` (dynamic species pack the full layer), `free_dyn = 1e-10`. The counterion concentration becomes:

```
c_k = c_b_k * exp(-z_k * phi_clamped) * 1e-10 / (theta_b + sum a_k' c_b_k' exp(-z_k' phi))
```

The denominator contains `theta_b` (a bulk constant, O(0.1)) and the exp term. Because `free_dyn` in the numerator is 1e-10 while the denominator is O(theta_b) ~ 0.1, the resulting `c_k ~ 1e-9 * exp(...)`. For SO4²⁻ (z=-2) at phi_clamped at the clamp limit (phi_clamp=50), `exp(100) ~ 2.7e43` — so `c_k ~ 2.7e34` (dimensionless). This is wildly non-physical but only activates when A_dyn simultaneously saturates AND phi is near the clamp limit.

**More important:** The floor is meant to prevent div-by-zero in the `free_dyn/denom` ratio, but the denominator `denom` (line 241) is always `>= theta_b > 0` by construction (validated at line 159), so the floor on free_dyn is not needed to prevent div-by-zero in `c_steric_k` — it exists to prevent negative `packing` later in `forms_logc_muh.py:452`. The floor is in the wrong place (on `free_dyn` entering `c_steric_k`) rather than only on `theta_inner` (the packing for the steric mu). As a result, when A_dyn saturates:
- `c_steric_k` is floored to 1e-10/denom (small but positive) rather than zero
- `packing_contribution = a_k * c_steric_k` is also small — so `theta_inner` is not numerically corrected
- The Poisson `charge_density = z_k * c_steric_k` contributes a tiny, discontinuous jump in the Poisson charge term

The practical risk: in the current production config, A_dyn_bulk = sum(a_i * c0_i) uses A_DEFAULT=0.01 for O2/H2O2/H+ (per CLAUDE.md Hard Rule #7), so the bulk packing is tiny. Saturation only occurs at high phi near the electrode. **At phi_clamped=50, the product gives c_k ~ 1e-9 * exp(100) — this is a numerical artifact that contributes spurious Poisson charge near the OHP under extreme cathodic polarization.**

**Trigger:** A_dyn_local near 1 AND |phi| near phi_clamp=50. In practice this requires unphysically large a_nondim for dynamic species, which the current A_DEFAULT=0.01 prevents from happening in bulk. Risk rises when physical-a variants (Hard Rule #7 bridge runs) assign a_h ~ r_H3O+/c_ref to H+ — in that case A_dyn_local at the OHP could approach saturation.

---

## Q3 — Bikerman counterion exponent overflow (SO4²⁻ cathodic)
**SEVERITY: LOW — CLAMP EXISTS BUT ASYMMETRIC RISK**  
**Location:** `boltzmann.py:224-228`, `config.py:412`

```python
phi_clamped_k = fd.min_value(
    fd.max_value(phi, fd.Constant(-phi_clamp_k)),
    fd.Constant(phi_clamp_k),
)
q_k = fd.exp(-z_k_const * phi_clamped_k)
```

For SO4²⁻: z_k = -2, phi_clamp = 50.0. At the cathodic clamp limit `phi_clamped_k = -50`:
```
q_k = exp(-(-2) * (-50)) = exp(-100) ~ 3.7e-44
```
This is fine (underflows gracefully).

At the anodic clamp limit `phi_clamped_k = +50`:
```
q_k = exp(-(-2) * 50) = exp(100) ~ 2.7e43
```
This is the problematic case: SO4²⁻ under **anodic** polarization (phi >> 0) piles up. exp(100) does not overflow float64 (max ~1.8e308), but it produces a dimensionless counterion concentration of `c_b * 2.7e43` — 43 orders of magnitude above bulk. This overflows the Poisson source term `z_k * c_steric_k * charge_rhs` in practice.

**However:** For SO4²⁻ (z=-2, anion) under **cathodic** polarization (phi < 0 at OHP), `phi < 0 → q_k = exp(+2*|phi|)`. At phi_clamped = -50, `q_k = exp(100) ~ 2.7e43`. This is the cathodic case (the physically relevant case for ORR). The clamp prevents SNES from evaluating `exp` at unbounded negative phi, but the clamped value itself is 2.7e43. With c_b ~ O(1), and free_dyn/denom ~ O(1) (when not at saturation), `c_steric_k ~ 2.7e43`. Multiplied by `charge_rhs` and integrated over a boundary ds, this would create enormous fictitious charge.

**Mitigation in place:** The `free_dyn/(theta_b + ...)` denominator saturates: when q_k is large, `denom ~ a_k * c_b * q_k` dominates, and `c_steric_k → free_dyn / a_k`. So the Bikerman saturation cap **is functioning correctly** as a ceiling: `c_steric_k_max ~ 1/(a_k)` regardless of how large `exp(...)` grows. This is the physically correct hard-packing ceiling. No overflow bug here — the denominator absorbs the exponential growth exactly as designed. The exp(100) value in the numerator and denominator cancel to give O(1/a_k).

**Residual concern:** The z_scale shared between ideal and Bikerman counterions (line 245-250 boltzmann.py) means Strategy-B z-ramp affects both simultaneously. Bikerman entries with large q_k and the Strategy-B z-ramp at z_scale=0 → 1 may cause a jump discontinuity in the Poisson term if the ramp is coarse.

---

## Q4 — Closed-form-functional invariant: no trial/test functions in Bikerman
**SEVERITY: CLEAN**  
**Location:** `boltzmann.py:203-268`

`build_steric_boltzmann_expressions` receives `phi` (the Newton-state subfunction from `fd.split(U)[indices.phi_index]`), `ci` (the exp-reconstructed UFL closures), and `a_dyn_funcs` (R-space fd.Function constants). The returned `c_steric_k` UFL expression (`p["c_const"] * p["q"] * free_dyn / denom`) depends only on:
- `fd.Constant` values (z_k, c_k, a_k)
- `phi` (Newton state — correct)
- `ci[i]` which equals `fd.exp(fd.min_value(fd.max_value(u_exprs[i], ...), ...))` — also depends on Newton state

No `TestFunction` or `TrialFunction` enters these expressions. The `w` test function (line 339 of forms_logc_muh.py) is only multiplied in when building the Poisson source term (`charge_density * w * dx`) inside `build_forms_logc_muh`, not inside `build_steric_boltzmann_expressions`. Invariant holds.

---

## Q5 — J_form derivation ordering: REAL BUG for ideal Boltzmann path
**SEVERITY: HIGH / REAL BUG (IDEAL PATH)**  
**Location:** `forms_logc_muh.py:839-881`

```python
J_form = fd.derivative(F_res, U)          # line 839: J taken BEFORE add_boltzmann
ctx.update({..., "J_form": J_form, ...})  # line 841-876: J_form stored (STALE for ideal counterions)
ctx.update(_step6_cation_hydrolysis_artifacts)
add_boltzmann_counterion_residual(ctx, params, skip_bikerman=True)  # line 881: mutates F_res, re-derives J
```

`add_boltzmann_counterion_residual` (boltzmann.py:384-386) does re-derive `J_form` from the updated `F_res`:
```python
ctx["F_res"] = F_res
ctx["J_form"] = fd.derivative(F_res, U)  # boltzmann.py:386
```

**So J_form ends up correct in ctx IF ideal (non-bikerman) counterions are present.** However, there is a window between line 841-876 (`ctx.update({"J_form": J_form, ...})`) and line 881 where `ctx["J_form"]` is stale. Any code that uses `ctx["J_form"]` between those two lines would get the wrong Jacobian (missing the ideal Boltzmann Poisson terms). In the current sequential execution of `build_forms_logc_muh`, nothing reads ctx["J_form"] in that window, so the bug is latent.

**More importantly:** When ALL counterions are Bikerman (skip_bikerman=True skips them all), `add_boltzmann_counterion_residual` returns 0 without updating J_form. The Bikerman contributions enter F_res via `steric_boltz` bundles assembled BEFORE line 839 (they are part of the Poisson term at lines 656-660). So for the pure-Bikerman path, J_form at line 839 **already includes the Bikerman contribution** — J_form is correct. The `skip_bikerman=True` call at line 881 is a no-op for fully Bikerman configs.

**Actual bug scenario:** Mixed config with BOTH ideal and Bikerman counterions. At line 839, F_res already has Bikerman terms (added via steric_boltz in lines 656-660) but does NOT yet have the ideal-path counterion terms (added at line 881). The J_form taken at line 839 is missing ideal-path Poisson derivatives. The call at line 881 corrects this. Execution is sequential, so the final J_form is always correct. The intermediate stale state is latent but not currently triggered.

**Verdict on Q5:** Not an active bug — J_form is always consistent at the point `build_forms_logc_muh` returns, because `add_boltzmann_counterion_residual` re-derives it at line 886 of boltzmann.py. The intermediate stale state at ctx line 841-876 is a maintainability hazard (future async callers could read stale J_form), but not a current correctness defect.

---

## Q6 — Picard fallback: U_prev correctly set
**SEVERITY: CLEAN**  
**Location:** `forms_logc_muh.py:1028-1039`

On Picard failure, `set_initial_conditions_debye_boltzmann_logc_muh` calls `set_initial_conditions_logc_muh(ctx, solver_params)`. That function (lines 980-994) explicitly writes all `U_prev.sub(i)` for i in range(n) and `U_prev.sub(n)` (phi), then does `ctx["U"].assign(U_prev)`. U_prev is fully initialized before SER step 0. No uninitialized state.

---

## Q7 — Picard singular Jacobian: explicit bailout
**SEVERITY: CLEAN**  
**Location:** `picard_ic.py:1416-1432` (general loop), `picard_ic.py:748-751` (2x2 loop)

Both paths detect singularity explicitly:
- 2x2 loop: `abs(det) < 1e-300 or not isfinite(det)` → returns `(False, "singular_jacobian_iter_k_det=...", k, state_dict_failure)`.
- N-reaction general loop: `abs(det) < 1e-300` (N=2) or `any(not isfinite(v) for v in R_solve)` (N>2) → same early-return pattern.

Both bail out with `ok=False`, which propagates to the caller (`_try_debye_boltzmann_ic_muh`) as a Picard failure, triggering the linear-phi IC fallback. No pseudo-inverse, no damped step on singularity — hard exit. This is correct behavior for an IC seeder (fallback exists), though it means any config that consistently produces singular Picard Jacobians silently falls back to the linear-phi IC without alarming the user beyond the `initializer_fallback=True` flag.

---

## Q8 — Picard oscillation detection: ABSENT
**SEVERITY: LOW / KNOWN GAP**  
**Location:** `picard_ic.py:1519-1521` (general loop), `picard_ic.py:804-806` (2x2 loop)

Convergence is checked only by `delta < tol` (relative rate-change). There is **no explicit oscillation detection**. If R alternates between two values with the same magnitude `|R_1 - R_2| = const`, `delta` never decreases below `tol` and the loop exhausts `max_iters` (default 50) before returning `(False, "picard_max_iters_delta=...", ...)`. This triggers the fallback IC.

**Risk:** Omega=0.5 damping (line 1445) is sufficient to suppress most oscillations in the convex regime, but configurations with strong Stern coupling (large C_S) or near-saturation packing can produce two-cycle oscillations that damping at omega=0.5 cannot quench. The outcome is max-iter exit and linear-phi fallback — functionally safe but diagnostically silent.

---

## Q9 — solve_stern_split bisection bracket guarantee: LATENT BUG
**SEVERITY: MEDIUM / LATENT DEFECT**  
**Location:** `picard_ic.py:253-270`

```python
lo, hi = 0.0, full_drop
f_lo = residual(lo)
f_hi = residual(hi)
if f_lo * f_hi > 0.0:
    # Linear-Debye limit fallback...
    psi_D_lin = stern_coeff_nondim * full_drop * lambda_D / denom
    psi_S_lin = full_drop - psi_D_lin
    return psi_S_lin, psi_D_lin, phi_applied_model - psi_S_lin
```

When the bracket fails (`f_lo * f_hi > 0`), the code falls back to the linear-Debye approximation. This approximation is `psi_D_lin = C_S * full_drop * lambda_D / (eps + C_S * lambda_D)`, valid only in the small-|psi_D| limit. **If the true root lies outside [0, full_drop] due to numerical precision or a non-monotone residual (e.g., Bikerman denominator becoming complex-shaped at extreme a_cl), the linear fallback returns a value that does NOT satisfy the Robin identity** — it is an approximation with no guaranteed error bound.

The comment "Falls through to a linear-Debye analytical solution when bisection cannot bracket the root (typically only at `psi_D ~ 0`)" acknowledges this. In practice, for the production config (a_cl = SO4 Bikerman size, C_S = 0.20), the residual is monotone and the bracket always holds. The risk activates when:
- `full_drop` is very small (near zero crossing) — covered by the `abs(full_drop) < 1e-12` guard at line 241
- Non-monotone `compute_surface_slope_signed` (can occur at large `nu_charged = 2*a_cl*c_clo4_bulk` combined with extreme psi_D)

**Verdict:** The linear-Debye fallback returns a non-root silently, which seeds the Picard IC with a physically incorrect Stern split. If this path is taken, the IC is wrong but the solver does not know. Picard will then converge from this wrong IC or fail → fallback to linear-phi IC. Net effect: extra Newton iterations or fallback IC, not incorrect converged solution (the solver independently enforces the Stern BC in the residual).

---

## Q10 — Numerical underflow on log-c: Jacobian near-zero
**SEVERITY: LOW / FLOAT64 SAFE**  
**Location:** `forms_logc_muh.py:338-341`

`ci[i] = fd.exp(fd.min_value(fd.max_value(u_exprs[i], -_U_CLAMP_C), _U_CLAMP_C))` with `_U_CLAMP = 30.0` (default). At clamp=-30, `exp(-30) ~ 9.4e-14` — well above float64 underflow (5e-324). The Jacobian term `d(exp(u))/du = exp(u) ~ 9.4e-14` is small but not zero. Rows corresponding to the NP flux of a trace species will be numerically small but not singular.

In log-c formulation the NP residual for species i has the form `D_i * c_i * grad(v_i) * dx` (roughly). When `c_i ~ 1e-13`, this row contributes `~1e-13` to J. The condition number of J is the ratio of largest/smallest pivot, so trace species inflate the condition number by ~`1/c_i_min`. At `c_i_min = exp(-30) ~ 1e-13`, condition number inflation is ~10^13 relative to a well-conditioned O2 row. SNES with ILU/GAMG should handle this, but it can cause iterative linear solver stagnation on coarse grids. Not an immediate bug — flagged for awareness.

---

## Q11 — mu_h_idx caching: computed fresh each build
**SEVERITY: CLEAN**  
**Location:** `forms_logc_muh.py:289-290`

```python
species_roles = _get_species_roles(params, n)
mu_h_idx = _resolve_mu_h_index(list(z_vals), roles=species_roles)
```

`_resolve_mu_h_index` is called at build time from `z_vals` and `roles`, which are inputs to `build_forms_logc_muh`. The index is NOT cached on ctx or as a module-level variable. Each call to `build_forms_logc_muh` recomputes it. Same pattern in `set_initial_conditions_logc_muh` (line 925) and `_try_debye_boltzmann_ic_muh` (line 1120). If species reorder between calls, each build independently re-resolves the correct index. No stale-cache risk.

---

## Q12 — Additional findings

### Q12a — log-rate branch: fd.ln(k0_j) when k0_j = 0
**SEVERITY: HIGH / REAL BUG (conditional on config)**  
**Location:** `forms_logc_muh.py:529-535, 554`

```python
if float(rxn["k0_model"]) <= 0.0 or bool(rxn.get("enabled", True)) is False:
    R_j = fd.Constant(0.0)
    bv_rate_exprs.append(R_j)
    continue  # <-- skips to next rxn; k0_j is already populated above

# ... later, for enabled reactions:
log_cathodic = fd.ln(k0_j) + u_exprs[cat_idx] - alpha_j * n_e_j * eta_j
```

The guard at line 532 correctly skips `fd.ln(k0_j)` when `k0_model <= 0`. However, `k0_j` is an `fd.Function` whose value is `.assign(float(rxn["k0_model"]))`. If `k0_model` is a very small positive number (e.g., `1e-30` — the mass-transport floor used in K0_FACTOR tests), `fd.ln(k0_j)` evaluates to `ln(1e-30) ~ -69.1`. This is fine. But if `k0_model` is exactly 0.0 for a reaction marked `enabled=True` (a config mistake), the guard `<= 0.0` catches it and returns `R_j = 0`. This guard is correct.

**Actual latent bug:** If someone passes `k0_model = -1e-30` (negative, due to a sign error in the nondim scaling that takes k0 negative), the guard `<= 0.0` catches it and sets `R_j = 0` silently. No exception is raised. The correct behavior would be to raise a ValueError, as negative k0 is unphysical.

### Q12b — Stoichiometry sign applied twice for BV flux in Poisson
**SEVERITY: CLEAN (verified)**  
The Poisson equation does not include BV fluxes — it only includes the dynamic species charge and the analytic Boltzmann counterion charge. The BV boundary flux appears only in the NP residual via `F_res -= stoi[i] * R_j * v_list[i] * ds(electrode_marker)`. The sign convention here: `F_res` is the NP residual (continuity weak form); the BV rate `R_j = cathodic - anodic`; minus sign gives a source when R_j > 0 (cathodic consumption). This is internally consistent.

### Q12c — Missing phi_prev in BV rate for non-Stern path
**SEVERITY: LOW / DOCUMENTATION INCONSISTENCY**  
For Stern path: `eta_raw = phi_applied_func - phi - E_eq_const` (line 394) uses the Newton-state `phi` (current iterate). This is correct for implicit time stepping. For the non-Stern `use_eta_in_bv=True` path: `eta_raw = phi_applied_func - E_eq_const` — a constant, independent of the Newton state. The BV rate then has zero Jacobian wrt phi via the eta pathway (it depends only on concentrations via `c_surf`). This is intentional (the `use_eta_in_bv` path is a fixed-eta approximation), but it means the Jacobian is missing the d(R)/d(phi) column for the BV boundary terms. Could cause slower Newton convergence when the BV rate is strongly phi-coupled. Not a bug per the intended design.

---

## VERDICT

| Q | Status | Severity |
|---|--------|----------|
| Q1 | Jacobian zeroed on clip plateau | LOW — production grid unclipped; risk at V < -0.8V |
| Q2 | free_dyn floor creates spurious c_k contribution at saturation+clamp | MEDIUM — latent, activated when physical-a bridge runs assigned |
| Q3 | Bikerman exponent handled by saturation denominator — overflow absorbed | LOW — design correct, z_scale ramp discontinuity residual risk |
| Q4 | No test/trial functions in Bikerman expressions | CLEAN |
| Q5 | J_form taken before ideal Boltzmann added; re-derived correctly at line 881 | LATENT (not active); final J_form always consistent |
| Q6 | U_prev fully initialized in fallback IC path | CLEAN |
| Q7 | Singular Jacobian: hard bailout to fallback IC | CLEAN |
| Q8 | No oscillation detection — max-iter exit only | LOW GAP |
| Q9 | Stern bisection fallback returns non-root silently | MEDIUM — Picard IC can be seeded incorrectly; downstream solver absorbs |
| Q10 | log-c underflow safe in float64; condition number inflation only | LOW |
| Q11 | mu_h_idx recomputed fresh each build — no cache | CLEAN |
| Q12a | Negative k0_model silently treated as disabled | LOW — unphysical config not raised |
| Q12b | BV stoichiometry signs consistent | CLEAN |
| Q12c | use_eta_in_bv=True gives zero dR/dphi in Jacobian | LOW / INTENTIONAL |

**Priority action items:**
1. **Q9 (MEDIUM):** Add a log warning when `solve_stern_split` falls back to linear-Debye so users know the Stern split is approximate.
2. **Q2 (MEDIUM):** When physical-a bridge runs proceed (Hard Rule #7 work), re-examine whether the `free_dyn` floor at 1e-10 producing large `c_steric_k` via exp(100)/denom near the saturation+clamp corner is physically meaningful, or whether `c_steric_k` should be capped independently.
3. **Q12a (LOW):** Add a `ValueError` when `k0_model < 0` rather than silent treatment as disabled.
