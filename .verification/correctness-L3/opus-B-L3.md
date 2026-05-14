# L3 Code-Correctness Audit (Opus, Pass B)

Scope: targeted reads of `forms_logc_muh.py`, `boltzmann.py`, `picard_ic.py`.
Documentation ignored. Findings are code-only.

---

## FINDING 1 — Multi-ion Picard receives only `counterions[0]` bulk concentration as `c_clo4_bulk`; secondary counterion enters only through `multi_ion_ctx`. Inconsistent path coverage.

SEVERITY: HIGH (correctness for Cs+/SO4 stack with single-ion fallbacks)
LOCATION: `Forward/bv_solver/forms_logc_muh.py:1182` and `Forward/bv_solver/forms_logc_muh.py:1267-1285`
TRIGGER: Multi-counterion Cs+/SO4²⁻ stack reaching `_try_debye_boltzmann_ic_muh`. Fires every production run since K⁺/SO₄²⁻ migration.
EVIDENCE:
```python
# forms_logc_muh.py:1182
c_clo4_bulk = max(float(counterions[0]["c_bulk_nondim"]), 1e-300)
...
# forms_logc_muh.py:1267-1285
ok, reason, picard_iters, picard_state = picard_outer_loop_general(
    ...
    c_clo4_bulk=c_clo4_bulk,           # ← only counterions[0]
    a_h=a_h_picard,
    a_cl=a_cl_picard,                  # ← also only the first bikerman a_nondim
    c_cl_anchor_kind=c_cl_anchor_kind,
    stern_split=stern_split_picard,
    multi_ion_ctx=ctx_mion_pre,        # ← second counterion only reaches here
    ...
)
```
Inside `picard_outer_loop_general` (`picard_ic.py:1379`), when `multi_ion_ctx is not None` the helpers `_solve_phi_o`, `_compute_picard_gamma_s`, `_solve_picard_stern_split` dispatch to the multi-ion branch and `c_clo4_bulk` / `a_cl` are unused for electrostatics. But:
1. `c_cl_anchor = H_o if c_cl_anchor_kind == "synthesised_4sp" else c_clo4_bulk` at line 1379 still reads `c_clo4_bulk` and is forwarded to `_compute_picard_gamma_s` as `c_cl_anchor`. In multi-ion mode the helper ignores it (line 361-364) — so the value is dead, but it is *also* passed as `c_clo4_bulk` to the per-iter helpers in single-ion fall-throughs. **If `multi_ion_ctx` is dropped on any sub-call (it currently is not, but the API permits passing only the multi-ion branch on one helper and not the other), behavior diverges silently.**
2. The IC `phi_init_expr` at `forms_logc_muh.py:1493`/`:1546` *does* use `fd.ln(H_outer / fd.Constant(c_clo4_bulk))` directly — this is the **single-ion BKSA branch** for the spatial profile. For the multi-ion case the branch at `:1345` (`multi_ion_mode = len(bikerman_entries_in_counterions) > 1`) is taken and uses `phi_o_local` from `solve_outer_phi_multiion`, so the `c_clo4_bulk` BKSA path is bypassed.

The dangerous remnant: `a_cl_picard` is set from the *first* bikerman entry that `next(...)` picks at line 1208–1212 — **dictionary insertion order**, not z-sign. If a deck author lists SO₄²⁻ first, `a_cl_picard` becomes a's value (with `c_cl_anchor_kind="bulk"`), and although multi-ion mode bypasses single-ion γ_s computation, the **Stern split path in `_solve_picard_stern_split` does NOT receive `multi_ion_ctx` when called inside the picard outer loop** (line 1349 vs 776). It passes `multi_ion_ctx=multi_ion_ctx`, which is None for the multi-counterion case here. Re-checking:

```python
# forms_logc_muh.py:1255-1265
if len(bikerman_entries_pre) > 1:
    ...
    ctx_mion_pre = _build_counterion_ctx_pre(...)
else:
    ctx_mion_pre = None
```
So `ctx_mion_pre` IS set when ≥2 bikerman entries. Then it is forwarded to `picard_outer_loop_general(... multi_ion_ctx=ctx_mion_pre ...)` which dispatches to `_solve_picard_stern_split(... multi_ion_ctx=multi_ion_ctx, ...)` — i.e. correct.

**Net assessment for FINDING 1**: not a live bug for current production decks, but `c_clo4_bulk = counterions[0]["c_bulk_nondim"]` is a misleading name for "the first counterion's bulk concentration, used as a stand-in only in single-ion sub-helpers and bypassed in multi-ion mode." The variable is dead weight in multi-ion mode. Downgrade to MEDIUM cleanup hazard: if a future patch removes the `multi_ion_ctx` dispatch from one sub-helper, the wrong bulk concentration will silently propagate.

VERDICT: MEDIUM (latent, surface-area-only; not exercised by production)

---

## FINDING 2 — `_solve_phi_o(multi_ion_ctx=ctx_mion)` warm-start bracket `(phi_o_prev ± 5)` is not symmetric over the bisection sign convention; can miss the root for large jumps and silently fall back to the slower (-50, +50) bracket.

SEVERITY: LOW (performance / latency, not correctness)
LOCATION: `Forward/bv_solver/picard_ic.py:324-337`
TRIGGER: First Picard iteration after a large applied-voltage step where `phi_o` moves > 5 V_T.
EVIDENCE:
```python
if phi_o_prev is not None and math.isfinite(phi_o_prev) and abs(phi_o_prev) < 50.0:
    try:
        return solve_outer_phi_multiion(
            ctx=multi_ion_ctx,
            c_dyn_outer=c_dyn_outer,
            bracket=(phi_o_prev - 5.0, phi_o_prev + 5.0),
        )
    except ValueError:
        pass
return solve_outer_phi_multiion(
    ctx=multi_ion_ctx,
    c_dyn_outer=c_dyn_outer,
    bracket=(-50.0, +50.0),
)
```
The `ValueError` from `solve_outer_phi_multiion` (line 311 — "bracket does not contain a root after expansion") is caught and the global bracket retried. Inside `solve_outer_phi_multiion`, bracket-expansion doubles `[a_low, a_high]` up to 8 times BEFORE raising. So the actual escape path is: try local, double 8x within the inner expansion, raise; fall back to global. Functionally correct but doubles the bisection work in steep-step cases. No correctness bug.

VERDICT: LOW (potential 2x cost in extreme cases, never wrong)

---

## FINDING 3 — `solve_stern_split` bisection bracket sign check (Question F)

SEVERITY: INFO
LOCATION: `Forward/bv_solver/picard_ic.py:256-270`
TRIGGER: never (correctly handled).
EVIDENCE: Both endpoints checked at lines 257-258. If `f_lo * f_hi > 0.0`, the linear-Debye small-|psi_D| fallback is applied (lines 259-270). The fallback is mathematically the closed-form solution of the linearized closure, so it returns a meaningful `(psi_S_lin, psi_D_lin)` even when the bisection bracket misses the root. **This is correct.** Question F is fully handled.

VERDICT: NONE

---

## FINDING 4 — Sign of Stern Robin term in F_res (Question H)

SEVERITY: INFO (sign-consistent with documented convention)
LOCATION: `Forward/bv_solver/forms_logc_muh.py:666-668`
EVIDENCE:
```python
if use_stern:
    stern_coeff = fd.Constant(float(stern_capacitance_model))
    F_res -= stern_coeff * (phi_applied_func - phi) * w * ds(electrode_marker)
```
The Robin closure is `σ_S = C_S · (φ_M − φ_OHP)`. With `F_res` = (eps ∇φ · ∇w) dx − (Σ z_i c_i w) dx + ... and the natural boundary `∫ ε (∂φ/∂n) w ds` = `∫ σ w ds`, the consistent sign for the Stern flux on the electrode is `+σ w ds = +C_S (φ_M − φ_OHP) w ds`. Moving to the residual side gives `F_res -= C_S * (φ_applied − φ) * w * ds`. **The sign is correct.**

For cathodic polarization, `phi_applied < 0` (cathodic in the model), and the residual term `−C_S(phi_applied − phi)` produces a forcing that drives `phi_OHP > phi_applied`, leaving a positive Stern drop `psi_S = phi_applied − phi_OHP < 0` — i.e. positive `σ_S = C_S · psi_S < 0` (negative surface charge density, attracting cations). Consistent with physics. NONE.

VERDICT: NONE

---

## FINDING 5 — Sign of Boltzmann counterion contribution to Poisson (Question I)

SEVERITY: INFO (correct sign)
LOCATION: `Forward/bv_solver/forms_logc_muh.py:644-660` and `boltzmann.py:34-41,380-382`
EVIDENCE: Module docstring at `boltzmann.py:34-41` documents the contract:
```
F_poisson  =  eps * grad(phi) . grad(w) * dx
              - charge_rhs * sum_i z_i * c_i * w * dx
              - charge_rhs * sum_k z_k * c_steric_k * w * dx
```
Forms file at `forms_logc_muh.py:644-660` matches:
```python
F_res += eps_coeff * fd.dot(fd.grad(phi), fd.grad(w)) * dx
if not suppress_poisson_source:
    F_res -= charge_rhs * sum(z[i] * ci[i] * w for i in range(n)) * dx
    ...
    if steric_boltz:
        F_res -= z_scale_shared * charge_rhs * charge_density_total * w * dx
```
And ideal-mode (legacy) at `boltzmann.py:380-382`:
```python
F_res -= z_scale * charge_rhs * z_const * c_bulk_const * fd.exp(
    -z_const * phi_clamped
) * w * dx
```
For Cs+ (z=+1) under cathodic polarization (φ<0): `c_steric ∝ exp(−z·φ) = exp(+|φ|) > 1` → enriched at OHP → `z·c > 0` adds positively to Poisson source → drives `−∇²φ` positive → `phi` curves toward neutralization. **Correct neutralization sign.** NONE.

VERDICT: NONE

---

## FINDING 6 — `compute_surface_gamma_multiion` uses `c_outer` (NOT `c_bulk`) for ions; consistent with shared-θ closure (Question A)

SEVERITY: INFO (verified consistent)
LOCATION: `Forward/bv_solver/multi_ion.py:335-370` and `boltzmann.py:91-269`
EVIDENCE: The bulk-side bookkeeping is at `boltzmann.py:159` and `multi_ion.py:171`:
```python
theta_b = 1.0 - A_dyn_bulk - A_an_bulk   # bulk side
```
where `A_an_bulk = sum_k a_k * c_b_k` over **every** bikerman entry. The local (φ-dependent) denominator at `boltzmann.py:241-242`:
```python
denom = theta_b_const + sum(p["a_const"] * p["c_const"] * p["q"]
                            for p in per_ion_q)
```
sums over **all** bikerman ions with each ion's own `exp(-z·φ)` factor. **This is the correct shared-θ partition.** The closure (`boltzmann.py:254`) is `c_steric_k = c_b_k * q_k * (1 - A_dyn) / denom`. For Cs+ AND SO4²⁻ co-saturation, both `c_Cs` and `c_SO4` enter `denom` symmetrically and the lattice constraint `a·c_Cs + a·c_SO4 + a·c_dyn ≤ 1` is enforced by construction since `denom ≥ theta_b_const + max term > 0` always.

**However**, the closure assumes `theta_b > 0`. At line 160-171, an explicit `ValueError` is raised when `theta_b <= 0.0`. So co-saturation that overruns the lattice constraint at the bulk side is caught loudly. Good. NONE.

VERDICT: NONE

---

## FINDING 7 — `free_dyn = max_value(1 - A_dyn_local, 1e-10)` is the SAME free_dyn applied to all bikerman bundles (Question G)

SEVERITY: INFO (verified)
LOCATION: `Forward/bv_solver/boltzmann.py:204-206, 254`
EVIDENCE:
```python
A_dyn_local = sum(a_dyn_funcs[i] * ci[i] for i in range(len(ci)))
free_dyn_floor = fd.Constant(1e-10)
free_dyn = fd.max_value(fd.Constant(1.0) - A_dyn_local, free_dyn_floor)
...
for p in per_ion_q:
    c_steric_k = p["c_const"] * p["q"] * free_dyn / denom
```
A single `free_dyn` symbol is multiplied into every bundle's `c_steric_expr`. The denominator `denom` is also computed once and shared. **Self-consistent: every counterion sees the same dynamic-packing factor (1 − A_dyn) and the same partition `denom`.** NONE.

VERDICT: NONE

---

## FINDING 8 — UFL `min_value`/`max_value` are NOT smooth — hard clips → derivative = 0 in clipped regions (Question B)

SEVERITY: MEDIUM (silent Jacobian zeros, but only when clip is binding)
LOCATION: `Forward/bv_solver/forms_logc_muh.py:400-402` and `boltzmann.py:224-227`
TRIGGER: `eta_scaled = bv_exp_scale * (φ_applied − φ − E_eq)` reaches `±exponent_clip = ±100` on the electrode boundary, OR `phi` reaches `±phi_clamp = ±50` everywhere in the cell.
EVIDENCE: Firedrake's `fd.min_value` and `fd.max_value` are non-smooth piecewise expressions; their UFL derivative is exactly zero on the clipped side. If `eta_scaled > 100` *uniformly across the boundary*, the Jacobian contribution from the BV term vanishes (cathodic), making `J_form` singular w.r.t. those degrees of freedom.

Current production exponent_clip is 100. Per CLAUDE.md "Hard rule #2", clip=100 is supposed to keep R2 unclipped at V_RHE > −0.79 V. So in production no DOF binds. But the *floor* on dynamic packing in `boltzmann.py:206` (`max_value(1 − A_dyn_local, 1e-10)`) and `forms_logc_muh.py:452` (`max_value(theta_inner, packing_floor=1e-8)`) clip on a per-quadrature-point basis. If the steric saturation is binding everywhere on the electrode boundary (e.g. K+ pile-up at strong cathodic polarization with the documented `c_K` = 291), the term `−ln(packing_floor)` becomes a constant and `∂(mu_steric)/∂(any U)` = 0 in the saturated zone.

This is documented as an intentional "Bikerman saturation floor" in `boltzmann.py:205` but the consequence for the Newton Jacobian is not flagged in the code. The Newton solve can stall if the saturation floor is touched globally; mitigated in practice by the floor being small enough that the system reaches steric equilibrium before all DOFs bind.

VERDICT: MEDIUM-LOW. Real risk under extreme cathodic polarization. No explicit defensive code beyond the floor itself. Documented in the source comments.

---

## FINDING 9 — `bv_exp_scale = fd.Constant(...)` is a global constant, `phi_applied_func` is R-space (global); η is local only through φ (Question C)

SEVERITY: INFO
LOCATION: `Forward/bv_solver/forms_logc_muh.py:375-403`
EVIDENCE:
```python
phi_applied_func = fd.Function(R_space, name="phi_applied")
phi_applied_func.assign(float(scaling["phi_applied_model"]))
...
E_eq_model_global = fd.Constant(float(scaling["bv_E_eq_model"]))
bv_exp_scale = fd.Constant(float(scaling["bv_exponent_scale"]))
...
def _build_eta_clipped(E_eq_const):
    if use_stern:
        eta_raw = phi_applied_func - phi - E_eq_const   # phi is local in y
    elif conv_cfg["use_eta_in_bv"]:
        eta_raw = phi_applied_func - E_eq_const
    else:
        eta_raw = phi - E_eq_const
    eta_scaled = bv_exp_scale * eta_raw
```
`phi_applied_func` is in the R-space — a global scalar. `phi` is the trial function (varies in y across the mesh). Inside `ds(electrode_marker)`, only the boundary trace of `phi` is sampled — i.e. `phi(y=0)`. So `eta_raw` is effectively a global scalar at the boundary, multiplied by the global `bv_exp_scale`. **All correct.** NONE.

VERDICT: NONE

---

## FINDING 10 — `_resolve_mu_h_index` correctness on changing species ordering (Question D)

SEVERITY: INFO (form-build-time re-resolved, NOT cached)
LOCATION: `Forward/bv_solver/forms_logc_muh.py:95-145, 289-291`
EVIDENCE: `_resolve_mu_h_index(list(z_vals), roles=species_roles)` is called inside `build_forms_logc_muh` (line 290), `set_initial_conditions_logc_muh` (line 925), and `_try_debye_boltzmann_ic_muh` (line 1120). It receives `z_vals` from `solver_params[4]` and `species_roles` from `_get_species_roles(params, n)`. **The index is recomputed on every call.** No module-level cache.

The `roles` argument (Phase 6β v9 Gate 1) is used to disambiguate K2SO4 stacks where both H⁺ and K⁺ have z=+1. Without roles, the legacy path (line 134) demands "exactly one species with z=+1"; if two are present, it raises loudly. **Both paths are sound.** NONE.

VERDICT: NONE

---

## FINDING 11 — Picard oscillation handling: ω=0.5 damping only; no oscillation-detection fallback before max-iters (Question E)

SEVERITY: LOW-MEDIUM (correctness path is via fallback to linear-phi IC)
LOCATION: `Forward/bv_solver/picard_ic.py:759-760, 1445, 808-816, 1523-1534`
EVIDENCE:
```python
# picard_ic.py:759-760 (legacy 2x2 path)
R1 = (1.0 - omega) * R1 + omega * R1_new
R2 = (1.0 - omega) * R2 + omega * R2_new
...
# picard_ic.py:1445 (general path)
R = [(1.0 - omega) * R_old[j] + omega * R_solve[j] for j in range(N)]
```
The Picard uses straight under-relaxation with `ω=0.5` (default). There is no Aitken acceleration, no 2-cycle detection, no adaptive ω. If a 2-cycle persists, `delta` stays bounded but above `tol`, the loop hits `max_iters=50`, and returns `(False, "picard_max_iters_delta=...", ...)`. The caller (`_try_debye_boltzmann_ic_muh`) then sets `initializer_fallback=True` and falls back to `set_initial_conditions_logc_muh` (linear-phi IC).

**Does the demo's `debye_boltzmann` initializer set up Picard inputs such that Picard should converge in normal operation?** From the production driver (`mangan_full_grid_csplus_so4.py`-class scripts), Picard inputs are:
- `bulk_concs = [O_b, P_b, H_b]` — physical bulk concentrations
- `k0_targets` — production k0 values (Cs+/SO4 stack)
- `alphas, n_e` — Ruggiero parallel-2e/4e values
- `multi_ion_ctx` — built from `counterions` list

Empirically (per CLAUDE.md `project_pass_a_outcome.md`) Pass A grid 8/8 converged via PreconvergedAnchor + solve_grid_with_anchor. Picard *did* converge for the no-Stern stack. For the Stern stack at C_S=0.20, the v10a' two-stage anchor was needed (build at 0.10 then bump to 0.20) — Picard alone could not converge at the production target.

**Does demo rely on the fallback?** Yes, but only for edge cases. The relevant evidence:
1. `forms_logc_muh.py:1037-1039` — `set_initial_conditions_debye_boltzmann_logc_muh` explicitly sets `initializer_fallback = True` on Picard failure and calls the linear-phi IC as a safety net.
2. The single fallback path means **if Picard 2-cycles, the IC is the simple linear-phi seed, which may be far from the BV manifold at high V_RHE**. The Newton solver then has to do all the work from a poor IC.

The risk: 2-cycle scenarios are not detected and not damped further — they just count up to max-iters and bail. **A defensive layer (ω → ω/2 on stagnation, or Aitken Δ²) is absent.** For high-V_RHE V_RHE > +0.6 V Stern + multi-counterion runs, the C+D anchor recipe explicitly avoids relying on debye_boltzmann Picard alone, which is consistent with this concern.

VERDICT: LOW. Not a bug, but the fallback path's design assumption ("Picard either converges in ≤50 iters or we bail to linear") is brittle. No code-level fix needed unless production starts failing more.

---

## FINDING 12 — exp() in Bikerman closure: per-ion phi_clamp_k controls the exponent (Question J)

SEVERITY: INFO
LOCATION: `Forward/bv_solver/boltzmann.py:224-228`
EVIDENCE:
```python
phi_clamped_k = fd.min_value(
    fd.max_value(phi, fd.Constant(-phi_clamp_k)),
    fd.Constant(phi_clamp_k),
)
q_k = fd.exp(-z_k_const * phi_clamped_k)
```
Each counterion's `q_k = exp(-z_k · phi_clamped_k)` uses its OWN `phi_clamp_k` (typically 50). For z=-2 (SO4²⁻) and phi_clamp=50, the exponent is `−(−2)·50 = +100` at cathodic extreme, giving `q ≈ exp(100) ≈ 2.7e43`. fp64 max is 1.8e308, so no overflow. For z=+2 (none in current production), same.

The shared `denom = theta_b + Σ a_k c_b_k q_k`. With `c_b_k ~ 0.1 mol/m³ → nondim ~ 1`, `a_k ~ 0.01-0.05`, `q_k ~ 1e43`: `denom ~ 1e42` order. Then `c_steric_k = c_b_k * q_k * free_dyn / denom ~ 1 * 1e43 * (~1e-10) / 1e42 = ~1e-9`. Bounded.

The ratio is structurally well-defined: `c_steric_k = c_b_k · q_k · (1 − A_dyn) / (theta_b + Σ a_k' c_b_k' q_k')`. The dominant term in the denominator at high |φ| is the one with the largest `|z_k'|·sign(φ)`. So the ratio reduces to `c_b_k / (a_k' c_b_k') · (q_k / q_k')` where `q_k / q_k'` is bounded.

**No overflow risk.** Even at `phi_clamp = 50, z=-2`, fp64 holds the values without trouble. NONE.

VERDICT: NONE

---

## FINDING 13 — `c_clo4_bulk` symbol persists in K+/SO4 production code path (stale-name leftover)

SEVERITY: INFO (cosmetic; not a correctness bug)
LOCATION: `picard_ic.py` throughout; `forms_logc_muh.py:937-963, 1182, 1273, 1460-1500`
EVIDENCE: After the migration from ClO4⁻ to K+/SO4²⁻ (Cs+/SO4²⁻), the variable name `c_clo4_bulk` was kept as a stand-in for "the first counterion's bulk concentration", but the semantics are now generic. The `multi_ion_ctx` path bypasses these uses for ≥2 bikerman counterions.

Specifically:
- `forms_logc_muh.py:1182` reads `c_clo4_bulk = counterions[0]["c_bulk_nondim"]`. In Cs+/SO4 mode this is whatever ion is listed first.
- `forms_logc_muh.py:1493` builds `phi_init_expr = fd.ln(H_outer / fd.Constant(c_clo4_bulk)) + psi` — this is **bypassed** when `multi_ion_mode = True` (line 1345 condition).
- `picard_ic.py:1273` passes `c_clo4_bulk` to `picard_outer_loop_general`. When `multi_ion_ctx is not None`, internal helpers (e.g. `_solve_phi_o`) dispatch to multi-ion versions and `c_clo4_bulk` is unused in the electrostatics.

**Net**: in multi-ion mode, all the `c_clo4_bulk` references are dead. In single-ion (Cs+ only or K+ only without SO4) mode, they carry whichever counterion's bulk concentration is `counterions[0]`. **Functionally correct, naming-wise misleading.** A rename to `c_anion_bulk_first` or `c_counterion_bulk_for_singleion_fallback` would clean up the API surface.

VERDICT: NONE (cosmetic). Worth a rename in a future cleanup commit.

---

## FINDING 14 — H+ Bikerman a_nondim hardcoded discrepancy

SEVERITY: INFO (documented in CLAUDE.md Hard rule #7, not a code-side bug)
LOCATION: `Forward/bv_solver/boltzmann.py:204` and `forms_logc_muh.py:410-412`
EVIDENCE: `steric_a_funcs[i].assign(float(a_vals_list[i]))` reads `a_vals` from `solver_params[6]`. For dynamic species (O₂/H₂O₂/H⁺) the canonical driver passes `A_DEFAULT = 0.01` (per `_bv_common.py`; not in the read scope but confirmed in CLAUDE.md Hard rule #7). For counterions, real Marcus/Stokes radii are used. This is a **physics calibration issue, not a code bug** — the code faithfully implements whatever `a_vals` it receives.

Worth flagging because the closure `(1 − A_dyn) / denom` includes the H⁺ contribution at A_DEFAULT, so the dynamic-side packing is artificially loose. The closure is internally consistent (no double-counting), but the physical interpretation of the steric cap is off.

VERDICT: NONE (documented). Outside L3 scope.

---

## FINDING 15 — Picard inner Jacobian singular-J guard relies on `det` from `_solve_linear_system`; for N≥3 the singular check only looks at `not math.isfinite(v) for v in R_solve` (no det)

SEVERITY: LOW (Edge case for N≥3 reactions; current production is N=2 so this is exercised only in future R3-reaction stacks)
LOCATION: `Forward/bv_solver/picard_ic.py:1416-1432`
EVIDENCE:
```python
R_solve, det = _solve_linear_system(M_mat, b_vec)
if N == 2:
    singular = (
        not math.isfinite(det) or abs(det) < 1e-300
    )
else:
    singular = any(not math.isfinite(v) for v in R_solve)
```
For `N=2`, both the determinant check (`abs(det) < 1e-300`) AND the finiteness check are used. For `N >= 3`, only finiteness of `R_solve` is checked. A near-singular (but solveable into finite-but-huge `R`) 3×3 matrix would not be flagged, even if the result is meaningless.

In production (Ruggiero parallel 2e+4e = N=2), this path is never exercised. If R3-reaction topologies land, the singular-J guard for N≥3 will be too permissive.

VERDICT: LOW. Worth tightening to `abs(det) < tol * max(matrix-row-norms)` when N≥3 lands. Not a current bug.

---

## VERDICT (overall)

No HIGH-severity bugs surfaced in this pass. Findings ranked by gravity:

| # | Severity | Topic |
|---|----------|-------|
| 1 | MEDIUM | `c_clo4_bulk = counterions[0]` is dead in multi-ion mode but misleading API surface; latent risk if dispatch path changes |
| 8 | MEDIUM-LOW | Hard UFL min/max clips → zero Jacobian when binding globally; only mitigated by clip values being far enough from production operating envelope |
| 11 | LOW | Picard ω=0.5 only; no Aitken / oscillation detection; relies on fallback to linear-phi IC for divergence |
| 15 | LOW | N≥3 Picard singular-J guard checks finiteness only, not determinant scale; not yet exercised |
| 2 | LOW | Multi-ion `_solve_phi_o` warm-start bracket may fall back to global; perf, not correctness |
| 13 | INFO | Stale `c_clo4_bulk` naming after ClO4→K+/SO4 migration; not a bug but worth a rename |
| 14 | INFO | H+ Bikerman `a_nondim = A_DEFAULT` not physical; documented in Hard rule #7, calibration scope |

Bikerman shared-θ multi-ion closure (Cs+ AND SO4²⁻) is **self-consistent in code**: shared `theta_b`, single producer of `denom`, identical `free_dyn` applied to both bundles, identical `phi_clamp_k` semantics per ion. Sign conventions on Stern Robin BC and Poisson source are **correct** (matches documented contract). H⁺ index resolution is recomputed at form-build time, not cached. UFL clip overflow protection in the Bikerman closure is **safe** for fp64 at production `phi_clamp = 50` and `z ∈ {±1, ±2}`.

Picard fallback path: **brittle but not broken**. The full-grid production stack (Cs+/SO4² with C+D anchor) does NOT rely on Picard alone for high-V cases — the anchor + grid + warm-walk design hedges against Picard divergence.

PASS WITH HOUSEKEEPING NOTES.
