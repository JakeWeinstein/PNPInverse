# Opus Chunk B — Code Correctness Audit (forms_logc_muh + boltzmann + picard_ic)

Scope: Q1–Q10 + Opus-deeper A–D. Targeted reads only. No docs consulted.

---

## Q1: `_build_eta_clipped` Jacobian information loss — NOT A BUG (by design)

SEVERITY: low (intentional, design-documented behavior)
LOCATION: Forward/bv_solver/forms_logc_muh.py:388-403
TRIGGER: All voltage points where `|bv_exp_scale * (phi_applied - phi - E_eq)|` saturates the `exponent_clip = 100` bound (i.e., reactions R_2e at deeply cathodic V_RHE, R_4e at deeply anodic V_RHE).

EVIDENCE: `fd.min_value(fd.max_value(eta_scaled, -clip_val), clip_val)` at line 402 makes `d(eta_clipped)/d(eta_scaled) = 0` on the saturated plateau. The Jacobian rows corresponding to the BV boundary face integral lose the `phi`-dependence of that reaction (since `d(R_j)/d(phi)` through the η-channel vanishes). HOWEVER:
- BV boundary contribution is one of MANY terms in the Jacobian; the bulk NP equations (`fd.grad(ui)` and `fd.grad(phi)` couplings) and Poisson contributions remain non-zero.
- η-clipped reactions still couple to φ via `c_surf[i]` (concentration channel) — only the exponential channel vanishes.
- `c_surf[i]` is itself NOT clipped beyond `u_clamp=100` (line 326-340), which is well beyond what's reached in production with `phi_clamp=50` on counterions.

The clip is `fd.min_value`/`fd.max_value` (UFL-smooth), NOT Python's `min`/`max`, so derivative semantics are well-defined as one-sided (cf. Opus item A — verified at lines 402, 224-227, 374-377; all are `fd.min_value`/`fd.max_value`, none use bare Python `min`/`max`).

NOT a Newton breakdown in practice. Confirmed by repo's "production unclipped above V_RHE > -0.79V at clip=100" note (Hard Rule #2).

---

## Q2: Bikerman saturation pathology + `free_dyn` floor — POTENTIAL ISSUE (numerical, not algebraic)

SEVERITY: medium
LOCATION: Forward/bv_solver/boltzmann.py:205-206
TRIGGER: Strong cathodic polarization with dynamic-species pile-up at the OHP saturating `A_dyn ≈ 1`.

EVIDENCE: Line 206: `free_dyn = fd.max_value(fd.Constant(1.0) - A_dyn_local, free_dyn_floor)` with `free_dyn_floor = 1e-10`.

When `A_dyn → 1` (dynamic species fully pack the layer):
- True Bikerman: counterions are EXCLUDED from this region → `c_steric → 0`.
- Computed: `c_steric_k = c_b_k * exp(-z_k * phi) * (free_dyn / denom)` where `denom = θ_b + Σ a·c_b·exp(-z·φ)`.

At deep cathodic φ ≈ -50 (saturated phi_clamp), for SO4²⁻ (z=-2): `exp(-(-2)·-50) = exp(-100) ≈ 3.7e-44` — DEEPLY suppressed numerator. For Cs⁺ (z=+1): `exp(50) ≈ 5e21` — enriched. Denom dominated by the Cs⁺ term ≈ a_Cs·c_Cs·5e21.

So `c_steric_Cs ≈ c_Cs · 5e21 · free_dyn / (θ_b + a_Cs·c_Cs·5e21) ≈ free_dyn / a_Cs`.

WITH `free_dyn = 1e-10` floor: `c_steric_Cs ≈ 1e-10/a_Cs`. This is **NOT** unphysically large — it's actually a sensible cap once dynamic species have evicted everything. **No algebraic blow-up.**

CAUTION: The 1e-10 floor is BELOW machine epsilon when multiplied by `exp(50)·c_Cs` in floating-point. Specifically, `free_dyn/denom = 1e-10/(1e21) = 1e-31`, then `c_steric = c_Cs · 1e21 · 1e-31 = c_Cs · 1e-10` — small but representable. The **Jacobian** `d(c_steric_Cs)/d(φ)` via the floor is identically zero where `(1 - A_dyn) < 1e-10`, which gives Newton zero gradient information from the counterion charge density in that region. Could cause slow Newton convergence in deeply saturated runs but not a correctness bug per se.

VERDICT: numerical-only concern; algebraic limit is correct.

---

## Q3: Bikerman exponent overflow — IN-SPEC (no separate clip; phi_clamp does it)

SEVERITY: low (works in fp64; would fail in fp32)
LOCATION: Forward/bv_solver/boltzmann.py:228, 380-382 (no separate exp clip)
TRIGGER: SO4²⁻ z=-2 with φ-clamp 50 → `exp(-z·φ) = exp(±100)`.

EVIDENCE: Grep for "clip"/"min_value"/"max_value"/"exponent" in boltzmann.py shows ONLY the `phi_clamped` rectifier (lines 224-227, 374-377). There is no second-stage clip on `fd.exp(-z * phi_clamped)`.

Numerical check: with default `phi_clamp = 50.0` (scripts/_bv_common.py line 729 for SO4²⁻) and z=-2:
- Maximum |z·φ| = 100.
- `exp(±100) ≈ {3.7e-44, 2.7e43}`.
- IEEE 754 double max ≈ 1.8e308, min subnormal ≈ 5e-324.
- **Both bounded values are well within fp64 range.** No overflow.

If a deployment ever sets `phi_clamp > 350` on z=±2 species (e.g., 400), `exp(800)` overflows fp64 → `+inf`. Then `denom → inf`, `c_steric → 0`, Newton sees zeros in the residual. NO RUNTIME GUARD against this — `phi_clamp` is validated only as `> 0` in config.py:417, not bounded above. Recommend hardening the validator.

VERDICT: no bug in current configs; recommend adding `phi_clamp ≤ 200` upper-bound validation.

---

## Q4: Closed-form-functional invariant — VERIFIED CLEAN

SEVERITY: n/a
LOCATION: Forward/bv_solver/boltzmann.py:91-269
TRIGGER: n/a
EVIDENCE: `build_steric_boltzmann_expressions` does NOT call `fd.TrialFunction`, `fd.TestFunction`, or create new `fd.Function` on the mixed space W. The only Functions created are:
- Line 248: `z_scale = fd.Function(R_space, name="boltzmann_z_scale")` (on **R_space**, not W — scalar real space, OK).
The expressions returned (`c_steric_expr`, `packing_contribution`, `charge_density`) are UFL closure expressions using the *existing* `phi` argument (the caller's `fd.split(U)[indices.phi_index]`).

Cs⁺/SO4²⁻ are NOT added to Newton state. ✓

---

## Q5: `J_form = fd.derivative(F_res, U)` ordering — NUANCED PASS (re-derivation on line 386 saves it)

SEVERITY: low (saved by helper re-derivation)
LOCATION: Forward/bv_solver/forms_logc_muh.py:839, 881; boltzmann.py:386
TRIGGER: Ideal-mode (non-bikerman) Boltzmann counterions present.

EVIDENCE: Line ordering:
1. Line 839: `J_form = fd.derivative(F_res, U)` and stored in ctx via update at line 843.
2. Line 881: `add_boltzmann_counterion_residual(ctx, params, skip_bikerman=True)` — for ideal-mode entries, mutates F_res via line 380 (`F_res -= ...`) AND re-derives J_form at line 386: `ctx["J_form"] = fd.derivative(F_res, U)`.

CONCLUSION: J_form is re-derived AFTER the ideal-Boltzmann residual is appended. **This is correct.** But it's brittle: if a future contributor adds another residual term AFTER line 881 without calling `fd.derivative` again, J_form becomes stale.

Bikerman-mode terms are added BEFORE line 839 (lines 657-660 and 447-449 for packing), so the line-839 derivative captures them.

VERDICT: correct, but brittle — recommend collapsing to a single end-of-function derivative call after all residual mutations complete.

---

## Q6: Picard fallback → linear-phi IC — U_prev properly initialized

SEVERITY: n/a (no bug)
LOCATION: Forward/bv_solver/forms_logc_muh.py:1036-1039, 980-994
TRIGGER: `_try_debye_boltzmann_ic_muh` returns `ok=False`.

EVIDENCE: Lines 1036-1039:
```
ctx["initializer_fallback"] = True
ctx["initializer_fallback_reason"] = reason
ctx["initializer_picard_iters"] = picard_iters
set_initial_conditions_logc_muh(ctx, solver_params)
```
`set_initial_conditions_logc_muh` (lines 889-994) fully interpolates **every** `U_prev.sub(i)` (lines 986, 991, 993) AND copies `U_prev → U` at line 994 (`ctx["U"].assign(U_prev)`). No partial-state contamination.

Caveat: if Picard returns failure AFTER having partially mutated `U_prev` mid-iteration (e.g., singular-Jacobian at iter k), `set_initial_conditions_logc_muh` cleanly overwrites all subfunctions. ✓

The `dt` referenced in the audit prompt is `dt_const` (line 314) — set ONCE at build time from `scaling["dt_model"]`, not driven by U_prev. SER logic is NOT in picard_ic.py at all (no matches for "SER", "initial_dt"). No "bogus dt from uninitialized U_prev" risk in this layer.

---

## Q7: Picard singular Jacobian handling — VERIFIED with explicit guard

SEVERITY: n/a (no bug)
LOCATION: Forward/bv_solver/picard_ic.py:748-757, 1416-1432
TRIGGER: 2x2 / N×N Picard inner solve produces `|det| < 1e-300` or non-finite.

EVIDENCE: Both legacy 2x2 (line 748) and general N-rxn (line 1418-1422) paths check the determinant (or finiteness of `R_solve` for N>2) and return `(False, "singular_jacobian_iter_{k}_det={det:.3g}", k, state)` — a clean failure that triggers fallback to linear-phi IC in the caller. No silent NaN propagation.

---

## Q8: Stern-split bisection bracket validation — VERIFIED with explicit fallback

SEVERITY: n/a (no bug)
LOCATION: Forward/bv_solver/picard_ic.py:181-287, especially 256-270
TRIGGER: `f_lo * f_hi > 0` (no sign change in [0, full_drop]).

EVIDENCE: Lines 257-270:
```
f_lo = residual(lo)
f_hi = residual(hi)
if f_lo * f_hi > 0.0:
    # Linear-Debye limit fallback ...
    psi_D_lin = stern_coeff_nondim * full_drop * lambda_D / denom
    return psi_S_lin, psi_D_lin, phi_applied_model - psi_S_lin
```
Bisection is NOT entered when endpoints have the same sign — replaced by closed-form linear-Debye approximation. No silent meaningless-midpoint convergence.

---

## Q9: `_resolve_mu_h_index` memoization — CALLED EACH BUILD, NOT MEMOIZED

SEVERITY: n/a (no bug)
LOCATION: Forward/bv_solver/forms_logc_muh.py:95-145, 290, 925, 1120
TRIGGER: n/a.

EVIDENCE: `_resolve_mu_h_index` is a pure module-level function with NO `@functools.cache` / `@lru_cache` decorator and NO module-level cache dict. It re-derives the index from `z_vals` (or `roles`) every call. Confirmed by reading the function body — no global state read/write. Three call sites (lines 290, 925, 1120) all re-invoke with current `z_vals`. ✓

---

## Q10: Constants vs floats for k0 / alpha / n_e — ALL MUTABLE FE OBJECTS

SEVERITY: n/a (no bug)
LOCATION: Forward/bv_solver/forms_logc_muh.py:522-537
TRIGGER: n/a.

EVIDENCE:
- Line 522-524: `k0_j = fd.Function(R_space, ...); k0_j.assign(float(rxn["k0_model"]))` — `fd.Function`, mutable via `.assign()`.
- Line 525-527: `alpha_j = fd.Function(R_space, ...); alpha_j.assign(...)` — `fd.Function`, mutable.
- Line 537: `n_e_j = fd.Constant(float(rxn["n_electrons"]))` — `fd.Constant`, mutable via `.assign()`.

All are FE objects, not baked-in Python floats. `set_reaction_k0_model` (anchor_continuation.py:246) operates on `ctx["bv_k0_funcs"][j]` which is `bv_k0_funcs[j]` = the same Function object used in the residual. **k0 ladder is wired.** ✓

NOTE: `bv_k0_funcs`/`bv_alpha_funcs` are populated BEFORE the disabled-reaction skip (line 532-535), so even a disabled reaction (`k0 ≤ 0`) gets a populated Function — ladders can "un-disable" via assignment.

---

## A. UFL min_value/max_value vs Python min/max — VERIFIED UFL-SAFE

SEVERITY: n/a
LOCATION: forms_logc_muh.py:338-341, 402, 224-227, 374-377, 452, 1316-1319, etc.
TRIGGER: n/a.
EVIDENCE: Grep shows ALL eta/phi/log clamps use `fd.min_value` / `fd.max_value` (UFL operators with well-defined one-sided derivatives). No bare Python `min(`/`max(` inside UFL expressions. The Python-level `max(c0_i, _C_FLOOR)` at line 825 / 982 operates on numeric values BEFORE wrapping in `fd.Constant(...)`, so it's fine.

---

## B. Shared-θ closure self-consistency for multiple bikerman counterions — VERIFIED

SEVERITY: n/a (mathematically self-consistent)
LOCATION: boltzmann.py:204-269
TRIGGER: 2+ bikerman counterions (Cs⁺ + SO4²⁻ deck-baseline production setup).

EVIDENCE: Tracing the closure (call φ ≡ phi_clamped, suppressing it):
- `c_k(φ) = c_b_k · exp(-z_k·φ) · (1 - A_dyn) / denom` where `denom = θ_b + Σ_k' a_k' · c_b_k' · exp(-z_k'·φ)`.
- `Σ_k a_k · c_k(φ) = (1 - A_dyn) · Σ_k a_k · c_b_k · exp(-z_k·φ) / denom = (1 - A_dyn) · (denom - θ_b) / denom`.
- Total packing: `Σ a_i · c_i + Σ_k a_k · c_k = A_dyn + (1 - A_dyn)·(denom - θ_b)/denom`.
- Vacancy `θ(φ) = 1 - total_packing = (1 - A_dyn) · θ_b / denom`.
- At bulk: `denom_bulk = θ_b + Σ a_k c_b_k = 1 - A_dyn_bulk`; `θ(bulk) = (1 - A_dyn_bulk) · θ_b / (1 - A_dyn_bulk) = θ_b`. ✓

The closure is the **algebraic steady-state reduction** of the coupled multi-ion Bikerman problem in the inert-counterion limit. It is exact (not an approximation) under the assumption that counterions are in instantaneous local equilibrium with φ — which is the WHOLE POINT of replacing them with an analytic profile. No breakdown at high concentration; the closure handles saturation correctly via `θ → 0`.

VERDICT: math is correct. The IC side (forms_logc_muh.py:1340-1440 multi-ion branch) uses the same closure consistently.

---

## C. Stern Robin sign convention — VERIFIED CORRECT

SEVERITY: n/a (no bug)
LOCATION: forms_logc_muh.py:666-668
TRIGGER: `use_stern=True` (production: `stern_capacitance_f_m2=0.20`).

EVIDENCE: Line 668: `F_res -= stern_coeff * (phi_applied_func - phi) * w * ds(electrode_marker)`.

Sign-checking: Take Poisson `-∇·(ε∇φ) = ρ`. Weak form residual `F = ∫ε∇φ·∇w dx - ∫ρ·w dx - boundary`. The Robin BC `-ε ∂φ/∂n = C_S·(φ_applied - φ)` on the electrode gives boundary term `+ ∫ C_S·(φ_applied - φ)·w ds` on the RHS, i.e., `- ∫ C_S·(φ_applied - φ)·w ds` on F_res. Line 668 matches exactly: `F_res -= C_S·(φ_applied - φ)·w ds`. ✓

(Sign check confirmed against compatible Poisson term line 644: `F_res += eps_coeff * fd.dot(fd.grad(phi), fd.grad(w)) * dx` — both terms have F_res on LHS=0 convention.)

---

## D. mu_H transformation in NP flux — VERIFIED CORRECT REDUCTION

SEVERITY: n/a (no bug)
LOCATION: forms_logc_muh.py:470-505
TRIGGER: n/a.

EVIDENCE:
- For mu species (H+), line 473: `ideal_grad = fd.grad(ui[i])` where `ui[i]` is the mu_H primary variable. So `ideal_grad = ∇μ_H`.
- Standard log-c flux: `J_i = -D_i · c_i · (∇u_i + em·z_i·∇φ)` (line 476-477 for non-mu species).
- For H+: `∇μ_H = ∇u_H + em·z_H·∇φ` (by definition of μ_H). So `J_H = -D_H · c_H · ∇μ_H` (line 499 or 501) = `-D_H · c_H · (∇u_H + em·z_H·∇φ)`. ✓
- `c_H = ci[mu_h_idx]` is reconstructed at line 338 from `u_exprs[mu_h_idx] = ui[mu_h_idx] - em·z_H·φ = μ_H - em·z_H·φ = u_H`, then clamped and exponentiated. Correct.

Steric activity coupling (line 499 vs 501) adds `+ fd.grad(mu_steric)` to the gradient when `steric_active`. Mathematically: `J_i = -D_i · c_i · (∇μ_i + ∇μ_steric)` which is the Bikerman chemical-potential-gradient flux. Same form for mu and non-mu species — consistent.

---

# VERDICT

**No correctness-breaking bugs found** in the audited slices of `forms_logc_muh.py`, `boltzmann.py`, and `picard_ic.py`. The code is defensively written with explicit guards on Picard singular Jacobians (Q7), Stern bisection bracket sign-changes (Q8), and Picard-fallback IC initialization (Q6).

## Non-blocking findings (recommendations only)

1. **Q5 brittleness (low):** `J_form = fd.derivative(F_res, U)` is computed at line 839 and again at line 386 inside `add_boltzmann_counterion_residual` — works correctly today but is brittle. A future contributor adding any residual term after line 881 without re-deriving J_form would silently desync the Jacobian. Recommend collapsing to a single J_form derivation at the end of `build_forms_logc_muh` after `add_boltzmann_counterion_residual` returns.

2. **Q3 hardening (low):** `phi_clamp` is validated `> 0` only (config.py:417). With `phi_clamp > ~350` on z=±2 species, `exp(2*350)=exp(700)` overflows fp64 to `+inf`, silently zeroing the steric closure denominator and shutting off Newton sensitivity. Recommend adding `phi_clamp ≤ 150` (or similar) upper-bound validation. **No production config currently triggers this** — all defaults are 50.0.

3. **Q2 numerical-only (medium):** The `free_dyn = max(1 - A_dyn, 1e-10)` floor zeros Jacobian rows from the counterion closure in saturated regions, possibly slowing Newton convergence in deep cathodic regimes. The algebraic limit is correct; the impact is purely on convergence speed. No fix needed unless empirically problematic.

4. **Item A reassurance:** All UFL clips use `fd.min_value`/`fd.max_value`; no bare Python `min`/`max` leaks into the UFL graph. Differentiability semantics are well-defined.

## Items VERIFIED CLEAN
- Q1 (eta clip Jacobian — by design),
- Q4 (no Newton-state contamination from Cs⁺/SO4²⁻),
- Q6 (Picard fallback U_prev),
- Q7 (singular Jacobian guard),
- Q8 (Stern bisection bracket check),
- Q9 (mu_h index resolver — no memoization),
- Q10 (k0/α/n_e are mutable FE objects, ladders wire correctly),
- Items A/B/C/D (UFL/closure math/Stern sign/μ_H NP reduction all correct).
