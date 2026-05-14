# Chunk B Verification Report
## Verifier: claude-sonnet-4-6
## Scope: forms_logc_muh.py, boltzmann.py, picard_ic.py
## Date: 2026-05-13

---

## Summary of Findings

8 issues found, ranging from critical misattributions to minor label inaccuracies.

---

## Claim-by-Claim Verification

---

### Claim 1: `build_context_logc_muh` at forms_logc_muh.py:152

**Doc**: "Constructs mixed FE space W = V_u × V_μH × V_φ; Creates U = Function(W) as Newton iterate; Creates U_prev for SER pseudo-time"

**SEVERITY: warning — partial inaccuracy in attribution**

**LOCATION**: forms_logc_muh.py:152–163; forms_logc.py:59–125

**DESCRIPTION**:
- Line number 152 is CORRECT — `build_context_logc_muh` is defined there.
- The function does NOT directly construct W, U, or U_prev. It delegates to `build_context_logc` (forms_logc.py:59) and only adds `ctx["logc_muh_transform"] = True`. The actual W/U/U_prev construction happens in forms_logc.py:105–108.
- The mixed space is `MixedFunctionSpace([V_scalar] * (n_species+1))` — all CG elements of the same order. The doc notation "V_u × V_μH × V_φ" is conceptually correct but misleadingly implies distinct spaces. In code it is n+1 copies of the same V_scalar.
- "U_prev for SER pseudo-time" — U_prev is created and used as a pseudo-time lag variable; this description is functionally correct.

**EVIDENCE** (forms_logc_muh.py:161–163):
```python
def build_context_logc_muh(solver_params, *, mesh=None):
    ctx = build_context_logc(solver_params, mesh=mesh)
    ctx["logc_muh_transform"] = True
    return ctx
```
The doc's attribution of W/U/U_prev construction to this function is stale — those live in `build_context_logc`.

---

### Claim 2: `build_forms_logc_muh` at forms_logc_muh.py:166

**Doc**: Calls build_model_scaling (from nondim.py); calls _resolve_mu_h_index; calls build_steric_boltzmann_expressions with (1−A_dyn) factor for Cs+ AND SO4²⁻; calls add_boltzmann_counterion_residual; assembles F_res = Poisson + NP + BV; J_form = derivative(F_res, U); NonlinearVariationalProblem; NonlinearVariationalSolver → ctx['_last_solver'].

**SEVERITY: critical — NonlinearVariationalProblem/Solver NOT in build_forms_logc_muh**

**LOCATION**: forms_logc_muh.py:166; anchor_continuation.py:1101–1107

**DESCRIPTION**:
The doc states `build_forms_logc_muh` "Wraps NonlinearVariationalProblem(F, U, bcs, J)" and "Builds NonlinearVariationalSolver(problem, options=SNES_OPTS_CHARGED)" and "Stores → ctx['_last_solver']". This is **false**.

`build_forms_logc_muh` assembles F_res, J_form, bcs, and stores them in ctx — but it does NOT construct NonlinearVariationalProblem or NonlinearVariationalSolver. Those are built in `solve_anchor_with_continuation` in anchor_continuation.py:1101–1107, **after** the dispatch calls build_forms. `ctx['_last_solver']` is set by anchor_continuation.py, not by build_forms_logc_muh.

This affects the convergence-mechanism table claim: Layer 3 says SNES_OPTS_CHARGED is applied inside build_forms_logc_muh. It is not — `solve_anchor_with_continuation` reads solver options from `sp[10]` at line 1099 and passes them when constructing the solver.

The label "SNES_OPTS_CHARGED" does not appear anywhere in the codebase at all (confirmed by grep). The solver parameters are sourced from `params_block = sp[10]` at anchor_continuation.py:1095 with `snes_error_if_not_converged` as the only default.

**Sub-items that ARE correct**:
- Line 166 for `build_forms_logc_muh` definition — CORRECT.
- Calls `build_model_scaling` — but from `Nondim.transform`, not `nondim.py`. The doc says "nondim.py"; the import is `from Nondim.transform import build_model_scaling`. There is also a local `./nondim.py` but `build_model_scaling` does not live there.
- Calls `_resolve_mu_h_index` — CORRECT (line 290).
- Calls `build_steric_boltzmann_expressions` — CORRECT (line 418).
- Calls `add_boltzmann_counterion_residual` — CORRECT (line 881, with `skip_bikerman=True`).
- Assembles F_res = Poisson + NP + BV — CORRECT.
- `J_form = derivative(F_res, U)` — CORRECT (line 839).

**EVIDENCE** (anchor_continuation.py:1101–1107):
```python
problem = fd.NonlinearVariationalProblem(
    ctx["F_res"], ctx["U"], bcs=ctx["bcs"], J=ctx["J_form"]
)
solver = fd.NonlinearVariationalSolver(
    problem, solver_parameters=solve_opts
)
ctx["_last_solver"] = solver
```
These lines are in `solve_anchor_with_continuation`, NOT in `build_forms_logc_muh`.

---

### Claim 2b: build_model_scaling "from nondim.py"

**SEVERITY: note — wrong module attribution**

**LOCATION**: forms_logc_muh.py:61

**DESCRIPTION**: The doc says "Calls build_model_scaling (from nondim.py)". The actual import is:
```python
from Nondim.transform import build_model_scaling
```
There is a local `Forward/bv_solver/nondim.py` that provides `_add_bv_scaling_to_transform` and `_add_bv_reactions_scaling_to_transform`, but `build_model_scaling` itself comes from the top-level `Nondim/transform.py`. The doc attribution is wrong.

---

### Claim 2c: (1−A_dyn) factor for "Cs+ AND SO4²⁻"

**SEVERITY: warning — ambiguous but technically defensible**

**LOCATION**: boltzmann.py:204–254

**DESCRIPTION**: The doc says `build_steric_boltzmann_expressions` applies "(1−A_dyn) factor for Cs+ AND SO4²⁻". In the code, `free_dyn = max(1.0 - A_dyn_local, 1e-10)` is the SHARED free-volume factor applied to ALL bikerman counterions via the same `free_dyn` variable (line 206, 254). This is correct — every bikerman ion in the list gets the same `(1-A_dyn)` numerator. However, the (1-A_dyn) factor is not individually listed as "(for Cs+) AND (for SO4²⁻)" in two separate code paths; it is computed once and applied to all. The claim is correct in spirit but slightly misleading about the implementation structure.

---

### Claim 3: `set_ic_debye_boltzmann_logc_muh` at forms_logc_muh.py:997

**Doc**: "Calls picard_outer_loop_general (in picard_ic.py); 2×2 scalar Picard on (ψ_S, ψ_D); composite-ψ + multispecies-γ seed; falls back to linear_phi if Picard oscillates"

**SEVERITY: warning — description is partially stale and imprecise**

**LOCATION**: forms_logc_muh.py:997; picard_ic.py:1254

**Line number**: 997 is CORRECT for `set_initial_conditions_debye_boltzmann_logc_muh`.

**Description issues**:

1. **"2×2 scalar Picard on (ψ_S, ψ_D)"** — This is incorrect. The Picard loop does NOT iterate on (ψ_S, ψ_D) as the primary unknowns. It iterates on the BV reaction rates R_1, R_2 (or more generally R_j). ψ_S and ψ_D are DERIVED from the Stern-split closure per iteration given the current H_o. The primary variables are the surface reaction rates / outer-region concentrations. ψ_D is solved from a bisection (solve_stern_split) not from the Picard itself. This is a meaningful description error.

2. **"2×2 scalar Picard"** — This is true for the legacy two-reaction path (picard_outer_loop), but `_try_debye_boltzmann_ic_muh` actually calls `picard_outer_loop_general` (picard_ic.py:1267), which is the generalized N-reaction loop. For the production parallel-2e/4e stack, N=2, so it is 2×2, but this is a topology coincidence, not the description of the general algorithm.

3. **"falls back to linear_phi if Picard oscillates"** — The fallback is to `set_initial_conditions_logc_muh` (forms_logc_muh.py:1039), which is the muh linear-phi IC. The label "linear_phi" from dispatch.py's initializer routing is correct as shorthand. The fallback fires on any `_try_debye_boltzmann_ic_muh` failure (not just oscillation): also fires on n<3, no_boltzmann_counterion, mu_h_idx_unsupported, and singular Picard Jacobian failures.

4. **"composite-ψ + multispecies-γ seed"** — CORRECT. The IC seeds the composite BKSA ψ profile (psi_gc for ideal, or psi_zone1/psi_zone2 for Bikerman) plus gamma_psi multispecies activity coefficient.

---

### Claim 4: `build_steric_boltzmann_expressions` at boltzmann.py:91

**Doc**: "Implements Tresset closure (Tresset Eq. 19) with (1−A_dyn) for Cs+ AND SO4²⁻; Closed-form functional of φ (no Newton state)"

**SEVERITY: note — Tresset label is in writeups only, not in boltzmann.py code or docstring**

**LOCATION**: boltzmann.py:91

**Line number**: 91 is CORRECT.

**Tresset reference**: The function docstring and code in boltzmann.py contain NO reference to "Tresset", "Eq. 19", or any paper citation. The Tresset connection is documented only in `writeups/May13th/analytic_counterion_derivation.tex` (which shows the closure coincides with Tresset 2008 Eq. 19). The code docstring says:
```
This is the steady-state algebraic reduction of the 4sp dynamic Bikerman
problem for an inert counterion under the sign-corrected Bikerman chemical
potential. See docs/steric_analytic_clo4_reduction_handoff.md
```
No Tresset citation in the source code. The doc's "Tresset Eq. (19)" annotation is aspirational/derived — it correctly describes the mathematical relationship but the function is not labeled that way in the code.

**"Closed-form functional of φ (no Newton state)"** — CORRECT. The counterions are eliminated from the Newton state vector. `build_steric_boltzmann_expressions` returns UFL expressions depending only on `phi` (the Newton-state phi subfunction) without adding new unknowns.

---

### Claim 5: `add_boltzmann_counterion_residual` at boltzmann.py:272

**SEVERITY: PASS**

**LOCATION**: boltzmann.py:272

Line 272 is CORRECT for `add_boltzmann_counterion_residual`. Function does what the doc claims: appends Boltzmann counterion residuals to ctx['F_res'] and re-derives J_form. It uses `skip_bikerman=True` when called from `build_forms_logc_muh` (forms_logc_muh.py:881) so that bikerman entries handled by `build_steric_boltzmann_expressions` are not double-counted.

---

### Claim 6: `_build_eta_clipped` — clip on η_scaled BEFORE α·n_e

**SEVERITY: PASS — confirmed correct**

**LOCATION**: forms_logc_muh.py:388–403

The function computes:
```python
eta_scaled = bv_exp_scale * eta_raw      # scale applied first
# clip is applied to eta_scaled
return fd.min_value(fd.max_value(eta_scaled, -clip_val), clip_val)
```
And then in the BV rate expression the clipped eta is used as:
```python
log_cathodic = ... - alpha_j * n_e_j * eta_j   # eta_j is already clipped
```
The clip is definitively applied to `eta_scaled` (the dimensionless overpotential = bv_exp_scale × η_raw) BEFORE the α·n_e multiplication. Hard Rule #2 is respected.

In picard_ic.py, `_eta_clipped` (line 47–70) mirrors this: "The clip is applied to the scaled raw eta, NOT to `alpha * n_e * eta`; see docs/clipping_conventions.md." — CONSISTENT.

---

### Claim 7: exponent_clip = 100

**SEVERITY: PASS — confirmed correct**

**LOCATION**: forms_logc_muh.py:871; picard_ic.py:1185

`exponent_clip` is read from `conv_cfg["exponent_clip"]` (forms_logc_muh.py:401). The default value in the Picard code is 100.0 (picard_ic.py:1185). The stored diagnostic value `_diag_exponent_clip` reflects what was used. Hard Rule #2 is correctly implemented.

---

### Claim: Layer 2 (debye_boltzmann IC) seeds ψ_S + ψ_D structure

**SEVERITY: PASS — confirmed correct**

The IC constructs both Stern (psi_S) and diffuse (psi_D) components via `solve_stern_split` in picard_ic.py and the BKSA composite-psi profile. The phi_init_expr includes both:
- `phi_init_expr = ln(H_outer/c_clo4_bulk) + psi` where psi is the composite Gouy-Chapman/BKSA profile (≈ psi_D profile)
- phi_surface accounts for psi_S when Stern is active

---

### Claim: Layer 4 (Bikerman closure — Cs+ and SO4²⁻ never enter Newton state)

**SEVERITY: PASS — confirmed correct**

`build_steric_boltzmann_expressions` returns pure UFL expressions (no new Function/FunctionSpace added to W). Both Cs+ and SO4²⁻ bikerman bundles are UFL expressions of the existing `phi` subfunction. They enter the Poisson residual additively. Newton's state vector remains unchanged.

---

### Cross-Chunk Reference: ctx['_last_solver'] for Chunk C (anchor_continuation.py)

**SEVERITY: critical (attribution) — correctly confirmed for Chunk C**

`ctx['_last_solver']` IS set correctly (anchor_continuation.py:1107 and 1828), and is available for `set_stern_capacitance_model` in Stage 2. However, the doc incorrectly says it is set by `build_forms_logc_muh` — it is set by the orchestrator in anchor_continuation.py. Chunk C's use of `ctx['_last_solver']` is valid; only the doc's claim about WHERE it is set is wrong.

---

## Issue Registry

| # | Severity | File:Line | Claim | Reality |
|---|----------|-----------|-------|---------|
| 1 | warning | forms_logc_muh.py:152 | build_context_logc_muh constructs W/U/U_prev | Delegates entirely to build_context_logc; only adds logc_muh_transform flag |
| 2 | **critical** | forms_logc_muh.py:166 | build_forms_logc_muh wraps NonlinearVariationalProblem + NonlinearVariationalSolver → ctx['_last_solver'] | These are in anchor_continuation.py:1101–1107, NOT in build_forms_logc_muh |
| 3 | **critical** | docs (no code ref) | "SNES_OPTS_CHARGED" | String does not exist in codebase; solver options sourced from sp[10] with one default |
| 4 | note | forms_logc_muh.py:61 | build_model_scaling "from nondim.py" | Imported from Nondim.transform, not Forward/bv_solver/nondim.py |
| 5 | warning | forms_logc_muh.py:997 | "2×2 scalar Picard on (ψ_S, ψ_D)" | Picard iterates on BV rates R_j; ψ_S/ψ_D are derived per-iter via Stern bisection, not primary Picard unknowns |
| 6 | warning | forms_logc_muh.py:997 | fallback "if Picard oscillates" | Fallback fires on any failure: n<3, no counterion, mu_h_idx mismatch, singular Jacobian — not just oscillation |
| 7 | note | boltzmann.py:91 | "Tresset Eq. (19)" annotation | No Tresset citation in boltzmann.py code or docstrings; relation established only in external writeup |
| 8 | warning | dispatch.py:82,89 | Doc says build_context:82 and build_forms:89 | dispatch.py lines confirmed correct — PASS |

---

## VERDICT: ISSUES FOUND

Two critical issues, three warnings, two notes.

**Critical #1** (forms/doc mismatch): The doc attributes `NonlinearVariationalProblem`, `NonlinearVariationalSolver`, and `ctx['_last_solver']` to `build_forms_logc_muh`. These are actually created in `solve_anchor_with_continuation` (anchor_continuation.py:1101–1107). This is the most significant structural error in the doc — it suggests the solver is built at form-build time, but it is actually built later in the Stage 1 orchestration.

**Critical #2** (phantom constant): "SNES_OPTS_CHARGED" appears nowhere in the codebase. The solver uses options from `sp[10]` (the params dict) with `snes_error_if_not_converged=True` as the only unconditional default.

All other functional claims (eta_scaled clip ordering, (1-A_dyn) factor, ψ_S + ψ_D IC structure, closed-form counterion closure, fallback to muh linear-phi IC, line number accuracy for most functions) are correct.
