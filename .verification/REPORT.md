# Verification Report — BV Forward-Solver Codepath

**Target:** Production logc_muh BV forward-solver codepath
**Date:** 2026-05-05
**Level:** 1 (Sonnet only)
**Scope:** 15 files, 5,267 lines, 2 subsystems (`Forward/bv_solver/`, `scripts/`)
**Agents:** Sonnet × 6 (run in parallel)
**Chunking:**
- Chunk 1: `forms_logc_muh.py` (1044 lines)
- Chunk 2: `forms_logc.py` + `boltzmann.py` (1336 lines)
- Chunk 3: `grid_per_voltage.py` + `sweep_order.py` + `dispatch.py` (911 lines)
- Chunk 4: `config.py` + `nondim.py` + `mesh.py` + `solvers.py` (588 lines)
- Chunk 5: `validation.py` + `diagnostics.py` + `observables.py` + `__init__.py` (826 lines)
- Chunk 6: `scripts/_bv_common.py` (562 lines)

**Verdict:** **PASS with caveats.** The production logc_muh forward-solver math is correct end-to-end. No bug was found that produces wrong physics on the canonical production call path (`make_bv_solver_params(...formulation='logc_muh', initializer='debye_boltzmann', boltzmann_counterions=[DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC]...)` → `solve_grid_per_voltage_cold_with_warm_fallback`). All 12 warnings are either latent (require non-standard caller patterns), live in adjacent diagnostic/validator code that does not feed back into the solver, or affect callers that already pass the right values.

---

## Summary

The mu_H bookkeeping (`mu_H = u_H + em·z_H·phi` reconstructed at every c_H touch site, `c_H_old` using `phi_prev` for transient correctness), the log-rate Butler–Volmer form, the `exponent_clip=100` clip-on-η-before-α·n_e convention, the analytic Bikerman ClO4⁻ counterion + Stern Robin BC, the C+D continuation orchestrator, and the flag wiring through `_bv_common.py` → `config.py` → dispatcher → `forms_logc_muh.py` are all internally consistent. The two backends share their BV-rate form byte-for-byte except for the documented `_u_expr` substitution. The Bikerman closure formula matches `docs/steric_analytic_clo4_reduction_handoff.md` and the IC/residual sides share the same `c_steric` UFL expression (no double-counting).

The findings below are real, but every one of them either (a) sits behind a fallback path that the production caller chain never hits, (b) lives in diagnostic/validator code rather than in the residual, or (c) is a defensive-validation gap rather than a bug in the math.

---

## Findings

| # | Severity | Location | Issue | Found By |
|---|----------|----------|-------|----------|
| 1 | warning | `forms_logc.py:702`, `forms_logc_muh.py:800` | Stale `conv_cfg.get("exponent_clip", 50.0)` fallback in `_try_debye_boltzmann_ic[_muh]`. Authoritative default is 100.0. Unreachable on the normal `_get_bv_convergence_cfg` path (which always populates the key); fires only if a test or helper passes a partial `conv_cfg`. clip=50 produces fictitious peroxide currents per `docs/clipping_conventions.md`. | Chunks 1, 2, 4 |
| 2 | warning | `forms_logc.py:211`, `forms_logc_muh.py:262` | Stale `conv_cfg.get("u_clamp", 30.0)` fallback in main build-forms. Authoritative default is 100.0. Same unreachable-in-production story as #1, but the inline doc at `forms_logc.py:198–202` warns "widen to u_clamp=100 for V_RHE > +0.30 V" — so the fallback would actively bind right where production runs. | Chunks 1, 2, 4 |
| 3 | warning | `FluxCurve/bv_point_solve/__init__.py:723`, `bv_point_solve/forward.py:318` | These callers of `validate_solution_state` pass `is_logc=True` but **omit** `mu_species=ctx.get('mu_species')` and `em=ctx['nondim'].get('electromigration_prefactor', 1.0)`. For a `logc_muh` context the validator reads raw `mu_H` DoFs and exponentiates without the phi correction → reported H⁺ concentration off by `exp(em·z_H·phi)` (many decades inside the Debye layer). Affects diagnostic checks only; the residual itself is correct. CLAUDE.md explicitly flags this gotcha. | Chunk 5 |
| 4 | warning | `validation.py:53` | `exponent_clip` is a declared keyword arg but **never read** in the 202-line function body. The W1 ("clip saturation") check listed in the docstring is unimplemented. All callers faithfully pass the value from `ctx["_diag_exponent_clip"]`; nothing consumes it. | Chunk 5 |
| 5 | warning | `validation.py:181–195` | W5 cation-depletion mask: applies a coordinate-derived boolean mask (sized to vertex count) to a concentration DOF array. Sizes agree only for CG1. Latent for the production stack (CG1) but breaks if `order > 1` is ever used. | Chunk 5 |
| 6 | warning | `config.py:65, 72, 119, 135` | `_VALID_FORMULATIONS` still lists `"concentration"` and `_default_bv_convergence_cfg`/`_get_bv_convergence_cfg` default to it. The concentration backend was removed in the May 2026 cleanup. Dispatcher silently falls through to `logc` for unknown formulations, so this never errors — but the stored config can claim a backend that no longer exists. | Chunk 4 |
| 7 | warning | `config.py:24–26` | `_get_bv_cfg` skips per-element range validation when `alpha` is supplied as a `list`/`tuple`. `_get_bv_reactions_cfg:316–318` does validate each entry — inconsistent. Could accept `alpha=[0.5, 1.5]` on the legacy single-reaction path. | Chunk 4 |
| 8 | warning | `config.py:285–326` | `cathodic_species` and `anodic_species` parsed via bare `int(...)` with no `[0, n_species)` bounds check (the `cathodic_conc_factors` indices DO get bounds-checked). Out-of-range index would surface as an obscure UFL `IndexError` at form-assembly time. | Chunk 4 |
| 9 | warning | `config.py:224` | `_get_bv_boltzmann_counterions_cfg` accepts `z=0` silently. Would produce `exp(0·phi)=1`, a constant Poisson source — physically meaningless. Bikerman closure was derived for `z<0`; even `z>0` should at minimum warn. | Chunk 4 |
| 10 | warning | `grid_per_voltage.py` Phase 2 (lines 503–600) | Phase 2 warm-walk uses `anchor_lo=cold_idxs[0]`, `anchor_hi=cold_idxs[-1]`. Cold-failed interior points (`anchor_lo < k < anchor_hi`) are never visited by either the cathodic or anodic walk and remain `method="cold-failed"` silently. Practical risk low for the production grid (failures cluster above `anchor_hi`); behavior is silent and undocumented. | Chunk 3 |
| 11 | warning | `_bv_common.py:407–411, 421` | `DEFAULT_CLO4_BOLTZMANN_COUNTERION{,_STERIC}` uses `phi_clamp=50.0`. Doesn't bite at `V_RHE ≤ +1.0 V` (clamp activates ~+1.28 V physical), but inconsistent with the raised `exponent_clip=100`/`u_clamp=100`. | Chunk 6 |
| 12 | warning | `_bv_common.py:313–314, 444–445` | Factory defaults `E_eq_r1=0.0`, `E_eq_r2=0.0`. Most production-relevant scripts (e.g. `peroxide_window_3sp_bikerman_muh.py`, `peroxide_window_stern_test.py`, `anodic_cold_start.py`, `ic_refinement_study.py`) omit these kwargs and silently run with `E_eq=0`. CLAUDE.md Hard Rule 4: "Use physical `E_eq` (R1 = 0.68 V, R2 = 1.78 V vs RHE), never `E_eq = 0`." Factory should export `E_EQ_R1=0.68`/`E_EQ_R2=1.78` as named constants and/or warn when omitted with a non-zero formulation. | Chunk 6 |
| 13 | note | `forms_logc.py:310` vs 329/332 | Sign convention in inline comments: line 310 writes physical flux `J_i = -D·c·(∇u + z∇φ)` (with leading minus); lines 329/332 label `Jflux = D·c·(...)` as `J` (no minus). Code is correct (`Jflux = −J_i` physically and `F_res += dot(Jflux, ∇v) dx` is the right IBP form), but the inconsistent labeling could mislead a future maintainer modifying the steric or migration term. | Chunk 2 |
| 14 | note | `boltzmann.py:360–362` (called from `forms_logc_muh.py:589`) | When all counterions are `bikerman` and `skip_bikerman=True`, the loop skips every entry but still re-derives `J_form` from an unchanged `F_res`. Wasted Jacobian computation at form-build time (not solve time). Harmless. | Chunk 1 |
| 15 | note | `grid_per_voltage.py` (entire body) | Adjoint tape annotation suppression is caller-controlled, not internal. Production driver wraps the call in `with adj.stop_annotating():`. The debye_boltzmann ICs self-suppress; the linear-phi IC and z-ramp solves do not. By design but easy to miss if a new caller forgets the wrapper. | Chunk 3 |
| 16 | note | `grid_per_voltage.py:66–76` | `PerVoltagePointResult` stores `U_data` and `diagnostics` but no assembled observables (current density, peroxide current). Production driver pulls them via `per_point_callback`. Future-facing improvement, not a current bug. | Chunk 3 |
| 17 | note | `grid_per_voltage.py:379–388` (`_solve_warm`) | `_build_for_voltage` computes an IC at `V_target` immediately overwritten by `_restore_U(anchor_snap)`. For the debye_boltzmann IC (a Picard solve) this is non-trivial wasted work per warm-walked voltage. No correctness impact. | Chunk 3 |
| 18 | note | `Forward/bv_solver/sweep_order.py` (whole file) | Neither `_build_sweep_order` nor `_apply_predictor` is used by the C+D path; they are re-exported only by `FluxCurve/bv_point_solve/predictor.py`. Both are correctly implemented. | Chunk 3 |
| 19 | note | `config.py:323` (also legacy path) | `k0` parsed as `float(...)` with no nonnegative check. Negative k0 would yield a negative BV flux — physically meaningless. | Chunk 4 |
| 20 | note | `nondim.py:8–101` | No idempotency guard on `_add_bv_scaling_to_transform`. If the function were ever called twice on the same scaling dict, it would re-scale `bv_k0_model_vals`. Currently called exactly once per build path; latent risk if call graph changes. | Chunk 4 |
| 21 | note | `nondim.py:30` | `thermal_voltage_v` fallback `0.02569` (RT/F at ~297.8 K) is a magic literal. Production `build_model_scaling` always populates the key from `temperature_k`; fallback is unreachable in production but stale at non-standard temperatures. | Chunk 4 |
| 22 | note | `Forward/bv_solver/solvers.py` (entire 21-line stub) | After the May 2026 cleanup, the actual PETSc/SNES options live in `scripts/_bv_common.py` (`DEFAULT_SOLVER_PARAMS` — SNES newtonls + L2 line search + direct LU/MUMPS, tolerances reasonable). The `solvers.py` stub is misleading as a scope entry point. | Chunk 4 |
| 23 | note | `diagnostics.py` (whole file) | Mass balance (`∫_Ω r_i dx − ∫_∂Ω_electrode J_i·n dA`) and Stern surface-charge consistency (`σ = C_S·(φ_metal − φ_solution)`) checks are not implemented. The file's docstring accurately scopes itself to "failure-mode information"; the planned checks live in `docs/physics_validation_plan.md`. Scope gap, not incorrect logic. | Chunk 5 |
| 24 | note | `__init__.py:58–67` | Six private `_get_bv_*` / `_add_bv_*` helpers imported into `__init__.py` but not in `__all__`. Accessible as `Forward.bv_solver._get_bv_cfg` etc. Looks like a leftover from before logic was split into submodules. | Chunk 5 |
| 25 | note | `validation.py` (NaN/Inf coverage) | NaN DoFs are not detected by any check inside `validate_solution_state` (`NaN < threshold` and `NaN > threshold` both evaluate False). SNES upstream catches NaN before the validator runs in practice; defense-in-depth gap only. | Chunk 5 |
| 26 | note | `_bv_common.py:276` vs `config.py:113, 132` | Factory `_make_bv_convergence_cfg` writes `conc_floor=1e-12`; `_default_bv_convergence_cfg`/`_get_bv_convergence_cfg` fallback is `1e-8`. Factory always wins through the params dict; mismatch only matters for tests passing manual params. | Chunk 6 |

---

## Agreement Analysis

- **Agreed on (all chunks):** The production logc_muh stack is mathematically correct on the canonical call path. mu_H bookkeeping, log-rate BV form, `exponent_clip=100` convention, Stern Robin BC, Bikerman analytic ClO4⁻ counterion (no double-count, shared `c_steric` between Poisson and NP saturation), debye_boltzmann IC with composite-ψ + multispecies-γ cancellation, C+D orchestrator (cold + warm-walk, NaN-safe per-voltage isolation, deterministic seeding), and flag wiring through `_bv_common.py → config.py → dispatch.py → forms_logc_muh.py` are all consistent.

- **Disagreement: severity of the IC `exponent_clip=50.0` fallback.**
  - Chunks 1 (forms_logc_muh) and 2 (forms_logc + boltzmann) classified the stale literal as **note** because the fallback is unreachable on the normal `_get_bv_convergence_cfg` path (which always populates the key).
  - Chunk 4 (config) classified it as **critical**, arguing the IC sets Newton's starting iterate and a wrong value would directly corrupt the cold-start solve.
  - **Resolution:** Reconciled to **warning** (Issue 1 above). Chunks 1 and 2 are right that production is unaffected because `build_forms_logc[_muh]` always passes a fully populated `conv_cfg`; chunk 4 is right that the magic literal is dangerous if a test or helper builds `conv_cfg` manually. The fix is one-line and should be done — but it is not currently producing wrong production output.

- **No other tier-internal disagreements.** Chunks generally complemented each other; cross-chunk interface checks (dispatcher imports vs. forms function names, factory keys vs. config parser keys, NP/Poisson saturation agreement, observable surface marker vs. residual surface marker) all aligned.

---

## Recommended Fixes (Priority Ordered)

1. **Issue 12 (factory `E_eq_r1/r2=0.0` defaults)** — Highest impact. Most production-relevant scripts run with unphysical `E_eq=0` because they omit the kwargs. Either change the factory defaults to the physical values, or export `E_EQ_R1=0.68`/`E_EQ_R2=1.78` as named constants and add a warning when omitted. **CLAUDE.md Hard Rule 4 explicitly requires this.**

2. **Issues 1, 2 (stale 50.0 / 30.0 fallbacks in `forms_logc[_muh].py`)** — One-line fixes per file. Either bump fallbacks to match `_default_bv_convergence_cfg` (100.0 / 100.0) or drop the `.get(..., default)` and use `conv_cfg["..."]` so missing keys fail loudly the same way the rest of the build path does.

3. **Issue 3 (validator omits `mu_species`/`em` for muh contexts)** — Two-line fix at each caller in `FluxCurve/bv_point_solve/`. CLAUDE.md already documents this footgun. Without the fix, validator-flagged H⁺ violations in muh runs are physically meaningless.

4. **Issue 4 (`exponent_clip` is a dead parameter on validator)** — Either implement W1 (clip-saturation warning) or remove the parameter from the signature so it stops misleading callers.

5. **Issue 9 (Boltzmann counterion `z=0` silently accepted)** — One-line guard.

6. **Issues 7, 8 (alpha-list and species-index bounds checks)** — Defensive validation gaps. Easy to add.

7. **Issue 6 (`"concentration"` still in formulation whitelist)** — Drop it, change defaults to `"logc"`. The dispatcher's silent fallthrough hides the inconsistency today.

8. **Issue 11 (factory `phi_clamp=50.0`)** — Bump to 100.0 for consistency with `exponent_clip` and `u_clamp`.

9. **Issue 10 (interior cold-failed gap in Phase 2)** — Document the behavior, or extend Phase 2 to walk gaps from the nearest-converged neighbor in either direction.

Issues 5, 13–26 are notes — fix opportunistically, no rush.

---

## Per-Chunk Reports

Detailed findings, evidence, and correctness arguments are in:
- `.verification/sonnet-chunk-1-muh-report.md`
- `.verification/sonnet-chunk-2-logc-bz-report.md`
- `.verification/sonnet-chunk-3-orchestrator-report.md`
- `.verification/sonnet-chunk-4-config-report.md`
- `.verification/sonnet-chunk-5-obs-diag-report.md`
- `.verification/sonnet-chunk-6-factory-report.md`
