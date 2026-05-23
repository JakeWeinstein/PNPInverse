# Sonnet Verification Sweep — Master Summary

**Date:** 2026-05-22
**Scope:** 13 parallel Sonnet agents auditing every component of the production
PNP-BV forward pipeline (formulation=`logc_muh`, parallel 2e/4e ORR, Bikerman
counterions, Stern Robin BC, `debye_boltzmann` IC, K⁺/SO₄²⁻ deck).
**Method:** Static analysis only. No code executed.

---

## TL;DR

The forward pipeline is **doing what it says it's doing**. No correctness-blocking
bugs were found by the Sonnet sweep. **Twelve of thirteen components got a PASS
(with at most LOW-severity advisories)** and one (forms residual) got FLAG-MINOR
for a dead `u_clamp=30.0` fallback that never triggers in production.

The single most important finding is **cross-agent corroboration that the Hard
Rule #7 fix (A_DEFAULT → physical Bikerman radii for O₂/H₂O₂/H⁺) is staged in
the working tree but uncommitted** — agents 03, 05, 08, 11 independently
confirmed this. A `git stash` or stale checkout would silently revert the
production presets. **Commit recommended urgently.**

A handful of MEDIUM-severity diagnostic/observability gaps were surfaced
(missing FEM mass balance, no Bikerman c_max cap check in `validate_solution_state`,
silent fallbacks in `solve_stern_split` and the Picard σ_S path) — none of these
falsify production runs but they erode confidence and should be closed.

---

## Per-component verdicts

| # | Component | Files | Verdict | Notes |
|---|-----------|-------|---------|-------|
| 01 | Residual (non-BV) | `forms_logc_muh.py` | **FLAG-MINOR** | Dead `u_clamp=30.0` fallback (lines 326, 274); μ_H change-of-variable faithfully implemented; Poisson ρ correct; H⁺ NP flux linear in ∇μ_H. |
| 02 | BV reactions + topology | `forms_logc_muh.py`, `_bv_common.py` | **FLAG-MINOR** | Reactions list correct (R2e E°=0.695, R4e E°=1.23); log-rate BV correct; clip=100 unclipped across production grid; stale comment values in `clipping_conventions.md` and `_bv_common.py:118` use legacy R1/R2 sequential numbers. |
| 03 | Initial conditions | `picard_ic.py`, IC paths in forms | **PASS** | Composite-ψ correct (GC, BKSA, multi-ion); μ_H assignment correct (ψ terms cancel exactly when em·z_H=1); IC↔residual Bikerman branch selection consistent. Caveat: A_DEFAULT fix uncommitted. |
| 04 | Boundary conditions | `forms_logc_muh.py`, `anchor_continuation.py` | **PASS** | Stern Robin BC correct (`F_res -= C_S·(φ_app−φ)·w·ds`); runtime bump real (`fd.Constant.assign`); species BC signs verified vs MMS derivation. |
| 05 | Bikerman + multi-ion | `boltzmann.py`, `multi_ion.py` | **PASS w/ MEDIUM (uncommitted fix)** | Formula correct, sign correction applied (`-fd.ln(packing)`), shared-θ closure correct, single-ion degeneration verified. A_DEFAULT fix is staged but uncommitted. |
| 06 | Anchor continuation | `anchor_continuation.py` | **PASS** | Ladder bisection correct; `c_s_ladder`/`kw_eff_ladder` now individually implemented (only pairwise combos NotImplementedError); adjoint `stop_annotating()` hygiene present; Strategy B absent. |
| 07 | Grid solver | `grid_per_voltage.py` | **PASS** | C+D algorithm correct; `solve_grid_with_anchor` lives here (not `anchor_continuation.py`); no aliasing in warm-start (deep copy); Strategy B confirmed removed. |
| 08 | Non-dimensionalization | `nondim.py`, `units.py` | **PASS** | V_T, F, R = exact CODATA; `em=1.0` algebraically guaranteed (`em = (F/RT)·V_T`); Stern C_S nondim factor implemented identically in `nondim.py` and `forms_logc_muh.py`. |
| 09 | Cation hydrolysis | `cation_hydrolysis.py`, `singh2016.py` | **PASS w/ MEDIUM (silent fallback)** | Singh constants exact; β(K⁺, Cu) = −45.61 pm² verified; σ-mapping cathodic sign correct; pKa factor sign correct; H⁺ source / K⁺ sink mass-balanced. Picard σ_S=0.0 fallback is silent — needs `warnings.warn`. |
| 10 | Water ionization | `water_ionization.py` | **PASS** | K_w_eff nondim correct; OH⁻ shadowed (algebraic, not FE DOF); `kw_eff_ladder` now fully implemented; default-off byte-equivalent. Watch: `kw_eff=0` at build with no ladder is silent no-op. |
| 11 | Canonical factory + v10b | `calibration/v10b.py`, `_bv_common.py` | **PASS w/ advisory** | v10b constants correct (Γ_max=0.047, k_des=1.0, C_S=0.20); `SMOKE = V10A_SMOKE` alias clean. `make_bv_solver_params` factory defaults lag production (formulation="logc", initializer="linear_phi", stern=None) — relies on caller overrides. |
| 12 | Validation + observables | `validation.py`, `observables.py`, `rrde_observables.py`, `diagnostics.py` | **PASS w/ MEDIUM (missing checks)** | Disc current, ring current, selectivity, n_e all correct. **Missing**: Bikerman c_max cap check for dynamic H⁺ (W9 needed), FEM global mass balance (∫Ω), charge balance (Σ z_i·c_i). N_COLLECTION inconsistency (0.2237 vs 0.224 in one script). |
| 13 | Dispatch + plumbing | `config.py`, `dispatch.py`, `forms_indexing.py`, `mesh.py`, `sweep_order.py`, `__init__.py` | **PASS** | Routing correct (`logc_muh` → `forms_logc_muh`, `logc` → `forms_logc`, conc backend deleted); `exponent_clip=100`/`u_clamp=100` defaults correct; cross-validation in `_get_bv_convergence_cfg` enforces ablation guards. `validate_solution_state` missing from `__init__.__all__`. |

---

## Cross-agent corroborations (multiple agents flagged same issue)

### 1. Hard Rule #7 fix is staged but uncommitted [HIGH]
**Agents:** 03, 05, 08, 11.
**File:** `scripts/_bv_common.py` (modified, not staged-committed).
**Evidence:** `THREE_SPECIES_LOGC_BOLTZMANN`, `FOUR_SPECIES_LOGC_DYNAMIC_K2SO4`,
`FOUR_SPECIES_LOGC_DYNAMIC` have been updated to use `[A_O2_HAT, A_H2O2_HAT,
A_HP_HAT, A_KPLUS_HAT]` (physical Marcus/Stokes radii) instead of `[A_DEFAULT]*N`.
Numerical verification (agent 05): A_DEFAULT=0.01 ↔ r=14.89 Å, c_max=120 mol/m³;
physical r=2.8 Å (H₃O⁺) ↔ A_HP_HAT=6.645e-5, c_max=18,059 mol/m³ — **150.5× clamp tighter** with A_DEFAULT.
**Risk:** Stash/checkout silently reverts to unphysical radii. Bridge runs cited
in CLAUDE.md §7 (`_phase_D_bridge_corrected_a*.py`) presume the fix is in.
**Action:** `git add scripts/_bv_common.py && git commit -m "fix(steric): replace A_DEFAULT with physical a_nondim for dynamic species (Hard Rule #7)"`.

### 2. Stale comment/doc values referencing legacy R0→R1 sequential topology [LOW]
**Agents:** 02, 13.
**Files:** `docs/solver/clipping_conventions.md:85–89,348` (R2 unclip threshold computed with legacy E_eq=1.78); `scripts/_bv_common.py:118` (comment cites Hard Rule #4 with legacy R1=0.68, R2=1.78); `dispatch.py` comments ("logc_muh experimental"); `__init__.py` docstring (ClO4⁻ in description).
**Risk:** Future readers compute the wrong unclip threshold from the doc. Conclusion (grid unclipped at clip=100) remains correct.
**Action:** Update threshold math to use production E°=0.695/1.23 and replace ClO4⁻ references in docstrings.

### 3. Silent fallbacks that swallow degenerate inputs [LOW-MEDIUM]
**Agents:** 03, 09.
**Files:**
- `picard_ic.py:260–270` — `solve_stern_split` falls through to linear-Debye when bisection can't bracket; no log/ctx flag.
- `cation_hydrolysis.py:966–968` — `_cation_hydrolysis_sigma_S_expr` missing from ctx silently substitutes `Constant(0.0)`, giving ΔpKa=0.
**Risk:** Quiet correctness hazard for any future code path that builds a partial ctx.
**Action:** Add `warnings.warn` (or `logger.warning`) at both fallthroughs.

---

## Other findings by severity

### MEDIUM

- **(12.1) No Bikerman c_max check for H⁺ in `validate_solution_state`** — `validation.py` W2 uses `c > c_bulk·5x` which won't fire on H⁺ pileup at pH 4 (c_bulk_H ≈ 10⁻⁴ mol/m³ trivially small). Needs a new W9 keyed to `c_max · a_nondim ≥ 0.9` with `a_nondim` passed per species.
- **(12.2) No FEM domain mass balance or charge balance** — `diagnostics.py` reports a *Langmuir surface ODE* balance (5e-13 cited in MEMORY) but no `∫Ω ∂c/∂t + ∫∂Ω J·n − ∫Ω S = 0` per-species check, and no `Σ z_i·c_i + σ_steric` electroneutrality check. Absent by design — but adding these would close the loop on "is the PDE really solved".

### LOW

- **(01.1) Dead `u_clamp=30.0` fallback** in `forms_logc_muh.py:326` and `forms_logc.py:274`. Never reached because `config.py` always populates 100.0. Trap for hand-crafted `conv_cfg`. Change to 100.0.
- **(06.1) Step 9.5 `warm_start_floor` asymmetry** — used in `solve_lambda_ramp_from_warm_start` but not in `lambda_hydrolysis_ladder` branch of `solve_anchor_with_continuation`. Undocumented inconsistency.
- **(06.2) `LadderExhausted` lacks structured result** — only string-encoded ladder history in exception. MVP-acceptable.
- **(07.1) No mesh y-extent validation at API boundary** — `solve_grid_with_anchor` asserts DOF count match but not `domain_height_hat`. Silent failure if caller rebuilds mesh with different L_eff.
- **(08.1) `A_SO4_HAT = 4.20e-5` hardcoded** vs computed 4.185e-5 for r=2.40 Å (0.35% off). Style inconsistency with K⁺ and OH⁻ (formula-computed). Same for `A_CSPLUS_HAT`.
- **(10.1) `kw_eff=0` silent no-op** when `enable_water_ionization=True` but no `kw_eff_ladder` provided — water-ionization layer is degenerate but the flag is set. Should warn.
- **(11.1) Factory defaults lag production** — `make_bv_solver_params` defaults `formulation="logc"`, `log_rate=False`, `initializer="linear_phi"`, `stern_capacitance_f_m2=None`, `multi_ion_enabled=False`. All five must be overridden for production. No assert or warn for a new caller falling back to the legacy stack.
- **(12.3) N_COLLECTION = 0.2237 vs 0.224** in `_phase_D_plot_vs_slide15.py:54` only. 0.15% drift. Standardize.
- **(12.4) No electrode-side φ sign check** in W3 — symmetric absolute band can't catch a flipped BC.
- **(13.1) `validate_solution_state` not in `__init__.__all__`** — callers import from `validation.py` directly, no breakage, just an API completeness gap.
- **(13.2) Legacy v18_*/v19_* scripts** call `_make_bv_convergence_cfg()` without `formulation`, hitting the stale `"concentration"` default → DeprecationWarning → fallthrough to logc. Documented as non-operational in CLAUDE.md; protection is in place.

### INFO

- **(12.5) Jensen approximation** in `surface_field_means` — `exp(mean(u))` vs `mean(exp(u))`. Acknowledged in comments. Affects `surface_pH_proxy` only, not observable chain.
- **(05.1) ClO₄⁻ Bikerman a_nondim still A_DEFAULT** in `DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC`. Used by legacy scripts only; not production.

---

## Gaps the Sonnet sweep cannot close (need Opus / Codex)

These are the open questions the static Sonnet pass could not authoritatively
answer. Good targets for the expensive sweep:

1. **MMS cross-check on the residual.** Does `forms_logc_muh.py` actually solve
   the PNP-BV system as derived in `docs/solver/mms_pnpbv_muh_multi_ion_stern_derivation.md`?
   Sonnet inspected the residual structurally but did not regenerate the
   weak form from the PDE and compare term-by-term against MMS. Codex with
   the derivation doc loaded would be the right tool.

2. **`forms_logc` vs `forms_logc_muh` byte-equivalence in the limit.** When
   `em·z_H·φ → 0` the two formulations should produce identical solutions.
   Sonnet did not derive or compute this. An MMS or unit-test cross-check
   between the two backends would be high value.

3. **Adjoint annotation correctness.** The codebase wraps continuation in
   `with adj.stop_annotating():`, but Sonnet did not verify the tape contains
   the correct *forward-only* operations versus the parameter-derivative-bearing
   operations. Inverse work resumes from this assumption.

4. **Phase D σ-mapping identifiability — physics vs. code.** Agent 09 confirmed
   the local Stern σ at V_kin is ~10⁵× smaller than Singh's cell-level σ,
   explaining the flat Δ_β loss. But is the σ extraction `C_S·(φ_metal − φ_OHP)`
   the right *physical* σ for Singh's formula? The formula was developed for
   cell-averaged σ, not local diffuse-layer σ. Could be a code-physics mismatch.
   Worth a derivation pass.

5. **Bikerman packing differentiability at saturation.** When `1 − A_dyn` approaches
   `1e-8` (floor), `mu_steric = -ln(packing)` blows up. Does Newton converge through
   this regime? Sonnet didn't profile. Likely needs runs.

6. **Strategy B orphan call sites.** Agent 07 confirmed Strategy B is structurally
   removed. Worth a wider grep across `scripts/` to make sure no driver still
   imports `solve_grid_with_charge_continuation`.

7. **Picard convergence and bisection in `solve_stern_split`.** Sonnet found the
   silent fallback but did not characterize when it fires. Needs runs.

8. **Phase 6α `enable_water_ionization=True` path.** Agent 10 confirmed the
   default-off path is byte-equivalent. The on-path (which Phase 6α used) was
   not exercised. Worth a run-based cross-check.

9. **The Jensen bias in `surface_field_means`** — agent 12 acknowledged but
   didn't quantify. How alkaline-shifted is the surface-pH proxy? Could affect
   reading of step 6/9 results.

10. **Adjoint gradient w.r.t. `Δ_β`, `k_hyd`, `λ`.** With inverse work paused
    (per CLAUDE.md), these aren't immediately needed, but the production
    stack's adjoint pathway across the continuation chain is unverified.

---

## What to commit before the next pass

A short "clean-up" commit closing the corroborated findings:

1. **Commit Hard Rule #7 fix** (`scripts/_bv_common.py` A_DEFAULT → physical radii). HIGH PRIORITY.
2. Bump dead `u_clamp=30.0` fallback to 100.0 in `forms_logc_muh.py:326` and `forms_logc.py:274`.
3. Add `warnings.warn` at `picard_ic.py:260` (Stern fallback) and `cation_hydrolysis.py:966` (σ_S=0.0 silent fallback).
4. Standardize `N_COLLECTION = 0.224` in `_phase_D_plot_vs_slide15.py:54`.
5. Update `clipping_conventions.md:85–89,348` to use production E°=0.695/1.23.
6. Optionally: add W9 Bikerman cap check to `validate_solution_state` (passing `a_nondim` per species).
