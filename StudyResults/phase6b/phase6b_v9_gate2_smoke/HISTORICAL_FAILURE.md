# Phase 6β v9 Gate 2 — cathodic dynamic-K⁺ smoke FAILURE report

**Date:** 2026-05-10
**Stack:** `phase6b_v9_gate2_4sp_dynamic_k2so4_cathodic`
**Outcome:** ALL THREE FALLBACK STEPS FAILED — cathodic 4sp dynamic K⁺
stack does NOT converge at any tested voltage in V_RHE ∈ [-0.40, -0.10] V.
**Plan trigger:** ".claude/plans/write-up-the-formal-joyful-papert.md" §Risk
callouts → "If Gate 2 smoke fails to converge at −0.40 V after the four
fallback steps in 2C, stop and re-plan. Gate 2 failure is the load-bearing
risk; do NOT proceed to Gates 3 + 4 until cathodic 4sp dynamic K⁺ converges.
Queue another GPT round with the smoke result as new evidence."

## What was tested

* **Stack:** `FOUR_SPECIES_LOGC_DYNAMIC_K2SO4` — O₂, H₂O₂, H⁺, K⁺ all
  dynamic; both H⁺ and K⁺ at z = +1 (the exact case Gate 1 unblocked).
  Roles `["neutral", "neutral", "proton", "counterion"]`.
* **Reactions:** `PARALLEL_2E_4E_REACTIONS_4SP` (Ruggiero parallel 2e/4e,
  K⁺ inert in both).
* **Counterion:** `DEFAULT_SULFATE_ANALYTIC_BIKERMAN_FOR_K2SO4` (analytic
  Bikerman, z=-2, c_bulk=83.33 nondim, a_nondim=4.20e-5).
* **Stern:** C_S = 0.10 F/m² production target.
* **Mesh:** Nx=8, Ny=200, beta=3.0, L_eff = 16 µm (domain_height_hat=0.16).
* **Formulation:** `logc_muh`, `bv_log_rate=True`, `enable_water_ionization=True`,
  `initializer="debye_boltzmann"`.
* **Voltage grids:**
  * Primary: V_RHE ∈ {-0.10, -0.20, -0.30, -0.40} V.
  * Decimal refinement: V_RHE ∈ {-0.10, -0.20, -0.30, -0.35, -0.40} V.
  * C_S ladder anchor at V_RHE = -0.40 V with C_S ramp `(1.0, 0.5, 0.25, 0.10)` F/m².

## Per-step results

| Step | Strategy | Result | Method |
|------|----------|--------|--------|
| 1    | Cold + z-ramp + warm-walk @ C_S=0.10 | **0/4 converged** | All `cold-failed`, `z_achieved=0.0` |
| 2    | Decimal V_RHE refinement @ C_S=0.10 | **0/5 converged** | All `cold-failed`, `z_achieved=0.0` |
| 3    | C_S ladder anchor @ V=-0.40 V | **LadderExhausted** at C_S=1.0 F/m² | `cs_history=[(1.0, 'fail')]` |

`z_achieved = 0.0` means the per-voltage z-ramp could not even take the
first step from z=0 — Newton fails immediately on the IC.

## Diagnostic signal

The recurring runtime warning across all three passes:

```
RuntimeWarning: overflow encountered in exp
  out[f"c{i}_surface_mean"] = float(np.exp(u_mean))
```

is the classic signature of an IC for which the dynamic K⁺ field's
`u_K = ln(c_K)` overflows in the surface-mean reconstruction.  K⁺ at
z=+1 with cathodic phi accumulates exponentially at the electrode, and
the symmetric `u_clamp = 100` allows `c_K(0) ~ exp(+|phi|)` which
overflows for `|phi| > ~50`.

## What was tried (and didn't work)

1. **Gate 1A + 1B + 1C role-aware species machinery** landed cleanly
   (619 fast tests pass, byte-equivalent for legacy stacks).  K⁺ at idx
   3 with explicit role "counterion" reaches the IC's bikerman branch.
2. **Gate 2A K2SO4 species + counterion constants** — 9 fast tests pass.
   Bulk electroneutrality verified algebraically.
3. **Gate 2B C_S continuation ladder** — `set_stern_capacitance_model` /
   `c_s_ladder` plumbing landed and validated with 6 fast tests.
4. **IC seed for U.sub(3) (K⁺) in the bikerman branch** — added in
   `forms_logc_muh.py:_try_debye_boltzmann_ic_muh` and
   `forms_logc.py:_try_debye_boltzmann_ic` for parity.  Seeds K⁺ with
   the Boltzmann anchor `u_K_init(y) = ln(c_K_bulk) - z_K · phi_init(y) +
   log_gamma(y)`.  Did NOT rescue convergence — the IC overflow happens
   at the very first z=0 Newton iterate.

## Root cause hypothesis

The matched-asymptotic Picard + composite-ψ IC machinery in
`_try_debye_boltzmann_ic[_muh]` was derived for **1:1 symmetric
electrolytes** (e.g. H⁺:ClO₄⁻ both at 0.1 mol/m³).  The K2SO4 stack
breaks this symmetry in two ways:

1. **Asymmetric salt** (1:2 ratio): H⁺ at 0.1 mol/m³, K⁺ at 199.9 mol/m³,
   SO₄²⁻ at 100 mol/m³ — the Picard's `c_clo4_bulk` anchor (= sulfate
   bulk = 83.33 nondim) doesn't align with H⁺ bulk for the
   `phi_init = ln(H_outer / c_clo4_bulk) + ψ` construction.
2. **Coexisting cations at z=+1** (H⁺ + K⁺ both accumulate at cathode):
   the composite-ψ closure assumes one accumulating species and one
   depleting species per Debye decay.  With two cations sharing the
   accumulation, the closure under-counts the cation packing fraction
   and over-estimates the surface ψ — leading to `u_K(0) = ln(c_K_bulk) +
   |z_K| · ψ_S → ∞` overflow at any meaningful cathodic voltage.

The multi-ion IC branch (`forms_logc_muh.py:1199+`) handles asymmetric
multi-counterion stacks via `Forward.bv_solver.multi_ion`, but it only
fires when **≥ 2 Bikerman counterion entries** are configured.  Our
K2SO4 stack has 1 Bikerman entry (sulfate) + 1 dynamic counterion (K⁺),
so neither the legacy bikerman branch nor the multi-ion branch fits
cleanly.

## Re-plan recommendations (queue for GPT round)

The architectural gap is clear: **the IC needs a "1 Bikerman + 1
dynamic counterion" branch** for the K2SO4 cathodic stack.  Three
candidate paths:

(A) **Promote SO₄²⁻ to the multi-ion path** — make K⁺ also analytic
    (Bikerman entry instead of dynamic NP species) so we have 2
    Bikerman entries and the existing multi-ion IC fires.  Tradeoff:
    K⁺ as analytic Bikerman drops the v9 architecture's premise of
    reading c_K(0) from a dynamic FE field for the cation hydrolysis
    residual.  Would require revisiting the v9 design.

(B) **Extend the multi-ion IC to handle ≥ 1 dynamic counterion** —
    refactor `forms_logc[_muh].py:_try_debye_boltzmann_ic[_muh]` so the
    multi-ion branch includes per-dynamic-species seeding via the
    multi-ion shared-θ closure.  Highest engineering cost; cleanest
    long-term solution.

(C) **Synthesize a 2-Bikerman-equivalent IC for the K2SO4 stack** —
    treat K⁺ as analytic for the IC seed only, then let Newton rebalance
    K⁺ at solve time.  Requires careful handling so the IC seeds K⁺ at
    the multi-ion-Boltzmann surface concentration rather than the naive
    `exp(+|phi|)` overflow.  Mid cost.

Recommendation: queue **(C)** as the next GPT round — minimal
architectural change while the v9 design premise stays intact.  If
Newton still struggles, escalate to **(B)**.

## Status

* Gates 1A, 1B, 1C, 2A, 2B all passing (618 fast + 70 slow tests, 6
  pre-existing fast failures + 8 pre-existing slow failures all
  unrelated).
* Gate 2C BLOCKED on cathodic IC convergence.
* Gate 2D regression test added but skips with `pytest.skip` until
  Gate 2C converges.

## Artifacts

* `iv_curve.json` — full pass-by-pass payload with diagnostics for each
  attempted voltage.
* `diagnostics.json` — diagnostics-only excerpt for grep-friendly
  inspection.
* This `FAILURE.md` — failure summary + re-plan recommendation.
