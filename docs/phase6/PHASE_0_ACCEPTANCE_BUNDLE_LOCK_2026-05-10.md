# Phase 0 — internal acceptance-bundle lock

**Date:** 2026-05-10
**Status:** Locked internally; no external send to the Seitz/Mangan
group is planned.  This document represents Claude's
best-evidence-based working hypothesis for what the deck-aligned
model deliverable should be, locked so v10a can begin without
ambiguous downstream targets.
**Upstream context:**
* `docs/phase6/PHASE_6B_V9_PHASES_A_B_RESULTS_2026-05-10.md` — the
  revised plan (post critique sessions 30 + 31 APPROVED).
* `docs/handoffs/CHATGPT_HANDOFF_31_phase6b-v9-strategic-pivot/` —
  R1 through R5 + FINAL_REVISION.md.

## Rationale for this lock decision

Critique session 31's R5 verdict APPROVED a plan that required
a Phase 0 deliverable contract sent to the experimental group
before any v10a code work began.  The user has stated that no
external send is feasible.  Without external confirmation, the
acceptance bundle becomes Claude's documented working hypothesis,
locked in this file so:

1. v10a code work has unambiguous downstream targets.
2. Future sessions can audit what was assumed without external
   input.
3. If a future communication with the group becomes possible,
   this doc IS the contract draft that can be sent.

## Decision: primary validation target

**Primary target: per-cation RRDE-equivalent H₂O₂ selectivity at
pH ≈ 4**, extracted from `data/EChem Reactor Modeling-Seitz-Mangan/
Brianna/20201024 CP Experiment Data-Code/Summary Data-Error.xlsx`
sheet `Cation Summary Table`.  Rationale:

* This is the deck's headline cation-effect observable (Linsey
  deck slide 9 + 27 frame the cation comparison as a selectivity
  story).
* It's measured per cation (K2, Cs2, Na2, Li2 sulfate) per pH
  bin in the experimental file — a direct experimental anchor.
* The model has all the ingredients to predict it: per-branch
  currents R_2e and R_4e, RRDE collection N=0.224, BV kinetics.
* If the model fails to reproduce this, the cation hydrolysis
  mechanism (v10 architecture) is falsified as the dominant
  driver of the deck's cation effect — that's still a valid
  research finding.

**Secondary targets (supporting evidence, looser tolerances):**

* Ring Onset Potential (`Ring Onset Pot (V, @ 0.01 mA/cm²)`).
* Max Ring Current (`Max Ring Current (mA/cm²)`).
* Number of e⁻ (`Number of e-`).
* ΔpKa ordering (Singh-mechanism check against slide 27 — note
  slide 27 IS Singh's Cu table reproduced, so this is a
  self-consistency check on the model's β parameterization).
* ΔpKa magnitude (also a self-consistency check on β).

## Acceptance criteria

### Primary criterion (must pass for Phase E)
1. **Per-cation max RRDE-equivalent H₂O₂% in V_RHE window** matches
   experimental value within **±10 percentage points absolute**.
2. Met for **all 4 cations** (K, Cs, Na, Li) at the pH ≈ 4 bin.

### Secondary criteria (≥2/3 must pass per cation)
1. **Ring Onset Potential**: match within ±50 mV absolute.
2. **Max Ring Current**: match within ±30% relative, with
   absolute floor 0.01 mA/cm² (below floor = exact-zero comparison).
3. **n_e_rrde**: match within ±0.5 absolute.

### Mechanism self-consistency (must pass for Phase D)
1. **ΔpKa ordering**: predicted ΔpKa magnitudes follow Li < Na < K
   < Cs (every adjacent pair correctly ordered).
2. **ΔpKa magnitude**: mean_i |ΔpKa_model_i − ΔpKa_deck_i| /
   mean_i |ΔpKa_deck_i| ≤ 0.30.
3. **β sign guard**: cathodic pKa lowering preserved (β · σ_local
   cathodic < 0 for all cations).

### Phase E pass rule
* All primary criteria pass for all 4 cations.
* All mechanism self-consistency criteria pass.
* ≥ 2 of 3 secondary criteria pass per cation.

If any of the primary criteria fail: the v10 model + calibrated β
falsifies the cation-hydrolysis-as-dominant-mechanism hypothesis
for the deck's cation effect.  That's a valid research finding;
report and document the falsification pathway (which sub-component
likely fails: C_S mapping, Γ_max/k_des calibration, ORR kinetic
parameters, sulfate closure).

## Phase E.0 data-reduction protocol

Frozen before Phase D fitting:

* **pH bin:** experimental pH ∈ [3.5, 4.5] (the deck's
  cation-comparison condition per Linsey deck slide 9).
  Cation Summary Table rows averaged within bin.  Records
  outside bin reported separately as out-of-target diagnostics.
* **Cycle aggregation:** mean of cycles 1 / 2 / 3 from
  `Brianna_ORR_Data.mat` and `0,1M K2SO4 data 8-15-19.xlsx`
  (where multiple cycles available); std reported as error bar.
  Cation Summary Table values used directly if no cycle
  decomposition is present in the table.
* **V scan window:** model V_RHE ∈ [−0.4, +1.0] V (solver
  C+D convergence window).  Experimental V window from
  Brianna LSV at pH 6.39 ≈ [−0.06, +1.14] V.  **Overlap window
  [−0.06, +1.0]** used for max-extraction of selectivity, ring
  current.
* **Per-observable extraction:** defined below; one function
  per observable; tested at v10a stage on synthetic data
  before any Phase D fit.

## Per-observable extraction functions

| Observable | Source | Extraction |
|---|---|---|
| Max H₂O₂ Selectivity (%) | model output → `max_H2O2_selectivity_in_window` | max of `RRDE-equivalent H₂O₂% = 100·I_disk_ring_branch / I_disk_total_branch` over overlap window |
| argmax V for selectivity | model output → `argmax_V_for_selectivity` | V at which `max_H2O2_selectivity_in_window` occurs |
| Ring Onset Potential | model output → `ring_onset_V_at_0.01_mA_cm2` | first V where `gross_h2o2_current` ≥ 0.01 mA/cm² when sweeping anodic→cathodic; linear interp between bracketing grid points |
| Max Ring Current | model output → `max_ring_current_in_window` | max of `ring_current_ring_basis_mA_cm2` over overlap window |
| n_e_rrde | model output → `n_e_rrde_at_argmax_V` | `4·|I_disk| / (|I_disk| + I_ring/N)`, N=0.224, evaluated at `argmax_V_for_selectivity` |
| ΔpKa per cation | model output (Phase D fit) | `β_per_cation · σ_local_at_V_kin` |

For the RRDE-equivalent H₂O₂% formula, see Ruggiero 2022 §2.
For N=0.224 collection efficiency, see Ruggiero 2022 §2 (paper)
and `_bv_common.py` documentation.

## β parameterization (locked)

Singh Eq. (4) geometric coefficient:
```
β_per_cation = 2 · A · z_eff · r_H_El · (1 − r_M-O² / r_H_El²)
            [pm²; consistent with σ in counts/pm²]
```
where A = 620.32 pm, z_eff and r_M-O per
`SINGH_2016_CATION_PARAMS` in `_bv_common.py`, and r_H_El is
the calibration knob (default = Cu prior).

Transfer rule (locked default):
* **Δ-rule:** β_X_carbon = β_X_Cu + Δ_β
* Phase D fits the single scalar Δ_β against K's experimental
  H₂O₂ selectivity at pH 4.
* Phase E applies the same Δ_β to Cs, Na, Li without refit.

ΔpKa = β · σ for the active σ mapping:
* **Coupled physical path:** σ_local_Stern from PNP/Stern solve.
* **Ablation path:** σ_imposed_Singh_counts_pm2 via the
  `pka_override_ablation` flag.

**Sign convention:** cathodic pKa lowering requires β · σ_local
cathodic < 0.  Add as regression test before Phase D fit.

## Σ_S mapping convention (locked default)

**Default:** local Stern σ_S from PNP/Stern solve via
`sigma_C_m2_to_counts_pm2(σ_C_m2)` (signed; helper to be added
in v10a alongside the Langmuir cap).  Conversion factor
`6.243e-6` (where 1 m² = 1e24 pm², F = 96485, N_A = 6.022e23).

**Ablation alternative:** `override_sigma_singh_counts_pm2`,
labelled `pka_override_ablation` in code, ablation only — does
not couple to Stern/Poisson.

If Phase D's Δ_β differs by > 30 % between the two conventions,
β is flagged non-identifiable; Phase E reports both.

## v10a → E sequence (per session 31 R5)

1. **Phase 0 — this document** (~1 hour; today, done).
2. **v10a — Langmuir cap + integrated diagnostics** (~5-7 days):
   * `build_forward_branch` and `update_gamma_from_solution`
     gain `(1 − Γ/Γ_max)` factor.
   * Langmuir Picard formula:
     `Γ_ss(λ) = λ·F₀ / ((1−λ) + λ·k_des + λ·B + λ·F₀/Γ_max)`.
   * Γ clamp `[0, Γ_max]` after Picard; `RuntimeWarning` if
     unclamped value out-of-bounds.
   * Integrated rung_callback: F₀_avg, Γ, θ = Γ/Γ_max,
     R_forward_capped, denominator, R_2e/R_4e separately, σ_S
     from solved fields.
   * `sigma_C_m2_to_counts_pm2` helper.
   * Tests: Γ → Γ_max as k_hyd → ∞; Γ_max → ∞ recovers v9
     byte-equivalent.
3. **Minimum V-sweep diagnostic** (~1 day): v10a across V_RHE
   walk at smoke + λ=0/1.
4. **V_kin selection** (predeclared rule per session 31 R5).
5. **Phase A.2** at V_kin (~1 day).
6. **Plumbing ablation matrix** at V_kin (~2 days; A1/A2/A3,
   BEFORE v10b).
7. **CMK-3 capacitance literature note** (v10b prerequisite).
8. **v10b** — Γ_max + k_des + C_S literature calibration
   (~1-2 weeks).
9. **B.2** — densified k_hyd × λ ramp at V_kin (~2 days).
10. **Phase D** — K-only Δ_β fit (~1 day).
11. **Phase E** — predictive holdout (~3-5 days).

## Risk register (no external validation)

| # | Risk | Mitigation |
|---|---|---|
| 1 | Locked bundle may not match group's actual deliverable | Document working-hypothesis status; future external send if/when possible uses this as contract draft |
| 2 | Primary observable (selectivity) may be out of reach if cd is O₂-Levich-limited per cation too | Fallback: report selectivity match as exploratory, downgrade ΔpKa ordering to primary, drop Phase E "must pass" → "best-effort" |
| 3 | β · σ_local may have wrong sign convention | Sign-guard regression test added in v10a |
| 4 | Singh σ-mapping (local Stern vs imposed) may give different Δ_β | Phase D reports both; if > 30% divergence, flag non-identifiable |
| 5 | CMK-3 C_S = 10 µF/cm² may be wrong by 5× | v10b's literature note OR C_S bracket sweep at {10, 30, 50} µF/cm² with explicit "no literature anchor" caveat |
| 6 | v10a's Γ_max smoke value (1 monolayer) may be wrong | v10b recalibrates from literature; v10a tests don't depend on the absolute value |
| 7 | Δ-rule transferability may not work for Li⁺ (gap-singular at r_H_El ≈ r_M-O) | Document; Phase E reports per-cation pass/fail individually; if Li⁺ fails, the 4-cation pass becomes 3-cation pass + Li⁺ ablation finding |

## Status

* **Phase 0 locked:** 2026-05-10.
* **Next action:** begin v10a (Langmuir cap residual + Picard
  formula + Γ clamp + helper + tests).
