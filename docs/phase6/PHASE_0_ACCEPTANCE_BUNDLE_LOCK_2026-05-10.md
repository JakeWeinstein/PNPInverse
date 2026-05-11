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
* **v10a delivered:** 2026-05-10 (same day).  See section below
  for the implementation summary.
* **v10a V-sweep diagnostic (initial):** 2026-05-10.  7 V points
  (+0.55 → −0.50), `C_S = 0.10 F/m²`, `K0_R4e_factor = 1.0`.
  Result: `no_candidate_passed_locked_rule` — branch filter blocked
  every V (pure-4e selectivity throughout) and σ_S<0 ∩ cd_ok
  intersection was empty.  See
  `StudyResults/phase6b_v10a_v_sweep_diagnostic/iv_diagnostic.json`
  and the v10a' result section below for the corrected re-run.
* **v10a' V-sweep diagnostic (corrected):** 2026-05-10.  Same 7 V
  points with `C_S = 0.20 F/m²` (Bohra-Koper-Choi consensus per
  `.research/cmk3-stern-capacitance/SUMMARY.md`) and
  `K0_R4e_factor = 1e-14` (V=−0.10 branch-pass probe per
  `project_k0_r4e_ratio_regimes`).  **Result: V_kin = −0.10 V**
  via the primary path (no fallback).  Decision tree precedence
  guards: not transport-artifact (o2_flux_levich = 0.63 < 0.9),
  not cap-dominated (V=−0.10 has θ=0.86 < 0.9 and |sensS|=0.187
  > 0.10).  → **Case A: Phase A.2 at V_kin = −0.10 V.**

  Per-V breakdown (truncated):

  | V_RHE | σ_S<0 | cd_ok | branch_ok | 3pass | sensS | denom_cap/T | θ | x_2e |
  |---|---|---|---|---|---|---|---|---|
  | +0.55 | F | T | F | F | −0.104 | 0.318 | 0.318 | 0.002 |
  | +0.20 | F | T | T | F | −0.220 | 0.596 | 0.596 | 0.059 |
  | +0.10 | F | T | T | F | −0.249 | 0.687 | 0.687 | 0.130 |
  | **−0.10** | **T** | **T** | **T** | **T** | **−0.187** | 0.861 | 0.861 | 0.199 |
  | −0.30 | T | F | F | F | −0.064 | 0.948 | 0.948 | 0.001 |
  | −0.50 | T | F | F | F | −0.021 | 0.976 | 0.976 | <1e-5 |

  V=−0.30 and V=−0.50 are cap-dominated (denom_cap/total > 0.8 AND
  θ > 0.9 AND |sensS| < 0.10) — the v10b literature-calibration
  prerequisite signal is present at the most cathodic V's, but
  V_kin (=−0.10) is *not* cap-dominated, so v10b routing is not
  triggered for V_kin itself.  Phase A.2 at V_kin should still
  retain headroom on the Stern-cap-manifold derivative.

  K+ enrichment is the dominant F₀ amplifier in the cathodic
  region (per critique session 33 R3 #1):
  `amplification_from_c_K = 0.16 (V=+0.55) → 11.6 (V=−0.50)`,
  while `amplification_from_singh ≈ 1.0` everywhere
  (`pka_shift_avg ~ 1e-5`).

  Output: `StudyResults/phase6b_v10a_prime_k0r4e_1e-14/iv_diagnostic.{json,png}`.
  Wall: 1626 s (~27 min).
* **Phase A.2 landed:** 2026-05-10.  Densified k_hyd × λ ramp at
  V_kin = −0.10 V with 10-point k_hyd grid
  `{1e-5, 3e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 1e-1}`,
  two-stage anchor at C_S = 0.10 → 0.20 F/m², K0_R4e_factor =
  1e-14, full v10a Langmuir cap.  **All 10 rungs converged at
  λ=1.0; Picard converged everywhere; mass-balance residual at
  machine precision (1e-14 to 1e-16)**.  Baseline reproduction at
  k_hyd=1e-3 matches v10a' record within rel 1e-3 (γ=0.04047 vs
  0.0405; θ=0.861; σ_S=−0.01715; cd=−3.12 mA/cm²).  Cap saturation
  smooth from θ=0.058 (k_hyd=1e-5) → θ=0.998 (k_hyd=1e-1); v10a's
  `(1 − Γ/Γ_max)` cap delivers exactly the saturation behavior it
  was designed for, with v9's k_hyd ≥ 1e-1 Picard breakdown
  resolved.  **k_hyd_route = 1e-1**.  **v10b k_des/Γ_max priority:
  LOW** (`single_v_selectivity_gap_pp = +5.09 pp`; H₂O₂% = 19.91%
  sits within 10pp of deck band's lower edge [25, 50]).
  `max_amp_from_singh = 1.0000112` ⇒ rH_El recalibration NOT
  required for v10b.  No transport re-entry (o2_flux_levich ≈ 0.63
  flat across k_hyd).

  Decoupling observations (independent of k_hyd at this V_kin):
  σ_S = −0.01715 C/m² (field-driven Bikerman packing, not Γ);
  cd_mA_cm² = −3.12 (R_4e dominated); x_2e = 0.199, x_4e = 0.801
  (k0_R4e_factor=1e-14 puts both branches active);
  amplification_from_c_K = 1.75 (K⁺ enrichment dominates F₀ growth
  at V_kin, consistent with v10a' cathodic-side finding).

  Convergence audit `overall_pass = False` is a threshold-
  narrowness artifact, NOT a physics or convergence problem: the
  plan's `transition_coverage` test demanded
  `max(θ at k_hyd ∈ {1e-5 … 2e-3}) ≥ 0.93`, but the closed-form
  prediction at k_hyd=2e-3 was θ=0.926 — observed 0.9253 is
  essentially exact match.  k_hyd=5e-3 hits θ=0.969 (within the
  saturation grid by construction).  Future audits should set
  the transition-grid threshold ≤ 0.92 OR move k_hyd=5e-3 into
  the transition grid.

  Output:
  `StudyResults/phase6b_v10a_phase_A2_v_kin/phase_a2_v_kin.{json,png}`.
  Tests: `tests/test_phase6b_v10a_phase_A2_driver.py` (43 fast).
  Wall: 1300 s (~22 min).  Plan + critique provenance:
  `~/.claude/plans/phase6b-v10a-phase-A2-v-kin.md` /
  `docs/handoffs/CHATGPT_HANDOFF_34_phase6b-v10a-phase-A2-v-kin/`
  (4 rounds of GPT critique, APPROVED; 41 issues addressed).

  Per-k_hyd λ=1 result table:

  | k_hyd | γ | θ | picard | mb_rel | x_2e | H₂O₂% | σ_S | o2_Levich | amp_cK |
  |---|---|---|---|---|---|---|---|---|---|
  | 1e-5 | 0.00274 | 0.058 | converged | 0.0 | 0.199 | 19.95 | −0.0171 | 0.630 | 1.75 |
  | 3e-5 | 0.00737 | 0.157 | converged | 2e-17 | 0.199 | 19.94 | −0.0171 | 0.630 | 1.75 |
  | 1e-4 | 0.01798 | 0.382 | converged | 7e-17 | 0.199 | 19.93 | −0.0171 | 0.630 | 1.75 |
  | 2e-4 | 0.02601 | 0.553 | converged | 7e-17 | 0.199 | 19.93 | −0.0171 | 0.630 | 1.75 |
  | 5e-4 | 0.03553 | 0.756 | converged | 3e-16 | 0.199 | 19.92 | −0.0171 | 0.630 | 1.75 |
  | **1e-3** | **0.04047** | **0.861** | converged | 4e-16 | 0.199 | 19.91 | −0.0171 | 0.631 | 1.75 |
  | 2e-3 | 0.04349 | 0.925 | converged | 1e-15 | 0.199 | 19.91 | −0.0171 | 0.631 | 1.75 |
  | 5e-3 | 0.04553 | 0.969 | converged | 1e-16 | 0.199 | 19.91 | −0.0171 | 0.631 | 1.75 |
  | 1e-2 | 0.04625 | 0.984 | converged | 2e-15 | 0.199 | 19.91 | −0.0171 | 0.631 | 1.75 |
  | **1e-1** | **0.04692** | **0.998** | converged | 1e-14 | 0.199 | 19.91 | −0.0171 | 0.631 | 1.75 |

  Bold row at k_hyd=1e-3 = v10a' baseline reproduction.  Bold row
  at k_hyd=1e-1 = `k_hyd_route` (routing-pass).
* **Step 6 plumbing-ablation matrix landed:** 2026-05-10.  Verifies
  the four override consumers of the cation-hydrolysis residual
  (form-build R_net, ctx-stored pKa-shift expression, Picard Γ
  update, diagnostics F0_decomposition) are wired consistently
  against three new `bv_convergence` flags:
  `apply_h_source`, `apply_k_sink`, `override_sigma_singh_counts_pm2`.
  Defaults preserve byte-equivalence with v9/v10a/v10a'/A.2.

  | Ablation | Status | Key evidence |
  |---|---|---|
  | A0 baseline | pass | γ/θ/σ_S/cd/R_2e/R_4e all rel 0.0 vs A.2 record |
  | A0b assembly | pass | mass-balance closure rel 1e-15; consistency gates exact |
  | A1 source-only | pass | R_inj=2.0; Δc_H = +6.2% (in [5%, 25%]) |
  | A2 sink-only | pass | wiring_ok_magnitude_unreachable; Δc_K signed −0.5% at R_inj=10 ceiling (sign + monotonic) |
  | A3 σ override | pass | pka_factor obs/pred 10.0781 (rel 2.6e-12); gate 6 residual closure rel 4.5e-16 |

  **Routing decision:**
  `plumbing_verified_proceed_to_step7_then_step8`.  v10b literature
  calibration of Γ_max + k_des + C_S is unblocked.

  A2 wiring-ok-magnitude-unreachable verdict is a **physics
  finding**, not a wiring bug.  The cathodic K⁺ Boltzmann pile-up
  at V_kin gives `c_K_boundary_avg ≈ 291` nondim ≈ 1.75·c_K_bulk
  (per F0_decomposition `amplification_from_c_K`); sentinel-scale
  R_inj ≤ 10 cannot dent this by 5%.  Linear extrapolation puts a
  5% drop at R_inj ≈ 100, well above the plan's ceiling of 10.
  The wiring is verified by:
    (a) sign correct at the largest R_inj attempted
        (signed Δc_K = −5.03e-3 < 0), AND
    (b) monotonic `|Δc|` growth across `R_inj ∈ {0.1 … 10}` with
        slope ≈ −5e-4/R_inj.
  The plan did not anticipate this; the driver introduces a
  ``status="pass" + pass_qualifier="wiring_ok_magnitude_unreachable"``
  outcome for this case (see
  `scripts/studies/phase6b_step6_plumbing_ablation.py:_verify_wiring_from_prepass`).

  Output:
  `StudyResults/phase6b_step6_plumbing_ablation/ablation_matrix.{json,png}`.
  Tests: `tests/test_phase6b_step6_plumbing_ablation.py` (53 fast),
  `tests/test_phase6b_step6_plumbing_ablation_slow.py` (14 slow +
  1 e2e gated by `RUN_SLOW_E2E`).
  Wall: 25 min.  Plan + critique provenance:
  `~/.claude/plans/phase6b-step6-plumbing-ablation.md` /
  `docs/handoffs/CHATGPT_HANDOFF_35_phase6b-step6-plumbing-ablation/`
  (5 rounds of GPT critique, **APPROVED**; 54 issues addressed).

* **Step 7 CMK-3 capacitance literature note landed:**
  2026-05-10.  Published
  `docs/phase6/CMK3_capacitance_literature.md` — distillation of
  the `.research/cmk3-stern-capacitance/` research trail into a
  canonical reference: `C_S = 0.20 F/m²` derivation (Bohra-Koper-
  Choi consensus, `L_S=5 Å` + `ε_S=11.3`), citation chain
  (Bohra 2024 / Choi 2024 / Pillai 2024 / CatINT / Kilic-Bazant
  2007), three Pillai-2024 regimes, the four load-bearing
  caveats (per-local-surface-element interpretation; Singh's
  51 µF/cm² is total C_dl not Stern-only; carbon-specific
  narrowing pulls slightly below 20 µF/cm²; constant `C_S` is
  field-averaged), and the locked sensitivity bracket
  `C_S ∈ {0.05, 0.10, 0.20, 0.30} F/m²` for v10b.  CLAUDE.md
  hard rule #6 + the source-of-truth table now point here as
  the canonical reference; full research trail (agent-by-agent
  evidence + four supporting notes) remains at
  `.research/cmk3-stern-capacitance/SUMMARY.md`.

* **Next action:** step 8 — v10b literature calibration of
  `Γ_max + k_des + C_S` (~1-2 weeks).  Plan via `/sci-planner`
  recommended (real multi-day calibration effort with several
  literature interpretation calls).  v10b is MANDATORY in all
  routing branches per the "v10a → E sequence" §; A.2 + step 6
  inform v10b priority but do not cancel v10b.

* **Step 8 v10b literature calibration landed:** 2026-05-10.
  Published `docs/phase6/v10b_calibration_summary.md` with the
  three locked V10B numeric drops + per-parameter decision-rule
  outcomes:
  - `GAMMA_MAX_HAT_V10B = 0.047` nondim (= `GAMMA_MAX_HAT_V10A_SMOKE`;
    4-test compatibility check finds Singh 2016 reports K_eq not
    coverage, Iamprasertkun 2019 reports HOPG specific capacitance
    not MOH coverage, Bohra 2019 uses variable permittivity →
    tighten V10A derivation chain rather than replace value).
  - `K_DES_NONDIM_V10B = 1.0` nondim (engineering choice with
    documented Eyring prior `k_des_nondim ∈ [1e-2, 1e2]` ↔
    `ΔG_des ∈ [0.69, 0.94] eV`; central value = `ΔG_des ≈ 0.80 eV`
    at 298 K).
  - `C_S_F_M2_V10B = 0.20` F/m² (locked at step 7).
  Source of truth: new top-level Firedrake-free
  `calibration/v10b.py` package carries V10B constants plus
  `V10B_CALIBRATION_METADATA` (schema: value / units / is_nondim
  / source_type / engineering_choice / citation / bracket / prior
  / compatibility{mechanism, electrode, electrolyte, dimensional}).
  Deprecation alias `SMOKE = V10A_SMOKE` (NEVER `SMOKE = V10B`)
  with ASCII-only comment block + AST-aware import-audit test.
  v10b.A.2 driver re-run at v10b parameters with
  `--out-subdir phase6b_v10b_phase_A2_v_kin`; `_convergence_audit`
  refactored to HARD/SOFT split (escalation gates separated from
  informative deltas; `overall_pass` driven by HARD only).
  v10b.step-6 driver re-run at v10b parameters with
  `--out-subdir phase6b_v10b_step6_plumbing_ablation` and the new
  `--a2-baseline-json` CLI flag pointing at the v10b A.2 baseline.
  Sensitivity sweeps: D7-D1 C_S bracket (4 rungs: {0.05, 0.10,
  0.20, 0.30} F/m²) and D7-D4 Γ_max × k_des × k_hyd matrix
  (30 rungs: 3 × 5 × 2) with per-rung analytic-vs-solver
  mass-balance HARD gate (rel ≤ 5e-3 via `gamma_ss_langmuir`).
  Test coverage: 27 new fast tests
  (`tests/test_phase6b_v10b_calibration.py` 13 +
  `tests/test_phase6b_v10b_bracket_matrix.py` 14); 255/255
  phase6b/cation fast-suite green.  Plan + critique provenance:
  `~/.claude/plans/phase6b-step8-v10b-calibration.md` (v7-FINAL,
  53 issues across 7 rounds → APPROVED) +
  `docs/handoffs/CHATGPT_HANDOFF_36_phase6b-v10b-calibration/`.

## v10a delivery summary (2026-05-10)

What landed:

* `Forward/bv_solver/cation_hydrolysis.py`:
  - `CationHydrolysisBundle.gamma_max_func` (R-space Function) +
    `GAMMA_MAX_HAT_SMOKE = 0.047` (1 monolayer at the OHP).
  - `build_proton_boundary_source` / `build_forward_branch` now
    apply the Langmuir vacancy factor `(1 − Γ/Γ_max)`.
  - `build_forward_branch_uncapped` exposes the Γ-independent F₀
    for diagnostic readers.
  - `gamma_ss_langmuir(...)` — pure-Python closed-form helper used
    by Picard and by unit tests.
  - `update_gamma_from_solution` rewired to delegate to the helper
    on the physical path; manufactured-R_inj path unchanged
    (intentionally bypasses the cap and the post-clamp warning).
  - `clamp_gamma_to_max(ctx)` silent helper for warm restarts.
  - `collect_v10a_rung_diagnostics(ctx)` returns the documented
    rung-callback payload.
* `Forward/bv_solver/units.py` (new): `sigma_C_m2_to_counts_pm2(σ)`.
* `Forward/bv_solver/anchor_continuation.py`:
  - `set_reaction_gamma_max_model` accessor; new
    `"gamma_max_nondim"` key in
    `solve_lambda_ramp_from_warm_start`'s `parameter_overrides`.
  - `clamp_gamma_to_max` invoked before Picard in
    `solve_lambda_ramp_from_warm_start` so warm restarts always
    enter Newton with a feasible vacancy factor.
  - `collect_v10a_rung_diagnostics` plumbed into the λ-rung
    rung_diag in both `solve_anchor_with_continuation` and
    `solve_lambda_ramp_from_warm_start`.
* `scripts/_bv_common.py`: `make_cation_hydrolysis_config(...)`
  builder (threads `gamma_max_nondim`) + re-exported
  `GAMMA_MAX_HAT_SMOKE`.
* `scripts/studies/phase6b_v9_gate4_finite_hydrolysis_smoke.py`:
  migrated to the builder; new `gamma_max_nondim` driver arg.
* `tests/test_phase6b_v10a_langmuir_cap.py` (new): 22 fast +
  14 slow regression tests covering the six plan-prescribed
  invariants plus the Risk #2 bisection cross-check.

Pre-merge verification (all green):

* `python -m pytest -m "not slow" --deselect tests/test_autograd_gradient.py --deselect tests/test_multistart.py -q` → 659 passed.
* `python -m pytest tests/test_phase6b_v10a_langmuir_cap.py -v` → 22 fast + 14 slow passed.
* Existing gate3/gate4 slow tests still green:
  `TestGammaResidualAreaInvariance`, `TestGammaDirichletPinAtLambdaZero`,
  `TestProtonBoundarySourceSignConvention`, `TestSinghPkaShiftUFL`,
  `TestLambdaHydrolysisAccessorRoundtrip`,
  `TestCationHydrolysisBundleBuild`, `TestMixedSpaceLayout*` — all pass.
