# Critique loop final — phase6b-sulfate-spike-planning

5 rounds, cap hit with `VERDICT: ISSUES_REMAIN`. Substantive
mid-loop redirects (sulfate-buffering → cation-hydrolysis per
handoff 26 §9; conjecture audit incorporation) shifted the artifact
twice; plan was rewritten clean as v5 to eliminate vestigial v3/v4
contradictions GPT flagged in R4#10/R5#7.

* **Revised artifact:** `docs/phase6b_next_steps_plan.md` (now v5)
* **Session dir:** `docs/CHATGPT_HANDOFF_28_phase6b-sulfate-spike-planning/`
* **Total issues raised across 5 rounds:** 65 (16 + 10 + 15 + 14 + 10)
* **Final architecture:** cation hydrolysis with boundary-only
  algebraic shadow + Stern surface-charge coupling

---

## Addressed (62)

The vast majority of GPT's points were accepted and rolled into the
plan. Grouped by topic:

### Mechanism redirect (rounds 1–2, then mid-loop user redirect)

* Sulfate-buffering hypothesis (HSO₄⁻ ⇌ SO₄²⁻ + H⁺) **retired entirely**
  per handoff 26 §9. v1/v2 of the plan in git for that excursion.
* Replaced with cation-hydrolysis (M(H₂O)ₙ⁺ ⇌ M(OH)⁰ + H⁺) at the
  OHP per Linsey 2025 deck slide 27 + Singh 2016 JACS framework.
* `(c_H)^n` correctly identified as a *multiplier* on the rate, not
  a *ceiling* — handoff 26 §4 had this backwards (R2#1).
* Acid-form BV + transport gives plateau, not decay — therefore 6δ
  kept active, not deferred (R2#8 / R3#12). 6δ split into
  6δ.1 (parallel alkaline channels) + 6δ.2 (pH-gated switching).

### Algebra spike (rounds 1–5)

* Original sulfate algebra spike was numerically wrong on multiple
  fronts (I=0.3M vs [SO4]_T conflation, n=0.5/1.0 vs n=2/4 powers,
  unconstrained closure, post-hoc multiplier ≠ cd) — all retired
  with sulfate.
* Cation-hydrolysis spike scope reduced to **plausibility screen**
  per R5#5: brackets Phase 6α surface-state inputs and Singh-2016 β_M
  uncertainty, predicts local pH range per cation. **Not a Branch
  A/B gate** (the actual gate is a 6β.1 solver smoke that includes
  Stern field feedback).
* Phase 6α surface c_H bracketed over [pre-water-ionization, current
  6α] per R4#14 to honor the §6 caveat that fast-Kw may overpredict.

### Architecture (rounds 3–5, the heavy lifting)

* `c_M+(y)` stays analytic Boltzmann — no DOF (R3#7 / R4#2).
* `c_MOH` is a **boundary algebraic shadow** on the BV electrode
  `ds`, not a volume DOF, not a separate function space (R4#9 /
  R5#10).
* Proton-condition residual stays Phase 6α `E = c_H − c_OH` in
  volume (no double-counting per R4#1; superseded R3#4's unified
  E proposal which was in fact double-counting at steady state).
* No volume `R_buf` source / `θ(y)` thin-layer kernel — both had
  undefined units and mis-attributed Stern physics (R4#3, #4, #5).
* **Stern surface-charge coupling** added as the steady-state
  driver: `σ_OHP_corrected = σ_OHP_existing + F·δ_OHP·(c_M+(0) −
  c_M_total(0))` (R5#1; see Unresolved below).
* Activation continuation `λ_hydrolysis` ramps **`log(Ka_M_eff)`**,
  not Ka_M_eff, to avoid logarithmic stiffness near small λ (R5#6).
* Disabled-path regression at λ=0 asserts **residual L²-norm** at
  original-DOF subset, not byte-level vector equivalence (R4#8).

### Literature / data (rounds 3–5)

* **Singh / Kwon / Lum / Ager / Bell 2016 JACS**
  (`10.1021/jacs.6b07612`) is the primary methodological reference
  for field-dependent pKa shift — not Co-Zhang 2019 §3 (R3#6).
* Co-Zhang 2019 used product/ring-current electrochemistry for CO₂RR,
  **not** an IrOx ring (R4#13). IrOx-on-RRDE-ring attribution
  belongs to Linsey 2025 deck slides 5–9.
* Linsey deck slide 27 pKa table is for Cu/CO₂RR — treat as
  cross-check target, not solver input (R3#7).
* CP data (`{Cs,K,Na,Li}2SO4_10-9-20.mat`) requires Ag/AgCl→RHE
  conversion, replicate averaging, and outlier QC; trends are by
  pH/current regime, not single monotone ordering (R3#13).
* Calibration / holdout split: K⁺ for calibration, Cs⁺/Na⁺/Li⁺ as
  holdout. Holdout is a **predictive screen, not decisive
  falsification** (R3#12 → tightened in R5#8).
* Per-cation config schema: full parameter set (Stokes radius,
  a_nondim, phi_clamp, c_bulk_nondim, D_M, pKa_bulk, pKa_shift_form,
  pKa_shift_params), not minimal counterion dicts (R3#14).

### Conjecture audit (incorporated post-R4)

* HIGH: Cs⁺ vs K⁺ apples-to-apples — step 6 smoke uses K⁺ as
  primary; Cs⁺ is sensitivity (R5#9).
* MED: defer K0_R4e / α_R4e re-tuning until cation hydrolysis lands.
* LOW-MED: Stern capacitance citation chain (queued).
* LOW: L_eff sweep is Claude/GPT framing; deck-comparable is
  L=16 µm only (RDE 1600 rpm Levich).

### Architectural details (rounds 1–5)

* Phase 6α cosmetic logging bug (`_config_dict` line 519) flagged
  in R1#1 — fix specified in step 5; data verified real via run log
  + summary.json + cd magnitudes that exceed H⁺ Levich.
* `c_H_neutral_water` formulation in v4's Phase 6α.1 finite-rate
  water rewrite was wrong; replaced with
  `R_water = k_r·(Kw_eff − c_H·c_OH)` per R4#11.
* Phase 6α's surface pH 10.58 is a model output, not measurement —
  documented in §1.5 caveat with the IrOx empirical-truth queued
  in step 10 (Phase 6α.1).

---

## Defended (1)

* **R1#1 (Phase 6α evidence basis).** GPT flagged that
  `iv_curve.json` config field shows `enable_water_ionization: false`
  and `kw_eff_hat: null`. Defended on **physics**: the run log
  shows the kw_eff ladder fired at every combo, the top-level
  `summary.json` correctly records `true`, and the cd magnitudes
  (4–25× the H⁺ Levich limit) can only come from the water-ionization
  pathway. Accepted on the **provenance bug**: per-combo `_config_dict`
  call at line 519 omits the kwarg — fix in step 5 of the plan.
  Net: data trusted, bug acknowledged and queued for fix.

---

## Unresolved (2)

* **R5#1 (Stern surface-charge coupling not GPT-reviewed).** The v5
  architecture's `σ_OHP_corrected = σ_OHP_existing + F·δ_OHP·(c_M+(0)
  − c_M_total(0))` is the proposed steady-state driver. GPT flagged
  in R5#1 that without this term, the boundary algebraic shadow has
  no effect on steady state. The fix was added in step 7 of the
  plan but the loop hit cap before GPT could review it. **The 6β.1
  implementer must verify in the smoke (step 6):**
  * Sign of the Stern correction (R5#2).
  * Unit consistency of `δ_OHP·(c_M+ − c_M_total)` mol/m³·m → mol/m²
    → ×F → C/m² (R5#3).
  * That the smoke shows c_H *moves* when λ_hydrolysis goes 0 → 1.
  
  If the smoke shows c_H doesn't move, queue another GPT round with
  the smoke result as new evidence; the architecture is wrong and
  needs further work.

* **R5#4 (Boltzmann c_M+ ≠ c_M_total when neutralization is
  significant).** The plan labels this as a controlled reduced-model
  assumption per R5#4's "either accept and label, or solve coupled."
  Accepted as labelled assumption; the 6β.1 smoke will reveal whether
  it holds. If c_MOH(0) becomes O(50%) of c_M_total(0), the Boltzmann
  distribution itself is no longer accurate and c_M may need to be
  upgraded to a dynamic NP species — explicit in §8 open question 4.

---

## Round-by-round timeline

* **R1** (sulfate plan): 16 issues. GPT killed n=0.5/1.0 powers, the
  Phase 6α evidence-basis question (correctly flagged the cosmetic
  bug), the I/[SO4] conflation, the local-algebra closure, and the
  branch logic.
* **R2** (sulfate refined to Levich balance): 10 issues. GPT killed
  the handoff 26 §4 ceiling-mechanism story, forced the Levich
  flux balance (which numerically showed sulfate-augmented acid
  supply is actually within the right range at deck-relevant L_eff),
  and tightened the branch criterion. Numbers in R3 §2.
* **R3** (substantive redirect to cation hydrolysis per handoff 26
  §9): 15 new issues on the new chemistry. GPT correctly identified
  the source-term form was wrong (it was equilibrium concentration,
  not net rate), demanded the unified E equation and OHP localization,
  redirected the literature reading list to Singh 2016, and forced
  the calibration/holdout split.
* **R4** (cation hydrolysis architectural rebuild): 14 issues. GPT
  exposed the architectural muddle in v4 (three incompatible
  representations: Boltzmann + OHP-local pool + finite-rate volume),
  forced collapse to one clean closure, killed the dimensional /
  unit errors in the spike's "self-consistent (c_M+, c_MOH, c_H,
  η_local)" claim, and demanded the Phase 6α c_H bracket.
* **R5** (final round, boundary-only shadow + audit incorporation):
  10 issues. GPT exposed the load-bearing flaw that boundary shadow
  alone has no steady-state effect (the Stern coupling fix in step
  7), forced the log(Ka) activation ramp, demanded the plan-file
  cleanup, and pointed out that the calibration/holdout split is
  improved but Singh parameters may not transfer from CO₂RR-on-Cu
  to ORR-on-carbon.

---

## Net assessment

The loop's primary value was **forcing the architectural rebuild**
through R3 → R4 → R5 of the cation-hydrolysis closure. Without GPT,
the plan would have shipped with a 2× source-counting bug (R3#1),
no steady-state coupling (R5#1), wrong literature attribution
(R3#6 / R4#13), and a confused Boltzmann / OHP-local / volume-source
mix (R4#1–5).

The two unresolved items above are **honest residuals** that depend
on solver-smoke evidence not yet available. The plan documents
them in §8/§9 as the things the 6β.1 implementer must verify
empirically rather than accept on theory.

The mid-loop sulfate-buffering retirement (handoff 26 §9 redirect)
was an external-to-loop ground-truth correction, not a critique-loop
finding. GPT had been correctly flagging that sulfate's algebra
didn't work (R2 critique was directionally right) — but the user
pointing at handoff 26 §9 supplied the chemistry that did work.
Both threads converged.
