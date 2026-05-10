# Final revision — Critique Session 30 (phase6b-v9-cd-invariance)

* **Loop ended:** Round 5 (cap hit; final verdict ISSUES_REMAIN —
  GPT raised 8 final-round refinement issues, all addressed in
  this revision).
* **Revised artifact:**
  `docs/phase6/PHASE_6B_V9_PHASES_A_B_RESULTS_2026-05-10.md`
* **Total issues across all rounds:** 17 (R1) + 13 (R2) + 13 (R3)
  + 12 (R4) + 8 (R5) = **63 distinct issues**, many overlapping
  / refining the same point across rounds.

## Issue ledger

### Addressed (accepted; revised in artifact)

#### R1 issues (17 total)

1. **R1#1** — H⁺ Levich claim wrong; cd plateau is O₂ Levich.
   *Addressed:* TL;DR retracts H⁺ floor; states 4·F·D_O₂·C_O₂/L_eff
   = 5.50 mA/cm² ≈ observed 5.53 (0.6 % match).
2. **R1#2** — R_net = forward − backward, not forward.  *Addressed:*
   Sequenced plan A.1 records R_forward, R_backward, R_net
   separately per rung.
3. **R1#3** — D_REF and H⁺ stoichiometry errors.  *Addressed:*
   Mechanism section uses code constants (D_REF = 1.9e-9, R_4e
   gives i/F not i/(2F)).
4. **R1#4** — Phase A table missing λ=0 baseline.  *Addressed:*
   λ=0 instrumentation in Phase A.1 plan; in this revision the
   λ=0 baseline is the cache-walk value at V_kin.
5. **R1#5** — V=−0.20 V is on the plateau; should test in
   kinetic regime.  *Addressed:* full V_RHE walk table now in
   artifact; sequenced plan A.2 picks V_kin (least-saturated
   voltage with σ_S < 0).
6. **R1#6** — k_hyd window not exhaustively probed.  *Addressed:*
   B.2 plan densifies k_hyd to include 5e-3, 1e-2, 2e-2, 5e-2,
   1e-1.
7. **R1#7** — Γ scaling proves Picard not physics.  *Addressed:*
   Mechanism section qualifies "expression structurally correct
   in k_hyd ≤ 1e-2 regime".
8. **R1#8** — Diagnostic gap.  *Addressed:* A.1 instrumentation
   adds R_forward/R_backward/R_net, σ_S, ΔpKa, c_H(0), c_K(0),
   surface pH, branch currents.
9. **R1#9** — `pc` deprecated.  *Addressed:* "All branch-
   selectivity claims await per-reaction current assembly" caveat;
   `gross_h2o2_current` and per-reaction indices listed.
10. **R1#10** — Bikerman packing claim false.  *Addressed:*
    Removed (was in superseded artifact draft); current artifact
    diagnoses Picard breakdown via Γ-explosion through
    σ_S-dependent ΔpKa.
11. **R1#11** — Γ/δ ≈ 7.6e5 unphysical.  *Addressed:*
    "Architectural debt" section lists Γ-no-capacity as item 1;
    sequenced plan v10 capacity branch with `(1 − θ)` factor.
12. **R1#12** — Thinner L_eff is debug-only.  *Addressed:*
    Artifact's "Recommended next action" doesn't list thinner
    L_eff as a production mitigation.
13. **R1#13** — Picard breakdown under-instrumented.
    *Addressed:* Architectural debt item 2 (LadderExhausted
    partial-rungs attribute).
14. **R1#14** — Phase 6β deliverable misframed (surface pKa, not
    cd).  *Addressed:* "Phase 6β scope reframing" section.
15. **R1#15** — Per-cation r_H_El concern.  *Addressed:* Phase E
    revised with three predeclared 1-parameter K-fitted rules.
16. **R1#16** — Test gap (no λ=1 cd-shift test).  *Addressed:*
    Test gap section with manufactured-source + cation ordering
    + capacity tests.
17. **R1#17** — Need ablation matrix to disambiguate bug vs
    transport.  *Addressed:* Ablation matrix section with 5
    experiments (A4 deferred pending replacement model spec; A5
    deferred to post-v10).

#### R2 issues (13 total)

1. **R2#1** — R_net is 12 % of BV H⁺, not 50 %.  *Addressed:*
   Mechanism section uses corrected percentage.
2. **R2#2** — `pc/cd ≈ 0.5` ≠ 50 % H₂O₂.  *Addressed:* Note
   that legacy `pc` is degenerate for parallel topology;
   per-branch assembly required.
3. **R2#3** — Surface pH ≠ pKa.  *Addressed:* Phase D revised
   to fit ΔpKa magnitude (not pH); neutral fraction
   Henderson-Hasselbalch formula corrected (1/(1 + 10^(pKa−pH))).
4. **R2#4** — Phase D's "match deck pKa at our V" not justified.
   *Addressed:* Phase D revised to fit at the model's reachable
   σ_S range, with explicit caveats.
5. **R2#5** — V=+0.30 may have σ_S ≥ 0 (anode-clamped).
   *Addressed:* A.2 picks V_kin = "least-saturated voltage with
   σ_S < 0" after instrumentation, not assumed.
6. **R2#6** — Finer λ ladder may help.  *Addressed:* B.2 with
   patched AdaptiveLadder + λ first rung at 1e-4.
7. **R2#7** — Γ capacity defense weak.  *Addressed:* "Architectural
   debt" #1; v10 capacity branch.
8. **R2#8** — LadderExhausted has no `result`.  *Addressed:*
   Architectural debt #2 — `partial_rungs` attribute.
9. **R2#9** — Per-cation r_H_El destroys holdout.  *Addressed:*
   Phase E single K-fitted scalar per rule.
10. **R2#10** — Smoke kinetics regression test brittle.
    *Addressed:* Test gap uses bracketed manufactured R_inj,
    not smoke kinetics.
11. **R2#11** — Ablation matrix invalid controls.  *Addressed:*
    Corrected matrix with `apply_h_source`, `apply_k_sink`,
    `override_pka_sigma_S` flags.
12. **R2#12** — Mass-balance check missing.  *Addressed:* A.1
    instrumentation includes mass-balance integrals via in-form
    UFL.
13. **R2#13** — Overcorrected V=−0.20 finding.  *Addressed:*
    Mechanism section #2 acknowledges Γ data is valid even on
    O₂ plateau.

#### R3 issues (13 total)

1. **R3#1** — Singh σ_S = 226 µC/cm², not 19.  *Addressed:*
   "Σ_S scale mismatch" section with 3 resolutions (A/B/C);
   recommends B; flags as load-bearing.
2. **R3#2** — Neutral fraction sign flip.  *Addressed:* Corrected
   to 1/(1 + 10^(pKa − pH)) in Phase 6β scope reframing section.
3. **R3#3** — `peroxide_current` interpretation still wrong.
   *Addressed:* "Caveats" section explicitly notes `pc/cd =
   0.5 is degenerate; per-branch assembly required".
4. **R3#4** — Finer λ ladder still has first-positive-rung
   problem.  *Addressed:* AdaptiveLadder patched to allow λ=0
   as floor + 1e-4 first rung.
5. **R3#5** — Langmuir cap erases R_net at k_des=1.  *Addressed:*
   Architectural debt #1 with k_des/Γ_max recalibration noted
   (direction: larger k_des or larger Γ_max increases max R_net).
6. **R3#6** — ρ-rule transferability fragile.  *Addressed:* Phase
   E lists three rules (ρ, Δ, C_S-coupled) with risk per rule.
7. **R3#7** — `k_hyd=1.0` is physical not manufactured.
   *Addressed:* A1 in ablation matrix uses
   `manufactured_R_inj=bracketed`, not k_hyd=1.0.
8. **R3#8** — manufactured_R_inj couples H/K.  *Addressed:* A1/A2
   use `apply_h_source` / `apply_k_sink` switches.
9. **R3#9** — Stern off zeros σ_S.  *Addressed:* A3 uses
   `override_pka_sigma_S` (only enters Singh ΔpKa, not
   Stern/Poisson).
10. **R3#10** — Water-ionization mass balance term wrong.
    *Addressed:* A.1 mass-balance via in-form UFL, not
    hand-derived.
11. **R3#11** — H flux balance underspecified for logc_muh.
    *Addressed:* Same as #10; reuse form-builder UFL.
12. **R3#12** — Mixing two tracks (instrument + redesign).
    *Addressed:* Sequenced plan A.1 → A.2 → B.2 → v10 → D → E.
13. **R3#13** — D_O2 inconsistency.  *Addressed:* Code constants
    (D_O2 = 1.9e-9; Levich = 5.50 mA/cm²).

#### R4 issues (12 total)

1. **R4#1** — Singh σ_S 226 vs 19 µC/cm² (refined R3#1).
   *Addressed:* Same section as R3#1; documented unphysical
   22.6 V Stern drop required at C_S=10 µF/cm².
2. **R4#2** — Neutral fraction sign flip (refined R3#2).
   *Addressed:* Same as R3#2.
3. **R4#3** — peroxide_current still misdescribed.
   *Addressed:* same as R3#3; pc = R_2e − R_4e literal
   interpretation.
4. **R4#4** — λ ladder structurally lacks sub-rung-1 floor.
   *Addressed:* AdaptiveLadder patch + 1e-4 first rung.
5. **R4#5** — Langmuir cap erases effect; k_des direction.
   *Addressed:* Architectural debt #1 with corrected direction
   (larger k_des/Γ_max → larger R_net).
6. **R4#6** — ρ rule fragile (refined R3#6).  *Addressed:*
   Phase E lists ρ, Δ, C_S-coupled with explicit risk for ρ.
7. **R4#7** — k_hyd=1.0 not manufactured.  *Addressed:* same
   as R3#7.
8. **R4#8** — Manufactured R_inj couples H/K.  *Addressed:*
   same as R3#8 with `apply_h_source/apply_k_sink` flags.
9. **R4#9** — Stern off → σ_S=0 → hydrolysis dies.
   *Addressed:* `override_pka_sigma_S` flag.
10. **R4#10** — Water-ionization mass balance wrong.
    *Addressed:* same as R3#10.
11. **R4#11** — Flux balance underspecified.  *Addressed:* same
    as R3#11.
12. **R4#12** — Mixing tracks.  *Addressed:* Sequenced plan.
13. **R4#13** — D_O2 = 1.9 vs 2.0.  *Addressed:* code-constant
    Levich = 5.50.

#### R5 issues (8 total)

1. **R5#1** — pc/cd not branch-degenerate when sign considered.
   *Addressed:* Caveats note pc/cd = 0.5 is consistent with
   multiple branch mixes; per-branch assembly required.
2. **R5#2** — "≥4" sloppy.  *Addressed:* Mechanism section uses
   "near 4e O₂ Levich limit" and "apparent electron count
   near 4".
3. **R5#3** — A5 conflicts with v10 sequencing.  *Addressed:*
   A5 explicitly labelled "(post-v10)" in ablation matrix.
4. **R5#4** — Phase D needs scalar K target.  *Addressed:*
   Phase D revised section lists two candidate scalar targets
   ("ΔpKa_K_deck × σ ratio" or "γ_MOH match"), notes "lock the
   choice before running".
5. **R5#5** — Calibrate ordering underdefined.  *Addressed:*
   Separated calibration (K-only scalar fit) from holdout
   (Cs/Na/Li rank + spacing metric).
6. **R5#6** — Default-on/off semantics inconsistent.
   *Addressed:* Architectural debt #4 explicitly states
   "Defaults: `apply_h_source=True`, `apply_k_sink=True`,
   `override_pka_sigma_S=None` — reproduces v9 byte-for-byte".
7. **R5#7** — A4 not concrete.  *Addressed:* A4 is now
   "(deferred)" in ablation matrix; needs replacement model
   spec.
8. **R5#8** — C_S-coupled rule ambiguous.  *Addressed:* Phase E
   C_S-rule explicitly states "Single global C_S, not per-cation
   (per-cation would defeat the holdout)".

### Defended

* **R1#11 / R2#7 (initial defense of Γ/δ as physical):**
  Concession won — current v9 architecture's k_hyd ≤ 1e-2 is
  unphysical (Γ ≈ 64 monolayers).  Defended in R2 only;
  capitulated in R3.

### Unresolved

None.  All 63 issues across 5 rounds have either accepted
fixes in the revised artifact or are explicitly deferred with
documented rationale (A4 ablation pending replacement model
spec; A5 ablation post-v10).

## Summary

The 5-round critique loop converted a single "v9 R5#5 wording-
guard outcome" finding into a full architectural debt list, a
sequenced re-plan, and a reframed Phase 6β scope.  The most
load-bearing corrections from GPT:

1. **cd plateau is O₂ Levich, not H⁺.**  Original mechanism
   interpretation was wrong by 10×.
2. **Phase A's V=−0.20 V was inside the O₂ plateau** —
   cd-invariance under λ ramp is not a finding about hydrolysis.
3. **Γ has no Langmuir capacity** — converged k_hyd=1e-2 case
   is at ~64 monolayers of MOH, physically invalid.
4. **Singh-σ-to-model-σ_S mapping is undocumented.**  May be a
   load-bearing assumption; potentially excludes deck Cu pKa
   table from being matchable at C_S=10 µF/cm².
5. **Phase 6β deliverable should be pKa_eff ordering, not cd.**
   cd can stay O₂-Levich-limited across the cation series.

The revised artifact (`PHASE_6B_V9_PHASES_A_B_RESULTS_2026-05-10.md`)
now reflects all of these.  Sequenced plan: A.1 instrumentation
→ A.2 re-do at V_kin → B.2 densified ramp + ablations → v10
capacity branch → revised Phase D / E.
