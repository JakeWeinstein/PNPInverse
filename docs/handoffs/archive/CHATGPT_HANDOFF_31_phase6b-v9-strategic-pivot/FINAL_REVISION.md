# Final revision — Critique Session 31 (phase6b-v9-strategic-pivot)

* **Loop ended:** Round 5 with VERDICT: APPROVED.
* **Revised artifact:**
  `docs/phase6/PHASE_6B_V9_PHASES_A_B_RESULTS_2026-05-10.md`
* **Total issues across all rounds:** 16 (R1) + 14 (R2) + 13 (R3)
  + 6 (R4) + 3 (R5 non-blocking) = **52 distinct issues**.

## Big-picture takeaway

This session converted a 3-pick strategic snapshot (read Singh
first; pKa_eff ordering as deliverable; A.1 instrumentation
first) into a fully-sequenced 11-step plan with explicit
acceptance metrics, primary/secondary observable designation,
predeclared extraction functions, σ-mapping conventions,
ablation-before-calibration ordering, and an external
deliverable contract gating the whole effort.

The largest reframings were:

1. **Slide 27 ≠ independent target.** Slide 27 reproduces
   Singh's Cu pKa table; calibrating to it is a self-consistency
   check.  Real validation is `Summary Data-Error.xlsx` Cation
   Summary Table.
2. **Sequence reordered to v10a-first.** The user's "A.1 first"
   pick was technically correct in spirit, but A.1's
   "instrument unphysical v9" was wasted effort.  v10a-first
   integrates the diagnostics that A.1 was supposed to add.
3. **Phase 0 deliverable contract added.** Before any 2-week
   commitment, send a one-page contract to Linsey/Brianna with
   explicit per-observable extraction + tolerance + Phase E.0
   data reduction.  GPT correctly flagged that the Seitz/Mangan
   group's acceptance is the actual deliverable boundary, not
   Claude's framing.

## Issue ledger (all rounds)

### Addressed (52 issues — all accepted; revised in artifact)

#### R1 (16 issues)

1. pKa overstated as deliverable → "mechanism subdeliverable"
   + experimental subdeliverable.
2. **Slide 27 = Singh reproduction (load-bearing).**
   Phase D fits K experimental data, not slide 27.
3. **Per-cation data in folder (load-bearing).**
   `Summary Data-Error.xlsx` Cation Summary Table = real
   validation target.
4. Spearman ≥ 0.9 with n=4 fake precision → exact ordering +
   mean fractional error.
5. Spacing ratio underdefined → ΔpKa vector L1 error +
   collapse failure rule.
6. **Singh σ already extracted** (`singh_2016_pka_formula.md`
   §5.2) → re-verify, don't re-read.
7. σ_local-vs-cell mapping not equivalent → bracket under
   both, treat as assumption.
8. **A.1 full diagnostics waste effort on unphysical v9** →
   v10a-first reorder.
9. v10 underestimated → split into v10a (5-7 days) + v10b
   (1-2 weeks).
10. Langmuir Picard formula derived explicitly:
    `Γ_ss(λ) = λF₀ / ((1−λ) + λk_des + λB + λF₀/Γ_max)`.
11. override_pka_sigma_S unit trap → signed-counts helper +
    explicit unit labels.
12. V_kin not reproducible → predeclared selection rule with
    explicit smoke parameter set + fallback.
13. Branch-current rates ≠ electron currents → record both
    separately.
14. r_H_El transfer near-singular → parameterize in β
    (R5 refined: β = 2·A·z·r_H_El·(1 − r_M-O²/r_H_El²)).
15. C_S not scoped → σ_S(V) bracket at C_S ∈ {10, 30, 50, 100}.
16. **Deliverable contract before 2-week commitment (load-
    bearing).**

#### R2 (14 issues)

1. Magnitude metric Σ→Σ/n fix.
2. **Unit conversion 1e16 → 1e24 pm² per m².**
3. Anode-OER scaling dropped; two mapping conventions only.
4. **ψ_S ≠ V_RHE.**  σ_S from solved fields.
5. v10a needs integrated diagnostics, not deferred to A.1.
6. Γ clamp [0, Γ_max] + warn on out-of-bounds Picard.
7. V_kin filter on hydrolysis sensitivity, not Levich-ratio.
8. RRDE selectivity uses ring/disk formula, not raw rates.
9. Phase E data reduction protocol predeclared (pH bin, cycle
   aggregation, V window, tolerance).
10. Negative-result phrasing softened: "falsifies this closure".
11. δ_HE = r_M-O − r_H_El sign convention.
12. CMK-3 literature uncited → v10b prerequisite note.
13. CP_data.csv = summary, .mat = raw.
14. Contract before D/E design, not 2-week commit.

#### R3 (13 issues)

1. Signed σ helper (anode-clamp at pKa layer).
2. Sequence: v10a → V-sweep → V_kin → A.2 (not A.2 direct).
3. V_kin selector params predeclared; fallback rule.
4. Ring Onset = output threshold, define interp function.
5. Observable-specific tolerances (mV / pp / relative).
6. Primary vs secondary observables.
7. CMK-3 lit needs area normalization documentation.
8. Imposed Singh σ = ablation only (`pka_override_ablation`).
9. Γ clamp on Picard with warning; warm-restart silent.
10. β parameterization (refined in R5 to σ-independent form).
11. Selectivity max over fixed window (no model-experiment
    contamination).
12. Phase 0 contract includes data-reduction protocol.
13. **Plumbing ablations BEFORE v10b literature calibration.**

#### R4 (6 issues)

1. β must be σ-independent: β = 2·A·z·r_H_El·(1 − r_M-O²/r_H_El²),
   units pm².
2. Max-selectivity: separate `max_in_window` and `argmax_V` obs.
3. Onset direction: sweep anode→cathode.
4. n_e_rrde = 4·|I_D|/(|I_D| + I_R/N) RRDE formula.
5. V_kin fallback from actual v10a walk, no hard-coded V.
6. CMK-3 RF range removed from default smoke priors.

#### R5 (3 non-blocking notes)

1. β units = pm² (not pm).
2. Ring current basis: ring-area normalization in extraction.
3. β sign guard test: cathodic pKa lowering preserved.

### Defended

None.  Every issue across 52 rounds was accepted.

### Unresolved

None.  VERDICT: APPROVED.

## Summary

The 5-round critique loop converted the user's strategic
pivot into a fully-specified 11-step plan with explicit
acceptance metrics, predeclared extraction functions,
σ-mapping conventions, and a Phase 0 deliverable contract
gating all downstream work.  The revised plan in
`docs/phase6/PHASE_6B_V9_PHASES_A_B_RESULTS_2026-05-10.md`
now contains the full sequence: Phase 0 contract → v10a +
diagnostics → V-sweep → V_kin selection → Phase A.2 → plumbing
ablations → CMK-3 lit note → v10b calibration → Phase B.2 →
Phase D K-only fit → Phase E predictive holdout.

The most actionable immediate next step is **Phase 0 contract
drafting** (~1 hour), to be routed to Linsey/Brianna before
any v10a code work begins.
