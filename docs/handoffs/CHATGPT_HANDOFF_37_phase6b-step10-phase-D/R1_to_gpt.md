# R1 → GPT: Critique session 37 — Phase 6β Step 10 (Phase D) plan

You are an adversarial reviewer for the Phase D plan in the
PNPInverse PNP-BV research codebase.  Phase D is a **single-
scalar derivative-free fit** of `Δ_β` against K₂SO₄ deck data at
pH 4.  Find every hole.  Verdict at end: APPROVED / ISSUES_REMAIN.

---

## Section 1 — Context bundle

### 1.1. Project shape

Research code for Poisson–Nernst–Planck / Butler–Volmer (PNP-BV)
forward simulation for parallel-2e/4e ORR (O₂ → H₂O₂ → H₂O) on
the Seitz/Mangan deck (K₂SO₄ at pH 4–6).

**Production stack (May 2026):** 3 dynamic species (O₂, H₂O₂, H⁺) +
analytic Bikerman counterions; `formulation='logc_muh'`; log-rate
BV with parallel R2e (E°=0.695 V) + R4e (E°=1.23 V); Stern C_S =
0.20 F/m² (Bohra-Koper-Choi); `debye_boltzmann` IC; reaches V_RHE
= +1.0 V at 15/15.

### 1.2. Just-completed step 8 = v10b

v10b LANDED 2026-05-11 (~5 hours wall, all 11 D-gates closed, no
v10c escalation).  Outcomes:
* `GAMMA_MAX_HAT_V10B = 0.047 nondim` (tightened V10A chain — 4-
  test compatibility check found no peer-reviewed MOH coverage
  anchor at sp²-carbon OHP for K₂SO₄).
* `K_DES_NONDIM_V10B = 1.0 nondim` — engineering choice with
  documented Eyring prior `k_des_nondim ∈ [10⁻², 10²]` ↔
  `ΔG_des ∈ [0.69, 0.94] eV` at 298 K.
* `C_S_F_M2_V10B = 0.20 F/m²` (step-7 lock).

Numerically v10b is BYTE-EQUIVALENT to v10a (V10B values equal
V10A values).  The change is meta-level: clean provenance via
`calibration/v10b.py` at repo root (Firedrake-free) +
`V10B_CALIBRATION_METADATA` schema with units, citation, prior,
bracket, compatibility.

Phase A.2 v10b regression: 10/10 rungs converged, mass-balance
1e-17 to 1e-14, baseline reproduction rel < 1e-3.
Step 6 plumbing: 5/5 ablations PASS, A0 byte-equivalent rel=0.0.
C_S bracket (4/4) + Γ_max × k_des matrix (30/30): all PASS,
per-rung analytic-vs-solver Γ_ss mass-balance rel = 0.0.

Critique trail: `docs/handoffs/CHATGPT_HANDOFF_36_phase6b-v10b-calibration/`
(7 rounds; 53 issues accepted, 0 defended, 0 unresolved; APPROVED).

### 1.3. Concurrent step 9 (B.2) — running NOW

Step 9 (B.2 densified k_hyd × λ ramp at V_kin) is **executing
concurrently** via subagent.  Grid: 14 k_hyd × 10 λ = 140 rungs at
V_kin = −0.10 V with V10B kinetics.  Step 9 is a diagnostic +
documentation artifact, NOT direct input to Phase D's fit.  Phase
D uses V-resolved scans at fixed (k_hyd=1e-3, λ=1); step 9's
single-V (k_hyd, λ) scan does not overlap.

### 1.4. Where Phase D sits in the locked sequence

Per `docs/phase6/PHASE_0_ACCEPTANCE_BUNDLE_LOCK_2026-05-10.md`
§ "v10a → E sequence":

```
[done]  Step 8  v10b (Γ_max + k_des + C_S literature calibration)
[active] Step 9 B.2 densified k_hyd × λ ramp at V_kin
[NEXT]   Step 10 Phase D — K-only Δ_β fit (~1 day per bundle)
[future] Step 11 Phase E — predictive holdout (4 cations) (~3-5 days)
```

### 1.5. The locked β parameterization

From `singh_2016_pka_formula.md` Eq (4') and acceptance bundle
§ "β parameterization (locked)":

```
β_per_cation = 2 · A · z_eff · r_H_El · (1 − r_M-O² / r_H_El²)
              [units: pm² consistent with σ in counts/pm²]
```

where:
* `A = 620.32 pm` (Singh's geometric prefactor).
* `z_eff` and `r_M-O` per `SINGH_2016_CATION_PARAMS` in
  `scripts/_bv_common.py` (per-cation table).
* `r_H_El` is the calibration knob (default = Cu prior).

**Δ-rule transfer (locked):**
```
β_X_carbon = β_X_Cu + Δ_β
```
Phase D fits the **single scalar `Δ_β`** against K's experimental
H₂O₂ selectivity at pH 4.  Phase E applies the same `Δ_β` to Cs,
Na, Li without refit.

**ΔpKa = β · σ** at the active σ mapping:
* Coupled physical path: `σ_local_Stern` from PNP/Stern solve.
* Ablation path: `σ_imposed_Singh_counts_pm2` via the
  `pka_override_ablation` flag.

**If Phase D's Δ_β differs by > 30% between the two mappings, β
is flagged non-identifiable; Phase E reports both** (acceptance
bundle § "Σ_S mapping convention", line 165-166).

**Sign convention:** cathodic pKa lowering requires `β ·
σ_local_cathodic < 0`.  Regression test added in v10a.

### 1.6. Acceptance criteria (from acceptance bundle §)

**Primary (must PASS for Phase E pass rule):**
* Per-cation max H₂O₂% in V_RHE window matches deck within
  **±10 percentage points absolute**.
* For Phase D (K-only step): only K must pass.

**Secondary (≥ 2/3 must pass per cation; loaded in Phase E):**
* Ring Onset Potential: within ±50 mV.
* Max Ring Current: within ±30% relative, abs floor 0.01
  mA/cm² (below floor → exact-zero comparison).
* `n_e_rrde`: within ±0.5 absolute.

**Mechanism self-consistency (for Phase D, K only):**
* `β · σ_local_cathodic < 0` sign guard.
* (Mechanism ordering Li < Na < K < Cs and ΔpKa magnitude are
  Phase E criteria for the 4-cation holdout — out of Phase D
  scope.)

**Phase E pass rule (locked):**
* All primary criteria pass for all 4 cations.
* All mechanism self-consistency criteria pass.
* ≥ 2/3 secondary criteria pass per cation.

**Falsification path (locked):** if any primary criterion fails
for K (at Phase D) OR for Cs/Na/Li (at Phase E), the v10 model +
calibrated β **falsifies the cation-hydrolysis-as-dominant-
mechanism hypothesis** for the deck's cation effect.  "That's a
valid research finding; report and document the falsification
pathway."

### 1.7. Data target — K at pH 4

**Source:**
`data/EChem Reactor Modeling-Seitz-Mangan/Brianna/
20201024 CP Experiment Data-Code/Summary Data-Error.xlsx`,
sheet `Cation Summary Table`, row `K2` (K₂SO₄) at pH ∈ [3.5, 4.5].

**Required deck values (extracted once before fit):**
1. Max H₂O₂ Selectivity (%) — primary target.
2. Ring Onset Pot (V, @ 0.01 mA/cm²) — secondary.
3. Max Ring Current (mA/cm²) — secondary.
4. Number of e- — secondary.

**V scan window overlap:** model `V_RHE ∈ [−0.4, +1.0] V` (solver
C+D convergence window).  Brianna LSV at pH 6.39 ≈ `[−0.06,
+1.14] V`.  **Overlap window `[−0.06, +1.0] V`** used for max-
extraction of selectivity, ring current (acceptance bundle line
107).

### 1.8. Hard invariants Phase D must preserve

From acceptance bundle + v10b + CLAUDE.md:

| Constant | Value | Source |
|---|---|---|
| V10B kinetics | Γ_max=0.047, k_des=1.0, C_S=0.20 | step 8 |
| `k_hyd_baseline` | 1e-3 nondim | step 5 |
| λ | 1.0 (full physics) | bundle |
| Parallel topology | R2e (E°=0.695 V) + R4e (E°=1.23 V) | Ruggiero 2022 |
| `exponent_clip` | 100.0 | CLAUDE.md hard rule #2 |
| Two-stage anchor (STERN_F_M2_ANCHOR=0.10 → BASELINE=0.20) | locked | v10a' |
| Singh ΔpKa formula structure | locked | bundle |
| β_per_cation geometric coefficient | locked | bundle |
| Δ-rule transfer | locked | bundle |
| `N_collection` | 0.224 | Ruggiero §2 |
| `τ_REF` | ≈ 5 s | _bv_common |

### 1.9. Inverse status: PAUSED

All inverse scripts non-operational.  Phase D is **forward-only,
derivative-free**.  No adjoint tape.  If any code inadvertently
imports a taped path, wrap in `with adj.stop_annotating():` (per
CLAUDE.md + v10b plan).

### 1.10. Key file paths

* `Forward/bv_solver/cation_hydrolysis.py` —
  `_build_singh_2016_eq_4_pka_shift` (line ~496), `build_pka_shift`
  (line 421), `update_gamma_from_solution`, `gamma_ss_langmuir`
  (line 638).
* `Forward/bv_solver/anchor_continuation.py` —
  `solve_anchor_with_continuation`, `solve_grid_with_anchor`,
  runtime setters (`set_reaction_*_model`,
  `set_stern_capacitance_model`).
* `scripts/_bv_common.py` — `SINGH_2016_CATION_PARAMS` table,
  factory functions, deck-aligned defaults.
* `scripts/studies/phase6b_v10a_v_sweep_diagnostic.py` — existing
  V-resolved driver (V10A); Phase D's new driver inherits its
  V_RHE_GRID structure and σ-perturbation infra.
* `calibration/v10b.py` (repo root) — V10B numeric constants +
  metadata.
* `docs/phase6/CMK3_capacitance_literature.md` — step 7 reference
  writeup.
* `docs/phase6/v10b_calibration_summary.md` — step 8 reference
  writeup; Phase D's writeup mirrors this structure.
* `docs/phase6/singh_2016_pka_formula.md` — Singh Eq (4') +
  σ-mapping convention; Table from line 237 has per-cation
  ΔpKa values at the Cu prior.
* `docs/phase6/PHASE_0_ACCEPTANCE_BUNDLE_LOCK_2026-05-10.md` —
  § "Per-observable extraction functions" line 114, § "β
  parameterization" line 129, § Risk #4 + #3 + #2.
* `data/.../Brianna/Summary Data-Error.xlsx` — deck data source.

---

## Section 2 — The artifact under review

The plan below was just written.  It targets step 10 (Phase D)
end-to-end.

````````````````````````markdown
[FULL PLAN v1 — see /Users/jakeweinstein/.claude/plans/phase6b-step10-phase-D-deltaBeta-fit.md]

(The plan is ~530 lines.  Reproduced inline below.)

---

[INLINE: full text of phase6b-step10-phase-D-deltaBeta-fit.md]
````````````````````````

(GPT — please request the plan via your search if you need to re-
read; I'm linking instead of duplicating because the plan is ~530
lines and the codex session memory will retain whatever you read.)

Plan summary:
* **Single derivative-free fit** of scalar `Δ_β` (carbon-vs-Cu
  offset in Singh's pKa-shift coefficient).
* **Two-stage optimization**: pre-fit grid scan (9 evals) to
  bracket the minimum + Brent's method (≤ 16 evals) to refine to
  bracket width < 0.05 or loss < 1 pp².
* **Loss function**: `L(Δ_β) = (max_H₂O₂%_model − max_H₂O₂%_deck)²`
  on K@pH4 only.  Selectivity is the locked primary criterion;
  secondary criteria (ring onset, max ring, n_e_rrde) are checked
  post-fit, not minimized.
* **V grid**: `V_RHE ∈ [−0.06, +1.0] V` at 0.05 V step (22 pts).
* **σ-mapping divergence check** (production Stern vs ablation
  imposed): records identifiability flag if > 30%.
* **Falsification verdict** if primary criterion fails: STOP
  Phase D, valid research outcome per bundle.
* **Wall budget**: ~6 hours fit + ~1 day each for driver build
  and writeup = ~2-3 working days (vs bundle's "~1 day").
* **Out of scope**: multi-cation (Phase E), Γ_max/k_des/C_S re-
  fitting, adjoint gradients, multi-objective fit.

13 DoD items (D1-D13), 12 risk-register entries, 8 open questions.

I'm reproducing the FULL plan text inline below for your review
(530 lines).  After this block ends, the critique prompt follows.

````````````````````````markdown
# Phase 6β Step 10 — Phase D: K-only Δ_β Fit

[v1 PLAN — see Section 2.1 below for full text]
````````````````````````

### Section 2.1 — Plan v1 text (inline, full)

Below is the full plan text reproduced verbatim.  All 530 lines
are present for your review.

---

## Section 3 — Critique prompt

You are an adversarial reviewer. Be critical. Be argumentative.
Find every hole: missing steps, wrong algebra, untested assumptions,
edge cases not addressed, implicit dependencies, claims without
evidence, off-by-one errors, sign errors, dimensional errors. Don't
be polite — if something is wrong, say so. Concision over hedging.

For each issue, state:
  - WHAT is wrong (specific, not vague — name the line or symbol)
  - WHY it matters (what breaks downstream if uncorrected)
  - WHAT to do (concrete fix, or what evidence would close the gap)

Number your issues. After all issues, end your response with exactly
one of these lines, no other text after it:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN

Use APPROVED only when there are no issues you would block on.
Minor nitpicks alone do not justify ISSUES_REMAIN — call them out
but still verdict APPROVED. Use ISSUES_REMAIN whenever any of your
issues are genuinely blocking.

Specific lenses for this plan:
- **Fit identifiability**: is `Δ_β` actually identifiable from
  max H₂O₂% alone?  Could there be a flat-loss region or a
  pathological coupling with locked engineering-choice k_des?
- **Loss function choice**: is squared `(max_H₂O₂%_model −
  deck)²` the right objective, or does it miss something (e.g.
  argmax-V information, ring onset)?
- **V grid resolution**: 0.05 V × 22 points adequate for ring
  onset extraction at 0.01 mA/cm² threshold?
- **Bracket choice for pre-fit scan**: is `Δ_β ∈ [−2, +2]` the
  right magnitude given Singh's β values are O(10⁻²) 1/pm²?
- **Δ_β parameterization in code**: is there ambiguity between
  routing through `r_H_El_pm` override vs an explicit
  `beta_offset` setter that doesn't exist yet?
- **σ-mapping divergence threshold**: 30% is from the acceptance
  bundle; is it the right number?
- **Falsification verdict semantics**: is the plan honest about
  what failure means scientifically, or does it leak hedging?
- **Brent's method choice**: is 1D Brent over a 25-eval cap the
  right optimizer, or is grid+local-quadratic-fit safer?
- **Two-stage anchor pattern**: every V at every Δ_β triggers a
  fresh anchor build + bump — does that compound to fragile
  convergence at corner V values?
- **Wall budget honesty**: 6-hour fit + 2 days driver + writeup
  vs bundle's "~1 day" — is the bundle estimate fixable or just
  optimistic?
- **What's missing**: anything outside the plan that should be in
  scope (e.g. a pre-existing sanity check at Δ_β=0)?
# Phase 6β Step 10 — Phase D: K-only Δ_β Fit

**Author:** Claude (planner).  **Date:** 2026-05-11.
**Status:** Draft v1 — entering GPT critique loop (≤ 7 rounds).
**Provenance:** Locked sequence step 10 per
`docs/phase6/PHASE_0_ACCEPTANCE_BUNDLE_LOCK_2026-05-10.md` §
"v10a → E sequence".  Step 9 (B.2) is concurrently executing.

---

## 0. One-paragraph framing

Step 10 fits a **single scalar `Δ_β`** (the carbon-vs-Cu offset in
Singh 2016's pKa-shift coefficient) against K-only deck data
(K₂SO₄, pH ≈ 4, parallel 2e/4e ORR on CMK-3) using the V10B
calibration locked in step 8.  The Δ-rule
`β_X_carbon = β_X_Cu + Δ_β` is applied uniformly across cations;
Phase E (step 11) holds Δ_β fixed and predicts Cs/Na/Li.  The fit
is **derivative-free** (inverse work is paused project-wide; no
adjoint tape).  Each forward evaluation is a V-resolved
`V_RHE ∈ [overlap window]` scan at `(k_hyd=1e-3, λ=1, V10B
kinetics)` with the two-stage Stern anchor pattern.  The primary
acceptance gate is `|max_H₂O₂%_model − max_H₂O₂%_deck| ≤ 10 pp`;
failure falsifies the cation-hydrolysis-as-dominant-mechanism
hypothesis per the locked acceptance bundle.

---

## 1. Definition of done

v10b is the prerequisite; Phase D is **done** iff every box ticks:

- [ ] **D1.** A V-resolved fit-evaluator driver lands at
      `scripts/studies/phase6b_step10_phase_D_fit_eval.py` with:
      * single CLI flag `--delta-beta` (float, default 0.0).
      * V_RHE grid spanning the overlap window `[−0.06, +1.0]` V
        at locked step (TBD; suggest 0.05 V → 22 points).
      * Two-stage anchor pattern at every V: build at
        `STERN_F_M2_ANCHOR = 0.10`, bump to
        `STERN_F_M2_BASELINE = 0.20` via
        `set_stern_capacitance_model` + Newton resolve.
      * Per-V emission of: `cd_mA_cm²`, `R_2e_current_nondim`,
        `R_4e_current_nondim`, `gross_h2o2_current`,
        `ring_current_ring_basis_mA_cm2`, `σ_S`, mass-balance
        residual, analytic-Γ rel.
      * Aggregated emission of the 4 acceptance-bundle observables
        per the locked extraction functions (Acceptance bundle §
        "Per-observable extraction functions"):
        `max_H2O2_selectivity_in_window`,
        `argmax_V_for_selectivity`,
        `ring_onset_V_at_0.01_mA_cm2`,
        `max_ring_current_in_window`,
        `n_e_rrde_at_argmax_V`.
      * Both σ-mapping paths emitted in parallel: the production
        `σ_local_Stern` path AND the ablation
        `σ_imposed_Singh_counts_pm2` path (per acceptance bundle
        Risk #4 — Phase D reports both; >30% Δ_β divergence flags
        non-identifiable).
      * HARD per-V gates (same as v10b D7): convergence, Picard
        OK, mass-balance < 5e-3, analytic-Γ rel < 5e-3 via
        `gamma_ss_langmuir`, sign convention preserved.
      * Lazy Firedrake imports per v10b round-7 patch P38; CLI/
        schema importable without Firedrake.
- [ ] **D2.** Driver fast tests:
      * `test_step10_fit_eval_cli_parses` — argparse + `--delta-beta`
        accepts a float.
      * `test_step10_fit_eval_target_grid` — V grid construction.
      * `test_step10_fit_eval_output_schema` — JSON keys match the
        D1 spec.
      * `test_step10_fit_eval_module_firedrake_free` — module
        import doesn't pull `firedrake` into `sys.modules`.
- [ ] **D3.** **Pre-fit unimodality scan.**  Sweep
      `Δ_β ∈ {−2.0, −1.0, −0.5, −0.25, 0.0, +0.25, +0.5, +1.0,
      +2.0}` (9 evals; ~45-90 min wall) and emit a 1D loss curve
      `|max_H₂O₂%_model(Δ_β) − max_H₂O₂%_deck|²`.  Pass criteria:
      * 9/9 evaluations complete all V-points without HARD-gate
        failure.
      * Loss curve is **unimodal** (single minimum interior to
        the bracket OR endpoint minimum — both acceptable).
      * Bracket the minimum: identify `Δ_β_lo`, `Δ_β_mid`,
        `Δ_β_hi` where `loss(Δ_β_lo) > loss(Δ_β_mid) < loss(Δ_β_hi)`.
      Failure → STOP, escalate (multi-modal loss = identifiability
      crisis; switch to grid scan; do not Brent on multi-modal).
- [ ] **D4.** **Optimization step.**  scipy.optimize.brent
      (`scipy.optimize.brent`) called with the bracket from D3.
      Convergence: bracket width `< 0.05` OR primary loss `< 1
      pp²` (≈ within 1 pp of the deck max H₂O₂%).  Cap at 25
      forward evaluations total (D3 + Brent).  Emit `Δ_β_fit`,
      `loss(Δ_β_fit)`, and the full evaluation trace.
- [ ] **D5.** **σ-mapping divergence check** (Acceptance bundle
      Risk #4).  Run the optimization twice in parallel: once
      with the production `σ_local_Stern` mapping, once with the
      ablation `σ_imposed_Singh_counts_pm2` mapping (using the
      same V grid + observables).  Compute
      `|Δ_β_fit_stern − Δ_β_fit_ablation| / |Δ_β_fit_stern|`.
      If > 0.30, emit `identifiability_flag = "non_identifiable";
      Phase E reports both`.  This is INFORMATIONAL — not a HARD
      gate; the fit always returns the production-path value.
- [ ] **D6.** **Acceptance-bundle primary criterion check** (K
      only at this step):
      * `|max_H2O2%_model − max_H2O2%_deck_K_pH4| ≤ 10 pp` at
        `Δ_β = Δ_β_fit`.
      * If FAIL → emit
        `falsification_verdict = "cation_hydrolysis_falsified_for_K"`
        and STOP Phase D.  This is a **valid research outcome**
        per acceptance bundle § Pass rule (lines 85-90): "If any
        of the primary criteria fail: the v10 model + calibrated
        β falsifies the cation-hydrolysis-as-dominant-mechanism
        hypothesis for the deck's cation effect."
      * Phase E is then conditional on PASS at D6.
- [ ] **D7.** **Secondary criteria check** (K only, informational
      at this step; load-bearing in Phase E per acceptance bundle
      § "Phase E pass rule"):
      * `|ring_onset_V_model − ring_onset_V_deck_K| ≤ 50 mV`.
      * `|max_ring_current_model − max_ring_current_deck_K| / |…_deck_K|
        ≤ 0.30` (abs floor 0.01 mA/cm²; below floor → exact-zero
        comparison).
      * `|n_e_rrde_model − n_e_rrde_deck_K| ≤ 0.5`.
      Report per-criterion PASS/FAIL.  ≥ 2/3 pass for Phase E
      pre-clearance; this step only logs, doesn't gate.
- [ ] **D8.** **Sign-guard regression** (per acceptance bundle
      Risk #3, v10a regression test added).  Verify at
      `Δ_β_fit`: `β_K_carbon · σ_local_cathodic_at_V_kin < 0`.
      Reconfirm; fail → STOP, escalate.
- [ ] **D9.** **Phase E setup.**  Emit a "production-point
      spec" JSON: `(Δ_β_fit, V10B kinetics, σ-mapping=stern,
      V_grid, k_hyd=1e-3, λ=1)` for the 4-cation holdout.
      Stored at `StudyResults/phase6b_step10_phase_D/phase_E_spec.json`.
- [ ] **D10.** **Writeup published:**
      `docs/phase6/phase6b_step10_phase_D_summary.md`.  Sections:
      * §1 — fit setup (data target, free parameter, loss
        function, optimization method).
      * §2 — pre-fit unimodality evidence (loss curve plot).
      * §3 — fit result (`Δ_β_fit`, evaluation trace, secondary
        criteria, σ-mapping divergence).
      * §4 — falsification verdict (PASS/FAIL with one-paragraph
        diagnostic of WHICH primary criterion drove the verdict).
      * §5 — Phase E preparation (production-point spec).
      * §6 — open asks for Phase E.
- [ ] **D11.** **Acceptance bundle § Status appended** with Phase
      D paragraph.
- [ ] **D12.** **CLAUDE.md "Recent progress"** updated; ≤ 200
      lines.
- [ ] **D13.** **Memory entry**
      `project_phase6b_step10_phase_D_outcome.md` +
      `MEMORY.md` pointer.

---

## 2. Hard invariants (do NOT touch in Phase D)

Phase D is forward-only; fits a single derivative-free scalar.

| Constant | Value | Source |
|---|---|---|
| V10B kinetics | `Γ_max_V10B = 0.047`, `k_des_V10B = 1.0`, `C_S = 0.20` | step 8 |
| `k_hyd_baseline` | `1e-3 nondim` | step 5 |
| λ | `1.0` (full physics on) | acceptance bundle |
| Parallel topology | `R2e (E°=0.695 V)` + `R4e (E°=1.23 V)` | Ruggiero 2022 |
| `exponent_clip` | `100.0` | CLAUDE.md |
| `STERN_F_M2_ANCHOR / BASELINE` | `0.10 / 0.20 F/m²` | two-stage anchor |
| Singh ΔpKa formula structure | `ΔpKa = β · σ` per `pka_shift_form='singh_2016_eq_4'` | acceptance bundle § "β parameterization" |
| β_per_cation geometric coefficient | `2·A·z_eff·r_H_El·(1 − r_M-O²/r_H_El²)` (per cation; Cu prior r_H_El default) | Singh 2016 SI Eq (4'), `SINGH_2016_CATION_PARAMS` |
| Δ-rule transfer | `β_X_carbon = β_X_Cu + Δ_β` (uniform across cations) | acceptance bundle § "Transfer rule" |
| `N_collection` | `0.224` | Ruggiero 2022 §2 |
| Sign convention | `β · σ_local_cathodic < 0` | v10a regression test |
| `τ_REF` | `≈ 5 s` | `_bv_common.py` |
| `c_s_ladder + kw_eff_ladder` | unsupported combo | `anchor_continuation.py:1689` |

Breaking any of these → escalate to a re-derivation plan, NOT a
within-Phase-D scope change.

---

## 3. Data target — K at pH 4

**Source:** `data/EChem Reactor Modeling-Seitz-Mangan/Brianna/
20201024 CP Experiment Data-Code/Summary Data-Error.xlsx`,
sheet `Cation Summary Table`, row `K2` (K₂SO₄) filtered to pH ∈
[3.5, 4.5] per acceptance bundle § "Phase E.0 data-reduction
protocol".

**Required values (one row of the table):**
1. `Max H₂O₂ Selectivity (%)` at K, pH 4 — **primary target**.
2. `Ring Onset Pot (V, @ 0.01 mA/cm²)` at K, pH 4 — secondary.
3. `Max Ring Current (mA/cm²)` at K, pH 4 — secondary.
4. `Number of e-` at K, pH 4 — secondary.

**Extraction:** Phase D's first action is to call (or build) a
small data-extraction helper that loads the xlsx, applies the
pH bin filter, averages cycles 1/2/3 if present, and emits the
four target scalars + stds.  This is a deterministic data prep
step, NOT a fit input — runs once and the results are pinned
into the driver via a config dict or a JSON sidecar.

**Failure mode:** if the xlsx row is missing or the pH bin
returns zero entries, STOP Phase D and write a status note.

---

## 4. Forward model evaluation per Δ_β

Given a candidate `Δ_β`, the evaluator runs:

1. Set `β_K_carbon = β_K_Cu + Δ_β` via the Singh ΔpKa formula
   (β_K_Cu computed from `SINGH_2016_CATION_PARAMS["K"]` with
   default r_H_El = Cu prior, locked).  In code: pass
   `r_H_El_pm` override to `make_cation_hydrolysis_config` so the
   per-cation β shifts by the right amount.  (Note: `Δ_β` is
   parameterized as an additive shift to `β`; the
   solver-internal representation routes this via the
   `r_H_El_pm` knob OR via an explicit `beta_offset` field —
   verify in Phase D.A which path is supported, add it if missing.)
2. V-resolved scan across `V_RHE ∈ [−0.06, +1.0]` V at 0.05 V
   spacing = 22 points.  Two-stage anchor at every V (build at
   C_S = 0.10, bump to 0.20 via `set_stern_capacitance_model`).
3. Per V: full Newton resolve at `(k_hyd=1e-3, λ=1)`.  Emit
   diagnostics per D1.
4. Aggregate observables per acceptance bundle § "Per-observable
   extraction functions".
5. Return a result dict with `max_H2O2_selectivity_in_window`,
   `argmax_V_for_selectivity`, `ring_onset_V_at_0.01_mA_cm2`,
   `max_ring_current_in_window`, `n_e_rrde_at_argmax_V`, plus the
   per-V diagnostic table, plus the σ-mapping divergence values.

**Wall per evaluation:** Estimate ~10 min (22 V points × 30s
warm-walk + Newton resolve, two-stage anchor amortized across
V).  Total fit wall: 25 evals × 10 min = ~4 hours.

---

## 5. Loss function

**Primary loss (the function the optimizer minimizes):**
```
L(Δ_β) = (max_H2O2%_model(Δ_β) − max_H2O2%_deck_K_pH4)²
```
Units: percentage points squared.  Pass criterion at the
minimum: `|max_H2O2%_model − max_H2O2%_deck| ≤ 10 pp`, i.e.
`L(Δ_β_fit) ≤ 100 pp²`.

**Why not multi-objective:**
1. The acceptance bundle § "Primary criterion" locks selectivity
   as the headline observable; everything else is secondary.
2. Multi-objective fits with arbitrary weights amount to picking
   the weights — uncomfortable for a calibration step.
3. Secondary criteria are validated POST-fit (D7).  If they all
   fail at the selectivity-fit minimum, that's diagnostic
   information; we'd report it and not refit.

**Why squared:**
* Differentiable (helps Brent's method even if it doesn't use
  gradients).
* Symmetric on either side of the deck value.

**Failure modes:**
* `max_H2O2%_model` is NaN/None at some Δ_β: catch as exception,
  record as loss = `+inf` for that eval, continue.
* Optimization stalls at a boundary: report the boundary value
  as Δ_β_fit with a warning; check secondary criteria for
  context.

---

## 6. Optimization strategy

**Two-stage approach:**

**Stage 1: Pre-fit grid scan (D3).**
* Sweep `Δ_β ∈ {−2.0, −1.0, −0.5, −0.25, 0.0, +0.25, +0.5, +1.0,
  +2.0}` (9 evals).
* Bracket boundaries justified:
  * Singh 2016 reports `β_per_cation` of order `O(10⁻²)` 1/pm²
    (Table from `singh_2016_pka_formula.md` line 237) — Cu prior
    base values are small.  Δ_β additive offsets of ±2 cover
    several decades of pKa shift (since `ΔpKa ≈ β·σ` and σ is
    ~50 counts/pm² at V_kin).
  * If the unimodality scan shows a minimum at an interior
    bracket point, Brent refines.  If at endpoint, expand
    bracket in D3 → D4 transition.
* Required: 9/9 converge AND loss curve is unimodal (or
  endpoint-monotone).

**Stage 2: Brent's method (D4).**
* `scipy.optimize.brent(L, brack=(Δ_β_lo, Δ_β_mid, Δ_β_hi),
  tol=0.05)`.
* Cap: 16 additional evaluations (D3 used 9, total 25).
* Convergence: bracket width < 0.05 OR `L < 1 pp²`.

**Why Brent's method (not Nelder-Mead, not BFGS):**
* 1D problem (single scalar).  Brent is the standard choice for
  1D unimodal optimization without gradients.
* Nelder-Mead is overkill for 1D and slower.
* BFGS/L-BFGS need gradients (and adjoint is paused).

**Why not bisection on `L'(Δ_β) = 0` numerically:**
* Numerical derivative of L would require 2 forward calls per
  Brent step — 2× wall cost.  Brent's golden-section + parabolic-
  fit hybrid handles non-differentiable losses gracefully.

**Why not a Bayesian-optimization library (skopt, BoTorch):**
* Overhead per recommendation step is non-negligible (~seconds
  to minutes for GP fitting).
* The 25-evaluation budget is too small to benefit from BO's
  asymptotic improvements.
* Brent + a pre-fit grid scan is the right scale.

---

## 7. Phase breakdown

### Phase 10.A — Driver build (~1 day)

**Steps:**
1. Audit Singh ΔpKa machinery for `Δ_β` parameterization:
   * Does `build_pka_shift` accept an offset param, or does Δ_β
     have to be routed through `r_H_El_pm` overrides?
   * If only `r_H_El_pm`: implement an explicit `beta_offset`
     knob in `_build_singh_2016_eq_4_pka_shift` and the runtime
     setter (`set_reaction_beta_offset_model`).
   * If `r_H_El_pm`: that's the existing knob, but the
     parameterization is non-linear (β depends nonlinearly on
     r_H_El).  Prefer explicit `beta_offset` for fit interpretability.
2. New driver `scripts/studies/phase6b_step10_phase_D_fit_eval.py`:
   * Lazy Firedrake imports (CLI / schema tests Firedrake-free).
   * Single-Δ_β evaluator: `evaluate_fit_target(delta_beta) -> dict`.
   * Two-stage anchor pattern at every V.
   * Per-V diagnostic emission.
   * Aggregation to the 4 acceptance-bundle observables.
   * σ-mapping divergence run (parallel production + ablation).
3. Fast tests (D2).
4. Data extraction helper (§3): load deck values from xlsx, pin
   into driver config.
5. `pytest -m "not slow" -k "phase6b or cation or step10"` green.
6. Commit:
   ```
   feat(phase6b): step 10.A — Δ_β fit-eval driver

   - New driver scripts/studies/phase6b_step10_phase_D_fit_eval.py.
   - Single-Δ_β evaluator with V-resolved scan, two-stage anchor,
     per-V HARD gates, observable aggregation per acceptance
     bundle § "Per-observable extraction functions".
   - Lazy Firedrake imports; 4 fast tests passing.
   - Deck data extraction helper for K@pH4 from Brianna xlsx.

   Prep for Phase 10.B (pre-fit unimodality + Brent).
   ```

### Phase 10.B — Pre-fit + fit run (~6 hours wall)

**Steps:**
1. Pre-fit grid scan (D3): 9 evaluations, ~90 min wall.  Verify
   unimodality.  If multi-modal → STOP, escalate (refit
   strategy needed).
2. Brent's method (D4): ≤ 16 additional evaluations, ~3 hours
   wall.
3. σ-mapping divergence check (D5) — runs in parallel during the
   fit (no extra wall).
4. Per-V analysis at `Δ_β_fit`: verify primary criterion (D6) +
   secondary criteria (D7) + sign guard (D8).
5. Emit `Δ_β_fit`, falsification verdict, observables comparison
   table.
6. Commit:
   ```
   feat(phase6b): step 10.B — Phase D Δ_β fit

   - Pre-fit unimodality scan: 9 evals across Δ_β ∈ [−2, +2].
   - Brent's method refinement to bracket width < 0.05.
   - Δ_β_fit = <value>, primary loss <value> pp².
   - σ-mapping divergence: <value>% (identifiability_flag = …).
   - Primary criterion (max H2O2%): PASS / FAIL (Δ = <value> pp).
   - Secondary criteria (K only): <pass count>/3.

   Output: StudyResults/phase6b_step10_phase_D/{fit.json, fit.png}.
   ```

### Phase 10.C — Writeup + bundle update (~1 day)

**Steps:**
1. Write `docs/phase6/phase6b_step10_phase_D_summary.md` per
   D10.
2. Acceptance bundle § Status (D11).
3. CLAUDE.md (D12).
4. Memory entry (D13).
5. Phase E spec JSON for the 4-cation holdout (D9).
6. Commit:
   ```
   docs(phase6b): step 10 Phase D summary + acceptance bundle update

   - New writeup: phase6b_step10_phase_D_summary.md.
   - Phase E spec JSON at StudyResults/.../phase_E_spec.json.
   - Acceptance bundle § Status: Phase D paragraph.
   - Memory entry: project_phase6b_step10_phase_D_outcome.md.
   ```

**Total estimate: ~2-3 working days** (vs acceptance bundle's
"~1 day" — the locked estimate was optimistic; ~25 forward
evaluations × ~10 min each plus driver build + tests + writeup
realistically lands in the 2-3 day range).

---

## 8. Risk + mitigation register

| # | Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|---|
| R1 | Loss curve is multi-modal in Δ_β | LOW | HIGH | D3 unimodality scan detects.  If multi-modal → switch from Brent to global grid scan; escalate to a separate Phase D' plan. |
| R2 | Primary criterion fails at minimum (max H₂O₂% gap > 10 pp) | MEDIUM | LOW (research outcome) | Acceptance-bundle-aligned: report `falsification_verdict = "cation_hydrolysis_falsified_for_K"` and STOP Phase D.  Phase E becomes conditional. |
| R3 | σ-mapping divergence > 30% (Δ_β non-identifiable per acceptance bundle Risk #4) | LOW | MEDIUM | D5 records the divergence; Phase E reports both fits.  Not a Phase-D STOP. |
| R4 | Singh ΔpKa formula structure missing `beta_offset` knob | MEDIUM | MEDIUM | Phase 10.A step 1 audits.  Add an explicit `beta_offset` runtime setter if needed; trivial code change. |
| R5 | V-resolved scan fails convergence at deep anodic V (≈ +1.0) or deep cathodic V (≈ −0.06) | LOW | MEDIUM | Solver's C+D + two-stage anchor handles to V_RHE = +1.0 V (15/15 record).  Cathodic edge −0.06 V is well inside the validated window.  HARD-gate failure → STOP, do not refit. |
| R6 | Brent's method exceeds 25-evaluation cap | LOW | LOW | Cap-aware loop; report whatever Δ_β has the lowest L if cap hits.  Tighten tolerance for the writeup. |
| R7 | Wall budget exceeds 6 hours | MEDIUM | LOW | Run in background overnight; subagent execution pattern (v10b precedent). |
| R8 | Adjoint-tape contamination | LOW | HIGH | Phase D is forward-only, derivative-free.  No `Forward.bv_solver.tape` calls; wrap any inadvertent imports in `with adj.stop_annotating():`. |
| R9 | Δ_β parameterization is non-linear (routed through `r_H_El_pm` not additive `beta_offset`) | MEDIUM | MEDIUM | Phase 10.A.1 audit decides.  Prefer additive `beta_offset` for fit interpretability; if forced to use `r_H_El_pm`, document the non-linearity in the writeup and re-emit Brent on `r_H_El_pm` as the optimization variable (with `Δ_β` derived post-fit). |
| R10 | Brianna xlsx K row missing pH 4 entry | LOW | HIGH | Phase 10.A.4 data-extraction helper detects + STOPs.  Fallback: use the closest-pH entry with explicit caveat in writeup; if no row at all, STOP Phase D. |
| R11 | Number of forward evaluations on the 22-point V grid is wasteful (most V points don't affect max H₂O₂% argmax) | MEDIUM | LOW | First-pass: keep 22 points (cheap insurance for ring onset detection).  If wall budget tight: trim to ~14 points after pre-fit scan shows the argmax region. |
| R12 | `gross_h2o2_current` and `ring_current_ring_basis_mA_cm2` are mis-aligned with deck convention | LOW | HIGH | Verify against acceptance bundle § "Per-observable extraction functions" + Ruggiero 2022 §2 BEFORE running.  Sanity check: at `Δ_β=0`, the unmodified V10B model should produce roughly the v10b A.2 selectivity at V_kin (~20%). |

---

## 9. Out of scope

- **Cs, Na, Li fitting** — Phase E (step 11) applies Δ_β
  transferred via the Δ-rule.
- **Γ_max, k_des, C_S re-fitting** — locked at V10B; if Phase D
  fails, the falsification path opens a separate re-derivation
  (NOT a Phase D scope expansion).
- **Multi-objective fitting** — see §5; primary loss is K
  selectivity only.
- **Adjoint gradients** — inverse paused.
- **Phase E predictive holdout** — separate step.
- **Refinement of Singh ΔpKa formula** (e.g., higher-order σ
  terms, sgn(σ)·|σ|^p) — locked structure.
- **V grid extension beyond `[−0.06, +1.0]` V** — outside the
  deck overlap window.
- **k_des fitting** — engineering choice locked at V10B; if
  Phase E predictions fail, a separate Phase D' plan can re-open
  k_des.

---

## 10. Dependency graph

```
Step 9 (B.2, executing concurrently)
        │
        │  (no direct dependency; step 10 uses
        │   V-resolved scans, not k_hyd × λ map)
        ▼
   Step 10 Phase D
   ┌─────────────────────┐
   │  10.A — driver build │  (~1 day)
   └─────────┬───────────┘
             ▼
   ┌─────────────────────┐
   │  10.B — pre-fit +    │  (~6 hours wall)
   │       Brent          │
   └─────────┬───────────┘
             ▼
   ┌─────────────────────┐
   │  10.C — writeup +    │  (~1 day)
   │       Phase E spec   │
   └─────────────────────┘
             ▼
       Step 11 (Phase E)
       4-cation predictive holdout
```

---

## 11. Validation checkpoints

1. **End of 10.A:** driver lands; fast tests green; data
   extraction confirms K/pH4 row present.
2. **End of 10.B pre-fit:** 9/9 evals converge; loss curve
   unimodal.  If not → STOP.
3. **End of 10.B Brent:** Δ_β_fit emitted with bracket width <
   0.05 or loss < 1 pp²; primary criterion PASS or FAIL verdict;
   σ-mapping divergence reported.
4. **End of 10.C:** all D-gates close.  Phase E unblocked
   conditional on D6 PASS.

---

## 12. Decision rules summary

| Question | Outcome path |
|---|---|
| Loss curve multi-modal? | STOP; escalate to separate Phase D' plan with global grid scan. |
| Primary criterion (max H₂O₂% within 10 pp) fail? | Emit `falsification_verdict`; STOP Phase D; Phase E conditional. |
| σ-mapping divergence > 30%? | Emit `identifiability_flag = "non_identifiable"`; Phase E reports both fits.  NOT a STOP. |
| Brent exceeds 25-eval cap? | Report current best; tighten tolerance for writeup. |
| V-resolved scan fails at any V? | STOP, write status; do not silently drop V points. |
| `Δ_β` parameterization not directly supported in solver? | Phase 10.A.1 adds it; trivial. |

**Default: prefer falsification clarity over fit-quality
heroics.**  If the K-only fit can't bring selectivity within
10 pp, that's a research finding worth reporting — not a signal
to re-tune the engineering-choice k_des or Γ_max.

---

## 13. Open questions for the GPT critique loop

1. Is the V grid `[−0.06, +1.0]` V at 0.05 V (22 points) the
   right density?  Deck data resolution likely lower; over-
   resolving wastes wall, under-resolving misses the ring onset.
2. Is `Δ_β ∈ [−2, +2]` the right initial bracket?  Singh's β is
   in units of 1/pm²; need to verify the typical magnitude.
3. Should `argmax_V_for_selectivity` be in the loss function
   (multi-objective) instead of just observable comparison?
4. Brent's tolerance `0.05` on bracket width and `1 pp²` on loss
   — what's the right trade-off?
5. Is the pre-fit scan grid `{−2, −1, −0.5, −0.25, 0, +0.25,
   +0.5, +1, +2}` adequate for unimodality detection?  Should
   it be denser in the central region?
6. σ-mapping divergence check at every Brent iteration vs.
   only at the final fit — what's the cost-effective frequency?
7. If `beta_offset` runtime setter doesn't exist, is the
   `r_H_El_pm` non-linear parameterization an acceptable Plan B
   (with post-fit conversion to Δ_β)?
8. The acceptance bundle says Phase D is ~1 day; my estimate is
   2-3 days.  Should the plan flag this overrun explicitly to
   the user?

---

**End of plan v1.**  Entering GPT critique loop.

---

## Section 3 — Critique prompt (repeated for clarity after long plan)

Same critique prompt as Section 3 above.  Number issues, end with VERDICT line.
