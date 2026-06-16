# Handoff 45 — Phase 7.3 pH-coupled ORR mechanism: implementation plan review

## What I want from you

This is the **follow-on to Handoff 44** (you approved the mechanism
brainstorm + 4-tier taxonomy). The user has now told me **we will NOT get
any new experimental data** (no rpm series, no N₂ H₂O₂-reduction scans, no
surface modification, no acid recipe). So I wrote an **implementation plan**
to pursue the best hypothesis (the Tier-1 backbone A+C+G) using only data
already on disk, and I want you to tear it apart before I execute.

Specifically:
1. **Is the no-new-data framing honest and sound?** I claim the pH axis is
   a second identifying dimension that partially substitutes for the
   missing rpm series (pH-flat channels = direct water routes;
   pH-dependent = c_H-coupled C). Is "trend attribution, not absolute
   identification" a legitimate goal, or am I fooling myself?
2. **Are the two kill-gates (G1 after A0, G2 after C) in the right place
   and decisive?** Will they actually catch a dead hypothesis cheaply, or
   can a wrong model sneak through?
3. **Is the anti-overfitting design real?** (shared kinetics +
   leave-one-pH-out + parameter-count ≪ per-pH). With only pH 2/4/6
   digitized curves + 6 scalars + one full LSV, is there enough signal to
   avoid fitting noise?
4. **Implementation feasibility / sequencing errors.** Is A0 really
   driver-only? Does the C2 O₂-regeneration coupling create a circular
   dependency? Anything mis-ordered, under-specified, or that will silently
   fail in this PNP-BV stack?
5. **Anything I'm still missing** given the constraint.

This is a PLAN review, not a code review. I want a plan I can execute
without discovering a fatal flaw in week 3.

---

## Section 1: Context bundle

### Carryover from Handoff 44 (the approved diagnosis)
- A frozen "water-route" dual-pathway ORR model (2e + 4e Butler–Volmer,
  water as proton donor, NO c_H factor) fits ONE condition (K₂SO₄ RRDE,
  bulk pH 6.39, disk + ring) but is **structurally pH-FLAT**, while the
  data has strong pH structure.
- **S1 (onset):** d(onset)/dpH = +41 mV/pH on RHE = −18 mV/pH on SHE
  (R²=0.80; leave-one-out slope 26–53; robust SIGN, soft magnitude). On
  the SHE axis onset is ~flat (mean +0.17 V, humped) → fingerprint of a
  **proton-uncoupled rate-determining first electron transfer**. The model
  is RHE-flat (η = V_RHE − E°(RHE), no c_H) → 0 mV/pH → fails.
- **S2 (selectivity):** rises 20% → ~73% with pH, midpoint ~pH 2.5–3.
- **S3 (ring collapse):** peroxide ring current collapses in strong acid.
- Approved mechanism taxonomy:
  - **Tier 1 (pursue):** A0/A1 = SHE-anchored first ET (onset); C1+C2 =
    pH-dependent peroxide consumption (selectivity + acid ring).
  - **Tier 2 (infrastructure):** G = surface-pH coupling; H = bisulfate.
  - **Tier 3 (confounds):** D site speciation, E cation/field, F pzc.
  - **Tier 4 (near-dead):** B superoxide pKa-4.8.
- **Identifiability ceiling (your catch):** at ONE rotation rate, direct-4e
  and series-2e+2e are degenerate; n_e(V) cannot separate them. The locked
  pH-6.39 "direct 2e/4e competition" holds only up to that degeneracy.

### Correct slope formula (agreed in H44)
`dE_RHE/dpH = 0.0592·[1 − m·β/(αn)]`, β = −∂log₁₀(c_H,surf)/∂pH_bulk.
A0 = pure m=0 → +59 mV/pH (parameter-free). A1 = + correction → ~41.
m≈0.18 only if β=1; with surface pH 9–10 under load, β≪1 and m is
larger / identifiable only via modeled surface c_H.

### The solver (facts I verified in code)
- 2D PNP-BV FE forward solver; species O₂, H₂O₂, H⁺; analytic Bikerman
  K⁺/SO₄²⁻; Stern C_S=0.20 F/m²; formulation `logc_muh`
  (c_H = exp(μ_H − em·z_H·φ)); log-rate BV.
- **Per-reaction E_eq is supported:** `forms_logc_muh.py:545-548` reads
  `rxn["E_eq_model"]` per reaction and builds its own
  `eta_j = _build_eta_clipped(E_eq_j)` where
  `eta_raw = phi_applied − phi − E_eq_const`. ⇒ **A0 = set
  E_eq_v = E0_SHE + 0.0592·pH per condition in the driver, no solver
  edit.**
- **Per-reaction conc factors are supported:** reactions carry
  `cathodic_species`, `cathodic_conc_factors` (species index + power +
  c_ref_nondim). The 4e acid route already uses a c_H factor
  (`c_ref_nondim=C_HP_HAT`). ⇒ **C1 = a new reaction with
  cathodic_species=H₂O₂(idx1) + a c_H,surf factor = config-level.**
- **Homogeneous source/sink terms exist:** `water_ionization.py` builds
  homogeneous residual terms for water self-ionization (H⁺ source/sink).
  ⇒ **C2 (H₂O₂ sink + O₂ source) = a new solver term modeled on it.**
- Potential convention (a known caveat): solver uses ψ_bulk = 0,
  V_model = V_RHE directly (no OCP shift). Deck convention would apply
  V_OCP_RHE ≈ 0.47+0.197+0.059·pH. The pH-series run used per-pH
  OCP = 0.664+0.059·pH. BV kinetics use absolute E° differences so a
  constant offset is benign, but a pH-dependent frame error could FAKE the
  onset slope.
- Anchor recipe (water routes): kw_eff_ladder=None (anchor at full Kw),
  k0 AdaptiveLadder ≥6 inserts, anchor at V_solver=0 with linear_phi IC
  (the debye_boltzmann Picard IC mis-seeds water routes). Studies cost
  minutes–hours.

### The data we actually have (the ENTIRE evidence base — no more coming)
| Source | What | Fidelity |
|---|---|---|
| `k2so4_ph6p39_rrde_binned.csv` | full raw disk+ring LSV, **pH 6.39** | high (numeric) — this is the LOCKED fit |
| `digitized_experimental_3panel.json` | disk/ring/sel vs V at **pH 2, 4, 6** (283–756 pts/panel) | medium (figure digitization) |
| `metrics.json:exp_info` | onset/max_ring/peak_sel scalars at pH 1.65, 2.35, 3.42, 4.21, 5.21, 6.39 | low (sheet convention; sel area-mixed, overweights ring ×1.786) |
| `mangan_deck_p15_h2o2_current_v2.csv` | Cs⁺ pH 4 ring volcano | high (vector SVG) |

Locked pH-6.39 fit (to be preserved): log f_2w=−1.009, log f_4w=−12.309,
α_2w=0.577, α_4w=0.305, L_eff=21.7 µm; the 2e water route dominates
(f_2w ≫ f_4w), so it dominates onset.

### Honesty constraints I'm trying to respect
- The rpm degeneracy is NOT broken by this plan — I only claim pH-TREND
  attribution.
- Digitized data is lower fidelity than the one raw LSV; sel scalars use a
  nonstandard area-mixed convention.
- The pH-6.39 fit must be PRESERVED (it's the one high-fidelity anchor).
- Overfitting is the central risk with so few conditions.

---

## Section 2: The artifact under review (the implementation plan)

### 0. Thesis
Best hypothesis = Tier-1 backbone: ONE SHE-anchored, ~proton-uncoupled
rate-determining first ET sets onset (A), followed by c_H-dependent
peroxide consumption setting branching (C), both reading surface pH (G),
H bounding the acid end. Goal = **sufficiency / trend-attribution** (can a
physical pH coupling reproduce the trends with shared kinetics + few
params, preserve pH-6.39, predict held-out pH), NOT absolute
identification (rpm-degenerate). Scope: pH 2–6.4 (alkaline 10/12 out).

### 1. Data inventory — as the table above.
Objective: fit RAW disk + RAW ring (model-side N=0.224); selectivity/n_e
diagnostics only.

### 2. Pre-work P0 (computational gates, no new data)
- **P0.1 single-convention potential-frame derivation.** Pin
  V_RHE↔V_SHE↔V_model, where E_eq_v lives, ONE place proton dependence
  enters (formal shift XOR kinetic c_H factor). Unit test: with E0_SHE
  tuned so E_eq(2e)=0.695 at pH 6.39, SHE-anchored driver reproduces the
  locked pH-6.39 fit byte-for-byte.
- **P0.2 onset re-extraction harness.** From digitized disk curves
  (pH 2/4/6) extract onset at multiple small absolute-current thresholds
  (0.05/0.1/0.2 mA/cm²); slope+CI vs threshold. (Digitized data is RHE
  axis only — cannot redo iR/axis variants; only threshold-robustness is
  available.) Combine with 6 scalars. Output: onset-vs-pH(±CI) = the
  A-gate baseline.
- **P0.3 fit/test split + scorer.** Calibrate at pH 6.39 (raw) + pH 4
  (3-panel); **hold out pH 2 and pH 6** for prediction. Extend
  `phase7_wls.py score_dual_series` to a pH-series aggregate.

### 3. Phase 1 (A0/A1) — SHE-anchored first ET = onset spine + KILL GATE
Mechanistic ideal: one RDS first ET shared by 2e & 4e; branching
downstream. Tractable approx this phase: keep parallel 2e/4e but
SHE-anchor BOTH routes' E_eq (full shared-first-ET refactor deferred).
- A0 (driver-only): E_eq_v = E0_SHE + 0.0592·pH per condition. Refit
  {E0_SHE(2e), k0(2e)} at calibration pH; PREDICT onset at pH 2/4/6 +
  scalars. Coarse grid first.
- A1 (needs G): if A0 over-predicts (+59 vs ~41), add c_H,surf^m kinetic
  prefactor; fit m; check sub-Nernstian slope + onset hump (with β
  caveat).
- **★ KILL GATE G1:** if A0+A1 can't reproduce onset POSITIVE SIGN within
  P0.2 CI while preserving pH-6.39 → SHE-anchoring FALSIFIED; STOP,
  re-enter brainstorm. Don't build C on a dead spine.

### 4. Phase 2 (G) — surface-pH kinetic coupling
Every c_H rate reads SURFACE c_H = exp(μ_H − em·z_H·φ) at boundary, not
bulk. Diagnostic: extract β across the curve; check m physical (0≤m≲1).
Test: byte-equivalent when no c_H rate active.

### 5. Phase 3 (C1+C2) — peroxide consumption + KILL GATE
- C1 (config): new reaction, cathodic_species=H₂O₂, c_H,surf factor; adds
  disk current, consumes peroxide.
- C2 (solver term ~water_ionization): homogeneous H₂O₂ sink + O₂ source
  (2H₂O₂→2H₂O+O₂), coupled so transport decides O₂ re-entry. pH law (if
  any) uses LOCAL conditions, not forced into C1's c_H^m form. Mass test:
  ∫O₂-source = ½∫H₂O₂-sink.
- Calibrate {C1-rate, C1-order, C2-rate} at low pH. C ∝ c_H,surf →
  negligible at pH 6.39 → locked fit preserved (verify).
- **★ KILL GATE G2:** if C1+C2 can't make sel-rise AND pH-2 ring
  suppression while preserving pH-6.39 → consumption insufficient; flag
  B/E.

### 6. Phase 4 — joint pH-series fit + leave-one-pH-out
Fit A+C+G across pH 2/4/6 + scalars, SHARED kinetics + {E0_SHE, m,
C1-rate, C1-order, C2-rate}, anchor weight on pH-6.39 raw. Success: 3-panel
trends with effective params ≪ per-pH brute force AND leave-one-pH-out
error within digitization σ. H sensitivity: bisulfate reservoir at pH 2 as
a bracket (acid recipe/activities unknown), not a fitted point.

### 7. Phase 5 — cross-condition (slide-15 Cs⁺ pH4) + write-up
Apply FROZEN A+C+G to slide-15; add E (cation OHP coupling via existing
Stern/Bikerman+Boltzmann) ONLY if the +0.09 V anodic-peak offset + bump
residuals demand it. Write up identified-vs-degenerate, prediction-vs-fit,
digitization caveats.

### 8. Risks register
identifiability ceiling not broken; digitization fidelity; overfitting
(shared kinetics + LOO + param-count); convention double-count (P0.1
gate); solver cost (coarse grid for gates, confirm before full runs;
adjoint tape hygiene); anchor robustness (reuse water-route recipe;
AdaptiveLadder may need re-tune for c_H-coupled stiffness).

### 9. Milestones
P0 gates → G1 → G wiring → G2 → Phase 4 LOO PASS → Phase 5. First step:
P0.1+P0.2 + A0 driver + G1 on coarse grid (cheapest validate-or-kill).

---

## Section 3: Critique prompt

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
