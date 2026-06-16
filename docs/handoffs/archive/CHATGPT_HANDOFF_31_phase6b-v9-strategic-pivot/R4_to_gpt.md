# R4 — counterreply

## Section 1: Per-issue responses

### **Re point 1** (signed σ helper) — **Accept**
```python
def sigma_C_m2_to_counts_pm2(sigma_signed_C_m2: float) -> float:
    """Convert signed Stern surface charge (C/m²) to signed
    counts per pm² (Singh's native unit).  The pKa layer is
    responsible for `sigma_singh = max(0.0, −signed_counts)` to
    apply Singh's anode-clamp convention."""
    return sigma_signed_C_m2 * 6.022e23 / 96485 / 1e24
```
* Helper returns signed.
* `build_pka_shift` applies the anode-clamp: pass
  `max(0.0, −signed_counts)` into Singh Eq. (4).
* Unit test: Singh Cu K⁺ values round-trip; anode bias
  (σ_signed > 0) → Singh σ = 0 → ΔpKa = 0.

### **Re point 2** (sequence skips σ_S(V)/V_kin diag) — **Accept**
Corrected sequence:

1. **Phase 0** — Deliverable contract drafted, sent, held.
2. **v10a** — Langmuir cap + integrated diagnostics
   (F₀, Γ, θ, capped forward rate, denominator); tests for
   Γ → Γ_max and Γ_max → ∞.
3. **Minimum V-sweep diagnostic** (~1 day): run v10a across the
   V_RHE walk at smoke kinetics + λ=0 baseline AND λ=1 with
   v10a's cap.  Record σ_S(V), per-branch currents R_2e/R_4e,
   dR_net/dσ_S sensitivity, c_H(0), c_K(0).  Hold-output the V_kin
   candidate per the gates below.
4. **A.2 re-run at V_kin** (~1 day) — λ ramp + densified k_hyd
   ramp at V_kin with v10a cap.
5. **Source/sink/override plumbing ablations** (~2 days) —
   `apply_h_source`/`apply_k_sink`/`override_pka_sigma_S_ablation`
   flags + 3-experiment ablation matrix at V_kin.  This BEFORE
   v10b literature calibration (point 13 fix).
6. **v10b** — literature calibration of Γ_max + k_des; document
   CMK-3 capacitance literature with area normalization (point 7);
   byte-equivalence at Γ_max → ∞.
7. **B.2** — densified k_hyd × λ ramp at V_kin with calibrated v10b.
8. **D / E** — Phase D K-only fit; Phase E predictive holdout.

### **Re point 3** (V_kin selector depends on smoke params) — **Accept**
You're right — the sensitivity index changes once Γ_max/k_des
are calibrated.

**Concrete fix:** V_kin selection uses an explicit, predeclared
parameter set: smoke baseline k_hyd = 1e-3, smoke k_des = 1.0,
Γ_max = 1 monolayer, λ = 1, all per-cation at K⁺.  Document this
explicitly in the V_kin selection record so post-v10b shifts of
V_kin are auditable.

Fallback if no V passes the filters: pick V_RHE = +0.30 V (the
"transition" voltage from Phase A's V_RHE walk) and flag the
result as "no σ-active V satisfies all filters at smoke
kinetics; using transition voltage as fallback."

### **Re point 4** (Ring Onset Potential extraction) — **Accept**
You're right.

**Concrete fix:** define per-observable extraction functions:
* **Ring Onset Potential (V@0.01 mA/cm²):** find smallest V where
  model `gross_h2o2_current` exceeds 0.01 mA/cm² in the cathodic
  sweep direction.  Linear interpolate between bracketing V grid
  points.
* **Max Ring Current:** max model ring current over a fixed
  V window (e.g. V ∈ [+0.2, −0.4]).
* **Highest H₂O₂ Selectivity:** max model RRDE-equivalent
  H₂O₂% over the same fixed V window.
* **Number of e- transferred:** model `j_total / (F · O₂ flux)`
  averaged over the cathodic plateau.

Each defined as a function of the model's V-walk output, not
of the experiment's measured potential.

### **Re point 5** (±30% invalid for voltages) — **Accept**
You're right.

**Concrete fix:** observable-specific tolerances (predeclare):
* Voltages (Ring Onset, etc.): ±50 mV absolute.
* Currents (Max Ring): ±30% relative, with absolute floor
  0.01 mA/cm² (below floor → exact-zero comparison).
* Selectivity (H₂O₂%): ±10 percentage points absolute.
* n_e: ±0.5 absolute.
* ΔpKa: ±30% relative on mean(|ΔpKa_deck|).

### **Re point 6** (N-1 of N too weak) — **Accept**
You're right.

**Concrete fix:** designate **primary observables** that MUST
pass for the per-cation result to be valid:
* H₂O₂ Selectivity (the deck's headline cation-effect observable).
* Ring Onset Potential.

**Secondary observables** (support, not blocking):
* Max Ring Current.
* Number of e-.
* Surface pH / pKa_eff.

Phase E pass criteria: ALL primaries pass + at least 2/3
secondaries pass per cation, across all 4 cations.

### **Re point 7** (CMK-3 lit ≠ Stern C_S) — **Accept**
You're right.  Porous carbon literature often reports F/g
or F/cm² normalized by BET surface area, not geometric electrode
area.  Stern model wants the local interfacial C_S per geometric
area of the electrode at the OHP.

**Concrete fix:** before importing literature C_S values:
1. Document each literature source's normalization convention
   (per gram, per BET m², per geometric cm², supercapacitor vs
   ORR conditions).
2. Compute the equivalent geometric C_S using the catalyst's
   roughness factor (typical CMK-3 RF ~ 50-500; deck-cited if
   available).
3. Sanity-bound the geometric C_S vs the model's current
   10 µF/cm² choice.
4. If literature is sparse, treat C_S as a fit parameter with
   priors from at most one defensible literature point.

This lives in `docs/phase6/CMK3_capacitance_literature.md`.

### **Re point 8** (imposed Singh σ = ablation only) — **Accept**
You're right.  An imposed σ bypasses the coupled Poisson +
Stern system; not a self-consistent physical model.

**Concrete fix:**
* Rename the flag to `pka_override_sigma_singh_counts_pm2` and
  label it `pKa_override_ablation` in code + docs.  Default None.
* Only the local-Stern path is a coupled physical run; the
  imposed-σ path is for ablation (A3 in the previous matrix:
  "what does ΔpKa look like at Singh's deck-cited σ?").
* Phase D fits against the COUPLED local-Stern run, not the
  imposed-σ ablation.

### **Re point 9** (silent Γ clamp hides bugs) — **Accept**
You're right.

**Concrete fix:**
```python
def update_gamma_from_solution(ctx, ...):
    ...
    gamma_new_unclamped = ...  # from Langmuir Picard formula
    gamma_new = max(0.0, min(Gamma_max, gamma_new_unclamped))
    if gamma_new != gamma_new_unclamped:
        warnings.warn(
            f"update_gamma_from_solution: clamped Γ "
            f"{gamma_new_unclamped:.6e} → {gamma_new:.6e} "
            f"(out of [0, {Gamma_max}])",
            RuntimeWarning,
        )
    ...
```
* Warm-start restore clamps without warning (expected on
  warm-restart with state from a different parameter set).
* Picard-update clamp warns if it activates — flags formula
  inconsistency or near-bound numerics for the rung_callback to
  log.
* Test: at v10's intended physical regime (k_hyd ≤ calibrated),
  the warning never fires.

### **Re point 10** (gap_Cu = 0 for Li is rounding) — **Accept**
You're right.  Storing r_H_El_Cu = 132.00 pm rounds the Li back-fit
value; high-precision is somewhere in the range that gives the
observed −0.44 ΔpKa, but my code drops the precision.

**Concrete fix:**
* Option A: store high-precision back-fit values
  (`r_H_El_pm_Cu` field rounded to 5+ decimal places).
  Recompute by inverting Singh Eq. 4 with the deck's −0.44
  ΔpKa target and Cu σ.
* Option B: parameterize directly in the geometric factor
  G = 2·A·z·σ·r_H_El·(1 − r_M-O²/r_H_El²).  Store G_per_cation_Cu
  (one number per cation at Cu σ) and use that as the calibration
  target.  Bypasses the gap singularity.

I prefer **Option B** — G is what enters Eq. (4) directly,
gap parameterization is an indirect proxy.  Re-parameterizing
in G also makes the transfer rules cleaner:
* G_carbon = G_Cu + ΔG (additive); apply same ΔG to all cations.
* G_carbon = ρ_G · G_Cu (multiplicative); apply same ρ_G to all.

### **Re point 11** (selectivity max needs fixed scan window) — **Accept**
You're right.

**Concrete fix:** experimental V scan window:
* Brianna 2019 K2SO4 LSV: V_RHE range from −0.06 to +1.14 V
  (per Phase F's extracted data); cathodic regime where
  selectivity is reported.
* Predeclared model scan: V_RHE ∈ [−0.4, +1.0] (model's
  C+D convergence window from CLAUDE.md).
* H₂O₂ Selectivity max-extraction: max over **the overlap**
  V_RHE ∈ [−0.06, +1.0], at the deck's experimental pH bin.

For per-cation max selectivity from the Cation Summary Table,
each cation's per-pH max-selectivity record has a "Corr Pot" —
take that as the model's lookup voltage too, with the
±50 mV tolerance.

### **Re point 12** (Phase 0 contract too vague) — **Accept**
You're right.

**Concrete fix — Phase 0 deliverable contract MUST include:**
1. Summary of v9 findings (cd is O₂-Levich-limited; Γ unphysical
   at all converged k_hyd; σ-mapping ambiguity).
2. Proposed acceptance bundle with EXPLICIT observable + extraction
   function + tolerance per row.
3. Phase E.0 data-reduction protocol (pH bin, cycle aggregation,
   V-window per observable, primary/secondary designation).
4. Ask: which observables are mandatory vs nice-to-have; what
   tolerance bounds are acceptable; whether the proposed
   experimental V window is correct; whether Phase E should be
   pure predictive holdout or a small per-cation refit is OK.
5. Estimated 2-week scope for v10a + minimum diagnostic + A.2 +
   plumbing ablations + v10b + B.2.
6. Decision lever: if group rejects pKa-only and wants
   ring/selectivity match too, scope grows another 1-2 weeks.

### **Re point 13** (plumbing ablations BEFORE v10b calibration) — **Accept**
You're right.  Fitting Γ_max and k_des before verifying the
source/sink plumbing risks calibrating against a sign bug.

**Concrete fix:** sequence reordered (already in R4 point 2 above):
* v10a → minimum V-sweep → A.2 at V_kin → **plumbing ablations
  (apply_h_source/apply_k_sink/override) at V_kin** → v10b
  literature calibration → B.2 → D / E.

The plumbing ablations verify:
* H source actually moves c_H (A1).
* K sink actually moves c_K (A2).
* Imposed σ at Singh's Cu σ produces Singh's Cu ΔpKa (A3 sanity
  check on Singh algebra; no coupled physics).

Only after these pass do we calibrate Γ_max/k_des against
experimental data.

---

## Section 2: Updated artifact (final synthesis if APPROVED)

The revised plan in
`docs/phase6/PHASE_6B_V9_PHASES_A_B_RESULTS_2026-05-10.md` will
absorb (when this loop ends):

* Phase 0: deliverable contract with explicit observables /
  tolerances / data-reduction protocol.
* Final sequence: 0 → v10a + integrated diags → V-sweep diag →
  A.2 at V_kin → plumbing ablations → v10b literature
  calibration → B.2 → D → E.
* Magnitude metric: mean fractional ΔpKa error ≤ 30%.
* Unit conversion: signed counts/pm² helper, anode-clamp at
  pKa layer.
* σ mapping: local Stern only is coupled physical; imposed
  Singh σ is ablation-only with `pKa_override_ablation` label.
* V_kin selector documented with explicit smoke parameter set
  + fallback rule.
* Per-observable extraction functions defined (onset = threshold
  crossing, max = max over window, etc.).
* Per-observable tolerances (mV for V, % for selectivity, etc.).
* Primary (selectivity + ring onset) vs secondary observables.
* CMK-3 capacitance literature with area normalization
  documented.
* Γ clamp + WARN-on-Picard-clamp; warm-restart clamp silent.
* Geometric factor G parameterization for r_H_El transfer rules.
* Selectivity max-extraction over predeclared V window overlap.

---

## Section 3: Continued critique prompt

Review the updated plan and my responses to your earlier issues.
Push back on responses where I defended poorly — name which point.
Raise any new issues the updated plan creates. Re-issue any earlier
issue you don't think I addressed. Same numbered format and same
verdict line at the end:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN
