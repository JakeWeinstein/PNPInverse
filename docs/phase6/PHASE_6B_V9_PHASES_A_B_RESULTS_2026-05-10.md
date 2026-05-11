# Phase 6β v9 post-Gate-4 plan — Phases A + B results

**Date:** 2026-05-10
**Plan:** `docs/phase6b_v9_post_gate4_plan.md` §A (observability),
§B (k_hyd ramp).
**Critique sessions (5+5 rounds):**
* `docs/handoffs/CHATGPT_HANDOFF_30_phase6b-v9-cd-invariance/`
  — cd-invariance finding (capped at ISSUES_REMAIN).
* `docs/handoffs/CHATGPT_HANDOFF_31_phase6b-v9-strategic-pivot/`
  — strategic pivot **(VERDICT: APPROVED on round 5)**.

## TL;DR

* The cd plateau at V ≤ +0.10 V is **at the O₂ Levich limit**:
  4·F·D_O₂·C_O₂/L_eff = 5.50 mA/cm² with code constants
  (D_O₂=1.9e-9, C_O₂=1.2 mol/m³, L_eff=16 µm); observed cd
  −5.531 mA/cm² is within 0.6 % of this.  Apparent electron
  count is near 4 → 4e O₂ ORR dominant on the plateau (per-
  branch assembly required for any selectivity claim; legacy
  `peroxide_current = R_2e − R_4e` is deprecated for the
  parallel topology).
* The H⁺ Levich floor hypothesis from the original artifact
  is **retracted**.  H⁺ Levich (1e at pH 4) ≈ 0.56 mA/cm² —
  10× too small to bottleneck the observed 5.5 mA/cm² cd.
* Phase A's V=−0.20 V test was **deep inside the O₂ plateau**;
  the cd-invariance under λ ramp is not a finding about
  hydrolysis at all.  Cation hydrolysis cannot move what's
  already O₂-bottlenecked.
* Phase B's "no usable k_hyd window" stands as a description
  of v9 numerics but is **not a physics conclusion**: at
  k_hyd ≤ 1e-2 we observe valid Picard data; at k_hyd ≥ 1e-1
  Γ explodes via the σ_S-dependent ΔpKa factor and Picard
  fails.  The high-k_hyd regime is unphysical anyway because
  Γ has no Langmuir capacity (architectural debt — see below).
* **Phase 6β scope reframing (session 31):** slide 27 is Singh's
  Cu pKa table reproduced (per
  `docs/phase6/singh_2016_pka_formula.md` line 221), NOT an
  independent experimental calibration target. The real
  experimental target is `Summary Data-Error.xlsx::Cation
  Summary Table`: per-cation Ring Onset Potential, Max Ring
  Current, **Highest H₂O₂ Selectivity (%)**, Number of e⁻ at
  4 cations × 5–6 pH values.  pKa_eff ordering is a
  **mechanism subdeliverable**; the **acceptance subdeliverable**
  is the per-cation experimental observable match.

## Phase A — observability check at V=−0.20 V (this run)

**Setup:** Smoke kinetics (k_hyd = k_prot = 1e-3 nondim, k_des = 1).
Anchor at V=+0.55 V at λ=0; warm-walk +0.55 → −0.20 V at λ=0
(9/9 converged); λ ramp 0.0 → 0.25 → 0.50 → 0.75 → 1.0 at
V=−0.20 V (all rungs converged with Picard 2 iters/rung).

**V_RHE walk (λ=0 baseline):**

| V_RHE | cd (mA/cm²) | pc (mA/cm²) |
|---:|---:|---:|
| +0.55 | −0.583 | +0.291 |
| +0.50 | −0.618 | +0.309 |
| +0.40 | −0.937 | +0.469 |
| +0.30 | −2.561 | +1.281 |
| +0.20 | −5.468 | +2.734 |
| +0.10 | −5.532 | +2.766 |
| 0.00 | −5.532 | +2.766 |
| −0.10 | −5.532 | +2.766 |
| −0.20 | −5.532 | +2.766 |

cd plateaus at the O₂ Levich limit by V=+0.10 V.  Kinetic
regime is V ≥ +0.30 V; transition near V=+0.20 V.

**λ ramp at V=−0.20 V:**

| λ | cd_observable | Γ |
|---|---|---|
| 0.25 | 12.572849749522968 | 0.07639157771401214 |
| 0.50 | 12.572851628193117 | 0.15277664154520300 |
| 0.75 | 12.572853502603921 | 0.22915519131001685 |
| 1.00 | 12.572855372992420 | 0.30552722681905870 |

Δcd from λ=0.25 → λ=1.00: 5.62e-6 (relative 4.5e-7).  Γ scales
linearly with λ (0.076 → 0.306 = 4×; expected for the closed-form
Γ_ss(λ) at fixed σ_S).  cd is invariant **because cd is
O₂-limited at this voltage**, not because hydrolysis is broken.

## Phase B — k_hyd ramp at V=−0.20 V (this run)

**Setup:** k_hyd ladder (1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3)
at V=−0.20 V from cached Phase A snapshot; baseline k_prot=1e-3,
k_des=1.0.

| k_hyd_nondim | converged | cd (mA/cm²) | pc (mA/cm²) | Γ_MOH | error |
|---|---|---|---|---|---|
| 1e-3 | True  | −5.531 | 2.766 | 0.306 | — |
| 1e-2 | True  | −5.531 | 2.766 | 3.051 | — |
| 1e-1 | False | —      | —     | —     | LadderExhausted at λ=0.25 |
| 1e+0 | False | —      | —     | —     | LadderExhausted at λ=0.25 |
| 1e+1 | False | —      | —     | —     | LadderExhausted at λ=0.25 |
| 1e+2 | False | —      | —     | —     | LadderExhausted at λ=0.25 |
| 1e+3 | False | —      | —     | —     | LadderExhausted at λ=0.25 |

cd is invariant across converged rungs (O₂ Levich pinning;
not hydrolysis-related).  Γ scales linearly with k_hyd (0.306 →
3.051 = 10×).  Picard breaks at k_hyd ≥ 1e-1 because the
σ_S-driven ΔpKa exponent makes Γ_ss explode in the first
positive λ rung.

## Mechanism (corrected)

1. **cd is bottlenecked by O₂ supply at V ≤ +0.10 V**, not
   H⁺ supply.  4e O₂ Levich ≈ 5.50 mA/cm²; observed 5.53 within
   0.6 %.  Adding H⁺ at the OHP via cation hydrolysis cannot
   move what's not the rate-limiting flux.
2. **Γ_MOH grows correctly** in the converged regime (linear
   in k_hyd; closed-form Picard formula reproduces analytic
   Γ_ss(λ) value).  The architecture's expression of the
   Singh hydrolysis branch is structurally correct in the
   k_hyd ≤ 1e-2 regime where Picard converges.
3. **Picard breakdown at k_hyd ≥ 1e-1 is physically meaningful:**
   Γ has no Langmuir capacity in the v9 architecture, so high
   k_hyd produces Γ values corresponding to many monolayers
   of MOH (Γ ≈ 3 nondim ≈ 64 monolayers physical).  This is
   simultaneously a numerical-stiffness failure AND a
   physically invalid regime.

## Architectural debt (must address before production)

1. **Γ has no Langmuir capacity.**  Forward rate
   `k_hyd · c_M · 10^(−ΔpKa)` is not multiplied by `(1 − θ_MOH)`,
   so Γ can grow unboundedly and the converged k_hyd=1e-2 case
   already corresponds to ~64 monolayers of MOH (physically
   invalid).  Fix: add `(1 − Γ/Γ_max)` factor; calibrate
   Γ_max and k_des from literature (note: max R_net at λ=1 =
   k_des·Γ_max → larger k_des or larger Γ_max gives larger
   max R_net; smoke baseline k_des=1.0 with Γ_max=0.047 caps
   R_net at 0.047 nondim ≈ 0.01 mA/cm² equivalent).
2. **`LadderExhausted` loses partial-rung diagnostics.**  Fix:
   attach `partial_rungs` attribute to the exception or change
   the orchestrator to return `converged=False` results.
3. **AdaptiveLadder structurally lacks a sub-rung-1 floor.**
   With first positive rung 0.25 (or any positive value), the
   ladder cannot insert below it on failure.  Fix: patch
   `record_failure_and_insert` to use λ=0 as a valid floor for
   linear-midpoint inserts.
4. **`apply_h_source`, `apply_k_sink`, `override_pka_sigma_S`
   ctx flags missing.**  These are needed for ablation matrix
   experiments (see below).  Defaults: `apply_h_source=True`,
   `apply_k_sink=True`, `override_pka_sigma_S=None` —
   reproduces v9 byte-for-byte.

## Σ_S scale mismatch (load-bearing scoping finding)

Singh 2016 SI's Cu calibration uses C_dl ≈ 51 µF/cm² ×
Δφ_cell ≈ 4.4 V → σ_S ≈ 226 µC/cm².  Our model: C_S = 0.10 F/m²
= 10 µF/cm²; reaching σ_S = 226 µC/cm² requires Δφ_Stern ≈
22.6 V — unphysical.  **Phase D cannot directly calibrate
r_H_El against Singh's Cu pKa table at the deck's σ_S.**

Resolutions (must instrument first; the above arithmetic
assumes Singh's "σ" is local Stern surface charge density,
which may not be the same object as the deck-level capacitance
integration):

A. **Match σ_S, raise C_S:** treat C_S as a free fit
   parameter at ~50 µF/cm².  Risk: C_S becomes a primary fit
   parameter, scope creep.
B. **Calibrate within accessible σ_S range:** fit r_H_El to
   K's absolute ΔpKa at the model's reachable σ_S range; then
   evaluate Cs/Na/Li ordering as a holdout.
C. **2D (r_H_El, C_S) calibration:** honest joint fit; turns
   Phase 6β into a 2-parameter calibration.

The artifact recommends **B** as the next step (single-fit on
K's ΔpKa magnitude; rank/spacing metric on Cs/Na/Li holdout)
contingent on the σ_S(V) instrumentation showing an accessible
σ_S < 0 range with non-trivial ΔpKa magnitudes.

## Phase 6β scope reframing (post session 31)

The deck deliverable has two layers:

* **Mechanism subdeliverable (pKa_eff):** per-cation pKa_eff =
  pKa_bulk + ΔpKa(σ_S) ordering consistent with Singh Eq. (4)
  + deck slide 27.  Note slide 27 IS Singh's Cu table reproduced
  (`docs/phase6/singh_2016_pka_formula.md` line 221) — calibrating
  to slide 27 = self-consistency check, NOT an independent
  experimental validation.
* **Acceptance subdeliverable (experimental):** per-cation Ring
  Onset Potential, Max Ring Current, **Highest H₂O₂ Selectivity
  (%)**, Number of e⁻ matching the per-cation Cation Summary
  Table in `Summary Data-Error.xlsx` at 4 cations × 5-6 pH values.
  This is the actual external validation target.

### Σ_S mapping conventions (two only)

Per `docs/phase6/singh_2016_pka_formula.md` §5.2: Singh's σ via
his Eq. (5) is computed from cell-level capacitance × total cell
voltage (couples cathode + anode + Nernst + iR drops).  The
existing extraction note proposed using local Stern σ_S as the
"equivalent" — but the equivalence is an assumption, not a fact.
Two conventions:

1. **Local Stern σ_S** — model's coupled physical path; σ_S is
   solved from PNP/Stern at each V_RHE via
   `σ_S = C_S · (φ_metal − φ_surface)` from the solved fields
   (NOT algebraically from V_RHE; ψ_S ≠ V_RHE).  Default path.
2. **Imposed Singh σ (ablation only)** — pin σ at Singh's Cu
   table value for each cation via the `pka_override_ablation`
   flag (`override_sigma_singh_counts_pm2`).  This is NOT a
   coupled physical run; it bypasses the Stern/Poisson coupling.
   Labelled "ablation" in code + docs.

A new helper `sigma_C_m2_to_counts_pm2(signed)` returns signed
counts/pm² using the conversion factor 6.243e-6 (where
1 m² = 1e24 pm²; signed return).  The pKa layer applies the
anode-clamp `sigma_singh = max(0, −signed_counts)`.

### Phase D — K-only calibration

Predeclared scalar target from K experimental data (NOT Singh's
table).  Acceptance metric: K's predicted RRDE-equivalent H₂O₂%
matches the Cation Summary Table within ±10 percentage points
at the deck's pH ≈ 4 condition.  Fits a single scalar:
**Δ_β_K** (additive shift on the geometric coefficient
β_per_cation = 2·A·z·r_H_El·(1 − r_M-O²/r_H_El²) in units of
**pm² when σ is in counts/pm²**).  ΔpKa = β · σ for whatever σ
mapping is active.  Sign convention: cathodic pKa lowering
requires σ_local > 0 (or imposed Singh σ positive), and β
preserves cathodic-lowering for all cations.

### Phase E — predictive holdout

Two transferability rules on β:
1. **Δ-rule (default):** β_carbon = β_Cu + Δ_β, applied uniformly.
2. **ρ-rule:** β_carbon = ρ_β · β_Cu.  Risk: at Li⁺ Cu prior,
   β_Cu ≈ 0 (gap-singular case), ρ_β has no effect → Δ-rule
   safer.

Both rules are **exploratory**; "validated" requires future
blinded re-evaluation.

Per-cation experimental observables (predeclared extraction):
* **Ring Onset Potential** = first crossing of model ring
  current 0.01 mA/cm² when sweeping V_RHE anodic→cathodic;
  linear interpolation between bracketing grid points.
  Tolerance: ±50 mV absolute.
* **Max Ring Current** = max of model ring current over
  predeclared V_RHE window (overlap of model + experiment).
  Tolerance: ±30% relative with absolute floor 0.01 mA/cm².
* **`max_H2O2_selectivity_in_window`** = max over predeclared
  V_RHE window.  Tolerance: ±10 percentage points absolute.
* **`argmax_V_for_selectivity`** = V at which model selectivity
  is max.  Tolerance: ±50 mV vs deck "Corr Pot".
* **n_e_rrde** = 4·|I_disk|/(|I_disk| + I_ring/N), N = 0.224.
  Tolerance: ±0.5 absolute.
* **ΔpKa magnitude metric:** mean_i |ΔpKa_model − ΔpKa_deck| /
  mean_i |ΔpKa_deck| ≤ 0.30.
* **ΔpKa ordering metric:** full Li < Na < K < Cs in predicted
  ΔpKa magnitude (every pair correctly ordered).

**Primary observables (must pass):** Ring Onset Potential,
Highest H₂O₂ Selectivity, ΔpKa ordering.
**Secondary observables (≥2/3 must pass):** Max Ring Current,
n_e_rrde, ΔpKa magnitude.

Phase E pass: all primaries + ≥2/3 secondaries pass per cation
across all 4 cations.

### Phase E.0 — data reduction protocol (predeclared)

Frozen before fitting:
* **pH binning:** experimental "pH ≈ 4" bin = pH ∈ [3.5, 4.5];
  Cation Summary Table records grouped by bin.
* **Cycle aggregation:** mean of cycles 1/2/3 from
  `0,1M K2SO4 data 8-15-19.xlsx`; report std as error bar.
* **V scan window:** model V_RHE ∈ [−0.4, +1.0] (solver's
  convergence window).  Experimental V window per the Brianna
  LSV (V_RHE ≈ [−0.06, +1.14]); overlap [−0.06, +1.0] used for
  max-extraction.
* **Tolerance per observable:** above.
* **File mapping:**
  * RRDE LSV raw (pH 6.39 only): `0,1M K2SO4 data 8-15-19.xlsx`
  * Cation summary statistics: `Summary Data-Error.xlsx`
  * CP summary: `CP_data.csv`
  * CP raw waveforms: `{K2,Cs2,Na2,Li2}SO4_10-9-20.mat`

## Sequenced re-do plan (post session 31)

1. **Phase 0 — Deliverable contract (1 hour to draft, then
   hold).**  One-page contract to Linsey/Brianna with:
   * v9 findings summary.
   * v10a scope + estimated timeline.
   * Proposed acceptance bundle (per-observable extraction +
     tolerance + primary/secondary designation).
   * Phase E.0 data-reduction protocol (pH binning, cycle
     aggregation, V window, file mapping).
   * Ask: which observables mandatory; whether tolerances are
     acceptable; whether Phase E is pure holdout or per-cation
     refit OK.
   * Hold v10a/A.2/D/E until contract returns OR user
     explicitly authorizes proceeding unconditionally.

2. **v10a — Langmuir cap + integrated diagnostics (~5-7 days).
   ✅ LANDED 2026-05-10.**
   * Code: `(1 − Γ/Γ_max)` factor in `build_forward_branch` and
     `build_proton_boundary_source`
     (`Forward/bv_solver/cation_hydrolysis.py`).
   * Langmuir Picard formula:
     `Γ_ss(λ) = λ·F₀ / ((1−λ) + λ·k_des + λ·B + λ·F₀/Γ_max)`
     where F₀ = k_hyd·⟨c_M·10^(−ΔpKa)⟩, B = k_prot·⟨c_H⟩/δ_OHP.
     Reduces to v9 when Γ_max → ∞ (tested via
     `test_reduces_to_v9_when_gamma_max_large` with Γ_max=1e10 →
     rel-diff < 1e-12).  Helper extracted as standalone
     `gamma_ss_langmuir` for unit-testability.
   * Γ clamp [0, Γ_max] after every Picard update with a
     `RuntimeWarning` if the unclamped value leaves bounds.
     Warm-restart clamp via `clamp_gamma_to_max(ctx)` is silent
     (orchestrator calls it in `solve_lambda_ramp_from_warm_start`
     before Picard runs).
   * Integrated diagnostics via `collect_v10a_rung_diagnostics(ctx)`
     in rung_callback: F₀_avg, Γ, θ = Γ/Γ_max, R_forward_capped,
     denominator decomposition (constant + k_des + k_prot proton
     flux + cap), per-reaction R_2e/R_4e currents, σ_S in both
     C/m² and Singh counts/pm² (via new
     `Forward/bv_solver/units.py::sigma_C_m2_to_counts_pm2`).
   * Smoke values for Γ_max (= 1 monolayer at OHP, `5.6e-6 mol/m²
     / (C_SCALE · L_REF) ≈ 0.047`) and k_des (= smoke baseline
     1.0); these are placeholders for v10b.
   * Tests: 22 fast + 14 slow regression tests in
     `tests/test_phase6b_v10a_langmuir_cap.py` cover:
     - (a) Γ → Γ_max as k_hyd → ∞ ✓
     - (b) Γ_max → ∞ recovers v9 byte-equivalent ✓
     - vacancy factor (1 − θ) ≥ 0 across parameter sweep ✓
     - σ unit helper round-trips Singh Cu values ✓
     - anode-clamp invariant (σ_S > 0 → ΔpKa = 0) ✓
     - β sign-guard (cathodic σ_S → ΔpKa < 0 for K⁺) ✓
     - bisection cross-check on closed form at λ=1 ✓
     - clamp_gamma_to_max silent / setter round-trips ✓
   * Public surface additions to `Forward/bv_solver/anchor_continuation.py`:
     `set_reaction_gamma_max_model` accessor + the
     `gamma_max_nondim` key in
     `solve_lambda_ramp_from_warm_start`'s `parameter_overrides`
     dispatch.
   * Config builder helper `make_cation_hydrolysis_config(...)` in
     `scripts/_bv_common.py` so drivers do not silently omit
     `gamma_max_nondim`; gate4 driver
     (`scripts/studies/phase6b_v9_gate4_finite_hydrolysis_smoke.py`)
     migrated to the builder.

3. **Minimum V-sweep diagnostic (~1 day).**  Run v10a across
   the V_RHE walk at smoke kinetics + λ=0 baseline AND λ=1 with
   v10a cap.  Record σ_S(V), per-branch currents,
   `dR_net/dσ_S` sensitivity at smoke parameters, c_H(0),
   c_K(0).  Output: σ_S(V) plot, V_kin candidate selection.

4. **V_kin selection (predeclared rule).**
   * Score each V by `dR_net/dσ_S` sensitivity at smoke params
     (k_hyd = 1e-3, k_des = 1.0, Γ_max = 1 monolayer, λ = 1).
   * Filters: cd/Levich < 0.9 (non-plateau); R_2e/(R_2e+R_4e)
     ∈ [0.05, 0.95] (both branches active); σ_S < 0.
   * V_kin = argmax(sensitivity) subject to filters.
   * Fallback: if no V passes all filters, pick argmax among
     V with σ_S < 0 AND cd/Levich < 0.9, even if branch filter
     fails.  Label as "filter-failed fallback".
   * Fail-stop: if no V has σ_S < 0, abort Phase A.2 and route
     to v10c (C_S bracket sweep) instead.
   * **Never hard-code a numeric V.**

5. **Phase A.2 — re-run at V_kin (~1 day).**  λ ramp +
   densified k_hyd ramp at V_kin with v10a cap.  Full
   diagnostics per the integrated rung_callback.

6. **Plumbing ablation matrix (~2 days, BEFORE v10b).**
   Ablations verify source/sink/override plumbing before
   calibrating Γ_max/k_des against experimental data.
   * **A1 — Source-only manufactured:** `apply_h_source=True`,
     `apply_k_sink=False`, manufactured_R_inj=bracketed
     (≥5% surface c_H shift fixture).  Expected: surface c_H
     rises; c_K(0) unchanged.
   * **A2 — Sink-only manufactured:** `apply_h_source=False`,
     `apply_k_sink=True`, manufactured_R_inj=bracketed.
     Expected: c_K(0) falls; surface c_H unchanged.
   * **A3 — Imposed Singh σ ablation:** physical Singh path
     with `override_sigma_singh_counts_pm2`=deck-Cu-cited value
     per cation.  Expected: ΔpKa = Singh table value; R_net
     flows; σ-mapping independent.  Does NOT couple to
     Stern/Poisson φ.  Labelled "pKa_override_ablation".
   * **A4 (deferred):** Sulfate analytic disabled with
     concrete replacement residual to define.
   * **A5 (after v10b):** Physical Singh hydrolysis at large
     k_hyd with v10b-calibrated capacity.

7. **CMK-3 capacitance literature note (v10b prerequisite).**
   File `docs/phase6/CMK3_capacitance_literature.md` with:
   * Per-source normalization convention (F/g, F/BET-m²,
     F/geometric-cm²).
   * Per-source measurement basis (supercapacitor / ORR / CV).
   * Geometric C_S = literature C_S / roughness factor
     (document RF source if used).
   * At least one defensible geometric-area C_S literature
     anchor.
   * If no anchor exists: v10b treats C_S as a fit parameter
     with σ_S(V) bracketing across `{10, 30, 50}` µF/cm²
     exploratory.

8. **v10b — literature calibration (~1-2 weeks).**
   * Recalibrate Γ_max from literature MOH adsorption (or
     a defensible monolayer estimate from a_MOH).
   * Recalibrate k_des from literature surface-desorption
     rate constants (max R_net = k_des·Γ_max; larger k_des
     OR larger Γ_max → larger max source).
   * If C_S literature anchor exists, set C_S from it; else
     run σ_S(V) bracket diagnostic and pick.
   * Byte-equivalence regression: v10b at Γ_max → ∞ matches
     v9 numerically.
   * Phase 6α water-ionization compatibility test.

9. **B.2 — densified k_hyd × λ ramp at V_kin (~2 days).**
   Patched AdaptiveLadder (λ=0-as-floor; 1e-4 first rung +
   linear-midpoint inserts).  Densified k_hyd: {1e-3, 5e-3,
   1e-2, 2e-2, 5e-2, 1e-1}.  Full diagnostics + ablation
   matrix repeated with v10b-calibrated parameters.

10. **Phase D — K-only fit (~1 day).**  Pre-declare K
    calibration target (single scalar Δ_β_K from K
    experimental data); fit; report.

11. **Phase E — predictive holdout (~3-5 days).**  Two
    transferability rules (Δ-rule, ρ-rule); apply
    K-fitted Δ_β to Cs/Na/Li without refit; report
    primary/secondary observable scores per cation, aggregate
    pass/fail matrix.

## Caveats / open assumptions

* **Singh σ mapping convention** documented in
  `docs/phase6/singh_2016_pka_formula.md` §5.2 (local Stern σ);
  treated as one of two candidate mappings — the alternative
  is the imposed-Singh ablation (`pka_override_ablation`).
  The mapping is an **assumption**, not a fact; if Phase D
  Δ_β_K differs by > 30 % between conventions, β is flagged
  non-identifiable without resolving σ.
* **Phase 6β v9 architecture** is now classified as a
  numerical/architectural diagnostic branch only; physics
  conclusions await v10a/b.
* **Slide 27 = Singh Cu reproduction**, not an independent
  experimental dataset.  Phase D/E validate against the
  experimental Cation Summary Table, not against slide 27
  directly.
* **CMK-3 C_S = 10 µF/cm²** is uncited.  v10b's literature
  note (step 7 above) is a prerequisite for any C_S claim.
* **All branch-selectivity claims** await per-reaction current
  assembly (`reaction_index=0` and `=1` in
  `_build_bv_observable_form`); legacy `peroxide_current` is
  deprecated and degenerate for parallel topology.
* **β units:** β = 2·A·z·r_H_El·(1 − r_M-O²/r_H_El²) has units
  of **pm²** (consistent with Singh's σ in counts/pm²); the
  product β·σ is dimensionless ΔpKa.
* **β sign guard:** Phase D's fitted β must preserve cathodic
  pKa lowering (i.e. β·σ_local_cathodic < 0) for all cations
  unless an exploratory rule explicitly allows sign reversal.
  Add as a regression test.
* **Ring current basis:** `Summary Data-Error.xlsx` ring
  values are mA/cm² on ring area; model `I_ring` extraction
  needs to use the same basis (ring-area, not disk-area) for
  the RRDE formulas.  Document in extraction function tests.

## Files

* Critique session 30 (5 rounds, capped at ISSUES_REMAIN;
  cd-invariance finding):
  `docs/handoffs/CHATGPT_HANDOFF_30_phase6b-v9-cd-invariance/`
* Critique session 31 (5 rounds, **VERDICT: APPROVED** on R5;
  strategic pivot):
  `docs/handoffs/CHATGPT_HANDOFF_31_phase6b-v9-strategic-pivot/`
* Driver:
  `scripts/studies/phase6b_v9_gate4_finite_hydrolysis_smoke.py`
  (now supports `--voltage`, `--lambda-only`, `--k-hyd-ramp`,
  `--k-hyd`, `--k-prot`, `--out-subdir` flags).
* Phase A output:
  `StudyResults/phase6b_v9_observability_v_minus_0_20/iv_curve.json`
  + `u_warmstart_at_v_-0.200.npz`.
* Phase B output:
  `StudyResults/phase6b_v9_k_hyd_ramp/iv_curve.json`.
* Phase F output (parallel-safe Tafel):
  `data/derived/k_plus_tafel_slopes_from_brianna_2019.xlsx`
  + `.json`.
* Cation hydrolysis impl:
  `Forward/bv_solver/cation_hydrolysis.py`.
* Singh formula extraction:
  `docs/phase6/singh_2016_pka_formula.md` (the σ mapping note
  is in §5.2; mathematical content was the basis for both
  critique sessions).
* Experimental targets (Phase E validation):
  `data/EChem Reactor Modeling-Seitz-Mangan/Brianna/20201024 CP Experiment Data-Code/Summary Data-Error.xlsx`
  (Cation Summary Table sheet).

## Recommended next action

**Phase 0 — draft the deliverable contract and route it to
the user for the Seitz/Mangan group.**  Hold v10a / A.2 / D / E
until contract returns OR user explicitly authorizes proceeding
unconditionally.  The contract must include the explicit
per-observable extraction functions, tolerances, primary/secondary
designation, and Phase E.0 data-reduction protocol — not just
"pKa ordering and selectivity".  Drafting effort ~1 hour.
