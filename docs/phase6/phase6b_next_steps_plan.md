# Phase 6β next-steps plan (v9, post-handoff-29 GPT-APPROVED staged investigation)

**Status basis:** Written 2026-05-09 22:55 PT (v5); revised
23:30 PT (v6) for conjecture-audit follow-through; major
architectural pivot 2026-05-10 02:30 PT (v7→v8→v9) after a second
5-round GPT critique loop in
`docs/CHATGPT_HANDOFF_29_phase6b-stern-coupling-and-audit-residuals/`.
Phase 6α sweep **complete** (8/8 × 13/13 V_RHE).

**Material change at v9 vs v6:** v5/v6/v7/v8's "boundary-only
algebraic shadow + Stern surface-charge coupling" architecture is
structurally impossible. Reasons (handoff 29 R1#4, R2#1, R2#2,
R3#1):
* At smoke target K⁺ pH 9.5, ~91% of cation is neutralized to MOH⁰
  (not 10% as v6 had assumed). The Boltzmann analytic c_M+
  reduced-model fails decisively.
* Algebraic equilibrium `c_MOH = c_M+·Ka/c_H` zeroes the net rate
  `R_hyd = k_hyd·[c_M+ − c_H·c_MOH/Ka]` for any finite k_hyd, so
  the architecture has no nonzero steady-state proton source —
  hydrolysis is purely transient.
* Boltzmann profile has zero NP flux by construction (drift-
  diffusion balance), so no cation supply rate is recoverable.
* Γ_MOH is **neutral**; it does not enter Stern σ. The v6/v7/v8
  Stern surface-charge correction `+F·δ·c_MOH(0)` was
  conceptually muddled (handoff 29 R4#6 Gauss-balance argument).

**v9 architecture (GPT-APPROVED on round 5):**
1. Promote c_K⁺ to a **dynamic NP species** (4-DOF stack,
   formulation='logc_muh').
2. Γ_MOH as a **global Real-element scalar Function** on the
   electrode marker (mol/m²), coupled through `ds`.
3. **Finite-rate hydrolysis kinetics** at the OHP:
   R_net = k_hyd · c_M+(0) − k_prot · c_H(0) · Γ_MOH/δ_OHP.
4. **Desorption removal path** R_des = k_des · Γ_MOH (open-reservoir;
   not full mass conservation — caveat documented).
5. **No Stern surface-charge correction.** Hydrolysis affects
   electrostatics purely via the dynamic c_K⁺ NP profile responding
   to the boundary sink R_net.
6. **6β.1 staged in 4 falsification-oriented gates** (build / λ=0
   dyn-K equilibrium / Γ source unit test / 1-V finite-rate
   hydrolysis).
7. **Calibration scope reduced** to two grouped parameters at Gate 4
   (`K_s = δ·Ka_eff` and `Da = k_des·δ/D_M`); other tunables held
   fixed at literature priors with sensitivity sweeps.

**Status of v6 unresolved items:**
* R5#1 Stern surface-charge coupling: **REJECTED** (handoff 29 R4#6).
* R5#4 Boltzmann reduced-model: **REJECTED at smoke target** (R1#4);
  c_K+ promoted to dynamic NP.
* All 11 v6 §5 audit items remain incorporated; expanded ledger in
  v9 §10.

Earlier versions (v1–v8) are preserved in git history; this v9 is
the canonical artifact post-handoff-29.

---

## 1. Verified Phase 6α final state

### 1.1 Verdict

* P1 plateau Levich linearity: **PASS** (slope ≈ 1.00).
* P2 no cathodic peak: **PASS** (no interior local minimum).
* P3 max_surface_pH < 9 at L=16 µm: **FAIL** (max pH = 10.58).

### 1.2 Plateau magnitudes (deck target ≈ −0.18 mA/cm²)

| L_eff | ratio 1e-18 | ratio 1e-30 |
|---|---|---|
| 100 µm | −0.737 (4.1×) | −0.440 (2.4×) |
| 66 µm | −1.122 (6.2×) | −0.667 (3.7×) |
| 21 µm | −3.541 (19.7×) | −2.095 (11.6×) |
| 16 µm | −4.651 (25.8×) | −2.749 (15.3×) |

### 1.3 Surface pH at V_RHE = −0.40 V

L_eff-independent at ~10.6 (1e-18) / ~10.3 (1e-30). Transport
doesn't move surface pH; only chemistry can.

### 1.4 Cosmetic logging bug

`_config_dict` line 519 of `l_eff_transport_sweep_csplus_so4.py`
omits `enable_water_ionization`. Per-combo JSON shows `false` while
`summary.json` and the run log correctly show `true`. Fixed in-tree
(handoff 26 §2.1) but the on-disk per-combo files reflect the
pre-fix state. Step 5 below.

### 1.5 Provenance caveats

* **Cation:** Phase 6α used Cs⁺ (`DEFAULT_CSPLUS_BOLTZMANN_COUNTERION_STERIC`).
  The deck's reference data (`Brianna/0,1M K2SO4 data 8-15-19.xlsx`)
  is K⁺. Steric properties near-identical (Stokes 2.2 vs 2.3 Å), so
  Phase 6α numbers are approximately apples-to-apples for
  *transport-only* claims, but they diverge once cation hydrolysis
  is active (pKa 4.32 vs 8.49). See §8 for the conjecture-audit
  finding.
* **Water self-ionization is sub-leading**, not dominant. The
  Seitz/Mangan group's documents don't endorse `H₂O ⇌ H⁺ + OH⁻` as
  the relevant ORR buffer (handoff 26 22:55 CDT update). Phase 6α's
  surface-pH 10.58 may itself over-predict H⁺ supply due to the
  fast-equilibrium assumption; textbook bulk water dissociation
  rate (~1.4 mol/m³/s) is ~20× smaller than Phase 6α's effective
  demand at L=16 µm. See §10 for a queued Phase 6α.1 finite-rate
  refinement.
* **L_eff sweep is a Claude/GPT framing**, not a Ruggiero parameter.
  Deck-comparable runs use L_eff ≈ 16–26 µm (RDE 1600 rpm Levich
  range); larger values are stagnant-film sensitivity, not deck.

---

## 2. Phase 6β chemistry: cation hydrolysis (handoff 26 §9)

Sulfate buffering is **out of scope** (handoff 26 §9 22:30 CDT
correction; see also v1/v2 of this plan in git for the full
sulfate-buffering excursion that was retired). The correct buffer
mechanism is cation hydrolysis at the OHP:

```
M(H₂O)ₙ⁺ ⇌ M(H₂O)ₙ₋₁(OH)⁰ + H⁺            (in OHP)

Ka_M(near cathode) = [M(OH)] · [H⁺] / [M(H₂O)]
```

A water molecule in the cation's hydration shell deprotonates to
release H⁺ at the OHP. The bulk pKa is ~13–14; near a polarized Cu
cathode it drops 5–10 units due to electrostatic stabilization of
the deprotonated form. The functional form for `Ka_M(φ_local)` is
in **Singh et al. 2016 JACS** `10.1021/jacs.6b07612` (must read SI
for exact expression). Co-Zhang 2019 Angewandte `10.1002/anie.201912637`
provides experimental support for cation-dependent local pH in CO₂RR
(via product/ring electrochemistry, **not** an IrOx ring — the IrOx
attribution is in Linsey 2025 deck slides 5–9).

### Predicted cation series (Linsey 2025 deck slide 27, Cu/CO₂RR)

| Cation | Bulk pKa | pKa near cathode | Stokes r |
|---|---|---|---|
| Li⁺ | 13.6 | 13.16 | 3.4 Å |
| Na⁺ | 14.2 | 11.44 | 2.8 Å |
| K⁺ | 14.5 | 8.49 | 2.3 Å |
| Cs⁺ | 14.7 | **4.32** | 2.2 Å |

These values are for Cu/CO₂RR conditions, **not** ORR-on-carbon
with this codebase's Stern. Treat as a *cross-check target* for the
solver-fit values, not a solver input. See §3 step 7 for per-cation
calibration / holdout split.

---

## 3. Plan steps (v9: 4-gate phased smoke)

### Step 1 — Done (Phase 6α complete)

### Step 2 — Postprocessing audit (no action)

`summary.json` / `verdict.json` confirmed coherent (timestamp 21:51,
`enable_water_ionization: true`).

### Step 3 — Cation-hydrolysis algebra spike (RETIRED in v9)

The v5/v6 plausibility-screen spike is retired. Reason: handoff 29
R1#4 found the v5/v6 algebra was off by an order of magnitude
(claimed ~10% K⁺ neutralized at smoke target; correct value is
~91% — Ka/c_H = 10, not 0.1). The algebraic-equilibrium architecture
the spike was probing (R3#1: `R_hyd = k_hyd[c_M+ − c_H·c_MOH/Ka] ≡
0` at equilibrium) is structurally impossible. The spike's "Boltzmann
gives total cation" assumption (R3#3) violates mass conservation
when the c_MOH/c_M+ ratio reaches 10. The successor is Gate 4 of
the staged 6β.1 plan (Step 6), which uses dynamic c_K+ NP +
finite-rate hydrolysis kinetics.

### Step 4 — Read first (priority order, R3#6 corrected)

1. **Singh / Kwon / Lum / Ager / Bell 2016 JACS**
   (`10.1021/jacs.6b07612`) — functional form for field-dependent
   pKa. SI especially.
2. **Co-Zhang 2019 Angewandte** (`10.1002/anie.201912637`) —
   experimental support for cation-dependent local pH in CO₂RR.
   **Note:** uses product/ring electrochemistry, not IrOx ring.
3. **Linsey 2025 ACS-CATL deck** — slides 5–9 (IrOx local-pH probe
   protocol for ORR), slide 27 (pKa-near-cathode table; cross-check
   target, not solver input).
4. **Ruggiero 2022 J.Catal.** —
   `docs/Ruggiero2022_JCatal_manuscript.pdf`. Discusses cation
   hydrolysis in ORR context. Note: paper has **no PNP/Stern
   modeling section** — for that, follow ref 71 to Bohra 2019.
5. **Bohra et al. 2019 EES** — `10.1039/c9ee02485a`,
   "Modeling the electrical double layer to understand the reaction
   environment in a CO₂ electrocatalytic system". Cited as
   Ruggiero 2022 ref 71. The methodologically-closest PNP+BV+Stern
   reference for compact-layer parameter values used in the
   group's literature; read for `C_S` priors and Stern BC
   conventions before defending v6 §5.3.
6. **MacDougall–Gupta 2005** — closest analog (cathodic load with
   carbonate buffer). Useful sanity check on the carbonate-vs-cation
   buffer-mechanism distinction.

### Step 5 — Fix `_config_dict` cosmetic bug

Single-line fix at `scripts/studies/l_eff_transport_sweep_csplus_so4.py:519`:

```python
"config": _config_dict(
    l_eff_m=l_eff_m, ratio=ratio,
    enable_water_ionization=enable_water_ionization,
),
```

Plus add a cross-check in `scripts/studies/score_l_eff_sweep.py`
asserting per-combo `config.enable_water_ionization` matches
`summary.enable_water_ionization`.

### Step 6 — 6β.1 staged solver smoke (4 falsification-oriented gates)

Per handoff 29 R3#12 + R4#12, 6β.1 is staged into 4 gates, each
independently passable, with explicit fallbacks. Gate 4 is
**falsification-oriented** — it is designed to expose architecture
failure modes; a Gate 4 pass is "necessary but not sufficient"
(R5#5 wording guard) — it shows the v9 coupled solver can express
a plausible branch without immediate contradiction; it does NOT
validate hydrolysis physics. Physics validation is 6β.2.

**Architecture:** see Step 7 (rewritten for v9).

**Gate 1 — Build/form test (estimated 1-2 days)**

Audit and refactor. Prereq for everything below.

* Refactor `Forward/bv_solver/forms_logc_muh.py:_resolve_mu_h_index`
  to take an explicit `h_index` argument from species config
  instead of inferring proton from `z=+1`. (R3#4 + R4#8: existing
  code hard-fails when more than one species has z=+1.)
* Same audit for `Forward/bv_solver/water_ionization.py:79`,
  `Forward/bv_solver/picard_ic.py` (Debye/Boltzmann IC paths),
  `Forward/bv_solver/multi_ion.py` (multi-ion charge inference),
  and any other `z=+1` callers in IC, diagnostics, post-processing.
* Update `_bv_common.py` species config to include explicit
  `role: "proton" | "counterion" | "neutral"` fields.
* Tests: existing 549 fast + 10 slow tests pass. New regression:
  4-species K2SO4-with-H stack assembles, H+ uses μ_H, K+ does not.

Gate 1 verdict: ALL prior tests pass + new role-aware test passes.

**Gate 2 — λ=0 dynamic-K equilibrium smoke (estimated 3-5 days)**

Establishes that 4-DOF dynamic K+/SO4 stack converges at cathodic
V_RHE *before* any hydrolysis is added. R4#7 flagged this as a
load-bearing feasibility risk: dynamic K+ at cathodic bias is the
sign-flipped analogue of dynamic ClO4- at anodic bias, which has
a known 5/15-7/15 V_RHE ceiling per CLAUDE.md Hard Rule #5.

**Smoke scope:**
* 4sp dynamic stack: O₂, H₂O₂, H⁺, K⁺ (all NP); SO₄²⁻ analytic Bikerman.
* V_RHE = −0.40 V, C_S = 0.10 F/m², L_eff = 16 µm, K2SO4 baseline.
* `λ_hydrolysis = 0` (no hydrolysis active). Phase 6α water-ionization
  enabled.

**Continuation fallbacks (R4#7 — "expected, not exceptional"):**
* z-ramp: start with z=0.1 K+ (effectively neutral) at λ_z=0, ramp
  z→1 over 5-10 rungs.
* k0 ramp: start with k0_R4e/R2e at low values, ramp.
* C_S ramp: start with C_S = 1.0 (effectively no Stern), ramp to 0.10.
* Warm-start from less-cathodic V_RHE (e.g. -0.1 → -0.4).
* Decimal V_RHE refinement on the grid.

**Verdict criteria (semantic tolerances per R3#6):**
* Newton converges to within standard tolerance.
* c_K+ NP profile matches analytic Boltzmann profile to within
  L²-norm difference < 1% at boundary (the Boltzmann is the
  steady state with no boundary sink).
* H+, O₂, H₂O₂ observables match Phase 6α 3sp+Boltzmann-K stack
  to within ε ≈ 1e-6 in c_H, c_O2, c_H2O2.
* `TestDynamicKplusAnalyticSO4MatchesAnalyticBaseline` passes
  (manufactured-equilibrium consistency proof per R3#9).

Gate 2 verdict: convergence + observables match + new test passes.
Failure means the 4sp dynamic K+ stack has its own convergence
problem to solve before any hydrolysis work — re-plan, possibly
queue another GPT round.

**Gate 3 — Γ-only manufactured-source unit test (estimated 1-2 days)**

Verifies sign and area-normalization for the Γ_MOH boundary scalar
DOF and the proton boundary flux machinery, *before* coupled
hydrolysis kinetics.

**Setup:**
* 3sp + dynamic K+ (Gate 2 architecture) + Γ_MOH as global
  Real-element scalar in mixed function space (R3#7 / R4#9).
* Replace the hydrolysis kinetics term with a fixed prescribed
  source `R_inj` independent of c_M+, c_H, Γ. (No Newton coupling
  back to c_H — just an injected proton flux.)

**Tests:**
* TestProtonBoundarySourceSignConvention: positive `R_inj` increases
  c_H(0). One-cell unit test (R2#5 implementation-form sign check).
* TestGammaResidualAreaInvariance: refining the mesh 2× should not
  change the steady-state Γ value (area-doubling test per R4#10).
* TestGammaDirichletPinAtLambdaZero: at λ=0 with Γ Dirichlet-pinned
  to 0, the steady state matches Gate 2 baseline byte-equivalent
  on the original-DOF subset (R4#11).

Gate 3 verdict: all 3 unit tests pass.

**Gate 4 — Finite-rate hydrolysis at one V_RHE (estimated 5-7 days)**

Couples Γ_MOH dynamics to c_K+(0) and c_H(0) via finite-rate
kinetics. **Falsification-oriented.**

**Smoke scope:**
* Full v9 architecture (Step 7): dynamic K+ + Γ_MOH global scalar
  + finite-rate hydrolysis + desorption + NO Stern σ correction.
* V_RHE = −0.40 V (one voltage), λ_hydrolysis ∈ {0, 0.5, 1}.
* C_S sensitivity: {0.05, 0.10, 0.20} × λ ∈ {0, 1} (6 extra solves).
* k_des sensitivity: {1e3, 1e5, 1e7} 1/s × λ = 1 (3 extra solves).
* Pre-Gate-4 prereq: extract Singh 2016 SI exact pKa-shift formula
  (handoff 29 R4#4 hard requirement). Until extracted, use a
  placeholder `ΔpKa = β_M·sgn(σ_S)·|σ_S|^p` with p ∈ {0.5, 1.0,
  2.0} sensitivity — explicitly unphysical, falsification-only.

**Diagnostics (signed quantities per R2#7 / R4 carry-forward):**
* Signed `ψ_S = φ_m − φ_s` at each (V_RHE, λ).
* Signed `η_4e = ψ_S − E°_4e`, signed `η_2e = ψ_S − E°_2e`.
* Γ_MOH steady-state value, Γ-implied surface inventory.
* `a_M·c_M+(0) + a_MOH·Γ_MOH/δ_OHP` packing fraction.
* Signed `Δψ_S(λ) = ψ_S(λ) − ψ_S(λ=0)` at each V_RHE.
* `α·n_e·Δη/V_T` predicted vs realized.
* Predicted vs realized `Δ ln R_4e` (algebra in §7).

**Verdict criteria (architecture-only per R1#11):**
* Newton converges at all (λ, C_S) combinations.
* Disabled-path λ=0 recovers Gate 2 baseline within semantic
  tolerance (R3#6).
* Packing fraction < 1.0 at every (V, λ, C_S, k_des) — fail otherwise.
* Surface pH at V_RHE = −0.40 V moves *in the right direction*
  (pH ↓ from Gate 2's 10.58-equivalent toward [4, 9]).
* Predicted-vs-realized `Δ ln R_4e` agreement within 30%.
* C_S sensitivity bounded — pH movement direction robust to
  C_S ∈ {0.05, 0.10, 0.20}.
* Branch diagnostic (R4#5): pKa driven by bare σ_S vs corrected σ_S
  yields qualitatively-similar Γ_MOH (no order-of-magnitude jump).
* **Plateau-magnitude target NOT gated** (calibration target
  blocked on Tafel xlsx delivery; v6 §5.6 carry-forward).

**v9 R5#5 wording guard:** A Gate 4 pass does NOT validate
hydrolysis physics; it only shows the v9 coupled solver can express
a plausible branch without immediate contradiction. Cation-series
validation (6β.2) is the actual physics check.

If Gate 4 fails on packing, on predicted-vs-realized disagreement,
or on positive-feedback divergence: re-queue GPT with smoke result
as evidence; the architecture is wrong and needs further iteration.

If all 4 Gates pass: 6β.1.b unblocked (full V_RHE grid, single
cation K⁺); cation-series validation deferred to 6β.2.

### Step 7 — 6β.1 implementation architecture (v9, GPT-APPROVED)

This replaces v5/v6/v7/v8 architectures wholesale. v9 is the result
of handoff 29's 5-round critique (`docs/CHATGPT_HANDOFF_29_phase6b-stern-coupling-and-audit-residuals/`).

**Key architectural commitments:**

* **c_K⁺ is now a dynamic NP species** (R2#2 / R3#11). The 4-DOF
  stack (O₂, H₂O₂, H⁺, K⁺) replaces the v6 3sp + analytic-K
  configuration. Bikerman steric closure on K+ is imposed via the
  dynamic-species Bikerman residual; the codebase's prior 4sp work
  (CLAUDE.md Hard Rule #5: 4sp dynamic Bikerman tested at anodic
  with ClO₄⁻) provides infrastructure but the cathodic K+ regime
  is unexplored — Gate 2 of Step 6 verifies feasibility.

* **Γ_MOH is a global Real-element scalar Function** on the BV
  electrode marker (R3#11 / R4#7 / R4#9). Units mol/m². For 1D
  RDE/RRDE the electrode is spatially uniform so global-scalar Γ
  is appropriate; defer facet-supported Γ to a future phase.
  Γ is a Newton unknown in the mixed function space, NOT a
  Constant updated outside Newton (R4#9: would Picard-lag).

* **Finite-rate hydrolysis kinetics at the OHP** (R3#1 net-rate form):

  ```
  R_net = k_hyd · c_M+(0) − k_prot · c_H(0) · Γ_MOH / δ_OHP

  k_hyd  units: m/s
  k_prot units: m³/(mol·s)            (R4#3: NOT m⁴/(mol·s))
  R_net  units: mol/m²/s
  ```

* **Desorption removal path** (R4#1 — required for nonzero
  steady-state turnover):

  ```
  R_des = k_des · Γ_MOH               [k_des units: 1/s — R4#1]
  ```

  Bulk MOH⁰ leaves the modeled system as a neutral species into
  an open reservoir. **Caveat:** this is NOT full mass conservation;
  full conservation would require tracking c_MOH(y) as a 5th NP
  species (deferred to 6β.2 if evidence warrants per ledger L4).

* **Boundary residuals:**

  ```
  c_M+ NP boundary BC:    D_M·∇c_M+·n |_{y=0} = R_net   (cation sink)
  
  Γ_MOH inventory:        ∂_t Γ_MOH = R_net − R_des
                          steady state: R_net = R_des = k_des · Γ_MOH
  
  Proton condition source:
                          F_res -= R_net_hat · v_H · ds(electrode_marker)
                          (R2#5: implementation-form sign;
                           positive R_net increases c_H — verified by
                           Gate 3 unit test)
  ```

* **NO Stern surface-charge coupling.** The v6/v7/v8 proposal
  `+F·δ·c_MOH(0)` (or `+F·Γ`) is REMOVED in v9 (R4#6 + R5#2
  Gauss-balance argument):

  ```
  Surface charge balance at electrode:
    σ_metal     (metal-side, set by Stern BC C_S·ψ_S)
    + Γ_MOH     (NEUTRAL — does NOT add to charge)
    + ∫_OHP_layer ρ_diffuse dy   (diffuse charge inside δ_OHP)
                                = 0  (Gauss)

  Dynamic c_K+ NP already accounts for OHP cation depletion via the
  Poisson source ρ(y). The neutral Γ_MOH contributes 0 to Stern σ.
  Hydrolysis affects electrostatics PURELY via the dynamic-cation
  Poisson coupling responding to the boundary R_net sink.
  ```

* **Field-dependent pKa formula — TBD on Singh SI extraction**
  (R4#4 + R5#4 ledger L1). Until Singh 2016 SI is read, Gate 4
  uses a placeholder `ΔpKa = β_M · sgn(σ_S) · |σ_S|^p` with
  p ∈ {0.5, 1.0, 2.0} sensitivity, β_M cation-specific. **Explicitly
  unphysical-by-fiat** — Gate 4 is falsification-only on the pKa
  formula side. Real physical interpretation requires Singh SI
  extraction before any production run.

  Likely SI form (NOT to be implemented from this guess; see L1):

  ```
  log10(Ka_M_eff/Ka_M_bulk) = (e · σ_S · r_M)
                              / (ln(10) · k_B · T · ε_OHP · ε_0)
                              − β_M

  with σ_S = stern_coeff · ψ_S  (signed: cathodic σ_S < 0)
       ε_OHP ≈ 8 (Bohra-aligned compact-layer dielectric)
  ```

  Driver: `σ_S` is the **bare** Stern surface charge; explicitly
  does NOT include any Γ correction (R4#5 positive-feedback risk).

* **Activation `λ_hydrolysis`** ramps `log(Ka_M_eff)`. At λ=0,
  Γ_MOH is Dirichlet-pinned to 0 (R4#11 hard-zero pin); R_net = 0;
  no hydrolysis effect.

* **Continuation ladder:**

  ```
  k0_scale ladder (existing)
    → kw_eff_ladder (Phase 6α, existing)
      → z_K_ramp / k0_ramp / C_S_ramp (Gate 2 fallbacks if needed)
        → λ_hydrolysis ladder (Gate 4): 0 → 0.5 → 1.0
  ```

* **Slow regression test** `TestHydrolysisActivationZeroReproducesPhase6aSemantics`
  asserts λ=0 recovers Gate 2 baseline within semantic tolerance
  (R3#6: byte-equivalence is impossible because DOF layout
  changes).

**Calibration scope reduced** (R3#10 / R4#3): at Gate 4, fit only
two grouped parameters:
* `K_s = δ_OHP · Ka_eff_volume` (equilibrium-strength composite)
* `Da = k_des · δ_OHP / D_M` (Damköhler-style turnover/supply ratio)

Other tunables held fixed at literature priors:
* `δ_OHP = 0.40 nm` (Bohra 2019 prior; R3#3)
* `r_K = 2.3 Å` (Linsey 2025 deck slide 13)
* `C_S = 0.10 F/m²` (existing production tunable; sensitivity-swept
  at {0.05, 0.10, 0.20} per Gate 4)
* `k_des = 10⁵ /s` (phenomenological prior; sensitivity-swept at
  {1e3, 1e5, 1e7} per Gate 4)
* `β_M_K = 1.0` (Singh placeholder; sensitivity at {0.5, 1.0, 2.0})

**Per-cation config schema (R3#14, expanded for dynamic NP):**

```python
DEFAULT_KPLUS_DYNAMIC_NP = {
    "label": "K+",
    "z": +1,
    "role": "counterion",         # NEW (R4#8) — explicit role
    "stokes_radius_m": 2.3e-10,
    "a_nondim": ...,
    "phi_clamp": ...,
    "c_bulk_nondim": 199.9,
    "D_M_m2_per_s": 1.96e-9,
    "pKa_bulk": 14.5,
    "pKa_shift_form": "singh_2016_TBD",   # placeholder until SI extracted
    "pKa_shift_params": {"beta_M": 1.0, "p": 1.0, "epsilon_OHP": 8.0},
    "stokes_diameter_m": 4.6e-10,         # for δ_OHP packing diagnostic
    "is_dynamic_NP": True,                 # NEW — flag for solver
}
```

Add explicit role fields ("proton" / "counterion" / "neutral") to
all species entries (R4#8); the inferred-from-z=+1 path is
deprecated.

**Algebra: predicted Δ ln R_4e for Gate 4 verdict (R2#8 carry-forward):**

```
ln R_4e(λ) − ln R_4e(λ=0) = 4 · Δ ln c_H − α · 4 · Δη/V_T

For K⁺ at V_RHE = −0.40 V smoke target:
  Δ pH ≈ −1 to −2 → Δ ln c_H ≈ +2.3 to +4.6
  Δ η must overcompensate via dynamic-K+ Poisson shift:
  Δη/V_T ≈ Δ ln c_H · n_p / (α · n_e) ≈ 1.15 to 2.3
  → Δη ≈ 0.03 to 0.06 V at V_T = 0.0257 V

Predicted vs realized agreement within 30% required.
```

Note this Δη is much smaller than the v6/v7 R2#8 erroneous estimate
of 0.29 V (which assumed Stern σ correction ≠ 0; v9 has no Stern
correction so the relevant shift is purely diffuse-layer Poisson).

### Step 8 — Cation-series validation (calibration / holdout split)

After 6β.1 smoke passes, run validation sweep:

* **Calibration cation:** K⁺. β_K tuned against:
  * `0,1M K2SO4 data 8-15-19.xlsx` LSV (cd vs V_RHE at pH 4)
  * `K2SO4_10-9-20.mat` chronopotentiometry
  * IrOx local-pH measurements per Linsey 2025 deck (where available)
* **Holdout cations:** Cs⁺, Na⁺, Li⁺. β_Cs/Na/Li from Singh 2016
  literature, **not** tuned to deck data. Compare predicted surface
  pH and cd shape against `{Cs,Na,Li}2SO4_10-9-20.mat` CP datasets
  and Summary Data-Error.xlsx error bars.

**Honesty caveat (R5#8):** β values from Singh 2016 may not transfer
from their CO₂RR-on-Cu electrode/electrolyte conditions to ORR-on-
carbon. Treat the holdout as a **predictive screen, not decisive
falsification**. If the holdout fails, first audit Stern/OHP field
mapping before rejecting the cation-hydrolysis hypothesis.

**CP data handling (R3#13):**

* Convert all CP voltages from Ag/AgCl to RHE:
  `V_RHE = V_AgCl + 0.197 + 0.0592·pH`.
* QC-reject outliers, use replicate averages with error bars from
  `Summary Data-Error.xlsx`.
* Compare cd vs V_RHE *trends* by pH/current regime, not as a single
  monotone ordering.

### Step 9 — 6δ kept active

Cation hydrolysis pinning local pH at the cation's pKa amplifies
acid-form cathodic rate; transport gives plateau, not decay (R3#12
+ R4 reaffirmed). **Cation hydrolysis fixes the pH source, not the
decay mechanism.** 6δ remains likely-required:

* If 6β.1 sweep produces only a plateau → 6δ.1 (parallel
  alkaline-form ORR reactions) for shape.
* If it produces peak + plateau but no decay → 6δ.1 still required.
* If it produces peak + decay (unlikely, possible if O₂ transport
  saturation interacts) → 6δ delayable but not eliminated.

6δ split:

* **6δ.1** — explicit `R2e_alk` / `R4e_alk` parallel reactions via
  existing reaction-list machinery. Same E°_RHE as acid form
  (RHE invariant). Different k0/α/concentration powers. Smaller
  change. Source-term derivations for the proton condition with
  alkaline-form OH⁻ production needed; HO₂⁻ algebraic closure
  decision tree.
* **6δ.2** — pH-gated switching, site coverage, adsorbed-intermediate
  kinetics. Reserve until 6δ.1 outcome decides.

Order: 6β.1 → measure → 6δ.1 → measure → 6δ.2 if needed.

### Step 10 — Phase 6α.1 finite-rate Kw (queued, R3#11 corrected)

Queued for after 6β.1 lands. Compares Phase 6α's fast-equilibrium
output to IrOx local-pH measurements; if the model still over-predicts
H⁺ supply, replace with a finite-rate residual:

```
R_water(y) = k_r · [Kw_eff − c_H(y) · c_OH(y)]
```

where `k_r` is the textbook backward water dissociation rate
(~1.4·10⁻³ M⁻¹s⁻¹ × unit factor). Goes to 0 at equilibrium; recovers
Phase 6α at `k_r → ∞` byte-equivalently. (Replaces v4's incorrect
`c_H_neutral_water` formulation per R4#11.)

---

## 4. Hard solver invariants (unchanged from CLAUDE.md)

* C+D continuation only.
* `exponent_clip = 100` only.
* Physical `E_eq` (R2e = 0.695 V, R4e = 1.23 V).
* `set_initial_conditions(blob=True)` ignored in log-c mode.
* Run from `PNPInverse/` with `venv-firedrake`.
* Cache env: `MPLCONFIGDIR=/tmp` etc.

---

## 5. Conjecture-audit findings (`docs/CONJECTURE_AUDIT_2026-05-09.md`)

### 5.1 HIGH — Cs⁺ vs K⁺ apples-to-apples (incorporated)

Phase 6α used Cs⁺. Deck reference (`0,1M K2SO4 data 8-15-19.xlsx`)
is K⁺. Steric near-identical (2.2 vs 2.3 Å) so transport-only claims
are approximately apples-to-apples, but cation hydrolysis (pKa 4.32
vs 8.49) makes them diverge sharply once 6β.1 lands. Step 6 smoke
runs K⁺ as the **primary** apples-to-apples test; Cs⁺ is high-effect
sensitivity (R5#9).

### 5.2 MED — defer K0_R4e and α_R4e calibration (incorporated)

`K0_R4E_RATIOS = (1e-18, 1e-30)` was a qualitative fit to Mangan-like
peroxide selectivity. `ALPHA_R4E = 0.5` is labelled placeholder.
Both calibrate cathodic 4e kinetics, which only become meaningful
once local pH is in the deck-realistic 4–7 window. Don't retune in
6β.1; carry current values forward. Calibration source = the
missing Tafel xlsx (§5.6).

### 5.3 LOW-MED — Stern capacitance citation chain (audited 2026-05-09 v6)

**Audit result:** `stern_capacitance_f_m2 = 0.10` has **no Ruggiero
2022 or Linsey 2025 deck citation.** Verified by full-text grep of
`docs/Ruggiero2022_JCatal_manuscript.pdf`: the paper has no Stern,
capacitance, F/m², or μF/cm² value beyond methods-section
"capacitive current subtraction"; it has no PNP modeling section
and cites Bohra 2019 (ref 71, EES 12 11) for the modified
Poisson–Boltzmann modeling approach.

**Actual provenance of 0.10 F/m²:**

* `docs/stern_layer_physics_and_next_steps.md` (2026-05-03) line
  214 lists C_S = [0.05, 0.10, 0.20, 0.40, 1.00] F/m² as the
  sweep design, drawn from the textbook compact-layer scale of
  5–100 µF/cm² (= 0.05–1.0 F/m²); 10 µF/cm² is the low end of
  Bockris/Reddy's typical aqueous range.
* The May 2026 Stern sweep (`docs/4sp_bikerman_ic_option_2b_results.md`)
  selected 0.10 as the smallest finite-Stern value that allowed
  Newton to cross the +1.0 V wall on the 4sp bikerman stack.
* The value has therefore been a **convergence-pinned engineering
  choice**, not a deck-calibrated parameter.

**v6 framing:** Treat 0.10 F/m² as a **labelled tunable** with
literature anchor [0.05, 0.50] F/m² (low–mid end of Bockris/Reddy
range; Bohra 2019 should be read for the value used in the
closest CO₂RR PNP+BV literature). Do not retune in 6β.1;
sensitivity-sweep `C_S ∈ {0.05, 0.10, 0.20}` only after the
hydrolysis architecture is validated. The 6β.1 smoke (Step 6)
should record the Stern drop diagnostic to bound how much c_H
movement comes from Stern vs hydrolysis.

### 5.4 LOW — L_eff sweep is Claude/GPT framing (incorporated)

Ruggiero has no L_eff parameter; deck uses RDE 1600 rpm Levich
length ~16 µm (CESR Seed Proposal + Trienens 2025 Report ground
the *concept*, not specific values). Deck-comparable runs use
L_eff = 16 µm only (or 16–26 µm bracket per R5#9); larger values
are stagnant-film sensitivity, not deck-comparable.

### 5.5 LOW — SO₄²⁻ Bikerman radius = 2.4 Å (provenance documented)

`scripts/_bv_common.py:594` cites "Marcus" (Marcus 1988
hydrated-ion radii — the standard textbook reference) as the
source for SO₄²⁻ at 2.4 Å. The conjecture audit's "STILL
UNVERIFIED" flag is overstated: the citation is a textbook
reference, not a wild guess. **v6 framing:** carry the Marcus
value forward; flag for cross-check against Linsey 2025 deck
slide 13 (which lists *cation* hydrated radii only — anion radii
are not in the deck), and against Bohra 2019 if it gives a
specific SO₄²⁻ Bikerman parameter. Since SO₄²⁻ at z=−2 is the
co-ion (depleted at the cathode), its precise radius matters
much less than the cation radius for the OHP packing. Defer
calibration to a later pass; not a 6β.1 priority.

### 5.6 External — Tafel slope xlsx data request (open, no in-tree action)

The conjecture audit's recommendation 5: the calibration source
for items 5.2 and (less directly) 5.3 — `Tafel slope analysis
cation-pH-Li-K-Cs.xlsx` — is the only piece of the data audit
(per `docs/seitz_mangan_data_folder_audit_2026-05-08.md`) still
**missing from the data folder**. Action: ask Linsey/Brianna for
the file directly (re-request after the original 2026-05-08 ask;
not yet delivered as of v6 timestamp). Until it lands, 6β.1 K⁺
calibration uses `0,1M K2SO4 data 8-15-19.xlsx` LSV +
`K2SO4_10-9-20.mat` chronopotentiometry as the only deck
calibration sources — Tafel-slope-side calibration of K0_R4e /
α_R4e is **blocked on data delivery**, not on solver work. This
matches v6 §5.2's "carry forward, don't retune".

---

## 6. Files / pointers

* Implementation prereq (Gate 1):
  * `Forward/bv_solver/forms_logc_muh.py` — refactor `_resolve_mu_h_index`
    to take explicit `h_index` argument (R3#4 + R4#8).
  * `Forward/bv_solver/water_ionization.py:79` — same z=+1 inference fix.
  * `Forward/bv_solver/picard_ic.py` — Debye/Boltzmann IC paths
    audit for z=+1 callers.
  * `Forward/bv_solver/multi_ion.py` — multi-ion charge inference
    audit.
  * `scripts/_bv_common.py` — add `role: "proton"|"counterion"|"neutral"`
    fields; add `DEFAULT_KPLUS_DYNAMIC_NP` config.
* Implementation (Gate 2-4):
  * New `Forward/bv_solver/cation_hydrolysis.py` (or merge into
    renamed `local_h_sources.py`) — finite-rate kinetics, Γ_MOH
    Real-element scalar, R_net + R_des assembly.
  * `Forward/bv_solver/anchor_continuation.py` — extend continuation
    ladder with Gate 2 fallbacks (z-ramp, k0-ramp, C_S-ramp) and
    Gate 4 λ_hydrolysis ladder.
  * `Forward/bv_solver/forms_logc_muh.py` — extend mixed function
    space to include global Real-element Γ_MOH.
* Tests: `tests/test_water_ionization_phase_6a.py` (existing)
  + new `tests/test_cation_hydrolysis_phase_6b_v9.py`:
  * `TestRoleFieldRoundtrip` (Gate 1)
  * `TestDynamicKplusAnalyticSO4MatchesAnalyticBaseline` (Gate 2)
  * `TestProtonBoundarySourceSignConvention` (Gate 3)
  * `TestGammaResidualAreaInvariance` (Gate 3)
  * `TestGammaDirichletPinAtLambdaZero` (Gate 3)
  * `TestHydrolysisActivationZeroReproducesPhase6aSemantics` (Gate 4)
* Outputs: `StudyResults/fast_realignment_2026-05-08/phase6b_v9_smoke/`
  (new) — Gate 4 outputs with diagnostics: signed ψ_S, η, Δ ln R
  predicted vs realized, packing fraction, branch-diagnostic
  comparison (bare σ_S vs corrected σ_S — though corrected is
  removed in v9, the diagnostic test sanity-checks the Gauss-balance
  argument).
* Reference papers: see Step 4 reading list. Singh 2016 SI is the
  Gate 4 prereq (ledger L1).
* Critique-loop sessions:
  * Handoff 28 (`docs/CHATGPT_HANDOFF_28_phase6b-sulfate-spike-planning/`)
  * Handoff 29 (`docs/CHATGPT_HANDOFF_29_phase6b-stern-coupling-and-audit-residuals/`)

---

## 7. What's gone vs prior versions

* **v1/v2 (sulfate buffering)** — out of scope per handoff 26 §9.
* **v3/v4 (volume R_buf source, θ(y) thin-layer kernel, "two NP
  species" framing, c_T primary variable)** — handoff 28 R3/R4
  killed all of these.
* **v5/v6 (boundary-only algebraic shadow)** — handoff 29 R1/R2/R3
  found this is structurally impossible (R5#4 algebra error: 91%
  not 10% neutralized; R3#1: equilibrium algebra zeroes net rate).
* **v6/v7/v8 Stern surface-charge coupling `+F·δ·c_MOH`** — handoff
  29 R4#6 Gauss-balance argument: Γ_MOH is neutral, doesn't enter
  Stern σ. The dynamic-K+ Poisson coupling already captures the
  electrostatic effect of cation depletion.
* **v6 cation-hydrolysis algebra spike (`scripts/studies/phase6b_cation_hydrolysis_spike.py`)**
  — retired in v9 since the algebra it was probing is structurally
  impossible. Successor is Gate 4 of the staged 6β.1 plan.
* **v6 step 6 deck-magnitude verdict gating** — replaced with
  architecture-only verdict (R1#11 + R5#5 wording guard).
* **Phase 6γ as a separate phase.** Subsumed into 6β.

---

## 8. v9 unresolved-physics ledger

These items are explicit unresolved physics that v9 carries forward
into 6β.1 execution. Each has a status, a decision-needed, and a
closure target. Gate 4 is falsification-oriented for L1/L2/L3/L5.

| ID | Item | Status | Decision needed | Closure target |
|---|---|---|---|---|
| L1 | Singh 2016 SI exact pKa-shift formula | TBD; placeholder `β_M·sgn(σ_S)·|σ_S|^p` with sensitivity sweep | Read Singh 2016 SI; extract exact functional form (sign, ln(10) factor, capacitance/surface-charge convention, distance dependence) | **Gate 4 prereq** + cation series 6β.2 |
| L2 | k_des desorption rate prior | Phenomenological 1/s, no literature anchor (R4#1) | Literature search for MOH⁰ desorption from polarized OHP; otherwise full-range sweep | Gate 4 sensitivity sweep |
| L3 | Γ-Stern double-counting | **REMOVED in v9 per R4#6 Gauss balance**; only electrostatic effect is via dynamic K+ Poisson coupling | Write Gauss-balance derivation in §7 (already inline) | §7 derivation, pre-Gate-4 |
| L4 | Open-reservoir desorption (not full mass conservation) | Documented caveat (R4#2) | Track c_MOH(y) as 5th NP species if any post-6β.1 evidence requires it | 6β.2 or later |
| L5 | Dynamic K+ cathodic convergence (Gate 2 feasibility) | Known risk; continuation fallbacks documented (R4#7) | Gate 2 outcome | Gate 2 |
| L6 | Cation-series transferability (β_M, p, K_s, Da) | Deferred to 6β.2 | 6β.2 holdout pass/fail | 6β.2 |
| L7 | Stern capacitance C_S = 0.10 F/m² | Labelled tunable [0.05, 0.50] F/m²; sensitivity only (v6 §5.3) | Deferred to 6β.2 calibration | 6β.2 |
| L8 | SO₄²⁻ Bikerman radius 2.4 Å (Marcus textbook prior) | Documented carry-forward (v6 §5.5; R1#10 wording adjusted to "named provisional Marcus value; exact source unchecked") | Cross-check Linsey deck slide 13 anion radii if/when delivered | Post-6β.1 (low priority) |
| L9 | Tafel slope xlsx (external data delivery) | Blocked on data delivery (v6 §5.6) | Re-request from Linsey/Brianna | 6β.2 K0_R4e + α_R4e calibration |

**v9 R5#5 wording guard (durable):** "Gate 4 pass does not validate
hydrolysis physics; it only shows the v9 coupled solver can express
a plausible branch without immediate contradiction." Cation-series
validation (6β.2) is the actual physics check.

---

## 9. Critique-loop sessions

* Handoff 28 (`docs/CHATGPT_HANDOFF_28_phase6b-sulfate-spike-planning/`)
  — 5 rounds, cap hit, retired sulfate buffering, settled v5
  boundary-only-shadow architecture (later rejected in handoff 29).
* Handoff 29 (`docs/CHATGPT_HANDOFF_29_phase6b-stern-coupling-and-audit-residuals/`)
  — 5 rounds, **APPROVED on round 5**, killed v5/v6/v7/v8 algebraic
  shadow + Stern σ coupling, established v9 dynamic-K+ + Γ_MOH
  global-scalar architecture, produced the L1-L9 unresolved-physics
  ledger above.
