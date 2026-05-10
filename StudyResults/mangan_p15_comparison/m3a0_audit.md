# M3a.0 Observable Audit — Mangan Deck p.15 vs Run C Reassembly

Date: 2026-05-07

## What this is

Per `docs/mangan_alignment_status_2026-05-07.md`§"Next concrete step" and CHATGPT_HANDOFF_18§"M3a.0".  No solver re-run.  Pure algebraic post-processing of `StudyResults/mangan_p15_comparison/run_C/iv_curve.json`.

Reassembled three observables from Run C and reapplied the B7 tolerance bands from `docs/m0_target_extraction.md` against each:

- `cd_mA_cm2` — total disk current = `-I_SCALE · (R_0 + R_1)` (already in JSON)
- `pc_mA_cm2` — net peroxide = `-I_SCALE · (R_0 - R_1)` (already in JSON; compared in Run D)
- `gross_R0_mA_cm2` — gross 2e production = `-I_SCALE · R_0 = (cd + pc) / 2` (NEW)

Algebraic identity for the sequential 2-step model with both reactions 2e:

```
R_0 + R_1 = -cd / I_SCALE       (sum of dimensionless rates)
R_0 - R_1 = -pc / I_SCALE       (difference)
R_0       = (-cd - pc) / (2·I_SCALE)
gross_R0_mA_cm2 = -I_SCALE · R_0 = (cd + pc) / 2
```

`I_SCALE = 0.1833 mA/cm²` (from `_bv_common.compute_i_scale()`: n_e=2, F=96485.33, D_O2=1.9e-9 m²/s, C_O2=0.5 mol/m³, L_REF=100 µm).

## Why this matters (post-Ruggiero reframing)

Ruggiero 2022 J. Catal. (Mangan co-author) shows the deck/paper uses **parallel 2e/4e ORR**, not sequential R_0 producing peroxide which R_1 then reduces.  The deck's "Peroxide Current Density" therefore maps to the **gross 2e production current** (single-rate observable), not the net `R_0 - R_1`.  This audit tests how much of the 0/5 Run D verdict was an observable-definition failure (cheap to fix) vs a physics gap requiring the parallel R_2e/R_4e rewrite (expensive).

## Run C metadata (reused)

```
catalyst               = CMK-3
geometry               = RRDE
pH_bulk                = 4.0
cation                 = Cs+
anion_model            = ClO4_protonic_surrogate
rotation_rate_rpm      = 1600.0
L_eff_m                = None
N_collection           = 0.224
electrolyte_model      = pH_countercharge_surrogate
comparison_status      = deck_proxy
source_authority       = Mangan2025_deck
target_curve           = Mangan2025_deck_p15_H2O2_current_density_pH4_Csplus
acceptance_tier        = semi_quant
```

## Tolerance bands — three channels

### net pc (R_0 - R_1) — original observable

| # | Feature | Experimental | Model | Error | Tolerance | Pass |
|---|---------|--------------|-------|-------|-----------|------|
| 1 | peak voltage | +0.1 | +0.55 | 450.0 mV | 50 mV | FAIL |
| 2 | peak magnitude | -0.4 | +1.543e-05 | 100.0% | 25% (rel) | FAIL |
| 3 | left plateau magnitude at V=-0.32 V | -0.17 | +1.551e-05 | 100.0% | 25% (rel) | FAIL |
| 4 | onset to zero | +0.45 | +nan | nan | 50 mV | FAIL |
| 5 | shoulder in V_RHE in (0.18, 0.3) | yes (visible on page 15) | no | no inflection / slope flattening detected | qualitative | FAIL |

**Quantitative bands passing**: 0 / 4   **Total bands passing (incl. shoulder)**: 0 / 5

### gross R_0 (single-rate 2e) — NEW post-Ruggiero candidate

| # | Feature | Experimental | Model | Error | Tolerance | Pass |
|---|---------|--------------|-------|-------|-----------|------|
| 1 | peak voltage | +0.1 | -0.4 | 500.0 mV | 50 mV | FAIL |
| 2 | peak magnitude | -0.4 | -0.09194 | 77.0% | 25% (rel) | FAIL |
| 3 | left plateau magnitude at V=-0.32 V | -0.17 | -0.09148 | 46.2% | 25% (rel) | FAIL |
| 4 | onset to zero | +0.45 | +nan | nan | 50 mV | FAIL |
| 5 | shoulder in V_RHE in (0.18, 0.3) | yes (visible on page 15) | no | no inflection / slope flattening detected | qualitative | FAIL |

**Quantitative bands passing**: 0 / 4   **Total bands passing (incl. shoulder)**: 0 / 5

### total cd (R_0 + R_1) — reference

| # | Feature | Experimental | Model | Error | Tolerance | Pass |
|---|---------|--------------|-------|-------|-----------|------|
| 1 | peak voltage | +0.1 | -0.4 | 500.0 mV | 50 mV | FAIL |
| 2 | peak magnitude | -0.4 | -0.1839 | 54.0% | 25% (rel) | FAIL |
| 3 | left plateau magnitude at V=-0.32 V | -0.17 | -0.183 | 7.6% | 25% (rel) | PASS |
| 4 | onset to zero | +0.45 | +nan | nan | 50 mV | FAIL |
| 5 | shoulder in V_RHE in (0.18, 0.3) | yes (visible on page 15) | no | no inflection / slope flattening detected | qualitative | FAIL |

**Quantitative bands passing**: 1 / 4   **Total bands passing (incl. shoulder)**: 1 / 5

## Sample dimensionless and dimensional rate values

| V_RHE (V) | cd (mA/cm²) | pc (mA/cm²) | R_0 (dimless) | R_1 (dimless) | gross R_0 (mA/cm²) |
|-----------|-------------|-------------|----------------|----------------|---------------------|
| -0.400 | -0.1839 | +1.551e-05 | +0.5015 | +0.5016 | -0.09194 |
| -0.320 | -0.183 | +1.551e-05 | +0.499 | +0.4991 | -0.09148 |
| -0.250 | -0.1821 | +1.551e-05 | +0.4965 | +0.4966 | -0.09102 |
| -0.100 | -0.1791 | +1.551e-05 | +0.4886 | +0.4886 | -0.08956 |
| +0.000 | -0.1752 | +1.55e-05 | +0.4778 | +0.4778 | -0.08758 |
| +0.100 | -0.1681 | +1.55e-05 | +0.4585 | +0.4586 | -0.08406 |
| +0.180 | -0.16 | +1.55e-05 | +0.4363 | +0.4364 | -0.07999 |
| +0.250 | -0.1457 | +1.549e-05 | +0.3972 | +0.3973 | -0.07282 |
| +0.300 | -0.1164 | +1.548e-05 | +0.3175 | +0.3176 | -0.05821 |
| +0.400 | -0.001839 | +1.544e-05 | +0.004974 | +0.005058 | -0.0009119 |
| +0.450 | -2.726e-05 | +1.544e-05 | +3.226e-05 | +0.0001165 | -5.914e-06 |

## Layered diagnostic decomposition

The mechanical band test scores 0/5 for both `pc_net` and `gross R_0`, but the structural reading under the parallel-2e/4e physics from Ruggiero is more informative than that summary line:

**(1) Observable switch (net pc -> gross R_0) is necessary.** The original `peroxide_current = R_0 - R_1` collapses to ~0 across the entire window because R_0 ~ R_1 ~ 0.5 in the saturation regime. Switching to single-rate gross R_0 (`mode="reaction", reaction_index=0`) makes the observable cathodic-negative everywhere with a finite saturated plateau -- qualitatively the right channel to compare against the deck's gross 2e production current.

**(2) Gross R_0's magnitude shortfall factors into the R_0/R_1 lock-in itself.** At the experimental left-plateau voltage V = -0.32 V, gross R_0 = -0.09148 mA/cm² vs experimental -0.17 mA/cm² -- short by a factor of 1.86x. The reason is mechanical: in saturation R_0 + R_1 ~ 1, so R_0 ~ 0.5. If R_1 were replaced by a parallel R_4e channel without a free-H₂O₂ surface intermediate (the Ruggiero §1 structural fix), or kinetically suppressed for CMK-3 2e selectivity, R_0 would absorb the full O₂ flux at saturation.

  - Saturation envelope (gross R_0 unlocked from R_1): -0.1839 mA/cm² ≡ -I_SCALE at full 2e Levich. This is total cd in the existing data, which numerically *already* matches the experimental left plateau within ~8% (well inside the ±25% band): -0.183 model vs -0.17 exp.

  - Read: total cd's left-plateau "PASS" is **not** a match in the current sequential physics -- it's the saturation envelope that **gross R_0 would reach** if the parallel-reaction structural fix landed. The 2x factor is mechanical, not physical.

**(3) Peak structure is real missing physics, not an observable bug.** Gross R_0 monotonically decays from saturation toward zero as V rises through E_eq_R1 = 0.68 V. The experimental peak at V_RHE ≈ +0.10 V is non-monotonic in V at fixed bulk pH -- the local-pH-driven mechanism shift in Ruggiero §3.1 (bulk pH 4 has the largest local pH excursion; cation identity modulates buffering). M3b (multi-ion electrolyte: Cs⁺/SO₄²⁻ + OH⁻ tracking) and M3c (local-pH validation against Ruggiero Fig 1B) work.

**(4) Mass-transport headroom check.** Experimental peak |j| = 0.40 mA/cm² vs our 2e Levich ceiling I_SCALE = 0.1833 mA/cm² → ratio 2.18x.
  - Experimental peak is **above** our 2e Levich ceiling. Even after the structural fix in (2), gross R_0 cannot reach −0.40 mA/cm² without L_eff alignment (M5: ~21 µm Levich δ at 1600 rpm vs current 100 µm) or local-pH amplification of the effective surface rate. M5 retune is **larger than the ~10% previously assumed** -- the ceiling itself needs to roughly double.

## Decision (per status doc "Next concrete step")

`m3a0_verdict = "magnitude_off_by_lock_in_shape_off_by_local_pH"`

**Layered finding -- closest to status doc's "most likely" branch, with structural refinement.**

Gross R_0 has the right sign and saturation/decay structure but is short by 1.86x at the left-plateau voltage *and* missing the peak feature in the middle of the V range. Both gaps factor into known structural/physics issues identified in the Ruggiero realignment, **not** a forward-solver bug:

- **Magnitude (~2x short on plateau):** the R_0/R_1 lock-in itself   (sequential model artifact). Resolved by replacing R_1 with   parallel R_4e (M3a.2 diagnostic + M3a.3 production) or by   CMK-3 2e-selective kinetic recalibration of k0_R2. After this,   gross R_0's saturation envelope = total cd ~ -0.183 mA/cm²   lands within +/-25% of the experimental left plateau. The   audit's reading: total cd is the projected *post-fix* gross R_0,   not a current match.
- **Peak structure (no peak in middle of V range):** local-pH   dynamics + cation buffering + multi-ion EDL screening. M3b   (multi-ion electrolyte: Cs⁺/SO₄²⁻ + OH⁻ tracking) and M3c   (local-pH validation against Ruggiero Fig 1B).
- **Peak magnitude (-0.40 vs our 2e ceiling -0.183):** experimental   peak is 2.18x our 2e Levich ceiling, so   even after the lock-in fix the peak cannot be reproduced at   L_REF = 100 µm. **M5 (L_eff alignment) is larger than the   ~10% retune previously assumed** -- the mass-transport ceiling   itself needs to roughly double to give gross R_0 enough headroom   to peak past the saturation plateau.

Implication for sequencing:
1. **M3a.0 (this audit): DONE.** Confirms the gap is structural +    missing physics, not a forward-solver bug.
2. **M3a.1 (electron-weighted current accounting):** proceed --    required before any R_4e is added to the residual.
3. **M3a.2 (diagnostic parallel R_2e/R_4e residual):** the clean    test of the gross R_0 channel. *Predicts:* gross R_0 plateau    lifts from -0.092 -> near -0.183, passing the left-plateau    band; peak structure still missing.
4. **M3a.3 (production IC generalization):** only after M3a.2.
5. **M3b/M3c (multi-ion + local-pH validation):** for the peak.
6. **M5 (L_eff alignment):** larger than +/-10%. Defer sizing    until M3b shows whether local-pH amplification accounts for    any of the peak-magnitude gap -- it may already be partly    contributing in the existing surface_pH_proxy field    (~9.7 at V=-0.40 in Run C, well above bulk pH 4).

## Comparison summary

Number of B7 tolerance bands passing under each candidate observable (out of 5 total; 4 quantitative + 1 qualitative shoulder):

| Channel                                    | Quantitative (/4) | Total (/5) |
|--------------------------------------------|--------------------|------------|
| net pc = R_0 - R_1 (original observable)   |  0                |  0         |
| gross R_0 (single-rate 2e, NEW)            |  0                |  0         |
| total cd = R_0 + R_1 (reference)           |  1                |  1         |

## Open follow-ups

- This audit only reinterprets the existing Run C *observable*. The underlying residual still uses the sequential R_0 + R_1 reaction list and both reactions are 2e (so `N_ELECTRONS=2` in I_SCALE is correct for the existing data).
- **M3a.1 (electron-weighted current accounting)** is required before any R_4e is added to the residual: once per-reaction `n_electrons` becomes heterogeneous (R_2e=2, R_4e=4), total disk current must be `Σ n_e_j · R_j · scale_per_unit_n_e`, not the current `Σ R_j · I_SCALE` with a global 2e prefactor (cf. Handoff 18 §2).
- **What replacing R_1 with parallel R_4e actually buys for the peroxide channel.** In the current sequential model, R_1 is specifically a peroxide *sink* (H₂O₂ + 2H⁺ + 2e⁻ → 2H₂O), so it couples directly to surface c_H2O2 and rate-limits gross R_0 through the H2O2 mass balance. A parallel R_4e (O₂ + 4H⁺ + 4e⁻ → 2H₂O) consumes O₂ independently; for a 2e-selective catalyst like CMK-3, R_4e's k0 is suppressed relative to R_2e, so R_2e absorbs the O₂ flux and gross R_2e rises toward the full 2e Levich saturation (-I_SCALE = -0.183 mA/cm²) at the cathodic plateau. That is the M3a.2 diagnostic prediction.
- **Surface pH excursion is already large in Run C.** `surface_pH_proxy` = -log10(c_H_surface · C_SCALE) sits at ~9.77 at V=-0.40 V and ~8.36 at V=+0.30 V — i.e., a 4-6 pH-unit alkaline excursion from bulk pH 4. This is in the ballpark of Ruggiero Fig 1B (~4 pH units at 3.25 mA/cm²), but applies under the surrogate (ClO4⁻ pH-countercharge) electrolyte, no Cs⁺ buffering, and may not translate quantitatively under M3b. The fact that the model's local pH is already alkaline yet produces no PC peak is consistent with the lock-in dominating: even with surface H⁺ depleted, R_2's intrinsic BV factor stays large enough to keep R_0/R_1 locked.

## Cross-references

- Status doc: `docs/mangan_alignment_status_2026-05-07.md`
- Source-paper finding: `docs/Ruggiero2022_JCatal_source_paper.md`
- Run C JSON: `StudyResults/mangan_p15_comparison/run_C/iv_curve.json`
- Run D verdict: `StudyResults/mangan_p15_comparison/run_D_verdict.md`
- Digitised target: `data/mangan_deck_p15_h2o2_current.csv`
- Plan H17/H18: `docs/CHATGPT_HANDOFF_17_RUGGIERO_REALIGNMENT_PLAN.md`, `docs/CHATGPT_HANDOFF_18_RUGGIERO_REALIGNMENT_COUNTERREPLY.md`
- Observable code: `Forward/bv_solver/observables.py:13-68`
- I_SCALE definition: `scripts/_bv_common.py:131-141`
- Audit derived observables JSON: `StudyResults/mangan_p15_comparison/m3a0_observables.json`
- Audit overlay plot: `StudyResults/mangan_p15_comparison/m3a0_audit.png`
