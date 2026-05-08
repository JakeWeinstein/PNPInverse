# Mangan Deck Page 15 Comparison — Summary (2026-05-07)

Self-contained summary of the Plan B "swirling-crunching-wren" workstream:
M0 extraction for the Mangan deck page-15 H₂O₂-current-density target,
Run C of the production stack against that target, Run D verdict, and the
load-bearing alignment items the verdict surfaced. Written for someone
picking up this thread later — it folds in the reframing that came out of
the post-verdict review.

## What the target is

Mangan 2025 deck page 15: "Current density of H₂O₂ production" at pH 4
with Cs⁺ ions on a CMK-3 carbon disk, RRDE geometry. The figure is the
sole Cs⁺ peroxide-current trace shown in the deck. Eyeballed features
from the digitisation:

| Feature                  | V_RHE      | j_H₂O₂        |
|--------------------------|------------|---------------|
| Left plateau             | ≈ −0.32 V  | ≈ −0.17 mA/cm² |
| Peak (most cathodic)     | ≈ +0.10 V  | ≈ −0.40 mA/cm² |
| Shoulder                 | +0.18–+0.30 V | (visible inflection) |
| Onset to zero            | ≈ +0.45 V  |  0             |

Digitised at 37 points: `data/mangan_deck_p15_h2o2_current.csv`.

## What was run

`scripts/studies/mangan_p15_comparison.py` against the production stack
per CLAUDE.md "Calling the production solver":

```
species              = THREE_SPECIES_LOGC_BOLTZMANN
boltzmann_counterions= [DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC]
formulation          = "logc_muh"
log_rate             = True
initializer          = "debye_boltzmann"
stern_capacitance    = 0.10 F/m²
exponent_clip        = 100  (clip=100 — only PC-trustworthy setting)
orchestrator         = solve_grid_per_voltage_cold_with_warm_fallback (C+D)
voltage grid         = 25 points across V_RHE ∈ [−0.40, +0.55] V
mesh_Ny              = 200
```

Result: **converged 25 / 25 in 887.5 s**. Output in
`StudyResults/mangan_p15_comparison/run_C/`.

## What we expected to learn vs. what we actually found

**Expected branch from Plan B§"Run D — Verdict".** Three outcomes were
articulated:

- *all_bands_met* — surrogate is fine, no electrolyte upgrade needed
  (judged unlikely)
- *magnitude_off_shape_correct* — most likely; M2 (multi-ion electrolyte)
  is the load-bearing fix
- *shape_wrong* — debug forward solver (BV constants, clip, IC) before
  M2

Run C scored **0 / 5** tolerance bands. Three rounds of data-look made
the right read iteratively clearer.

### First read (wrong): "shape_wrong → forward-solver bug"

Initial verdict followed the Plan B branch literally and recommended
debugging the BV residual / observable code before M2. It also flagged
the flat-PC behavior as something that "had not previously been
benchmarked against experimental peroxide current".

### Second read: "the surrogate is missing the load-bearing physics"

The user pushed back: *"doesn't the current stack still use the same 3sp
+ ClO4- model? I would assume that the qualitative shape matching would
come from bringing the solver setup more in line with the experimental
setup."* This reframed the question — the model isn't broken, it just
isn't the experimental cell. **"Surrogate"** here is the simplified
forward-model bundle of approximations recorded in
`experiment_metadata.electrolyte_model = "pH_countercharge_surrogate"`,
not anything related to the ML `Surrogate/` system.

| Aspect           | Production solver (today)                    | Mangan p.15 experiment                  |
|------------------|----------------------------------------------|-----------------------------------------|
| Dynamic species  | 3 (O₂, H₂O₂, H⁺)                              | 4–5 (Cs⁺, H⁺, SO₄²⁻, OH⁻ + neutrals)    |
| Counterion       | analytic ClO₄⁻ Boltzmann (single ion)         | dynamic Cs⁺ + SO₄²⁻ + OH⁻               |
| Cation effects   | none (just net charge)                        | Cs⁺-specific OHP layering (deck pp.10–11, 18) |
| Kinetics         | generic carbon, k0_R1 = 2.4e−8, k0_R2 = 1e−9 | CMK-3 (high 2e-H₂O₂ selectivity)       |
| Transport length | L_REF = 100 µm stagnant film                  | Levich δ ≈ 21 µm at 1600 rpm            |

### Third read (clean): R_0 / R_1 lock-in, 4e-pathway-dominant ORR

Backing out the rates from assembled (cd, pc) via `R_0 + R_1 = −cd/I_SCALE`,
`R_0 − R_1 = −pc/I_SCALE`:

| V_RHE  | cd (mA/cm²) | pc (mA/cm²) |    R_0    |    R_1    | R_1/R_0   |
|--------|-------------|-------------|-----------|-----------|-----------|
| −0.40  | −0.184      | +1.55e−5    | +0.5015   | +0.5016   | 1.0002    |
| 0.00   | −0.175      | +1.55e−5    | +0.4778   | +0.4778   | 1.0002    |
| +0.10  | −0.168      | +1.55e−5    | +0.4585   | +0.4586   | 1.0002    |
| +0.30  | −0.116      | +1.55e−5    | +0.3175   | +0.3176   | 1.0003    |
| +0.40  | −0.0018     | +1.54e−5    | +0.0050   | +0.0051   | 1.0169    |
| +0.50  | −1.5e−5     | +1.54e−5    | +1.6e−7   | +8.4e−5   | 540.7     |

Two regimes:

1. **V_RHE ∈ [−0.40, +0.30] V — mass-transport-coupled 4e steady state.**
   R_0 and R_1 lock at ~half the O₂ diffusion limit each. Every H₂O₂
   produced by R_0 is consumed by R_1. Net peroxide ≈ 0.
2. **V_RHE > +0.30 V — R_0 shuts off, R_1 lingers.** R_0 falls
   exponentially as V → E_eq_R1 = +0.68 V; R_1 still has a finite
   cathodic factor (η_R2 = V − 1.78 V is far below E_eq_R2), so R_1
   stays alive consuming the H₂O₂ seed.

Why R_0 ≈ R_1 in regime 1 (kinetic arithmetic at V = −0.40 V):

```
η_R1 = (V − 0.68)/V_T = −42.0   (within ±100 clip)
η_R2 = (V − 1.78)/V_T = −84.8   (within ±100 clip)
R_0 cathodic BV factor = exp(−0.627·2·−42.0) ≈ 7.5e+22
R_1 cathodic BV factor = exp(−0.5·  2·−84.8) ≈ 6.7e+36
ratio R_1_BV / R_0_BV ≈ 9e+13
k0_R2 / k0_R1 = 1e−9 / 2.4e−8 = 0.0417
```

R_2's intrinsic cathodic BV factor is ~14 OoM larger than R_1's; k0_R2
is only 24× smaller. So once any H₂O₂ exists at the surface, R_2 is
overwhelmingly cathodic and consumes it as fast as it is produced. The
mass-transport coupling enforces R_0 ≈ R_1 at half the diffusion limit
each, total cd at the 4e Levich ceiling.

This is **the model correctly producing 4-electron-pathway-dominant ORR**.
For a 2e-selective catalyst like CMK-3, the *effective* k0_R2 is much
smaller (or R_2 is suppressed by some surface mechanism), so net peroxide
leaves the disk. Our parameters describe a generic 4e catalyst, so the
PC observable lands at zero.

## Reframed M2 implication

Plan B's "shape_wrong → debug forward solver before M2" branch is the
wrong mapping for this target. The forward solver is producing exactly
what its parameters and surrogate electrolyte tell it to produce. The
shape mismatch is the surrogate gap.

**Closing the gap is a multi-piece alignment, not a single milestone.**
Three load-bearing items must land together:

1. **CMK-3 kinetic recalibration.** Drop k0_R2 (or otherwise suppress
   R_2) so the H₂O₂ → H₂O pathway no longer dominates. Without this,
   *no* electrolyte upgrade alone produces a peak in PC — R_0 ≈ R_1
   lock-in remains the steady state regardless of φ-profile changes.
2. **M2 multi-ion electrolyte.** Cs⁺ + SO₄²⁻ analytic closure for the
   peak voltage and shoulder shape per deck pages 14, 17, 18 (size
   factor 0.16/0.20/0.24 → measurable peak/onset shifts). Brings I to
   literal experimental ≈ 0.3 M, λ_D ≈ 0.6 nm — pushes solver into the
   1-nm Debye conditioning regime that option 2 (ideal salt pair) was
   meant to validate before M2.
3. **Levich-δ alignment.** Current `L_REF = 100 µm` gives I_SCALE ≈
   0.183 mA/cm² (2e ceiling). Experimental Levich δ at 1600 rpm,
   c_O₂_sat ≈ 1.2 mM gives j_lim ≈ 0.9 mA/cm² (2e). Our diffusion
   ceiling is 5–10× below the experimental peak of −0.40 mA/cm² — even
   after items 1 + 2, the magnitude target is unreachable without this.

Sequencing note: **M2 should land alongside item 1, not before.**
M2-only output will still show flat PC (R_0/R_1 lock-in unchanged by
electrolyte), and the electrolyte upgrade will look ineffective if
landed first. Items 1 and 2 are mutually reinforcing once paired.

## What does work in Run C

`cd_mA_cm2` (total disk current density) matches the experimental shape
qualitatively:

- Mass-transport-limited at ≈ −0.18 mA/cm² for V ≲ −0.10 V
- Rises monotonically toward zero
- Crosses near zero around V ≈ +0.42 V (within ~30 mV of the
  experimental onset at +0.45 V)
- Magnitude at V = −0.32 V matches the experimental left-plateau within
  ~10% (model −0.183 vs experimental −0.17)

The bulk shape of total current is captured. Only the peroxide-current
*observable channel* is flat at zero, and the R_0/R_1 lock-in
arithmetic explains why.

## Open question worth confirming with the Mangan team

Page 15's vertical axis is labeled "Peroxide Current Density (mA/cm²)".
The deck does not state whether this is:

- the **gross** R_0 contribution (i.e. what an RRDE Pt ring at H₂O₂-
  oxidation potential would collect, before subsequent R_2 consumption),
  or
- the **net** (R_0 − R_1) at the disk surface.

Our `peroxide_current` observable computes the *net* per `Forward/bv_solver/
observables.py:45–52`. If the deck means *gross*, the matching observable
is `R_0` alone (`mode="reaction", reaction_index=0`), and the model's
R_0 in Run C peaks at −0.184 mA/cm² at V = −0.40 V — close to the
experimental left plateau of −0.17 mA/cm². If so, the comparison surface
needs to switch and the entire diagnosis above changes.

## Artefacts produced

| Path                                                           | What it is                                              |
|----------------------------------------------------------------|---------------------------------------------------------|
| `data/mangan_deck_p15_h2o2_current.csv`                        | 37-pt digitisation of slide 15 (provenance in header)   |
| `docs/m0_target_extraction.md`                                 | M0 extraction (B1–B10 outputs, RRDE constants, B7 bands) |
| `scripts/studies/mangan_p15_comparison.py`                     | Run C study script (production stack, 25-V grid)        |
| `scripts/studies/mangan_p15_run_D_verdict.py`                  | Run D tolerance-band computation                        |
| `StudyResults/mangan_p15_comparison/run_C/iv_curve.json`       | Run C model output (cd, pc, RRDE observables, metadata) |
| `StudyResults/mangan_p15_comparison/run_C/diagnostics.json`    | Per-V solver diagnostics                                |
| `StudyResults/mangan_p15_comparison/run_C/comparison.png`      | Side-by-side experimental + model PC overlay            |
| `StudyResults/mangan_p15_comparison/run_D_verdict.md`          | Verdict report with tolerance bands and reframed M2 implication |
| `memory/project_mangan_m0_extraction_complete.md`              | Memory entry — page-15 M0 done; surrogate-gap diagnosis |

## What did not get built

Run A (pre-IC-fix baseline) and Run B (post-IC-fix baseline against the
2b voltage grid) were optional in the plan once the verdict landed; they
quantify the IC fix's effect on PC, which is a separate scientific
question and not load-bearing for the M2 decision now that the diagnosis
is on the rate constants and electrolyte rather than the IC seed.

`comparison_status` stays at `"deck_proxy"` per the M1 deferred-parameter
convention. Promotion past requires both the R_0/R_1 lock-in fix
(item 1 above) and the M2 electrolyte upgrade (item 2).

## Cross-references

- Plan: `~/.claude/plans/swirling-crunching-wren.md`
- M0 doc: `docs/m0_target_extraction.md`
- Verdict: `StudyResults/mangan_p15_comparison/run_D_verdict.md`
- Memory: `memory/project_mangan_m0_extraction_complete.md`
- BV rate construction: `Forward/bv_solver/forms_logc_muh.py:393–475`
- BV observable wiring: `Forward/bv_solver/observables.py:45–52`
- Production kinetic constants: `scripts/_bv_common.py` (`K0_PHYS_R1`,
  `K0_PHYS_R2`, `ALPHA_R1`, `ALPHA_R2`, `E_EQ_R1_V`, `E_EQ_R2_V`)
- M1 deferred-parameter convention: `memory/project_mangan_m1_deferred_parameters.md`
- Clipping convention authority: `docs/clipping_conventions.md`,
  CLAUDE.md hard rule 2
- Existing reference run with identical R_0/R_1 lock-in (Stern pass):
  `StudyResults/peroxide_window_3sp_bikerman_muh_2b/iv_curve.json`
