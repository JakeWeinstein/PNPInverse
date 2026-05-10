# Run D Verdict — Mangan Deck Page 15 Comparison

Plan B §"Run D — Verdict". Compares the model from
`StudyResults/mangan_p15_comparison/run_C/iv_curve.json` against
`data/mangan_deck_p15_h2o2_current.csv` (digitised page 15) using the
B7 tolerance bands.

## TL;DR

**Outcome: `shape_wrong`. 0 / 5 bands pass.**

The production stack (3sp + bikerman + muh + Stern 0.10 F/m² + clip=100,
C+D orchestrator) converged 25 / 25 across V_RHE ∈ [−0.40, +0.55] V but
produces `pc_mA_cm2 ≈ +1.5e−5 mA/cm² flat` across the full window. The
total disk current `cd_mA_cm2` does match the experimental shape
qualitatively. **The flat PC is not a bug; it is the surrogate model
correctly predicting a 4-electron-pathway-dominant ORR** given its
kinetic constants. The experiment runs on CMK-3 (2e-selective) in
0.1 M Cs₂SO₄ + 0.1 M H₂SO₄ + 0.1 M CsOH; our model runs on a generic
carbon catalyst with a ClO₄⁻ pH-countercharge surrogate. **The
shape mismatch is the surrogate gap, not a forward-solver defect.**
Closing it is a **broad alignment effort**, not a single M2 milestone.

## Run C metadata

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
converged              = 25 / 25
mesh_Ny                = 200
exponent_clip          = 100.0   (CLAUDE.md hard rule 2 — PC-trustworthy)
stern_capacitance_F_m2 = 0.10
formulation            = logc_muh
initializer            = debye_boltzmann
wall_seconds           = 887.5
```

## Tolerance band measurements

| # | Feature                                     | Experimental | Model               | Error           | Tolerance      | Pass |
|---|---------------------------------------------|--------------|---------------------|-----------------|----------------|------|
| 1 | peak voltage                                | +0.10 V      | +0.55 V (argmin of nearly-flat array — noise) | 450 mV | 50 mV  | FAIL |
| 2 | peak magnitude                              | −0.40 mA/cm² | +1.54e−05 mA/cm²    | 100.0% (sign-flipped) | 25% (rel) | FAIL |
| 3 | left plateau magnitude at V=−0.32 V         | −0.17 mA/cm² | +1.55e−05 mA/cm²    | 100.0% (sign-flipped) | 25% (rel) | FAIL |
| 4 | onset to zero (j first ≥ 0 from below)      | +0.45 V      | NaN (never crosses, pc > 0 everywhere) | NaN | 50 mV | FAIL |
| 5 | shoulder in V_RHE ∈ (+0.18, +0.30)          | yes (visible on page 15) | no (flat) | no inflection / slope flattening detected | qualitative | FAIL |

**Quantitative bands passing**: 0 / 4
**Total bands passing (incl. shoulder)**: 0 / 5

## Diagnosis: R_0 / R_1 lock-in to 4e-pathway steady state

Backing out the dimensionless reaction rates from the assembled (cd, pc)
via `R_0 + R_1 = −cd / I_SCALE` and `R_0 − R_1 = −pc / I_SCALE` reveals
the structural pattern (`I_SCALE = 0.1833 mA/cm²`):

| V_RHE (V) |   cd (mA/cm²) |     pc (mA/cm²) |     R_0 |     R_1 | R_1 / R_0 |
|-----------|---------------|-----------------|---------|---------|-----------|
| −0.400    | −0.184        | +1.55e−05       | +0.5015 | +0.5016 | 1.0002    |
| −0.250    | −0.182        | +1.55e−05       | +0.4965 | +0.4966 | 1.0002    |
|  0.000    | −0.175        | +1.55e−05       | +0.4778 | +0.4778 | 1.0002    |
| +0.100    | −0.168        | +1.55e−05       | +0.4585 | +0.4586 | 1.0002    |
| +0.250    | −0.146        | +1.55e−05       | +0.3972 | +0.3973 | 1.0002    |
| +0.350    | −0.0496       | +1.55e−05       | +0.1351 | +0.1352 | 1.0006    |
| +0.400    | −0.00184      | +1.54e−05       | +0.0050 | +0.0051 | 1.0169    |
| +0.450    | −2.7e−05      | +1.54e−05       | +3.2e−05 | +1.2e−04 | 3.61      |
| +0.500    | −1.5e−05      | +1.54e−05       | +1.6e−07 | +8.4e−05 | 540.7     |
| +0.550    | −1.5e−05      | +1.54e−05       | +6.1e−10 | +8.4e−05 | 137,746   |

Two regimes:

1. **V_RHE ∈ [−0.40, +0.30] V — mass-transport-coupled "4e steady
   state".** R_0 (O₂ → H₂O₂) and R_1 (H₂O₂ → H₂O) lock to each other
   at ~half the O₂ diffusion limit (R_1/R_0 ≈ 1.0002). Every H₂O₂
   produced by R_0 is consumed by R_1; no net peroxide leaves the
   surface. Total cd is at the 4-electron Levich limit
   (cd ≈ −I_SCALE since R_0 + R_1 ≈ 1).
2. **V_RHE > +0.30 V — R_0 shuts off, R_1 lingers.** R_0 falls
   exponentially as V → E_eq_R1 = +0.68 V, but R_1 still has a finite
   cathodic exp factor at η_R2 = V − 1.78 V, so R_1 ≫ R_0 in the
   tail. Model PC stays positive (anodic-leaning) at the seed level.

Why this happens (kinetic arithmetic at V = −0.40 V):

- η_R1 = (V − 0.68 V) / V_T = −42.0 (well within ±100 clip)
- η_R2 = (V − 1.78 V) / V_T = −84.8 (also within ±100; this is why
  CLAUDE.md hard rule 2 marks clip=100 as PC-trustworthy here — but
  trustworthy means "what the model says is what it physically computes",
  not "what the model says matches CMK-3 experiment")
- R_0 cathodic BV factor = exp(−0.627 × 2 × −42.0) ≈ 7.5e+22
- R_1 cathodic BV factor = exp(−0.5 × 2 × −84.8) ≈ 6.7e+36
- Ratio R_1_BV / R_0_BV ≈ 9e+13 (per unit concentration)
- k0 ratio: k0_R2 / k0_R1 = 1e−9 / 2.4e−8 = 0.0417

R_2's intrinsic BV factor at V = −0.40 V is ~14 OoM larger than R_1's;
k0_R2 is only 24× smaller. So once any peroxide exists at the surface,
R_1 (R_2) is overwhelmingly cathodic and consumes it as fast as it is
produced. The mass-transport coupling enforces R_0 ≈ R_1.

**For a 2e-selective catalyst like CMK-3**, the *effective* k0_R2 is
much smaller (or R_2 is suppressed by some other surface mechanism), so
R_2 cannot keep up with R_1's H₂O₂ production rate, and net peroxide
leaves the disk. Our model has no such suppression — the parameters
make it look like a generic 4e-selective ORR catalyst.

## Reframing the M2 implication

Plan B §"Conditional next phase" lays out three branches; the original
verdict-matrix mapped `shape_wrong` → "debug forward solver before any
M2 work". That mapping is wrong for the present case, because the
forward solver is producing exactly what its parameters and surrogate
electrolyte tell it to produce. The corrected reading:

**The shape mismatch is the surrogate gap, not a forward-solver bug.**
Closing it requires a *combined* alignment of the model with the
experiment, not a single milestone:

### Load-bearing alignment items

1. **Catalyst kinetic calibration to CMK-3 (2e selectivity).**
   - Reduce `k0_R2` (or otherwise suppress R_2) so the H₂O₂ → H₂O
     pathway no longer dominates. Plausible fixes: a CMK-3-specific
     k0_R2 from Ruggiero / Mangan deck supplementary; or an explicit
     2e-selectivity constraint baked into the surrogate.
   - Without this, R_0 ≈ R_1 lock-in is the model's steady state and
     no electrolyte upgrade alone will produce a peak in PC.
2. **M2 multi-ion electrolyte (Cs⁺ + SO₄²⁻ analytic closure).**
   - Required for the *peak voltage and shoulder shape* per deck
     pages 14, 17, 18 (size factor 0.16/0.20/0.24 → measurable shifts
     in onset and peak).
   - Brings ionic strength to literal experimental I ≈ 0.3 M, λ_D ≈ 0.6 nm
     — pushes solver into the 1-nm Debye conditioning regime that
     option 2 (ideal salt pair) was meant to validate before M2.
3. **Diffusion length alignment.**
   - Current `L_REF = 100 µm` gives I_SCALE ≈ 0.183 mA/cm² (2e ceiling
     ≈ 0.183 mA/cm²; 4e ceiling ≈ 0.367 mA/cm²). Levich δ at
     1600 rpm, ν_water = 1e−6 m²/s, D_O₂ = 1.9e−9 m²/s is
     δ ≈ 0.62 × D_O₂^(1/3) × ν^(1/6) × ω^(−1/2) ≈ 21 µm with c_O₂_sat
     ≈ 1.2 mM giving experimental j_lim ≈ 0.9 mA/cm² (2e) / 1.8 mA/cm²
     (4e). Our diffusion ceiling is therefore 5–10× below experiment.
     The experimental peak of −0.40 mA/cm² is roughly half the 2e
     Levich limit at deck conditions — *cannot* be reproduced by our
     model magnitude even after items 1–2.

### Useful diagnostic side-quests (not load-bearing)

These are still worth doing because they harden the comparison surface,
but they are not what is causing the flat PC:

- **Probe R_0 and R_1 separately** (mode="reaction"). Already implicitly
  done via the `R_0 = (−cd − pc)/(2·I_SCALE)`,
  `R_1 = (−cd + pc)/(2·I_SCALE)` decomposition above; an explicit
  per-reaction observable would just give the same answer in JSON form.
- **Inspect c_H2O2_surface diagnostic** (`c1_surface_mean`) along the V
  grid. Hypothesis: it sits at H2O2_SEED_NONDIM = 1e−4 because R_2
  consumes H₂O₂ as fast as R_1 produces it. Confirming this would
  formalise the "R_0 / R_1 lock-in" interpretation here.
- **Verify the deck's H₂O₂-current derivation.** If page 15's "Current
  density of H₂O₂ production" is the *gross* 2e contribution rather
  than the *net*, our matching observable is `R_0` alone, not
  `R_0 − R_1`. This would reframe the model output entirely (R_0 in
  Run C peaks at −0.184 mA/cm² at V = −0.40, not far from the
  experimental −0.18 left-plateau magnitude). Worth confirming
  with whoever has access to the Mangan team.

### What this means for the M2 decision

- **M2 is necessary but not sufficient.** Plan B's option 3 (steric
  multi-ion analytic closure) is still load-bearing for peak voltage
  / shoulder shape, but pairing it with item 1 (CMK-3 kinetic
  calibration) is required for any peak to appear in PC at all.
- **M2 should land alongside** the kinetic recalibration, not before
  it. Otherwise the M2 output will still show flat PC and the
  electrolyte upgrade will look ineffective.
- **Plan B's "debug forward solver before M2" branch was the wrong
  read of `shape_wrong`** for this target. The right read here is
  "the surrogate is fundamentally not the experimental setup; align
  with experiment along the kinetics and electrolyte axes together,
  not the diffuse-layer-screening axis alone".

## Verdict

`outcome = "shape_wrong"`

`shape_wrong_reason = "surrogate_4e_pathway_dominant"`

(extends the Plan B verdict matrix with a sub-tag distinguishing this
case from the original "BV constants / clip / IC" debug branch)

## What this verdict does NOT do

- Does not advance `comparison_status` past `"deck_proxy"`.
- Does not modify `Forward/bv_solver/forms_logc{,_muh}.py`,
  `observables.py`, or `validation.py`. The forward solver is fine;
  the disagreement is at the *model parameter / electrolyte* level.
- Does not regenerate Run A / Run B baselines.

## Cross-references

- Plan: `~/.claude/plans/swirling-crunching-wren.md`
- M0 extraction: `docs/m0_target_extraction.md`
- Run C output: `StudyResults/mangan_p15_comparison/run_C/iv_curve.json`,
  `comparison.png`, `diagnostics.json`
- Digitised target: `data/mangan_deck_p15_h2o2_current.csv`
- Existing reference run with the same R_0 ≈ R_1 lock-in:
  `StudyResults/peroxide_window_3sp_bikerman_muh_2b/iv_curve.json`
  (the `3sp_bikerman_muh_stern_0p10_clip50` pass)
- BV rate construction: `Forward/bv_solver/forms_logc_muh.py:393–475`
  (log-rate cathodic + anodic terms per reaction)
- BV observable wiring: `Forward/bv_solver/observables.py:45–52`
- Production kinetic constants: `scripts/_bv_common.py:K0_PHYS_R1, K0_PHYS_R2,
  ALPHA_R1, ALPHA_R2, E_EQ_R1_V, E_EQ_R2_V`
- Clipping convention authority: `docs/clipping_conventions.md`,
  CLAUDE.md hard rule 2
