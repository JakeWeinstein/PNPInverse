# M0 Target Extraction — Mangan Deck Page 15 (H₂O₂ Production Current Density, pH 4, Cs⁺)

This document captures the B1–B10 outputs from the Plan B "swirling-crunching-wren"
target-curve workstream. It promotes the deferred-parameter placeholders from
`scripts/_bv_common.make_experiment_metadata` to extracted values **for this
specific target** (Mangan2025 deck p.15). Other future targets remain
placeholder until their own M0 sweep is run.

Scope: Plan B§"M0 extraction tasks (anchored to deck page 15)".

---

## B1 — Target lock (DONE)

```
target_curve = "Mangan2025_deck_p15_H2O2_current_density_pH4_Csplus"
```

The figure plots net peroxide-producing current density at the disk —
directly comparable to our `peroxide_current` observable
(`Forward/bv_solver/observables.py:_build_bv_observable_form` with
`mode="peroxide_current"` plus `scale=-I_SCALE`). No ring-machinery
post-processing needed for this comparison.

## B2 — Digitization

- File: `data/mangan_deck_p15_h2o2_current.csv` (37 points).
- Source: `writeups/WeekOfFeb25/assets/mangan_slide15.png` (2933×1650 PNG).
- Eyeballed uncertainty: ±0.01 V on V_RHE; ±10% relative on j (or
  ±0.02 mA/cm² absolute floor near zero).
- Sampling: denser around peak (V_RHE ∈ [+0.05, +0.20]) and shoulder
  (V_RHE ∈ [+0.18, +0.30]).

Key features captured:

| Feature                      | V_RHE (V)    | j (mA/cm²)   |
|------------------------------|--------------|--------------|
| Left plateau                 | ~ −0.32      | ~ −0.17      |
| Peak (most cathodic)         | ~ +0.10–0.13 | ~ −0.40      |
| Shoulder begin               | ~ +0.18      | ~ −0.30      |
| Shoulder end                 | ~ +0.25      | ~ −0.18      |
| Onset to zero                | ~ +0.42–0.45 | ~ 0.00       |

Note on peak magnitude: Plan B's eyeballed prior was −0.35 mA/cm². The
high-resolution PNG resolves the local minimum to ~ −0.40 mA/cm².
Page 16 of the deck (model overlay) shows experimental peak at
~ −0.36 mA/cm². The −0.40 value is within the ±10% digitization
uncertainty of −0.36 and within the ±25% B7 tolerance band.

## B3 — RRDE constants (Ruggiero 2022 cross-reference)

Page 15 does not state experimental RRDE parameters. The deck cites
Ruggiero et al. *Journal of Catalysis* 414 (2022): 33–43 on multiple
slides (10, 11, 19); the corresponding Northwestern accepted manuscript
is at https://www.osti.gov/servlets/purl/2418971. The deck page 14
schematic explicitly lists the same modeling species (Cs⁺, H⁺, SO₄²⁻,
OH⁻) that Ruggiero used in the 0.1 M H₂SO₄ + 0.1 M Cs₂SO₄ + 0.1 M CsOH
mixture, so we adopt Ruggiero's experimental constants pending direct
confirmation from the deck author.

| Quantity            | Value                              | Source                                 |
|---------------------|------------------------------------|----------------------------------------|
| Rotation rate       | 1600 rpm                            | Ruggiero ¶"Methods" line 460, 508       |
| Collection N        | 0.224                               | Ruggiero ¶"determined to be 0.224"     |
| Ring electrode      | Au + electrodeposited IrOx          | Ruggiero ¶"IrOx-Au ring"                |
| Ring usage          | OCP for local pH sensing (NOT H₂O₂ detection) | Ruggiero §"To detect local pH" |
| Disk scan range     | 1.1 V → 0.05 V vs RHE               | Ruggiero ¶"Methods" line 460            |
| Disk scan rate      | 20 mV/s                             | Ruggiero ¶"Methods" line 460, 506–508   |
| Disk catalyst       | CMK-3 mesoporous carbon             | Ruggiero ¶"Methods" line 364            |
| Electrolyte (Cs)    | 0.1 M H₂SO₄ + 0.1 M Cs₂SO₄ + 0.1 M CsOH | Ruggiero ¶"electrolytes used"      |
| [SO₄²⁻] anion conc. | 0.1 M                               | Ruggiero ¶"anion concentration of 0.1 M" |
| Total [cation]      | ~0.2 M (Cs⁺ + H⁺)                   | Ruggiero ¶"total cation concentration"  |
| Bulk pH (Cs⁺)       | ~4 (mixed acid+salt+base)           | Page 15 + Ruggiero pH list (2,4,6,9,10,12) |
| Bipotentiostat      | Biologic VSP 200                    | Ruggiero ¶"Methods" line 433            |
| Cell                | Heart-shaped electrochemical cell   | Ruggiero ¶"Methods" line 433            |

**Caveat on j_H₂O₂ derivation.** Ruggiero's RRDE uses the Au-IrOx ring
for pH sensing, not for H₂O₂ oxidation. The "H₂O₂ Production Current
Density" plotted on deck page 15 must therefore be derived from the
disk current and an assumed (or separately measured) electron count.
The deck does not state how j_H₂O₂ was computed. For the comparison
on page 15, we treat the experimental curve as a black-box `j_H₂O₂`
target and compare directly against our model's `peroxide_current`
observable. If a downstream comparison needs to reconstruct ring/disk
splits, additional Ruggiero supplementary tables would be needed.

## B4 — IrOx calibration — SKIP

Page 15 has no local-pH data. IrOx calibration (Au-IrOx OCP → pH) is
not on the comparison surface. Defer until a target figure with
local-pH data is added (see Ruggiero Figure 1A and deck page 10 for
where this would re-enter scope).

## B5 — Source authority per quantity

Per the Plan B convention:

```
target_curve, j_H2O2 trace        → source_authority = "Mangan2025_deck"
RRDE rotation, N, scan rate,
  ring potential, scan range,
  disk catalyst, electrolyte,
  cell                             → source_authority = "Ruggiero_manuscript"
```

This mapping is enforced at the `experiment_metadata` field level
through a single top-level `source_authority` string. Where the run
covers both, set the field to `"Mangan2025_deck"` (the figure being
compared) and rely on this document to record the per-quantity split.

## B6 — Voltage range

**Truncate comparison at V_RHE ∈ [−0.32, +0.50] V** (the data range
visible on page 15 with non-zero peroxide current). Run the solver
with **V_RHE ∈ [−0.40, +0.55] V** to provide context on either side.
No solver-window extension milestone is required: the full target
range fits comfortably below the +1.0 V production ceiling
(`docs/4sp_bikerman_ic_option_2b_results.md`).

## B7 — Acceptance tier — semi-quant

Tolerance bands:

| Feature                                                                    | Tolerance         |
|----------------------------------------------------------------------------|-------------------|
| Peak voltage (digitized ≈ +0.10 V)                                          | ±50 mV            |
| Peak magnitude (digitized ≈ −0.40 mA/cm²)                                   | ±25%              |
| Left plateau magnitude at V_RHE ≈ −0.32 V (digitized ≈ −0.17 mA/cm²)        | ±25%              |
| Onset position where j → 0 (digitized ≈ +0.45 V)                            | ±50 mV            |
| Shoulder visibility in V_RHE ∈ [+0.18, +0.30]                               | qualitative — yes/no |

The ±25% magnitude band is calibrated to deck page 16's L_eff = 86 µm
overshoot (model peak ~−0.47 vs experimental −0.36 ≈ +30%): tighter
would set us above the deck's own model performance; looser would
admit agreement when we are worse than the deck.

`acceptance_tier = "semi_quant"`.

## B8 — Deck-side ionic strength

Deck page 14 explicitly draws Cs⁺ + H⁺ + SO₄²⁻ + OH⁻ as model species.
Pages 13–14 describe spectral methods + nonlinear spatial mapping to
resolve nm-scale EDL — consistent with running at literal experimental
ionic strength (I ≈ 0.3 M, λ_D ≈ 0.6 nm). **Conservative assumption:
deck runs literal experimental ionic strength.**

This locks M2 toward option 3 (steric multi-ion analytic closure)
minimum if we target semi-quant on this figure with a screening-
sensitive observable. Option 2 (ideal salt pair) is a useful
intermediate diagnostic — same architecture as our current single-ion
analytic, just two ideal Boltzmann ions instead of one — to validate
that our solver tolerates 1-nm Debye conditioning without sterics.

## B9 — Stern + cation priors

| Quantity                       | Value range                    | Notes / authority                                                        |
|--------------------------------|--------------------------------|--------------------------------------------------------------------------|
| Stern capacitance (carbon)     | 0.05–0.5 F/m² (5–50 µF/cm²)     | Memory; production default 0.10 F/m² is mid-range                        |
| Cs⁺ bare radius                | 1.67 Å (Shannon)                | Standard ionic radius literature                                          |
| Cs⁺ hydrated (Marcus) radius   | 3.29 Å                          | More appropriate for Bikerman steric closure (excluded volume)            |
| Cs⁺-hydrated `a_nondim`        | ~10⁻⁴ at I = 0.3 M             | a = N_A · (2·r_Cs)³ · C_SCALE → ~100× smaller than current `A_DEFAULT=0.01` (which assumed I = 0.1 mol/m³ ≪ deck) |

Mapping radius → a_nondim:

```
a [m³ / mol] = N_A · (2·r_eff_m)³                         (excluded volume per mol)
a_nondim     = a · C_SCALE [mol/m³]                       (with C_SCALE = C_O2 = 0.5 mol/m³)
```

For Cs⁺-hydrated (r_eff = 3.29 Å) at C_SCALE = 0.5 mol/m³:

```
a_phys ≈ 6.022e23 · (6.58e-10)³ ≈ 1.7e-7 m³/mol
a_nondim ≈ 1.7e-7 · 0.5 ≈ 8.6e-8
```

This is much smaller than `A_DEFAULT = 0.01`. Note the relevant
`c·a_nondim` saturation product is what matters: at deck-correct
[ClO₄⁻] = 0.1 M = 100 mol/m³ (or [SO₄²⁻] = 0.1 M, scaled by 2 for
charge), `c_b_nondim ≈ 200`, so `c·a ≈ 1.7e-5` — far from steric
saturation. Consistent with Cs⁺ steric not being a strong load-bearing
control on this figure (c.f. deck page 18 which shows size-factor
sensitivity is moderate over 0.16–0.24).

The current production stack runs `A_DEFAULT = 0.01` with
`c_b_nondim = C_CLO4_HAT = 0.2` (i.e. 0.1 mol/m³ at C_SCALE = 0.5),
giving a saturation product of `c·a = 0.002` — also unsaturated.
Switching to the deck-correct ionic strength does not push us into
saturation; it's the screening length, not the steric, that changes.

## B10 — RDE fallback — N/A

Page 15 plots peroxide current density (RRDE-derived). We compare
directly against `peroxide_current` without going through the
disk-only/RDE channel. Skip.

---

## Promoted experiment_metadata for the page-15 target

Used by `scripts/studies/mangan_p15_comparison.py` (the Run C study
script):

```python
experiment_metadata = make_experiment_metadata(
    catalyst="CMK-3",
    geometry="RRDE",
    pH_bulk=4.0,
    cation="Cs+",
    anion_model="ClO4_protonic_surrogate",   # current solver carries ClO4-, not SO4^2-
    rotation_rate_rpm=1600.0,
    L_eff_m=None,                             # not used by current solver (stagnant-film geometry)
    N_collection=0.224,
    electrolyte_model="pH_countercharge_surrogate",  # honest until M2 lands
    comparison_status="deck_proxy",           # not yet "deck_quantitative_candidate"
    source_authority="Mangan2025_deck",       # primary; per-quantity split documented above
    target_curve="Mangan2025_deck_p15_H2O2_current_density_pH4_Csplus",
    acceptance_tier="semi_quant",
)
```

The `electrolyte_model` field stays at `"pH_countercharge_surrogate"`
until M2 implements the SO₄²⁻ + Cs⁺ multi-ion analytic closure. M1.5
fixes the Stern-aware IC seed; it does NOT change the electrolyte
model.

The `anion_model` field stays at `"ClO4_protonic_surrogate"` for the
same reason: our solver tracks an analytic ClO₄⁻ counterion, not the
deck's SO₄²⁻. This is the central honesty bookkeeping — promotion past
`"deck_proxy"` requires aligning anion_model and electrolyte_model
with the actual experimental composition.

## References for downstream consumers

- Plan: `~/.claude/plans/swirling-crunching-wren.md`
- Plan A (M1) executed-output convention: `memory/project_mangan_m1_deferred_parameters.md`
- Run C output: `StudyResults/mangan_p15_comparison/run_C/`
- Run D verdict: `StudyResults/mangan_p15_comparison/run_D_verdict.md`
- Production stack docs: `docs/bv_solver_unified_api.md`, `docs/4sp_bikerman_ic_option_2b_results.md`
- Clipping convention (clip=100 mandatory for PC trustworthiness):
  `docs/clipping_conventions.md`, CLAUDE.md hard rule 2
