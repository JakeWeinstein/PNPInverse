# Ruggiero et al. 2022 (J. Catal.) — Source Paper for Mangan Deck

Date compiled: 2026-05-07
Why this exists: the Mangan 2025 catalysis deck (`docs/Mangan2025_Catalysis.pdf`)
is built on top of this peer-reviewed paper. Mangan is a co-author. This doc
captures the load-bearing experimental and modeling facts that the deck assumes
but doesn't restate — including a structural finding about the reaction model
that contradicts the assumption built into our current solver. Read this before
doing any further Mangan-alignment work.

## Citation

Brianna N. Ruggiero, Kenzie M. Sanroman Gutierrez, Jithin D. George,
**Niall M. Mangan**, Justin M. Notestein, Linsey C. Seitz.
**"Probing the Relationship Between Bulk and Local Environments to
Understand Impacts on Electrocatalytic Oxygen Reduction Reaction."**
*Journal of Catalysis* (Elsevier), 2022.

PII: `S0021951722003591`. Affiliations: Department of Chemical and
Biological Engineering (Ruggiero, Sanroman Gutierrez, Notestein, Seitz)
and Department of Engineering Sciences and Applied Mathematics (George,
Mangan), Northwestern University, Evanston IL 60208.

Mangan is the same author whose name is on the deck. The deck's modeling
slides (pages 14, 16+ of `docs/Mangan2025_Catalysis.pdf`) extend this
experimental paper.

## Where to find the paper

- **Local PDF (copy of the OSTI accepted manuscript)**:
  `docs/Ruggiero2022_JCatal_manuscript.pdf`
- **OSTI canonical URL**: `https://www.osti.gov/servlets/purl/2418971`
- **Publisher version of record**: `https://www.sciencedirect.com/science/article/pii/S0021951722003591`
  (paywalled; we use the OSTI accepted manuscript instead)

## Hard experimental facts (extract these, don't re-derive)

### Catalyst and electrode

| Quantity                | Value                       | Section |
|-------------------------|-----------------------------|---------|
| Catalyst material       | CMK-3 (mesoporous carbon, ACS Material) | §2.1    |
| Disk loading            | 0.5 mg/cm²                  | §2.2    |
| Disk substrate          | Glassy carbon, 0.196 cm²    | §2.2    |
| Catalyst ink            | 2 mg CMK-3 + 197.3 µL EtOH + 2.7 µL cation-exchanged Nafion; sonicated 3 h; 10 µL drop-cast | §2.2    |

### Electrolyte composition

| Quantity                | Value                       |
|-------------------------|-----------------------------|
| Acid                    | 0.1 M H₂SO₄                |
| Base                    | 0.1 M MOH                   |
| Salt                    | 0.1 M M₂SO₄                |
| Cation set M⁺           | Li⁺, Na⁺, K⁺, Cs⁺ (99.999% trace metals basis) |
| Anion concentration     | **0.1 M [SO₄²⁻]**          |
| Total cation concentration | **0.2 M** (e.g. [H⁺] + [K⁺] in pH-4 K-electrolyte) |
| Bulk pH range studied   | 2-12                        |
| Ionic strength I        | **0.5·Σz²c = 0.3 M**        |
| Debye length λ_D at I = 0.3 M | ≈ 0.55 nm            |

**Anion choice rationale (§3.1, verbatim from §3.1):** sulfate was chosen
because the application is H₂O₂ for environmental remediation, and
"perchlorate ions...are well known as persistent environmental pollutants."
**The paper explicitly rejects perchlorate.** Our solver's
`pH_countercharge_surrogate` ClO₄⁻ is the wrong anion identity by design
in this context.

### RRDE protocol

| Quantity                | Value                       |
|-------------------------|-----------------------------|
| Geometry                | RRDE with Au-IrOx ring (IrOx for local pH sensing) |
| Rotation rate           | **1600 rpm**                |
| Pt ring potential       | **+1.2 V vs RHE** (for H₂O₂ oxidation) |
| **Collection efficiency `N`** | **0.224** (Fe(CN)₆⁴⁻/³⁻ standardized) |
| LSV scan range          | **1.1 V → 0.05 V vs RHE**   |
| LSV scan rate           | **20 mV/s**                 |
| O₂ purge time           | 30 minutes                  |
| Capacitive correction   | Background subtracted using 0.8-1.0 V window |
| Reference electrode     | Ag/AgCl (sat'd KCl); E_RHE = E_AgCl + 0.197 V + 0.059·pH |

### Selectivity and electron-number formulas (§2.4, Eq 1)

```
H₂O₂(%) = 200 · (I_R / N) / ( |I_D| + I_R / N )
n_e     =   4 · |I_D|       / ( |I_D| + I_R / N )
```

with `I_R` = ring current, `I_D` = disk current, `N = 0.224`. These are
the standard RRDE selectivity / electron-number formulas; the factor of
200 (not 100) comes from the 2e peroxide pathway in a 4e-vs-2e
decomposition.

### Mass-transport / Levich parameters (§2.4, Eq 6)

| Quantity                | Value                       |
|-------------------------|-----------------------------|
| C(O₂) at pH 5-13        | 1.2 × 10⁻⁶ mol/cm³ = 1.2 mM |
| C(O₂) at pH 2           | 1.1 × 10⁻⁶ mol/cm³          |
| D(O₂) at pH 5-13        | 1.9 × 10⁻⁵ cm²/s            |
| D(O₂) at pH 2           | 1.8 × 10⁻⁵ cm²/s            |
| Kinematic viscosity     | 0.01 cm²/s                  |
| Levich diffusion-limited current | j_lim = 0.62·n·F·D^(2/3)·ν^(-1/6)·ω^(1/2)·C |

For 1600 rpm and 4-electron full ORR at pH 5-13 these constants give
j_disk_lim ≈ 5.7 mA/cm² (4e ceiling). For pure 2e peroxide pathway:
j_disk_lim ≈ 2.85 mA/cm² (half).

## Reaction model — STRUCTURAL FINDING

The paper's reaction set is **parallel 2e and 4e ORR**, not sequential
peroxide-then-water. From §1, Eqs 1-2 (acidic):

```
4e:  O₂ + 4H⁺ + 4e⁻ → 2H₂O      E⁰ = 1.23  V_RHE
2e:  O₂ + 2H⁺ + 2e⁻ → H₂O₂      E⁰ = 0.695 V_RHE
```

Per §1 (their associative-mechanism description), selectivity between 2e
and 4e pathways "is determined by preference for breaking the *-OOH bond
(between the catalyst surface and first oxygen) or the *-O-OH bond
(between the first and second oxygen atoms) to produce H₂O₂ or H₂O,
respectively." **Free H₂O₂ never forms in the 4e channel** — the 4e
pathway goes through *-OOH → *-OH → H₂O directly. Once H₂O₂ leaves the
disk surface in the 2e channel, no surface reaction consumes it.

**Implications for our model:** our current solver uses a sequential
two-step BV reaction set:

```
R_0:  O₂ + 2H⁺ + 2e⁻ → H₂O₂           ✓ matches the paper's 2e
R_1:  H₂O₂ + 2H⁺ + 2e⁻ → 2H₂O         ✗ NOT in the paper's model
```

Our R_1 is "homogeneous H₂O₂ reduction at the cathode boundary after it's
already produced." This is a different physical process from the paper's
4e pathway. The R_0/R_1 lock-in we diagnosed in
`docs/mangan_p15_comparison_summary.md` is a model-structural artifact,
not a physical phenomenon present in the actual experiment. Setting
`k0_R2 → 0` in our model effectively converts our sequential model into
a parallel-2e-only model, but loses the 4e contribution to total disk
current.

The correct fix is to **replace R_1 (sequential peroxide reduction) with
a separate parallel R_4e** that consumes O₂ + 4H⁺ + 4e⁻ → 2H₂O directly.
Both reactions consume O₂ but they are independent — there is no shared
H₂O₂ intermediate at the surface.

The deck's "Peroxide Current Density" on page 15 is therefore best
interpreted as the **gross 2e-pathway production current**
(2F · R_2e), not as the net (R_0 − R_1) that our existing
`peroxide_current` observable computes.

## Local pH dynamics — the central claim of the paper

The paper's central thesis (abstract; §3.1):

> "Local pH more strongly dictates changes in mechanisms and performance,
> while the cation identity modulates local pH to influence these trends."

Specific findings from Figure 1B and §3.1:

- **Bulk pH 4 has the largest local pH excursion** of any bulk pH in the
  2-12 range studied.
- At bulk pH 4 with K⁺, **local pH shifts from 4 → ~8-9** at disk current
  density of ≈ 3.25 mA/cm². Roughly 4 pH units of local alkalinization
  under modest current.
- Local pH evolves nearly linearly with disk current density at bulk
  pH 4 across the 0 → 3.25 mA/cm² range.
- Bulk pH 6 also shows large local pH change but onset is at lower
  current.
- Bulk pH 2, 10, 12 show only small local pH changes (high buffer
  capacity from H⁺ or OH⁻).

**This means:** the deck page-15 figure's behavior across V_RHE
∈ [-0.32, +0.50] V is dominantly a **local pH evolution story**, not a
fixed-electrolyte story. The peak in peroxide current at V_RHE ≈ +0.10 V
likely corresponds to the regime where local pH starts shifting alkaline
under increasing cathodic current, and the falloff toward zero at higher
V is a combination of kinetic onset and cation/local-pH-mediated
mechanism shift. To reproduce this with our solver we need to model
local pH dynamics correctly under load.

## Cation-effect mechanism (per the paper)

§1 introduction-paragraph review of cation effects, plus the abstract
claim:

- Cation identity provides **variable pH-buffering**, modulating the
  onset and magnitude of local-pH changes.
- For weakly-binding catalysts like carbon, "the influence of cations on
  ORR activity cannot be simply due to blocking effects of clusters
  formed via noncovalent interactions, but may also involve
  cation-induced modification of reaction intermediate energies which
  would further impact reaction kinetics."
- Larger hydrated cations (Li⁺ has the largest hydrated radius despite
  being the smallest bare ion) interact more strongly with adsorbed
  hydroxide intermediates.
- The deck's "cation effects" story is therefore primarily a
  **chemistry story (buffering capacity, cation-OH interactions)**, not
  purely an EDL physics story (Bikerman sterics).

This is a more nuanced mechanism than "Cs⁺ at OHP changes screening."
For our PNP-Bikerman framework, capturing it requires:

- Tracked OH⁻ as a species (or analytic equivalent) so local pH can
  swing alkaline with proton consumption.
- Cation-specific OHP localization (this is what Bikerman gives us).
- A coupling between cation OHP density and the proton/OH-balance,
  capturing the cation's "buffering" modulation.

## What we still don't have from this paper

The paper has a Results & Discussion section beyond the pages we
extracted (1-15 above) that may contain:

- The actual modeling framework (Poisson-Boltzmann, Bikerman, BV
  parameters used by the authors).
- Cation-specific selectivity numbers and Tafel slopes.
- The H₂O₂ current-density curve at pH 4 with Cs⁺ that maps to deck
  page 15 — likely as the experimental data alongside the model fit.
- Cation-radius assumptions used in any modeling.

Action item: extract pages 16-end from
`docs/Ruggiero2022_JCatal_manuscript.pdf` to capture the modeling section
and any specific selectivity numbers for the page-15 condition.
The deck pages 14, 16-18 also contain modeling details that should be
cross-referenced.

## Implications for our solver and plan

The full reassessment is in `docs/mangan_p15_comparison_summary.md`
(updated section). Short version:

**Three-piece fix (revised from the page-15 verdict's framing):**

1. **Reaction-set fix (replaces "kinetic recalibration"):** swap our
   sequential R_0 + R_1 for parallel R_2e + R_4e. This is a config /
   observable rewiring change, not a kinetic parameter tweak. The
   `peroxide_current` observable should assemble as a single-reaction
   rate (R_2e), not a difference. Once this lands, the R_0/R_1 lock-in
   is structurally gone and a peroxide curve appears.

2. **Multi-ion electrolyte upgrade (M2/M3):** Cs⁺ (analytic), SO₄²⁻
   (analytic z = -2), OH⁻ (tracked species — needed for local-pH
   alkaline excursions). Drop ClO₄⁻. Total ionic strength = 0.3 M, not
   0.1 M as we previously assumed. This unlocks the cation-mediated
   local-pH buffering story.

3. **L_eff alignment:** target ~90 µm to match the deck's empirical
   range (66-86 µm bracket the experimental peak — see deck p.16). Bare
   Levich at 1600 rpm gives δ ≈ 21 µm but the deck calibrates higher.

Sequencing: piece 1 should land first as an isolated check — it should
unblock peroxide-current production with the surrogate electrolyte (no
shape match expected). Pieces 2-3 then follow to capture the cation /
pH dynamics that produce the peak shape.

## Cross-references

- Plan: `~/.claude/plans/swirling-crunching-wren.md`
- Page-15 comparison summary: `docs/mangan_p15_comparison_summary.md`
- Existing M0 doc: `docs/m0_target_extraction.md`
- Earlier gap audit: `docs/Mangan2025_experimental_alignment.md`
- Mangan deck PDF: `docs/Mangan2025_Catalysis.pdf`
- BV reaction-rate construction:
  `Forward/bv_solver/forms_logc_muh.py:393-475`
- BV observable wiring: `Forward/bv_solver/observables.py:13-68`
- Production constants: `scripts/_bv_common.py`
- Memory entry covering the page-15 work:
  `memory/project_mangan_m0_extraction_complete.md` (will need update
  to reflect the parallel-reaction reframing)
