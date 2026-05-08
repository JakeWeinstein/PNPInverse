# Seitz/Mangan experimental data folder — deep audit (2026-05-08)

Audit of `data/EChem Reactor Modeling-Seitz-Mangan/`, the real
experimental data drop received from the Seitz/Mangan group on
2026-05-08. Catalogues setup, data files, and contradictions with
the current PNPInverse production stack.

Use as a companion to `Mangan2025_experimental_alignment.md` and
`Ruggiero2022_JCatal_source_paper.md`. This audit is
multi-document, multi-author, multi-year (2019→2025), so its
findings carry more weight than a single-paper extraction.

## TL;DR

1. **Counterion is sulfate, not perchlorate.** Every Seitz/Mangan
   document in this folder — 8+ across 2019→2025 — uses K₂SO₄ (or
   Na₂SO₄ / Cs₂SO₄ / Li₂SO₄, or 0.5 M H₂SO₄ for kinetics-only). ClO₄⁻
   is **never mentioned**. The current `DEFAULT_CLO4_BOLTZMANN_…`
   stack is structurally wrong relative to the experiment.

2. **Mechanism is parallel 2e⁻/4e⁻, not sequential R₀+R₁.** Linsey's
   2025 ACS-CATL deck (slide 13) is explicit: *"Two Butler–Volmer
   relationships: 2e⁻ (H₂O₂) & 4e⁻ (H₂O) ORR — calculated O₂
   concentration used for both reactions."* Equilibrium potentials
   are E°(2e⁻) = 0.67 V and E°(4e⁻) = 1.23 V vs RHE. The current
   model's R₁ = H₂O₂/H₂O at 1.78 V is a *different reaction*; that
   topology cannot reproduce a 4e⁻ ORR onset at 1.23 V. (Already
   flagged from the Ruggiero 2022 paper in
   `project_mangan_m0_extraction_complete`; this audit confirms
   independently from multiple group documents.)

3. **There is a parallel modeling code in this folder.**
   `Yash-Trends/Data and Plotting.zip` (last touched 2026-05-03)
   contains a 6-species PNP+BV solver with explicit
   [H⁺, K⁺, OH⁻, SO₄²⁻, O₂, H₂O₂] and per-point fits to Cs⁺ pH 4
   RRDE. Cross-validation reference for the PNPInverse stack;
   profiles can be diff'd point-for-point.

4. **The actual LSV data Yash is fitting against is missing.** His
   `plotting.ipynb` reads `Tafel slope analysis cation-pH-Li-K-Cs.xlsx`
   which is **not present** in this folder. Request from the
   Mangan/Seitz team.

5. **C_O₂ provenance is weak.** Brianna 2019 spreadsheet and the
   MATLAB BV reference both use 5 mol/m³ (saturation @ 20 °C);
   Linsey's 2020 modeling slides use 0.5 mol/m³. **No primary-source
   document in this folder supports the current 1.2 mol/m³.** That
   value lives in the M3a.0 Mangan-page-15 audit only.

## Folder layout

```
data/EChem Reactor Modeling-Seitz-Mangan/
├── Articles/                     # Background literature (Norskov,
│                                 # Bohra, Singh, Lewis, Kocha, etc.)
├── Books & Lectures/             # Bard & Faulkner, course slides
├── Brianna/                      # Brianna Ruggiero's experimental data
│   ├── 0,1M K2SO4 data 8-15-19.xlsx              ★ raw RRDE LSV
│   ├── 20201024 CP Experiment Data-Code/         ★ CP holds + cation comparison
│   │   ├── CP_data.csv
│   │   ├── {K2,Cs2,Na2,Li2}SO4_10-9-20.mat
│   │   └── Summary Data-Error.xlsx               ★ averaged across cycles
│   └── (notes, PDFs, kinetics derivations)
├── CESR_Report_2020/2022_Seitz_Mangan*.docx      Year-end reports
├── CESR_Seed_Proposal_*.pdf                      2019 funding proposal
├── Jithin/                       # Early Mangan-side modeling notes
├── Linsey/                       # Linsey Seitz's modeling + decks
│   ├── 20200407_Electrochemical Double Layer Modeling_LSeitz.pdf  ★ slab geometry
│   └── ButlerVolmer MATLAB/      ★ reference BV implementation
│       ├── ButlerVolmerKinetics.m
│       ├── ButlerVolmerFitting.m
│       └── Brianna_ORR_Data.mat (= ORR_K2SO4_pH6_cyc1)
├── MES_Analysis_Proposal/        Empty / unrelated
├── NSF-proposal/                 NSF DAC reactive-capture proposal
├── Parameters_Seitz_Mangan.xlsx  ★ species mobilities, ICs, voltage ranges
├── Reaction modeling overleaf documents/         Jithin's 2020 PNP write-ups
├── Trienens_Report_2025/         ★ most recent narrative
│   └── 20250818-ACS-CATL-EChem Rxn Enviro for ORR-LSeitz.pptx  ★ load-bearing slides
└── Yash-Trends/                  ★ parallel modeling code + plots
    ├── Data and Plotting.zip     ★ 200 sim .npy files + plotting.ipynb
    ├── Results.zip               Sweep result SVGs
    └── *.png/*.svg               Trend plots
```
★ = highest-signal files for the PNPInverse audit.

## Experimental setup (consistent across documents)

| Parameter | Value | Source |
|---|---|---|
| Catalyst | CMK-3 carbon-based (Brianna's RRDE); Pt in MATLAB BV demo | `Brianna Research Intro ORR.pptx`, MATLAB |
| Counter / reference | Pt counter; reference electrode separate (V_OCP shift consistent with Hg/HgSO₄ + Ag/AgCl conversions) | Brianna deck |
| Disk area | **0.196 cm²** (5 mm-class disk; geometric 19.63 mm² noted in cycle headers) | `0,1M K2SO4 data 8-15-19.xlsx` |
| Ring area | **0.11 cm²** | same |
| Collection efficiency N | **0.224** | Brianna deck slide 13; Linsey 2020 slide 13; matches CLAUDE.md Ruggiero callout |
| Rotation | **1600 rpm** | Brianna deck; Linsey 2025 deck |
| Electrolyte (modeling) | **0.1 M K₂SO₄ family**: K⁺/Na⁺/Cs⁺/Li⁺ as cation; SO₄²⁻ as anion. Total cation 0.2 M, [SO₄²⁻] = 0.1 M | every doc |
| Brianna RRDE base | **0.5 M H₂SO₄** | Brianna Research Intro deck |
| Ionic strength | I ≈ 0.3 M | Linsey 2025 deck |
| pH probed (RRDE LSV) | **1.65 / 2.35 / 3.42 / 4.21 / 5.21 / 6.39** | `0,1M K2SO4 data 8-15-19.xlsx` |
| pH probed (modeling slides) | 1, 3, 5 | Linsey 2020 slide 3 |
| Disk LSV range | 1.1 → 0.05 V vs RHE | `Parameters_Seitz_Mangan.xlsx` |
| Ring potential | held at +1.2 V vs RHE | same |
| Bulk [O₂] | **5 mol/m³** (saturation, 20 °C) — Brianna 2019, MATLAB; **0.5 mol/m³** — Linsey 2020 modeling | spreadsheet, MATLAB, Linsey deck |
| 2e⁻ ORR E° | **0.67 V vs RHE** | Brianna deck; Linsey 2020 slide 10; Brianna 20200303 mass-action PDF |
| 4e⁻ ORR E° | **1.23 V vs RHE** | Linsey 2025 deck; Norskov–Viswanathan 2012 article in folder |
| Slab thickness L | ≈ 1 µm | Jithin reaction-modeling notes |
| Mobility u_i | ≈ 10⁻⁷ m²/(V·s) | Jithin notes |
| Diffusivity D_i | ≈ 10⁻⁹ m²/s | Jithin notes |
| ε_r,bulk | 80.1 (Booth) | Trienens 2023 proposal |
| ε_r,surface | ~6 (Aim 3 — variable permittivity flagged future work) | Trienens 2023 proposal |
| Cation pKa near cathode | Li⁺ 13.16 / Na⁺ 11.44 / K⁺ 8.49 / Cs⁺ 4.32 | Linsey 2025 deck slide 13 |
| Cation hydrated radius (Å) | Li⁺ 3.4 / Na⁺ 2.8 / K⁺ 2.3 / Cs⁺ 2.2 | Linsey 2025 deck slide 13 |

## Actual data we're trying to match

### Direct measurements present in the folder

- **`Brianna/0,1M K2SO4 data 8-15-19.xlsx`** — full RRDE LSV at 6 pH
  values, both cycles. Columns per pH: `E_disk (V vs RHE)`,
  `j_disk (mA/cm²)`, `j_ring (mA/cm²)`, `H₂O₂%`, `n_e`,
  `Overpotential`. ~1054 rows per pH. **Highest-density direct-match
  dataset for the model.**

- **`Brianna/20201024 CP Experiment Data-Code/`** — chronoamperometry
  (CP) holds at −0.02, −0.2, −0.4, −0.6, −0.65 V (vs RHE) for 3600 s,
  both cycles, across K₂SO₄ / Na₂SO₄ / Cs₂SO₄ / Li₂SO₄ at pH 2 / 4 / 6.
  - `*_10-9-20.mat` files store raw waveforms (`AllData_disk` /
    `AllData_ring` cell arrays, ~20k samples per hold).
  - `Summary Data-Error.xlsx` aggregates across 3 cycles per
    (cation, pH, hold-V) — **the error bars are here**.
  - `CP_data.csv` is the condensed (cation, pH, hold-V) → (Disk CP,
    Ring OCP, Disk Potential) table for K and Cs at pH 2 / 4 / 6.

- **`Linsey/ButlerVolmer MATLAB/Brianna_ORR_Data.mat`** —
  `ORR_K2SO4_pH6_cyc1` (1047 × 2: V_RHE, j_disk). Single curve.

### Curated/derived dataset Yash is fitting against

`Yash-Trends/.../plotting.ipynb` reads the **Cs⁺ pH 4** column from
`Tafel slope analysis cation-pH-Li-K-Cs.xlsx`. Conversions used:

```python
j_H2O2 = j_ring * 0.11 / (0.224 * 0.196)   # ring → disk-basis H₂O₂ partial current
V_RHE  = v_sim + 0.47 + 0.197 + 0.059 * pH  # = 0.903 V at pH 4; v_sim is in raw potentiostat coords
```

**The xlsx itself is NOT in this folder** — request from the
Mangan/Seitz team. Without it we're matching against a derived
figure, not the measurement.

## Yash-Trends parallel modeling code

`Yash-Trends/Data and Plotting.zip` (extract first; 200 sim files +
`plotting.ipynb`).

**Species set:** [H⁺, K⁺, OH⁻, SO₄²⁻, O₂, H₂O₂] — 6 dynamic species
explicit, **no Bikerman analytic-counterion shortcut**.

**Per `.npy` (loaded with `allow_pickle=True`, `.item()`):**

| Key | Meaning |
|---|---|
| `estimated_voltage` | scalar; v_sim — needs `+V_OCP` to plot vs RHE |
| `current_density` | scalar; mA/cm² |
| `selectivity` | scalar; H₂O₂ % |
| `conc_H+`, `conc_K+`, `conc_OH-`, `conc_SO42-`, `conc_O2`, `conc_H2O2` | 121-pt species profiles |
| `psi`, `x`, `s` | 121-pt potential and grid |
| `bulk_concentrations`, `charges`, `volumes` | 6-element arrays |
| `size_factor`, `L_Stern`, `c_ratio` | scalars |
| `error`, `error_without_nBV` | residual diagnostics |

**Best-fit parameter set (folder-name encoding):**
`0.16_0.5e-9_0.065_4e-6_6e-6` — `(size_factor=0.16, L_Stern=0.5 nm,
?, ?, L_bulk?)`. The `best_fit_base/` folder in the zip uses these
parameters and sweeps over 200 voltages from estimated_V ≈ −1.245
to −0.105.

**Use:** When validating PNPInverse against experimental Cs⁺ pH 4
RRDE, cross-check Yash's sim outputs at matched (V_RHE, pH, cation).
The conc/psi profile diffs localize whether the gap is sulfate vs
perchlorate (EDL-side), parallel-2e/4e vs sequential
(reaction-side), or both.

## Reference MATLAB BV implementation

`Linsey/ButlerVolmer MATLAB/ButlerVolmerKinetics.m`:

```matlab
V_ORR_2e   = 0.695        % V vs RHE (note: 0.695 here, vs 0.67 in slides)
n_ORR_2e   = 2
J0_ORR_2e  = 1            % A/m² = 0.1 mA/cm²
alpha_ORR_2e = 0.1
C_O2_bulk  = 5            % mol/m³, saturation @ 25 °C

eta = abs(V_RHE − V_ORR_2e)
J = J0 * (C_O2_0/C_O2_bulk) * exp(α·F·η·n / RT)   % cathodic-only single exp
```

Reference (commented-out) 4e⁻ branch: `J0=0.09 mA/cm², α=0.056, n=4`.
Form is unidirectional Tafel (single-exponential), not symmetric BV.

## Contradictions with the current PNPInverse stack

| Current model | Documents say | Severity | Action |
|---|---|---|---|
| **ClO₄⁻ counterion (analytic Bikerman)** | Universal SO₄²⁻ + explicit K⁺/Na⁺/Cs⁺/Li⁺; ClO₄⁻ never mentioned | **HIGH** | Structural change: divalent anion + cation chemistry. Yash's code already does this explicitly |
| **Sequential R₀ (O₂→H₂O₂ at 0.68 V) + R₁ (H₂O₂→H₂O at 1.78 V)** | Parallel 2e⁻ (0.67 V) + 4e⁻ (1.23 V) with shared O₂ consumption | **HIGH** | The model's R₁ is a different reaction than the experimental 4e⁻ couple. Replace with parallel topology. Already flagged in `project_mangan_m0_extraction_complete`; this audit reaffirms with multi-source confirmation |
| **Fixed pH 4** | KIE > 1 at low pH; Tafel slope shifts; selectivity 19→77 % as pH 1→6.4 | MEDIUM | Model captures one slice of pH-dependent mechanism — not contradictory at pH 4 specifically, but limits transferability |
| **C_O₂ = 1.2 mol/m³** (post-2026-05-07 migration) | Older Brianna spreadsheet + MATLAB: 5 mol/m³; Linsey 2020 slides: 0.5 mol/m³. **No document supports 1.2** | MEDIUM | The M3a.0 anchor is the only justification; needs paper-trail link to Mangan/Ruggiero or revert |
| **E°(2e⁻) = 0.68 V** | All docs say 0.67 V (one MATLAB file says 0.695) | LOW (10–25 mV) | Cite-able mismatch |
| **Symmetric Bikerman (single a_i)** | Cation hydrated radii are asymmetric (3.4 / 2.8 / 2.3 / 2.2 Å); Trienens 2023 Aim 2 explicitly flags need for asymmetric charged-hard-spheres | MEDIUM | Known limitation |
| **Constant ε_r ≈ 80** | Trienens 2023 Aim 3: ε ≈ 6 near surface; field-dependent | MEDIUM | Known limitation |
| **No SO₄²⁻ speciation** | At low pH, HSO₄⁻ ↔ SO₄²⁻ matters (pKa₂ ≈ 1.99) | NEW | Unmodeled physics if we move to sulfate |

## Recommended actions

1. **Get `Tafel slope analysis cation-pH-Li-K-Cs.xlsx` from the
   Mangan/Seitz team.** Without it we're matching against a derived
   figure rather than the measurement.

2. **Cross-validate against Yash's solver outputs.** Diff the
   conc/psi profiles at matched (V_RHE, pH) between PNPInverse and
   Yash's 6-species code. The divergences localize whether the
   model gap is sulfate, parallel-2e/4e, or both.

3. **Topology change (parallel 2e/4e) is the single
   highest-impact correction.** R₀(0.67 V) + an independent 4e⁻
   branch at 1.23 V — sharing O₂ — replaces the current sequential
   R₀+R₁(1.78 V).

4. **Anion change (ClO₄⁻ → SO₄²⁻)** is a structural physics change,
   not a parameter tweak — divalent counterion + possible HSO₄⁻
   speciation.

5. **C_O₂ provenance gap.** The current 1.2 mol/m³ has no direct
   primary-source support in this folder; either link to a specific
   Mangan/Ruggiero figure caption, or revert to 0.5 (Linsey 2020) /
   5 (saturation) and document the choice.

## Source files

- `data/EChem Reactor Modeling-Seitz-Mangan/Parameters_Seitz_Mangan.xlsx`
- `data/EChem Reactor Modeling-Seitz-Mangan/Brianna/0,1M K2SO4 data 8-15-19.xlsx`
- `data/EChem Reactor Modeling-Seitz-Mangan/Brianna/20201024 CP Experiment Data-Code/Summary Data-Error.xlsx`
- `data/EChem Reactor Modeling-Seitz-Mangan/Brianna/20201024 CP Experiment Data-Code/CP_data.csv`
- `data/EChem Reactor Modeling-Seitz-Mangan/Brianna/20201024 CP Experiment Data-Code/{K2,Cs2,Na2,Li2}SO4_10-9-20.mat`
- `data/EChem Reactor Modeling-Seitz-Mangan/Linsey/ButlerVolmer MATLAB/ButlerVolmerKinetics.m`
- `data/EChem Reactor Modeling-Seitz-Mangan/Linsey/ButlerVolmer MATLAB/Brianna_ORR_Data.mat`
- `data/EChem Reactor Modeling-Seitz-Mangan/Linsey/20200407_Electrochemical Double Layer Modeling_LSeitz.pdf`
- `data/EChem Reactor Modeling-Seitz-Mangan/Trienens_Report_2025/Trienens_Report_2025_Seitz_Mangan.docx`
- `data/EChem Reactor Modeling-Seitz-Mangan/Trienens_Report_2025/20250818-ACS-CATL-EChem Rxn Enviro for ORR-LSeitz.pptx`
- `data/EChem Reactor Modeling-Seitz-Mangan/Trienens_Report_2025/Mangan_SeitzProposal 1.pdf`
- `data/EChem Reactor Modeling-Seitz-Mangan/CESR_Seed_Proposal_Mangan_Seitz_final.pdf`
- `data/EChem Reactor Modeling-Seitz-Mangan/CESR_Report_{2020,2022}_Seitz_Mangan*.docx`
- `data/EChem Reactor Modeling-Seitz-Mangan/Brianna/Brianna Research Intro ORR.pptx`
- `data/EChem Reactor Modeling-Seitz-Mangan/Brianna/literature-notes.pptx`
- `data/EChem Reactor Modeling-Seitz-Mangan/Reaction modeling overleaf documents/reaction_modeling_April22.pdf`
- `data/EChem Reactor Modeling-Seitz-Mangan/Yash-Trends/Data and Plotting.zip`
- `data/EChem Reactor Modeling-Seitz-Mangan/Yash-Trends/Results.zip`
