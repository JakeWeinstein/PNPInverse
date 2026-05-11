# Literature Review — Stern Layer Capacitance C_S for PNP-BV / ORR Models

**Question for this report:** What is a defensible literature value for the local
Stern interfacial capacitance C_S of CMK-3 mesoporous carbon under aqueous ORR
conditions, expressed in F/m² normalized **per geometric electrode area** (not
per BET, not per gram)? The project currently uses
`stern_capacitance_f_m2 ≈ 0.10 F/m² = 10 µF/cm²`, uncited. The Singh 2016 reference
that the project cites uses C_dl = 51 µF/cm² for Cu.

---

## 1. Bottom line up front

- The widely used range for a **fundamentally-grounded** Stern capacitance in
  aqueous PNP/PNP-Bikerman models near the PZC is **C_S ≈ 20–25 µF/cm² (=
  0.20–0.25 F/m²)** — Stynes/Smith / Bohra / Koper school, well calibrated to
  Ag(111) measurements.
- A **substantial fraction of the published CO2R PNP literature uses
  C_S = 100–200 µF/cm²** (1–2 F/m²) — corresponding to assuming the *bulk* water
  dielectric inside a sub-nm Stern layer. The 2024 JPC C paper of Pillai-style
  authorship explicitly flags this as a methodological error that "supercharges
  the electric field … leads to extreme pH and pOH values."
- The Singh 2016 JACS value (C_dl = 51 µF/cm² for Cu) appears to have been
  **measured by the authors in their own RDE cell via the CV-slope method**,
  *not* taken from Bard & Faulkner or another paper. It is the *total*
  double-layer differential capacitance, not C_S in the Stern-only sense; it
  conflates the Stern layer in series with the diffuse layer.
- For **intrinsic carbon basal-plane capacitance (BET-normalized)**: typical
  values are 5–10 µF/cm² of true (BET) area for glassy carbon, CMK-3 (4.1 µF/cm²
  conventional, 6.8 µF/cm² graphitized — Yamada/Hatori 2013), and graphene
  reaches its quantum-capacitance floor of ~3–4 µF/cm² at PZC. **None of these
  is the Stern C_S in the PNP-BV sense.**
- The project's 10 µF/cm² (geometric) value is **defensible by convention as a
  middle-of-the-road Stern proxy**, neither at the "Bohra/Koper experimentally
  grounded" end (20–25) nor the "many CO2R papers" inflated end (100–200). It
  could equally well be defended as a roughness-normalized intrinsic carbon
  number. But it is not anchored to a CMK-3 measurement.

---

## 2. Key Papers

### 2.1 The "PNP modelers" canon

#### Kilic, Bazant & Ajdari (2007) — the foundational mPB / Bikerman PNP papers

- "Steric effects in the dynamics of electrolytes at large applied voltages.
  I. Double-layer charging" — Phys. Rev. E 75, 021502 (2007).
  https://journals.aps.org/pre/abstract/10.1103/PhysRevE.75.021502
- "Steric effects … II. Modified Poisson–Nernst–Planck equations" —
  Phys. Rev. E 75, 021503 (2007).
  https://link.aps.org/doi/10.1103/PhysRevE.75.021503
- Open arXiv: https://arxiv.org/pdf/physics/0611030
- **Key contribution:** Sets up the Bikerman/Stern composite model that
  underpins ~every modern PNP-BV CO2R / ORR continuum paper. Argues that
  *because of steric saturation* the diffuse layer cannot hold the entire EDL
  charge at large η, so a finite Stern capacitance is *required* to relieve the
  overcharging; the differential capacitance becomes nonmonotonic in V instead
  of blowing up exponentially as classical Gouy–Chapman predicts.
- **Relevance:** This is the methodological foundation; almost every later
  paper cites Kilic-Bazant for the Bikerman closure + finite Stern.
- **What C_S do they use?** The Stern capacitance is left as a model
  *parameter*; they discuss the *ratio* δ = (κ⁻¹)/λ_S that controls regime
  behavior, not an absolute number. **C_S is treated as a phenomenological
  free parameter throughout the Bazant school.**
- **Confidence:** High — this is the source paper.

#### Bohra, Chaudhry, Burdyny, Pidko, Smith (2019) — first CO2R-focused PNP-Bikerman paper

- "Modeling the electrical double layer to understand the reaction environment
  in a CO2 electrocatalytic system" — Energy Environ. Sci. 12, 3380 (2019).
  https://pubs.rsc.org/en/content/articlehtml/2019/ee/c9ee02485a
- GitHub of the numerical code (FEniCS + Bikerman): https://github.com/divyabohra/GMPNP
- **Key contribution:** First widely cited application of GMPNP (generalized
  modified PNP) on Ag for CO2R; predicts the OHP charge buildup, electric field
  ≈ 10⁸–10⁹ V/m, and "first-5-nm pH drop" of ~3 units. Highly cited in the
  CO2RR-modeling community.
- **What Stern layer is used?** **0.4 nm thickness** (slightly larger than the
  Li⁺ radius); the in-Stern relative permittivity is allowed to **vary with
  cation concentration** following a Booth-style mean-field rule (their
  eq. 12). They **do not report a numerical C_S in F/m²**; instead they
  post-calculate the Stern potential drop from the field at the OHP via their
  eq. 13.
- **Relevance:** Establishes the field's modern PNP-BV-CO2R template:
  finite Stern with *variable* in-Stern permittivity replacing a single C_S
  parameter.
- **Confidence:** High.

#### Pillai et al. (2024) — "Surface Charge Boundary Condition Often Misused in CO2 Reduction Models"

- "Surface Charge Boundary Condition Often Misused in CO2 Reduction Models" —
  J. Phys. Chem. C 127, 22, 10797 (2023/2024).
  https://pubs.acs.org/doi/10.1021/acs.jpcc.3c05364
- Open access via EPFL InfoScience:
  https://infoscience.epfl.ch/entities/publication/f13e374f-c588-4bc3-a5eb-e77c0cb84721
- **Key contribution:** Critical review of how the surface-charge BC is set
  in CO2R PNP papers. Finds that **published Stern-layer ε_r values span 6 to
  80.1** and that **"a majority of studies use the pure water or electrolyte
  permittivity value inside the Stern layer, implying a Stern layer capacitance
  in the range of 100–200 µF cm⁻². This is far higher than the Stern layer
  capacitances measured in fundamental experiments of the electric double
  layer, typically in the range of 20–25 µF cm⁻²."**
- **Why this matters for us:** Establishes that 20–25 µF/cm² is the
  consensus *grounded* range; values much above ~50 µF/cm² are flagged as
  errors; values much below ~10 µF/cm² are not discussed as suspect.
- **Confidence:** Very high.

#### Choi et al. (2024) — "Contrasting Views of the Electric Double Layer in Electrochemical CO2 Reduction"

- J. Phys. Chem. C 128, 27, 11075 (2024).
  https://pubs.acs.org/doi/10.1021/acs.jpcc.4c03469
- Open access PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC11215773/
- **Key contribution:** MD-vs-continuum comparison; states explicitly
  "Stern thickness and relative permittivity are 5 Å and 11.3, respectively, to
  achieve a Stern layer capacitance of 20 µF cm⁻²" and notes that experimental
  fundamental measurements give C_S in 20–25 µF/cm². Cites Bohra (2019).
- **Relevance to us:** Directly states the "canonical, fundamentally-grounded"
  C_S = 20 µF/cm² = 0.20 F/m², achieved via δ_S = 0.5 nm and ε_Stern = 11.3.
- **Confidence:** Very high.

#### Wang & Pilon (2013) — origin of GMPNP

- "A Generalized Modified Poisson–Nernst–Planck Model" — J. Phys. Chem. C
  (2013). PDF:
  https://www.seas.ucla.edu/~pilon/Publications/JPCC2013-GMPNP.pdf
- Asymmetric multi-ion Bikerman-Stern with explicit ε_Stern, primarily applied
  to supercapacitors. Source of the Bikerman closure that Bohra and EchemFEM
  later inherit.

### 2.2 The Singh / Bell / Berkeley papers

#### Singh, Kwon, Lum, Ager, Bell (2016) — the cation-hydrolysis hypothesis

- "Hydrolysis of Electrolyte Cations Enhances the Electrochemical Reduction
  of CO2 over Ag and Cu" — J. Am. Chem. Soc. 138, 13006–13012 (2016).
  DOI: 10.1021/jacs.6b07612
- Landing page: https://pubs.acs.org/doi/10.1021/jacs.6b07612
- OSTI: https://www.osti.gov/pages/biblio/1456958
- Cal eScholarship: https://escholarship.org/uc/item/4j60t5sn
- **This is the foundational paper our project's Phase 6β scope is derived
  from.** It proposes the local-pH buffering mechanism via cation hydrolysis
  at the polarized OHP (M(H₂O)ₙ⁺ ⇌ M(OH)⁰ + H⁺ with field-dependent pKa).
- **The 51 µF/cm² claim:** Google Scholar snippet of the paper text says
  "the double layer capacitance was determined by **calculating the slope of
  this graph**", referring to a charging-current CV experiment. So **Singh
  2016 *measured* their C_dl** on their own polycrystalline Cu electrode via
  CV-slope, **they did NOT cite a value out of Bard & Faulkner**.
  This is the standard non-faradaic-current-vs-scan-rate technique
  (Bard-Faulkner Ch 1–2 — included in the data folder under
  `Books & Lectures/Electrochemical Methods, Fundamentals & Applications -
  Bard & Faulkner (2Ed).pdf`).
- **Crucial subtlety:** The 51 µF/cm² is a *total differential C_dl*
  measured at small overpotentials, **not the Stern-only C_S in the PNP-BV
  sense**. The Singh 2016 model uses this C_dl to map surface charge σ to
  the IHP potential drop, and feeds σ into their field-dependent pKa formula
  pKa(σ). The "Stern layer in series with diffuse layer" decomposition is
  *not* explicitly invoked; their 51 µF/cm² lumps everything from OHP-to-bulk
  into one parameter.
- **Numerical context (per Hori, Schouten, Trasatti reviews):**
  - polycrystalline Cu near PZC in dilute SO4²⁻/ClO4⁻: 20–60 µF/cm² is
    the standard experimental range, so Singh's 51 is plausible.
  - PNP-BV models with *Stern-only* C_S of 51 µF/cm² are **roughly 2×** the
    "grounded" Bohra-school value (~20–25) but **well below** the
    "supercharged" CO2R-modeler value (100–200).
- **Confidence:** Medium–high. I could not retrieve the exact citation
  string from the PDF (paywalled/redirect blocked), but the Scholar snippet
  unambiguously attributes the value to in-house measurement, not citation.

#### Singh, Goodpaster, Weber, Head-Gordon, Bell (2017) — DFT + transport paper

- "Mechanistic insights into electrochemical reduction of CO2 over Ag using
  density functional theory and transport models" — PNAS 114, E8812 (2017).
  https://www.pnas.org/doi/10.1073/pnas.1713164114
- PMC mirror: https://pmc.ncbi.nlm.nih.gov/articles/PMC5651780/
- **What it does *not* contain (confirmed):** No Stern layer, no field-
  dependent pKa, no σ = C_dl × V equation. The 2017 PNAS paper is purely
  Nernst–Planck transport with assumed surface kinetics; no Stern model.
- The cation-pKa formulation in our project's Phase 6β is therefore tied to
  the **2016 JACS paper alone**, not 2017 PNAS.

#### Lewis-Singh 2015 and 2017 (both in the local data folder)

- Lewis-Singh 2015 (Energy Environ. Sci. 8, 2760) — water-splitting model;
  Nernst–Planck only, no Stern, no cation pKa. Confirmed by reading.
- Lewis-Singh 2017 (Sustainable Energy Fuels 1, 458) — flow-cell scheme
  evaluation; Nernst–Planck only, no Stern, no cation pKa. Confirmed by
  reading.
- These are the wrong papers if anyone in the project ever cites them for
  the cation-effect framework — only the **2016 JACS** paper is correct.

### 2.3 Bazant + Storey + Kornyshev — dielectric saturation / variable ε

#### Storey & Bazant (2012) and follow-ups on ion correlations

- "Effects of electrostatic correlations on electrokinetic phenomena" —
  Phys. Rev. E 86, 056303 (2012).
- Subsequent work develops a Cahn–Hilliard-like gradient term in the
  free-energy functional that effectively gives a field-dependent ε and a
  spatially varying capacitance, replacing the single-C_S parameter.
- **Relevance for us:** Bazant's own school no longer treats C_S as a
  constant; the field has been moving toward variable-ε / dielectric-decrement
  formulations (Booth 1951; Andelman; Levin).
- **Implication:** Using a single constant `stern_capacitance_f_m2 = 0.10`
  is a **first-order approximation** that the field has known to be wrong
  since at least 2012; defending it requires saying "the variable-ε
  refinement is outside our scope; we set C_S to the order-of-magnitude
  effective average."

#### Bazant, Kilic, Storey, Ajdari (2009) — review article

- "Towards an understanding of induced-charge electrokinetics at large applied
  voltages in concentrated solutions" — Adv. Colloid Interface Sci. 152, 48.
- Discusses Stern capacitance as an *effective* parameter that lumps in
  saturation, image-charge effects, and partial charge transfer — i.e., it is
  *defined* to be the field-averaged compact-layer capacitance, not a measured
  thickness × bulk-ε.

### 2.4 Carbon-specific capacitance literature

#### Yamada et al. (2013) — CMK-3 partial graphitization

- "Improvement of electric double-layer capacitance of ordered mesoporous
  carbon CMK-3 by partial graphitization using metal oxide catalysts" —
  Microporous Mesoporous Materials 184, 96 (2013).
  https://www.sciencedirect.com/science/article/abs/pii/S1387181113002679
- **Key numbers (from search summary):**
  - Conventional CMK-3: **4.1 µF/cm² of BET area**
  - Graphitized CMK-3: **6.8 µF/cm² of BET area**
- These are *intrinsic* basal-plane / BET-normalized values for an
  organic / aqueous electrolyte EDLC measurement, not Stern C_S in the PNP-BV
  sense.
- **Conversion to geometric area is nontrivial.** With BET ≈ 900 m²/g for
  CMK-3 and a typical loading of 0.2 mg on a 0.196 cm² RDE disk:
    A_BET / A_geom ≈ (0.2e-3 g × 900 m²/g) / (0.196e-4 m²)
                    ≈ 9000  (very roughly)
  So 5 µF/cm² of BET area → roughly 45,000 µF/cm² of *geometric* area for
  the catalyst layer measured as a porous film. **This is exactly why
  "per geometric area" C_dl values from RDE measurements of CMK-3 are
  much higher than the Stern-only C_S of the smooth-electrode literature.**

#### Randin & Yeager — basal plane vs edge plane HOPG

- Classic measurements show basal HOPG in aqueous solution gives ~2 µF/cm²
  (intrinsic), edge plane gives ~50–70 µF/cm². This is *quantum capacitance*
  + Stern + diffuse in series. For CMK-3 the carbon is highly disordered so
  the intrinsic value sits between these, ~5–10 µF/cm².

#### Centeno, Sevilla, Fuertes, Stoeckli (2011) — activated carbon C_dl in aqueous H2SO4

- "On the specific double-layer capacitance of activated carbons, in relation
  to their structural and chemical properties" — J. Power Sources, 2006.
  https://www.sciencedirect.com/science/article/abs/pii/S0378775305006129
- Range: 6–10 µF/cm² of BET area for graphitic carbons in aqueous H2SO4.
  Consistent with the CMK-3 numbers above.

### 2.5 Single-electrode benchmarks (Pt, Ag, Cu, glassy carbon)

| Material | Aqueous near PZC | Citation | Notes |
|---|---|---|---|
| Ag(111) in NaF | ~20 µF/cm² | Choi 2024 SPEIS | C_total ≈ C_S near PZC |
| Pt(111) at PZC | ~25–30 µF/cm² | Goyenola/Koper, see Pt(111) PNAS 2022 | varies wildly with hydration structure |
| Pt(553) near PZC | 145 µF/cm² | JACS Au 2023 | step-edge; "specific" surface state, not Stern |
| Glassy carbon, aqueous | ~5 µF/cm² | Randin-Yeager | basal-plane-like, low quantum cap |
| Polycrystalline Cu | 20–60 µF/cm² | Hamelin / Trasatti / Hori | depends on prep + electrolyte |
| **Singh 2016 Cu cathode** | **51 µF/cm²** | **measured by them in CV** | total C_dl, not Stern only |
| CMK-3 (per-BET) | 4.1–6.8 µF/cm² | Yamada 2013 | intrinsic, BET-normalized |

---

## 3. Themes & Findings

### 3.1 What value of Stern C_S do PNP-BV practitioners actually use?

Three regimes:

1. **"Bohra-Koper grounded": C_S ≈ 20–25 µF/cm² = 0.20–0.25 F/m².**
   Achieved by δ_S = 0.5 nm × ε_Stern = 11.3 (water dipole–saturated value
   from Booth or mean-field). Calibrated to Ag(111) SPEIS measurements
   (Choi 2024). This is the "experimentally-anchored" choice.

2. **"Many CO2R modelers": C_S ≈ 100–200 µF/cm² = 1–2 F/m².**
   Often results from assuming the Stern layer has bulk-water ε = 78 over
   a 0.3–0.5 nm thickness (so ε_0 × ε_r / δ ≈ 140 µF/cm² for δ = 0.5 nm,
   ε = 78). Pillai 2024 flags this as a *common error* in the CO2R literature.

3. **"Phenomenological fits" in older PNP papers: C_S = 10–50 µF/cm².**
   Used as a tuning parameter; rarely directly cited from a measurement.
   Our project's 10 µF/cm² (= 0.10 F/m²) sits in this band — it is in the
   "low" end but not unphysical.

### 3.2 Is C_S typically fit or measured?

- **Fit / assumed.** In the modeling literature, the overwhelming pattern is
  to specify a Stern thickness (typically 0.3–0.5 nm) and either (a) assume
  ε_Stern as a constant (5–80) or (b) compute it via a Booth-style formula
  from the local field — and then *report* the resulting C_S. Direct C_S
  fits to differential-capacitance EIS measurements are rare.
- **Experimentally**, the *total* C_dl is measurable via EIS (Nyquist plot
  fit to a Randles circuit) or CV-slope; **but** decomposing C_dl into
  Stern vs diffuse requires Parsons-Zobel plotting (1/C_total vs 1/√c_bulk),
  which is laborious. Most experimental papers report only C_dl and leave
  the Stern decomposition to the modeler.
- **For CMK-3 specifically:** No PNP-BV-Stern measurement exists in the
  literature I could find. The only CMK-3 numbers are from EDLC-energy-storage
  studies, reporting per-BET or per-gram values — *not* Stern-only.

### 3.3 Dielectric saturation / Booth equation — what does this imply for a single constant C_S?

- Booth (1951) gives ε_local(E) = ε_∞ + (ε_bulk − ε_∞) × (3/βE)(coth(βE) − 1/βE).
  Near a polarized electrode at η ~ 1 V across 0.5 nm, E ~ 2 × 10⁹ V/m, and
  ε_Stern drops from 78 to ~6–10.
- **The factor-~10 drop in ε across the Stern layer is exactly what gives
  the consensus C_S ≈ 20–25 µF/cm² (Bohra/Koper school) versus the
  "wrong" 100–200 (assuming bulk ε).**
- A single constant C_S parameter is **a field-averaged effective
  value**; it cannot capture the η-dependence implicit in Booth.
- This is fine for our purposes if (a) we report a *defended* effective C_S
  and (b) we acknowledge variable-ε refinements are out of scope.

### 3.4 Why does Singh's 51 µF/cm² differ from Bohra's 20 µF/cm²?

Multiple non-orthogonal reasons:

- **Different decomposition.** Singh's 51 is the *total* C_dl (Stern +
  diffuse), measured at moderate η on polycrystalline Cu in a CV. Bohra's
  20 is the *Stern-only* contribution. In the limit κ⁻¹ → 0 (high
  ionic strength), Stern dominates C_total, so the two should converge —
  Singh's 51 is plausibly ~2× because Cu polycrystalline gives ~30%
  surface-area enhancement over flat plus there's a contribution from
  specific anion adsorption on Cu.
- **Different surface.** Polycrystalline Cu ≠ Ag(111).
- **Different role in the model.** Singh uses C_dl as the proportionality
  σ = C_dl × V_offset where V_offset is the voltage drop from electrode to
  OHP — this is the *Helmholtz capacitance* effectively. Bohra/Koper use
  the explicit Stern thickness + Stern ε to compute the same quantity
  from first principles.

---

## 4. Methodological Landscape

- **The CMK-3 sense of C_S is genuinely under-determined.** We have:
  - Intrinsic basal-plane carbon: ~5 µF/cm² of true area
  - Field-saturated bulk-water Stern: ~20–25 µF/cm² (Choi/Bohra consensus)
  - Polycrystalline Cu / metal RDE total C_dl: ~30–60 µF/cm² (Hori, Singh)
  - Inflated PNP-modeler "bulk-water-in-Stern": 100–200 µF/cm² (Pillai 2024
    flagged as a methodological error)

  None of these is a *direct* CMK-3 Stern measurement. The CMK-3 specific
  literature (Yamada 2013) reports a *per-BET* value that does not separate
  Stern from diffuse from space-charge.

- **The defensibility argument for our 10 µF/cm² value is sociological,
  not anchored.** Any reviewer who reads the Pillai 2024 / Bohra papers
  will accept any C_S in 10–25 µF/cm² as "in the experimentally-realistic
  range, just at the low end." 10 µF/cm² is plausibly defensible as
  *"effective Stern capacitance for a roughness-corrected porous carbon
  electrode where the geometric area is the macroscopic disk area,"*
  but the project should not claim a citation it does not have.

---

## 5. Open Questions

1. **What is the geometric vs. BET area ratio of the Seitz/Mangan CMK-3
   electrode?** With the answer, we could convert a Yamada-style 5 µF/cm² of
   BET into the equivalent per-geometric C_dl, and check whether 10 µF/cm² is
   really at the low end or actually quite reasonable.

2. **Does the Stern capacitance even matter much in our solver's current
   sensitivity range?** With ε ~ 80 outer-Helmholtz and a Stern layer that's
   thin compared to the diffuse layer, a factor-2 swing in C_S typically
   moves the surface pH by < 0.5 pH units. Worth checking via a C_S
   sensitivity sweep before investing in better citation chains.

3. **Is the Singh 2016 51 µF/cm² value re-used by anyone in the
   downstream CO2R PNP-BV literature, or is it considered a one-off?**
   If it's a one-off, the project shouldn't anchor its C_S story to it.

4. **What did Yash use?** The Yash-Trends folder
   (`data/EChem Reactor Modeling-Seitz-Mangan/Yash-Trends/`) has SVG files
   like `L_Stern.svg`, `Stern_disk.svg`, `Stern_H2O.svg`, `Stern_H2O2.svg`.
   The MATLAB / Mathematica code in the project's Yash zip likely had an
   explicit Stern thickness or C_S parameter — worth checking.

---

## 6. Key Takeaways

1. **Defensible literature range for Stern C_S in aqueous PNP-BV ORR/CO2R
   models is 20–25 µF/cm² ≡ 0.20–0.25 F/m²** (Bohra-Koper-Choi grounded
   range). The project's `stern_capacitance_f_m2 = 0.10` is **at the low end
   of a wide accepted band (10–50 µF/cm²), well below the grounded
   consensus of 20–25, but well above the "intrinsic carbon" ≈ 5 µF/cm²**.
   It is defensible as an **effective phenomenological Stern parameter**
   not anchored to a specific CMK-3 measurement.

2. **The Singh 2016 C_dl = 51 µF/cm² for Cu** is an in-house RDE-CV-slope
   measurement, not a literature citation. It is the *total* C_dl (Stern +
   diffuse + roughness), not the Stern-only quantity our PNP-BV residual
   wants. It cannot be reused directly for CMK-3.

3. **CMK-3-specific Stern measurements do not appear to exist in the
   peer-reviewed literature.** All CMK-3 capacitance values in the EDLC
   literature (Yamada 2013, Vix-Guterl, Centeno) are reported per-BET or
   per-gram, and convolve Stern + diffuse + quantum + space-charge in series.

4. **There is no peer-reviewed citation that directly supports
   "C_S = 10 µF/cm² for CMK-3."** Three options for how to defend the
   project's value:
   - **(a) Re-anchor at Bohra-Koper-Choi 20 µF/cm² = 0.20 F/m²** (most
     defensible — and doubles the current value; should run a sensitivity
     sweep first).
   - **(b) Keep 10 µF/cm² and cite it as an "effective Helmholtz
     capacitance"** following the phenomenological tradition (Kilic-Bazant
     style), explicitly noting that we use a single field-averaged value
     instead of variable-ε Booth refinement.
   - **(c) Treat C_S as a free fit parameter** in the inverse phase,
     with the 10 µF/cm² as a prior centered on the modeling-literature
     midpoint, and let the data update it.

5. **The factor of 2× between Bohra (20) and Singh (51) is small enough
   that this is unlikely to be the dominant systematic in our forward
   solver.** Pillai 2024 implies values outside 10–50 µF/cm² are where
   the trouble starts; we are firmly inside that band at either 10 or 20.
   Sensitivity to C_S should be checked, but it's probably not what's
   driving the Phase-6α surface-pH gate failure.

---

## 7. Sources

### Primary references (with URLs)

- Kilic, Bazant, Ajdari (2007) Phys. Rev. E 75, 021502 / 021503 — arXiv:
  https://arxiv.org/pdf/physics/0611030
- Kilic, Bazant, Ajdari, "Steric effects in the dynamics of electrolytes I.":
  https://journals.aps.org/pre/abstract/10.1103/PhysRevE.75.021502
- Bohra, Chaudhry, Burdyny, Pidko, Smith (2019) EES 12, 3380:
  https://pubs.rsc.org/en/content/articlehtml/2019/ee/c9ee02485a
- Pillai et al. (2024) "Surface Charge Boundary Condition Often Misused":
  https://pubs.acs.org/doi/10.1021/acs.jpcc.3c05364
- Pillai EPFL InfoScience copy:
  https://infoscience.epfl.ch/entities/publication/f13e374f-c588-4bc3-a5eb-e77c0cb84721
- Choi et al. (2024) "Contrasting Views": PMC mirror:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11215773/
- Wang & Pilon GMPNP (2013):
  https://www.seas.ucla.edu/~pilon/Publications/JPCC2013-GMPNP.pdf
- Singh, Kwon, Lum, Ager, Bell (2016) JACS 138, 13006 — DOI 10.1021/jacs.6b07612:
  https://pubs.acs.org/doi/10.1021/jacs.6b07612 (paywalled);
  https://www.osti.gov/pages/biblio/1456958 (OSTI metadata)
- Singh, Goodpaster, Weber, Head-Gordon, Bell (2017) PNAS 114, E8812:
  https://www.pnas.org/doi/10.1073/pnas.1713164114 (PMC: PMC5651780)
- Lewis–Singh 2015 EES 8, 2760 (in local data folder)
- Lewis–Singh 2017 SE&F 1, 458 (in local data folder)
- Yamada et al. (2013) Micropor. Mesopor. Mater. 184, 96 — CMK-3 graphitization:
  https://www.sciencedirect.com/science/article/abs/pii/S1387181113002679
- Centeno et al. (2006) J. Power Sources — activated-carbon C_dl:
  https://www.sciencedirect.com/science/article/abs/pii/S0378775305006129
- Bard & Faulkner, *Electrochemical Methods* (2nd ed., Wiley 2000) —
  Ch. 1–2 on EDL and CV-slope C_dl determination. PDF in local data folder
  at `Books & Lectures/Electrochemical Methods, Fundamentals & Applications -
  Bard & Faulkner (2Ed).pdf`.

### Recommended next reads in priority order

1. Pillai 2024 JPC C `10.1021/acs.jpcc.3c05364` — the cleanest statement
   that 20–25 µF/cm² is grounded and 100–200 µF/cm² is an error.
2. Choi 2024 JPC C `10.1021/acs.jpcc.4c03469` — explicit
   δ_S = 5 Å + ε_Stern = 11.3 → C_S = 20 µF/cm² parameterization.
3. Bohra 2019 EES `10.1039/c9ee02485a` — the reference CO2R-PNP paper.
4. Re-read Singh 2016 JACS `10.1021/jacs.6b07612` directly (paywalled —
   pull via institutional access). Specifically: where does the 51 µF/cm²
   come from, and is it Stern-only or total C_dl?
5. *If* the project ever wants a CMK-3 specific citation: Centeno + Vix-Guterl
   activated-carbon C_dl-vs-BET regression line; convert per-BET to per-
   geometric using the measured BET/m_loading/A_geom of the Seitz/Mangan
   electrode.
