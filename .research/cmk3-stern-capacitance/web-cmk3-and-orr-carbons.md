## Defensible literature value for C_S of CMK-3 mesoporous carbon under aqueous ORR conditions

### Top-line answer

The project's current value **C_S = 0.10 F/m² (= 10 µF/cm²)** is *defensible but low* for a PNP modeling convention. The well-established PNP-modeling community convention for an aqueous metal/electrolyte interface is **C_S ≈ 0.20 F/m² (= 20 µF/cm²)**, derived from Stern thickness ≈ 5 Å and Stern permittivity ε_r ≈ 11.3. For a *carbon* electrode specifically, the literature shows a wider, lower range (intrinsic 2–10 µF/cm² per BET area for ordered carbons; ~10–25 µF/cm² for glassy carbon per geometric area). Neither Ruggiero 2022 (the source paper) nor any Seitz-group publication I could locate publishes a C_S value for CMK-3.

**Recommended for v10b lock:** **C_S = 0.20 F/m² (20 µF/cm²)** as a literature-cited mid-range value, with **0.10–0.30 F/m²** as a defensible sensitivity bracket. The 0.10 F/m² currently in production is at the *low* edge of this range; if you want to stay close to current production, **0.10 F/m² remains defensible if you cite the carbon-electrode literature (intrinsic basal-plane HOPG, glassy-carbon background, and ordered-mesoporous-carbon supercapacitor numbers below)**.

---

### Sub-topic 1: Does Ruggiero 2022 (J Catal 414:33-43) report or cite a C_S value for CMK-3?

**No.** I read all 31 pages of the manuscript at `/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/docs/papers/Ruggiero2022_JCatal_manuscript.pdf`.

- The paper purchases CMK-3 from ACS Material, prepares disk electrodes by drop-casting 2 mg CMK-3 + 197.3 µl ethanol + 2.7 µl cation-exchanged Nafion ink on a 0.196 cm² polished glassy carbon disk, **catalyst loading 0.5 mg cm⁻² geometric** (page 9).
- The only mention of capacitance is on page 11: "background current (capacitive current) was collected between 0.8V to 1.0V vs. RHE" for current subtraction. **No numerical C_dl is reported.**
- The paper does **not** report BET surface area of the CMK-3, roughness factor, or Stern/Helmholtz capacitance.
- `pdftotext`-grep for "capacit", "stern", "helmholtz", "F/cm", "µF", "BET" confirms: zero numerical capacitance value in the entire paper.

This is consistent with the project's own audit at `/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/docs/phase6/CONJECTURE_AUDIT_2026-05-09.md` line 21, which already flagged the 0.10 F/m² as "STILL UNVERIFIED, no specific C_S value cited in folder."

**Confidence:** high. Verified directly against the PDF.

---

### Sub-topic 2: CMK-3 capacitance from supercapacitor literature (per-BET normalization)

The relevant literature reports C_dl for CMK-3 in two forms: gravimetric (F/g) and per-BET-area (µF/cm²-BET). Converting either to per-geometric-area requires the catalyst-film roughness factor.

**Reported values for CMK-3 capacitance:**

| Source | Electrolyte | F/g | µF/cm²-BET | BET (m²/g) | Notes |
|---|---|---|---|---|---|
| Lota et al. via Vix-Guterl/Frackowiak (graphitization study, 2013) | Et₄NBF₄ (organic) | — | **4.1** (pristine) | ~1000 | Pristine CMK-3, per BET |
| Lota et al. via graphitization paper | Et₄NBF₄ (organic) | — | **6.8** (graphitized CMK-3-Fe) | similar | After partial graphitization |
| Lee/Singh group review of CMK-3 in 1 M H₂SO₄ | 1 M H₂SO₄ | 60–90 F/g | n/a (60/900 ≈ **6.7 µF/cm²-BET**) | ~900 | Three-electrode cell |
| Ettifri et al. "Supercapacitive Performance of CMK-3 in Neutral Aqueous Electrolyte" | Na₂SO₄ aq | 285 F/g (at 10 A/g, on Ni foam) | **~10 µF/cm²-BET** equivalent | ~1000 | Areal value referenced |
| Korean J Chem Eng (Ca(NO₃)₂ aqueous) | Ca(NO₃)₂ aq | 210 F/g | ~14 µF/cm²-BET | ~1000–1500 | Higher because of pseudo-capacitive Ca²⁺ |

**Per-BET-area for CMK-3 in aqueous = 5–15 µF/cm²-BET** (consistent across multiple sources).

- **Source:** [Improvement of electric double-layer capacitance of ordered mesoporous carbon CMK-3 by partial graphitization](https://www.sciencedirect.com/science/article/abs/pii/S1387181113002679) (Microporous Mesoporous Mater 178:171, 2013) — explicit 4.1 µF/cm²-BET pristine, 6.8 µF/cm²-BET graphitized
- **Source:** [Supercapacitive Performance of Ordered Mesoporous Carbon (CMK-3) in Neutral Aqueous Electrolyte](https://www.ijcce.ac.ir/article_27404.html) — 285 F/g at 10 A/g, areal ~10 µF/cm²-BET
- **Source:** [Supercapacitive behavior of mesoporous carbon CMK-3 in calcium nitrate aqueous electrolyte](https://link.springer.com/article/10.1007/s11814-013-0289-z) — 210 F/g
- **Source (BET range):** [ACS Materials CMK-3 datasheet](https://www.acsmaterial.com/ordered-mesoporous-carbon-cmk-3.html) — Type B "≥ 900 m²/g", pore diameter 3.8–4.0 nm, pore volume 1.2–1.5 cm³/g
- **Source (original synthesis):** Jun, Joo, Ryoo et al., *JACS* 2000, 122, 10712 ([DOI 10.1021/ja002261e](https://pubs.acs.org/doi/10.1021/ja002261e)) — first CMK-3 synthesis, BET typically 1000–1500 m²/g

**Important caveat:** these numbers are *full-double-layer* C_dl, not pure Stern (compact-layer) capacitance. In a Bouwer-Stern decomposition C_dl⁻¹ = C_Stern⁻¹ + C_diff⁻¹, the Stern part is typically larger than C_dl by a modest factor when the diffuse layer is fully developed. At high ionic strength (0.1–1 M typical for both supercapacitor measurements and the Ruggiero deck), the diffuse capacitance saturates and C_dl ≈ C_Stern within a factor of ~2.

**Confidence:** high for the per-BET numbers; medium for the equivalence to "Stern" in the PNP-model sense.

---

### Sub-topic 3: Comparable ORR-relevant carbon electrodes (per-geometric area)

Critical because the PNP model's `stern_capacitance_f_m2` is a *per-geometric-area* boundary condition, not per-BET.

| Material | C_dl (per geometric area) | Electrolyte | Source |
|---|---|---|---|
| HOPG basal plane (intrinsic, minimum) | **2.5–3 µF/cm²** | concentrated aq | [Differential capacitance study on stress-annealed pyrolytic graphite](https://www.sciencedirect.com/science/article/abs/pii/S0022072872802493) |
| HOPG basal plane (at PZC + 0.5 V) | **4.7–9.4 µF/cm²** | aq KF, NaF | [Capacitance of Basal Plane and Edge HOPG, J Phys Chem Lett 2019](https://pubs.acs.org/doi/10.1021/acs.jpclett.8b03523) |
| HOPG basal plane (intrinsic, ref value) | **6.0 µF/cm²** | aq, near minimum | quoted widely; cited from HOPG/graphene PNAS 2023 |
| Glassy carbon (background) | **10–25 µF/cm²** | aq H₂SO₄, KOH | [Electrical double layer formation on glassy carbon, EA 2021](https://www.sciencedirect.com/science/article/abs/pii/S0013468621007064); textbook ranges |
| Glassy carbon (low-end, polished, pristine) | **2–5 µF/cm²** | aq | textbook and Wikipedia consensus |
| Carbon black (Vulcan XC-72R) per atomic BET | **16 µF/cm²-BET** | 30% aq H₂SO₄ | [Frackowiak, J Appl Electrochem 2001](https://link.springer.com/article/10.1023/A:1017529920916) — quoted as `C_A,DL = 16 µF cm⁻²` per atomic surface area |
| Activated carbons, CNTs in aq NaCl | **5–10 µF/cm²-BET** (asymptotic 4–5 at SSA > 1500 m²/g) | dilute NaCl | [Nature Comm 2014 review of carbon EDLCs](https://www.nature.com/articles/ncomms4317) |
| N-doped activated graphene | 6 → 22 µF/cm²-BET (increases with N) | aq | [Nature Comm 2014 review](https://www.nature.com/articles/ncomms4317) |

For a **thin, porous catalyst-film electrode** like the Ruggiero CMK-3 ink (0.5 mg cm⁻² on a 0.196 cm² glassy carbon disk), the *geometric* C_dl is amplified by the catalyst-film roughness factor (RF). Order-of-magnitude calc:

- 0.5 mg cm⁻² × 1000 m²/g (BET) = 5 m²/cm²-geometric = RF ≈ 50,000 if the *entire* internal BET surface is electrochemically wetted and active.
- 5 µF/cm²-BET × 50,000 RF ≈ **250,000 µF/cm²-geometric = 2.5 F/cm²-geometric = 25,000 F/m²-geometric.**

That is wildly unphysical for a PNP boundary condition. **The whole-electrode C_dl in F/m²-geometric for a typical porous carbon ink is 10²–10⁴ F/m²-geometric**, which is what the supercapacitor community reports (you see 200–700 F/g × ~0.5 mg/cm² ≈ 0.1–0.4 F/cm²-geometric, which gives ~1000–4000 F/m²-geometric).

**This is the heart of the conceptual issue:** the PNP-Bikerman-BV model's `stern_capacitance_f_m2` is not the same physical quantity as the experimentally measured C_dl of the porous catalyst film. The model treats the glassy-carbon-disk interface as if it were a flat metal/electrolyte interface; `C_S` is the *intrinsic compact-layer* capacitance of one geometric cm², not a quantity that scales with BET area. The porous ink's amplification is folded into the kinetic rate constants (k₀ and α) and the effective catalyst loading separately.

**Confidence:** high.

---

### Sub-topic 4: PNP-modeling community convention (the most defensible chain for v10b)

The strongest citation chain — and the one I recommend locking — comes from the PNP-continuum-modeling literature, which is what your model actually is.

**Standard PNP value: C_S = 20 µF/cm² = 0.20 F/m².**

Derivation: Stern thickness L_S = 5 Å, Stern permittivity ε_S = 11.3 (water dielectric reduced by structural ordering at the interface). Then:

  C_S = ε_S · ε_0 / L_S = 11.3 × 8.854×10⁻¹² F/m / 5×10⁻¹⁰ m = 0.200 F/m² = 20 µF/cm².

- **Source (definitive recent review):** [Bohra et al., "Contrasting Views of the Electric Double Layer in Electrochemical CO2 Reduction: Continuum Models vs Molecular Dynamics," *J Phys Chem C* 2024 (PMC11215773)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11215773/) — directly states: *"Experimentally found values of C_Stern are often reported in the range of 20 to 25 μF cm⁻² at potentials well below (e.g., 0.5 V below) the PZC."* Flags historical PNP studies that used 100–200 µF/cm² as 5–10× too high. Their own modeling uses C_S = 20 µF/cm² with L_S = 5 Å, ε_S = 11.3. Applied to Ag, but framing is generic for an aqueous metallic interface.
- **Source (canonical tool):** [Ringe et al., CatINT (Stanford) input file](https://github.com/sringe/CatINT/blob/master/docs/source/tutorials/co2r_au_catmap/catint_input.rst) — `'Stern capacitance': 20., #micro F/cm2`. CatINT is one of the most-used CO₂R continuum modeling tools; this is the *default* they recommend.
- **Source (Wikipedia consensus + Helmholtz model):** [Double-layer capacitance, Wikipedia](https://en.wikipedia.org/wiki/Double-layer_capacitance) — "The Helmholtz model predicts a differential capacitance value of about 18 μF/cm²" with ε_r = 6 and 0.3 nm separation. With slightly different choices (ε_r ≈ 11, L ≈ 0.5 nm), the same model gives 20 µF/cm².

**The 20 µF/cm² value is generic for an aqueous metal/electrolyte interface.** For carbon, the literature suggests it should be *slightly lower* because (a) carbon basal-plane intrinsic capacitance is 2–6 µF/cm², and (b) carbon has weaker water polarization at the interface than metals (lower density of states at the Fermi level, less screening). So a defensible carbon-specific narrowing would be **C_S ≈ 10–20 µF/cm² (0.10–0.20 F/m²)** for sp²-rich carbon.

**Confidence:** high for the 20 µF/cm² baseline; medium for the carbon-specific downward adjustment.

---

### Sub-topic 5: What about the Singh 2016 JACS reference (51 µF/cm²)?

The user's prompt cites "Singh 2016 used C_dl = 51 µF/cm² for Cu cathode." I was unable to access the paper directly ([JACS 138:13006, 2016, doi:10.1021/jacs.6b07612](https://pubs.acs.org/doi/10.1021/jacs.6b07612), 403 Forbidden via WebFetch). However:

- This is a **measured C_dl on Cu** used for ECSA estimation (cyclic voltammetry slope of double-layer charging current vs scan rate in a non-faradaic window), not a Stern-layer-only value.
- 51 µF/cm² is *too high* for a flat Cu interface in the classical Stern sense (Cu electropolished is typically 20–40 µF/cm² in aqueous bicarbonate). This is consistent with the Cu surface having ~1.3–2× roughness factor from electropolishing — a real measurement on a slightly textured surface.
- **It is not directly applicable as C_S for CMK-3.** Carbon's intrinsic compact-layer capacitance is meaningfully lower than Cu's because of carbon's reduced density of states at the Fermi level (the "quantum capacitance" effect contributes a series term that drops effective C_dl).

The relevant Singh paper for our project's purposes is **Singh, Goddard, Bell 2016 JACS 138:13006** (cation hydrolysis at the OHP), which gives the *mechanism* the Phase 6β plan is implementing — but it does not provide a transportable C_S for CMK-3.

**Confidence:** medium (could not verify the exact 51 µF/cm² number directly, but the broader claim about Cu C_dl being in this range is consistent with Cu CO₂R literature).

---

### Sub-topic 6: Other Seitz-group / Mangan publications

I searched for Seitz-group follow-on papers:

- **Sanroman Gutierrez, Seitz et al. 2024** — "Efficient electrosynthesis of hydrogen peroxide in neutral media using boron and nitrogen doped carbon catalysts" ([J Mater Chem A 2024, DOI 10.1039/d4ta04613g](https://pubs.rsc.org/en/content/articlelanding/2024/ta/d4ta04613g)). This is the natural Ruggiero follow-on. I did not fully verify the body text but the abstract focuses on B/N doping effects on selectivity, not on EDL/Stern capacitance.
- **Mangan2025 deck** (`/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/docs/papers/Mangan2025_Catalysis.pdf`) — by `pdftotext` grep this is actually a talk titled "Inferring biological networks by sparse identification of nonlinear dynamics" produced from a PowerPoint by Niall Mangan, with bullet points "Mesoporous carbon black CMK-3 catalyst / Rotating Ring-Disk electrode" repeated several times. No capacitance value present. This is *not* the Linsey-deck the project refers to; it's a math/SINDy-flavored seminar version.

**Confidence:** medium. The Seitz lab's published record (as accessible to me) does not contain a primary CMK-3 C_S measurement.

---

### Quick Reference: Recommended C_S decision tree

```
For v10b literature calibration of PNP-Bikerman-BV with CMK-3:

  PRIMARY RECOMMENDATION:
    C_S = 0.20 F/m² (20 µF/cm²)
      cite: Bohra et al. J Phys Chem C 2024 (PMC11215773); CatINT default; Helmholtz model
      rationale: standard PNP-community value, derived from Stern thickness 5 Å + ε_r 11.3,
                experimentally consistent with C_Stern at potentials |E - E_PZC| > 0.3 V.

  CARBON-CONSERVATIVE ALTERNATIVE (KEEPS PRODUCTION CURRENT VALUE):
    C_S = 0.10 F/m² (10 µF/cm²)
      cite: Joo/Ryoo & follow-on CMK-3 supercapacitor literature (4-10 µF/cm²-BET);
            HOPG basal plane intrinsic 6 µF/cm² (Iamprasertkun et al. JPCL 2019);
            glassy-carbon low-end background 10 µF/cm² (multiple).
      rationale: lower end of carbon-specific range; reflects intrinsic sp²-carbon
                Helmholtz capacitance + quantum-capacitance reduction; matches current
                production setting without recalibration.

  SENSITIVITY BRACKET (the right interval to sweep):
    C_S ∈ [0.05, 0.30] F/m²  ≡  [5, 30] µF/cm²
      0.05: edge-case low (basal-plane minimum, intrinsic graphene)
      0.30: edge-case high (Helmholtz upper bound for water at metals, ε_r ≈ 11, L 0.3 nm)
      this is the right interval for v10b sensitivity / FIM analysis.
```

---

### Key Takeaways (for v10b lock decision)

- **Ruggiero 2022 does NOT report a C_S value for CMK-3.** Confirmed by direct read of all 31 pages of the manuscript PDF. The project's existing audit note (`docs/phase6/CONJECTURE_AUDIT_2026-05-09.md` line 21) is correct. Source: `/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/docs/papers/Ruggiero2022_JCatal_manuscript.pdf`.

- **The PNP-modeling community standard is C_S = 0.20 F/m² (20 µF/cm²).** This is the value to cite if you want the most defensible literature chain: Bohra et al. *J Phys Chem C* 2024 (PMC11215773), CatINT default, and the Helmholtz model with Stern thickness ≈ 5 Å, ε_r ≈ 11.3.

- **CMK-3-specific data from the supercapacitor literature gives 4–10 µF/cm²-BET in aqueous electrolytes** (1 M H₂SO₄, Na₂SO₄, Ca(NO₃)₂). These are *full C_dl per BET area*, not pure Stern, but at I ≥ 0.1 M the diffuse contribution is small and C_dl ≈ C_Stern within ~2×. Per-geometric-area is obscured by the porous-film roughness factor and should not be applied directly as the boundary-condition C_S.

- **0.10 F/m² is defensible if you cite carbon-specific intrinsic values** (HOPG basal plane 6 µF/cm², glassy-carbon low-end 10 µF/cm², CMK-3 supercapacitor per-BET 4–10 µF/cm²). It is at the *low* edge of the PNP-community range. If you want a citation chain that is "carbon-aware and conservative" rather than "generic PNP standard," 0.10 F/m² works.

- **Recommended action for v10b:** Either (a) **move to C_S = 0.20 F/m²** with the citation chain Bohra et al. 2024 + CatINT, declaring this as the "PNP-community standard for aqueous metal/electrolyte interfaces with a carbon-electrode caveat," or (b) **keep 0.10 F/m²** and add the citation chain Iamprasertkun et al. *JPCL* 2019 ([10.1021/acs.jpclett.8b03523](https://pubs.acs.org/doi/10.1021/acs.jpclett.8b03523)) for the HOPG basal-plane intrinsic value, plus Lota et al. via [Microporous Mesoporous Mater 2013](https://www.sciencedirect.com/science/article/abs/pii/S1387181113002679) for CMK-3 supercapacitor numbers. Plan v10b to sweep **C_S ∈ {0.05, 0.10, 0.20, 0.30} F/m²** as part of sensitivity analysis (this matches the existing `PHASE_6B_V9_GATES_3_4_SUMMARY.md` Stern sensitivity range, which already used {0.05, 0.10, 0.20}).

---

### Sources

- Ruggiero et al. 2022, *J Catalysis* 414:33–43 — direct PDF read, no C_dl value present. DOI: [10.1016/j.jcat.2022.08.040](https://www.sciencedirect.com/science/article/pii/S0021951722003591). Local: `/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/docs/papers/Ruggiero2022_JCatal_manuscript.pdf`
- Bohra et al. 2024, "Contrasting Views of the EDL in CO2R: Continuum vs MD," *J Phys Chem C*. [PMC11215773](https://pmc.ncbi.nlm.nih.gov/articles/PMC11215773/) — recommends 20–25 µF/cm² Stern, flags 100–200 µF/cm² as too high.
- CatINT (Ringe / Bell group) — [github.com/sringe/CatINT default Stern 20 µF/cm²](https://github.com/sringe/CatINT/blob/master/docs/source/tutorials/co2r_au_catmap/catint_input.rst)
- Iamprasertkun et al. 2019, *J Phys Chem Lett* — [HOPG basal-plane capacitance 4.7–9.4 µF/cm²](https://pubs.acs.org/doi/10.1021/acs.jpclett.8b03523)
- Randin/Yeager 1972, *J Electroanal Chem* — [HOPG basal-plane minimum ~3 µF/cm²](https://www.sciencedirect.com/science/article/abs/pii/S0022072872802493) (classical reference)
- Lota et al. 2013, "Improvement of EDL capacitance of CMK-3 by partial graphitization," *Microporous Mesoporous Mater* — [CMK-3 pristine 4.1 µF/cm²-BET, graphitized 6.8 µF/cm²-BET](https://www.sciencedirect.com/science/article/abs/pii/S1387181113002679)
- Ettifri et al., "Supercapacitive Performance of CMK-3 in Neutral Aqueous Electrolyte," *Iranian J Chem Chem Eng* — [285 F/g, ~10 µF/cm²-BET](https://www.ijcce.ac.ir/article_27404.html)
- Korean J Chem Eng 2013, "CMK-3 in Ca(NO₃)₂ aqueous" — [210 F/g](https://link.springer.com/article/10.1007/s11814-013-0289-z)
- Frackowiak et al. 2001, *J Appl Electrochem* — [Vulcan XC-72R in 30% H₂SO₄, C_A,DL = 16 µF/cm² per atomic BET](https://link.springer.com/article/10.1023/A:1017529920916)
- ACS Materials CMK-3 datasheet — [BET ≥ 900 m²/g, pore diameter 3.8–4.0 nm](https://www.acsmaterial.com/ordered-mesoporous-carbon-cmk-3.html)
- Jun, Joo, Ryoo et al. 2000 — [*JACS* 122, 10712 (original CMK-3 synthesis)](https://pubs.acs.org/doi/10.1021/ja002261e)
- Sanroman Gutierrez, Seitz et al. 2024 — [B,N-doped carbon for H2O2 (Seitz follow-on)](https://pubs.rsc.org/en/content/articlelanding/2024/ta/d4ta04613g)
- Project audit doc — `/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/docs/phase6/CONJECTURE_AUDIT_2026-05-09.md` (line 21) confirming "no specific C_S value cited in folder"
- Nature Communications 2014 review of carbon EDLCs — [universal 4–10 µF/cm²-BET asymptote](https://www.nature.com/articles/ncomms4317)
- Wikipedia — [Helmholtz model predicts 18 µF/cm² for aqueous interfaces (ε_r=6, L=0.3 nm)](https://en.wikipedia.org/wiki/Double-layer_capacitance)
- MIT OCW lecture 26 (Bazant, EES) — [discussion of compact-layer in PNP modeling](https://ocw.mit.edu/courses/10-626-electrochemical-energy-systems-spring-2014/8b6f18475743cb823a8ff45238a04228_MIT10_626S14_S11lec26.pdf)
