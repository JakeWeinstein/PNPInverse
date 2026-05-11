## Angle: General porous-carbon EDLC literature with normalization conventions — implications for CMK-3 Stern C_S in the PNP model

**Bottom line up front:** the textbook "10 μF/cm²" for sp² carbons is universally **per BET area**, not per geometric disk area. For a CMK-3 RRDE film at the Ruggiero 2022 / Mangan deck loading of **0.5 mg/cm²-geom** with a BET area of ~1000–1500 m²/g, the *effective per-geometric-area* double-layer capacitance is **on the order of 0.5 to 1.5 F/cm²-geom (≈ 5,000–15,000 F/m²-geom)** — i.e. 3–4 orders of magnitude larger than the model's current `stern_capacitance_f_m2 = 0.10` value. The current value behaves quantitatively like a *per-BET* number that has been silently treated as per-geometric. The Mangan/Seitz 1D-slab PNP setup (Jithin's overleaf docs) never resolves this distinction explicitly — it just writes `j = i/(zFA)` with a single "A" that is implicitly the geometric disk area, with no roughness factor.

---

### (A) Specific double-layer capacitance per BET area for graphitic / porous carbons

| Material / surface | Specific C (μF/cm²-BET) | Source / confidence |
|---|---|---|
| HOPG basal plane (1 M aqueous, pmin point) | 4.3–6.0 μF/cm² (FEG=6.0, AAG=4.3, IAG=4.7); 4.72 → 9.39 μF/cm² across Li⁺→Cs⁺ | High — Velický et al. 2019 *J. Phys. Chem. C* "Electrochemistry of the Basal Plane vs Edge Plane of Graphite Revisited"; Iamprasertkun et al. 2019 *J. Phys. Chem. Lett.* "Capacitance of Basal Plane and Edge-Oriented HOPG" |
| HOPG edge plane (oxygenated, aqueous) | ~50–70 μF/cm² intrinsic; ≥100× basal when pseudocapacitance counted | High — Velický 2019; Iamprasertkun 2019 |
| Porous activated / microporous carbons (sp² basal-dominated inner surface) | 10 μF/cm² (textbook rule of thumb); micropore wall ≈ 15–20 μF/cm² | High — Pandolfo & Hollenkamp 2006 *J. Power Sources* 157:11; Frackowiak & Béguin 2001 *Carbon* 39:937; Centeno & Stoeckli 2010 *Carbon* / "Capacitance and surface of carbons in supercapacitors" 2017 |
| Carbide-derived carbons (sub-nm pores) | up to 20–30 μF/cm² in narrow pores | Medium — Largeot et al. 2008 *JACS*; Chmiola et al. *Science* 2006 |
| "Clean" graphite reference | 20 μF/cm² often cited as a clean-graphite reference | Medium — review consensus |

**Synthesis (confidence: high):**
- The textbook rule of thumb in the supercapacitor literature is that aqueous double-layer capacitance for porous sp² carbons saturates around **10 μF/cm²-BET**, with a defensible range of ~5–25 μF/cm²-BET depending on pore accessibility, surface chemistry, and edge vs basal exposure. This is the basis of the canonical "1000 m²/g × 10 μF/cm² → 100 F/g" gravimetric estimate that Pandolfo & Hollenkamp and the Wikipedia/supercapacitor consensus literature both cite.
- CMK-3 has ~3–5 nm cylindrical mesopore walls. Those walls are graphitic sp² and predominantly basal-plane-like, with a modest edge contribution from disordered junctions. A defensible **per-BET-area** double-layer estimate for CMK-3 is therefore ~**5–15 μF/cm²-BET**, with 10 μF/cm² as the central textbook value.

### (B) RRDE catalyst film loading and roughness factor for CMK-3

**CMK-3 BET surface area (confidence: high):**
Reported values for commercial CMK-3 (e.g., ACS Material — the same vendor Ruggiero 2022 used) cluster in 1000–1500 m²/g, with individual reports of 1005, 1067, 1115, 1371, 1420, and 1688 m²/g across literature. A defensible representative value is **≈ 1200 m²/g**.
- Sources: Tuning the Pore Geometry of Ordered Mesoporous Carbons (PMC5507042); Lee et al. *Carbon* 2014 on AlSBA-15-templated CMK-3; ACSMaterial CMK-3 product datasheet.

**RRDE loading for CMK-3 in the Ruggiero/Mangan/Seitz deck (confidence: high — directly from the paper):**
Ruggiero, Sanroman Gutierrez, George, Mangan, Notestein, Seitz 2022 *J. Catal.*, "Probing the Relationship Between Bulk and Local Environments to Understand Impacts on Electrocatalytic Oxygen Reduction Reaction," §2.2 Electrode Preparation (quoted verbatim from the manuscript at `data/EChem Reactor Modeling-Seitz-Mangan/.../Manuscript_a684a021c0cfe03561a01550e490f19f.pdf`):
> "The disk working electrode was prepared by mixing 2 mg of CMK-3, 197.3 μL of ethyl alcohol and 2.7 μL of cation-exchanged Nafion solution. The resultant catalyst ink was sonicated for three hours. 10 μL of the catalyst ink was then drop-casted onto a clean and polished glassy carbon (GC, 0.196 cm²) disk… **to obtain a catalyst loading of 0.5 mg cm⁻²**."

That is **500 μg/cm²-geom of CMK-3 dry mass** on the disk — 2.5–5× the "typical" thin-film RDE loading range (the broader carbon-ORR literature cites 80–800 μg/cm² as a survey range; Bonakdarpour et al. *ECS* and review of NNMC RRDE protocols).

**Roughness factor arithmetic:**

Plugging in:
- Loading: m = 0.5 mg/cm²-geom = 5.0 × 10⁻⁴ g/cm²-geom
- BET: S_BET = 1200 m²/g = 1.2 × 10⁷ cm²-BET / g
- Roughness factor RF = S_BET × m = (1.2 × 10⁷ cm²/g) × (5.0 × 10⁻⁴ g/cm²-geom) = **6.0 × 10³ cm²-BET / cm²-geom**

Even at the low end of the CMK-3 BET range (1000 m²/g) and a conservative loading match: RF ≈ 5 × 10³. At a higher loading or higher BET (1500 m²/g): RF ≈ 7.5 × 10³.

This is *much larger* than the "RF = 130–260" figure in the research prompt — the prompt was assuming loading ≈ 100 μg/cm². The deck-actual loading (500 μg/cm²) is 5× higher, which scales RF to ~6000.

> **Caveat (confidence: medium):** Not all of the BET area is electrochemically accessible. The reviews (Centeno 2017; Béguin/Frackowiak 2014 book "Supercapacitors: Materials, Systems, Applications") emphasize that for microporous carbons, only the fraction of pores wider than the (de-)solvated ion is "wet" and contributes to double-layer charging. CMK-3's primary pore diameter is ~3.9–4 nm, well above the sub-nm cutoff where BET vs. electrochemically-active-area diverges, so the fraction of accessible area is high — probably 60–90%. The Nafion-binder fraction (~3 wt% in the Ruggiero ink) modestly blocks micropore mouths but the mesoporous bulk is reachable. Net: an *effective* RF of ~3000–6000 is a defensible band.

### Geometric C_S for the Ruggiero CMK-3 RRDE film

Geometric C_S = (specific C per BET area) × RF
- Central estimate: C_S^geom = 10 μF/cm²-BET × 6 × 10³ cm²-BET/cm²-geom = **6 × 10⁴ μF/cm²-geom = 6 × 10⁻² F/cm²-geom = 60,000 μF/cm² ≈ 0.6 F/cm² ≈ 6000 F/m²**
- Low-end (5 μF/cm²-BET × RF=3000): **15,000 μF/cm²-geom ≈ 0.15 F/cm²-geom ≈ 1500 F/m²-geom**
- High-end (15 μF/cm²-BET × RF=7500): **112,500 μF/cm²-geom ≈ 1.1 F/cm²-geom ≈ 11,000 F/m²-geom**

The CMK-3 RRDE film's effective C_S per geometric area is **~10² – 10⁴ F/m²-geom**, with a single best estimate near **10³ to 10⁴ F/m²-geom**.

**The current model uses `stern_capacitance_f_m2 = 0.10 F/m²` — which is ~10⁻⁵ × the defensible per-geometric estimate, but is in the right ballpark to be a per-BET-microscopic-Stern number (a *true* Stern-layer compact-layer capacitance for a flat, smooth, basal-like graphite surface is typically 20–100 μF/cm² = 0.2–1.0 F/m²; this is the *Stern* part, distinct from the *total* double-layer capacitance that adds the diffuse contribution in series).**

### (C) "Effective area" in PNP-Bikerman 1D-slab modeling for porous catalysts

This is where the load-bearing convention lives. The conclusion is that the Mangan/Seitz overleaf documents (the Jithin reaction-modeling PDFs from 2020) and the broader Bazant-school 1D PNP-Stern literature **do not resolve the porous vs. geometric area question** — they all silently assume a flat, single-area electrode.

Specific findings (confidence: high — sourced from the Jithin overleaf doc at `data/EChem Reactor Modeling-Seitz-Mangan/Reaction modeling overleaf documents/reaction_modeling_April22.pdf`):

- The Jithin model writes `j = i / (zFA)` (boundary flux from current) with a single area `A` that is treated as the cross-section of a 1D box (his Figure 1 explicitly shows the system as cuboidal cross-section A perpendicular to x).
- The "Boundary conditions" section §5 explicitly states "our problem occurs in 3D dimensions even if we are looking at only along one dimension" — i.e., A is the projected/geometric area of the flat 1D-slab cross-section. There is no roughness factor R_f anywhere in the document.
- This means the Mangan PNP setup is implicitly modeling a **flat electrode of geometric area A = 0.196 cm²** — not a porous electrode of BET area ≈ 6 × 10³ × 0.196 cm² ≈ 1200 cm²-BET.

The Bazant/Kilic/Storey-school mPNP-Stern papers (Kilic & Bazant 2007 PRE; Bazant, Chu & Bayly 2005; Bonnefont, Argoul & Bazant 2001) likewise treat the electrode as a single flat surface and write the Stern BC as:

φ_electrode − φ_OHP = (σ_surface / C_S)

where σ_surface is the surface charge density per unit *electrode area*, and C_S is the compact-layer capacitance per unit *electrode area*. **In their derivations the electrode is by construction a smooth planar surface — there is no concept of "geometric" vs "BET" area, because there is only one surface.** This is fine for the Pt single-crystal and HOPG basal-plane experiments that those papers target, where the geometric and active areas are equal.

For porous-electrode capacitor modeling (Biesheuvel & Bazant 2010 *Phys Rev E*, "Nonlinear dynamics of capacitive charging and desalination by porous electrodes"; Mirzadeh, Gibou, Squires 2014 *Phys Rev Lett*), the formulation switches to a volume-averaged "macrohomogeneous" framework where the *Stern capacitance per unit electrode volume* is C_S × a, with `a` the volumetric area density (m²-pore/m³-electrode). That is fundamentally a different model than the 1D-slab one used here — and the bridge between them is precisely the roughness factor / area-amplification problem.

**Implication (high confidence):** The PNPInverse model is a **flat-electrode 1D-slab PNP-Stern model**. It is mathematically consistent if interpreted as modeling the *flat planar disk surface* (e.g., the glassy carbon support, or a single planar facet of the CMK-3 particle), and the per-geometric-area current it predicts is what would be measured on a *smooth* planar electrode. To compare against an RRDE measurement on a porous CMK-3 film, one of three things must be true:

1. The model's `current_density_geom` output already represents the *current at the catalyst surface element*, and matching the deck's per-disk current density requires scaling by RF — which the model doesn't do explicitly. The factor-of-thousands discrepancy this would normally produce is hidden because k₀ is fit/freely adjusted.
2. The Stern capacitance and the kinetic rate constants are *both* being interpreted as effective per-geometric-area quantities — i.e., the model is implicitly absorbing the RF into both C_S and k₀. This is internally consistent for fitting one current-density curve but will fail any independent test of either parameter.
3. The model is treating the porous film as a "pseudo-flat" interface where the effective compact layer is the same as a single graphite-wall Stern layer (~20–100 μF/cm²) but the *geometric* current is enhanced by RF. This requires C_S ~ 0.2–1.0 F/m² in the residual *and* an RF-multiplication on the kinetic flux — the current code base does the former but not the latter.

---

### Key Takeaways

1. **Per-BET specific double-layer capacitance for CMK-3** is best estimated at **10 μF/cm²-BET** (defensible range 5–15 μF/cm²-BET), based on the Pandolfo & Hollenkamp / Frackowiak & Béguin consensus and the HOPG basal-plane measurements (4–9 μF/cm²-basal) bracketing the low end.

2. **CMK-3 BET surface area** ≈ **1000–1500 m²/g**, central value ~1200 m²/g (Tuning the Pore Geometry 2017; Lee 2014).

3. **Ruggiero/Mangan deck loading is 0.5 mg/cm²-geometric** (directly from the paper, §2.2). The often-quoted "100–200 μg/cm²" thin-film RDE convention does **not** apply here — the deck uses 5× more material.

4. **Roughness factor for the deck-actual film:** RF = S_BET × m = (1.2 × 10⁷ cm²/g)(5 × 10⁻⁴ g/cm²-geom) = **~6000 cm²-BET / cm²-geom**. Accounting for ~30% inaccessible area gives an *effective* RF of 3000–6000.

5. **Geometric double-layer capacitance estimate for the CMK-3 RRDE film:** C_dl^geom ≈ 10 μF/cm²-BET × 6000 ≈ **60 mF/cm²-geom ≈ 600 F/m²-geom** (central; defensible 150–1100 F/m²-geom). This is the *total* double-layer capacitance per geometric disk area.

6. **The pure Stern (compact-layer) capacitance for a single graphite-wall surface** is ~**20–100 μF/cm²-Stern-surface = 0.2–1.0 F/m²-Stern-surface** (this is *per unit surface element*, not per geometric disk). The model's `stern_capacitance_f_m2 = 0.10` is at the *low* end of this single-surface Stern range, but is **not** the per-geometric-area double-layer capacitance.

7. **Verdict on the model's current 0.10 F/m² Stern value:**
   - **If C_S is intended as "per local catalyst surface element" (i.e. the local compact-layer drop on a single graphite wall):** 0.10 F/m² is plausible but at the low end. Standard mPNP-Stern fits in the Bazant school give 0.2–1.0 F/m² for graphitic/sp² surfaces in dilute aqueous electrolyte. Defensible literature value: **0.2–0.5 F/m²** as a central estimate.
   - **If C_S is intended as "per geometric disk area" (i.e. the EIS-measured total CdI lumped onto the disk):** 0.10 F/m² is 3–4 orders of magnitude too low. The defensible value would be **150–1100 F/m²-geom**.
   - The model's residual is a 1D-slab Robin BC `φ_electrode − φ_OHP = σ/C_S` where σ is the diffuse-layer surface charge density from a flat 1D solve. That residual structurally treats C_S as **per local surface element**, not per geometric disk. The 0.10 F/m² value is therefore self-consistent *within the flat-electrode interpretation* — but the model then cannot be calibrated against per-geometric-disk RRDE currents without an additional RF multiplier on the kinetic flux (which the code does not apply).

8. **The 1D-slab PNP-Bikerman convention (Jithin overleaf; Bazant/Kilic/Storey lineage) treats the electrode as flat with a single area `A`** that is implicitly geometric. There is no built-in roughness factor. Porous-electrode-aware variants (Biesheuvel/Bazant 2010; macrohomogeneous capacitor models) use a volumetric area density `a` and per-pore-area C_S — that is a different formulation than the current code.

9. **Recommended action for the C_S calibration question:** the cleanest interpretation is to keep `stern_capacitance_f_m2 ≈ 0.2 F/m²` (mid of the Bazant-school single-surface compact-layer band for sp² carbon in aqueous K₂SO₄) and treat all kinetic prefactors (k₀_R2e, k₀_R4e) as *per-local-surface-element* rate constants — accepting that the comparison against the deck's per-geometric-disk current will require an explicit RF multiplier (~3000–6000) somewhere, or equivalently that the fit k₀ values are absorbing the RF. Document this in the inversion conventions explicitly. **The current 0.10 F/m² is too low by ~2× relative to the consensus per-surface Stern value, but is not "secretly per-BET" in any defensible reading of the model equations.**

---

### Sources (with confidence ratings)

**High confidence (peer-reviewed primary literature, directly relevant):**
- Ruggiero, Sanroman Gutierrez, George, Mangan, Notestein, Seitz 2022, *J. Catal.* (the deck paper itself), https://www.sciencedirect.com/science/article/pii/S0021951722003591 — loading 0.5 mg/cm² CMK-3, K₂SO₄ + Li⁺/Na⁺/K⁺/Cs⁺ electrolytes, pH 2–12, parallel 2e/4e ORR (E° = 0.695 V / 1.23 V vs RHE), iridium-oxide ring pH probe. Local manuscript at `data/EChem Reactor Modeling-Seitz-Mangan/.../Manuscript_a684a021c0cfe03561a01550e490f19f.pdf`.
- Pandolfo & Hollenkamp 2006, *J. Power Sources* 157:11, "Carbon properties and their role in supercapacitors," https://www.sciencedirect.com/science/article/abs/pii/S0378775306003442 — foundational review; ~10 μF/cm²-BET consensus, 100 F/g for 1000 m²/g carbons.
- Frackowiak & Béguin 2001, *Carbon* 39:937, "Carbon materials for the electrochemical storage of energy in capacitors," https://www.sciencedirect.com/science/article/abs/pii/S0008622300001834 — same consensus, with explicit microporous-vs-mesoporous accessibility caveats.
- Velický et al. 2019, *J. Phys. Chem. C* 123:11677, "Electrochemistry of the Basal Plane vs Edge Plane of Graphite Revisited," https://pubs.acs.org/doi/abs/10.1021/acs.jpcc.9b01010 — HOPG basal 4–9 μF/cm², edge ~50–70 μF/cm² intrinsic.
- Iamprasertkun et al. 2019, *J. Phys. Chem. Lett.* 10:617, "Capacitance of Basal Plane and Edge-Oriented HOPG: Specific Ion Effects," https://pubs.acs.org/doi/abs/10.1021/acs.jpclett.8b03523 — 4.72–9.39 μF/cm²-basal across Li⁺→Cs⁺; edge ~100× basal.
- Kilic, Bazant, Ajdari 2007, *Phys. Rev. E* 75:021503 (Part II, "Modified PNP"), https://link.aps.org/doi/10.1103/PhysRevE.75.021503 — flat-electrode Stern BC formulation; no roughness factor; C_S as per-electrode-area.
- Bazant, Chu, Bayly 2005, *SIAM J. Appl. Math.*, https://web.mit.edu/bazant/www/papers/pdf/Bazant_2005_SIAM_J_Appl_Math_MTF2.pdf — Stern + Butler-Volmer thin-film PNP; compact-layer C_S as a single per-area boundary parameter.

**Medium confidence (review-level / vendor data):**
- "Capacitance and surface of carbons in supercapacitors" 2017 review, *Carbon* https://www.sciencedirect.com/science/article/abs/pii/S0008622317306619 — 15–20 μF/cm² on micropore surface; emphasizes BET vs accessible-area distinction.
- Centeno, Sereda, Stoeckli 2010, *J. Power Sources* / 2017 *Carbon* "Capacitance and surface of carbons in supercapacitors" — accessible-area normalization.
- ACSMaterial CMK-3 product page https://www.acsmaterial.com/ordered-mesoporous-carbon-cmk-3.html — BET ≈ 1100 m²/g, mesopore 3.9 nm (Ruggiero source).
- Tuning the Pore Geometry of Ordered Mesoporous Carbons for Enhanced Adsorption of Bisphenol-A, https://pmc.ncbi.nlm.nih.gov/articles/PMC5507042/ — CMK-3 1115 m²/g representative.

**Lower confidence (general context / review snippets):**
- "Electrochemical surface area" Wikipedia, https://en.wikipedia.org/wiki/Electrochemical_surface_area — distinguishes ECSA from BET; references for "1000 m²/g × 10 μF/cm² → 100 F/g."
- Frontiers in Chemical Engineering 2022, "A review of transport models in charged porous electrodes," https://www.frontiersin.org/journals/chemical-engineering/articles/10.3389/fceng.2022.1051594/full — confirms macrohomogeneous porous-electrode models use per-pore-area C_S × volumetric area density `a`, not per-geometric-disk.

**Local docs (for verification of conventions):**
- `/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/data/EChem Reactor Modeling-Seitz-Mangan/Reaction modeling overleaf documents/reaction_modeling_April22.pdf` — Jithin's PNP-Stern formulation; flat-electrode 1D-slab, area `A` is the geometric disk cross-section, no roughness factor, no per-BET treatment.
- `/Users/jakeweinstein/Desktop/ResearchForwardSolverClone/FireDrakeEnvCG/PNPInverse/data/EChem Reactor Modeling-Seitz-Mangan/Manuscript_a684a021c0cfe03561a01550e490f19f.pdf` — Ruggiero 2022 manuscript (verified loading = 0.5 mg/cm², disk = 0.196 cm², CMK-3 from ACS Material).
