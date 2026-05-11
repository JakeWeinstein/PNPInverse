# Research Summary: CMK-3 Stern Capacitance for PNP-BV Model

**Query:** Defensible literature value for the local Stern interfacial capacitance C_S of CMK-3 mesoporous carbon under aqueous ORR conditions (deck: 0.1 M K₂SO₄ / 0.2 M K⁺, pH 4–6, parallel 2e⁻/4e⁻ topology per Ruggiero 2022), expressed in F/m² per geometric electrode area. Project currently uses `stern_capacitance_f_m2 = 0.10 F/m²`, uncited.

**Date:** 2026-05-10
**Sources:** 4 research agents (2 web, 1 literature, 1 codebase)

---

## Executive Summary

1. **No CMK-3-specific Stern capacitance exists in the literature or the Seitz/Mangan data folder.** Ruggiero 2022 reports zero capacitance value; the supercapacitor literature only reports per-BET (4–10 µF/cm²-BET) values that don't cleanly map to a PNP-Stern parameter. The field treats Stern C_S as an effective Helmholtz parameter, often fit to data.

2. **The convergent literature-anchored single value is C_S = 0.20 F/m² (20 µF/cm²)**, derived from Stern thickness L_S = 5 Å and Stern permittivity ε_S = 11.3 (dielectric saturation from Booth). Three independent sources agree: Bohra et al. 2024 *JPC C* (PMC11215773), CatINT default (Stanford, Ringe/Bell group), and Choi et al. 2024. Pillai 2024 *JPC C* (`10.1021/acs.jpcc.3c05364`) flags the "supercharged" CO2R-modeler convention of 100–200 µF/cm² as a methodological error and identifies 20–25 µF/cm² as the experimentally-grounded range.

3. **The project's current 0.10 F/m² is defensible-but-low-end.** It sits at the bottom of Pillai 2024's "safe band" (10–50 µF/cm²) and at the low edge of the carbon-electrode-specific range (HOPG basal ~6 µF/cm², CMK-3 supercap ~4–10 µF/cm²-BET, sp²-carbon Stern ~10–20 µF/cm²). It is *not* unphysical, but it is not citation-anchored — it was selected as the smallest finite-Stern value that let Newton cross the +1.0 V wall on the 3sp+bikerman+muh stack (15/15 convergence).

4. **Architectural finding (load-bearing for v10b):** The 1D-slab PNP-Stern formulation (Bazant/Kilic/Storey/Jithin lineage) has **no roughness factor**. C_S in the residual is treated as a per-local-surface-element compact-layer capacitance. The RF ≈ 6000 amplification from the deck's 0.5 mg/cm² CMK-3 loading on BET ≈ 1200 m²/g is *implicit in the fitted k₀ values*, never explicit in the boundary condition.

5. **The Singh 2016 σ_S "mismatch" concern is partially based on a misapplication.** Singh's 51 µF/cm² for Cu was an in-house CV-slope C_dl measurement (total Stern + diffuse, not Stern-only), measured on polycrystalline Cu in CO₂R conditions. It is **not** a transportable Stern-only value for CMK-3.

---

## 1. Recommended C_S Anchor (Literature-Cited Convergence)

### The 20 µF/cm² consensus and how it's derived

| Source | DOI / Link | Position |
|---|---|---|
| **Bohra et al. 2024** "Contrasting Views of the EDL in CO2R" *JPC C* | [PMC11215773](https://pmc.ncbi.nlm.nih.gov/articles/PMC11215773/) | "Experimentally found values of C_Stern are often reported in the range of 20 to 25 µF cm⁻²." Flags 100–200 µF/cm² as 5–10× too high. |
| **Choi et al. 2024** *JPC C* 128, 27, 11075 | [10.1021/acs.jpcc.4c03469](https://pubs.acs.org/doi/10.1021/acs.jpcc.4c03469) | Explicit derivation: "Stern thickness and relative permittivity are 5 Å and 11.3, respectively, to achieve a Stern layer capacitance of 20 µF cm⁻²." |
| **CatINT** (Ringe/Bell, Stanford) | [github.com/sringe/CatINT](https://github.com/sringe/CatINT/blob/master/docs/source/tutorials/co2r_au_catmap/catint_input.rst) | Default config: `'Stern capacitance': 20., #micro F/cm2` |
| **Pillai et al. 2024** "Surface Charge BC Often Misused" *JPC C* | [10.1021/acs.jpcc.3c05364](https://pubs.acs.org/doi/10.1021/acs.jpcc.3c05364) | Critical review: published ε_Stern span 6 to 80.1; majority of CO2R papers use bulk-water ε implying 100–200 µF/cm² — "far higher than the Stern layer capacitances measured in fundamental experiments, typically in the range of 20–25 µF cm⁻²." |

**Derivation:** `C_S = ε_S · ε_0 / L_S = 11.3 × 8.854×10⁻¹² F/m / 5×10⁻¹⁰ m = 0.200 F/m² = 20 µF/cm²`

The ε_S ≈ 11 (down from bulk ε ≈ 78) reflects **dielectric saturation** via Booth (1951): at η ~ 1 V across 0.5 nm, E ~ 2 × 10⁹ V/m, and water dipoles are fully aligned. The factor-~10 drop in ε is exactly what distinguishes the "grounded" 20–25 µF/cm² consensus from the "supercharged" 100–200 µF/cm² convention.

### Where the carbon-specific evidence narrows the range

For sp²-carbon specifically, the literature suggests C_S should be *slightly lower* than the metal-electrode 20 µF/cm² because of (a) lower density-of-states at the Fermi level (the quantum-capacitance series term lowers effective C_dl) and (b) weaker water polarization at carbon than at metals.

| Material | Specific C | Source |
|---|---|---|
| HOPG basal plane (intrinsic minimum) | 2.5–3 µF/cm² | Randin–Yeager 1972 *JEAC* |
| HOPG basal plane (Li⁺→Cs⁺, at PZC+0.5V) | 4.7–9.4 µF/cm² | Iamprasertkun 2019 *JPCL* [10.1021/acs.jpclett.8b03523](https://pubs.acs.org/doi/10.1021/acs.jpclett.8b03523) |
| HOPG edge plane (oxygenated) | ~50–70 µF/cm² (>100× basal) | Velický 2019 *JPC C* |
| Glassy carbon (background) | 10–25 µF/cm² | textbook |
| CMK-3 pristine | **4.1 µF/cm²-BET** | Yamada 2013 *Micropor Mesopor Mater* [10.1016/j.micromeso.2013.05.024](https://www.sciencedirect.com/science/article/abs/pii/S1387181113002679) |
| CMK-3 graphitized | **6.8 µF/cm²-BET** | Yamada 2013 |
| CMK-3 in Na₂SO₄ aqueous | ~10 µF/cm²-BET | Ettifri *Iranian J Chem* |
| Activated carbon/CNT asymptote, dilute aq | 4–10 µF/cm²-BET | Nature Comm 2014 |
| Bazant-school per-surface-element Stern (sp² in dilute aq) | 0.2–1.0 F/m² | Kilic-Bazant 2007 *Phys Rev E* 75:021503 |

**Synthesized carbon-specific Stern range:** 0.10–0.20 F/m² per local surface element. The carbon-specific narrowing pulls the generic 20 µF/cm² PNP value *downward by a factor of ~2*.

---

## 2. Literature Landscape: Three Regimes (per Pillai 2024)

1. **"Bohra-Koper-Choi grounded": C_S ≈ 20–25 µF/cm² (0.20–0.25 F/m²).** δ_S = 5 Å × ε_Stern = 11.3 (water dipole-saturated, Booth-style mean-field). Calibrated to Ag(111) SPEIS (Choi 2024). **Experimentally anchored.**

2. **"Many CO2R modelers": C_S ≈ 100–200 µF/cm² (1–2 F/m²).** Assumes Stern retains bulk water ε ≈ 78 over sub-nm thickness. **Pillai 2024 flags as methodological error. Avoid.**

3. **"Phenomenological fits": C_S = 10–50 µF/cm² (0.10–0.50 F/m²).** Kilic-Bazant 2007 style — C_S left as model parameter, field-averaged effective compact-layer capacitance lumping in saturation/image-charge/partial-charge-transfer. **Project's current 0.10 sits at the low edge of this band — in the "safe range" per Pillai but at the bottom.**

**Priority reading:** Pillai 2024 → Choi 2024 → Bohra 2019 EES (the latter is cited by both Ruggiero 2022 ref 71 and Linsey 2025 deck slide 13 but is **NOT in the data folder Articles/** — open literature ask).

---

## 3. The Singh 51 µF/cm² Recalibration (Load-Bearing Correction)

The project's prior σ_S mismatch concern was: "Singh 2016 cites C_dl = 51 µF/cm² for Cu and σ_S = 226 µC/cm²; with C_S = 10 µF/cm² we'd need Δφ_Stern ≈ 22.6 V which is unphysical → 0.10 F/m² is ~5× too low" (Risk #5 in `PHASE_0_ACCEPTANCE_BUNDLE_LOCK_2026-05-10.md:205`).

**This concern is partially based on a misapplication:**

1. **Singh's 51 µF/cm² is an in-house CV-slope measurement, not a citation.** Per Google Scholar snippet: "the double layer capacitance was determined by calculating the slope of this graph." Singh, Kwon, Lum, Ager, Bell measured their *own* polycrystalline Cu in NaHCO₃/CO₂R conditions.

2. **It is the *total* differential C_dl, not Stern-only.** It conflates Stern + diffuse + roughness + specific adsorption. The PNP-BV residual's `C_S = σ / Δφ_Stern` wants the Stern-only value.

3. **It is for Cu, not carbon.** Polycrystalline Cu gives 20–60 µF/cm² (Hori/Schouten/Trasatti); carbon's intrinsic Stern is meaningfully lower because of quantum-capacitance reduction.

4. **Singh 2016 vs Singh 2017 disambiguation:** The cation-hydrolysis paper is **Singh, Kwon, Lum, Ager, Bell 2016 *JACS* 138:13006** ([10.1021/jacs.6b07612](https://pubs.acs.org/doi/10.1021/jacs.6b07612)). NOT Lewis-Singh 2015/2017 (which are Nernst-Planck only with no Stern). NOT Singh-Goodpaster 2017 PNAS (no Stern, no field-dependent pKa). Phase 6β cation-pKa formulation is tied to the 2016 JACS paper alone.

5. **Context:** Singh's 51 (Cu total) ≈ 2× Bohra's 20 (Ag Stern-only) — plausible because of Cu polycrystalline roughness + specific anion adsorption + Stern+diffuse vs Stern-only decomposition. The two are not directly comparable.

**Recommendation:** Re-derive σ_S mismatch using Stern-only Bohra/Koper value (20 µF/cm² for metal; ~10 µF/cm² for sp²-carbon). If σ_S = 226 µC/cm² still requires unphysical Δφ_Stern at C_S = 0.20 F/m², then the σ_S target itself is non-transportable from polycrystalline Cu to CMK-3. **Before treating σ_S mismatch as load-bearing, the Stern-only mapping needs clarification.**

---

## 4. The Roughness-Factor Architectural Finding

### The deck-actual numbers
- **CMK-3 loading (Ruggiero 2022 §2.2, verbatim):** "2 mg CMK-3 + 197.3 µL ethanol + 2.7 µL Nafion; 10 µL drop-cast on 0.196 cm² GC disk → **catalyst loading 0.5 mg cm⁻²-geometric**."
- **CMK-3 BET:** 1000–1500 m²/g (representative ~1200 m²/g).
- **Initial "100–200 µg/cm² typical thin-film RDE" assumption is WRONG.** The deck uses 5× more material.

### Roughness factor arithmetic
```
RF = S_BET × m_loading = (1.2 × 10⁷ cm²/g) × (5.0 × 10⁻⁴ g/cm²-geom) = 6.0 × 10³ cm²-BET / cm²-geom
```
With ~30% inaccessible BET (Nafion blockage), **effective RF ≈ 3000–6000**.

### Geometric C_dl implied
- C_dl^geom = 10 µF/cm²-BET × 6000 = **60 mF/cm²-geom ≈ 6000 F/m²-geom** (defensible 1500–11000).
- Supercap community sees this: 200–700 F/g × 0.5 mg/cm² ≈ 0.1–0.4 F/cm²-geom ≈ 1000–4000 F/m²-geom.

### What this means for the 1D-slab PNP-Stern formulation
- The Jithin overleaf docs (`data/EChem Reactor Modeling-Seitz-Mangan/Reaction modeling overleaf documents/reaction_modeling_April22.pdf`) write `j = i/(zFA)` with A = geometric, no roughness factor.
- The Bazant/Kilic/Storey lineage (Kilic-Bazant 2007 PRE; Bazant-Chu-Bayly 2005 SIAM) likewise treat the electrode as smooth planar with one area; C_S is per-electrode-area.
- **The PNPInverse model is a flat-electrode 1D-slab PNP-Stern model.** It is mathematically consistent if interpreted as modeling a single planar CMK-3 facet (or the GC support). To match deck RRDE currents, one of three things must be true:
  1. **Per-local-surface-element interpretation (most defensible):** Current is per surface element; RF amplification is implicit in fitted k₀.
  2. **Per-geometric-disk interpretation:** C_S = 0.10 F/m² would be ~10⁵× too low (would need 1500–11000 F/m²-geom). Breaks the model.
  3. **Hybrid:** C_S per-local-surface-element AND explicit RF on kinetic flux. Code currently does the former, not the latter.

**Verdict:** Current C_S = 0.10 F/m² is **self-consistent within per-local-surface-element interpretation** but at the *low* end of the Bazant single-surface Stern range (0.2–1.0 F/m²). Cross-stack comparison (Yash, Bohra, deck-measured C_dl) requires explicit RF accounting on the kinetic side or comparing per-local-surface-element only. **Document this in inversion conventions.**

---

## 5. Provenance of the Project's Current 0.10 F/m²

### Where 0.10 F/m² entered (2026-05-05)
- **Commit `247c56e`** (2026-05-05): `docs/solver/stern_layer_physics_and_next_steps.md` (dated 2026-05-03) line 204 proposed sweep `C_S = [None, 0.05, 0.10, 0.20, 0.40, 1.00] F/m²`. Lines 207–214 justification: "1 F/m² = 100 µF/cm². So the range above covers roughly 5 to 100 µF/cm², with 0.10–0.40 F/m² being a reasonable compact-layer scale to inspect." **No literature citation.**
- **Commit `4026b70`** (2026-05-05): `scripts/studies/peroxide_window_stern_test.py` with `CS_DEFAULT = (None, 0.05, 0.10, 0.20, 0.40, 1.00)`. Commit msg: "stern_test: finite Stern capacitance sweep (informs the 0.10 F/m^2 [default])."
- **Lock-in commit `793b73e`**: `docs/ic_studies/4sp_bikerman_ic_option_2b_results.md` recorded C_S = 0.10 F/m² gave 15/15 clean convergence to V_RHE = +1.00 V on the 3sp+bikerman+muh stack while no-Stern (None) failed.
- **Propagation:** 23+ call sites pass `stern_capacitance_f_m2=0.10` (the default in `scripts/_bv_common.py:539, 1082` is `None`).

### Self-described provenance (verbatim from `docs/phase6/phase6b_next_steps_plan.md` §5.3)
> "0.10 F/m² has no Ruggiero 2022 or Linsey 2025 deck citation. Actual provenance: a sweep midpoint... The value has therefore been a **convergence-pinned engineering choice**, not a deck-calibrated parameter."

### What the data folder cites for C_S
The codebase agent grep'd every PDF, DOCX, pptx in `data/EChem Reactor Modeling-Seitz-Mangan/` for capacitance/Stern/µF/F-per-area. Findings:

| File | Content |
|---|---|
| Ruggiero 2022 manuscript | No Stern, no µF/cm², no F/m² beyond "capacitive current subtraction" methods. Cites Bohra 2019 (ref 71). |
| Linsey 2025 ACS-CATL deck slides 12–13 | Stern modification named; cites Bohra 2019 + Borukhov-Orland 1997 + Kilic-Ajdari 2007 + Butt-Hartkamp 2023. **No specific C_S value.** |
| Linsey 2020 deck, Seitz NU lecture, Tang Drexel lecture, Bard & Faulkner Ch 13 | Educational / qualitative. No CMK-3 measurement. |
| Co-Zhang 2019 Angewandte | "Buffer capacity" only. No numerical C_S. |
| **Bohra 2019 EES** | **NOT in Articles/ folder** despite being cited by both Ruggiero ref 71 and Linsey deck slide 13. |
| Brianna's slides | OHP / cation hydration qualitatively. No numeric C_S. |
| **Yash modeling code** | Uses Stern **thickness** `L_Stern` ∈ {0.6, 0.8, 1.0 nm}, **not capacitance**. Different functional form. |
| Jithin reaction modeling overleaf docs (4 files) | No EDL-capacitance content. |

**Zero numeric C_S value** attributable to CMK-3 or generic carbon in the entire data folder.

### Yash thickness vs project capacitance
Yash L_Stern = 0.6 nm with ε_r = 6 (Conway oriented-water) → C_S ≈ 0.088 F/m² ≈ 8.8 µF/cm² (close to project's 0.10). With ε_r = 11.3 (Choi-Bohra saturation) → C_S = 0.17 F/m². Whether dielectric saturation is fully invoked determines which value is canonical.

---

## 6. Decision Options for v10b — Trade-offs

### Option (a): Raise to C_S = 0.20 F/m² (literature-anchored) — **PRIMARY RECOMMENDATION**
- **Citation chain:** Bohra et al. 2024 *JPC C* (PMC11215773) + CatINT default + Choi et al. 2024 (δ_S = 5 Å, ε_S = 11.3 → 20 µF/cm²) + Pillai 2024 safe-band + Kilic-Bazant 2007 foundational.
- **Pros:** Most defensible single value. Aligns with experimentally-grounded PNP-BV CO2R/ORR consensus. 2× the current value but in the same Pillai-safe band.
- **Cons:** Doubles current value. Requires validating Newton convergence at +1.0 V wall regression (should *improve* since higher C_S → less Stern drop). Carbon-specific narrowing pulls slightly below 0.20.

### Option (b): Keep C_S = 0.10 F/m² (carbon-conservative)
- **Citation chain:** Kilic-Bazant 2007 phenomenological + Iamprasertkun 2019 HOPG basal 4.7–9.4 + Lota/Yamada 2013 CMK-3 4.1–6.8 µF/cm²-BET + Pillai 2024 safe-band + project's empirical convergence-pinned choice.
- **Pros:** Carbon-aware, conservative. No code changes. Preserves convergence record.
- **Cons:** At the *low* edge of PNP-community range. Cannot be cited as "standard PNP-modeling value."

### Option (c): Treat C_S as free fit parameter
- **Setup:** Prior at 0.10 F/m²; bracket-sweep `C_S ∈ {0.05, 0.10, 0.20, 0.30}` (extends v9 Gates 3-4 plan `{0.05, 0.10, 0.20}` with one high-end point).
- **Pros:** Most honest; lets data update prior. Captures full Pillai-safe band.
- **Cons:** Adds one fit parameter, identifiability concern with other Stern params. Defers literature anchoring.

---

## 7. Contradictions and Open Questions

### Where the agents disagree
- **Web 1** recommends 0.20 F/m² primary, 0.10 F/m² as carbon-conservative alternative.
- **Web 2** recommends 0.2–0.5 F/m² (mid of Bazant single-surface range), weights per-local-surface-element interpretation more strongly. Sees 0.10 as "too low by 2×".
- **Literature agent** is more agnostic — presents three options without picking a winner; emphasizes 10 µF/cm² is "in the low end but not unphysical."
- All three converge on 0.20 F/m² as the *literature anchor* with the Bohra/CatINT/Choi citation chain.

### Stern thickness vs capacitance convention
- Bazant/Kilic/Bohra/Choi/Pillai use **C_S directly**.
- Yash code uses **L_Stern thickness** (0.6–1.0 nm); Bohra 2019 fine print also uses L_Stern + ε_Stern (often field-dependent via Booth).
- **Open: Does v10b switch to (L_Stern, ε_Stern) parameterization or keep C_S as single effective parameter?**

### Booth equation / variable permittivity
- Storey-Bazant 2012 onward, Bohra 2019, Choi 2024 acknowledge ε_S is field-dependent (Booth 1951).
- Constant C_S is a **field-averaged effective value**; cannot capture η-dependence implicit in Booth.
- Factor-~10 drop from bulk ε = 78 to dipole-aligned Stern ε ≈ 6–11 is what gives the 20 µF/cm² consensus.
- **Open for v10b:** Stay with constant-C_S (simpler, defensible if explicitly field-averaged) or move to variable-ε?

### Does C_S even matter much in current sensitivity range?
The literature agent flags: with ε ~ 80 outer-Helmholtz and Stern thin compared to diffuse layer, a factor-2 swing in C_S typically moves surface pH by < 0.5 pH units. **Worth checking via v10b sensitivity sweep before investing further in citation chains.** Pillai's safe band (10–50 µF/cm²) is wide; values inside this band may not drive the dominant systematic in the forward solver.

### Singh 2016 σ_S re-derivation needed
- Is σ_S = 226 µC/cm² target itself transportable from polycrystalline Cu in NaHCO₃ to CMK-3 in K₂SO₄?
- With Stern-only C_S = 20 µF/cm² instead of total C_dl = 51 µF/cm², does the σ_S mismatch persist?
- Both need addressing before treating Risk #5 as load-bearing.

---

## 8. Codebase Relevance

- `stern_capacitance_f_m2 = 0.10 F/m²` is supplied by ~23 call sites; default in `scripts/_bv_common.py:539, 1082` is `None`.
- Bracket-sweep `{0.05, 0.10, 0.20}` already planned at v10b/Gate 4B per `docs/phase6/missing_data.md` M2 and `docs/phase6/phase6b_next_steps_plan.md` §5.3.
- σ-scale consistency check vs Singh 2016 logged as Risk #5 in `PHASE_0_ACCEPTANCE_BUNDLE_LOCK_2026-05-10.md:205`.

---

## 9. Recommendations for v10b

### Primary: **Option (a) — Raise to C_S = 0.20 F/m²**

Convergent evidence from three independent PNP-modeling sources supports 0.20 F/m² as the literature-anchored single value.

**Concrete steps:**
1. **Lock C_S = 0.20 F/m² in v10b production.** Add citation chain to `scripts/_bv_common.py` docstring and CLAUDE.md "Hard rules":
   - Bohra et al. 2024 *JPC C* PMC11215773
   - Choi et al. 2024 *JPC C* `10.1021/acs.jpcc.4c03469`
   - Pillai et al. 2024 *JPC C* `10.1021/acs.jpcc.3c05364` (safe-band defense)
   - CatINT default (Stanford Bell group)
   - Kilic, Bazant, Ajdari 2007 *Phys Rev E* 75:021503 (foundational mPNP-Stern)
2. **Run convergence regression first.** Verify 15/15 V_RHE = +1.0 V wall still clears at 0.20 (should improve since higher C_S → less Stern drop).
3. **Bracket sweep for v10b:** `C_S ∈ {0.05, 0.10, 0.20, 0.30} F/m²` (extends v9 `{0.05, 0.10, 0.20}` with one high-end point).
4. **Document per-local-surface-element interpretation** in `docs/solver/bv_solver_unified_api.md`:
   - 1D-slab PNP-Stern treats C_S as per-local-surface-element compact-layer cap.
   - No explicit RF in BC; deck-actual RF ≈ 6000 implicit in fitted k₀.
   - Cross-stack comparison (Yash, Bohra, deck C_dl) requires explicit RF accounting.

### Secondary if (a) regresses
- **Option (c) free-fit** with 0.10 prior; report bracket-sweep results. Document constant-C_S as first-order approximation; variable-ε / Booth refinements (Storey-Bazant 2012, Bohra 2019) out of scope for v10b.

### Architecture follow-up (separate from C_S lock)
- **Pull Bohra 2019 EES** `10.1039/c9ee02485a` to Articles/. Cited by Ruggiero ref 71 + Linsey slide 13; absence is open ask. Code at [github.com/divyabohra/GMPNP](https://github.com/divyabohra/GMPNP) (FEniCS + Bikerman).
- **Re-derive Risk #5 σ_S mismatch** using Stern-only C_S = 20 µF/cm². If mismatch persists, treat σ_S target as non-transportable from polycrystalline Cu.
- **Document implicit RF in fitted k₀** in CLAUDE.md so future inverse work doesn't double-count.
- **Cross-check Yash convention:** C_S from L_Stern = 0.6 nm with ε_S = 11.3 → 0.17 F/m² (Choi-consistent) vs. ε_S = 6 → 0.088 F/m² (Conway-consistent). Pick convention and stick with it.

---

## 10. Key Sources

**PNP-BV modeling — primary citations for C_S = 0.20 F/m²:**
- Bohra et al. 2024 *JPC C* — [PMC11215773](https://pmc.ncbi.nlm.nih.gov/articles/PMC11215773/)
- Choi et al. 2024 *JPC C* — [10.1021/acs.jpcc.4c03469](https://pubs.acs.org/doi/10.1021/acs.jpcc.4c03469)
- Pillai et al. 2024 *JPC C* — [10.1021/acs.jpcc.3c05364](https://pubs.acs.org/doi/10.1021/acs.jpcc.3c05364)
- CatINT default — [github.com/sringe/CatINT](https://github.com/sringe/CatINT/blob/master/docs/source/tutorials/co2r_au_catmap/catint_input.rst)
- Kilic, Bazant, Ajdari 2007 *Phys Rev E* 75:021502/021503 — [arXiv physics/0611030](https://arxiv.org/pdf/physics/0611030)
- Bohra et al. 2019 EES 12:3380 — [10.1039/c9ee02485a](https://pubs.rsc.org/en/content/articlehtml/2019/ee/c9ee02485a) (open literature ask)

**Singh / Bell group:**
- Singh, Kwon, Lum, Ager, Bell 2016 *JACS* 138:13006 — [10.1021/jacs.6b07612](https://pubs.acs.org/doi/10.1021/jacs.6b07612) (in-house CV-slope, NOT Stern-only)

**Carbon EDLC:**
- Iamprasertkun 2019 *JPCL* — [10.1021/acs.jpclett.8b03523](https://pubs.acs.org/doi/abs/10.1021/acs.jpclett.8b03523) (HOPG basal 4.7–9.4)
- Yamada/Lota 2013 *Micropor Mesopor Mater* — [10.1016/j.micromeso.2013.05.024](https://www.sciencedirect.com/science/article/abs/pii/S1387181113002679) (CMK-3 4.1/6.8 µF/cm²-BET)
- Pandolfo & Hollenkamp 2006 *J Power Sources* 157:11 — supercap carbons 10 µF/cm²-BET consensus

**Ruggiero 2022 (deck source):**
- *J Catal* 414:33–43 — [10.1016/j.jcat.2022.08.040](https://www.sciencedirect.com/science/article/pii/S0021951722003591). Loading 0.5 mg/cm² CMK-3 on 0.196 cm² GC disk, K₂SO₄ pH 2–12, parallel 2e/4e ORR. **No C_S value.**

**Open literature asks:**
- Bohra 2019 EES `10.1039/c9ee02485a` — cited by Ruggiero ref 71 and Linsey 2025 slide 13, not in Articles/
- Bockris & Reddy "Modern Electrochemistry" Vol 2A — informal citation for "5–100 µF/cm² typical aqueous range"
- Any CMK-3-specific EIS in ORR conditions — appears not to exist in peer-reviewed literature
- Tafel slope analysis cation-pH-Li-K-Cs.xlsx (`missing_data.md` M1) — still owed by group
