## Project's existing citation trail for C_S = 0.10 F/m²

Internal audit of where the project's `stern_capacitance_f_m2 = 0.10` (= 10 µF/cm²) default value comes from, what the Seitz/Mangan data folder cites for it, and what (if anything) the group's own published / draft work says about a CMK-3 specific value.

## Where does the value come from in the project?

**First appearance in the code/docs (chronological):**

1. `docs/solver/stern_layer_physics_and_next_steps.md` line 204 (first committed in `247c56e`, 2026-05-05, "docs: investigation log + Bikerman/Stern/muh research notes"). The doc is dated **2026-05-03**. It proposes the sweep range:

   ```
   C_S = [None, 0.05, 0.10, 0.20, 0.40, 1.00] F/m²
   ```

   with inline justification (lines 207–214): "1 F/m² = 100 µF/cm². So the range above covers roughly 5 to 100 µF/cm², with 0.10–0.40 F/m² being a reasonable compact-layer scale to inspect." No literature citation — it is presented as a sweep design, not a chosen value.

2. `scripts/studies/peroxide_window_stern_test.py` (committed in `4026b70`, 2026-05-05, "feat(scripts): study + verification scripts for the 2b workstream"). Lines 80–81: `CS_DEFAULT = (None, 0.05, 0.10, 0.20, 0.40, 1.00)`. Commit message itself acknowledges: "stern_test: finite Stern capacitance sweep (informs the 0.10 F/m^2 [default])".

3. `docs/ic_studies/4sp_bikerman_ic_option_2b_results.md` (later in commit `793b73e`) records the post-sweep outcome: with `Stern 0.10 F/m²` the **3sp + bikerman + muh** stack achieves **15/15 clean convergence to V_RHE = +1.00 V**, while no-Stern fails. This is the empirical result that locked 0.10 in as the *de facto* production default.

4. The value then propagates to **23+ call sites** via the `stern_capacitance_f_m2=0.10` keyword: `scripts/diagnose_db_ic_distance.py:333`, `scripts/studies/l_eff_transport_sweep_csplus_so4.py:194,230`, `scripts/studies/parallel_2e_4e_warmstart_probe.py:117,151`, plus 17 test files (`test_grid_per_voltage_helpers.py`, `test_anchor_continuation.py`, `test_phase6b_v9_gate{1,2,3,4}_*`, `test_phase6b_v10a_langmuir_cap.py`, `test_water_ionization_phase_6a.py`, `test_l_eff_smoke.py`, `test_steric_saturation.py`). The default in `scripts/_bv_common.py:539, 1082` is still `None`; the **literal `0.10` is supplied by callers**.

**Self-described provenance** (verbatim, `docs/phase6/phase6b_next_steps_plan.md` §5.3, lines 633–652):

> Audit result: `stern_capacitance_f_m2 = 0.10` has no Ruggiero 2022 or Linsey 2025 deck citation. Verified by full-text grep of `docs/Ruggiero2022_JCatal_manuscript.pdf`: the paper has no Stern, capacitance, F/m², or μF/cm² value beyond methods-section "capacitive current subtraction"; it has no PNP modeling section and cites Bohra 2019 (ref 71, EES 12 11) for the modified Poisson-Boltzmann modeling approach.
>
> Actual provenance of 0.10 F/m²:
> - `docs/stern_layer_physics_and_next_steps.md` (2026-05-03) line 214 lists C_S = [0.05, 0.10, 0.20, 0.40, 1.00] F/m² as the sweep design, drawn from the **textbook compact-layer scale of 5–100 µF/cm² (= 0.05–1.0 F/m²); 10 µF/cm² is the low end of Bockris/Reddy's typical aqueous range**.
> - The May 2026 Stern sweep (`docs/4sp_bikerman_ic_option_2b_results.md`) selected 0.10 as the smallest finite-Stern value that allowed Newton to cross the +1.0 V wall on the 4sp bikerman stack.
> - The value has therefore been a **convergence-pinned engineering choice**, not a deck-calibrated parameter.

`docs/phase6/missing_data.md` M2 (lines 92–123) confirms the same: "the production solver currently uses `C_S = 0.10 F/m²` which has **no Ruggiero / deck citation** … It traces to `docs/stern_layer_physics_and_next_steps.md` (2026-05-03) line 214 — a sweep design picked from the textbook 5–100 µF/cm² range; the May 2026 sweep selected 0.10 because it was the smallest finite-Stern that crossed the +1.0 V wall. It's a convergence-pinned engineering choice, NOT a deck-calibrated parameter."

`docs/realignment/m0_target_extraction.md:158` lists Stern capacitance (carbon) at 0.05–0.5 F/m² (5–50 µF/cm²) with authority just labeled **"Memory; production default 0.10 F/m² is mid-range"** — i.e. unsourced.

## What does the data folder cite for C_S?

Full-text grep across `data/EChem Reactor Modeling-Seitz-Mangan/` for "capacit", "stern", "µF", "F/cm", "F/m", "10 µF", "inner layer", "Helmholtz" (PDF + DOCX + PPTX text extraction, including pptx XML strings):

| File | Stern / C_S content |
|---|---|
| `Linsey/20200407_Electrochemical Double Layer Modeling_LSeitz.pdf` | Qualitative deck — covers Stern layer concept and "ORR in pH-adjusted K2SO4 electrolyte" baseline; no numeric C_S. |
| `Trienens_Report_2025/20250818-ACS-CATL-EChem Rxn Enviro for ORR-LSeitz.pptx` (Linsey 2025 ACS-CATL deck) | Slide 12 "Stern Modification (1924) Inner & Outer Helmholtz Planes ‡ Prevents ions from getting arbitrarily close to electrode. Width depends on permittivity of electrolyte and concentrations of various species." Slide 13 cites Bohra 2019 EES 12(11):3380–3389 for the modified PNP+BV approach. **No numeric C_S anywhere in the deck.** |
| `Trienens_Report_2025/Mangan_SeitzProposal 1.pdf` | No Stern/capacitance hits. |
| `Trienens_Report_2025/Trienens_Report_2025_Seitz_Mangan.docx` | No Stern/capacitance/CMK hits via docx XML grep. |
| `CESR_Seed_Proposal_Mangan_Seitz_final.pdf` | Qualitative: "double layer (110 nm of the catalyst surface)" — describes the EDL region width, no C_S. |
| `Books & Lectures/EChemCourse_Seitz_NU_L5 Electrode Kinetics - Double Layer.pdf` (Seitz's own NU lecture) | Educational — "Stern Layer: layer of counter ions that shield electrode surface charge", "Inner Helmholtz Layer", "Capacitive current". No specific C_S numerical value cited for carbon. |
| `Books & Lectures/EChemCourse_Tang_Drexel_Double Layer Structure 20160125.pdf` | Educational; covers Helmholtz / Gouy-Chapman / Stern models with parallel-plate capacitor formulas, but explicitly notes "this is how Grahame's data was measured … only works for mercury and other liquid electrodes" — Hg-specific, not carbon. |
| `Books & Lectures/Electrochemical Methods, Fundamentals & Applications - Bard & Faulkner (2Ed).pdf` | Chapter 13 ("Double-Layer Structure and Adsorption") covers EDL theory, but a targeted grep across the text-extractable pages for carbon-with-numeric-µF pairings turned up nothing definitive. The Tang slides cite Bard Chapter 13 for Helmholtz/Gouy-Chapman/Stern formalism (a generic textbook reference, not a CMK-3 measurement). |
| `Articles/2019-Co-Zhang-…-Angewandte.pdf` (referenced in Phase 6β plan) | "Buffer capacity" hits only; no numerical C_S or Stern capacitance value for carbon. |
| `Articles/` (all 10 PDFs) | **No Bohra 2019 EES paper present.** Bohra 2019 is cited in the deck (slide 13) and Ruggiero 2022 (ref 71) but **not in the data drop** — the Phase 6β plan still flags it as the next literature anchor to chase (`docs/phase6/phase6b_next_steps_plan.md:656`). |
| `Brianna/` (literature-notes.pptx, Brianna Research Intro ORR.pptx, all docx) | "OHP" and "outer Helmholtz plane" discussions (cation-coverage effects, citing Resasco/Bell 2017 and Strmcnik 2009), but no C_S value. CMK-3 is named as the catalyst material; the only quoted Stern parameter is conceptual. |
| `Yash-Trends/Data and Plotting.zip` + `Yash-Trends/Results/Stern_*.svg`, `L_Stern.svg` | Yash's PNP+BV solver uses **Stern *thickness*** `L_Stern` (not capacitance), with values `{0.6 nm, 0.8 nm, 1 nm}` shown on the sensitivity SVGs (plus a wider range `7.6e-6 / 7.6e-5 / 7.6e-4` m in `L_bulk_Disk.svg`, possibly Stokes/diffuse layer). The plotting notebook `plotting.ipynb` reads `Tafel slope analysis cation-pH-Li-K-Cs.xlsx` (= missing-data item M1) and does not define a Stern capacitance. |
| `Reaction modeling overleaf documents/reaction_modeling_*.pdf` (Jithin's docs, 4 files, 2020-04 era) | No capacitance / Stern / µF / F/m / EDL hits via pdftotext. |

**Bottom line on the data folder:** zero numeric C_S value attributable to CMK-3 or even to carbon electrodes generically. The deck and lectures describe the Stern layer *concept* (with Bohra 2019 cited as the modeling reference), and Yash's solver instead parameterizes a Stern *thickness* in nanometers rather than a capacitance. The missing `Tafel slope analysis cation-pH-Li-K-Cs.xlsx` (`missing_data.md` M1) and the **absent-from-folder Bohra 2019 EES paper** are the two upstream literature anchors the project would chase if calibrating C_S.

## What does the Seitz/Mangan group's own published work say?

- **Ruggiero 2022 (J. Catal., the peer-reviewed source paper):** Explicit full-text check confirms (`docs/phase6/phase6b_next_steps_plan.md:634-639`): **no Stern, no specific capacitance, no µF/cm² or F/m² value** beyond standard CV methodology ("capacitive current subtraction" between 0.8–1.0 V vs RHE). The paper has no PNP/Stern modeling section. Ruggiero ref 71 = Bohra 2019 EES, cited for the modified PB approach.

- **Linsey 2020 deck (`20200407_Electrochemical Double Layer Modeling_LSeitz.pdf`):** Qualitative EDL pedagogy. No numeric C_S.

- **Linsey 2025 ACS-CATL deck (Trienens_Report 2025 dir, `20250818-ACS-CATL-EChem Rxn Enviro for ORR-LSeitz.pptx`):** Slide 12 names the Stern modification; slide 13 cites Bohra 2019 EES, Borukhov-Orland PRL 79 1997, Kilic-Ajdari PRE 75 2007, Butt-Hartkamp Sust. Energy & Fuels 7 2023 as the modified PB lineage. **No specific C_S F/m² value on any slide.** The deck's quantitative numbers are concentrations (`[SO₄²⁻]=0.1 M`, `[K⁺]=0.2 M`), per-cation pKa-near-cathode (Li 13.16, Na 11.44, K 8.49, Cs 4.32 — Singh 2016 Cu values reproduced), and Tafel kinetics — not C_S.

- **Trienens 2023 award letter + Trienens 2025 report (DOCX):** No Stern/capacitance hits. The report covers boundary-layer thickness / rotation-rate dependence (Levich) for mass-transport context, not the compact layer.

- **Brianna's docs (intro deck, lit-notes deck, mechanism / units / Tafel / conv-diff docx files):** Mentions OHP / cations / hydration sheath qualitatively; no numeric C_S for CMK-3.

- **CESR Seed Proposal + Mangan_SeitzProposal:** Describe the inner layer as "<1 nm" and the screening layer as "~50 nm" qualitatively (per `CONJECTURE_AUDIT_2026-05-09.md` line 21); no numeric C_S.

- **Jithin's 2020 overleaf modeling docs (reaction_modeling_*.pdf, 4 files):** No EDL-capacitance content extractable.

- **Yash's modeling code** (cross-validation reference per `MEMORY.md:reference_yash_modeling_code.md`): Parameterizes Stern as a **thickness `L_Stern`** ({0.6, 0.8, 1.0 nm} in the SVG sensitivity sweeps), not as a capacitance. This is a different functional form than the project's `C_S = σ_metal / ψ_Stern` Robin BC.

## Gap analysis

**What is documented (and consistent):**
- The 0.10 F/m² default originated as a sweep midpoint in `docs/solver/stern_layer_physics_and_next_steps.md` on 2026-05-03 (textbook 5–100 µF/cm² range, attributed loosely to "Bockris/Reddy's typical aqueous range").
- It was locked in as the production default because it was the smallest finite-Stern value that let the 3sp+bikerman+muh stack cleanly cross the +1.0 V wall in the May 2026 stern sweep (15/15 vs no-Stern's failure).
- Multiple downstream docs (CONJECTURE_AUDIT 2026-05-09 §"Stern capacitance = 0.10 F/m²"; missing_data.md M2; PHASE_6B_V9_PHASES_A_B_RESULTS line 448; phase6b_next_steps_plan §5.3, §L7) all label this as **uncited / labelled tunable**.
- A Σ_S scale-consistency check (`PHASE_6B_V9_PHASES_A_B_RESULTS_2026-05-10.md:149-160`) flags that the project's `C_S = 0.10 F/m² = 10 µF/cm²` is **5× smaller than Singh 2016's Cu calibration C_dl ≈ 51 µF/cm²**, and reaching Singh's tabulated σ_S = 226 µC/cm² would require Δφ_Stern ≈ 22.6 V — unphysical. This is the project's strongest internal evidence that **0.10 F/m² may be wrong by 5×** (logged as Risk #5 in `PHASE_0_ACCEPTANCE_BUNDLE_LOCK_2026-05-10.md:205`).
- Sensitivity-only mitigation plan: bracket sweep `C_S ∈ {0.05, 0.10, 0.20} F/m²` at v10b/Gate 4B (`missing_data.md` M2; phase6b_next_steps_plan §5.3); also an alternative bracket `{10, 30, 50} µF/cm²` proposed at Phase 0 risk register.

**What is missing:**
- **No CMK-3-specific C_S measurement** in any Seitz/Mangan group document. The group's published work (Ruggiero 2022) does not measure or report it; the deck describes the Stern concept and cites Bohra 2019 for the modeling form; Brianna's slides and Trienens reports are silent on numeric C_S.
- **Bohra 2019 EES 12(11):3380–3389** — the closest CO₂RR PNP+BV literature reference, cited by both the deck and Ruggiero — is **not in the data folder Articles/ directory**, even though it's named as the closest C_S anchor by phase6b_next_steps_plan §5.3. Per `missing_data.md` M2, it would be the natural literature anchor to chase but "doesn't directly transfer (different cathode material)".
- **No published experimental EIS / capacitance number for CMK-3 ORR conditions** in the data folder — neither the K₂SO₄ pH-4 RRDE deck baseline nor any sweep across the cation series provides a measured C_dl or C_S.
- **Functional-form ambiguity:** Yash's solver uses Stern *thickness*; the current project uses Stern *capacitance*. The two are related (C_S ≈ ε_r·ε_0 / L_Stern); a quick check at `L_Stern = 0.6 nm` with ε_r ≈ 6 (oriented water at the IHP, standard Conway value) gives C_S ≈ ε_0·6 / 0.6e-9 ≈ 0.088 F/m² ≈ 8.8 µF/cm² — close to 0.10 F/m² but the project doesn't actually cite this derivation anywhere.

## Key Takeaways

- `stern_capacitance_f_m2 = 0.10 F/m²` (= 10 µF/cm²) entered the codebase on **2026-05-05** via `docs/solver/stern_layer_physics_and_next_steps.md` (commit `247c56e`, doc dated 2026-05-03) and the companion `peroxide_window_stern_test.py` study script (commit `4026b70`). It was a **sweep midpoint**, not a chosen value.
- It became the production default because it was the smallest finite-Stern value that crossed the **+1.0 V Newton convergence wall** on the 3sp+bikerman+muh stack (15/15 convergence per `docs/ic_studies/4sp_bikerman_ic_option_2b_results.md`). The provenance is explicitly a "convergence-pinned engineering choice" per `docs/phase6/phase6b_next_steps_plan.md:651`.
- **Zero numeric C_S value** for CMK-3 (or even generic carbon) anywhere in the Seitz/Mangan data folder, after grepping all PDFs, DOCX files, and pptx XML.
- Ruggiero 2022 (the deck's peer-reviewed source paper) has **no Stern modeling content** beyond methods-section "capacitive current subtraction"; it cites Bohra 2019 EES for the modified PB form. **Bohra 2019 itself is not in the Articles/ folder** and remains an open literature ask.
- The project's own σ-scale consistency check vs Singh 2016 Cu calibration (`PHASE_6B_V9_PHASES_A_B_RESULTS_2026-05-10.md:149-160`) suggests **0.10 F/m² may be ~5× too low**; this is logged as Risk #5 in the Phase 0 acceptance bundle. The current mitigation is a sensitivity bracket sweep `{0.05, 0.10, 0.20}` or `{10, 30, 50} µF/cm²` at v10b/Gate 4B, not a recalibration.
- For the literature search proper, the natural anchors to chase are: **Bohra 2019 EES** (closest CO₂RR PNP+BV precedent — different cathode material though); **Bockris & Reddy** "Modern Electrochemistry" (vaguely cited as the "5–100 µF/cm² typical aqueous range"); any CMK-3-specific EIS measurement (none in the data folder); and the open `Tafel slope analysis cation-pH-Li-K-Cs.xlsx` (M1) which the group still owes the project.
