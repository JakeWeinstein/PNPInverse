# CMK-3 Stern Capacitance — Literature Note

**Status:** locked recommendation (2026-05-10). Acceptance-bundle
step 7 of the "v10a → E sequence"
(`PHASE_0_ACCEPTANCE_BUNDLE_LOCK_2026-05-10.md`).

**Question.** What value should the project use for the local Stern
compact-layer capacitance ``stern_capacitance_f_m2`` (the residual
parameter in `Forward/bv_solver/forms_logc{,_muh}.py`) on the
Seitz/Mangan CMK-3 deck (K₂SO₄ pH 4–6, parallel 2e/4e ORR)?

**Recommendation.** **`C_S = 0.20 F/m² (= 20 µF/cm²)`**, derived
from Stern thickness `L_S = 5 Å` and Booth-saturated permittivity
`ε_S = 11.3`:

```
C_S = ε_S · ε_0 / L_S = 11.3 · 8.854e-12 F/m / 5e-10 m
    = 0.200 F/m² = 20 µF/cm²
```

Backed by three independent PNP-modeling sources that converge on
this value: **Bohra-Koper-Choi consensus**.

Full research trail (literature search, codebase provenance, web
review of CMK-3 + ORR carbon EDLC ranges) in
`.research/cmk3-stern-capacitance/SUMMARY.md` and four supporting
notes in the same directory.

---

## 1. Citation chain for `C_S = 0.20 F/m²`

| Source | DOI / Link | Contribution |
|---|---|---|
| **Bohra et al. 2024** *JPC C* | [PMC11215773](https://pmc.ncbi.nlm.nih.gov/articles/PMC11215773/) | "Experimentally found values of C_Stern are often reported in the range of 20 to 25 µF cm⁻²." |
| **Choi et al. 2024** *JPC C* 128, 27, 11075 | [10.1021/acs.jpcc.4c03469](https://pubs.acs.org/doi/10.1021/acs.jpcc.4c03469) | Explicit derivation: `L_S = 5 Å`, `ε_S = 11.3` ⇒ `C_S = 20 µF/cm²`. |
| **Pillai et al. 2024** "Surface Charge BC Often Misused" *JPC C* | [10.1021/acs.jpcc.3c05364](https://pubs.acs.org/doi/10.1021/acs.jpcc.3c05364) | Methodological critique: flags 100–200 µF/cm² CO2R-modeler convention as wrong; identifies 20–25 µF/cm² as experimentally-grounded "safe band" (10–50 µF/cm² wider envelope). |
| **CatINT** (Ringe/Bell, Stanford) | [github.com/sringe/CatINT](https://github.com/sringe/CatINT/blob/master/docs/source/tutorials/co2r_au_catmap/catint_input.rst) | Default config: `Stern capacitance: 20. # micro F/cm2`. |
| **Kilic, Bazant, Ajdari 2007** *Phys Rev E* 75:021503 | [arXiv physics/0611030](https://arxiv.org/pdf/physics/0611030) | Foundational mPNP-Stern formulation; phenomenological band 10–50 µF/cm². |

The factor-~10 drop from bulk water `ε ≈ 78` to Stern
`ε_S ≈ 11` reflects **dielectric saturation** via Booth (1951): at
`η ~ 1 V` across 0.5 nm, `E ~ 2 × 10⁹ V/m`, water dipoles are
fully aligned. This is the saturation step that distinguishes the
"grounded" 20–25 µF/cm² consensus from the "supercharged" 100–200
µF/cm² convention.

## 2. Three regimes (per Pillai 2024)

1. **Grounded (`C_S ≈ 20–25 µF/cm² = 0.20–0.25 F/m²`).** `L_S = 5
   Å`, `ε_S ≈ 11` (Booth-saturated). Calibrated to Ag(111) SPEIS
   (Choi 2024). **Experimentally anchored. Use this.**
2. **"Supercharged" (`C_S ≈ 100–200 µF/cm² = 1–2 F/m²`).** Assumes
   Stern retains bulk water `ε ≈ 78` over sub-nm thickness. **Pillai
   2024 flags as methodological error. Avoid.**
3. **Phenomenological (`C_S = 10–50 µF/cm² = 0.10–0.50 F/m²`).**
   Kilic-Bazant style — `C_S` left as a model parameter, field-
   averaged effective compact-layer capacitance lumping in
   saturation, image charge, partial charge transfer. The legacy
   `0.10 F/m²` sits at the low edge of this band.

## 3. Caveats that matter for the model (read before changing things)

### 3a. `C_S` is per-local-surface-element, not per-geometric area

The 1D-slab PNP-Stern formulation (Bazant/Kilic/Storey/Jithin
lineage) has **no roughness factor** in the BC. `C_S` in the
residual is treated as a per-local-surface-element compact-layer
capacitance. The deck's `RF ≈ 6000` (= CMK-3 BET ≈ 1200 m²/g ×
loading 0.5 mg/cm²-geom) is *implicit in fitted `k₀`*, never
explicit in the Stern BC.

This means:

- Cross-stack comparison to **Yash** (uses `L_Stern` thickness
  directly, not `C_S`) or **Bohra 2019** (variable `ε_S` via
  Booth) requires explicit RF accounting on the kinetic side.
- A per-geometric interpretation of `C_S = 0.10 F/m²` would need
  ~1500–11000 F/m²-geom (supercap territory); that's not what the
  residual computes. The model is mathematically consistent only
  as a flat-electrode / per-local-surface-element formulation.

### 3b. Singh's 51 µF/cm² Cu is total C_dl, not Stern-only

`PHASE_0_ACCEPTANCE_BUNDLE_LOCK_2026-05-10.md` Risk #5 once flagged
`σ_S = 226 µC/cm² / C_S = 0.10 F/m²` as implying unphysical
`Δφ_Stern ≈ 22.6 V`. This was based on Singh 2016's reported
`51 µF/cm² for Cu` — but that value is an **in-house CV-slope
total `C_dl`** measurement (Stern + diffuse + roughness + specific
adsorption), measured on polycrystalline Cu in NaHCO₃/CO₂R, **not**
a transportable Stern-only value for CMK-3 in K₂SO₄/ORR. Verbatim
from Google Scholar snippet: "the double layer capacitance was
determined by calculating the slope of this graph."

Before treating Risk #5 as load-bearing, re-derive the σ_S
mismatch with **Stern-only** values (Bohra/Koper 20–25 µF/cm² for
metal; ~10–20 µF/cm² for sp²-carbon). At `C_S = 0.20 F/m²` the
mismatch may not even be the right comparison anymore.

### 3c. Carbon-specific narrowing pulls slightly below 20 µF/cm²

For sp² carbon (HOPG basal, CMK-3 pristine + graphitized), the
intrinsic Stern is **slightly lower** than the metal-electrode
consensus because of quantum-capacitance reduction (lower DOS at
E_F) and weaker water polarization. Material-specific values:

| Material | Specific C | Source |
|---|---|---|
| HOPG basal (Li⁺ → Cs⁺, PZC+0.5V) | 4.7–9.4 µF/cm² | Iamprasertkun 2019 *JPCL* [10.1021/acs.jpclett.8b03523](https://pubs.acs.org/doi/abs/10.1021/acs.jpclett.8b03523) |
| Glassy carbon | 10–25 µF/cm² | textbook |
| CMK-3 pristine | 4.1 µF/cm²-BET | Yamada 2013 *Micropor Mesopor Mater* [10.1016/j.micromeso.2013.05.024](https://www.sciencedirect.com/science/article/abs/pii/S1387181113002679) |
| CMK-3 graphitized | 6.8 µF/cm²-BET | Yamada 2013 |
| sp²-carbon Stern (1D-slab) | 0.10–0.20 F/m² | this work, synthesised |

The per-local-surface-element interpretation (caveat 3a) means
these specific-C values are **not** directly comparable to the
residual's `C_S` — they're EDLC measurements that lump roughness.
The Stern-only 1D-slab `C_S` for carbon converges on ~0.10–0.20
F/m².

### 3d. Constant `C_S` is field-averaged

Storey-Bazant 2012 onward, Bohra 2019, and Choi 2024 acknowledge
`ε_S` is field-dependent (Booth 1951). Constant `C_S` is a
**field-averaged effective value**; it cannot capture η-dependence
implicit in Booth. Going to variable-`ε` is out of scope for
v10b; document constant-`C_S` as a first-order approximation.

## 4. Sensitivity bracket for v10b

```
C_S ∈ {0.05, 0.10, 0.20, 0.30} F/m²
```

Spans the carbon-conservative low edge (0.05–0.10), the locked
production target (0.20), and a high-end excursion (0.30) for
sensitivity-bracket sanity. The four values all sit inside
Pillai 2024's "safe band" (10–50 µF/cm²).

## 5. Implementation status

- **Production target locked (2026-05-10):** `C_S = 0.20 F/m²`.
  Live in `scripts/_bv_common.py:make_bv_solver_params` callers,
  driver `scripts/studies/phase6b_v10a_v_sweep_diagnostic.py`
  (`STERN_F_M2_BASELINE`), Phase A.2, step 6 plumbing-ablation.
- **Legacy `C_S = 0.10 F/m²`:** carbon-conservative low rung.
  Pre-2026-05-10 historical StudyResults remain interpretable as
  the bracket low rung; pre-existing convergence record at +1.0 V
  wall (15/15) was at this value.
- **Two-stage anchor pattern** (Phase A.2 + step 6): build the
  anchor at `STERN_F_M2_ANCHOR = 0.10 F/m²` (convergence-friendly
  for the k0/Kw_eff ladder), then runtime-bump to
  `STERN_F_M2_BASELINE = 0.20 F/m²` via
  `set_stern_capacitance_model` plus a Newton resolve. Needed
  because `c_s_ladder` raises `NotImplementedError` when combined
  with `kw_eff_ladder` in the current
  `solve_anchor_with_continuation` API.
- **v10b unblocked (step 6 plumbing verified):** v10b literature
  calibration of `Γ_max + k_des + C_S` will run with this
  recommendation as the central rung of the C_S bracket sweep.

## 6. Open asks (carry into v10b)

- **Pull Bohra 2019 EES** `10.1039/c9ee02485a` into `Articles/`.
  Cited by Ruggiero 2022 ref 71 *and* Linsey 2025 deck slide 13 —
  the absence is a true open ask. Code at
  [github.com/divyabohra/GMPNP](https://github.com/divyabohra/GMPNP)
  (FEniCS + Bikerman).
- **Re-derive Risk #5 σ_S mismatch** using Stern-only `C_S = 20
  µF/cm²`. If the mismatch persists, treat the Singh `σ_S = 226
  µC/cm²` target as non-transportable from polycrystalline Cu in
  CO₂R to CMK-3 in K₂SO₄/ORR.
- **Yash convention cross-check:** `L_Stern = 0.6 nm` with
  `ε_S = 11.3` (Choi-consistent) ⇒ `C_S = 0.17 F/m²`. With
  `ε_S = 6` (Conway oriented-water) ⇒ `C_S = 0.088 F/m²`. Decide
  which convention the cross-stack equivalence uses.
- **Field-dependent `ε_S` / variable-`C_S`** (Storey-Bazant 2012,
  Bohra 2019): out of scope for v10b. Could land post-v10b if
  Pillai-safe-band sensitivity sweep shows constant-`C_S` is a
  meaningful systematic.

## 7. Cross-references

- Full research trail:
  `.research/cmk3-stern-capacitance/SUMMARY.md` (this doc is a
  publishable distillation; the SUMMARY has the agent-by-agent
  evidence, contradictions, and codebase grep results).
- Supporting notes in the same dir:
  `literature-pnp-bv-stern-ranges.md`,
  `codebase-cs-provenance.md`, `web-cmk3-and-orr-carbons.md`,
  `web-porous-carbon-edlc-conventions.md`.
- `CLAUDE.md` hard rule #6 — references this doc + SUMMARY.
- `PHASE_0_ACCEPTANCE_BUNDLE_LOCK_2026-05-10.md` § "Σ_S mapping
  convention (locked default)" + § "Status" step 7 / step 8.
- Original missing-data ask (now resolved):
  `docs/phase6/missing_data.md` M2.
