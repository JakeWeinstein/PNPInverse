# Phase 6α — investigation summary (2026-05-09)

Concise results from the water-self-ionization landing + L_eff
validation sweep. Long-form planning context is in
`docs/CHATGPT_HANDOFF_26_phase6a_outcome_and_phase6b_scoping.md`.

## TL;DR

* **Phase 6α (water self-ionization) infrastructure landed clean**:
  proton-condition residual on `E = c_H − c_OH` with closure
  `c_OH = K_w_eff / c_H`, default-off, byte-equivalent to baseline
  when disabled. 549 fast tests + 10 slow tests pass. Default-off
  byte-equivalence + a `Kw_eff = 0` reduces-to-baseline regression
  test lock the sign convention.
* **Validation sweep ran 8/8 combos × 13 V_RHE points to convergence
  in 58.5 min** (within plan estimate after the optimized double-
  ladder). Verdict P1 PASS, P2 PASS, **P3 FAIL** (`max_surface_pH =
  10.58 at L=16 µm`, threshold 9.0). Verdict file:
  `StudyResults/fast_realignment_2026-05-08/l_eff_transport_sweep/verdict.json`.
* **Critical mechanism correction**: the Seitz/Mangan group's own
  documents (and Co-Zhang 2019 Angewandte) identify **cation
  hydrolysis at the OHP**
  (`M(H₂O)ₙ⁺ ⇌ M(H₂O)ₙ₋₁(OH)⁰ + H⁺`, Cs⁺ pKa ≈ 4.32 near Cu) — not
  water self-ionization, not sulfate buffering — as the dominant
  cation-dependent buffer for the deck's electrolyte. Phase 6α is
  universal aqueous physics (Kw is always Kw) but is **not the
  primary buffer mechanism the deck data probes**. Phase 6β should
  be cation hydrolysis, not sulfate.

## Results from the L_eff sweep

| Quantity | Baseline (Phase 5α) | Phase 6α (L=16 µm cathodic) | Deck target |
|---|---|---|---|
| `max_surface_pH` cathodic | 13.72 (saturated) | 10.58 | ~6 |
| `cd_plateau` (L=100 µm) | −0.090 mA/cm² (Levich-locked) | −0.737 | −0.18 |
| `cd_plateau` (L=16 µm) | −0.562 | **−4.65** (26× over deck) | (interpolation) |
| Levich slope `log|cd| vs log(1/L_eff)` | 1.000 | **1.005** (preserved) | (n/a) |
| Cathodic peak in `cd` | absent | absent | present at V_RHE ≈ +0.1 V |
| Peroxide-current peak position | absent | V_RHE ≈ −0.20 V | V_RHE ≈ +0.10 V |

## What we learned about the physics

1. **Phase 6α drops surface pH by 3.5 units (14.14 → 10.58) but the
   pH gap to the deck's 4-7 window is NOT closable by transport
   alone.** Surface pH is **L_eff-independent at ≈10.58**: same
   value at L = 100, 66, 21, 16 µm to ≤0.05 units. The water-source-
   vs-ORR equilibrium pins surface pH at a value set by the
   kinetic-transport balance, not by L_eff. Smaller boundary layer
   only scales magnitudes; it doesn't shift pH.

2. **Levich slope is *preserved*, not broken.** The cathodic
   limiting mechanism shifted from H⁺ delivery from bulk (baseline)
   to OH⁻ removal back to bulk (Phase 6α), but both are Levich-like
   `1/L_eff` scalings. The plateau magnitude is now 8-26× the
   baseline H⁺-Levich ceiling.

3. **The peroxide-current peak position (−0.20 V) is shifted ~0.30 V
   more cathodic than the deck (~+0.10 V)**. Root cause: at our
   surface pH 10.58, the `c_H⁴` stoichiometry penalty on R_4e is
   ~16 OOM more severe than at deck-like pH 6, so R_4e/R_2e
   crossover happens at much more cathodic V than the deck shows.
   *Same defect* drives the missing `cd` cathodic peak: a soft
   `c_H` ceiling at high local pH lets the BV exponential always
   win over the stoichiometry suppression.

4. **Phase 6α's fast-equilibrium closure is quantitatively
   optimistic.** Bulk water dissociation rate ≈ 1.4 mol/m³/s; our
   modeled H⁺-deficit at L=16 µm cathodic demands ≈ 30 mol/m³/s —
   **~20× faster than textbook water-equilibrium can support**. The
   real surface pH may be higher than 10.58 because water can't
   actually keep up. Doesn't change Phase 6α correctness as a sub-
   model, but it argues the residual gap to deck is *larger* than
   our 10.58 → 6 estimate.

5. **The actual buffer mechanism (per group documents):**
   `M(H₂O)ₙ⁺ ⇌ M(H₂O)ₙ₋₁(OH)⁰ + H⁺`, with **field-dependent pKa**
   that drops 5-10 units near a polarized cathode. Cation pKa near
   Cu cathode (per Linsey deck / Co-Zhang 2019): Li⁺ 13.16, Na⁺
   11.44, K⁺ 8.49, **Cs⁺ 4.32**. At deck pH 4-6, Cs⁺ is the only
   alkali cation actively buffering. This is what Phase 6β should
   model.

## Bugs caught and locked down

* **Sign error in `build_proton_condition_flux`** (early Phase 6α
  development): helper returned `+J_NP` instead of the residual-side
  `−J_NP`. Newton converged at small k0, stalled at k0 ≳ 5e-4 (where
  transport becomes load-bearing). Fixed; regression locked by
  `tests/test_water_ionization_phase_6a.py::TestKwZeroReducesToBaseline`
  which asserts the residual L²-norm at `Kw_eff = 0` matches the
  default-off baseline byte-for-byte.

* **Cosmetic logging bug in sweep script**: success-path
  `_config_dict` call omitted `enable_water_ionization` flag, so
  `iv_curve.json` config blocks showed `False` even when the
  residual was using it. Fixed in tree; the in-flight sweep's
  iv_curves are mislabeled but the physics is correct.

## Files & pointers

* **Implementation**: `Forward/bv_solver/water_ionization.py` (new),
  edits in `forms_logc.py`, `forms_logc_muh.py`,
  `anchor_continuation.py`, `config.py`, `scripts/_bv_common.py`,
  `scripts/studies/l_eff_transport_sweep_csplus_so4.py`.
* **Tests**: `tests/test_water_ionization_phase_6a.py` (35 cases).
* **Outputs**:
  `StudyResults/fast_realignment_2026-05-08/l_eff_transport_sweep/`
  (per-combo `iv_curve.json` + `summary.json` + `verdict.json`).
* **Baseline (preserved)**:
  `StudyResults/fast_realignment_2026-05-08/l_eff_transport_sweep_baseline_phase5_alpha_failure/`.
* **Plots**:
  `StudyResults/fast_realignment_2026-05-08/l_eff_transport_sweep_phase6a_final.png`
  (4-panel × 2-ratio overlay vs baseline).
* **Phase 6β planning context**:
  `docs/CHATGPT_HANDOFF_26_phase6a_outcome_and_phase6b_scoping.md`
  §9 (corrected mechanism: cation hydrolysis, NOT sulfate, NOT
  carbonate). Read that handoff before scoping the next plan.
* **Group's own writeup of the buffer mechanism**:
  `data/EChem Reactor Modeling-Seitz-Mangan/Trienens_Report_2025/20250818-ACS-CATL-EChem Rxn Enviro for ORR-LSeitz.pptx`
  slide 27 (Linsey deck cation pKa table); CESR_Report_2022 §pH-
  buffering; `Articles/2019-Co-Zhang-Direct Evidence ...` (the
  methodological reference).
