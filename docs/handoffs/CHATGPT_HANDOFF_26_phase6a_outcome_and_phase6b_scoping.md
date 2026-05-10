# Handoff 26 — Phase 6α outcome (water self-ionization) and Phase 6β scoping

**Status as of 2026-05-09 22:00 CDT — sweep COMPLETE.** Phase 6α
landed end-to-end; the L_eff transport-domain validation sweep ran
all 8 combos × 13 V_RHE points to convergence in **58.5 min** (within
the 45-60 min plan estimate after the optimized double-ladder).
Verdict: **P1 PASS, P2 PASS, P3 FAIL** — `overall_pass = false`, with
the failure being the headline insight, not a regression.

> **22:30 CDT update — Phase 6β mechanism revised.** Earlier sections
> of this doc proposed sulfate buffering (HSO₄⁻ ⇌ SO₄²⁻ + H⁺) as the
> Phase 6β chemistry. **That is incorrect per the Seitz/Mangan group's
> own analysis.** The correct buffer mechanism is **cation hydrolysis**
> at the OHP: M(H₂O)ₙ⁺ ⇌ M(OH)⁰ + H⁺, with field-dependent pKa
> (Li 13.2 / Na 11.4 / K 8.5 / Cs 4.3 near Cu cathode per Ruggiero
> 2022 / Co-Zhang 2019). See §9 "Phase 6β mechanism correction" at the
> bottom of this doc — that section supersedes the sulfate-buffering
> recommendations earlier in the doc.

## 1. What's landed (Phase 6α)

Implementation matches `.claude/plans/formally-write-the-plan-wondrous-seal.md`
(now `docs/PHASE_6A_OH_WATER_IONIZATION_PLAN.md` in spirit):

* **Residual rewrite**: H⁺ NP equation replaced by the proton-condition
  residual on `E = c_H − c_OH` with closure `c_OH = K_w_eff / c_H`.
  Both backends — `Forward/bv_solver/forms_logc.py` and `forms_logc_muh.py`
  — go through the new helper module
  `Forward/bv_solver/water_ionization.py`. Default-off via
  `bv_convergence['enable_water_ionization']`; the disabled path is
  byte-equivalent to the pre-Phase-6α stack (locked down by
  `tests/test_water_ionization_phase_6a.py::TestDisabledPathByteEquivalence`).
* **Kw_eff continuation**: `set_reaction_kw_eff_model` and a
  `kw_eff_ladder` outer loop on `solve_anchor_with_continuation`. After
  bug-hunt the sweep settled on a structure of "full k0 ladder once at
  Kw_eff=0, then a single SS at production k0 per Kw_eff>0 rung" —
  ~9 SS solves per anchor instead of ~25. See
  `Forward/bv_solver/anchor_continuation.py:780+`.
* **Sweep CLI**: `scripts/studies/l_eff_transport_sweep_csplus_so4.py
  --enable-water-ionization` threads the flag through and applies the
  5-rung ladder `(0, KW_HAT·1e-6, KW_HAT·1e-3, KW_HAT·0.1, KW_HAT)`.
* **Tests**: 549 fast + 10 slow Phase 6α tests pass. The MMS
  convergence test is scoped & skipped with reason — derivation of the
  proton-condition manufactured-solution forcing is queued for
  follow-up (do NOT take its absence as a sign Phase 6α math is
  unverified — `TestKwZeroReducesToBaseline` regression-locks the sign
  convention against the no-water residual).

### A bug we fixed mid-sweep, worth flagging

The first sign error in `build_proton_condition_flux` returned `+J_NP`
instead of the residual-side `−J_NP` (the form convention is
`Jflux = +D·c·∇μ` so `F_res += dot(Jflux, ∇v) dx` matches the weak
form). Newton converged at tiny k0 (transport irrelevant) but stalled
at k0 ≈ 5e-4 because the residual was enforcing reversed transport.
The new `TestKwZeroReducesToBaseline` slow regression catches this
exactly — adding `enable_water_ionization=True, Kw_eff=0` and asserting
the residual L²-norm matches the disabled path. **Future flux helpers
in this codebase must add a parallel test.**

## 2. What we have (verified, combo 1/8)

`StudyResults/fast_realignment_2026-05-08/l_eff_transport_sweep/L100um_ratio_1e-18/iv_curve.json`,
13 V_RHE points on `[-0.40, +0.55]` V, 13/13 converged. Comparison
against the archived baseline at
`l_eff_transport_sweep_baseline_phase5_alpha_failure/`:

| V_RHE | cd_now | cd_base | pH_now | pH_base | Δ pH |
|---|---|---|---|---|---|
| −0.400 | −0.737 | −0.090 | 10.61 | 14.14 | **−3.53** |
| −0.242 | −0.455 | −0.090 | 10.56 | 12.59 | −2.03 |
| −0.083 | −0.313 | −0.090 | 10.54 | 11.03 | −0.49 |
| +0.075 | −0.101 | −0.090 |  9.44 |  9.47 | −0.03 |
| +0.550 | −0.066 | −0.066 |  5.06 |  5.06 |  0.00 |

* Anodic regime byte-equivalent to baseline (cd matches to ≤1e-4).
* Cathodic plateau magnitude jumps **8.2×** at V=−0.40 V; the
  H⁺ Levich ceiling is broken.
* Surface pH at the deepest cathodic V drops from **14.14 → 10.61**.
* Anchor wall: 34 s (vs. baseline ~30 s; the kw>0 rungs each add
  ~0.1-0.3 s of Newton perturbation). Grid wall: 14 min for 13 points.
* Plot: `StudyResults/fast_realignment_2026-05-08/l_eff_transport_sweep_phase6a_combo1.png`.

### Cosmetic logging bug (verify before drawing conclusions)

The success-path `_config_dict` call in
`scripts/studies/l_eff_transport_sweep_csplus_so4.py` line 519 omitted
`enable_water_ionization`, so combo-1's `iv_curve.json` config block
shows `enable_water_ionization: False` even though the residual *is*
using it (`kw=…` heartbeats in the log prove the ladder fired). Fixed
in-tree but the in-flight sweep is reading the old script — combos 2-8
will also have the misleading config field. The actual physics is
correct; only the JSON label is wrong.

## 3. Full sweep results (verified, all 8 combos, 22:00 CDT)

**Verdict** (`StudyResults/.../l_eff_transport_sweep/verdict.json`):

| Gate | Result | Detail |
|---|---|---|
| P1 plateau Levich linearity | **PASS** | slope = 1.0048 (ratio 1e-18), 1.0000 (ratio 1e-30); both inside [0.85, 1.15] |
| P2 no cathodic peak | **PASS** | no interior local minimum of cd at any (L_eff, ratio) |
| P3 max_surface_pH < 9 at L=16 µm | **FAIL** | max pH = **10.583** (ratio 1e-18), **10.580** (ratio 1e-30) |

**Plateau magnitudes** (deck target ≈ −0.18 mA/cm²):

| L_eff | ratio 1e-18 | ratio 1e-30 |
|---|---|---|
| 100 µm | −0.737 (4.1× over) | −0.440 (2.4× over) |
| 66 µm | −1.122 (6.2×) | −0.667 (3.7×) |
| 21 µm | −3.541 (19.7×) | −2.095 (11.6×) |
| 16 µm | **−4.651 (25.8×)** | **−2.749 (15.3×)** |

**Surface pH at deepest cathodic V_RHE = −0.40 V**:

| L_eff | ratio 1e-18 | ratio 1e-30 | baseline |
|---|---|---|---|
| 100 µm | 10.61 | 10.34 | 14.14 |
| 66 µm | 10.60 | 10.34 (≈ same) | 14.04 |
| 21 µm | 10.58 | 10.31 | 13.78 |
| 16 µm | **10.58** | **10.31** | 13.72 |

The **L_eff-independence of surface pH** at deep cathodic V is the
load-bearing physical observation. Compared to the original verdict's
`max_surface_pH = 13.72` at L=16 µm, Phase 6α moves it to 10.58 — a
**3.14-unit drop** but still 1.6 above the threshold. The pH gap to
the experimental window (4-7) is *chemical*, not transport.

Final overlay plot (all 4 L_eff × 2 ratios):
`StudyResults/fast_realignment_2026-05-08/l_eff_transport_sweep_phase6a_final.png`.

## 3a. Earlier prediction history (kept for the postmortem)

**Earlier predictions in this doc were wrong on two of three gates;
keep this section as the authoritative one.** Combos 2 (L=100 / 1e-30)
and 3 (L=66 / 1e-18) are now in:

| | cd_min (mA/cm²) | pH at V_RHE = −0.40 |
|---|---|---|
| L=100 / 1e-18 | −0.737 | **10.61** |
| L=100 / 1e-30 | −0.440 | **10.34** |
| L=66  / 1e-18 | −1.122 | **10.60** |

Two findings change the verdict prediction:

1. **Levich slope is *preserved*, not broken.** `|cd|@L=66 / |cd|@L=100
   = 1.521` vs. pure-Levich `100/66 = 1.515` ⇒ slope ≈ **1.010**. The
   limiting mechanism shifted from H⁺ delivery (baseline) to OH⁻
   removal back to bulk (Phase 6α), but both are 1/L_eff-Levich-like.
   **P1 (Levich linearity) → likely PASS** (was predicted FAIL).
2. **Surface pH is L_eff-independent at ≈10.6.** Comparing L=100 / 1e-18
   (pH 10.61) to L=66 / 1e-18 (pH 10.60) — *same pH, different cd*.
   The water-source-vs-ORR equilibrium pins surface pH at a value that
   doesn't move with L_eff. The L=21 and L=16 combos will almost
   certainly land at pH ≈ 10.5-10.7 cathodically too.
   **P3 (`max_surface_pH < 9` at L=16 µm) → likely FAIL** (was
   predicted PASS).

Revised gate prediction:
- **P1 Levich linearity: PASS** (slope ≈ 1.0 preserved through the
  OH⁻-Levich path).
- **P2 No peak: PASS** (Phase 6α still has no peak-forming kinetics).
- **P3 surface_pH < 9 at L=16 µm: FAIL** at ~10.6; the deck's pH 4-7
  operating window is not reachable by water self-ionization alone.

The original verdict had P1 PASS / P2 PASS / P3 FAIL with `max_surface_pH
= 13.72`. Phase 6α moves P3's `max_surface_pH` from 13.72 → ~10.6
(big improvement) but does not cross the < 9 threshold. **The headline
goal of Phase 6α — surface pH inside experimental window — is NOT
achieved by Phase 6α alone.**

## 4. Gap to deck shape (the actual ask)

What the deck (Ruggiero 2022 / Mangan 2025 / Linsey 2025) shows that
we still don't reproduce:

| Feature | Status | What it needs |
|---|---|---|
| Anodic decay to zero | ✅ matches | (preserved from baseline) |
| Cathodic onset at V_RHE ≈ +0.15 V | ✅ matches | (parallel-2e/4e topology) |
| Plateau magnitude ≈ −0.18 mA/cm² | ❌ 4-15× over | **sulfate buffering (6β)** |
| Surface pH in 4-7 (deck operating window) | ❌ ~10-11 | **sulfate buffering (6β)** |
| Cathodic **peak** at V_RHE ≈ +0.10 V | ❌ monotonic, no peak | **6β to set the (c_H)^n ceiling**, possibly **6δ** for sharpness |
| Cathodic **decay** past peak | ❌ no decay | **6δ alkaline-form ORR**, or local-pH-dependent BV stoichiometry switch |
| Cation-identity-dependent peak height | ❌ not modeled | **6γ** explicit cation-OHP physics |

**Key insight for 6β scoping:** the cathodic peak is *not* a kinetic
artifact. It's the V_RHE at which the cathodic Butler-exponential
growth `exp(−α·n·η/V_T)` first hits the `(c_H/c_H_ref)^n` ceiling
imposed by surface pH. With the surface pH stuck at ~10-11 (water
alone), `c_H ≈ 1e-7 mol/m³` is too soft a ceiling — the BV exponential
wins and cd just grows. With sulfate buffering pinning surface pH
near 6-7, `c_H ≈ 1e-3 mol/m³` is a hard ceiling and the curve peaks.

**Sharpened insight after combos 2-3 came in (21:25 CDT):** The
surface-pH equilibrium under Phase 6α is **L_eff-independent at
~10.6**. This means transport (smaller L_eff) does NOT bring surface
pH down — it only scales the magnitudes. Water-source-vs-ORR pins
the local pH at the value where OH⁻-Levich removal balances ORR's H⁺
consumption. Phase 6β's job is to introduce a **chemical** H⁺ source
that's stronger and equilibrates at lower pH. **Sulfate buffering
isn't optional for deck-shape recovery — it's required to hit the
experimental pH 4-7 window at all.**

## 5. Recommended scope for Phase 6β

Mostly notes for the next-steps agent, not a plan:

* **Closure shape**: `HSO₄⁻ ⇌ SO₄²⁻ + H⁺` with `K_a ≈ 1.0e-2` (pKa₂ ≈ 1.99).
  At pH > 3 SO₄²⁻ dominates; the buffer activates in a narrow window
  around the surface as cathodic flux tries to crash c_H.
* **Two implementation paths**:
  - **Path B (recommended for 6β.1)**: Treat the buffer with an
    *algebraic equilibrium closure* mirroring Phase 6α. Add a buffer
    term `R_buf = k·(c_HSO4 − c_SO4·c_H/K_a)` to the H⁺ flux that
    relaxes toward equilibrium fast. Probably easiest to wire as an
    additional source on the proton-condition residual:
    `J_E gains a buffer flux, Poisson source gains charge from HSO4⁻
    and SO4²⁻ separately`. SO₄²⁻ is *already* in the multi-ion
    Bikerman counterion stack — `DEFAULT_SULFATE_BOLTZMANN_COUNTERION_STERIC`.
    HSO₄⁻ would need to enter as either a third counterion or a fully
    dynamic species.
  - **Path A (more correct, for 6β.2)**: Make HSO₄⁻ a fully dynamic
    Nernst-Planck species with explicit ↔ SO₄²⁻ exchange via a
    finite-rate residual. Roughly the work of Phase 6α scaled up.
* **What 6β should preserve**: every byte-equivalence we have today
  (default-off path, anodic regime, Phase 6α residual at Kw_eff=0).
  The pattern that worked for 6α — guarded helper module + UFL gate
  off the cfg dict + matching slow regression test — should be the
  template.
* **Continuation**: 6β will likely need its own sub-ladder (analogous
  to Kw_eff). Buffer activation is its own stiffness source. Mirror
  the `kw_eff_ladder` design in `solve_anchor_with_continuation`.
* **Phase 6δ ordering**: do *not* try to land 6δ before 6β. The
  alkaline-form ORR rewrite changes the BV reaction set across the
  cathodic regime; without 6β-set surface pH it'll over-correct in
  the wrong direction. 6β first, then evaluate whether 6δ is needed
  for the decay past peak.

## 6. Open questions for the next-steps agent

1. **HSO₄⁻ as dynamic species or third counterion?** The plan's
   §5 deferred-list said "HSO₄⁻ as a separate dynamic species: Phase
   6β" — but a Boltzmann-equilibrium closure (mirror of Phase 6α water
   ionization) might capture the buffering behavior with much less
   solver work. Need to think about whether HSO₄⁻ deviates from
   Boltzmann equilibrium under cathodic load.
2. **Buffer activation rate**: experimentally, is the equilibrium fast
   enough that the algebraic closure is justified? Aqueous sulfate
   protonation is essentially diffusion-limited (k_f ~ 1e10 M⁻¹s⁻¹),
   so the equilibrium-closure path should be valid. Confirm with the
   Ruggiero/Mangan literature before committing.
3. **Cathodic peak verification**: the deck peak position depends on
   Cs⁺ vs K⁺ vs Na⁺ vs Li⁺ identity. Phase 6β alone (sulfate
   buffering) may give a peak at the wrong V_RHE. **Plan a sub-task**
   to compare cd vs V_RHE peak position across the 4 cation choices
   the deck has, against Linsey 2025 ACS-CATL data (referenced in
   `docs/seitz_mangan_data_folder_audit_2026-05-08.md`).
4. **Whether Phase 6α's `cd` overshoot at small L_eff blows up
   Newton**: predicted |cd| at L=16 µm is ~−1.5 to −3 mA/cm². If
   that runs out of mass-transport headroom and crashes Newton mid-
   sweep, Phase 6β becomes urgent — *and* the Phase 6α validation
   sweep needs a smaller cathodic V_RHE band until 6β lands.
   Re-check the running sweep before scoping.

## 7. Files / pointers

* Implementation:
  `Forward/bv_solver/water_ionization.py`,
  `Forward/bv_solver/forms_logc.py`,
  `Forward/bv_solver/forms_logc_muh.py`,
  `Forward/bv_solver/anchor_continuation.py`,
  `Forward/bv_solver/config.py`,
  `scripts/_bv_common.py`,
  `scripts/studies/l_eff_transport_sweep_csplus_so4.py`.
* Tests: `tests/test_water_ionization_phase_6a.py`.
* Outputs (after sweep finishes):
  `StudyResults/fast_realignment_2026-05-08/l_eff_transport_sweep/`
  (per-combo `iv_curve.json`, `summary.json`, `verdict.json`).
* Baseline preserved at:
  `StudyResults/fast_realignment_2026-05-08/l_eff_transport_sweep_baseline_phase5_alpha_failure/`.
* Plot of combo 1:
  `StudyResults/fast_realignment_2026-05-08/l_eff_transport_sweep_phase6a_combo1.png`.
* Sweep log:
  `StudyResults/fast_realignment_2026-05-08/l_eff_transport_sweep_phase6a_run.log`.

## 8. What should be the next plan deliverable

Either:
- **Plan 6β.1 (algebraic-buffer)**: HSO₄⁻ ⇌ SO₄²⁻+H⁺ as an algebraic
  closure on the proton-condition residual, mirroring Phase 6α. Scope
  ~3-5 days. Should yield the cathodic peak and bring magnitudes
  within 2× of deck.
- **Plan 6β.2 (dynamic-buffer)**: HSO₄⁻ as a fully dynamic NP species
  with finite-rate equilibration. Scope ~5-7 days. Same expected
  outcome but more degrees of freedom.

Recommend starting with 6β.1 unless the next agent identifies a
specific physics reason the algebraic closure is invalid. The full
sweep should finish before the next plan lands; verdict.json + the
L=16 µm cd/pH numbers should be the empirical input that decides
whether 6β alone is sufficient or 6δ also needs scoping.

## 9. Phase 6β mechanism correction — CATION HYDROLYSIS, not sulfate

**Read this section before §5-§8 above; it supersedes them.**

After the sweep finished, I checked the Seitz/Mangan group's own docs
in `data/EChem Reactor Modeling-Seitz-Mangan/` for explicit discussion
of buffer chemistry. They have a clear, documented hypothesis — and
it's not sulfate.

### Verbatim from group's documents

**Linsey ACS-CATL deck 2025**
(`Trienens_Report_2025/20250818-ACS-CATL-EChem Rxn Enviro for ORR-LSeitz.pptx`,
slide 27):

> "If the **pKa of a hydrated cation is lower than local pH at cathode,
> hydrated cations will act as pH buffers**. Hydrated cations with
> larger radii (Cs+) → have much lower pKa near charged cathode surface
> (~4.3) → and will more effectively buffer lower pH environments."

With a table of cation pKa near a Cu cathode:

| Cation | Bulk pKa | pKa near cathode |
|---|---|---|
| Li⁺ | 13.6 | 13.16 |
| Na⁺ | 14.2 | 11.44 |
| K⁺ | 14.5 | 8.49 |
| Cs⁺ | 14.7 | **4.32** |

**Seitz/Mangan CESR Report 2022**
(`CESR_Report_2022_Seitz_Mangan_v1.docx`):

> "different cations provide variable pH-buffering, thereby modulating
> overall magnitude and onset of local pH changes. The ability of
> cations to buffer the local pH more strongly is dictated by the
> **hydrolysis pK_a of water molecules within the hydration shells of
> each cation**"

**Brianna's literature notes**
(`Brianna/literature-notes.pptx`, summarizing Co-Zhang 2019, Angew.
Chem., included as `Articles/2019-Co-Zhang-Direct Evidence...`):

> "pKa of hydrolysis of the cation [...] **dramatically decreases near
> a polarized cathode surface due to the increase in electrostatic
> interaction**"

### The actual buffer chemistry

Equilibrium:

```
   M(H₂O)ₙ⁺  ⇌  M(H₂O)ₙ₋₁(OH)⁰  +  H⁺            (in OHP)
        Ka_M = [M(OH)][H⁺] / [M(H₂O)]
```

The hydrated cation acts as a Brønsted acid. A water molecule in the
cation's hydration shell deprotonates to release H⁺. The pKa of this
hydrolysis is ~13-14 in bulk solution but **drops by 5-10 units near
a polarized cathode** because the cathodic field stabilizes the
deprotonated form (electrostatic interaction with the negative charge
on the deprotonated cation).

**Cation-identity scaling** (pKa near cathode, per the deck):
Cs⁺ (4.3) << K⁺ (8.5) < Na⁺ (11.4) < Li⁺ (13.2). At local pH 5-6
(deck operating window), Cs⁺ is the only cation with pKa BELOW the
local pH, so Cs⁺ is the only one actively buffering. K⁺/Na⁺/Li⁺ have
pKa > local pH and don't deprotonate — they're inert observers in
the OHP.

### Why this changes Phase 6β from §5

The §5 plan was wrong on three counts:

1. **Wrong reaction.** §5 said HSO₄⁻ ⇌ SO₄²⁻ + H⁺ at pKa 1.99. The
   correct reaction is the cation hydrolysis above, at a pKa that
   depends on cation identity *and* surface field strength.
2. **Wrong reservoir.** §5 said HSO₄⁻ is a transported species with
   bulk concentration ~1 mol/m³ at pH 4 (1% of total sulfate),
   diffusion-limited at small L_eff. The cation hydrolysis reservoir
   is the OHP cation population — which is *already in our Bikerman
   counterion stack* (Cs⁺ at c_bulk = 199.9 mol/m³, packed at the
   OHP). No new transported species needed.
3. **Wrong cation behavior.** §5 made cation identity an independent
   Phase 6γ concern. It's not — cation identity *is* the buffer. The
   Cs⁺/K⁺/Na⁺/Li⁺ trend in the deck is direct evidence of varying
   pKa_M near cathode.

### Revised Phase 6β scope

The architectural template is still Phase 6α (proton-condition
residual + analytic equilibrium closure on the H⁺ activity), but the
*closure expression* is different:

* Pre-Phase-6α (baseline): no chemical H⁺ source. Surface pH → 14.
* Phase 6α (water): `c_H = K_w / c_OH(y)`, fast equilibrium with
  c_OH determined by transport. Surface pH → 10.6.
* **Phase 6β (cation hydrolysis):** add a NEW H⁺ source from cation
  hydrolysis local to the OHP. The closure looks like

  ```
  Total H⁺ source rate density(y) =
       water-source rate (Phase 6α)
     + sum over cations: Ka_M(φ_local) · c_M(y)·θ_M(y)·γ(...)
       where γ enforces equilibrium with local c_H
  ```

  with `Ka_M(φ_local)` being the cation's intrinsic pKa shifted
  by the local potential drop relative to bulk. The functional form
  for `Ka_M(φ)` is in Co-Zhang 2019 §3 (use that as the citation
  contract).

  Because the cation already lives in the OHP via the Bikerman
  closure, this introduces **no new transported species**. The
  reaction is local-only.

### Implementation pointers

1. The cation entries in `scripts/_bv_common.py` already carry
   `c_bulk_nondim`, `a_nondim`, etc. Add a `pKa_near_cathode_default`
   field per cation (Cs⁺ 4.32, K⁺ 8.49, Na⁺ 11.44, Li⁺ 13.16) so the
   field-dependent shift can be parameterized.
2. The `bv_convergence` cfg gets a new flag
   `enable_cation_hydrolysis: bool = False` (default off, mirror
   Phase 6α's plumbing).
3. The proton-condition residual in `water_ionization.py`
   (rename to `local_h_sources.py` or similar) gains an additive
   cation-hydrolysis source term. The OH⁻ closure stays — they
   coexist; H⁺ now has TWO equilibrium-controlled sources.
4. **Continuation:** parameterize the cation pKa shift as a ramped
   parameter analogous to Kw_eff. Ramp the shift from 0 (bulk pKa)
   up to the near-cathode value over 4-5 rungs to give Newton a
   continuous path.
5. **Validation gate:** run the Phase 6β-enabled sweep with all four
   alkali sulfate counterions (Cs/K/Na/Li, holding the rest constant)
   and check that surface pH at L=16 µm scales:
   pH(Cs) ~ 5-6 < pH(K) ~ 7-8 < pH(Na) ~ 9-10 < pH(Li) ~ 10.5.
   That's a falsifiable prediction tied directly to the deck's data.

### Caveat on Phase 6α itself (added 22:55 CDT)

A folder grep also revealed that water self-ionization (the Phase 6α
mechanism we just landed) is **not actually endorsed by the group's
documents** as the relevant buffer for this deck. Group docs and 9
relevant articles contain zero substantive mentions of water
dissociation / Kw / `H₂O ⇌ H⁺ + OH⁻` as a buffer source for ORR.
MacDougall-Gupta 2005 (the closest analog — surface concentrations
under cathodic load with carbonate buffer) tracks CO₂/HCO₃⁻/CO₃²⁻/H⁺
equilibria but explicitly does NOT list water self-ionization as a
modeled reaction.

This isn't fatal — Kw=1e-14 is universal and including it isn't
wrong. But it argues that:

1. Phase 6α is a **sub-leading** contribution to the actual deck
   chemistry, not the dominant one.
2. The Phase 6α modeled rate (fast equilibrium) may be quantitatively
   too generous: textbook bulk water dissociation rate ~1.4 mol/m³/s
   vs. our model's effective demand at L=16 µm ~30 mol/m³/s. A
   slower closure (kinetic-rate-limited water dissociation) would
   give a smaller water contribution and a higher predicted surface
   pH — which would actually *strengthen* the case for cation
   hydrolysis as the missing physics.
3. **Don't conclude from Phase 6α's surface pH = 10.58 that
   "transport is set; just buffer chemistry is missing."** The 10.58
   number is a model output that depends on the fast-equilibrium
   assumption being approximately valid. If it's not, the residual
   gap is *larger* than 10.58 → 4-7. Plan empirical validation:
   the IrOx-ring local pH probe data Brianna referenced (Linsey deck
   slides 5-9) is the ground truth — get the in-situ-measured local
   pH at known cathodic currents in known electrolytes and compare.

### Other physics pieces that go AWAY now

* §5 "Phase 6β.2 dynamic HSO₄⁻": **out of scope.** Sulfate is *not*
  the buffer. Keep SO₄²⁻ inert in the multi-ion Bikerman stack as
  it is.
* §5 "Phase 6γ cation-dependent selectivity": **subsumed by Phase
  6β.** The cation-dependent peak height the deck shows IS the
  cation-pKa-dependent buffering, not a separate selectivity
  mechanism.
* §5 "Phase 6δ alkaline-form ORR": still possibly relevant for the
  cathodic decay past peak, but **defer until Phase 6β empirics
  decide**. With Cs⁺ buffering the local pH near 4-5, the cathodic
  decay may emerge naturally without needing alkaline-form rate
  laws.

### Empirical priorities for Phase 6β planning agent

* **Read first**: `Articles/2019-Co-Zhang-Direct Evidence of Local pH
  Change and the Role of Alkali Cation during CO₂ Electroreduction
  in Aqueous Media-Angewandte.pdf`. This is the methodological
  reference for both (a) the cation-hydrolysis-buffer mechanism
  and (b) the IrOx-ring-RRDE local pH probe used to validate it.
* **Read second**: Linsey 2025 ACS-CATL deck slides 8, 9, 27 — has
  the pKa-near-cathode table for Li/Na/K/Cs and the experimental
  protocol.
* **Read third**: Ruggiero 2022 J.Catal. (`docs/Ruggiero2022_JCatal_manuscript.pdf`,
  the source paper underlying the Mangan deck) — has the parallel-2e/4e
  topology but ALSO discusses cation hydrolysis.
* **Available data**: `Brianna/20201024 CP Experiment Data-Code/{Cs,K,Na,Li}2SO4`
  has the four-cation × three-pH chronopotentiometry that's the
  primary calibration source. Cs₂SO₄ pH 4 V → −1.55 V is the buffer-
  active signature.

This handoff originally pointed at sulfate; the corrected mechanism
(cation hydrolysis) is the one actually in the literature, and the
one that explains the deck's cation series. Sorry for the
mis-direction in §5 — it was conjecture from chemistry intuition,
not from the data folder.
