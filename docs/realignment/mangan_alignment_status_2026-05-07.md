# Mangan Alignment — Status & Handoff to Next Claude

Date: 2026-05-07

## What this is

Brief status doc for a Claude session picking up the Mangan-deck
alignment work. Read this first, then pull the linked docs as needed.

## Where we are

Goal: align the PNP/BV forward solver to **Mangan deck page 15**
(peroxide current density vs V_RHE at pH 4 with Cs⁺ on CMK-3, RRDE).

**Completed:**
- M1 (observable infrastructure with provenance metadata) — landed.
- M1.5 (Stern-aware IC seed fix) — landed (separate agent).
- M0 extraction for the page-15 target — done. Digitized curve at
  `data/mangan_deck_p15_h2o2_current.csv` (37 points). Constants in
  `docs/m0_target_extraction.md`.
- Run C executed: production stack (3sp + Bikerman ClO₄⁻ + logc_muh +
  log-rate + Stern, clip=100, C+D orchestrator) across 25 voltages in
  V_RHE ∈ [−0.40, +0.55] V. Converged 25/25.
- Run D verdict: **0/5 tolerance bands met.** Output flat at zero in
  the peroxide channel, total disk current matches experiment shape
  qualitatively. See `StudyResults/mangan_p15_comparison/` and
  `docs/mangan_p15_comparison_summary.md`.

**Not done yet:**
- Diagnosis-driven follow-up work (M3a.0 onward — see "Next step"
  below).

## The diagnosis (current best read)

Run D verdict was initially "shape_wrong" with R_0/R_1 lock-in: our
sequential model has R_0 producing H₂O₂ and R_1 consuming it at the
boundary, with R_2's BV cathodic factor ~14 OoM larger than R_1's, so
surface H₂O₂ is consumed as fast as produced and net peroxide → 0.

**Source paper found (this is the load-bearing realization):**
Ruggiero, Sanroman Gutierrez, George, **Mangan**, Notestein, Seitz.
*J. Catal.* 2022. Mangan is co-author. Local PDF and full notes in
`docs/Ruggiero2022_JCatal_source_paper.md`.

The paper's reaction model (per §1 mechanism description) is **parallel
2e and 4e ORR**, not sequential:
- 2e: O₂ + 2H⁺ + 2e⁻ → H₂O₂ at E⁰ = 0.695 V_RHE
- 4e: O₂ + 4H⁺ + 4e⁻ → 2H₂O at E⁰ = 1.23 V_RHE (no free H₂O₂
  intermediate)

So the R_0/R_1 lock-in is a **model-structural artifact** of our
sequential reaction set — there's no "R_1 consumes the H₂O₂ R_0
produces" coupling in the actual experiment. The deck's "Peroxide
Current Density" maps to **gross R_2e** (single-rate observable), not
to the net (R_0 − R_1) we've been computing.

Other key facts from Ruggiero 2022 (full extraction in the source-
paper notes doc):
- Electrolyte: 0.1 M H₂SO₄ + 0.1 M MOH + 0.1 M M₂SO₄. **Sulfate, not
  perchlorate (explicitly rejected as environmental pollutant).**
- Total ionic strength I = 0.3 M (not 0.1 M). λ_D ≈ 0.55 nm.
- Collection efficiency N = 0.224, rotation 1600 rpm, ring at +1.2 V.
- Local pH at bulk pH 4 swings from 4 → 8-9 at ~3.25 mA/cm² disk
  current (Fig 1B).

## The plan (post-Ruggiero, post-GPT-pushback)

Two GPT review rounds (handoffs 17 → 18) on the realigned plan.
GPT's H18 critique is the current best version. Key adjustments:

- **M3a is not a 1-2 day reaction-list edit.** Production parallel
  2e/4e requires generalizing the `debye_boltzmann` Picard
  initializer, which is currently hard-coded around two sequential
  rates with a 2×2 O₂/H₂O₂ rate-transport solve. Split into
  substages M3a.0 → M3a.3.
- **Electron-weighted disk current is mandatory.** Current
  `current_density` observable assumes uniform n_e = 2; with R_4e
  added, total disk current must be `Σ n_e_j · R_j`.
- **Cation buffering may not fall out from PNP+Bikerman.** The
  paper's "buffering" mechanism may need hydrolysis/activity
  chemistry beyond what multi-ion PNP captures.

Realigned milestone sequence:

1. **M3a.0** — Observable audit on existing Run C state.
2. **M3a.1** — Electron-weighted current accounting.
3. **M3a.2** — Diagnostic parallel R_2e/R_4e residual.
4. **M3a.3** — Production IC generalization.
5. **M3b** — Multi-ion electrolyte (Cs⁺/SO₄²⁻ + IC rederivation).
6. **M3c** — Local-pH validation against Ruggiero Fig 1B.
7. **M4** — Cation specificity sweep.
8. **M5** — L_eff alignment (~10% retune).
9. **M6** — Stern + cation joint sensitivity.

Full plan in `~/.claude/plans/swirling-crunching-wren.md`. The
realignment dialogue is in handoffs 17-18; my assessment of GPT's H18
counterreply is in this conversation but not saved as a doc.

## Next concrete step: M3a.0 observable audit

**Cost: hours, not days. No code changes. No new solver runs.**

The existing Run C output is on disk in
`StudyResults/mangan_p15_comparison/run_C/`. The solver state was
saved per voltage. The observable function in
`Forward/bv_solver/observables.py:13-68` already supports
`mode="reaction", reaction_index=N` to assemble a single-reaction
rate (without computing differences).

Action:
1. Load Run C's per-voltage state.
2. Re-assemble three observables off the existing state:
   - `cd_mA_cm2` (already there): total disk current = Σ R_j
   - `pc_mA_cm2` (already there): net peroxide = R_0 − R_1
   - **`gross_R0_mA_cm2` (NEW)**: just R_0 alone
     (`mode="reaction", reaction_index=0`)
3. Plot all three vs V_RHE alongside the digitized page-15 target
   from `data/mangan_deck_p15_h2o2_current.csv`.
4. Compute the page-15 tolerance bands (peak voltage, peak magnitude,
   plateau, onset, shoulder) for the gross R_0 channel.

What the answer tells us:
- **gross R_0 matches both magnitude and shape**: scope reduction.
  The "problem" was observable definition; we may not need
  parallel reactions or multi-ion at all for THIS figure.
- **gross R_0 matches magnitude but not shape** (most likely
  outcome — page-15 summary noted gross R_0 was near the
  experimental left plateau): the parallel-reaction work is real,
  but we're chasing a real physics gap, not an observable bug.
- **gross R_0 doesn't match anything**: deeper diagnosis required;
  debug forward solver / kinetics before more milestones.

This is the cheapest test that informs everything downstream.

## Reference docs (priority order for pickup)

| Doc | Purpose |
|---|---|
| `CLAUDE.md` (project root) | Hard rules, production-stack flags, convergence ceilings |
| `docs/mangan_p15_comparison_summary.md` | Run C/D verdict + the three reads of the diagnosis (canonical narrative for the comparison) |
| `docs/Ruggiero2022_JCatal_source_paper.md` | Full source-paper extraction; the parallel 2e/4e structural finding |
| `docs/Ruggiero2022_JCatal_manuscript.pdf` | The actual paper (838 KB local copy) |
| `~/.claude/plans/swirling-crunching-wren.md` | Plan B — page 15 comparison plan (now superseded by GPT H18 splitting but still anchors target/protocol) |
| `docs/m0_target_extraction.md` | M0 extraction outputs (RRDE constants, source authority per quantity, acceptance bands) |
| `docs/CHATGPT_HANDOFF_17_RUGGIERO_REALIGNMENT_PLAN.md` | My realignment plan post-Ruggiero |
| `docs/CHATGPT_HANDOFF_18_RUGGIERO_REALIGNMENT_COUNTERREPLY.md` | GPT's counterreply with the M3a substage splitting (current best version of the plan) |
| `docs/Mangan2025_Catalysis.pdf` | Original deck; pages 14, 16-18 are the modeling-parameter source (NOT YET extracted — open task) |
| `docs/Mangan2025_experimental_alignment.md` | Original gap-audit doc (historical context) |

### Memory entries

- `memory/project_mangan_m1_deferred_parameters.md` — convention for
  experiment_metadata placeholders (don't promote past `deck_proxy`
  without M0 done).
- `memory/project_mangan_m0_extraction_complete.md` — page-15
  resolved values + the post-Ruggiero diagnosis update (parallel
  2e/4e, not kinetic recalibration).

### Run artifacts

- `StudyResults/mangan_p15_comparison/run_C/iv_curve.json` — Run C
  output (cd, pc, RRDE observables, metadata).
- `StudyResults/mangan_p15_comparison/run_C/diagnostics.json` —
  per-V solver diagnostics (surface fields, SNES reasons).
- `StudyResults/mangan_p15_comparison/run_C/comparison.png` — side-by-
  side experimental + model PC overlay.
- `StudyResults/mangan_p15_comparison/run_D_verdict.md` — verdict.

### Code anchors

- BV reaction-rate construction:
  `Forward/bv_solver/forms_logc_muh.py:393-475` and
  `forms_logc.py:350-435`.
- BV observable wiring: `Forward/bv_solver/observables.py:13-68`.
  Note: already supports `mode="reaction", reaction_index=N` for
  single-reaction assembly — needed for M3a.0.
- Production constants: `scripts/_bv_common.py` (K0, ALPHA, E_EQ,
  C_HP, C_CLO4, C_SCALE, L_REF, A_DEFAULT, etc.).
- Sequential-IC Picard initializer (the structural barrier to
  parallel-reaction production model): `forms_logc.py:623-811`,
  `forms_logc_muh.py:704-908`.

## Open tasks visible from this status

- **M3a.0 observable audit** (next concrete step).
- **Extract Mangan deck slides 14, 16-18** for the actual
  computational parameters (k0, α, L_eff, etc. — Ruggiero paper
  doesn't have them; deck does).
- **Decide on M3a.0 result.** Don't commit to M3b (multi-ion) until
  the audit clarifies how much of the gap is observable-side vs
  physics-side.
- **Round 9 with GPT** — only after M3a.0 produces empirical data,
  not before.

## Don't get confused by

- The earlier rounds 11-16 dialogue was about the WRONG framing
  (sequential reactions, single-counterion shortcut, etc.). Ignore
  the diagnostic recommendations there; the post-Ruggiero diagnosis
  in handoffs 17-18 is the live one.
- "Surrogate" in `experiment_metadata.electrolyte_model =
  "pH_countercharge_surrogate"` refers to the simplified forward-
  model bundle (single ClO₄⁻ at protonic concentration), NOT the ML
  `Surrogate/` system in this repo.
- The plan file `swirling-crunching-wren.md` was written for the
  page-15 comparison execution. It's not a complete current plan —
  refer to handoff 18 for the M3a substaging.
