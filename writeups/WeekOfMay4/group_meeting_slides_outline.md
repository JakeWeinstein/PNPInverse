# Group Meeting — Slide Outline (May 2026)

For Niall's group-meeting talk on the PNP-BV inverse project. Audience
already saw the Winter Quarter v13 deck. Goal: (1) update on progress
since March; (2) make the case for an "Agentic AI for Scientific
Computing" course and gather feedback.

Proposed length: ~15 slides total — 1 reused recap, 8 progress, 6 AI.

Notation below: `[reuse]` = lift slide from Winter Quarter deck;
`[new]` = build new; `[fig]` = figure or table that needs to be made
or pulled from an existing artifact path.

---

## Part 1 — Progress since the v13 pipeline (~8 slides)

### Slide 1 — Title `[new]`
- **PNP-BV Inverse Solver — Progress since March 2026**
- Jake Weinstein · Niall Mangan group · May 2026
- Subtitle: "Forward solver rebuild + a vision for agentic AI in
  scientific computing"

### Slide 2 — Where we left off `[reuse, with a caveat banner]`
- Lift the v13 "Error progression" slide and the 5.26% max-error
  table verbatim.
- **Add a red caveat banner:** *"Subsequent investigation found
  E_eq was set to 0 for both reactions in this run. With physical
  E_eq (R1 = 0.68 V, R2 = 1.78 V vs RHE), the onset region moves
  outside the original solver window — the v13 result was on a
  much easier (and unphysical) problem."*
- One line of takeaway: this discovery triggered the forward-solver
  rebuild that follows.

### Slide 3 — The E_eq fix changed the problem `[new]`
- Two-column "before/after" table:
  - **Before:** `E_eq = 0`, onset at V_RHE ≈ 0 V, well inside
    z-convergence window.
  - **After:** physical `E_eq`, onset at V_RHE ≈ +0.68 V, well
    outside the original [-0.50, +0.10] V window.
- Bullet: kinetic information lives where the old solver could not
  converge. The "5.3% recovery" was identifying the wrong physics.
- One line on what this implied: forward solver had to be rebuilt
  before the inverse story could continue.

### Slide 4 — Forward solver rebuild: three stacked changes `[new]` `[fig]`
- Title: *"From 4-species PNP to 3-species + Boltzmann + log-c +
  log-rate BV"*
- Three small panels, one bullet each:
  1. **3 species + analytic Boltzmann counterion** — drop ClO4⁻ as
     a dynamic NP unknown; replace with Boltzmann factor in
     Poisson source. (Same idea as the PBNP hybrid in
     Mangan's group, applied to ORR.)
  2. **Log-concentration** `u_i = ln c_i` — positivity by
     construction; SNES no longer steps to negative iterates.
  3. **Log-rate Butler–Volmer** — assemble `log r_j` additively
     before a single `exp`; removes the c_surf clamp inside BV
     that was producing a phantom R2 sink at high anodic V.
- Headline number: usable voltage grid expands from
  V_RHE ≤ +0.20 V (pre-rebuild) to V_RHE ≤ +0.60 V at the
  Apr 27 writeup, and now V_RHE ≤ +1.00 V (15/15 production
  stack with Bikerman-IC + Stern, May 4) — with an existing
  log-c + steric + Debye–Boltzmann sweep already
  demonstrating warm-walk convergence to V_RHE = +2.00 V.
  Caveat: above ~+0.5 V the converged currents are at
  machine epsilon (kinetic dead zone, see slide 8).
- Source figure: pull from `writeups/WeekOfApr27/PNP Inverse
  Solver Revised.pdf` if a clean schematic is wanted.

### Slide 5 — Aligned at the physics-framework level `[new]` `[fig]`
- Frame: the rebuilt forward solver now uses the same physics
  framework the group's reactor-modeling work uses (PNP +
  Bikerman + finite Stern + BV/Tafel), in a more general setting
  (full PNP rather than reduced PB, two reactions, both BV
  branches, adjoint-ready).
- Bullets:
  - **Stern compact layer** — Robin BC; lets the metal-vs-
    solution potential split rather than collapsing onto the
    diffuse drop.
  - **Bikerman steric** — finite ion size in the modified
    Poisson–Boltzmann source.
  - **Debye–Boltzmann analytical IC** — closed-form initial
    guess that cuts Newton iters 1.7×–6.3× and unblocks
    +0.6 V cold starts. (`StudyResults/ic_refinement_study/`.)
  - **Bikerman-consistent IC (Option 2b, today)** — extends
    the Debye–Boltzmann IC with the BKSA matched-asymptotic
    composite ψ profile (saturated zone + outer exponential
    decay) and a multispecies activity γ-correction so all
    seeded `u_i` land on the saturated steric manifold.
    Production sweep is now **15/15 cold/warm to V_RHE = +1.0 V**
    on the production stack with Stern; **14/15** on the same
    stack without Stern.
    (`docs/4sp_bikerman_ic_option_2b_results.md`,
    `StudyResults/peroxide_window_3sp_bikerman_muh_2b/`.)
  - **µ_H electrochemical-potential formulation** —
    `µ_H = ln c_H + φ` for H⁺; smooth across the Debye layer.
- Reuse Niall's "Bulk → Boundary → Double layer → Catalyst"
  cartoon as the visual.
- One-line caveat that sets up the next slide: this is
  alignment at the *physics framework* level — not yet at the
  level of the actual deck experiment.

### Slide 6 — Closing the gap to the actual deck experiment `[new]`
- Honest framing: the most important next moves toward the
  deck/Ruggiero experiment are **physical**, not numerical.
  Whether to swap the FE solver for a deck-style PDE-to-ODE +
  spectral strategy is a second-order question and should
  wait. Source: `docs/Mangan2025_experimental_alignment.md`.
- Concrete physical-model gaps:
  - **Electrolyte identity** — the Ruggiero study uses
    **sulfate-family** electrolytes (`H₂SO₄ + MOH + M₂SO₄`,
    `M = Li, Na, K, Cs`). Current solver carries `ClO₄⁻` as a
    generic inert surrogate — useful for a generic supporting
    electrolyte, wrong for a deck-matched run. Need a
    sulfate-family balancing anion path.
  - **Cation specificity** — the deck story is explicitly
    `Li⁺/Na⁺/K⁺/Cs⁺` and ion-size–dependent. Current solver
    has a generic steric size (`A_DEFAULT = 0.01`); needs
    cation-specific radii and an explicit cation species (or
    cation-specific analytic closure).
  - **Local pH observable** — deck reports IrOx local-pH
    sensing. Cheap to derive from the existing surface H⁺
    field: `pH_local = −log₁₀(c_H_surf / 1000)`. Worth adding
    as a first-class output.
  - **RRDE transport** — deck varies the diffusive-region
    length via rotation rate. Tying `L_REF` to a Levich-style
    mapping makes rotation a real experimental knob, not just
    an abstract nondim length.
  - **Ring channel + selectivity** — RRDE ring-collection
    efficiency for peroxide selectivity is a post-processing
    step on the existing peroxide-current observable.
- Pitch: this is a concrete and short list, and most of it is
  cheap to add. Open question for the group: which
  experimental condition should be the first deck-matched run?

### Slide 7 — Forward-solver known caveats `[new]`
The price of stacking five physics layers and a clip on top of
each other — things that will bite future me, or anyone picking
this up:

- **Log-rate removed *one* clamp, not all clipping.** Three
  distinct clips live in the production code; log-rate eliminated
  only the inner `c_surf` clamp inside the BV residual. The
  **η-clip at ±50** on `(V−E_eq)/V_T` and the **`_U_CLAMP=30`**
  on bulk `u_i = ln c_i` are both still active. Widen `_U_CLAMP`
  to 100 at V_RHE > +0.30 V or it binds at SS and distorts the
  bulk PDE coefficient. (`docs/clipping_conventions.md`.)
- **R₂ is frozen for V_RHE < +0.495 V** at clip=50 — the cathodic
  exponent is held at α₂·n_e·50 independent of V and α₂, so
  `dR_2/dα_2 ≈ 0` there. *(Older handoffs say "+1.14 V"; that's
  an arithmetic error — the clip is on `η_scaled` **before** the
  α·n_e multiplication.)* Practical consequence: **α₂ is only
  data-identifiable from voltages above +0.495 V.**
- **PC observable is qualitatively wrong at V_RHE < −0.1 V** —
  sign flip plus 3–4 OOM magnitude shift vs unclipped truth. CD
  is preserved to ~0.2%; PC is not. Cathodic-regime inverse fits
  using PC fit the clip artifact, not kinetics.
  (`docs/clip_observable_investigation.md`.)
- **`debye_boltzmann` analytical IC is a no-op below
  V_RHE = +0.20 V.** Picard outer loop oscillates against the
  H⁺ mass-transport limit and silently falls back to `linear_phi`.
  Direct z=1 SS also fails at cathodic V — the z-ramp is doing
  real basin work, not just a warm-up.
  (`docs/ic_refinement_study.md`.)
- **Verify adjoints with cold-ramp FD, not warm-start FD,** near
  the R₂ unclip threshold. Warm-start FD from a TRUE-parameter
  cache lands the perturbed solve in a metastable basin with
  exactly half the slope; "adjoint FAIL" reports based on
  warm-start FD are inconclusive. (`CLAUDE.md` rule #2.)
- **Doc drift to watch for.** Older notes still mention a
  "+0.66 V convergence wall" and a "+1.14 V R₂ unclip"; both
  predate the May 4 production stack and the corrected clip
  arithmetic. `CLAUDE.md` and `docs/bv_solver_unified_api.md`
  are the source of truth.

### Slide 8 — Inverse status: deferred `[new]`
- One-line headline: *"Inverse runs are deferred until I am more
  confident in the forward solver."*
- Why: the v13 result was on the wrong physics (E_eq = 0). Not
  eager to repeat that mistake with log-c, Boltzmann, log-rate,
  Stern, and steric all stacked on top of each other.
- The forward-solver consolidation work on the next slide has to
  land first. Inverse picks up after that.

### Slide 9 — Forward-solver next steps `[new]`
Two parallel tracks before re-engaging the inverse:
- **Numerical / convergence track**
  - **Peroxide window — largely solved.** As of May 4, the
    production stack reaches V_RHE = +1.00 V (15/15 with
    `C_S = 0.10` F/m², 14/15 without); an existing log-c +
    steric + Debye–Boltzmann sweep
    (`StudyResults/iv_curve_unified_logc_steric_debye_clip50_v_0p2_to_2p0/`)
    has already demonstrated warm-walk convergence all the
    way to V_RHE = +2.00 V. The remaining frontier above the
    kinetic dead zone may eventually want Option 2c
    (numerical ODE integration of the Bikerman first integral
    rather than the closed-form composite).
  - **Kinetic dead zone above ~+0.5 V** — the converged
    currents drop to machine epsilon (10⁻¹² to 10⁻¹⁶ mA/cm²)
    once `c_H_surf ≲ exp(−ψ_D)` underflows the BV cathodic
    terms. Not a clip artifact, real physics. May change the
    set of useful observables in that range.
  - **4sp dynamic stack still ceiling at V_RHE = +0.50 V**
    (5/15 no-Stern, 7/15 Stern, unchanged from 2a′). This is
    a validation reference, not a production blocker —
    `solve_grid_with_charge_continuation` is no longer the
    production path. Worth chasing if we want full
    cross-stack agreement above +0.5 V.
  - **Validation breadth** — extend the 4sp vs. 3sp +
    Boltzmann agreement check on the cathodic overlap; cross-
    check log-c against log-c-µH on the production grid;
    clip-threshold sensitivity above +0.5 V (currents are
    near zero so PC observable definitions need rechecking).
  - **Documentation lock-in** — `CLAUDE.md` and the
    unified-API doc are the source of truth; the +0.66 V wall
    description is stale and needs updating.
- **Experimental-closure track** (per slide 6)
  - Local-pH observable from the existing surface H⁺ field.
  - Sulfate-family balancing anion path; explicit/effective
    cation species with ion-specific sterics.
  - `L_REF` tied to RRDE transport.
- Once these land, re-engage the inverse with multi-experiment
  Fisher design (bulk-O₂ variation, H₂O₂-fed R₂ isolation,
  L_ref / rotation, cation sweeps).

---

## Part 2 — Agentic AI for Scientific Computing (~6 slides)

> Framing: the previous AI slide ("changes the problem from 'can I
> write code' to 'can I verify it does what I want'") was right but
> understated. Two months later it's the dominant mode of work, and
> the group should consider building a course around it.

### Slide 10 — Vision: agentic AI is changing how research code gets written `[new]`
- One-screen claim: *"Every line of code in this project since the
  v13 paper was written by an LLM. I am the architect, reviewer, and
  scientific lead — not the typist."*
- Three sub-bullets:
  - The forward-solver rebuild + ten ChatGPT/Claude collaboration
    rounds + the inverse studies happened in roughly two months.
  - The bottleneck is now **scientific judgment and verification**,
    not implementation throughput.
  - This is generalizable: most scientific computing groups could
    reorganize around this pattern.

### Slide 11 — Repo stats since March 4 `[new]` `[fig: dashboard]`
Pull these directly from `git`:
- 174 commits since the v13 paper.
- 237 Python source files; 4,700 lines in `Forward/bv_solver/`
  alone (the production stack).
- 79 study scripts under `scripts/studies/`, 95 corresponding
  result directories under `StudyResults/`.
- 32 test files (MMS convergence, F2 diffusion-limit regressions,
  adjoint-vs-FD, multi-noise-model FIM, four-vs-three-species
  agreement…).
- 48 internal design/handoff docs in `docs/`, 22 writeups.
- Optional one-line caveat: a portion of the lines are generated
  artifacts (CSVs, JSON, PNGs); the *source code* is ~30k lines.

### Slide 12 — The Claude ↔ ChatGPT handoff loop `[new]` `[fig: arc diagram]`
- Diagram: **Me ↔ Claude (implementer) ↔ ChatGPT (advisor)**.
- Pattern: when stuck on a research direction, write a long
  "handoff" — current state, what was tried, what failed, open
  questions — and pass it to the *other* model. The two models
  catch each other's blind spots.
- Concrete arc, ten handoff documents in `docs/CHATGPT_HANDOFF_*.md`:
  - #2 peroxide result → #3 LSQ ridge → #4 precision &
    next → #5 FIM "definitive next, anodic" → **#6 log-rate
    breakthrough** → #7 log-rate validation → #8 multi-init →
    #9 adjoint resolved + grid done → #10 LM/Tikhonov, basin
    geometry as the new bottleneck.
- Each handoff round took **hours, not weeks.**
- Punchline: this is closer to "ensemble of advisors" than "code
  assistant".

### Slide 13 — Speed: a concrete case study `[new]` `[fig: timeline]`
Pick one rebuild as a story:
- **Log-rate BV breakthrough → validated production stack**
  in ≈ 1 week, end-to-end:
  - Day 0: ChatGPT identifies the c_surf clip as the suspect for
    the +0.30 V Newton failures. Handoff written.
  - Day 1: Claude refactors `forms_logc.py` to assemble
    `log r_j` additively; toggle exposed via
    `bv_log_rate=True`.
  - Day 2: regression suite extended; cold-ramp FD adjoint check
    written; +0.60 V converges for the first time.
  - Day 3–5: multi-init inverse runs; Fisher analysis at TRUE;
    the old "log k₀,₂ unidentifiable" failure is resolved.
  - Day 6–7: ChatGPT response and next-path handoff (#10) — the
    bottleneck has moved from forward solver to global basin
    geometry.
- The honest caveat: most of that week is *me* deciding what to
  validate, not waiting on code.

### Slide 14 — Verification habits — keeping the AI honest `[new]`
The single biggest reason this hasn't gone off the rails:
- **Tests written in parallel with code, not after.**
  - MMS convergence tests for log-c form
    (`tests/test_mms_convergence.py`).
  - F2 diffusion-limit tolerance gates on every
    forward-solver change.
  - Adjoint vs cold-ramp finite-difference regression at
    every voltage that matters.
  - Strategy A / B / C / D side-by-side studies (V24, V25,
    V26) — never trust a refactor without comparing to the
    prior production build.
- **Lessons captured as durable rules** in `CLAUDE.md` so they
  don't have to be relearned. Examples currently in force:
  - "Always use adjoint gradients; do not switch to
    derivative-free."
  - "Verify adjoints with cold-ramp FD, not warm-start FD,
    near R2 unclip."
  - "Default FIM noise model is `local_rel + abs_floor`."
- **Two-LLM cross-checks** — Claude and ChatGPT review each
  other's diagnoses before code lands.

### Slide 15 — A course proposal + ask for feedback `[new]`
- Pitch: **"Agentic AI for Scientific Computing"** — a course
  for ESAM grads working on PDE / inverse / simulation
  research code.
- Tentative modules:
  1. **Context engineering** — `CLAUDE.md`, persistent memory,
     handoff docs.
  2. **Multi-model dialog** — when to use one model vs. an
     adversarial pair.
  3. **Verification-first prompting** — tests, MMS, regression
     suites as the real specification.
  4. **Failure modes** — silent drift, mocked physics,
     plausible-but-wrong derivations; how to detect them.
  5. **Hybrid scientific judgment** — what *only* the human
     should still do (problem framing, physical sanity,
     publication-grade claims).
- Closing ask: *"Would you want this course? What's missing
  from the outline? Who in the group has war stories — good
  or bad — that should go in?"*

---

## Reused supporting slides (decide based on time)

If the talk runs long, drop one of these. If it runs short,
keep all three:

- v13 forward setup (PNP equations + 4-species table). `[reuse]`
- v13 observables (CD/PC vs V_RHE plot). `[reuse]`
- v13 inverse-problem statement (BV equation + 4 unknowns).
  `[reuse]`

---

## Notes on figures to make / pull

Most figures already exist as artifacts; only the timeline /
arc / dashboard panels are new:

- Slide 4 schematic: `writeups/WeekOfApr27/PNP Inverse Solver
  Revised.pdf` already has the three stacked changes laid out;
  may be liftable as-is.
- Slide 11 dashboard: generate from `git log --since=2026-03-04
  --shortstat` and a quick `find ... | wc -l` panel.
- Slide 12 handoff arc: just a stack of ten labeled boxes —
  no real figure required.
- Slide 13 timeline: same — calendar-style strip with ChatGPT
  Claude / me lanes.

---

## Open questions for Jake before drafting slides

1. Should the AI section be **half** the deck (as outlined here)
   or **dominant**, given that the course pitch is "the more
   important section"? Could push to ~8 AI slides if course
   discussion is the goal.
2. How honest do we want to be about what *didn't* work with AI?
   (Would strengthen the course pitch; risks distracting from
   the science update.)
3. Want a slide on the cost / token / wall-clock numbers? They
   are striking but potentially a distraction.
4. Slide 6 ("closing the gap to the actual deck experiment")
   asks the group to nominate a first deck-matched run. Worth
   pre-deciding internally what the desired answer is (e.g.
   "pH 4 / `Cs⁺` / sulfate"), or genuinely leave the choice
   open to the group?
5. The experimental-closure list on slide 6 spans
   electrolyte, cation, observable, transport, and ring
   channel. If any of those are already on someone else's
   plate in the group, the slide should redirect the asks
   accordingly — worth a quick check with Niall before the
   talk.
