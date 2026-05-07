# CHATGPT Handoff 11 — Mangan 2025 Experimental Alignment Plan

Date: 2026-05-06
Status: Forward-only planning. Inverse work is paused.

## Why you're reading this

We have a Firedrake-based PNP/Butler-Volmer forward solver for a two-step ORR
(O₂ → H₂O₂ → H₂O) at pH 4. We want to bring it closer to the Mangan 2025
catalysis deck, which describes RRDE peroxide-production experiments on
mesoporous carbon black (CMK-3) with alkali-cation electrolytes and IrOx
local pH sensing.

A Claude sub-agent already wrote a gap audit
(`docs/Mangan2025_experimental_alignment.md`). Then Claude (the same model,
different conversation) sketched a reorganized plan. This doc is the
combined writeup. We want **substantive disagreement** from you, not
validation. Where you think we are wrong, say so. Where you would reorder
the plan, say so. Where you think we are missing a question, say so.

This doc is meant to be self-contained — you should not need to read the
codebase. We list the source documents at the end if you want to anchor
specific claims, and we flag which numerical values are "from memory and
worth verifying."

---

## Part 1 — The experiment (Mangan 2025)

### Setup

- **Catalyst**: mesoporous carbon black, CMK-3.
- **Geometry**: rotating ring-disk electrode (RRDE).
- **Local pH sensor**: IrOx oxide film, used to infer local pH near the
  electrode surface separately from bulk pH.
- **Bulk pH**: 4 (acidic regime).
- **Electrolyte**: H₂SO₄ + M₂SO₄ + MOH, with M⁺ = Li⁺, Na⁺, K⁺, Cs⁺.
  This is sulfate-family, **not perchlorate**. Source for the cited
  Ruggiero et al. study: `https://www.osti.gov/servlets/purl/2418971`.
- **Headline condition**: pH 4 with Cs⁺ as the working cation.

### Reaction model

Two-step oxygen reduction to peroxide and then to water:

- R1: `O₂ + 2H⁺ + 2e⁻ → H₂O₂`, equilibrium potential `E_eq = 0.68 V` vs RHE.
- R2: `H₂O₂ + 2H⁺ + 2e⁻ → 2H₂O`, equilibrium potential `E_eq = 1.78 V` vs RHE.

The ratio of R1 to R2 sets the H₂O₂ selectivity. Cs⁺ at pH 4 is
peroxide-selective in the deck.

### Observables

- RRDE **disk current** — total current density.
- RRDE **ring current** — peroxide-collection signal at a downstream ring
  electrode operated at a peroxide-oxidation potential. Maps to peroxide
  flux from the disk via collection efficiency `N` (typically 0.20–0.42,
  geometry-dependent — verify per cell).
- IrOx **local pH** — local proton activity near the electrode surface,
  used to track surface acidification/alkalinization under load.
- Local-vs-bulk pH offset.
- **H₂O₂ selectivity** — the canonical RRDE expression is
  `S_H2O2 = 2 (I_ring / N) / (I_disk + I_ring / N)`, with `I_ring` and
  `I_disk` as ring and disk currents.

### Cation-effects story (the deck's headline)

Across Li⁺, Na⁺, K⁺, Cs⁺ at fixed pH 4, the deck reports monotonic shifts in
peroxide-current curves and in selectivity. The proposed mechanism is
ion-specific steric exclusion at the OHP / outer compact layer. **Bare vs
hydrated radii matter and reverse the ordering** (verify against the deck's
actual figures — this is from memory):

| Cation | Bare ionic radius (Å, Shannon CN=6) | Hydrated radius (Å, ~Marcus 1988) |
|--------|-------------------------------------|-----------------------------------|
| Li⁺    | 0.76                                | 3.82                              |
| Na⁺    | 1.02                                | 3.58                              |
| K⁺     | 1.38                                | 3.31                              |
| Cs⁺    | 1.67                                | 3.29                              |

Bare radii are monotonic Li < Na < K < Cs. Hydrated radii are approximately
the reverse (Cs ≈ K < Na < Li). A "larger ion = more steric exclusion"
narrative is only meaningful if the deck is specific about which radius it
uses for the steric size parameter. This is one of the sharper points
where you might tell us we're wrong.

### Modeling approach in the deck (their solver, not ours)

- Steady-state, 1D.
- Outer **diffusive** region solved analytically/explicitly.
- Inward integration from the bulk.
- **PDE → ODE switch** at the screening-region boundary.
- Nonlinear **spatial mapping** to resolve nm-scale double-layer structure.
- **Spectral methods** for spatial discretization.
- Modified Poisson-Boltzmann + Bikerman-style steric closure.
- Butler-Volmer / Tafel boundary kinetics.

This is a different numerical strategy from what we have. We treat that as
secondary — see Part 4, Milestone D — but you may disagree.

---

## Part 2 — Current solver state (May 2026)

### Production stack

- **3 dynamic species**: O₂, H₂O₂, H⁺.
- **Analytic Bikerman counterion** for ClO₄⁻
  (`steric_mode='bikerman'`). The analytic counterion contributes to
  Poisson, to the steric packing fraction, and to the bulk
  electroneutrality closure, so it is more than a passive Boltzmann
  correction — it is wired into the residual.
- **Proton electrochemical-potential primary variable**:
  `formulation='logc_muh'`, with `μ_H = u_H + em·z_H·φ` where `u_H = ln c_H`.
- **Log-rate Butler-Volmer** at the cathode boundary.
- **Finite Stern compact layer**, capacitance ≈ 0.10 F/m².
- **`debye_boltzmann` initial condition** — composite-ψ + multispecies-γ.
- **C+D orchestrator**: `solve_grid_per_voltage_cold_with_warm_fallback`.

### What this stack reaches

- **V_RHE = +1.0 V at 15/15** convergence on the C+D orchestrator
  (cold ceiling +0.60 V, warm-walk to +1.00 V).
- **V_RHE ∈ [−0.5, +1.0] V** is the trusted convergence window with the
  production stack. Below −0.5 V we have not pushed; above +1.0 V the
  warm-walk fails.

### Existing observables

- **Total current density** — sum of BV reaction rates.
- **Peroxide current** — `R1 − R2` (so positive when net peroxide is
  produced and consumed at less than the production rate).

We do **not** yet model RRDE ring collection efficiency, IrOx sensor
response, local pH as a first-class observable, selectivity, or
rotation-rate-dependent transport.

### Solver internals (mostly numerical, listed for context)

- Firedrake finite elements.
- Coupled log-concentration PNP weak form.
- Graded interval (1D) / rectangle (2D) meshes.
- Pseudo-time stepping to steady state.
- Newton / SNES nonlinear solves.
- Per-voltage z-ramp continuation, plus warm-walk continuation between
  voltages.
- Reference length scale `L_REF = 100 µm`.

### Hard rules — do not violate without re-reading the docs

These are lessons that cost real time. They are anchored in the project's
source-of-truth docs.

1. **Use C+D orchestrator, not Strategy B.**
   `solve_grid_per_voltage_cold_with_warm_fallback` works; the older
   `solve_grid_with_charge_continuation` (B) fails 3/13 at production
   resolution (Ny=200) on the logc + Boltzmann stack. B's Phase-1
   V-sweep at z=0 hands bisection a mismatched species IC it cannot
   recover from on the saturated manifold.

2. **`exponent_clip = 100`, not 50.** The clip applies to
   `eta_scaled = (V_RHE − E_eq)/V_T` *before* the α·n_e multiplication.
   - At clip=50, R2 unclips at V_RHE > +0.495 V. Below that, the
     reported peroxide current is **fictitious** (sign-flipped at
     V_RHE < −0.1 V; magnitude artefact across the cathodic grid). Do
     not compare clip=50 PC against experiment.
   - At clip=100, R2 unclips at V_RHE > −0.79 V. Production grid is
     fully unclipped. **Only configuration where negative-voltage PC
     is trustworthy.**
   - Some configurations cold-fail more often at clip=100 than at
     clip=50 (no-Stern bikerman near V_RHE ≈ +0.1 V); recover with C+D
     warm-walk or Stern, not by lowering the clip.
   - Older handoffs (numbered ≤ 10) cite an unclipping voltage of
     ~+1.14 V — that was wrong (missed the α factor). Ignore that.

3. **IC and residual must agree about steric saturation.**
   `debye_boltzmann` IC + `steric_mode='bikerman'` is the matched pair.
   A Bikerman IC without the matching residual closure (or vice-versa)
   cold-fails on the saturated counterion manifold.

4. **The 4sp dynamic stack ceiling is unchanged by the IC fix.**
   When ClO₄⁻ is treated as a fully dynamic NP species rather than an
   analytic counterion, the stack tops out at 5/15 (no Stern) and 7/15
   (Stern). The binding constraint is the dynamic anion's NP equation,
   not the IC. Going fully dynamic on a counterion is **not** a fix
   for the anodic ceiling.

5. **Use physical `E_eq`** (R1 = 0.68 V, R2 = 1.78 V vs RHE), never
   `E_eq = 0`.

6. **Trusted convergence window**: V_RHE ∈ [−0.5, +1.0] V.

### Inverse-mode status

**Paused.** All scripts in `scripts/studies/v*.py` are legacy and
non-operational. This planning is forward-only. Richer observables will
plug back into inverse work when it resumes, but no inverse coupling is
in scope now.

---

## Part 3 — Gap summary

Distilled from `docs/Mangan2025_experimental_alignment.md`:

| Aspect              | Deck                                          | Solver                                | Severity                          |
|---------------------|-----------------------------------------------|---------------------------------------|-----------------------------------|
| Observables         | RRDE disk + ring + IrOx local pH + selectivity | total current + peroxide current      | Important — blocks comparison     |
| Electrolyte         | H₂SO₄ / M₂SO₄ / MOH (sulfate family)          | ClO₄⁻ analytic counterion             | Important — wrong identity        |
| Cation              | Li/Na/K/Cs, headline focus on Cs⁺             | generic `A_DEFAULT = 0.01`             | **Critical — deck's main story**  |
| Anion charge        | SO₄²⁻ dominates at pH 4 (z = −2)              | ClO₄⁻ (z = −1)                         | EDL screening ~2× stronger        |
| Mass transport      | RRDE rotation-rate-dependent diffusion layer  | `L_REF` as abstract scale             | Important — couples to disk       |
| Numerical strategy  | 1D, outer-explicit, PDE→ODE, spectral         | Firedrake FE + Newton + warm-walk     | Lower priority — efficiency, not physics |

### The existing doc's recommended order

1. Local pH observable (cheap).
2. Experiment metadata config.
3. Sulfate-family effective anion.
4. Dynamic Cs⁺.
5. Cation-specific sterics.
6. Tie `L_REF` to RRDE rotation rate.
7. Ring current / selectivity post-processing.
8. Finite-Stern calibration (revisit).
9. Reduced 1D ODE/spectral solver (deferred).

---

## Part 4 — Proposed reorganization (Claude's pushback)

### Reordering

1. **Bundle stages 1–3 into one "deck-matched config v1" landing.**
   The local-pH observable, experiment-metadata config, and analytic
   anion swap (ClO₄⁻ → effective sulfate, z = −2) are mechanically
   independent and don't make sense one at a time. The config layer is
   what makes the new observable and anion meaningful as a deck-matched
   *mode* rather than scattered flags.

2. **Stages 4–5 (cation identity + cation sterics) are the load-bearing
   physics change.** This is the deck's actual story. Once we commit to a
   Cs⁺-at-pH-4 run, the analytic counterion has to either (a) become a
   Cs⁺-bearing closure, or (b) sit alongside a dynamic Cs⁺ species. The
   Bikerman analytic-counterion derivation in
   `docs/steric_analytic_clo4_reduction_handoff.md` was done for `z = −1` —
   generalizing to a `z = −2` sulfate closure plus a Cs⁺ closure is **not
   just a sign flip**. The residual algebra needs to be re-checked.

3. **Stage 6 (`L_REF` ↔ rotation) has a pitfall.** `L_REF` is the
   nondimensionalization length, so retuning it rescales every other term
   in the problem. Cleaner approach: keep `L_REF` fixed as a numerical
   scale, expose `omega_rpm` as an experimental input, and let the Levich
   relation set the *position* of the bulk Dirichlet BC (or equivalently
   an experimentally-meaningful `L_eff`). Decouples physical transport
   from solver discretization.

   Levich for context (verify constants): `δ = 1.61 D^(1/3) ν^(1/6)
   ω^(−1/2)`. With `D ~ 2 × 10⁻⁹ m²/s`, `ν ~ 10⁻⁶ m²/s`, and
   `ω = 2π·1600/60 ≈ 168 rad/s`, `δ ~ 17 µm`. The deck's cited 66–86 µm
   range corresponds to slower rotation (200–400 rpm).

### Open questions that determine plan shape

a. **Immediate goal**: reproduce a specific Mangan figure quantitatively
   (e.g., the Cs⁺ pH-4 peroxide-current curve) and compare numerically,
   or build deck-matched infrastructure so that future runs can be
   matched? Different scope.

b. **Cation set**: just Cs⁺ at pH 4 (deck's headline result), or the full
   Li/Na/K/Cs sweep (deck's radius-vs-selectivity story)? The latter is
   the more honest reproduction.

c. **Effective anion vs full speciation**: at pH 4 we are 2 pKa units
   above pKa₂(H₂SO₄) ≈ 1.99 (verify), so SO₄²⁻ dominates ~99%. Effective
   `z = −2` at half ClO₄⁻'s bulk concentration *should* be enough — it
   captures the EDL-screening change without acid-base buffering. Full
   HSO₄⁻/SO₄²⁻ speciation matters only if surface pH could swing into
   the pKa region, which is plausible under load. Open.

d. **Dynamic Cs⁺ vs analytic Cs⁺**: analytic is cheaper and reuses the
   existing Boltzmann-counterion plumbing. Dynamic is honest but
   inherits the **5/15 ceiling** we already saw on the dynamic-anion
   stack (Hard Rule 4). Analytic-Cs⁺ first seems right to us, but you
   may argue the deck's story requires real cation transport.

e. **Inverse pause**: still firm? Confirming so we don't accidentally
   design observables in a way that locks out future inverse work.

### Proposed milestones

**Milestone A — Deck-matched config v1**
- **Local pH observable** as post-processing:
  `pH_local = -log10(c_H_surface / 1000)` with `c_H_surface` in mol/m³
  (convert from nondim through the concentration scale).
- **Experiment metadata config schema**: `catalyst`, `geometry`,
  `pH_bulk`, `cation`, `anion_model`, `rotation_rate_rpm`, `L_eff_m`,
  `observables`. Drives parameter construction so study scripts stop
  re-assembling species and scales by hand.
- **Effective sulfate-family anion**: analytic Boltzmann counterion with
  `z = −2`, bulk concentration set by electroneutrality against the
  cation and proton bulk concentrations.
- **Sanity diff** against current ClO₄⁻ baseline at the same pH 4. We
  expect EDL screening to be stronger and the peroxide-current curve to
  shift by a measurable but not catastrophic amount.
- **Risk**: the Bikerman analytic-counterion residual closure was
  derived for `z = −1`. Re-deriving for `z = −2` is a real task, not a
  single-character change.

**Milestone B — Cation specificity**
- **Cation-specific steric radii** as an experiment parameter, with
  defensible bare or hydrated values per cation (decision pending — see
  open question on which radius the deck uses).
- **Analytic Cs⁺ first** — treat as a Boltzmann species with
  experimentally-grounded steric size. Reuses A's plumbing.
- **Optionally dynamic Cs⁺** as a second pass, with the explicit
  acknowledgement that the dynamic stack inherits the 5/15 ceiling.
- **Reproduce a deck-style steric-radius sweep** across Cs/K/Na/Li
  effective radii, holding pH and transport fixed.
- **Risk**: dynamic Cs⁺ convergence ceiling. If hit, fall back to
  analytic-only.

**Milestone C — RRDE transport**
- **`omega_rpm` as input**, Levich-derived `L_eff` setting BC position.
- **Ring current** as a derived observable from peroxide flux at the
  electrode and a configured collection efficiency `N`.
- **Selectivity** post-processing.
- **Risk**: collection efficiency `N` is normally a per-cell calibration;
  treat as configurable, not fitted, until we have ring data to
  calibrate against.

**Milestone D — Deferred**
- Finite Stern as a *physical* parameter (calibrated nuisance with
  bounds, not a numerical knob).
- Reduced 1D ODE/spectral solver only if the Firedrake stack hits a
  wall that's structural, not numerical.

---

## Part 5 — What we want from you

We are explicitly looking for **substantive disagreement and missing
considerations**, not validation of the plan.

### Specific points where we want pushback

1. **Reordering**. Where would you reorder A–C? Is anything in D actually
   a *precondition* for A or B that we're wrongly deferring? Concrete
   example: is finite-Stern calibration actually load-bearing for a
   defensible cation-specific steric story, given that Stern capacitance
   and steric size both control where the OHP sits?

2. **Anion question**. Is effective `z = −2` sulfate enough at pH 4, or
   does the deck require explicit HSO₄⁻ ↔ SO₄²⁻ + H⁺ buffering? Surface
   pH can swing under load — does that push us across the pKa region and
   force speciation? If so, how invasive is it (it's another transported
   species plus an equilibrium constraint)?

3. **Cation question**. Is analytic Cs⁺ defensible, or is the cation's
   transport (its actual diffusion away from the cathode at large
   overpotentials, and its accumulation under hydroxide formation
   conditions) part of what the deck is measuring? If transport
   matters, dynamic Cs⁺ is mandatory and we need to confront the
   convergence ceiling head-on.

4. **Steric-radius question**. Bare versus hydrated for the cation
   sterics. The deck likely reports one. Which does Bikerman
   theoretically demand? Hydrated radii reverse the ordering — which
   way does the deck's data go?

5. **Boltzmann-counterion derivation**. Will the residual closure in
   `docs/steric_analytic_clo4_reduction_handoff.md` generalize cleanly
   from `z = −1` to `z = −2`, or is there a structural issue (e.g.,
   Stern-layer charge balance, a missed nonlinearity in the saturation
   density, or a problem with the bulk-concentration counter-equation)?

6. **Numerical strategy**. We currently defer the deck's reduced
   1D ODE/spectral approach (Stage 9 → Milestone D). Are we right to
   defer it, or is the Firedrake stack's 5/15 ceiling on dynamic
   counterions a *structural* symptom that the spectral approach with
   nonlinear spatial mapping would solve cleanly? If yes, Milestone B
   collapses into a solver-rewrite milestone and the cost-benefit
   changes a lot.

7. **Observables we have not considered**. The deck mentions local-vs-
   bulk pH offset and selectivity. Are there other RRDE-specific
   observables — e.g., **Levich plots** (`I_disk vs ω^(1/2)`),
   **Koutecký-Levich extraction** of kinetic current, or
   **n_e (electron number) inferred from disk/ring ratio** — that we
   should be planning to produce as first-class outputs because that's
   how this kind of paper is read?

8. **What did we miss**: anything in the experimental setup, the
   electrochemistry, the ion-specific physics, or the standard practice
   for comparing models to RRDE data that isn't on our radar?

---

## Source documents (for verification)

- `docs/Mangan2025_experimental_alignment.md` — the original gap audit.
- `docs/Mangan2025_Catalysis.pdf` — the deck itself (we have not
  re-read every slide for this writeup; lean on it for ground truth).
- `README.md` — solver narrative.
- `CLAUDE.md` — operational rules and convergence ceilings.
- `docs/bv_solver_unified_api.md` — production-solver call surface.
- `docs/4sp_bikerman_ic_option_2b_results.md` — production-target sweep.
- `docs/4sp_drop_boltzmann_investigation.md` §11–§13 — investigation
  log behind the production target.
- `docs/clipping_conventions.md` — the clip=50 vs clip=100 PC story.
- `docs/steric_analytic_clo4_reduction_handoff.md` — the existing
  `z = −1` Bikerman analytic-counterion derivation.
- `docs/CONTINUATION_STRATEGY_HANDOFF.md` — why C+D, not B.
- OSTI accepted manuscript for Ruggiero et al. (sulfate-family
  electrolyte source): `https://www.osti.gov/servlets/purl/2418971`.

### Caveats on this doc

- Numerical estimates in this doc (ionic radii, Levich constants,
  pKa(HSO₄⁻), Cs⁺ hydrated radius) are from memory and worth verifying
  before relying on them quantitatively.
- We have **not** opened `Forward/bv_solver/boltzmann.py` or
  `scripts/_bv_common.py` for this writeup. Claims about the *ease* of
  particular changes (e.g., "z = −2 swap is ~one config line") are
  inferred from the docs and CLAUDE.md, not from direct code reads.
  Treat them as estimates.
- The "5/15 ceiling on dynamic counterions" number is anchored in
  CLAUDE.md and the `4sp_drop_boltzmann_investigation` doc; it is the
  most load-bearing empirical fact in this writeup, so flag if you
  think it's been misinterpreted.

---

## Bottom line we expect to defend or revise

The core of our pushback to the existing gap-audit doc is:

- The first three stages should land together as a single "deck-matched
  config v1" milestone.
- Cation specificity (with bare-vs-hydrated radius decision) is the
  load-bearing physics change and deserves its own milestone.
- The `L_REF` ↔ rotation linkage should be done by exposing `omega_rpm`
  and using Levich to set BC position, **not** by retuning the
  nondim length scale.
- The numerical-strategy rewrite (deck's 1D ODE/spectral approach)
  stays deferred unless someone shows the FE stack has a structural
  rather than numerical wall.

Tell us which of these is wrong, and what we missed.
