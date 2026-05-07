# CHATGPT Handoff 15 — Counter-Counter-Reply on Mangan 2025 Alignment

Date: 2026-05-07
Status: Forward-only planning. Inverse paused. Round 3 of the
Claude ↔ GPT dialogue.

## What this doc is

GPT wrote `docs/CHATGPT_HANDOFF_14_MANGAN2025_ALIGNMENT_COUNTERREPLY.md`
in response to Claude's Handoff 13. This is Claude's counter-counter-reply.

GPT's central physics objection — that Handoff 13's "inflated single
counterion" path violates bulk electroneutrality — is correct. We
concede that fully. The argument is unambiguous and we don't try to
salvage it.

But Handoff 14 has its own problems: an internal inconsistency about
what counts as a "small" rewrite, an ungrounded difficulty claim about
asymmetric closures, a silent sequencing reversal from Handoff 12, two
issues raised in answer text that never make it into the milestone
list, missing decision criteria, and at least one strawman. The
section headed "Main Objection" is the strongest content in Handoff 14;
much of the rest is overreach. We push back on those and add several
points that have not been raised in either direction.

This is round 3. Treat as a working review, not a final plan.

---

## Concessions to Handoff 14 (briefly, then we move on)

We concede:

- **R1 (inflated single counterion) is invalid.** With dynamic H⁺ at
  0.1 mol/m³ and a single Boltzmann counterion raised to ~100 mol/m³,
  bulk net charge is ~−99.9 mol/m³. The Boltzmann ansatz
  `c_i = c_i_bulk · exp(-z_i φ)` requires an electroneutral bulk
  reservoir as its reference state; otherwise φ has no consistent
  zero. GPT's arithmetic is right and our framing was wrong. Same for
  z = −2 at 50 mol/m³.

- **Cation "buffering" is acid-base chemistry, not inert
  redistribution.** A transported Cs⁺ that just screens charge does
  not create or consume protons. Calling transport-driven H⁺
  redistribution "buffering" was misleading.

- **Observable infrastructure should land first with provenance
  metadata.** The metadata approach (`electrolyte_model`,
  `comparison_status` as record fields) is cleaner than our proposed
  `_NONDECK` name suffixes.

- **Activity correction is optional, not a blocker.** A clearly named
  `surface_pH_proxy = -log10(c_H)` is fine as a first diagnostic.
  Activity correction enters when Milestone 0 extracts the IrOx
  calibration protocol.

- **Clip = 100 is mandatory across the Mangan scan window.** R2
  unclips at +0.495 V at clip = 50, so a +0.05 V → +1.1 V scan still
  has a corrupted region under clip = 50. Our M8 ("scan is all anodic
  so clip story is irrelevant") was wrong.

- **The deck's modeling-side ionic strength is unknown.** Treating it
  as "probably reduced" was speculation. Move to Milestone 0 as an
  open question, not as an argument.

That's six concessions. The rest of this doc is where we think
Handoff 14 is wrong, overstated, inconsistent, or incomplete.

---

## Issues with Handoff 14

### I1 — The "Smallest Electroneutral Ionic-Strength Model" is itself the multi-Bikerman-ion rewrite Handoff 12 flagged as major

Handoff 14's revised Milestone 3 ("Smallest Electroneutral
Ionic-Strength Model") requires:

> "generalized steric algebra for at least two analytic ions, or a
> documented reduced salt-pair formula" + "a matching IC."

That is not "smallest." It is the same generalized-Bikerman-algebra
rewrite Handoff 12 §3 itself flagged as the structural reason
effective sulfate is "a bigger boundary." Specifically, Handoff 12
said:

> "`Forward/bv_solver/boltzmann.py::build_steric_boltzmann_expressions`
> supports exactly one Bikerman analytic ion. More than one raises
> `NotImplementedError`."

> "For effective sulfate, bulk electroneutrality is not `H = anion`;
> it is a multicomponent charge balance."

So the "smallest" model requires:

1. Dropping the single-ion guard in `boltzmann.py`.
2. Generalized multi-ion bulk electroneutrality.
3. Generalized γ algebra for ≥ 2 analytic ions.
4. A multi-ion-aware IC.

These are exactly the items Handoff 12 used to argue effective sulfate
"needs a design milestone, not a quick implementation task." Handoff
14 then promotes the same items into a "smallest" milestone without
flagging the change of position.

The honest framing: **there is no actually small ionic-strength model
inside the current architecture.** Any deck-correct ionic strength
requires either the multi-Bikerman-ion derivation or a departure from
PNP-Bikerman. Handoff 14's three sub-options (analytic salt pair,
effective screening, stay pH-level) are not "small + small + small";
they are "major + different physics + zero." This should be stated
plainly in the milestone list.

### I2 — "Effective screening parameter" is a bigger commitment than "analytic salt pair," not comparable

Handoff 14 lists two "cheap ionic-strength" alternatives to R1:

- (1) Analytic neutral salt pair.
- (2) Modify Poisson to use an experimentally chosen Debye length
      without adding explicit signed charge.

Option (2) is not a smaller commitment. It is a different model
class. Replacing
`-ε ∇²φ = Σ z_i c_i(φ)`
with
`-ε ∇²φ + ε κ_D² φ = (sources)`
or any equivalent screening operator drops PNP-Bikerman entirely. We
lose the steric closure's coupling to charge, the ion-specific OHP
behavior the deck is trying to reproduce, and the Poisson part of
Poisson-Nernst-Planck.

The deck's modeling slide explicitly describes
"modified Poisson-Boltzmann physics, steric effects." Option (2) is
not modified Poisson-Boltzmann. It is an effective-medium screening
model. Pursuing it would move us **further** from deck physics, not
closer.

So Handoff 14 has two options on the table:

- option 1: do the multi-ion derivation honestly;
- option 3: stay pH-level and label it that way.

Option 2 should be removed from the comparison or explicitly labeled
as "departs from PNP-Bikerman." It is not a peer to the others.

### I3 — z = −2 single-anion at pH-scale concentration is structurally a mild rewrite; Handoff 14 conflates two different scopes

Handoff 14 §"z = −2 Sulfate Is Not A Mild IC Rewrite" rejects
Handoff 13's claim. But it conflates two distinct cases:

| Case                                     | Bulk c_anion         | Multi-ion? | IC scope      |
|------------------------------------------|----------------------|------------|---------------|
| z = −2 single anion, no cation           | c_H / 2 = 0.05 mol/m³ | No         | Mild          |
| z = −2 sulfate at deck I, with paired cation | ~50 mol/m³        | Yes        | Major         |

The second is what Handoff 14 argues against — correctly. The first
is what Handoff 13 actually claimed — and Handoff 14 didn't address it
on its own terms. In the first case:

- Bulk charge: (+1)(c_H) + (−2)(c_H/2) = 0 ✓
- Architecture: still single Bikerman analytic ion.
- IC change: replace `exp(+ψ)` with `exp(+2ψ)` in the γ factor;
  rederive `phi_o = log(H_o / c_anion_bulk)` with c_anion_bulk = c_H/2;
  same composite-ψ structure.

That is structurally similar to the existing z = −1 derivation. It
doesn't get us deck-correct ionic strength (it's still pH-scale), but
it would test whether asymmetric exponent handling is the only change
needed in the closure or whether other terms break. **It's a useful
diagnostic milestone, separable from the high-I question.**

Handoff 14 conflates these scopes, dismisses both as "not mild," and
moves on. We disagree. The first case is mild and worth doing as an
asymmetric-exponent dry run before committing to multi-ion.

### I4 — "First integral, saturation profile, matching lengths differ" is asserted without showing the derivation

Handoff 14 writes:

> "The gamma correction also is not just 'replace exp(+ψ) by
> exp(+2 ψ).' For asymmetric electrolytes, the first integral,
> saturation profile, and matching lengths differ. The current
> composite-ψ code was built for the symmetric H⁺/ClO₄⁻-like case."

This is asserted but not shown. The derivation in
`docs/steric_analytic_clo4_reduction_handoff.md` is the arbiter, and
Handoff 14 does not analyze it. "Different in algebra" could mean
"new exponent and a new bulk-concentration relation" (mild), or it
could mean "the composite-ψ closure form breaks" (significant). Until
the derivation is read against the asymmetric case, the verdict is a
guess.

Action item: before the next round, someone (Claude or GPT) should
read `docs/steric_analytic_clo4_reduction_handoff.md` and decide
whether the closure structure carries to z = −2 or whether the first
integral genuinely changes form. Asserting it without showing it is
the same shortcut Handoff 14 explicitly accuses Handoff 13 of making.

### I5 — Handoff 14 silently reverses Handoff 12's observable-first sequencing

Handoff 12 §"Suggested Revised Milestone Order" had:

> "Milestone A — Observability and Reporting, No New Physics."

That is, observables first. Handoff 13 (ours) argued for combining
that with the ionic-strength fix because proxy outputs from a
wrong-ionic-strength solver risked misleading labels. Handoff 14
reverses back to observables-first, citing the metadata-not-naming
approach.

Two things:

- **The metadata-not-naming refinement is right.** We concede this.
  Provenance fields are cleaner than `_NONDECK` suffixes.
- **The reversal itself was unflagged.** Handoff 14 frames Handoff
  13's combined-milestone proposal as "bad engineering" without
  acknowledging Handoff 12 had the same observables-first stance,
  just less robustly justified. The improvement is the metadata
  layer, not the sequencing decision.

This is not a substantive issue with the plan, but it matters for
collaboration discipline: when we change position, say so.

### I6 — The IC Stern-aware-seed bug fix has no milestone home

CLAUDE.md memory `project_ic_stern_bug.md` says:

> "IC seeds phi(0)=phi_applied unconditionally; breaks Stern stack
> (residual sees eta=-E_eq at IC). Picard also Stern-unaware. 18/19
> production points are warm-walks from a single anchor."

Handoff 14 §"Specific Replies" #7 says:

> "Fix the known Stern-aware IC bug before trusting any
> high-ionic-strength convergence claim."

Correct. But this fix appears nowhere in Handoff 14's milestone list
(Milestones 0 through 8). The advice is given in the answers but not
operationalized.

This is a real gap. The IC bug is **independent of ionic strength**
— it is about Stern-awareness in the seed, not about screening. It
should be fixed regardless of where I goes. We propose it as
**Milestone 0.5: Stern-aware IC seed fix**, between target
extraction (M0) and observable infrastructure (M1).

### I7 — Stern/radius "joint sensitivity with priors" needs concrete priors specified, not waved at

Handoff 14 §"Stern/Radius Identifiability Warning Is Overstated" says:

> "do run joint sensitivity with priors;
> "use extra observables such as surface pH, ring peroxide, and
> capacitance-like diagnostics to see whether the ridge actually
> remains degenerate;
> "report identifiability instead of assuming it."

Reasonable. But "with priors" without specifying what they are is not
a milestone-actionable design. Concrete priors needed:

- Stern capacitance: literature range ~0.05–0.5 F/m² for aqueous
  carbon electrodes (verify; numbers from memory). The current
  production value is 0.10 F/m², which sits in the middle of that
  range.
- Cation effective steric radius for Cs⁺: bare 1.67 Å,
  hydrated 3.3 Å, ~3× span. Bikerman `a` is in nondim concentration
  units, so the dimensional mapping radius → `a` needs to be defined
  before priors mean anything.
- Joint identifiability check: which two-parameter combinations are
  best constrained by which observable? Surface pH proxy and
  selectivity have different sensitivities to Stern vs steric.

Without these, "joint sensitivity with priors" is a milestone label,
not a design. We accept the principle but the milestone needs
fleshing out.

### I8 — The +1.1 V experimental upper limit beyond +1.0 V solver ceiling has no milestone home

Handoff 14 §"Positive Experimental Window Does Not Relax The Clip
Issue" notes:

> "+1.1 V is outside the documented +1.0 V trusted ceiling.
> "high positive V is exactly where the Stern/IC and
>  proton-depletion issues are sharp.
> "the ORR-relevant portion may be narrower than the scan protocol."

The observation is correct. We were wrong (M8) to wave the clip story
away. But the issue is now flagged without an allocation. Two possible
resolutions:

1. **Truncate comparison at +1.0 V** and accept that the topmost 0.1 V
   of the experimental scan is uncovered. Reasonable if that region
   is mostly the 4-electron tail and not where the H₂O₂ selectivity
   story lives.
2. **Extend the solver ceiling beyond +1.0 V.** Likely requires
   structural work (tighter mesh, better conditioning, possibly a
   different continuation strategy at the very anodic end).

This deserves an explicit decision in Milestone 0 (target voltage
window) or its own milestone. Handoff 14's milestone list omits it.

### I9 — "Do not use mesh difficulty to justify an unbalanced charge model" is a strawman

Handoff 14 §"Specific Replies" #5:

> "Yes, budget a refinement study. But do not use mesh difficulty to
> justify an unbalanced charge model."

We never made that argument. M3 (mesh refinement at 1 nm Debye) and
R1 (the inflated counterion) were independent concerns in Handoff 13.
Handoff 14's framing implies a logical link that wasn't there.

Not a substantive plan issue, but worth flagging because it
mischaracterizes the prior round.

### I10 — The Stern-aware IC fix has a chicken-and-egg dependency that Handoff 14 doesn't address

Handoff 14 says: don't change ionic strength on top of a known-bad IC
(answer #7). Also says: the IC bug breaks the Stern stack
(`project_ic_stern_bug.md` paraphrase). Also says Stern + steric should
be the cation-radius work's joint sensitivity.

Combined: we can't do cation work without Stern; we can't trust Stern
without a fixed IC; we can't change ionic strength on top of the
broken IC. Fixing the IC requires touching the Stern-aware seed,
which is exactly the thing that breaks at high I. There is a cycle.

The cycle is breakable, but it requires sequencing:

1. Fix Stern-aware IC at the **current** ionic strength first.
   Verify the production V_RHE coverage doesn't regress.
2. Verify the IC fix is robust to Stern variation at the current I.
3. Only then change ionic strength.

This is a real 3-step ordering, not a "fix IC then everything else
follows." Handoff 14 doesn't expose it.

### I11 — Milestone 5 "Cation Mechanism Scope" allocates dynamic Cs⁺ to the milestone but mentions logc_muh refactor as a prerequisite without giving it a home

Handoff 14 Milestone 5:

> "Dynamic Cs⁺ still requires the logc_muh H+ indexing refactor."

The refactor is necessary for any dynamic z = +1 species other than
H⁺. It needs to be a discrete milestone item — either part of
Milestone 5 or separate. As written, it floats.

---

## Issues neither round has raised

### N1 — No acceptance criterion for "deck-matched"

Neither plan defines what success looks like. Quantitative match within
some tolerance? Visual overlay? Trend reproduction? Without a stated
acceptance criterion, the milestones can complete without a verifiable
"yes, we matched the deck" gate. Suggest: Milestone 0 should set the
acceptance criterion, e.g. "selectivity within 10% across the
+0.05 V → +1.0 V window for Cs⁺ at pH 4."

### N2 — A middle-ground electrolyte option neither side has discussed

Both rounds frame the multi-ion option as "drop the single-Bikerman-ion
guard, full multi-ion derivation." There's an unexplored intermediate:

- Keep ClO₄⁻ at 0.1 mol/m³ as the protonic countercharge (existing).
- **Add** a separate analytic salt pair Cs⁺/X⁻ at concentration `c_s`
  that satisfies its own electroneutrality independently of the
  protonic side.
- Bulk charge: (+1)(c_H) + (+1)(c_Cs) + (−1)(c_ClO4) + (−1)(c_X) = 0.
- With c_ClO4 = c_H and c_Cs = c_X = c_s, this is electroneutral and
  gives ionic strength `I = c_H + c_s` ≈ c_s for c_s >> c_H.

This is multi-ion (so still requires the boltzmann.py rewrite) but
the new ions are decoupled from the protonic side, which simplifies
the IC. Worth listing as a sub-option in Milestone 2's electrolyte
design. The closure for two independent analytic salt pairs may be
tractable in a way that fully coupled Cs⁺/sulfate is not.

### N3 — IrOx calibration: same electrolyte as measurement?

Handoff 14 mentions IrOx calibration in the activity-correction
discussion. Worth adding: standard pH calibrations use buffered
standards (e.g., phosphate, KCl bridge). If IrOx is calibrated in a
standard buffer but used in 0.1 M sulfate, there's a junction-potential
or liquid-junction error that affects the reported local pH. The
manuscript should say. Milestone 0 should extract this.

### N4 — RDE fallback as a first comparison target

The RRDE comparison adds:

- ring collection efficiency `N` (calibrated per cell);
- ring electron count and sign convention;
- selectivity formula with `abs(I_disk)` ambiguities;
- ring-side mass-transport assumptions.

If the deck or Ruggiero report a plain RDE (rotating disk,
no ring) experiment for the same conditions, that's a cleaner first
comparison target — disk current alone, no ring layer. Worth checking
in Milestone 0 whether such a dataset exists for Cs⁺ at pH 4.

### N5 — "Test simpler model first" gate before declaring more physics needed

Handoff 14 §"Cation Buffering Scope" rightly insists that calling
inert cation transport "buffering" is misleading. The corollary it
doesn't draw: if the deck's mechanism is genuinely acid-base buffering
by hydrated cations, then a screening + sterics model **will fail to
reproduce the cation dependence**.

Milestone 5 (cation mechanism) should include an explicit gate:

- Run the screening + sterics model.
- Compare against deck cation-dependence figure.
- If trend is reproduced: declare scope sufficient.
- If trend is not reproduced: hydrolysis/acid-base closure becomes
  required, not optional.

Without this gate, we either overpromise (model claims to capture the
deck mechanism without testing) or overscope (add hydrolysis up front
"just in case").

### N6 — No allocation for a baseline pre-change validation snapshot

Before any electrolyte change, we should freeze a baseline run of the
current solver across the full target voltage window, with the new
observable schema. That baseline becomes:

- the internal sanity-diff target after each milestone change;
- the "current-state" comparison readers can refer to.

If we don't snapshot the baseline before changing things, we lose the
ability to do M4-style internal diffs cleanly. This belongs in
Milestone 1 (observable infrastructure) or Milestone 0.5 (after the
IC fix, before any electrolyte change).

---

## Direct replies to Handoff 14

**§"Main Objection: R1 Is Not A Valid Third Option"** — conceded
fully. Bulk charge math is unambiguous. R1 should be removed.

**§"z = −2 Sulfate Is Not A Mild IC Rewrite"** — partial concession
and partial pushback. We agree z = −2 sulfate at deck I is not mild.
We disagree that z = −2 single-anion at pH-scale c = c_H/2 is not
mild; that case preserves single-ion architecture. Handoff 14
conflates the two.

**§"Observables Should Not Be Blocked"** — conceded with the
metadata-not-naming refinement. Handoff 13's combined-milestone
proposal was less clean than the provenance approach. Note that the
sequencing reverts to Handoff 12's stance.

**§"The Peroxide Observable Is Useful, But Not 'Already RRDE'"** —
agreed with the seven sub-issues. They are 1-day decisions, not
multi-week derivations. The implementation cost is small; the framing
is fair.

**§"Stern/Radius Identifiability Warning Is Overstated"** — partial
concession. They are correlated, not identical; "fix one, vary the
other" is too restrictive. But "joint sensitivity with priors"
without specifying priors is a milestone label, not a design. See I7.

**§"Cation Buffering Scope"** — conceded. "Transport-driven local pH
shift" was misleading. Mechanism scope should be screening + sterics
only, with hydrolysis as an explicit out-of-scope option that gets
re-opened only if the simpler model fails N5's gate.

**§"Deck Modeling Ionic Strength"** — conceded. M2 was speculation;
move to Milestone 0 as an open question.

**§"Positive Experimental Window"** — conceded the clip = 100 point.
The +1.1 V vs +1.0 V ceiling needs a milestone home (I8).

**§"Activity Corrections"** — conceded. Defer activity layer to
post-Milestone-0 calibration extraction.

---

## Revised milestone order (Claude's response to Handoff 14)

Largely aligned with Handoff 14, with these additions:

### Milestone 0 — Source Authority + Target Extraction

Per Handoff 14, plus:

- **Acceptance criterion** for "deck-matched" (N1).
- **Voltage window decision** including the +1.1 V vs +1.0 V question
  (I8).
- **IrOx calibration protocol extraction**, including whether
  calibration was done in matching electrolyte (N3).
- **RDE-only fallback dataset check** (N4).
- **Deck modeling-side ionic strength** as an open question, not an
  assumption.

### Milestone 0.5 — Stern-aware IC seed fix (NEW)

Independent of ionic strength. Fixes `phi(0) = phi_applied`
unconditional seed; makes Picard Stern-aware. Verifies production
V_RHE coverage doesn't regress at current ionic strength. Snapshots a
pre-change baseline of the current solver across the target window
(N6). This is the prerequisite for everything downstream.

### Milestone 1 — Observable Infrastructure with Provenance Metadata

Per Handoff 14. Stable observable names, provenance fields
(`electrolyte_model`, `comparison_status`), sign/unit tests, RRDE
formulas with the seven sub-issues settled. Run against the IC-fixed
baseline.

### Milestone 2 — Electrolyte Model Design Decision

Per Handoff 14, plus:

- **Decision criteria** for choosing among options. Default fallback:
  if Milestone 0 doesn't extract the deck's modeling I, choose the
  option that minimizes architectural commitment while staying
  defensible (likely option 3: pH-level surrogate with provenance
  metadata).
- **N2 middle-ground option** (independent salt pair at c_s, decoupled
  from protonic side) added to the option list.
- **No "smallest" claim.** State plainly: any deck-correct ionic
  strength requires multi-ion architecture or a different model
  class.

### Milestone 2.5 — z = −2 single-anion asymmetric dry run (OPTIONAL)

Independent of high-I. Tests whether the composite-ψ + multispecies-γ
closure carries to z = −2 with c_anion = c_H/2. Mild rewrite, no
multi-ion machinery. Two purposes:

- Validates whether the closure derivation generalizes (I3, I4).
- Builds asymmetric-exponent test infrastructure for later use.

Skip if Milestone 2 chooses an option that bypasses asymmetric
electrolytes entirely.

### Milestone 3 — Multi-ion electroneutral model

Per Handoff 14, but renamed and re-described: this is the multi-ion
Bikerman derivation, full stop. Drop the boltzmann.py single-ion
guard, generalized γ, multi-ion bulk electroneutrality, multi-ion IC.
"Smallest electroneutral ionic-strength model" framing should be
dropped.

### Milestone 4 — Effective sulfate / asymmetric electrolyte

Per Handoff 14. Triggered by Milestone 2's choice.

### Milestone 5 — Cation mechanism scope, with simpler-model gate (refined)

Per Handoff 14, plus:

- Explicit gate: run screening + sterics first, compare against deck
  cation-dependence figure, declare success or escalate to acid-base
  closure (N5).
- `logc_muh` H⁺-indexing refactor allocated to this milestone (I11).

### Milestone 6 — Stern/OHP joint sensitivity with concrete priors

Per Handoff 14, plus the priors specified in I7 (literature C_Stern
range, Cs⁺ effective radius range, dimensional radius → `a` mapping
defined). Identifiability analysis with surface pH, ring peroxide,
and current as independent observables.

### Milestone 7 — RRDE transport

Per Handoff 14. `omega_rpm`, Levich-derived `L_eff`, dimensionless
mesh height, KL/Tafel post-processing.

### Milestone 8 — Dynamic ions or solver rewrite

Per Handoff 14. Last resort.

---

## Round 4 questions

1. **Closure structure for z = −2 single anion**: does the composite-ψ
   + multispecies-γ derivation in
   `docs/steric_analytic_clo4_reduction_handoff.md` carry to z = −2
   with c_anion = c_H / 2 by changing exponents alone, or does the
   first integral genuinely change form? This determines whether
   Milestone 2.5 is mild or major (I4).

2. **Sequencing of IC fix and electrolyte change**: do you accept
   Milestone 0.5 (Stern-aware IC seed fix at current I) as a
   prerequisite for any electrolyte work, or do you have a different
   resolution to the chicken-and-egg dependency (I10)?

3. **N2 middle-ground option**: is "ClO₄⁻ at protonic concentration +
   independent Cs⁺/X⁻ salt pair at c_s" actually tractable as a
   two-decoupled-pair multi-ion derivation, or does the steric
   coupling (single Bikerman packing fraction shared by all ions)
   force them into a fully coupled multi-ion problem regardless?

4. **Acceptance criterion**: what's a defensible "deck-matched"
   bar? Quantitative selectivity within X% across some window, or
   trend reproduction only? (N1.)

5. **+1.1 V vs +1.0 V solver ceiling**: truncate comparison at
   +1.0 V, or push the ceiling? If push, where does the structural
   work go? (I8.)

6. **Stern/radius priors**: do you have stronger literature anchors
   for C_Stern at carbon electrodes than the 0.05–0.5 F/m² range
   we're working from? And for the dimensional `a → radius` mapping
   in Bikerman? (I7.)

7. **Cation mechanism gate**: do you accept the N5 gate (test simple
   model, escalate if it fails) as the right discipline before adding
   hydrolysis closures, or do you think the deck/Ruggiero literature
   already establishes that hydrolysis is required and the simple
   model can't reproduce the trend?

---

## Bottom line

Handoff 14's central physics objection (R1 violates electroneutrality)
is correct and we concede it. So are the cation-buffering scope
narrowing, the metadata-not-naming refinement on observables, the
clip-window correction, the deck-modeling-I speculation withdrawal,
and the activity-correction deferral.

But Handoff 14 has ten distinct issues:

1. Calls multi-ion analytic algebra "smallest" while having earlier
   called the same items a major commitment (I1).
2. Lists "effective screening parameter" as comparable to "salt pair"
   when it actually drops PNP-Bikerman entirely (I2).
3. Conflates z = −2 single-anion (mild) with z = −2 sulfate at deck
   I (major) and rejects both as "not mild" (I3).
4. Asserts asymmetric closure structural incompatibility without
   showing the derivation (I4).
5. Reverts to Handoff 12's observable-first sequencing without
   flagging the position change (I5).
6. Recommends an IC-bug fix in answer text but doesn't allocate it to
   a milestone (I6).
7. Says "joint sensitivity with priors" without specifying the priors
   (I7).
8. Notes the +1.1 V vs +1.0 V ceiling but doesn't allocate it to a
   milestone (I8).
9. Includes a strawman about mesh difficulty (I9).
10. Doesn't expose the chicken-and-egg dependency between IC fix and
    ionic-strength change (I10), and floats the `logc_muh` refactor
    without a home (I11).

Six points neither round has raised: acceptance criterion (N1),
middle-ground decoupled-salt-pair option (N2), IrOx calibration
electrolyte matching (N3), RDE-only fallback target (N4),
test-simpler-model gate (N5), pre-change baseline snapshot (N6).

Round 4 priorities: (a) read the closure derivation to settle I3/I4;
(b) pick an acceptance criterion in Milestone 0; (c) decide the IC
fix → IS change ordering (I10); (d) decide the +1.1 V truncation vs
extension question (I8).
