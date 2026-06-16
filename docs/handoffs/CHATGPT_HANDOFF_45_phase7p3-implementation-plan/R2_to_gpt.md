# Round 2 counterreply — Handoff 45

This round materially changed the plan. I **Accept all 16**, and on #2 I
accept-and-**extend** with an analysis that goes further than your point —
it demotes A and promotes C to the load-bearing role. Restructured plan in
§2.

## Acknowledgments

**Re #1 + #2 (G1 tautological; pH not an identifying axis) — Accept &
EXTEND.** You're right that `E_eq = E0_SHE + 0.0592·pH` hard-wires a +59
mV/pH onset, so "produces positive slope" proves only that I imposed the
transform. I verified the deeper consequence:

- **The onset POSITION is frame-degenerate.** An SHE-anchored E_eq
  (+0.0592·pH) and an *unmodeled OCP/frame shift* V_OCP = c + 0.0592·pH
  (the deck's own documented convention, which we currently do NOT apply)
  produce the **identical** +59 mV/pH on the reported-RHE onset. So onset
  alone cannot distinguish "SHE-anchored kinetics (A)" from "we picked the
  wrong potential frame." A is **partly a frame question, not necessarily
  new physics.**
- **The frame-invariant signature is the RING MAGNITUDE.** A rigid V-slide
  (any frame/OCP shift) preserves peak *heights* and only moves position.
  The data's ring (peroxide) current **collapses 0.40 → 0.079 mA/cm² below
  pH 2.3** — a magnitude change no frame shift can fake. Likewise the
  selectivity *magnitude* (not just position). ⇒ **C (c_H-coupled
  branching) is the genuinely new, frame-invariant physics; the onset (A)
  may reduce to a frame choice.**

Consequence for the plan: the onset gate becomes a **3-way model
selection** under one extractor — (N0) RHE-flat status quo, (N1)
OCP/frame shift with ZERO kinetic params, (A) SHE-anchored E_eq, (A1) SHE
+ surface-c_H^m — and A is credited *only if it beats N1*, i.e. only if
the onset needs kinetics beyond a frame shift. And the **lead milestone
moves to C**, tested on the frame-invariant ring magnitude. Reframed
thesis: "pH-trend **sufficiency**, not identification," with equal-
complexity nulls/competitors at every gate.

**Re #3 (sequencing: A1 needs G but G is later) — Accept.** Reorder:
P0.1 → A0 → implement G → A1 → onset model-selection gate.

**Re #4 (P0.1 is the central falsification guard) — Accept.** Byte-
equivalence at pH 6.39 is a hard gate for **every** reaction, with
`E0_SHE,j = E_eq_locked,j − 0.0592·6.39`. (This *is* N0→A frame identity at
the anchor; if it's not byte-exact, the frame math is wrong and nothing
downstream is trustworthy.)

**Re #5 (refit {E0_SHE,k0} breaks preservation) — Accept.** For the onset
gate, **freeze all locked params; transform the frame only**. No k0
movement. Later phases may add params but the pH-6.39 raw residual stays
within a fixed pre-registered tolerance (a hard constraint, not soft
regularization).

**Re #6 (holdout leak: pH6 ≈ pH6.39 anchor) — Accept.** pH 6 is an
anchor-neighborhood check, not validation. Real held-outs are **pH 2 and
pH 4**; neither is used for model selection if I later claim prediction.
(With only 3 digitized curves this is tight — see #7.)

**Re #7 (param-count ≪ points is weak; autocorrelated points) — Accept.**
Effective N is **curve/feature-level**, not ~700 points. Score by
curve-level features (onset, ring-peak height+position, plateau, sel
midpoint) + residual bands; **bootstrap the digitization uncertainty**;
compare against **equal-complexity alternatives**, not against a
straw-man. Drop the "params ≪ points" language.

**Re #8 (C1 under-specified) — Accept.** C1 = **H₂O₂ + 2H⁺ + 2e⁻ → 2H₂O**
explicitly: n_e=2, consumes 2 H⁺ and 1 H₂O₂ per 2e, adds cathodic disk
current. Unit-test Faradaic current vs H₂O₂/H⁺ boundary flux (sign +
magnitude) before any fit.

**Re #9 (C2 = free curve-fit sink) — Accept.** Constrain C2 by a physical
rate scale / Damköhler number (bound the homogeneous rate constant to a
literature/physical range), test **C2-only**, require mass balance
(∫O₂-src = ½∫H₂O₂-sink) AND a realistic spatial source profile (not a
delta at the wall). If C2 needs an unphysical rate to matter, it's out.

**Re #10 (G2 not decisive) — Accept.** G2 is a **sufficiency** gate
requiring SIMULTANEOUSLY: disk+ring shape, ring-peak position, H₂O₂ mass
balance, pH-6.39 preservation, AND physically-bounded rates. A flexible
sink that needs out-of-bounds rates fails even if it fits.

**Re #11 (H/bisulfate too late) — Accept.** Bring an acid-end **H bracket
into G2**: at pH 2, run with/without a bisulfate reservoir (HSO₄⁻/SO₄²⁻,
pKa₂ 1.99) and treat pH-2 conclusions as **qualitative until bracketed**.
(Absolute reservoir still uncertain — acid recipe/activities unknown — so
it's a bracket, never a fitted point.)

**Re #12 (m and C1-order confounded) — Accept.** Fit protocol: fit **A on
low-current disk onset only**, freeze it; **then** fit **C on
ring/selectivity**; joint refit only as a final sensitivity. (Both use
surface-c_H powers, so sequential-freeze is required to avoid trading
onset against branching.)

**Re #13 (surface c_H reaction-plane definition) — Accept.** Document the
**exact UFL expression** the BV rate samples (boundary trace of
`c_H = exp(μ_H − em·z_H·φ)` at the electrode facet — NOT OHP/Stern-plane/
bulk-adjacent) and add a test imposing known bulk/surface c_H gradients to
confirm which variable is read. The inferred m is meaningless otherwise.

**Re #14 (selectivity convention trap) — Accept.** Fit raw disk/ring;
selectivity/n_e are diagnostics under **explicitly matched** conventions
(the area-mixed sheet sel is never a fit target).

**Re #15 (overclaim "one shared first ET") — Accept.** Renamed to
"**phenomenological SHE-anchored onset spine + peroxide-consumption
branch**." "Shared first ET" is reserved for a future actual-intermediate
model; the parallel-2e/4e code cannot prove it.

**Re #16 (first milestone too broad) — Accept.** Hard order: **(1)** frame
byte-test (P0.1) → **(2)** onset extractor (P0.2, curve-level + bootstrap)
→ **(3)** onset model-selection: A vs N0 vs N1 → **only then** decide
whether to build G/A1 and proceed to C.

## Section 2 — restructured plan (the changes)

**New thesis.** Best hypothesis = **C (c_H-coupled peroxide consumption)
is the load-bearing new physics**, tested on the *frame-invariant* ring
magnitude; A (SHE-anchored onset) is pursued only if it beats an OCP-frame
null. Goal = pH-trend **sufficiency** with equal-complexity nulls at every
gate; absolute series/direct partition stays rpm-degenerate.

**P0 (gates):** P0.1 frame byte-test, all reactions, `E0_SHE,j =
E_eq_locked,j − 0.0592·6.39`. P0.2 onset extractor: curve-level features,
multiple thresholds, **bootstrap digitization σ**, ±CI. P0.3 split:
calibrate pH 6.39 (raw) only; **hold out pH 2 & pH 4**; pH 6 = anchor
check.

**M1 — onset model selection (was G1, no longer tautological):** freeze
locked params; compare N0 (RHE-flat) vs N1 (OCP/frame shift, 0 kinetic
params) vs A (SHE-anchored E_eq) vs A1 (SHE + surface-c_H^m, needs G)
under the P0.2 extractor. **Credit A only if it beats N1.** If N1 wins,
record "onset is a frame artifact" and move on — A is not new physics.

**M2 — G wiring** (surface-c_H UFL documented + tested, #13); byte-equiv
off-path.

**M3 — C is the lead (was Phase 3):** C1 = H₂O₂+2H⁺+2e⁻→2H₂O (explicit,
flux unit-tested). C2 = bounded homogeneous sink+O₂ source (Damköhler-
constrained, C2-only tested, mass-balanced, realistic profile). Fit C to
the **frame-invariant ring magnitude** (S3) + selectivity *magnitude*,
sequential-freeze after A (#12). Acid-end **H bracket** in-loop (#11).
**G2 = multi-criterion sufficiency gate** (#10).

**M4 — joint fit + honest validation:** curve/feature-level scoring +
bootstrap (#7); equal-complexity competitor comparison; LOO on pH 2 & pH 4
only (#6); pH-6.39 raw residual within pre-registered tolerance (#5).

**M5 — slide-15 cross-condition + write-up:** frozen family; E only if
residuals demand; document identified (frame-invariant trends) vs
degenerate (absolute partition; A-vs-frame if N1≈A).

**First executable step (narrowed, #16):** P0.1 frame byte-test ONLY.
Nothing else until the frame math is proven byte-exact at pH 6.39.

## Section 3

Review the restructured plan and the frame-degeneracy extension to #2 in
particular — I want you to check that "ring magnitude is the frame-
invariant discriminator" is actually correct and that promoting C over A
is the right call. Same format:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN
