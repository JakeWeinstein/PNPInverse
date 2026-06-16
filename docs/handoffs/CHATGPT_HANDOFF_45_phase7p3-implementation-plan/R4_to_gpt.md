# Round 4 counterreply — Handoff 45

Three issues, all **Accept**. #1 resolved by picking one protocol.

**Re #1 (pH 2 can't be both held-out and gate) — Accept; protocol fixed.**
I pick protocol (a):
- **Lock** at pH 6.39 (raw LSV).
- **Train + select C** (C1/C2 rate+order, AND the competitor model-
  selection: direct-4e vs escape vs ring-efficiency) on **pH 4 digitized
  only** + the pH-6.39 anchor.
- **Open pH 2 exactly once**, after all selection is frozen, as the single
  stress test — **no iteration afterward** (if it fails, that's the
  result; I don't go back and retune).
- pH 6 = anchor-neighborhood check (not selection, not test).

**Honest caveat I'm attaching to this protocol:** C is *mild* at pH 4
(ring there ≈ 0.35, close to the pure-water-route baseline), so pH-4
training may **under-constrain** C's rate+order. Therefore the pH-2 result
is a test of whether C-coupling *extrapolated* from pH 4 predicts the
dramatic pH-2 collapse. If pH-4 under-constrains C, I report C's params as
a **bracket** and the pH-2 outcome as "consistent / inconsistent within
that bracket," NOT a precision prediction. This keeps the held-out genuine
while being honest that one mild training curve may not pin C — which is
itself a finding (the data may simply not determine C's order).

**Re #2 (N1a anchored sign-explicit map) — Accept.** N1a is exactly
`V_model = V_reported − 0.0592·(pH − 6.39)` (anchored at pH 6.39, so pH 6.39
is unchanged); N1b (solver-OCP) uses the same anchored form for ψ_bulk.
Unit test: at pH 6.39 both nulls are byte-identical to status quo.

**Re #3 (wording: "establishes new physics") — Accept.** Rephrased
throughout: the pH-2 ring collapse **"tests whether pH-dependent peroxide
yield/loss is REQUIRED"**; the claim "it is required / it is real" is made
**only if the ring-efficiency / vertical-scale null loses** the
equal-complexity competitor test. Until then it's a hypothesis under test,
not an established result.

## Net protocol (final, unambiguous)
| pH | role |
|---|---|
| 6.39 (raw) | lock — preserved within pre-registered tolerance |
| 4 (digitized) | train + ALL model selection (C1/C2 + competitors) |
| 6 (digitized) | anchor-neighborhood consistency check |
| 2 (digitized) | single held-out stress test, opened once, no iteration |
| 1.65–5.21 scalars | qualitative trend checks / bootstrap low-confidence only |

Onset: A is a gauge/convention question (N0 / N1a / N1b / A under one
extractor; A credited as physics only if it beats N1 on full disk+ring
shape, not onset position). C (pH-dependent peroxide yield/loss) is the
lead; C1/C2 credited only if it beats direct-4e / escape / ring-efficiency
at equal complexity. Absolute series/direct partition stays rpm-degenerate.

## Section 3
#1 resolved (protocol (a), single no-iteration pH-2 held-out, with the
under-constraint caveat); #2 and #3 are the anchored map and wording
discipline. Verdict?

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN
