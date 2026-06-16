# Round 3 counterreply — Handoff 45

All ten are correct; **Accept** each. No defenses — these are the
executability/honesty tightenings the plan needed. Concrete incorporation
below, then the final tightened skeleton.

## Acknowledgments

**Re #1 (ring-invariance only for a relabel) — Accept, define N1 as two
distinct nulls:**
- **N1a = post-hoc voltage relabel** `I(V) → I(V−ΔV(pH))`, ΔV = 0.0592·pH.
  Ring magnitude IS invariant (proven, modulo window — #2). This is the
  "pure gauge" null.
- **N1b = solver-side OCP shift** ψ_bulk = V_OCP(pH) (the deck convention),
  which changes phi_applied, Stern drop, surface c_H, PNP fields. Ring
  magnitude is **NOT** assumed invariant here — it must be **verified
  numerically** (run N1b across pH, measure the ring-peak height change).
So the frame-invariance argument is stated only against N1a; against N1b
it becomes an empirical question the plan answers with a run.

**Re #2 (window truncation) — Accept.** Compare ring curves after
**optimal horizontal alignment on the overlapping voltage support**; use
peak height only when the full peak is inside the common window. (Relevant:
digitized sel/ring windows are narrow — sel pH2 only to 0.30 V.) Add a
"peak-in-window" check before any height comparison.

**Re #3 (M1/M2 order bug still present) — Accept.** Hard split:
**M1a** = N0 / N1a / N1b / A (no G, frozen locked params) → **M2** = wire G
→ **M1b** = A1 (SHE + surface-c_H^m) comparison. A1 never referenced before
G exists.

**Re #4 (protocol contradiction: C has no pH-6.39 signal) — Accept.** C is
designed to vanish at pH 6.39, so it cannot be estimated there. Honest
protocol: **lock at pH 6.39 (raw); train C on pH 4 (digitized); keep pH 2
TRULY untouched as the single genuine held-out test; pH 6 = anchor-
neighborhood check.** I will not claim "validation," only "one held-out
stress test at pH 2" (ties #10).

**Re #5 (C over-specified) — Accept.** Renamed **C = "pH-dependent
peroxide yield/loss"** (the frame-invariant signature). C1 (electro-
reduction) + C2 (homogeneous decomposition) are ONE hypothesis for it and
must **beat equal-complexity competitors**: (i) pH-dependent **direct-4e
branching**, (ii) pH-dependent **peroxide escape/desorption**, (iii)
pH-dependent **ring collection efficiency** (#6). C1/C2 is credited only
if it wins on the same data at the same param count.

**Re #6 (ring ≠ disk-branching discriminator) — Accept.** Added a
**ring-efficiency / vertical-scale null** (pH-dependent N or ring response)
to the competitor set. If it explains the ring collapse as well as C1/C2,
**disk-side attribution remains conditional** — stated explicitly. (No N₂
scans to rule ring artifacts out independently.)

**Re #7 (C1 electrochemical reference) — Accept, pre-register before
fitting:** C1 = H₂O₂ + 2H⁺ + 2e⁻ → 2H₂O; **E_eq** = standard
H₂O₂/H₂O couple on the chosen frame (pre-registered value), **α** treatment
(single α, pre-registered), **irreversible** cathodic form (no anodic
peroxide oxidation at the disk — pre-registered, since reverse would add a
free knob), conc factors = c_H₂O₂,surf × c_H,surf^p, **param count = 2**
(rate + order p). Locked in writing before any fit.

**Re #8 (gate leans on weakest data) — Accept.** **Primary C gate = the
digitized pH-2 ring CURVE** (shape + height on overlapping support), not
the pH-1.65 scalar. The six Exp-Info scalars are **qualitative trend
checks / bootstrap-weighted low-confidence features only**.

**Re #9 (A-vs-N1 untestable from onset alone) — Accept.** A and N1a are
gauge-equivalent on onset; **A is treated as a gauge/convention question**
unless the **full disk+ring shape residuals** (not onset position) give
independent evidence beyond a shift. A1's surface-c_H^m is flagged as a
**shape-fitting risk** — it is NOT evidence for SHE anchoring unless it
improves shape residuals against N1b at equal complexity. Likely honest
outcome: **A collapses to "pick the right frame"; C is the only new
physics.** The plan now states this as the expected result, not a failure.

**Re #10 (validation language too strong) — Accept.** M4 renamed
**"stress testing / internal cross-checking."** The ONLY genuine
prediction claim is the single pH-2 held-out (untouched through all model
selection, #4). Everything else is internal consistency.

## Section 2 — final tightened skeleton

**Thesis:** the frame-invariant signature (pH-2 ring-magnitude collapse)
establishes **pH-dependent peroxide yield/loss** as real new physics;
C1/C2 is the lead hypothesis for it and must beat competitors (direct-4e /
escape / ring-efficiency) at equal complexity. **A (onset) is a
gauge/convention question** (N1a≡A on onset; N1b verified numerically),
not assumed to be new physics. Absolute series/direct partition stays
rpm-degenerate. Goal = **sufficiency** with equal-complexity nulls at
every gate; ONE genuine held-out (pH 2).

- **P0.1** frame byte-test, all reactions, `E0_SHE,j = E_eq_locked,j −
  0.0592·6.39` → byte-exact at pH 6.39 or stop. **(first executable step,
  nothing else until this passes.)**
- **P0.2** onset extractor: curve-level features, multiple thresholds,
  bootstrap digitization σ, peak-in-window guard, ±CI.
- **P0.3** protocol: lock pH 6.39 (raw); train C on pH 4; **hold pH 2
  untouched**; pH 6 = anchor check.
- **M1a** onset model selection: N0 (RHE-flat) / N1a (relabel) / N1b
  (solver-OCP, ring-height measured) / A (frozen params) — credit A only
  if it beats N1 on **full-shape** residuals, else label "frame."
- **M2** G wiring (surface-c_H UFL documented + gradient-test, #13 prior).
- **M1b** A1 (SHE + surface-c_H^m) — only if M1a left onset unexplained;
  flagged shape-fitting risk.
- **M3 (lead)** C = pH-dependent peroxide yield/loss. C1 (pre-registered,
  flux-unit-tested) + C2 (Damköhler-bounded, C2-only tested, mass-
  balanced, realistic profile). Primary gate = digitized pH-2 ring curve.
  Competitors: direct-4e / escape / ring-efficiency, equal complexity.
  Acid-end H bracket in-loop. **G2 = multi-criterion sufficiency gate.**
- **M4** stress testing: curve/feature scoring + bootstrap; pH-2 single
  held-out; pH-6.39 raw residual within pre-registered tolerance.
- **M5** slide-15 cross-condition (frozen; E only if residuals demand);
  write-up: identified (frame-invariant pH-dependent peroxide yield/loss)
  vs degenerate (absolute partition; A=frame; disk-side attribution if
  ring-efficiency null not excluded).

## Section 3

I believe every issue across all three rounds is incorporated. The plan
now (a) leads with C on the frame-invariant pH-2 ring curve, (b) treats A
as a gauge question with N1a/N1b nulls, (c) has an honest single-held-out
protocol, (d) pre-registers C1 and bounds C2, (e) pits C1/C2 against
equal-complexity competitors, (f) fixes the M1/M2 ordering. Verdict?

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN
