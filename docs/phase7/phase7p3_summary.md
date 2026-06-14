# Phase 7.3 — pH-coupled ORR mechanism: summary & honest taxonomy

Capstone for `~/.claude/plans/phase7p3-pH-coupled-orr-mechanism.md`
(no-new-data; GPT-hardened Handoff 45). Detail: `phase7p3_P0_protocol_lock.md`
(P0), `phase7p3_M_series_log.md` (M1a/M2/M3). Memory:
[[project-phase7p3-mechanism-shortlist]].

## Question

The frozen pH-6.39 water-route fit (θ_L) is **pH-flat**; the K₂SO₄ RRDE data
is not (onset moves, ring/selectivity swing, ring collapses in strong acid).
Can a physical pH-coupling reproduce the **frame-invariant** ring/selectivity
*magnitude* trends with shared kinetics + few params, preserve the locked
pH-6.39 fit, and survive a single held-out (pH 2)? Goal = **pH-trend
SUFFICIENCY, not identification.**

## What was done (all no-new-data, on disk)

- **P0.1 frame byte-test (STOP gate) — PASS.** An SHE-anchored formal
  potential reproduces the pH-6.39 lock byte-for-byte (max|Δcd|=max|Δpc|=0.0).
  The frame math is sound; proton dependence enters in ONE place (formal-
  potential shift XOR a kinetic c_H factor, XOR-guarded).
- **P0.2 onset extractor.** Disk onset is **not a robust discriminator**:
  pH 4 ≈ pH 6 (RHE slope +9…+11 mV/pH); the negative 3-pt slope is a
  pH-2-only artifact and disagrees with the ring-onset.
- **M1a onset selection (N0/N1a/N1b/A).** A (SHE-anchored kinetics) is
  **gauge-equivalent** to the frame nulls — it wins only at pH 2 by ~4%
  (coarse-grid noise) and loses at pH 4/6, carrying no shape info beyond a
  rigid relabel. No frame variant matches the ring **magnitude**.
- **M2/G surface-pH coupling.** c_H rates read the electrode-facet boundary
  trace (verified). Surface c_H is a **threshold switch** vs bulk pH (acidic
  ~10² mol/m³ at bulk ≤2.35; alkaline ~10⁻⁷ at ≥3.42) — the physical gate.
- **M3/C (LEAD).** C1 = electrochemical H₂O₂ reduction reading surface c_H
  (E°=1.765 V, deck-consistent). Within a **lock-preserving bracket
  (k0_factor ~[1e-20, 1e-15])**, C1 is OFF at pH 6.39 (lock ring 0.361 ≈
  N0 0.367) and ON at pH 2 (ring → 0). A single c_H-gated rate reproduces the
  frame-invariant pH-2 ring collapse while preserving the lock.

## Identified vs degenerate vs conditional (the honest taxonomy)

**Identified (supported by frame-invariant evidence):**
- The pH-2 **ring-magnitude collapse is real new physics**, not a gauge/frame
  artifact (M1a: no potential-axis operation reproduces a magnitude).
- It is a **pH-dependent peroxide yield/loss** that **switches on in strong
  acid** and is gated by the **surface** (not bulk) proton activity (M2/G).
- A **single surface-c_H-gated consumption rate is SUFFICIENT** to reproduce
  the collapse while preserving the pH-6.39 lock (M3 C1 bracket).

**Degenerate / conditional (NOT claimed):**
- **The onset/first-ET frame (A) is a gauge question** — SHE-anchored E_eq
  and an OCP/relabel frame are equivalent on the full disk+ring shape (M1a).
  No SHE-anchoring physics claim is made.
- **Disk-side attribution of the collapse is CONDITIONAL (faradaic favored,
  not proven).** When C1 is tuned so the pH-2 ring hits the data (0.069), the
  pH-2 deep-cathodic disk is −4.06 (C1) vs −3.94 (data) vs −3.57 (C1-off) —
  the data is ~3× closer to C1-ON, i.e. the ring collapse IS accompanied by
  the extra cathodic current faradaic C1 predicts (the disk was not tuned, so
  this is an independent check). But C1 over-shoots ~3% and the absolute scale
  is rpm-conditional, so **faradaic C1 vs a non-faradaic loss (C2 homogeneous
  decomposition / peroxide escape / ring-detection artifact) is favored but
  not cleanly separated** — an N₂ H₂O₂-reduction scan / rpm series / per-pH
  ring calibration would settle it (the top data-asks; unavailable here).
- **C's rate magnitude and proton order p are not pinned** — C is OFF at the
  pH-4 TRAIN condition (alkaline surface), so it is a lock-derived BRACKET,
  not a pH-4 fit; the bracket is wide and p is unidentifiable (β is a step,
  not a slope — M2/G). "The data does not pin C's order" is itself a finding.
- The absolute direct-4e vs series-2e+2e partition stays **rpm-degenerate**
  (single rotation rate; never claimed).

**Prediction vs fit.** pH 2 is the held-out: C1 was NOT fit to it (it cannot
be trained at pH 4). Within the lock-derived bracket C1 **predicts** the pH-2
collapse — consistent with the held-out — but over-predicts the magnitude at
full collapse; the honest status is "consistent within the C bracket," NOT a
precision prediction (plan §6).

**Cross-condition (M5).** At the slide-15 condition (Cs⁺, pH 4) the surface
is alkaline under load, so **C1 is OFF there** and leaves the Phase-7
slide-15 fit unchanged — consistent (no acid collapse expected at pH 4). The
slide-15 residuals (+0.09 V anodic-peak offset, secondary bump) are the
separate **E (cation/field)** Tier-3 question, not addressed by C and deferred.

## Digitization caveats

Fit the raw pH-6.39 LSV (the lock); the pH 2/4/6 3-panel is a medium-fidelity
figure (the digitized pH-4/6 ring under-reads vs the Exp Info scalars
≈0.35 ≈ model; the real signal is the pH<2.3 collapse). Selectivity/n_e are
diagnostics, never fit targets.

## Bottom line

Phase 7.3 closes the no-new-data question: **the pH-resolved discrepancy is a
real, surface-pH-gated, pH-dependent peroxide yield/loss (mechanism C) — the
onset (A) is a frame question, and a single c_H-gated rate is sufficient to
reproduce the frame-invariant trends and preserve the lock.** What it
CANNOT do without new experiments: attribute the collapse uniquely to
faradaic C1 vs a non-faradaic loss, or pin C's rate/order. Those are
N₂-scan / rpm / ring-calibration questions — the standing top data-asks.
