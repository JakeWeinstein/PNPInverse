# Phase 7.3 — P0 pre-work: frame byte-test, onset extractor, protocol lock

Executes the **P0 block** of `~/.claude/plans/phase7p3-pH-coupled-orr-mechanism.md`
(GPT-hardened, Handoff 45 APPROVED). P0 is the no-data computational
foundation that must hold before any mechanism (M1–M5) is built.

Memory: [[project-phase7p3-mechanism-shortlist]].

---

## P0.1 — single-convention frame byte-test ✅ PASS (the STOP gate)

**Claim under test (plan §0/§2):** the model is RHE-referenced (V_OCP
cancels in η because `_build_reactions` shifts every E_eq by −V_OCP *and*
the V-grid by −V_OCP). An SHE-anchored formal potential
`E_eq_RHE,j(pH) = E0_SHE,j + S·pH = E_eq_locked,j + S·(pH − pH_anchor)`
(S = V_T·ln10 ≈ 0.05916 V/pH) must, at the pH-6.39 lock, reproduce the
locked fit **byte-for-byte** — else the frame math is wrong and nothing
downstream is trustworthy.

**Implementation (single shift point, opt-in, backward-compatible):**
`scripts/studies/solver_demo_slide15_dual_pathway_cs.py`
- `_she_eeq_shift_v(opts)` adds the SHE offset to every reaction E_eq inside
  `_build_reactions` (`E_eq_v + she_shift − V_OCP`). Default frame is `rhe`
  (offset 0.0); `--proton-frame she` opts in.
- **Byte-exactness mechanism:** the offset is anchored on the bulk **c_H**
  (`bulk_h_anchor_mol_m3`, default 4.07e-4 = pH 6.39), not a rounded pH, so
  at the anchor `pH_bulk − pH_anchor` flows through one helper → delta is
  **exactly 0.0** → identical E_eq float → identical solver output.
- **XOR guard:** in the SHE frame, an *enabled* reaction carrying a kinetic
  c_H factor (`cathodic_conc_factors`) raises — "formal-potential shift XOR
  kinetic c_H factor, never both." Water routes carry none (clean).
- Callers that never set `proton_frame` (the fit harness, the pH-series
  generalization) are byte-equivalent via `getattr(..., "rhe")`.

**Verification:**
- Reaction-level (instant, Firedrake-free): `tests/test_phase7p3_frame_byte.py`
  — 6/6 pass. Offset is exactly 0.0 at anchor; reaction dicts byte-identical
  RHE↔SHE at anchor; live off-anchor with correct Nernstian sign+magnitude;
  XOR guard fires; invalid frame rejected; default ≡ explicit RHE.
- Solver-level (end-to-end): `scripts/studies/phase7p3_p0_1_frame_byte_test.py`
  ran the locked pH-6.39 forward model (θ_L, V_OCP 1.019, K⁺, L_eff 21.7,
  water routes) in both frames. **Result: `max|Δcd| = max|Δpc| = 0.0`** over
  all 13 voltages, 13/13 converged in both, grids identical.
  `StudyResults/phase7p3_p0_1_frame_byte_test/verdict.json` (`PASS: true`).
- No regression: existing Phase 7 fast suite 35/35.

**Verdict: the frame math is byte-exact at the lock. Gate cleared.**

---

## P0.2 — disk-onset extractor ✅ (defensible onset ± CI)

`scripts/studies/phase7p3_p0_2_onset_extractor.py` →
`StudyResults/phase7p3_p0_2_onset/{onset_metrics.json, onset_vs_pH.png}`.
Unit-tested: `tests/test_phase7p3_onset_extractor.py` (5/5).

Extracts DISK onset (the first-ET current onset, signature S1) from the
digitized disk curves (pH 2/4/6) at thresholds {0.05, 0.10, 0.20} mA/cm²
**after per-pH background subtraction** (the curves carry a pH-dependent
high-V baseline −0.100 / −0.076 / −0.057 mA/cm²), with a **plateau-connected
onset** definition (isolated high-V digitization blips ignored) and a
digitization bootstrap (σ_V = 5 mV axis-reading, σ_y = 15 µA/cm² small
absolute — a full-scale fraction swamps the small thresholds and pins the
bootstrap at the window edge).

**Disk onsets (V_RHE, threshold-robust bootstrap CIs):**

| pH | thr 0.05 | thr 0.10 | thr 0.20 |
|----|----------|----------|----------|
| 2  | 0.679 | 0.598 | 0.581 |
| 4  | 0.500 | 0.440 | 0.414 |
| 6  | 0.474 | 0.457 | 0.436 |

**Key finding — onset is NOT a robust mechanism discriminator:**
- pH 4 ≈ pH 6 (clean, sustained): **pH-2-excluded RHE slope = +9…+11 mV/pH**
  (flat / slightly positive), and even its sign is threshold-unstable.
- Including the **noisy pH-2 high-V shoulder** flips the 3-pt RHE slope to
  **−35…−51 mV/pH**. ⇒ sign **not stable to dropping pH 2**.
- The disk-onset sign **disagrees** with the low-confidence Exp Info
  RING-onset slope (+41 mV/pH RHE, 6 pH) — a different feature (ring = when
  surviving peroxide appears; at pH 2 the ring turns on ~0.36 V, far below
  the disk's ~0.58–0.68 V shoulder, consistent with acid-accelerated
  peroxide consumption, mechanism C).
- On the SHE axis the disk onset *falls* steeply with pH (≈ −95…−110 mV/pH),
  i.e. it is **not** "SHE-flat" either.

**This is the expected outcome (plan §3 / brainstorm §2):** the onset is a
**gauge question**, fragile and frame/feature-ambiguous; the
**frame-invariant RING MAGNITUDE collapse** (0.40 → 0.079 below pH 2.3) is
the load-bearing new physics → **mechanism C is the lead (M3)**, not A.

---

## P0.3 — protocol (LOCKED, unambiguous)

Fixed roles for the entire evidence base (plan §1) — **no role may change
after this point**:

| pH | source | role |
|----|--------|------|
| 6.39 | `k2so4_ph6p39_rrde_binned.csv` raw disk+ring | **LOCK** — preserved within pre-registered tolerance throughout |
| 4 | `digitized_experimental_3panel.json` | **TRAIN + ALL model selection** |
| 6 | `digitized_experimental_3panel.json` | anchor-neighborhood **consistency check** |
| 2 | `digitized_experimental_3panel.json` | **single HELD-OUT** — opened once, after selection is frozen, **no iteration after** |
| 1.65–5.21 | `metrics.json:exp_info` scalars | low-fidelity trend / bootstrap-low-confidence only |
| Cs⁺ pH4 | `mangan_deck_p15_h2o2_current_v2.csv` | cross-condition (M5) |

**Objective everywhere:** fit **raw disk + raw ring** (model-side N=0.224);
selectivity/n_e are derived diagnostics under matched conventions only —
never fit targets. The area-mixed Exp Info `peak_sel` is never a target.

**Proton-dependence convention (locked by P0.1):** proton coupling enters in
exactly ONE place per reaction — a formal-potential (SHE-anchor) shift XOR a
kinetic c_H factor, never both. Water routes are c_H-free (Tafel); any
c_H-coupled route (A1/C1) reads the **surface** c_H trace (M2/G).

**Single Nernst constant:** S = V_T·ln10 (≈ 0.05916 V/pH) is shared by the
model E_eq shift and the data-side RHE↔SHE conversion (no spurious offset).

**Scope:** pH 2–6.4 (alkaline 10/12 excluded — no digitized data, anchor
fails the acidic-tuned recipe).

---

## Status & next

P0 complete and gated. **P0.1 PASS** unblocks the build; **P0.2** confirms
onset is a gauge question, reinforcing **C (M3)** as the lead and **A (M1)**
as a frame check. Downstream (M1a → M2/G → M1b → M3/C → M4 held-out →
M5 cross-condition) requires per-pH Firedrake solver campaigns
(~minutes–hours each) — confirm scope before launching.
