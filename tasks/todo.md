# Phase 7.3 — pH-coupled ORR mechanism (executing `~/.claude/plans/phase7p3-pH-coupled-orr-mechanism.md`)

Gated plan. **P0.1 is the STOP-or-go gate**: prove the SHE-anchored frame
math is byte-exact at the pH-6.39 lock before building anything else.

## P0.1 — single-convention frame byte-test (THE central falsification guard)

- [x] Map the frame plumbing: `_build_reactions` shifts `E_eq_v` by −V_OCP
      and the V-grid by −V_OCP ⇒ OCP cancels in η ⇒ model is RHE-referenced.
      `E_eq_model = E_eq_v / potential_scale` (affine; shift composes).
- [x] Add an opt-in SHE-anchored formal-potential frame to `_build_reactions`
      (the single shift point, reused by driver + fit harness + pH-series):
      `E_eq_RHE,j(pH) = E_eq_locked,j + S·(pH − pH_anchor)`, `S = V_T·ln10`.
      Anchored on bulk c_H so the shift is EXACTLY 0.0 at the anchor.
- [x] Enforce the XOR: when frame=she, no enabled reaction may carry a
      kinetic c_H factor (`cathodic_conc_factors`) — never both.
- [x] Backward-compat: `getattr(opts,"proton_frame","rhe")` ⇒ fit harness /
      pH-series byte-equivalent (they never set it).
- [x] Reaction-level byte test (instant, no Firedrake): rhe vs she at the
      anchor ⇒ byte-identical reaction dicts; she at pH 4 ⇒ E_eq shifted by
      exactly S·(4−6.39) (machinery live, not a no-op).
- [x] pytest regression locking the byte-test.
- [x] Confirmatory solver byte test: `dp._run` at pH 6.39 (θ_L, v_ocp 1.019,
      cation k, water, L_eff 21.7) in both frames ⇒ cd/pc per V byte-identical.
      RESULT: max|Δcd|=max|Δpc|=0.0 over 13 pts, 13/13 conv both, grids match.
      `StudyResults/phase7p3_p0_1_frame_byte_test/verdict.json`.
- [x] GATE DECISION: **byte-exact PASS** ⇒ frame math sound ⇒ proceed to P0.2.

## Lock config (reference for the byte-test)
θ_L = (lg f_2w −1.008700, lg f_4w −12.308927, α_2w 0.577014, α_4w 0.304854);
V_OCP 1.019 V; bulk_H 4.07e-4 mol/m³ (pH 6.39); cation k; routes water;
L_eff 21.7 µm; coarse grid; enable_water_ionization=True.

## P0.2 — disk-onset extractor ✅ (milestone 2)
- [x] `scripts/studies/phase7p3_p0_2_onset_extractor.py` — disk onset at
      {0.05,0.10,0.20} mA/cm² post-background, plateau-connected onset,
      digitization bootstrap (σ_V 5mV, σ_y 15µA/cm²), RHE+SHE, ±CI.
- [x] Unit-tested `tests/test_phase7p3_onset_extractor.py` (5/5).
- [x] FINDING: onset is NOT a robust discriminator — pH4≈pH6 (+9..+11
      mV/pH), sign flips to −35..−51 only via the noisy pH-2 shoulder;
      disagrees with ring-onset (+41). Reinforces C (M3) as lead, A as gauge.

## P0.3 — protocol lock ✅ (milestone 2)
- [x] `docs/phase7/phase7p3_P0_protocol_lock.md` — roles frozen (6.39 LOCK,
      4 TRAIN+select, 6 anchor-check, 2 HELD-OUT once), raw disk+ring
      objective, one-place proton convention, single Nernst constant.

## M-series (full plan order; user-approved, commit-to-main)

### M1a — onset selection N0/N1a/N1b/A  ⏳ RUNNING
- `scripts/studies/phase7p3_m1a_onset_selection.py` — reuses cached N0
  (pH-series), runs N1b (solver OCP shift) + A (SHE frame) at pH 2/4/6,
  scores disk+ring RMS + ring-peak vs digitized. Preview: N0 ring peak flat
  ~0.36 while data collapses to 0.025 at pH 2 → no frame variant fixes the
  magnitude (expected: A = gauge question, C load-bearing).

### M2 (G) — surface-pH kinetic coupling  ✅
- `scripts/studies/phase7p3_m2_surface_ph_coupling.py` (+ json/png). c_H
  factors read the electrode-facet boundary trace (verified in
  forms_logc_muh.py). β diagnostic: surface c_H is a THRESHOLD (acidic
  ≤pH2.35 surface ~10² mol/m³; alkaline ≥pH3.42 surface ~10⁻⁷) — the switch
  that gates C1. β is NOT a single slope ⇒ C's proton order non-identifiable
  from bulk pH (a finding).

### M3 (LEAD) — C: pH-dependent peroxide yield/loss  🔨 wiring done
- C1 wired: `make_c1_reaction` in `_bv_common.py` (H₂O₂+2H⁺+2e⁻→2H₂O,
  E°=1.765 V deck-consistent, reads surface c_H); driver `--routes c1`
  + `--k0-c1-factor/--alpha-c1/--c1-h-order`; purely additive (default
  byte-equiv). Tests `tests/test_phase7p3_c1_wiring.py` (7/7); net-peroxide
  observable (gross − C1) validated.
- [x] C1 smoke: C1 VIABLE. k0_factor~1 saturates (E°=1.765→exp(53)); the
  c_H gate spans 8.5 decades → at k0_factor=1e-18 C1 is OFF at pH6.39 (lock
  ring 0.361≈0.367) and ON at pH2 (ring→0). Lock-preserving bracket
  ~[1e-20,1e-15]. SUFFICIENCY shown.
- [x] M3/M4 verdict: C is load-bearing; C1 sufficient (reproduces pH-2
  collapse + preserves lock); **disk-side attribution CONDITIONAL** — pH-2
  deep disk (−3.89) is between ring-only null (N0 −3.57) and full-C1
  (−4.06), within absolute-current uncertainty → can't exclude non-faradaic
  loss without N₂ scans/rpm. C1 NOT fit on pH2 (held-out; off at pH4 so
  bracket from lock-preservation only). p not pinned (finding).
- [ ] C2 (homogeneous decomposition) — needs forms-level dx sink/source; NOT
  wired (the non-faradaic alternative; deferred).

### Pending: M5 slide-15 Cs⁺ cross-condition + honest write-up.
(M1b A1 NOT pursued — M1a showed onset is a frame question.)
