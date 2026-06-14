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

## Downstream (NOT started — M-series = per-pH Firedrake campaigns, mins–hrs)
**HOLD for scope confirmation.** M1a N0/N1a/N1b/A (onset selection on full
disk+ring shape) · M2 G surface-pH coupling · M1b A1 · **M3 C1/C2 (LEAD)** ·
M4 pH-2 held-out · M5 slide-15 + write-up.
