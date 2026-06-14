# Phase 7.3 — M-series running log

Executes the mechanism milestones of
`~/.claude/plans/phase7p3-pH-coupled-orr-mechanism.md` after the P0 gate
(`phase7p3_P0_protocol_lock.md`, P0.1 byte-test PASS). Roles are frozen:
pH 6.39 raw = LOCK, pH 4 = TRAIN+select, pH 6 = anchor check, pH 2 =
held-out (not yet opened). Memory: [[project-phase7p3-mechanism-shortlist]].

---

## M1a — onset selection N0 / N1a / N1b / A  ✅

`scripts/studies/phase7p3_m1a_onset_selection.py` →
`StudyResults/phase7p3_m1a_onset_selection/` (m1a_scores.json,
m1a_disk_ring_overlay.png). Frozen-θ_L water routes, L_eff 21.7, K⁺, coarse
13-pt grid; N0 reused from the pH-series (13/13), N1b + A solved fresh
(all 13/13). Scored on the digitized disk+ring at pH 2/4/6 over V_RHE
[0.20, 0.75]; Δ(pH)=S·(pH−6.39), S=V_T·ln10.

| pH | Δ (V) | metric | N0 | N1a (relabel) | N1b (OCP) | A (SHE) |
|----|------|--------|----|----|----|----|
| 2 | −0.260 | disk RMS | 1.284 | 0.718 | 1.145 | **0.704** |
|   |        | ring RMS | 0.323 | 0.024 | 0.307 | **0.007** |
|   |        | combined | 1.607 | 0.742 | 1.452 | **0.710** |
|   |        | ring peak (model/data) | 0.357 / 0.025 | 0.006 | 0.365 | 0.016 |
| 4 | −0.141 | combined | 1.237 | **0.582** | 1.099 | 0.723 |
|   |        | ring peak | 0.362 / 0.222 | 0.327 | 0.365 | 0.333 |
| 6 | −0.023 | combined | 0.434 | **0.388** | 0.424 | 0.452 |
|   |        | ring peak | 0.365 / 0.186 | 0.365 | 0.365 | 0.367 |

**Verdict — A is a gauge question, C is load-bearing:**
1. **A does NOT beat the frame nulls on shape.** A wins only at pH 2 (0.710
   vs N1a 0.742, a ~4% edge inside coarse-grid noise) and loses at pH 4
   (0.723 vs 0.582) and pH 6 (0.452 vs 0.388). The SHE-anchored kinetic
   re-solve carries no shape information beyond a rigid relabel ⇒ A is
   gauge-equivalent to N1 (#9). N1b ≈ N0 (the η-invariant OCP shift barely
   moves the c_H-free, neutral-O₂ curve), confirming numerically that an
   OCP-frame shift does not reproduce the data.
2. **No frame variant matches the ring-peak MAGNITUDE at any pH.** N0/N1b
   over-predict the ring everywhere (pH 2: 0.36 vs data 0.025 = 14×; pH 4:
   0.36 vs 0.22; pH 6: 0.37 vs 0.19 — the last two within the digitization /
   pH-6-vs-6.39 caveat, the lock is the RAW 6.39 data). N1a/A's tiny pH-2
   ring peaks (0.006/0.016) are an artifact of the relabel shifting the flat
   curve out of the scoring window — not a physical magnitude match.

The **frame-invariant ring-magnitude collapse** is unexplained by any
potential-axis operation, exactly as the plan predicted. ⇒ **mechanism C
(pH-dependent peroxide yield/loss, M3) is the load-bearing new physics; A is
recorded as "pick the right frame," not new physics.** (M1b/A1 is therefore
not pursued unless C fails to close the shape — plan §3.)

---

## M2 (G) — surface-pH kinetic coupling  ✅

`scripts/studies/phase7p3_m2_surface_ph_coupling.py` →
`StudyResults/phase7p3_m2_surface_ph/`.

**Infrastructure (verified by code read).** Every c_H-dependent rate already
reads the **electrode-facet boundary trace** of c_H: a `cathodic_conc_factor`
on the proton contributes `power·(u_exprs[2] − ln c_ref)` to the log-rate,
integrated on `ds(electrode_marker)`, and for the proton `u_exprs[2]` is the
muh-reconstructed `log c_H = μ_H − em·z_H·φ` (forms_logc_muh.py ~L567/L620).
NOT bulk, NOT OHP/Stern-plane. Water routes carry no c_H factor (byte-equiv
off-path).

**β diagnostic (from the frozen pH-series, no new solve).** Surface c_H is a
**threshold function of bulk pH, not a line:**

| bulk pH | surface pH @ V=0.30 | surface c_H (mol/m³) |
|---|---|---|
| 1.65–2.35 | 0.70–1.37 | **42–198** (acidic plateau) |
| 3.42–6.39 | 9.6–9.8 | **~2×10⁻⁷** (alkaline plateau) |

A ~9-order-of-magnitude jump across bulk pH 2.35→3.42. The H⁺ reservoir +
cathodic field enrichment hold the surface acidic below ~pH 2.3; above ~3.4
ORR H⁺-consumption outruns bulk supply and the surface alkalizes. ⇒ a
surface-c_H-reading consumption rate (C1) is **ON only at bulk pH ≲ 2.3** and
**OFF at ≳ 3.4** — the exact switch that reproduces the data's ring collapse
at pH 2 and survival at pH 4/6. Consequence: the brainstorm's bulk-limiting
proton order (β=1) is **invalid**; β≈0 on each plateau and large only across
the switch, so **C's apparent proton order is not identifiable as a simple
c_H^m power from bulk pH** (a legitimate finding, plan §6).

---

## M3 (LEAD) — C: pH-dependent peroxide yield/loss  🔨 in progress

**C1 wired** (`scripts/_bv_common.py::make_c1_reaction`, driver `--routes c1`
+ `--k0-c1-factor/--alpha-c1/--c1-h-order`): electrochemical H₂O₂ reduction
`H₂O₂ + 2H⁺ + 2e⁻ → 2H₂O`, n_e=2, cathodic_species=1, E°=1.765 V vs RHE
(=2·E_R4e − E_R2e, deck-consistent H₂O₂/H₂O couple), reads surface c_H
(`cathodic_conc_factors` on the proton, power p). Purely additive (default
routes byte-equivalent); single-convention (RHE E° + kinetic c_H^p, never an
SHE shift too — XOR-guarded). Tests `tests/test_phase7p3_c1_wiring.py` (7/7).
**Net peroxide escape** = gross 2e production − C1 consumption =
Σ_j stoich_H2O2[j]·rate_2e_units[j] (recovers gross when C1 absent).

Pending: C1 smoke (lock-preservation at pH 6.39 + ring collapse at pH 2),
then the C1/C2 fit on pH 4 with the equal-complexity competitor set
(pH-dep direct-4e / peroxide escape / ring-collection-efficiency null) and
the G2 multi-criterion sufficiency gate. C2 (homogeneous decomposition) not
yet wired.
