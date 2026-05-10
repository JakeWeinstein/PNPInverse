# Phase 6β v9 — post-Gate-4 plan (calibration + 6β.1.b + 6β.2)

**Date:** 2026-05-10
**Status basis:** Gate 4B sweep complete
(`docs/PHASE_6B_V9_GATE_4B_SWEEP_RESULTS.md`). Architecture PASS;
calibration OPEN.

## Context

Gate 4B's 9-combination sweep shows:

* **Architecture is sound.** All 9 combinations converge in ≤45 s
  λ-ramp wall. Picard outer loop hits the analytic fixed point
  exactly (`Γ·k_des = const` to 4 decimals across the k_des sweep).
  Singh Eq. (4) directionality is correct.
* **Calibration cannot proceed at V=−0.40 V.** BV is mass-transport
  saturated there (cd = −5.532 mA/cm² for *every* combination) —
  hydrolysis adds H⁺ at the OHP, but BV consumes it as fast as
  produced and the bulk-side H⁺ Levich floor pins cd. **You cannot
  calibrate `r_H_El` against deck cd at V=−0.40 V** because cd
  doesn't move.
* **Smoke kinetics are intentionally tame** (`k_hyd = 1e-3` nondim)
  to ensure Picard convergence. The plan-spec'd `k_hyd = 1e2 m/s`
  physical translates to ~5×10⁶ nondim, which breaks Picard.
  Production-tier `k_hyd` is somewhere between these and
  needs calibration.
* **r_H_El sweep produces ~8% Γ swing** across a 70 pm bracket
  at smoke kinetics — far short of the deck's implied 10⁶× Ka_M
  shift. Direction is right; magnitude is gated on `k_hyd`.

The post-Gate-4 work is therefore: **find the operating point and
kinetic regime where hydrolysis is observable, calibrate there
against deck data, then produce the full IV curve and the
cation-series holdout**.

---

## Phase A — Observability check at V=−0.20 V (1–2 days)

**Goal:** verify that *somewhere* in the cathodic V grid, cation
hydrolysis actually shifts cd by a measurable amount. This is the
load-bearing physics check that Gate 4B couldn't run at V=−0.40 V.

**Setup:**

* Re-use the cached U snapshot from Gate 4B
  (`StudyResults/phase6b_v9_gate4_smoke/u_warmstart_at_v_target.npz`).
  Walk to V=−0.20 V at λ=0 first (warm-start chain), snapshot at
  V=−0.20 V, then run the λ ramp from there.
* Smoke kinetics same as Gate 4B (`k_hyd = 1e-3, k_prot = 1e-3,
  k_des = 1.0` nondim) — keep things stable while we're checking
  observability, not magnitude.
* λ ladder (0.0, 0.5, 1.0). Record cd, pc, Γ, surface pH per rung.

**Decision tree:**

| Outcome at V=−0.20 V, λ=0 → 1 | Action |
|---|---|
| cd shifts by > 5% | Calibrate at V=−0.20 V (Phase B at V=−0.20). Architecture confirmed observable. |
| cd shifts by 1–5% | Marginal — try V=−0.15 V or V=−0.10 V. Walk back toward onset until a clean signal emerges. |
| cd plateau invariant | Either k_hyd is too tame even at non-saturated V (ramp k_hyd in Phase B), or the architecture has zero physical effect on cd at any V (would be a v9 R5#5 wording-guard "expressed plausible branch" + "no physics validation" outcome → re-queue GPT round). |

**Verification:**

```bash
python -u scripts/studies/phase6b_v9_gate4_finite_hydrolysis_smoke.py \
    --voltage -0.20 --lambda-only
```

Add `--voltage` flag to the existing smoke driver if not present.

**Files:**

* `scripts/studies/phase6b_v9_gate4_finite_hydrolysis_smoke.py` —
  add `--voltage` to override the target V from the default
  V=−0.40.
* `StudyResults/phase6b_v9_observability_v_minus_0_20/iv_curve.json`
  — output.

**Out of scope:** changing kinetics. Phase A is geometry-only:
"does ANY V in the cathodic band show observable hydrolysis at
the smoke kinetic baseline?"

---

## Phase B — k_hyd production-tier calibration (2–4 days)

**Goal:** find the k_hyd value (in nondim) that produces a Γ_MOH
saturation regime where the deck's ΔpKa ≈ −6 (10⁶× Ka_M shift)
maps to a measurable Γ swing. Constraint: Picard outer loop must
still converge (cap 8 iterations, 1e-4 relative tolerance).

**Strategy:**

The closed-form Γ_ss is:

```
Γ_ss(λ=1) = k_hyd · ⟨c_M · 10^(−ΔpKa)⟩
            / (k_des + k_prot · ⟨c_H⟩ / δ_OHP)
```

At smoke kinetics k_hyd = 1e-3 nondim and Γ ≈ 0.55, the system is
nowhere near saturation — Γ is linear in k_hyd. As k_hyd grows,
either:
1. Γ grows linearly until it saturates the Bikerman packing
   (Bikerman A_dyn at boundary) — then we hit the packing > 1
   wall.
2. The denominator `k_prot · ⟨c_H⟩ / δ_OHP` term grows non-linearly
   via c_H feedback (BV depletes H⁺ → c_H drops → reverse rate
   shrinks → Γ grows further) → bistability or runaway.
3. Picard outer loop's analytic fixed point becomes unstable
   (more than 8 iterations to converge, or oscillation between
   two branches).

The k_hyd ramp explores which of these dominates first, and where
the Picard convergence boundary is.

**Setup:**

* Operating point from Phase A (V=−0.20 V or whatever showed
  observability).
* k_hyd ladder: `(1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3)` nondim
  — 7 rungs spanning 6 orders of magnitude. Stop at the first
  rung where Picard fails or packing exceeds 0.95.
* λ=1 (production target).
* r_H_El = 200.98 (Cu prior; recalibrate in Phase D).
* Other params at smoke baselines.

**Diagnostics per rung:**

* Γ_MOH steady value
* Bikerman packing fraction `a_K·c_K(0) + a_MOH·Γ/δ_OHP`
* Picard iteration count and final residual
* Surface pH
* cd, pc

**Decision tree:**

| Outcome | Action |
|---|---|
| Picard converges, packing < 0.95, Γ scales linearly with k_hyd up to ~1e1 | Use k_hyd = 1e0 or 1e1 as production. Move to Phase C. |
| Picard converges but packing exceeds 0.95 | Production k_hyd is bounded by Bikerman saturation. Either (a) accept the bounded value and move on, or (b) increase δ_OHP (which Singh's geometry fixes; would need explicit deviation note). |
| Picard fails before packing limit | The Singh formula's `10^(−ΔpKa)` exponential blows up faster than Picard can track. Need either (a) damping factor on the Picard step (typical Anderson acceleration knob), or (b) inner Newton on the Γ residual at each rung. Both are implementable but add complexity. |

**Verification:**

```bash
python -u scripts/studies/phase6b_v9_gate4_finite_hydrolysis_smoke.py \
    --voltage <from-Phase-A> --k-hyd-ramp \
    --lambda 1.0 --r-h-el 200.98
```

Add `--k-hyd-ramp` flag if not present.

**Files:**

* `scripts/studies/phase6b_v9_gate4_finite_hydrolysis_smoke.py` —
  add `--k-hyd-ramp` flag.
* `StudyResults/phase6b_v9_k_hyd_ramp/iv_curve.json` — output.

---

## Phase C — Full V_RHE grid at λ=1 (6β.1.b proper) (3–5 days)

**Goal:** produce the deck-comparable IV curve at production
kinetics. This is the deliverable that closes 6β.1.b.

**Strategy: ramp-at-anchor pattern.** Different from Gate 4B's
ramp-at-V=−0.40 — at this stage the architecture is validated
and we want the full IV curve.

**Setup:**

1. Anchor at V=+0.55 V at λ=0 (Gate 2 SUCCESS recipe; ~157 s).
2. **Ramp λ to 1 at the anchor** where σ_S ≈ 0 → ΔpKa ≈ 0 →
   essentially free continuation rung. Picard still runs (Γ
   updates per rung) but it's a no-op physics-wise.
3. Warm-walk down to V=−0.40 V at λ=1 with calibrated kinetics
   from Phase B. 11 V_RHE points.

If λ=1 walk fails partway down, fall back to the Gate 4B pattern
(walk at λ=0 first, then re-ramp λ at each V).

**Setup:**

* Singh kinetics from Phase B (k_hyd at calibrated value).
* Singh formula `r_H_El = 200.98 pm` (Cu prior; calibration in
  Phase D may revise this, after which Phase C re-runs).
* Mesh Ny=80 production. One-off Ny=200 spot-check at V=−0.40 V
  for mesh-refinement direction (per the v9 plan §"Risk callouts").

**Diagnostics:**

* Per-V cd, pc, surface pH, Γ_MOH, Bikerman packing
* Plot vs Linsey 2025 deck slide reference (if extractable from
  the pptx; else just stand-alone)
* Compare to Gate 2's λ=0 baseline to quantify the hydrolysis
  effect across V

**Verification:**

```bash
python -u scripts/studies/phase6b_v9_6beta1b_full_iv.py
```

(New driver script; mirror the Gate 4B smoke structure but use
ramp-at-anchor pattern.)

**Files:**

* `scripts/studies/phase6b_v9_6beta1b_full_iv.py` — new driver.
* `StudyResults/phase6b_v9_6beta1b/iv_curve.json` — output.
* `StudyResults/phase6b_v9_6beta1b/iv_plot_vs_deck.png` — overlay.

**Out of scope:** other cations (Phase E). Just K⁺ here.

---

## Phase D — r_H_El calibration for ORR-on-carbon (1–2 days)

**Goal:** given Phase A's observability point + Phase B's
calibrated k_hyd, find the `r_H_El_pm_carbon` that reproduces
the Linsey deck slide 27 ΔpKa(K) ≈ −6 unit drop at the cathode
within 30%.

**Strategy:**

The deck slide 27 K⁺ near-cathode pKa = 8.49 (vs bulk 14.5,
implies ΔpKa = −6.01). The model's surface pH at λ=1 should drop
toward the K⁺ pKa once `r_H_El_pm_carbon` is correctly calibrated.
Iterate r_H_El at fixed (V, k_hyd) until the model surface pH at
V_target matches the deck.

**Setup:**

* Operating point: Phase A's observability V (V=−0.20 V or wherever).
* Kinetics: Phase B's calibrated k_hyd.
* r_H_El sweep: `{180, 195, 200.98, 215, 250}` pm, plus refinement
  rounds (binary search) once we bracket the deck target.
* λ=1 (no λ ramp needed; warm-start from Phase B).

**Deck target:**

* Linsey 2025 ACS-CATL deck slide 27 transcribed in
  `docs/CONJECTURE_AUDIT_2026-05-09.md`: pKa near Cu cathode,
  K⁺ = 8.49, Cs⁺ = 4.32, etc. **These are Cu values; the
  v9 plan §"Risk callouts" flags this as the load-bearing
  Cu→ORR-on-carbon transferability check.**

**Decision tree:**

| Outcome | Action |
|---|---|
| Some r_H_El in the sweep produces ΔpKa within 30% of deck | Lock that value as `r_H_El_pm_carbon`. Re-run Phase C with the calibrated value. Pass to Phase E. |
| Sweep brackets but doesn't hit the deck target | Refine via binary search inside the bracket. ~3 more rungs. |
| No r_H_El in the swept range works | The Cu→carbon transferability has failed for K⁺. Fall back to Co-Zhang 2019 derivation (already in `data/.../Articles/`). Re-queue GPT round with this finding. v9 R5#5 wording guard: Gate 4 PASSed architecturally but physics-validation FAILS at calibration. |

**Files:**

* `scripts/studies/phase6b_v9_r_h_el_calibration.py` — new driver.
* `StudyResults/phase6b_v9_r_h_el_calibration/calibration_curve.json`
  — output.

**Output:** updated `_bv_common.py::SINGH_2016_CATION_PARAMS` row
with `r_H_El_pm_ORR_carbon` for K⁺.

---

## Phase E — Cation-series holdout 6β.2 (5–10 days)

**Goal:** apply the calibrated `r_H_El_pm_ORR_carbon` (from K⁺
in Phase D) to Cs⁺ / Na⁺ / Li⁺ and check whether their predicted
local pH matches the deck slide 27 cation series. **No re-fitting
per cation** — this is a *predictive* holdout, not a calibration.

**Setup:**

* Three new species lists in `_bv_common.py`:
  * `FOUR_SPECIES_LOGC_DYNAMIC_CS2SO4`
  * `FOUR_SPECIES_LOGC_DYNAMIC_NA2SO4`
  * `FOUR_SPECIES_LOGC_DYNAMIC_LI2SO4`
* Each uses Singh Table S1 row for that cation's z_eff, r_M, n_hyd,
  pKa_bulk.
* `r_H_El_pm` from Phase D, applied uniformly across all 4 cations
  (the Cu → carbon shift should be cation-independent if the
  geometry is set by the adsorbed-CO sandwich; if cation-specific,
  this is a finding).
* Operating point: V=−0.40 V production target (or Phase C's full
  V grid if we want the full per-cation IV curve).

**Deck targets (Linsey slide 27, Cu cathode → ORR/carbon):**

| Cation | pKa near Cu | Expected at ORR/carbon |
|---|---|---|
| Li⁺ | 13.16 | small ΔpKa, weak hydrolysis |
| Na⁺ | 11.44 | moderate |
| K⁺ | 8.49 | calibration anchor |
| Cs⁺ | 4.32 | strong hydrolysis (deepest cathodic shift) |

The model should predict the same monotone ordering (Li⁺ < Na⁺ <
K⁺ < Cs⁺) in surface pH drop magnitude at fixed V.

**Verdict criteria:**

* All 4 cations converge at λ=1.
* Predicted surface pH order matches deck order.
* Magnitude within 30% (loose, because we're not re-fitting).
* If holdout fails: ledger L6 not closed; re-queue GPT round.

**Files:**

* `_bv_common.py` — three new species list constants.
* `scripts/studies/phase6b_v9_6beta2_cation_series.py` — new driver.
* `StudyResults/phase6b_v9_6beta2/{Cs,Na,Li}_iv_curve.json` —
  per-cation outputs.
* `StudyResults/phase6b_v9_6beta2/cation_series_overlay.png` —
  comparison plot.

---

## Phase F — Independent: K⁺ Tafel slope extraction (1 day, parallel-safe)

**Goal:** unblock M1 (`docs/missing_data.md` ledger) for K⁺ without
waiting for the missing xlsx delivery. Extract Tafel slopes from
`Brianna/0,1M K2SO4 data 8-15-19.xlsx` directly.

**Strategy:**

The xlsx has full RRDE LSV at 6 pH values (1.65–6.39) for K⁺ in
0.1 M K₂SO₄. Each LSV scan provides:
* `E_disk (V vs RHE)` — voltage axis
* `j_disk (mA/cm²)` — current axis

Tafel slope = `dV / d(log|j|)` in the linear region (typically
±0.1 V around onset, where the Butler-Volmer kinetic regime
dominates).

**Setup:**

* Read xlsx, parse 6 sheets (one per pH), extract E vs j curves.
* Clip to cathodic regime (j < 0) and exclude the mass-transport
  plateau and the ohmic-drop region.
* Fit log|j| vs E linearly in the kinetic region. Report slope
  in mV/decade per pH.
* Expected: slope around 60–120 mV/decade (single-electron-transfer
  rate-limiting step) at low pH; may shift at higher pH.

**Output:**

* `data/derived/k_plus_tafel_slopes_from_brianna_2019.xlsx` — six
  rows, one per pH, with slope, R², linear-region V range. Schema
  matches what Yash's `plotting.ipynb` would have read from the
  missing xlsx, so his pipeline can run unchanged.

**Why now:**

* Gate 4 is closed; the K⁺ kinetic priors (k_hyd, k_prot) currently
  are phenomenological. Once we have a measured Tafel slope, we can
  inform `K0_R4e / α_R4e` — though this is technically a 6β.2
  calibration step, the K⁺ data is in hand and extracting it is
  cheap.
* Useful for Phase D's calibration sanity check: deck-derived
  Tafel slope should be reproducible by the model at the calibrated
  `r_H_El`.

**Files:**

* `scripts/derive/extract_k_plus_tafel_slopes.py` — new analysis
  script. Uses `pandas.read_excel` + linear regression.
* `data/derived/k_plus_tafel_slopes_from_brianna_2019.xlsx` —
  output.
* Update `docs/missing_data.md` M1 with "K⁺-only workaround
  delivered as `data/derived/...`" status note.

**Parallel-safe** — runs independent of A/B/C/D/E. Fits in 1 day
of work.

---

## Order of operations

```
Phase A (V=−0.20 observability)            ──── 1–2 days
 │
 ├─ if observable: Phase B (k_hyd ramp)    ──── 2–4 days
 │   │
 │   ├─ Phase C (full V grid at λ=1)       ──── 3–5 days  (parallel
 │   │                                              with D)
 │   │
 │   └─ Phase D (r_H_El calibration)       ──── 1–2 days
 │       │
 │       └─ Phase E (cation-series 6β.2)   ──── 5–10 days
 │
 └─ if not observable: re-queue GPT round (architecture is
    "expressed plausible branch" but no physics signal — would
    need revisit of v9 R5#5 wording-guard interpretation)

Phase F (K⁺ Tafel extraction) ──────────── 1 day, parallel-safe with all of A–E
```

Total wall: 2–3 weeks for the full 6β.1.b + 6β.2 closure if every
phase converges first try; longer if any phase needs a re-design
loop.

---

## Risk callouts

* **Phase A may show "no observability anywhere"** — an outcome the
  Gate 4 sweep didn't anticipate. The smoke at V=−0.40 saturated
  cd at Gate 2's H⁺ Levich floor; if that floor is actually the
  rate-limiting step at every cathodic V, then no amount of
  hydrolysis at the OHP will move cd. This would be a v9 R5#5
  wording-guard finding: "Gate 4 architecture works, expresses a
  plausible branch, but the branch has no measurable cd impact" —
  meaning the *deck-magnitude calibration is unattainable* with
  v9's transport-saturated regime. Mitigation: try thicker
  L_eff (smaller boundary layer → less Levich limitation), or
  alternative L_eff from the deck if specified differently.

* **Phase B may not find a stable k_hyd window** — between
  `k_hyd ≤ 1e-3` (no signal) and `k_hyd ≥ 1e2` (Picard breaks),
  there may be no value where (a) signal is observable AND (b)
  Picard converges AND (c) packing < 1 simultaneously. Mitigation:
  Anderson acceleration on the Picard step (off-the-shelf
  numerical recipe), or migrate Γ to inner Newton (revisit the
  R-space-in-mixed Firedrake limitation that drove the Phase 6β
  v9 architectural deviation in the first place — see
  `docs/PHASE_6B_V9_GATES_3_4_SUMMARY.md` §"Architectural deviation").

* **Phase D Cu→carbon transferability failure is the most likely
  physics-side failure mode.** The Singh formula is fit to Cu/CO₂RR;
  ORR/carbon has no adsorbed CO and a different surface electronic
  structure. If no `r_H_El_pm` in [180, 250] pm reproduces the deck
  ΔpKa, we have to fall back to Co-Zhang 2019. Co-Zhang is already
  in `data/.../Articles/` so the fallback exists, but it's another
  formula-extraction round.

* **Phase E cation-series may not transfer cleanly even with
  calibrated K⁺** — Cs⁺/Na⁺/Li⁺ have different hydration
  geometries, and Singh's r_H_El for Cu is a different value per
  cation (not constant). Our v9 simplification uses one
  `r_H_El_pm_carbon` for all 4 cations, which may not match deck
  cation-specific values. Mitigation: cation-specific `r_H_El`
  fitting at 6β.2 (each cation's deck pKa as a separate
  calibration anchor), at the cost of becoming "fitted everywhere"
  rather than "predictive holdout".

---

## Critical files (modified or created)

* `scripts/studies/phase6b_v9_gate4_finite_hydrolysis_smoke.py` —
  add `--voltage`, `--k-hyd-ramp` flags.
* `scripts/studies/phase6b_v9_6beta1b_full_iv.py` — **NEW**, full V
  grid at λ=1 (Phase C).
* `scripts/studies/phase6b_v9_r_h_el_calibration.py` — **NEW**,
  r_H_El calibration driver (Phase D).
* `scripts/studies/phase6b_v9_6beta2_cation_series.py` — **NEW**,
  cation-series holdout driver (Phase E).
* `scripts/derive/extract_k_plus_tafel_slopes.py` — **NEW**, Tafel
  extraction (Phase F).
* `_bv_common.py` — three new K2SO4 → CsSO4/NaSO4/LiSO4 species
  list constants (Phase E); update SINGH_2016_CATION_PARAMS with
  calibrated `r_H_El_pm_ORR_carbon` (Phase D).
* `data/derived/k_plus_tafel_slopes_from_brianna_2019.xlsx` —
  **NEW**, K⁺ Tafel slopes (Phase F).
* `docs/missing_data.md` — update M1 with Phase F workaround
  delivery.
* `docs/phase6b_next_steps_plan.md` — update §8 ledger with Phase D
  outcome (closes L6 cation-series transferability if Phase E
  passes).

---

## End-to-end verification

```bash
# After Phase A: observability at V=−0.20
ls StudyResults/phase6b_v9_observability_v_minus_0_20/iv_curve.json

# After Phase B: k_hyd calibrated
ls StudyResults/phase6b_v9_k_hyd_ramp/iv_curve.json

# After Phase C: full IV curve at λ=1
ls StudyResults/phase6b_v9_6beta1b/iv_curve.json
ls StudyResults/phase6b_v9_6beta1b/iv_plot_vs_deck.png

# After Phase D: r_H_El calibrated
grep -A2 "r_H_El_pm_ORR_carbon" scripts/_bv_common.py

# After Phase E: cation series
ls StudyResults/phase6b_v9_6beta2/{Cs,Na,Li}_iv_curve.json
ls StudyResults/phase6b_v9_6beta2/cation_series_overlay.png

# After Phase F: Tafel slopes (parallel-safe)
ls data/derived/k_plus_tafel_slopes_from_brianna_2019.xlsx

# Full test suite still green throughout
pytest tests/ -m "not slow"
pytest tests/ -m "slow"
```

**6β.1.b PASS:** Phase A confirms observability, Phase B finds
production k_hyd, Phase C produces deck-comparable IV at K⁺,
Phase D calibrates r_H_El.

**6β.2 PASS:** Phase E shows the Cs⁺/Na⁺/Li⁺ holdout reproduces
the deck cation series within 30% with a single calibrated
r_H_El value.

**6β.2 FAIL** (any phase A–E hits a "stop and re-plan" branch):
queue another GPT round with the failure data as new evidence;
likely fall back to Co-Zhang 2019 derivation or revisit the
Cu→carbon transferability assumption.
