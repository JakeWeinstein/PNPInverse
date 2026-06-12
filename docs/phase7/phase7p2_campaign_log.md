# Phase 7.2 campaign log — K₂SO₄ pH 6.39 dual-series fit

**Live document — updated as results land.** Plan:
`~/.claude/plans/phase7p2-k2so4-dual-series-fit.md` (session 43,
APPROVED, 40 issues). Data provenance:
`docs/phase7/k2so4_ph6p39_provenance.md`.

Last updated: 2026-06-12 (profiles in flight).

---

## TL;DR (current state)

A 4-parameter water-route dual-pathway model (no acid routes, no
bisulfate, no surface corrections) fits BOTH series of a real
numeric RRDE dataset — total disk current AND raw ring current —
across the full ORR window, passing all five pre-registered feature
gates, several by an order of magnitude. The χ² ladder:

| Configuration | raw χ² (60 residuals) | J/pt |
|---|---|---|
| θ* (slide-15 Cs⁺/pH4 params, transferred) | 44,515 | 1483.8 |
| dual fit at L_eff = 15.4 µm | 8,801 | 293.4 |
| **dual fit at L_eff = 21.7 µm (accepted)** | **366** | **12.20** |

**θ_L = (log f_2w = −1.009, log f_4w = −12.309, α_2w = 0.577,
α_4w = 0.305)**, L_eff-conditional and OCP-conditional per the lock
tiers (OCP empirically near-immaterial, see § Sensitivities).

Two findings with paper weight:

1. **Tafel structure transfers; prefactors don't.** The fitted α's
   (0.577, 0.305) sit within 0.03 of the slide-15 Cs⁺/pH-4 values
   (0.550, 0.285) across a 2.4-pH-unit, cation-swapped, different-
   electrode condition; Δ(αn) = +0.066 vs +0.042 (the volcano
   slope-competition mechanism survives). The k0's shift by 2.7 /
   1.2 decades — pre-registered outcome (ii): site-density /
   catalyst-history difference (catalyst ask gates interpretation).
2. **The disk series identifies the partition.** First profile
   points put the Δχ²=4 half-width of log f_4w at ≈ 0.11 decades
   (±25% on k0_4e) under the dual objective — a parameter that was
   structurally non-identified on peroxide-only data. (Ring-only
   comparison profiles pending to complete the claim.)

---

## Chronology + analysis

### Session 43 critique loop (5 rounds, APPROVED)
`docs/handoffs/CHATGPT_HANDOFF_43_k2so4-dual-series-fit/`. 40 issues
+ 3 non-blocking, all accepted. Decisive design changes from the
loop: OCP composed with the file's MEASURED Ag/AgCl→RHE cal
(0.47 + 0.549 = 1.019 V, not 1.044); raw-ring objective with the
collection model on the MODEL side (N variants = pure refits);
raise-and-restart eval policy (no penalty values into L-BFGS-B);
true re-optimized profiles incl. the derived 4e fraction; refit-vs-
rescore table; lock tiers with a claim→blocking-ask map.

### Stage 0 — extraction (gate PASS, commit 8f898c7)
Re-derived every processed column of Brianna's workbook exactly
(0 failures). Two substantive data findings:

- **iR sign audit:** the sheet's "iR New" column uses V + I·Rs;
  with the file's own anodic-positive convention the physical
  correction is V − I·Rs — 0.19 V apart at the plateau (Rs=125 Ω).
  Canonical axis = physical; sheet axis kept as refit variant.
  ASK: which sign the paper pipeline used.
- **Capacitive LSV shelf:** ring-silent −0.21..−0.33 mA/cm² over
  0.6–1.0 V (60× the recorded cap band) — porous-carbon LSV
  background a steady-state model must not fit. Linear background
  (anchored 0.65–1.0 V) subtracted; validated by the down/up
  sweep-average cross-check (median residual 0.03 mA/cm²).

Also: canonical Sel%/n_e recomputed from currents (sheet formulas
mix areas — sheet peak 73.2% → canonical 63.4%; sheet n_e 2.8 →
band 2.85–3.52); air saturation EXCLUDED by the data at every
L_eff bracket end; fit window [0.132, 0.559] V_phys, 30 bins,
replicate-based σ (conservative single-observation scale).

### Stage 1 — K⁺/pH 6.39 config (gate PASS, commit 64ae476)
Driver gained --cation/--v-ocp-rhe/--v-grid. The pH-4 anchor recipe
transferred UNMODIFIED (38 s, 5 rungs, no escalation): 13/13 smoke
grid, electron residual 5e-15, anodic share 0, acid sink 0.
Analysis: at bulk pH 6.39 the acid routes are dead by physics (H⁺
cap ≈ 2.4e-3 mA/cm², 3 OOM below data) — the freeze-out is not a
modeling choice here.

### Stage 2 — pre-registered θ* transferability diagnostic (b2cf542)
θ* (slide-15 fit) evaluated unchanged at the new condition BEFORE
any tuning: raw χ² 44,515; every feature ~0.15–0.18 V too cathodic,
plateau pushed toward the 4e transport ceiling, ring 2.4× small.
Coherent (axis-shift-like) misfit, not noise — and the 0.15 V onset
offset ÷ 54 mV/dec ≈ 2.8 decades predicted the k0_2e move the fit
later made (+3.4 dec). Locked as the Δχ² baseline.

### Stage 3 — objective + adjoint + gates (29bd4ac, 8b519b2)
- score_dual_series (Firedrake-free) + 10 fast tests; harness with
  two taped observables (cd = mode "current_density", pc =
  role-resolved production), collection model on tape, EvalFailure
  raise-and-restart runner, adaptive 17-pt grid + 30-bin polish.
- FD gate: 3/4 components pass at default h; α_4w fails at h=0.005
  with the KNOWN truncation signature, then h-converges 14.9% →
  0.52% → 0.13% — adjoint correct.
- Observable contracts 4/4: cd ≡ manual electron-weighted sum;
  isolated 2e/4e; **escape ≡ production to <1e-4** via the
  discrete-consistent flux (residual paired with ψ = y/H on the
  H₂O₂ slot). Two methodological findings: the naive −D∂c/∂n
  estimator misses the Bikerman steric-activity flux (~4.5% at the
  test state; raw P1 boundary gradients are junk), and hand-built
  test configs drift from production (clip/clamp/SNES/α) in ways
  that can silently kill a reaction branch — tests now build
  through the driver's own config path.

### Stage 4 — fit, robustness, structure (7877e4b → f3a9b02)
- **Main fit (15.4 µm):** 83 evals, 0 solver restarts, converged
  flat. J 1484 → 293. Onsets matched to ≤10 mV; plateau overshot
  +38% sitting exactly AT the model's mixed-n_e transport ceiling →
  read as L_eff too thin; predicted L ≈ 21.3 µm.
- **L_eff bracket (REFITS, not rescores):** {12: J = 813 (k0_4e
  boundary-pinned), 15.4: 293, 21.7: 12.195}. Decisive thick-film
  preference. θ moves substantially with L ⇒ parameters are
  **L_eff-conditional**; rpm ask is the single most load-bearing
  data ask for parameter VALUES (not for fit-quality/mechanism
  statements). Physical readings of 21.7 µm: rotation < 1600 rpm
  (~850 rpm), or the δ_OH-film bias (21.7 µm IS the OH⁻-equivalent
  film at 1600 rpm). Distinguishable once rpm is confirmed.
- **Feature gates at θ_L: ALL 5 PASS** — disk onset Δ21 mV, ring
  onset Δ0.5 mV, ring peak position Δ29 mV, ring peak height
  −3.3%, plateau −5.1%.
- **Robustness:** at 15.4 µm the landscape is multi-modal (restart
  A found a boundary-pinned high-k0/low-α basin at J=1474,
  rejected 5×; B returned to base). At the ACCEPTED 21.7 µm
  optimum: A returns EXACTLY (4 decimals), B same basin along the
  flat k0-ratio valley — unimodal in the perturbation box.

### Sensitivities (refit table; in progress)
| Variant | J | θ shift | verdict |
|---|---|---|---|
| OCP 1.044 V (+25 mV, theoretical cal) | 12.184 | ≤0.09 in log k0, <0.001 in α | NOT load-bearing |
| OCP 0.994 V (−25 mV) | running | — | — |
| N = 0.20 | ≈9.86 (converging) | mild | data slightly prefers lower N; partition carries the band |
| N = 0.25 | queued | — | — |
| bulk c_H ×{0.7, 1.4}, weights, window edge, bg-scale, sheet-axis | queued | — | — |

### Identifiability (in progress)
log f_4w profile (fix + reoptimize 3): ±0.1 dec → Δχ²_raw ≈ +3.1
both sides ⇒ Δχ²=4 half-width ≈ 0.11 decades (PRAGMATIC width, not
a CI — σ is a conservative predictive scale). α_4w ladder + the
ring-only-objective mirror (the actual partition-identified
comparison) + derived 4e-fraction constrained profiles: queued.

---

## Conditional labels currently attached to θ_L

- **L_eff-conditional** (bracket spread >> profile widths) — until
  rpm confirmed.
- **OCP-conditional** formally; empirically insensitive at ±25 mV.
- **transport-conditional** (c_O₂ protocol unconfirmed; air ruled
  out by the data).
- Partition scale carries the **N band** until ring calibration
  confirmed.
- Cross-condition transferability claim is
  **digitized-slide-conditional** (raw Tafel xlsx ask) and
  **catalyst-conditional**.

## Data asks (status)

rpm (NOW CRITICAL — sets L_eff and parameter values); catalyst
identity; gas/T/O₂ protocol (air excluded by data already);
ring hold + calibration (N band); sibling pH xlsx files; iR sign
used in the paper pipeline (new, from the Stage-0 audit); raw Tafel
xlsx; 0.47 V OCP component (empirically weak lever here).

## Run ledger

| StudyResults dir | What | Key number |
|---|---|---|
| phase7p2_stage0_extraction | extraction QA + report | gate PASS |
| phase7_dual_pathway/phase7p2_stage1_smoke_k_ph6p39 | Stage-1 smoke | 13/13 |
| phase7p2_theta_star_prediction | Stage-2 baseline | χ² 44,515 |
| phase7p2_fd_check{,_comp3_hconv} | FD gate | 4/4 (h-conv) |
| phase7p2_fit_dual | main fit @15.4 | J 293.365 |
| phase7p2_fit_restart_{plus,minus} | robustness @15.4 | multi-modal flagged |
| phase7p2_refit_leff21p7{,_cont} | accepted fit @21.7 | J 12.195, 5/5 gates |
| phase7p2_refit_leff12 | bracket | J 813 |
| phase7p2_l21_restart_{plus,minus} | robustness @21.7 | unimodal |
| phase7p2_refit_ocp1p044 | OCP +25 mV | J 12.184 |
| phase7p2_refit_n0p20 | N 0.20 | ≈9.86 |
| phase7p2_prof_k4w_{p,m}0p1 | log f_4w profile ±0.1 | Δχ² ≈ 3.1 |
| phase7p2_leff21p7_final_curves | θ_L curves | best_fit_vs_data.png |

## Remaining queue

1. Finish sensitivity refits (OCP 0.994 + ±50 mV pair, N 0.25,
   c_H, weights, window edge, bg-scale, sheet-axis variant — the
   last three need extractor parametrization).
2. α_4w profile ladder; ring-only (w_disk_scale=0) profile mirror;
   derived 4e-fraction constrained profiles; BFGS Hessian + FD
   diagonal.
3. Stage 5: θ_L → slide-15 Cs⁺/pH4 re-prediction (two L variants);
   mini-ablation cell (water-2e off, water-4e off, acid-on probe,
   Kw 0.5×); lock docs + memory; computational lock declared with
   the conditional-label table (paper-grade claims await asks).
