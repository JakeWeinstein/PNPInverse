# Phase 7.2 — dual-series K₂SO₄ pH 6.39 fit: summary & lock

**Computational lock.** Plan
`~/.claude/plans/phase7p2-k2so4-dual-series-fit.md` (session 43,
APPROVED). Full chronology + per-run analysis:
`docs/phase7/phase7p2_campaign_log.md`. Data provenance:
`docs/phase7/k2so4_ph6p39_provenance.md`.

## Result

A 4-parameter water-route dual-pathway ORR model (water-as-proton-
donor 2e + 4e Butler–Volmer; NO acid routes, NO bisulfate, NO surface
corrections) simultaneously fits the **total disk current** and the
**raw ring (peroxide) current** of a real numeric RRDE dataset
(Brianna Ruggiero 2019, 0.1 M K₂SO₄, bulk pH 6.39) across the full
ORR window.

**Accepted θ_L** (L_eff = 21.7 µm): log f_2w = −1.009, log f_4w =
−12.309, α_2w = 0.577, α_4w = 0.305. raw χ² = 366 over 60 residuals
(J = 12.20). **All five pre-registered feature gates pass:** disk
onset Δ21 mV, ring onset Δ0.5 mV, ring-peak position Δ29 mV, ring-peak
height −3.3%, plateau −5.1% (gates: 50 mV / 30%).

## The three scientific findings

1. **Mechanism (ablation-proven).** Both water routes are necessary
   and play distinct roles:
   - water-2e off → peroxide → 0 (the 2e route IS the ring signal);
     disk rises to −4.05 (4e captures the freed O₂).
   - water-4e off → current becomes pure 2e (cd ≡ pc), plateau drops
     to −2.03, and peroxide goes MONOTONIC — no volcano. The 4e route
     creates the descending flank by out-competing 2e for O₂ at
     cathodic potentials. **The volcano is a 2e/4e O₂-competition.**
   - acid-on probe → adds 0.001 mA/cm² (e-sink share 0.0007%) at
     pH 6.39, vs −3.6 mA/cm² of Kw-laundering artifact at pH 4. The
     near-neutral condition is free of the acid-route pathology that
     made pH 4 intractable; the acid freeze-out is physically exact
     here.

2. **Cross-condition transferability — clean outcome (ii).** Same
   water-route kinetics explain slide-15 (Cs⁺ pH 4) and this dataset
   (K⁺ pH 6.39):
   - frozen-θ_L → slide-15 (no refit): χ²/pt 5464 → outcome (i)
     rejected (prefactors condition-specific);
   - **α frozen at θ_L, k0 refit at slide-15's L: χ²/pt 35.6 vs the
     native free-α 29.8** → the Tafel coefficients TRANSFER;
   - at matched L_eff the genuine k0 shift is only ~0.26 dec on the
     dominant 2e route — the large raw 2.67-dec gap was MOSTLY the
     transport-film difference (21.7 vs 15.4 µm).
   Δ(αn) = +0.066 (here) vs +0.042 (slide-15): the slope-competition
   that makes the volcano is shared.
   Figure: `StudyResults/phase7p2_xcond_slide15_alphafrozen/slide15_generalization.png`
   (slide-15 data vs θ_L cold-fail vs α-frozen k0-refit recovery).

3. **What two series buy vs one (identifiability).** True
   re-optimized profiles (Δχ² = 4, pragmatic widths):

   | quantity | dual width | ring-only width | disk gain |
   |---|---|---|---|
   | log f_4w | ±0.23 dec | ±0.28 dec | 1.2× |
   | α_4w | ±0.0046 | (Tafel tilt, ring-soft) | — |
   | **4e current fraction** | 0.719 | 0.729 (ring-only optimum) | ~none |

   **The partition FRACTION (≈0.72 4e) is ring-determined** — the
   volcano peak shape already encodes the 2e/4e competition. The disk
   adds the ABSOLUTE total-current scale and shifts the prefactor
   optimum 0.68 dec to fix it (at fixed fraction), with only a 1.2×
   prefactor tightening. The original "disk breaks the partition
   non-identifiability" motivation held for the DIGITIZED pH-4 case,
   not for clean numeric pH-6.39 ring data. (Caught by the
   derived-quantity profile the plan mandated.)

## Robustness & sensitivities

- **Optimum:** 83 evals, 0 solver restarts; at θ_L both perturbed
  restarts return to basin (A exact to 4 dp) — unimodal. (At the
  15.4 µm sub-optimum the landscape was multi-modal; the accepted
  21.7 optimum is not.)
- **L_eff bracket (REFITS):** {12 µm: J 813, 15.4: 293, 21.7: 12.2}
  — decisive thick-film preference. The plateau-overshoot at 15.4
  diagnosed it (model pinned at its transport ceiling).
- **OCP {0.994, 1.019, 1.044}:** J 12.18–12.21, flat — NOT
  load-bearing.
- **N {0.20, 0.224, 0.25}:** absorbs monotonically into log f_4w
  (±0.25 dec); partition absolute scale is ring-calibration-limited,
  identifiability at fixed N is tight.
- **Weight swap w_r ×{0.5, 2}:** [pending — confirmatory].

## Conditional labels on θ_L (lock tiers)

| Claim | Status | Blocking ask |
|---|---|---|
| model fits dual-series data | **LOCKED** (computational) | none |
| both routes necessary (mechanism) | **LOCKED** | none |
| partition fraction ≈0.72 4e | **LOCKED** (ring-robust) | none |
| α transfers across condition | **LOCKED** | none |
| absolute θ_L k0 VALUES | conditional | rpm (sets L_eff), ring cal (N) |
| 21.7 µm physical interpretation | conditional | rpm |
| absolute-potential statements | conditional | 0.47 V OCP (empirically weak) |
| cross-condition "one catalyst" | conditional | catalyst identity + raw Tafel xlsx |

## Data asks (priority order)

1. **rpm** for the 2019 K₂SO₄ run — sets L_eff, hence θ_L's absolute
   k0; distinguishes "sub-1600 rpm" from "δ_OH-film bias" for the
   21.7 µm.
2. **catalyst identity** (CMK-3 vs GC) — gates the cross-condition
   "same catalyst" interpretation.
3. **ring hold potential + calibration** — gates the partition's
   absolute scale (N band).
4. **iR sign** used in the paper pipeline (Stage-0 audit found the
   sheet uses a nonstandard +I·Rs; 0.19 V at plateau).
5. gas/T/O₂ protocol (air already excluded by the data); sibling pH
   xlsx; raw Tafel xlsx; 0.47 V OCP component.

## Deferred confirmatory runs (not blocking the computational lock)

Kw ×0.5 ablation (needs a driver kw-scale flag); window-edge ±0.05,
background-scale ×{0.5,1.5}, sheet-iR-axis variants (need extractor
re-parametrization); BFGS-Hessian cross-check at θ_L (FD diagonal
already verified at θ*). All expected confirmatory given the OCP/N/
L_eff sensitivities already mapped.

## pH-series generalization (frozen θ_L, water-only)

Frozen θ_L applied across bulk pH (only c_H + per-pH OCP changed;
water routes only). 9/11 conditions converged (pH 1.65–6.39, all
13/13 points); **alkaline pH 10 & 12 failed the acidic-tuned anchor**
(LadderExhausted at c_H=1e-7 — needs a bulk-H continuation ramp;
follow-up). Compared to Brianna's `Exp Info` metrics (same
electrode/cation/run) and the ACS 3-panel range.

**Key result — the model does NOT generalize across pH:**
- The water-route onset is **pH-independent on the RHE axis** (η =
  V_RHE − E°_RHE; the OCP shift cancels). Model onset is flat at
  ~0.52 V (0.514→0.531 over pH 6.39→1.65); the DATA onset moves
  −0.221 V (0.472→0.251), ≈ −0.047 V/pH.
- Model peak selectivity is flat ~55%; data swings 73%→20% (and a
  91% spike at pH 4.21).
- Model max-ring is flat ~0.36; data varies 0.08–0.50.

The frozen water-only model reproduces the volcano MAGNITUDE in the
right ballpark everywhere but carries **no pH-dependence** — because
water-route BV is c_H-independent. The strong measured pH-trends in
onset and selectivity therefore live in physics the frozen model
omits: the **c_H-dependent acid routes** (active at genuinely low
bulk pH, NOT Kw-laundering there) and **local-pH / cation coupling**.
This is the quantitative motivation for a pH-coupled extension —
and it means the single-condition pH-6.39 fit, while excellent, is
NOT yet a transferable pH-resolved model.

Figures:
`StudyResults/phase7p2_ph_series_generalization/model_vs_expinfo_metrics.png`
(metric trends), `.../model_3panel_acidic.png` +
`.../ACS_experimental_3panel_reference.png` (3-panel layout).
