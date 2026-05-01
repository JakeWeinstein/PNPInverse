# Follow-up to ChatGPT: Path A (LM) and Path C (Tikhonov) both run; basin geometry is fundamental

This responds to `PNP Positive Voltage Extension Handoff.md`'s recommendation
order — Path A (LM on G0) → Path C (Tikhonov σ=log(3), log(10)) → noisy seeds.
A and C are both done.  Outcome is decisive but not the one we hoped for: the
single-experiment CD+PC inverse problem has fundamental basin geometry that
neither optimizer choice nor literature-defensible priors fix.

## Headline

1. **LM on G0 is decisively worse than TRF on every init.**  Total `<5%`
   parameter recoveries across 4 inits: TRF=6, LM=0.  Without bounds, LM's
   damped Gauss-Newton step slides far down the (k0, α) Tafel ridge — k0_2
   ends 100-15000× TRUE on minus20 and k0high_alow inits (the two best TRF
   results).  TRF's bounds on log_k0 (±2 from TRUE) were acting as an
   implicit prior keeping it near TRUE on the ridge.

2. **Tikhonov σ=log(3) and σ=log(10) on log_k0 don't break the basin
   geometry.**  Total `<5%` counts: TRF=6, log10 Tikhonov=6, log3 Tikhonov=6
   — identical.  Each init lands in the same basin under all three settings;
   the prior shifts position WITHIN a basin (monotonically toward TRUE as σ
   tightens) but does NOT cross basin barriers.

3. **The basin barrier is data-cost in nature.**  At minus20's basin
   endpoint with σ=log(3): data cost ≈ 0.002, prior cost ≈ 0.078, total
   0.080.  At TRUE both are 0.  TRF doesn't go to TRUE because the path
   between basins crosses a data-cost peak ≈ 5-10 (visible in the LM v21
   minus20 line profile).  For the prior to push TRF over a barrier of
   height H ≈ 5, σ ≲ log(1.15) is required — a "factor-of-15%" prior,
   physically indefensible.

4. **The combined V20+V21+V22 conclusion: α is data-identifiable, k0_1 and
   k0_2 are not jointly identifiable from a single CD+PC experiment.**
   The Fisher information at TRUE is well-conditioned (cond=1.79e+7,
   ridge_cos=0.031) but local; the global cost surface has multiple
   Tafel-ridge basins separated by data-cost barriers that no
   literature-defensible prior can bridge.

---

## Path A — LM on G0 (V21)

`StudyResults/v21_lm_g0_clean/`

### Setup

bv_log_rate=True, V_GRID=G0, observables=CD+PC, init_cache=cold-solve at
INIT, σ_data=2%×max|target|, no prior, 4 inits.  scipy.optimize.least_squares
method='lm' has no bounds, so we added an unphysical-parameter guard to
`compute_residuals_and_jacobian`: when α exits (0, 1] or |log_k0 - log_k0_TRUE|
> 5, the function returns huge residuals (1e3) and a back-pointing Jacobian
instead of letting the forward solver raise.

### Results

| init | evals | wall | guards | cost | k0_1 err | k0_2 err | α_1 err | α_2 err | <5% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| plus20 | 27 | 3.5min | 3 | 1.99e+02 | -17.25% | -43.25% | +28.39% | +18.32% | 0 |
| minus20 | 24 | 2.2min | 5 | 5.34 | +709.77% | +12252.66% | -5.99% | -10.00% | 0 |
| k0high_alow | 22 | 1.7min | 6 | 8.06 | +1388.83% | +14481.96% | -7.80% | -10.44% | 0 |
| k0low_ahigh | 24 | 3.0min | 3 | 1.87e+02 | -63.75% | -61.99% | +29.42% | +18.27% | 0 |

### Comparison to TRF G0 baseline

| init | TRF cost | LM cost | TRF <5% | LM <5% | verdict |
|---|---:|---:|---:|---:|---|
| plus20 | 141 | 199 | 1 | 0 | LM worse |
| minus20 | **0.011** | **5.34** | **3** | **0** | LM catastrophically worse |
| k0high_alow | 0.40 | 8.06 | 2 | 0 | LM catastrophically worse |
| k0low_ahigh | 143 | 187 | 0 | 0 | LM worse |

### Mechanism

Along the Tafel ridge `k0_j · exp(-α_j · n_e · η / V_T) = const`, log_k0 can
change a lot for a tiny α nudge while CD/PC stay nearly constant.  TRF's
bounds (log_k0 within ±2) kept it parked on the ridge near TRUE.  LM has no
bounds; its damped Gauss-Newton step rode the ridge until α hit the (0, 1]
boundary (where the guard kicked in) and log_k0 ended 100-15000× TRUE.

All 4 LM inits terminated with `xtol` (status=3) at non-zero residual — the
optimizer converged to a stationary point of the cost on the Tafel ridge.
The line profiles from recovered→TRUE show TRUE has lower J than the LM
endpoint in every case: TRUE IS reachable; LM walked away from it.

### Conclusion

**Route 1(a) [LM] from handoff #9 is dead.**  LM-without-bounds is
fundamentally incompatible with this Tafel-ridge problem.  TRF's bounds
were the only thing keeping it useful — and even those bounds didn't make
TRF recover all 4 params from any init.

---

## Path C — Tikhonov on log_k0 (V22)

`StudyResults/v22_tikhonov_g0/`

### Setup

Same as V21 but with TRF (bounded) and 2 Tikhonov prior residual rows
appended:

```
r_prior_j = (log_k0_j - log_k0_prior_center_j) / σ_log_k0
```

prior_center = TRUE (literature/EIS measurement at the actual k0_j).  Two
σ values: log(3) ≈ 1.099 (factor-of-3 uncertainty) and log(10) ≈ 2.303
(factor-of-10).  4 inits × 2 σ values = 8 runs.

Code change: added `--prior-sigma-log-k0` and `--prior-center` to
`v18_logc_lsq_inverse.py`; prior rows appended in a new `_eval_with_prior`
helper inside `fun`/`jac` wrappers; `compute_residuals_and_jacobian` itself
unchanged (data residuals only).

### Results

#### log3 prior (σ = log(3) ≈ 1.099)

| init | evals | wall | cost | k0_1 err | k0_2 err | α_1 err | α_2 err | <5% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| plus20 | 16 | 1.8min | 1.35e+02 | -32.77% | **+0.60%** | -8.55% | +13.93% | 1 |
| minus20 | 25 | 2.7min | 8.01e-02 | **-1.56%** | +54.39% | **+0.02%** | **-0.88%** | 3 |
| k0high_alow | 29 | 3.1min | 1.68e-01 | +25.01% | +58.37% | **-0.65%** | **-0.97%** | 2 |
| k0low_ahigh | 16 | 1.8min | 1.36e+02 | -45.54% | -18.09% | -7.79% | +13.32% | 0 |

#### log10 prior (σ = log(10) ≈ 2.303)

| init | evals | wall | cost | k0_1 err | k0_2 err | α_1 err | α_2 err | <5% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| plus20 | 16 | 1.8min | 1.39e+02 | -41.73% | **+0.78%** | -8.28% | +13.94% | 1 |
| minus20 | 21 | 2.2min | 2.60e-02 | **-2.80%** | +57.91% | **+0.05%** | **-0.94%** | 3 |
| k0high_alow | 20 | 2.0min | 4.01e-01 | +63.40% | +69.60% | **-1.33%** | **-1.13%** | 2 |
| k0low_ahigh | 16 | 1.8min | 1.42e+02 | -51.70% | -26.37% | -7.67% | +13.56% | 0 |

#### TRF baseline (no prior, from `v20_best_grid_trf_clean/`)

| init | cost | k0_1 err | k0_2 err | α_1 err | α_2 err | <5% |
|---|---:|---:|---:|---:|---:|---:|
| plus20 | 141 | -45.06% | +0.85% | -8.16% | +13.94% | 1 |
| minus20 | 0.011 | -3.72% | +59.09% | +0.07% | -0.96% | 3 |
| k0high_alow | 0.40 | +73.80% | +71.30% | -1.52% | -1.16% | 2 |
| k0low_ahigh | 143 | -53.88% | -29.93% | -7.62% | +13.67% | 0 |

### The basin endpoints are identical

Total `<5% counts` are 6 for all three settings.  Per-init `<5% counts` are
identical: (1, 3, 2, 0).  Each init lands in the same basin under TRF,
Tikhonov-log10, and Tikhonov-log3.  The prior just shifts position within
the basin.

### Within-basin shifts ARE monotone in σ

| init | param | TRF (∞) | log10 (2.30) | log3 (1.10) |
|---|---|---:|---:|---:|
| plus20 | k0_1 | -45.06% | -41.73% | -32.77% |
| plus20 | k0_2 | +0.85% | +0.78% | +0.60% |
| minus20 | k0_1 | -3.72% | -2.80% | -1.56% |
| minus20 | k0_2 | +59.09% | +57.91% | +54.39% |
| k0high_alow | k0_1 | +73.80% | +63.40% | +25.01% |
| k0high_alow | k0_2 | +71.30% | +69.60% | +58.37% |
| k0low_ahigh | k0_1 | -53.88% | -51.70% | -45.54% |
| k0low_ahigh | k0_2 | -29.93% | -26.37% | -18.09% |

All 8 (init, param) pairs show monotone movement toward TRUE as the prior
tightens.  The prior IS pulling — it's just not strong enough to push across
basin barriers.

### The basin barrier height calculation

For minus20 with σ=log(3):
- Recovered point: data cost ≈ 0.002, prior cost ≈ 0.078, total 0.080
- TRUE: data cost = 0, prior cost = 0, total 0

TRUE has lower total cost.  TRF doesn't go there because the recovered→TRUE
path has a DATA cost peak ≈ 5-10 (LM line profile in
`v21_lm_g0_clean/lm_minus20/line_profile.json`).

For the prior to push over a barrier of height H ≈ 5 at log_k0_2 error
≈ 0.434, we need `0.5 · (0.434/σ)² > 5`, i.e., σ ≲ 0.137.  That's
σ ≈ log(1.15), a "factor-of-15%" prior.  That's an unreasonably strong claim
for an electrochemical k0 — even high-quality EIS rarely gives k0 to ±15%.

### Conclusion

**Route 2 [Tikhonov on log_k0] at literature-defensible σ does not break the
basin geometry.**  σ = log(3) and σ = log(10) only refine within-basin
position; they don't change which basin each init falls into.  Multi-init +
voting picks the lowest-cost basin (minus20), which has k0_2 robustly +54-59%
off across all settings.

---

## Combined picture from V20+V21+V22

The Fisher information of CD + peroxide current under log-rate BV at
V_RHE ∈ [-0.10, +0.60] is:

- **Well-conditioned at TRUE** (cond=1.79e+7, ridge_cos=0.031).
- **Robust to noise model** — every voltage grid + noise model tested in V20
  has cond < 1e+9 and ridge_cos < 0.10.
- **Locally informative about all 4 parameters** — the weak eigenvector is
  almost-pure log_k0_1 (component ≥ 0.99 across all grids and noise models)
  but the weakest singular direction is still informative enough that LOCAL
  recovery near TRUE is well-posed.

But the global cost surface has **multiple (k0, α) Tafel-ridge basins**:

- α is robustly data-identifiable across most inits to 0.02-2% (8-13% in
  basins where TRF's α-bound saturates).
- k0_1 and k0_2 are reachable individually from specific inits but not
  jointly.  No init recovers all 4 parameters to <5%.
- The minimum-cost basin under any reasonable optimizer + literature prior
  has k0_2 robustly +50-60% off.

This is the honest publishable claim:

> Under a single CD+PC experiment with log-rate BV BCs, multi-init TRF
> recovers α to ≤ 2% in 2 of 4 inits and a single k0 to ≤ 5% in additional
> inits, but not all 4 parameters jointly.  The local Fisher information at
> TRUE is well-conditioned (cond=1.79e+7) and the global cost surface
> contains multiple basins separated by data-cost barriers.  Soft priors at
> literature-defensible uncertainty (σ_log_k0 = log(3) or log(10)) reduce
> within-basin position bias but cannot bridge the inter-basin barriers
> (which would require σ ≲ log(1.15)).  Joint 4-parameter recovery requires
> either (a) tighter prior information than is realistic from EIS/literature
> alone, or (b) richer observables — multi-experiment data (different bulk
> c_O2, temperature, EIS impedance), which is a separate study.

---

## Specific questions for you

1. **Is the "α robust, k0 multi-modal" framing the right paper story?**
   Reframing from "ridge fully broken" (post-handoff #6) and "k0_2 needs a
   prior" (post-handoff #5) to "the data fixes α robustly + each individual
   k0 in a specific init, but the (k0_1, k0_2) joint recovery requires
   tighter prior or multi-experiment data."  Stronger than #5, more honest
   than #6.

2. **Is σ ≲ log(1.5) ever a defensible Bayesian prior on k0 in this field?**
   If EIS impedance or RDE Tafel slope analysis CAN yield k0 to ±50% (σ ≈
   log(1.5) ≈ 0.41), running V22 with σ=log(2) and σ=log(1.5) would
   characterize where the basin breaks.  Worth doing?

3. **Restart-with-perturbation as a Route 1(c) sanity check?**  From the
   minus20 endpoint (k0_1 -1.6%, k0_2 +54%, α correct), perturb log_k0_2 by
   ±0.5 (factor-of-1.65) and re-run TRF.  If TRF returns to the same point,
   basin barrier confirmed; if it escapes, restart-with-perturbation might
   recover all 4.  Cheap (~30 min for ±0.5 × 2-3 init perturbations).

4. **Is the positive-voltage FIM diagnostic still on the path?**  Per your
   handoff, "stop if the weak direction remains pure log_k0_1."  It does
   (V20 confirmed |log_k0_1| ≥ 0.99 across all grids).  Do we run the cheap
   diagnostic at G0+[0.70], G0+[0.70, 0.80] anyway to formally close it out,
   or skip?

5. **Multi-experiment / different-observables strategy.**  The most natural
   second experiment is a different bulk c_O2 — same electrode, same setup,
   different c_O2.  This breaks the (k0_1, α_1) Tafel ridge directly because
   r_1 ∝ k0_1 · c_O2 · exp(-α_1·...).  Worth scoping as a separate
   investigation, or stay single-experiment and write up the current result?

## Proposed next steps

The matrix:

```
A. Stronger Tikhonov sweep [σ = log(2), log(1.5), log(1.2)]
   ~30 min × 4 inits × 3 σ = 6 hours wall (sequential)
   Characterizes the prior strength needed to break the basin.
   Defensible only if EIS k0 measurements at this precision exist.

B. Restart-with-perturbation
   ~30 min, cheap, decisive on Route 1(c).

C. Positive-voltage FIM diagnostic
   ~30 min new script, low expected payoff.

D. Stop here, write paper around current results
   "α-robust, k0 multi-modal, single-experiment limit".
   Multi-experiment / EIS as future work.

E. Multi-experiment design
   New scope: forward-solve at different bulk c_O2, joint inverse.
```

My recommendation: **A (stronger Tikhonov)** if you have a use-case for
"factor-of-2 or factor-of-1.5 prior on k0" — it would tell us at exactly
what prior strength the basin breaks, which is a concrete numerical claim.

Otherwise: **D (stop and write up)** is the honest scientific endpoint of
the single-experiment story.

Tell me which path, or weigh in on questions 1-5 first.

## Files for reference

- `scripts/studies/v18_logc_lsq_inverse.py`
  - Added unphysical-parameter guard in `compute_residuals_and_jacobian`
    (V21 prerequisite).
  - Added `--prior-sigma-log-k0` and `--prior-center` flags.
  - Tikhonov rows appended in `_eval_with_prior` helper inside `fun`/`jac`
    wrappers.
- `StudyResults/v21_lm_g0_clean/` — Path A results.
  - `summary.md` — full LM-vs-TRF comparison.
  - `lm_{plus20,minus20,k0high_alow,k0low_ahigh}/{result.json,
    history_partial.json, line_profile.json, true_curve.npz, targets.npz}`.
  - `_master_run.log` — full sequential run log including 17 GUARD events
    across 4 inits.
- `StudyResults/v22_tikhonov_g0/` — Path C results.
  - `summary.md` — full Tikhonov-vs-TRF comparison.
  - `trf_log{3,10}_<init>/result.json` — includes `use_prior`,
    `prior_sigma_log_k0`, `prior_center`, `log_k0_prior_center` fields.
  - `_master_run.log` — full sequential run log for 8 runs.
- `docs/PNP Positive Voltage Extension Handoff.md` — your prior message.
- `docs/CHATGPT_HANDOFF_9_ADJOINT_RESOLVED_GRID_DONE.md` — Claude's response
  that motivated this round.
