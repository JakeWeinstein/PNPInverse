# V22 — Tikhonov on log_k0, TRF on G0, 4 inits × 2 σ values

Per Route 2 / Path C in `docs/CHATGPT_HANDOFF_9_ADJOINT_RESOLVED_GRID_DONE.md`,
launched after V21 LM showed unbounded LM is decisively worse than TRF
(catastrophic Tafel-ridge slide).

Setup: bv_log_rate=True, observables=CD+PC, init_cache=cold-solve at INIT,
σ_data=2%×max|target|, V_GRID=G0=[-0.10, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60].
TRF (bounded) with 2 prior residual rows appended:
  `r_prior_j = (log_k0_j - log_k0_prior_center_j) / σ_log_k0`
prior_center = TRUE (literature/EIS measurement at the TRUE k0_j).

## Results

### log3 prior (σ_log_k0 = log(3) ≈ 1.099 — "factor-of-3 uncertainty")

| init | evals | wall | cost_total | k0_1 err | k0_2 err | α_1 err | α_2 err | <5% count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| plus20 | 16 | 1.8min | 1.35e+02 | -32.77% | **+0.60%** | -8.55% | +13.93% | 1 |
| minus20 | 25 | 2.7min | 8.01e-02 | **-1.56%** | +54.39% | **+0.02%** | **-0.88%** | 3 |
| k0high_alow | 29 | 3.1min | 1.68e-01 | +25.01% | +58.37% | **-0.65%** | **-0.97%** | 2 |
| k0low_ahigh | 16 | 1.8min | 1.36e+02 | -45.54% | -18.09% | -7.79% | +13.32% | 0 |

### log10 prior (σ_log_k0 = log(10) ≈ 2.303 — "factor-of-10 uncertainty")

| init | evals | wall | cost_total | k0_1 err | k0_2 err | α_1 err | α_2 err | <5% count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| plus20 | 16 | 1.8min | 1.39e+02 | -41.73% | **+0.78%** | -8.28% | +13.94% | 1 |
| minus20 | 21 | 2.2min | 2.60e-02 | **-2.80%** | +57.91% | **+0.05%** | **-0.94%** | 3 |
| k0high_alow | 20 | 2.0min | 4.01e-01 | +63.40% | +69.60% | **-1.33%** | **-1.13%** | 2 |
| k0low_ahigh | 16 | 1.8min | 1.42e+02 | -51.70% | -26.37% | -7.67% | +13.56% | 0 |

## TRF G0 baseline (no prior, from `v20_best_grid_trf_clean/`)

| init | cost | k0_1 err | k0_2 err | α_1 err | α_2 err | <5% count |
|---|---:|---:|---:|---:|---:|---:|
| plus20 | 141 | -45.06% | +0.85% | -8.16% | +13.94% | 1 |
| minus20 | 0.011 | -3.72% | +59.09% | +0.07% | -0.96% | 3 |
| k0high_alow | 0.40 | +73.80% | +71.30% | -1.52% | -1.16% | 2 |
| k0low_ahigh | 143 | -53.88% | -29.93% | -7.62% | +13.67% | 0 |

## Headline

**Tikhonov σ=log(3) and σ=log(10) priors do NOT break the basin geometry.**
Total `<5% counts` across the 4 inits: TRF=6, log10=6, log3=6 — identical.
Every init lands in the same basin under all three settings; the prior only
shifts the optimizer's position WITHIN that basin.

The shifts ARE monotone in σ — stronger prior pulls k0 closer to TRUE within
a basin:

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

Every (init, param) shows monotone movement toward TRUE as prior tightens. The
prior IS doing what it's supposed to — it's just not strong enough to push
across basin barriers.

## Why the prior cannot break the ridge at σ=log(3)

For minus20, the recovered point has data cost ≈ 0.002 (essentially perfect
fit), prior cost ≈ 0.078, total 0.080. At TRUE, both costs are 0. So TRUE
has lower total cost — but TRF doesn't go there because the path between
the basins crosses a DATA cost barrier (LM's line profile in
`v21_lm_g0_clean/lm_minus20/line_profile.json` shows J peak ~5-10 along the
recovered→TRUE direction).

For the prior to push TRF over a barrier of height H, we'd need
`0.5 · (log(k0_err)/σ)² > H` at the wrong-basin endpoint. With H ≈ 5 and
log_k0_2_err ≈ 0.434 at minus20, we'd need σ < 0.137 — i.e., σ ≈ log(1.15),
a "factor-of-15%" prior. That's an unreasonably strong claim for an
electrochemical rate constant.

## Reframing the result

The combined picture from V20 (FIM ablation), V21 (LM), V22 (Tikhonov):

1. **Local Fisher information at TRUE is well-conditioned** (cond=1.79e+7,
   ridge_cos=0.031) — data CONTAINS information about all 4 parameters
   *locally near TRUE*.
2. **Global cost surface has multiple (k0, α) Tafel-ridge basins.** Each
   init falls into one; basin endpoints are different points on the
   `k0_j · exp(-α_j · n_e · η / V_T) = const` manifolds.
3. **α is data-identifiable across most inits** to 0.02-2% (8-13% in
   stalled basins where TRF's bounds saturate α).
4. **k0_1 and k0_2 are not data-identifiable from a single init** — TRF
   endpoint depends on which basin the init starts in.
5. **Multi-init voting picks the wrong answer for k0_2.** Minimum-cost
   basin (minus20) has k0_2 robustly +54-59% off across all prior strengths;
   the true basin (lower total cost at the global minimum) is unreachable.
6. **Tikhonov σ=log(3) and log(10) on log_k0** do not change which basin
   each init falls into. Stronger priors (σ ≈ log(1.15) or tighter) would
   work but are physically indefensible without an actual EIS measurement.

## Implications

The handoff #9's question "Path A (LM) vs Path C (Tikhonov) first?" is now
answered for both:
- LM (V21): catastrophically worse than TRF (no-bounds slide).
- Tikhonov σ=log(3), log(10) (V22): same basin endpoints as TRF baseline.

The honest publishable claim from a single CD+PC experiment is:

> The Fisher information of CD + peroxide current under log-rate BV at
> V_RHE ∈ [-0.10, +0.60] is well-conditioned at TRUE (cond=1.79e+7), but the
> global cost surface contains multiple (k0, α) Tafel-ridge basins. α is
> robustly data-identifiable to ≤2% across multiple inits; k0_1 and k0_2
> are reachable individually from specific inits but not jointly without a
> tight (σ ≲ log(1.5)) prior or richer observables. Soft priors at literature
> uncertainty (σ = log(3) or log(10)) reduce within-basin position bias but
> do not change which basin each init converges to.

## What to try next (priority order)

1. **Stronger Tikhonov σ=log(2) ≈ 0.69 and σ=log(1.5) ≈ 0.41** — characterize
   the prior strength needed to break the ridge. Defensible if EIS measurement
   for k0 is available. Cheapest experiment (~25 min for 4 inits × 2 σ).

2. **Restart-with-perturbation from minus20 endpoint.** From the current
   (k0_2 +54%, others <5%) point, perturb log_k0_2 by ±0.5 (factor of e^0.5
   ≈ 1.65) and re-run TRF. If it returns to the same point, basin barrier
   confirmed; if it escapes, restart-with-perturbation is a viable strategy.

3. **Positive-voltage FIM diagnostic** (handoff side branch). Test G0+[0.70],
   G0+[0.70, 0.80] at FIM level only. Stop if weak eigvec stays |log_k0_1|
   ≈ 1. Per handoff Question 5: low expected payoff.

4. **Multi-experiment / different observables (e.g. different bulk c_O2,
   EIS) as a separate study.** Out of scope per handoff #9. This is the
   only path to a fully data-identifiable 4-param recovery.

## Files

- `scripts/studies/v18_logc_lsq_inverse.py` — added `--prior-sigma-log-k0`
  and `--prior-center` args. Tikhonov rows appended in `_eval_with_prior`
  helper inside `fun`/`jac` wrappers; `compute_residuals_and_jacobian` is
  unchanged (data residuals only).
- `trf_log3_*/`, `trf_log10_*/` — per-init result.json + history_partial.json
  + line_profile.json + true_curve.npz + targets.npz. result.json includes
  `use_prior`, `prior_sigma_log_k0`, `prior_center`, `log_k0_prior_center`.
- `_master_run.log` — full sequential run log with all 8 inits.
