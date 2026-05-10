# V20 Task C — TRF on G2 grid (clean data)

Compares TRF inverse on G0 (7-V grid) vs G2 (10-V grid with +0.00, +0.15, +0.25 added).

Setup: bv_log_rate=True, observables=CD+PC, regularization=none, init_cache=cold-solve at INIT, σ=2%×max|target|.

## Results

| init | grid | k0_1 err | k0_2 err | α_1 err | α_2 err | <5% count | cost | wall (min) | status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| plus20 | G0 | -45.06% | +0.85% | -8.16% | +13.94% | 1 | 141 | 2.5 | 3 |
| plus20 | G2 | -85.01% | -45.94% | +5.84% | +1.86% | 1 | 4.84 | 5.3 | 3 |
| minus20 | G0 | -3.72% | +59.09% | +0.07% | -0.96% | 3 | 0.0107 | 2.1 | 3 |
| minus20 | G2 | -20.00% | -19.98% | -19.98% | -19.33% | 0 | 1.61e+03 | 3.7 | 3 |
| k0high_alow | G0 | +73.80% | +71.30% | -1.52% | -1.16% | 2 | 0.404 | 2.4 | 3 |
| k0high_alow | G2 | +135.04% | +46.19% | -13.56% | -13.48% | 0 | 817 | 3.2 | 3 |
| k0low_ahigh | G0 | -53.88% | -29.93% | -7.62% | +13.67% | 0 | 143 | 2.4 | 3 |
| k0low_ahigh | G2 | -61.11% | -61.92% | +2.71% | +2.14% | 2 | 1.45 | 4.9 | 3 |

## Per-init verdict (G2 vs G0)

| init | Δcost | Δ(<5% count) | verdict |
|---|---:|---:|---|
| plus20 | -136 | +0 | BETTER |
| minus20 | +1.61e+03 | -3 | WORSE |
| k0high_alow | +817 | -2 | WORSE |
| k0low_ahigh | -142 | +2 | BETTER |

## Handoff Task C pass criterion

- ≥1 init recovers all 4 params to <10%: **FAIL** (no init under both grids meets this)
- α stays <5% in most inits: G0: 2/4, G2: 2/4 (different inits each)
- Stalled high-cost endpoints reduced: G0: 2/4, G2: 2/4 (just shifted to different inits)

## Interpretation

**G2 does not pass.** Replacing 2 of the 4 inits' "good" results (minus20,
k0high_alow) with worse outcomes is a net loss. The "best" inits in each
grid:
- G0 best: minus20 (cost 0.011, 3/4 params <5%)
- G2 best: k0low_ahigh (cost 1.45, 2/4 params <5%) — but k0_1, k0_2 each
  off by 60+%

**The basin structure shifts with grid choice.** Adding voltages 0.00,
0.15, 0.25 changes which init lands in which basin:
- plus20 G0→G2: cost 141 → 4.8, but k0 errors go from -45/+1 to -85/-46
- minus20 G0→G2: cost 0.011 → 1610. minus20 GOT STUCK AT INIT under G2.
  Likely cause: G2 added voltages where minus20-init cold-solve has
  weak convergence, contaminating residuals at init.
- k0high_alow G0→G2: cost 0.40 → 817. Similar pattern — broke a
  previously-working init.
- k0low_ahigh G0→G2: cost 143 → 1.45 (improvement)

This confirms the FIM analysis prediction: the marginal cond improvement
(1.79e7 → 1.75e7, ~2%) does NOT translate into a uniform inverse-problem
improvement. Grid changes shift the cost landscape's basin structure
unpredictably.

## Recommendation

**Grid choice is not the lever.** The TRF-stall pattern is fundamental to
the cost landscape, not the grid. Per the V20 FIM ablation, log_k0_1 is
robustly the weak direction across all sensible grids; voltage choice
alone won't break it.

Next steps for the inverse problem:
1. **Try LM optimizer** (--method lm in v18 script) — different step
   strategy may handle the basin transitions better.
2. **FIM-eigenbasis x_scale**: rotate parameters via FIM eigenvectors so
   TRF takes equal-information steps. Should help the log_k0_1 weak
   direction.
3. **Restart-with-perturbation**: when TRF hits xtol-stop with non-zero
   residual, perturb θ randomly and re-optimize. May find better basin.
4. **Multi-init voting / ensemble**: run many inits and treat the cluster
   centroid as the answer, with the spread as the uncertainty.
5. **Weak prior (Tikhonov)**: per Task E in handoff, a weak log-k0 prior
   (σ_log_k0 ≈ log(3)) will likely make all 4 inits converge to TRUE.
   This is the publishable Bayesian framing.

Sticking with **G0 (current grid)** for downstream work makes sense
since G2 doesn't reliably help and breaks 2/4 inits.
