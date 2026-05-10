# V23 Phase 0.1 — Restart-with-perturbation from minus20 wrong basin

Per HANDOFF_11 §3.0.1. Tests whether perturbing log_k0_2 only causes
TRF to escape to TRUE basin or return to same wrong basin.

## Setup

Wrong-basin endpoint (V19 G0 minus20):
```
log_k0_1 = -6.712083  (TRUE -6.673770)
log_k0_2 = -9.387864  (TRUE -9.851186, +0.464 above)
alpha_1  = 0.627423  (TRUE 0.627000)
alpha_2  = 0.495219  (TRUE 0.500000)
```

Perturbations applied to log_k0_2 only; other params held.
TRF (bounded), G0, log-rate, no prior, σ_data=2%×max|target|.

## Results

| perturb | log_k0_2 start | cost | k0_1 err | k0_2 err | α_1 err | α_2 err | <5% | basin |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| -0.75 | -10.138 | 1.269e-02 | +5.33% | -19.35% | -0.26% | +0.31% | 2 | OTHER |
| -0.50 | -9.888 | 4.558e-02 | -3.61% | +0.68% | +0.00% | -0.11% | 4 | TRUE_BASIN |
| -0.25 | -9.638 | 8.048e-03 | -3.38% | +26.55% | +0.06% | -0.49% | 3 | OTHER |
| +0.25 | -9.138 | 1.873e-02 | -3.66% | +94.75% | +0.06% | -1.36% | 3 | WRONG_HIGH |
| +0.50 | -8.888 | 3.701e-03 | -2.95% | +145.32% | +0.05% | -1.82% | 3 | WRONG_HIGH |
| +0.75 | -8.638 | 6.486e-02 | -3.79% | +199.57% | +0.23% | -2.10% | 3 | WRONG_HIGH |

## Verdict

PARTIAL — some perturbations escaped to TRUE basin; restart-with-perturbation IS a viable diagnostic strategy.
