# V19 — adjoint vs FD at extended V_GRID with log-rate

V tested: [0.2]

Pass criterion: rel_err < 1% for components |FD| > 1e-6 · max|FD|;
absolute error < 1e-6 · max|FD| for near-zero components.

**Result: 4/8 PASS, 4 FAIL**

## Per-component results

| V | obs | param | adjoint | FD | rel_err | verdict |
|---:|:---:|:---|---:|---:|---:|:---|
| +0.20 | cd | log_k0_1 | -1.6284e-02 | -1.6234e-02 | 3.115e-03 | PASS |
| +0.20 | cd | log_k0_2 | -4.3546e-12 | +7.2625e-09 | 1.198e-08 | PASS-NEAR0 |
| +0.20 | cd | alpha_1 | -6.0846e-01 | -6.0657e-01 | 3.115e-03 | PASS |
| +0.20 | cd | alpha_2 | -4.3546e-10 | +2.5384e-09 | 4.903e-09 | PASS-NEAR0 |
| +0.20 | pc | log_k0_1 | -5.1184e-09 | -4.8438e-09 | 5.671e-02 | FAIL |
| +0.20 | pc | log_k0_2 | +6.0630e-12 | -7.2510e-09 | 1.001e+00 | FAIL |
| +0.20 | pc | alpha_1 | -1.9125e-07 | -1.8099e-07 | 5.671e-02 | FAIL |
| +0.20 | pc | alpha_2 | +6.0630e-10 | -1.3908e-09 | 1.436e+00 | FAIL |
