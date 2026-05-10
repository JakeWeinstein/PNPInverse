# V19 — adjoint vs FD at extended V_GRID with log-rate

V tested: [0.3]

Pass criterion: rel_err < 1% for components |FD| > 1e-6 · max|FD|;
absolute error < 1e-6 · max|FD| for near-zero components.

**Result: 2/8 PASS, 6 FAIL**

## Per-component results

| V | obs | param | adjoint | FD | rel_err | verdict |
|---:|:---:|:---|---:|---:|---:|:---|
| +0.30 | cd | log_k0_1 | -3.7628e-06 | -3.7631e-06 | 9.573e-05 | PASS |
| +0.30 | cd | log_k0_2 | -5.2650e-13 | +1.1524e-09 | 1.000e+00 | FAIL |
| +0.30 | cd | alpha_1 | -1.1130e-04 | -1.1138e-04 | 6.991e-04 | PASS |
| +0.30 | cd | alpha_2 | -5.2650e-11 | +2.2970e-07 | 1.000e+00 | FAIL |
| +0.30 | pc | log_k0_1 | -1.2367e-12 | +4.1634e-11 | 1.030e+00 | FAIL |
| +0.30 | pc | log_k0_2 | +5.2467e-13 | -1.1524e-09 | 1.000e+00 | FAIL |
| +0.30 | pc | alpha_1 | -3.6582e-11 | +1.0951e-08 | 1.003e+00 | FAIL |
| +0.30 | pc | alpha_2 | +5.2467e-11 | -2.2970e-07 | 1.000e+00 | FAIL |
