# V19 — adjoint vs FD at extended V_GRID with log-rate

V tested: [0.3]

Pass criterion: rel_err < 1% for components |FD| > 1e-6 · max|FD|;
absolute error < 1e-6 · max|FD| for near-zero components.

**Result: 2/8 PASS, 6 FAIL**

## Per-component results

| V | obs | param | adjoint | FD | rel_err | verdict |
|---:|:---:|:---|---:|---:|---:|:---|
| +0.30 | cd | log_k0_1 | -3.7628e-06 | -3.7630e-06 | 6.166e-05 | PASS |
| +0.30 | cd | log_k0_2 | -5.2650e-13 | +6.4708e-10 | 1.001e+00 | FAIL |
| +0.30 | cd | alpha_1 | -1.1130e-04 | -1.1135e-04 | 4.011e-04 | PASS |
| +0.30 | cd | alpha_2 | -5.2650e-11 | +7.2696e-08 | 1.001e+00 | FAIL |
| +0.30 | pc | log_k0_1 | -1.2367e-12 | +2.3102e-11 | 1.054e+00 | FAIL |
| +0.30 | pc | log_k0_2 | +5.2467e-13 | -6.4708e-10 | 1.001e+00 | FAIL |
| +0.30 | pc | alpha_1 | -3.6582e-11 | +6.1481e-09 | 1.006e+00 | FAIL |
| +0.30 | pc | alpha_2 | +5.2467e-11 | -7.2696e-08 | 1.001e+00 | FAIL |
