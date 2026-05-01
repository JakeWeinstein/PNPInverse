# V19 — adjoint vs FD at extended V_GRID with log-rate

V tested: [0.3]

Pass criterion: rel_err < 1% for components |FD| > 1e-6 · max|FD|;
absolute error < 1e-6 · max|FD| for near-zero components.

**Result: 3/8 PASS, 5 FAIL**

## Per-component results

| V | obs | param | adjoint | FD | rel_err | verdict |
|---:|:---:|:---|---:|---:|---:|:---|
| +0.30 | cd | log_k0_1 | -3.7628e-06 | -3.7664e-06 | 9.714e-04 | PASS |
| +0.30 | cd | log_k0_2 | -5.2650e-13 | +3.2821e-11 | 2.971e-07 | PASS-NEAR0 |
| +0.30 | cd | alpha_1 | -1.1130e-04 | -1.1224e-04 | 8.327e-03 | PASS |
| +0.30 | cd | alpha_2 | -5.2650e-11 | +1.0029e-06 | 1.000e+00 | FAIL |
| +0.30 | pc | log_k0_1 | -1.2367e-12 | +5.1881e-10 | 1.002e+00 | FAIL |
| +0.30 | pc | log_k0_2 | +5.2467e-13 | -3.2823e-11 | 1.016e+00 | FAIL |
| +0.30 | pc | alpha_1 | -3.6582e-11 | +1.3592e-07 | 1.000e+00 | FAIL |
| +0.30 | pc | alpha_2 | +5.2467e-11 | -1.0029e-06 | 1.000e+00 | FAIL |
