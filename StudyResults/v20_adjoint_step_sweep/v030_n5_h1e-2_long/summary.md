# V19 — adjoint vs FD at extended V_GRID with log-rate

V tested: [0.3]

Pass criterion: rel_err < 1% for components |FD| > 1e-6 · max|FD|;
absolute error < 1e-6 · max|FD| for near-zero components.

**Result: 1/8 PASS, 7 FAIL**

## Per-component results

| V | obs | param | adjoint | FD | rel_err | verdict |
|---:|:---:|:---|---:|---:|---:|:---|
| +0.30 | cd | log_k0_1 | -3.7628e-06 | -1.8816e-06 | 9.997e-01 | FAIL |
| +0.30 | cd | log_k0_2 | -5.2650e-13 | +2.8620e-10 | 1.002e+00 | FAIL |
| +0.30 | cd | alpha_1 | -1.1130e-04 | -1.1133e-04 | 1.882e-04 | PASS |
| +0.30 | cd | alpha_2 | -5.2650e-11 | +1.4235e-08 | 1.004e+00 | FAIL |
| +0.30 | pc | log_k0_1 | -1.2367e-12 | -1.8816e-06 | 1.000e+00 | FAIL |
| +0.30 | pc | log_k0_2 | +5.2467e-13 | -2.8620e-10 | 1.002e+00 | FAIL |
| +0.30 | pc | alpha_1 | -3.6582e-11 | +2.7198e-09 | 1.013e+00 | FAIL |
| +0.30 | pc | alpha_2 | +5.2467e-11 | -1.4235e-08 | 1.004e+00 | FAIL |
