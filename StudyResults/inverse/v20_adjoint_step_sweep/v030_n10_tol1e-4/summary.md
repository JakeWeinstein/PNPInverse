# V19 — adjoint vs FD at extended V_GRID with log-rate

V tested: [0.3]

Pass criterion: rel_err < 1% for components |FD| > 1e-6 · max|FD|;
absolute error < 1e-6 · max|FD| for near-zero components.

**Result: 0/8 PASS, 8 FAIL**

## Per-component results

| V | obs | param | adjoint | FD | rel_err | verdict |
|---:|:---:|:---|---:|---:|---:|:---|
| +0.30 | cd | log_k0_1 | -3.7627e-06 | -1.8816e-06 | 9.997e-01 | FAIL |
| +0.30 | cd | log_k0_2 | +1.6347e-13 | -1.7319e-05 | 1.000e+00 | FAIL |
| +0.30 | cd | alpha_1 | -1.1130e-04 | -5.5659e-05 | 9.997e-01 | FAIL |
| +0.30 | cd | alpha_2 | +1.6347e-11 | +2.8620e-08 | 9.994e-01 | FAIL |
| +0.30 | pc | log_k0_1 | -1.9590e-12 | -1.8816e-06 | 1.000e+00 | FAIL |
| +0.30 | pc | log_k0_2 | -1.6539e-13 | +1.7319e-05 | 1.000e+00 | FAIL |
| +0.30 | pc | alpha_1 | -5.7949e-11 | -5.5659e-05 | 1.000e+00 | FAIL |
| +0.30 | pc | alpha_2 | -1.6539e-11 | -2.8620e-08 | 9.994e-01 | FAIL |
