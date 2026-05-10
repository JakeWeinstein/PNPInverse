# V19 — adjoint vs FD at extended V_GRID with log-rate

V tested: [0.3]

Pass criterion: rel_err < 1% for components |FD| > 1e-6 · max|FD|;
absolute error < 1e-6 · max|FD| for near-zero components.

**Result: 1/8 PASS, 7 FAIL**

## Per-component results

| V | obs | param | adjoint | FD | rel_err | verdict |
|---:|:---:|:---|---:|---:|---:|:---|
| +0.30 | cd | log_k0_1 | -3.7628e-06 | -3.7702e-06 | 1.963e-03 | PASS |
| +0.30 | cd | log_k0_2 | -5.2650e-13 | +1.4235e-10 | 1.004e+00 | FAIL |
| +0.30 | cd | alpha_1 | -1.1130e-04 | -1.1260e-04 | 1.153e-02 | FAIL |
| +0.30 | cd | alpha_2 | -5.2650e-11 | +7.5787e-09 | 1.007e+00 | FAIL |
| +0.30 | pc | log_k0_1 | -1.2367e-12 | +1.0611e-09 | 1.001e+00 | FAIL |
| +0.30 | pc | log_k0_2 | +5.2467e-13 | -1.4235e-10 | 1.004e+00 | FAIL |
| +0.30 | pc | alpha_1 | -3.6582e-11 | -3.3180e-07 | 9.999e-01 | FAIL |
| +0.30 | pc | alpha_2 | +5.2467e-11 | -7.5789e-09 | 1.007e+00 | FAIL |
