# V19 — adjoint vs FD at extended V_GRID with log-rate

V tested: [0.3]

Pass criterion: rel_err < 1% for components |FD| > 1e-6 · max|FD|;
absolute error < 1e-6 · max|FD| for near-zero components.

**Result: 0/8 PASS, 6 FAIL**

## Per-component results

| V | obs | param | adjoint | FD | rel_err | verdict |
|---:|:---:|:---|---:|---:|---:|:---|
| +0.30 | cd | log_k0_1 | -3.7628e-06 | -3.9214e-06 | 4.046e-02 | FAIL |
| +0.30 | cd | log_k0_2 | -5.2650e-13 | +6.9568e-10 | 1.001e+00 | FAIL |
| +0.30 | cd | alpha_1 | -1.1130e-04 | -1.5652e-04 | 2.889e-01 | FAIL |
| +0.30 | cd | alpha_2 | -5.2650e-11 | +nan | nan | FD-NAN |
| +0.30 | pc | log_k0_1 | -1.2367e-12 | -8.2805e-12 | 8.506e-01 | FAIL |
| +0.30 | pc | log_k0_2 | +5.2467e-13 | -6.9568e-10 | 1.001e+00 | FAIL |
| +0.30 | pc | alpha_1 | -3.6582e-11 | -4.6057e-11 | 2.057e-01 | FAIL |
| +0.30 | pc | alpha_2 | +5.2467e-11 | +nan | nan | FD-NAN |
