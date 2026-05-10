# V19 — adjoint vs FD at extended V_GRID with log-rate

V tested: [0.3]

Pass criterion: rel_err < 1% for components |FD| > 1e-6 · max|FD|;
absolute error < 1e-6 · max|FD| for near-zero components.

**Result: 6/8 PASS, 2 FAIL**

## Per-component results

| V | obs | param | adjoint | FD | rel_err | verdict |
|---:|:---:|:---|---:|---:|---:|:---|
| +0.30 | cd | log_k0_1 | -3.7628e-06 | -3.7628e-06 | 2.691e-06 | PASS |
| +0.30 | cd | log_k0_2 | -5.2650e-13 | -7.4815e-13 | 1.991e-09 | PASS-NEAR0 |
| +0.30 | cd | alpha_1 | -1.1130e-04 | -1.1130e-04 | 3.982e-06 | PASS |
| +0.30 | cd | alpha_2 | -5.2650e-11 | -7.4816e-11 | 1.991e-07 | PASS-NEAR0 |
| +0.30 | pc | log_k0_1 | -1.2367e-12 | -1.2399e-12 | 2.551e-03 | PASS |
| +0.30 | pc | log_k0_2 | +5.2467e-13 | +7.4635e-13 | 2.970e-01 | FAIL |
| +0.30 | pc | alpha_1 | -3.6582e-11 | -3.6676e-11 | 2.543e-03 | PASS |
| +0.30 | pc | alpha_2 | +5.2467e-11 | +7.4636e-11 | 2.970e-01 | FAIL |
