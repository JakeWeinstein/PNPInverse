# V19 — adjoint vs FD at extended V_GRID with log-rate

V tested: [0.3]

Pass criterion: rel_err < 1% for components |FD| > 1e-6 · max|FD|;
absolute error < 1e-6 · max|FD| for near-zero components.

**Result: 0/8 PASS, 8 FAIL**

## Per-component results

| V | obs | param | adjoint | FD | rel_err | verdict |
|---:|:---:|:---|---:|---:|---:|:---|
| +0.30 | cd | log_k0_1 | -3.7628e-06 | -1.8816e-06 | 9.998e-01 | FAIL |
| +0.30 | cd | log_k0_2 | -1.8069e-12 | -1.7319e-05 | 1.000e+00 | FAIL |
| +0.30 | cd | alpha_1 | -1.1131e-04 | -5.5659e-05 | 9.998e-01 | FAIL |
| +0.30 | cd | alpha_2 | -1.8069e-10 | +2.8620e-08 | 1.006e+00 | FAIL |
| +0.30 | pc | log_k0_1 | -6.5101e-13 | -1.8816e-06 | 1.000e+00 | FAIL |
| +0.30 | pc | log_k0_2 | +1.8052e-12 | +1.7319e-05 | 1.000e+00 | FAIL |
| +0.30 | pc | alpha_1 | -1.9257e-11 | -5.5659e-05 | 1.000e+00 | FAIL |
| +0.30 | pc | alpha_2 | +1.8052e-10 | -2.8620e-08 | 1.006e+00 | FAIL |
