# V23 -- Negative-V FIM ablation (extend grid to V <= -0.30)
Per ``docs/TODO_extend_inverse_v_range_negative.md``: extend the V20 unified FIM dataset with FD sensitivity rows at the missing negative voltages (V20 cold-ramp failed at V = -0.30 and V = -0.50, which voltage continuation now reaches in 1-2 s per V) and rerun a V20-style ablation across G0 + extended grids. Pass criterion: weak eigvec |log_k0_1| component drops below 0.95 under at least one noise model on at least one extended grid.
Anchor cold-solve at V = +0.000, warm-continuation outward.

## Combined V grid (rows available)
| V | cd | pc |
|---:|:---|:---|
| -0.500 | -1.8410e-01 | -1.8131e-01 |
| -0.400 | -1.8305e-01 | -1.6393e-01 |
| -0.300 | -1.8172e-01 | -6.5069e-02 |
| -0.200 | -1.8017e-01 | -1.5678e-03 |
| -0.100 | -1.7802e-01 | +2.9058e-06 |
| +0.000 | -1.7379e-01 | +1.5288e-05 |
| +0.100 | -1.6307e-01 | +1.5388e-05 |
| +0.150 | -1.4504e-01 | +1.5394e-05 |
| +0.200 | -7.8607e-02 | +1.5414e-05 |
| +0.250 | -2.0216e-03 | +1.5437e-05 |
| +0.300 | -1.9201e-05 | +1.5438e-05 |
| +0.400 | -1.5436e-05 | +1.5436e-05 |
| +0.500 | -1.1882e-05 | +1.1882e-05 |
| +0.600 | +1.8240e-09 | +2.7043e-09 |

## Results -- noise model: global_max
| grid | NV | sv_min | cond(F) | ridge_cos | |k0_1| weak | weak_eigvec |
|---|---:|---:|---:|---:|---:|---|
| G0 | 7 | 2.510e+00 | 1.79e+07 | 0.031 | 0.999 | [+0.999, +0.031, -0.018, -0.001] |
| G_neg1 | 8 | 1.763e-01 | 1.16e+09 | 0.043 | 0.999 | [+0.999, +0.042, -0.027, -0.008] |
| G_neg2 | 9 | 1.921e-02 | 1.96e+10 | 0.972 | 0.236 | [-0.236, -0.972, +0.006, +0.012] |
| G_neg3 | 11 | 2.610e-02 | 1.54e+09 | 0.999 | 0.032 | [-0.032, -0.999, +0.001, +0.010] |

## Results -- noise model: local_rel
| grid | NV | sv_min | cond(F) | ridge_cos | |k0_1| weak | weak_eigvec |
|---|---:|---:|---:|---:|---:|---|
| G0 | 7 | 5.356e+00 | 1.10e+08 | 0.011 | 1.000 | [+1.000, +0.011, -0.018, -0.001] |
| G_neg1 | 8 | 6.477e+00 | 7.59e+07 | 0.007 | 1.000 | [+1.000, +0.007, -0.017, -0.000] |
| G_neg2 | 9 | 8.487e+00 | 4.43e+07 | 0.003 | 1.000 | [+1.000, +0.003, -0.016, +0.000] |
| G_neg3 | 11 | 8.552e+00 | 4.36e+07 | 0.003 | 1.000 | [+1.000, +0.003, -0.016, +0.000] |

## Verdict
**WEAK PASS** -- weak direction stays log_k0_1 but cond(F) improves >= 2.0x and sv_min improves >= 1.5x on: G_neg2 [local_rel] (cond/2.48x, sv_min x1.58), G_neg3 [local_rel] (cond/2.52x, sv_min x1.60). Geometry is unchanged but local curvature along log_k0_1 is sharper -- worth a single inverse run on the extended grid as a sanity test, but unlikely to break basin barriers by itself.
**Artifact flips (do not count)** -- weak direction rotated but cond(F) got WORSE on: G_neg2 [global_max] (cond x1092.32 WORSE -- noise-model artifact), G_neg3 [global_max] (cond x85.66 WORSE -- noise-model artifact). This is the V20 sigma_pc-inflation trap: V <= -0.30 has |PC| ~ 0.06-0.18, which under global_max sets sigma_pc by 2% x max|pc| and demotes every V > 0 PC row by ~10^4. The flip is mathematical, not informative.
FAIL on: G_neg1 [global_max] (cond/0.02x, sv_min x0.07), G_neg1 [local_rel] (cond/1.45x, sv_min x1.21).

## Recommendation
Geometry is essentially unchanged but conditioning improves under local_rel. Suggested order: (a) optionally rerun the multi-init clean inverse on the best extended grid as a low-cost sanity check (TODO Step 2), but do not expect it to recover all 4 parameters; (b) proceed to HANDOFF_11 Phase 2A (bulk-O2 variation) regardless, since the weak direction is unchanged.
