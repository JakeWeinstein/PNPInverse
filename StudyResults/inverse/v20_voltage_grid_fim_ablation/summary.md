# V20 — Voltage-grid FIM ablation

Per Task B in `CHATGPT_HANDOFF_8_LOGRATE_MULTIINIT.md`: test whether adding low/onset voltages (0.00, 0.15, 0.25) and mild negatives reduces the new weak direction (log_k0_1).

Setup: cap=50, log-rate ON, observables=CD+PC. Computed under two noise models (global 2% max and local 2% rel).

Note: V=-0.50 and V=-0.30 failed cold-ramp; G3 and G4 are evaluated on the converged subset.


## Results — noise model: global_max

| grid | NV | sv_min | cond(F) | ridge_cos | |k0_1| weak | weak_eigvec |
|---|---:|---:|---:|---:|---:|---|
| G0_current | 7 | 2.510e+00 | 1.79e+07 | 0.031 | 0.999 | [+0.999, +0.031, -0.018, -0.001] |
| G1_add_zero | 8 | 2.512e+00 | 1.79e+07 | 0.031 | 0.999 | [+0.999, +0.031, -0.018, -0.001] |
| G2_densify_R1_onset | 10 | 2.539e+00 | 1.75e+07 | 0.031 | 0.999 | [+0.999, +0.031, -0.018, -0.001] |
| G3_add_mild_negative | 11 | 2.158e-01 | 7.77e+08 | 0.043 | 0.999 | [+0.999, +0.043, -0.027, -0.008] |
| G4_add_strong_negative | 10 | 2.539e+00 | 1.75e+07 | 0.031 | 0.999 | [+0.999, +0.031, -0.018, -0.001] |
| G_best_candidate | 11 | 2.158e-01 | 7.77e+08 | 0.043 | 0.999 | [+0.999, +0.043, -0.027, -0.008] |

## Results — noise model: local_rel

| grid | NV | sv_min | cond(F) | ridge_cos | |k0_1| weak | weak_eigvec |
|---|---:|---:|---:|---:|---:|---|
| G0_current | 7 | 6.690e+01 | 7.03e+05 | 0.056 | 0.998 | [+0.998, +0.056, -0.021, -0.003] |
| G1_add_zero | 8 | 6.690e+01 | 7.03e+05 | 0.056 | 0.998 | [+0.998, +0.056, -0.021, -0.003] |
| G2_densify_R1_onset | 10 | 6.767e+01 | 6.87e+05 | 0.093 | 0.995 | [+0.995, +0.093, -0.025, -0.006] |
| G3_add_mild_negative | 11 | 6.805e+01 | 6.88e+05 | 0.086 | 0.996 | [+0.996, +0.086, -0.024, -0.006] |
| G4_add_strong_negative | 10 | 6.767e+01 | 6.87e+05 | 0.093 | 0.995 | [+0.995, +0.093, -0.025, -0.006] |
| G_best_candidate | 11 | 6.805e+01 | 6.88e+05 | 0.086 | 0.996 | [+0.996, +0.086, -0.024, -0.006] |

## Per-voltage leverage (fraction of ||S||², global_max noise)

| grid | V=-0.20 | V=-0.10 | V=+0.00 | V=+0.10 | V=+0.15 | V=+0.20 | V=+0.25 | V=+0.30 | V=+0.40 | V=+0.50 | V=+0.60 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G0_current | — | 0.979 | — | 0.000 | — | 0.000 | — | 0.000 | 0.000 | 0.021 | 0.000 |
| G1_add_zero | — | 0.979 | 0.000 | 0.000 | — | 0.000 | — | 0.000 | 0.000 | 0.021 | 0.000 |
| G2_densify_R1_onset | — | 0.979 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.021 | 0.000 |
| G3_add_mild_negative | 0.999 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| G4_add_strong_negative | — | 0.979 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.021 | 0.000 |
| G_best_candidate | 0.999 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Per-voltage contribution to smallest singular direction (global_max)

| grid | V=-0.20 | V=-0.10 | V=+0.00 | V=+0.10 | V=+0.15 | V=+0.20 | V=+0.25 | V=+0.30 | V=+0.40 | V=+0.50 | V=+0.60 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G0_current | — | 0.001 | — | 0.000 | — | 0.370 | — | 0.213 | 0.375 | 0.041 | 0.000 |
| G1_add_zero | — | 0.001 | 0.001 | 0.000 | — | 0.370 | — | 0.213 | 0.375 | 0.040 | 0.000 |
| G2_densify_R1_onset | — | 0.001 | 0.001 | 0.000 | 0.014 | 0.360 | 0.007 | 0.206 | 0.368 | 0.042 | 0.000 |
| G3_add_mild_negative | 0.000 | 0.398 | 0.000 | 0.002 | 0.245 | 0.027 | 0.077 | 0.003 | 0.010 | 0.239 | 0.000 |
| G4_add_strong_negative | — | 0.001 | 0.001 | 0.000 | 0.014 | 0.360 | 0.007 | 0.206 | 0.368 | 0.042 | 0.000 |
| G_best_candidate | 0.000 | 0.398 | 0.000 | 0.002 | 0.245 | 0.027 | 0.077 | 0.003 | 0.010 | 0.239 | 0.000 |

## Decision

### Under global_max
- Best (lowest cond): **G2_densify_R1_onset** (cond=1.75e+07, sv_min=2.54e+00)
- Worst: G3_add_mild_negative (cond=7.77e+08)

### Under local_rel
- Best (lowest cond): **G2_densify_R1_onset** (cond=6.87e+05, sv_min=6.77e+01)
- Worst: G0_current (cond=7.03e+05)

## Interpretation

**Adding V=+0.00 (G1) does nothing.** Already covered by neighbors at V=±0.10.
The cd values at V=-0.10/0/+0.10 are -0.178/-0.174/-0.163 — all in the
plateau regime where R1, R2 rates are ≈ equal. The new row is redundant.

**Adding V=+0.15, +0.25 (G2) gives marginal improvement.** cond improves
~2% under both noise models. ridge_cos goes 0.031→0.031 (global_max) or
0.056→0.093 (local_rel). The R1 onset transition does provide some info,
but most of it is already captured by V=+0.20.

**Adding V=-0.20 (G3, G_best) is bad under global_max, neutral under
local_rel.** At V=-0.20, |pc|=1.57e-3 (1000× larger than the high-V pc
≈ 1.5e-5). Under global_max σ_pc = 2% × max|pc|, the new big-pc V
inflates σ_pc by 100×, demoting all other pc rows. This bumps cond by
40× (1.79e7 → 7.77e8). Under local_rel each row has its own σ, so the
new row doesn't contaminate others — cond improves marginally.

**The weak direction is robustly log_k0_1 (|component|=0.99+) across
ALL grids and noise models.** Voltage-grid changes alone cannot break
this — additional observables or different bulk conditions would be
needed. This means the FIM ridge-breaking on log_k0_2 (the original
problem) survives, but log_k0_1 is now the bottleneck and no V choice
fixes it.

## Recommendation for Task C

**Use G2 = [-0.10, 0.00, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60].**

- Best cond under both noise models.
- Avoids V=-0.20 contamination of global_max σ_pc.
- 10 voltages — modest computational increase from G0 (7 voltages).
- ridge_cos under local_rel improves slightly (0.056 → 0.093), suggesting
  the new V=+0.15, +0.25 rows do extract a bit more info.

Under local_rel noise, G3 (with V=-0.20) is essentially tied with G2.
If the publishable noise model is local_rel, V=-0.20 can be added but
gives no real benefit. If it's global_max (or has significant absolute
floor), V=-0.20 should be excluded.

The TRF stalls observed in `CHATGPT_HANDOFF_8_LOGRATE_MULTIINIT` are
NOT addressable via grid choice — log_k0_1 is fundamentally the weak
direction. Task C should focus on optimizer choices (LM, FIM-eigenbasis
x_scale, restart-with-perturbation) rather than re-running TRF on a
"better" grid.
