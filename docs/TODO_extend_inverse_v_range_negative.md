# TODO — Extend the inverse solver's V grid to V<=-0.10 to capture PC dynamic range

**Date raised:** 2026-04-30
**Source:** `StudyResults/plot_iv_curves_3sp_true/iv_curves.npz`, generated
by `scripts/plot_iv_curves_3sp_true.py` at TRUE parameters with the
production 3sp + Boltzmann + log-c + log-rate forward solver.

## Observation

The current inverse-problem voltage grid is

```
G0 = [-0.10, +0.10, +0.20, +0.30, +0.40, +0.50, +0.60]   V_RHE
```

This grid catches the R1 kinetic-onset region (V ∈ [+0.10, +0.30]) and the
R2-only tail (V ∈ [+0.30, +0.60]) but stops at V = -0.10 on the cathodic
side. The full I-V scan over V ∈ [-0.5, +0.6] in 0.05 V steps shows that
**PC carries ~5 orders of magnitude of additional information at
V <= -0.10** that the current inverse grid does not use:

| V_RHE  | CD          | PC          |
|-------:|:------------|:------------|
| -0.500 | -1.8410e-01 | -1.8131e-01 |
| -0.450 | -1.8360e-01 | -1.7723e-01 |
| -0.400 | -1.8305e-01 | -1.6393e-01 |
| -0.350 | -1.8243e-01 | -1.2835e-01 |
| -0.300 | -1.8172e-01 | -6.5069e-02 |
| -0.250 | -1.8097e-01 | -1.4553e-02 |
| -0.200 | -1.8017e-01 | -1.5679e-03 |
| -0.150 | -1.7923e-01 | -1.2660e-04 |
| -0.100 | -1.7802e-01 | +2.9058e-06 |

Key features:

- PC **changes sign** between V = -0.10 (positive, peroxide-net-produced
  by R1) and V = -0.15 (negative, peroxide-net-consumed at the bath
  side from the small H2O2 seed).
- |PC| spans **+3e-6 to -1.8e-1** across V ∈ [-0.10, -0.50] -- almost
  five decades of dynamic range that the inverse currently ignores.
- CD over the same range varies by less than 4% (-0.178 to -0.184),
  so adding these voltages costs little in CD information density but
  could substantially improve PC-driven identifiability.

## Why this matters for identifiability

Per HANDOFF_10 / V20 FIM analysis, the post-rebuild weak Fisher direction
is almost-pure log_k0_1 (component >= 0.99 across all tested grids). A
sensitivity argument suggests V <= -0.10 is exactly where R1's
peroxide-consumption term should rotate the weak direction:

- R1 produces H2O2 at V > E_eq^{(1)} ≈ 0.68 V; consumes it at V < E_eq.
- At V ∈ [-0.5, -0.1], R1 is in its anodic / peroxide-consuming branch
  while R2 is essentially off (R2 needs V > +0.5 to unclip).
- The PC observable is therefore dominated by R1's anodic peroxide
  consumption -- a regime that directly probes (k0_1, alpha_1) jointly
  with the H2O2 stoichiometric coupling.

Plausible hypothesis worth testing with FIM only first: adding V = -0.20,
-0.30, -0.40 to G0 may break the log_k0_1 weak direction without needing
the multi-experiment bulk-O2 design proposed in HANDOFF_11 § 6. This is
a single-experiment grid extension, much cheaper than multi-experiment.

## Prerequisites

The 3sp log-c forward solver historically failed to **cold-ramp** at
V = -0.30 and V = -0.50 (per V20 voltage-grid FIM ablation,
`StudyResults/v20_voltage_grid_fim_ablation/summary.md`).

The I-V plot just generated shows that **voltage continuation
succeeds at all of V ∈ [-0.5, +0.6]**: the full grid converges in
1-2 s per voltage once warm-started from the V=0 cold-ramp anchor. So:

- The current `v18_logc_lsq_inverse.py` Step 1 (TRUE-curve solve) uses
  cold-ramp at every V independently and therefore fails at V <= -0.30.
- The forward solve at TRUE in this regime is **not actually impossible**;
  it just needs voltage continuation in Step 1.
- The optimizer's inv_cache machinery already does warm-starting per
  voltage *during* TRF iterations, but Step 1 doesn't.

## Proposed work order

1. **FIM-only first.** Recompute the V20 FIM ablation with the extended
   grids G_neg1 = G0 + [-0.20], G_neg2 = G0 + [-0.20, -0.30], G_neg3 =
   G0 + [-0.20, -0.30, -0.40, -0.50]. Use the I-V data from
   `iv_curves.npz` to construct S_white directly (no fresh forward
   solves needed; we already have CD, PC at all these V's).
   - Pass condition: weak eigvec |log_k0_1| component drops below 0.95.
   - This is the cheapest thing to try; ~15 minutes of analysis.

2. **If FIM passes**, add voltage continuation to v18's Step 1
   (TRUE-curve solve loop) and rerun the multi-init clean-data inverse
   with the extended grid. Compare against the V20 G0 baseline.

3. **Inverse-side voltage continuation.** Step 2.5 (INIT cold-solve) and
   the per-voltage warm-starting inside TRF already exist; only Step 1
   needs fixing. The `solve_warm_unann()` machinery is already there.

## Caveats

- PC noise model: at V = -0.10, PC = +3e-6 (positive); at V = -0.20,
  PC = -1.6e-3 (negative). Under a global-max noise model, sigma_pc is
  set by max|PC| across the grid. If we extend to V = -0.50 where
  |PC| = 0.18, the global sigma_pc jumps from ~3e-7 (current) to ~3.6e-3,
  which would **demote all the V > 0 PC rows by 4 orders of magnitude**
  -- a worst-case. Local-rel noise model is much safer here. This was
  the same trap that made V20 G3/G_best worse than G0 under global_max:
  see `StudyResults/v20_voltage_grid_fim_ablation/summary.md`.
- So if we extend V <= -0.10, we should also pin the noise model to
  local_rel (or the global-max-with-floor variant from V19).

## Status

Not blocking; logged for the next inverse-side iteration.
