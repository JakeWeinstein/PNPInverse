# Noise Model Conventions for FIM / Inverse Analysis

**Date:** 2026-04-30
**Source:** V20 voltage-grid FIM ablation + V23 negative-V FIM ablation
findings; documented to fix the convention for the paper writeup and
future identifiability studies in this project.

## TL;DR

Use **`local_rel + abs_floor`** as the default noise model for FIM and
inverse analysis. Report results under **all three** models
(`global_max`, `local_rel`, `local_rel + abs_floor`) so robustness is
visible. Treat `global_max` flips as suspect when the grid spans many
decades of |y|.

## Why this convention exists

Across V20 and V23, voltage-grid changes interacted with the noise
model in ways that made one model say PASS and another say FAIL on the
same data. The honest publishable claim has to be invariant to a
defensible noise model choice. `global_max` alone is **not** defensible
when the observable spans many decades across the V grid.

## The three noise models

For an observable `y` measured at voltages `V_1, ..., V_NV`:

### 1. `global_max`

```
sigma_y = c_rel * max_i |y(V_i)|
```

One scalar `sigma_y` for the whole grid. Each whitened residual row is
`y(V_i) / sigma_y`.

- **Pro:** simple; matches the 2% relative noise convention used in
  early identifiability studies.
- **Con:** fragile when |y| spans many decades. The biggest-|y| voltage
  fixes `sigma_y` for everyone else and demotes the small-|y| rows
  proportionally to the dynamic range.
- **In this project:** PC spans ~8 decades across V in [-0.5, +0.6]
  (|PC| from 3e-9 to 0.18). Adding V=-0.50 inflates `sigma_pc` by ~10^4
  and demotes every V > 0 PC row. V20 documented this with V=-0.20
  alone (cond went 1.79e7 -> 7.77e8); V23 confirmed for V=-0.30
  (cond -> 1.96e10).

### 2. `local_rel`

```
sigma_y(V_i) = c_rel * |y(V_i)|
```

Per-voltage `sigma`. Each whitened residual row is
`y(V_i) / sigma_y(V_i)` -- effectively the relative residual.

- **Pro:** every row contributes to the FIM at its own scale; not
  dominated by the largest |y|.
- **Con:** undefined / numerically unstable as |y| -> 0. The small-|y|
  rows get unbounded whitened sensitivity if `c_rel * |y|` underflows.

### 3. `local_rel + abs_floor` (RECOMMENDED DEFAULT)

```
sigma_y(V_i) = sqrt( (c_rel * |y(V_i)|)^2 + sigma_abs^2 )
```

Smooth interpolation: relative noise dominates at large signals,
absolute floor dominates at small signals.

- **Pro:** matches real instrument noise structure (potentiostat noise
  has both gain-multiplied and quantization components); finite
  everywhere.
- **Con:** `sigma_abs` must be defended -- requires citing instrument
  specs.

This is what should be used as the default.

## Choosing `c_rel` and `sigma_abs`

### `c_rel` (relative noise component)

Typical values for steady-state RDE/RRDE current measurements with a
quality potentiostat:
- 1% (0.01) for high-end, well-calibrated setups
- 2% (0.02) is the standard FIM literature default
- 5% if explicit drift / electrode preparation variability is included

This project uses **`c_rel = 0.02`** in V19/V20/V23 to stay consistent
with the FIM identifiability literature. Document the value used and
keep it consistent across studies.

### `sigma_abs` (absolute floor)

Driven by potentiostat range / quantization:

- Typical 16-bit potentiostat at 1 mA range: LSB ~ 30 nA absolute.
- After dimensionless scaling by `I_SCALE` (see `scripts/_bv_common.py`),
  this becomes an absolute floor in nondimensional CD/PC units.
- Good first cut: `sigma_abs = c_rel * |y_min|` where `|y_min|` is the
  smallest |y| at which the data is still informative -- this prevents
  noise-floor rows from drowning out real information.
- Better: cite the potentiostat datasheet noise spec divided by
  `I_SCALE`.

For this project, V23 used `sigma_abs = 1e-8` in nondimensional units
as a first-cut floor. This matches roughly:
- |y_min| ~ 1e-9 to 1e-8 nondim (e.g. PC at V=+0.6) below which we
  expect noise to dominate.

A more defensible choice would be derived from the actual instrument
spec, expressed in physical units, then divided by `I_SCALE`. Update
this doc when the spec is settled.

## How `global_max` artifacts manifest

A **`global_max` flip** is when the FIM weak direction rotates between
`global_max` and `local_rel` results -- and the rotation is driven by
which voltages are included, not by the data itself. Diagnostic:

- Did `cond(F)` get **worse** in the rotated case?
- Did the new "weak direction" land on the parameter most associated
  with the demoted rows?

If both yes, the flip is a noise-model artifact. Do not report it as
an identifiability gain.

V23 example (G_neg2 = G0 + [-0.20, -0.30]):
- `global_max`: weak |log_k0_1| 0.999 -> 0.236 (looks like rotation)
  but cond(F) 1.79e7 -> 1.96e10 (1100x WORSE). Artifact.
- `local_rel`: weak |log_k0_1| stays 1.000, cond(F) 1.10e8 -> 4.43e7
  (2.5x better). Real but modest.

## How to defend the choice in the paper

1. State the noise model explicitly with both `c_rel` and `sigma_abs`
   numerical values, cite the instrument datasheet for `sigma_abs`.
2. Report main FIM / inverse-recovery results under
   `local_rel + abs_floor`.
3. Include a sensitivity table: same metrics under `global_max` and
   pure `local_rel` for comparison. Note any rotations and explain
   them as noise-model artifacts when cond degrades.
4. The headline claim should hold under all three: e.g. "weak Fisher
   direction is log_k0_1 across all noise models tested."

## Where this lives in code

- Whitening conventions are factored in
  `scripts/studies/v23_negative_v_fim_ablation.py::whiten_rows`.
  Currently supports `global_max` and `local_rel` (with floor=1e-8 in
  the local case). To extend, add a `local_rel_with_floor` mode that
  reads `sigma_abs` from a config or CLI flag.
- V19/V20 outputs report `sigma_cd` and `sigma_pc` in JSON; treat them
  as `global_max` values unless the producing script's noise mode is
  documented otherwise.

## Studies referenced

- `StudyResults/v20_voltage_grid_fim_ablation/summary.md` -- first
  surfaced the `global_max` fragility (G3 / G_best worse under
  `global_max`, neutral under `local_rel`).
- `StudyResults/v23_negative_v_fim_ablation/summary.md` -- confirmed
  the artifact pattern at V <= -0.30 and added the three-tier verdict
  (STRONG PASS / WEAK PASS / artifact / FAIL) to distinguish real ID
  gains from noise-model rotations.
- `docs/TODO_extend_inverse_v_range_negative.md` -- predicted the
  `global_max` trap before V23 ran it; verified.
- `docs/CHATGPT_HANDOFF_10_LM_TIKHONOV_BASIN_GEOMETRY.md` -- shows
  Tikhonov / LM analysis is also robust to noise-model choice; the
  basin geometry doesn't change with whitening.
