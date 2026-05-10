# V24 — 3sp+Boltzmann log-c vs 4sp standard PNP

Apples-to-apples regeneration of the writeup's overlap claim using the *current production* forward solver from `scripts/studies/v18_logc_lsq_inverse.py` (most recent inverse) and the 4sp standard PNP via `Forward.bv_solver.solve_grid_with_charge_continuation`.

## Setup

- V_GRID = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.05, 0.1]
- log-rate (3sp side) = False
- mesh: graded rectangle Nx=8, Ny=200, beta=3.0
- TRUE params: K0_HAT_R1=1.263158e-03, K0_HAT_R2=5.263158e-05, α1=0.627, α2=0.5
- E_eq R1/R2 = 0.68 / 1.78 V (RHE)
- pass threshold: |Δobs|/max(|obs_4sp|) ≤ 5.00% per voltage (matches F2 tolerance used in `Forward/bv_solver/validation.py`)

## Per-voltage comparison

| V_RHE (V) | cd_4sp | cd_3sp | |Δcd|/cd_max% | pc_4sp | pc_3sp | |Δpc|/pc_max% | 3sp method | verdict |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| -0.500 | -1.8390e-01 | -1.8410e-01 | 0.108 | -1.8112e-01 | -1.8131e-01 | 0.104 | warm←-0.40 | PASS |
| -0.400 | -1.8283e-01 | -1.8305e-01 | 0.122 | -1.6374e-01 | -1.6391e-01 | 0.094 | warm←-0.30 | PASS |
| -0.300 | -1.8133e-01 | -1.8172e-01 | 0.212 | -6.4999e-02 | -6.5039e-02 | 0.022 | warm←-0.20 | PASS |
| -0.200 | -1.7968e-01 | -1.8017e-01 | 0.268 | -1.5838e-03 | -1.5678e-03 | 0.009 | cold | PASS |
| -0.100 | -1.7753e-01 | -1.7802e-01 | 0.266 | -1.2483e-05 | +2.9058e-06 | 0.008 | cold | PASS |
| +0.000 | -1.7330e-01 | -1.7379e-01 | 0.266 | -9.6937e-08 | +1.5288e-05 | 0.008 | cold | PASS |
| +0.050 | -1.6936e-01 | -1.6985e-01 | 0.269 | -8.6209e-09 | +1.5378e-05 | 0.008 | cold | PASS |
| +0.100 | -1.6289e-01 | -1.6307e-01 | 0.100 | -7.7618e-10 | +1.5388e-05 | 0.008 | cold | PASS |

## Aggregate

- overlap voltages where both solvers converged: 8 / 8
- max |Δcd|/cd_max: 0.269%
- mean |Δcd|/cd_max: 0.201%
- max |Δpc|/pc_max: 0.104%
- mean |Δpc|/pc_max: 0.033%
- per-voltage verdicts: PASS=8, FLAG=0

## Verdict

All 8 overlap voltages pass the 5.0% F2-style tolerance on both CD and PC.  This regenerates and quantifies the writeup's "matched closely within the validation tolerances" claim from real data and saves the artefacts under `v24_3sp_logc_vs_4sp_validation/`.

Wall time: 4sp = 17.8s, 3sp logc = 74.3s.

Artefacts:

- `raw_values.json`
- `comparison.csv`
- `comparison.png`
