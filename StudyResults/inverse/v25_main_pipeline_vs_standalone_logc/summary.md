# V25 — main-pipeline log-c parity vs standalone log-c

Verifies that the patched ``Forward.bv_solver`` dispatcher (formulation=logc + bv_log_rate=True + bv_bc.boltzmann_counterions) reproduces the production standalone path used in ``v18_logc_lsq_inverse.py`` / ``v24_3sp_logc_vs_4sp_validation.py`` to numerical precision.

## Setup

- V_GRID = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.05, 0.1]
- mesh: graded rectangle Nx=8, Ny=200, beta=3.0
- TRUE: K0_HAT_R1=1.263158e-03, K0_HAT_R2=5.263158e-05, α1=0.627, α2=0.5
- E_eq R1/R2 = 0.68 / 1.78 V (RHE)

## Per-voltage comparison

| V_RHE (V) | cd_main | cd_stand | |Δcd|/cd_max% | pc_main | pc_stand | |Δpc|/pc_max% | verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
| -0.500 | -0.1841027049115712 | None | — | -0.18130803074852705 | None | — | MISSING |
| -0.400 | -0.18305411697878235 | None | — | -0.1639104687710057 | None | — | MISSING |
| -0.300 | -0.18172225585484408 | None | — | -0.06503908565477808 | None | — | MISSING |
| -0.200 | -1.8017e-01 | -1.8017e-01 | 0.0002 | -1.5674e-03 | -1.5678e-03 | 0.0003 | PASS |
| -0.100 | -1.7802e-01 | -1.7802e-01 | 0.0001 | +2.9076e-06 | +2.9058e-06 | 0.0000 | PASS |
| +0.000 | -1.7379e-01 | -1.7379e-01 | 0.0000 | +1.5288e-05 | +1.5288e-05 | 0.0000 | PASS |
| +0.050 | -1.6985e-01 | -1.6985e-01 | 0.0000 | +1.5378e-05 | +1.5378e-05 | 0.0000 | PASS |
| +0.100 | -1.6307e-01 | -1.6307e-01 | 0.0000 | +1.5388e-05 | +1.5388e-05 | 0.0000 | PASS |

## Aggregate

- overlap voltages: 5 / 8
- max |Δcd|/cd_max: 0.0002%
- mean |Δcd|/cd_max: 0.0001%
- max |Δpc|/pc_max: 0.0003%
- mean |Δpc|/pc_max: 0.0001%
- per-voltage verdicts: PASS=5, FLAG=0
- wall: main=72.2s, standalone=63.9s
