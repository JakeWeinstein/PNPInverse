# V26 — strategy B (grid_charge_continuation) at Ny=200

Tests whether `solve_grid_with_charge_continuation` (Phase 1 V-sweep at z=0, Phase 2 per-V z-ramp) reaches the full Apr 27 target grid V_RHE in [-0.50, +0.60] V at production resolution after the `boltzmann_z_scale` patch.

## Setup

- V_GRID = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
- mesh: graded rectangle Nx=8, Ny=200, beta=3.0
- TRUE: K0_HAT_R1=1.263158e-03, K0_HAT_R2=5.263158e-05, α1=0.627, α2=0.5

## Per-voltage results

| V_RHE (V) | cd | pc | z_factor | converged |
|---:|---:|---:|---:|---|
| -0.500 | (none) | (none) | 0.0000 | False |
| -0.400 | (none) | (none) | 0.0000 | False |
| -0.300 | (none) | (none) | 0.0000 | False |
| -0.200 | (none) | (none) | 0.0000 | False |
| -0.100 | +5.1974e+103 | +5.1974e+103 | 0.0000 | False |
| +0.000 | -1.7379e-01 | +1.5288e-05 | 1.0000 | True |
| +0.050 | -1.6985e-01 | +1.5378e-05 | 1.0000 | True |
| +0.100 | -1.6307e-01 | +1.5388e-05 | 1.0000 | True |
| +0.200 | +7.0881e+131 | +7.0881e+131 | 0.0000 | False |
| +0.300 | (none) | (none) | 0.0000 | False |
| +0.400 | (none) | (none) | 0.0000 | False |
| +0.500 | (none) | (none) | 0.0000 | False |
| +0.600 | (none) | (none) | 0.0000 | False |

## Aggregate

- converged: 3/13
- wall: 25.4s
- full target grid covered: False

