# Surrogate Training v12 -- Targeted Gap-Fill

## Purpose

The Phase 1 training data audit (`StudyResults/training_data_audit/REPORT.md`)
revealed coverage holes in the 3,194-sample training set: a max-empty-ball
radius of 0.58 (threshold 0.15) located in the high-k0_1 / low-k0_2 /
high-alpha_1 / low-alpha_2 corner.  Convergence failures clustered heavily in
the low-k0_2 decades (84 of 251 failures in [1e-7, 1e-6)).  This script
generates 1,000 targeted samples to fill those gaps and merges them with the
existing dataset.

## Sampling Strategy

| Region | Samples | k0_1 | k0_2 | alpha_1 | alpha_2 | Rationale |
|--------|---------|------|------|---------|---------|-----------|
| 1 -- Gap corner | 500 | [0.1, 1.0] | [1e-7, 1e-5] | [0.6, 0.9] | [0.1, 0.4] | Fills the largest empty-ball hole |
| 2 -- Low k0_2 | 300 | [1e-6, 1.0] | [1e-7, 1e-6] | [0.1, 0.9] | [0.1, 0.9] | Backfills the failure-depleted decade |
| 3 -- Wide supplement | 200 | [1e-6, 1.0] | [1e-7, 0.1] | [0.1, 0.9] | [0.1, 0.9] | General coverage improvement |

All regions use Latin Hypercube Sampling with log-space k0 values and
deterministic seeds (401, 402, 403).

## Solver Hardening

**Aggressive SNES options** (vs charged-system defaults):

| Parameter | Default | Gap-fill |
|-----------|---------|----------|
| `snes_max_it` | 300 | 500 |
| `snes_atol` | 1e-7 | 1e-6 |
| `snes_rtol` | 1e-10 | 1e-8 |
| `snes_linesearch_maxlambda` | -- | 0.8 |
| `snes_divergence_tolerance` | -- | 1e14 |

**Aggressive recovery config** (via `ForwardRecoveryConfig`):
- 7 total attempts (vs 4 default), with 3 max-it-only, 3 tolerance-relaxation
- Max iteration cap raised to 800
- Tolerance relaxation factor 10x per level (atol, rtol, ksp_rtol)
- Full line-search cycling: bt -> l2 -> cp -> basic

## 6-Level Escalation Ladder

Each sample is attempted through up to six levels before being marked failed:

| Level | Strategy | max_eta_gap | Notes |
|-------|----------|-------------|-------|
| L1 | Direct solve, warm-started from previous sample | 3.0 | Fastest path |
| L2 | k0_2 continuation (8 steps from anchor 1e-3) | 3.0 | Log-space homotopy |
| L3 | k0_2 continuation, denser bridge (12 steps) | 1.5 | Finer step size |
| L4 | Multi-param continuation (also walks alpha from 0.5) | 1.5 | For extreme alpha values |
| L5 | Multi-param continuation + lenient steady-state (16 steps, anchor 1e-4) | 1.5 | Relaxed SS tolerances |
| L6 | Cold start, lenient steady-state | 1.0 | Last resort |

Levels 2-5 only trigger when k0_2 < 1e-3. Level 4 additionally requires at
least one alpha to be extreme (< 0.2 or > 0.8).

## Convergence Techniques

- **Parameter continuation in k0_2**: Walk from an "easy" anchor (k0_2 = 1e-3)
  toward the target in log-space steps, propagating converged initial conditions
  forward.  The PDE solution varies smoothly in log(k0_2).
- **Multi-parameter continuation (alpha walk)**: Simultaneously walk alpha_1 and
  alpha_2 from 0.5 midpoints toward target values to avoid Butler-Volmer
  exponential overflow at extreme alphas.
- **Bridge point densification**: Increase continuation steps from 8 to 12 or 16
  and tighten `max_eta_gap` from 3.0 to 1.5 for finer voltage interpolation.
- **Lenient steady-state fallback**: Relaxed config with `rel_tol=1e-3`,
  `max_steps=200`, `consecutive_steps=2` to grind through stiff transients.
- **Cross-sample warm-starting**: Nearest-neighbor ordering of the full sample
  set so each worker solves samples close in parameter space sequentially,
  reusing converged solutions as initial conditions.

## Output Files

| File | Description |
|------|-------------|
| `StudyResults/surrogate_v12/training_data_gapfill.npz` | Gap-fill samples (valid + metadata) |
| `StudyResults/surrogate_v12/training_data_merged_v12.npz` | Merged dataset (existing + gap-fill) |
| `StudyResults/surrogate_v12/split_indices_v12.npz` | Train/test split (85/15, seed 888) |
| `StudyResults/surrogate_v12/training_data_gapfill.npz.checkpoint.npz` | Checkpoint for resume |

The `.npz` files contain: `parameters`, `current_density`, `peroxide_current`,
`phi_applied`, plus region metadata and convergence flags.

## How to Run

```bash
cd /path/to/PNPInverse
/path/to/venv-firedrake/bin/python \
    scripts/surrogate/overnight_train_v12_gapfill.py 2>&1 \
    | tee StudyResults/surrogate_v12/run.log
```

**Checkpoint/resume**: Re-run the same command.  The script detects existing
checkpoints and skips completed samples automatically.  Phase 2 (merge) is also
idempotent.

## Estimated Runtime

1.5--3 hours for 1,000 samples on 8 parallel workers, depending on how many
samples require continuation retries.  Heartbeat messages print every 5 minutes
during long waits.
