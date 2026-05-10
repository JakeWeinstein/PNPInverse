# PNP-BV Log-Rate Multi-Init Status + Next Tasks for Claude Code

## Purpose of this handoff

This summarizes the latest log-rate BV validation results and gives a concrete next-step plan. The key new question from ChatGPT/user is whether adding back omitted negative/low voltages could help stabilize the inverse problem.

Short answer: **test omitted low/onset voltages, but do not expect strongly negative transport-limited points to fix the remaining k0 ambiguity.** The immediate priority is still to debug the extended-voltage adjoint mismatch before running noisy inverse studies.

---

## Current state

### What is now solid

1. **Log-rate BV remains the essential numerical fix.**

   In the log-c solver, the standard BV path evaluated rates through

   ```python
   c_surf = exp(clamp(u, ±30))
   rate = k0 * c_surf * exp(...)
   ```

   This created an artificial H2O2 floor inside the BV residual. At high anodic voltage, the floor multiplied by the large R2 exponential generated a phantom R2 sink and caused Newton failures.

   The log-rate path instead evaluates

   ```python
   log_rate = ln(k0) + u_H2O2 + 2*(u_H - ln(c_ref)) - alpha*n_e*eta
   rate = exp(log_rate)
   ```

   so H2O2 can decay smoothly as `u_H2O2 -> -inf`. This enabled the solver to reach the extended voltage grid up to `V_RHE = +0.60`.

2. **The extended voltage grid fixes the old purely local FIM problem.**

   The old V ≤ +0.20 grid had a k0_2-dominated weak direction. With log-rate BV and V up to +0.60:

   ```text
   baseline V≤+0.20:     cond(F) ≈ 2.03e11, sv_min ≈ 2.35e-2, ridge_cos = 1.000
   extended V≤+0.60:     cond(F) ≈ 1.79e7,  sv_min ≈ 2.51,    ridge_cos = 0.031
   ```

   The local FIM at TRUE no longer says "k0_2 is hopeless." The weak direction shifted mostly to log_k0_1.

3. **Noise-model FIM audit passed.**

   Even with an absolute noise floor that buries the tiny high-V currents, the FIM remains much better than the old baseline:

   ```text
   noise model                    sv_min      cond(F)     ridge_cos
   global 2% max                  2.510e+00   1.79e+07    0.031
   local 2% rel                   6.690e+01   7.03e+05    0.056
   local 2% + floor 1e-6          2.874e+00   1.32e+06    0.020
   local 2% + floor 1e-7          5.218e+00   2.93e+07    0.011
   local 2% + floor 1e-8          5.355e+00   1.07e+08    0.011
   local 2% + floor 1e-9          6.060e+00   8.56e+07    0.011
   ```

4. **The TRUE-cache-as-IC bug was correctly fixed.**

   The inverse pipeline should not initialize the optimizer's per-voltage cache from TRUE-parameter steady states. That can make offset-parameter solves fail or mislead TRF.

   Correct architecture:

   ```python
   # TRUE solve is only for target observables
   true_cache = solve_all_voltages_at_true_params()

   # NEW: cold-solve at the actual INIT parameters
   init_cache = solve_all_voltages_at_init_params()

   # optimizer starts from init_cache, with true_cache only as last fallback
   inv_cache = [init_cache[i] if init_cache[i] is not None else true_cache[i]
                for i in range(NV)]
   ```

---

## Latest multi-init TRF result

Run settings:

```text
bv_log_rate = True
V_GRID = [-0.10, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
observables = CD + PC
free params = [log_k0_1, log_k0_2, alpha_1, alpha_2]
regularization = none
noise = clean data
sigma weighting = uniform 2% × max|target|
init cache = cold-solve at INIT params
```

Results:

```text
init          k0_1 err    k0_2 err    alpha_1 err   alpha_2 err   cost
+20%          -45.06%     +0.85%      -8.16%        +13.94%       141
-20%          -3.72%      +59.09%     +0.07%        -0.96%        0.011
k0high_alow   +73.80%     +71.30%     -1.52%        -1.16%        0.40
k0low_ahigh   -53.88%     -29.93%     -7.62%        +13.67%       144
```

Interpretation:

- `alpha_1` and `alpha_2` are the most robustly recoverable parameters.
- Each individual k0 is recoverable from at least one initialization:
  - `k0_2` to <1% from +20% init.
  - `k0_1` to <4% from -20% init.
- No single initialization recovers all four parameters.
- The old single k0_2 ridge is no longer the whole story, but curved Tafel-like manifolds remain.
- TRF can stall even when a monotone downhill path to TRUE exists.

The headline should **not** be "ridge fully broken." A more accurate statement is:

> Log-rate BV + extended V_GRID makes the local Fisher information much better and makes alpha robust, but the global single-experiment inverse still has multiple basins/ridge manifolds. k0 values are reachable but not uniquely recovered from arbitrary initialization without additional help.

---

## Critical unresolved issue: adjoint vs FD mismatch

At the new important voltages, the adjoint check is not yet clean:

```text
V_RHE   verdict
+0.30   FAIL — nonzero components adjoint ≈ 2.000 × FD
+0.40   FAIL — nonzero components adjoint ≈ 2.000 × FD
+0.50   FAIL — nonzero components adjoint ≈ 1.819 × FD
+0.60   PASS — adjoint matches FD to ~1e-7 relative
```

Likely hypotheses:

1. **FD is seeing unconverged transient response.**
   Perturbed FD solves may stop before H2O2 fully relaxes, especially around +0.30 where R2 is transport-limited.

2. **The tape is differentiating a short pseudo-time relaxation, not the true steady-state map.**
   Five annotated SNES/pseudo-time steps may carry a residual `c_old -> U_prev_step` dependency. If the warm start is close but not exactly steady, this can amplify gradients. The exact 2× pattern points strongly toward this.

3. **Less likely: pyadjoint/log-rate tape issue.**
   Possible, but the fact that +0.60 passes and k0_2 can be recovered from +20% init makes this less likely.

### Do this before noisy seeds

Run the `annotate_final_steps` sweep:

```text
annotate_final_steps = [1, 2, 3, 5, 10]
V_TEST = [0.30, 0.40, 0.50]
```

For each parameter and observable, report:

```text
adjoint
FD
adjoint / FD
residual norm before annotated segment
residual norm after annotated segment
steady-state stopping criterion
```

Decision rule:

- If adjoint/FD ratio scales with number of annotated steps, fix the tape/steady-state differentiation architecture before any more inverse studies.
- If FD changes with tighter steady-state tolerances while adjoint is stable, the FD check was contaminated by transient/unrelaxed solves.
- If neither explains it, isolate the log-rate path in a tiny single-voltage scalar test.

---

## Should we add back omitted negative voltages?

### Recommendation

Yes, but test this as a **voltage-grid ablation** first. Do not assume more negative points help.

Strongly negative voltages are mostly transport-limited. If diffusion/bulk parameters are fixed, those points mostly anchor the plateau and may add little direct information about k0 or alpha. They can even worsen conditioning by increasing the largest singular value without increasing the weakest one.

The more promising omitted points are the **low/onset voltages**:

```text
0.00, 0.15, 0.25
```

These densify the R1 kinetic/transition region and may help with the new weak direction, which is mostly `log_k0_1` in the extended FIM.

### Proposed FIM-only grid ablation

Run FIM diagnostics only first. Do not run TRF on every grid until the adjoint issue is resolved.

Compare:

```text
G0 current:
[-0.10, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]

G1 add zero:
[-0.10, 0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]

G2 densify R1 onset:
[-0.10, 0.00, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]

G3 add mild negative:
[-0.30, -0.20, -0.10, 0.00, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]

G4 add strong negative:
[-0.50, -0.30, -0.10, 0.00, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]
```

Also test the recommended compact grid:

```text
G_best_candidate:
[-0.20, -0.10, 0.00, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]
```

### Metrics to report for each grid

```text
sv_min(S_white)
cond(F)
weak eigenvector
ridge_cos against old k0_2/alpha_2 direction
component of weak eigvec along log_k0_1
parameter correlation matrix
per-voltage row leverage
per-voltage contribution to smallest singular direction
```

The most important question:

> Do the added low/onset rows increase sensitivity to log_k0_1 and reduce the log_k0_1 weak direction, or do they only add redundant plateau information?

### Expected outcome

Likely helpful:

```text
0.00, 0.15, 0.25
```

Possibly mildly helpful:

```text
-0.20, -0.10
```

Probably low value:

```text
-0.50, -0.40, -0.30
```

Do not add many strongly cathodic points unless the leverage analysis shows they improve the weakest direction.

---

## Proposed next-task order

### Task A — adjoint diagnostic

Run `annotate_final_steps` sweep before noisy inverse runs.

Deliverable:

```text
StudyResults/v20_adjoint_step_sweep/
    summary.md
    ratios_by_voltage.csv
    ratios_by_param.csv
    residuals_by_step.csv
```

Pass/fail:

- PASS: identify whether the 2× mismatch is tape-step amplification, FD transient contamination, or true log-rate adjoint bug.
- FAIL: cannot interpret inverse results reliably.

### Task B — voltage-grid FIM ablation

Run the grid set G0-G4 plus G_best_candidate.

Deliverable:

```text
StudyResults/v20_voltage_grid_fim_ablation/
    summary.md
    fim_by_grid.json
    weak_eigvec_by_grid.csv
    leverage_by_voltage.csv
```

Decision:

- If G2 or G_best_candidate improves `sv_min` and weak eigvec relative to G0, use it for the next TRF clean-data sweep.
- If strongly negative points do not improve the weakest direction, omit them.

### Task C — clean-data TRF on best grid

Only after Task A explains/fixes the Jacobian issue.

Run four inits again:

```text
+20%
-20%
k0high_alow
k0low_ahigh
```

Use:

```text
bv_log_rate = True
regularization = none
observables = CD + PC
init_cache = cold-solve at INIT
```

Deliverable:

```text
StudyResults/v20_best_grid_trf_clean/
    result_by_init.json
    line_profile_by_init.json
    residual_by_voltage.csv
```

Pass criteria:

- At least one init recovers all four parameters to <10%.
- Alpha remains <5% in most inits.
- Stalled high-cost endpoints are reduced or explained.

### Task D — noisy seeds

Only after clean-data behavior is interpretable.

Run 10 seeds first, not 50.

Use 2-3 most informative inits:

```text
-20%
+20%
best basin from Task C
```

Deliverable:

```text
StudyResults/v20_best_grid_noise_seeds/
    summary.md
    errors_by_seed.csv
    failure_modes.md
```

### Task E — weak prior/Tikhonov as Bayesian check

After data-only behavior is understood, add weak log-k0 priors:

```text
sigma_log_k0 = log(3)
prior centers:
    true
    0.3x true
    3x true
```

Purpose:

- not to claim data-only recovery;
- to show how much prior information is needed to stabilize k0 across basins.

---

## Suggested scientific framing for now

Do not say yet:

> The four-parameter inverse problem is solved.

Say:

> Log-rate BV evaluation and the extended voltage grid repair the severe k0_2 Fisher singularity and make alpha robustly data-identifiable. However, multi-init TRF reveals a nonconvex global landscape with curved Tafel-like manifolds. Additional grid design, adjoint cleanup, and/or weak priors are needed for consistent four-parameter recovery.

This is a stronger and more honest story than both earlier extremes:

- not "k0_2 is impossible without a prior";
- not "ridge fully broken."

The current best interpretation is:

> The data now contain the missing information locally, but the optimizer/Jacobian/global-basin problem is not yet solved.

---

## One-line instruction to Claude Code

Prioritize:

```text
A. Debug adjoint/FD mismatch via annotate_final_steps sweep.
B. Run FIM voltage-grid ablation to test adding 0.00, 0.15, 0.25 and mild negative voltages.
C. Only then rerun clean-data multi-init TRF on the best grid.
D. Defer noisy seeds and Tikhonov until A-C are resolved.
```
