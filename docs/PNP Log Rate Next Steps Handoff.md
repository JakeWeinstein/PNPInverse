# PNP-BV Log-Rate Breakthrough — Next Steps for Claude Code

## Purpose

This handoff updates the project direction after the `CHATGPT_HANDOFF_6_LOGRATE_BREAKTHROUGH.md` results.

The key finding is that the previous `k0_2` / `alpha_2` non-identifiability was **not necessarily fundamental**. The log-rate Butler–Volmer evaluation removes a numerical artifact in the BV boundary rate, extends the convergent voltage grid to `V_RHE = +0.60 V`, and the Fisher information matrix now shows that the old `log_k0_2` ridge has been broken.

However, this is not yet a full inverse-problem victory. Before we update the project storyline, we need two confirmation steps:

1. **Noise-model FIM audit** — make sure the high-voltage information survives a realistic measurement-noise model.
2. **Clean-data TRF inverse** — verify that the optimizer actually recovers all four parameters without priors.

Do **not** start the FV/Scharfetter–Gummel prototype yet. It is no longer the urgent path unless the log-rate extended-grid inverse fails.

---

## Current interpretation

The important correction is:

> The problem was not that CG1 + log-c + Boltzmann could never expose `k0_2`. The problem was that the BV rate was still being evaluated through a clamped concentration path. That clamp created a phantom H2O2 floor inside the R2 boundary rate, which prevented the solver from reaching the voltage range where R2 provides independent sensitivity.

The log-rate form fixes this by replacing rate construction of the form

```python
cathodic = k0_j * c_surf[cat_idx] * exp(-alpha_j * n_e_j * eta_j)
cathodic *= (c_surf[sp_idx] / c_ref)**power
```

with

```python
log_cathodic = (
    ln(k0_j)
    + u[cat_idx]
    + power * (u[sp_idx] - ln(c_ref))
    - alpha_j * n_e_j * eta_j
)
cathodic = exp(log_cathodic)
```

The crucial difference is that `u[i]` enters additively. The rate can decay smoothly as `u_H2O2 -> -infinity`. It no longer hits an artificial concentration floor from `exp(clamp(u, ±30))`.

Where both formulations converge, the log-rate and standard forms agree to five significant figures. So this is not a physics change in the normal regime. It is a numerically better equivalent formulation.

---

## Major result so far

Previous extended-grid FIM result with log-rate BV:

```text
V_GRID up to +0.60 V
bv_log_rate = True
cap = 50

sv_min:    2.51
cond(F):   1.79e7
ridge_cos: 0.031
weak eigvec: mostly log_k0_1, not log_k0_2
```

Compare to the previous single-experiment CD+PC baseline:

```text
V_GRID up to +0.20 V
no log-rate

sv_min:    2.35e-2
cond(F):   2.03e11
ridge_cos: 1.000
weak eigvec: pure log_k0_2
```

This is a very large improvement:

```text
sv_min increased by ~107x
cond(F) improved by ~11,000x
old k0_2 ridge rotated away
```

The current best interpretation:

> Extending the voltage grid beyond `+0.30 V`, enabled by log-rate BV evaluation, exposes R2-dominated current and eventually R2 voltage-shape sensitivity. That makes `k0_2` and `alpha_2` locally identifiable in the Fisher analysis.

---

## Correct R2 unclipping threshold

The previous estimate that R2 would not unclip until about `+1.14 V` was based on the wrong clipping convention.

The code clips

```python
eta_scaled = (V_RHE - E_eq) / V_T
eta_scaled_clipped = clamp(eta_scaled, ±50)
```

before multiplying by `alpha * n_e`.

Therefore R2 unclips when:

```text
abs((V - E_eq_2) / V_T) < 50
```

or

```text
V > E_eq_2 - 50 * V_T
```

Using:

```text
E_eq_2 = 1.78 V
V_T    = 0.02569 V
```

gives:

```text
V_unclip_2 = 1.78 - 50 * 0.02569
           = 1.78 - 1.2845
           = +0.4955 V
```

So R2 begins to unclip around:

```text
V_RHE ≈ +0.495 V
```

This is why `V=+0.50` and `V=+0.60` matter.

Important caveat: document exactly what is being clipped. If the code later changes to clip the full final exponent instead of `eta_scaled`, the threshold formula changes.

---

## Important caution: the FIM may be noise-model sensitive

At `V=+0.60`, the current is very small. The existing FIM uses:

```text
sigma = 2% * max(abs(target))
```

per observable type.

That is useful for continuity with previous studies, but it may overstate the usefulness of tiny high-voltage signals if the real experiment has an absolute current noise floor.

Redo the FIM with multiple noise models before declaring the inverse problem solved.

### Noise model A: current convention

```text
sigma_y = 0.02 * max_V(abs(y(V)))
```

Keep this for comparison with existing results.

### Noise model B: local relative noise

```text
sigma_y(V) = 0.02 * abs(y(V))
```

Warning: this upweights tiny high-V signals because sigma becomes tiny. It does **not** represent a realistic instrument floor by itself.

### Noise model C: local relative noise plus absolute floor

Use:

```text
sigma_y(V) = sqrt((0.02 * abs(y(V)))**2 + sigma_abs**2)
```

Run several absolute floors in the same units as the observable:

```text
sigma_abs ∈ [1e-6, 1e-7, 1e-8, 1e-9]
```

or choose values based on the actual dimensional current-density scale if that conversion is already available.

### Required output table

Create:

```text
StudyResults/v19_lograte_noise_model_fim/
    summary.md
    fim_by_noise_model.json
    fim_by_noise_model.csv
    weak_eigvecs.csv
```

Include:

| noise model | sv_min | cond(F) | ridge_cos | weak eigvec |
|---|---:|---:|---:|---|
| global 2% max | | | | |
| local 2% | | | | |
| local 2% + floor 1e-6 | | | | |
| local 2% + floor 1e-7 | | | | |
| local 2% + floor 1e-8 | | | | |
| local 2% + floor 1e-9 | | | | |

### Interpretation rule

If the FIM still shows:

```text
cond(F) <= ~1e8 or 1e9
ridge_cos no longer near 1
weak eigvec no longer pure log_k0_2
```

under a realistic absolute floor, the result is robust.

If the improvement disappears under realistic absolute noise, then the new high-V information exists mathematically but may be experimentally unusable.

---

## Next required inverse test: clean-data TRF

The FIM is a local information test at TRUE parameters. The actual inverse test is whether TRF recovers the parameters from an offset initial guess.

Use:

```text
script base: scripts/studies/v18_logc_lsq_inverse.py
new options:
    bv_log_rate = True
    V_GRID = [-0.10, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
    regularization = None
    observables = CD + PC
    params = [log_k0_1, log_k0_2, alpha_1, alpha_2]
    optimizer = scipy.optimize.least_squares(method="trf")
```

Run clean data first.

### Initial guesses

At minimum:

```text
+20% all params
-20% all params
```

Also useful:

```text
k0 high / alpha low
k0 low / alpha high
mixed signs by reaction
```

The old ridge could hide failures if only one initial direction is tested.

### Clean-data pass criteria

Do not move to noisy synthetic tests unless clean data passes.

A clean-data run passes if:

```text
all four parameter errors < 10%
final residual near solver precision
projected gradient small
same solution from +20% and -20% starts
no monotone downhill path remains from recovered point to TRUE
```

Also save the line profile from recovered to TRUE as a diagnostic. If `J(t)` decreases monotonically from recovered to TRUE, the optimizer still failed even if the endpoint looks decent.

### Required output

```text
StudyResults/v19_lograte_extended_trf_clean/
    result_plus20.json
    result_minus20.json
    result_mixed_init_*.json
    summary.md
    line_profiles/
```

The summary should include:

| init | k0_1 err | k0_2 err | alpha_1 err | alpha_2 err | final cost | status |
|---|---:|---:|---:|---:|---:|---|

---

## Verify adjoint/Jacobian on the extended grid

Before trusting the TRF inverse, verify the Jacobian at the newly important voltages:

```text
V = +0.30, +0.40, +0.50, +0.60
```

Especially check:

```text
dCD/dalpha_2
dPC/dalpha_2
dCD/dlog_k0_2
dPC/dlog_k0_2
```

These are the sensitivities that supposedly re-enter after log-rate and voltage extension.

Use central finite differences against the adjoint Jacobian.

Required output:

```text
StudyResults/v19_lograte_extended_adjoint_check/
    adjoint_vs_fd.csv
    summary.md
```

Pass criterion:

```text
relative error < 1% for meaningful components
absolute error small for near-zero components
```

If `dCD/dalpha_2` at `V=+0.50/+0.60` is the key new direction, verify it directly.

---

## Widen `_U_CLAMP`, but compare before and after

The log-rate BV path no longer uses clamped `c_surf`, but `_U_CLAMP=30` is still active in the bulk PDE residual. This may or may not matter.

Run:

```text
_U_CLAMP = 30
_U_CLAMP = 60
_U_CLAMP = 100
```

with:

```text
bv_log_rate=True
V_GRID up to +0.60
cap=50
```

Compare:

```text
CD/PC curves
min/max u_H2O2
min/max c_H2O2
FIM
clean TRF endpoint if affordable
```

Required output:

```text
StudyResults/v19_lograte_uclamp_sweep/
    summary.md
    curves_by_clamp.csv
    fim_by_clamp.json
```

Interpretation:

- If results are unchanged, `_U_CLAMP=30` in the bulk residual is not contaminating the conclusion.
- If results change materially, the current FIM is partly clamp-dependent and must be reported with that caveat.

Also add a catastrophic guard rather than a new model clip:

```python
if abs(log_rate) > 120:
    mark_solve_unreliable
```

Do not silently hard-clip the log-rate again.

---

## Noisy inverse test — only after clean TRF passes

If clean-data TRF succeeds, run noisy synthetic tests.

Use at least:

```text
10 seeds at 2% noise
```

Prefer:

```text
20 seeds
```

Use the realistic noise model selected from the FIM audit. If possible, test both:

```text
global 2% max
local 2% + absolute floor
```

Required output:

```text
StudyResults/v19_lograte_extended_trf_noisy/
    seed_*.json
    summary.csv
    summary.md
```

Report:

```text
median error
IQR
failure rate
initialization sensitivity
correlation between failures and high-V signal/noise
```

Noisy-data success criteria:

```text
alpha_1, alpha_2 median error < 5-10%
k0_1, k0_2 median error < 20%
no strong pinning to initialization
failure cases explainable by noise floor / low high-V signal
```

Do not demand <1% k0 recovery under realistic noise. That is probably too strict.

---

## Stage 4 FV/Scharfetter-Gummel status

Do **not** start the FV/SG prototype now.

The log-rate breakthrough may be enough for the inverse-problem result. FV/SG remains valuable, but it is now a lower-priority methods branch.

Keep FV/SG in the backlog for one of these cases:

1. log-rate extended TRF fails clean data;
2. the FIM collapses under realistic noise floors;
3. the solver cannot stably handle the widened `_U_CLAMP`;
4. we want a separate numerical-methods paper on positivity-preserving high-overpotential PNP;
5. future systems require voltages beyond the log-rate CG1/Boltzmann window.

The original CG1 positivity issue is still real for full 4sp PNP. The log-rate fix solves the H2O2 boundary-rate pathology in the 3sp + log-c + Boltzmann model. It does not make CG1 generally positivity-preserving.

---

## Revised project storyline — current safe version

Do not yet claim:

> “We recover all four parameters from single-experiment data.”

That claim requires clean TRF plus noise-model confirmation.

Safe current claim:

> “The previous `k0_2` non-identifiability was not fundamental to the chemistry. It was caused in part by a numerical formulation that evaluated Butler–Volmer rates through clamped surface concentrations, creating an artificial H2O2 floor and preventing convergence into the voltage range where R2 contributes independent sensitivity. A log-rate BV evaluation extends the convergent voltage window to about `+0.60 V` and, in Fisher analysis, removes the old `k0_2`-dominated weak direction.”

If clean TRF succeeds and the noise-floor FIM survives, then the stronger claim becomes available:

> “A log-rate BV evaluation in a log-concentration PNP-BV solver makes four-parameter ORR kinetic inference possible from CD+PC data on an extended voltage grid, without Tikhonov priors.”

---

## Decision tree

### If noise-model FIM is robust and clean TRF succeeds

Proceed to noisy synthetic tests and write this up as the main inverse-problem breakthrough.

### If FIM is robust but clean TRF fails

The information is there, but optimizer/Jacobian conditioning remains an issue. Try:

```text
FIM eigenbasis parameterization
x_scale based on FIM/SVD
tighter solver tolerances
line-profile-guided diagnostics
```

Do not go to noisy data yet.

### If FIM collapses under realistic absolute noise

The high-V information is mathematically present but experimentally weak. Reframe:

```text
log-rate BV fixes the solver and exposes the missing sensitivity,
but practical k0_2 recovery requires lower-noise high-V measurements,
priors, or additional observables.
```

### If widening `_U_CLAMP` changes the result

The current extended-grid FIM is partly affected by bulk clamp inconsistency. Stabilize the formulation before inverse claims.

---

## Immediate task list

### Task 1 — archive unclipping correction

Document:

```text
V_unclip_2 = E_eq_2 - 50 * V_T ≈ +0.495 V
```

for the current `eta_scaled` clipping convention.

### Task 2 — noise-model FIM audit

Implement and run the FIM table described above.

### Task 3 — extended-grid adjoint check

Verify adjoint/FD sensitivities at:

```text
V = +0.30, +0.40, +0.50, +0.60
```

### Task 4 — clean-data TRF

Run the extended-grid inverse from multiple initial guesses.

### Task 5 — `_U_CLAMP` sweep

Compare `_U_CLAMP=30,60,100`.

### Task 6 — noisy TRF

Only if Tasks 2–4 pass.

---

## Suggested output directories

```text
StudyResults/v19_lograte_noise_model_fim/
StudyResults/v19_lograte_extended_adjoint_check/
StudyResults/v19_lograte_extended_trf_clean/
StudyResults/v19_lograte_uclamp_sweep/
StudyResults/v19_lograte_extended_trf_noisy/
```

---

## Final recommendation

The log-rate result is important enough to pause the FV/SG plan. The next work should be confirmation, not more solver architecture.

Priority order:

1. Noise-model FIM audit.
2. Extended-grid adjoint/Jacobian verification.
3. Clean-data TRF from multiple starts.
4. `_U_CLAMP` sweep.
5. Noisy TRF only after clean success.
6. FV/SG only if the above fails or if we decide to pursue a separate numerical-methods contribution.

The key question now is no longer “Can the forward solver be extended?” It appears it can, enough for the inverse problem. The key question is:

> Is the new high-voltage sensitivity experimentally and numerically strong enough to recover `k0_2` and `alpha_2` without priors?
