# PNP-BV Inverse Solver — Peroxide Negative Result and Next Steps

**Audience:** Claude Code / implementation agent  
**Purpose:** Translate the peroxide-current negative result into concrete next coding steps.  
**Core message:** Do **not** keep trying to force disk current + peroxide current alone to recover `k0`. The result shows that peroxide is computable from the 3-species model, but in the accessible voltage window it is almost redundant with disk current. The next useful work is sensitivity/Fisher analysis and multi-experiment design.

---

## 1. Executive conclusion

The peroxide-current second-observable experiment was implemented correctly enough to be informative.

The 3-species + Boltzmann ClO4- model still contains the dynamic species:

```text
O2, H2O2, H+
```

and still contains both reactions:

```text
R1: O2 + 2H+ + 2e- -> H2O2
R2: H2O2 + 2H+ + 2e- -> 2H2O
```

Therefore peroxide current can be computed from the BV boundary rates:

```text
peroxide molar production = r1 - r2
peroxide current          = -2 F (r1 - r2)       # depending on sign convention
```

The problem is not that peroxide current is unavailable. The problem is that, in the current voltage window, peroxide current does not add enough independent information to identify `k0_1` or `k0_2`.

The clean-data result confirms this:

| Parameter | Init | Recovered | Interpretation |
|---|---:|---:|---|
| `k0_1` | +20% | +20.52% | essentially stuck at init |
| `k0_2` | +20% | +18.25% | essentially stuck at init |
| `alpha_1` | +20% | -5.95% | optimizer moves alpha along ridge |
| `alpha_2` | +20% | -4.68% | optimizer moves alpha along ridge |

Even with clean data, where the true parameters give `J = 0`, the optimizer reduced the loss by moving mostly in `alpha`, not `k0`. That is strong evidence of a practical ridge/conditioning problem.

---

## 2. Mechanistic explanation

R2 has:

```text
E_eq,2 = 1.78 V vs RHE
```

The tested voltage window was approximately:

```text
V_RHE = [-0.10, 0.00, +0.10, +0.15, +0.20] V
```

So the R2 overpotential is:

```text
eta_2 = V_RHE - E_eq,2
      ≈ -1.88 to -1.58 V
```

For the cathodic BV exponential, with `n_e = 2` and `V_T = 25.69 mV`, this exponent is enormous in magnitude. With the existing exponent clip at `±50`, R2 is clipped throughout the whole voltage window:

```text
exp(cathodic exponent) = exp(50)
```

Thus, in the whole fitted voltage window, R2 behaves approximately like:

```text
r2 ≈ k0_2 * exp(50) * (c_H_surf / c_ref)^2 * c_H2O2_surf
```

Consequences:

1. `r2` has little or no useful voltage-shape information.
2. `alpha_2` is largely hidden by the exponent clip.
3. `r2` is slaved to `c_H2O2_surf`, which is mostly determined by R1 production.
4. Peroxide current,

```text
PC = -2F(r1 - r2)
```

is approximately a second copy of R1 information when `r2 << r1` or when R2 has no independent voltage shape.

Therefore disk current and peroxide current are not independent enough in this window:

```text
CD = -2F(r1 + r2)
PC = -2F(r1 - r2)
```

In principle these are complementary. In this operating regime they are nearly redundant.

---

## 3. Important correction

If you write this up elsewhere, do **not** say that accessing “less anodic R2 conditions” would help.

To move R2 closer to its equilibrium potential, the model would need to access **more anodic** potentials, closer to:

```text
V_RHE ≈ E_eq,2 = 1.78 V
```

The current solver cannot reach anything close to that. Extending from `+0.20` to `+0.30 V` may help R1 curvature slightly, but it will not put R2 near its kinetic regime.

---

## 4. Do not spend more time on these paths yet

Do **not** keep tuning objective weights in the disk+peroxide inverse problem hoping that this alone will recover `k0`.

Do **not** interpret Tikhonov recovery with a true prior as data-only identifiability.

Do **not** assume the peroxide observable failed because the formula was wrong. The formula is fine. The operating window makes it weak.

Do **not** make the full 4-species CG1 solver the main inference path right now. The ClO4- positivity failure is structural, not a damping issue.

---

## 5. Immediate next task: sensitivity and Fisher-rank diagnostics

Before running more inverse optimizations, quantify whether peroxide adds rank.

Use the parameter vector:

```text
theta = [log_k0_1, log_k0_2, alpha_1, alpha_2]
```

Use observables:

```text
y = [CD(V_1), ..., CD(V_N), PC(V_1), ..., PC(V_N)]
```

Build the sensitivity matrix:

```text
S[m, j] = d y_m / d theta_j
```

Prefer normalized/whitened sensitivities:

```text
S_white[m, j] = (1 / sigma_m) * d y_m / d theta_j
```

If true measurement variances are not known, also compute range-normalized diagnostics, but label them as numerical diagnostics, not statistical Fisher information.

Then compute:

```text
F = S_white.T @ S_white
```

Report:

1. singular values of `S_white`;
2. eigenvalues of `F`;
3. condition number of `F`;
4. smallest eigenvector(s) of `F`;
5. correlation between CD and PC sensitivity rows/columns;
6. rank comparison for:
   - CD only;
   - PC only;
   - CD + PC.

### Expected result

If the negative-result interpretation is right, then adding PC will not materially increase the smallest singular value. The weakest eigenvector should still be dominated by combinations like:

```text
log_k0_1 <-> alpha_1
log_k0_2 <-> alpha_2
```

or a coupled version of those ridge directions.

---

## 6. Suggested script: `v18_logc_sensitivity_fim.py`

Create a new script:

```text
scripts/studies/v18_logc_sensitivity_fim.py
```

It should reuse the existing working forward model:

```text
forms_logc.py
3 dynamic species: O2, H2O2, H+
Boltzmann ClO4- background
H2O2 seed = 1e-4
V_GRID = [-0.10, 0.00, 0.10, 0.15, 0.20]
```

### Inputs

- `theta_true`
- optional `theta_eval`, default `theta_true`
- voltage grid
- observable mode: `cd`, `pc`, or `both`
- normalization mode:
  - `none`
  - `range`
  - `noise`

### Outputs

Write JSON and CSV to:

```text
StudyResults/v18_logc_sensitivity_fim/
```

Suggested files:

```text
sensitivity_matrix_cd.csv
sensitivity_matrix_pc.csv
sensitivity_matrix_both.csv
fim_cd.json
fim_pc.json
fim_both.json
singular_values.json
eigenvectors.json
summary.md
```

### Core pseudocode

```python
PARAMS = ["log_k0_1", "log_k0_2", "alpha_1", "alpha_2"]
V_GRID = [-0.10, 0.00, 0.10, 0.15, 0.20]

# y_cd, y_pc shape: (N_V,)
y_cd, y_pc = forward_observables(theta_eval, V_GRID)

# Build sensitivities either by adjoint gradients per observable
# or finite differences initially if easier.
S_cd = zeros((N_V, 4))
S_pc = zeros((N_V, 4))

for k, V in enumerate(V_GRID):
    for j, p in enumerate(PARAMS):
        S_cd[k, j] = d_cd_dtheta(V, theta_eval, p)
        S_pc[k, j] = d_pc_dtheta(V, theta_eval, p)

S_both = vstack([S_cd, S_pc])

S_cd_w = normalize(S_cd, observable="cd")
S_pc_w = normalize(S_pc, observable="pc")
S_both_w = normalize(S_both, observable="both")

for name, S in [("cd", S_cd_w), ("pc", S_pc_w), ("both", S_both_w)]:
    U, svals, VT = np.linalg.svd(S, full_matrices=False)
    F = S.T @ S
    evals, evecs = np.linalg.eigh(F)
    save_results(name, svals, evals, evecs, condition_number=evals[-1]/evals[0])
```

### Adjoint vs finite difference

Adjoint is preferred once convenient, but finite differences are acceptable for the first diagnostic because this is not an optimizer loop.

Use central differences in the transformed parameter space:

```text
log_k0_j step: 1e-3 to 1e-2
alpha_j step: 1e-4 to 1e-3
```

Check step-size stability. Save at least two step sizes.

---

## 7. Second task: line/profile checks on the clean-data endpoint

The clean-data optimizer stopped at nonzero `J` even though TRUE gives `J = 0`. Before calling the recovered point a genuine local minimum, diagnose the geometry.

Create or add a script:

```text
scripts/studies/v18_logc_joint_profile_checks.py
```

### Required checks

Given:

```text
theta_true
theta_recovered_clean
```

Evaluate:

```text
theta(t) = (1 - t) * theta_recovered_clean + t * theta_true
```

for:

```text
t in [0.0, 0.05, 0.10, ..., 1.0]
```

Save:

```text
J(t), J_cd(t), J_pc(t)
```

Interpretation:

- If `J(t)` decreases monotonically toward TRUE, the endpoint was probably optimizer/conditioning failure, not a true local minimum.
- If `J(t)` rises before falling, there may be a real local basin or nonconvex geometry.

Also compute the gradient norm at the recovered point:

```text
||grad J||
projected gradient norm if bounds are active
```

Save all results to:

```text
StudyResults/v18_logc_joint_profile_checks/
```

---

## 8. Third task: multi-experiment synthetic design

The best next non-prior path is multi-experiment fitting.

The idea is to vary transport conditions so that kinetic and transport effects shift differently. The simplest model-side knob is:

```text
L_ref
```

standing in for rotation-rate / diffusion-layer thickness variation.

### Minimum synthetic test

Use three `L_ref` values, for example:

```text
L_ref_set = [0.5 * L0, 1.0 * L0, 2.0 * L0]
```

or, if numerical stability permits:

```text
L_ref_set = [0.5 * L0, 1.0 * L0, 4.0 * L0]
```

Do not start with two experiments only unless runtime is a problem. Three gives a cleaner FIM test.

### Joint observable vector

For each experiment `e` and voltage `V_k`:

```text
y[e, k] = CD_e(V_k)
```

Optionally add peroxide current after the CD-only multi-L test:

```text
y[e, k, obs] = [CD_e(V_k), PC_e(V_k)]
```

### Compare FIMs

Run FIM diagnostics for:

1. single `L_ref`, CD only;
2. single `L_ref`, CD + PC;
3. three `L_ref`, CD only;
4. three `L_ref`, CD + PC.

Key metric:

```text
smallest eigenvalue of F
condition number of F
smallest eigenvector composition
```

If three `L_ref` values increase the smallest eigenvalue substantially and rotate the weak eigenvector away from pure `log_k0`/`alpha` compensation, then multi-experiment fitting is worth implementing fully.

---

## 9. Fourth task: H2O2-fed experiment for R2

For `k0_2`, the current ORR setup is weak because bulk H2O2 is zero and R2 is only observed through H2O2 generated by R1.

Design a synthetic R2-focused experiment:

```text
c_O2_bulk    = 0 or very small
c_H2O2_bulk  > 0
c_H_bulk     same as baseline
ClO4 handled by Boltzmann background
```

Candidate values:

```text
c_H2O2_bulk = 0.05, 0.1, 0.5 mol/m^3
```

Start with one nonzero H2O2 level, then test multiple levels if useful.

Goal:

- isolate R2 more directly;
- test whether `k0_2` becomes visible in FIM;
- compare against ORR-only data.

FIM comparison:

1. ORR experiment only;
2. H2O2-fed experiment only;
3. ORR + H2O2-fed jointly;
4. ORR + H2O2-fed + multi-L.

---

## 10. Tikhonov / Bayesian prior guidance

Regularization is still useful, but only if framed honestly.

Do **not** say:

```text
Tikhonov proves the data recover k0.
```

Say:

```text
The data have a ridge. A physically justified prior on log(k0) selects a point on that ridge and yields a MAP estimate. Prior sensitivity must be reported.
```

Use a noise-normalized objective plus log-prior:

```text
J = sum_m ((y_sim_m - y_obs_m) / sigma_m)^2
    + sum_j ((log_k0_j - log_k0_prior_j) / sigma_log_k0_j)^2
```

Recommended prior-width tests:

| Prior center | Prior width | Purpose |
|---|---:|---|
| true `k0` | factor 3 | best-case prior |
| `0.3 * true` | factor 3 | wrong-low prior |
| `3.0 * true` | factor 3 | wrong-high prior |
| true `k0` | factor 10 | weak prior |
| wrong prior | factor 10 | stress test |

Use:

```text
sigma_log_k0 = log(3)    # factor-of-3 prior
sigma_log_k0 = log(10)   # factor-of-10 prior
```

This is preferable to an arbitrary `lambda = 0.01`, unless that lambda is derived from normalized residuals and prior variance.

---

## 11. Objective weighting guidance

The peroxide run switched from inverse-variance weighting to range-normalized weighting because the PC noise scale made gradients explode.

For numerical optimization, scaling is allowed. But for inference, do not change the statistical meaning of the objective accidentally.

Preferred approach:

1. Use physical/noise-based residual whitening:

```text
r_m = (y_sim_m - y_obs_m) / sigma_m
```

2. If gradients are too large, multiply the **entire objective** by a constant:

```text
J_scaled = c * J
```

This does not change the minimizer.

3. Also scale parameters internally if needed.

Do not replace noise whitening with range normalization unless the result is explicitly labeled as a numerical conditioning experiment.

---

## 12. Possible dimensionally different observables

If steady-state I-V remains under-informative, consider observables with time or frequency scales.

Potential additions:

1. chronoamperometry after voltage steps;
2. transient H2O2 production after voltage steps;
3. EIS / small-signal impedance at selected biases;
4. H2O2-fed chronoamperometry;
5. temperature variation.

These may help because `k0` affects kinetic timescales, not just steady-state amplitudes.

Do not implement these before the FIM/multi-experiment tests unless requested. They are higher-effort.

---

## 13. Recommended execution order

### Phase A: Diagnostics, low effort

1. Implement `v18_logc_sensitivity_fim.py`.
2. Compute CD-only, PC-only, and CD+PC sensitivities.
3. Report singular values/eigenvectors.
4. Run line/profile check from clean recovered point to TRUE.

Expected time: hours, not days.

### Phase B: Synthetic multi-experiment, medium effort

5. Add multi-`L_ref` forward loop.
6. Compute FIM for one vs three `L_ref` values.
7. Only if FIM improves, run inverse optimization.

Expected time: 1-2 days if the solver parameterization is clean; longer if `L_ref` is hard-coded.

### Phase C: R2-focused experiment

8. Add H2O2-fed synthetic experiment.
9. Compute FIM for ORR-only vs H2O2-fed vs joint.
10. If promising, run inverse optimization.

### Phase D: Priors

11. Add Bayesian/log-prior objective.
12. Run prior-sensitivity tests.
13. Report posterior/MAP dependence on prior width and prior center.

---

## 14. Short instruction to Claude Code

Implement diagnostics before more optimization.

Specifically:

```text
1. Stop trying to make disk+peroxide alone recover k0.
2. Build sensitivity/FIM diagnostics for CD, PC, and CD+PC.
3. Check whether PC adds rank or only duplicates CD information.
4. Profile the clean-data recovered point along the line to TRUE.
5. Build a three-L_ref synthetic experiment and compare FIM eigenvalues.
6. Add an H2O2-fed synthetic experiment to isolate R2.
7. Treat Tikhonov only as a Bayesian prior and run wrong-prior stress tests.
```

The most useful next result is not another recovered-parameter table. It is a table of Fisher eigenvalues and weakest eigenvectors showing which experiments actually break the ridge.

