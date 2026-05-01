# PNP-BV Inverse Solver — Recommended Next Path

**Purpose:** This handoff summarizes the recommended strategy after the CD+PC/FIM/TRF diagnostics. The goal is to decide how to keep pushing toward reliable recovery of `k0_1`, `k0_2`, `alpha_1`, and `alpha_2`, without spending more time on optimizer tuning that cannot overcome the current conditioning limit.

---

## 1. Executive conclusion

The project has now moved past the question of whether peroxide current is computable from the 3-species model. It is computable, and the latest diagnostics show it is genuinely informative.

However, **single-experiment CD+PC is still too ill-conditioned to recover `k0` reliably**.

Current refined picture:

1. **CD-only is essentially singular** for the 4-parameter inverse problem.
2. **CD+PC adds real rank** and improves the smallest singular value dramatically.
3. But the joint CD+PC Fisher matrix remains badly conditioned, with the weak direction still dominated by `log_k0_2`.
4. TRF/least-squares with per-observable adjoint Jacobians recovers `alpha_1` and `alpha_2` reasonably well, but still lands on the `k0`–`alpha` ridge.
5. Therefore, the next step should not be more L-BFGS-B tuning or objective-weight tinkering.
6. The next step should be **multi-observable + multi-experiment design**, screened first by Fisher information.

Recommended strategy:

> Keep CD+PC, add physically different experiments that rotate or lift the weak Fisher direction. Start with multi-`L_ref` / rotation-rate variation, then add an H2O2-fed experiment to isolate R2 and improve `k0_2` identifiability.

---

## 2. What the latest diagnostics mean

The previous interpretation that peroxide current was redundant was too pessimistic. The FIM diagnostic showed that PC adds rank.

But the inverse is still numerically/practically ill-conditioned.

The most accurate current statement is:

> Disk current plus peroxide current makes the inverse problem formally identifiable in clean arithmetic, but the weakest parameter direction is so poorly conditioned that `k0` is not reliably recoverable from a single steady-state experiment with the current forward/Jacobian precision.

Use the phrase:

> **solver-noise-limited practical identifiability**

or, if mesh/time discretization is clearly the dominant source,

> **discretization-limited practical identifiability**.

Avoid saying:

> `k0` is mathematically unidentifiable from CD+PC.

The FIM says it is technically identifiable. The problem is that the useful `k0` information is too weak relative to numerical precision and optimizer conditioning.

---

## 3. What not to prioritize next

### 3.1 Do not spend major effort rescuing L-BFGS-B

TRF/least-squares with per-observable adjoint Jacobians already gave the cleaner result. L-BFGS-B may improve with scaling, but it is not the main bottleneck.

### 3.2 Do not try to fix `k0` by objective weighting alone

Changing weights may move the optimizer to a different point on the ridge, but it does not create new information. Weighting should reflect measurement noise or a clearly stated engineering choice.

### 3.3 Do not present Tikhonov as data-only recovery

Tikhonov can be useful, but only as a Bayesian/MAP prior. It should be reported as prior-assisted inference, not as proof that the data alone identify `k0`.

### 3.4 Do not make voltage extension past +0.30 V the near-term main path

Pushing the solver farther anodic may ultimately help, but the previous work suggests the obstruction is structural: singular Poisson perturbation plus positivity failure of the full 4-species CG1 formulation. This is high-effort solver research.

The near-term path should be experimental-design/FIM screening, not full-PNP rescue.

---

## 4. Main recommendation: multi-observable + multi-experiment

The best path is not simply “add more observables from the same experiment.” The best path is:

> **Keep CD+PC and add experimental perturbations that change the transport/kinetic balance.**

The two highest-value additions are:

1. **multi-`L_ref` / rotation-rate variation**
2. **H2O2-fed experiment to isolate R2**

These should be tested first by Fisher information, before expensive inverse runs.

---

## 5. Stage 1 — Fisher-screen candidate designs

Before running full inverse optimizations, build a design-screening script.

Parameter vector:

```python
theta = [log_k0_1, log_k0_2, alpha_1, alpha_2]
```

For each experiment and observable, compute whitened sensitivities:

```text
S[m, j] = d y_m / d theta_j / sigma_m
```

For a joint dataset, stack the sensitivity rows:

```text
S_joint = [
    S_CD_exp1,
    S_PC_exp1,
    S_CD_exp2,
    S_PC_exp2,
    ...
]
```

Then compute:

```text
F = S_joint.T @ S_joint
```

Report:

```text
sigma_min(S_joint)
condition(F)
condition(S_joint)
weak eigenvector of F
weak singular vector of S_joint, if convenient
```

### Success criterion

Do not run a full inverse unless the design improves the weak direction substantially.

Suggested threshold:

```text
sigma_min(S_joint) improves by at least 100x to 1000x
```

and/or

```text
condition(F) drops from ~1e11 to <= 1e8, preferably <= 1e7
```

Also check whether the weak eigenvector is still almost pure `log_k0_2`. If it is, the design has not solved the core problem.

---

## 6. Stage 2 — Multi-L_ref / rotation-rate experiment

First candidate design:

```text
L_ref values: L, 2L, 4L
observables: CD + PC at each L_ref
voltage grid: same working grid initially
```

Rationale:

Changing `L_ref` changes the transport limitation and surface concentrations without changing intrinsic kinetic parameters. That should rotate the `k0`–`alpha` ridge.

Do not use nearly identical `L_ref` values. If `L2 = 1.2L1`, the sensitivity directions may be too similar. Use a wide spread first.

### Designs to screen

Run FIM-only diagnostics for:

```text
A. baseline single L, CD only
B. baseline single L, CD + PC
C. two L values: L and 4L, CD + PC
D. three L values: L, 2L, 4L, CD + PC
```

Compare:

```text
sigma_min(S)
condition(F)
weak eigenvector
```

Expected local scaling:

For small perturbations in `L_ref`, the improvement in the weakest Fisher eigenvalue will often scale roughly like the square of the change in sensitivity direction. In practice, that means small changes in `L_ref` may help only weakly. Use spread-out transport conditions.

---

## 7. Stage 3 — Add an H2O2-fed experiment for R2

The persistent weak direction is still mostly `log_k0_2`. This makes sense: in the ORR experiment, H2O2 is generated by R1 and then consumed by R2, so R2 is downstream and weakly observed.

Add a synthetic H2O2-fed experiment:

```text
c_O2_bulk = 0 or very small
c_H2O2_bulk > 0
same 3sp + Boltzmann + log-c formulation
observable: CD initially; PC if meaningful/available
```

Purpose:

Make R2 directly visible instead of inferring it only from H2O2 produced by R1.

### Designs to screen

Run FIM diagnostics for:

```text
A. ORR CD+PC, single L
B. ORR CD+PC, L/2L/4L
C. H2O2-fed CD, single L
D. ORR CD+PC + H2O2-fed CD
E. ORR CD+PC at L/2L/4L + H2O2-fed CD
```

If the H2O2-fed condition lifts the `log_k0_2` weak direction, that is probably the cleanest route to `k0_2` recovery.

---

## 8. Stage 4 — Only after FIM improves: clean-data TRF

If the FIM improves enough, run full inverse tests.

Use:

```text
optimizer = scipy.optimize.least_squares(method="trf")
residuals = whitened CD and PC residuals
jacobian = per-observable adjoint Jacobian
params = [log_k0_1, log_k0_2, alpha_1, alpha_2]
```

Initial guesses:

```text
+20% all parameters
-20% all parameters
mixed-sign offsets, e.g. k0 high / alpha low
```

Pass criteria for clean data:

```text
alpha_1, alpha_2 within ~5%
k0_1, k0_2 within ~10-20%
converges to similar solution from multiple starts
final cost close to TRUE discretization floor
```

Do not expect <1% `k0` immediately. The first win is getting `k0` to move consistently toward truth rather than staying pinned to initialization or sliding freely along the ridge.

---

## 9. Stage 5 — Noisy synthetic tests

After clean-data success, run 2% noise tests.

Recommended:

```text
10-20 noise seeds
same multi-experiment design
same optimizer/Jacobian setup
```

Report:

```text
median error
90% interval
failure rate
bias vs variance
correlation between recovered log_k0 and alpha
```

Expected realistic success:

```text
alpha stable within 5-10%
k0 no longer pinned to initialization
k0 confidence intervals finite and honest
```

A good result does not require <1% `k0` recovery under 2% noise. It requires demonstrating that the added experiments materially reduce ridge sensitivity.

---

## 10. Tikhonov / Bayesian prior path

Tikhonov should be kept as a secondary path, not the main proof.

If used, express it as a prior:

```text
J_prior = ((log_k0 - log_k0_prior) / sigma_log_k0)^2
```

Recommended prior widths:

```text
sigma_log_k0 = log(3)     # factor-of-3 prior
sigma_log_k0 = log(10)    # factor-of-10 prior
```

Run prior stress tests:

```text
prior centered at true
prior centered at 0.3x true
prior centered at 3x true
factor-3 width
factor-10 width
```

Interpretation:

If the recovered `k0` follows the prior, say so. That means the data are not sufficiently informative. It is not a failure; it is the correct uncertainty statement.

---

## 11. Optional solver-precision diagnostic

This is useful, but should not dominate the next phase.

Run one “gold precision” clean-data TRF case:

```text
same single-experiment CD+PC setup
tighter SNES tolerances
stricter Newton residual tolerance
consistent warm-start/cache path
maybe finer mesh if feasible
```

Purpose:

Determine whether `k0` recovery changes substantially when forward precision improves.

Interpretation:

- If `k0` moves much closer to truth, the inverse is strongly solver-noise-limited.
- If `k0` remains wrong, the ridge is too flat even with tighter solves.

This is a diagnostic, not the main recovery strategy.

---

## 12. Optional FIM eigenbasis optimization

This is also useful diagnostically.

At TRUE or near the initial guess, compute:

```text
F = Q Lambda Q.T
```

Then optimize in transformed coordinates:

```text
theta = theta0 + Q z
```

or use locally whitened coordinates:

```text
theta = theta0 + Q Lambda^{-1/2} z
```

Purpose:

Make the strong and weak directions explicit. This probably will not magically recover `k0`, but it will show which coordinate stalls and will make the identifiability story cleaner.

---

## 13. Possible later observables

### 13.1 Transient current

Voltage-step chronoamperometry could help because it adds timescale information. Steady-state curves mostly encode algebraic balances; transient relaxation can help separate kinetics from transport.

This is promising but higher effort because it requires reliable time-dependent adjoints or many forward solves.

### 13.2 EIS

EIS could also help because frequency-domain response can separate charge transfer, diffusion, and double-layer effects.

However, this is a new forward model and a more complicated inverse problem. It should come after the multi-`L_ref` and H2O2-fed steady-state tests.

### 13.3 Temperature

Temperature changes `V_T = RT/F` and may help disentangle BV parameters, but `k0` itself is temperature-dependent. This adds Arrhenius modeling and extra assumptions. Deprioritize for now.

### 13.4 pH

pH changes H+ availability and could add information, but it may also change mechanism, activity coefficients, and physical interpretation. Promising later, not the cleanest immediate test.

---

## 14. Best immediate task list for Claude Code

### Task 1 — Multi-design FIM script

Create a script, e.g.

```text
scripts/studies/v18_logc_multiexperiment_fim.py
```

Inputs:

```text
experiment_specs = [
    {"name": "ORR_L", "L_scale": 1.0, "c_O2": 0.5, "c_H2O2": 0.0, "observables": ["cd", "pc"]},
    {"name": "ORR_2L", "L_scale": 2.0, "c_O2": 0.5, "c_H2O2": 0.0, "observables": ["cd", "pc"]},
    {"name": "ORR_4L", "L_scale": 4.0, "c_O2": 0.5, "c_H2O2": 0.0, "observables": ["cd", "pc"]},
    {"name": "H2O2_fed", "L_scale": 1.0, "c_O2": 0.0, "c_H2O2": some_positive_value, "observables": ["cd"]},
]
```

Outputs:

```text
StudyResults/v18_logc_multiexperiment_fim/results.json
StudyResults/v18_logc_multiexperiment_fim/summary.md
```

For each design, report:

```text
sigma_min(S)
sigma_max(S)
condition(S)
condition(F)
weak eigenvector
parameter correlations
```

### Task 2 — Design comparison table

Generate a summary table for:

```text
CD only, single L
CD+PC, single L
CD+PC, L and 4L
CD+PC, L/2L/4L
H2O2-fed only
CD+PC single L + H2O2-fed
CD+PC L/2L/4L + H2O2-fed
```

### Task 3 — Clean TRF only for promising designs

If FIM improves enough, create:

```text
scripts/studies/v18_logc_multiexperiment_lsq_inverse.py
```

Use TRF, whitened residuals, per-observable adjoint Jacobians.

Run clean data from multiple starts.

### Task 4 — Noise seeds only after clean success

Run 2% noise only after clean-data recovery is acceptable.

---

## 15. Recommended project framing

The emerging publishable story may be:

> A PNP-BV inverse model can robustly recover transfer coefficients from disk and peroxide observables, but exchange-rate constants are practically unidentifiable from a single steady-state experiment because the relevant Fisher direction is far too ill-conditioned. Multi-experiment design or physically justified priors are required to recover `k0`.

If multi-`L_ref` and H2O2-fed experiments work, the story becomes stronger:

> Adding observables alone is not enough; adding physically distinct experiments that rotate the sensitivity ridge makes `k0` recoverable.

That is the path most likely to make the project work.

