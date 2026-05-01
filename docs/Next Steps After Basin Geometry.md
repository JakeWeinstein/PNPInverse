# HANDOFF_11 — PNP-BV inverse solver next steps after HANDOFF_10

## Copy/paste instruction for Claude Code

```text
Read HANDOFF_10 as the current state. Do not re-run LM, L-BFGS-B tuning, or log(k0) Tikhonov at sigma=log(3)/log(10). Those routes have already shown that the single-experiment CD+PC inverse has multi-basin Tafel-ridge geometry that ordinary optimizers and literature-defensible priors do not fix. Implement the next path in this order: (1) cheap closure diagnostics on the single-experiment basin geometry, (2) anchored Tafel-coordinate parameterization, (3) multi-experiment FIM design screening with bulk O2 variation first, H2O2-fed R2 isolation second, and L_ref/rotation variation third, and (4) full clean-data inverse only for designs whose FIM actually rotates/lifts the remaining weak log_k0 direction. Do not run noisy seeds until clean-data multi-experiment recovery works from multiple initializations.
```

---

## 1. Current state from HANDOFF_10

The project is no longer blocked by the old `log_k0_2` Fisher singularity. The log-rate Butler-Volmer formulation extended the usable voltage grid to roughly:

```text
V_GRID = [-0.10, +0.10, +0.20, +0.30, +0.40, +0.50, +0.60]  # G0
bv_log_rate = True
observables = CD + PC
parameters = [log_k0_1, log_k0_2, alpha_1, alpha_2]
```

The local FIM at TRUE is much better than the old clipped/low-voltage model:

```text
cond(F) ≈ 1.79e7
ridge_cos ≈ 0.031
weak eigvec ≈ almost-pure log_k0_1, not old log_k0_2
```

But HANDOFF_10 showed that the single-experiment global inverse is still not solved:

```text
LM on G0 is worse than TRF on every init.
Tikhonov sigma=log(3) and sigma=log(10) shift within a basin but do not cross basin barriers.
A prior strong enough to force basin crossing would need roughly sigma <= log(1.15), i.e. factor-of-15% k0 prior, which is not defensible as a generic literature/EIS prior.
alpha_1 and alpha_2 are robustly data-identifiable in many inits.
k0_1 and k0_2 are individually reachable from some inits but not jointly recoverable from one CD+PC experiment.
```

The correct current conclusion is:

```text
Single-experiment CD+PC with log-rate BV has useful local information, but the global cost surface contains multiple Tafel-ridge basins. alpha is robust; joint k0 recovery is not reliable without either unrealistically tight prior information or additional experiments/observables.
```

---

## 2. What not to do next

Do **not** spend time on these as main paths:

```text
more LM
more L-BFGS-B tuning
more objective reweighting
more log(k0) Tikhonov at sigma=log(3) or log(10)
noisy seeds on the single-experiment inverse
positive-voltage extension beyond +0.60 as the main route
more negative/low voltages as the main route
FV/SG solver rewrite for this immediate inverse question
```

Reason:

```text
The bottleneck is not a missing gradient, adjoint error, or ordinary optimizer choice. It is global basin geometry in the single-experiment steady-state objective.
```

Positive voltages beyond +0.60 may be useful later, but they mainly probe R2. The current weak direction is mostly `log_k0_1`, so higher positive voltage is not the most targeted next lever.

---

## 3. Phase 0 — Cheap closure diagnostics for the single-experiment basin

These are not expected to solve the problem. They are to close the single-experiment story cleanly.

### 0.1 Restart-with-perturbation from the best wrong basin

Use the best low-cost wrong basin from HANDOFF_10, especially the minus20/log3 or minus20/no-prior endpoint where approximately:

```text
k0_1 is close to TRUE
k0_2 is about +54% to +59% high
alpha_1 and alpha_2 are close to TRUE
```

Perturb `log_k0_2` only:

```text
perturbations = [-0.75, -0.50, -0.25, +0.25, +0.50, +0.75]
```

For each perturbation:

```text
start theta = recovered_wrong_basin_theta
start theta[log_k0_2] += perturbation
run bounded TRF on G0, no prior
save endpoint, cost, parameter errors, convergence status
```

Suggested output:

```text
StudyResults/v23_restart_perturbation/
    summary.md
    restart_results.csv
    restart_results.json
```

Interpretation:

```text
If all runs return to the same k0_2-high basin:
    basin barrier confirmed; stop trying restart tricks.

If some runs escape to TRUE/all-4 recovery:
    implement restart-with-perturbation as a practical multi-start diagnostic,
    but still do not claim single-experiment structural identifiability without explaining basin dependence.
```

### 0.2 Small basin-of-attraction map around TRUE

Use a small, finite design rather than a huge sampling study.

Sample initial guesses at controlled radii:

```text
log_k0 radius: [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]  # fractional/log equivalent OK
alpha radius:  [0.02, 0.05, 0.10, 0.20]
```

Use 5-10 Latin-hypercube starts per radius pair, or a smaller structured grid if runtime is high.

Suggested output:

```text
StudyResults/v23_single_experiment_basin_map/
    basin_map.csv
    summary.md
```

Report:

```text
initial distance from TRUE
converged basin label
final cost
parameter errors
whether all 4 params <5%, <10%, <20%
whether alpha recovered but k0 failed
```

Main question:

```text
How close must the initial guess/prior already be before clean-data TRF recovers all four parameters?
```

If recovery requires very close starts, that supports the HANDOFF_10 conclusion that the single-experiment problem is locally informative but globally multi-basin.

---

## 4. Phase 1 — Anchored Tafel-coordinate parameterization

This is the one optimizer-side idea still worth testing because it changes the parameter coordinates to match what the data actually sees.

### 4.1 Motivation

The steady-state data mostly sees effective BV rate combinations of the form:

```text
log_rate_j(V) ≈ log_k0_j - alpha_j * n_e * (V - E_eq_j) / V_T + concentration terms
```

`log_k0_j` is an extrapolated intercept far from much of the measured voltage information. This makes `log_k0` and `alpha` highly compensatory.

Instead of optimizing directly in:

```text
[log_k0_1, log_k0_2, alpha_1, alpha_2]
```

optimize in anchored kinetic coordinates:

```text
beta_j = log_k0_j - alpha_j * n_e * (V_anchor_j - E_eq_j) / V_T
```

Then recover:

```text
log_k0_j = beta_j + alpha_j * n_e * (V_anchor_j - E_eq_j) / V_T
```

### 4.2 Suggested anchors

Use anchors near voltages where each reaction actually contributes information:

```text
V_anchor_1 = +0.25 or +0.30   # R1 transition/tail region
V_anchor_2 = +0.50 or +0.55   # R2 informative/unclipped region
```

Test a small anchor grid:

```text
(V_anchor_1, V_anchor_2) in [
    (+0.25, +0.50),
    (+0.30, +0.50),
    (+0.30, +0.55),
]
```

### 4.3 Implementation

Create or extend a script:

```text
scripts/studies/v23_anchored_tafel_lsq_inverse.py
```

Internally optimize:

```text
x = [beta_1, beta_2, alpha_1, alpha_2]
```

but convert to physical parameters before each forward solve:

```text
log_k0_1 = beta_1 + alpha_1 * n_e1 * (V_anchor_1 - E_eq_1) / V_T
log_k0_2 = beta_2 + alpha_2 * n_e2 * (V_anchor_2 - E_eq_2) / V_T
```

Keep the same physical bounds by transforming them appropriately or rejecting impossible mapped values:

```text
0 < alpha_j <= 1
|log_k0_j - log_k0_TRUE_j| <= 2  # same TRF-style physical guard/bound as before
```

Run the same four initializations used in HANDOFF_10:

```text
plus20
minus20
k0high_alow
k0low_ahigh
```

Suggested output:

```text
StudyResults/v23_anchored_tafel_parameterization/
    summary.md
    by_anchor_pair.csv
    endpoints.json
```

### 4.4 Interpretation

```text
If beta_1, beta_2, alpha_1, alpha_2 recover consistently but transformed k0 does not:
    The data identifies effective rates in the observed voltage window, not formal exchange rates k0.
    This is scientifically useful and should be reported.

If transformed k0 also stabilizes across inits:
    The problem was partly bad coordinates, and anchored Tafel variables become the preferred inverse parameterization.

If neither beta nor transformed k0 stabilizes:
    Close the optimizer/parameterization route and move to multi-experiment design.
```

---

## 5. Phase 2 — Multi-experiment FIM design screening

This is the main next path. Do FIM screening before expensive inverse runs.

### 5.1 Baseline setup

Use the validated log-rate solver:

```text
bv_log_rate = True
V_GRID = [-0.10, +0.10, +0.20, +0.30, +0.40, +0.50, +0.60]
observables = CD + PC where meaningful
parameters = [log_k0_1, log_k0_2, alpha_1, alpha_2]
```

Compute whitened sensitivities:

```text
S[m, j] = d y_m / d theta_j / sigma_m
F = S.T @ S
```

Use the same noise models already used in the log-rate work, especially:

```text
A. global 2% max
B. local 2% relative
C. local 2% plus absolute floor
```

At minimum, use the current standard global 2% max model so results compare directly to HANDOFF_10.

### 5.2 Required FIM outputs

For each candidate design, report:

```text
sigma_min(S)
sigma_max(S)
condition(S)
condition(F)
weak eigenvector of F
right singular vector of S for sigma_min, if convenient
parameter correlation matrix or covariance proxy
predicted standard deviations for log_k0_1, log_k0_2, alpha_1, alpha_2
```

Suggested output directory:

```text
StudyResults/v23_multiexperiment_fim/
    summary.md
    design_table.csv
    fim_by_design.json
    weak_vectors.csv
    covariance_by_design.json
```

### 5.3 Success criteria before running inverse

Do **not** run a full multi-experiment inverse unless the FIM changes the weak direction meaningfully.

A design is promising if it does several of these:

```text
weak eigenvector no longer has |log_k0_1| >= 0.95
condition(F) improves materially from the single-experiment baseline
sigma_min(S) improves materially
predicted uncertainty in log_k0_1 and log_k0_2 both decrease
correlations between log_k0_j and alpha_j are reduced
```

Do not use condition number alone. HANDOFF_10 showed that the local FIM can look acceptable while global basin geometry remains difficult. The weak-vector composition and parameter correlations matter.

---

## 6. Phase 2A — Bulk O2 variation first

This should be the first multi-experiment design because the current weak direction after log-rate is mostly `log_k0_1`.

### 6.1 Rationale

R1 depends directly on O2 availability. Changing bulk O2 changes the R1 transport/kinetic balance without changing intrinsic kinetic parameters. That is the most targeted way to attack the remaining `log_k0_1` ambiguity.

### 6.2 Candidate experiments

Start with:

```text
ORR_O2_low:   c_O2_bulk = 0.25 mol/m^3, c_H2O2_bulk = 0
ORR_O2_base:  c_O2_bulk = 0.50 mol/m^3, c_H2O2_bulk = 0
ORR_O2_high:  c_O2_bulk = 1.00 mol/m^3, c_H2O2_bulk = 0
```

Keep pH/H+ and supporting electrolyte fixed initially.

If convergence is robust, optionally expand to:

```text
c_O2_bulk = [0.125, 0.25, 0.50, 1.00]
```

But do not over-expand before seeing whether the basic three-level design rotates the weak direction.

### 6.3 Designs to compare

```text
A. baseline single O2, CD+PC
B. low + base O2, CD+PC
C. base + high O2, CD+PC
D. low + base + high O2, CD+PC
```

Pass condition:

```text
The weak eigenvector should stop being almost pure log_k0_1.
```

If O2 variation does not rotate `log_k0_1`, that is a serious sign that steady-state ORR CD+PC may not be enough.

---

## 7. Phase 2B — H2O2-fed R2 isolation second

This is still valuable, but it should come after O2 variation because the current weak direction is no longer old `log_k0_2`.

### 7.1 Candidate experiments

Use:

```text
c_O2_bulk = 0 or very small
c_H2O2_bulk = [0.05, 0.10, 0.50] mol/m^3
same pH/H+
same 3sp + Boltzmann + log-c formulation
```

Start with one level:

```text
c_H2O2_bulk = 0.10 mol/m^3
```

Then add more levels only if the first one converges and provides distinct sensitivity.

### 7.2 Observables

Use CD. Use PC only if the sign convention and physical meaning are clear in the H2O2-fed configuration.

Compare:

```text
A. ORR baseline only
B. H2O2-fed only
C. ORR baseline + H2O2-fed
D. ORR O2-variation + H2O2-fed
```

Expected benefit:

```text
H2O2-fed data should directly constrain R2 rather than observing R2 only downstream of R1-generated peroxide.
```

---

## 8. Phase 2C — L_ref / rotation variation third

This has already been considered, but in the new HANDOFF_10 state it should be secondary to O2 variation.

Candidate designs:

```text
L_ref scales = [0.5, 1.0, 2.0, 4.0]
```

Do not use nearly identical values. Small changes in L_ref may not rotate the sensitivity enough.

Compare:

```text
A. baseline L only
B. L and 4L
C. 0.5L, L, 2L, 4L
D. O2 variation + L variation
E. O2 variation + H2O2-fed + L variation
```

Only keep L_ref in the main path if it adds distinct sensitivity beyond O2/H2O2 concentration variation.

---

## 9. Phase 3 — Full clean-data inverse only for promising designs

If and only if Phase 2 FIM screening improves the weak direction, create:

```text
scripts/studies/v23_multiexperiment_lsq_inverse.py
```

Use:

```text
optimizer = scipy.optimize.least_squares(method="trf")
residuals = whitened residuals stacked across experiments
jacobian = per-observable adjoint Jacobian stacked across experiments
parameters = [log_k0_1, log_k0_2, alpha_1, alpha_2]
```

Also test the anchored Tafel coordinates if Phase 1 showed any benefit:

```text
parameters = [beta_1, beta_2, alpha_1, alpha_2]
```

Run clean data from:

```text
plus20
minus20
k0high_alow
k0low_ahigh
additional mixed starts if cheap
```

Pass criteria:

```text
alpha_1 and alpha_2 within ~5%
k0_1 and k0_2 within ~10-20%
most starts converge to the same basin
final cost near TRUE discretization floor
no obvious monotone downhill path from recovered endpoint to TRUE
```

Do not demand <1% k0 recovery. The first win is no longer being pinned to initialization or selecting different k0 basins from different starts.

---

## 10. Phase 4 — Noisy synthetic tests only after clean success

Only after clean multi-experiment inverse succeeds:

```text
10-20 noise seeds
2% noise
selected realistic absolute floor if applicable
same experiment design as clean success
same optimizer and parameterization
```

Report:

```text
median parameter error
IQR or 90% interval
failure rate
bias vs variance
correlation between recovered log_k0 and alpha
basin-switching frequency
```

Expected realistic success:

```text
alpha stable within 5-10%
k0 confidence intervals finite and honest
k0 no longer pinned to initialization
```

---

## 11. Phase 5 — If steady-state multi-experiment still fails

If O2 variation, H2O2-fed experiments, and L_ref variation do not remove the basin problem, then the next genuinely different information source is time/frequency response.

Consider, in this order:

```text
1. voltage-step chronoamperometry
2. transient H2O2 production after voltage steps
3. EIS / small-signal impedance at selected biases
4. H2O2-fed chronoamperometry
```

Reason:

```text
Steady-state I-V curves mostly encode algebraic balances. Transient and frequency-domain data can expose kinetic timescales, which may separate k0 from alpha better than more steady-state amplitudes.
```

Do not start this until the steady-state multi-experiment FIM screen is complete.

---

## 12. Summary of immediate implementation tasks

### Task 1 — Single-experiment closure diagnostics

Create:

```text
scripts/studies/v23_restart_perturbation.py
scripts/studies/v23_single_experiment_basin_map.py
```

Outputs:

```text
StudyResults/v23_restart_perturbation/
StudyResults/v23_single_experiment_basin_map/
```

### Task 2 — Anchored Tafel parameterization

Create:

```text
scripts/studies/v23_anchored_tafel_lsq_inverse.py
```

Outputs:

```text
StudyResults/v23_anchored_tafel_parameterization/
```

### Task 3 — Multi-experiment FIM screen

Create:

```text
scripts/studies/v23_multiexperiment_fim.py
```

with experiment specs for:

```text
bulk O2 variation
H2O2-fed R2 isolation
L_ref variation
combined designs
```

Outputs:

```text
StudyResults/v23_multiexperiment_fim/
```

### Task 4 — Multi-experiment inverse only if Task 3 passes

Create only if FIM screening is promising:

```text
scripts/studies/v23_multiexperiment_lsq_inverse.py
```

Outputs:

```text
StudyResults/v23_multiexperiment_lsq_inverse_clean/
```

### Task 5 — Noisy seeds only after clean success

Outputs:

```text
StudyResults/v23_multiexperiment_lsq_inverse_noisy/
```

---

## 13. Reporting language

Use this wording if Phase 0/1 do not solve single-experiment recovery:

```text
The log-rate BV formulation removes the old clipped-R2 local identifiability failure, but a single steady-state CD+PC experiment still produces a multi-basin Tafel-ridge objective. Transfer coefficients are robustly identifiable, but joint exchange-rate recovery requires either much tighter prior information than is usually defensible or additional independent experiments/observables.
```

Use this wording if anchored Tafel coordinates recover beta but not k0:

```text
The data identify effective kinetic rates at measured voltages more robustly than formal exchange constants extrapolated to E_eq. The appropriate identifiable quantities may be anchored Tafel rates rather than k0 itself.
```

Use this wording if multi-experiment FIM succeeds:

```text
Adding independent transport/concentration perturbations rotates the remaining weak Fisher direction and converts the problem from single-experiment multi-basin recovery to a better-posed multi-experiment inverse.
```

---

## 14. Final priority order

```text
1. Restart-with-perturbation from best wrong basin.
2. Small basin-of-attraction map around TRUE.
3. Anchored Tafel-coordinate TRF.
4. Multi-experiment FIM: bulk O2 variation first.
5. Multi-experiment FIM: H2O2-fed R2 isolation second.
6. Multi-experiment FIM: L_ref/rotation variation third.
7. Clean multi-experiment inverse only after FIM improvement.
8. Noisy seeds only after clean success.
9. Transient/EIS only if steady-state multi-experiment still fails.
```
