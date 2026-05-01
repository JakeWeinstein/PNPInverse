# PNP-BV Inverse Solver — Positive Voltage Extension Diagnostic

## Purpose

This handoff answers the question: **Is it worth pushing the log-rate BV solver to more positive voltages beyond +0.60 V?**

Short answer: **yes, but only as a cheap diagnostic / secondary solver-extension branch.** It should not replace the current main path: LM on G0, then Tikhonov / Bayesian prior tests if LM does not solve the multi-init ridge problem.

The latest validated state is:

- Log-rate BV + extended grid to +0.60 V makes `k0_2` locally identifiable by FIM.
- The old `log_k0_2` weak direction is gone.
- The remaining weak direction is now robustly `log_k0_1`.
- Voltage-grid changes in the lower/onset range did **not** fix this.
- TRF stalls are not caused by adjoint error or bad grid choice; they reflect the cost landscape / Tafel-ridge basins.

Therefore, more positive voltages are worth testing only if they improve the new weak direction or reduce basin ambiguity.

---

## Main recommendation

Do **not** make positive-voltage extension the main next step.

Main path remains:

```text
A. Run LM on G0 with 4 inits.
B. If LM succeeds, compare LM vs TRF and then run noisy seeds.
C. If LM fails or is mixed, run Tikhonov on log(k0) with sigma = log(3), then log(10).
D. Run noisy seeds only after choosing the successful framing.
```

Positive-voltage extension should be a **parallel/secondary diagnostic**:

```text
E. Test whether adding V > +0.60 improves the FIM weak direction.
```

---

## Why more positive voltage might help

The log-rate breakthrough showed that reaching +0.50 and +0.60 V matters because R2 begins to unclip and contributes real voltage-shape sensitivity. This was enough to eliminate the old `log_k0_2` ridge.

Adding +0.70, +0.80, or +1.00 V may provide more R2-side curvature and further constrain the R2 pathway.

Potential benefits:

- more direct R2 kinetic information;
- more separation between R1-dominated and R2-dominated regimes;
- possible improvement in global basin structure;
- possible reduction in `k0_2` ambiguity under noisy data.

---

## Why it may *not* solve the current problem

The current weak direction after the log-rate fix is **not `log_k0_2` anymore**. It is almost pure `log_k0_1` across every tested voltage grid and noise model.

Higher positive voltages mostly probe the region where R1 is already gone and R2 dominates. That may add little or no new information about `k0_1`.

So the key test is not simply whether the solver converges at +0.70 or +0.80. The key test is:

> Does the FIM weak eigenvector stop being almost pure `log_k0_1`?

If the weak direction remains `|log_k0_1| ≈ 1`, then more positive voltages are not the lever.

---

## Diagnostic voltage grids to test

Use G0 as the baseline:

```text
G0 = [-0.10, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
```

Then test increasingly positive extensions:

```text
Gpos1 = G0 + [0.70]
Gpos2 = G0 + [0.70, 0.80]
Gpos3 = G0 + [0.70, 0.80, 1.00]
Gpos4 = G0 + [0.70, 0.80, 1.00, 1.20]   # only if earlier grids converge cleanly
```

Do not jump directly to +1.20 or +1.50. Work upward.

---

## For each positive-voltage grid, report

### 1. Solver convergence

For each new voltage, record:

```text
V_RHE
cold convergence status
warm convergence status
number of continuation steps
number of Newton failures / fallbacks
min/max u_H2O2
min/max c_H2O2
min/max log_rate_R1
min/max log_rate_R2
```

Flag any voltage where:

```text
abs(log_rate) > 120
```

as numerically unreliable unless a deliberate guard/continuation strategy is used.

### 2. Signal size versus noise floor

For each new voltage, report:

```text
cd(V)
pc(V)
abs(cd) relative to max abs(cd)
abs(pc) relative to max abs(pc)
```

Important: if `cd` or `pc` at high V is far below a realistic absolute current noise floor, the FIM may show mathematical information that is experimentally unusable.

Use at least these noise models:

```text
A. global sigma = 2% * max(|observable|)
B. local sigma = 2% * |observable(V)|
C. local sigma with absolute floor:
   sigma(V) = sqrt((0.02 * |observable(V)|)^2 + sigma_abs^2)
```

Test several `sigma_abs` values in the same units used for cd/pc.

### 3. FIM diagnostics

For each grid and noise model, report:

```text
sv_min
cond(F)
weak eigenvector
|log_k0_1| component of weak eigenvector
|log_k0_2| component of weak eigenvector
ridge_cos versus old k0_2/alpha_2 ridge
```

The main pass/fail criterion:

```text
PASS if positive voltages reduce |log_k0_1| in the weak eigvec
     and improve sv_min / cond(F) without relying only on below-noise rows.

FAIL if weak eigvec remains |log_k0_1| ≈ 1.
```

### 4. Optional clean-data inverse only if FIM improves

Do **not** run full TRF/LM on extended positive grids unless the FIM improves meaningfully.

If FIM passes, run clean-data inverse on the best positive grid with:

```text
method = LM and/or TRF
observables = CD + PC
regularization = none first
inits = plus20, minus20, k0high_alow, k0low_ahigh
```

Then compare directly against G0.

---

## Expected outcome

Prediction:

- +0.70 and +0.80 may improve R2-related sensitivity.
- They may help stabilize `k0_2` under noise.
- They probably will **not** fix the current `log_k0_1` weak direction.
- If the weak direction remains `log_k0_1`, stop pursuing positive-voltage extension for this inverse problem.

---

## What not to do

Do not:

- replace LM/Tikhonov tests with positive-voltage exploration;
- spend time pushing to +1.50 before testing +0.70/+0.80;
- trust FIM gains from rows where cd/pc are below realistic noise floors;
- run noisy seeds on a positive-voltage grid before clean FIM and clean inverse pass;
- revive FV/SG just to reach higher positive voltages unless the FIM shows those voltages would fix the weak direction.

---

## Suggested one-line instruction to Claude Code

```text
Run a cheap positive-voltage FIM diagnostic on G0+[0.70], G0+[0.70,0.80], and G0+[0.70,0.80,1.00]. For each grid, check convergence, cd/pc magnitude vs noise floor, sv_min, cond(F), and the weak eigvec. Stop if the weak direction remains pure log_k0_1. Do not run full inverse or noisy seeds unless the FIM clearly improves the log_k0_1 weakness.
```

---

## Recommended priority

Priority order remains:

```text
1. LM on G0 with 4 inits.
2. Tikhonov / Bayesian log(k0) prior tests if LM is mixed.
3. Noisy seeds on the chosen framing.
4. Positive-voltage FIM diagnostic as a side branch.
5. FV/SG or higher-voltage methods only if positive-voltage FIM proves valuable.
```
