# Phase 3: Charged System Inference Summary

## System Configuration

- 4-species charged system: O2, H2O2, H+, ClO4- (z=[0,0,+1,-1])
- Two BV reactions: R1 (O2 -> H2O2, reversible), R2 (H2O2 -> H2O, irreversible)
- True parameters: k0_1=0.001263, k0_2=5.263e-5, alpha_1=0.627, alpha_2=0.5
- Voltage range: eta_hat = -1.0 to -10.0 (10 points)
- 2% Gaussian noise on synthetic targets
- Mesh: 8x200, beta=3.0 (graded)
- L-BFGS-B optimizer, up to 30 iterations

---

## Method 1: Regularized Joint Inference

Script: `Infer_BVJoint_charged_regularized.py`
Output: `StudyResults/bv_joint_regularized_charged/`

Tikhonov regularization with prior: k0_prior = [1.5x true_k0_1, 2x true_k0_2], alpha_prior = [0.5, 0.4]

| lambda | k0_1 err % | k0_2 err % | alpha_1 err % | alpha_2 err % | loss |
|--------|-----------|-----------|--------------|--------------|------|
| 0.0000 | 199.5 | 377.3 | 74.4 | 76.9 | 1.92e-05 |
| 0.0010 | 50.2 | 100.7 | 21.7 | 19.4 | 2.80e-05 |
| 0.0100 | 49.6 | 100.0 | 20.7 | 20.0 | 2.82e-05 |
| 0.1000 | 49.9 | 100.0 | 20.3 | 20.0 | 2.83e-05 |
| 1.0000 | 50.0 | 100.0 | 20.3 | 20.0 | 2.84e-05 |

### Key Findings

1. Regularization dramatically reduces parameter error vs unregularized (e.g., k0_1: 200% -> 50%)
2. Results are insensitive to lambda in [0.001, 1.0] -- all give nearly identical parameter values
3. The converged alpha values (0.50, 0.40) match the prior, not the truth (0.627, 0.5)
4. k0_2 error is exactly 100% for all lambda > 0, meaning it converges to the prior value
5. The regularization essentially pulls toward the prior and stops -- the data provides insufficient signal to move away from the prior for R2 parameters

### Interpretation

The unregularized problem has severe k0-alpha correlation, causing the optimizer to find a
false minimum with compensating errors. Regularization removes this correlation by anchoring
to the prior, but the resulting solution inherits the prior's bias. The problem is
fundamentally ill-conditioned for joint (k0, alpha) recovery of both reactions.

---

## Method 2: Staged Inference

Script: `Infer_BVStaged_charged_from_current_density_curve.py`
Output: `StudyResults/bv_staged_inference_charged/`

| Stage | k0_1 err % | k0_2 err % | alpha_1 err % | alpha_2 err % | loss | time |
|-------|-----------|-----------|--------------|--------------|------|------|
| S1: Alpha (k0 fixed true) | 0.0 | 0.0 | 0.3 | 67.1 | 3.33e-05 | 157s |
| S2: k0 (alpha fixed S1) | 5.0 | 96.9 | 0.3 | 67.1 | 3.28e-05 | 988s |
| S3: Joint warm start | 19.2 | 96.9 | 5.8 | 66.9 | 3.21e-05 | 112s |
| S4: Direct joint baseline | 221.3 | 270.2 | 75.4 | 90.0 | 1.90e-05 | 347s |

### Key Findings

1. **Staging greatly improves R1 parameter recovery** (k0_1: 19% vs 221%, alpha_1: 5.8% vs 75.4%)
2. **R2 parameters are unidentifiable from total current density** regardless of method
   - alpha_2 error is 67% even when k0 is fixed at true values (Stage 1)
   - k0_2 collapses to near-zero (96.9% error) in all stages
3. The staged approach finds a **physically more meaningful minimum** despite higher loss (3.21e-05 vs 1.90e-05)
4. The direct joint baseline (Stage 4) finds a lower-loss solution, but with much worse
   parameter values -- a classic **false minimum** in an ill-conditioned inverse problem
5. Total staged time (1258s) is much longer than direct joint (347s) due to Stage 2's slow convergence on k0_2

### Why Staging Helps

Staging breaks the k0-alpha correlation for R1 by first recovering alpha_1=0.629 (with k0 fixed),
then using that correct alpha_1 to recover k0_1=0.001326 (5% error). The joint refinement
(Stage 3) slightly adjusts both but stays near the warm-start solution.

### Why R2 Fails

The second reaction (H2O2 -> H2O) contributes very little to the total current density at
the voltage range tested. Its signal is buried in noise, making both k0_2 and alpha_2
fundamentally unidentifiable from total current alone.

---

## Method 3: Peroxide Current Observable

### k0-only from peroxide current

Script: `Infer_BVk0_charged_peroxide_current.py`
Output: `StudyResults/bv_k0_peroxide_current_charged/`

| Parameter | True | Recovered | Error % |
|-----------|------|-----------|---------|
| k0_1 | 0.001263 | 0.001331 | 5.4 |
| k0_2 | 5.263e-5 | 1.139e-8 | 99.98 |

### Joint (k0, alpha) from peroxide current

Script: `Infer_BVJoint_charged_peroxide_current.py`
Output: `StudyResults/bv_joint_peroxide_current_charged/`

| Parameter | True | Recovered | Error % |
|-----------|------|-----------|---------|
| k0_1 | 0.001263 | 0.001495 | 18.3 |
| k0_2 | 5.263e-5 | 1.830e-6 | 96.5 |
| alpha_1 | 0.627 | 0.593 | 5.4 |
| alpha_2 | 0.500 | 0.050 | 90.0 (hit bound) |

### Key Findings

1. **Peroxide current observable does NOT resolve R2 identifiability**
2. k0_2 still collapses to near-zero in both k0-only and joint modes
3. alpha_2 hits the lower bound (0.05) in joint mode -- completely unidentifiable
4. R1 recovery is similar to total current density methods (k0_1: 5-18%, alpha_1: 5%)

### Why Peroxide Observable Fails

The peroxide current I_pxd = -(R0 - R1) * scale is dominated by R0 (the O2->H2O2 reaction)
because at these overpotentials R1 << R0. The net peroxide signal is essentially the same as
the O2 reduction signal, providing no additional information about R2.

---

## Comprehensive Comparison

| Method | Observable | k0_1 err % | alpha_1 err % | k0_2 err % | alpha_2 err % | Time |
|--------|-----------|-----------|--------------|-----------|--------------|------|
| Regularized (lam=0) | current_density | 199.5 | 74.4 | 377.3 | 76.9 | ~5min |
| Regularized (lam=0.001) | current_density | 50.2 | 21.7 | 100.7 | 19.4 | ~5min |
| Staged (S3 final) | current_density | 19.2 | 5.8 | 96.9 | 66.9 | ~21min |
| Direct joint (S4) | current_density | 221.3 | 75.4 | 270.2 | 90.0 | ~6min |
| k0-only peroxide | peroxide_current | 5.4 | N/A | 99.98 | N/A | ~11min |
| Joint peroxide | peroxide_current | 18.3 | 5.4 | 96.5 | 90.0 | ~13min |

---

## Answers to Key Questions

### 1. Does regularization help charged joint inference?

**YES, dramatically for R1.** Regularization reduces k0_1 error from 200% to 50% and alpha_1
from 74% to 22%. However, the regularized solution converges to the prior values rather than
the true values, so accuracy depends on prior quality. For R2, regularization simply anchors
at the prior (100% error), which is better than the unregularized 377% but still poor.

### 2. Does staging break the k0-alpha correlation?

**YES, for R1.** By fixing k0 at true values and recovering alpha_1 first (0.3% error),
then using that correct alpha to recover k0_1 (5% error), staging achieves the best R1
recovery of any method tested. The final joint refinement (Stage 3) gives k0_1=19.2%,
alpha_1=5.8%. This is far superior to direct joint (221%, 75%).

### 3. Does the peroxide observable improve R2 identifiability?

**NO.** The peroxide current is dominated by R1 at the voltages tested. k0_2 collapses to
near-zero and alpha_2 hits its lower bound in all peroxide-based experiments. The R2 reaction
simply does not generate enough current to produce a distinguishable signal.

### 4. What is the best method for the charged system?

**Staged inference is the best approach for R1 parameter recovery.**

For a complete recommendation:
- Use staged inference to accurately recover k0_1 and alpha_1 (errors <20%)
- R2 parameters (k0_2, alpha_2) cannot be reliably recovered from ANY tested observable
- To improve R2 identifiability, consider: (a) using H2O2 concentration profiles directly,
  (b) testing at voltages where R2 dominates, or (c) using separate experiments that
  isolate the H2O2 -> H2O reaction

---

## Files Generated

```
StudyResults/
  bv_joint_regularized_charged/     -- 5 lambda subdirectories
  bv_staged_inference_charged/      -- 4 stage subdirectories + comparison CSV
  bv_k0_peroxide_current_charged/   -- k0-only peroxide inference
  bv_joint_peroxide_current_charged/ -- joint peroxide inference
  phase3_inference_summary.md       -- this file
```
