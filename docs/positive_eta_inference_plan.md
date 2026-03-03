# Positive Overpotential (Anodic) Extension for BV Parameter Inference

**Date:** 2026-03-01
**Context:** 4-species PNP-BV system (O2, H2O2, H+, ClO4-), two coupled reactions

---

## 1. Physics Analysis: Why Positive Eta Helps

### 1.1 The Alpha Identifiability Problem

At cathodic-only overpotentials (eta < 0), the Butler-Volmer rate expression for
reaction j reduces to:

```
R_j ~ k0_j * c_cathodic * (c_H+/c_ref)^2 * exp(-alpha_j * eta_hat)
```

The observable (current density) depends on `k0 * exp(-alpha * eta)`. This creates
a fundamental **k0-alpha correlation**: changing k0 by a factor A is indistinguishable
from shifting alpha by `ln(A)/eta`. At any fixed eta, the objective function has a
**valley** in (log k0, alpha) space with slope `-1/eta`. Different cathodic eta values
have different slopes, but all slopes are negative -- the constraint from multiple
cathodic points narrows the valley but does not eliminate it.

### 1.2 How Anodic Data Breaks the Correlation

At anodic (positive) overpotentials, R1 runs in reverse (H2O2 -> O2), and the
dominant BV branch becomes:

```
R_1,anodic ~ -k0_1 * c_ref * exp((1-alpha_1) * eta_hat)
```

The key difference: the observable now depends on `exp(+(1-alpha)*eta)` instead of
`exp(-alpha*eta)`. This gives a Tafel slope of **(1-alpha)** instead of **alpha**.

The k0-alpha correlation at anodic eta has slope `+1/((1-alpha)*eta)`, which is
**positive** -- the opposite sign from the cathodic correlation slope. Combining
cathodic and anodic data means the correlation valleys cross at a well-defined point,
pinpointing both k0 and alpha simultaneously.

Quantitatively:
- Cathodic Tafel slope: `d(ln|I|)/d(eta_hat) = -alpha_1 = -0.627`
- Anodic Tafel slope: `d(ln|I|)/d(eta_hat) = (1-alpha_1) = +0.373`
- These are **two independent measurements** of alpha from the same system.
- The ratio `alpha / (1-alpha) = 1.681` can be verified from the slope ratio.

### 1.3 Near-Equilibrium Region: Exchange Current Density

At eta ~ 0, both BV branches contribute comparably. The linearized BV equation gives:

```
R ~ k0 * (alpha + (1-alpha)) * eta_hat = k0 * eta_hat    (for small eta_hat)
```

The **slope of I-V at eta=0** directly measures k0 (the exchange current density,
up to constants), independent of alpha. This is the most direct constraint on k0
available in the data.

In practice, the "linear regime" extends to roughly |eta_hat| < 1. Dense data
placement in this region constrains k0 via the slope and alpha via the curvature.

### 1.4 R2 Isolation at Positive Eta

R2 (H2O2 -> H2O) is irreversible -- it has no anodic branch. At positive eta:
- R2 cathodic term: `k0_2 * c_H2O2 * exp(-alpha_2 * eta)` -- this decays
  exponentially to zero as eta increases.
- At eta_hat >= +2, R2 is effectively zero.

This means **anodic data is pure R1 signal**, completely free of R2 contamination.
This is extremely valuable because:
1. R1 parameters can be determined from anodic data without interference from R2.
2. Once R1 is pinned, cathodic data (where R1 and R2 both contribute) can be used
   to constrain R2 parameters via subtraction.

### 1.5 R2 Identifiability Outlook

R2 (k0_2, alpha_2) remains fundamentally difficult because:
- R2 is irreversible, so there is no anodic branch to provide an independent
  constraint on alpha_2.
- R2's contribution to total current is always smaller than R1's in the cathodic
  range (k0_2 << k0_1 by ~24x).
- At the plateau, R2 is mass-transport limited (not kinetics-limited), so k0_2
  and alpha_2 affect only the TRANSITION region.

However, with R1 well-pinned from anodic data, the RESIDUAL current (I_total - I_R1)
isolates R2. This should improve R2 recovery compared to the current joint inference
where R1 and R2 parameter uncertainties compound.

### 1.6 Steric Parameter Sensitivity

The Bikerman steric chemical potential is `mu_steric = ln(1 - sum(a_j * c_j))`.

At anodic potentials (eta > 0):
- O2 accumulates near electrode (c_O2 ~ 1.004 at eta=+3, slight increase)
- H2O2 is depleted (c_H2O2 goes slightly negative -- see convergence section)
- H+ is strongly depleted (c_H+ drops from 0.2 to 0.01 at eta=+3)
- Net packing: sum(a*c) changes mainly through H+ depletion

The steric signal at anodic potentials is WEAK because:
1. Concentration changes are small (only H+ changes significantly)
2. The steric term is most informative when packing fractions are intermediate
   (neither too high nor too low)

The steric sensitivity is still concentrated in the cathodic transition region
(eta = -3 to -8) where H2O2 accumulation creates significant packing. Anodic
data adds little steric information directly, but by improving k0/alpha recovery,
it indirectly improves steric parameter recovery in a full (k0+alpha+steric)
inference by reducing parameter cross-talk.

---

## 2. Forward Solver Convergence at Positive Eta

### 2.1 Empirical Results

Dense sweep from eta_hat = 0 to +7 in steps of 0.5 V_T:

| eta_hat | eta (mV) | Steps | I_pxd (mA/cm2) | c_O2  | c_H2O2  | c_H+   |
|---------|----------|-------|-----------------|-------|---------|--------|
| +0.5    | +12.8    | 7     | +2.164e-04      | 1.001 | -0.001  | 0.122  |
| +1.0    | +25.7    | 6     | +3.194e-04      | 1.002 | -0.002  | 0.074  |
| +2.0    | +51.4    | 6     | +4.871e-04      | 1.003 | -0.003  | 0.027  |
| +3.0    | +77.1    | 6     | +7.089e-04      | 1.004 | -0.005  | 0.010  |
| +5.0    | +128.5   | 7     | +1.495e-03      | 1.008 | -0.010  | 0.001  |
| +6.0    | +154.2   | 8     | +2.171e-03      | 1.012 | -0.014  | 0.0005 |
| +6.5    | +167.0   | 8     | +2.616e-03      | 1.014 | -0.017  | 0.0003 |
| +7.0    | --       | FAIL  | DIVERGED_DTOL   | --    | --      | --     |

Cathodic comparison:

| eta_hat | eta (mV)  | Steps | I_pxd (mA/cm2) | c_O2  | c_H2O2 | c_H+  |
|---------|-----------|-------|-----------------|-------|--------|-------|
| -0.5    | -12.8     | 8     | -6.535e-04      | 0.996 | 0.004  | 0.327 |
| -1.0    | -25.7     | 10    | -2.802e-03      | 0.985 | 0.018  | 0.527 |
| -2.0    | -51.4     | 13    | -2.249e-02      | 0.877 | 0.146  | 1.130 |
| -5.0    | -128.5    | 12    | -1.210e-01      | 0.299 | 0.784  | 1.796 |
| -10.0   | -256.9    | 7     | -1.466e-01      | 0.114 | 0.950  | 0.683 |

### 2.2 Convergence Limits and Root Causes

**Maximum converged anodic eta: +6.5** (167 mV).

Failure mode at +7.0: DIVERGED_DTOL. Root causes:

1. **H+ depletion**: c_H+ drops from 0.2 (bulk) to 0.0003 at eta=+6.5. At eta=+7,
   c_H+ approaches the machine epsilon regime, making the Poisson equation stiff
   (recall (lambda_D/L)^2 ~ 9e-8 with the original c_H+; near-zero c_H+ makes
   the effective Debye length diverge, destabilizing the Poisson coupling).

2. **Negative H2O2 concentration**: Since c_H2O2_bulk = 0, the anodic reaction
   (H2O2 -> O2) drives c_H2O2 negative. The concentration floor (`conc_floor=1e-12`)
   only clips concentrations in the BV rate expression, not in the transport equation.
   Negative concentrations do not violate the PDE (diffusion can produce negative
   values transiently) but create unphysical BV rates.

3. **H+ cathodic concentration factors**: The `(c_H+/c_ref)^2` factor in the BV
   rate multiplies the already-small H+ concentration, further reducing the reaction
   rate and making the Jacobian difficult.

### 2.3 Practical Operating Range

For inference, we need reliable convergence AND meaningful signal. The useful
range is **eta_hat in [-10, +5]** (or [-6, +5] for a focused study):

- eta_hat in [+0.5, +5.0]: Anodic Tafel regime, pure R1 signal, converges in 6-9 steps
- eta_hat in [-0.5, +0.5]: Near-equilibrium, linear regime, exchange current density
- eta_hat in [-1, -5]: Cathodic onset/transition, R1 + R2
- eta_hat in [-5, -10]: Cathodic knee/plateau, mass-transport limited

The anodic regime (eta > +5) is not needed because:
1. The Tafel slope information is already well-determined by eta=+5
2. The currents are small (10x smaller than cathodic plateau), so SNR degrades
3. Convergence becomes unreliable above +6.5

### 2.4 Pipeline Changes for Mixed-Sign Eta

**Updated (2026-03-01)**: The naive `np.argsort(np.abs(...))` sweep DOES NOT work
for mixed positive/negative eta. Empirical testing showed SNES divergence at
eta=+2.0 and +3.0 when warm-started from a cathodic solution at eta=-2.0. The
concentration profiles are inverted: at cathodic eta, O2 is depleted and H2O2
accumulates; at anodic eta, O2 accumulates and H2O2 depletes. The solver cannot
bridge this inversion in a single pseudo-time step.

**Fix**: `FluxCurve/bv_point_solve.py` now uses a two-branch sweep order for
mixed-sign eta arrays. See Section 5.1 for details.

The BV form itself correctly handles both signs of eta through `eta_clipped` --
the issue was only with the warm-start continuation strategy between branches.

---

## 3. Proposed Voltage Placement

### 3.1 Primary Configuration: 20-point Symmetric Placement

```python
eta_values = np.array([
    # Anodic region (pure R1, alpha identification)
    +5.0, +3.0, +2.0, +1.0, +0.5,
    # Near-equilibrium (exchange current density, k0)
    -0.25, -0.5,
    # Cathodic onset/Tafel (R1 kinetics + R2 onset)
    -1.0, -1.5, -2.0, -3.0,
    # Cathodic transition (R1 + R2 kinetics, steric signal peak)
    -4.0, -5.0, -6.5, -8.0,
    # Cathodic knee (mass-transport onset)
    -10.0, -13.0,
    # Cathodic plateau (mass-transport limited, weak k0/alpha info)
    -17.0, -22.0, -28.0,
])
```

Rationale:
- **5 anodic points** (eta = +0.5 to +5): Determine anodic Tafel slope = (1-alpha)
  for R1. Pure R1 signal, no R2 contamination. Gap of 2.0 at most, well within
  warm-start capability.
- **2 near-equilibrium points** (eta = -0.25, -0.5): Constrain exchange current
  density (proportional to k0). The slope d(I)/d(eta) at eta~0 gives k0 directly.
- **4 cathodic onset points** (eta = -1 to -3): Cathodic Tafel slope = -alpha.
  Combined with anodic slope, pins alpha precisely.
- **4 cathodic transition points** (eta = -4 to -8): Where R2 begins to contribute
  and steric signal peaks. Best region for k0_2/alpha_2 constraints.
- **2 cathodic knee points** (eta = -10, -13): Transition to plateau, constrains
  limiting current.
- **3 cathodic plateau points** (eta = -17, -22, -28): Constrain plateau height
  (affected by steric parameter). Requires bridge points.

### 3.2 Secondary Configuration: 12-point Focused Placement

```python
eta_values_focused = np.array([
    +5.0, +3.0, +1.0,           # anodic (3 points)
    -0.5,                         # near-equilibrium (1 point)
    -1.0, -2.0, -3.0,           # cathodic onset (3 points)
    -5.0, -8.0,                  # transition (2 points)
    -10.0, -15.0, -20.0,        # knee + plateau (3 points)
])
```

Use this for quick testing (less compute, still captures the essential information).

### 3.3 Minimal Configuration: 8-point "Symmetric Tafel"

```python
eta_values_minimal = np.array([
    +5.0, +2.0,                  # anodic
    -0.5,                         # near-equilibrium
    -2.0, -5.0,                  # cathodic onset
    -8.0, -13.0, -20.0,         # transition + plateau
])
```

Use this as the fastest possible test of whether anodic data helps.

---

## 4. Inference Experiments to Run

### Experiment 1: Alpha Recovery (Symmetric Tafel)

**Goal**: Demonstrate that anodic data improves alpha_1 recovery.

**Setup**:
- Control mode: "alpha" (k0 fixed at true values)
- Two runs: (a) cathodic-only eta in [-1, -10], (b) symmetric eta as in Section 3.1
- Compare alpha errors

**Expected outcome**: Alpha_1 error drops from ~0.1% (already good) to essentially
zero. Alpha_2 may not improve (R2 has no anodic signal). The key test is whether
the k0-alpha correlation is broken in joint inference.

### Experiment 2: Joint k0 + Alpha Recovery (Symmetric)

**Goal**: Test whether joint inference benefits from anodic data.

**Setup**:
- Control mode: "joint" (infer k0 and alpha simultaneously)
- Three runs:
  (a) cathodic-only [-1, -10], 10 points
  (b) symmetric [-5 to +5] + cathodic extension to -10, 12 points
  (c) full 20-point placement from Section 3.1
- Same initial guesses (wrong by ~10x for k0, ~0.2 for alpha)

**Expected outcome**: Joint inference with symmetric data should show:
- k0_1 error: reduced from ~5% to < 2%
- alpha_1 error: reduced from ~5% (joint) to < 1%
- Faster convergence (fewer optimizer iterations, better-conditioned Hessian)
- k0_2 and alpha_2: modest improvement via R1 pinning

### Experiment 3: Staged Inference (Symmetric)

**Goal**: Determine if staged inference still helps when anodic data is available.

**Setup**:
- Stage 1: Alpha from symmetric data (k0 fixed)
- Stage 2: k0 from full range (alpha fixed from Stage 1)
- Stage 3: Joint refinement
- Compare against direct joint on symmetric data

**Expected outcome**: With symmetric data, direct joint may match staged inference
because the k0-alpha correlation is already broken. If so, staged inference becomes
unnecessary, simplifying the pipeline.

### Experiment 4: Steric Parameter Inference (Extended Range)

**Goal**: Test steric parameter recovery with symmetric voltage placement.

**Setup**:
- Control mode: "steric" (k0 and alpha fixed, infer a_vals)
- 20-point placement including cathodic plateau
- Compare against cathodic-only extended [-1, -46.5]

**Expected outcome**: Steric recovery is mainly driven by cathodic transition data.
Anodic data should have minimal direct effect on steric parameter recovery. The
indirect benefit (better k0/alpha -> less cross-talk) may help in "full" mode.

### Experiment 5: Full Parameter Inference (k0 + alpha + steric)

**Goal**: Ultimate test -- all 6 parameters simultaneously.

**Setup**:
- Control mode: "full"
- 20-point symmetric placement
- Compare against cathodic-only

**Expected outcome**: Full inference benefits most from symmetric data because the
6-parameter problem has the most severe cross-correlation issues.

---

## 5. Convergence Strategy

### 5.1 Sweep Order

**Updated (2026-03-01)**: The naive `np.argsort(np.abs(eta))` approach that
interleaves positive and negative eta values causes SNES divergence at moderate
|eta|. Empirical testing showed that warm-starting from eta=-2.0 to eta=+2.0
fails because the concentration profiles are inverted (O2 depleted/H2O2 accumulated
at cathodic vs the reverse at anodic). The solver cannot bridge this in a single
pseudo-time step.

**Fix implemented**: `bv_point_solve.py` now uses a two-branch sweep:

1. **Negative branch**: All eta <= 0, sorted ascending in |eta|.
   ```
   -0.25, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -5.0, -6.5, -8.0,
   -10.0, -13.0, -17.0, -22.0, -28.0
   ```
2. **Save hub state**: The converged solution at eta = -0.25 (closest to equilibrium)
   is saved as a "hub state".
3. **Positive branch**: Restore hub state, then sweep eta > 0 ascending.
   ```
   +0.5, +1.0, +2.0, +3.0, +5.0
   ```

The only sign-change transition occurs from eta = -0.25 to eta = +0.5 (a small
0.75 V_T jump near equilibrium). The predictor history is cleared at the branch
transition to prevent extrapolation from cathodic solutions.

Bridge points between branches are inserted automatically if `max_eta_gap > 0`
and the gap exceeds the threshold.

### 5.2 Bridge Points for Large Cathodic Gaps

Bridge points are needed only in the cathodic range (gaps > 3.0 V_T):
- eta=-13 to -17: bridge at -15
- eta=-17 to -22: bridge at -19.5
- eta=-22 to -28: bridge at -25

No bridges needed in the anodic range (max gap is 2.0 V_T, from +3 to +5).

Use `max_eta_gap=3.0` in the `BVFluxCurveInferenceRequest`.

### 5.3 Negative H2O2 Concentration Handling

At anodic potentials, c_H2O2 goes slightly negative (down to -0.017 at eta=+6.5).
This is physically unphysical but numerically benign for the forward solver because:

1. The magnitude is small (|c_H2O2| < 0.02, vs bulk c_O2 = 1.0)
2. The BV rate expression uses `c_surf = max(c_H2O2, eps_c)` which clips to the
   floor, preventing BV blow-up
3. The transport equation does not require c >= 0 (diffusion can produce negative
   transients)

For the adjoint, negative concentrations may cause gradient issues. If this becomes
a problem, two mitigations are available:

**Mitigation A**: Use softplus regularization in the BV expression (already supported
via `"softplus_regularization": True` in `bv_convergence`). This provides a smooth
C-infinity approximation to the concentration floor.

**Mitigation B**: Add a small positive bulk H2O2 concentration (e.g., c_H2O2_bulk = 0.01)
so the anodic reaction has some H2O2 to consume. This changes the physics slightly
but may be justified if the experiment involves pre-electrolysis.

For now, proceed without mitigation. The forward solver converges, and the adjoint
should handle the small negative concentrations.

### 5.4 H+ Depletion at Anodic Potentials

At eta=+5, c_H+ drops to 0.001 (0.5% of bulk). This is the same phenomenon as
O2 depletion at large cathodic eta -- the species being consumed in the dominant
reaction is depleted at the electrode. The solver handles this via:

1. Mesh grading (beta=3, Ny=200) provides fine resolution near the electrode
2. Exponent clipping (clip at +/-50) prevents overflow
3. Concentration floor (1e-12) prevents division by zero

The convergence limit at eta~+7 is not a fundamental issue for inference because
we only need eta up to +5.

---

## 6. Implementation Plan

### 6.1 Files to Create

1. **`scripts/inference/Infer_BVStaged_charged_symmetric.py`**
   - Staged inference with symmetric 20-point placement
   - Same 4-stage structure as `Infer_BVStaged_charged_from_current_density_curve.py`
   - Uses `eta_values` from Section 3.1

2. **`scripts/inference/Infer_BVJoint_charged_symmetric.py`**
   - Direct joint inference with symmetric placement
   - Comparison baseline

3. **`scripts/inference/Infer_BVSteric_charged_symmetric.py`**
   - Steric inference with symmetric + cathodic placement

4. **`scripts/inference/Infer_BVFull_charged_symmetric.py`**
   - Full (k0 + alpha + steric) inference with symmetric placement

5. **`scripts/bv/bv_iv_curve_symmetric.py`**
   - Forward I-V sweep covering both positive and negative eta
   - Generates a complete I-V curve for validation

### 6.2 Files Modified

**`FluxCurve/bv_point_solve.py`** -- Two changes:

1. **Two-branch sweep order** (`_build_sweep_order` function):  When phi_applied
   contains both positive and negative values, the sweep is split into two branches
   (negative ascending |eta|, then positive ascending eta) instead of interleaving
   by |eta|. Single-sign arrays retain the original ascending-|eta| behaviour.

2. **Branch transition handling**: At the sign-change transition between branches,
   the predictor history is cleared and the warm-start state is restored from the
   "hub" (the first converged point near eta = 0) rather than from the last point
   of the first branch (which may be at eta = -28).

No changes to `Forward/bv_solver.py` or `FluxCurve/bv_run.py` -- the BV form
already handles both signs of eta via the `eta_clipped` expression, and the
adjoint tape recording is agnostic to the sign of eta.

### 6.3 Forward Convergence Verification

Before running inference, verify forward convergence at all 20 points:
```bash
python scripts/bv/bv_iv_curve_symmetric.py
```

This should complete in ~2 minutes and produce a CSV + plot showing the I-V curve
spanning both anodic and cathodic regions.

---

## 7. Predictions

### 7.1 R1 Parameters (k0_1, alpha_1)

| Metric | Cathodic-only [-1,-10] | Symmetric [-5,+5,-10] | Improvement |
|--------|----------------------|----------------------|-------------|
| alpha_1 error (alpha-only) | 0.1% | < 0.05% | ~2x |
| alpha_1 error (joint) | ~5% | < 1% | ~5x |
| k0_1 error (joint) | ~5% | < 2% | ~2.5x |
| alpha_1 error (staged S1) | 0.1% | < 0.05% | ~2x |
| k0_1 error (staged S2) | 4.9% | < 2% | ~2.5x |

**Mechanism**: The anodic Tafel slope provides an independent constraint on alpha_1,
breaking the k0-alpha correlation that limits joint inference at cathodic-only data.

### 7.2 R2 Parameters (k0_2, alpha_2)

| Metric | Cathodic-only | Symmetric | Improvement |
|--------|--------------|-----------|-------------|
| k0_2 error | > 95% | ~50-80% | Modest (~2x) |
| alpha_2 error | > 50% | ~30-50% | Modest (~1.5x) |

**Mechanism**: R2 has no anodic branch, so no direct benefit. The indirect benefit
comes from pinning R1 parameters, allowing residual analysis to isolate R2. The
improvement is modest because R2 is fundamentally constrained only by the cathodic
transition region.

### 7.3 Steric Parameter (a)

| Metric | Cathodic-only | Symmetric | Improvement |
|--------|--------------|-----------|-------------|
| a error (steric-only) | ~5-10% | ~5-10% | None |
| a error (full) | ~20-30% | ~10-15% | ~2x |

**Mechanism**: Steric sensitivity is concentrated in the cathodic transition region.
Anodic data does not directly constrain steric parameters. In full inference, better
k0/alpha determination reduces cross-talk, indirectly improving steric recovery.

### 7.4 Optimizer Convergence

| Metric | Cathodic-only | Symmetric | Improvement |
|--------|--------------|-----------|-------------|
| Joint iterations | ~30 | ~20 | ~1.5x |
| Joint time | ~5 min | ~6 min* | Similar |
| Staged total time | ~8 min | ~10 min* | Similar |

*Symmetric has more points (20 vs 10), so per-evaluation time increases, but fewer
optimizer iterations are needed due to better-conditioned objective landscape.

### 7.5 Summary

The primary benefit of positive eta is **breaking the k0-alpha correlation for R1**.
This is a structural improvement in the information content of the data, not just
a numerical tweak. The benefit is most pronounced in joint and full inference
(which suffer most from parameter correlation) and minimal for alpha-only inference
(which already performs well with cathodic-only data).

R2 identifiability remains the hardest problem. Positive eta does not directly help,
but the indirect benefit of pinning R1 is expected to provide modest improvement.

---

## 8. References

- [Butler-Volmer Equation and Tafel Analysis (Fiveable)](https://fiveable.me/electrochemistry/unit-5/butler-volmer-equation-tafel-analysis/study-guide/ZmhYJiafdALjEYFQ)
- [The Butler-Volmer equation in electrochemical theory (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S1572665720303283)
- [Revisiting Butler-Volmer: Separating Anodic and Cathodic Components (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S1572665724000870)
- [Butler-Volmer equation (Wikipedia)](https://en.wikipedia.org/wiki/Butler%E2%80%93Volmer_equation)
- [MIT OCW: Butler-Volmer equation lecture notes](https://ocw.mit.edu/courses/10-626-electrochemical-energy-systems-spring-2014/56cfa6e0f28bc8fc1a647cbe679384d1_MIT10_626S14_S11lec13.pdf)
- [Tafel Analysis Guide (Admiral Instruments)](https://www.admiralinstruments.com/_files/ugd/dc5bf5_800a2edead684e01b9c0f50083a7fbb3.pdf?index=true)
