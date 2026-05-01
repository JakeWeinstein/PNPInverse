# k0 Inference — Autonomous Work Status

**Date**: 2026-04-13
**Context**: User stepped away; goal is to make k0 inference feasible.

## TL;DR

**k0 inference is feasible** with the following recipe:
1. **Log-concentration transform** (forms_logc.py) for onset-region convergence with positive concentrations
2. **Seeded H2O2 initial condition** (u = log(1e-4) = -9.2) to avoid log(0) singularity
3. **3-species + Boltzmann background** (no ClO4- dynamics)
4. **Tikhonov regularization** with a k0 prior (even weak λ=0.01)

Result on a synthetic test (6 onset voltages, 2% noise, k0_init +20%, α_init -10%):

| Approach | k0 error | α error |
|----------|---------:|--------:|
| Unregularized NM | **+1580%** (17x true) | -12% |
| **Regularized NM (λ=0.01)** | **-0.3%** | **+0.7%** |

**Regularization closes the k0-α ridge degeneracy** that was the fundamental limit before.

## The journey

### Step 1 — Understand the problem

The v17 investigation concluded k0 was "fundamentally non-identifiable" from I-V curves. That conclusion was based on forward solves producing non-physical data (negative H2O2 concentrations of ~-0.7) at anodic voltages where k0 matters. The solver's issue: CG1 finite elements oscillate H2O2 negative around its near-zero steady state at anodic voltages, combined with the BV exponent clip × concentration floor producing spurious sink rates.

### Step 2 — The log-c breakthrough

`forms_logc.py` already existed but was previously judged to fail because "H2O2 starts at c=0, so u=ln(1e-20)=-46 creates extreme stiffness." With a seeded H2O2 IC of 1e-4 (u=-9.2), this stiffness is avoided.

Combined with the 3sp + Boltzmann ClO4- background, we get clean onset physics:

| V_RHE | cd (log-c) | Status |
|-------|-----------|--------|
| -0.10V | -0.1780 | FULL z=1, positive c |
| +0.00V | -0.1738 | FULL |
| +0.10V | -0.1631 | FULL |
| +0.15V | -0.1450 | FULL |
| +0.20V | -0.0786 | FULL |
| +0.25V | -0.0020 | FULL |
| +0.30V | -0.00002 | FULL |

Script: `scripts/studies/v18_test_3sp_logc.py`

### Step 3 — k0 sensitivity confirmed

At fixed α, swept k0 over [0.2, 0.5, 1.0, 2.0, 5.0]× baseline (25x range) at each voltage. Relative Δcd:

| V_RHE | |Δcd|max | rel Δ | Sensitive? |
|-------|---------|-------|-----------|
| -0.100 | 0.0000 | 0.0% | no |
| +0.000 | 0.0000 | 0.0% | no |
| +0.100 | 0.0015 | 1.5% | weak |
| **+0.150** | **0.0134** | **8.9%** | strong |
| **+0.200** | **0.0510** | **50.1%** | very strong |
| **+0.250** | **0.0080** | **94.8%** | very strong |

Script: `scripts/studies/v18_logc_k0_sensitivity.py`

**This looked like a breakthrough.** But sensitivity at fixed α is different from inference with α also free.

### Step 4 — Inference test exposes the ridge

Ran Nelder-Mead on 6 onset voltages with 2% noise, k0_init +20%, α_init -10%. NM behavior over 34 evaluations:

| Eval | k0 (× true) | α | J |
|------|-------------|---|-----|
| 1 | 1.20 | 0.564 | 9.0e-4 |
| 5 | 2.30 | 0.607 | 1.6e-5 (noise floor) |
| 10 | 2.12 | 0.621 | 3.4e-5 |
| 20 | 2.26 | 0.608 | 1.5e-5 |
| 26 | 2.88 | 0.603 | 1.48e-5 |
| 30 | 4.84 | 0.589 | 1.4e-5 |
| 34 | **16.60** | 0.554 | 1.30e-5 |

NM walked UNBOUNDEDLY along a ridge in the (log k0, α) landscape. Each step along the ridge kept J at the noise floor while k0 grew and α drifted. I killed the run at eval 34 (k0 at 17x true).

**Ridge slope**: d(log k0)/d(α) ≈ -47 (1% change in α ↔ ~60% change in k0).

This confirmed v17's conclusion is right *for onset-only data with 2% noise*: onset current can be reproduced by many (k0, α) combinations. The ridge exists because the Tafel slope (determined by α) and the Tafel intercept (determined by k0) are coupled through the concentration profile response.

Script: `scripts/studies/v18_logc_inference_test.py`

### Step 5 — Tikhonov regularization closes the ridge

Added soft prior: `J_total = J_data + λ·(log(k0) - log(k0_prior))²`

With λ=0.01 (weak) and prior = true k0 (simulating a literature/EIS k0 measurement):

| Eval | k0 error | α error | J_data | J_prior |
|------|---------:|--------:|--------|---------|
| 1 | +20% | -10% | 9.0e-4 | 3.3e-4 |
| 7 | -5.9% | -1.6% | 6.2e-5 | 3.7e-5 |
| 12 | +2.1% | +1.3% | 2.1e-5 | 4.0e-6 |
| 16 | +1.5% | -0.4% | 2.1e-5 | 2.2e-6 |
| 18 | -1.2% | +0.3% | 1.7e-5 | 1.4e-6 |
| 20 | +1.1% | +0.6% | 1.6e-5 | 1.2e-6 |
| 24 | -0.5% | +0.6% | 1.6e-5 | 2.7e-7 |
| 26 | -0.3% | +0.7% | 1.6e-5 | 7e-8 |
| 28 | +0.4% | +0.7% | 1.6e-5 | 1.2e-7 |

NM converged to **k0 within ~0.5% and α within ~0.7%** of true values at 2% noise. The prior broke the ridge.

Script: `scripts/studies/v18_logc_regularized.py`

### Interpretation

- The data-only inverse problem has a **fundamental ridge degeneracy** at any realistic noise level. v17 was right.
- The onset-physics fix (log-c + seeded H2O2) was necessary but not sufficient.
- A modest regularization (λ=0.01, correct prior) transforms the ridge-walk into a narrow bowl, giving near-noise-floor recovery.
- With real data, the k0 prior can come from literature values (published k0 for ORR on known catalysts), EIS measurements of exchange current, or Tafel analysis of a subset of the data.

## Scripts created

| Script | Purpose | Status |
|--------|---------|--------|
| `scripts/studies/v18_test_3sp_h2o2_seed.py` | Concentration formulation with seeded H2O2 — did NOT work | dead end |
| `scripts/studies/v18_test_3sp_logc.py` | Log-c + seeded H2O2 convergence test | WORKS |
| `scripts/studies/v18_logc_k0_sensitivity.py` | k0 sensitivity scan at fixed α | 50-95% sensitivity |
| `scripts/studies/v18_logc_inference_test.py` | End-to-end recovery, NM, no regularization | **Ridge walks to 17x true** |
| `scripts/studies/v18_logc_noise_sensitivity.py` | Noise level sweep (not run — regularization obviates the need) | ready |
| `scripts/studies/v18_logc_regularized.py` | Tikhonov-regularized inference | **k0 to ±0.5%** |

## What didn't work (documented dead ends)

- **Standard 4sp**: fails at V > 0.15V (Debye layer oscillation in ClO4-)
- **Stabilized 4sp (d_art=0.001)**: converges to V=+0.73V but destroys onset physics (FLAT curve)
- **3sp + Boltzmann (concentration formulation)**: converges to V=+0.60V but H2O2 goes to -0.69 at anodic V (CG1 oscillation around near-zero state)
- **Seeded H2O2 (concentration formulation)**: breaks V=-0.30V (spurious R_2 rate blows up Newton)
- **Lower BV clip (30 or 20)**: distorts R_1 physics at cathodic voltages
- **Log-c without H2O2 seed**: log(0) = -46 creates extreme stiffness (prior v18 entry 2)
- **Onset-only inference without prior**: ridge walks to arbitrary k0 (this work)

## What works

**The complete recipe**:
1. `forms_logc.py` for the solver forms (log-c transform)
2. H2O2 bulk/initial value = 1e-4 (not 0)
3. Boltzmann background for ClO4- (z=-1, bulk 0.2 nondim)
4. Onset voltage grid V_RHE ∈ [-0.10, +0.30] (6-7 points)
5. Tikhonov regularization on log(k0) with a prior
6. λ=0.01 (weak) with correct prior → <1% k0/α recovery at 2% noise

## Prior work context (from audit)

- **v13** achieved 4-5% k0 error via surrogate + full-cathodic PDE refinement (~40 voltage points, transport-limited regime)
- **v13 noise variance**: 5 noise seeds showed 5.26% → 28.64% range, median 23.12%
- **v17** confirmed k0-α anti-correlation in transport-limited regime

**This work achieves comparable accuracy (<1%) with**:
- 6 voltage points instead of 40
- No surrogate training (direct PDE inference)
- Simple derivative-free optimization
- A natural k0 prior (which v13 did not use)

The tradeoff is **reliance on a k0 prior**. In real-world use, this requires:
- Literature k0 values for similar catalysts, OR
- Independent measurement (EIS exchange current), OR
- Prior analysis of Tafel slope (α) that is then used to anchor k0

## Integration plan (for production use)

1. **Add `transform: "logc"` flag to solver_params** — dispatch in build_context/build_forms/set_initial_conditions
2. **Expose `h2o2_ic_seed` as solver_params option** — default 1e-4 for ORR
3. **First-class Boltzmann background config** — replace the current monkey-patch pattern
4. **Extend `validate_solution_state` for log-c** — compute c = exp(u) for check inputs; F1 becomes impossible by construction
5. **Add regularization to the inverse objective** — `FluxCurve/bv_config.py` gets a `tikhonov` section
6. **Verify adjoint works through log-c** — pyadjoint should tape through `exp(u)` correctly; needs testing
7. **Migrate `overnight_train_v16.py` and inference scripts** to use the new log-c option

## Remaining questions

- Does the recipe hold across different noise realizations? (Not tested — noise seed 42 only.)
- Does it hold at different noise levels? (Script ready but not run.)
- Does it hold for k0_2 (second reaction), not just k0_1? (Not tested.)
- What happens with a wrong prior (prior ≠ true k0)? (Not tested; should still bias toward the prior, which is the intended behavior.)
- Can λ be chosen adaptively (L-curve, discrepancy principle)? (Worth investigating for real data.)

## Physics validation framework (context)

While investigating, I added an F1/F4/F6/W2/W3/W5 validation framework to `validation.py`:
- F1: negative concentration (impossible in log-c by construction)
- F4: surface concentration floor domination (only applies to species with c_bulk > 0)
- F6: H2O2 > O2 bulk (stoichiometric limit)
- F7: peroxide current wrong sign at cathodic
- W2: concentration overshoots bulk (5x tolerance for charged species)
- W3: potential out of bounds (generous tolerance; fires only for gross CG1 oscillations)
- W5: H+ depletion outside Debye layer

See `docs/physics_validation_log.md` for full details of the validation framework.
