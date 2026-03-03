# Convergence Research Report: PNP-BV Electrochemistry Solver

**Date:** 2026-02-28
**Author:** Agent R (Researcher)
**Context:** 4-species PNP system (O2, H2O2, H+, ClO4-) with Butler-Volmer BCs,
Firedrake FEM + PETSc SNES, MUMPS direct solver, l2 linesearch.

---

## 1. Convergence Tricks for Coupled PNP with Exponential BV Kinetics

### 1.1 The Core Problem

The Butler-Volmer exponential `exp(-alpha * eta_hat)` grows as `exp(|alpha * eta_hat|)`.
At V_RHE = -1.0 V, the overpotential `eta = V_RHE - 0.695 ~ -1.695 V` gives
`eta_hat = eta / V_T ~ -66` (at 25C), so the BV exponent reaches `exp(66 * 0.627) ~ 10^18`.
This makes the Jacobian extremely ill-conditioned.

### 1.2 Implemented Strategies (Already in `bv_solver.py`)

The codebase already has three continuation strategies:

1. **Voltage continuation** (`solve_bv_with_continuation`): Ramps eta from 0 to target in
   `eta_steps` increments. Has bisection on failure (up to 6 levels). This is the workhorse.

2. **Pseudo-transient continuation** (`solve_bv_with_ptc`): Adds mass-matrix regularization
   via (c-c_old)/dt, adaptively grows dt. Current limitation: dt is baked into the UFL form
   as a float, not as a mutable `fd.Constant`, so the PTC inner loop cannot truly vary the
   pseudo-timestep.

3. **Charge continuation** (`solve_bv_with_charge_continuation`): Two-phase approach --
   solve neutral (z=0) then ramp charges. Avoids Poisson stiffness `(lambda_D/L)^2 ~ 9e-8`.

### 1.3 New Strategies to Consider

#### A. True Pseudo-Transient Continuation with Mutable dt

**Priority: HIGH.** The current PTC implementation has a structural limitation -- `dt_model` is
a Python float embedded in the UFL form. To get genuine PTC behavior:

```python
dt_const = fd.Constant(dt_model)  # mutable!
# In build_forms, replace float(dt_model) with dt_const in the weak form
# Then in PTC loop:
dt_const.assign(dt_ptc_new)
```

This enables the classic SER (Switched Evolution/Relaxation) strategy from Coffey, Kelley & Keyes:

```
dt_{k+1} = dt_k * (||F(x_{k-1})|| / ||F(x_k)||) * growth_factor
```

Start with `dt_ptc = 0.001` (heavily regularized), grow toward `dt_ptc > 1e6` (steady state).
The mass matrix term `(1/dt_ptc) * M * (u - u_prev)` dominates at small dt, making the
Jacobian well-conditioned (essentially a scaled identity), then fades as dt grows.

**Practical parameters for PNP-BV:**
- `dt_initial`: 1e-3 to 1e-2 (start very regularized)
- `growth`: 1.5 to 2.0
- `max_dt`: 1e8 (generous, let convergence criterion stop it)
- `ss_tol`: 1e-8 (relative change in L2 norm)
- `max_ptc_steps`: 500

#### B. Damped Newton via Line Search Tuning

**Priority: MEDIUM.** The current solver uses `l2` line search. Consider:

- **Backtracking (`bt`)**: More conservative, guaranteed descent. Good for preventing
  overshoot on the exponential BV terms. Key parameters:
  ```python
  "snes_linesearch_type": "bt",
  "snes_linesearch_maxstep": 1.0,  # prevent huge Newton steps
  "snes_linesearch_minlambda": 1e-12,  # allow very small steps
  ```

- **Critical point (`cp`)**: Minimizes `F(x) . Y` where `Y` is the search direction.
  Can be more effective than l2 when the residual has a complex landscape.

- **Basic (no line search)**: Fastest per iteration but least robust. Consider for
  the final Newton steps when close to convergence.

**Recommendation:** Try `bt` line search first. It is the PETSc default for `newtonls`
for good reason -- it has built-in safeguards against large steps that cause the
BV exponentials to blow up.

#### C. NGMRES or Composite Solver

**Priority: LOW (advanced).** PETSc's `SNESNGMRES` combines the last m iterates plus a
new fixed-point iteration to minimize the residual. It can be composed with Newton:

```python
"snes_type": "ngmres",
"npc_snes_type": "newtonls",
"npc_snes_max_it": 3,
```

This uses 3 Newton steps as a "nonlinear preconditioner" for NGMRES. The NGMRES outer
loop provides globalization. This pattern is documented as improving robustness for
highly nonlinear problems.

**Note:** Firedrake's interface to composite SNES types may be limited. Test with
`solver_parameters` dictionary first.

#### D. Arc-Length Continuation

**Priority: LOW (for future).** PETSc 3.24+ has `SNESNEWTONAL` for arc-length continuation.
This traces the solution curve in (eta, u) space using a constraint:

```
||delta_u||^2 + psi * delta_lambda^2 * ||F_ext||^2 = (delta_s)^2
```

Benefits: can follow turning points and S-shaped I-V curves. Not needed for monotonic
voltage sweeps, but useful if the solver encounters limit points at extreme overpotentials.

Implementation requires `SNESNewtonALSetFunction` with a tangent load callback, which is
non-trivial in Firedrake. Defer unless standard continuation fails.

#### E. Exponent Clipping / Regularization

**Priority: HIGH (quick win).** The `bv_convergence` config already supports
`clip_exponent` and `regularize_concentration`. Ensure these are active:

```python
"bv_convergence": {
    "clip_exponent": 50.0,      # clip |alpha * eta_hat| to [-50, 50]
    "regularize_concentration": 1e-10,  # floor for c in BV expression
}
```

`exp(50) ~ 5e21` is still large but within float64 range. Clipping at 50 prevents
overflow without affecting the physics at practical overpotentials.

For the cathodic branch at very negative eta: the cathodic exponent
`exp(-alpha_c * eta_hat)` is the one that blows up (eta_hat < 0, -alpha_c < 0, so
the product is positive and large). Clipping this is physically justified because
at extreme overpotentials, the reaction becomes mass-transport limited anyway.

---

## 2. Adaptive Continuation Strategies for Voltage Sweeps

### 2.1 Current Implementation

The existing `solve_bv_with_continuation` uses uniform voltage steps with bisection on
failure (up to 6 levels). This is solid but can be improved.

### 2.2 Recommended Improvements

#### A. Non-Uniform (Graded) Voltage Steps

**Priority: HIGH.** The BV exponential changes slowly near equilibrium and rapidly at
large |eta|. Use geometric or power-law spacing:

```python
# Geometric grading: more steps at large |eta|
t = np.linspace(0, 1, eta_steps + 1)[1:]
path = eta_target * t**grading_power  # grading_power = 0.5 clusters toward end

# Or hyperbolic tangent grading:
path = eta_target * np.tanh(3.0 * t) / np.tanh(3.0)
```

This puts more steps where the exponential is steepest, reducing bisection triggers.

#### B. Adaptive Step Sizing Based on Newton Iterations

**Priority: HIGH.** After each successful step, count Newton iterations `n_its`:

```python
if n_its <= 3:
    step_factor *= 1.5   # easy solve, can take bigger steps
elif n_its <= 6:
    step_factor *= 1.0   # keep current step size
elif n_its <= 10:
    step_factor *= 0.5   # getting hard, slow down
else:
    step_factor *= 0.25  # very hard, much smaller steps
```

This is the standard approach in semiconductor device simulation (continuation methods
in semiconductor device simulation, Scharfetter-Gummel literature).

#### C. Predictor Step

**Priority: MEDIUM.** Use linear extrapolation from the last two solutions:

```python
# After solving at eta_k and eta_{k-1}:
U_predicted = U_k + (eta_{k+1} - eta_k) / (eta_k - eta_{k-1}) * (U_k - U_{k-1})
ctx["U"].assign(U_predicted)
```

This gives Newton a better starting point, reducing iterations by 30-50% in typical
PNP problems. The predictor-corrector approach is standard in continuation methods.

**Caution:** The predictor can overshoot, especially for concentrations that must remain
positive. Clip the predicted solution: `c_predicted = max(c_predicted, epsilon)`.

#### D. Improved Bisection

**Priority: MEDIUM.** The current bisection always tries the midpoint. A better approach:

1. On failure at `eta_target`, try `eta_current + 0.75 * (eta_target - eta_current)` first
   (optimistic).
2. If that fails, try the midpoint.
3. If midpoint fails, try `eta_current + 0.25 * (eta_target - eta_current)` (conservative).

This avoids unnecessary bisection depth when the solver is close to converging.

---

## 3. Bikerman Steric Exclusion Model

### 3.1 Physics of the Steric Parameter

The Bikerman model adds an entropy term for solvent packing to prevent unphysical
concentration blow-up at the electrode surface. The steric potential is:

```
Strc(r) = ln(Gamma(r) / Gamma_B)
```

where `Gamma(r)` is the void (solvent) volume fraction. The modified flux becomes:

```
J_i = -D_i * grad(c_i) + beta_i * c_i * grad(phi) - (v_i/v_0) * c_i * grad(Strc)
```

The last term is an "entropic pressure" that resists crowding.

### 3.2 Physically Reasonable Values for the Steric Parameter `a`

The parameter `a` represents the effective ion diameter. Physically reasonable values:

| Species | Crystal radius | Hydrated radius | Effective `a` |
|---------|---------------|-----------------|---------------|
| H+      | ~0.25 A       | 2.82 A          | 3-6 A         |
| ClO4-   | 2.40 A        | 3.38 A          | 3-7 A         |
| O2      | ~1.5 A (vdW)  | N/A             | 3-4 A         |
| H2O2    | ~1.5 A (vdW)  | N/A             | 3-4 A         |
| H2O     | 1.4 A         | N/A             | 2.75 A        |

In the nondimensionalized model, `a` is typically `a_dim / L_ref`. With `L_ref = 300 um`:
- `a = 3 A / 300 um = 1e-6` in nondim units

However, the solver parameter `a` in the codebase appears to be on a different scale
(0.01 to 0.20 nondim). This suggests it may be `a_eff = a_dim / lambda_D` or a modified
steric volume fraction parameter, not a raw ion diameter. **Clarify the nondimensionalization
of `a` in the code before changing values.**

### 3.3 Effect of Large `a` on Solver Conditioning

**Why `a = 0.20` diverges but `a <= 0.10` works:**

The steric potential `Strc = ln(Gamma/Gamma_B)` diverges as `Gamma -> 0` (all space
occupied by ions). At large `a`:

1. The void fraction `Gamma = 1 - sum(a_i^3 * c_i)` approaches zero faster, creating
   a near-singular logarithm.
2. The gradient `grad(Strc) = -grad(Gamma)/Gamma` blows up, introducing extreme stiffness
   in the Jacobian.
3. At large overpotentials, the Debye layer concentrations are already extreme; steric
   repulsion creates a sharp boundary layer within the Debye layer.

**The Liu & Eisenberg (2020) paper recommends continuation on the steric parameter:**

```python
# Steric continuation: ramp a from 0 to a_target
for a_step in np.linspace(0, a_target, n_steric_steps + 1)[1:]:
    for i in range(n_species):
        ctx["a_consts"][i].assign(a_step)
    solver.solve()
```

This is directly analogous to the charge continuation already implemented. Start with
`a = 0` (dilute solution, no steric effects), then gradually increase to the target value.

**Implementation requirement:** The `a` values must be stored as `fd.Constant` objects
(not Python floats) in the weak form, similar to how `z_consts` are handled in
`solve_bv_with_charge_continuation`.

### 3.4 Asymmetric Steric Parameters

The original Bikerman model uses one global ion size. The modified model allows
species-specific sizes `a_i`:

```
Gamma = 1 - sum_i (v_i * c_i)  where v_i = (4/3) * pi * (a_i/2)^3
```

This is physically important because H+ is much smaller than ClO4-. The current codebase
already supports per-species `a_vals` (the solver_params[6] entry is a list). This is
the right design.

**Practical recommendation:** Use asymmetric steric parameters:
- `a_H+ = 0.02` (small, minimal steric effect for the tiny proton)
- `a_ClO4- = 0.08` (larger anion)
- `a_O2 = 0.03` (neutral, small molecule)
- `a_H2O2 = 0.03` (neutral, small molecule)

These should be more numerically tractable than `a = 0.20` for all species.

### 3.5 Steric Regularization at Extreme Overpotentials

At very cathodic potentials, cation (H+) accumulation in the Debye layer is the primary
source of blow-up. Steric exclusion physically limits the maximum concentration:

```
c_max = 1 / v_i  (when Gamma -> 0)
```

This acts as a **natural regularizer** -- it prevents the unphysical infinite concentrations
that cause Newton divergence. However, this regularization is helpful only when `a` is
moderate. When `a` is too large, the steric term itself becomes the source of numerical
difficulty.

**Sweet spot for regularization:** `a` values that give `c_max` about 10-100x the bulk
concentration. This prevents blow-up without making the steric gradient too stiff.

---

## 4. Regularization in PDE-Constrained Optimization

### 4.1 Tikhonov Regularization

For the inverse problem `min_p ||F(u(p)) - d||^2`, Tikhonov regularization adds:

```
min_p ||F(u(p)) - d||^2 + lambda * ||p - p_ref||^2
```

where `lambda` is the regularization weight, `p` is the parameter vector (e.g., D, kappa),
and `p_ref` is a prior/reference value.

### 4.2 Methods for Choosing lambda

#### A. L-Curve Method

Plot `log(||r||)` vs `log(||p||)` for a range of lambda values. The "corner" of the
L-shaped curve gives the optimal lambda that balances data fit and regularization.

**Corner detection:** Use maximum curvature (Hansen & O'Leary 1993) or minimum-product
method. In practice:

```python
lambdas = np.logspace(-10, 2, 50)
residuals, param_norms = [], []
for lam in lambdas:
    p_opt = solve_inverse(data, lam)
    residuals.append(compute_residual(p_opt))
    param_norms.append(norm(p_opt - p_ref))

# Find corner by maximum curvature
curvature = compute_curvature(np.log(residuals), np.log(param_norms))
lambda_opt = lambdas[np.argmax(curvature)]
```

**Cost:** Requires solving the inverse problem ~50 times. Expensive but reliable.

#### B. Morozov Discrepancy Principle

Choose lambda so that the residual equals the noise level:

```
||F(u(p_lambda)) - d|| = delta  (noise estimate)
```

**Requirement:** Must have an estimate of the measurement noise delta. For synthetic
data, this is known. For real EIS/voltammetry data, estimate from experimental
reproducibility.

**Practical implementation:** Solve a 1D root-finding problem (bisection on lambda).

#### C. Generalized Cross-Validation (GCV)

Minimize `GCV(lambda) = ||r||^2 / (trace(I - A_lambda))^2` where `A_lambda` is the
influence matrix. Does not require noise estimate.

**For PDE-constrained optimization:** The trace computation requires the Hessian of
the Lagrangian, which can be approximated via randomized trace estimators.

### 4.3 Practical Recommendations for This Project

For the `make_diffusion_objective_and_grad` and `make_robin_kappa_objective_and_grad`
functions:

1. **Start with L-curve for D inference:** D has physical bounds (1e-10 to 1e-8 m^2/s).
   Use `p_ref = D_literature` and sweep lambda from 1e-10 to 1.0.

2. **Morozov for kappa inference with noisy data:** When noise level is known (e.g.,
   from `Forward/noise.py`), Morozov is computationally cheaper than L-curve.

3. **Regularization for multi-parameter problems:** When inferring (D, kappa) jointly,
   use separate regularization weights for each parameter:
   ```
   lambda_D * ||D - D_ref||^2 + lambda_kappa * ||kappa - kappa_ref||^2
   ```
   This accounts for different scales and sensitivities.

4. **Log-transform for strictly positive parameters:** Instead of regularizing D directly,
   regularize `log(D)`. This ensures D > 0 and puts all parameters on similar scales:
   ```
   min_theta ||F(u(exp(theta))) - d||^2 + lambda * ||theta - theta_ref||^2
   ```

---

## 5. Multi-Species BV Systems: Coupled O2/H2O2 Reactions

### 5.1 Coupling Structure

The two-reaction system:
- R1: O2 + 2H+ + 2e- -> H2O2 (oxygen reduction)
- R2: H2O2 + 2H+ + 2e- -> 2H2O (peroxide reduction)

Creates coupling through:
1. **Shared intermediate H2O2:** R1 produces it, R2 consumes it.
2. **Shared electrode potential eta:** Both reactions respond to the same overpotential.
3. **Shared H+ consumption:** Both consume protons.

### 5.2 Numerical Challenges

The two BV rates have different `E_eq` values and kinetics. At intermediate
overpotentials, R1 may be in the mass-transport-limited regime while R2 is still
in the kinetic regime. This creates a two-scale problem.

### 5.3 Strategies for Robust Convergence

#### A. Reaction Continuation

Instead of turning on both reactions simultaneously, ramp them sequentially:

```python
# Phase 1: Solve with R1 only (O2 -> H2O2)
# Phase 2: Gradually introduce R2 (H2O2 -> H2O) via reaction_factor

for rxn_factor in np.linspace(0, 1, rxn_steps + 1)[1:]:
    k0_R2_current = k0_R2_target * rxn_factor
    # Update BV rate expression for R2
    solver.solve()
```

This is analogous to charge continuation but for reaction kinetics.

**Implementation:** Store `k0` for each reaction as `fd.Constant` objects, enabling
gradual ramping without form/solver rebuild.

#### B. Sequential Voltage Sweeps per Reaction

Sweep R1 first (single reaction, simpler problem), then add R2:

1. Solve single-species O2 BV to target eta.
2. Add H2O2 species with source from R1, solve diffusion-only.
3. Turn on R2 BV kinetics for H2O2, re-solve.
4. Iterate until self-consistent.

This is a Gummel-like (block-iterative) approach. Each sub-problem is easier than
the monolithic coupled system.

#### C. Staggered vs Monolithic Solve

**Monolithic (current approach):** All species and reactions in one Newton system. Fast
convergence when it works, but fragile at extreme conditions.

**Staggered (Gummel):** Iterate between:
- Fix concentrations, solve Poisson for phi.
- Fix phi, solve Nernst-Planck for concentrations.
- Repeat until self-consistent.

The Gummel method is more robust but converges only linearly (not quadratically like
Newton). It is most effective when coupling is weak. For PNP with strong coupling
(Debye layer), Gummel may converge very slowly or not at all.

**Recommendation:** Stick with monolithic Newton but use the other continuation
strategies (voltage, charge, steric, reaction) for globalization. Gummel is a fallback
option.

---

## 6. Performance: Addressing 500-Step Timeout at 300s

### 6.1 Root Cause Analysis

If 200 steps complete in ~85s (0.425 s/step) but 500 steps timeout at 300s, the
per-step cost is increasing at large |eta|. Likely causes:

1. **More Newton iterations per step:** At large |eta|, the BV exponential makes the
   Jacobian more ill-conditioned, requiring more Newton (SNES) iterations.
2. **More line search backtracks:** The l2 line search may need many iterations to
   find a good step length.
3. **MUMPS fill-in:** The Jacobian sparsity pattern doesn't change, but the magnitude
   of entries affects pivoting and scaling. `mat_mumps_icntl_8: 77` (auto-scaling)
   helps but adds overhead.

### 6.2 Actionable Recommendations

#### A. Reduce Time Steps per Continuation Step

The solver currently runs `num_steps = t_end / dt` time steps at each voltage. For
continuation, only 1-3 time steps per voltage increment may suffice (the previous
solution is already a good initial guess). This is the single biggest speedup opportunity.

```python
# Instead of full time stepping at each eta:
num_steps_per_continuation = 3  # or even 1
```

#### B. SNES Tolerances

Tighten SNES stopping criteria to avoid unnecessary iterations:

```python
"snes_atol": 1e-8,    # absolute tolerance
"snes_rtol": 1e-8,    # relative tolerance
"snes_max_it": 15,    # cap Newton iterations
```

If Newton needs >15 iterations, it's better to take a smaller voltage step than to
keep iterating.

#### C. MUMPS Optimizations

```python
"mat_mumps_icntl_14": 50,     # increase working space (% above estimated)
"mat_mumps_icntl_24": 1,      # null pivot detection
"mat_mumps_icntl_8": 77,      # auto row/col scaling (already used)
"mat_mumps_cntl_1": 0.01,     # relative threshold for pivoting (lower = faster)
```

#### D. Consider Iterative Solver for Inner Linear System

For large meshes, MUMPS (direct) becomes expensive. An iterative solver with a
good preconditioner can be faster:

```python
"ksp_type": "gmres",
"pc_type": "ilu",    # or "asm" for domain decomposition
"ksp_rtol": 1e-6,
"ksp_max_it": 100,
```

For PNP, ILU(1) or block ILU with field-split (splitting concentration from potential)
is effective. However, this is a bigger change and may not be needed on the current
2D problem.

---

## 7. Summary of Prioritized Recommendations

| Priority | Action | Expected Benefit | Implementation Effort |
|----------|--------|------------------|-----------------------|
| 1 (HIGH) | Non-uniform voltage stepping | 2-3x fewer bisections | Small (change `np.linspace` to graded) |
| 2 (HIGH) | Adaptive step sizing by Newton iteration count | 2x fewer total steps | Small (add iteration counter) |
| 3 (HIGH) | Exponent clipping at 50 | Prevent overflow at extreme eta | Tiny (config change) |
| 4 (HIGH) | Reduce time steps per continuation step | 3-5x speedup | Small (parameter change) |
| 5 (HIGH) | Make dt_model a mutable fd.Constant for true PTC | Robust convergence at extreme eta | Medium (modify build_forms) |
| 6 (MED)  | Predictor step (linear extrapolation) | 30-50% fewer Newton iters | Small-medium |
| 7 (MED)  | Switch to `bt` line search | Better robustness | Tiny (config change) |
| 8 (MED)  | Steric continuation (ramp `a` from 0) | Enable a=0.20 | Medium (like charge continuation) |
| 9 (MED)  | Store k0 as fd.Constant for reaction continuation | Decouple multi-reaction | Medium |
| 10 (LOW) | NGMRES composite solver | Robustness at extreme conditions | Medium (PETSc config) |
| 11 (LOW) | L-curve for regularization weight | Principled lambda selection | Medium |
| 12 (LOW) | Arc-length continuation | Handle turning points | High (PETSc 3.24+) |

---

## 8. References

- Coffey, Kelley & Keyes, "Pseudo-transient continuation and differential-algebraic equations," SIAM J. Sci. Comput. (2003)
- Liu & Eisenberg, "Poisson-Nernst-Planck-Bikerman Model," Entropy 22(5):550 (2020)
- Hansen & O'Leary, "The use of the L-curve in the regularization of discrete ill-posed problems," SIAM J. Sci. Comput. (1993)
- Scharfetter & Gummel, "Large-signal analysis of a silicon Read diode oscillator," IEEE Trans. Electron Devices (1969)
- PETSc SNES documentation: https://petsc.org/release/manual/snes/
- FEniCS-arclength: https://fenics-arclength.readthedocs.io/
- PETSc TSPSEUDO: https://petsc.org/release/manualpages/TS/TSPSEUDO/
- PETSc SNESNEWTONAL: https://petsc.org/release/manualpages/SNES/SNESNEWTONLS/
