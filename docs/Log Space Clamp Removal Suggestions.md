# Suggestions for Removing or Reducing Clamps in the PNP Log-Concentration Solver

## Purpose

This document proposes concrete implementation directions for reducing or eliminating the remaining clamps in the current log-concentration PNP solver.

The current production stack uses

```text
u_i = log(c_i)
c_i = exp(u_i)
```

as the primary dynamic variable. This already removes negative concentration Newton iterates, but it does **not** remove all overflow and underflow issues because the bulk weak form still needs the actual concentration `c_i = exp(u_i)` as a coefficient.

The main goal should not be “remove every guard at all costs.” The goal should be:

1. avoid changing the physical residual through artificial clamps,
2. preserve positivity and mass conservation,
3. prevent floating-point overflow during bad nonlinear iterations,
4. improve solver stability without hiding real parameter sensitivity.

The most important distinction is:

```text
A clamp changes the equation being solved.
A line search / trust region rejects bad Newton steps but preserves the equation.
```

That distinction should guide the implementation.

---

## Current situation

The log-concentration transition changed the dynamic unknown from concentration to log concentration:

```text
c_i = exp(u_i)
```

The Nernst–Planck flux becomes

```text
J_i = -D_i exp(u_i) * (grad(u_i) + z_i F / RT * grad(phi))
```

or equivalently

```text
J_i = -D_i c_i * (grad(u_i) + z_i F / RT * grad(phi))
```

The backward Euler time term is currently written in concentration space:

```text
(exp(u_i^{n+1}) - exp(u_i^n)) / dt
```

This is the correct conservative form. Do **not** replace it with

```text
exp(u_i^{n+1}) * (u_i^{n+1} - u_i^n) / dt
```

because that would no longer preserve discrete species mass in the same way.

The current remaining guards are approximately:

```text
1. bulk u clamp, currently around +/-30
2. eta clip in the Butler–Volmer exponent
3. legacy surface concentration clamp, mostly avoided by bv_log_rate=True
```

The log-rate BV branch already solves the surface concentration clamp problem for the boundary reaction. The remaining hard problem is the bulk appearance of `exp(u_i)`.

---

# Recommendation 1: Remove the lower bulk u-clamp

## Problem

A symmetric clamp such as

```python
u_safe = min_value(max_value(u_i, -30.0), 30.0)
c_i = exp(u_safe)
```

creates an artificial concentration floor:

```text
c_i >= exp(-30)
```

That is dangerous for product species whose physical concentration can be extremely small or whose bulk concentration is zero. It can create artificial sinks/sources and can distort the inverse problem.

The lower clamp is not needed for overflow protection. Very negative `u_i` values simply underflow toward zero, which is usually much less damaging than flooring them to a nonzero concentration.

## Suggested change

Replace the symmetric clamp with an **upper-only overflow guard**:

```python
U_MAX = 650.0  # safely below double precision exp overflow near 709
c_i = exp(min_value(u_i, U_MAX))
```

Do not clamp the lower side:

```python
# Do not do this unless a separate regularization is explicitly intended:
u_safe = max_value(u_i, U_MIN)
```

## Why this is better

- Avoids artificial lower concentration floors.
- Preserves near-zero product concentrations better.
- Still prevents floating-point overflow.
- Separates true overflow safety from numerical regularization.

## Implementation target

Look for code that currently does something like:

```python
u_clamped = min_value(max_value(u_i, -u_clip), u_clip)
c_i = exp(u_clamped)
```

and change it to:

```python
u_upper_safe = min_value(u_i, U_MAX)
c_i = exp(u_upper_safe)
```

Use this in the bulk residual, Poisson source, and any coefficient evaluation where the purpose is only floating-point safety.

---

# Recommendation 2: Replace eta clipping with stable log-rate evaluation plus step rejection

## Problem

The current eta clip appears to be used to prevent overflow in the Butler–Volmer exponentials:

```text
exp(alpha * n_e * eta_scaled)
```

However, clipping `eta_scaled` changes the reaction rate. This may hide exactly the voltage sensitivity we are trying to infer.

The current clip value of around +/-50 may also be much more conservative than the actual floating-point limit. Double precision overflow for `exp(x)` occurs near `x = 709`, not near `x = 50`.

So the eta clip is likely acting as a stiffness control, not merely an overflow guard.

## Suggested change

Evaluate the BV rate using a stable log-difference form.

If

```text
R = exp(a) - exp(b)
```

where

```text
a = log cathodic rate
b = log anodic rate
```

then compute:

```python
m = max_value(a, b)
R = exp(m) * (exp(a - m) - exp(b - m))
```

This prevents unnecessary overflow when one rate is much larger than the other.

## Near-cancellation case

When `a` and `b` are close, the expression

```python
exp(a - m) - exp(b - m)
```

can suffer cancellation. If Firedrake/UFL supports a stable `expm1` equivalent, use it. Conceptually:

```python
if a >= b:
    R = exp(a) * (-expm1(b - a))
else:
    R = -exp(b) * (-expm1(a - b))
```

In UFL this may need to be written using `conditional`.

Approximate UFL-style pseudocode:

```python
delta = a - b

R_pos = exp(a) * (-expm1(-delta))   # for delta >= 0
R_neg = -exp(b) * (-expm1(delta))   # for delta < 0

R = conditional(ge(delta, 0.0), R_pos, R_neg)
```

If `expm1` is unavailable in UFL, use the `m = max(a,b)` version first. It is still better than independently evaluating `exp(a)` and `exp(b)`.

## Overflow handling

Do not clip eta directly. Instead, reject or damp the Newton step if the maximum log-rate is too large:

```text
max(a, b) > RATE_LOG_MAX
```

where, for double precision,

```python
RATE_LOG_MAX = 650.0
```

This changes solver behavior, not the physical residual.

## Implementation target

Replace any code of the form:

```python
eta_safe = min_value(max_value(eta_scaled, -eta_clip), eta_clip)
rate = k0 * c_cat * exp(-alpha * n_e * eta_safe) \
     - k0 * c_anod * exp((1.0 - alpha) * n_e * eta_safe)
```

with a log-rate assembly:

```python
log_cath = log(k0) + log_c_cat + other_terms - alpha * n_e * eta_scaled
log_anod = log(k0) + log_c_anod + other_terms + (1.0 - alpha) * n_e * eta_scaled

m = max_value(log_cath, log_anod)
rate = exp(m) * (exp(log_cath - m) - exp(log_anod - m))
```

Then add nonlinear step rejection/backtracking if `m` gets too large.

---

# Recommendation 3: Add a Newton trust-region / line-search rule on u

## Problem

Many clamps are protecting against bad intermediate Newton iterates, not against the final solution.

If Newton proposes a step that sends some `u_i` to `+1000`, the right answer is not to silently clamp the residual. The right answer is to reject or shrink the step.

## Suggested change

Add a step acceptance rule based on quantities such as:

```text
max(u_i) <= U_MAX
max(abs(delta_u_i)) <= DELTA_U_MAX
max(log_BV_rate) <= RATE_LOG_MAX
residual norm decreases sufficiently
```

Suggested starting values:

```python
U_MAX = 650.0
DELTA_U_MAX = 5.0  # try 5 first; maybe 10 if too restrictive
RATE_LOG_MAX = 650.0
```

If any criterion fails, reduce the Newton step length:

```text
u_trial = u_old + lambda * delta_u
phi_trial = phi_old + lambda * delta_phi
```

with

```text
lambda = 1, 1/2, 1/4, 1/8, ...
```

until the trial step is acceptable.

## Why this is better than clamping

Clamping changes the residual.

Backtracking preserves the residual and only controls how aggressively the nonlinear solve moves through variable space.

This is especially important for inverse problems, because clamps can create artificial plateaus, artificial sensitivities, or hidden discontinuities in the parameter-to-observable map.

---

# Recommendation 4: Keep backward Euler in concentration space, but evaluate exp differences more stably

## Current correct form

The current time term should remain:

```text
(exp(u_new) - exp(u_old)) / dt
```

This is mass-conservative in concentration space.

## Numerical issue

When `u_new` and `u_old` are close, direct subtraction can suffer cancellation:

```python
exp(u_new) - exp(u_old)
```

## Suggested stable form

Use the identity:

```text
exp(u_new) - exp(u_old)
= exp(u_old) * expm1(u_new - u_old)
```

or symmetrically:

```text
exp(u_new) - exp(u_old)
= exp(m) * (exp(u_new - m) - exp(u_old - m))
```

where

```text
m = max(u_new, u_old)
```

Possible implementation:

```python
m = max_value(u_new, u_old)
dc_dt = exp(m) * (exp(u_new - m) - exp(u_old - m)) / dt
```

This does not remove the need for upper overflow control, but it improves numerical stability and avoids direct subtraction of nearly equal exponentials.

If UFL supports `expm1`, prefer:

```python
dc_dt = exp(u_old) * expm1(u_new - u_old) / dt
```

but only if `u_old` is safely bounded and the nonlinear step is trust-region controlled.

---

# Recommendation 5: Try electrochemical-potential variables

## Motivation

The current unknown is

```text
u_i = log(c_i)
```

The flux is

```text
J_i = -D_i exp(u_i) * (grad(u_i) + z_i F / RT * grad(phi))
```

Define the electrochemical-potential-like variable

```text
mu_i = u_i + z_i F / RT * phi
```

Then

```text
u_i = mu_i - z_i F / RT * phi
c_i = exp(mu_i - z_i F / RT * phi)
```

and the flux becomes

```text
J_i = -D_i c_i * grad(mu_i)
```

## Why this may help

In Debye layers or near-Boltzmann regions, `u_i` and `phi` may each vary strongly, but the electrochemical potential `mu_i` may be smoother.

This may improve nonlinear conditioning and reduce the size of Newton excursions.

## Important limitation

This does **not** eliminate exponentials. The Poisson source and concentration coefficient still require:

```text
c_i = exp(mu_i - z_i F / RT * phi)
```

But it may reduce how often the solver wanders into extreme exponent values.

## Possible implementation sketch

Use unknowns:

```text
mu_1, mu_2, ..., mu_N, phi
```

Define:

```python
u_i = mu_i - z_i * F / (R * T) * phi
c_i = exp(min_value(u_i, U_MAX))
```

Flux:

```python
J_i = D_i * c_i * grad(mu_i)
```

Weak form:

```python
F_np_i = ((c_i - c_old_i) / dt) * v * dx \
       + dot(D_i * c_i * grad(mu_i), grad(v)) * dx \
       - boundary_flux_i * v * ds_elec
```

Poisson:

```python
F_phi = epsilon * dot(grad(phi), grad(w)) * dx \
      - charge_rhs * sum(z_i * c_i * w for i in species) * dx
```

Boundary conditions need to be translated carefully:

```text
bulk c_i = c0_i
means mu_i = log(c0_i) + z_i F / RT * phi_bulk
```

If `phi_bulk` is fixed, this is straightforward.

---

# Recommendation 6: Consider a softplus concentration variable instead of pure log concentration

## Motivation

The pure log transform gives positivity:

```text
c = exp(u)
```

but it also creates exponential growth for large positive `u`.

An alternative is to use a positive transform that behaves like log concentration near zero but grows only linearly for large positive values.

Define:

```text
c = c_scale * softplus(y)
```

where

```text
softplus(y) = log(1 + exp(y))
```

Use the stable implementation:

```python
softplus_y = max_value(y, 0.0) + ln(1.0 + exp(-abs(y)))
c_i = c_scale_i * softplus_y
```

If `abs` is awkward in UFL, use `conditional`:

```python
softplus_y = conditional(
    ge(y, 0.0),
    y + ln(1.0 + exp(-y)),
    ln(1.0 + exp(y)),
)
```

## Behavior

For very negative `y`:

```text
softplus(y) ~ exp(y)
```

so the variable behaves like a log concentration near zero.

For very positive `y`:

```text
softplus(y) ~ y
```

so the concentration grows linearly rather than exponentially.

## Weak form

Use concentration as a function of `y`:

```text
c_i = c_scale_i * softplus(y_i)
```

Then keep the original concentration-form NP structure:

```text
int ((c_i_new - c_i_old) / dt) v dx
+ int D_i (grad(c_i) + z_i F / RT c_i grad(phi)) dot grad(v) dx
= boundary flux
```

In code:

```python
c_i = c_scale_i * softplus(y_i)
grad_c_i = c_scale_i * sigmoid(y_i) * grad(y_i)

flux_factor = grad_c_i + beta_i * c_i * grad(phi)
F_np_i = ((c_i - c_old_i) / dt) * v * dx \
       + D_i * dot(flux_factor, grad(v)) * dx \
       - boundary_flux_i * v * ds_elec
```

where

```text
beta_i = z_i F / RT
sigmoid(y) = 1 / (1 + exp(-y))
```

Use a stable sigmoid implementation to avoid overflow.

## Advantages

- Positivity is preserved.
- Large positive unknowns do not create exponential concentration blowup.
- It stays close to the current weak-form FEM implementation.
- It may be easier than a full Scharfetter–Gummel rewrite.

## Disadvantages

- It changes the variable transform, so it needs careful verification.
- Exact zero concentration still corresponds to `y = -infinity`, so product species still need a tiny initialization/bulk floor or a special boundary treatment.
- It may weaken the clean Boltzmann interpretation of `u = log(c)`.

## Suggested use

This is a good second-line experiment if upper-only clamps and trust-region damping are not enough.

---

# Recommendation 7: Prototype a Slotboom / Scharfetter–Gummel bulk flux discretization

## Motivation

The most principled way to avoid instability from the drift-diffusion term may be to change the spatial discretization rather than just the variable.

The Nernst–Planck flux has the form:

```text
J_i = -D_i (grad(c_i) + beta_i c_i grad(phi))
```

where

```text
beta_i = z_i F / RT
```

A Slotboom transform rewrites the operator using an integrating factor. This leads naturally to Scharfetter–Gummel-type fluxes, commonly used in semiconductor drift-diffusion and related PNP systems.

## 1D edge flux idea

Across an edge/cell with potential difference

```text
DeltaPsi = beta_i * (phi_R - phi_L)
```

an SG-style flux is approximately:

```text
J_edge = -D_i / h * [ B(DeltaPsi) c_R - B(-DeltaPsi) c_L ]
```

where

```text
B(x) = x / (exp(x) - 1)
```

The Bernoulli function must be evaluated stably:

```python
B(x) = x / expm1(x)
```

with a Taylor expansion near zero:

```text
B(x) = 1 - x/2 + x^2/12 - x^4/720 + ...
```

## Why this may help

- Drift is handled analytically along edges/cells.
- Positivity is easier to preserve.
- It avoids treating steep drift layers with plain polynomial concentration interpolation.
- It may reduce the need for artificial clamps in the bulk transport operator.

## Cost

This is a bigger rewrite. It may require moving from the current continuous Galerkin weak form toward one of:

```text
finite volume
DG
edge-averaged finite element method
exponentially fitted FEM
```

## Suggested implementation strategy

Do not rewrite the full production solver immediately.

Instead:

1. Build a minimal 1D SG transport prototype.
2. Test it on a fixed-potential NP equation.
3. Add Poisson coupling.
4. Add BV boundary flux.
5. Compare to the current log-c FEM solver on a simple voltage sweep.

If the prototype shows better robustness, then consider porting the idea into the production 2D solver.

---

# Recommendation 8: Treat zero product bulk concentrations explicitly

## Problem

The log formulation requires

```text
u = log(c)
```

so a species with physical bulk concentration zero cannot be represented exactly as a finite Dirichlet value.

The current code uses a floor such as:

```python
c0_i = max(c0_i, 1e-20)
u0_i = log(c0_i)
```

This may be unavoidable for a pure log variable, but it should be treated as a modeling/numerical convention, not as a physical concentration.

## Suggested options

### Option A: Keep a tiny initialization only, but avoid lower clamps afterward

Use the tiny floor only for initial and boundary data where log is undefined:

```python
c0_eff = max(c0_physical, C_INIT_FLOOR)
u0 = log(c0_eff)
```

But do not impose a lower clamp inside the bulk residual.

### Option B: Use a no-flux or reaction-generated product boundary treatment

For product species whose physical bulk concentration is zero, consider whether the far boundary should be:

```text
Dirichlet c = 0
```

or instead a different physical condition, such as:

```text
product is generated at electrode and leaves through transport
far-field concentration is approximately zero
```

In pure log variables, exact `c=0` is awkward. In concentration or softplus variables it may be easier.

### Option C: Use softplus for product species only

A hybrid formulation may be possible:

```text
reactant species: log concentration
product species with zero bulk concentration: softplus variable
```

This is more complex but may avoid artificial product floors.

---

# Suggested implementation order

## Phase 1: Minimal changes to current log-c solver

These are the highest-value changes and should be tried first.

1. Replace symmetric bulk `u` clamp with upper-only overflow guard.
2. Increase upper guard to a true floating-point safety value, e.g. `U_MAX = 650`.
3. Remove lower bulk clamp entirely.
4. Keep the conservative backward Euler time term.
5. Replace direct exponential differences with stable difference forms where feasible.
6. Remove or widen eta clipping in BV.
7. Evaluate BV using stable log-difference form.
8. Add Newton line search / trust-region rejection based on:
   - max `u`,
   - max `delta_u`,
   - max BV log-rate,
   - residual decrease.

## Phase 2: Alternative variable experiments

If Phase 1 is not robust enough:

1. Try electrochemical-potential variables:

   ```text
   mu_i = log(c_i) + z_i F phi / RT
   ```

2. Try softplus concentration variables:

   ```text
   c_i = c_scale_i * log(1 + exp(y_i))
   ```

3. Consider a hybrid variable choice for product species with zero bulk concentration.

## Phase 3: Larger discretization rewrite

If the weak-form FEM approach remains unstable:

1. Prototype a 1D Scharfetter–Gummel / Slotboom flux solver.
2. Verify positivity, mass conservation, and voltage sweep robustness.
3. Add BV boundary conditions.
4. Compare observables against the current production solver.
5. Only then consider a 2D production rewrite.

---

# Concrete tasks for Claude Code

## Task 1: Locate all remaining clamps

Search for:

```text
max_value
min_value
clip
clamp
exp(
eta_clip
u_clip
_C_FLOOR
```

Create a table listing:

```text
file
line/function
quantity being clamped
current bounds
reason for clamp
whether it changes physics or only prevents overflow
recommended replacement
```

Separate:

```text
1. initial/boundary floors needed because log(0) is undefined
2. residual-level clamps that change the solved PDE
3. solver-level safety checks that only reject bad Newton steps
```

## Task 2: Implement upper-only concentration exponentiation

Create a helper:

```python
def safe_exp_upper(u, U_MAX=650.0):
    return fd.exp(fd.min_value(u, U_MAX))
```

Use it wherever the current code does symmetric clamping for bulk concentration coefficients.

Do not apply a lower clamp inside the residual.

## Task 3: Preserve a separate log-zero convention

Keep a separate helper for initialization and Dirichlet data:

```python
def log_concentration_data(c0, floor=1e-20):
    return np.log(max(float(c0), floor))
```

This is different from residual clamping. Keep the distinction explicit in names and comments.

## Task 4: Implement stable BV log-difference rate

Create a helper that takes two log-rates:

```python
def logdiffexp_rate(log_cath, log_anod):
    m = fd.max_value(log_cath, log_anod)
    return fd.exp(m) * (fd.exp(log_cath - m) - fd.exp(log_anod - m))
```

Use this in the BV residual.

If possible, later add an `expm1`-based version for near-cancellation.

## Task 5: Add nonlinear step diagnostics

During each nonlinear solve, log:

```text
max u_i
min u_i
max abs(delta_u_i)
max eta_scaled
max BV log-rate
number of backtracking reductions
residual norm before and after step
```

This is critical for determining whether the remaining failures are physical stiffness, bad scaling, bad initial guess, or an actual formulation issue.

## Task 6: Add step rejection / damping

Implement a backtracking rule that rejects trial steps when:

```text
max u_i > U_MAX
max abs(delta_u_i) > DELTA_U_MAX
max BV log-rate > RATE_LOG_MAX
residual norm does not decrease
```

Suggested starting values:

```python
U_MAX = 650.0
DELTA_U_MAX = 5.0
RATE_LOG_MAX = 650.0
```

This should be implemented as solver control, not as a residual clamp.

## Task 7: Run an A/B comparison

Compare these solver variants:

```text
A. current production log-c solver
B. upper-only bulk clamp, current eta clip
C. upper-only bulk clamp, stable BV logdiff, no eta clip
D. C plus Newton trust-region damping
```

For each variant, record:

```text
voltage range completed
Newton iterations per voltage
failed voltages
mass conservation error
min/max concentration per species
surface concentrations
BV rates R1/R2
observable currents
objective landscape if used in inverse solve
```

Pay special attention to high positive voltages and product species such as H2O2.

## Task 8: Optional prototype softplus variable branch

Create a separate branch or experimental file. Do not mix this into production initially.

Implement:

```python
def softplus(y):
    return fd.conditional(
        fd.ge(y, 0.0),
        y + fd.ln(1.0 + fd.exp(-y)),
        fd.ln(1.0 + fd.exp(y)),
    )
```

Then define:

```python
c_i = c_scale_i * softplus(y_i)
```

and use the concentration-form NP residual:

```python
grad_c_i = c_scale_i * sigmoid(y_i) * grad(y_i)
flux_factor = grad_c_i + beta_i * c_i * grad(phi)
```

This branch should be evaluated only after the simpler upper-only clamp and solver damping changes are tested.

---

# Key principle

Do not aim for “no exponentials anywhere.” That is impossible for a log-concentration PNP formulation because the physical concentration still appears in the Poisson source, the mass term, the mobility coefficient, and the BV boundary rate.

The better goal is:

```text
Keep the physical residual unclamped.
Evaluate exponentials stably.
Reject bad nonlinear steps instead of clipping the equations.
Only use floors where the mathematical transform itself requires them, such as log(0) data.
```

That is the path most likely to improve both forward-solver robustness and inverse-problem reliability.
