# PNP-BV Forward Solver Extension Handoff for Claude Code

## Purpose

This handoff converts the latest ChatGPT review into concrete implementation tasks. The goal is to extend the forward solver to higher anodic voltages so that `k0_2` and `alpha_2` become data-identifiable rather than prior-selected.

Current conclusion from the FIM work:

- Multi-experiment steady-state designs did **not** rotate the weak `k0_2` direction.
- CD+PC adds information, but not enough under the current accessible-voltage model.
- The main missing information is direct R2 voltage-shape sensitivity.
- R2 is currently hidden because its Butler–Volmer exponent is hard-clipped at `exp(50)` throughout the accessible voltage window.
- Therefore the next path is not more optimizer tuning. The next path is to attack the forward solver / BV formulation so that the model can either:
  1. expose R2 sensitivity without artificial clipping, or
  2. reach higher anodic voltages where R2 naturally unclips.

The recommended path is staged:

1. **BV exponent clip audit** — cheap, do first.
2. **Log-rate BV evaluation and removal of concentration floors inside BV** — still cheap.
3. **1D finite-volume / Scharfetter–Gummel prototype** — medium effort, highest-confidence numerical path.
4. **Extended-voltage FIM/TRF validation** — only after the forward model reaches higher voltages.
5. **Optional long-term Firedrake DG/FV port** — only if FV/SG works and adjoints are needed.

---

## Current diagnosis

### What failed

The latest FIM screen tested:

- single rotation;
- multiple `L_ref` values;
- ORR + H2O2 co-fed variants;
- CD-only and CD+PC observables.

None rotated the `k0_2` weak direction. The weak eigenvector stayed essentially aligned with pure `log_k0_2`.

### Mechanism

R2 has:

```text
E_eq,2 = 1.78 V vs RHE
```

In the accessible solver window:

```text
V_RHE ≈ -0.10 to +0.20 V
eta_2 = V_RHE - E_eq,2 ≈ -1.88 to -1.58 V
```

The cathodic exponent is roughly:

```text
alpha_2 * n_e * eta_2 / V_T ≈ -73 to -62
```

The current code hard-clips the BV exponent to `±50`, so the R2 exponential is frozen at:

```text
exp(50)
```

Consequences:

```text
dr2/dalpha_2 = 0 exactly in clipped regions
dr2/dlog_k0_2 = r2
```

So alpha_2 has no direct first-order kinetic sensitivity, and k0_2 only appears as a multiplicative factor in a sensitivity direction that does not rotate across experiments.

This means the latest “multi-experiment cannot fix k0_2” result is definitive **for the clipped accessible-voltage model**, but before accepting it as physical, audit the clip.

---

# Stage 1 — BV exponent clip audit

## Goal

Determine whether the `k0_2` identifiability failure is truly physical or partly caused by the hard BV exponent clip.

## Add configuration options

Add a BV exponent mode:

```python
bv_exp_mode = "hard_clip" | "soft_clip" | "no_clip" | "cap_sweep"
```

Add a cap parameter:

```python
bv_exp_cap = 50.0  # default current behavior
```

Run caps:

```python
caps = [50, 60, 70, 80, 100, None]
```

Where:

- `50` reproduces current behavior.
- `60-100` tests whether sensitivity returns gradually.
- `None` means no model clip, only catastrophic overflow protection.

## Required outputs

For each cap, record:

```text
convergence status by voltage
min/max H2O2 surface concentration
min/max H2O2 in domain
CD(V)
PC(V)
r1(V)
r2(V)
dr2/dalpha_2 or finite-difference proxy
dPC/dalpha_2
FIM singular values
FIM weak eigenvector
ridge_cos
```

Suggested result folder:

```text
StudyResults/v19_bv_clip_audit/
    convergence_by_cap.json
    fim_by_cap.json
    observables_by_cap.csv
    h2o2_min_by_cap.csv
    rates_by_cap.csv
    run.log
```

## Acceptance test

If increasing the cap from 50 to 70/80/100 changes any of the following:

```text
dPC/dalpha_2
weak eigenvector
sv_min
ridge_cos
```

then the current `k0_2` failure is at least partly a clip artifact.

If nothing changes and the solver fails badly without the cap, then proceed to the FV/SG prototype.

---

# Stage 2 — Evaluate BV rates in log space

## Goal

Avoid numerically fragile products of the form:

```python
huge_exp * tiny_concentration
```

The log-c formulation already represents concentrations as:

```python
c_i = exp(u_i)
```

So the BV rate should be evaluated in log-rate form where possible.

## R2 cathodic branch

Instead of:

```python
r2_cath = k0_2 * exp(exponent) * c_H2O2_surf * c_H_surf**2
```

compute:

```python
log_r2_cath = (
    log_k0_2
    + u_H2O2_surf
    + 2.0 * u_H_surf
    - alpha_2 * n_e * eta_2 / V_T
)
r2_cath = exp(log_r2_cath)
```

Use the project’s nondimensional conventions, but keep the structure.

## Remove concentration floors inside BV

Do **not** use:

```python
c_surf = max(c, floor)
```

inside the reaction rate when using log-c.

The concentration is already positive by construction:

```python
c = exp(u)
```

A concentration floor inside BV creates artificial reactive mass and can generate a spurious H2O2 sink.

If a guard is needed, it should be a solve-failure guard, not a physical floor.

## Catastrophic guard

Instead of hard-clipping at 50, use something like:

```python
if abs(log_rate_component) > 120:
    mark_solve_as_unreliable_or_enter_guarded_mode()
else:
    use_exp_directly()
```

The point is: `120` is not a physics clip. It is a catastrophic numerical guard.

Do not silently flatten derivatives in the production inverse model.

---

# Stage 3 — Continuation strategies before changing discretization

Try these before building a new solver.

## 3.1 Cap continuation

Warm-start sequentially:

```text
cap = 50 → 60 → 70 → 80 → 100 → None
```

At each voltage, use the previous cap’s solution as the initial condition.

## 3.2 R2 reaction continuation

Introduce:

```python
gamma_R2 in [0, 1]
```

and define:

```python
r2_eff = gamma_R2 * r2
```

Continuation:

```text
gamma_R2 = 0.0 → 0.1 → 0.25 → 0.5 → 0.75 → 1.0
```

Use this at high voltage where R2 stiffness destabilizes the solve.

## 3.3 Voltage continuation

Use the best available solution as a warm start:

```text
V = 0.30 → 0.35 → 0.40 → 0.50 → 0.60 → 0.80 → 1.00 → 1.20 → 1.50
```

Use small increments around convergence cliffs.

## 3.4 Combined continuation order

Recommended order for a hard case:

```text
known solution at lower V, cap=50, gamma_R2=1
increase V slightly
reduce gamma_R2 to 0 if needed
solve
ramp gamma_R2 to 1
increase cap
remove cap
```

This is diagnostic. If it only works with an active hard clip, it is not good enough for `alpha_2` inference.

---

# Stage 4 — 1D finite-volume / Scharfetter–Gummel prototype

## Why this is the highest-confidence numerical path

The core failure is positivity loss in steep drift-diffusion layers:

```text
CG1 finite elements are not positivity-preserving.
Depleting species goes negative.
BV exponentials applied to negative/near-zero concentrations blow up Newton.
```

Because the problem is 1D, do not start with full DG. Build a separate finite-volume prototype first.

## Scope

Create a new solver separate from Firedrake:

```text
Forward/fv_sg_solver/
    mesh_1d.py
    bernoulli.py
    flux_sg.py
    residual.py
    solve_steady.py
    validate_overlap.py
```

Do not implement adjoints initially.

Do not implement inverse optimization initially.

The goal is only:

```text
Can this positivity-preserving discretization reach higher V_RHE?
```

## Model

Start with the current best physics:

```text
3 dynamic species: O2, H2O2, H+
Boltzmann ClO4- background in Poisson source
log or positive concentration variables
BV boundary fluxes at electrode
bulk Dirichlet BC at x = L_ref
```

Do **not** return to dynamic ClO4- at first.

## Scharfetter–Gummel flux

For charged species, use an exponentially fitted flux.

For interface `j+1/2`:

```text
Delta psi = z_i * (phi_{j+1} - phi_j)
```

Use Bernoulli function:

```text
B(x) = x / (exp(x) - 1)
```

Flux form:

```text
J_i,j+1/2 =
    -D_i / h_j+1/2 * [
        B(-Delta psi) * c_i,j+1
        - B( Delta psi) * c_i,j
    ]
```

Handle small `x` with Taylor expansion:

```python
B(x) ≈ 1 - x/2 + x**2/12
```

For neutral species, this reduces to standard diffusion.

## Boundary condition at electrode

Use conservative flux balance:

```text
J_i(0) = -sum_r s_i,r * r_r
```

Use the project’s existing sign convention carefully.

Compute and store both:

```text
CD = electron current from r1 + r2
PC = peroxide current from r1 - r2
```

## Solver strategy

Use pseudo-transient backward Euler to steady state.

Pseudo-code:

```python
for V in voltage_grid:
    U = warm_start_from_previous_voltage_or_lower_cap()
    dt = dt_initial

    for step in range(max_steps):
        solve implicit FV residual with Newton/trust-region
        if converged:
            increase dt
        else:
            decrease dt / retry
        if steady_residual < tol:
            break
```

## Validation ladder

Do not start at 1.5 V.

Run in this order:

```text
1. Match Firedrake 3sp+Boltzmann/log-c over [-0.10, +0.30] V.
2. Confirm positivity of all species.
3. Confirm CD and PC match overlap to acceptable tolerance.
4. Try +0.40 V.
5. Try +0.50 V.
6. Try +0.60 V.
7. Try +0.80 V.
8. Try +1.00 V.
9. Try +1.14 V.
10. Try +1.20 to +1.50 V.
```

Why `+1.14 V` matters:

R2 begins to unclip when:

```text
|alpha_2 * n_e * eta_2 / V_T| < 50
```

For `alpha_2 ≈ 0.5`, this means roughly:

```text
eta_2 > -0.64 V
V_RHE > 1.14 V
```

So `V_RHE ≈ 1.14 V` is the first major target.

## Success criteria

Minimum success:

```text
positive concentrations
no hard BV clip required
stable CD/PC curves
reaches V_RHE >= 0.60 V
```

Strong success:

```text
reaches V_RHE >= 1.14 V
R2 no longer clipped
FIM weak direction rotates away from pure log_k0_2
TRF recovers k0_2 in clean synthetic data
```

---

# Stage 5 — Extended-voltage FIM and inverse validation

Only do this after a solver reaches higher voltage.

## FIM grids

Test voltage grids:

```python
V_grid_1 = [-0.10, 0.00, 0.10, 0.20, 0.30]
V_grid_2 = [-0.10, 0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
V_grid_3 = [-0.10, 0.10, 0.30, 0.50, 0.80, 1.00, 1.14]
V_grid_4 = [-0.10, 0.10, 0.30, 0.60, 0.90, 1.14, 1.30, 1.50]
```

For each grid, compute FIM for:

```text
CD only
PC only
CD + PC
```

Report:

```text
sv_min(S_white)
cond(F)
weak eigenvector
ridge_cos
dPC/dalpha_2
dPC/dlog_k0_2
```

## Clean-data TRF

Only run TRF when FIM improves.

Initial guesses:

```text
+20% all params
-20% all params
mixed: k0 high, alpha low
mixed: k0 low, alpha high
```

Pass conditions:

```text
alpha_1, alpha_2 within 5%
k0_1, k0_2 within 10-20%
same solution from multiple initial guesses
cost near zero for clean data
```

Then run 2% noise seeds.

---

# Stage 6 — Matched-asymptotic EDL model, optional but promising

This is a separate model-development path.

## Motivation

The full PNP solve spends enormous effort resolving a Debye layer that may not be needed for the inverse problem.

The reduced model would solve only the outer electroneutral transport problem and represent the EDL analytically through boundary corrections.

## Concept

At the reaction plane:

```text
c_i_surf = c_i_outer(0) * exp(-z_i * Delta_phi_D)
```

For neutral species:

```text
c_O2_surf ≈ c_O2_outer(0)
c_H2O2_surf ≈ c_H2O2_outer(0)
```

For protons:

```text
c_H_surf = c_H_outer(0) * exp(-Delta_phi_D)
```

Use effective BV overpotential:

```text
eta_r_eff = V_applied - E_eq,r - Delta_phi_Stern - Delta_phi_D
```

The unknown EDL potential drop `Delta_phi_D` is solved algebraically or through a compact boundary-layer relation.

## Why this matters

The 3sp + Boltzmann ClO4- model already partially does this and was the first model that captured the onset while extending the convergence window. A fully reduced EDL model may eliminate the singular Poisson perturbation entirely.

## Risk

This becomes a different model. It may be more publishable but requires careful physical justification.

---

# Stage 7 — Long-term Firedrake DG/FV port

Only do this if the 1D FV/SG prototype works.

## Recommended Firedrake direction

Use finite-volume-like DG0 first:

```text
concentrations: DG0 or DG1 with limiter
potential: CG1 or compatible DG/mixed space
fluxes: numerical fluxes at interfaces
charged fluxes: SG/exponential fitting if possible
neutral fluxes: standard diffusive flux
boundary fluxes: BV
```

The goal is positivity and conservation by construction.

Do not start with high-order DG.

Do not add positivity penalties as the main fix. Positivity must be structural.

---

# What not to do next

Do not spend major time on:

```text
L-BFGS-B tuning
objective reweighting
more multi-rotation FIMs under the same clipped model
more H2O2 co-fed variants under the same clipped model
artificial diffusion that smears onset physics
positivity penalties that fight the PDE
```

These have either already failed or do not address the missing R2 voltage-shape information.

---

# Immediate Claude Code task list

## Task 1 — BV clip audit

Implement cap sweep and produce:

```text
StudyResults/v19_bv_clip_audit/
```

with convergence, observables, rates, and FIM diagnostics.

## Task 2 — Log-rate BV branch

Implement log-rate evaluation for R2 and ideally R1.

Remove concentration floors inside BV when using log-c.

## Task 3 — Continuation experiments

Run:

```text
cap continuation
R2 reaction continuation
voltage continuation
```

Track whether any route reaches:

```text
V_RHE = 0.40, 0.50, 0.60
```

without hard clipping.

## Task 4 — FV/SG prototype

If Tasks 1-3 do not solve the issue, build the standalone 1D finite-volume / Scharfetter–Gummel solver.

No adjoint. No inverse. Just forward convergence and positivity.

## Task 5 — Extended FIM

If the forward model reaches `V_RHE >= 1.14 V`, immediately run FIM before any inverse optimization.

---

# Strategic recommendation

Use hybrid path C:

## Short-term paper path

Run Tikhonov / Bayesian prior tests honestly:

```text
alpha and k0_1 are data-driven
k0_2 requires a prior under the accessible clipped-voltage model
```

This is a defensible electrochemistry/inverse-problem result.

## Long-term methods path

Attack the anodic-voltage frontier using:

```text
clip audit → log-rate BV → FV/SG prototype → extended FIM → inverse validation
```

If this works, the contribution becomes much stronger:

```text
A positivity-preserving PNP-BV solver extends the voltage range enough to make k0_2 data-identifiable.
```

That is potentially publishable as both numerical methods and electrochemistry.

---

# Core takeaway

The missing information is not another optimizer trick and not another steady-state experiment under the same clipped voltage regime.

The missing information is **direct R2 voltage-shape sensitivity**.

To get it, either:

```text
1. use a prior for k0_2, or
2. extend the forward solver anodically until R2 is no longer clipped.
```

The next implementation should therefore begin with the BV clip audit and then move to a positivity-preserving 1D FV/SG prototype if the audit confirms that CG1/log-c cannot reach the needed voltage window.
