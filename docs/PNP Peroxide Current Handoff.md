# PNP 3-Species + Boltzmann Model: Peroxide Current Handoff for Claude Code

## Purpose

We need to recover the peroxide-current / peroxide-selectivity observable in the newer **3-species + Boltzmann ClO4-** PNP-BV solver.

Claude previously seemed to think the peroxide observable could not be computed from the 3sp model. That is probably a misunderstanding. The 3sp model removes only the **nonreactive ClO4- co-ion** as a dynamic species. It still solves for:

- O2
- H2O2
- H+

and still includes both Butler-Volmer reactions:

- **R1:** O2 + 2H+ + 2e- -> H2O2
- **R2:** H2O2 + 2H+ + 2e- -> 2H2O

Therefore peroxide information is still available directly from the **reaction rates** at the electrode, and also from the **H2O2 boundary flux** as a consistency check.

The removed ClO4- variable is not needed to compute peroxide production, peroxide consumption, disk current, ring-equivalent peroxide current, or H2O2 selectivity.

---

## Key conceptual correction

Do **not** try to get peroxide current from ClO4- or from the eliminated charge variable.

The peroxide observable should be computed from either:

1. the two electrode BV reaction rates, `r1` and `r2`, or
2. the H2O2 flux at the electrode boundary.

Because the 3sp model still contains H2O2 and both reactions, the peroxide-current observable is not lost.

---

## Rate definitions

Let:

- `r1` = rate of R1, O2 -> H2O2
- `r2` = rate of R2, H2O2 -> H2O

For the formulas below, assume `r1 > 0` and `r2 > 0` mean cathodic reduction in the forward ORR direction.

Then:

```text
R1 consumes 2 electrons and produces H2O2.
R2 consumes 2 electrons and consumes H2O2.
```

The total cathodic disk current density is:

```text
i_disk = -2 F (r1 + r2)
```

The individual reaction-current components are:

```text
i_R1 = -2 F r1

i_R2 = -2 F r2
```

The net peroxide production rate at the disk is:

```text
r_H2O2_net = r1 - r2
```

The ring-equivalent peroxide current before applying collection efficiency is:

```text
i_ring_equiv = +2 F (r1 - r2)
```

The sign here assumes the ring oxidizes H2O2 and therefore gives a positive anodic ring current. If the code stores all currents with the disk cathodic sign convention, then store the disk-signed version separately:

```text
i_H2O2_escape_disk_sign = -2 F (r1 - r2)
```

If there is a collection efficiency `N`, then the modeled RRDE ring current is:

```text
i_ring = N * 2 F (r1 - r2)
```

---

## Important distinction: peroxide current vs peroxide selectivity

Do not overload the variable name `PC` unless we define exactly what it means.

There are several related but distinct observables:

```text
i_R1              = current through the peroxide-producing first 2e step
                   = -2 F r1

I_R2              = current through the peroxide-consuming second 2e step
                   = -2 F r2

i_disk            = total disk current
                   = -2 F (r1 + r2)

i_H2O2_escape     = disk-signed net peroxide escape current
                   = -2 F (r1 - r2)

i_ring_equiv      = ring-equivalent peroxide oxidation current before collection efficiency
                   = +2 F (r1 - r2)

i_ring            = N * i_ring_equiv
```

For inverse fitting, the best second observable is probably not `i_R1` alone. It is usually either:

1. the peroxide/ring current, proportional to `r1 - r2`, or
2. the peroxide selectivity / H2O2 yield.

A useful H2O2 selectivity formula in terms of the rates is:

```text
H2O2_percent = 100 * (r1 - r2) / r1
```

This is equivalent to the standard RRDE expression:

```text
H2O2_percent = 200 * (i_ring / N) / (abs(i_disk) + i_ring / N)
```

because:

```text
abs(i_disk) = 2 F (r1 + r2)

i_ring / N = 2 F (r1 - r2)
```

so:

```text
200 * [2F(r1-r2)] / [2F(r1+r2) + 2F(r1-r2)]
= 100 * (r1-r2) / r1
```

This is a better selectivity definition than simply `(r1-r2)/(r1+r2)`, which is only the ring-equivalent peroxide current divided by the disk-current magnitude, not the usual H2O2 percent.

---

## Where to compute it in the code

The safest implementation is to expose the BV rates from the same code path that constructs the electrode boundary terms.

Do **not** duplicate a different BV formula in a diagnostic file unless absolutely necessary. The diagnostic observable and the PDE residual should use the same symbolic rate expressions.

Recommended refactor:

```python
# Pseudocode only

def bv_rates_at_electrode(state, params):
    """
    Return UFL expressions for r1 and r2 on the electrode boundary.
    These should be the exact same rate expressions used in the BV flux boundary condition.
    """
    c_O2_s = surface_concentration(state, "O2")
    c_H2O2_s = surface_concentration(state, "H2O2")
    c_H_s = surface_concentration(state, "H+")
    phi_s = surface_potential(state)

    r1 = bv_rate_R1(c_O2_s, c_H2O2_s, c_H_s, phi_s, params)
    r2 = bv_rate_R2(c_H2O2_s, c_H_s, phi_s, params)

    return r1, r2
```

Then compute observables from those rates:

```python
# Pseudocode only

def electrode_observables(state, params, ds_electrode, dimensional=True):
    r1_hat, r2_hat = bv_rates_at_electrode(state, params)

    # Integrate over electrode boundary. In 1D this may amount to point/boundary
    # evaluation, but use the same boundary measure convention as the existing
    # disk-current code.
    R1_hat = assemble(r1_hat * ds_electrode)
    R2_hat = assemble(r2_hat * ds_electrode)

    if dimensional:
        # For nondimensional PNP with c scaled by c_ref and x by L_ref,
        # reaction-rate/flux scale is D_ref * c_ref / L_ref.
        flux_scale = params.D_ref * params.c_ref / params.L_ref
        R1 = flux_scale * R1_hat
        R2 = flux_scale * R2_hat
    else:
        R1 = R1_hat
        R2 = R2_hat

    F = params.F
    n_e = 2.0

    i_R1 = -n_e * F * R1
    i_R2 = -n_e * F * R2
    i_disk = -n_e * F * (R1 + R2)

    # Net peroxide production / escape from disk.
    i_h2o2_escape_disk_sign = -n_e * F * (R1 - R2)
    i_ring_equiv = +n_e * F * (R1 - R2)

    N = getattr(params, "collection_efficiency", None)
    i_ring = None if N is None else N * i_ring_equiv

    # H2O2 percent, with safety for small R1.
    h2o2_percent = 100.0 * (R1 - R2) / R1

    return {
        "R1_rate": R1,
        "R2_rate": R2,
        "i_R1": i_R1,
        "i_R2": i_R2,
        "i_disk": i_disk,
        "i_h2o2_escape_disk_sign": i_h2o2_escape_disk_sign,
        "i_ring_equiv": i_ring_equiv,
        "i_ring": i_ring,
        "h2o2_percent": h2o2_percent,
    }
```

The exact scaling should be checked against the current existing current-density calculation. The important rule is:

> Use the same dimensional/nondimensional scaling for disk current and peroxide current.

---

## Log-c formulation caveat

For the log-concentration model, the physical concentrations are:

```text
c_i = exp(u_i)
```

The BV rate diagnostic should use the same `exp(u_i)` expressions used in the PDE residual.

Avoid computing the diagnostic using a hard concentration floor such as:

```python
c_surf = max(c, eps)
```

if the log-c model already guarantees positivity. A floor can create artificial peroxide consumption/production, especially for H2O2 near zero. This was already identified as a major source of trouble: when H2O2 is near zero, a concentration floor combined with a large clipped BV exponential can create a fake R2 sink.

If the residual still uses a floor or softplus safeguard, the diagnostic should first match the residual exactly for consistency, but the better longer-term fix is to remove unnecessary floors from log-c reaction expressions.

---

## Boundary-flux consistency check

As a test, compute peroxide production two ways:

### Method 1: From reaction rates

```text
r_H2O2_net = r1 - r2
```

### Method 2: From the H2O2 flux boundary condition

At the electrode:

```text
J_H2O2 · n = stoichiometric boundary flux
```

Depending on the code's normal/sign convention, this should be equal to either `r1 - r2` or `-(r1 - r2)`.

Add a diagnostic assertion that compares:

```text
H2O2_flux_from_boundary
```

against:

```text
r1 - r2
```

up to sign.

This is important because sign errors in boundary fluxes and current observables are easy to miss.

---

## Recommended output columns

Add these columns to the I-V curve output CSV / diagnostics table:

```text
V_RHE
z_achieved
converged
cd_total
r1_rate
r2_rate
i_R1
i_R2
i_h2o2_escape_disk_sign
i_ring_equiv
i_ring   # if collection efficiency is provided
h2o2_percent
```

If the code currently has an old `PC` column, replace it or supplement it with explicitly named columns. Avoid using `PC` without defining whether it means:

- R1 pathway current,
- net escaping peroxide current,
- ring-equivalent current,
- ring current after collection efficiency, or
- H2O2 selectivity percent.

---

## Inverse objective recommendation

Once the peroxide observable is available, use a multi-observable objective:

```text
J = J_disk + J_peroxide
```

For real or synthetic RRDE-style data:

```text
J_disk = sum_k [ (i_disk_sim(V_k) - i_disk_obs(V_k)) / sigma_disk(V_k) ]^2

J_ring = sum_k [ (i_ring_sim(V_k) - i_ring_obs(V_k)) / sigma_ring(V_k) ]^2
```

or, if using selectivity:

```text
J_h2o2 = sum_k [ (h2o2_percent_sim(V_k) - h2o2_percent_obs(V_k)) / sigma_h2o2(V_k) ]^2
```

Then:

```text
J_total = J_disk + J_ring
```

or:

```text
J_total = J_disk + J_h2o2
```

Do not use arbitrary weights if noise estimates are available. Normalize each residual by its measurement uncertainty.

This should improve identifiability of `k0_2`, because total disk current mostly sees the sum `r1 + r2`, while peroxide/ring current sees the difference `r1 - r2` and therefore gives information about the partitioning between peroxide escape and peroxide reduction.

---

## Suggested implementation tasks for Claude Code

### Task 1: Locate existing BV rate construction

Find the functions in the 3sp + Boltzmann solver that construct the electrode BV boundary terms for R1 and R2.

Refactor them so that `r1` and `r2` can be returned as reusable UFL expressions.

### Task 2: Add an observable function

Add a function similar to:

```python
def compute_electrode_observables(solution, params, mesh_info):
    ...
```

It should return disk current, R1/R2 components, net peroxide escape current, ring-equivalent current, and H2O2 percent.

### Task 3: Ensure nondimensional scaling is correct

Compare the new `i_disk` with the existing disk-current diagnostic. They should match to numerical precision.

If they do not, fix scaling/sign conventions before trusting peroxide output.

### Task 4: Add flux-vs-rate consistency test

For a few voltages in the working window, verify that H2O2 boundary flux matches `r1 - r2` up to the code's sign convention.

### Task 5: Compare against old 4sp model where both converge

In the overlap voltage range where both 4sp and 3sp+Boltzmann converge, compare:

```text
i_disk
i_R1
i_R2
i_h2o2_escape / i_ring_equiv
h2o2_percent
```

The 3sp model should agree closely if the Boltzmann replacement is not altering the ORR chemistry.

### Task 6: Add inverse-fitting option

Add an option to the inverse script, for example:

```text
observable_mode = "disk_only"
observable_mode = "disk_plus_ring"
observable_mode = "disk_plus_h2o2_percent"
```

This will let us directly test whether adding peroxide information breaks the `k0_2` identifiability problem.

---

## Limiting-case tests

These are useful sanity checks.

### Test A: k0_2 = 0

If R2 is disabled:

```text
r2 = 0

i_disk = -2F r1

i_ring_equiv = +2F r1

H2O2_percent = 100%
```

This means all reduced O2 exits as H2O2.

### Test B: R2 very fast

If R2 is very fast and consumes nearly all H2O2:

```text
r2 ≈ r1

r1 - r2 ≈ 0

i_ring_equiv ≈ 0

H2O2_percent ≈ 0%
```

The disk current approaches the 4-electron limit:

```text
i_disk ≈ -4F r1
```

### Test C: collection efficiency

If collection efficiency `N` is provided:

```text
i_ring = N * i_ring_equiv
```

If `N = 1`, `i_ring` should equal `i_ring_equiv`.

If `N = 0.25`, ring current should be exactly one quarter of the ring-equivalent peroxide current.

---

## Why this matters for k0 inference

The current inverse problem has a ridge in `(log k0, alpha)` when fitting disk current alone. Disk current mostly constrains:

```text
r1 + r2
```

Peroxide/ring current constrains:

```text
r1 - r2
```

Together, these two observables provide much more information about the individual rates and especially about the balance between R1 and R2.

This is especially important for `k0_2`, because R2 is only indirectly visible in the total disk current but directly affects the amount of H2O2 that escapes the disk.

Therefore, restoring peroxide current in the 3sp+Boltzmann model is likely one of the best ways to improve `k0_2` identifiability without relying only on Tikhonov regularization.

---

## Summary instruction

Please implement peroxide observables in the 3sp + Boltzmann solver by exposing the electrode BV rates `r1` and `r2`, computing disk and peroxide/ring observables from those rates, and adding consistency tests against the H2O2 boundary flux and the old 4sp model in the overlap voltage window.

The key point is:

> Removing dynamic ClO4- does not remove peroxide information. The peroxide observable lives in the O2/H2O2/H+ reaction network, which is still present in the 3sp model.
