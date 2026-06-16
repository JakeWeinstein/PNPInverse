# Codex Review of Handoff #12 -- IC/Picard Bugs

**Date:** 2026-05-07  
**Reviewer:** Codex  
**Subject:** Critical review of `CHATGPT_HANDOFF_12_IC_PICARD_BUGS.md`

## Executive summary

The handoff's two main diagnoses are substantially correct:

1. The Debye-Boltzmann IC Picard loop is not Stern-aware, while the residual is.
2. The Bikerman IC writes gamma-shifted dynamic concentrations, while the Picard surface-rate algebra still solves against gamma-free concentrations.

However, I do **not** recommend implementing proposed Fix A as written. Dropping `log_gamma` from the IC is a quick way to make the Picard and IC agree, but it does so by moving the IC off the Bikerman zero-flux / packing manifold that the residual is actually solving. The lower-risk correct fix is to keep gamma in the IC and update the Picard algebra to use the same reaction-plane concentrations the residual sees.

Recommended order:

1. Keep gamma in the IC and make the Picard surface BV rates gamma-aware.
2. Add Stern-aware diffuse/Stern splitting and anchor `phi(0)` at the solution-side surface potential.
3. Make the linear fallback Stern-aware too.
4. Only after those fixes, address the cathodic Picard limit cycle as a separate nonlinear iteration problem.

## Code-verified findings

### 1. Stern eta mismatch is real

Residual eta is Stern-aware in both logc and muh:

- `Forward/bv_solver/forms_logc.py:243-249`
- `Forward/bv_solver/forms_logc_muh.py:298-304`

When Stern is active, the residual uses:

```text
eta_raw = phi_applied_func - phi - E_eq
```

The IC Picard loop instead uses:

```text
eta = bv_exp_scale * (phi_applied_model - E)
```

at:

- `Forward/bv_solver/forms_logc.py:718-722`
- `Forward/bv_solver/forms_logc_muh.py:816-820`

The Picard IC then constructs:

```text
phi_init = log(H_outer / c_clo4_bulk) + psi
```

and the current diffuse drop is set so that, at the electrode, `phi_init(0)` collapses back to `phi_applied_model`.

That means the Stern residual evaluates the IC as:

```text
eta_raw = phi_applied - phi_applied - E_eq = -E_eq
```

This explains the voltage-independent fallback residual signature in the handoff.

### 2. Gamma/Picard mismatch is real

The Bikerman IC writes gamma-shifted concentrations:

- `Forward/bv_solver/forms_logc.py:924-939`
- `Forward/bv_solver/forms_logc_muh.py:996-1024`

For the muh path, the proton concentration reconstructed by the residual is:

```text
u_H = mu_H - em*z_H*phi
```

via:

- `Forward/bv_solver/forms_logc_muh.py:266-274`

So the residual really does see the gamma-shifted `u_i` written by the IC.

The Picard loop does not. It computes the BV rates using:

```text
H_s = H_o * exp(-psi_D)
O_s = O_b - R1 / D_O
P_s = P_b + (R1 - R2) / D_P
```

with no surface `log_gamma` term in the reaction-rate logs.

That is the direct mismatch.

## Why I would not drop `log_gamma`

The current residual-side Bikerman closure is not gamma-free. It uses dynamic concentrations in the total packing and in the steric flux:

- `Forward/bv_solver/boltzmann.py:215-225`
- `Forward/bv_solver/forms_logc.py:283-303`
- `Forward/bv_solver/forms_logc.py:328-330`
- `Forward/bv_solver/forms_logc_muh.py:312-323`

The analytic counterion closure is:

```text
c_steric = c_b * q * (1 - A_dyn) / (theta_b + a_b*c_b*q)
```

where:

```text
A_dyn = sum_i a_i*c_i
```

That means the dynamic species concentrations are part of the steric state. A matched Bikerman zero-flux IC should use:

```text
c_i = c_outer_i * gamma(psi) * exp(-z_i*psi)
```

For neutral species this reduces to:

```text
c_i = c_outer_i * gamma(psi)
```

So `log_gamma` on O2 and H2O2 is not obviously a bug. It is the expected steric chemical-potential offset for a common-size Bikerman-style volume fraction. Dropping it may recover a small no-Stern residual at moderate voltages, but that appears to be because it reverts toward the older ideal-counterion IC, not because it matches the current Bikerman residual.

The better fix is to keep the IC on the Bikerman manifold and make the scalar Picard problem solve the same boundary-rate problem.

## Recommended implementation details

### Fix 1: make Picard gamma-aware

Inside the Picard iteration, after updating or predicting `psi_D`, compute a scalar surface gamma consistent with the IC branch:

For 3sp + analytic Bikerman counterion:

```text
gamma_s = 1 / (
    1
    + a_h  * H_o          * (exp(-psi_D) - 1)
    + a_cl * c_clo4_bulk  * (exp(+psi_D) - 1)
)
```

For synthesized 4sp dynamic ClO4:

```text
gamma_s = 1 / (
    1
    + a_h  * H_o * (exp(-psi_D) - 1)
    + a_cl * H_o * (exp(+psi_D) - 1)
)
```

Then build BV logs using the same reaction-plane concentrations the IC writes:

```text
log_O_rxn = log(O_s) + log(gamma_s)
log_P_rxn = log(P_s) + log(gamma_s)
log_H_rxn = log(H_o) - psi_D + log(gamma_s)
```

The current 2x2 diffusion-balance solve assumes `A1`, `B1`, and `A2` multiply gamma-free `O_s`, `P_s`, and `H_s`. The coefficients should instead be derived from the gamma-shifted rate laws. The minimal algebraic update is to fold the appropriate gamma factors into `A1`, `B1`, and `A2`:

```text
A1 uses O_rxn and H_rxn factors
B1 uses P_rxn
A2 uses P_rxn and H_rxn factors
```

This needs care because the H factors already appear through `_h_factor_log`; do not double-count `log_gamma` there. The cleanest route is to replace `_h_factor_log(H_s, factors)` with a helper that accepts `log_H_rxn`, and to make the base cathodic/anodic species logs explicit.

### Fix 2: make the IC Stern-aware

When Stern is active, the IC should not enforce:

```text
phi_surface = phi_applied
```

It should solve the split:

```text
psi_total = phi_applied - phi_o
psi_total = psi_D + psi_S
phi_surface = phi_o + psi_D = phi_applied - psi_S
```

The scalar closure should match the residual Robin coefficient:

```text
stern_coeff * psi_S = poisson_coefficient * surface_slope(psi_D)
```

where `surface_slope(psi_D)` must be consistent with the IC profile:

- ideal/GC branch: use the GC first integral or equivalent derivative at y=0;
- Bikerman/composite branch: use the same BKSA surface slope `alpha_d` already computed for the saturated-zone profile.

Then the Picard eta should be:

```text
eta = bv_exp_scale * (phi_applied_model - phi_surface - E_eq)
```

not:

```text
eta = bv_exp_scale * (phi_applied_model - E_eq)
```

### Fix 3: make the linear fallback Stern-aware

The fallback IC also currently sets:

```text
phi(y) = phi_applied * (1 - y)
```

in:

- `Forward/bv_solver/forms_logc.py:572-578`
- `Forward/bv_solver/forms_logc_muh.py:640-656`

With Stern active, this has the same `phi(0)=phi_applied` problem. If Picard fails and falls back, the fallback should still set `phi(0)` to an estimated solution-side surface potential rather than to the metal potential.

A simple first pass can use the same Stern split helper with bulk outer values:

```text
phi_o = log(H_bulk / c_clo4_bulk)
psi_total = phi_applied - phi_o
solve psi_D + psi_S = psi_total
phi_surface = phi_applied - psi_S
phi(y) = phi_surface * (1 - y)
```

This will not fix cathodic Picard failure, but it prevents fallback rows from all seeing `eta=-E_eq`.

## Test and validation plan

### Fast tests

Add pure-Python scalar tests for helper functions before running Firedrake:

1. Gamma helper:
   - `gamma_s -> 1` when `psi_D=0`.
   - high positive `psi_D` gives counterion saturation behavior.
   - neutral species get the common `log_gamma` shift.

2. Stern split helper:
   - no-Stern or infinite-capacitance limit gives `psi_S=0`, `phi_surface=phi_applied`.
   - finite Stern gives `phi_surface < phi_applied` for positive diffuse charge under the current sign convention.
   - the residual identity `stern_coeff*psi_S ~= poisson_coefficient*surface_slope` holds.

3. Picard rate logs:
   - no-Bikerman/ideal path is unchanged.
   - Bikerman path includes `log_gamma` exactly once in O, P, and H reaction-plane logs.

### Slow Firedrake checks

Run targeted IC residual diagnostics, not the full production sweep first:

1. no-Stern + ideal counterion at `V_RHE=+0.5`: should remain near the old good baseline.
2. no-Stern + Bikerman at `V_RHE=+0.5`: should drop substantially from the current `~1e3` residual if gamma/Picard consistency is fixed.
3. Stern + Bikerman at `V_RHE=+0.5`: should drop relative to the current production residual once the Stern split is implemented.
4. Stern fallback case such as `V_RHE=0.0` or `+0.1`: species residual should no longer be the voltage-independent `exp(alpha*n_e*E_eq)` signature.

Then rerun:

```text
scripts/diagnose_db_ic_distance.py
```

for both `logc` and `logc_muh`, with and without Stern.

## Notes on the cathodic Picard limit cycle

The cathodic Picard limit cycle described in the handoff is likely a third issue. It should not be mixed into the first Stern/gamma patch unless those fixes fail in the converged-Picard regime.

The observed cycle is plausible:

```text
large R -> H_o floor -> huge psi_D -> rates collapse -> R decays
small R -> H_o recovers -> psi_D normal -> rates explode -> repeat
```

Best next step after Fixes 1-3:

1. Instrument scalar Picard history for `R1`, `R2`, `H_o`, `psi_D`, `gamma_s`, and `eta`.
2. Try a log-domain variable for `H_o` or a bounded nonlinear solve for the scalar reduced problem.
3. Try Anderson acceleration only after confirming the fixed point exists.
4. Treat smoothing the `H_o` floor as diagnostic, not as the production fix.

## Bottom line

The handoff correctly found the two inconsistencies, but the smallest patch is not the safest physical patch. The consistent path is:

```text
keep Bikerman gamma in the IC
make Picard use the same gamma-shifted reaction-plane concentrations
make Stern split/eta/phi anchoring consistent with the residual
then solve the cathodic Picard cycle separately
```

