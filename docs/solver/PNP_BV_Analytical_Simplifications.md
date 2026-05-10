# Analytical simplifications for the production PNP-BV system

Date: 2026-05-03

## Context

This note uses the current production model described in
`writeups/WeekOfApr27/PNP Inverse Solver Revised.tex`,
`writeups/WeekOfApr27/PNP Log Concentration Transition.tex`,
`docs/CONTINUATION_STRATEGY_HANDOFF.md`, and the live implementation in
`Forward/bv_solver/forms_logc.py` plus `Forward/bv_solver/boltzmann.py`.

The active forward stack is:

- dynamic species `[O2, H2O2, H+]` with charges `[0, 0, +1]`;
- inert `ClO4-` removed as an NP unknown and added to Poisson as
  `c_-(x) = c_-^b exp(phi)`;
- primary variables `u_i = log c_i`;
- log-rate Butler-Volmer for
  `R1: O2 + 2H+ + 2e- <-> H2O2` and
  `R2: H2O2 + 2H+ + 2e- -> 2H2O`;
- nondimensional potential `phi = Phi/V_T`, bulk at `y=1`,
  electrode at `y=0`.

The key question is whether there are analytical reductions analogous to the
Boltzmann counterion simplification, whether any part of the system is
closed-form, and whether this can produce a strong voltage-specific initial
condition.

## Main conclusion

A closed-form solution of the full production system is not realistic. The
full problem is a two-dimensional, singularly perturbed, steric, nonlinear
Poisson-Nernst-Planck system with two nonlinear reactive boundary rates.
Even the one-dimensional steady PNP-BV problem with nonzero reactive fluxes is
not generally expressible as elementary closed form.

There is, however, a useful analytical reduction:

1. Use Poisson-Boltzmann/Gouy-Chapman theory to represent the charged Debye
   layer analytically.
2. Use electroneutral outer transport for the slowly varying region outside
   the Debye layer.
3. In that outer region, the analytic Boltzmann `ClO4-` constraint collapses
   proton electromigration into an ambipolar diffusion law with effective
   diffusivity `2D_H`.
4. The steady neutral/species transport then reduces to linear concentration
   profiles coupled to a two-rate Butler-Volmer algebraic system.

This is the breakthrough: the best analytical object is not an exact full
solution, but a matched-asymptotic composite state. It should be good enough
to initialize Newton near the correct Debye-layer basin at a specified
voltage, including voltages where the current linear-potential IC is far from
the physical branch.

## Exact zero-flux simplification

In the log-c formulation, ignoring steric for the moment,

```text
J_i = -D_i c_i grad(u_i + z_i phi).
```

Define the electrochemical-potential variable

```text
mu_i = u_i + z_i phi.
```

Then

```text
J_i = -D_i c_i grad(mu_i).
```

For any species with zero flux in a connected region, `grad(mu_i)=0`, so

```text
c_i = C_i exp(-z_i phi).
```

This is the exact reason the inert supporting electrolyte can be removed as a
dynamic species. It also says that `H+` is locally Boltzmann distributed in
the diffuse layer whenever its normal reaction flux is small compared with
double-layer equilibration:

```text
c_H(y) ~= c_H^outer exp(-(phi(y) - phi_outer)).
```

This is not exact globally because `H+` is consumed by both BV reactions, but
it is an excellent inner-layer approximation. That is exactly where the
current Newton failures occur.

With steric active, the zero-flux variable becomes

```text
mu_i = u_i + z_i phi + mu_steric(c).
```

The Boltzmann formula becomes modified by the steric activity coefficient.
For the current dilute `a=0.01` production cases, the nonsteric formula is the
right first analytical IC; steric can be added later as one Picard correction.

## Analytical Debye layer

For the symmetric `H+`/`ClO4-` layer, take an outer proton concentration
`h_o` and outer potential `phi_o = log(h_o / c_-^b)`. In the inner layer let

```text
psi = phi - phi_o.
```

The zero-flux Boltzmann profiles are

```text
c_H = h_o exp(-psi),
c_- = h_o exp(+psi).
```

The one-dimensional Poisson equation becomes the Poisson-Boltzmann equation

```text
eps * psi'' = 2 h_o sinh(psi),
lambda_D,o = sqrt(eps / (2 h_o)).
```

For a thin layer on a semi-infinite domain, the Gouy-Chapman solution is

```text
tanh(psi(y) / 4) = tanh(psi_D / 4) exp(-y / lambda_D,o),
psi_D = phi_electrode - phi_o.
```

Equivalently,

```text
psi(y) = 4 atanh(tanh(psi_D / 4) exp(-y / lambda_D,o)).
```

This is much better than the current linear initial potential profile. At
`V_RHE = +1.0 V`, `phi_electrode ~= 38.9`; the physically relevant initial
proton value at the electrode is therefore closer to
`log(h_o) - 38.9`, not to `log(h_bulk)`. The current docs already observed
that Newton fails because it is asked to discover this enormous Boltzmann
depletion from a non-Boltzmann IC.

For small voltage, the linear Debye-Huckel version is enough:

```text
psi(y) ~= psi_D exp(-y / lambda_D,o),
c_H ~= h_o (1 - psi),
c_- ~= h_o (1 + psi).
```

For the production grid and especially for positive voltage, the nonlinear
Gouy-Chapman expression is the better default.

## Combined Boltzmann-Tafel factor

Both production reactions contain the cathodic proton factor
`(c_H/c_ref)^2`. In the diffuse-layer approximation,

```text
c_H,s = H_o exp(-psi_D).
```

Therefore every cathodic BV branch carries the multiplier

```text
(c_H,s/H_ref)^2 exp(-alpha_j n eta_j)
  = (H_o/H_ref)^2 exp(-2 psi_D - alpha_j n eta_j).
```

This is a compact analytical explanation for the severe positive-voltage
conditioning wall. At anodic `psi_D >> 1`, the cathodic branches are
suppressed not only by the usual Tafel factor but also by proton Boltzmann
depletion. For the two-proton reactions here, the extra suppression is
`exp(-2 psi_D)`. At `V_RHE = +1.0 V`, `psi_D` can be `O(40)`, so the IC must
place `H+` tens of log-units below bulk at the electrode.

## Outer electroneutral simplification

Outside the Debye layer, electroneutrality gives

```text
c_H ~= c_-^b exp(phi).
```

Therefore

```text
phi = log(c_H / c_-^b).
```

Insert this into the proton NP flux:

```text
J_H = -D_H c_H grad(log c_H + phi)
    = -D_H c_H grad(log c_H + log(c_H/c_-^b))
    = -2 D_H grad(c_H).
```

So the outer `H+` equation is ordinary diffusion with effective diffusivity
`2D_H`. This is the main simplification beyond the existing Boltzmann
counterion: once the counterion is analytic, proton electromigration no
longer has to be solved explicitly in the outer electroneutral region.

The neutral species are already ordinary diffusion in the outer region:

```text
J_O = -D_O grad O,
J_P = -D_P grad P.
```

Thus, the one-dimensional steady outer solution has linear profiles.

## Two-rate algebraic outer model

Let `y=0` be the electrode and `y=1` the bulk. Let `R1` and `R2` be positive
in the cathodic directions used by the code:

```text
R1: O2 -> H2O2,
R2: H2O2 -> H2O.
```

The production stoichiometries are

```text
s_R1 = [-1, +1, -2],
s_R2 = [ 0, -1, -2].
```

For linear outer profiles, the reaction-plane-side outer concentrations are

```text
O_s = O_b - R1 / D_O,
P_s = P_b + (R1 - R2) / D_P,
H_o = H_b - (R1 + R2) / D_H.
```

The `H_o` expression uses the ambipolar `2D_H` and the `-2` proton
stoichiometry, so the factors cancel.

The actual proton concentration entering BV at the reaction plane is the
outer proton value multiplied by the diffuse-layer Boltzmann factor:

```text
H_s = H_o exp(-psi_D),
psi_D = phi_electrode - phi_o,
phi_o = log(H_o / c_-^b).
```

For the current no-Stern production stack, the code evaluates BV exponents
with `eta_j = phi_applied - E_eq,j`; it does not subtract the diffuse-layer
solution potential. If a Stern/Frumkin formulation is enabled later, replace
that with the physically corrected overpotential used by that form.

Define

```text
A1 = k1 (H_s/H_ref)^2 exp(-alpha1 n eta1),
B1 = k1 exp((1-alpha1) n eta1),
A2 = k2 (H_s/H_ref)^2 exp(-alpha2 n eta2).
```

Then

```text
R1 = A1 O_s - B1 P_s,
R2 = A2 P_s.
```

If `H_s` is held fixed during one Picard step, this is a 2x2 linear system:

```text
[1 + A1/D_O + B1/D_P    -B1/D_P      ] [R1] = [A1 O_b - B1 P_b]
[-A2/D_P                 1 + A2/D_P  ] [R2]   [A2 P_b          ].
```

This small algebraic solve gives a voltage-specific approximation to the
surface rates and surface concentrations. Updating `H_o`, `phi_o`, `psi_D`,
and the coefficients `A1`, `A2` by fixed point gives a two-variable nonlinear
reduced model. It is still cheap and much better conditioned than the full
PNP-BV solve.

Useful limiting cases:

- If R1 is one-way cathodic and `H_s` is fixed,
  `R1 = A1 O_b / (1 + A1/D_O)`. This recovers the usual transition from
  kinetic control to the O2 diffusion limit.
- If R2 is one-way and R1 has already been estimated,
  `R2 = A2 (P_b + R1/D_P) / (1 + A2/D_P)`.
- If anodic R1 dominates, the same linear system enforces peroxide
  depletion through `P_s`; the rate cannot exceed the peroxide supply
  implicit in `P_s >= 0`.

This algebraic model is also a compact way to interpret the inverse problem:
steady I-V data mostly observes these effective rate combinations, which is
why the `log k0`/`alpha` Tafel ridges persist even after the numerical
log-rate fix.

## Voltage-specific analytical initial condition

This is the practical output. For a given experiment and voltage:

1. Convert voltage to nondimensional `phi_e = V_RHE/V_T`.
2. Start with `R1=R2=0`, `H_o=H_b`, `phi_o=0`, `psi_D=phi_e`.
3. Compute `H_s = H_o exp(-psi_D)`.
4. Build `A1`, `B1`, `A2` from the production BV parameters.
5. Solve the 2x2 algebraic system for `R1`, `R2`.
6. Update
   `O_s = O_b - R1/D_O`,
   `P_s = P_b + (R1-R2)/D_P`,
   `H_o = H_b - (R1+R2)/D_H`,
   `phi_o = log(H_o/c_-^b)`,
   `psi_D = phi_e - phi_o`.
7. Under-relax and repeat steps 3-6 until the rates stop changing.
8. Construct outer linear profiles:

   ```text
   O_outer(y) = O_s + (O_b - O_s)y,
   P_outer(y) = P_s + (P_b - P_s)y,
   H_outer(y) = H_o + (H_b - H_o)y.
   ```

9. Construct the Debye correction:

   ```text
   lambda_D,o = sqrt(eps / (2 max(H_o, floor))),
   psi(y) = 4 atanh(tanh(psi_D/4) exp(-y/lambda_D,o)).
   ```

10. Build the composite IC:

    ```text
    phi_IC(y) = log(H_outer(y)/c_-^b) + psi(y),
    H_IC(y)   = H_outer(y) exp(-psi(y)),
    O_IC(y)   = O_outer(y),
    P_IC(y)   = max(P_outer(y), floor).
    u_i_IC    = log(c_i_IC).
    ```

For the 2D rectangle, use this as a `y`-only profile and interpolate it
across `x`.

This IC should place Newton much closer to the correct branch because it
already contains the exponentially depleted/enriched Debye-layer state. It
also respects the outer reaction-diffusion mass balances instead of seeding
every concentration at its bulk value.

## Can H+ be removed analytically like ClO4-?

Not exactly in the full model. `H+` participates in both BV reactions through
the `(c_H/c_ref)^2` factors and has nonzero boundary flux. Removing it
globally would erase a real reactant transport limitation.

But three controlled reductions are available:

1. **Inner-layer Boltzmann H+.** Use `H+` Boltzmann only inside the Debye
   layer. This is the best initialization strategy and the least invasive
   model change.
2. **Outer ambipolar H+.** Replace outer proton NP with `J_H=-2D_H grad H`.
   This keeps proton depletion while removing outer Poisson stiffness.
3. **All-Boltzmann H+ surrogate.** Set `H_s = H_b exp(-psi_D)` directly and
   drop proton transport. This is useful as a fast surrogate or diagnostic,
   but it is a different physical model.

The recommended next model-development step is the first two together: an
outer electroneutral diffusion/BV solve plus analytical Debye-layer matching.

## Relationship to Stern layer support

The current no-Stern production mode imposes Dirichlet `phi=phi_applied` at
the electrode. That forces the full applied potential into the resolved
diffuse layer and makes large positive voltages extremely stiff.

Physically, this is a strong assumption. It identifies the solution-side
potential at the reaction plane with the metal potential:

```text
phi_s = phi_m = phi_applied.
```

Real electrode/electrolyte interfaces normally contain a compact Stern layer:
ions have a finite closest-approach distance, solvent is structured near the
surface, and some voltage can drop across this compact region before the
diffuse PNP layer begins. The applied metal-to-bulk voltage is therefore more
naturally split as

```text
phi_m - phi_bulk = Delta_phi_Stern + Delta_phi_Diffuse.
```

The current Dirichlet model effectively sets `Delta_phi_Stern = 0`, so the
resolved diffuse layer must absorb the whole interfacial drop. A Stern model
lets `phi_s` float and uses a capacitance law to decide how much voltage sits
in the compact layer versus the diffuse layer:

```text
sigma = C_S (phi_m - phi_s).
```

In the PNP weak form this is a Robin-type electrostatic boundary condition,
not a prescribed-potential condition. In words:

```text
Dirichlet:  prescribe solution potential phi_s.
Stern:      prescribe metal potential phi_m and relate diffuse-layer charge
            to the compact-layer voltage drop phi_m - phi_s.
```

This is generally more physical for an electrochemical interface, but it
introduces a new physical parameter, the Stern capacitance `C_S`. If `C_S` is
poorly chosen, the model can be structurally more physical while still being
less predictive.

The limiting behavior is useful:

```text
C_S -> infinity:  phi_m - phi_s -> 0, recovering the current Dirichlet-like
                  no-Stern behavior.

finite C_S:       phi_s is not pinned to phi_m; some applied voltage can drop
                  in the compact Stern layer.
```

For the positive-voltage solver problem, finite `C_S` matters because it can
keep the resolved solution-side potential `phi_s` much smaller than the metal
voltage `phi_m`. That reduces the extreme Boltzmann source
`c_-^b exp(phi_s)` that currently drives the Poisson residual wall.

A Stern formulation replaces the electrode Dirichlet potential with a
capacitance relation and evaluates BV using

```text
eta = phi_m - phi_s - E_eq.
```

The same analytical framework still applies, but `psi_D` is no longer simply
`phi_e - phi_o`. It is solved with a compact capacitance relation:

```text
sigma_GC(psi_D) = C_S (phi_m - phi_s),
phi_s = phi_o + psi_D,
```

where the Gouy-Chapman surface charge in nondimensional units follows from
the first integral,

```text
sigma_GC ~= sign(psi_D) sqrt(8 eps h_o) sinh(|psi_D|/2)
```

up to the exact sign convention used in the weak Poisson boundary term.

This may be more than an initializer: it is a route to replacing the explicit
Debye layer with an algebraic boundary condition.

## What is solved vs what is analytical

The safest first use is **not** to replace the model. Use the analytical
Debye/Stern construction as an initial condition for the existing full PNP-BV
solver.

In the current resolved model, the PDE solver computes:

```text
bulk/outer transport + explicit Debye layer + BV surface flux
```

The proposed analytical-IC workflow keeps the same full PDE and BV equations,
but starts them from a better state:

```text
analytical formula computes: Debye-layer phi and H+ correction
PDE solver still solves:     full PNP-BV after initialization
```

For a later reduced model, the split would be stronger:

```text
PDE solver solves:
  outer O2, H2O2, and H+ transport
  possibly an outer electroneutral potential relation

analytical boundary map supplies:
  reaction-plane phi_s from Stern/diffuse-layer charge balance
  reaction-plane H_s = H_o exp(-(phi_s - phi_o))
  neutral surface values O_s ~= O_o and P_s ~= P_o

BV still imposes:
  O2 flux    = -R1
  H2O2 flux  = R1 - R2
  H+ flux    = -2R1 - 2R2
```

So the algebraic Debye/Stern relation does not replace the Butler-Volmer
boundary condition. It only replaces, or initially approximates, the thin
electrostatic layer that maps outer electrolyte values to reaction-plane
values. BV is still the reactive flux law.

This distinction matters. As an **initial condition**, the analytical Debye
profile cannot change the physical solution if the full solver converges; it
only improves basin access. As a **reduced boundary model**, it becomes a
modeling approximation and must be validated against resolved PNP-BV wherever
the resolved solver is available.

## Recommended implementation path

1. Add an optional `debye_boltzmann` initializer for the log-c stack.
   It only needs the current mesh coordinate `y`, nondimensional scaling,
   BV reaction config, and Boltzmann counterion config.
2. Start with the no-Stern formula above and ignore steric in the IC.
3. Clamp only for floating-point safety:
   `O_s`, `P_s`, `H_o`, `H_s` should be floored before taking logs, but the
   reduced solve should report when it hits a floor because that signals a
   diffusion-limit regime.
4. Use it first as a cold-start replacement at voltages near and above
   `+0.60 V`, where the current linear IC is known to be very far from the
   Boltzmann layer.
5. Compare against existing converged points on `[-0.50,+0.60] V` by
   measuring SNES iterations and final observables. The IC should not change
   the converged solution; it should only improve basin access.
6. If it works, use the same reduced algebraic model as a cheap per-voltage
   predictor before full PNP-BV solves.

## What this can and cannot answer

Can we get an analytical solution to the whole system?

No, not for the full 2D production PNP-BV model with finite reaction fluxes,
steric terms, and two coupled BV reactions.

Can we get an analytical solution for part of the system?

Yes. The zero-flux charged layer has a Boltzmann form; the symmetric
one-dimensional Debye layer has a Gouy-Chapman/Poisson-Boltzmann solution;
the electroneutral outer proton equation reduces to ordinary diffusion with
effective `2D_H`; and the outer steady neutral/reactive transport reduces to
a two-rate algebraic system.

Can we get a good analytical IC for a given voltage and experiment?

Yes. The matched Debye-layer plus outer algebraic profile above is likely the
highest-value analytical simplification for the current solver. It directly
targets the failure mode documented in `docs/peroxide_window_investigation.md`:
Newton is failing because it starts far from the exponentially depleted
Boltzmann proton layer and the exponentially enriched Boltzmann counterion
layer. The proposed IC starts on that manifold.
