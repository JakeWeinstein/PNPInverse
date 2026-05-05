# 4sp Bikerman-Corrected IC Math Review

**Date:** 2026-05-04  
**For:** Claude Code handoff  
**Reviewed context:** `docs/4sp_drop_boltzmann_investigation.md`,
`docs/PNP_BV_Analytical_Simplifications.md`,
`Forward/bv_solver/forms_logc.py`, and `scripts/_bv_common.py`.

## Short verdict

The failure diagnosis in `docs/4sp_drop_boltzmann_investigation.md` is
basically right: the pure Boltzmann 4sp `debye_boltzmann` IC can seed dynamic
`ClO4-` far above the steric packing limit, so Newton starts from a state that
is outside the intended Bikerman manifold.

However, the proposed "Bikerman-corrected IC" in section 8 is not
mathematically consistent with the current residual as written. The gamma
formula in the note is the conventional Bikerman/Kornyshev modified
Poisson-Boltzmann formula, but the code currently uses the opposite steric
sign from that conventional formula:

```python
packing = max(1 - sum_j a_j*c_j, packing_floor)
mu_steric = ln(packing)
Jflux = D*c*(grad(u) + grad(z*phi) + grad(mu_steric))
```

Before implementing the proposed IC as "the Bikerman IC", decide whether the
code's steric sign is intentional. If not, fix the sign first. If the sign is
intentional, the standard MPB gamma formula is not the zero-flux relation for
the implemented equations.

## What is correct in the investigation note

The pure Boltzmann seed is incompatible with a finite-volume steric packing
constraint at anodic voltage.

At `V_RHE = +0.3 V`, the note estimates:

```text
psi_D ~= 11.5
c_ClO4 = 0.2 * exp(11.5) ~= 2e4
a = 0.01, so 1/a ~= 100
```

That is enough to put `1 - a*c_ClO4` far below zero before including any other
species. The residual clamps `packing` to `packing_floor`, but that just hides
the invalid state behind `ln(1e-8)` and a huge local derivative. The direct
`z=1` Newton failure and fallback to `linear_phi` are therefore plausible and
consistent with the observed diagnostics.

The note is also right that a steric-aware IC is the right class of next step.
The current pure Boltzmann IC is helpful for 3sp+analytic-Boltzmann because
the analytic `ClO4-` is not a dynamic species and does not enter
`mu_steric`. In 4sp dynamic mode, `ClO4-` is a primary variable and the IC must
respect the packing domain from the start.

## Main math issue: steric sign convention

The current code's zero-flux condition is, schematically:

```text
grad(u_i + z_i*phi + ln(theta)) = 0
theta = 1 - sum_j a_j*c_j
```

That implies:

```text
c_i * theta * exp(z_i*phi) = constant
```

For an enriched counterion, decreasing `theta` forces `c_i` larger, not
smaller. This is the inverse of the usual lattice-gas Bikerman correction.

The conventional Bikerman/Kornyshev MPB relation comes from:

```text
grad(u_i + z_i*phi - ln(theta)) = 0
```

which gives:

```text
c_i = gamma(psi) * c_i_bulk * exp(-z_i*psi)
gamma(psi) = 1 / (1 + sum_j a_j*c_j_bulk*(exp(-z_j*psi) - 1))
```

That is the formula written in section 8 of the investigation note. It is
standard, and it saturates counterion concentration as expected. But it is
standard for `-ln(theta)` in the electrochemical potential, not for the
currently implemented `+ln(theta)` term.

Claude should not blindly add the gamma correction to `_try_debye_boltzmann_ic`
without first resolving this sign mismatch. Otherwise the IC will be
"Bikerman" in the literature sense while the residual remains a different
steric model.

## Secondary issue: cap must be total-packing aware

Even if the conventional sign is adopted, the proposed option 2a is slightly
too loose because it only targets:

```text
c_ClO4 <= 1/a
```

The residual actually requires:

```text
1 - sum_j a_j*c_j > 0
```

For the production nondimensional values:

```text
C_O2_hat = 1.0
C_HP_hat = C_CLO4_hat = 0.2
a = 0.01
```

So `O2` alone consumes about `0.01` of the packing fraction. A binary
H+/ClO4- gamma correction can put `c_ClO4` near `99.5` at `psi ~= 11.5`.
Adding `O2 ~= 1` then makes the total packing slightly negative. The IC tests
should check total packing, not only `c_ClO4 < 100`.

For a first implementation, either:

1. Build gamma using all species that are assumed to participate in the local
   lattice-gas equilibrium, or
2. Keep the simple charged-species gamma but add a packing-safe cap using the
   current neutral/H+ occupancy:

```text
c_ClO4 <= (1 - margin - sum_{j != ClO4} a_j*c_j) / a_ClO4
```

Option 2 is more heuristic, but it directly prevents an invalid IC.

## Recommended path forward

### Step 1: audit and decide the steric sign

Check every active backend:

- `Forward/bv_solver/forms_logc.py`
- `Forward/dirichlet_solver.py`
- `Forward/robin_solver.py`
- any MMS or weak-form documentation that mirrors this term

If the intended model is conventional Bikerman sterics, change the
electrochemical potential contribution to `-ln(packing)` or equivalently
change the flux contribution from `+grad(mu_steric)` to `-grad(mu_steric)` if
`mu_steric` remains `ln(packing)`.

This is a model-level change and should get its own tests. It may affect old
4sp equivalence expectations, because the previous "Bikerman" behavior was not
the conventional saturation law.

### Step 2: implement a packing-safe 4sp IC

After the sign decision, update only the synthesized 4sp counterion branch in
`_try_debye_boltzmann_ic`.

Do not change the 3sp+analytic-Boltzmann path unless the goal is also to add
steric saturation to the analytic Poisson source.

A minimal conventional-sign IC can still start with option 2a:

```text
psi(y): keep the existing Gouy-Chapman profile
c_H(y): gamma * H_outer * exp(-psi)
c_ClO4(y): gamma * c_ClO4_bulk * exp(+psi)
```

But add a total-packing safety check/cap before interpolating `u_3`.

### Step 3: validate IC quality before long voltage sweeps

Add or update tests so they assert:

```text
all nodes: packing = 1 - sum_j a_j*c_j > margin
bulk: c_ClO4 ~= c_bulk
surface: c_ClO4 is near the steric scale but below the total-packing limit
3sp+Boltzmann initializer behavior remains unchanged
```

Then run the smallest useful solve:

```text
4sp dynamic + debye_boltzmann + V_RHE=+0.3
```

Only after that passes should the peroxide-window sweep be repeated.

## What I would tell Claude to change in the existing note

Section 8 should be softened from "the math is correct, implement option 2a"
to:

```text
The MPB gamma formula is the conventional Bikerman correction, but it assumes
the steric term enters as -ln(packing) in the electrochemical potential. The
current code appears to use +ln(packing), so first audit/fix the residual sign
or derive the IC for the implemented sign. Also make the IC cap total-packing
aware, because c_ClO4 < 1/a alone is not sufficient when O2, H+, and H2O2 also
occupy volume.
```

The overall direction is still reasonable: 4sp dynamic with steric saturation
is the physically cleaner path than high-voltage 3sp+unbounded-Boltzmann.
But the next implementation should treat the steric sign and total packing
domain as blockers, not details.

