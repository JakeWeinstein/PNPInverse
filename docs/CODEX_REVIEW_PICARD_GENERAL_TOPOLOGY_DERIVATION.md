# Codex Review - Picard General Topology Derivation

Date: 2026-05-08

Reviewed document:

- `docs/picard_general_topology_derivation.md`

Related implementation references:

- `Forward/bv_solver/picard_ic.py`
- `Forward/bv_solver/forms_logc.py`
- `Forward/bv_solver/forms_logc_muh.py`
- `scripts/_bv_common.py`

## Executive summary

The core idea is viable for the immediate M3a.3 target: replacing the hardcoded
legacy 2x2 sequential Picard algebra with a small N-reaction linear solve over
the BV turnover rates. The derivation correctly reproduces the legacy
sequential 2x2 matrix and gives the right O2-coupled matrix for the
parallel `R_2e + R_4e` topology.

However, the current writeup overstates the generality. It is not a fully
general reaction-network solve. It is a Picard-linearized initializer where
the selected cathodic/anodic substrate concentrations are treated linearly,
while H+, gamma, eta, and all non-substrate concentration factors are frozen
during each inner linear solve and updated only in the outer Picard iteration.

Before implementing from this derivation, Claude should address the four
issues below.

## Finding 1 - Signed proton stoichiometry is lost

Location:

- `docs/picard_general_topology_derivation.md:68-69`
- also affects the formulas at lines 89-95, 232-236, 257-260, and 316-318

The derivation writes the generalized proton outer balance using
`abs(s_H,j)`:

```text
H_o = H_b - sum_j |s_H,j| / 2 * R_j / D_H
```

That is correct for the current acid ORR reactions because cathodic-direction
stoichiometry always consumes protons:

```text
s_H,R2e = -2
s_H,R4e = -4
```

But it is not a general topology formula. It silently gives the wrong sign
for any reaction whose signed cathodic-direction stoichiometry produces H+,
or for any rate convention where a reaction can contribute with the opposite
signed stoichiometric effect.

The signed ambipolar formula should be:

```text
H_o = H_b + sum_j s_H,j * R_j / (2 D_H)
```

For the current sequential reactions:

```text
H_o = H_b + (-2 R1 - 2 R2) / (2 D_H)
    = H_b - (R1 + R2) / D_H
```

For the parallel `R_2e + R_4e` reactions:

```text
H_o = H_b + (-2 R_2e - 4 R_4e) / (2 D_H)
    = H_b - (R_2e + 2 R_4e) / D_H
```

Recommendation:

- Replace the absolute-value formula with the signed formula everywhere.
- If the implementation deliberately supports only proton-consuming acid ORR
  topologies, say that explicitly and validate `s_H,j <= 0` instead of calling
  the formula general.

## Finding 2 - Constant anodic branch is omitted

Location:

- `docs/picard_general_topology_derivation.md:31`
- also affects lines 108-127 and the matrix/RHS definition in section 4

The derivation states:

```text
anodic_species: int | None
None => irreversible
```

That does not match the residual. Both residual paths support a reversible
reaction with `anodic_species is None` and positive `c_ref_model`:

- `Forward/bv_solver/forms_logc.py:407-413`
- `Forward/bv_solver/forms_logc.py:433-437`
- `Forward/bv_solver/forms_logc_muh.py:446-452`
- `Forward/bv_solver/forms_logc_muh.py:473-477`

That branch is an affine constant anodic term, not a linear surface-species
term:

```text
R_j = A_j c_cat,s - C_j
```

where, matching the residual,

```text
C_j = k_j c_ref_model exp((1 - alpha_j) n_e,j eta_j)
```

There is no gamma factor in this constant branch because the residual does not
multiply the reference concentration by a surface activity coefficient.

A general affine form that matches all currently legal residual branches is:

```text
R_j = A_j c_cat,s - B_j c_anod,s - C_j
```

with:

```text
B_j = 0 if anodic_species is None
C_j = reversible_j and anodic_species is None and c_ref_model > 0
      ? k_j c_ref_model exp((1 - alpha_j) n_e,j eta_j)
      : 0
```

Then:

```text
M_jk = delta_jk
     - A_j s_cat,k / D_cat
     + B_j s_anod,k / D_anod

b_j  = A_j c_cat,b
     - B_j c_anod,b
     - C_j
```

Recommendation:

- Either add the affine `C_j` branch to the derivation and implementation, or
  validate it away before entering the generalized Picard.
- Do not assume `anodic_species=None` implies irreversible unless config
  validation enforces that invariant.

## Finding 3 - Generality and convergence are overstated

Location:

- `docs/picard_general_topology_derivation.md:173-177`
- also affects section 7 convergence language

The N x N solve only handles linear dependence on the selected cathodic and
anodic substrate concentrations. It does not include derivatives or direct
linear coupling for:

- H+ concentration factors
- any other `cathodic_conc_factors`
- gamma
- psi_D
- psi_S
- eta_j

Those are frozen during the inner solve and updated by the outer Picard loop.
That is a reasonable matched-asymptotic initializer strategy, but it is not a
general reaction-network solve. For example, if a future reaction puts a
dynamic species into `cathodic_conc_factors` with nontrivial powers, the rate
dependence becomes nonlinear in the rates and the current N x N derivation is
only a Picard lagging approximation.

Recommendation:

- Rename the scope in the document to something like "general linear-substrate
  BV topology Picard initializer" or "ORR topology Picard initializer".
- State explicitly that non-substrate concentration factors are Picard-lagged.
- Add implementation guards or warnings for unsupported topologies, especially
  if a factor species is also strongly changed by the reaction network.
- Keep robust fallback behavior: singular matrix, nonfinite rates, negative or
  floored concentrations, and max-iteration failure should fall back to the
  existing linear-phi IC path rather than hard-failing production sweeps.

## Finding 4 - Pure-2e regression is underspecified

Location:

- `docs/picard_general_topology_derivation.md:341-342`

The verification ladder says:

```text
T2 - pure-2e via parallel preset: k0_R4e = 0 => R_4e approx 0,
(R_2e, O_s, P_s, H_o) = (R_1, O_s, P_s, H_o)_legacy
```

That comparison is underspecified. The legacy sequential topology also has
the second peroxide-consuming reaction:

```text
R2: H2O2 + 2H+ + 2e- -> 2H2O
```

So a pure `R_2e` parallel run cannot match the full legacy sequential run
unless the legacy `R2` branch is disabled.

Recommendation:

- Change T2 to compare against a legacy sequential config with `k0_R2 = 0`.
- In that comparison, map `R_2e` to legacy `R1`.
- Expected matching identities are:

```text
R_4e = 0
R_2e = R1_legacy_with_R2_disabled
O_s  = O_b - R_2e / D_O
P_s  = P_b + R_2e / D_P
H_o  = H_b - R_2e / D_H
```

The current T3 pure-4e test is otherwise correctly specified:

```text
k0_R2e = 0
R_2e approx 0
P_s approx P_b
H_o = H_b - 2 R_4e / D_H
O_s = O_b - R_4e / D_O
```

## What is mathematically sound

The following parts of the derivation look correct for the intended ORR
topologies:

1. The ordinary-species surface balance:

```text
c_i,s = c_i,b + sum_j s_i,j R_j / D_i
```

This matches the current sequential formulas:

```text
O_s = O_b - R1 / D_O
P_s = P_b + (R1 - R2) / D_P
```

2. The legacy 2x2 reduction in section 5.

The matrix entries match `Forward/bv_solver/picard_ic.py:489-495`:

```text
m11 = 1 + A1 / D_O + B1 / D_P
m12 = -B1 / D_P
m21 = -A2 / D_P
m22 = 1 + A2 / D_P
rhs1 = A1 O_b - B1 P_b
rhs2 = A2 P_b
```

3. The parallel `R_2e + R_4e` matrix in section 6.

Both reactions consume O2, and only `R_2e` touches H2O2. The documented matrix:

```text
M = [ 1 + A_2e/D_O + B_2e/D_P    A_2e/D_O       ]
    [ A_4e/D_O                    1 + A_4e/D_O   ]

b = [ A_2e O_b - B_2e P_b ]
    [ A_4e O_b             ]
```

is consistent with the signed stoichiometry:

```text
R_2e stoich = [-1, +1, -2]
R_4e stoich = [-1,  0, -4]
```

4. The gamma powers.

The residual uses reaction-plane log concentrations, so the cathodic branch
gets one gamma from the cathodic substrate and one gamma per concentration
factor power. For the current acid ORR factors:

```text
R_2e cathodic gamma power = 1 + 2 = 3
R_4e cathodic gamma power = 1 + 4 = 5
```

The reversible anodic surface-species branch gets one gamma from the anodic
species. The constant `c_ref_model` anodic branch gets no gamma.

## Suggested implementation contract for Claude

Implement the generalized Picard as a conservative initializer, not as a new
global nonlinear solver.

Minimum contract:

1. Use a rate vector `R` of length `N = len(bv_reactions)`.
2. Build per-reaction `A_j`, `B_j`, and optional constant `C_j` in log-space.
3. Assemble the affine linear system:

```text
M_jk = delta_jk
     - A_j s_cat,k / D_cat
     + B_j s_anod,k / D_anod

b_j = A_j c_cat,b
    - B_j c_anod,b
    - C_j
```

4. Use signed stoichiometry for all post-solve surface balances:

```text
c_i,s = c_i,b + sum_j s_i,j R_j / D_i
H_o   = H_b   + sum_j s_H,j R_j / (2 D_H)
```

5. Keep H+, gamma, Stern split, eta, and concentration factors Picard-lagged
   exactly as the current two-reaction initializer does.
6. Preserve the legacy closed-form post-loop `P_s` and `O_s` reconstruction
   only behind a sequential-topology hint.
7. Default all unknown topologies to naive signed flux reconstruction.
8. On singular matrix, nonfinite output, failed convergence, or physically
   unusable concentrations, return a clear failure reason so the adapter can
   fall back to the linear-phi initializer.
9. Keep all Picard work under `firedrake.adjoint.stop_annotating()`.
10. Maintain sequential regression tolerances before enabling the parallel
    topology in production sweeps.

## Verification changes

Recommended verification ladder changes:

1. T1: legacy sequential byte-equivalence with both reactions enabled.
2. T2: pure-2e parallel with `k0_R4e = 0` compared against legacy sequential
   with `k0_R2 = 0`.
3. T3: pure-4e parallel with `k0_R2e = 0`, checking `P_s approx P_b` and
   `H_o = H_b - 2 R_4e / D_H`.
4. T4: gamma-power probe:

```text
R_2e cathodic prefactor scales as gamma^3
R_4e cathodic prefactor scales as gamma^5
R_2e anodic surface-species prefactor scales as gamma^1
constant c_ref anodic branch, if tested, scales as gamma^0
```

5. Add one unit test for the signed H+ formula using a synthetic reaction with
   positive `s_H` if the document continues to claim general topology support.
6. Add one unit test for the reversible `anodic_species=None, c_ref_model>0`
   branch, or add validation proving that branch cannot reach Picard.

## Bottom line

Proceed with the generalized Picard for the ORR sequential and parallel
2e/4e cases. The method is mathematically coherent as a Picard-lagged,
linear-substrate initializer. Fix the signed H+ balance, account for or reject
the constant anodic branch, and narrow the stated scope before treating the
document as implementation source-of-truth.
