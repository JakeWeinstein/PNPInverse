# Codex Review - Picard General Topology Derivation Round 2

Date: 2026-05-08

Reviewed documents:

- `docs/CHATGPT_HANDOFF_19_PICARD_DERIVATION_ROUND2.md`
- `docs/picard_general_topology_derivation.md` v2

Prior review:

- `docs/CODEX_REVIEW_PICARD_GENERAL_TOPOLOGY_DERIVATION.md`

## Approval status

Do not start implementation yet. The v2 algebra is sound for the intended
ORR sequential and parallel 2e/4e Picard initializer, but three doc-level
issues should be fixed before treating `docs/picard_general_topology_derivation.md`
as the implementation contract.

All three are small, but they affect either legacy byte-equivalence or the
verification ladder.

## Finding 1 - Delta uses the wrong rate symbol

Location:

- `docs/picard_general_topology_derivation.md:294-301`

The pseudocode currently says:

```text
R_new            <- numpy.linalg.solve(M, b)
R_j              <- (1 - omega) R_j_old + omega R_new_j
...
delta            <- sum_j |R_j_new - R_j_old| / max(|R_j|, 1e-30)
```

This names the raw linear-solve output `R_new`, then uses `R_j_new` in the
convergence check after `R_j` has already been relaxed. That is ambiguous,
and if implemented literally it changes the current convergence semantics.

The existing two-reaction Picard loop computes delta from the relaxed rate:

```text
R1 = (1 - omega) R1_old + omega R1_solve
R2 = (1 - omega) R2_old + omega R2_solve
delta = |R1 - R1_old| / max(|R1|, 1e-30)
      + |R2 - R2_old| / max(|R2|, 1e-30)
```

If the generalized implementation uses the raw solve output in delta, T1
byte-equivalence can fail and the effective tolerance shifts by about
`omega`.

Required edit:

```text
R_solve          <- numpy.linalg.solve(M, b)
R_old            <- copy(R)
R                <- (1 - omega) R_old + omega R_solve
...
delta            <- sum_j |R_j - R_old_j| / max(|R_j|, 1e-30)
```

Also update any prose that refers to `R_new` so the raw solve and relaxed
Picard state are distinct.

## Finding 2 - Failure contract conflicts with legacy floors

Location:

- `docs/picard_general_topology_derivation.md:383-389`

The v2 implementation contract says that physically unusable surface
concentrations, for example negative `O_s`, should return failure and fall
back to linear-phi.

That conflicts with the existing legacy Picard behavior. The current loop
floors surface concentrations after the flux update:

```text
O_s = max(O_b - R1 / D_O, 1e-300)
P_s = max(P_b + (R1 - R2) / D_P, P_FLOOR)
H_o = max(H_b - (R1 + R2) / D_H, 1e-300)
```

Failing immediately on a negative raw concentration would break sequential
byte-equivalence and can unnecessarily reject diffusion-limited states where
the legacy initializer intentionally floors the value.

Required edit:

- State that legacy-compatible floors are applied first.
- Failure should be reserved for nonfinite values, singular linear solves,
  invalid post-floor states, or unrecoverable states that cannot be safely
  floored.
- Do not make negative raw `O_s`, `P_s`, or `H_o` an automatic failure in
  the sequential-compatible path.

Suggested replacement wording:

```text
On singular M, nonfinite solve output, nonfinite post-update state, or
max_iters reached without convergence, return failure. Surface concentrations
must be reconstructed with the same legacy-compatible floors used by the
current 2x2 loop before deciding whether the state is unusable.
```

## Finding 3 - Slow pure-2e regression still compares to full sequential

Location:

- `docs/picard_general_topology_derivation.md:445-453`

T2 correctly fixes the pure-2e unit comparison:

```text
parallel k0_R4e = 0
vs.
legacy sequential k0_R2 = 0
```

But T8 still says that the slow Firedrake pure-2e parallel run should match
T7, while T7 is the full legacy sequential reaction list. That reintroduces
the round-1 ambiguity at the slow-test level.

Required edit:

Add a disabled-R2 slow reference, or make T8 compare against that reference
instead of T7.

Suggested ladder:

```text
T7  - Firedrake single-V cold, full legacy sequential.
T7b - Firedrake single-V cold, legacy sequential with k0_R2 = 0.
T8  - Firedrake single-V cold, parallel pure-2e with k0_R4e = 0;
      compare against T7b, not T7.
```

This keeps the full legacy regression while making the pure-2e parallel
comparison mathematically equivalent.

## Answers to round-2 questions

Q1 - Gamma power for constant-anodic branch:

Preserve `C_hat_j` proportional to `gamma^0` in the initializer because the
residual currently does that. If the physics is later changed, change the
residual and IC together. For this implementation, the IC must mirror the
residual.

Q2 - T11 signed-H_o framing:

Keep the signed formula and keep T11. Do not enforce `s_H <= 0` unless the
project deliberately wants ORR-only scope narrowing. The signed formula is
cheap, correct, and prevents future sign mistakes.

Q3 - T12 constant-anodic branch disposition:

Choose path (a): implement `C_hat_j`, test it, and leave it dormant in the
current ORR production presets. It is low implementation cost and keeps the
Picard contract aligned with residual branches that are already legal.

Q4 - Post-loop closed form for parallel:

Wait for empirical failure. The parallel peroxide concentration formula

```text
P_s = P_b + R_2e / D_P
```

does not have the sequential cancellation source `R1 - R2`, so preemptively
adding a topology-specific closed form is unnecessary.

Q5 - Per-reaction relaxation:

Keep a single `omega = 0.5`. The inner N x N solve already handles the direct
cross-reaction coupling. Per-reaction relaxation adds a new tuning knob with
unclear benefit.

Q6 - Picard tolerance:

Do not tighten the default tolerance preemptively. Keep `tol = 1e-6` for
legacy compatibility and only tighten after T7-T10 show that Newton is
sensitive to Picard residual distance.

Q7 - Anything missing:

Add one guard note before implementation: if future configs allow H+ to be
the `cathodic_species` or `anodic_species`, then the matrix coefficient for
that substrate must use the ambipolar transport factor `1 / (2 D_H)` rather
than the ordinary species factor `1 / D_H`. Current ORR configs do not trigger
this because H+ only appears in `cathodic_conc_factors`, but the source doc
should either state the restriction or define the transport coefficient by
species.

Suggested wording:

```text
For ordinary species, lambda_i = 1 / D_i. For H+ used as a linear substrate,
lambda_H = 1 / (2 D_H) under the ambipolar outer transport law. Current ORR
configs never use H+ as cathodic_species or anodic_species; if that remains
out of scope, reject such configs at the adapter site.
```

## Final approval condition

After the three findings above are incorporated, v2 is approved as the
implementation contract for M3a.3.

The approved scope is:

- Linear-substrate Picard initializer.
- ORR sequential and parallel 2e/4e topologies.
- Three anodic branches matching the residual.
- Signed stoichiometric flux reconstruction.
- Legacy-compatible floors and post-loop sequential closed form.
- Clean failure fallback to linear-phi IC.

Do not expand into generalized nonlinear reaction-network handling,
multi-counterion gamma, or non-sequential closed-form reconstruction until
the current verification ladder shows a concrete need.
