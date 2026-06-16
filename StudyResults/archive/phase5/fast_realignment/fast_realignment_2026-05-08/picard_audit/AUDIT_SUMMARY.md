# Phase 1.2 Audit — `picard_outer_loop_general` for parallel-2e/4e

Date: 2026-05-08
V_RHE: +0.55 (weakest cathodic drive within page-15 grid)
Stern: NONE (audit-only; production uses C_S = 0.10 F/m²)
Bikerman: NONE (audit-only; production uses Cs⁺/SO₄²⁻)
counterion: ClO₄⁻ ideal (audit baseline; production target swaps to Cs⁺ + SO₄²⁻)

## Diagnosis

**Picard converges geometrically for parallel-2e/4e topology.**

All three passes (pure-2e, pure-4e, mixed) converge at iter 20 with
`delta < 1e-6`. The plan-referenced "parallel topology cold-fails
universally" claim is stale — that referred to the M3a.2 sequential-only
Picard which has since been replaced by `picard_outer_loop_general`
(M3a.3 silently landed).

| Pass | iters | R_2e_model | R_4e_model | phi_o | psi_D | gamma_s |
|---|---|---|---|---|---|---|
| A pure-2e | 20 | -1.875e-09 | 0 | +4.59e-09 | +21.41 | 1.0 |
| B pure-4e | 20 | 0 | +7.99e-18 | 0 | +21.41 | 1.0 |
| C mixed   | 21 | -1.875e-09 | +7.99e-18 | +4.59e-09 | +21.41 | 1.0 |

## Why the rates are tiny

`psi_D = +21.41` (full applied potential, no Stern). For H⁺ (cation)
near a positive electrode (V_RHE = +0.55 V is above the pH-4 reference
phi_bulk = 0), the diffuse-layer Boltzmann factor depletes H⁺ at the
OHP:

```
c_H_OHP = c_H_bulk · exp(-ψ_D) = 0.0833 · exp(-21.41) ≈ 4.2e-11
```

The cathodic 2e rate has an `(c_H/c_H_ref)^2` factor, so the H⁺
depletion suppresses the forward ORR by `~exp(-43)` ≈ `2e-19`. The
anodic reverse (peroxide oxidation H₂O₂ → O₂) lacks this factor and
wins: R_2e ≈ -1.87e-9 (anodic-dominated).

This is **correct physics for the bare ideal-counterion model** at
V_RHE = +0.55 V. The production stack's Stern (C_S=0.10 F/m²) absorbs
most of the drop into ψ_S, leaving a small ψ_D, which makes the H⁺
depletion mild and the cathodic rate physically reasonable. The audit
does NOT exercise that path; it exercises only the bare Picard.

## Implications for the realignment plan

- **Phase 1.2 done.** No bug in `picard_outer_loop_general`; no need to
  rewrite generic Picard.
- **Phase 1.3 next.** Add `_is_reaction_disabled` + `_is_parallel_2e_4e`
  helpers and the trivial-row treatment in `_assemble_n_reaction_system`.
- **Phase 2 (multi-ion) is the load-bearing work.** Without
  Stern + multi-ion Bikerman, the bare ideal-counterion IC at the
  production V_RHE grid will give zero rates due to H⁺ depletion.
  Cs⁺/SO₄²⁻ at I=0.3 M with Stern is the physically correct setup.

## Per-iter trace excerpt (pass A, k=1..5)

```
[picard k=  1] delta=1.000e+00  R=[-9.375e-10 +0.000e+00]  c_s=[+1.000e+00 +1.000e-04 +8.333e-02]  phi_o=+2.296e-09  psi_D=+2.141e+01  psi_S=+0.000e+00  gamma_s=1.000e+00  eta=[-5.644e+00 -2.647e+01]
[picard k=  2] delta=3.333e-01  R=[-1.406e-09 +0.000e+00]  c_s=[+1.000e+00 +1.000e-04 +8.333e-02]
[picard k=  3] delta=1.429e-01
[picard k=  4] delta=6.667e-02
[picard k=  5] delta=3.226e-02
...
[picard k= 20] delta=9.537e-07  ← below tol=1e-6, converged
```

Geometric convergence at omega=0.5 (delta halves each iter as expected).

## Conclusion

Picard generalization (M3a.3) is solid. The parallel-2e/4e topology
dispatches correctly through `topology_hint='general'`. Proceed to
Phase 1.3 (add helpers) and Phase 2 (multi-ion infrastructure) without
rewriting the Picard.
