# Follow-up: precision diagnostic + multi-experiment FIM next

## TL;DR

Following your "Recommended Next Path" handoff, ran the optional solver-precision diagnostic first. Result: **precision matters substantially but doesn't solve the ridge**. Now moving to the main path — multi-experiment FIM screening.

## Solver-precision diagnostic

Same TRF + per-observable adjoint Jacobian inverse as before; clean data; +20% init; only difference is forward-solve tolerances.

| Param | Standard (rtol 1e-10, ss 1e-4) | Tight (rtol 1e-13, ss 1e-6) |
|-------|-------------------------------:|----------------------------:|
| k0_1  | -65.3%                          | **-11.1%** |
| k0_2  | -23.9%                          | -33.0% |
| α_1   | +4.5%                           | +11.6% |
| α_2   | +1.9%                           | +5.4% |
| cost  | 1.15e-02                        | 4.88e+01 |
| evals | 26                              | 25 |

TRUE forward observables agreed to 5+ significant figures between standard and tight, so the precision wall isn't about cd/pc accuracy at TRUE — it's about how reliably the optimizer can resolve directional gradients on the ridge during iteration.

### Mechanism (proposed)

At standard precision, the SS criterion (`rel_tol=1e-4`) absorbs ~1e-4 wiggle in cd/pc that's stable across optimizer iterations. So successive evals at the same θ give the same observables, low residuals, and TRF can drive cost to ~zero by sliding along the ridge to whichever point happens to satisfy `||r||² ≈ 0` first. The recovered point is essentially "the closest ridge point to the init guess given the standard-precision residual landscape."

At tight precision (`rel_tol=1e-6`, 5 consec convergence), the residual landscape is more rugged because tiny iteration-history-dependent variations in the converged state become visible. The optimizer can no longer slide along the ridge cleanly because it now sees that "moving along the ridge" produces small but nonzero residuals. So it moves *off* the ridge in the k0_1 direction, recovering k0_1 substantially closer to TRUE — but α_1 and k0_2 drift to compensate, and the final cost stays large because the new "valley" the optimizer settles in is also non-true.

### Key takeaway

**Precision shifts which wrong-point the optimizer lands at, not whether it finds TRUE.** It's not numerical — the FIM cond ~ 1e11 ridge is structural. Tight precision exposed previously-buried information about k0_1, but α and k0_2 followed the (rotated) ridge to compensate.

This rules out "tighter solver = solve k0" as a viable single-experiment path. It also reinforces the multi-experiment framing — the ridge has to be *bent*, not just *exposed at higher resolution*.

## Now executing your Phase B (multi-experiment FIM screening)

Building `v18_logc_multiexperiment_fim.py` with the design list you proposed:

| Design | Experiments | Observables |
|--------|------------|-------------|
| A | ORR single L₀ | CD only |
| B | ORR single L₀ | CD + PC |
| C | ORR L₀ + 2L₀ + 4L₀ | CD + PC each |
| D | H2O2-fed at L₀ (c_O2=0, c_H2O2>0) | CD only |
| E | ORR L₀ + H2O2-fed L₀ | CD+PC ⊕ CD |
| F | ORR L₀/2L₀/4L₀ + H2O2-fed L₀ | full design |

Per-design metrics: σ_min(S_white), cond(F), smallest eigenvector of F, *and* the inner product of that eigvec with the canonical ridge direction `[+1, 0, 0.02, 0]` (per breakthrough analysis: 1% α shift ↔ 47% k0 shift on the canonical ridge). If the weak eigvec rotates *away* from this canonical direction in a multi-experiment design, that's quantitative proof the ridge has been broken.

Runtime estimate: 4 experiments × ~8 min FIM each = ~30–40 min total. Compute-bound by 8 forward solves per experiment (central FD with two step sizes per parameter; reusing the existing infrastructure).

Decision rule (yours): only run TRF inverse on a design if cond(F) drops from ~1e11 to ≤ 1e8, AND the weak eigvec rotates significantly off the canonical ridge.
