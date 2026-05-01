# V19 — _U_CLAMP sweep with log-rate BV

Per GPT's `PNP Log Rate Next Steps Handoff.md` Task 5.  Tests whether the
bulk PDE residual's `_U_CLAMP` (the symmetric clamp on `u = ln c` used in
the time-derivative and diffusion terms) is contaminating the extended-V
FIM result.

The log-rate BV path uses `ui[i]` directly and bypasses the clamp on the
boundary residual.  But the bulk PDE still uses `c_i = exp(clamp(u_i, ±_U_CLAMP))`.
If the clamp is biting, results should change between clamp=30 and 100.

## Configuration

```
bv_log_rate = True
cap = 50
V_GRID = [-0.10, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
u_clamp ∈ {30, 60, 100}
```

## Result

```
u_clamp     sv_min       cond(F)     ridge_cos   weak_eigvec
30          2.510e+00    1.79e+07    0.031       [+1.00,+0.03,-0.02,-0.00]
60          2.510e+00    1.79e+07    0.031       [+1.00,+0.03,-0.02,-0.00]
100         2.510e+00    1.79e+07    0.031       [+1.00,+0.03,-0.02,-0.00]
```

**Identical to all printed digits across all three clamp values.**  cd, pc,
r1, r2 also identical (verified by comparing the per-V observables CSV).

## Interpretation

The bulk PDE clamp is **not** contaminating the FIM result.  Even at the
hardest voltage in the grid (V=+0.60, where c_H2O2_min reaches 1e-9 at
cap=50), the bulk `u_H2O2` stays well above -30 in the converged solution.
The clamp value only affects what happens when Newton transiently steps
into very low-c states during iteration; at convergence those transients
have relaxed.

So the extended-V FIM result reported in `extended_v_to_60/` is robust
to the clamp value.  No follow-up needed.

## Recommendation

- Keep `u_clamp = 30` as the default; widening it has no effect on
  converged solutions in this regime.  (Widening could matter for harder
  V > +1.0 V cases or for adjoint computations through tightly-coupled
  states, which we haven't tested here.)
- The "U_CLAMP still active in bulk PDE" caveat from
  `CHATGPT_HANDOFF_6_LOGRATE_BREAKTHROUGH.md` §"Caveats I'm aware of"
  can be **closed**: it's not a meaningful confound for the published
  results.
