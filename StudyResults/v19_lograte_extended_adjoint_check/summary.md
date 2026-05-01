# V19 — adjoint vs FD at extended V_GRID with log-rate

Per GPT's `PNP Log Rate Next Steps Handoff.md` Task 3.  Verifies pyadjoint
gradient against central FD at the new important voltages.

**Configuration:** `bv_log_rate=True`, cap=50, V_TEST = [+0.30, +0.40, +0.50, +0.60].
Adjoint pattern: warm-start unannotated, then 5 annotated SNES iterations
(matches `v18_logc_lsq_inverse.py:solve_warm_annotated`, validated at V=+0.20
with the standard form).

## Result: 14/32 PASS, 18 FAIL — but with a clear pattern.

```
V_RHE   verdict at log_k0_2, alpha_2 components
+0.30   FAIL (adjoint ≈ 0; FD says ~1e-5)
+0.40   FAIL (adjoint 1e-4× smaller than FD)
+0.50   FAIL (adjoint ~5× smaller than FD)
+0.60   PASS (1e-7 relative error, all 8 components)
```

The non-zero-component failures at V<+0.60 share a striking pattern:

```
V_RHE  component         adjoint        FD            ratio
+0.30  cd dlog_k0_1      -3.76e-6      -1.88e-6      adjoint = 2.000 × FD
+0.30  cd alpha_1        -1.11e-4      -5.57e-5      adjoint = 2.000 × FD
+0.40  cd dlog_k0_1      -1.19e-11     -5.94e-12     adjoint = 2.000 × FD
+0.40  cd alpha_1        -2.59e-10     -1.30e-10     adjoint = 2.000 × FD
+0.50  cd dlog_k0_1      +5.18e-11     +2.85e-11     adjoint = 1.819 × FD
+0.50  cd alpha_1        +7.26e-10     +3.99e-10     adjoint = 1.819 × FD
+0.60  cd dlog_k0_1      +2.26e-9      +2.26e-9      adjoint = 1.000 × FD ✓
```

The ~2× factor on the *non-zero-clipped* components and a near-zero
adjoint on the *clipped* components looks like either (a) a tape-level bug
specific to the log-rate path on a "clipped + transport-limited" steady
state, or (b) a steady-state-convergence issue where the FD perturbed
solves don't fully relax to the new steady-state under the warm-start
convergence criterion.

I do not have a clean diagnosis here.  Possibilities not yet ruled out:

1. **FD picks up unconverged transient.**  At V=+0.30, c_H2O2 should adjust
   to absorb a k0_2 perturbation while keeping r_2 ≈ const (transport-
   limited).  Steady-state argument: dr_2/dlog_k0_2 → 0, so dcd/dlog_k0_2
   → 0.  The adjoint reports ~0; FD reports -1.7e-5.  If the FD warm-start
   relaxation is incomplete, FD would see the partial response (k0_2 doubles
   → r_2 doubles before c_H2O2 catches up) and report a larger derivative.
   This would mean **the adjoint is correct and the FD-based FIM result is
   biased**.

2. **Pyadjoint tape has a bug at log-rate clipped state.**  fd.ln(k0_j)
   composed with min/max value operations (the eta clip) and a few annotated
   SNES iterations might not be back-propagated correctly.

3. **Annotation pattern issue.**  v18's pattern is 5 annotated SNES
   iterations after warm-up.  My script copies that pattern, but at V=+0.30
   the warm-start may be in a regime where the 5 annotated solves don't
   represent the implicit-function gradient correctly.

## Implication for TRF and the FIM claim

The TRF inverse uses per-observable adjoint Jacobian following the same
v18 pattern.  Two caveats apply:

- If issue 1 above is correct, then **the FIM result from
  `extended_v_to_60/` is biased upward** (the ridge-breaking is overstated
  relative to true steady-state sensitivities).  The adjoint-Jacobian TRF
  will then *not* converge to the parameters that best fit the FD-based
  residuals.

- If issue 2 or 3 is correct, the ridge-breaking is real but the TRF with
  adjoint Jacobian will converge *to the wrong gradient direction*, likely
  stalling on a non-true-minimum.

Either way, the TRF result is informative — it will tell us which of these
hypotheses is right.

## Recommendation (TBD per next handoff)

A. Re-run this adjoint check at V=+0.20 with both standard and log-rate
   forms (control: standard at V=+0.20 should pass per v18; log-rate at
   V=+0.20 should also pass if the issue is V-specific, not log-rate-
   specific).

B. Re-run at V=+0.30 with much tighter steady-state convergence
   (`ss_rel_tol = 1e-9`, `sc_consec = 10`).  If FD shrinks toward 0 with
   tighter convergence, the "FD unconverged" hypothesis is supported and
   the adjoint is correct.

C. If A and B don't resolve, switch the TRF to FD-Jacobian (slower but
   trusted) for the next pass.

## Outputs

- `adjoint_vs_fd.csv`, `adjoint_vs_fd.json` — raw per-component data.
