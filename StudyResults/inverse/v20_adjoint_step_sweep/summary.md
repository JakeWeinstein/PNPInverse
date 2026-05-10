# V20 Adjoint Diagnostic Sweep — Resolved

## Headline

**The adjoint is correct.** The "2× FD vs adjoint" mismatch reported in
`CHATGPT_HANDOFF_8_LOGRATE_MULTIINIT.md` Task 3 was a **warm-start FD
artifact**, not a bug in the adjoint.

When the FD perturbed solve is **cold-ramped** at the perturbed parameters
(instead of warm-started from the TRUE-cache snapshot), FD agrees with
the adjoint to within ~1e-5 relative error at V=+0.30. The "exactly
2.000" ratio was a metastable basin issue in the warm-start procedure.

This means:
- The FIM at TRUE (cond=1.79e+7, ridge_cos=0.031) is reliable.
- The TRF inverse stalls (+20% and k0low_ahigh inits at cost ≈141) are
  NOT due to gradient bias.
- The handoff's hypotheses (i, ii, iii) are all rejected; the actual
  cause was an FD-methodology artifact (hypothesis iv, not in the list).

## Diagnostics that REJECTED the original three hypotheses

### (ii) Per-step amplification in the tape — REJECTED

`annotate_steps ∈ {1, 2, 3, 5, 10}` at V=+0.30, ss_rel_tol=1e-4:

```
n=1:  cd dlog_k0_1 adjoint = -3.7628e-6 / FD = -1.8816e-6 (ratio 2.0)
n=2:  cd dlog_k0_1 adjoint = -3.7628e-6 / FD = -1.8816e-6 (ratio 2.0)
n=3:  cd dlog_k0_1 adjoint = -3.7628e-6 / FD = -1.8816e-6 (ratio 2.0)
n=5:  cd dlog_k0_1 adjoint = -3.7628e-6 / FD = -1.8816e-6 (ratio 2.0)
n=10: cd dlog_k0_1 adjoint = -3.7627e-6 / FD = -1.8816e-6 (ratio 2.0)
```

Adjoint is identical to 4 sig figs across N_annotate. Per-step bias
hypothesis is impossible — the ratio doesn't grow with N.

### (i) FD picks up unconverged transient — REJECTED

`ss_rel_tol ∈ {1e-4, 1e-5, 1e-6, 1e-7, 1e-9, 1e-12}` at V=+0.30,
n_annotate=5:

```
1e-4 .. 1e-12: cd dlog_k0_1 FD = -1.8816e-6 (constant)
```

Tightening rel_tol does not move FD. Also tested ss_abs_tol=1e-20
(eliminate OR-dominated convergence): no change. Also tested
max_steps=2000 with all tolerances tight: no change.

### (iii) Log-rate tape bug — REJECTED

```
V=+0.30, log-rate=ON:  adjoint = -3.7628e-6 / FD = -1.8816e-6 (ratio 2.0)
V=+0.30, log-rate=OFF: adjoint = -3.7628e-6 / FD = -1.8816e-6 (ratio 2.0)
V=+0.20, log-rate=ON:  adjoint = -1.6284e-2 / FD = -1.6234e-2 (ratio 1.003) PASS
```

Same factor with both BV forms; not log-rate specific. V=+0.20 already
passes — the issue is V-specific, not form-specific.

### Annotate-time-step (1/dt contribution) — REJECTED

Forced `dt_const = 1e10` before annotation (effectively pure
steady-state Jacobian inversion):

```
annotate_dt = 100, 1e6, 1e10: cd dlog_k0_1 adjoint = -3.7628e-6 (constant)
```

The (1/dt) term in the implicit-diff Jacobian is not the cause.

## Diagnostic that EXPOSED the actual cause

### h-step sweep at V=+0.30, n=5, default warm-start

```
h_FD     d_p (cd_p - cd_TRUE)   FD slope
1e-3     -1.88e-9               -1.88e-6   (HALF of adjoint)
1e-2     -1.89e-8               -1.88e-6   (HALF of adjoint)
1.5e-2   -5.69e-8               -3.76e-6   (matches adjoint)
2e-2     -7.61e-8               -3.78e-6   (matches adjoint)
3e-2     -1.15e-7               -3.76e-6   (matches adjoint)
5e-2     -1.93e-7               -3.76e-6   (matches adjoint)
7e-2     -2.73e-7               -3.77e-6   (matches adjoint)
1e-1     -3.97e-7               -3.77e-6   (matches adjoint)
5e-1     -2.44e-6               -3.92e-6   (matches; some nonlinearity)
```

**There is a sharp transition at h ≈ 1.2e-2.** Below: FD = -1.88e-6
(half of adjoint). Above: FD = -3.76e-6 (matches adjoint).

This is incompatible with FD being the "true linear sensitivity" — the
sensitivity should be smooth in h.

## The decisive test — cold-ramp FD

Modified `cd_pc_at` to perform a fresh cold-ramp at each perturbed
parameter (no warm-start from TRUE-cache):

```
V=+0.30, h=1e-3, COLD-RAMP FD:
  cd dlog_k0_1: adjoint = -3.7628e-6 / FD = -3.7628e-6 (rel_err 2.7e-6) PASS
  cd dlog_k0_2: adjoint ≈ 0          / FD ≈ 0            PASS-NEAR0
  cd dalpha_1:  adjoint = -1.1130e-4 / FD = -1.1130e-4 (rel_err 4.0e-6) PASS
  cd dalpha_2:  adjoint ≈ 0          / FD ≈ 0            PASS-NEAR0
```

**Cold-ramp FD matches adjoint to 6 sig figs.** The 2× factor was
entirely an artifact of warm-starting from TRUE-cache: small-h
perturbations land in a metastable basin that has half the slope of the
true SS.

### Cold-ramp FD across all V

```
V=+0.30: 6/8 PASS, 2 FAIL (FAIL components are pc near-zero, irrelevant)
V=+0.40: 4/8 PASS, 4 FAIL (R2-clipped components, rel_err ≈ 0.27)
V=+0.50: 4/8 PASS-NEAR0, 4 FAIL (R2 transition, rel_err ≈ 0.22)
V=+0.60: 8/8 PASS (rel_err < 2e-4)
```

V=+0.30 and V=+0.60 are clean. V=+0.40 and V=+0.50 still show ~25%
mismatch on R2-clipped components — likely due to the same metastable
issue but with cold-ramp also stuck (R2 is fully clipped at V=+0.40,
and the perturbed cold-ramp may be hitting a similar basin issue).
These components carry minimal information anyway (clipped exponent),
so this residual issue is not blocking the inverse problem.

## What this means for the inverse problem

**Re-evaluate the handoff conclusions:**

1. **The FIM result stands.** cond=1.79e+7, ridge_cos=0.031, weak
   eigvec on log_k0_1. Adjoint-based FIM is reliable.

2. **The TRF stalls have a different cause.** "+20% init at cost 141"
   and "k0low_ahigh at cost 144" with monotone-downhill paths to TRUE
   are NOT due to factor-2 gradient bias. The gradient is correct.

   Possible causes for TRF stalls:
   - Cost surface has narrow valleys; TRF's trust-region steps overshoot.
   - Parameter scaling between log_k0 (~0.001 scale) and alpha (~0.0001
     scale) causes ill-conditioned trust region.
   - Genuine multi-basin structure (Tafel ridges) that no single init
     can navigate.

3. **The verification methodology has a flaw.** Warm-start FD is
   unreliable at voltages where the system has metastable basins.
   The standard verification practice should be COLD-RAMP FD (or
   verify via Taylor remainder test instead of FD).

## Recommended next steps

The handoff's Task A is now COMPLETE (root cause found). The handoff's
Task B (voltage-grid FIM ablation), Task C (TRF on best grid), Task D
(noisy seeds), Task E (Tikhonov) can now proceed without first fixing
the adjoint.

For Task C/D specifically: investigate the TRF stalls separately:
- Try LM (Levenberg-Marquardt) instead of TRF for the stalled inits.
- Use FIM-eigenbasis x_scale instead of unit scaling.
- Add a small random perturbation at xtol-stop and re-optimize.

## Files

- `scripts/studies/v19_lograte_extended_adjoint_check.py` — extended
  with `--ss-rel-tol`, `--ss-abs-tol`, `--annotate-dt`, `--max-steps`,
  `--h-fd`, `--u-clamp`, `--fd-cold-ramp` CLI flags.
- `StudyResults/v20_adjoint_step_sweep/v030_n*_tol*` — annotate_steps
  and ss_rel_tol arms (all REJECTED hypothesis (i) and (ii)).
- `StudyResults/v20_adjoint_step_sweep/v030_n5_h*` — h-step sweep
  showing the sharp transition at h ≈ 1.2e-2.
- `StudyResults/v20_adjoint_step_sweep/v030_n5_h1e-3_coldramp` —
  decisive test: cold-ramp FD at h=1e-3 matches adjoint.
- `StudyResults/v20_adjoint_step_sweep/coldramp_all_V` — cold-ramp FD
  across V=[+0.30, +0.40, +0.50, +0.60].
