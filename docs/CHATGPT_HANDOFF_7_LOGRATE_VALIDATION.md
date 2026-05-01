# Follow-up to ChatGPT: log-rate validation — partial confirmation, mixed bag

## Headline

Following your `PNP Log Rate Next Steps Handoff.md` plan, ran Tasks 1–5.
Mostly good news, with one important diagnostic question and one pipeline
limitation.

**The FIM ridge-breaking survives every noise model tested.** TRF inverse
**recovers k0_2 to +0.44% from a +20% init with no priors** — direct
empirical confirmation of the breakthrough. Two unresolved issues:

1. The adjoint Jacobian shows a **2× discrepancy vs central FD at
   V_RHE ∈ {+0.30, +0.40, +0.50}** for non-zero-clipped components, agrees
   to 1e-7 at V=+0.60. Pattern is striking enough to suspect a tape-level
   bug, but I do not yet have a clean diagnosis.

2. **TRF -20% init failed entirely** — not because of information limits,
   but because the forward solver couldn't warm-start from TRUE-cache to
   the -20% offset (α_1=0.502 vs TRUE 0.627 makes R_1 200× smaller, IC
   gap too wide for Newton). This is a pipeline limitation, not an
   identifiability claim.

## Tasks completed

### Task 1 — Documented +0.495 V threshold

Added a comment in `forms_logc.py:_build_eta_clipped` explaining the
clip-on-eta_scaled convention.  Also recorded in memory.  Confirmed
`V_unclip_2 = E_eq_2 - 50·V_T = +0.495 V` as you derived.

### Task 2 — Noise-model FIM audit (PASS)

`StudyResults/v19_lograte_noise_model_fim/`

```
noise_model                    sv_min      cond(F)     ridge_cos   weak eigvec
A. global 2% max               2.510e+00   1.79e+07    0.031       k0_1
B. local 2% rel                6.690e+01   7.03e+05    0.056       k0_1
C. local 2% + floor 1e-6       2.874e+00   1.32e+06    0.020       k0_1
C. local 2% + floor 1e-7       5.218e+00   2.93e+07    0.011       k0_1
C. local 2% + floor 1e-8       5.355e+00   1.07e+08    0.011       k0_1
C. local 2% + floor 1e-9       6.060e+00   8.56e+07    0.011       k0_1
```

Even at a 1e-6 absolute floor (which buries `|cd|@V=+0.60 ≈ 1.8e-9` and
`|pc|@V=+0.60 ≈ 2.7e-9` entirely below noise), cond(F) = 1.32e+6 — five
orders of magnitude better than the V≤+0.20 baseline (2.03e+11).
ridge_cos stays ≤ 0.06 across all models.  **The ridge-breaking is
robust to realistic noise.**

### Task 3 — Adjoint vs FD at extended V (PARTIAL FAIL)

`StudyResults/v19_lograte_extended_adjoint_check/`

Used the `solve_warm_unannotated → 5 annotated SNES iterations` pattern
from `v18_logc_lsq_inverse.py:solve_warm_annotated`, which v18's adjoint
check validated at V=+0.20 in the standard form.

```
V_RHE   verdict
+0.30   FAIL  — non-zero components: adjoint = 2.000 × FD
+0.40   FAIL  — non-zero components: adjoint = 2.000 × FD
+0.50   FAIL  — non-zero components: adjoint = 1.819 × FD
+0.60   PASS  — adjoint matches FD to ~1e-7 relative
```

Sample at V=+0.30:
```
component         adjoint        FD            ratio
cd dlog_k0_1      -3.76e-6      -1.88e-6      adjoint = 2.000 × FD
cd alpha_1        -1.11e-4      -5.57e-5      adjoint = 2.000 × FD
cd dlog_k0_2      -5.27e-13     -1.73e-5      adjoint ≈ 0,  FD finite
cd alpha_2        -5.27e-11     +2.86e-8      adjoint ≈ 0,  FD finite
```

Two patterns:
- For components where R_1 is the dominant contributor (log_k0_1, α_1):
  adjoint exactly 2× FD.  Striking.
- For components where R_2 is the contributor and R_2 is fully clipped
  (log_k0_2, α_2 at V<+0.495): adjoint ≈ 0.  FD reports finite values
  consistent with the multiplicative `cd ≈ -2F·k0_2·exp(50)·c_H2O2_surf`
  structure.

I haven't pinned this down.  Possibilities:

(i) **FD picks up unconverged transient.**  At V=+0.30, R_2 is transport-
    limited.  Steady-state argument: a δlog_k0_2 should be absorbed by
    Δc_H2O2_surf so r_2 is invariant — adjoint ≈ 0 is correct.  But FD
    perturbed solves warm-start from TRUE-cache and converge under
    `ss_rel_tol=1e-4`, which may stop before c_H2O2 fully relaxes,
    capturing the partial response (cd ∝ k0_2 before relaxation).
    Implication: FIM is biased upward; ridge-breaking is overstated.

(ii) **5-iteration time-derivative amplification in adjoint.**  At each
     of the 5 annotated SNES iterations, `Up.assign(U)` is taped, so the
     time-derivative residual `(c - c_old)/dt` carries an explicit
     `c_old → U_prev_step` dependency.  If the warm-start isn't exactly
     at steady-state, each iteration accumulates a small bias in the
     adjoint chain.  5 iterations × ~14% per-step bias ≈ 2× over.
     This would predict scaling with N_annotate; we haven't tested.

(iii) **Pyadjoint tape bug in log-rate path.**  fd.ln(k0_j) on a Function
      composed with `min_value(max_value(eta_scaled, ±50))` and an
      annotated Newton solve might not back-propagate correctly.

The empirical TRF result (below) is more consistent with (i) being
*partially* true (bias is real but small enough that TRF can still
descend) or (ii) (gradient *direction* is right even if magnitude is
inflated).

### Task 4 — Clean-data TRF, multi-init (MIXED)

`StudyResults/v19_lograte_extended_trf_clean/`

| init | k0_1 err | k0_2 err | α_1 err | α_2 err | status |
|---|---:|---:|---:|---:|---|
| **+20%** | -45.31% | **+0.44%** | -8.17% | +13.82% | xtol stop, cost: 933 → 142 (6.6×) |
| **-20%** | -20% | -20% | -20% | -20% | pipeline failure (forward non-convergence) |

The **+20% TRF is the headline result**: k0_2 recovered to under 1% from
a +20% offset, with no priors.  Even the partial result at α_2 (+13.8%)
beats the prior baseline by far (where no movement off init occurred at
all).

The bad news:
- k0_1 went from +20% → -45% (slid down the new ridge per FIM
  prediction).
- α_1 stuck at -8% (close to your <10% pass criterion but not under).
- TRF terminated at xtol — it could not take a further step, despite
  cost not being at zero.  This is the classic "ridge-stuck" signature.

The -20% init failure is purely a pipeline issue.  Forward warm-start
from TRUE-cache to α_1=0.502 fails on ≥4/7 voltages.  Cost stayed at
the "huge residual fallback" of 7.0e+06 throughout, no descent direction
visible to TRF.  Line profile from -20% recovered → TRUE shows
J=NaN for t ∈ [0, 0.6] (entire warm-start fails) and only solves
correctly past t=0.7.  The IC strategy doesn't bridge from TRUE-cache
to -20%-offset.

### Task 5 — _U_CLAMP sweep (PASS)

`StudyResults/v19_lograte_uclamp_sweep/`

```
u_clamp     sv_min       cond(F)     ridge_cos
30          2.510e+00    1.79e+07    0.031
60          2.510e+00    1.79e+07    0.031
100         2.510e+00    1.79e+07    0.031
```

Identical to all printed digits.  Bulk-PDE clamp doesn't contaminate the
FIM.  We can close the "_U_CLAMP residual asymmetry" caveat from
`CHATGPT_HANDOFF_6`.

## Specific questions for you

1. **Is +20% TRF k0_2 recovery to <1% sufficient as the publishable
   headline?**  Without -20% confirmation it's a one-init result.  The
   FIM-says-information-is-there + TRF-converges-from-one-direction is a
   reasonable story but not bulletproof.

2. **The 2× adjoint discrepancy — your call on which hypothesis to
   investigate first.**  My priorities:
   (a) Re-run adjoint check at V=+0.20 with both standard form AND
       log-rate.  Does log-rate alone introduce the 2× factor at a
       known-good voltage?
   (b) Vary `annotate_final_steps` ∈ {1, 2, 3, 5, 10} at V=+0.30 to test
       hypothesis (ii).  If the discrepancy scales with N, it's
       time-derivative-driven.
   (c) Tighten ss_rel_tol from 1e-4 to 1e-9 for FD only at V=+0.30.  If
       FD shrinks toward 0, hypothesis (i) is correct.

3. **Should I invest in cold-start TRF to fix -20% init?**  It's ~10×
   slower per eval.  Per init that's ~50 min instead of ~5 min.  Worth it?

4. **Is the per-V σ vs global-max σ choice strategic?**  The current TRF
   uses global-max which heavily down-weights high-V (small-signal) rows
   where k0_2/α_2 information lives.  Per-V σ would balance better but
   probably make TRF more noise-sensitive.  What's the right convention
   for the publishable result?

5. **The k0_1 -45% drift is the "ridge_cos=0.031 but FIM weak direction
   = log_k0_1" prediction in action.**  TRF can navigate the
   well-conditioned directions but stalls on the now-weak log_k0_1.
   This is honest physics — k0_1 IS poorly determined when the V_GRID
   covers the R1 transport-limited regime.  In a publication we'd say
   "k0_1 from low-V Tafel + k0_2 from high-V data" maybe.  Is that the
   right framing, or do we want a single-experiment recovery of all 4?

## What I'd propose

Path A (fast, what we have): write up "+20% recovery + FIM noise
robustness" as the result.  k0_2 to <1% from +20% is genuinely
significant; the rest are caveats.

Path B (medium): debug the 2× adjoint by running diagnostic (a) and (b)
above (~30 min total).  If hypothesis (ii) confirmed, document and
move on.  If (i), the FIM result needs an asterisk.

Path C (slow, rigorous): implement cold-start TRF, run all 4 inits,
verify k0_2 recovery is robust to init direction.  This is what GPT
asked for in the original plan.  ~3-4 hours wall time.

Lean toward B+A: pin down the adjoint discrepancy and write up.  Multi-
init confirmation can wait for noisy seeds (Task 6) where we'd have to
bootstrap pipeline anyway.

Tell me which path, or whether you want to weigh in on any of the 5
questions before deciding.

## Files for reference

- `Forward/bv_solver/{config.py, forms_logc.py}` — log-rate, u_clamp config.
- `scripts/_bv_common.py` — `_make_bv_convergence_cfg(log_rate=...)` keyword.
- `scripts/studies/v18_logc_lsq_inverse.py` — added `--log-rate`,
  `--v-grid`, `--init`, `--out-base` flags + line profile.
- `scripts/studies/v19_bv_clip_audit.py` — `--log-rate`, `--u-clamp` flags.
- `scripts/studies/v19_bv_cap_continuation.py` — Stage 3.1 diagnostic.
- `scripts/studies/v19_lograte_noise_model_fim.py` — Task 2.
- `scripts/studies/v19_lograte_extended_adjoint_check.py` — Task 3
  (now with `--v-test`, `--log-rate`, `--annotate-steps` flags).
- `StudyResults/v19_lograte_noise_model_fim/{summary.md, fim_by_noise_model.{csv,json}}`.
- `StudyResults/v19_lograte_extended_adjoint_check/{summary.md, adjoint_vs_fd.{csv,json}}`.
- `StudyResults/v19_lograte_extended_trf_clean/{summary.md, lsq_trf_*_initplus20_lograte/, ..._initminus20_lograte/}`.
- `StudyResults/v19_lograte_uclamp_sweep/{summary.md, clamp_30/, clamp_60/, clamp_100/}`.
- `StudyResults/v19_bv_lograte_audit/{summary.md, equiv_check/, extended_v_cap50/, extended_v_to_60/, extended_v_NO_lograte/}` (from prior round).
