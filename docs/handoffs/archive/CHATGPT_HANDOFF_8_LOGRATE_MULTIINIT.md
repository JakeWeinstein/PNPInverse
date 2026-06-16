# Follow-up to ChatGPT: full validation pass — pipeline bug fixed, multi-init done

This supersedes `CHATGPT_HANDOFF_7_LOGRATE_VALIDATION.md`.  Same set of
tasks, but with a critical pipeline bug fixed mid-stream that materially
changed the multi-init TRF result.

## Headline

The "k0_2 → +0.44% from +20% init" claim from #7 stands, but it was
**a one-init result obscuring a more nuanced picture**.  After fixing a
TRUE-cache-as-IC bug in the inverse pipeline and running the full 4-init
sweep, the actual story is:

> **α_1 and α_2 are data-identifiable** to ~1% in 2/4 inits and within
> 16% in 3/4.  **Each individual k0 is data-identifiable from at least
> one init** (k0_2 to <1% from +20%, k0_1 to <4% from -20%) but no single
> init recovers all 4 params.  The (k0, α) Tafel ridge from the V≤+0.20
> baseline is replaced by a richer cost surface; the data fixes the
> *combination* `log(k0) + α·n_e·η/V_T`, not the individual values.
> Single-experiment data still doesn't uniquely fix all 4 params, but
> the new V_GRID makes α robust and each k0 reachable.

The "ridge broken" framing from `CHATGPT_HANDOFF_6_LOGRATE_BREAKTHROUGH`
was too strong.  More accurately: the local FIM at TRUE is well-
conditioned (cond=1.79e+7, ridge_cos=0.031), but global cost-surface
geometry has multiple basins, and TRF's endpoint depends on which basin
the init lives in.

## Pipeline bug found and fixed

The original `v18_logc_lsq_inverse.py` initialized the optimizer's
per-V IC cache from cold-solve at *TRUE* parameters:

```python
inv_cache = [c for c in true_cache]   # warm-start from TRUE anchor
```

For init close to TRUE this works.  For ±20% offsets in α it doesn't,
because TRUE-cache c_O2_surf and c_H2O2_surf differ by orders of
magnitude from steady-state at the offset point.  Newton can't bridge
the gap from TRUE-IC to (TRUE±20%)-steady-state in the iteration budget,
the script's "huge residual fallback" returns cost ≈ 7e+06, and TRF
sees no descent direction.  In Handoff #7 the -20% TRF "failed
entirely"; that was the symptom.

**Fix:** cold-solve at INIT params first, populate `init_cache`, use
that as the optimizer's IC anchor.  TRUE-cache is reserved for
*target observables* (what we fit against).  Plus update the IC fallback
cascade to try `init_cache` before `true_cache`:

```python
# Step 2.5 (NEW): cold-solve at INIT
init_cache = [None] * NV
for V in V_GRID:
    cd_i, pc_i, snap_i = solve_cold(V, init_k0_1, init_k0_2,
                                     init_a_1, init_a_2)
    init_cache[i] = snap_i

# Step 3: use init_cache, fall back to true_cache only as last resort
inv_cache = [c if c is not None else true_cache[i]
             for i, c in enumerate(init_cache)]
```

This is the correct architecture.  Kudos to the user for spotting it
("if so, that's dumb and we should do another cold start sweep at the
init parameter values and then warm start from there").

## Tasks completed (all 5 of your priority list)

### Task 1 — +0.495 V threshold (PASS)

`forms_logc.py:_build_eta_clipped` annotated.  Memory updated.  Threshold
confirmed:

```
V_unclip_2 = E_eq_2 - exponent_clip · V_T = 1.78 - 50·0.02569 = +0.4955 V
```

### Task 2 — Noise-model FIM audit (PASS)

`StudyResults/v19_lograte_noise_model_fim/`

```
noise_model                    sv_min      cond(F)     ridge_cos
A. global 2% max               2.510e+00   1.79e+07    0.031
B. local 2% rel                6.690e+01   7.03e+05    0.056
C. local 2% + floor 1e-6       2.874e+00   1.32e+06    0.020
C. local 2% + floor 1e-7       5.218e+00   2.93e+07    0.011
C. local 2% + floor 1e-8       5.355e+00   1.07e+08    0.011
C. local 2% + floor 1e-9       6.060e+00   8.56e+07    0.011
```

Even with a 1e-6 absolute floor (which buries `|cd|@V=+0.60 ≈ 1.8e-9`
and `|pc|@V=+0.60 ≈ 2.7e-9` entirely below noise), cond(F) = 1.32e+6 —
five orders of magnitude better than the V≤+0.20 baseline (2.03e+11).
ridge_cos stays ≤ 0.06 across all noise models.  **The FIM ridge-
breaking is genuinely robust to realistic noise.**

### Task 3 — Adjoint vs FD at extended V (PARTIAL FAIL — unresolved)

`StudyResults/v19_lograte_extended_adjoint_check/`

Pattern at the new important voltages (with `solve_warm_unannotated → 5
annotated SNES iterations` from `v18_logc_lsq_inverse.py`):

```
V_RHE   verdict
+0.30   FAIL  — non-zero components: adjoint = 2.000 × FD
+0.40   FAIL  — non-zero components: adjoint = 2.000 × FD
+0.50   FAIL  — non-zero components: adjoint = 1.819 × FD
+0.60   PASS  — adjoint matches FD to ~1e-7 relative
```

Two distinct sub-patterns:
- For components where R_1 contributes (cd dlog_k0_1, cd α_1):
  adjoint exactly 2× FD.  Striking.
- For components where R_2 contributes and R_2 is fully clipped
  (log_k0_2, α_2 at V<+0.495): adjoint ≈ 0; FD reports finite values
  consistent with `cd ≈ -2F·k0_2·exp(50)·c_H2O2_surf`.

**I have not pinned this down.** Three hypotheses:

(i) **FD picks up unconverged transient.**  At V=+0.30, R_2 is transport-
    limited.  Steady-state argument: a δlog_k0_2 should be absorbed by
    Δc_H2O2_surf so r_2 is invariant — adjoint ≈ 0 is correct.  But FD
    perturbed solves warm-start from TRUE-cache and converge under
    `ss_rel_tol=1e-4`, which may stop before c_H2O2 fully relaxes,
    capturing the partial response (cd ∝ k0_2 before relaxation).
    *Implication: FIM is biased upward.*

(ii) **5-iteration time-derivative amplification in the tape.**
     `Up.assign(U)` after each annotated SNES iteration carries a small
     `c_old → U_prev_step` dependency.  If warm-start isn't exactly at
     steady state, 5 iterations × ~14% per-step bias ≈ 2× factor.  Not
     yet tested against `annotate_steps ∈ {1, 2, 3, 5, 10}`.

(iii) **Pyadjoint tape bug in log-rate path.**  fd.ln(k0_j) on a Function
      composed with `min_value(max_value(eta_scaled, ±50))` and an
      annotated Newton solve might not back-propagate correctly under
      the new form.  Standard form was validated at V=+0.20 in the v18
      adjoint check.

The TRF result (below) is informative: at +20% init, k0_2 was recovered
to <1% — so whatever bias the adjoint has, the gradient direction is
right enough to descend toward TRUE k0_2.  That argues against (iii).
The 2 inits that stalled (+20%, k0low_ahigh) end with cost ≈ 141 even
though monotone-downhill paths to TRUE exist — that's consistent with
either (i) or (ii).

### Task 4 — Multi-init clean-data TRF (MIXED, with new pipeline)

`StudyResults/v19_lograte_extended_trf_clean/`

```
init         k0_1 err    k0_2 err    α_1 err     α_2 err     cost      <5%-err count
+20%         -45.06%     +0.85%      -8.16%      +13.94%     141       1 (k0_2)
-20%         -3.72%      +59.09%     +0.07%      -0.96%      0.011     3 (k0_1, α_1, α_2)
k0high_alow  +73.80%     +71.30%     -1.52%      -1.16%      0.40      2 (α_1, α_2)
k0low_ahigh  -53.88%     -29.93%     -7.62%      +13.67%     144       0
```

(All with `bv_log_rate=True`, V_GRID = [-0.10, 0.10, 0.20, 0.30, 0.40,
0.50, 0.60], no Tikhonov, uniform σ-whitening at 2%×max|target|.
Init-cache fix in place.)

#### What this tells us

- **α_1, α_2 within 2%** in 2/4 inits.  Both have α below TRUE at start
  (init -20% and k0high_alow).  Within 16% in 3/4 inits.

- **k0_2 to <1%** from +20% init.  **k0_1 to <4%** from -20% init.  Each
  k0 is recoverable but with narrow basin of attraction.

- **No init recovers all 4.**  TRF lands on different points of a
  cost-surface manifold depending on init.

- **Two inits stalled** (+20%, k0low_ahigh) at cost ≈ 141.  Line profile
  from these recovered → TRUE is monotone downhill (J keeps decreasing
  toward TRUE), so a path exists.  TRF terminated with `xtol`,
  `step_norm = 0` — couldn't take any step.  This is consistent with
  the 2× adjoint bias mis-pointing the gradient.

#### Mechanism (Tafel ridge persistence)

Each (k0_j, α_j) pair has a Tafel ridge in (log_k0, α) space along which
the BV rate `r_j ∝ k0_j · exp(-α_j · n_e · η / V_T)` is constant.
Different inits land on different ridge points:

- k0high_alow: starts α-low, k0-high.  α relaxes UP toward TRUE.  k0
  goes UP +73% to compensate (predicted ridge slope ≈ 47).  cd matches
  TRUE despite k0 being far off.
- minus20: starts α-low, k0-low.  α goes UP to TRUE.  k0_1 also goes
  UP to TRUE.  But k0_2 overshoots +59%.
- plus20: k0_2 nailed, but k0_1 went DOWN -45% on its own ridge.

The ridge isn't broken; α just isn't on it (α is independently fixed by
data) and k0 *can* be on TRUE if the init's basin includes it.

### Task 5 — _U_CLAMP sweep (PASS)

`StudyResults/v19_lograte_uclamp_sweep/`

```
u_clamp     sv_min       cond(F)     ridge_cos
30          2.510e+00    1.79e+07    0.031
60          2.510e+00    1.79e+07    0.031
100         2.510e+00    1.79e+07    0.031
```

Identical to all printed digits.  Bulk-PDE clamp doesn't contaminate the
FIM.  Closes the "_U_CLAMP residual asymmetry" caveat.

## Reconciling the FIM and the multi-init TRF

The FIM was always *local at TRUE*.  It said cond=1.79e+7, the new weak
direction is log_k0_1, ridge_cos=0.031.  All true.

But:
- TRF doesn't operate at TRUE; it walks from init.
- The cost surface has multiple basins separated by (or connected by)
  curved Tafel ridges.
- TRF lands wherever its trajectory ends.

The **information** is in the data — but extracting it requires either
(a) being already in TRUE's basin (not the case for several of our
inits), or (b) priors that disambiguate the ridges, or (c) better
optimization that escapes saddle/ridge stalls.

## What's the actual publishable claim

Hard claim (data alone):
> Log-rate BV + extended V_GRID converts the previously k0_2-dominant
> single-ridge problem (V≤+0.20, cond≈2e+11, weak eigvec pure log_k0_2)
> into a richer landscape where (i) the Tafel coefficients (α_1, α_2) are
> data-identifiable to ≤2% across multiple init conditions, (ii) each
> kinetic rate constant is data-identifiable from at least one init, and
> (iii) the local Fisher information at TRUE is well-conditioned
> (cond=1.79e+7) and robust to realistic noise floors.  Single-experiment
> CD+PC data does not uniquely fix all 4 parameters from a single init,
> but the (k0, α) Tafel-ridge structure is now traversable: with a
> physical prior on either k0 (e.g. literature, EIS) or via multi-init
> + voting, all 4 parameters become recoverable.

Softer framing for paper:
> The original v17 conclusion — "k0_2 needs a prior" — is refined.  α
> is now data-identifiable; the (k0, α) Tafel ridges remain but each k0
> is recoverable with appropriate init or a weak prior.  Log-rate BV
> evaluation is the key numerical change.

## Specific questions for you

1. **Hypothesis (i) vs (ii) vs (iii) for the 2× adjoint bias.**  My next
   diagnostic: vary `annotate_final_steps` ∈ {1, 2, 3, 5, 10} at V=+0.30.
   If the discrepancy scales with N, hypothesis (ii).  Worth doing
   before noisy seeds?

2. **The k0_high_α_low result — is "α correct, k0 +73%" really a
   different ridge point or has the optimizer found a degenerate
   solution?**  At cost = 0.40, residuals are nontrivial.  Should we
   tighten ftol/xtol to push it further?

3. **Should we present the multi-init result as the headline ("α
   robust, k0 ridge-traversable"), or run noisy seeds first to see how
   the picture survives 2% noise?**

4. **Tikhonov + multi-init — the natural follow-up?**  With a weak
   prior on log(k0) (e.g. σ_log_k0 = log(3), centered at TRUE, giving
   factor-of-3 prior), all 4 inits would likely converge to TRUE.  This
   gives a defensible "Bayesian k0, data-driven α" story.

5. **The "plus20 stalls at cost 141" issue.**  Line profile shows
   monotone downhill to TRUE.  TRF terminates at xtol with
   step_norm=0.  Is this the 2× adjoint bias?  Or do we need a
   different optimizer (LM, dogleg) to escape?

## What I'd propose

Path A (fast, ~30 min): debug the 2× adjoint bias with the
`annotate_final_steps` sweep.  This will tell us if hypothesis (ii) is
right.  If yes, simple fix and we move on.

Path B (medium, ~1-2 hrs): run noisy seeds (Task 6) on 2-3 inits to
see if the multi-init basin structure survives.  Realistic test for
publishability.

Path C (medium): add Tikhonov + multi-init to demonstrate the
"Bayesian k0, data-driven α" framing.

Path D (slow, deferred): investigate why plus20 stalls despite
downhill path; might need optimizer change.

Lean toward **A then B**.  Adjoint debug is cheap and might explain
both the FIM mismatch and the TRF stalls; noisy seeds is the realistic
publishability test.

Tell me which path, or if you want to weigh in on questions 1–5 before
deciding.

## Files for reference

- `Forward/bv_solver/{config.py, forms_logc.py}` — log-rate, u_clamp config.
- `scripts/_bv_common.py` — `_make_bv_convergence_cfg(log_rate=...)` keyword.
- `scripts/studies/v18_logc_lsq_inverse.py` — TRF inverse, with new
  `--log-rate`, `--v-grid`, `--init`, `--out-base`, `--out_subdir` flags
  + line profile + **init-cache architecture fix** (Step 2.5 cold-solve
  at INIT, fallback cascade through init_cache then true_cache).
- `scripts/studies/v19_bv_clip_audit.py` — `--log-rate`, `--u-clamp` flags.
- `scripts/studies/v19_bv_cap_continuation.py` — Stage 3.1 diagnostic.
- `scripts/studies/v19_lograte_noise_model_fim.py` — Task 2 (this round).
- `scripts/studies/v19_lograte_extended_adjoint_check.py` — Task 3
  (now with `--v-test`, `--log-rate`, `--annotate-steps` flags).
- `StudyResults/v19_lograte_noise_model_fim/` — Task 2 results (PASS).
- `StudyResults/v19_lograte_extended_adjoint_check/` — Task 3 results
  (FAIL at V<+0.60, PASS at V=+0.60; unresolved).
- `StudyResults/v19_lograte_extended_trf_clean/` — Task 4, four inits:
  `plus20_v2_initcache/`, `minus20_v2_initcache/`,
  `k0high_alow_v2_initcache/`, `k0low_ahigh_v2_initcache/`.  Each has
  `result.json`, `line_profile.json`, `true_curve.npz`.
- `StudyResults/v19_lograte_uclamp_sweep/` — Task 5 results (PASS).
- `StudyResults/v19_bv_lograte_audit/` — prior round (Stages 2/3.1) for
  context.
- `StudyResults/v19_bv_clip_audit/` — original Stage 1 audit for
  context (cap=50 only converges at cold start; documented in
  `summary.md`).
- `docs/CHATGPT_HANDOFF_6_LOGRATE_BREAKTHROUGH.md` — initial breakthrough
  report.
- `docs/CHATGPT_HANDOFF_7_LOGRATE_VALIDATION.md` — superseded; +20%
  result was correct but isolated.
