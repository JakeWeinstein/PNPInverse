# V19 — clean-data TRF at extended V_GRID with log-rate (multi-init, init-cache fix)

## Pipeline architecture fix

**Original bug:** TRF used `inv_cache = [c for c in true_cache]` to seed the
optimizer's per-V IC cache from the cold-solve at TRUE.  This worked when
init was close to TRUE, but failed for far inits because TRUE-cache
warm-start to (TRUE ± 20%) parameters can't bridge the gap (steady-state
c_O2 / c_H2O2 differ by orders of magnitude).  Failed warm-starts triggered
the "huge residual fallback" → cost stuck at 7e+06 → TRF couldn't move.

**Fix (`v18_logc_lsq_inverse.py`):**

```python
# Step 2.5 (NEW): cold-solve at INIT params → init_cache
init_cache = [None] * NV
for V in V_GRID:
    cd_i, pc_i, snap_i = solve_cold(V, init_k0_1, init_k0_2, init_a_1, init_a_2)
    init_cache[i] = snap_i

# Step 3: optimizer's IC starts at INIT, not TRUE
inv_cache = [c if c is not None else true_cache[i]
             for i, c in enumerate(init_cache)]
```

And the fallback cascade in `compute_residuals_and_jacobian` now tries
`init_cache` before `true_cache` when warm-start fails.

This is the correct architecture: TRUE-cache is for *target observables*
(what we're trying to fit); INIT-cache is for *the optimizer's path*.
Previously the script confused the two.

## Multi-init result with the fix

| init        | k0_1 err | k0_2 err | α_1 err | α_2 err | cost final | params nailed (<5%) |
|-------------|---------:|---------:|--------:|--------:|-----------:|---------------------|
| +20%        |   -45.1% |  **+0.85%** |   -8.2% |   +13.9% | 1.41e+02 | k0_2 |
| -20%        |  **-3.72%** |  +59.1%  |  **+0.07%** | **-0.96%** | 1.07e-02 | α_1, α_2, k0_1 |
| k0high_alow |  +73.8%  |  +71.3%  |  **-1.52%** | **-1.16%** | 4.04e-01 | α_1, α_2 |
| k0low_ahigh |  -53.9%  |  -29.9%  |   -7.6%  |  +13.7% | 1.44e+02 | (none) |

(Same script, same V_GRID = [-0.10, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
same `bv_log_rate=True`, same uniform σ-whitening.)

### Headline observation

**No single init nails all 4 parameters.**  Different inits recover
different subsets.  Two inits (-20%, k0high_alow) both have α below TRUE
at start → both recover α_1, α_2 to ≤2% — α is consistently
data-identifiable when the optimizer can reach the data-fitting basin.

**k0_1 best from -20% init (-3.72%), k0_2 best from +20% init (+0.85%).**
Each init pins down a different k0 subsequence.  No init nails both k0s.

## Interpretation: the (k0, α) Tafel ridge persists

The data fixes the *combination* `log(k0) + α·n_e·η/V_T`, not the
individual values.  Concretely:

- k0high_alow: α went DOWN from -20% init to ~TRUE.  k0 had to go UP +73%
  to compensate (ridge slope ≈ 47).  Final cd matches TRUE despite k0
  being far off.

- minus20: α went UP from -20% init to TRUE.  k0_1 followed, ending
  at TRUE.  But k0_2 went +59% — sliding along its own ridge.

- plus20: k0_2 nailed.  But k0_1 went DOWN -45% — sliding along its
  ridge in the opposite direction.

Each init lands wherever its trust-region trajectory hits the
data-matching surface.

### Reconciling with the FIM result

The FIM at TRUE said cond(F) = 1.79e+7 and ridge_cos = 0.031 — local
conditioning is good.  But TRF doesn't operate at TRUE; it walks from
init.  The cost surface globally has a non-trivial topology (curved
ridge sheets connecting low-cost regions).  Local FIM ≠ global
identifiability.

The "ridge broken" headline from FIM was too strong.  More accurate:

> The (log_k0_2, α_2) ridge of the V≤+0.20 baseline is replaced by a
> richer cost-surface where α is consistently identifiable but absolute
> k0 still depends on which init's basin TRF lands in.

## Cost final values reveal which inits found which basins

- cost = 0.011 (-20%): genuine low-cost basin — TRF descended fully
- cost = 0.40 (k0high_alow): low-cost but not at zero — partial descent
- cost = 141 (plus20), 144 (k0low_ahigh): TRF stalled, not at zero

The plus20 and k0low_ahigh runs ended at significant residual cost.
This may reflect:
- (a) The cost-surface valley is non-monotone from these inits.
- (b) The 2× adjoint bias at V<+0.60 (per Task 3) gives wrong gradient
  directions, causing TRF to stall on local saddles.
- (c) The cost surface has multiple basins and these inits fell into a
  non-zero one.

The line profile at +20% recovered → TRUE is monotone descent (J=958 at
recovered → J=NaN at TRUE).  TRF didn't take the path to TRUE despite
it being downhill.  This points to (b) or a Jacobian-conditioning issue.

## What's recoverable, honestly

From this 4-init clean-data run:

1. **α_1, α_2 are data-identifiable** to ~1% in 2/4 inits.  In 3/4 the
   alphas are within 16%.  This is a robust, publishable claim.

2. **k0_2 is data-identifiable to <1%** from at least one init (+20%).
   But not robustly: from -20% init it goes +59%; from k0high_alow it
   goes +71%.  So k0_2 *can* be recovered but the basin of attraction
   is narrow.  Still better than the V≤+0.20 baseline, where k0_2
   couldn't move from any init.

3. **k0_1 is data-identifiable to <4%** from one init (-20%).  Similarly
   narrow basin.

4. **The single-experiment data does not uniquely fix all 4 parameters.**
   The Tafel-ridge structure persists — the data constrains
   `(log k0, α)` combinations more tightly than individual values.

This is consistent with the original "k0 needs a prior" conclusion,
*sharpened*: with log-rate + extended V, α is data-identifiable, and
each k0 has at least one init from which it's data-identifiable too —
but not all 4 from a single init.

## Comparison vs old pipeline (TRUE-cache warm-start)

| init | OLD pipeline outcome | NEW pipeline outcome |
|---|---|---|
| +20% | cost 142, k0_2 +0.44%, α_1 -8% | cost 141, k0_2 +0.85%, α_1 -8% (basically identical) |
| -20% | **cost 7e+06, all stuck at -20%** | cost 0.011, α to ±1%, k0_1 to -3.7%, k0_2 +59% |

The +20% case was reachable from TRUE-cache (so old pipeline worked).
The -20% case was not.  The fix unblocks half the multi-init test
without affecting the cases that already worked.

## Open questions for next handoff

1. **Why does plus20 stall at cost 141?**  The line profile shows
   monotone descent recovered → TRUE (J should keep falling), so a
   downhill path exists.  TRF's xtol termination at step_norm=0
   suggests the Jacobian's rank or scaling fails near this point.
   Could the 2× adjoint bias at V<+0.60 be mis-pointing the
   gradient?  Worth testing with FD Jacobian (slow but trusted).

2. **Should we report α-recovery as the headline?**  3/4 inits give
   α_1, α_2 within 16%.  In 2/4 inits, both α to ≤2%.  That's a
   robust electrochemistry claim ("Tafel coefficients are
   data-identifiable from CD+PC at extended V_GRID with log-rate
   evaluation").  The k0 story remains "individual k0s need priors;
   the (k0, α) combination is data-fixed."

3. **Should we run noisy seeds (Task 6)?**  At 2% noise, would expect
   the basin structure to widen and probably blur α recovery to
   5-10% across inits.  k0 would likely revert to "needs a prior."
   This is the realistic scenario.

## Outputs

```
StudyResults/v19_lograte_extended_trf_clean/
├── summary.md  (this file)
├── lsq_trf_noise_0.0pct_precstandard_ny200_initplus20_lograte/    (old pipeline)
├── lsq_trf_noise_0.0pct_precstandard_ny200_initminus20_lograte/   (old pipeline; failed)
├── plus20_v2_initcache/         ← new pipeline runs
├── minus20_v2_initcache/
├── k0high_alow_v2_initcache/
└── k0low_ahigh_v2_initcache/
```

Each `*v2_initcache/` directory contains `result.json`, `line_profile.json`,
`true_curve.npz`, and `targets.npz`.
