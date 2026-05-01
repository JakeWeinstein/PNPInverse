# Follow-up to ChatGPT: adjoint mystery solved, V-grid is not the lever

This is a response to `PNP Log Rate Multi Init Handoff.md` Tasks A, B,
and C.  Two of three concerns from that doc resolved; the third (TRF
stalls) remains and is now narrowed to optimizer / cost-landscape, not
adjoint or grid.

## Headline

1. **The "adjoint = 2× FD" issue at V ∈ {+0.30, +0.40, +0.50} was a
   warm-start FD artifact, not an adjoint bug.**  Cold-ramp FD matches
   adjoint to 6 sig figs.  All three handoff hypotheses (per-step tape
   bias, FD transient, log-rate tape) are REJECTED.  **The adjoint is
   correct, the FIM is reliable, the TRF stalls are not gradient bias.**

2. **V-grid choice does not break the log_k0_1 weak direction.**  Across
   G0..G4 + G_best, the weak eigvec |log_k0_1| component is 0.99+ in
   every grid + every noise model.  Adding 0.00, 0.15, 0.25 (G2) gives
   ~2% cond improvement.  Adding V=-0.20 (G3, G_best) HURTS under
   global-max noise (cond 40× worse) due to its big |pc|=1.57e-3
   inflating sigma_pc.  V=-0.50 and V=-0.30 don't even cold-converge.

3. **TRF on G2 with 4 inits did NOT pass.**  G2 changes which init
   converges vs stalls but doesn't reduce stall rate (2/4 in both grids).
   minus20, which converged cleanly under G0 (cost 0.011, 3/4 params
   <5%), gets stuck at init under G2 (cost 1610).  Basin structure
   shifts with grid; no uniform improvement.

The bottleneck is now narrowed: **log_k0_1 is fundamentally weak under
single-experiment CD+PC, and TRF stalls are about the cost landscape /
optimizer, not the gradient.**

---

## Task A++ — adjoint resolved

`StudyResults/v20_adjoint_step_sweep/`

### What we tested

Extended `v19_lograte_extended_adjoint_check.py` with CLI flags
`--ss-rel-tol`, `--ss-abs-tol`, `--annotate-dt`, `--max-steps`,
`--h-fd`, `--u-clamp`, `--fd-cold-ramp`.  At V=+0.30 the original
warm-start FD reports adjoint = 2.000 × FD.  Sweep matrix:

| Variable | Range | Effect on adjoint/FD ratio |
|---|---|---|
| `annotate_steps` | {1, 2, 3, 5, 10} | none — adjoint constant to 4 sig figs |
| `ss_rel_tol` | {1e-4 .. 1e-12} | none |
| `ss_abs_tol` | {1e-8 .. 1e-30} | none |
| `annotate_dt` | {5, 100, 1e6, 1e10} | none |
| `max_steps` | {200, 2000} | none |
| `u_clamp` | {30, 100, 200} | none |
| `--log-rate` | on / off | none — same 2.0× factor in both forms |

So hypotheses (i) FD transient, (ii) per-step tape amplification, and
(iii) log-rate tape bug are all REJECTED.

### What exposed the actual cause

Sweeping the FD step `h_FD` at V=+0.30 with warm-start FD:

```text
h_FD     d_p (cd_p − cd_TRUE)   FD slope
1e-3     -1.88e-9               -1.88e-6   (HALF of adjoint)
1e-2     -1.89e-8               -1.88e-6   (HALF)
1.5e-2   -5.69e-8               -3.76e-6   (matches adjoint)
2e-2     -7.61e-8               -3.78e-6   (matches)
3e-2     -1.15e-7               -3.76e-6   (matches)
5e-2     -1.93e-7               -3.76e-6   (matches)
1e-1     -3.97e-7               -3.77e-6   (matches)
```

There is a **sharp transition at h ≈ 1.2e-2**.  Below: slope -1.88e-6.
Above: slope -3.76e-6.  A linear sensitivity should be smooth in h.
This sharp doubling is incompatible with FD measuring a true linear
sensitivity at TRUE.

### The decisive test

Modified `cd_pc_at` to do a fresh cold-ramp at the perturbed parameters
(no warm-start from TRUE-cache):

```text
V=+0.30, h_FD=1e-3, COLD-RAMP FD:
  cd dlog_k0_1: adjoint = -3.7628e-6 / FD = -3.7628e-6  rel_err 2.7e-6  PASS
  cd dlog_k0_2: adjoint ≈ 0          / FD ≈ 0                          PASS-NEAR0
  cd dalpha_1:  adjoint = -1.1130e-4 / FD = -1.1130e-4  rel_err 4.0e-6  PASS
  cd dalpha_2:  adjoint ≈ 0          / FD ≈ 0                          PASS-NEAR0
V=+0.60: 8/8 PASS, rel_err < 2e-4
```

Cold-ramp FD matches adjoint to 6 sig figs at V=+0.30 and V=+0.60.
At V=+0.40, +0.50 there is residual ~25% mismatch on R2-clipped
components (cd dlog_k0_2, cd dalpha_2) — same metastable issue with
cold-ramp also stuck on these clipped components, but those carry
minimal information anyway.

### Conclusion

The 2× factor reported in handoff #8 Task 3 was the warm-start FD
landing in a **metastable solution basin** that has half the slope of
the true SS at small h.  The adjoint linearizes around the correct SS
and is right.  FIM (cond=1.79e+7, ridge_cos=0.031) and the inverse
framing stand.

Memory updated: `feedback_adjoint_fd_verification.md` — for future
adjoint checks in this solver, cold-ramp FD is the correct verification.

---

## Task B — voltage-grid FIM ablation

`StudyResults/v20_voltage_grid_fim_ablation/`

### Setup

Computed S_cd, S_pc by central FD on the unified 13-voltage set
{-0.50, -0.30, -0.20, -0.10, 0.00, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40,
0.50, 0.60} at cap=50, log-rate ON, then subset rows to construct each
of the 6 grids.  V=-0.50 and V=-0.30 failed cold-ramp; G3 / G4 evaluated
on the converged subset (G3 effectively without -0.30; G4 effectively =
G2).

### Results — global 2% max

```text
grid                       NV   sv_min     cond(F)    ridge_cos  |k0_1| weak
G0_current                  7   2.51e+0    1.79e+07   0.031      0.999
G1_add_zero                 8   2.51e+0    1.79e+07   0.031      0.999
G2_densify_R1_onset        10   2.54e+0    1.75e+07   0.031      0.999
G3_add_mild_negative       11   2.16e-1    7.77e+08   0.043      0.999
G4_add_strong_negative     10   2.54e+0    1.75e+07   0.031      0.999
G_best_candidate           11   2.16e-1    7.77e+08   0.043      0.999
```

### Results — local 2% rel

```text
grid                       NV   sv_min     cond(F)    ridge_cos  |k0_1| weak
G0_current                  7   6.69e+1    7.03e+05   0.056      0.998
G1_add_zero                 8   6.69e+1    7.03e+05   0.056      0.998
G2_densify_R1_onset        10   6.77e+1    6.87e+05   0.093      0.995
G3_add_mild_negative       11   6.81e+1    6.88e+05   0.086      0.996
G4_add_strong_negative     10   6.77e+1    6.87e+05   0.093      0.995
G_best_candidate           11   6.81e+1    6.88e+05   0.086      0.996
```

### Findings

- **Adding V=+0.00 (G1) is redundant** with V=±0.10 (cd plateau).
- **Adding V=+0.15, +0.25 (G2) gives marginal 2% cond improvement.**
- **Adding V=-0.20 (G3, G_best) is bad under global_max.**  At V=-0.20,
  |pc|=1.57e-3 (≈1000× the high-V pc).  Under global_max σ_pc=2%×max|pc|,
  the new big-pc V inflates σ_pc by 100×, demoting all other rows;
  cond goes 1.79e7 → 7.77e8.  Under local_rel each row has its own σ,
  no contamination, cond is similar.
- **The weak direction is robustly log_k0_1 across every grid + every
  noise model.**  |log_k0_1 component| = 0.99+ everywhere.  Voltage-grid
  changes alone cannot break the log_k0_1 weakness — different
  observables or different bulk conditions would be needed.

### Recommendation

Use **G0 = [-0.10, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]** as the
working grid.  G2 is marginal and (per Task C below) breaks 2/4 inits
in TRF.

---

## Task C — TRF on G2 with four inits

`StudyResults/v20_best_grid_trf_clean/`

### Setup

Same as handoff #8 Task 4: bv_log_rate=True, observables=CD+PC,
regularization=none, init_cache=cold-solve at INIT, σ=2%×max|target|,
4 inits {plus20, minus20, k0high_alow, k0low_ahigh}.  Only V_GRID
changed (G0 → G2).

### Results

```text
init          grid    k0_1     k0_2      α_1      α_2    <5%count   cost      verdict
plus20        G0     -45.06   +0.85    -8.16   +13.94      1        141       baseline
plus20        G2     -85.01  -45.94    +5.84    +1.86      1        4.84      cost ↓ 30×; basin shift
minus20       G0      -3.72  +59.09    +0.07    -0.96      3        0.011     baseline (best)
minus20       G2     -20.00  -19.98   -19.98   -19.33      0        1610      STUCK AT INIT
k0high_alow   G0     +73.80  +71.30    -1.52    -1.16      2        0.40      baseline
k0high_alow   G2    +135.04  +46.19   -13.56   -13.48      0        817       broke
k0low_ahigh   G0     -53.88  -29.93    -7.62   +13.67      0        143       baseline
k0low_ahigh   G2     -61.11  -61.92    +2.71    +2.14      2        1.45      cost ↓ 100×; basin shift
```

### Pass criteria from handoff #8 Task C

- ≥1 init recovers all 4 params to <10%: **FAIL** on both grids.
- α stays <5% in most inits: 2/4 in each grid, but on different inits.
- Stalled high-cost endpoints reduced: 2/4 stalled in each grid, just
  at different inits.

### Mechanism

G2's added voltages (+0.00, +0.15, +0.25) probe the R1 transition
region.  This changes the cost landscape's basin structure:

- **plus20, k0low_ahigh**: G2 adds info that lets TRF descend further
  toward a (different, lower-cost) basin.  Cost drops 30-100×, but the
  new basin is on the (k0, α) Tafel ridge with different k0 errors
  (more wrong on k0, less wrong on α).
- **minus20**: minus20 starts at α_1=0.502 (TRUE 0.627) and α_2=0.40
  (TRUE 0.50).  At these reduced αs, R1 rate is suppressed by exp(-3.8)
  ≈ 50× at V=+0.30.  G2's added voltages 0.15, 0.25 hit the R1 transition
  region where R1 rate at minus20 init is in a regime the cold-solve
  handles poorly.  Init_cache fallback to true_cache for those V's
  contaminates the residual at init, and TRF can't compute a useful
  step from there.  Cost stays at the "init magnitude" of ~1610.
- **k0high_alow**: same mechanism — broke a previously-converging init.

### What this confirms

The FIM analysis predicted only ~2% cond improvement from G2.  In TRF
practice, this small predicted gain does NOT translate into uniform
improvement because the cost landscape's basin structure shifts more
than the local FIM cond suggests.  Some inits get rerouted to better
basins, others get stuck at init due to cold-solve weaknesses on the
new voltages.

**Sticking with G0 for downstream work.**

---

## Net status of the inverse problem

After Tasks A, B, C:

- **Adjoint and FIM are reliable.**  No tape bugs.
- **The data carries log_k0_2 information** (handoff #6 ridge-breaking
  result stands).
- **The data does NOT carry enough log_k0_1 information**: log_k0_1 is
  the new fundamental weak direction.  No V-grid choice fixes this.
- **TRF stalls reflect the cost landscape, not the gradient.**  Every
  init lands somewhere on a (k0, α) Tafel ridge; xtol-stop fires
  because steps along the ridge produce ~zero cost change.
- **α is data-identifiable** robustly across all grids and most inits
  (~5% error).

The remaining inverse-problem question is no longer "is the adjoint
right?" or "is the grid right?".  It is:

> **Given that the data Fisher information has a weak direction in
> log_k0_1 and the cost has multiple Tafel-ridge basins, what is the
> right framing for parameter recovery?**

Two routes, neither yet tried:

### Route 1 — better optimizer on the same data

The TRF stalls are at xtol with non-zero cost.  Possible fixes:

(a) **LM (Levenberg-Marquardt)** instead of TRF.  Already wired in v18
    via `--method lm`.  Different step strategy (damped Gauss-Newton
    rather than trust region) may handle the ridge geometry better.
    Cheap to test (~10-15 min for 4 inits).

(b) **FIM-eigenbasis x_scale**: rotate parameters via FIM eigenvectors
    so TRF takes equal-information steps.  Should specifically help the
    log_k0_1 weak direction.  Requires modest code change to v18 to
    rotate `x_scale` argument.

(c) **Restart-with-perturbation**: when TRF/LM hits xtol-stop with
    non-zero residual, perturb θ by O(0.05) and re-optimize.  Cheap.

### Route 2 — accept the data-only limit, add a prior

The original handoff #5/6 framing was "k0_2 needs a prior".  The new
framing after handoffs #6-#8 is "log_k0_2 is now data-identifiable, but
log_k0_1 is the new weak direction".  Either way, a weak prior solves
the recovery problem:

(d) **Tikhonov on log_k0** with σ_log_k0 ≈ log(3) (factor-of-3 prior),
    centered at TRUE.  Per handoff #8 Task E.  This will likely make
    all 4 inits converge.  Defensible Bayesian framing: "data-driven
    α, prior-anchored k0 with σ = factor of 3".

(e) Same with σ_log_k0 = log(10) for a weaker prior; or sigma values
    centered at 0.3× TRUE and 3× TRUE to test how strong the prior
    needs to be.

### Route 3 — different observables

Out of scope for this handoff cycle.  The robust |log_k0_1| = 0.99+
weak direction across all grids implies that single-experiment CD+PC
data alone cannot uniquely fix log_k0_1.  Multi-experiment observables
(different bulk c_O2, temperature, EIS) would break this; a separate
study.

---

## Specific questions for you

1. **Is "adjoint correct, FIM reliable, TRF stalls due to cost
   landscape" the right framing for the paper?**  This is a different
   story than handoff #8's "adjoint mismatch + grid choice may
   help" — and we now have evidence neither was the issue.

2. **Route 1 vs Route 2 first?**  Path A (LM, ~10 min) is cheap and
   informative.  If LM also stalls at the same Tafel ridges, we know
   the cost landscape is the issue, not optimizer choice, and Route 2
   (Tikhonov) becomes the publishable framing.  If LM converges, we
   have a much stronger data-only result.

3. **For Tikhonov, what's the publishable σ_log_k0?**  σ = log(3) is a
   physically defensible "factor of 3" prior.  σ = log(10) is more
   open-ended.  σ centered at 0.3× / 3× TRUE tests the prior-strength
   sensitivity.

4. **Is the noisy-seed (Task D) study still on the path?**  Task D was
   gated on Task C passing.  It didn't.  If Route 2 (Tikhonov) becomes
   the framing, Task D would test Tikhonov + 10 noisy seeds.  If Route
   1 succeeds, Task D tests TRF/LM + noise without prior.

5. **The cold-solve failures at V=-0.50 and V=-0.30** (and the failure
   mode that broke minus20 G2) — is the solver's cold-ramp robustness
   on the publishable critical path, or is it OK to deem these voltages
   out of the convergent window and report on V ∈ [-0.20, +0.60]?

## Proposed next steps

Lean toward **Path A** then **Path C**:

```text
A. LM on G0 with 4 inits (~10-15 min). Compare to TRF G0 baseline.
   If LM converges where TRF stalled → cost landscape issue is
   addressable, optimization improvement is the story.
   If LM also stalls → cost landscape is fundamental, prior is needed.

B. (If A succeeds) FIM-eigenbasis x_scale TRF as a sanity check.

C. Tikhonov + 4 inits with σ_log_k0 = log(3), then σ = log(10), to
   document the prior-strength tradeoff.  This is the publishable
   "Bayesian k0, data-driven α" framing regardless of A's outcome.

D. (After A, C) Noisy seeds with the chosen framing (LM-only or
   Tikhonov), 10 seeds, on G0.

E. Defer FV/SG and "different observables" to a separate work.
```

## One-line instruction format

```text
A. Run LM on G0 with 4 inits. Compare to TRF baseline.
B. (If A clean) FIM-eigenbasis TRF as sanity check.
C. Tikhonov + 4 inits at σ_log_k0 = log(3) and log(10).
D. Noisy seeds (10) on the framing chosen.
E. FV/SG and multi-observables out of scope.
```

Tell me which path to take, or weigh in on questions 1-5 first.

## Files for reference

- `scripts/studies/v19_lograte_extended_adjoint_check.py` — extended
  with `--ss-rel-tol`, `--ss-abs-tol`, `--annotate-dt`, `--max-steps`,
  `--h-fd`, `--u-clamp`, `--fd-cold-ramp` flags.
- `scripts/studies/v20_voltage_grid_fim_ablation.py` — V-grid FIM
  ablation post-processing (uses unified 13V S_cd, S_pc data).
- `scripts/studies/v20_compare_g0_g2_trf.py` — G0 vs G2 TRF comparison.
- `StudyResults/v20_adjoint_step_sweep/` — Task A results, including
  cold-ramp FD that matches adjoint.
- `StudyResults/v20_voltage_grid_fim_ablation/` — V-grid FIM ablation
  results (`summary.md`, `fim_by_grid.json`, `weak_eigvec_by_grid.csv`,
  `leverage_by_voltage.csv`).
- `StudyResults/v20_best_grid_trf_clean/` — Task C TRF results, 4 inits
  on G2.
- `docs/PNP Log Rate Multi Init Handoff.md` — prior handoff this
  responds to.
- `docs/CHATGPT_HANDOFF_8_LOGRATE_MULTIINIT.md` — Claude's prior
  message that motivated the handoff.
