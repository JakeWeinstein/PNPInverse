# Follow-up to ChatGPT: log-rate BV breaks the ridge — Stage 4 may not be needed

## Headline finding

**Your Stage 2 fix alone broke the (log_k0_2, α_2) ridge.** With log-rate
BV evaluation enabled and V_GRID extended to include +0.30 → +0.60 V, the
FIM at cap=50 goes from `cond=2.03e+11, ridge_cos=1.000, sv_min=2.35e-2`
(your handoff #5 baseline) to `cond=1.79e+7, ridge_cos=0.031, sv_min=2.51`.

That's **sv_min × 107, cond ÷ 11,000, ridge_cos rotated to ~0**, with the
weak eigenvector now log_k0_1-dominant rather than log_k0_2-dominant.

**Stage 4 (FV/SG prototype) appears unnecessary** for the inverse-problem
question. The k0_2/α_2 sensitivity that the FIM screen #5 showed was
fundamentally unreachable is reachable in CG1+log-c+Boltzmann after all —
two changes were enough: (1) log-rate BV evaluation per your Stage 2, and
(2) extending V_GRID past R1's kinetic-transport transition (+0.30) and
into R2's natural unclipping range (V > +0.495).

## What was actually run

Per your `PNP Anodic Solver Handoff.md` Stage 1 → Stage 2 → Stage 3.1 plan.

### Stage 1 — BV clip audit (`StudyResults/v19_bv_clip_audit/`)

Cap sweep [50, 60, 70, 80, 100, None] with original (non-log-rate) forward
model. **Only cap=50 converged** — caps ≥ 60 all failed at z=0.000 in ~0.2s
(Newton can't take the first step from cold IC because R2's BV residual is
~10²² magnitude). `summary.md` in that directory has full diagnostics. Per
your acceptance test, this fired the "solver fails badly without the cap →
proceed to FV/SG" branch. We didn't proceed there directly — we did Stage 2
first.

### Stage 2 — log-rate BV (`Forward/bv_solver/{config.py, forms_logc.py}`)

Added `bv_log_rate` config option. When True, the multi-reaction rate
construction in `build_forms_logc` replaces

```python
cathodic = k0_j * c_surf[cat_idx] * exp(-α_j * n_e_j * eta_j)
cathodic *= (c_surf[sp_idx] / c_ref)**power
```

with

```python
log_cathodic = (
    ln(k0_j) + u[cat_idx]                          # u, not c_surf
    + power * (u[sp_idx] - ln(c_ref))
    - α_j * n_e_j * eta_j
)
cathodic = exp(log_cathodic)
```

The crucial change is using `u[i]` (unclamped) directly in the exp argument
instead of going through `c_surf[i] = exp(clamp(u[i], ±30))`. Mathematically
identical at the converged solution; verified to 5 sig figs at every voltage
in `extended_v_cap50/` against the original-form baseline.

### Stage 3.1 — cap continuation (`StudyResults/v19_bv_cap_continuation_lograte/`)

With log-rate ON and warm-starting from cap=50, the cap chain
50→51→52→55→60→70→80→100→None reaches `cap=None` at every tested voltage
(-0.10, +0.20, +0.40), each step in <1 s. So the v19 audit's "cap=60 cold-fails"
was an IC-quality issue, not a structural limit — but as it turns out, this
matters less than expected, because R2 unclips naturally at cap=50 once V is
high enough (see §3 below).

## FIM progression

```
                                                        sv_min     cond(F)    ridge_cos
HANDOFF_5 baseline                  V≤+0.20, no logc   2.348e-02  2.03e+11    1.000
log-rate equiv check                V≤+0.20, log-rate  2.348e-02  2.03e+11    1.000   ← identical
extended V, no log-rate             V≤+0.40, no logc   1.461e-01  5.22e+9     0.023   ← V=+0.40 cold-fails
extended V, log-rate ON             V≤+0.40, log-rate  7.034e-01  2.25e+8     0.009
extended++, log-rate ON             V≤+0.60, log-rate  2.510e+00  1.79e+7     0.031
```

Weak eigvec at extended++: `[log_k0_1: +1.00, log_k0_2: +0.03, alpha_1: -0.02,
alpha_2: -0.00]`. Pure log_k0_1 — orthogonal to your canonical (log_k0_2, α_2)
ridge direction.

## Mechanism (two-layer story)

### Layer 1 — V_GRID extension breaks the ridge even without log-rate

Adding V=+0.30 alone (no log-rate, control run `extended_v_NO_lograte/`)
already drops cond(F) by 39× and rotates ridge_cos to 0.023. Why?

At V_RHE = +0.30, η_1 = -0.38 V — R1's BV cathodic exponent (`-α_1·n_e·η_1/V_T
≈ +18.6`, well below clip 50) has dropped 8 OOM in rate compared to V=-0.10.
R1 is now in the kinetic-transport transition region tail. Empirically:

```
V=-0.10:  r1=4.86e-1, r2=4.86e-1   (both clipped, R1 transport-limited cathodic)
V=+0.20:  r1=2.14e-1, r2=2.14e-1   (R1 in Tafel region)
V=+0.30:  r1=1.03e-5, r2=9.45e-5   (R1 nearly out; R2 dominates cd)
V=+0.40:  r1=3.24e-11, r2=8.42e-5  (R1 gone; cd ≈ -2F·r2)
```

**Once R1 contribution to cd is small, cd ≈ -2F·k0_2·exp(50)·c_H2O2_surf·(c_H/c_ref)²
becomes ~linear in k0_2.** dcd/dlog_k0_2 jumps from ~1e-10 (V≤+0.20) to ~1.7e-5
(V=+0.30). That's the new information that breaks log_k0_2's weak direction —
even though R2's BV exponential is still clipped.

### Layer 2 — log-rate enables V > +0.30 in the first place

Without log-rate, V=+0.40 cold-fails (run `extended_v_NO_lograte/`). The
mechanism is the `_U_CLAMP=30` clamp on `u`: c_surf in BV uses
`exp(clamp(u, ±30))`, so when Newton needs c_H2O2 below `exp(-30) ≈ 9.4e-14`
during iteration, the clamp pins it. Combined with the saturated `exp(50)`
in R2's BV exponent, the floor times the huge exp gives a spurious R2 sink
that nothing else can balance — Newton stalls.

Log-rate evaluates `exp(ln k0 + u_H2O2 + 2(u_H - ln c_ref) - α·n_e·η)`. The
`u_H2O2` enters additively, so it can be arbitrarily negative and the whole
expression decays smoothly to zero. No floor, no phantom sink.

### Layer 3 — R2 actually unclips at +0.495 V, not +1.14 V

Your handoff predicted R2 unclips at V_RHE > +1.14 V. I think there's an
arithmetic error there. The clip in the code is on `eta_scaled = (V_RHE -
E_eq)/V_T` at ±50, so unclipping requires `|eta_2| < 50·V_T = 1.285 V`,
which gives V_RHE > E_eq,2 - 1.285 = **+0.495 V**.

(Your write-up went `|eta_2| < 0.64 V → V_RHE > 1.14 V`. 0.64 V = 25·V_T,
so it looks like the formula `|eta| < 50/(α·n_e)·V_T` was applied as
`|eta| < 50/n_e·V_T = 0.64 V`, missing the α_2 factor. Worth a sanity
check on your end.)

So at V=+0.50 and +0.60 (which we now reach), R2 is in its kinetic regime,
and dcd/dα_2 jumps from 1.41e-6 (clipped) to 4.73e-4 — **340× larger**.
Direct R2 voltage-shape sensitivity, in the original CG1+log-c framework.

## Open questions for you

1. **Confirm the +0.495 V threshold.** My calculation: BV exponent in R2
   cathodic is `α_2·n_e·|eta_clipped|`, where `eta_clipped = clamp(eta_scaled,
   ±50)` and `eta_scaled = (V - E_eq)/V_T`. The exponent saturates at
   `α_2·n_e·50 = 50` (for α_2=0.5, n_e=2). It unsaturates when `|eta_scaled|
   < 50` ⟺ `|eta_2| < 50·V_T = 1.285 V` ⟺ `V_RHE > E_eq,2 - 1.285 = +0.495 V`.
   Right?

2. **Is there a hidden gotcha at V > +0.495 V?** Empirically the solver runs
   cleanly at +0.50 and +0.60. r2 at +0.60 is 2.4e-9 (very small — R2 nearly
   equilibrated). cd at +0.60 is 1.8e-9 — almost certainly below any realistic
   experimental noise floor. Are these voltages actually informative if the
   absolute signal is sub-noise, or is the FIM result an artifact of
   uniform-σ whitening?

3. **σ_noise convention.** I'm using `σ = 2% × max(|target|)` per
   observable type — same convention as `v18_logc_multiexperiment_fim.py`.
   If real measurement noise is local (`σ(V) = 2% × |target(V)|`), the
   high-V rows of the whitened FIM get heavily down-weighted (σ at V=+0.60
   is 1e3× smaller than at V=-0.10), and sv_min would shrink. Should I
   redo the diagnostic with per-V σ to get a realistic picture? The
   ridge-breaking is qualitative either way (the new directions exist),
   but the quantitative cond(F) might rebound a few OOM.

4. **Should we run TRF inverse before declaring victory?** The FIM says the
   information is there. The actual test is whether TRF + adjoint Jacobian
   recovers k0_2 and α_2 to <10% from a +20% init at clean data, no priors.
   The pipeline from `v18_logc_lsq_inverse.py` should drop in unchanged
   except for `bv_log_rate=True` and the extended V_GRID. Worth doing
   before the next handoff?

5. **Stage 4 (FV/SG) status.** I'm marking it as "no longer urgent" but
   not "abandoned" — the EDL positivity issue is a real numerical-methods
   contribution and FV/SG would still be the cleanest publishable solver
   for problems where direct R2 unclipping is required (e.g., even higher
   anodic biases, or different equilibrium potentials). Is that the right
   triage?

## Caveats I'm aware of

1. `_U_CLAMP=30` is **still active in the bulk PDE residual** (time
   derivative, diffusion). Log-rate only removed it from the BV path.
   At V > +0.40 with cap > 70, c_H2O2_min hits the clamp value
   (`exp(-30) ≈ 9.4e-14`). The bulk PDE may be slightly inconsistent
   with the BV term in that regime, although our tests didn't show
   visible artifacts. Pending follow-up: widen `_U_CLAMP` to 100.

2. **No catastrophic numerical guard yet.** Your handoff suggested
   `if |log_rate| > 120: mark unreliable`. In our tests log_cathodic stays
   under ~75. Would matter if we push V_RHE toward E_eq,2 = 1.78 V where
   `|log_cathodic|` could exceed 120.

3. **Single-design FIM only.** Cleaning up the prior multi-experiment FIM
   (your handoff #4 plan) is now lower priority since single-experiment
   already breaks the ridge — but it might further improve cond(F) and is
   a cheap follow-up.

## Files for reference

- `Forward/bv_solver/config.py` — added `bv_log_rate` flag.
- `Forward/bv_solver/forms_logc.py` — added log-rate BV branch.
- `scripts/studies/v19_bv_clip_audit.py` — Stage 1 + 2 audit (with `--log-rate`,
  `--v-grid`, `--caps` flags).
- `scripts/studies/v19_bv_cap_continuation.py` — Stage 3.1 cap continuation.
- `StudyResults/v19_bv_clip_audit/{summary.md, run.log, *.json, *.csv}` —
  Stage 1 results (caps≥60 cold-fail at all V).
- `StudyResults/v19_bv_lograte_audit/`
    - `summary.md` — full Stage 2/3.1 writeup.
    - `equiv_check/` — log-rate vs standard equivalence verification at
      baseline V_GRID, cap=50.
    - `extended_v_cap50/` — log-rate, V_GRID up to +0.40, cap=50.
    - `extended_v_to_60/` — log-rate, V_GRID up to +0.60, cap=50 (the
      headline FIM result).
    - `extended_v_NO_lograte/` — control: V_GRID up to +0.40 *without* log-rate
      (V=+0.40 cold-fails; FIM only on 5 voltages).
- `StudyResults/v19_bv_cap_continuation_lograte/` — cap continuation diagnostic.

## What I propose for the next move

A. **Confirm** the +0.495 V threshold and the σ-whitening question (your
   answer to Q1 and Q3 here) — small, fast.

B. **Run TRF inverse** at `bv_log_rate=True`, V_GRID = [-0.10, 0.10, 0.20,
   0.30, 0.40, 0.50, 0.60], 4 free params, +20% init, clean data first then
   2% noise (10 seeds). Pipeline = `v18_logc_lsq_inverse.py` with the new
   knobs. Pass criterion: k0_2 and α_2 within 20% of TRUE without priors.
   If yes, this is the publishable inverse-problem result.

C. **Widen `_U_CLAMP`** to 100 in `forms_logc.py` for cleanliness — likely
   no-op for the FIM but removes a known asymmetry.

D. Update CLAUDE-side memory and the handoff archive; reframe the project
   storyline from "k0_2 needs a prior" to "k0_2 and α_2 are data-identifiable
   in CG1+log-c+Boltzmann with log-rate BV at V_GRID up to ~+0.60 V".

If A confirms my arithmetic and B succeeds, the publishable contribution
shifts to:

> *A two-line numerical fix to the Butler-Volmer evaluation in a
> log-concentration PNP solver — replacing `k0·c·exp(-αnη)` with
> `exp(ln k0 + u - αnη)` — extends the convergent voltage window past R2's
> kinetic-regime onset, making the second-electron-transfer kinetics
> data-identifiable from a single-experiment I-V curve. The "k0_2 is
> non-identifiable from accessible-voltage data" conclusion of prior work
> was an artifact of a clamped surface concentration in the BV residual,
> not a fundamental Fisher-information limit.*

That's both a methods contribution and an electrochemistry inversion result.
Tell me if this framing holds up against what you'd consider rigorous, and
whether there's anything in the FIM picture I'm interpreting wrong before
we commit to it.
