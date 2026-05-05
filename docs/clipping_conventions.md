# Clipping Conventions in the Production BV Solver

**Last updated:** 2026-05-03 (corrected η-clip role and CD/PC observable-
fidelity asymmetry; see `docs/clip_observable_investigation.md`)

> ## Operational rule (read this even if you read nothing else)
>
> - **`exponent_clip = 100` is the only setting where PC is
>   trustworthy.** This is the production default since 2026-05-04.
> - **`exponent_clip = 50` produces fictitious PC** at V_RHE < −0.1 V
>   — sign-flipped and 3–4 OOM off the true value. CD is approximately
>   correct under both clips, but PC is not. See §5.2 of
>   `docs/clip_observable_investigation.md` for the per-voltage data,
>   and the "Is the solver non-physical below +0.495 V?" section
>   below for the mechanism.
> - **Do not compare clip=50 PC against experimental data**, and
>   **do not run inverse fits against PC at clip=50** (the v18–v24
>   inverse runs predate this discovery and used clip=50; their PC-
>   based conclusions need re-running at clip=100 before being
>   trusted). Use clip=100 even when it costs cold-start convergence
>   on a few voltages — recover with C+D warm-walk or Stern, not by
>   lowering the clip.

A common misreading: "log-rate removed exponent clipping." It did
not. Log-rate is an algebraic rearrangement that lets the BV residual
avoid going through `c_surf = exp(clamp(u, ±30))` — it does not
eliminate `exp`, and it does not eliminate the η-clip. Two clips
remain active in the production codepath; only a third, separate
clamp was eliminated.

## The three clips/clamps

### 1. η-clip at ±50 — STILL ACTIVE in log-rate

`forms_logc.py:_build_eta_clipped` (lines 217-236):

```python
eta_scaled = bv_exp_scale * (phi_applied − E_eq)   # = (V − E_eq) / V_T
if conv_cfg["clip_exponent"]:
    return min_value(max_value(eta_scaled, −50), +50)
```

The clip is on `η_scaled` *before* the `α·n_e` multiplication. The
log-rate branch (`forms_logc.py:330`) and the legacy branch
(`forms_logc.py:360`) both consume the same clipped η. Log-rate never
sees an unclipped value.

**Why it stays — corrected understanding (2026-05-03):** The
"would-otherwise-overflow" framing is incomplete in the log-rate world.
On the production V grid V_RHE ∈ [-0.5, +0.6] V, `α·n_e·|η|` reaches
at most ~89 without the clip; `exp(89) ≈ 10^39` is well inside IEEE 754
double range (max ~10^308). Strict FP overflow only happens at V_RHE
more cathodic than ~-3 V, far outside production. So in practice the
clip is not preventing `exp` from returning Inf; it is doing three
other things, in order of importance:

1. **Newton stability during non-SS iterates.** During the cold ramp,
   `u_cat` has not yet depleted. With `u_cat ≈ ln(c_bulk) ≈ -9` and an
   unclipped `−α·n·η ≈ +88`, the log-rate sum is
   `log_cath = ln(k0) + u_cat − α·n·η ≈ +69`, so `exp(+69) ≈ 10^30`. The
   BV Jacobian entries scale with the rate (`dR/du_cat = R`,
   `dR/dφ = -α·n_e/V_T·R`), giving condition numbers ~10^33. Even
   MUMPS direct solve loses precision and the `l2` line search
   (`maxlambda=0.3`) either backs off to no progress or steps into a
   region where one species' `log_cath` does overflow.
2. **Cold-start basin of attraction.** Empirically (see clip-threshold
   sensitivity test, 2026-05-03): clip=50 gives 10/13 cold-converged
   voltages on the production grid; clip=100 gives only 3/13
   cold-converged (rest rescued by warm-walk continuation). The clip
   is doing real solver-robustness work, not just FP-guarding.
3. **Strict overflow protection** kicks in at V_RHE more cathodic
   than ~-3 V. This matters for any future use of the solver outside
   production range, and as a deterministic ceiling regardless of
   solver state.

At converged SS in log-rate, `u_cat` depletes to ~`-α·n·|η|` and the
sum cancels: `log_cath ≈ ln(k0)` (small), so the clip is effectively
inactive at SS. The clip bites during Newton transients, not at the
fixed point. **Caveat:** the *converged SS itself* depends on whether
the clip was active during the iteration only through the basin
selected — at clip=50 vs clip=100 the converged SS *is* different (PC
and internal c_H2O2 differ; see "Is the solver non-physical below
+0.495 V?" below).

**What "R2 unclips at V_RHE > +0.495 V" means**:

- R2 has `E_eq_2 = 1.78 V`, `n_e = 2`.
- Clip becomes inactive when `|V − E_eq| < 50·V_T = 1.285 V`.
- For R2 cathodic side: `V_unclip = 1.78 − 1.285 = +0.495 V`.
- Below +0.495 V, R2's BV cathodic exponent is frozen at
  `α_2·n_e·50` — independent of both V and α_2.
- Above +0.495 V, real values flow through and α_2 sensitivity
  recovers (v19 audit measured `dcd/dα_2` jumping ~340× across this
  threshold).

**Common arithmetic error** in older handoffs (e.g.
`docs/PNP Anodic Solver Handoff.md`): applying the clip to
`α·n_e·η_scaled` instead of `η_scaled` alone gives `V_unclip ≈ +1.14 V`.
That's wrong — the clip in the code is on η_scaled before the α·n_e
factor. The correct number is **+0.495 V**.

### 2. `_U_CLAMP=30` on bulk u_i — STILL ACTIVE

`forms_logc.py:194-198`:

```python
_U_CLAMP = float(conv_cfg.get("u_clamp", 30.0))
ci = [exp(min_value(max_value(ui[i], −_U_CLAMP), _U_CLAMP)) for i in range(n)]
```

Reconstructs `c_i = exp(clamp(u_i, ±30))` for use as a coefficient
inside the bulk PDE residual on `dx`: time derivative, diffusion, EM
drift. At the default ±30, c_i is bounded to `[9.4e-14, 1.07e+13]`.

**What log-rate changed about this**: nothing in the bulk PDE. This
clamp is still applied to the c_i used inside `dx` integrals. What
log-rate did change is the BV boundary residual on
`ds(electrode_marker)` — see #3. For the structural reason log-rate
*cannot* eliminate this bulk clamp, see "Why log-rate cannot
eliminate the bulk `_U_CLAMP`" below.

**Operational note**: the default `u_clamp=30` is inactive at SS in
V_RHE ∈ [−0.5, +0.6] V. **Widen to `u_clamp=100` for V_RHE > +0.30 V**
— at higher voltages R2 consumes H₂O₂ faster than R1 produces it,
the SS `u_H2O2` approaches −30, and the clamp starts binding at SS
(distorting the bulk PDE coefficient, not just guarding transient
Newton iterates).

### 3. The c_surf clamp inside BV — REMOVED by log-rate

Pre-log-rate (still present in the legacy branch at
`forms_logc.py:359-381`):

```python
cathodic = k0 * c_surf[cat_idx] * exp(−α · n_e · η)
# where c_surf[i] = exp(clamp(u_i, ±30))
```

When V > +0.30 V drives c_H2O2 below `exp(−30) ≈ 9.4e-14` during
Newton iteration, `c_surf` gets pinned at the floor. Combined with the
saturated `exp(α_2·n_e·50)` from clip #1 above, floor × huge produces
a spurious R2 sink that overwhelms physical H2O2 production.

Log-rate (`forms_logc.py:325-357`):

```python
log_cathodic = ln(k0) + ui[cat_idx] + Σ power*(ui[sp] − ln c_ref)
              − α · n_e · η
cathodic = exp(log_cathodic)
```

`ui[cat_idx]` is used directly — unclamped — and the only `exp` is
the final one applied to the assembled log-rate. There is no
`c_surf` in the BV residual. The artificial R2 sink at high V is
gone.

**This is what log-rate actually fixed.** A clamp on the BV
boundary residual specifically, not "exponent clipping" in general.

## Summary table

| Clip | Where applied | Active in log-rate? | What it guards |
|---|---|---|---|
| η-clip ±50 | `_build_eta_clipped` | **YES** | Newton stability + cold-start basin (NOT strict overflow on production V); selects an SS basin that diverges from physical for PC and internal c_H2O2 below threshold |
| `_U_CLAMP=30` on bulk u_i | bulk PDE residual on `dx` | **YES** | `exp(u)` in bulk integrals |
| c_surf clamp inside BV | legacy BV branch only | **NO** (algebra-eliminated) | Was: `exp(u)` underflow corrupting BV at high V |

## Why log-rate cannot eliminate the bulk `_U_CLAMP`

A natural question: log-rate removed the analogous clamp from the BV
residual; why does the same trick not apply to the bulk?

**BV term — one scalar rate per quadrature point:**

```
k · c · exp(−α·n·η)  =  exp( ln k + u − α·n·η )
```

The whole rate collapses to a single `exp(scalar_sum)`. You can defer
the only `exp` to the end after assembling everything in log-space.
That is what log-rate does.

**Bulk PDE — `exp(u)` is a coefficient:**

```
∫ D · exp(u_i) · (∇u_i + z_i·∇φ) · ∇v · dx        (Nernst–Planck)
∫ (exp(u_i) − exp(u_old)) / dt · v · dx           (time derivative)
∫ z_i · exp(u_i) · w · dx                          (Poisson source)
```

`exp(u_i)` is a coefficient multiplying gradient/test-function
expressions that vary across the domain. There is no single `exp` to
defer — `c_i = exp(u_i)` *is* the coefficient. No algebraic
rearrangement absorbs it into one `exp(scalar_sum)`.

So the bulk clamp's role is purely floating-point safety on `exp(u_i)`
during Newton iteration: stiff residuals or bad initial guesses can
produce wild iterates, and `exp(u)` past ~700 hits Inf and breaks
assembly.

### When to widen `u_clamp`

At default 30, `exp(±30) ∈ [9.4e−14, 1.07e+13]`. At converged SS in
V_RHE ∈ [−0.5, +0.6] V the SS u_i stays well inside ±30 and the clamp
is inactive (no physical effect on the observable — only fires during
transient Newton iterates as overflow safety).

Widen to `u_clamp=100` (`exp(±100) ≈ 2.7e±43`, still inside double
precision) when:

- **V_RHE > +0.30 V** — R2 consumes H₂O₂ faster than R1 produces it,
  so the SS `u_H2O2` itself approaches −30 and the clamp starts
  binding *at SS*, distorting the bulk PDE coefficient.
- The V grid is extended into more extreme regimes where any `u_i`
  may exceed ±30 at convergence.

### Could it be removed entirely?

Only via one of:

- (a) a smooth saturation that is exact at SS but bounded
  asymptotically — changes physics smoothly even at SS, undesirable.
- (b) a globalization / line-search strong enough to provably keep
  Newton iterates bounded under stiff residuals — hard to guarantee.
- (c) a bulk reformulation that doesn't have `exp(u)` as a
  coefficient — a much larger change than log-rate.

The clamp is the cheap correct answer; widen the knob if `u`
approaches it at SS.

## "Is the solver non-physical below +0.495 V?"

**Short answer (corrected 2026-05-03):**

- **CD (total current density):** approximately physical (~0.2% max
  distortion at the deepest production V).
- **PC (peroxide current):** **qualitatively wrong** at V_RHE < -0.1 V
  — sign flip plus 3–4 OOM magnitude shift vs unclipped.
- **Internal c_H2O2 surface concentration:** wrong by ~10⁹–10¹², with
  the qualitative behavior inverted (clip=50 says H2O2 *accumulates*
  at the electrode; unclipped truth says it *depletes*).
- **α_2 gradient identifiability:** zero in the clipped regime,
  because the exponent is frozen w.r.t. α_2.

The previous version of this doc claimed "forward CD/PC predictions
are usable" below threshold. That's correct for **CD**, **wrong for
PC**. The correction is from the clip-threshold sensitivity test
(2026-05-03; data in `StudyResults/clip_threshold_sensitivity/`,
investigation log in `docs/clip_observable_investigation.md`):

| V_RHE | ΔCD/CD (50→100) | PC@50 | PC@100 (true) | c_H2O2_surf @50 | c_H2O2_surf @100 |
|---|---|---|---|---|---|
| -0.500 | -0.17% | -1.81e-1 | +1.54e-5 | 1.17 (10⁴× above bulk) | 3.5e-13 (10⁹× below bulk) |
| -0.300 | -0.06% | -6.50e-2 | +1.54e-5 | 4.20e-1 | 4.9e-14 |
| -0.100 | -1e-6 | +2.91e-6 | +1.54e-5 | 8.06e-5 | 7.0e-15 |
| ≥-0.100 | <1e-6 | matches @100 | (matches @50) | converging | converging |

**Long answer.** In the deeply clipped regime (V well below R2's
E_eq = 1.78 V), R2 *should* be in the kinetics-saturated /
mass-transport-limited regime: c_H2O2 at the surface depletes to
balance R2's cathodic flux against the diffusive supply of H2O2 from
the bulk. The product `k0_2 · c_surf · exp(α·n·|η|)` should match the
diffusion supply.

With the clip, this rebalancing is *incomplete*. R2's clipped rate is
~10⁴× slower than R2's true unclipped rate (clip=50 freezes
`α·n·|η|` at 50, while the true value at V=-0.5 is 88.7 → ratio
`exp(88.7-50) = exp(38.7) ≈ 6e16` … but the converged SS doesn't
reach that ratio because c_surf can't reach `exp(-88)` and instead
overshoots to ~1.17). So:

- **R2 effective rate at clip=50 is roughly i_lim_O2** (matching
  diffusion supply of H2O2 from R1's local production).
- **R2 effective rate at clip=100 is much higher** — it consumes
  every H2O2 molecule R1 produces *plus* every H2O2 molecule diffusing
  in from the seed bulk.

CD ≈ R1 + (small R2 contribution) is set by O2 diffusion to the
electrode in either case → CD preserved to ~0.2%.

PC ≈ (R1 production - R2 consumption) at the electrode = boundary
flux of H2O2:

- At clip=50: c_H2O2 accumulates to 1.17, R2 consumption is throttled
  by the clipped exp, the *net* H2O2 boundary flux is dominated by
  outflow toward bulk → PC large negative (cathodic-signed by
  production convention) ≈ -0.18 mA/cm².
- At clip=100 (true physics): c_H2O2 depletes, R2 consumes everything
  including the diffusion supply, PC = +seed_diffusion ≈ +1.5e-5
  mA/cm² (small positive — anodic-signed because the electrode is
  *receiving* a small H2O2 inflow from the seeded bulk).

These differ by 4 orders of magnitude *and have opposite sign*. Any
inverse fit that uses PC data at V_RHE < -0.1 V at clip=50 is fitting
the clip artifact, not the physical kinetics.

Where the clip *does* break things at the observable level:

- **CD**: approximately preserved (~0.2% max distortion). Inverse
  fits using CD only are not materially affected.
- **PC**: qualitatively distorted at V_RHE < -0.1 V (sign flip +
  3–4 OOM magnitude shift). Use clip=100 if peroxide data in the
  cathodic regime is essential.
- **α_2 sensitivity**: `dR_2/dα_2 ≈ 0` in the clipped regime because
  the exponent is frozen w.r.t. α_2. Inverse problems that rely on
  V < +0.495 V data to recover α_2 will fail — not because the data
  is wrong, but because the forward model is locally insensitive to
  α_2 there.
- **Internal c_H2O2 profile**: not physically meaningful in the
  clipped regime (artifact differs from true SS by ~10⁹–10¹², with
  the *sign* of the deviation inverted: clip=50 accumulates above
  bulk; truth depletes below bulk). Don't trust surface concentration
  plots as diagnostics there.

**So the prior "approximately physical" claim splits as: CD yes, PC
no, internal fields no, gradients no.**

## Implications for the inverse pipeline

- α_2 is data-identifiable only from voltages where R2 is unclipped
  (V > +0.495 V at clip=50; V > -0.79 V at clip=100; never on the
  production grid at clip=50).
- α_1 unclips at `V < 0.68 − 50·V_T = −0.605 V` (anodic side at
  +1.965 V, irrelevant). Since the production grid covers V ∈
  [−0.5, +0.6], R1 is essentially always unclipped at clip=50 — α_1
  is robustly identifiable.
- This is why CD data above +0.495 V matters for α_2 recovery, and
  why V23's grid extension to −0.5 V gave only WEAK PASS (CLAUDE.md
  rule #7) — extending below didn't unclip anything new at clip=50.
- **PC data below V_RHE = -0.1 V is unsafe for fitting at clip=50**
  (clip-threshold sensitivity, 2026-05-03). The PC observable carries
  a sign-flipped 4-OOM clip artifact in this voltage range. Two
  remedies: (a) restrict PC fitting to V ≥ -0.1 V, (b) re-run the
  forward at clip=100 — Newton basin shrinks (10/13 → 3/13
  cold-converged on production grid) but warm-walk continuation
  succeeded at every production voltage in the test run.
- **The Tafel-slope test (Tier 1 hole #1 in
  `docs/forward_solver_test_coverage.md`) should be run at clip=100**
  for any voltage below +0.495 V. At clip=50 the Tafel slope below
  threshold reflects the frozen `α_2·n_e·50` exponent, not the
  physical α_2.

### Clip-activation thresholds at different `exponent_clip`

`V_unclip_below = E_eq − exponent_clip · V_T` (cathodic side; analytic
side is +clip·V_T above E_eq and not relevant on the production grid).

| `exponent_clip` | R2 cathodic threshold (E_eq=1.78) | R1 cathodic threshold (E_eq=0.68) |
|---|---|---|
| 50 (production) | V_RHE = +0.495 V | V_RHE = -0.605 V |
| 100 | V_RHE = -0.789 V | V_RHE = -1.889 V |
| 200 | V_RHE = -3.358 V | V_RHE = -4.458 V |

So at clip=100 the entire production grid V ∈ [-0.5, +0.6] V is
*unclipped* for both reactions. clip=100 is the cleanest "high-
fidelity reference" production-V configuration; clip=200 is a fortiori
unclipped and useful as a paranoia check that we're at the
clip-independent limit.

## Where this lives in code

- `Forward/bv_solver/forms_logc.py:_build_eta_clipped` — η-clip
  construction.
- `Forward/bv_solver/forms_logc.py:194-198` — `_U_CLAMP` on bulk
  u_i.
- `Forward/bv_solver/forms_logc.py:325-357` — log-rate BV branch
  (no c_surf clamp).
- `Forward/bv_solver/forms_logc.py:359-381` — legacy BV branch (c_surf
  clamp present, deprecated for production).
- `Forward/bv_solver/config.py:_get_bv_convergence_cfg` — defaults
  for `exponent_clip=50`, `u_clamp=30`, `bv_log_rate=True`.
- `scripts/_bv_common.py:_make_bv_convergence_cfg` — the
  `make_bv_solver_params` factory plumbing; override
  `params["bv_convergence"]["exponent_clip"]` after construction (or
  use `SolverParams.with_solver_options`) to change the clip threshold
  for sensitivity studies.

## Investigation log

- `docs/clip_observable_investigation.md` — 2026-05-03
  CD-vs-PC-vs-internal-fields experiment that established the
  CD/PC asymmetry and corrected this doc. Quantitative data in
  `StudyResults/clip_threshold_sensitivity/` (clip ∈ {50, 100}).
- `scripts/verification/mms_voltage_sweep.py` — 2026-05-03 MMS sweep
  of clipped vs unclipped equations (shows the equations differ
  materially below threshold but cannot directly measure SS-observable
  difference).
- `scripts/studies/clip_threshold_sensitivity.py` — runner that
  produced the data in `StudyResults/clip_threshold_sensitivity/`.
