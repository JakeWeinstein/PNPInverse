# Clip Effect on Observable Physics — Investigation Log

**Date:** 2026-05-03
**Question:** In the voltage range where the η-clip applies (V_RHE < +0.495 V
for R2), does the clip distort the steady-state observable (CD, PC) compared
to the true unclipped physical system?

> ## TL;DR / operational rule
>
> **CD: yes, approximately physical at clip=50. PC: no, fictitious at
> clip=50.** Below V_RHE = −0.1 V, PC at clip=50 is sign-flipped and
> 3–4 OOM off the true (clip=100) value. The full per-voltage table
> is in §5.2.
>
> **Use `exponent_clip = 100` for any PC observable comparison or
> inverse fit.** v18–v24 inverse runs used clip=50 and therefore have
> fictitious PC; their PC-based conclusions need re-running at
> clip=100 before being trusted. CLAUDE.md hard rule 2 codifies this;
> `docs/clipping_conventions.md` has the same rule in callout form.

This log captures (a) what was learned by extending the MMS test to multiple
voltages, (b) why that test does **not** answer the SS-observable question,
(c) the corrected understanding of what the clip is actually protecting in
the log-rate code, and (d) the threshold-sensitivity test that **does** answer
the question.

---

## 1. The MMS voltage-sweep test

Built `scripts/verification/mms_voltage_sweep.py` on top of
`scripts/verification/mms_bv_3sp_logc_boltzmann.py`. The underlying MMS
script was minimally refactored to accept `eta_hat`, `v_rhe`, and
`clip_source` as kwargs (defaults preserve `tests/test_mms_convergence.py`
behaviour).

For each V_RHE in `[+0.55, +0.495, +0.40, +0.20, -0.10, -0.30, -0.50] V` the
sweep runs the production-faithful MMS convergence study on
`UnitSquareMesh(N)` for N ∈ [8, 16, 32, 64] in two modes:

- `unclipped` (default): manufactured BV source uses unclipped
  `η = (V−E_eq)/V_T`. Solver applies the production η-clip at ±50.
- `consistent`: source applies the same ±50 clip as the solver.

Artifacts: `StudyResults/mms_voltage_sweep/` (JSON + 4 PNGs, two modes).

### 1.1 Three regimes observed

| V_RHE | η_R1 | η_R2 | unclipped Newton | consistent Newton | consistent L2 rate |
|---|---|---|---|---|---|
| +0.55 | -5.06 | -47.87 | 4/4, h² ✓ | 4/4, h² ✓ | (1.98, 2.00, 2.06, 2.20) |
| +0.495 | -7.20 | -50.01 | 4/4, H2O2 plateaus | 4/4, slight FP noise at kink | (1.73, 1.73, 1.72, 2.13) |
| +0.40 | -10.9 | -53.7 | 0/4 (DIVERGED_DTOL) | 4/4, h² ✓ | (2.04, 2.04, 1.91, 2.08) |
| +0.20 | -18.7 | -61.5 | 0/4 | 4/4, h² ✓ | (1.96, 1.96, 1.73, 2.04) |
| -0.10 | -30.4 | -73.2 | 0/4 | 4/4 (false convergence: u0 ≡ u1) | (-0.08, -0.08, +0.28, -0.20) |
| -0.30 | -38.1 | -81.0 | 0/4 | 0/4 (DIVERGED_DTOL) | — |
| -0.50 | -45.9 | -88.7 | 0/4 | 0/4 | — |

Three regimes:

- **V_RHE ≥ +0.20 V (consistent mode):** clipped operator achieves h^p
  convergence to U_manuf. The clipped equations are well-posed and the
  discrete solver faithfully approximates them.
- **V_RHE ≤ -0.10 V (consistent mode):** Newton converges to a wrong fixed
  point at -0.10 (u_O2 and u_H2O2 errors are *exactly equal* across all
  meshes — a stoichiometric-coupling lock from R1 when R2 saturates), and
  diverges entirely at V ≤ -0.30. This is an MMS artifact: U_manuf does
  not have c_H2O2 depleted, so it sits in the non-physical
  high-residual region rather than the SS basin.
- **Unclipped mode below threshold:** Newton diverges immediately because
  source–solver mismatch in the BV rate is `exp(|η_unclipped|−50)` =
  factor 40 at V=+0.40 up to factor 10^16 at V=-0.5.

## 2. Why the MMS sweep doesn't answer the SS-observable question

The MMS test verifies the discrete solver recovers a **manufactured shape**
under two source-construction choices. It does **not** measure the
difference between SS observables of the clipped vs. unclipped systems:

1. **U_manuf is not on either system's SS manifold.** It's a smooth
   designed shape `c_i = c0·(1 + 0.3·cos(πx)·(1−y)²)` that satisfies
   production Dirichlet BCs. The errors I reported at U_manuf are
   "discrete-vs-continuum at an arbitrary shape" — sensitive to the
   specific manufactured solution, not a SS-difference quantity.

2. **The unclipped solver has no usable SS in the clipped regime.**
   That's why the clip exists. Any test that needs both SS solutions
   in hand can't be run.

3. **The unclipped-mode plateau errors (e.g., H2O2 L2 ≈ 7e-3 at V=+0.495)
   reflect the BV-flux mismatch evaluated at U_manuf**, where
   c_H2O2_surf ≈ 1e-4 — far from the depleted physical SS value. The
   plateau magnitude scales with U_manuf's distance from SS, not with
   any SS-observable difference.

The sweep DID establish: **the clipped and unclipped equations differ
materially as PDEs in V<+0.495 V**, and the clipped equations are
internally h^p-consistent. That's an equation-level statement, not an
observable one.

## 3. Corrected understanding of what the clip protects

The framing in `docs/clipping_conventions.md` line 31:

> **Why it has to stay**: the final `exp(α·n_e·η)` would otherwise
> overflow double precision. `exp(±138) ≈ 10^±60`. There is no clean
> way around this in floating point; clipping is the standard guard.

is **incomplete** for the production V range. `exp(138)` is
representable in IEEE 754 double (`~10^308` is the max). For
V_RHE ∈ [-0.5, +0.6] V, the BV exponent `α·n·|η|` reaches at most ~89
without the clip — `exp(89) ≈ 10^39`, well within double range.

What log-rate actually fixed (vs. the legacy direct form):

- **Legacy** had an *explicit product* `c_surf · exp(α·n·|η|)` where
  `c_surf = exp(clamp(u_cat, ±30))`. At deeply-clipped SS, physical
  c_surf is `~exp(-88)` but the clamp pins it at `exp(-30)`. The product
  comes out wrong by ~58 OOM.
- **Log-rate** assembles `log_cath = ln(k0) + u_cat − α·n·η + ...` and
  exponentiates once. At SS, `u_cat → ln(c_surf) ≈ -88` and
  `−α·n·η ≈ +88` so they cancel: `log_cath ≈ ln(k0)`, a small number.
  No clamp is needed at SS; the c_surf clamp inside BV was eliminated.

What the η-clip is **really** doing in the log-rate world:

- **Bounding Jacobian conditioning during non-SS Newton iterates.** At a
  fresh / non-equilibrated Newton iterate, `u_cat` has not yet depleted
  (e.g., U_manuf has u_cat ≈ ln(1e-4) = -9, not -88). Then
  `log_cath = ln(k0) + u_cat − α·n·η = -10 + (-9) + 88 = +69`,
  and `exp(+69) ≈ 10^30`. The Jacobian entries scale with this rate
  (`dR/du = R`, `dR/dφ = -α·n_e/V_T · R`), giving a Jacobian whose row
  scales span ~10^30 (BV-coupled) to ~10^-3 (bulk diffusion). Condition
  number ~10^33; even MUMPS direct solve loses precision.
- **Without the clip, full Newton steps overshoot.** `snes_linesearch_l2`
  with `maxlambda=0.3` either backs off to no progress or steps into a
  region where some species' `log_cath` exceeds 700 and `exp` overflows
  to Inf, triggering DIVERGED_DTOL.
- **Production avoids this in practice via cold-ramp continuation:**
  each warm-start has c_surf already depleted from the previous voltage,
  so the new Newton sequence starts with `log_cath ≈ ln(k0)` already and
  the clip never bites at SS.

The MMS sweep confirmed this picture: at V=-0.5 V even *consistent-clip*
mode failed Newton, despite log-rate + matching source. R1 at V=-0.5 V
is unclipped (`η_R1 = -45.9`, `|η|<50`) and `R1_manuf ~ exp(51) ~ 10^22`
crashes Newton from U_manuf regardless of how the BV rate was assembled.
The clip is providing **transient Newton stability**, not strict
overflow protection.

The doc text in `clipping_conventions.md` should be amended; in the
production V range the clip is a Newton-stability safeguard, and only
becomes a strict overflow guard at V_RHE more cathodic than ~-3 V.

## 4. The test that DOES answer the SS-observable question

**Vary the clip threshold and compare SS observables** at the production
voltage grid. Specifically:

- Run `scripts/plot_iv_curve_unified.py` with `exponent_clip = 50`
  (production default — already in `StudyResults/iv_curve_unified/`).
- Re-run with `exponent_clip = 100`. At this threshold, the entire
  production grid V ∈ [-0.5, +0.6] V is *unclipped* for both reactions
  (R2 needs |η| < 100 → V > 1.78 − 100·V_T = -0.79 V; R1 even further).
- Optionally re-run with `exponent_clip = 200` to confirm convergence
  to a clip-independent limit.

Compare CD(V) and PC(V) across thresholds:

- If they're identical to within solver tolerance: the clip is
  **observable-neutral** in the production grid; the SS physics doesn't
  depend on the BV exponent value once the system is in any post-Tafel
  regime.
- If they differ in some V range: that range is where the clip is
  **distorting observable physics**, and the magnitude of the difference
  quantifies the distortion.
- If `clip=100` Newton fails at some voltages but `clip=50` succeeds,
  that itself is information: the clip is doing real work for solver
  robustness in those voltages, even if observable values agree where
  both converge.

Cost: ~3 minutes per full V grid (13 voltages, MESH_NY=200,
production solver), so ~10 minutes for the 3-threshold sweep.

Implementation: `scripts/studies/clip_threshold_sensitivity.py`
(parameterizes `exponent_clip` via `params["bv_convergence"]` after
`make_bv_solver_params` returns; otherwise mirrors
`plot_iv_curve_unified.py`).

## 5. Results

Ran `scripts/studies/clip_threshold_sensitivity.py --clips 50 100` on the
production V grid V_RHE ∈ {-0.5, …, +0.6} V at MESH_NY=200. Both runs
converged at all 13 voltages. Total runtime ~3 minutes. Artifacts in
`StudyResults/clip_threshold_sensitivity/` (per-clip JSON+CSV +
`comparison.png` + `summary.json`).

At `exponent_clip = 100` the entire production grid runs **unclipped**
for both reactions (R2 needs |η| < 100 → V > 1.78 − 100·V_T = -0.79 V;
R1 always unclipped on production grid). So the clip=100 columns below
are the "true unclipped physics" reference.

### 5.1 CD (total current density) — clip is approximately observable-neutral

| V_RHE | CD@50 [mA/cm²] | CD@100 [mA/cm²] | ΔCD/CD |
|---|---|---|---|
| -0.500 | -1.8410e-01 | -1.8379e-01 | -1.67e-03 |
| -0.400 | -1.8305e-01 | -1.8278e-01 | -1.52e-03 |
| -0.300 | -1.8172e-01 | -1.8161e-01 | -6.09e-04 |
| -0.200 | -1.8017e-01 | -1.8017e-01 | -1.74e-05 |
| -0.100 | -1.7802e-01 | -1.7802e-01 | -1.16e-06 |
| 0.000 to +0.5 | (identical) | (identical) | < 1e-6 |

Maximum CD distortion is **0.17%** at V=-0.5 V (the deepest cathodic
voltage on the production grid). At V≥-0.2 V the difference is below
typical solver tolerance. **For total current the clip is observable-
neutral for inverse fitting purposes** — well below experimental noise.

### 5.2 PC (peroxide current) — clip QUALITATIVELY changes the answer

| V_RHE | PC@50 [mA/cm²] | PC@100 [mA/cm²] | sign / magnitude shift |
|---|---|---|---|
| -0.500 | **-1.81e-01** | **+1.54e-05** | sign flip + 4 OOM |
| -0.400 | -1.64e-01 | +1.54e-05 | sign flip + 4 OOM |
| -0.300 | -6.50e-02 | +1.54e-05 | sign flip + 3 OOM |
| -0.200 | -1.57e-03 | +1.54e-05 | sign flip + 2 OOM |
| -0.100 | +2.91e-06 | +1.54e-05 | small positive shift |
| 0.000 to +0.6 | (matches) | (matches) | <1% |

PC@100 holds at ~+1.5e-5 mA/cm² across the cathodic grid — exactly the
diffusion supply `D_H2O2·c_H2O2_seed/L ≈ 0.193·1e-4 = 1.9e-5` from the
H2O2_SEED_NONDIM=1e-4 bulk. That's the **true mass-transport-limited
PC**: every H2O2 molecule diffusing in from the seed bulk gets consumed
by R2 (running at its full unclipped rate). Net peroxide flux at the
electrode equals the diffusion supply.

PC@50 in the same regime sits at ~-0.18 mA/cm² — opposite sign,
4 orders of magnitude larger. The clip throttles R2's rate so c_H2O2
must accumulate at the electrode to balance R2 against R1's production;
PC is then dominated by H2O2 *outflow* from the electrode toward the
bulk (cathodic-signed convention).

**This is a qualitative physical difference, not a small distortion.**
Anyone using PC data at V_RHE < -0.1 V for inverse fitting (k0_R2,
α_R2) would get systematically different answers depending on the clip.

### 5.3 Internal c_H2O2 surface concentration — opposite-sign artifact

| V_RHE | <c_H2O2_surf>@50 (nondim) | <c_H2O2_surf>@100 (nondim) | ratio |
|---|---|---|---|
| -0.500 | **1.17e+00** (accumulates 10⁴× above bulk seed) | **3.5e-13** (depleted 10⁹× below seed) | ~3e12 |
| -0.400 | 1.06e+00 | 1.3e-13 | ~8e12 |
| -0.300 | 4.20e-01 | 4.9e-14 | ~9e12 |
| -0.200 | 1.02e-02 | 1.8e-14 | ~6e11 |
| -0.100 | 8.06e-05 | 7.0e-15 | ~1e10 |
| 0.000 | 6.26e-07 | 2.6e-15 | ~2e08 |

At deep cathodic V the clipped and unclipped solvers give **opposite
qualitative pictures** of internal physics:

- **clip=50:** c_H2O2_surf = 1.17 (10⁴× above bulk seed). H2O2
  *accumulates* at the electrode because R2's clipped exponent can't
  consume it fast enough.
- **clip=100 (true physics):** c_H2O2_surf = 3.5e-13 (10⁹× below
  seed). H2O2 is *depleted* at the electrode, the standard
  mass-transport-limited picture.

The doc text in `clipping_conventions.md` lines 200-210 anticipated
this for *internal* fields ("not physically meaningful in the clipped
regime — artificially elevated"), but missed that PC inherits the
artifact and so is **not** approximately physical.

### 5.4 Newton convergence basin — the clip does real solver-robustness work

Phase-1 cold-convergence rate:

- **clip=50: 10/13 cold-converged** (only 3 cathodic edge points needed
  warm-walk rescue).
- **clip=100: 3/13 cold-converged** (only V_RHE ≥ +0.40 V cold-
  converged; the entire cathodic half of the grid required warm-walk
  continuation from the +0.5/+0.55/+0.60 V cold anchors).

This is *direct* evidence that the η-clip is doing real Newton-basin
work, not just FP overflow protection. Without the clip, production
would still converge (warm-walk rescues every voltage in this run),
but the cold-start basin shrinks dramatically. For any workflow that
attempts cold solves at perturbed parameters (the inverse pipeline at
non-TRUE k0/α, FIM gradient checks, parameter sweeps) the clip's
basin contribution may be load-bearing.

### 5.5 Bottom-line answer to the question

**Yes, the clip qualitatively changes physical behaviour in the
voltage range where it applies — but only for some observables.**

- **CD: approximately preserved (~0.2% max distortion at V=-0.5).**
  Inverse fits using CD only would not be materially affected by the
  clip in production.
- **PC: qualitatively distorted at V_RHE < -0.1 V** (sign flip plus
  3-4 OOM magnitude shift). Inverse fits using PC data in this voltage
  range are contaminated by the clip artifact.
- **Internal c_H2O2 profile: artifact ~10⁹–10¹² × physical** at the
  surface in deep cathodic V. Don't trust electrode concentration plots
  there.
- **Solver convergence basin: clip does real work**, expanding the cold-
  start basin from 3/13 to 10/13 voltages on the production grid.

The `clipping_conventions.md` claim that forward observables are
"approximately physical" in the clipped regime needs splitting: **CD
yes, PC no**. The α_2 sensitivity collapse predicted there is a
related but distinct phenomenon (gradient identifiability), which the
present test does not directly measure but which is consistent with
the PC distortion.

> **Spin-off question — what about the small PC dip near V=+0.6 V?**
> Investigating that led into a separate solver-architecture
> question about the peroxide window past E_eq_R1 = +0.68 V. That
> investigation is logged in
> `docs/peroxide_window_investigation.md` and concluded that the
> peroxide window is structurally inaccessible to the current
> production stack — but that finding is **independent** of the
> clip-effect question (it's about Newton basin navigation across
> the R1 reversal, not about the clip), so it does not change the
> conclusions of this log.

### 5.6 What this means for the inverse pipeline

- **PC data below V_RHE = -0.1 V is unsafe for k0_R2 / α_R2 fitting**
  using the production clip=50 — fitted values absorb the clip-induced
  PC artifact rather than reflecting physical kinetics.
- **CD data at V_RHE < +0.495 V is approximately safe for fitting**
  (~0.2% bias at the deepest point, much less above), provided one
  accepts that α_2 sensitivity is essentially zero there (CLAUDE.md
  rule #6, `docs/clipping_conventions.md` α_2 discussion).
- **A clip=100 production run is a viable "high-fidelity reference"**
  for any future inverse work that wants to use peroxide data in the
  cathodic regime — Newton basin shrinks, but warm-walk continuation
  succeeds at every production voltage in this run.
- **The Tafel-slope test (Tier 1 hole #1 in
  `docs/forward_solver_test_coverage.md`) should be run at clip=100,
  not clip=50.** At clip=50 the slope below +0.495 V is set by the
  clip-frozen α_2·n_e·50 exponent, not by the physical α_2.

## 6. Pointers

- Production driver: `scripts/plot_iv_curve_unified.py`
- Clip plumbing: `scripts/_bv_common.py:_make_bv_convergence_cfg`
  (hardcodes `exponent_clip=50.0`); `Forward/bv_solver/config.py`
  (default also 50.0); applied in
  `Forward/bv_solver/forms_logc.py:_build_eta_clipped`.
- Existing convention doc (now amended): `docs/clipping_conventions.md`.
- MMS sweep script: `scripts/verification/mms_voltage_sweep.py`.
- MMS sweep artifacts: `StudyResults/mms_voltage_sweep/`.
- Clip-threshold sensitivity runner:
  `scripts/studies/clip_threshold_sensitivity.py`.
- Clip-threshold artifacts: `StudyResults/clip_threshold_sensitivity/`.
- Spin-off (peroxide window past E_eq_R1):
  `docs/peroxide_window_investigation.md`.
- Hard rules: `CLAUDE.md` rule #4 (R2 unclips at V_RHE > +0.495 V).
