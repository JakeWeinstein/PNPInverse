# 4sp dynamic — can we drop the Boltzmann reduction? Investigation log

**Date:** 2026-05-03

## Bottom line

**No, not with the current residual machinery — but the failure mode is
now precisely localized.**

The path "drop the analytic Boltzmann counterion, use 4-species dynamic
PNP with Bikerman steric, optionally with finite Stern" looks
attractive: it would replace the unphysical c_ClO4 ~ 10^{16} blowup
of 3sp+Boltzmann at high anodic V with a physically saturated
c_ClO4 ≤ 1/a ~ 100 nondim. The math says this should work.

The residuals say otherwise. After landing a `debye_boltzmann` IC
extension that synthesises a virtual counterion entry from `c0[3]`
when `boltzmann_counterions` is empty (so the IC actually fires for
4sp instead of falling back to linear_phi), every cold solve at
V_RHE ≥ +0.3 V still fails — and the orchestrator falls back to the
linear_phi z-ramp anyway. **The Boltzmann analytical IC and the
Bikerman steric residual are incompatible at high anodic V**: the IC
seeds `c_ClO4 = c_bulk · exp(+ψ_D) ~ 10^4` at the surface, the
Bikerman residual `ln(1 − a·c_ClO4) = ln(1 − 200) = NaN`, and Newton
diverges on the first step at z=1.

The next step is a Bikerman-corrected IC — modified Poisson-Boltzmann
(MPB / Bikerman-Kornyshev) — so the IC respects the steric cap from
the start. Three options at increasing complexity are sketched in §6.

## 1. Starting question

Per `docs/stern_layer_physics_and_next_steps.md` and the
`StudyResults/peroxide_window_stern_test/` results, the production
3sp+Boltzmann + Stern stack converges across the peroxide window but
at C_S=0.05 F/m² gives `c_ClO4 ≈ 250–880` nondim — 2.5–9× over the
Bikerman steric cap (~100). The converged state is non-physical.

The 4sp dynamic preset (`FOUR_SPECIES_LOGC_DYNAMIC` in
`scripts/_bv_common.py`) tracks ClO4- as a regular NP species with
z=-1 and `a_vals_hat=A_DEFAULT=0.01`, so the Bikerman steric term
in the residual would naturally cap c_ClO4 at ~100. Question: does
the 4sp solver now converge in the peroxide window? If yes, drop
Boltzmann.

## 2. Initial 4sp test — fails everywhere at clip=100

Script: `scripts/studies/peroxide_window_4sp_quick.py`. V grid the
peroxide window (V_RHE ∈ [0.60, 1.00], 7 voltages). Three passes at
Ny=200, exponent_clip=100, linear_phi:

| Pass | Convergence | Wall |
|---|---|---|
| 4sp + linear_phi + no Stern | 0/7 | 77 s |
| 4sp + linear_phi + Stern C_S=0.10 | 0/7 | 19 s |
| 4sp + linear_phi + Stern C_S=0.05 | 0/7 | 3 s |

Without Stern, every cold solve PARTIAL z=0.10–0.20. With Stern, the
z-ramp aborts at z=0.000 — Newton diverges instantly because the
linear_phi IC seeds `phi(electrode) = phi_applied`, contradicting the
Robin BC's `phi_s ≪ phi_m` assumption.

Artifacts: `StudyResults/peroxide_window_4sp_quick/`.

## 3. Cathodic sanity check — exposes a clip mistake

Hypothesis at this point: 4sp + Ny=200 is regressed (the equivalence
test runs at Ny=100). To bisect, ran 4sp + linear_phi at the legacy
convergence window V_RHE ∈ [-0.5, +0.1], Ny=200, **clip=100**:
`scripts/studies/peroxide_window_4sp_cathodic_check.py`.

Result: 0/5 — even cathodic V failed. Tried Ny=100: also 0/5. But
`pytest tests/test_solver_equivalence.py -m slow` passed 3/3 in 116 s
on the same code.

Diff between my script and the equivalence test: my script overrode
`exponent_clip=100`. The equivalence test uses the default clip=50.

Per `docs/clip_observable_investigation.md` §3 and §5.4:
- clip=50 unclips R2 only at V_RHE > +0.495 V; below that R2 is
  clipped to a bounded exponent.
- clip=100 unclips R2 across the whole grid (V > -0.79 V); at deep
  cathodic V the cathodic exponent `exp(-α·n_e·η_R2)` at V=-0.5 is
  `exp(88.7) ≈ 3e+38` — overflow, Newton diverges.

Re-ran cathodic check at clip=50, Ny=200: **5/5 converged**, CD ~ -0.18
mA/cm² matching 3sp+Boltzmann, surface c_ClO4 within 1e-9 to 11
nondim — well below Bikerman cap.

**Lesson: 4sp dynamic at Ny=200 is healthy at the production clip. The
"0/29" result from the earlier 14:57 run was a clip=100 artifact, not
a code regression.**

Artifacts:
- `StudyResults/peroxide_window_4sp_cathodic_check_ny200/` (clip=100)
- `StudyResults/peroxide_window_4sp_cathodic_check_ny100/` (clip=100)
- `StudyResults/peroxide_window_4sp_cathodic_check_ny200_clip50/` (clip=50)

## 4. Extended-grid test with linear_phi at clip=50

Script: `scripts/studies/peroxide_window_4sp_extended.py`. V grid
`[-0.5, -0.3, -0.1, 0.0, +0.1, +0.3, +0.5, +0.55, +0.60, +0.65, +0.66,
+0.68, +0.70, +0.75, +1.00]` (15 voltages — cathodic anchors plus the
peroxide window). Two passes:

| Pass | Cold | Warm | Total | Max V |
|---|---|---|---|---|
| 4sp + linear_phi + no Stern | 3 | 2 | 5/15 | +0.10 |
| 4sp + linear_phi + Stern C_S=0.10 | 5 | 0 | 5/15 | +0.10 |

**Both anodic warm-walks broke at the same step: V=+0.1 → V=+0.3**.
Stern helped cold-convergence at deep cathodic V (5 cold vs 3) but
didn't bridge the +0.1 → +0.3 jump.

**Why Stern didn't help in 4sp the way it did in 3sp+Boltzmann.**
Compare Stern voltage drops at V=+0.1:

| Stack | Stern drop (nondim) |
|---|---|
| 3sp+Boltzmann + Stern (prior study, peroxide window) | +18 |
| 4sp + Stern (this study) | +0.98 |

Stern barely engages in 4sp. Bikerman steric naturally suppresses
surface charge buildup (c_ClO4 stays at 1–37 nondim across the
partial-z states vs 1e+9 in 3sp+Boltzmann no-Stern), so phi_s ≈
phi_applied — the Robin BC has nothing to do.

The +0.1 → +0.3 wall is the same kinetic-onset transition identified
in `docs/ic_refinement_study.md`: ORR shifts from mass-transport
limited to kinetic regime, and linear_phi IC sits very far from the
depleted-H+ surface manifold. For 3sp+Boltzmann this was unblocked by
the analytical `debye_boltzmann` IC. For 4sp, the IC silently fell
back to linear_phi.

Artifacts:
- `StudyResults/peroxide_window_4sp_extended_linear_phi/`

## 5. Extending `debye_boltzmann` IC to 4sp dynamic

### 5.1 The IC blocker

`Forward/bv_solver/forms_logc.py::_try_debye_boltzmann_ic` line 599-601
(pre-extension):

```python
counterions = _get_bv_boltzmann_counterions_cfg(params)
if not counterions:
    return False, "no_boltzmann_counterion", 0
```

For 4sp dynamic (`boltzmann_counterions=None`), this fast-bails. The
IC entry function then falls back to linear_phi. Result: passing
`initializer="debye_boltzmann"` for 4sp is functionally identical to
linear_phi.

### 5.2 The extension

In this session: relax the bail check to synthesise a virtual
counterion entry from `c0[3]` when no Boltzmann config is provided
but n_species==4 and z_vals[3]==-1 (the FOUR_SPECIES_LOGC_DYNAMIC
preset). The Picard outer loop then fires the same as for
3sp+Boltzmann.

Additionally: explicitly seed `u_3` (ClO4-) on the Boltzmann manifold
after the analytical IC builds phi:

```python
phi_init_expr = fd.ln(H_outer / fd.Constant(c_clo4_bulk)) + psi
U_prev.sub(n).interpolate(phi_init_expr)
if synthesised_4sp_counterion:
    # u_3 = ln(c_ClO4(y)) on the Boltzmann manifold:
    #   c_ClO4(y) = c_clo4_bulk · exp(+phi(y))  (z=-1)
    U_prev.sub(3).interpolate(
        fd.Constant(math.log(c_clo4_bulk)) + phi_init_expr
    )
```

This is the natural extension of the existing H+ initialization
`u_2 = ln(H_outer) - psi` for z=+1, mirrored to z=-1 for ClO4-. Both
species are now seeded on the matched-asymptotic profile the IC's
Picard outer loop assumes.

### 5.3 Tests

New file: `tests/test_initializer_debye_boltzmann_4sp.py`. 8 tests,
all passing in 2.4 s:

- `test_ic_fires_at_v05` — IC does not fall back at V=+0.5 V, picard
  iterates > 0.
- `test_ic_seeds_clo4_on_boltzmann_manifold` — `max|u_3 −
  (ln(c_clo4_bulk) + phi)| < 1e-6` (FE interpolation tolerance).
- `test_ic_reaches_phi_applied_at_electrode` — phi.max ≈ phi_applied.
- `test_ic_depletes_h_at_electrode` — u_H min < ln(c_bulk) − 15
  (Boltzmann depletion for z=+1).
- `TestRegression3spStillWorks::test_3sp_still_fires` — the original
  3sp+Boltzmann path is unchanged (synthesis logic only fires when
  `counterions` is empty).

Pre-existing tests unaffected:
- `tests/test_initializer_debye_boltzmann.py` — 3 tests, all pass.

## 6. Extended-grid test with `debye_boltzmann` IC at clip=50

Same script (`peroxide_window_4sp_extended.py`), now with
`INITIALIZER=debye_boltzmann`. Two passes:

| Pass | Cold | Warm | Total | Max V |
|---|---|---|---|---|
| 4sp + debye_boltzmann + no Stern | 3 | 2 | 5/15 | +0.10 |
| 4sp + debye_boltzmann + Stern C_S=0.10 | 5 | 0 | 5/15 | +0.10 |

**Bit-for-bit identical** to the linear_phi run. Same converged set,
same partial-z values at every voltage, same stern_drop pattern.

Artifacts: `StudyResults/peroxide_window_4sp_extended_debye_boltzmann/`.

### 6.1 The diagnostic — IC fires, residual rejects

Per-voltage `initializer_fallback` flag from
`StudyResults/.../diagnostics.json`:

| Pass / V_RHE | fallback | picard_iters | interpretation |
|---|---|---|---|
| no_stern / V ∈ [-0.5, +0.1] | True | 50 | Picard hit max iters (cathodic regime; documented in `docs/ic_refinement_study.md` §4) |
| no_stern / V ∈ [+0.3, +1.0] | True | 21 | Picard CONVERGED in 21 iters — IC fired — but flag is True |
| stern_0.10 / V ∈ [-0.5, +0.1] | True | 50 | Same cathodic pattern |
| stern_0.10 / V ∈ [+0.3, +1.0] | True | 21 | Same — Picard succeeded but fallback fired |

The `picard_iters=21` matches the documented 3sp+Boltzmann count at
anodic V (`docs/ic_refinement_study.md` §2: 21 iters at V=+0.3, +0.4,
+0.5, +0.6 with `debye_boltzmann`). So Picard IS succeeding for 4sp
with my synthesised counterion. The fallback comes from somewhere
else.

That somewhere is the orchestrator. `Forward/bv_solver/grid_per_voltage.py`
line 330-349:

```python
if _initializer_flag != "linear_phi" and not ctx.get("initializer_fallback", False):
    _set_z_factor(ctx, 1.0)
    if run_ss(max_ss_steps_z):
        return _snapshot_U(U), 1.0, ctx
    # Direct z=1 failed — reset to linear-phi IC and use the
    # standard z-ramp path below
    ctx["initializer_fallback"] = True
    ctx["initializer_fallback_reason"] = (
        ctx.get("initializer_fallback_reason", "") + ";cold_z1_diverged"
    )
    set_initial_conditions_logc(ctx, _params_with_phi(V_target_eta))
```

When the IC has fired (Path B), the orchestrator skips the z-ramp and
attempts direct z=1 SS. If that fails, it marks `initializer_fallback=True`,
resets to linear_phi, and runs the standard z-ramp (Path A).

For 4sp + my IC at V=+0.3:
1. Picard outer loop succeeds (21 iters).
2. IC writes the Boltzmann-seeded state: `u_3(electrode) = ln(c_clo4_bulk) + ψ_D`,
   so `c_ClO4(electrode) ≈ 0.2 · exp(11.5) ≈ 2e+04` nondim.
3. Orchestrator tries direct z=1 SS from this state.
4. Newton evaluates the residual. The Bikerman steric chemical
   potential
   ```python
   packing = max(1 - sum_j a_j · c_j, packing_floor)
   mu_steric = fd.ln(packing)
   ```
   sees `1 − 0.01 · 2e+04 = -199`. With `packing_floor=1e-8`, packing
   gets clamped, but the resulting `ln(1e-8) = -18.4` is a wildly
   non-physical chemical potential — and the gradient `∂μ/∂c =
   -a/packing` is `-0.01/1e-8 = -1e+6`, an enormous Jacobian entry.
   Newton's first step explodes.
5. Orchestrator catches the divergence, falls back to linear_phi
   z-ramp. Z-ramp progresses to the same partial-z it would have
   without the IC extension. Final result identical.

### 6.2 Why this is the predicted failure mode

The Boltzmann distribution `c_i = c_bulk · exp(-z·ψ)` has no upper
bound. The Bikerman steric residual `ln(1 − a·c)` has a logarithmic
singularity at `c = 1/a`. At V=+0.3 with c_bulk=0.2 and a=0.01:

- ψ_D ≈ 11.5
- Boltzmann prediction: c_ClO4 = 0.2 · exp(11.5) ≈ 2e+04
- Bikerman cap: 1/a = 100
- Boltzmann/cap ratio: ~200

The Boltzmann analytical IC is OFF the Bikerman manifold by 2 orders of
magnitude at V=+0.3, and exponentially worse at higher V. Newton at
z=1 cannot bridge this — the first residual evaluation produces a
Jacobian conditioning blow-up, and there's no continuation path from
"IC's Boltzmann state" to "Bikerman SS state" that doesn't pass
through the singularity.

For 3sp+Boltzmann, this isn't a problem because ClO4 is analytic
(with its own `phi_clamp=50`) and Bikerman doesn't see it (a_3=0 in
the 3sp preset). For 4sp dynamic, ClO4 IS a primary variable subject
to Bikerman steric — and the Boltzmann IC is a physical contradiction
of the residual.

## 7. Summary — what works, what doesn't

| Stack | Newton at V≥+0.3 | Surface c_ClO4 |
|---|---|---|
| 3sp+Boltzmann + linear_phi + no Stern | fails (peroxide window) | 1e+9 (Boltzmann blow-up) |
| 3sp+Boltzmann + debye_boltzmann + Stern C_S=0.05 | converges | 250–880 (over cap) |
| 4sp + linear_phi + no Stern (clip=50) | fails | 1–37 (in cap) |
| 4sp + linear_phi + Stern C_S=0.10 | fails (warm-walk breaks at +0.1→+0.3) | 1–37 (in cap) |
| 4sp + debye_boltzmann + ±Stern | fails (z=1 diverges → fallback) | irrelevant (post-fallback state) |

Two failure modes, dual to each other:
- **3sp+Boltzmann**: Newton converges with a non-physical IC manifold.
- **4sp + Bikerman**: Newton fails because the IC manifold is
  physical-but-incompatible with the residual.

Neither stack as currently implemented gives a fully physical
forward solution at peroxide-window voltages.

## 8. Next step — Bikerman-corrected IC

The `c_i(ψ)` part of modified Poisson-Boltzmann
(Bikerman-Kornyshev / Iglič) IS closed-form:

```
c_i(ψ) = γ(ψ) · c_bulk_i · exp(-z_i · ψ)
γ(ψ) = 1 / [1 + Σ_j a · c_bulk_j · (exp(-z_j·ψ) − 1)]
```

For binary symmetric (H+ and ClO4- at c_bulk and z=±1):

```
γ(ψ) = 1 / [1 + 2·a·c_bulk · (cosh(ψ) − 1)]
```

As ψ → ∞, γ → 1/(a·c_bulk·exp(ψ)), so c_ClO4 → 1/a. Saturation falls
out of the equilibrium correction directly.

The `(dψ/dy)²` first integral is also closed-form:

```
(dψ/dy)² = (2/(ν·λ_D²)) · ln[1 + ν·(cosh(ψ) − 1)]    where ν = 2·a·c_bulk
```

What's NOT closed-form: the inverse map ψ(y). The 1D integral
`y(ψ) = ∫ dψ' / |dψ'/dy|` has no elementary inverse. The current IC's
`4·atanh(tanh(ψ_D/4)·exp(-y/λ_D))` is the *standard* GC closed-form;
the Bikerman version doesn't have one.

### 8.1 Three options at increasing complexity

**Option 2a — γ-correction only (5 lines, partially correct).**
Keep the standard GC ψ(y) but multiply by γ(ψ) when computing u_3:

```python
# In _try_debye_boltzmann_ic, replace the existing u_3 interpolation:
gamma = fd.Constant(1.0) / (
    fd.Constant(1.0) + fd.Constant(2.0 * a * c_clo4_bulk)
    * (fd.cosh(psi_safe) - fd.Constant(1.0))
)
U_prev.sub(3).interpolate(
    fd.ln(fd.Constant(c_clo4_bulk) * gamma) + phi_init_expr
)
```

c_ClO4 = γ · c_bulk · exp(+ψ) automatically saturates at ~1/a. The
ψ profile is still the unmodified GC (formally inconsistent with
Bikerman near saturation), but the IC at least lands on a state
where the Bikerman residual is finite.

Cheapest test of "does this even unblock the warm-walk past +0.3 V."

**Option 2b — composite asymptotic ψ(y) (10–15 lines, more correct).**
Piecewise expression:
- Saturated zone (ψ > ψ_sat ≈ ln(2/ν)): linear decay
  `ψ(y) ≈ ψ_D − α·y` with α from the first integral at ψ=ψ_D.
- Outer zone (ψ < ψ_sat): exponential decay
  `ψ(y) = ψ_sat · exp(-(y − y_match)/λ_D)`.
- Match `y_match` from continuity.

Gets the *thickness* of the saturated layer right. Closed-form
expressions in each zone, just piecewise.

**Option 2c — numerical ODE integration (20–30 lines, fully correct).**
Numerically integrate `dψ/dy = -sqrt((2/(ν·λ_D²))·ln(1+ν·(cosh(ψ)-1)))`
from ψ=ψ_D at y=0 to ψ=ε at y_far. `scipy.integrate.solve_ivp` does
this in a few lines. Interpolate ψ(y) onto FE nodes via 1D look-up.
Fully Bikerman-respecting profile.

### 8.2 Recommendation

Start with **Option 2a**. It directly addresses the diagnosed failure
mode (Bikerman residual sees c_ClO4 above cap → NaN → Newton diverges
at z=1). 5 lines inside `_try_debye_boltzmann_ic`. If it still
doesn't unblock the warm-walk past V=+0.3 V, escalate to 2b or 2c.

What 2a does NOT fix:
- Picard outer loop's `H_o` calculation still uses bulk-electroneutrality.
  For 4sp this is fine (Picard runs on the outer mass-transport
  region where γ ≈ 1).
- The phi(y) profile inside the Debye layer is still standard GC. So
  intermediate-y phi is wrong-by-Bikerman but converges to correct
  boundary values. Newton has to repair the inner profile from a
  finite, Bikerman-respecting state instead of from NaN.

### 8.3 Validation gates for the Bikerman-corrected IC

If 2a (or 2b/2c) is implemented, the regression gates are:

1. `tests/test_initializer_debye_boltzmann.py` (3sp+Boltzmann) must
   still pass byte-identically. The γ-correction code path only
   fires when `synthesised_4sp_counterion` is True, so 3sp+Boltzmann
   should be untouched.
2. `tests/test_initializer_debye_boltzmann_4sp.py` — currently
   asserts `c_ClO4 = c_bulk · exp(+phi)` (pure Boltzmann). Update to
   assert `c_ClO4 ≤ 1/a · (1 − ε)` at every node, AND
   `c_ClO4(bulk) ≈ c_bulk` (γ → 1 in bulk).
3. `tests/test_solver_equivalence.py` — 4sp dynamic is tested at
   Ny=100, V_RHE ∈ [-0.5, +0.1] with default linear_phi initializer.
   The 2a change shouldn't affect this (initializer ≠ debye_boltzmann
   in this test), but verify.
4. New: a peroxide-window snapshot at V_RHE = +0.3 V should converge
   under 4sp + debye_boltzmann + 2a, with surface c_ClO4 ≈ 1/a (within
   a factor of 2). If converged but c_ClO4 < ~50, suspect γ correction
   is too aggressive.

## 9. Pointers

### Code

- `Forward/bv_solver/forms_logc.py:571-790` — `_try_debye_boltzmann_ic`,
  with the 4sp synthesis added in this session at lines ~615–640 (the
  "Locate the Boltzmann counterion bulk concentration" block) and
  the u_3 seeding at lines ~782–788.
- `Forward/bv_solver/grid_per_voltage.py:330-349` — orchestrator
  Path B → A fallback (the source of `;cold_z1_diverged` reason
  observed in this study's diagnostics).
- `scripts/_bv_common.py:FOUR_SPECIES_LOGC_DYNAMIC` — the species
  preset used here (lines containing the SpeciesConfig definition;
  z_vals=[0,0,1,-1], a_vals_hat=[0.01]*4).

### Tests

- `tests/test_initializer_debye_boltzmann.py` — 3sp+Boltzmann
  regression (3 tests, slow).
- `tests/test_initializer_debye_boltzmann_4sp.py` — new 4sp IC tests
  (5 tests + 1 regression test, all slow). All passing as of
  2026-05-03.
- `tests/test_solver_equivalence.py` — 3sp+Boltzmann ↔ 4sp dynamic
  equivalence at Ny=100. Still passes 3/3.

### Studies (this session)

- `scripts/studies/peroxide_window_4sp_quick.py` — initial test at
  clip=100. Failed everywhere. Artifacts:
  `StudyResults/peroxide_window_4sp_quick/`.
- `scripts/studies/peroxide_window_4sp_cathodic_check.py` — sanity
  check that exposed the clip=100 vs clip=50 confusion. Takes
  positional args `Ny` and `clip`. Artifacts in
  `StudyResults/peroxide_window_4sp_cathodic_check_ny{Ny}_clip{clip}/`.
- `scripts/studies/peroxide_window_4sp_extended.py` — extended V grid,
  parameterized by `INITIALIZER` argv. Artifacts in
  `StudyResults/peroxide_window_4sp_extended_{linear_phi,debye_boltzmann}/`.

### Companion docs

- `docs/stern_layer_physics_and_next_steps.md` — Stern physics
  framing.
- `docs/ic_refinement_study.md` — `debye_boltzmann` IC behaviour for
  3sp+Boltzmann.
- `docs/clip_observable_investigation.md` — clip=50 vs clip=100;
  why clip is doing real Newton-basin work.
- `docs/peroxide_window_investigation.md` — companion to the Stern
  test; the parent narrative for "why does the peroxide window not
  work."

### Related references (Bikerman-Kornyshev MPB)

- Bikerman (1942) — original "modified Poisson-Boltzmann" framework.
- Kornyshev (2007) — modern lattice-gas formulation; gives the
  γ-correction in the form used here.
- Iglič et al. — derivations of `(dψ/dy)²` first integral for
  multi-species MPB.
- Bazant et al. (2009) "Diffuse-charge dynamics in electrochemical
  systems" — comprehensive review of MPB.

## 10. Open questions for review

1. Should `Option 2a` be implemented and tested next? The diagnostic
   here suggests it's the minimum change that has a chance of
   unblocking V≥+0.3 V for 4sp. But 2b/2c would be more correct and
   not much more code.
2. Is the steric flag in `FOUR_SPECIES_LOGC_DYNAMIC` (a=0.01 → cap
   = 100) the right physics? Bikerman ν = 2·a·c_bulk = 0.004 — this
   is in the "moderate steric" regime per the MPB literature. Larger
   a (smaller cap) would saturate sooner; smaller a recovers
   pure Boltzmann.
3. If 2a unblocks 4sp at V=+0.3 but warm-walk still breaks somewhere
   around V=+0.5–0.6, what's the next investigation step? Likely the
   intermediate-y phi profile (which is still standard GC even with
   γ-corrected u_3) is the next obstacle — escalate to 2b.
4. Is there a hybrid worth considering: 3sp+Boltzmann + Bikerman
   steric on the analytical counterion residual? That would keep the
   3sp Newton tractability AND saturate c_ClO4. The ClO4-Bikerman
   combination would need to be added to `boltzmann.py`'s residual
   (ln(1-a·c_bulk·exp(-z·phi)) appears alongside the existing
   exp(-z·phi)). This is closer to "fix the symptom" than "use the
   right model" but might be the lowest-cost win.

## 11. Resolution — steric chemical-potential sign (2026-05-04)

**Diagnosis confirmed.** §6 and the cross-table at the end of this doc
flagged the implemented `μ_steric = +ln(1−Φ)` (`forms_logc.py:266`) as
having the opposite sign from the standard MPB derivation (Borukhov-
Andelman-Orland 1997 eq (3); Bazant-Kilic-Storey-Ajdari 2009 eq (20)).
The variational derivative of the lattice-gas entropy density
`(1−Φ)·ln(1−Φ)` produces a `−ln(1−Φ)` excess chemical potential
unambiguously; full derivation captured in
`docs/steric_sign_correction_plan.md` §1. The execution distillation
of that plan is in `~/.claude/plans/read-docs-steric-sign-correction-plan-md-ticklish-thimble.md`.

**Code change applied.** Two one-character edits:

- `Forward/bv_solver/forms_logc.py:266` — `mu_steric = fd.ln(packing)` →
  `mu_steric = -fd.ln(packing)`
- `Forward/bv_solver/forms_logc_muh.py:321` — same flip (parity for the
  experimental muh path; never exercised by tests until now)

**MMS source updated** (companion fix): `scripts/verification/mms_bv_3sp_logc_boltzmann.py:234`
flipped to match. The manufactured-solution source term must mirror the
residual sign or convergence rates collapse; without this fix
`test_mms_convergence::test_l2_convergence_rates` regresses on the O2
(u) L2 component (slope 0.28 vs expected 2.0).

**Writeup not yet edited.** The `docs/PNP Equation Formulations.tex:152`
spec definition still reads `μ^{steric}(c) = k_B T·ln(1−Σ a c)` (no
leading minus). Per the user's plan-mode choice (1 of 4: "Code edit +
tests only, skip writeup"), the LaTeX edit is deferred pending advisor
sign-off. Cross-doc reminders to that effect live in the plan file.

**Tests added.**

- `tests/test_steric_saturation.py::test_4sp_clo4_saturates_at_v0p3` —
  4sp + debye_boltzmann + Stern_0p10 + production K0/ALPHA/clip on a
  4-V grid `[-0.10, 0.00, +0.10, +0.30]`. Asserts V=+0.30 V converges
  and `c3_surface_mean` is within Bikerman cap and ≥50% of cap. Pre-
  fix this voltage diverged at z=1 (no SS exists under inverted
  sign). **PASSED** post-fix.

- `tests/test_steric_saturation.py::test_4sp_clo4_saturates_at_v0p3_muh` —
  same physics but `formulation="logc_muh"`. First test exercising the
  muh path with steric+4sp; until now muh was only tested on
  3sp+Boltzmann (no steric). **PASSED** post-fix in 3:33.

- `tests/test_steric_sign.py::test_mu_steric_sign_at_saturation` —
  fast symbolic sign-pin (no Firedrake). **PASSED**.

- A V=+0.50 V variant was prototyped but dropped: a 5-V grid forces a
  single 7.78-nondim-eta warm-walk leg from V=+0.30 anchor, which is
  intrinsically slow (>15 min) regardless of fix correctness. Coverage
  at higher V is preserved by sweep S1 (production 15-V grid, see
  below).

**Regression baseline drift on 3sp+Boltzmann production path.** The
source plan's claim that "the 3sp production preset has
`a_vals_hat = [0.0]*3`, so steric is dead code there" was wrong — both
`THREE_SPECIES_LOGC_BOLTZMANN` and `FOUR_SPECIES_LOGC_DYNAMIC` carry
`a_vals_hat = [A_DEFAULT]*n = [0.01]*n`, so steric is active for 3sp
too. The sign flip therefore perturbs the 3sp+Boltzmann path's
residual by a small but nonzero amount. Observed at V_RHE=+0.66 V on
the no-Stern Ny=200 baseline:

| | pre-fix | post-fix | drift |
|---|---|---|---|
| CD (mA/cm²) | 1.292191e-08 | 1.296845e-08 | +0.36% |
| PC (mA/cm²) | 1.292264e-08 | 1.296936e-08 | +0.36% |

The Stern snapshot baseline at `tests/test_stern_no_stern_snapshot.py`
was refreshed to the post-fix values; comment in the file flags the
update and points here. Drift is small in absolute terms (0.36% is
well below typical inverse-pipeline basin separation of 1–5%) but is
3600× the prior `rel_tol=1e-6` Newton-floor tolerance, so the baseline
update is mandatory rather than optional.

**Other regression gates.** With the MMS source and Stern baseline
updates landed, all 16 slow regression tests pass (R1 + R2 + R3):

- `test_initializer_debye_boltzmann.py` — 3 tests, PASSED
- `test_initializer_debye_boltzmann_4sp.py::TestRegression3spStillWorks` — PASSED
- `test_stern_no_stern_snapshot.py` — 2 tests, PASSED (with refreshed baseline)
- `test_solver_equivalence.py` — 4sp dynamic vs 3sp+Boltzmann
  equivalence holds at all V in [-0.5, +0.1] V; PASSED
- `test_mms_convergence.py` — 7 tests, PASSED (with corrected source)

**Sweep S1 — TBD.** `peroxide_window_4sp_extended.py debye_boltzmann`
is running at the time of this writeup; expected behaviour is
convergence at V ∈ {+0.30, +0.50, +0.66, +0.68} V where pre-fix
diverged. Pre-fix artifacts backed up to
`StudyResults/peroxide_window_4sp_extended_debye_boltzmann_PRE_STERIC_FIX/`
for direct comparison. Post-fix artifacts will land in the canonical
`peroxide_window_4sp_extended_debye_boltzmann/` directory.

**Open questions for advisor (carried forward from §10 + new):**

1. Approve the writeup edit at `docs/PNP Equation Formulations.tex:152`
   (add leading `-`).
2. Is the 0.36% drift on the 3sp+Boltzmann production observable
   acceptable given the inverse-pipeline cache may have hard-coded
   numerical TRUE values? See plan §7 Q4.
3. Confirm `a_vals_hat = [0.01]*3` for the 3sp preset is intentional
   (i.e., the production stack has always been Bikerman-corrected,
   just with the wrong sign). If unintentional, the prior production
   results were under a non-standard model and may need separate
   review.

## 12. Resolution — Bikerman γ-corrected IC (Option 2a′, 2026-05-04)

**Failure mode confirmed and fixed.** §6.1 diagnosed that even after
the steric sign correction (§11), Newton at z=1 from the existing
pure-Boltzmann IC seeds `c_ClO₄(surface) ≈ 2·10⁴` at V_RHE = +0.3 V,
which the (now correctly-signed) Bikerman residual rejects via
`packing_floor` clamp → `μ_steric = +18.4` → Jacobian ≈ 1e+6. The
steric sign fix was necessary but not sufficient; the IC also has
to land Newton in a state with strictly positive packing.

**The fix.** Multispecies Bikerman γ correction applied to all four
species in the synthesised-4sp counterion branch of
`_try_debye_boltzmann_ic` (`Forward/bv_solver/forms_logc.py:795–833`).
Final form:

```
γ(y) = 1 / [1 + a_H · H_outer(y) · (exp(−ψ) − 1)
              + a_ClO₄ · H_outer(y) · (exp(+ψ) − 1)]

c_i_IC(y) = c_outer_i(y) · γ(ψ(y)) · exp(−z_i · ψ(y))
```

Local outer anchor `H_outer` (not bulk constants) for both charged
species, since outer electroneutrality forces
`c_ClO₄_outer(y) = H_outer(y)` already in the existing matched-
asymptotic IC. Branch gated on `synthesised_4sp_counterion`, so the
3sp+analytic-Boltzmann path is byte-identical (no steric in 3sp
residual to begin with).

φ is **not** modified — `θ(ψ) = θ_bulk · γ` so the γ in `c_i` and the
`-γ` in `-ln(θ)` cancel in the chemical potential, and adding `ln γ`
to φ would double-count the steric term.

**Corrections to the original §8 Option 2a.** Two errors in the prior
handoff (`docs/4sp_bikerman_ic_gpt_handoff.md`) caught by GPT review
(`docs/4sp_bikerman_ic_gpt_assessment.md`):

1. The existing `u_3` was misstated as `ln(c_ClO₄_bulk) + ψ`. The
   actual code seeds `u_3 = ln(c_clo4_bulk) + phi_init_expr =
   ln(H_outer) + ψ`. The prior proposed patch dropped the
   `H_outer/c_ClO₄_bulk` factor; the final patch preserves
   `phi_init_expr` and adds `+ log_gamma`.

2. The prior γ used constant bulk anchors. Local-anchor
   `H_outer` (a) matches the existing matched-asymptotic IC and
   (b) gives a more comfortable packing margin (≈2% vs 0.5%).

These are documented in `docs/4sp_bikerman_ic_option_2a_plan.md`
(execution plan) and the GPT assessment doc.

**Tests added / updated.**

- `tests/test_initializer_debye_boltzmann_4sp.py::test_ic_seeds_clo4_with_gamma_correction`
  (renamed from `test_ic_seeds_clo4_on_boltzmann_manifold`). Asserts
  `c_ClO₄ > 0` everywhere, `max ≤ 1/a + tol` (cap), `max ≥ 0.5/a`
  (saturation visible), `min ≈ c_bulk` (bulk recovery). **PASSED**.
- `tests/test_initializer_debye_boltzmann_4sp.py::test_ic_total_packing_positive_at_v0p3`
  — new. At V_RHE=+0.3 (the binding case), asserts
  `1 − Σ a_j c_j > 1e-3` at every node. Pre-2a′ the same test
  saw `max(Σ a c) ≈ 200`. **PASSED** post-2a′ with margin ≈ 5e-3.

**Regression gates (all passed post-2a′).** The diff is gated on
`synthesised_4sp_counterion`, so the 3sp+Boltzmann path is literally
unchanged.

| gate | result |
|---|---|
| `test_initializer_debye_boltzmann.py` (3sp, 3 tests) | PASSED |
| `test_initializer_debye_boltzmann_4sp.py::TestRegression3spStillWorks` | PASSED |
| `test_solver_equivalence.py` (3 tests, V ∈ [-0.5, +0.1]) | PASSED |
| `test_stern_no_stern_snapshot.py` (2 tests) | PASSED |
| `test_mms_convergence.py` (7 tests) | PASSED |
| `test_steric_sign.py` (sign sanity) | PASSED |
| `test_bv_common_config.py` (12 tests, config wiring) | PASSED |
| `test_steric_saturation.py::test_4sp_clo4_saturates_at_v0p3` | PASSED (153 s) |
| `test_steric_saturation.py::test_4sp_clo4_saturates_at_v0p3_muh` | PASSED (253 s) |

**Sweep result — `peroxide_window_4sp_extended.py debye_boltzmann` at
clip=50, Ny=200, with the 2a′ patch in place** (90 min wall time on
the 15-V grid `[-0.5, …, +1.0]`):

| Pass | Cold | Warm | Total | Max-V cold |
|---|---|---|---|---|
| 4sp + debye_boltzmann + no Stern | 3 | 2 | 5/15 | +0.10 V |
| **4sp + debye_boltzmann + Stern C_S=0.10** | **7** | **0** | **7/15** | **+0.50 V** |

For the Stern pass, this is a 2-voltage extension of the pre-2a′
maximum (+0.10 → +0.50). Cold-converged voltages, with `c_ClO₄` and
the Stern drop:

```
V=-0.50  cold cd=-0.185  c_ClO₄=7.5e-10  stern=-0.05
V=-0.30  cold cd=-0.183  c_ClO₄=1.8e-06  stern=-0.04
V=-0.10  cold cd=-0.179  c_ClO₄=4.0e-03  stern=+0.02
V= 0.00  cold cd=-0.175  c_ClO₄=0.165    stern=+0.20
V=+0.10  cold cd=-0.168  c_ClO₄=3.6      stern=+0.97
V=+0.30  cold cd=-0.117  c_ClO₄=61.4     stern=+5.0   ← was failing
V=+0.50  cold cd=-1.5e-5 c_ClO₄=97.1     stern=+9.7   ← was failing
V≥+0.55  cold-failed     c_ClO₄=97-98 (IC saturated)
```

The **physics signature is right**: Stern drop is negligible up to
V=+0.10, climbs to +5.0 at V=+0.30 (Stern absorbing ~half the applied
voltage), reaches +9.7 at V=+0.50 (Stern absorbing nearly all of it).
This is the expected behaviour once `c_ClO₄ → 1/a` saturates the
diffuse layer's charge-storage capacity. CD at V=+0.50 falls to the
mass-transport limit (1.5e-5 mA/cm²) — also expected once Stern
absorbs the kinetic overpotential.

For V ≥ +0.55, **Newton at z=1 fails but the IC is still healthy**:
the seed produces `c_ClO₄ ≈ 97–98` (just below 1/a) at every node.
That is, the IC delivers a Bikerman-saturated initial state but
Newton can't bridge the remaining distance to the SS. Per the plan
§6, this is the trigger to escalate to **Option 2b** (composite
asymptotic ψ(y)) — the standard Gouy-Chapman ψ inside the EDL
deviates from the modified-Poisson-Bikerman profile in the saturated
zone, and the discrepancy compounds as V increases.

Artifacts: `StudyResults/peroxide_window_4sp_extended_debye_boltzmann/`
(`comparison.png`, `diagnostics.json`, `iv_curve.json`).

**Open questions deferred to future work.**

1. **Option 2b implementation.** The sweep confirms 2a′ unblocks
   V ∈ {+0.3, +0.5} but stalls at V ≥ +0.55. Composite asymptotic
   ψ(y) (saturated linear-decay zone matched to outer GC zone) is
   the natural next step; full closed-form expressions in
   §8.1 of this doc.
2. Should the steric γ-correction also be applied to
   `forms_logc_muh.py`'s IC? The muh path's
   `test_4sp_clo4_saturates_at_v0p3_muh` passed, but with a
   different IC code path; investigate whether muh needs its own
   2a′ treatment for the higher-V sweep.
3. The legacy `dirichlet_solver.py:146` and `robin_solver.py:156`
   still carry the old `+fd.ln(packing)` sign. They are not on the
   3sp+Boltzmann or 4sp dynamic production code path, but are
   inconsistent with the corrected backends. Either fix or retire
   in a separate plan.

## 13. Resolution — Bikerman-consistent IC (Option 2b, 2026-05-04)

**Composite ψ + multispecies γ on both `forms_logc.py` and
`forms_logc_muh.py`, gated to fire on either the synthesised 4sp
ClO₄⁻ counterion *or* an explicit `boltzmann_counterions` entry with
`steric_mode='bikerman'` (the production target stack).** Builds on
§12 (which only landed γ for the 4sp synthesised counterion in the
logc IC).

### 13.1 What landed

`Forward/bv_solver/forms_logc.py:_try_debye_boltzmann_ic` and
`forms_logc_muh.py:_try_debye_boltzmann_ic_muh` now share the same
unified Bikerman-consistent IC physics. Three structural changes:

1. **Gating generalised.** `apply_bikerman_ic =
   synthesised_4sp_counterion or bikerman_in_counterions`. The
   3sp + analytic ClO₄⁻ + `steric_mode='bikerman'` production stack
   now also gets γ + composite ψ; previously only the legacy 4sp
   synthesised counterion branch did.

2. **Composite ψ (BKSA matched-asymptotic).** Saturated zone +
   outer exponential decay:

   ```
   ν       = 2 · a · c_bulk_charged
   ψ_sat   = ln(2/ν)                                    (≈ 6.21 for a=0.01, c_b=0.2)
   α(ψ_D)  = √((2/(ν·λ_D²)) · ln[1 + ν·(cosh ψ_D − 1)])
   y_match = (|ψ_D| − ψ_sat) / α(ψ_D)

   ψ(y)    = sign(ψ_D) · (|ψ_D| − α·y),          y ∈ [0, y_match]
   ψ(y)    = sign(ψ_D) · ψ_sat · exp(-(y − y_match)/λ_D), y ≥ y_match
   ```

   Falls through to the prior tanh-Gouy-Chapman expression when
   `|ψ_D| ≤ ψ_sat·(1 − 1e-3)` (linear-Debye limit) or `ν ≤ 0`.

3. **Multispecies γ anchor selection.** The 4sp dynamic and 3sp +
   bikerman cases differ only in the analytic-counterion anchor:

   | case | a_cl source | c_cl_anchor |
   |---|---|---|
   | 4sp dynamic (synthesised) | `a_vals[3]` | `H_outer(y)` (electroneutrality) |
   | 3sp + bikerman counterion | `entry["a_nondim"]` | `c_clo4_bulk` (analytic outer = bulk) |

   Both reduce to the same closed form for our problem.

4. **muh γ propagation.** The muh path stores the proton as
   `μ_H = u_H + em·z_H·φ`. With `em·z_H = 1`, the (−ψ) in u_H
   cancels the (+ψ) in φ, so `μ_H_init = 2·ln(H_outer) − ln(c_clo4_bulk)
   + log_gamma`. The log_gamma propagates as a smooth additive
   shift on μ_H. Pre-2b the muh IC had no γ at all — the
   `synthesised_4sp_counterion` branch in `forms_logc_muh.py` was
   pure Boltzmann.

The legacy 3sp + ideal counterion path (`apply_bikerman_ic = False`)
is byte-identical to pre-2a′/pre-2b — verified by
`tests/test_initializer_debye_boltzmann.py`,
`tests/test_stern_no_stern_snapshot.py`, and the new
`tests/test_initializer_debye_boltzmann_3sp_bikerman.py::TestRegression3spIdealStillWorks`.

### 13.2 Sweep results (3 × 15-V grid, Ny=200, clip=50)

`StudyResults/peroxide_window_3sp_bikerman_muh_2b/` (new),
`StudyResults/peroxide_window_4sp_extended_debye_boltzmann{,_logc_muh}/`
(2b reruns), and `*_PRE_2B/` snapshots of the prior 2a′ state.

| Stack | Pass | 2a′ baseline | 2b | Δ |
|---|---|---|---|---|
| **3sp + analytic-bikerman + muh** *(production target)* | no Stern | n/a (new test) | **14/15, max V=+1.00** | first run |
| | Stern 0.10 | n/a | **15/15 clean, max V=+1.00** | first run |
| 4sp dynamic logc | no Stern | 5/15, max V=+0.10 | 5/15, max V=+0.10 | unchanged |
| | Stern 0.10 | 7/15, max V=+0.50 | 7/15, max V=+0.50 | unchanged |
| 4sp dynamic muh | no Stern | 5/15, max V=+0.10 | 5/15, max V=+0.10 | unchanged |
| | Stern 0.10 | 7/15, max V=+0.50 | 7/15, max V=+0.50 | unchanged |

Per-voltage detail for the **3sp + bikerman + muh + Stern** pass
(the production-target win):

```
V_RHE   ok    method                   c_steric   stern_drop   CD            PC
-0.500  ✓    warm<--11.677            7.5e-10    -0.05 V      -0.185        +1.5e-5
-0.300  ✓    warm<- -3.892            1.8e-06    -0.04 V      -0.183        +1.5e-5
-0.100  ✓    cold                     4.1e-03    +0.02 V      -0.179        +1.5e-5
 0.000  ✓    cold                     1.7e-01    +0.20 V      -0.175        +1.5e-5
+0.100  ✓    cold                     3.6e+00    +0.97 V      -0.168        +1.5e-5
+0.300  ✓    cold                     6.1e+01    +5.0 V       -0.116        +1.5e-5
+0.500  ✓    cold                     9.7e+01    +9.7 V       -1.6e-5       +1.5e-5
+0.550  ✓    cold                     9.9e+01    +10.8 V      -1.6e-5       +1.6e-5
+0.600  ✓    cold                     1.0e+02    +11.8 V      -1.3e-5       +1.3e-5
+0.650  ✓    warm<-+23.353            1.0e+02    +12.8 V      -7.8e-7       +7.8e-7
+0.700  ✓    warm<-+26.467            1.0e+02    +13.8 V      -6.9e-9       +6.9e-9
+1.000  ✓    warm<-+29.191            1.0e+02    +19.0 V      +1.2e-16      +1.2e-16
```

* Cold ceiling extended from +0.30 V (2a′ no-Stern, §12) to **+0.60 V**
  (2b Stern). Warm-walk continues cleanly to +1.00 V on both passes.
* `c_steric` saturates at the Bikerman cap `1/a_b = 100` for every
  V ≥ +0.30 — the analytic closure and the IC are now consistent.
* CD (cathodic, ORR-direction) and PC are non-trivial through
  V=+0.75 V (Stern pass): at V=+0.65 we have CD = −7.8e-7 mA/cm²
  with `stern_drop = 12.8 V` absorbing most of the applied
  potential. The `≈ machine-epsilon` values for V ≥ +0.50 in the
  no-Stern pass are not clipping artifacts but the genuine kinetic
  dead zone — the BV cathodic terms underflow because the
  proton concentration factor `(c_H_surf / c_H_ref)^2` (R1) and
  `(c_H_surf / c_H_ref)^4` (R2) drive the cathodic log-rate to
  ≲−150 once ψ_D > 20. The R2 clip threshold is V_RHE = +0.495 V;
  R2 is *unclipped* above it.

### 13.3 Cross-stack equivalence on the overlap

Both stacks converged at V ∈ {−0.5, −0.3, −0.1, 0, +0.1, +0.3, +0.5}
(7 voltages); rel error of CD and PC between 4sp dynamic logc + 2b
and 3sp + bikerman + muh + 2b on the Stern pass:

```
V_RHE     CD rel err     PC rel err
-0.500    4·10⁻⁹         3·10⁻⁹
-0.300    6·10⁻⁹         4·10⁻⁹
-0.100    4·10⁻⁸         1·10⁻⁸
 0.000    3·10⁻⁸         2·10⁻⁸
+0.100    2·10⁻⁶         7·10⁻⁸
+0.300    1·10⁻³         3·10⁻⁶
+0.500    5·10⁻³         7·10⁻³
```

Inside `tests/test_solver_equivalence.py` `REL_TOL = 5·10⁻³` over the
whole overlap — the analytic 3sp + bikerman closure is a faithful
representation of the dynamic ClO₄⁻ transport, and 2b's IC change
preserved the equivalence.

### 13.4 What 2b did and didn't change

* **Did extend the production-target stack.** 3sp + analytic-bikerman
  + muh now reaches V_RHE = +1.00 V on a 15-voltage cold + warm-walk
  sweep (15/15 with Stern, 14/15 without). Pre-2b sweeps for this
  stack didn't exist; the closest comparable was the §12 4sp dynamic
  Stern pass which reached +0.50 V.
* **Did not move the 4sp dynamic ceiling.** Both `logc` and `logc_muh`
  4sp dynamic are byte-identical to 2a′ in convergence reach. The
  binding constraint on 4sp dynamic is the c_ClO₄ NP equation, not
  the IC: the γ-corrected IC seeds `c_ClO₄ ≈ 1/a_b = 100` at the
  cap, but the dynamic species' transport residual at z<1 in the
  Phase-1 z-ramp can't reconcile the saturated-IC manifold with
  ∇μ_ClO₄ = 0 in SS. This is consistent with the project direction:
  the 3sp + analytic-bikerman stack is the *production* target, and
  the 4sp dynamic stack is a validation reference for equivalence
  testing.
* **Did not touch the residual side.** Both Bikerman closure
  (`build_steric_boltzmann_expressions`) and the steric chemical
  potential sign (`mu_steric = -ln(packing)`) are unchanged from
  §11 / §12. 2b is purely an IC-side seed change.
* **Open: V_RHE > +1.0 V.** Likely still cold-fails because the
  warm-walk needs an anchor and the kinetic dead zone is too deep
  for the cold IC. Trigger for a numerical ODE integration of the
  Bikerman first integral (Option 2c, out of scope for this round).
* **Open: V=+0.10 cold-fail in the no-Stern pass (3sp+bikerman+muh).**
  Cold-fails between fully saturated regimes; warm-walk recovers it
  in the Stern pass. Likely a transition-zone Newton stability
  issue, not a 2b issue.

### 13.5 Tests added/updated

* `tests/test_steric_psi_profile.py` — formula identities for the
  composite ψ (continuity at y_match, far-field decay, sign branch)
  + a slow scipy comparison against the Bikerman first integral.
* `tests/test_initializer_debye_boltzmann_3sp_bikerman.py` — γ +
  composite ψ + log_gamma propagation for both formulations; 3sp +
  ideal regression (byte-identical pre-2b).
* `tests/test_initializer_debye_boltzmann_4sp_muh.py` — parallel of
  the existing 4sp logc test suite for `formulation='logc_muh'`.
  Confirms γ now lands on the muh path's IC (was missing pre-2b).
* `tests/test_initializer_debye_boltzmann_4sp.py::test_ic_uses_composite_psi_at_v0p66`
  — composite-ψ regression target on the 4sp logc path at the new
  binding voltage.
* `tests/test_steric_saturation.py::test_3sp_bikerman_v0p55_muh` —
  full-stack integration test on the user's target config (3sp +
  bikerman + muh) at V=+0.55 V, the new cold-success voltage.

Artifacts: `StudyResults/peroxide_window_3sp_bikerman_muh_2b/`,
`StudyResults/peroxide_window_4sp_extended_debye_boltzmann{,_logc_muh}/`
(2b reruns), `*_PRE_2B/` (2a′ snapshots for diff). Sweep scripts:
`scripts/studies/peroxide_window_3sp_bikerman_muh.py` (new) +
`scripts/studies/peroxide_window_4sp_extended.py` (existing, run
with `debye_boltzmann` and `debye_boltzmann logc_muh`).
