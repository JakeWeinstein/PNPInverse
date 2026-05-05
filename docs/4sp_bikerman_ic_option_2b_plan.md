# Option 2b — composite asymptotic ψ(y) for the 4sp Bikerman IC — implementation plan

**Date:** 2026-05-04
**Status:** ready to execute (after Option 2a′ has landed)
**Background:**
* `docs/4sp_drop_boltzmann_investigation.md` §8.1 (original Option 2b sketch)
* `docs/4sp_bikerman_ic_option_2a_plan.md` (executed; the prerequisite)
* `docs/4sp_drop_boltzmann_investigation.md` §12 (sweep result post-2a′:
  cold ceiling V=+0.50, anodic warm-walk still fails at first step
  past anchor)

This plan replaces the standard Gouy-Chapman ψ(y) profile in
`_try_debye_boltzmann_ic` with a **piecewise closed-form** profile
matched to the Bikerman-modified Poisson-Boltzmann first integral.
The change is **gated on `synthesised_4sp_counterion`** so the 3sp +
analytic-Boltzmann path keeps its GC profile. Together with 2a′'s
γ-correction on the species concentrations, the IC becomes internally
consistent with the implemented Bikerman residual: c_i are
γ-corrected on a Bikerman-respecting ψ profile.

## 0. Goal in one sentence

Push the 4sp + debye_boltzmann + Stern cold ceiling from V_RHE=+0.50
to V_RHE=+0.66+ (peroxide window centre) by giving Newton an IC whose
ψ(y) profile thickens the saturated zone as V increases — matching
the Bikerman-modified first integral, not the standard GC closed form.

## 1. Pre-flight

### 1.1 ✅ Option 2a′ is in place

Verified by `git log` and the post-2a′ sweep at
`StudyResults/peroxide_window_4sp_extended_debye_boltzmann/`
showing cold convergence at V=+0.30 and V=+0.50.

### 1.2 ✅ The diagnosed failure mode is IC-side, not warm-walk-side

Per `docs/4sp_drop_boltzmann_investigation.md` §12 and the diagnostic
analysis after the post-2a′ sweep: warm-walk fails at the first
anodic substep past the highest cold anchor in **every** 4sp+Bikerman
run, pre and post 2a′; the pattern is consistent with Newton-stiffness
in the saturated regime, not a code regression. 2b is therefore the
right target: improve the IC profile so cold convergence reaches
V=+0.66 directly, eliminating the need for warm-walk in the peroxide
window.

### 1.3 Locate the ψ-building block in `_try_debye_boltzmann_ic`

`Forward/bv_solver/forms_logc.py:765–784` (line numbers may shift after
the 2a′ patch — confirm before editing). Existing block:

```python
EPS_TANH = 1e-15
T = math.tanh(psi_D / 4.0)
T_clamp = math.copysign(min(abs(T), 1.0 - EPS_TANH), T)

coords = fd.SpatialCoordinate(mesh)
ndim = mesh.geometric_dimension()
y = coords[0] if ndim == 1 else coords[1]

E_expr = fd.exp(-y / fd.Constant(lambda_D))
arg = fd.Constant(T_clamp) * E_expr
arg_safe = fd.min_value(
    fd.max_value(arg, fd.Constant(-1.0 + EPS_TANH)),
    fd.Constant(1.0 - EPS_TANH),
)
psi = fd.Constant(2.0) * fd.ln(
    (fd.Constant(1.0) + arg_safe) / (fd.Constant(1.0) - arg_safe)
)
```

This builds standard GC ψ(y). Strategy: keep this block as `psi_gc`
(used by the 3sp branch and as a fallback), and additionally compute
`psi_bikerman` in the 4sp branch.

## 2. Math

### 2.1 The Bikerman first integral

For a binary symmetric electrolyte (z=±1) at equilibrium under
Bikerman/Kornyshev MPB,

```
(dψ/dy)² = (2 / (ν·λ_D²)) · ln[1 + ν·(cosh ψ − 1)]    where ν = 2·a·c_bulk.
```

(Bazant-Kilic-Storey-Ajdari 2009 eq (32); Borukhov-Andelman-Orland
1997 §3.) Compare standard GC:

```
(dψ/dy)²_GC = (2/λ_D²) · (cosh ψ − 1).
```

For ψ ≪ 1 the two agree (linearised PB). For ψ ≫ ψ_sat the Bikerman
form grows as `ln(cosh ψ) ~ |ψ|` while GC grows as `cosh ψ ~ exp |ψ|/2`.
That is: in the saturated regime |dψ/dy| grows ~ √ψ, not exponentially
— so ψ has to extend much further into the bulk to absorb a given
applied voltage.

### 2.2 The saturation crossover

Define

```
ψ_sat = ln(2/ν) = ln(1/(a·c_bulk_charged)).
```

(For our setup `a = 0.01`, `c_bulk_charged = 0.2`, so
`ν = 0.004`, `ψ_sat ≈ 6.21` nondim, which equals `0.16 V` in
dimensional form via V_T ≈ 0.0257 V.)

For `|ψ| > ψ_sat` the Bikerman ν·(cosh ψ − 1) ≫ 1 and ν·cosh ψ ≈
ν·exp|ψ|/2 dominates; (dψ/dy)² ≈ (2/(ν·λ_D²))·(|ψ| − ln 2 + ln ν) =
(2/(ν·λ_D²))·(|ψ| − ψ_sat). For `|ψ| < ψ_sat`, ν·(cosh ψ − 1) ≪ 1
and the log linearises to ν·(cosh ψ − 1) ≈ ν·(ψ²/2), recovering the
Debye exponential decay.

### 2.3 The composite asymptotic profile

Two zones, matched at the y-coordinate where ψ first reaches ψ_sat:

```
[zone 1, saturated]   y ∈ [0, y_match]:   ψ(y) = ψ_D − α(ψ_D)·y · sgn(ψ_D)
                       slope: α(ψ_D) = sqrt((2/(ν·λ_D²)) · ln[1 + ν·(cosh ψ_D − 1)])

[zone 2, outer]       y ≥ y_match:        ψ(y) = sgn(ψ_D)·ψ_sat·exp(-(y - y_match)/λ_D)

continuity:                                y_match = (|ψ_D| − ψ_sat) / α(|ψ_D|)
```

Sign convention: `sgn(ψ_D)` carries the cathodic-vs-anodic distinction.
The slope α and ψ_sat are computed from |ψ_D|; the final ψ is then
sign-flipped if cathodic. (`cosh` is even in ψ, so the saturation
physics is symmetric.)

The match is C⁰ at `y_match` (zones agree on ψ value) but C¹ has a
small kink (ψ' jumps from −α to −ψ_sat/λ_D). For typical Bikerman
parameters, `α ≈ ψ_sat/λ_D · sqrt(2·(|ψ_D|/ψ_sat − 1)/ν·...)` — the
kink is small (~10–20%) and FE-tractable.

### 2.4 Edge cases

**Low-V case `|ψ_D| ≤ ψ_sat`.** No saturated zone. Composite degenerates
to zone 2 only:

```
ψ(y) = sgn(ψ_D) · ψ_D · exp(-y/λ_D)        if |ψ_D| ≤ ψ_sat
```

This is the linear-Debye limit; the standard GC formula reduces to
the same thing for small ψ_D. So in the low-V regime composite ≈ GC,
no surprise.

For the threshold check, use a tiny safety margin (`ψ_sat * (1 - eps)`)
to avoid numerical issues when ψ_D ≈ ψ_sat.

**Zero or near-zero ψ_D case.** `α(0) = 0` would divide by zero in
`y_match`. Guard with `if abs(ψ_D) < 1e-6: ψ ≡ 0`. Falls into the
trivial linear case which the existing GC handles fine; for 2b just
defer to GC in this regime.

**Boundary: y > y_match where ψ → 0.** The exponential decay tail
naturally goes to zero, no clamp needed (unlike GC's atanh saturation
at ψ_D > 30).

**ψ_sat boundary: ν·(cosh ψ − 1) ≈ 1 transition.** The composite
formula uses the asymptotic limits of the first integral; near the
crossover the actual integrand is intermediate. The C⁰-matched
composite is correct at ψ=ψ_D and ψ=0 limits, and a few-percent
approximation in the crossover region. FE-acceptable.

### 2.5 Verifying against numerical integration

Sanity-check 2b's ψ(y) profile against scipy.integrate.solve_ivp of
the full ODE:

```
dψ/dy = -sgn(ψ_D) · sqrt((2/(ν·λ_D²)) · ln[1 + ν·(cosh ψ − 1)])
ψ(0) = ψ_D
```

at a few representative voltages (V=+0.30, +0.50, +0.66, +1.00). Plot
or assert pointwise agreement within 5% relative error. This is a
**dev-time validation script**, not a permanent test (numerical ODE
adds scipy dependency overhead in the test suite).

## 3. Test-first sequencing (TDD per project rules)

### 3.1 Tests to add (new file `tests/test_steric_psi_profile.py`)

**P_2b_1 — composite ψ at saturation (fast, pure-Python).**

```python
def test_composite_psi_zone1_match():
    """At ψ_D = 11.5 (V=+0.3), the saturated zone extends from y=0
    to y_match; ψ at y_match equals ψ_sat exactly.  Tests the
    closed-form match condition."""
    a, c_bulk, lambda_D = 0.01, 0.2, 0.1  # representative
    psi_D = 11.5
    psi_sat = math.log(1.0 / (a * c_bulk))
    nu = 2.0 * a * c_bulk
    alpha_D = math.sqrt((2.0/(nu*lambda_D**2)) * math.log(1.0 + nu*(math.cosh(psi_D)-1.0)))
    y_match = (psi_D - psi_sat) / alpha_D
    psi_at_match = psi_D - alpha_D * y_match
    assert abs(psi_at_match - psi_sat) < 1e-10
```

**P_2b_2 — composite ψ at the bulk recovers zero (fast, pure-Python).**

Decay tail at y = 5·λ_D should give `ψ < 0.01·ψ_sat`.

**P_2b_3 — composite vs scipy ODE agreement (slow, scipy).**

```python
def test_composite_psi_matches_ode_within_5pct():
    """Numerically integrate the Bikerman first integral ODE and
    compare to the composite at the same y nodes. Pointwise relative
    error should be < 5% across V_RHE ∈ {+0.3, +0.5, +0.66, +1.0}."""
    # uses scipy.integrate.solve_ivp
```

**P_2b_4 — IC saturation visible at higher V (slow, Firedrake).**

Same shape as the existing 2a′ test, but at V=+0.66 (peroxide window
centre, where 2a′ alone left the IC's saturated zone too thin).
Asserts `c_ClO4` profile shows wider saturated band than under 2a′.

```python
def test_ic_extends_saturation_zone_at_v0p66():
    """At V_RHE=+0.66 V the 2b composite ψ produces a thicker saturated
    zone (where c_ClO4 ~ 1/a) than 2a′'s GC ψ.  The number of mesh
    nodes with c_ClO4 > 0.95/a should be >= some minimum thickness
    (e.g., > 5% of mesh height for Ny=80)."""
```

**P_2b_5 — bulk recovery still works.**

Same as 2a′'s `test_ic_bulk_recovery`: `c_i(y=1) ≈ c_bulk_i`. Should
trivially pass since both zones decay to ψ→0 at y_match → ∞.

### 3.2 Tests to update

**`test_initializer_debye_boltzmann_4sp.py::test_ic_seeds_clo4_with_gamma_correction`**:
this test already passes with 2a′. Should continue to pass with 2b
(same upper/lower bounds, possibly tighter). Run as regression.

**`test_initializer_debye_boltzmann_4sp.py::test_ic_total_packing_positive_at_v0p3`**:
margin should *improve* under 2b (saturated zone extends further into
mesh, packing margin similar at electrode but cleaner). Tighten the
margin assertion to `> 1e-2` if 2b genuinely improves it; otherwise
keep at `> 1e-3`.

**`test_initializer_debye_boltzmann_4sp.py::test_ic_depletes_h_at_electrode`**:
H+ depletion threshold (`u_h.min() < u_h_bulk - 15`) should hold
trivially with 2b — saturation is wider but the electrode value is
similar.

### 3.3 Tests to leave alone

**`test_initializer_debye_boltzmann.py`** — 3sp+Boltzmann path. Gated
out by `synthesised_4sp_counterion`. Byte-identical.

**`test_solver_equivalence.py`** — uses `linear_phi`, not
`debye_boltzmann`. Unaffected.

**`test_steric_sign.py`, `test_steric_saturation.py`** — sign sanity
and full-stack at V=+0.3. 2b should make V=+0.3 converge faster (less
Newton work) but the existing pass thresholds are loose. Run as
regression; tighten thresholds only if 2b clearly improves on them.

### 3.4 Regression gates (must continue to pass)

Same as 2a′'s gates §3.4 — listed for completeness:
* `test_initializer_debye_boltzmann.py` (3 tests, slow)
* `test_initializer_debye_boltzmann_4sp.py::TestRegression3spStillWorks`
* `test_solver_equivalence.py` (3 tests, slow)
* `test_stern_no_stern_snapshot.py` (2 tests, slow)
* `test_mms_convergence.py` (7 tests, slow)
* `test_steric_sign.py` (1 test, fast)
* `test_bv_common_config.py` (12 tests, fast)
* `test_steric_saturation.py::test_4sp_clo4_saturates_at_v0p3` (slow, integration)
* `test_steric_saturation.py::test_4sp_clo4_saturates_at_v0p3_muh` (slow, integration)

## 4. Code change

### 4.1 The diff (in `_try_debye_boltzmann_ic`)

Replace the existing GC ψ block with a conditional that builds either
GC or composite Bikerman ψ depending on the branch.

```python
# ----- ψ(y) profile ------------------------------------------------
# Default: standard Gouy-Chapman.  For the synthesised 4sp counterion
# branch, switch to the composite asymptotic profile that respects the
# Bikerman-modified Poisson-Boltzmann first integral
# (Bazant-Kilic-Storey-Ajdari 2009 eq (32)).  See
# docs/4sp_bikerman_ic_option_2b_plan.md.
EPS_TANH = 1e-15
T = math.tanh(psi_D / 4.0)
T_clamp = math.copysign(min(abs(T), 1.0 - EPS_TANH), T)

coords = fd.SpatialCoordinate(mesh)
ndim = mesh.geometric_dimension()
y = coords[0] if ndim == 1 else coords[1]

E_expr = fd.exp(-y / fd.Constant(lambda_D))
arg = fd.Constant(T_clamp) * E_expr
arg_safe = fd.min_value(
    fd.max_value(arg, fd.Constant(-1.0 + EPS_TANH)),
    fd.Constant(1.0 - EPS_TANH),
)
psi_gc = fd.Constant(2.0) * fd.ln(
    (fd.Constant(1.0) + arg_safe) / (fd.Constant(1.0) - arg_safe)
)

if synthesised_4sp_counterion:
    # Composite asymptotic ψ(y): saturated linear-decay zone matched
    # to outer exponential (Debye-like) zone at the saturation
    # crossover ψ_sat.
    a_h_for_psi   = float(a_vals_full[2])
    a_cl_for_psi  = float(a_vals_full[3])
    # Charged-pair effective ν: assume H+ and ClO4- balance at bulk
    # (electroneutrality), so we pick c_clo4_bulk as the reference.
    nu = 2.0 * a_cl_for_psi * c_clo4_bulk
    psi_d_abs = abs(psi_D)
    if nu <= 0.0 or psi_d_abs < 1e-6:
        # Degenerate: no Bikerman contribution, fall back to GC.
        psi = psi_gc
    else:
        psi_sat_val = math.log(2.0 / nu)
        if psi_d_abs <= psi_sat_val * (1.0 - 1e-3):
            # Low-V regime: no saturated zone; pure exponential decay.
            # Use exp(-y/λ_D) anchored at ψ_D rather than GC's atanh
            # form (they agree to leading order; this is just simpler).
            psi = fd.Constant(psi_D) * fd.exp(-y / fd.Constant(lambda_D))
        else:
            # General case: composite (saturated || outer).
            arg_cosh = math.cosh(psi_d_abs)
            alpha_d = math.sqrt(
                (2.0 / (nu * lambda_D**2)) * math.log(
                    1.0 + nu * (arg_cosh - 1.0)
                )
            )
            y_match_val = (psi_d_abs - psi_sat_val) / alpha_d
            sign_psi_D = 1.0 if psi_D > 0.0 else -1.0
            # Zone 1: ψ_D − α·y, sign-adjusted
            psi_zone1 = fd.Constant(sign_psi_D) * (
                fd.Constant(psi_d_abs) - fd.Constant(alpha_d) * y
            )
            # Zone 2: ψ_sat·exp(-(y - y_match)/λ_D), sign-adjusted
            psi_zone2 = fd.Constant(sign_psi_D * psi_sat_val) * fd.exp(
                -(y - fd.Constant(y_match_val)) / fd.Constant(lambda_D)
            )
            psi = fd.conditional(
                y < fd.Constant(y_match_val), psi_zone1, psi_zone2
            )
else:
    psi = psi_gc
```

Then the `O_outer`, `P_outer`, `H_outer` blocks (which use psi
implicitly via H_outer linear envelope only) and the seeding block
(which uses `psi` and `phi_init_expr = ln(H_outer/c_clo4_bulk) + psi`)
remain **unchanged** — they read `psi` regardless of which form was
built.

### 4.2 Things explicitly NOT in scope

* No changes to `O_outer`, `P_outer`, `H_outer` envelope construction.
* No changes to the multispecies γ formula (already correct in 2a′).
* No changes to phi_init_expr (already correct).
* No changes to the seeding block (4sp branch already γ-corrected).
* No changes to the 3sp + analytic-Boltzmann path (gated out via
  `synthesised_4sp_counterion`).
* No changes to other backends (`forms_logc_muh.py`,
  `dirichlet_solver.py`, `robin_solver.py`).
* No changes to the Picard outer loop (it uses `psi_D` scalar, not
  `psi(y)` field).

### 4.3 Risk register

* **`fd.conditional` performance.** Composite uses
  `fd.conditional(y < y_match, zone1, zone2)`. Firedrake compiles
  this into a runtime branch in the assembly kernel; it's OK but
  slightly slower than a single expression. For an IC-only
  computation this is negligible.
* **`y_match` outside [0, 1].** If the saturated zone is bigger than
  the entire mesh height (only at extreme ψ_D, e.g., ψ_D = 100 with
  ψ_sat = 6, y_match = 100/α which could exceed 1 if α < 100), the
  outer zone never fires inside the domain. The composite then
  reduces to pure linear decay throughout, which is correct (entire
  mesh is in saturation). The conditional handles this naturally.
* **C¹ kink at y_match in FE.** First-order CG elements can represent
  C⁰ kinks exactly at element boundaries but not within elements.
  For Ny=200 graded mesh with element thickness ~λ_D/few near the
  electrode, the kink is well-resolved. If Ny < 50 or the kink falls
  on a coarse element, the FE projection smears it slightly — not
  Newton-blocking.
* **`a_h` ≠ `a_cl`** (asymmetric Bikerman). Plan above assumes
  symmetric `a` for the binary pair. If they differ, ν must be
  computed as `a_h·c_h_bulk + a_cl·c_clo4_bulk` (sum of partial
  packing fractions). For our production setup `A_DEFAULT=0.01`
  applies uniformly, so this is moot — but document the assumption.
* **ψ_D and lambda_D computed as floats outside Firedrake.** They're
  already locals in `_try_debye_boltzmann_ic` (Picard outer-loop
  outputs); same scope. No new symbolic differentiation issue.

## 5. Smoke + sweep

### 5.1 Smoke at V=+0.55 (the new binding case)

Add a new test `test_steric_saturation.py::test_4sp_clo4_saturates_at_v0p55`,
mirroring the V=+0.3 production-stack test but at V_RHE=+0.55 V on a
5-V grid `[-0.10, 0.00, +0.10, +0.30, +0.55]`. Pass criteria:

1. V=+0.55 converges (cold or warm — either is fine since the cold
   ceiling moving up alone is the win).
2. `c3_surface_mean ≤ 1/a + tol` (still respects steric cap).
3. `c3_surface_mean ≥ 0.5/a` (saturation visible).

If V=+0.55 converges, mark 2b as smoke-passing. If not, **stop and
diagnose** before running the full sweep.

### 5.2 Sweep at clip=50 with 2b

Re-run `peroxide_window_4sp_extended.py debye_boltzmann` (with the
clip=50 default landed in 2a′). Compare against the 2a′ artifact at
`StudyResults/peroxide_window_4sp_extended_debye_boltzmann/`. Expected
behaviour:

| V_RHE | 2a′ (post-fix) | 2b (target) |
|---|---|---|
| ≤ +0.10 V | converges 5/5 | converges 5/5 (same; ψ_D < ψ_sat regime) |
| +0.30 V | converges cold | converges cold (faster Newton) |
| +0.50 V | converges cold | converges cold |
| **+0.55 V** | cold-failed | **converges cold (target)** |
| +0.60 V | cold-failed | converges cold (target) |
| +0.66 V (peroxide centre) | cold-failed | converges cold (stretch target) |
| +0.70 V | cold-failed | best effort; maybe needs Stern_0p20 or 2c |
| +1.00 V | cold-failed | best effort |

Acceptance: at least V ∈ {+0.55, +0.60, +0.66} converge cold. If
V ≥ +0.70 still fails, that's the trigger to escalate to **Option
2c** (numerical ODE for ψ(y)) or to investigate further — out of
scope here.

## 6. Acceptance criteria for merge

All of:

1. Regression gates (§3.4) pass.
2. New tests P_2b_1 through P_2b_5 pass.
3. Smoke solve at V=+0.55 (§5.1) passes.
4. Sweep at V ∈ {+0.55, +0.60, +0.66} converges cold (§5.2).
5. `docs/4sp_drop_boltzmann_investigation.md` gets a §13 "Resolution
   — Option 2b composite ψ" appended quoting the new sweep table.

## 7. Rollback

The diff is local to one block (~30 lines) inside
`_try_debye_boltzmann_ic`. To revert:

1. Remove the `if synthesised_4sp_counterion: ... composite ... else:
   psi = psi_gc` conditional, restore `psi = psi_gc` unconditionally.
2. New tests P_2b_1 to P_2b_5 will fail under revert; that's the
   correct signal that 2b is no longer in effect.
3. Re-run the §5.2 sweep at clip=50 to confirm cold ceiling drops
   back to V=+0.50 (the 2a′-only state).

## 8. Sequencing — concrete TODO order

```
[ ] 1. Verify forms_logc.py:_try_debye_boltzmann_ic line numbers
       (post-2a′, may have shifted slightly — look for the GC ψ block).
[ ] 2. Write tests/test_steric_psi_profile.py with P_2b_1, P_2b_2
       (fast pure-Python).  Run: should pass without code change since
       they only test math identities.
[ ] 3. Write P_2b_3 (slow, scipy).  Run: should pass without code
       change (purely numerical sanity).
[ ] 4. Write P_2b_4 (slow, Firedrake).  Run on current code: should
       FAIL (2a′-only IC has thin saturated zone at V=+0.66).  This
       is the regression-target test.
[ ] 5. Write P_2b_5 (slow, Firedrake).  Run: should already pass on
       2a′ code (bulk recovery is preserved).
[ ] 6. Apply the §4.1 diff to forms_logc.py.
[ ] 7. Run tests from steps 2–5.  All should pass now.  P_2b_4 in
       particular validates the saturated zone has thickened.
[ ] 8. Run regression gates (§3.4).  All should pass.
[ ] 9. Add tests/test_steric_saturation.py::test_4sp_clo4_saturates_at_v0p55
       (§5.1).  Run: should pass under 2b.
[ ] 10. Re-run sweep `peroxide_window_4sp_extended.py debye_boltzmann`
        (§5.2).  Compare to 2a′ artifact.
[ ] 11. Append "§13 Resolution — Option 2b" to investigation doc.
[ ] 12. Commit (3 atomic commits per project conventions):
          test: composite asymptotic ψ tests for 2b
          feat: option 2b composite ψ in 4sp debye_boltzmann IC
          docs: §13 resolution — 2b extends cold ceiling to +0.66
```

## 9. Open questions deferred

These are NOT blockers for this plan; record answers in §13 of the
investigation doc once 2b is in.

1. **Does V ≥ +0.70 converge under 2b alone, or is Option 2c
   (numerical ODE) needed?** Empirical — the §5.2 sweep answers it.
2. **Is the warm-walk failure at the highest cold anchor still
   observed under 2b?** If 2b's cold ceiling reaches V=+0.66 directly,
   the warm-walk failure becomes irrelevant for the peroxide window;
   document either way.
3. **Should the same composite ψ profile be added to the 3sp +
   analytic-Boltzmann path now that the user added Bikerman steric
   to the analytic counterion?** That stack now has Bikerman physics
   on the residual side; its IC could benefit from a Bikerman ψ too.
   Scope decision for a follow-up plan.
4. **Asymmetric `a` (different a_H vs a_ClO4) — does the binary-ν
   formula still hold?** The plan above assumes uniform `a`. For
   future work with asymmetric a, ν should be the sum of charged
   species' partial packing fractions — but verify the resulting
   first integral still has the closed-form composite structure.

## 10. Pointers

* Code (target of the change):
  `Forward/bv_solver/forms_logc.py:_try_debye_boltzmann_ic`,
  ψ-building block (was lines 765–784 pre-2a′; verify exact lines
  post-2a′).
* Solver param indexing: `solver_params[6] = a_vals` (same as 2a′).
* Bikerman first integral: Bazant-Kilic-Storey-Ajdari (2009)
  *Adv. Colloid Interface Sci.* 152, 48 — eq (32). arXiv:0903.4790.
  Borukhov-Andelman-Orland (1997) *PRL* 79, 435 — §3.
* Existing 4sp IC tests (regression):
  `tests/test_initializer_debye_boltzmann_4sp.py`.
* New test file: `tests/test_steric_psi_profile.py` (P_2b_1..5).
* Sweep script:
  `scripts/studies/peroxide_window_4sp_extended.py debye_boltzmann`.
* Reference artifacts:
  - 2a′ post-fix:
    `StudyResults/peroxide_window_4sp_extended_debye_boltzmann/`
  - Pre-steric-sign-fix (for full historical comparison):
    `StudyResults/peroxide_window_4sp_extended_debye_boltzmann_PRE_STERIC_FIX/`
* Companion docs:
  - `docs/4sp_drop_boltzmann_investigation.md` — failure-mode history
    and §11/§12 resolutions.
  - `docs/4sp_bikerman_ic_option_2a_plan.md` — the prerequisite.
  - `docs/4sp_bikerman_ic_gpt_assessment.md` — original GPT review;
    flagged the warm-walk-vs-cold-ceiling distinction.
