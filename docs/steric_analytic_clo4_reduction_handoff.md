# Steric Analytic ClO4 Reduction Handoff

**Date:** 2026-05-04  
**For:** Claude Code implementation  
**Goal:** keep the convergence benefit of analytic inert `ClO4-`, but replace
the unbounded pure-Boltzmann closure with a steric-aware algebraic closure.

## Short answer

Yes: the analytic `ClO4-` reduction can include Bikerman sterics. For the
steady-state inert-counterion problem, this is physically equivalent to
solving dynamic `ClO4-` and eliminating it algebraically, provided the same
steric model is used in both places.

The current closure is:

```text
c_- = c_b exp(phi)        # z = -1
```

The steric-corrected closure should be:

```text
c_-(x) = c_b exp(phi) * (1 - A(x)) / (theta_b + a_- c_b exp(phi))
```

where:

```text
c_-      = ClO4 concentration
c_b      = bulk ClO4 concentration
a_-      = ClO4 steric coefficient
A(x)     = sum_dynamic a_j c_j(x)      # O2, H2O2, H+
theta_b  = 1 - sum_dynamic a_j c_j_bulk - a_- c_b
```

At high anodic `phi`, this saturates to:

```text
c_- -> (1 - A) / a_-
```

so the analytic counterion fills only the free volume left by the dynamic
species. This is the key improvement over a hard `min(c_b exp(phi), 1/a)`.

## Derivation

Use the sign-corrected Bikerman chemical potential:

```text
mu_i = ln(c_i) + z_i phi - ln(theta)
theta = 1 - sum_j a_j c_j
```

For inert `ClO4-`, `z=-1`. At steady state with no `ClO4-` source/sink and
zero normal flux at non-bulk boundaries:

```text
grad(mu_-) = 0
```

So:

```text
ln(c_-) - phi - ln(theta) = ln(c_b) - ln(theta_b)
```

or:

```text
c_- exp(-phi) / theta = c_b / theta_b
```

Let:

```text
A = sum_dynamic a_j c_j
theta = 1 - A - a_- c_-
B = c_b exp(phi) / theta_b
```

Then:

```text
c_- = B theta
    = B(1 - A - a_- c_-)

c_-(1 + a_- B) = B(1 - A)

c_- = B(1 - A) / (1 + a_- B)
    = c_b exp(phi) * (1 - A) / (theta_b + a_- c_b exp(phi))
```

This is the reduced steady-state solution of the dynamic `ClO4-`
Nernst-Planck equation under the assumptions above.

## Physical equivalence to dynamic 4sp

This closure is physically the same as simulating `ClO4-` dynamically when
all of the following hold:

```text
1. steady state
2. ClO4- is inert: no BV reaction, source, sink, adsorption, or imposed flux
3. bulk boundary fixes c_ClO4 = c_b and phi = 0
4. same steric chemical potential is used:
   mu_steric = -ln(theta)
5. same total packing theta includes dynamic species plus analytic ClO4-
```

It is not equivalent for transient double-layer charging, nonzero `ClO4-`
flux, ion-specific adsorption, finite closest-approach physics, or any model
where `ClO4-` does more than remain an inert supporting electrolyte.

For the peroxide-window steady-state ORR solves, this is likely the best
physical reduction: retain the analytic inert counterion, but make it
steric-aware.

## Required implementation change

Do not only modify `Forward/bv_solver/boltzmann.py` as a capped Poisson
source. The analytic `ClO4-` must also enter the dynamic species' steric
packing. Otherwise the Poisson source and NP steric term are solving different
models.

The residual should use:

```text
A_dynamic = sum_i a_i c_i                 # dynamic species only
c_boltz_steric = c_b exp(phi) * (1 - A_dynamic) /
                 (theta_b + a_b c_b exp(phi))
theta_total = 1 - A_dynamic - a_b c_boltz_steric
mu_steric = -ln(theta_total)
```

Then Poisson should include the same `c_boltz_steric`:

```text
F_poisson -= charge_rhs * z_b * c_boltz_steric * w * dx
```

For `ClO4-`, `z_b = -1`, so this has the same sign convention as the current
pure-Boltzmann source.

## Suggested code shape

Current code separates:

- dynamic steric packing in `Forward/bv_solver/forms_logc.py`
- analytic pure-Boltzmann Poisson source in `Forward/bv_solver/boltzmann.py`

The steric analytic closure needs shared access to both:

- dynamic concentrations `ci`
- dynamic steric coefficients `a_vals`
- analytic counterion config
- `phi`
- Poisson test function `w`

So the cleanest implementation is likely one of these:

1. Extend `add_boltzmann_counterion_residual(...)` to optionally return the
   analytic counterion concentration expression before the dynamic steric
   term is built. This may require moving the call earlier in
   `forms_logc.py`.
2. Add a new helper, for example
   `build_steric_boltzmann_counterions(ctx, params, ci, a_vals, phi)`, that
   returns:

   ```text
   analytic_charge_density = sum_k z_k c_k_steric
   analytic_packing        = sum_k a_k c_k_steric
   metadata / expressions for diagnostics
   ```

   Then `forms_logc.py` can include `analytic_packing` in `theta_total`, and
   `boltzmann.py` or the forms module can include `analytic_charge_density`
   in Poisson.

Option 2 is easier to reason about because total packing and Poisson source
are built from the same expression.

## Numerically stable expression

Do not evaluate an unbounded `exp(phi)` without the existing clamp policy.
Start with the same `phi_clamp` convention used by the current analytic
Boltzmann source:

```python
phi_clamped = fd.min_value(
    fd.max_value(phi, fd.Constant(-phi_clamp_val)),
    fd.Constant(phi_clamp_val),
)
q = fd.exp(phi_clamped)  # for z=-1
```

Then:

```python
A_dyn = sum(a_dyn[j] * ci[j] for j in range(n))
free_dyn = fd.max_value(fd.Constant(1.0) - A_dyn, fd.Constant(packing_floor))
theta_b = 1.0 - sum(a_dyn[j] * c0[j] for j in range(n)) - a_b * c_b

c_boltz = (
    fd.Constant(c_b) * q * free_dyn
    / (fd.Constant(theta_b) + fd.Constant(a_b * c_b) * q)
)
theta_total = fd.max_value(
    fd.Constant(1.0) - A_dyn - fd.Constant(a_b) * c_boltz,
    fd.Constant(packing_floor),
)
mu_steric = -fd.ln(theta_total)
```

Notes:

- `theta_b` must be positive. Validate this at config parse/build time.
- `free_dyn` clamping is a numerical guard, not a physical fix. If
  `1 - A_dyn <= 0`, the dynamic species alone are overpacked.
- For multiple analytic counterions, the algebra becomes coupled. This
  handoff only targets the current single inert `ClO4-`.

## Config/API suggestion

Do not change existing `boltzmann_counterions` behavior silently. Add an
explicit mode so old tests remain meaningful:

```python
bv_bc = {
    "boltzmann_counterions": [
        {
            "z": -1,
            "c_bulk_nondim": C_CLO4_HAT,
            "phi_clamp": 50.0,
            "steric_mode": "bikerman",   # new; default "ideal"
            "a_nondim": A_DEFAULT,       # optional; default from species/config
        }
    ]
}
```

Alternative names:

```text
steric_boltzmann = True
counterion_model = "ideal_boltzmann" | "bikerman"
```

The important point is to keep the current ideal analytic Boltzmann path
available for regression comparisons.

## Tests

Add focused tests before running peroxide sweeps.

### Algebra/unit tests

1. **Bulk recovery**

   At `phi=0` and dynamic species at bulk:

   ```text
   c_boltz = c_b
   theta_total = theta_b
   ```

2. **Dilute limit**

   For `a_b -> 0` and `A_dynamic -> 0`:

   ```text
   c_boltz -> c_b exp(phi)
   ```

3. **High-voltage saturation**

   For large positive `phi`:

   ```text
   c_boltz -> (1 - A_dynamic) / a_b
   theta_total -> 0 from above
   ```

   With finite `phi_clamp`, check it is below that limit and monotone.

4. **Packing positivity**

   For representative dynamic concentrations:

   ```text
   1 - A_dynamic - a_b c_boltz > 0
   ```

### Solver/regression tests

1. Existing 3sp+ideal-Boltzmann tests remain unchanged when
   `steric_mode` is absent or `"ideal"`.
2. New 3sp+steric-Boltzmann smoke solve at `V_RHE = +0.3`.
3. Compare 3sp+steric-Boltzmann against 4sp dynamic on the low-voltage overlap
   where 4sp already converges. Observables and surface `ClO4-` should agree
   within the existing equivalence tolerance or a documented tolerance.
4. Check high-voltage surface `ClO4-` stays below the total-packing limit, not
   merely below `1/a`.

## Expected impact

This should preserve the main numerical benefit of the current 3sp+analytic
Boltzmann model: no dynamic `ClO4-` unknown that Newton must transport through
the Debye layer.

But it removes the high-anodic nonphysicality where pure Boltzmann gives
`c_ClO4 ~ 10^16` while the finite-size model allows only the remaining free
volume. In other words, it should be closer to the intended steady-state 4sp
Bikerman physics than either:

```text
3sp + ideal unbounded Boltzmann
4sp dynamic with difficult ClO4 transport basin
```

## Caveats

- This assumes `forms_logc.py` uses the sign-corrected steric term:

  ```text
  mu_steric = -ln(theta)
  ```

- Other backends may still have the old sign. Do not claim this is global
  until those are audited.
- This is a steady-state/inert-ion reduction. It should not be used to study
  transient supporting-electrolyte charging.
- If Stern/Frumkin physics is enabled, this closure remains valid for the
  diffuse-layer solution potential `phi`; Stern changes the boundary relation
  between metal voltage and solution-side `phi`, not the algebraic
  zero-flux relation itself.

## Bottom line

Implement a new explicit `steric_mode="bikerman"` analytic counterion path.
Use the algebraic `ClO4-` closure in both:

```text
1. Poisson charge density
2. total steric packing seen by dynamic species
```

If both use the same expression, the reduced 3sp model should represent the
same steady-state inert-`ClO4-` physics as the full 4sp dynamic Bikerman model,
without carrying the difficult `ClO4-` NP unknown.


## Resolution (2026-05-04)

Implemented per the recommendation. Closure construction is inlined in
both `Forward/bv_solver/forms_logc.py` and
`Forward/bv_solver/forms_logc_muh.py` (between the `eta_clipped` block
and the steric residual). The shared symbolic helper
`build_steric_boltzmann_expressions` lives in
`Forward/bv_solver/boltzmann.py`.

**Config opt-in:** the existing `boltzmann_counterions[j]` entry gains
two optional fields:

```python
{
    "z": -1,
    "c_bulk_nondim": C_CLO4_HAT,
    "phi_clamp": 50.0,
    "steric_mode": "ideal" | "bikerman",   # default "ideal"
    "a_nondim": 0.01,                       # required when bikerman
}
```

A drop-in steric-aware preset is published as
`scripts/_bv_common.DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC` next to
the existing `DEFAULT_CLO4_BOLTZMANN_COUNTERION` (which is byte-
identical to the pre-change shape, so all existing call sites are
unchanged).

**Z-ramp consistency:** both Poisson source and packing contribution
are multiplied by the shared `ctx['boltzmann_z_scale']` Function so
`grid_per_voltage._set_z_factor` ramps both alongside dynamic-species
`z_consts`. Both ideal- and bikerman-mode counterions on the same ctx
share the single z_scale Function (the legacy path was patched to
reuse `ctx['boltzmann_z_scale']` if already present).

**Diagnostics:** `Forward/bv_solver/diagnostics.py::collect_diagnostics`
and `check_steric_saturation` now branch on `steric_mode`; for bikerman
entries the per-counterion `c_counterion{j}_surface_mean` is evaluated
via the closure (using `phi_surf` and per-species `c{i}_surface_mean`
from `surface_field_means`), and an extra `c_counterion{j}_steric_mode`
field is published. The `surface_counterion_within_steric` flag is
therefore meaningful for both modes.

**MMS:** `scripts/verification/mms_bv_3sp_logc_boltzmann.py` accepts
an optional `bikerman_counterion=` kwarg in `_solve_mms_on_mesh` and
`run_mms`; when supplied, the manufactured Poisson source uses the
closure expression and the manufactured packing fraction includes the
counterion's contribution. `tests/test_mms_steric_boltzmann_convergence.py`
exists with the bikerman manufactured-source plumbing wired but is
marked `xfail` — Newton diverges from `U=U_manuf` at the currently-
tractable test voltages (the production η=21 saturates the closure so
`packing_manuf → 0`; lower η leaves Newton out of basin for the same
reason that affects the ideal MMS at low η). A redesigned manufactured
solution that keeps both the closure unsaturated and Newton in basin
is a follow-up plan. The existing
`tests/test_mms_convergence.py` (ideal path) is the byte-identical
regression gate.

### Tests

| Path | File | Coverage |
|---|---|---|
| Algebra (fast) | `tests/test_steric_boltzmann_closure_algebra.py` | bulk recovery, dilute limit, saturation, packing positivity, theta_b validator (15 tests) |
| Config (fast) | `tests/test_config_steric_mode.py` | parse/validate `steric_mode` + `a_nondim` (14 tests) |
| Wiring (fast) | `tests/test_steric_boltzmann_closure.py::test_double_counting_rejected`, `…_theta_b_negative_rejected`, `…_multi_bikerman_rejected`, `…_no_bikerman_returns_none` | helper validators + early-return path (4 tests) |
| Byte-identity (slow) | `tests/test_steric_boltzmann_closure.py::test_ideal_path_byte_identical` | ideal-path regression to existing snapshot baseline at V=+0.66 V, `rel_tol=1e-6` |
| Smoke (slow) | `tests/test_steric_boltzmann_closure.py::test_bikerman_smoke_cathodic_window` | 3sp + bikerman cathodic V_RHE in [-0.5, +0.1], converges and produces finite CD/PC |
| Diagnostics (slow) | `tests/test_steric_boltzmann_closure.py::test_diagnostics_reports_bikerman_mode_at_converged_voltage` | `c_counterion0_steric_mode='bikerman'`; surface c bounded by 1/a_b; `surface_counterion_within_steric=True` at cathodic V |
| Equivalence (slow) | `tests/test_steric_boltzmann_closure.py::test_steric_boltzmann_equiv_to_4sp_dynamic` | 3sp+bikerman ↔ 4sp dynamic on the cathodic overlap, hybrid abs/rel tol 5e-3 |
| muh formulation (slow) | `tests/test_steric_boltzmann_closure.py::test_bikerman_smoke_muh` | same physics on `formulation="logc_muh"` |
| MMS (slow) | `tests/test_mms_steric_boltzmann_convergence.py` | bikerman manufactured source, L2 rate ≥ 1.95−tol on UnitSquareMesh sweep |

Existing regression gates preserved byte-identical:
`tests/test_stern_no_stern_snapshot.py`,
`tests/test_solver_equivalence.py`,
`tests/test_initializer_debye_boltzmann.py`,
`tests/test_initializer_debye_boltzmann_4sp.py`,
`tests/test_steric_saturation.py`,
`tests/test_steric_sign.py`,
`tests/test_mms_convergence.py`.

### Known limits / out of scope

1. **Multi-counterion bikerman:** the closure algebra couples when more
   than one counterion is steric-aware; the helper raises
   `NotImplementedError`. Single-bikerman + multiple-ideal entries is
   supported.
2. **`debye_boltzmann` IC for the bikerman path at high anodic V:** the
   analytical IC seeds `phi_init = ln(H_outer/c_ClO4_bulk) + psi`
   (ideal-Boltzmann electroneutrality), which saturates `c_steric` at
   the IC and blows up `mu_steric = -ln(packing→0)`. The smoke test
   uses the cathodic window + `linear_phi` initializer to avoid this;
   high-V production runs need a bikerman-aware IC, deferred to a
   follow-up plan analogous to the 4sp Option 2a' fix.
3. **`Forward/dirichlet_solver.py` and `robin_solver.py`** still use the
   old `+ln(packing)` sign. Not in the production dispatch graph; the
   bikerman path is not wired into them. Sign correction and bikerman
   support there are separate plans.
4. **Stern/Frumkin reformulation:** closure operates on the diffuse-
   layer φ unchanged. Stern's Robin BC remains as-is.
5. **Inverse-pipeline TRUE-cache rebake:** the inverse uses 3sp+ideal by
   default; switching to bikerman would require a separate study and
   cache regeneration.

### Pointers

- Helper: `Forward/bv_solver/boltzmann.py::build_steric_boltzmann_expressions`
- Wiring: `Forward/bv_solver/forms_logc.py` (search for `steric_boltz`)
- Wiring (muh): `Forward/bv_solver/forms_logc_muh.py` (same)
- Config parser: `Forward/bv_solver/config.py:_get_bv_boltzmann_counterions_cfg`
- Diagnostics: `Forward/bv_solver/diagnostics.py:collect_diagnostics`
- MMS source: `scripts/verification/mms_bv_3sp_logc_boltzmann.py:_build_manufactured_source`
- Preset: `scripts/_bv_common.py:DEFAULT_CLO4_BOLTZMANN_COUNTERION_STERIC`
