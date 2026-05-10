# Electrochemical-potential solver feasibility and implementation plan

Date: 2026-05-04

## Bottom line

Using electrochemical-potential variables is feasible and worth prototyping,
but it should be treated as an experimental formulation branch first, not as a
silent replacement for the production solver.

The most conservative first implementation is a hybrid proton branch:

```text
formulation = "logc_muh"
unknowns    = [u_O2, u_H2O2, mu_H, phi]
mu_H        = u_H + em*z_H*phi
c_H         = exp(mu_H - em*z_H*phi)
J_H         = D_H*c_H*(grad(mu_H) + grad(mu_steric))
```

This is algebraically equivalent to the current log-c PDE when the same
solution is reached, but it gives Newton a smoother primary variable in the
Debye layer. At Boltzmann equilibrium, `mu_H` is nearly flat while `u_H` and
`phi` each vary by tens of log units.

Do not oversell this as a complete peroxide-window fix. The recent Stern
sweep shows the wall can be crossed numerically, but the analytic Boltzmann
counterion still exceeds the Bikerman steric scale. A `mu_H` branch improves
the proton block; it does not by itself saturate analytic ClO4- or remove the
large Poisson source.

## Recent context to read first

Read these before coding:

```text
docs/PNP_BV_Analytical_Simplifications.md
docs/Log Space Clamp Removal Suggestions.md
docs/Peroxide Solver Convergence.md
docs/stern_layer_physics_and_next_steps.md
docs/4sp_drop_boltzmann_investigation.md
StudyResults/peroxide_window_stern_test/summary.md
```

Current live implementation to inspect:

```text
Forward/bv_solver/forms_logc.py
Forward/bv_solver/config.py
Forward/bv_solver/dispatch.py
Forward/bv_solver/boltzmann.py
Forward/bv_solver/grid_per_voltage.py
Forward/bv_solver/diagnostics.py
scripts/_bv_common.py
tests/test_initializer_debye_boltzmann.py
tests/test_initializer_debye_boltzmann_4sp.py
tests/test_solver_equivalence.py
tests/test_mms_convergence.py
```

Important current status:

- Production is `3sp + analytic Boltzmann ClO4- + log-c + log-rate BV`.
- `forms_logc.py` uses `u_i = log(c_i)` and reconstructs `c_i` with a
  symmetric `u_clamp`.
- `debye_boltzmann` initialization is implemented and helps at anodic
  voltages, but no-Stern still fails at `V_RHE >= +0.68 V`.
- Stern support is already wired in the current tree, despite older docs
  describing it as a pending gap.
- The Stern sweep converged at peroxide-window voltages for finite `C_S`, but
  surface analytic `c_ClO4` exceeded the steric cap of about `1/a = 100`, so
  those states are Newton-converged but physically suspect.
- The 4sp dynamic stack is physically attractive because ClO4- is steric, but
  current residual/IC machinery fails past about `V_RHE = +0.1 V`.

## Feasibility assessment

### What is likely to work

The `logc_muh` branch is a low-to-medium risk formulation change. It is mostly
a change of variables for H+:

```text
u_H = mu_H - em*z_H*phi
c_H = exp(u_H)
```

The current H+ flux term:

```text
D_H*c_H*(grad(u_H) + grad(em*z_H*phi) + grad(mu_steric))
```

becomes:

```text
D_H*c_H*(grad(mu_H) + grad(mu_steric))
```

That is exactly the same ideal electrochemical-potential gradient, just
represented by a smoother unknown.

This branch should be straightforward to test against existing log-c results
on the overlap window where both formulations converge.

### What is uncertain

It is uncertain whether `logc_muh` alone crosses the no-Stern `+0.68 V` wall.
The current failure has two coupled parts:

1. H+ must deplete by enormous factors near the electrode.
2. Analytic ClO4- contributes `c_bulk*exp(phi)` to Poisson and can exceed the
   steric scale by many orders of magnitude.

`mu_H` directly targets the first part. It does not directly target the second.

### What it cannot solve alone

`logc_muh` does not make the high-voltage 3sp+Boltzmann model physically valid
when analytic `c_ClO4` is above the Bikerman cap. If it makes Newton converge
at `V_RHE >= +0.68 V`, still gate the output with:

```text
c_ClO4_surface = c_bulk*exp(phi_surface) <= 100 nondim
```

If this fails, the result is useful as a numerical diagnostic but not as a
production physical answer.

### Most promising long-term route

Use `logc_muh` as the small first step. If it proves stable, generalize to
charged-species electrochemical potentials for the 4sp dynamic stack:

```text
formulation = "logc_mu_charged"
unknowns    = [u_O2, u_H2O2, mu_H, mu_ClO4, phi]
mu_i        = u_i + em*z_i*phi  for charged species
u_i         = mu_i - em*z_i*phi
```

That route is more physically attractive because ClO4- remains dynamic and
steric. It is also higher risk and should not be the first patch.

## Implementation plan for Claude Code

### Phase 0 - Baseline check

Before editing, record the current dirty tree and do not revert unrelated
changes.

Run lightweight checks if the environment is available:

```bash
pytest tests/test_bv_common_config.py -q
pytest tests/test_initializer_debye_boltzmann.py -q
pytest tests/test_initializer_debye_boltzmann_4sp.py -q
```

If Firedrake is unavailable, note that and continue with static/unit tests
that do not import Firedrake.

### Phase 1 - Add formulation routing

Patch `Forward/bv_solver/config.py`:

1. Extend valid formulations to include canonical lower-case names:

   ```python
   _VALID_FORMULATIONS = ("concentration", "logc", "logc_muh")
   ```

2. Keep accepting `formulation="logc_muH"` from old docs by relying on
   `.lower()` normalization to `logc_muh`.

Patch `scripts/_bv_common.py`:

1. Update `_make_bv_convergence_cfg()` docstring to mention `logc_muh`.
2. Update `make_bv_solver_params()` docstring similarly.
3. Do not change the default; it remains `formulation="logc"`.

Patch `Forward/bv_solver/dispatch.py`:

1. Dispatch `build_context`, `build_forms`, and `set_initial_conditions` on
   `params["bv_convergence"]["formulation"]`.
2. Route:

   ```text
   "logc"      -> forms_logc
   "logc_muh"  -> forms_logc_muh
   ```

3. Keep `initializer` dispatch separate from `formulation` dispatch.
4. Keep old behavior exactly for `logc`.

Patch `Forward/bv_solver/__init__.py`:

1. Export the new explicit `*_logc_muh` helpers.
2. Update the module docstring to say production default is still `logc`.

### Phase 2 - Implement `forms_logc_muh.py`

Create:

```text
Forward/bv_solver/forms_logc_muh.py
```

Start by copying `forms_logc.py`. Keep the first version intentionally close
to the original; refactor common code only after tests pass.

Required variable mapping:

```python
raw = fd.split(U)
raw_prev = fd.split(U_prev)

phi = raw[-1]
phi_prev = raw_prev[-1]

unknowns = list(raw[:-1])
unknowns_prev = list(raw_prev[:-1])

mu_h_idx = 2
u_exprs = list(unknowns)
u_prev_exprs = list(unknowns_prev)

u_exprs[mu_h_idx] = unknowns[mu_h_idx] - em * z[mu_h_idx] * phi
u_prev_exprs[mu_h_idx] = (
    unknowns_prev[mu_h_idx] - em * z[mu_h_idx] * phi_prev
)
```

Use `u_exprs` everywhere the current code means physical log concentration:

```text
ci
ci_prev
BV log-rate expressions
Poisson charge source
steric packing
surface concentration expressions
```

Do not use raw `unknowns[2]` as `log(c_H)`.

Flux assembly:

```python
if i == mu_h_idx:
    ideal_grad = fd.grad(unknowns[i])
else:
    ideal_grad = fd.grad(u_exprs[i]) + fd.grad(em * z[i] * phi)

if steric_active:
    Jflux = D[i] * c_i * (ideal_grad + fd.grad(mu_steric))
else:
    Jflux = D[i] * c_i * ideal_grad
```

This preserves the original steric term. Do not fold `mu_steric` into the
primary variable in the first implementation.

Boundary conditions:

- Neutral species bulk BCs stay `u_i = log(c0_i)`.
- H+ bulk BC becomes:

  ```text
  mu_H_bulk = log(c_H_bulk) + em*z_H*phi_bulk
  ```

  In the current solver `phi_bulk = 0`, so this is numerically the same as
  `log(c_H_bulk)`. Keep the expression/comment explicit.

Context keys:

Add these to `ctx.update(...)`:

```python
"u_exprs": u_exprs,
"mu_species": [2],
"logc_muh_transform": True,
"logc_transform": True,
"ci_exprs": ci,
```

These keys let diagnostics and observables know that raw `U.sub(2)` is not
`log(c_H)`.

### Phase 3 - Initial conditions

Implement:

```text
set_initial_conditions_logc_muh
set_initial_conditions_debye_boltzmann_logc_muh
```

For `linear_phi`:

1. Build the same linear `phi_init` as `forms_logc.py`.
2. For O2 and H2O2, assign `log(c0_i)`.
3. For H+, assign:

   ```text
   mu_H_init = log(c_H_bulk) + em*z_H*phi_init
   ```

   This preserves the old physical concentration initial condition:

   ```text
   exp(mu_H_init - em*z_H*phi_init) = c_H_bulk
   ```

For `debye_boltzmann`:

1. Reuse the same Picard/Gouy-Chapman calculation as `forms_logc.py`.
2. Change only the final assignment:

   ```text
   u_H_init  = log(H_outer) - psi
   phi_init  = log(H_outer/c_clo4_bulk) + psi
   mu_H_init = u_H_init + em*z_H*phi_init
   ```

3. With `em*z_H = 1`, the Debye-layer `psi` cancels out:

   ```text
   mu_H_init = log(H_outer) + log(H_outer/c_clo4_bulk)
   ```

   That is the desired smooth initial variable.

Do not change the existing `forms_logc.py` initializer except for shared
helper extraction if you choose to refactor carefully.

### Phase 4 - Diagnostics and validation helpers

Patch `Forward/bv_solver/diagnostics.py`:

1. If `ctx["ci_exprs"]` exists, compute surface concentration means by
   assembling those expressions rather than using `exp(mean(raw U.sub(i)))`.
2. If `ctx["u_exprs"]` exists, report `u{i}_surface_mean` from those
   expressions.
3. Also report `mu2_surface_mean` when `2 in ctx.get("mu_species", [])`.
4. Preserve the old output keys for the current log-c branch.

Patch `Forward/bv_solver/validation.py` if any `logc_muh` code path calls it:

- The existing `is_logc=True` logic assumes raw `U.sub(i)` is `u_i`.
- Either add optional `u_exprs/ci_exprs` support or avoid using it for
  `logc_muh` until it is updated.

### Phase 5 - Tests

Add a focused test file:

```text
tests/test_logc_muh_formulation.py
```

Minimum tests:

1. Config accepts `formulation="logc_muH"` and stores canonical
   `logc_muh`.
2. Dispatcher routes `formulation="logc_muh"` to the new module.
3. `linear_phi` IC reconstructs constant H+ concentration:

   ```text
   max|exp(mu_H - phi) - c_H_bulk| < tolerance
   ```

4. `debye_boltzmann` IC makes `mu_H` much smoother than `u_H` at an anodic
   test point, for example:

   ```text
   range(mu_H) < 0.25 * range(u_H)
   ```

5. A small overlap solve compares `logc` and `logc_muh` at a voltage where
   the current solver is known to converge. Compare CD and PC with a hybrid
   tolerance, not exact equality.

Run existing regression tests:

```bash
pytest tests/test_bv_common_config.py -q
pytest tests/test_initializer_debye_boltzmann.py -q
pytest tests/test_initializer_debye_boltzmann_4sp.py -q
pytest tests/test_solver_equivalence.py -q
```

If time allows, add an MMS variant for `logc_muh` by extending:

```text
scripts/verification/mms_bv_3sp_logc_boltzmann.py
```

Manufactured exact fields:

```text
u_O2_exact
u_H2O2_exact
u_H_exact
phi_exact
mu_H_exact = u_H_exact + em*z_H*phi_exact
```

The expected convergence rates are unchanged.

### Phase 6 - Study scripts

Create:

```text
scripts/studies/peroxide_window_muh_test.py
```

Use the production 3sp stack:

```text
species = THREE_SPECIES_LOGC_BOLTZMANN
boltzmann_counterions = [DEFAULT_CLO4_BOLTZMANN_COUNTERION]
formulation = "logc_muh"
log_rate = True
initializer = "debye_boltzmann"
exponent_clip = 100.0
mesh Ny = 200
```

Run:

```text
V_RHE = [0.60, 0.66, 0.68, 0.70, 0.75, 0.80, 1.00]
Stern C_S = [None, 0.05, 0.10, 0.20]
```

Save:

```text
StudyResults/peroxide_window_muh_test/iv_curve.json
StudyResults/peroxide_window_muh_test/diagnostics.json
StudyResults/peroxide_window_muh_test/results.csv
StudyResults/peroxide_window_muh_test/summary.md
StudyResults/peroxide_window_muh_test/comparison.png
```

Summary gates:

```text
1. Does no-Stern logc_muh cross +0.68?
2. Does Stern+logc_muh reduce SNES work or extend convergence vs Stern+logc?
3. Does any converged 3sp+Boltzmann state keep analytic c_ClO4_surface <= 100?
4. Are CD/PC smooth against the existing no-Stern and Stern baselines?
5. Are eta-clip flags inactive at clip=100 over the peroxide-window points?
```

If the answer to gate 3 is no, write that the branch improves numerical
conditioning but does not solve physical validity.

## Optional Phase 7 - Charged-species mu branch for 4sp

Only attempt this after `logc_muh` passes overlap tests.

Goal:

```text
formulation = "logc_mu_charged"
unknowns    = [u_O2, u_H2O2, mu_H, mu_ClO4, phi]
```

Mapping:

```text
u_i = raw_i - em*z_i*phi  for charged species
u_i = raw_i               for neutral species
c_i = exp(u_i)
```

Flux:

```text
charged: D_i*c_i*(grad(mu_i) + grad(mu_steric))
neutral: D_i*c_i*(grad(u_i)  + grad(mu_steric))
```

Use the 4sp dynamic preset:

```text
species = FOUR_SPECIES_LOGC_DYNAMIC
boltzmann_counterions = None
```

Do not use the existing pure-Boltzmann 4sp `debye_boltzmann` IC without
modification. `docs/4sp_drop_boltzmann_investigation.md` shows that it can
seed ClO4- above the Bikerman cap and drive the steric residual into a
singular regime.

For 4sp, either:

1. Start with `linear_phi` and test whether charged mu variables alone improve
   the `+0.1 -> +0.3 V` wall.
2. Or add the `gamma`-corrected modified Poisson-Boltzmann IC from
   `docs/4sp_drop_boltzmann_investigation.md` before using a Debye-layer IC.

Validation gates for 4sp:

```text
1. Existing 3sp+Boltzmann vs 4sp equivalence still passes on [-0.5, +0.1].
2. 4sp logc_mu_charged converges at V=+0.3.
3. Surface c_ClO4 remains below about 100 nondim.
4. Warm-walk reaches at least +0.5, then retry peroxide-window points.
```

This is the path most likely to produce a physically valid high-voltage
solver, but it is larger than the proton-only branch.

## Acceptance criteria

Do not call the implementation complete until:

```text
1. logc behavior is unchanged by default.
2. logc_muh has its own config, dispatch, IC, diagnostics, and tests.
3. logc_muh reproduces logc CD/PC on the overlap window within documented
   tolerances.
4. Existing initializer and solver-equivalence tests still pass.
5. Peroxide-window study artifacts are written with explicit physical-validity
   flags.
6. The summary clearly separates:
   - numerical convergence,
   - observable smoothness,
   - steric physical validity.
```

## Suggested Claude Code prompt

```text
Read docs/electrochemical_potential_solver_plan.md and the referenced recent
docs. Implement the experimental proton electrochemical-potential formulation
as formulation="logc_muh".

Keep the production default as formulation="logc". Add routing in
Forward/bv_solver/config.py, dispatch.py, and __init__.py. Create
Forward/bv_solver/forms_logc_muh.py by closely mirroring forms_logc.py, but
store H+ as mu_H = u_H + em*z_H*phi and reconstruct u_H = mu_H - em*z_H*phi
for concentrations, BV log-rate factors, Poisson, and steric packing. The H+
flux should use grad(mu_H), plus grad(mu_steric) when steric is active.

Implement linear_phi and debye_boltzmann initializers for logc_muh. For
linear_phi, assign mu_H = log(c_H_bulk) + em*z_H*phi_init so c_H remains bulk.
For debye_boltzmann, reuse the existing Picard/Gouy-Chapman profiles but
assign mu_H = u_H + em*z_H*phi.

Update diagnostics so c/u surface means are reconstructed from ctx["ci_exprs"]
and ctx["u_exprs"] when present, and report mu2_surface_mean.

Add tests in tests/test_logc_muh_formulation.py for config normalization,
dispatch, linear IC concentration reconstruction, smoother debye mu_H, and an
overlap CD/PC comparison against logc. Run the existing BV config,
initializer, and solver-equivalence tests.

Finally create scripts/studies/peroxide_window_muh_test.py. Sweep
V_RHE=[0.60,0.66,0.68,0.70,0.75,0.80,1.00] and C_S=[None,0.05,0.10,0.20]
with the 3sp+Boltzmann+log-rate stack at exponent_clip=100 and
initializer=debye_boltzmann. Save JSON/CSV/summary artifacts under
StudyResults/peroxide_window_muh_test/. The summary must report convergence
separately from physical validity, especially whether analytic
c_ClO4_surface exceeds the steric cap around 100 nondim.
```
