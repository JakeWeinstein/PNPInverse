# Round 3: Counterreply on the v2 plan

All 18 of round 2's issues land.  Below: per-issue responses, then plan v3.

## Per-issue responses

**Re point 1** (re-point 7 was wrong; code uses `make_run_ss(...)` with
adaptive dt + plateau detection + `max_ss_steps`, not a fixed dt-march):
**Accept.**  Verified — `Forward/bv_solver/anchor_continuation.py:1033`
and `:1728` both import `make_run_ss` from `grid_per_voltage`, and
`grid_per_voltage.py:41-42` exposes `max_ss_steps_cold=200`,
`max_ss_steps_z=120` knobs.  The Picard inner steady-state solve must
**call `make_run_ss(ctx, ...).ss_step(max_ss_steps)`** verbatim, not
invent a parallel convention.  Plan v3 explicitly uses `make_run_ss`.

**Re point 2** (custom per-V loop underspecified vs nearest-source
ordering and source pool):
**Accept.**  The grid `[-0.40, +0.55]` is built from `np.linspace` (low
to high V).  `solve_grid_with_anchor` (per `grid_per_voltage.py:892+`)
visits voltages by distance-from-anchor and maintains a source pool of
`(U_snapshot, phi)` candidates, picking the nearest for warm-walk.  My
v2 "for each V in grid" was a regression.  Plan v3 fixes this — the
Picard wrapper FORKS `solve_grid_with_anchor` minimally, keeping the
nearest-source ordering and source pool; only the per-V solve call is
replaced.  See plan v3 §3.

**Re point 3** (don't hand-wave warm-walk/bisection/z-ramp away; either
fork `solve_grid_with_anchor` or expose Picard-aware `warm_walk_phi`):
**Accept** — pick the first option.  Fork
`solve_grid_with_anchor → solve_grid_with_anchor_picard`, preserving
the entire C+D continuation machinery (source pool, nearest ordering,
bisection, z-ramp).  Inside each per-V `warm_walk_phi` call, after
Newton+`make_run_ss` completes, run the outer Picard loop using
`make_run_ss` for each inner iter.  Picard sits as an outer wrapper
around `run_ss`, NOT around `warm_walk_phi`.

**Re point 4** (`_build_bv_observable_form(mode="reaction", scale=1.0)`
assembles `∫ R_j ds`, NOT a surface mean):
**Accept** — silent units bug.  Fix:

  `R_j_mean_hat = fd.assemble(R_j_form) / fd.assemble(fd.Constant(1.0) * fd.ds(electrode_marker))`

Same correction applied to all surface-integrated diagnostics (R, θ_OHP).
Add a test that the bare assemble equals `R_j_mean × electrode_measure`
to catch regressions.

**Re point 5** (algebra error: residual is `ξ + R·I/D − c_b/θ_b`, no θ
factor):
**Accept** — straight algebra slip.  The θ_OHP factor enters Eq B's
denominator because `R = K · θ_OHP · ξ` substitutes into Eq A', but the
residual of Eq A' itself has no θ_OHP factor.  Correct residual:

  `residual = ξ + R_O2_hat · I_hat / D_O2_hat − c_b_hat / θ_b`

This goes into the test assertion AND the per-iter diagnostic.

**Re point 6** (`< 1e-6` test conflicts with Picard tol of `1e-3`):
**Accept.**  Test runs Picard to a tighter tolerance for the assertion
case: `PICARD_TOL_TEST = 1e-8`, then asserts residual `< 1e-6`.  This
verifies the closed-form Eq B is being implemented correctly.  The
production tol stays at `1e-3` for speed.

**Re point 7** (θ=1 path not specified; `packing` only defined when
`steric_active=True`):
**Accept.**  In `forms_logc_muh.py`, when `bv_picard_mode=True`:
  - If `steric_active=True`: use existing `packing` and `theta_inner`.
  - Else: define `packing = theta_inner = fd.Constant(1.0)` and proceed
    with closure using these.
  - Always expose `ctx["packing_expr"]`, `ctx["theta_inner_expr"]`,
    `ctx["closure_theta_b"]` (set to 1.0 in ideal case),
    `ctx["closure_bulk_c_hat"][species]` (from `c0_model_vals`).

**Re point 8** (don't dig through `steric_boltz[0].metadata` for θ_b):
**Accept.**  Store at form-build:

  `ctx["closure_theta_b"] = float(theta_b_bulk_val)
   ctx["closure_bulk_c_hat"] = {s: float(scaling["c0_model_vals"][s])
                                for s in cathodic_species_set}`

`closure_picard.py` reads from ctx, no metadata heuristics.

**Re point 9** (`bv_picard_mode` may silently no-op with
`formulation="logc"`):
**Accept.**  In `config.py:_get_bv_convergence_cfg`, when
`bv_picard_mode=True`, validate `formulation == "logc_muh"`.  Raise
`ValueError` otherwise.  The logc backend was removed in May 2026
per `CLAUDE.md` so this is mostly a tripwire for future regressions.

**Re point 10** ("no positive fixed point" overstates "numerical
underflow"):
**Accept.**  When ξ_target falls below `ξ_floor = exp(-50)`, log
"PICARD: ξ_target underflow at iter N, V=X — clamping to floor.
Indicates K·I·θ/D ratio extreme; verify K_old, I_hat, θ_OHP values."
NOT "no fixed point exists."  Clamp + continue, report `picard_converged
= True` only if subsequent iter still hits floor (= persistent target).

**Re point 11** (semi-implicit requires `R > 0`; reversible/anodic gives
`R ≤ 0`):
**Accept.**  Gate the formula:
  - If `R_s_hat > 0`: use Eq B `ξ_target = (c_b/θ_b)/(1 + K·I·θ/D)`.
  - Else: `ξ_target = c_b_hat/θ_b` (no-flux equilibrium; net production
    or zero net flux means c_OHP relaxes to bulk equilibrium).
For Jithin emulation (Tafel-only R2e), `R > 0` always at V < E_eq = 0.695 V,
so this branch is dormant.  But the gate prevents future-deck use from
silently producing `K < 0` and divergence.

**Re point 12** ("per cathodic species" is too broad; could silently
patch charged species):
**Accept.**  Add explicit `z_species == 0` gate:

  for s in cathodic_species_set:
      if int(z_vals[s]) != 0:
          if not explicit_override_for_charged.get(s, False):
              raise ValueError(
                  f"bv_picard_mode: closure substitute on species index {s} "
                  f"requires z=0 (Bikerman closure derived for neutrals). "
                  f"Got z={z_vals[s]}.  For charged species, the Boltzmann
                  f"pile-up in PDE already matches the closure form."
              )

Default behavior: closure substitute applies ONLY to z=0 cathodic species.
For Jithin: O₂ is the only one.  H⁺ stays at PDE value (z=+1, Boltzmann
pile-up captures the closure).

**Re point 13** (x-invariance check not implementable as stated):
**Accept.**  Skip the runtime check in v2.  Replace with:
  - A one-time **setup assertion** that the script configuration is
    uniform in x (Stern BC uniform on bottom; Dirichlet bulk uniform
    on top; no x-localized source).  Encoded as a comment + assert
    on the relevant params dict keys.
  - For v3 (out-of-scope here): implement DG0-on-x binning if needed.

**Re point 14** (incomplete x-invariance check ignores other fields):
**Accept** — moot under the skip-for-v2 resolution above.  Noted as
follow-up if v2 results suggest x-variance.

**Re point 15** (no-flux equivalence test at "R≈0 anodic V" is not
clean):
**Accept.**  Replace with:
  - `test_no_flux_equivalent_to_v1`: set `k0_hat = 0` for the reaction
    (literally `bv_rate_exprs[j] = 0`), run Picard, assert 1 iter to
    convergence + cd matches v1's k0=0 baseline to machine precision
    (no flux at all).
  - Separate: `test_small_rate_anodic_regression`: at V=+0.50 V (well
    anodic), run Picard, assert cd matches v1 within 1% (small but
    nonzero flux; Picard correction is small).

**Re point 16** (`log_c_cat = ln(packing) + log_ξ` changes behavior
when packing floors):
**Accept.**  In validation runs (`bv_picard_mode=True`):
  - Hard failure if `floor_hit_area_fraction > 0` at converged state
    (the closure formula assumes packing > floor; floor engagement
    means closure doesn't reflect physics).
  - Configurable via `bv_picard_strict_floor: bool` (default True for
    Picard mode).  Set False only for floor-sensitivity experiments.
  - Diagnostic: `max(floor_hit_area_fraction)` over Picard iters
    recorded in JSON.

**Re point 17** (Picard residual only on scalar ξ ignores PDE drift):
**Accept.**  Add per-iter state-change norm:
  `dU_norm = sqrt(assemble(inner(U − U_prev_picard, U − U_prev_picard) * dx))`
  where `U_prev_picard` is the U snapshot at start of this Picard iter.
  Convergence requires:
    - ξ residual `< PICARD_TOL_RESIDUAL` AND
    - ξ step `< PICARD_TOL_STEP` AND
    - `make_run_ss` returned True (plateau detected) AND
    - `dU_norm / max(|U|_inf, 1e-12) < PICARD_TOL_STATE = 1e-4`
  All four must hold.

**Re point 18** (parallel 2e/4e form changes need at least a smoke test
NOW):
**Accept.**  Add to test suite:
  - `test_parallel_2e_4e_shared_supply`: build a minimal 2-reaction
    config (both consuming O₂, different stoich and α), run Picard at
    one V, assert that both reactions see the SAME ξ_O2 and that
    `R_O2_total = R_2e + R_4e` matches the closure update.  Doesn't
    need the full Jithin Fig 4.36 deck — just confirms shared-species
    plumbing works.

## Updated artifact (v3)

```markdown
# Jithin Closure Outer-Picard Implementation Plan (v3, post-GPT-round-2)

## Goal

Implement an outer Picard wrap on top of the existing
`bv_jithin_closure_form` patch.  Test whether the **continuum-MPNP
analog** of Jithin's Eq 4.31 closure reproduces his Fig 4.36 cliff.

Scope: single-R2e Jithin emulation as primary; parallel-2e/4e
smoke test as plumbing validation.  Full Fig 4.36 parallel run is
follow-up.

## Math (code convention)

Symbols:
- `c_b_hat` = bulk O₂ concentration nondim
- `θ_OHP`, `θ_b` = packing fraction at OHP / bulk
- `R_O2_hat` = molar rate of O₂ consumption at OHP, nondim, ≥0 for cathodic
  = `Σ_j (−stoich[O₂, j]) · R_j_mean_hat`
- `R_j_mean_hat = assemble(R_j_form) / assemble(1·ds(electrode))`
- `I_hat = assemble((1/packing) · dx) / Lx_hat`
- `D_O2_hat` = O₂ diffusivity nondim
- `η = (V_app − E_eq) / V_T` (cathodic → η<0)
- Code BV cathodic rate: `R_j_hat = k₀_hat · c_cat_hat · exp(−α·n·η)` (grows as V drops)
- Supply variable: `ξ = c_OHP_hat / θ_OHP` (R-space `fd.Function`)

### Closure (Eq A'): continuum-MPNP boundary supply for z=0 species

  **ξ = c_b_hat / θ_b − R_O2_hat · I_hat / D_O2_hat**

### Picard target (Eq B): semi-implicit positivity-preserving update

For `R_O2_hat > 0`:

  K_old = R_O2_hat / c_eff_hat_old   where c_eff_hat_old = θ_OHP · ξ_old
  **ξ_target = (c_b_hat/θ_b) / (1 + K_old · I_hat · θ_OHP / D_O2_hat)**

For `R_O2_hat ≤ 0` (reversible/anodic; not exercised in Jithin):
  **ξ_target = c_b_hat / θ_b**

Damped update: `ξ_new = exp((1−α)·log(ξ_old) + α·log(max(ξ_target, ξ_floor)))`,
default α = 0.5, ξ_floor = exp(−50) ≈ 2e−22.

### Fixed-point residual (for tests and diagnostics)

  **residual = ξ + R_O2_hat · I_hat / D_O2_hat − c_b_hat / θ_b**

(No θ_OHP factor; cancels because R already includes θ_OHP via
`R = k₀ · θ_OHP · ξ · exp(−αnη)`.)

### Pre-analysis prediction (unchanged from v2)

Smooth S-curve from kinetic onset to ~Levich, no cliff.  Continuum
correction ≈ 6% sub-Levich at our geometry.

## Implementation

### Files modified

1. **`Forward/bv_solver/config.py`** (~15 lines)
   - Add `bv_picard_mode: bool` (default False).
   - Add `bv_picard_strict_floor: bool` (default True, ignored unless
     bv_picard_mode True).
   - Validate:
     - `bv_picard_mode=True` requires `bv_jithin_closure_form=True`
       AND `bv_log_rate=True` AND `formulation == "logc_muh"`.

2. **`Forward/bv_solver/forms_logc_muh.py`** (~60 lines)
   - When `bv_jithin_closure_form=True` AND `bv_picard_mode=True`:
     - Validate `z_vals[s] == 0` for every distinct cathodic species s
       across reactions (raise ValueError otherwise).
     - Allocate `xi_func_s = fd.Function(R_space, name=f"picard_log_xi_sp{s}")`
       per distinct cathodic species s.  Initial value: `log(c_b_hat_s / θ_b_const)`.
     - Replace `log_c_cat = u_exprs[cat_idx]` with
       `log_c_cat = ln(packing) + xi_func_s`.
     - Store: `ctx["picard_log_xi_funcs"] = {s: xi_func_s}`.
   - When `bv_picard_mode=True`:
     - If `steric_active=False`: define `packing = theta_inner = fd.Constant(1.0)`.
     - Expose: `ctx["packing_expr"] = packing`,
       `ctx["theta_inner_expr"] = theta_inner`,
       `ctx["closure_theta_b"] = float(theta_b_bulk_val_or_1)`,
       `ctx["closure_bulk_c_hat"] = {s: float(scaling["c0_model_vals"][s])}`,
       `ctx["closure_cathodic_species_set"] = {s, ...}`.

3. **`Forward/bv_solver/closure_picard.py`** (NEW, ~300 lines)
   - `@dataclass PicardState`: `xi_per_species`, `U_snapshot`,
     `U_prev_snapshot`, `iter_history`.
   - `extract_picard_diagnostics(ctx, electrode_marker, Lx_hat) → dict`:
     uses surface-mean form `assemble(R · ds) / assemble(1 · ds)`,
     volume form `assemble((1/packing) · dx) / Lx_hat`.
   - `compute_picard_target(c_b, θ_b, θ_OHP, R_O2, I, D_O2) → ξ_target`:
     applies Eq B with R>0 gate.
   - `picard_iterate(ctx, sp, *, run_ss_max_steps=200, max_picard_iters=15,
     tol_residual=1e-3, tol_step=1e-3, tol_state=1e-4, damping=0.5,
     damping_min=0.05, max_damping_retries=3, strict_floor=True)`:
     - Initialize ξ from current ctx state.
     - Snapshot U, U_prev, xi_funcs.
     - Loop:
       - `run_ss = make_run_ss(ctx, ...)`; `ok = run_ss(run_ss_max_steps)`.
       - On failure: restore snapshot, halve damping, retry (up to
         max_damping_retries).  Persistent fail → return PicardState with
         `converged=False, reason="run_ss_failed"`.
       - Extract diagnostics.
       - If `strict_floor` and any iter shows `floor_hit_area > 0`: 
         return `converged=False, reason="packing_floored"`.
       - Compute ξ_target per species, damped update.
       - Snapshot iter state.
       - Convergence: residual + step + state-norm + run_ss-ok all clear.
     - Return PicardState.

4. **`Forward/bv_solver/grid_per_voltage.py`** — extend
   `solve_grid_with_anchor` (NOT fork; add opt-in flag).
   - Add param `picard_iterate_fn: Optional[Callable] = None`.
   - When provided, after the normal `make_run_ss` call inside each
     per-V solve (`grid_per_voltage.py:613, 686, 734, 825, 1112`),
     call `picard_iterate_fn(ctx, sp, ...)` and use its convergence
     status as the per-V `converged` flag.
   - Source pool stores `(U_snapshot, xi_snapshot, phi)` instead of
     `(U_snapshot, phi)`.  When warm-walking, restore both U AND xi.
   - `PreconvergedAnchor` extended (or `PreconvergedAnchorWithPicard`
     subclass added) with `xi_snapshots: dict[int, np.ndarray]`.

5. **`scripts/studies/_run_jithin_closure_picard.py`** (NEW, ~500 lines)
   - Fork `_run_jithin_closure_exact.py`.
   - `BV_PICARD_MODE = True`, `PICARD_*` constants set.
   - Setup assertion: confirm Stern BC + bulk Dirichlet are x-uniform
     and no x-localized source is wired (assert keys in params).
   - Stage 1: anchor build at V=+0.55 V, then call `picard_iterate`
     to converge ξ at anchor.  Anchor V is anodic, expect 1-2 Picard
     iters.
   - Stage 2: Stern bump ladder.  Per rung: `set_stern_capacitance_model`,
     `run_ss`, then `picard_iterate` to converge ξ at this rung.
   - Stage 3: `solve_grid_with_anchor(..., picard_iterate_fn=picard_iterate)`.
   - JSON: per-V `picard_iters`, `picard_converged`, `picard_failure_reason`,
     `min_theta_inner_per_iter`, `floor_hit_max_area_frac`,
     `h_closure_quality_per_iter`.
   - `converged_overall = grid_ok AND picard_converged AND no_strict_floor_hit`.
   - cd/pc plot masked by `converged_overall`.

## Tests (`tests/forward/bv/test_jithin_picard_closure.py`)

1. **`test_no_flux_equivalent_to_v1`** — set `k0_hat=0`, run Picard,
   assert: 1 iter to convergence, cd matches v1 k0=0 baseline to
   machine precision.

2. **`test_small_rate_anodic_regression`** — at V=+0.50 V, run Picard,
   assert: cd within 1% of v1's V=+0.50 cd (small correction expected).

3. **`test_theta_unity_recovers_levich`** — `a_vals_hat=[0,0,0]`, no
   counterions, Picard plateau within 1% of standard Levich
   `2F·D·c_bulk/L_eff`.

4. **`test_function_update_no_rebuild`** — after `xi_func.assign(...)`,
   `assemble(rate_form)` changes; no `solve()` invocation between assign
   and re-assemble.

5. **`test_fixed_point_residual_at_convergence`** — at converged ξ at
   one test V with `tol_residual=1e-8`, assert
   `|ξ + R·I/D − c_b/θ_b| < 1e-6` (no θ factor!).

6. **`test_raw_rate_to_molar_flux_conversion`** — extract R via
   `mode="reaction", scale=1.0`, divide by `assemble(1·ds)`, multiply by
   stoich; assert matches Levich molar rate at θ=1 within 1%.

7. **`test_parallel_2e_4e_shared_supply`** — minimal 2-reaction config
   (R2e + R4e, both consume O₂), run Picard at one V, assert both
   reactions reference the same xi_func_O2 and R_O2_total = R2e + R4e
   matches the Picard target.

8. **`test_z_zero_gate_blocks_charged`** — try to enable Picard with a
   cathodic species `z=+1`, assert ValueError raised.

## Risk areas (updated)

1. **`make_run_ss` failure inside Picard.**  Snapshot/restore/halve-damping
   protocol covers transient failures.  Persistent failure → abort V.
2. **Packing floor hit under Picard.**  Hard failure in strict mode.
   Indicates closure assumption breakdown; need finer mesh or different
   closure form.
3. **ξ underflow.**  Floored at exp(−50); logged as "K·I·θ/D ratio
   extreme."  Picard treats as not-converged if persistent across iters.
4. **PDE state drift not captured by ξ residual.**  Added state-norm
   `dU_norm` check (point 17).
5. **Surface integrals must use `mean = ∫(...)·ds / ∫1·ds`.**  Tests
   verify against bare assembly (point 4).
6. **z=0 gate too narrow if future deck needs H+ closure substitute.**
   Override flag exists but tripwire by default.

## Expected outcomes (unchanged)

Smooth S-curve from kinetic onset to ~Levich plateau; no cliff.
Validates solver via independent code path.  If cliff appears: pre-
analysis was wrong, re-examine I integral.

## Out of scope (gated follow-ups)

- Full parallel 2e/4e Fig 4.36 run (after this plan + smoke test pass).
- Surface-coverage / Frumkin isotherm patch (path 2).
- Hard-sphere mixture D(packing) (path 3).
- H+ closure-form substitute (if h_closure_rel_err diagnostic > 25%).
- Runtime x-invariance check (DG0 binning; if x-variance suspected).
- Custom from-scratch Chebyshev spectral solver.
```

## Continued critique prompt

Review the updated plan and my responses to your earlier issues.
Push back on responses where I defended poorly — name which point.
Raise any new issues the updated plan creates. Re-issue any earlier
issue you don't think I addressed. Same numbered format and same
verdict line at the end:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN
