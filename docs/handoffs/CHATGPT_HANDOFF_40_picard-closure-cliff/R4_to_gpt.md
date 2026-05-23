# Round 4: Counterreply on the v3 plan

All 12 issues land.  Point 1 is the structural one — Picard has to be a
wrapper around `run_ss` itself, not invoked at warm-walk convergence.
The rest are concrete corrections.  Per-issue responses + plan v4.

## Per-issue responses

**Re point 1** (Picard insertion still too late; failed substeps bypass
it; need to wrap `run_ss` itself):
**Accept — fundamental fix.**  Confirmed `grid_per_voltage.py:135 makes
run_ss`; called at every warm-walk substep at lines 613, 686, 734, 825, 1112.
The right insertion is: wrap `make_run_ss` to return a `picard_run_ss`
that does `run_ss → Picard update → run_ss → ...` until ξ-converged
(or max-iters), THEN returns success/failure to the warm-walk caller.

Plan v4 introduces `make_picard_run_ss(ctx, sp, solver, of_cd, ...,
picard_kwargs)` that returns a callable with the same signature as
`run_ss` (`(max_steps) -> bool`).  Inside:
  - Call `run_ss(max_steps)` (with packed-in solver, of_cd, etc.).
  - If False: return False (let warm-walk bisect).
  - If True: Picard inner loop — compute ξ_target, update xi_funcs,
    re-call `run_ss(max_steps)`, repeat until ξ-converged or max iters.
  - On Picard non-convergence: return False (let warm-walk bisect).
  - On Picard convergence: return True.

Every existing `run_ss(...)` call in `grid_per_voltage.py` is replaced
with the new wrapper when `picard_mode=True`, else uses bare `run_ss`
(byte-equivalent — addresses point 12).

**Re point 2** (first-iter retry on damping alone is futile if no
target was computed):
**Accept.**  Refined failure protocol:
  - If `run_ss` fails BEFORE any Picard target was computed (i.e., on
    the very first inner `run_ss` call): return False immediately —
    let warm-walk bisect the voltage substep.  Don't retry with damping.
  - If `run_ss` fails AFTER a Picard target was computed (subsequent
    inner iters): roll back ξ to last successful value, halve damping,
    retry up to `max_damping_retries=3`.  Persistent fail → return False.
  - Damping is only meaningful when there's a previous ξ to roll back to.

**Re point 3** (`picard_iterate` doesn't have access to `solver`, `of_cd`,
SS knobs):
**Accept.**  Restructure: the wrapping happens at the caller site
(`grid_per_voltage.py`), where these objects are already in scope.
The Picard logic is a thin closure capturing them.  Specifically, the
new signature:

  `make_picard_run_ss(*, ctx, solver, of_cd,
                      ss_rel_tol, ss_abs_tol, ss_consec,
                      dt_*, max_ss_steps,  # all passed explicitly
                      xi_funcs, closure_theta_b, closure_bulk_c_hat,
                      packing_expr, theta_inner_expr,
                      electrode_marker, Lx_hat, D_O2_hat, stoichiometry,
                      max_picard_iters, tol_residual, tol_step, tol_state,
                      damping_init, damping_min, max_damping_retries,
                      strict_floor, floor_tol)`

All `make_run_ss` knobs forwarded explicitly to the inner `run_ss`.
The Picard knobs are separate.  No silent defaults.

**Re point 4** (`PreconvergedAnchor` is frozen; extension must be
concrete):
**Accept.**  Use frozen-dataclass `replace`-friendly extension:

  ```python
  @dataclass(frozen=True)
  class PreconvergedAnchor:
      # ... existing fields ...
      xi_snapshots: tuple[tuple[int, tuple[float, ...]], ...] = ()
      # tuple of (species_idx, ξ_dof_values) pairs;
      # default () keeps byte-equivalence for non-Picard callers.
  ```

`isinstance(anchor, PreconvergedAnchor)` checks still pass.  Add
`snapshot_xi(xi_funcs) → tuple[tuple[int, tuple[float, ...]], ...]`
and `restore_xi(xi_funcs, snapshots) → None` helpers in
`closure_picard.py`.

**Re point 5** (snapshot must be after Picard convergence, not after
plain warm-walk):
**Accept.**  In each warm-walk success branch in
`grid_per_voltage.py:613+`, the snapshot is taken from `ctx["U"]` AFTER
the wrapping `picard_run_ss` returned True (which only happens after
ξ converged).  Equivalent change for the xi snapshot via
`snapshot_xi(ctx["picard_log_xi_funcs"])`.  Both packed into the
source pool entry.

**Re point 6** (R_s_hat ≤ 0 case is wrong for net production):
**Accept.**  For this plan iteration, **explicitly reject `R_s_hat < 0`
in Picard mode**.  Raise ValueError if encountered.  Plan scope is
irreversible cathodic O₂ only.  For future reversible cases, signed
closure derivation is its own task — not in this plan.  Comment in code:

  ```python
  if R_s_hat < 0.0:
      raise ValueError(
          f"bv_picard_mode: closure substitute supports irreversible "
          f"cathodic only (R_O2_hat >= 0).  Got R={R_s_hat} for "
          f"species {s}.  For reversible/anodic, implement signed "
          f"closure (out of scope for plan v4)."
      )
  ```

**Re point 7** (`k0=0` test conflicts with continuation helpers):
**Accept.**  `test_no_flux_equivalent_to_v1` becomes a direct form/unit
test:
  - Build the form with a disabled reaction (`enabled: False` in the
    rxn dict, which sets `R_j = fd.Constant(0.0)` per
    `forms_logc_muh.py:543-545`).
  - Skip the continuation entirely (no `solve_anchor_with_continuation`).
  - Solve at one V with the disabled reaction, assert Picard target equals
    initial ξ (no flux → no correction); assert 1 Picard iter.

**Re point 8** (V=+0.50 V is cathodic — E_eq=0.695, mislabel):
**Accept** — embarrassing.  Rename to `test_weak_cathodic_low_rate_regression`.
Use V=+0.60 V (still cathodic but η small).  Or set a value clearly anodic
of E_eq for the truly anodic test (out of plan scope — separate test if
reversible Picard is implemented later).

**Re point 9** (`inner(U, U)` on mixed Function is unit-mixed and
suspect):
**Accept.**  Use DOF-array norm:

  ```python
  with ctx["U"].dat.vec_ro as v_now, U_prev_picard.dat.vec_ro as v_prev:
      dU_inf = max(abs(a - b) for a, b in zip(v_now.array_r, v_prev.array_r))
      U_inf  = max(abs(a) for a in v_now.array_r)
  state_norm = dU_inf / max(U_inf, 1e-12)
  ```

L∞-norm on the raw DOF arrays.  Unit-clean (DOF values are all of the
same scale in log-c primary).  Tolerance: `tol_state = 1e-4`.

**Re point 10** (ξ underflow / floor handling contradictory):
**Accept** — clarify:
  - In strict-validation mode (default for `bv_picard_mode=True`):
    persistent ξ floor (≥2 consecutive iters at floor) → return
    `picard_converged=False, reason="xi_floored_persistent"`.
  - Non-strict mode (opt-in for floor-sensitivity experiments): floor
    is logged but iteration continues; final `picard_converged` reflects
    the residual/step/state convergence regardless of floor.

The Jithin emulation runs in strict-validation mode.

**Re point 11** (floor-hit "> 0" too brittle):
**Accept.**  Define:

  ```python
  floor_hit_form = fd.conditional(
      ctx["theta_inner_expr"] <= fd.Constant(packing_floor * (1.0 + 1e-6)),
      fd.Constant(1.0), fd.Constant(0.0)
  )
  total_measure = fd.assemble(fd.Constant(1.0) * fd.dx)
  floor_hit_area_frac = fd.assemble(floor_hit_form * fd.dx) / total_measure
  ```

Strict default: `floor_tol = 1e-10` (essentially zero, but
tolerates rounding).  Failure on `floor_hit_area_frac > floor_tol`.

**Re point 12** (byte-equivalence guarantee + smoke test):
**Accept.**  Add `tests/forward/bv/test_solve_grid_with_anchor_byte_equiv.py`:
  - Build any existing run config (e.g., v10b smoke).
  - Run `solve_grid_with_anchor(...)` with and without
    `picard_iterate_fn` arg (omitting/None vs setting).
  - Both runs called twice, identical except for the arg.
  - Assert: per-V `(method, U_snapshot)` equal byte-for-byte
    (using `np.array_equal`) when the arg is None/omitted.

The wrap-or-bare switch in `picard_run_ss` factory ensures the wrap
function is identical to plain `run_ss` when `picard_mode=False`.

## Updated artifact (v4 — full restatement)

```markdown
# Jithin Closure Outer-Picard Implementation Plan (v4, post-GPT-round-3)

## Goal

Implement a Picard wrap *around `run_ss`* (not around `warm_walk_phi`)
so the Picard inner loop interleaves at every warm-walk substep, before
the substep declares failure.  Test whether the continuum-MPNP analog
of Jithin Eq 4.31 closure reproduces his Fig 4.36 cliff.

Scope: irreversible cathodic single-R2e Jithin emulation + a parallel
2e/4e plumbing smoke test.  Reversible/anodic, full Fig 4.36 deck:
follow-up.

## Math (unchanged from v3, code convention)

  ξ ≡ c_OHP_hat / θ_OHP        (R-space supply variable)
  c_eff_hat (in BV) = θ_OHP · ξ = exp(ln(packing) + ln(ξ))
  log_c_cat = ln(packing) + xi_func        (used in BV rate UFL)

  Eq A' (continuum closure):
    ξ = c_b_hat/θ_b − R_O2_hat · I_hat / D_O2_hat,   R_O2_hat ≥ 0

  Eq B (semi-implicit Picard target):
    K_old = R_O2_hat / c_eff_hat_old
    ξ_target = (c_b_hat/θ_b) / (1 + K_old · I_hat · θ_OHP_avg / D_O2_hat)

  Residual (for tests, diagnostics):
    residual = ξ + R_O2_hat · I_hat / D_O2_hat − c_b_hat / θ_b
    (No θ_OHP factor — cancels via R = k₀ · θ_OHP · ξ · exp(−αnη))

Damped log-space update:
  ξ_new = exp( (1−α)·log(ξ_old) + α·log(max(ξ_target, ξ_floor)) )
  α default 0.5, ξ_floor = exp(-50).

Failure-mode signed-R policy:
  R_O2_hat < 0 in `bv_picard_mode` → raise ValueError (out of scope).

## Implementation

### Files modified

1. **`Forward/bv_solver/config.py`** (~15 lines)
   - Add `bv_picard_mode: bool` (default False).
   - Add `bv_picard_strict_floor: bool` (default True, only consumed
     when `bv_picard_mode=True`).
   - Validate:
     - `bv_picard_mode=True` requires `bv_jithin_closure_form=True`,
       `bv_log_rate=True`, `formulation == "logc_muh"`.

2. **`Forward/bv_solver/forms_logc_muh.py`** (~60 lines)
   - When `bv_jithin_closure_form=True` AND `bv_picard_mode=True`:
     - Collect distinct cathodic species across reactions; validate each
       has `z=0` (ValueError otherwise).
     - Per species s: `xi_func_s = fd.Function(R_space, name="picard_log_xi_sp{s}")`,
       initialized to `log(c_b_hat_s / θ_b_const)`.
     - Replace `log_c_cat = u_exprs[cat_idx]` with
       `log_c_cat = ln(packing) + xi_func_s`.
     - Expose in ctx:
       - `picard_log_xi_funcs: dict[int, fd.Function]`
       - `packing_expr` (the floored UFL `packing` expression)
       - `theta_inner_expr` (the unfloored `theta_inner` UFL expression)
       - `closure_theta_b: float`
       - `closure_bulk_c_hat: dict[int, float]`
       - `closure_cathodic_species_set: frozenset[int]`
       - `closure_cathodic_stoich: dict[int, dict[int, int]]`
         (species → (rxn_idx → stoich))
   - When `bv_picard_mode=True` AND `steric_active=False`:
     define `packing = theta_inner = fd.Constant(1.0)` and set
     `closure_theta_b = 1.0` so the ideal-mixture test path works.

3. **`Forward/bv_solver/closure_picard.py`** (NEW, ~350 lines)
   - `@dataclass PicardIterRecord`: per-iter snapshot.
   - `@dataclass PicardResult`: `converged`, `n_iters`, `reason`,
     `final_residual_per_species`, `final_step`, `final_state_norm`,
     `iter_history`, `min_theta_inner_per_iter`, `floor_hit_max`,
     `h_closure_relative_error`.
   - `snapshot_xi(xi_funcs) → tuple`, `restore_xi(xi_funcs, snap) → None`.
   - `compute_picard_diagnostics(ctx, electrode_marker, Lx_hat) → dict`:
     - `theta_OHP = assemble(packing_expr · ds(em)) / assemble(1·ds(em))`
     - `I_hat = assemble((1/packing_expr) · dx) / Lx_hat`
     - `R_j_mean_hat = assemble(R_j_form) / assemble(1·ds(em))` per reaction
     - Floor-hit area fraction (Eq from re point 11 above).
   - `compute_picard_target(c_b, θ_b, θ_OHP, R_O2, I, D_O2,
        xi_old) → ξ_target`:
     - Reject `R_O2 < 0`.
     - Apply Eq B if `R_O2 > 0`, else `ξ_target = c_b/θ_b`.
   - `make_picard_run_ss(*, ctx, solver, of_cd, ss_rel_tol, ss_abs_tol,
        ss_consec, dt_initial, dt_max, max_ss_steps,
        xi_funcs, closure_theta_b, closure_bulk_c_hat,
        cathodic_species_set, cathodic_stoich, D_per_species_hat,
        packing_expr, theta_inner_expr, electrode_marker, Lx_hat,
        max_picard_iters=15, tol_residual=1e-3, tol_step=1e-3,
        tol_state=1e-4, damping_init=0.5, damping_min=0.05,
        max_damping_retries=3, strict_floor=True, floor_tol=1e-10,
        xi_floor=math.exp(-50))`:
     - Returns `picard_run_ss(max_steps: int) → bool` closure.
     - Inside:
       - Snapshot xi.
       - Call inner `run_ss = make_run_ss(...)`.
       - `ok = run_ss(max_steps)`. If False: return False (warm-walk
         bisects).  No damping retry — no target yet.
       - Picard loop (iter 1..max):
         - Compute diagnostics, ξ_target per species.
         - Check strict_floor: if hit > floor_tol, set
           reason="packing_floored", return False.
         - Check residual + step + state-norm; if all clear: return True.
         - Damped update; assign new xi.
         - Snapshot U/xi before re-solve.
         - `ok = run_ss(max_steps)`.  If False: restore snapshot, halve
           damping, retry (max max_damping_retries).  Persistent fail:
           reason="run_ss_failed_with_target", return False.
       - Max iters without convergence: reason="max_picard_iters", return False.
   - All diagnostics written to a Picard-trajectory log file in
     run output dir; PicardResult also returned to the caller for
     per-V JSON capture.

4. **`Forward/bv_solver/grid_per_voltage.py`** (~30 lines)
   - Add opt-in `picard_make_run_ss_fn: Optional[Callable] = None` arg.
   - Inside `solve_grid_with_anchor`, the existing `make_run_ss(...)`
     call becomes:
     ```python
     run_ss = (picard_make_run_ss_fn(...) if picard_make_run_ss_fn
               else make_run_ss(...))
     ```
   - When None (the default for all existing callers): byte-equivalent
     to current behavior.
   - Source pool entries extended to `(U_snapshot, xi_snapshot, phi)`
     when `picard_mode=True`; otherwise legacy `(U_snapshot, phi)` —
     the xi entry is None and skipped in restore.

5. **`Forward/bv_solver/anchor_continuation.py`** (~20 lines)
   - `PreconvergedAnchor` adds defaulted `xi_snapshots: tuple = ()` field.
   - Frozen dataclass `replace`-friendly.
   - `solve_anchor_with_continuation` and `solve_grid_with_anchor`
     accept optional Picard wrapper arg, default None → byte-equivalent.

6. **`scripts/studies/_run_jithin_closure_picard.py`** (NEW, ~500 lines)
   - Fork `_run_jithin_closure_exact.py`.
   - `BV_PICARD_MODE=True`, PICARD_* tolerances.
   - Setup assertion: confirm Stern BC + bulk Dirichlet x-uniform.
   - Build `make_picard_run_ss_fn` closure with all required args
     captured from local scope.
   - Stage 1: `solve_anchor_with_continuation(..., make_run_ss_fn=
     make_picard_run_ss_fn)` — anchor uses Picard wrapper.
   - Stage 2: Stern bump ladder — each rung re-anchors via picard wrapper.
   - Stage 3: `solve_grid_with_anchor(..., picard_make_run_ss_fn=
     make_picard_run_ss_fn)` — every per-V substep uses Picard wrapper.
   - JSON output: per-V `picard_iters`, `picard_converged`,
     `picard_failure_reason`, `min_theta_inner_per_iter`,
     `floor_hit_max_area_frac`, `h_closure_quality_per_iter`,
     `picard_n_iters`.
   - `converged_overall = grid_converged AND picard_converged`.

## Tests (`tests/forward/bv/test_jithin_picard_closure.py`)

1. **`test_no_flux_disabled_reaction`** — build form with reaction
   `enabled=False` (R_j → 0); solve once; assert Picard converges in
   1 iter, ξ = c_b/θ_b (no correction).

2. **`test_weak_cathodic_low_rate_regression`** — V=+0.60 V (just
   cathodic of E_eq=0.695); assert cd matches v1 closure-exact run
   within 1%.

3. **`test_theta_unity_recovers_levich`** — `a_vals_hat=[0,0,0]`, no
   counterions → `steric_active=False` path; assert Picard plateau
   within 1% of standard Levich.

4. **`test_function_update_no_rebuild`** — `xi_func.assign(...)`,
   `assemble(rate_form)` changes; no `solver.solve()` invocation needed.

5. **`test_fixed_point_residual_at_convergence`** — run Picard with
   `tol_residual=1e-8`; assert
   `|ξ + R·I/D − c_b/θ_b| < 1e-6` (no θ factor).

6. **`test_rate_to_mean_flux_conversion`** — assemble `R_j_form` and
   `assemble(1·ds)`; assert `R_j_mean = R_total/ds_area`; assert
   stoich-multiplied molar flux matches Levich at θ=1.

7. **`test_parallel_2e_4e_shared_supply`** — minimal 2-reaction config,
   one Picard iter; assert both reactions reference the same xi_func_O2;
   assert `R_O2_total = R_2e + R_4e` matches the Picard target formula.

8. **`test_z_nonzero_cathodic_species_rejected`** — try `bv_picard_mode=True`
   with a cathodic species `z=+1`; assert ValueError raised.

9. **`test_negative_R_O2_rejected`** — synthetic test where R<0 is forced;
   assert ValueError raised in Picard target compute.

10. **`test_solve_grid_with_anchor_byte_equiv`** — run existing v10b smoke
    config twice (with `picard_make_run_ss_fn=None` and arg omitted);
    assert per-V `(method, U_snapshot)` byte-equal via `np.array_equal`.

## Risk areas

1. **Inner `run_ss` failure WITHOUT Picard target.**  Return False
   immediately, let warm-walk bisect.  No damping retry.
2. **Inner `run_ss` failure WITH Picard target.**  Roll back ξ, halve
   damping, retry; max 3 retries.
3. **Persistent ξ floor.**  Strict mode: 2 consecutive floor iters →
   not converged.  Non-strict: continue, mark in JSON.
4. **Packing floor hit.**  Strict mode: hard failure.  Tolerance 1e-10
   on area fraction.
5. **State norm on mixed Function.**  Use L∞ on raw DOF arrays.
6. **Byte-equivalence broken by accidental code path change.**  Smoke
   test asserts identical (method, U) when Picard not wired.
7. **R<0 by mistake.**  Explicit ValueError in target computation.

## Expected outcomes

Per pre-analysis: smooth S-curve from kinetic onset to ~Levich plateau,
NO cliff.  Picard rescues all v1-failed deep-cathodic points and
converges to a clean curve.  Validates solver via independent code path.

If cliff appears: pre-analysis was wrong; re-examine I integral and
θ(y) profile.

## Out of scope

- Reversible/anodic Picard (signed closure derivation).
- Full parallel 2e/4e Fig 4.36 deck run.
- H+ closure-form substitute.
- Surface-coverage / Frumkin (path 2).
- Hard-sphere D(packing) (path 3).
- Runtime x-invariance check (DG0 binning).
- Custom Chebyshev spectral solver.
```

## Continued critique prompt

Review the updated plan and my responses to your earlier issues.
Push back on responses where I defended poorly — name which point.
Raise any new issues the updated plan creates. Re-issue any earlier
issue you don't think I addressed. Same numbered format and same
verdict line at the end:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN
