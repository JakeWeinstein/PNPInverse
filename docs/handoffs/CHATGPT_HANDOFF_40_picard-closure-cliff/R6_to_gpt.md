# Round 6: Counterreply on R5 state-management issues + revised PLAN.md

User extended the cap to 10 rounds.  All 6 R5 issues land — folded
into the now-revised PLAN.md (auto-revision after the cap-5 termination,
re-opened for continued critique).  Per-issue acks then full updated
artifact.

## Per-issue responses to R5

**Re R5#1** (warm_walk_phi rollback checkpoints still U-only; need
`ckpt_inner` / `ckpt_outer` to include ξ):
**Accept.**  PLAN.md §4 now states explicitly:
> Make `ckpt_inner` / `ckpt_outer` checkpoints **ξ-aware** when
> `picard_log_xi_funcs in ctx`: snapshot `(U, U_prev, xi)` tuple,
> restore all three on bisection rollback.

And the contract on `picard_run_ss` (§3) requires that every False
return restores `(U, U_prev, xi)` to entry state — so warm-walk's
own rollback only needs to handle the case where it explicitly
checkpoints (e.g., before trying a bisected substep) and rolls back
on failure.  Both layers cover state.

**Re R5#2** (anchor/k0/Stern ladder rollback also needs ξ):
**Accept.**  PLAN.md §5:
> Ladder rung rollback snapshots include ξ when
> `picard_log_xi_funcs in ctx`.

Applied at every `run_ss = make_run_ss_factory(...)` site in
`anchor_continuation.py` (10 sites enumerated in §5).  Snapshot
helpers in `closure_picard.py` (`snapshot_state` / `restore_state`)
operate on `(U, U_prev, xi)` tuple uniformly.

**Re R5#3** (`converged_overall = grid_converged AND all Picard wrap
calls returned True` is wrong because bisection creates failed attempts
before success):
**Accept — important.**  PLAN.md §6 now defines:
> **Per-V `converged_overall`:** True iff
> - `warm_walk_phi` returned True for this V, AND
> - The final accepted Picard result (corresponding to the accepted
>   target state) is `reason="converged"`.
> Failed intermediate Picard attempts during bisection are recorded
> as iter_history but do NOT individually fail the V.

The PicardResult list `ctx["_picard_run_ss_history"]` tracks all
attempts including bisection-rejected ones; only the *last accepted*
one (the one whose state is left in `(U, U_prev, xi)`) gates
convergence.

**Re R5#4** (picard_run_ss must contractually restore entry state on
any False return):
**Accept.**  PLAN.md §3 main factory contract now lists every False
return path explicitly with `restore entry_snap`:
- Step 4 (early `run_ss` failure)
- Step 5 (strict-floor failure)
- Step 5 (xi_floored_persistent)
- Step 5 (run_ss_failed_with_target after damping retries)
- Step 6 (max_picard_iters)

The contract section ends with:
> **Contract:** every False return of `picard_run_ss` MUST restore the
> entry state `(U, U_prev, xi)`.  Every True return leaves the converged
> state in `(U, U_prev, xi)`.  Picard history appended to
> `ctx["_picard_run_ss_history"]` regardless of outcome.

Test §13 (`test_picard_failure_restores_entry_state`) verifies this.

**Re R5#5** (need `if make_run_ss_factory is None: make_run_ss_factory
= make_run_ss` at every entry point):
**Accept.**  PLAN.md §4 (`warm_walk_phi` and `solve_grid_with_anchor`)
and §5 (`solve_anchor_with_continuation`) both state:
> Normalize: `if make_run_ss_factory is None: make_run_ss_factory =
> make_run_ss`.

at the function entry, before any usage.  Test §10
(`test_byte_equiv_omitted_vs_none`) covers the normalization.

**Re R5#6** (pinned-baseline test depends on untracked
`StudyResults/.../iv_curve.json`):
**Accept.**  PLAN.md Test §11 now reads:
> **`test_byte_equiv_self_generated_baseline`** — fixture that runs
> a minimal config (e.g., 3-V single-reaction config built in the
> test) with the BARE `make_run_ss` factory (pre-modification
> behavior); save baseline arrays inline in the test.  Then re-run
> same config through MODIFIED `solve_grid_with_anchor` with
> `make_run_ss_factory=None` (which normalizes to bare).  Assert
> identical to 1e-12 relative.  Uses committed-in-test fixture
> rather than untracked `StudyResults/*.json`.

The fixture is self-contained — test builds the config inline (small
3-V grid), runs once for baseline, runs again through modified code,
asserts identical.  No dependency on outside files.

## Updated artifact

The full revised PLAN.md (post-cap auto-revision incorporating all R1-R5
fixes) is at `.planning/jithin_picard_plan/PLAN.md`.  Quoted in full below
for in-session review.

```markdown
# Jithin Closure Outer-Picard Implementation Plan (FINAL — post-GPT-critique-loop-40)

> **Status:** Final after 5-round GPT critique loop (session 40).  All 71
> accepted issues across rounds 1–5 are folded in.

## Goal

Wrap `make_run_ss` (`Forward/bv_solver/grid_per_voltage.py:135`) with a
Picard factory that interleaves boundary-supply ξ updates between
steady-state Newton solves at *every* warm-walk substep and *every*
anchor/Stern/k₀ ladder rung.  Test whether the continuum-MPNP analog
of Jithin's Eq 4.31 boundary closure reproduces his Fig 4.36 cliff
in our gradient-form solver.

Scope:
- Single-R2e Jithin emulation (primary).
- Parallel 2e/4e shared-supply plumbing smoke test.

Out of scope: reversible/anodic signed closure; full parallel 2e/4e
Fig 4.36 deck run; H+ closure substitute; surface-coverage; D(packing);
runtime x-invariance check; custom Chebyshev spectral solver;
emergency ξ backoff before substep-failure declaration.

## Math

Convention:
- `c_b_hat` = bulk concentration nondim
- `θ_OHP`, `θ_b` = packing fraction at OHP / bulk
- `R_O2_hat ≥ 0` = mean molar consumption rate of O₂ at OHP, nondim
  = `Σ_j (−stoich[O₂, j]) · R_j_mean_hat`
  with `R_j_mean_hat = assemble(R_j · ds(em)) / assemble(1 · ds(em))`
- `I_hat = assemble((1/packing) · dx) / Lx_hat`
- Code BV cathodic rate: `R_j_hat = k₀_hat · c_cat_hat · exp(−α·n·η)`
  (cathodic → η<0)
- Supply variable `ξ = c_OHP_hat / θ_OHP` (R-space `fd.Function`)
- BV substitution: `log_c_cat = ln(packing) + xi_func_s`

Continuum closure (Eq A'), z=0 species:
  ξ = c_b_hat/θ_b − R_O2_hat · I_hat / D_O2_hat,   R_O2_hat ≥ 0

This is the continuum-MPNP **analog** of Jithin Eq 4.31 — structural
similarity asserted, term-by-term mapping NOT claimed.

Picard target (Eq B), semi-implicit, positivity-preserving:
  K_old = R_O2_hat / c_eff_hat_old    (c_eff_hat_old = θ_OHP · ξ_old)
  ξ_target = (c_b_hat/θ_b) / (1 + K_old · I_hat · θ_OHP / D_O2_hat)

R_O2 < 0 → ValueError (out of scope).

Damped log update:
  ξ_new = exp((1−α)·log(ξ_old) + α·log(max(ξ_target, ξ_floor)))

Residual (no θ_OHP factor):
  residual = ξ + R_O2_hat · I_hat / D_O2_hat − c_b_hat / θ_b

Pre-analysis prediction: smooth S-curve to ~Levich, NO cliff (continuum
correction ≈ 6%).  Run still valuable as solver-correctness validation.

## Implementation

### 1. `config.py` (+15 lines)
- `bv_picard_mode: bool = False`
- `bv_picard_strict_floor: bool = True`
- Validate: `bv_picard_mode` requires `bv_jithin_closure_form`,
  `bv_log_rate`, `formulation == "logc_muh"`.

### 2. `forms_logc_muh.py` (+60 lines)
- When `bv_jithin_closure_form=True` AND `bv_picard_mode=True`:
  - Validate cathodic species z=0 (raise otherwise).
  - Per species s: `xi_func_s = fd.Function(R_space)`, init
    `log(c_b_hat_s / θ_b_const)`.
  - Replace `log_c_cat = u_exprs[cat_idx]` with
    `log_c_cat = ln(packing) + xi_func_s`.
- When `bv_picard_mode=True`:
  - If `steric_active=False`: `packing = theta_inner = fd.Constant(1.0)`,
    `closure_theta_b = 1.0`.
  - Expose in ctx: `picard_log_xi_funcs`, `packing_expr`,
    `theta_inner_expr`, `closure_theta_b`, `closure_bulk_c_hat`,
    `closure_cathodic_species_set`, `closure_cathodic_stoich`,
    `closure_packing_floor`.

### 3. `closure_picard.py` (NEW, ~400 lines)

Dataclasses (frozen):
- `StateSnapshot(U, U_prev, xi)` — all numpy arrays, deep copies.
- `PicardIterRecord` — per-iter diagnostics including residual, step,
  state_norm, floor_hit_area_frac, h_closure_rel_err.
- `PicardResult(converged, n_iters, reason, iter_history, ...)`.

Helpers:
- `snapshot_state(ctx, xi_funcs) → StateSnapshot`
- `restore_state(ctx, xi_funcs, snap) → None`
- `compute_picard_diagnostics(ctx, electrode_marker, Lx_hat,
   packing_floor, theta_inner_expr) → dict`:
  - mean θ_OHP via surface ratio
  - I_hat via volume ratio (divided by Lx_hat)
  - R_j_mean_hat via surface ratio
  - floor_hit_area_frac via conditional(theta_inner ≤ floor·(1+1e-6))
- `compute_picard_target(c_b, θ_b, θ_OHP, R_O2, I, D_O2, xi_old)
   → ξ_target` (rejects R<0).

Main:
```python
def make_picard_run_ss(*, ctx, solver, of_cd,
    dt_init=0.25, dt_growth_cap=4.0, dt_max_ratio=20.0,
    ss_rel_tol=1e-4, ss_abs_tol=1e-8, ss_consec=4,
    xi_funcs, closure_theta_b, closure_bulk_c_hat,
    cathodic_species_set, cathodic_stoich, D_per_species_hat,
    packing_expr, theta_inner_expr, electrode_marker, Lx_hat,
    packing_floor,
    max_picard_iters=15, tol_residual=1e-3, tol_step=1e-3,
    tol_state=1e-4, damping_init=0.5, damping_min=0.05,
    max_damping_retries=3, strict_floor=True, floor_tol=1e-10,
    xi_floor=math.exp(-50),
) -> Callable[[int], bool]:
```

Returned `picard_run_ss(max_steps)` contract:
1. **Snapshot entry state:** `entry_snap = snapshot_state(ctx, xi_funcs)`.
2. Build `run_ss = make_run_ss(...)`.
3. `ok = run_ss(max_steps)`.
4. If False: append PicardResult `reason="run_ss_failed_before_picard_target"`
   to history, **restore entry_snap**, return False.
5. Picard loop iter 1..max:
   - Compute diagnostics.
   - If strict_floor and floor_hit > floor_tol: append
     `reason="packing_floored"`, **restore entry_snap**, return False.
   - If R<0: ValueError.
   - Compute ξ_target; check persistent xi_floor (≥2 consecutive):
     append `reason="xi_floored_persistent"`, **restore entry_snap**,
     return False.
   - Convergence: all four clear (residual, step, state_norm, run_ss=True)
     → append `reason="converged"`, **return True** (state stays).
   - Damped log update; assign new ξ.
   - Snapshot pre-solve `iter_snap`.
   - `ok = run_ss(max_steps)`.
   - On False with target: `restore_state(ctx, xi_funcs, iter_snap)`,
     halve damping, retry up to max_damping_retries.  Persistent fail:
     append `reason="run_ss_failed_with_target"`, **restore entry_snap**,
     return False.
6. Max iters: append `reason="max_picard_iters"`, **restore entry_snap**,
   return False.

**Contract:** every False return MUST restore `(U, U_prev, xi)` to
entry state.  No file I/O — all data via returned/stored structures.

### 4. `grid_per_voltage.py` (+30 lines)
- `warm_walk_phi(*, ..., make_run_ss_factory=None)`:
  - Normalize: `if make_run_ss_factory is None: make_run_ss_factory = make_run_ss`.
  - Build `run_ss = make_run_ss_factory(...)` with full kwargs forwarded.
  - `ckpt_inner` / `ckpt_outer` are ξ-aware when
    `picard_log_xi_funcs in ctx`: snapshot/restore `(U, U_prev, xi)`.
- `solve_grid_with_anchor(*, ..., make_run_ss_factory=None)`:
  - Normalize None → bare.
  - Pass through to `warm_walk_phi`.
  - Clear `ctx["_picard_run_ss_history"] = []` at start of each per-V.
  - Source pool: when ξ-aware, store `(U_snapshot, xi_snapshot, phi)`
    after `warm_walk_phi` returns True (Picard-converged state).

### 5. `anchor_continuation.py` (+30 lines)
- `solve_anchor_with_continuation(*, ..., make_run_ss_factory=None)`:
  - Normalize None → bare.
  - Every `run_ss = make_run_ss(...)` site (1033, 1112, 1196, 1413,
    1561, 1728, 1845, 1860, 1880, 1927) becomes
    `run_ss = make_run_ss_factory(...)`.
  - Ladder rung rollback snapshots include ξ when ξ-aware.
- `PreconvergedAnchor`: add defaulted
  `xi_snapshots: tuple = ()` (frozen-dataclass compatible).

### 6. `_run_jithin_closure_picard.py` (NEW, ~500 lines)
- `BV_PICARD_MODE=True`, PICARD_* constants set.
- Setup assertion: confirm Stern + bulk Dirichlet x-uniform.
- Build `make_picard_run_ss_factory` partial capturing Picard-specific args.
- All stages use the factory.
- Per-V callback reads `ctx["_picard_run_ss_history"]` for JSON.
- **`converged_overall`** per V: `warm_walk_phi True AND final accepted
  Picard result reason="converged"`.  Failed intermediate bisection
  attempts do NOT individually fail the V.

## Tests (14 in `tests/forward/bv/test_jithin_picard_closure.py`)

1. `test_no_flux_disabled_reaction` — disabled reaction, 1 iter, ξ
   unchanged.
2. `test_function_update_no_rebuild` — enabled reaction, assert rate
   scales linearly with assign-to-xi.
3. `test_weak_cathodic_low_rate_regression` — V=+0.60, match v1 ±1%.
4. `test_theta_unity_recovers_levich` — `a_vals=[0,0,0]`, Levich ±1%.
5. `test_fixed_point_residual_at_convergence` — tol_residual=1e-8,
   residual `< 1e-6`.
6. `test_rate_to_mean_flux_conversion` — surface mean correctly computed.
7. `test_parallel_2e_4e_shared_supply` — run to convergence, assert
   shared xi_func_O2 + closure satisfied.
8. `test_z_nonzero_cathodic_species_rejected` — ValueError at form build.
9. `test_negative_R_O2_rejected` — ValueError in target compute.
10. `test_byte_equiv_omitted_vs_none` — identical results.
11. `test_byte_equiv_self_generated_baseline` — self-contained fixture,
    NOT untracked StudyResults files.
12. `test_U_prev_restored_on_picard_rollback` — manual injection test.
13. `test_picard_failure_restores_entry_state` — every False return
    restores (U, U_prev, xi).
14. `test_picard_history_captures_all_attempts` — bisection scenario
    produces multiple PicardResults; final accepted one drives convergence.

## Risk areas (unchanged from v5)
[...elided for brevity; see PLAN.md...]

## Expected outcomes
Smooth S-curve to ~Levich, no cliff.  Solver-correctness validation
via independent Picard+Newton code path.  If cliff: pre-analysis wrong.

## Out of scope (gated)
Reversible/anodic signed closure; full Fig 4.36 deck; H+ substitute;
surface coverage; D(packing); runtime x-invariance; Chebyshev spectral;
emergency ξ backoff.
```

## Continued critique prompt

Review the updated plan and my responses to R5.  Push back on remaining
holes — corner cases in the contract, test gaps, anything you can find.
Same numbered format and verdict line:

  VERDICT: APPROVED
  VERDICT: ISSUES_REMAIN
