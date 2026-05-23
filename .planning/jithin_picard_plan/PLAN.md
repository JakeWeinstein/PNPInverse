# Jithin Closure Outer-Picard Implementation Plan (FINAL — post-GPT-critique-loop-40 APPROVED)

> **Status:** APPROVED at round 7 of GPT critique loop (session 40, see
> `docs/handoffs/CHATGPT_HANDOFF_40_picard-closure-cliff/`).  All 81
> accepted issues across rounds 1–7 are folded in.  Three R7 nitpicks
> (non-blocking) addressed below.

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

Out of scope (explicit gated follow-ups):
- Reversible/anodic signed closure (own plan).
- Full parallel 2e/4e Fig 4.36 deck run (gated on this plan).
- H+ closure-form substitute (if h_closure_rel_err diagnostic > 25%).
- Surface-coverage / Frumkin isotherm (path 2 candidate).
- Hard-sphere D(packing) (path 3 candidate).
- Runtime x-invariance check (DG0 binning).
- Custom Chebyshev spectral solver.
- Emergency ξ backoff before substep-failure declaration.

## Math

### Convention (code-aligned)

- `c_b_hat` = bulk concentration nondim (= `c_O2_bulk_mol_m3 / C_SCALE`)
- `θ_OHP`, `θ_b` = packing fraction at OHP / bulk
- `R_O2_hat ≥ 0` = mean molar consumption rate of O₂ at OHP, nondim
  = `Σ_j (−stoich[O₂, j]) · R_j_mean_hat`
  with `R_j_mean_hat = assemble(R_j · ds(em)) / assemble(1 · ds(em))`
- `I_hat = assemble((1/packing) · dx) / Lx_hat`
- `D_O2_hat` = O₂ diffusivity nondim
- `η = (V_app − E_eq) / V_T` (cathodic → η<0)
- Code BV cathodic rate: `R_j_hat = k₀_hat · c_cat_hat · exp(−α·n·η)`
  (grows as V drops)
- Supply variable `ξ` = `c_OHP_hat / θ_OHP` (R-space `fd.Function`)
- BV substitution: `log_c_cat = ln(packing) + xi_func_s`
  (keeps `packing` inline / Newton-coupled; ξ Picard-controlled)

### Closure equations

**Continuum boundary closure (Eq A')** for z=0 species, derived from
steady-state `J = -D · θ · ∇(c/θ)` integrated y=0 to y=L:

  ξ = c_b_hat/θ_b − R_O2_hat · I_hat / D_O2_hat,    R_O2_hat ≥ 0

This is the **continuum-MPNP analog** of Jithin's Eq 4.31 (equilibrium
term + flux-supply correction).  We do NOT claim term-by-term mapping
to his κ_5·φ·g spectral form; we test whether the continuum-derived
structural analog reproduces his cliff.

**Picard target — semi-implicit, positivity-preserving (Eq B):**

  K_old = R_O2_hat / c_eff_hat_old    (c_eff_hat_old = θ_OHP · ξ_old)
  ξ_target = (c_b_hat/θ_b) / (1 + K_old · I_hat · θ_OHP / D_O2_hat)

For `R_O2_hat < 0` (reversible/anodic): **raise ValueError**.  Out of
scope for this plan; signed closure is its own derivation.

**Damped log-space update:**

  ξ_new = exp( (1−α)·log(ξ_old) + α·log(max(ξ_target, ξ_floor)) )

Default α=0.5, ξ_floor = exp(−50) ≈ 2e−22.

**Residual (for tests + diagnostics, NO θ_OHP factor):**

  residual = ξ + R_O2_hat · I_hat / D_O2_hat − c_b_hat / θ_b

The θ_OHP cancels because `R = k₀ · θ_OHP · ξ · exp(-αnη)` substituted
into Eq A' already contains the θ_OHP factor through R.

### Limits

- Kinetic regime (K·I·θ/D << 1): ξ → c_b_hat/θ_b
  → R = k₀ · θ_OHP · c_b/θ_b · exp(-αnη)   (= v1 equilibrium closure)
- Transport regime (K·I·θ/D >> 1):
  ξ → D_O2_hat / (K · I · θ_OHP_avg)
  → R → c_b_hat · D_O2_hat / (θ_b · I_hat)  (= continuum Levich)

### Pre-analysis prediction

For our geometry (L=10 µm, λ_D ≈ 0.5 nm, sat layer ~few nm, θ_OHP≈0.034,
θ_b≈0.94):

  I_hat ≈ L_hat/θ_b + tiny_sat_correction (≈ 6%)
  Continuum Levich ≈ standard Levich

**Predicted outcome:** smooth S-curve from kinetic onset to ~Levich
plateau, **NO cliff**.  If true → rules out continuum-Bikerman as cliff
mechanism; pivot to surface-coverage (Strmcnik non-covalent cation
effect, path 2).  If unexpected sub-Levich plateau / cliff: re-examine
I integral and θ(y) profile assumptions.

## Implementation

### 1. `Forward/bv_solver/config.py` (+15 lines)

- `bv_picard_mode: bool = False`
- `bv_picard_strict_floor: bool = True` (consumed only when
  `bv_picard_mode=True`)
- Validate: `bv_picard_mode=True` requires
  `bv_jithin_closure_form=True`, `bv_log_rate=True`,
  `formulation == "logc_muh"`.

### 2. `Forward/bv_solver/forms_logc_muh.py` (+60 lines)

When `bv_jithin_closure_form=True` AND `bv_picard_mode=True`:

- Collect distinct cathodic species across reactions; validate each
  has `z=0` (raise ValueError otherwise).
- Per species s: `xi_func_s = fd.Function(R_space, name=f"picard_log_xi_sp{s}")`,
  initialized to `log(c_b_hat_s / θ_b_const)`.
- In the BV log-rate construction, replace
  `log_c_cat = u_exprs[cat_idx]` with
  `log_c_cat = ln(packing) + xi_func_s`.

When `bv_picard_mode=True`:
- If `steric_active=False`: define `packing = theta_inner = fd.Constant(1.0)`
  and `closure_theta_b = 1.0` so the θ=1 test path works.
- Expose in ctx (always under Picard mode):
  - `picard_log_xi_funcs: dict[int, fd.Function]`
  - `packing_expr` (the floored UFL `packing` expression)
  - `theta_inner_expr` (the unfloored `theta_inner` UFL expression)
  - `closure_theta_b: float`
  - `closure_bulk_c_hat: dict[int, float]`
  - `closure_cathodic_species_set: frozenset[int]`
  - `closure_cathodic_stoich: dict[int, dict[int, int]]`
    (species → rxn_idx → stoich)
  - `closure_packing_floor: float`

### 3. `Forward/bv_solver/closure_picard.py` (NEW, ~450 lines)

Dataclasses (frozen):
- `@dataclass(frozen=True) PicardConfig`: ctx-invariant Picard knobs:
  `D_per_species_hat: dict[int, float]`, `electrode_marker: int`,
  `Lx_hat: float`, `max_picard_iters`, `tol_residual`, `tol_step`,
  `tol_state`, `damping_init`, `damping_min`, `max_damping_retries`,
  `strict_floor: bool`, `floor_tol`, `xi_floor`.
  Note: `D_per_species_hat` is treated as deck-invariant for this
  feature.  If future code mutates `logD_funcs` mid-run, the factory
  asserts `ctx[D_per_species].value == picard_config.D_per_species_hat[s]`
  for each cathodic species at factory call time and raises if stale.
- `@dataclass(frozen=True) StateSnapshot`: `U_snap: tuple` (from
  existing `snapshot_U`), `xi_snap: tuple[tuple[int, tuple[float,
  ...]], ...]`.  Reuses existing serialization helpers; no parallel
  `.dat` path.
- `@dataclass(frozen=True) PicardIterRecord`: per-iter diagnostics
  (`iter`, `xi_per_species`, `R_per_species`, `R_O2_total`, `theta_OHP`,
  `I_hat`, `residual_per_species`, `step_per_species`, `state_norm`,
  `damping`, `floor_hit_area_frac`, `h_closure_rel_err`).
- `@dataclass(frozen=True) PicardResult`: `converged`, `n_iters`,
  `reason`, `iter_history: tuple[PicardIterRecord, ...]`, summary fields.

Helpers:
- `snapshot_state(ctx, xi_funcs) → StateSnapshot`:
  ```python
  return StateSnapshot(
      U_snap=snapshot_U(ctx["U"]),                  # existing helper
      xi_snap=snapshot_xi(xi_funcs),
  )
  ```
- `restore_state(ctx, xi_funcs, snap) → None`:
  ```python
  restore_U(snap.U_snap, ctx["U"], ctx["U_prev"])   # existing helper;
                                                    # also sets U_prev = U
  restore_xi(xi_funcs, snap.xi_snap)
  ```
- `snapshot_xi(xi_funcs) → tuple[tuple[int, tuple[float, ...]], ...]`.
- `restore_xi(xi_funcs, snap) → None`.

User-facing factory adapter (ctx-aware):
```python
def make_picard_run_ss_factory(picard_config: PicardConfig) -> Callable:
    """Returns a make_run_ss-compatible factory:
       (ctx, solver, of_cd, **ss_kwargs) -> Callable[[int], bool]
    The returned factory pulls ctx-specific Picard objects FRESH from
    the passed ctx, so it remains correct across rebuilt ctx instances
    (anchor, Stern rungs, per-V continuation).
    """
    def _factory(ctx, solver, of_cd, **ss_kwargs):
        # Assert D consistency to catch stale config:
        for s in ctx["closure_cathodic_species_set"]:
            assert (
                abs(picard_config.D_per_species_hat[s] -
                    float(np.exp(ctx["logD_funcs"][s].dat.data_ro[0])))
                < 1e-12
            ), f"PicardConfig.D[{s}] stale vs ctx logD"
        return make_picard_run_ss(
            ctx=ctx, solver=solver, of_cd=of_cd, **ss_kwargs,
            xi_funcs=ctx["picard_log_xi_funcs"],
            packing_expr=ctx["packing_expr"],
            theta_inner_expr=ctx["theta_inner_expr"],
            closure_theta_b=ctx["closure_theta_b"],
            closure_bulk_c_hat=ctx["closure_bulk_c_hat"],
            cathodic_species_set=ctx["closure_cathodic_species_set"],
            cathodic_stoich=ctx["closure_cathodic_stoich"],
            D_per_species_hat=picard_config.D_per_species_hat,
            electrode_marker=picard_config.electrode_marker,
            Lx_hat=picard_config.Lx_hat,
            packing_floor=ctx["closure_packing_floor"],
            **picard_config.runtime_kwargs(),
        )
    return _factory
```

Diagnostics + target functions:
- `compute_picard_diagnostics(ctx, electrode_marker, Lx_hat,
    packing_floor, theta_inner_expr) → dict`:
  - `theta_OHP = assemble(packing_expr · ds(em)) / assemble(1·ds(em))`
  - `I_hat = assemble((1/packing_expr) · dx) / Lx_hat`
  - `R_j_mean_hat = assemble(R_j_form · ds(em)) / assemble(1·ds(em))`
  - `floor_hit_area_frac = assemble(conditional(theta_inner ≤ packing_floor·(1+1e-6), 1, 0) · dx)
                            / assemble(1·dx)`
- `compute_picard_target(c_b, θ_b, θ_OHP, R_O2, I, D_O2, xi_old) → ξ_target`:
  rejects `R_O2 < 0` (ValueError); applies Eq B.

Main factory:
```python
def make_picard_run_ss(
    *,
    # make_run_ss-compatible kwargs (forwarded verbatim):
    ctx, solver, of_cd,
    dt_init=0.25, dt_growth_cap=4.0, dt_max_ratio=20.0,
    ss_rel_tol=1e-4, ss_abs_tol=1e-8, ss_consec=4,
    # Picard-specific:
    xi_funcs, closure_theta_b, closure_bulk_c_hat,
    cathodic_species_set, cathodic_stoich, D_per_species_hat,
    packing_expr, theta_inner_expr, electrode_marker, Lx_hat,
    packing_floor,
    max_picard_iters=15, tol_residual=1e-3, tol_step=1e-3,
    tol_state=1e-4, damping_init=0.5, damping_min=0.05,
    max_damping_retries=3, strict_floor=True, floor_tol=1e-10,
    xi_floor=math.exp(-50),
) -> Callable[[int], bool]:
    """Returns `picard_run_ss(max_steps)` with signature matching bare run_ss."""
```

Contract for `picard_run_ss(max_steps) -> bool`:

1. **Entry snapshot** (always): `entry_snap = snapshot_state(ctx, xi_funcs)`.
2. Build inner `run_ss = make_run_ss(ctx, solver, of_cd, dt_init,
   dt_growth_cap, dt_max_ratio, ss_rel_tol, ss_abs_tol, ss_consec)`.
3. `ok = run_ss(max_steps)`.
4. If False: append `PicardResult(reason="run_ss_failed_before_picard_target")`
   to `ctx["_picard_run_ss_history"]`, **restore entry_snap**, return False.
5. Picard loop (1..max_picard_iters):
   - Compute diagnostics.
   - **Strict floor check:** if `floor_hit_area_frac > floor_tol`:
     append `reason="packing_floored"`, **restore entry_snap**, return False.
   - **Negative-R check:** if any `R_s_hat < 0`, raise ValueError.
   - Compute ξ_target per species.
   - **ξ_floor underflow detection:** count consecutive iters where any
     species hits floor; if ≥2 consecutive, append
     `reason="xi_floored_persistent"`, **restore entry_snap**, return False.
   - **Convergence check (iter 0 vs iter N≥1):**
     - **iter 0** (first post-initial-solve check): no prior Picard iter
       state exists to compute state_norm.  Gate on three only:
       residual + step + run_ss_ok.  If all clear: `reason="converged"`,
       return True.
     - **iter N≥1**: gate on all four:
       - `|ξ + R·I/D − c_b/θ_b| < tol_residual` (per species)
       - `|log(ξ_new) − log(ξ_old)| < tol_step` (per species, damped)
       - `state_norm = L∞(U_after_current_run_ss − U_after_prev_picard_run_ss)
                       / max(L∞(U_after_current_run_ss), 1e-12) < tol_state`
         (Picard iter-to-iter, NOT entry-to-current.)
       - `run_ss returned True for this iter`
       If all clear: append `reason="converged"`, return True.
   - Damped log update; assign new ξ to xi_funcs.
   - Snapshot pre-solve state (`iter_snap`).
   - `ok = run_ss(max_steps)`.
   - On False **with target**: `restore_state(ctx, xi_funcs, iter_snap)`,
     halve damping, retry up to `max_damping_retries`.  Persistent fail:
     append `reason="run_ss_failed_with_target"`, **restore entry_snap**,
     return False.
6. **Max iters without convergence:** append `reason="max_picard_iters"`,
   **restore entry_snap**, return False.

**Contract:** every False return of `picard_run_ss` MUST restore the
entry state `(U, U_prev, xi)`.  Every True return leaves the converged
state in `(U, U_prev, xi)`.  Picard history appended to
`ctx["_picard_run_ss_history"]` regardless of outcome.

No file I/O in this module — all data via returned/stored structures.

### 4. `Forward/bv_solver/grid_per_voltage.py` (+30 lines)

- Add module-level helper:
  ```python
  def _normalize_make_run_ss_factory(factory):
      """Returns the supplied factory or bare make_run_ss when None.
      Exposed so unit tests can assert normalization semantics directly."""
      return make_run_ss if factory is None else factory
  ```
- `warm_walk_phi(*, ..., make_run_ss_factory=None)`:
  - Normalize via helper: `make_run_ss_factory = _normalize_make_run_ss_factory(make_run_ss_factory)`.
  - Replace `run_ss = make_run_ss(...)` with
    `run_ss = make_run_ss_factory(ctx=ctx, solver=solver, of_cd=of_cd,
        dt_init=dt_init, dt_growth_cap=dt_growth_cap,
        dt_max_ratio=dt_max_ratio,
        ss_rel_tol=ss_rel_tol, ss_abs_tol=ss_abs_tol,
        ss_consec=ss_consec)`.
  - Make `ckpt_inner` / `ckpt_outer` checkpoints **ξ-aware** when
    `picard_log_xi_funcs in ctx`: snapshot `(U, U_prev, xi)` tuple,
    restore all three on bisection rollback.

- `solve_grid_with_anchor(*, ..., make_run_ss_factory=None)`:
  - Normalize via `_normalize_make_run_ss_factory` helper.
  - Pass through to every `warm_walk_phi` call.
  - Clear `ctx["_picard_run_ss_history"] = []` at start of each per-V solve.
  - Source pool entries: when `picard_log_xi_funcs in ctx`, store
    `(U_snapshot, xi_snapshot, phi)`; on restore, reset both U and ξ.
  - The xi entry from snapshot is taken AFTER `warm_walk_phi` returns
    True (so xi is at the Picard-converged value).

### 5. `Forward/bv_solver/anchor_continuation.py` (+30 lines)

- `solve_anchor_with_continuation(*, ..., make_run_ss_factory=None)`:
  - Normalize via `_normalize_make_run_ss_factory` helper.
  - Every `run_ss = make_run_ss(...)` in the file (1033, 1112, 1196,
    1413, 1561, 1728, 1845, 1860, 1880, 1927) becomes
    `run_ss = make_run_ss_factory(...)` with identical kwargs.
  - Ladder rung rollback snapshots include ξ when
    `picard_log_xi_funcs in ctx`.

- `PreconvergedAnchor` (frozen dataclass):
  - Add defaulted `xi_snapshots: tuple[tuple[int, tuple[float, ...]], ...] = ()`.
  - Default empty tuple → byte-equivalent for non-Picard callers.
  - `isinstance(anchor, PreconvergedAnchor)` checks still pass.

### 6. `scripts/studies/_run_jithin_closure_picard.py` (NEW, ~500 lines)

Fork `_run_jithin_closure_exact.py`.  Changes:
- `BV_PICARD_MODE = True`
- `PICARD_MAX_ITERS = 15`, `PICARD_TOL_RESIDUAL = 1e-3`,
  `PICARD_TOL_STEP = 1e-3`, `PICARD_TOL_STATE = 1e-4`,
  `PICARD_DAMPING_INIT = 0.5`, `PICARD_DAMPING_MIN = 0.05`,
  `PICARD_MAX_DAMPING_RETRIES = 3`, `PICARD_STRICT_FLOOR = True`
- Setup assertion: confirm Stern BC + bulk Dirichlet are x-uniform and
  no x-localized source is wired (assert keys/no per-x perturbation
  in params).  Comment: "Picard closure uses scalar ξ per species; valid
  only when solution is x-invariant.  Runtime check via DG0-on-x binning
  is out of scope for v1; assert setup-level homogeneity here."
- Build `make_picard_run_ss_factory` as `functools.partial(
    make_picard_run_ss, xi_funcs=..., closure_theta_b=..., ...)`
  that captures all Picard-specific args.
- Stage 1: `solve_anchor_with_continuation(..., make_run_ss_factory=
  make_picard_run_ss_factory)`.
- Stage 2: Stern bump ladder — same factory.
- Stage 3: `solve_grid_with_anchor(..., make_run_ss_factory=
  make_picard_run_ss_factory)`.
- Per-V callback reads `ctx["_picard_run_ss_history"]` (the list
  of all PicardResults from all warm-walk substep attempts) and
  captures into JSON.
- **Per-V `converged_overall`:** True iff
  - `warm_walk_phi` returned True for this V, AND
  - The final accepted Picard result (corresponding to the accepted
    target state) is `reason="converged"`.
  Failed intermediate Picard attempts during bisection are recorded
  as iter_history but do NOT individually fail the V.

### Diagnostics (per V, in JSON)

- `cd_mA_cm2`, `pc_mA_cm2` (post-converged state)
- `c_O2_OHP_nondim`, `c_H2O2_OHP_nondim`, `c_H_OHP_nondim`, `phi_OHP_nondim`
- `picard_history: list[PicardResult]` — every wrap call's result
- `picard_converged_overall: bool` — per definition above
- `picard_total_iters_accepted: int` — iters in the accepted Picard run
- `picard_total_attempts: int` — len(picard_history)
- `picard_failure_reasons: list[str]` — reasons across all attempts
- `min_theta_inner_overall: float`
- `max_floor_hit_area_frac: float`
- `h_closure_quality_per_iter: list[float]`

## Tests (`tests/forward/bv/test_jithin_picard_closure.py`)

1. **`test_no_flux_disabled_reaction`** — build form with reaction
   `enabled=False` (R_j=0); single solve; Picard converges in 1 iter
   at ξ = c_b/θ_b.  Smoke test for Picard init.

2. **`test_function_update_no_rebuild`** — build form with `bv_picard_mode=True`,
   enabled reaction, k₀>0; assemble `rate_form` at `xi_func.assign(log(0.1))`
   → R_a; assemble at `xi_func.assign(log(0.01))` → R_b; assert
   `R_b / R_a ≈ 0.1 within 1e-9` (linearity in ξ).  Exercises
   `log_c_cat = ln(packing) + xi_func` end-to-end without rebuild.

3. **`test_weak_cathodic_low_rate_regression`** — V=+0.60 V (just
   cathodic of E_eq=0.695), Picard run; cd matches v1 closure-exact
   `iv_curve.json` at same V within 1%.

4. **`test_theta_unity_recovers_levich`** — `a_vals_hat=[0,0,0]`, no
   counterions (`steric_active=False` path), Picard plateau within 1%
   of standard Levich `2F·D·c_bulk/L_eff`.

5. **`test_fixed_point_residual_at_convergence`** — Picard with
   `tol_residual=1e-8`, assert `|ξ + R·I/D − c_b/θ_b| < 1e-6`.

6. **`test_rate_to_mean_flux_conversion`** — assert
   `R_j_mean = assemble(R_j · ds) / assemble(1·ds)`; stoich-multiplied
   molar flux matches Levich at θ=1 within 1%.

7. **`test_parallel_2e_4e_shared_supply`** — minimal 2-reaction config
   (R2e + R4e, both consume O₂), Picard **run to convergence**, then:
   (a) both reactions reference the SAME xi_func_O2; (b) at converged
   ξ, residual `|ξ + R_O2_total·I/D − c_b/θ_b| < 1e-3`; (c) `R_O2_total
   = R_2e_converged + R_4e_converged` matches Picard target.

8. **`test_z_nonzero_cathodic_species_rejected`** — try `bv_picard_mode=True`
   with cathodic species `z=+1`; assert ValueError raised at form build.

9. **`test_negative_R_O2_rejected`** — synthetic test where `R_O2_hat`
   is forced negative; assert ValueError raised in `compute_picard_target`.

10. **`test_byte_equiv_omitted_vs_none`** — `solve_grid_with_anchor`
    with `make_run_ss_factory` omitted vs explicit `=None`; assert
    identical per-V `(method, U_snapshot)` arrays.

11. **`test_byte_equiv_self_generated_baseline`** — fixture that runs
    a minimal config (e.g., 3-V single-reaction config built in the
    test) with the BARE `make_run_ss` factory (pre-modification
    behavior); save baseline arrays inline in the test.  Then re-run
    same config through MODIFIED `solve_grid_with_anchor` with
    `make_run_ss_factory=None` (which normalizes to bare).  Assert
    identical to 1e-12 relative.  Uses committed-in-test fixture
    rather than untracked `StudyResults/*.json`.

12. **`test_U_prev_restored_on_picard_rollback`** — manually inject
    a U_prev snapshot, force run_ss failure in a Picard iter, assert
    restored `U_prev` byte-equal to snapshot.

13. **`test_picard_failure_restores_entry_state`** — entry state with
    known U/U_prev/ξ; trigger a Picard failure (e.g.,
    `max_picard_iters=0` or persistent run_ss fail); assert after
    `picard_run_ss` returns False, ctx U/U_prev and xi_funcs match
    entry state exactly.

14. **`test_picard_history_captures_all_attempts`** — run a scenario
    that triggers warm-walk bisection (forcing multiple Picard wrap
    calls per V); assert `ctx["_picard_run_ss_history"]` contains
    one PicardResult per call, with the final accepted one's
    `reason="converged"`.

15. **`test_normalize_make_run_ss_factory_helper`** — unit-test the
    helper directly: `_normalize_make_run_ss_factory(None) is
    make_run_ss` (identity check on the exact imported callable);
    `_normalize_make_run_ss_factory(custom) is custom`.  Catches
    accidental refactor that breaks the None→bare contract.

16. **`test_non_picard_mode_does_not_touch_picard_keys`** — build form
    with `bv_picard_mode=False` (default); assert NONE of the
    **Picard-specific** ctx keys are present:
    ```python
    PICARD_ONLY_KEYS = {
        "picard_log_xi_funcs", "closure_theta_b", "closure_bulk_c_hat",
        "closure_cathodic_species_set", "closure_cathodic_stoich",
        "closure_packing_floor", "_picard_run_ss_history",
    }
    assert not (PICARD_ONLY_KEYS & set(ctx.keys()))
    ```
    Note: `packing_expr` and `theta_inner_expr` are NOT in this set —
    they may be exposed by other (non-Picard) diagnostics in the future
    without breaking this assertion.

## Risk areas

1. **Inner `run_ss` failure BEFORE Picard target** → False return
   (warm-walk bisects).  Documented limitation; no Picard rescue
   without prior target.
2. **Inner `run_ss` failure AFTER Picard target** → rollback U/U_prev/ξ,
   halve damping, retry (max 3).  Persistent fail → False, entry state
   restored.
3. **Persistent ξ floor (≥2 consecutive iters)** → not-converged in
   strict mode; entry state restored.
4. **Packing floor hit > floor_tol (default 1e-10)** → not-converged in
   strict mode; entry state restored.
5. **State norm**: L∞ on raw DOF arrays of `U − U_prev_picard`
   (unit-clean in log-c primary).
6. **Byte-equivalence** when `make_run_ss_factory=None`: normalized at
   every public entry point + smoke test + self-generated-fixture test.
7. **R<0 by mistake** → ValueError in target computation.
8. **PicardResult propagation** via `ctx["_picard_run_ss_history"]`
   list.  Caller clears at per-V start.
9. **`converged_overall` definition**: warm_walk True AND final accepted
   Picard result converged.  Failed intermediate attempts (bisection
   rejects) do NOT individually fail the V.

## Expected outcomes

Per pre-analysis: smooth S-curve from kinetic onset to ~Levich plateau,
NO cliff.  Picard rescues v1's deep-cathodic failures via interleaved
ξ updates at every warm-walk substep; all 25 V points converge to a
clean curve.  Validates solver via independent code path (Picard +
Newton fixed point = direct Newton baseline).

If cliff appears: pre-analysis was wrong, I integral or θ(y) profile
much different from estimate; investigate via per-iter diagnostics.

If Picard non-convergence persists at deep V: consistent with "no
positive scalar fixed point of the continuum closure at this k₀/V
combo"; informative either way.

## Out of scope (explicit gated follow-ups)

- Reversible/anodic Picard (signed closure derivation).
- Full parallel 2e/4e Fig 4.36 deck run (gated on smoke test + main
  run completing).
- H+ closure-form substitute (gated on `h_closure_rel_err` diagnostic
  showing > 25% at any converged V).
- Surface-coverage / Frumkin isotherm patch (path 2 candidate if cliff
  doesn't appear).
- Hard-sphere mixture D(packing) (path 3 candidate).
- Runtime x-invariance check (DG0-on-x binning; if x-variance suspected).
- Custom from-scratch Chebyshev spectral solver matching Jithin literally.
- Emergency ξ backoff before substep-failure declaration.
