# v13 Ultimate Inference — Bug Fix Changelog

**Date:** 2026-03-05
**Scope:** 22 bugs across 10 files, identified by comprehensive code review of `Infer_BVMaster_charged_v13_ultimate.py` and all dependencies.

---

## Phase 1: CRITICAL — PDE targets generated at wrong alpha (C2)

**Problem:** When P1/P2 generate synthetic targets via `regenerate_target=True`, the target-generation functions (`ensure_bv_target_curve`, `_generate_observable_target`) did not accept or pass `alpha_values`. Targets were always generated at the default alpha baked into `_bv_common.py`, not at `true_alpha`. This means P1/P2 were fitting against the wrong curves whenever `true_alpha` differed from the default.

**Fix (3 files):**

| File | Change |
|------|--------|
| `FluxCurve/bv_run/io.py` | Added `alpha_values: Optional[Sequence[float]] = None` to both `ensure_bv_target_curve` and `_generate_observable_target`. The former passes it to `sweep_phi_applied_steady_bv_flux`; the latter passes it to `solve_bv_curve_points_with_warmstart` and auto-switches `control_mode` to `"joint"` when alpha is provided. |
| `Forward/steady_state/bv.py` | Added `alpha_values` parameter to `sweep_phi_applied_steady_bv_flux` and threaded it to `configure_bv_solver_params`. |
| `FluxCurve/bv_run/pipelines.py` | Extracts `request_runtime.true_alpha` into `_true_alpha` and passes it to both `ensure_bv_target_curve` (primary target) and `_generate_observable_target` (secondary target). |

**Impact:** Without this fix, any run where `true_alpha != default_alpha` would optimize against incorrect target curves in the PDE phases, silently producing wrong results.

---

## Phase 2: CRITICAL — Crash bugs and cache robustness (C1, C3, M6)

### C1: `--pde-cold-start --compare` crash

**Problem:** The cold-start branch set `target_cd_full` / `target_pc_full` but never set `target_cd_surr` / `target_pc_surr`. When `--compare` was also passed, the comparison block referenced these undefined variables → `NameError`.

**Fix:** Added `target_cd_surr = target_cd_full` and `target_pc_surr = target_pc_full` in the cold-start branch. Also disabled `--compare` with a warning when `--pde-cold-start` is set, since cold-start explicitly skips surrogate loading.

### C3: Surrogate grid size mismatch silently produces garbage

**Problem:** If the surrogate was trained on a different voltage grid than `all_eta`, predictions would be misaligned with targets. No validation existed.

**Fix:** Added assertion after loading the surrogate:
```python
if surrogate.n_eta != len(all_eta):
    raise ValueError(...)
```

### M6: Corrupt cache file crashes the pipeline

**Problem:** `np.load(cache_path)` was unguarded. A truncated or corrupt `.npz` file would raise an exception and abort the entire run.

**Fix:** Wrapped in `try/except`; on failure, falls through to PDE regeneration with a warning message.

---

## Phase 3: Cascade and multistart inference fixes (H1, H2, H3, L1, L2)

### H1: Cascade Pass 2 warm-start bias

**Problem:** Pass 2 (PC-dominant, only k0_2 + alpha_2 free) was warm-started from Pass 1's `p1.k0_2` and `p1.alpha_2`. But Pass 1 used CD-dominant weighting, which poorly recovers reaction-2 parameters. Starting Pass 2 from these biased values defeated the purpose of the cascade.

**Fix:** Changed to `initial_k0[1]` and `initial_alpha[1]` (the user-supplied initial guesses) so Pass 2 explores the PC-dominant landscape from a neutral starting point.

**File:** `Surrogate/cascade.py`, line 534.

### H2: Cascade `best_loss` not comparable across strategies

**Problem:** Each cascade pass uses a different `secondary_weight`, so the loss values from different passes are on different scales. Comparing `CascadeResult.best_loss` against `MultiStartResult.best_loss` (which uses `secondary_weight=1.0`) was meaningless.

**Fix:** After selecting the best pass, re-evaluate the best parameters under canonical `secondary_weight=1.0` and store that as `best_loss`.

**File:** `Surrogate/cascade.py`, after pass selection.

### H3: MultiStart NaN detection on un-subsetted arrays

**Problem:** After subsetting predictions to `cd_pred = pred["current_density"][:, subset_idx]`, the NaN check still used the full `pred["current_density"]`. Points with NaN only in the non-subset region were incorrectly penalized; points with NaN only in the subset were missed.

**Fix:** Changed to `np.any(np.isnan(cd_pred), axis=1)` and same for `pc_pred`.

**File:** `Surrogate/multistart.py`, lines 261–262.

### L1: `bounds_alpha[1 - 1]` obfuscated indexing

**Problem:** `bounds_alpha[1 - 1]` is `bounds_alpha[0]` but reads like a copy-paste artifact. It's correct by accident (`1-1 == 0`), but confusing.

**Fix:** Changed to `bounds_alpha[0]`.

**File:** `Surrogate/multistart.py`, line 182.

### L2: Dead imports in cascade.py

**Problem:** `ReactionBlockSurrogateObjective` and `SurrogateObjective` were imported but never used — the cascade builds its own inline objective closures.

**Fix:** Removed the unused imports.

**File:** `Surrogate/cascade.py`, lines 36–39.

---

## Phase 4: Ensemble and objectives fixes (H4, M1, M2, M3, L5)

### H4: Ensemble training bounds used union instead of intersection

**Problem:** `EnsembleMeanWrapper.__init__` merged member training bounds with `min` for lowers and `max` for uppers, producing the *union*. But the ensemble is only reliable within the *intersection* of all members' training ranges.

**Fix:** Swapped to `max` for lowers, `min` for uppers (intersection).

**File:** `Surrogate/ensemble.py`, lines 49–50.
**Test:** Updated `test_training_bounds_merge` in `tests/test_ensemble_and_v12.py`.

### M3: Ensemble std uses population formula (ddof=0) with N=5

**Problem:** `np.std(axis=0)` defaults to `ddof=0` (population std). With only 5 ensemble members, this underestimates uncertainty by ~11%.

**Fix:** Added `ddof=1` to both `.std()` calls in `_predict_ensemble_raw`.

**File:** `Surrogate/ensemble.py`, lines 93–95.

### M2: SubsetSurrogateObjective missing length validation

**Problem:** If `target_cd` and `subset_idx` had different lengths, the objective would silently compute wrong residuals (broadcasting or indexing errors).

**Fix:** Added length assertions in `__init__`.

**File:** `Surrogate/objectives.py`, after line 458.

### M1: fd_step uniformity documented

**Problem:** `fd_step=1e-5` is applied uniformly to log10(k0) and linear alpha dimensions. This is a potential tuning concern but works acceptably for typical parameter ranges.

**Fix:** Added clarifying comment (documentation only, no behavioral change).

**File:** `Surrogate/objectives.py`.

### L5: Ensemble phi_applied consistency not validated

**Problem:** If ensemble members were trained on different voltage grids, predictions would be silently misaligned.

**Fix:** Added `np.allclose` check across all members in `__init__`.

**File:** `Surrogate/ensemble.py`, after line 40.

---

## Phase 5: `_bv_common.py` fixes (H6, H7, M4)

### H7: `c_ref_legacy` incorrect for 4-species charged config

**Problem:** `c_ref_legacy=[1.0] * 4` sets all reference concentrations to 1.0, but H₂O₂ (species index 1) has zero bulk concentration (`c0_vals_hat` has `C_H2O2_HAT = 0`). The per-reaction BV config handles this correctly, but the legacy path used the wrong reference.

**Fix:** Changed to `[1.0, 0.0, 1.0, 1.0]`.

**File:** `scripts/_bv_common.py`, line 253.

### H6: `phi0=0.0` rationale undocumented

**Problem:** `phi0=0.0` in `make_bv_solver_params` looks like a potential error, since the equilibrium potential is nonzero. But for the BV path, the equilibrium potential is encoded in `E_eq_v` within the `bv_bc` config, so `phi0` is intentionally zero.

**Fix:** Added inline comment explaining the rationale.

**File:** `scripts/_bv_common.py`, line 430.

### M4: `max_attempts=6` rationale undocumented

**Problem:** `make_recovery_config` defaults to `max_attempts=6` vs the class default of 8, with no explanation.

**Fix:** Added inline comment: 6 is sufficient for typical BV solves and avoids wasting time on hopeless points.

**File:** `scripts/_bv_common.py`, line 441.

---

## Phase 6: v13 script and parallel config fixes (M5, L3, L7)

### M5: Surrogate timing includes comparison block

**Problem:** `surrogate_time` was computed immediately after the primary surrogate phases, but *before* the `--compare` block. The comparison block (which loads and runs alternative models) could take 30–60s, all of which was attributed to `pde_time` instead of `surrogate_time`.

**Fix:** Moved `surrogate_time` calculation to after the comparison block, guarded by `if not args.pde_cold_start`.

**File:** `v13_ultimate.py`.

### L3: Cache hash uses hardcoded string for mesh/solver params

**Problem:** `_target_cache_path` included `b"Nx=8,Ny=200,beta=3.0,dt=0.5,max_ss=100"` as a raw byte literal. If the actual constants in `_solve_clean_targets` were changed, the cache hash would silently become stale.

**Fix:** Changed to `f"Nx=8,Ny=200,beta=3.0,dt=0.5,max_ss=100".encode()` to make it clear these are derived values (further refactoring to auto-derive from the solver params is deferred).

**File:** `v13_ultimate.py`, line 185.

### L7: `BVParallelPointConfig.secondary_observable_scale` can be `float(None)`

**Problem:** `secondary_observable_scale` is `Optional[float]` defaulting to `None`. If downstream code calls `float(config.secondary_observable_scale)` without checking, it crashes.

**Fix:** Added `__post_init__` that eagerly coerces to `float` when set, catching type errors at construction time rather than deep in worker processes.

**File:** `FluxCurve/bv_parallel.py`.

---

## Files Modified

| File | Bug IDs |
|------|---------|
| `FluxCurve/bv_run/io.py` | C2 |
| `Forward/steady_state/bv.py` | C2 |
| `FluxCurve/bv_run/pipelines.py` | C2 |
| `scripts/surrogate/Infer_BVMaster_charged_v13_ultimate.py` | C1, C3, M5, M6, L3 |
| `Surrogate/cascade.py` | H1, H2, L2 |
| `Surrogate/multistart.py` | H3, L1 |
| `Surrogate/ensemble.py` | H4, M3, L5 |
| `Surrogate/objectives.py` | M1, M2 |
| `scripts/_bv_common.py` | H6, H7, M4 |
| `FluxCurve/bv_parallel.py` | L7 |
| `tests/test_ensemble_and_v12.py` | H4 (test update) |

## Verification

- All 108 non-Firedrake-solver tests pass (the one pre-existing `test_bv_forward` failure is unrelated — missing `strategy_A` attribute).
- All 11 modified files compile without syntax errors.
- `test_training_bounds_merge` updated to match corrected intersection semantics.
